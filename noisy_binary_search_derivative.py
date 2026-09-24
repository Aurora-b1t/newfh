"""
基于导数指标的噪声二分搜索（Derivative NBS）——跳速搜索核心算法
================================================================

实现 DerivativeNoisyBinarySearch：基于 MWU（乘性权重更新，Multiplicative
Weights Update）的噪声二分搜索算法的"导数"变体。与已删除的基础版 NBS
（noisy_binary_search.py）用"BER 升/降"这一二元信号做方向决策不同，
本变体改用基于 BER 与 hoprate 的*导数*（变化率）指标决定移动方向。

研究背景
--------
FHSS 对抗反应式干扰机时，干扰机需要时间锁定当前跳频信道：跳频太慢
（低 hoprate）时干扰机能跟上并压制通信（高 BER）；跳得足够快（高
hoprate）时干扰机跟不上（低 BER）。BER–hoprate 关系因此近似阈值型
单调下降，"找跳速阈值"可建模为一维有序候选集上的带噪搜索——正是
noisy binary search 的适用场景，MWU 权重分布扮演"阈值在哪"的后验
信念。

MWU 框架
--------
维护候选 hoprate 网格上的权重 w（初始均匀、和为 1）。每轮：
  1. 以加权中位数（带随机化，论文 Algorithm 3.1）选择查询 hoprate；
  2. 环境执行后返回该 hoprate 下的 BER 观测；
  3. 由观测推导方向答案（目标在查询点左侧 / 右侧）；
  4. 与答案兼容的候选权重 ×2(1−p)，不兼容的 ×2p（p 为假定的答案噪声
     概率，0 ≤ p < 0.5；只要答案以 > 1/2 的概率正确，兼容侧的期望乘子
     就占优，权重质量向真值附近指数级集中）；
  5. 权重归一化回 1。
收敛判据：max(w) ≥ 1 − δ。收敛后可用 MAP（最大权重网格点）或加权平均
两种方式给出阈值估计。

导数指标与参考点选择的原因
--------------------------
核心想法：不用"BER 升了/降了"的二元信号，而是计算

    metric = ΔBER_percent / Δhoprate

其中 Δ 的参考点不是"上一步"，而是最近一次与当前 hoprate **不同**的
先验观测：

  - ΔBER_percent = (BER_curr − BER_ref) × 100
    （BER 变化的百分点数，例如 0.1 → 0.02 对应 ΔBER_percent = −8）
  - Δhoprate = hoprate_curr − hoprate_ref

与紧邻的上一步作差在 hoprate 未变化时无意义（Δhoprate = 0，梯度不
存在），因此保留一小段历史，始终相对"上一个不同 hoprate"的观测计算
梯度。例如观测序列 (500 Hz, 0.20), (520 Hz, 0.15), (520 Hz, 0.16)，
第三步以第一步为参考：ΔBER% = (0.16 − 0.20)×100，Δhoprate = 20。

历史最多裁剪为两条记录——最新观测 + 最近一个不同 hoprate 的观测——
因为一旦存在不同 hoprate 的参考点，更早的记录（含更早的同 hoprate
记录）都不再需要。

方向决策规则（阈值可配置：类默认 -0.002，settings.NBS_CONFIG 与入口
脚本 CLI 默认 -0.005）：

    metric > threshold → 答案 = "目标在左侧（LEFT）"
                          → h < h_curr 兼容    → w × 2(1−p)
                          → h ≥ h_curr 不兼容  → w × 2p
    metric ≤ threshold → 答案 = "目标在右侧（RIGHT）"
                          → h ≥ h_curr 兼容    → w × 2(1−p)
                          → h < h_curr 不兼容  → w × 2p

  （分裂点 h_curr 的归属：LEFT 答案下查询点自身划为不兼容、RIGHT 答案
   下划为兼容，与 _apply_metric 的 favoured 掩码一致。）

由于参考点的 hoprate 恒与当前观测不同，Δhoprate 永不为 0，无需任何
除零兜底（基础版曾需为 Δhoprate == 0 设计兜底，本变体从参考点机制上
结构性避免了该情形）。

其余部分（权重归一化、加权中位数查询选择、收敛判据）与原始实现完全
一致。

参考论文
--------
Dereniowski et al., "Noisy (Binary) Searching: Simple, Fast and Correct"
(STACS 2025)。查询选择与 MWU 更新对应论文中的 Algorithm 3.1。

配置入口：settings.NBS_CONFIG（p / delta / hoprate_step /
derivative_threshold 的项目级默认值）。
"""

import numpy as np
from typing import Tuple, Optional

import settings


class DerivativeNoisyBinarySearch:
    """
    在区间 [hoprate_min, hoprate_max] 上做基于导数指标的噪声二分搜索，
    用于定位反应式干扰机的跳速跟踪阈值。

    设计意图
    --------
    把"寻找跳速阈值"建模为带噪一维搜索：候选 hoprate 构成有序网格，
    MWU 维护网格上的权重分布（对"阈值位置"的信念）；每轮用加权中位数
    选查询点，用 ΔBER%/Δhoprate 导数指标（而非单纯 BER 升降）推导方向
    答案并更新权重，使分布逐步集中到阈值附近。导数恒相对最近一次
    *不同* hoprate 的观测计算（不一定是紧邻的上一步）。

    与已删除的基础版 NBS 的区别：方向答案来自导数指标，且梯度参考点
    取"最近一次不同 hoprate"的观测（基础版直接比较相邻观测的 BER），
    因此 Δhoprate 恒非零，无需 Δhoprate == 0 的兜底逻辑。

    关键状态
    --------
    weights:
        候选 hoprate 上的权重分布（长度 n_candidates，和为 1）。
    _current_idx / _current_hoprate:
        下一个（或刚完成的）环境 step 的查询点，由 _select_hoprate 设置。
    _obs_hoprate / _obs_ber:
        最新完成的观测（实际测试的 hoprate 及其 BER）。
    _ref_hoprate / _ref_ber:
        最近一次 hoprate 不同于 _obs_hoprate 的观测，即梯度参考点。

    不变量
    ------
    1. 对外可见状态中 sum(weights) == 1（step() 末尾必经 _normalize；
       数值退化全零时回退均匀分布）。
    2. _ref_hoprate 非 None 时必有 _ref_hoprate != _obs_hoprate，因此
       _update_weights 中 Δhoprate ≠ 0 恒成立（无需除零保护）。
    3. 观测历史至多保留两条有效记录——最新观测 + 一个不同 hoprate
       参考；更早记录在状态转移时被丢弃（见 step 的裁剪逻辑）。
    4. 所有观测的 hoprate 都是网格点（形如 hoprate_min + k·hoprate_step）；
       外部传入值由 seed_observation 吸附到最近网格点。

    Parameters
    ----------
    hoprate_min : float
        候选 hoprate 下限（Hz）。
    hoprate_max : float
        候选 hoprate 上限（Hz）。
    hoprate_step : float
        离散步长（Hz）。默认 10，与环境 _apply_hoprate 的 10 Hz 量化
        一致——NBS 提议的网格点不会被环境侧二次取整。
    p : float
        假定的答案噪声概率，0 ≤ p < 0.5。p 越小收敛越快，但对 BER
        读数噪声的容忍度越低。
    delta : float
        置信 / 收敛阈值，0 < δ ≤ 1。最大权重达到 1 − δ 即报告收敛。
    derivative_threshold : float
        导数指标的方向决策阈值。类默认 -0.002（settings.NBS_CONFIG
        与入口脚本 CLI 默认 -0.005）。metric > threshold → 向左移动；
        metric ≤ threshold → 向右移动。
    seed : int or None
        查询随机化 RNG 的种子，用于复现；None 时查询序列不可复现。

    Raises
    ------
    ValueError
        p 不在 [0, 0.5) 或 delta 不在 (0, 1] 时抛出。
    """

    def __init__(
        self,
        hoprate_min: float = 10.0,
        hoprate_max: float = 1000.0,
        hoprate_step: float = 10.0,
        p: float = 0.1,
        delta: float = 0.01,
        derivative_threshold: float = -0.002,
        seed: Optional[int] = None,
    ):
        if not (0.0 <= p < 0.5):
            raise ValueError(f"p must be in [0, 0.5), got {p}")
        if not (0.0 < delta <= 1.0):
            raise ValueError(f"delta must be in (0, 1], got {delta}")

        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.hoprate_step = float(hoprate_step)
        self.p = float(p)
        self.delta = float(delta)
        self.derivative_threshold = float(derivative_threshold)

        # ---- candidate grid ---------------------------------------------------
        n = int(round((self.hoprate_max - self.hoprate_min) / self.hoprate_step)) + 1
        self.candidates = np.linspace(
            self.hoprate_min, self.hoprate_max, n, dtype=np.float64
        )
        self.n_candidates = len(self.candidates)

        # ---- internal state ---------------------------------------------------
        self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates

        # The query selected for the next environment step (the hoprate that will
        # be, or just was, tested).  Set by ``_select_hoprate``.
        self._current_idx: Optional[int] = None
        self._current_hoprate: Optional[float] = None

        # Latest completed observation (hoprate actually tested + its BER).
        self._obs_hoprate: Optional[float] = None
        self._obs_ber: Optional[float] = None

        # Most recent completed observation whose hoprate differs from
        # ``_obs_hoprate`` — the reference against which the gradient is taken.
        # Its hoprate is guaranteed != ``_obs_hoprate`` whenever it is not None.
        self._ref_hoprate: Optional[float] = None
        self._ref_ber: Optional[float] = None

        self._step_count: int = 0

        self._rng = np.random.RandomState(seed)

        # ---- diagnostic state -------------------------------------------------
        self._last_derivative: Optional[float] = None
        self._last_delta_hoprate: Optional[float] = None
        self._last_delta_ber: Optional[float] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> float:
        """
        重置全部内部状态，并返回初始待测 hoprate。

        Returns
        -------
        float
            由（均匀权重的）加权中位数选出的第一个 hoprate。
        """
        self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates
        self._current_idx = None
        self._current_hoprate = None
        self._obs_hoprate = None
        self._obs_ber = None
        self._ref_hoprate = None
        self._ref_ber = None
        self._step_count = 0
        self._last_derivative = None
        self._last_delta_hoprate = None
        self._last_delta_ber = None
        return self._select_hoprate()

    def seed_observation(self, hoprate: float, ber: float) -> None:
        """
        用一组初始 (hoprate, BER) 观测为算法播种。

        适用于第一个测试 hoprate 由外部（而非 ``reset()``）选定的场景。
        播种的观测记录为最新观测；若此前已存在不同 hoprate 的观测，则保留
        为参考点，使下一次 ``step()`` 能立即计算导数。

        Parameters
        ----------
        hoprate : float
            实际测试过的 hoprate。
        ber : float
            在该 hoprate 上观测到的平均 BER。
        """
        idx = int(np.argmin(np.abs(self.candidates - float(hoprate))))
        h_new = float(self.candidates[idx])
        ber = float(ber)

        if self._obs_ber is not None and h_new != self._obs_hoprate:
            # Promote the previous latest observation to the reference.
            self._ref_hoprate = self._obs_hoprate
            self._ref_ber = self._obs_ber
        # If h_new == _obs_hoprate, keep the existing reference and just refresh
        # the latest BER below.

        self._current_idx = idx
        self._current_hoprate = h_new
        self._obs_hoprate = h_new
        self._obs_ber = ber

    def step(self, ber: float) -> float:
        """
        用观测到的 BER 更新权重，并返回下一个待测 hoprate。

        BER 通过与"最近一次 hoprate 不同的观测"（参考点）比较得到导数指标。
        若本次测试的 hoprate 与已存最新观测相同，则只刷新该观测的 BER
        （参考点沿用）；若 hoprate 发生变化，则原最新观测转为新的参考点。

        首次调用（尚无任何历史观测）时权重保持均匀，只选出新 hoprate；
        在拿到两个 *不同* hoprate 的观测之前不做任何权重更新。

        Parameters
        ----------
        ber : float
            刚结束的环境 step 中观测到的平均 BER。

        Returns
        -------
        float
            下一个环境 step 使用的 hoprate。
        """
        self._step_count += 1
        ber = float(ber)

        # Hoprate that was just tested (selected at the end of the previous step).
        h_new = self._current_hoprate

        # First observation: record it, no reference to compare against yet.
        if self._obs_ber is None:
            self._obs_hoprate = h_new
            self._obs_ber = ber
            return self._select_hoprate()

        if h_new == self._obs_hoprate:
            # Same hoprate as the latest observation: refresh its BER to the
            # latest reading (older readings at this hoprate are discarded).
            # The reference (different hoprate) is reused as-is.
            self._obs_ber = ber
        else:
            # Hoprate changed: the previous latest observation becomes the new
            # reference (its hoprate differs from h_new).  The old reference is
            # no longer needed and is discarded.
            self._ref_hoprate = self._obs_hoprate
            self._ref_ber = self._obs_ber
            self._obs_hoprate = h_new
            self._obs_ber = ber

        # Update weights whenever a different-hoprate reference is available.
        if (self._ref_ber is not None and self._ref_hoprate is not None
                and self._ref_hoprate != h_new):
            self._update_weights(ber, self._ref_ber, h_new, self._ref_hoprate)
        else:
            # No different-hoprate reference yet (only one hoprate seen so far):
            # a real gradient cannot be formed.  Force a move so the weighted
            # median shifts and a different hoprate gets sampled next — this is
            # the same bootstrap the original algorithm used for Δhoprate == 0.
            self._force_move()
        self._normalize()

        return self._select_hoprate()

    def get_best_hoprate(self) -> float:
        """返回当前权重最大的候选 hoprate（MAP 估计）。"""
        return self.candidates[np.argmax(self.weights)]

    def get_weighted_average(self) -> float:
        """返回按权重加权的平均 hoprate（软估计，对噪声更平滑）。"""
        return float(np.average(self.candidates, weights=self.weights))

    def is_converged(self) -> bool:
        """最大权重是否 ≥ 1−δ，即是否达到收敛判据。"""
        return bool(np.max(self.weights) >= 1.0 - self.delta)

    def get_distribution(self) -> Tuple[np.ndarray, np.ndarray]:
        """返回 ``(candidates, weights)`` 副本，用于诊断输出与绘图。"""
        return self.candidates.copy(), self.weights.copy()

    def get_last_derivative(self) -> Optional[float]:
        """返回最近一次权重更新使用的导数值（未更新过则为 None）。"""
        return self._last_derivative

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def current_hoprate(self) -> Optional[float]:
        return self._current_hoprate

    @property
    def current_idx(self) -> Optional[int]:
        return self._current_idx

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # Internal: query selection  (paper Algorithm 3.1)
    # ------------------------------------------------------------------

    def _select_hoprate(self) -> float:
        """
        带随机化的加权中位数查询选择（论文 Algorithm 3.1）。

        1. 找到累计权重跨过 W/2 的下标 *k*；
        2. 在中位数两侧权重不平衡时，按概率 α 在 *k* 与 *k+1* 之间随机选择，
           以保留噪声二分搜索所需的不确定性。
        """
        total = np.sum(self.weights)
        cumulative = np.cumsum(self.weights)

        # weighted median index
        k = int(np.searchsorted(cumulative, total / 2.0))
        k = min(k, self.n_candidates - 1)

        if k < self.n_candidates - 1 and self.weights[k] > 0:
            sum_left = cumulative[k] - self.weights[k]   # Σ₁ᵏ⁻¹
            sum_right = total - cumulative[k]             # Σₖ₊₁ⁿ

            # α = (Σ₁ᵏ − Σₖ₊₁ⁿ) / (2 w_k)
            alpha = (sum_left + self.weights[k] - sum_right) / (2.0 * self.weights[k])
            alpha = float(np.clip(alpha, 0.0, 1.0))

            chosen = k if self._rng.random() < alpha else k + 1
        else:
            chosen = k

        self._current_idx = int(chosen)
        self._current_hoprate = float(self.candidates[chosen])
        return self._current_hoprate

    # ------------------------------------------------------------------
    # Internal: MWU weight update — DERIVATIVE VERSION
    # ------------------------------------------------------------------

    def _update_weights(self, ber_curr: float, ber_ref: float,
                        h_curr: float, h_ref: float) -> None:
        """
        用导数指标执行一次 MWU 权重更新。

        导数在"当前观测（``h_curr``, ``ber_curr``）"与"最近一次不同 hoprate
        的观测（``h_ref``, ``ber_ref``）"之间计算；当前 hoprate *h_curr*
        即查询元素（分裂点）。方向答案由下式导出：

            delta_ber_percent = (ber_curr - ber_ref) * 100
            delta_hoprate     = h_curr - h_ref          （保证 != 0）
            metric            = delta_ber_percent / delta_hoprate

            metric > threshold   →  answer = LEFT   →  兼容 h ≤ h_curr
            metric ≤ threshold   →  answer = RIGHT  →  兼容 h ≥ h_curr

        由于 ``h_ref`` 恒为与 ``h_curr`` 不同的 hoprate，``delta_hoprate``
        不会为零，无需任何 clamp。兼容侧权重乘以 2(1−p)，不兼容侧乘以 2p；
        分裂点始终取当前 hoprate 的下标。
        """
        curr_idx = self._current_idx
        if curr_idx is None:
            return

        # Δhoprate is guaranteed non-zero: the reference hoprate always differs
        # from the current hoprate (enforced by ``step`` / ``seed_observation``).
        delta_hoprate = h_curr - h_ref
        delta_ber = ber_curr - ber_ref
        self._last_delta_hoprate = delta_hoprate
        self._last_delta_ber = delta_ber

        metric = (delta_ber * 100) / delta_hoprate
        self._last_derivative = metric

        self._apply_metric(metric, curr_idx)

    def _force_move(self) -> None:
        """
        在尚无"不同 hoprate 参考点"时的引导性回退。

        此时只观测过一个 hoprate，无法形成真实导数。强制一次 RIGHT 移动
        （令 metric 略低于阈值），使加权中位数偏移、下一步采样到不同的
        hoprate——与原算法对 Δhoprate == 0 情形的处理保持一致。
        """
        curr_idx = self._current_idx
        if curr_idx is None:
            return

        self._last_delta_hoprate = 0.0
        self._last_delta_ber = 0.0
        self._last_derivative = self.derivative_threshold - 0.001  # ≤ threshold → RIGHT

        self._apply_metric(self._last_derivative, curr_idx)

    def _apply_metric(self, metric: float, curr_idx: int) -> None:
        """对给定导数 metric 与分裂点执行一次 MWU 权重更新。

        方向决策：``metric > threshold`` 视为"左侧更优"（支持更小 hoprate），
        否则视为"当前点及右侧更优"；被支持的一半权重乘以 2(1-p)，
        另一半乘以 2p（p 为假设噪声概率），随后由调用方归一化。
        """
        indices = np.arange(self.n_candidates)

        # 基于阈值的方向决策
        if metric > self.derivative_threshold:
            # metric > threshold → answer = LEFT → 偏向左半区
            favoured = indices < curr_idx
        else:
            # metric ≤ threshold → answer = RIGHT → 偏向左半区
            favoured = indices >= curr_idx

        n_fav = np.sum(favoured)
        if n_fav == 0 or n_fav == self.n_candidates:
            return

        self.weights[favoured] *= 2.0 * (1.0 - self.p)
        self.weights[~favoured] *= 2.0 * self.p

    def _normalize(self) -> None:
        """把权重重新归一化到和为 1；退化时重置为均匀分布。"""
        s = np.sum(self.weights)
        if s > 0:
            self.weights /= s
        else:
            # 退化兜底：重置为均匀分布
            self.weights = np.ones(self.n_candidates, dtype=np.float64) / self.n_candidates


# ------------------------------------------------------------------
# Quick smoke test
# ------------------------------------------------------------------
if __name__ == "__main__":
    def _sim_ber(hoprate: float, optimal: float, max_hr: float,
                 baseline: float = 0.1, scale: float = 0.3,
                 rng: np.random.RandomState = None) -> float:
        """模拟含噪 BER：越接近 *optimal* hoprate 越低，并叠加高斯噪声。"""
        dist = abs(hoprate - optimal) / max_hr
        noise = rng.normal(0, 0.02) if rng else 0.0
        return max(0.0, min(1.0, baseline + scale * dist + noise))

    def run_test(label: str, optimal: float, steps: int = 80, threshold: float = -0.002):
        nbs = DerivativeNoisyBinarySearch(
            hoprate_min=10.0, hoprate_max=1000.0, hoprate_step=10.0,
            p=0.1, delta=0.05,
            derivative_threshold=threshold,
            seed=settings.RANDOM_SEED,
        )
        print(f"\n{'='*60}")
        print(f"Mode: {label}")
        print(f"True optimal hoprate: {optimal:.0f} Hz")
        print(f"Derivative threshold: {threshold}")
        print(f"{'='*60}")

        h = nbs.reset()
        print(f"Step {nbs.step_count:3d}  init hoprate = {h:6.0f} Hz")

        for i in range(steps):
            ber = _sim_ber(nbs.current_hoprate, optimal, nbs.hoprate_max,
                           rng=nbs._rng)
            h = nbs.step(ber)

            metric = nbs.get_last_derivative()
            metric_str = f"{metric:10.4f}" if metric is not None else "      N/A"

            if nbs.is_converged():
                best = nbs.get_best_hoprate()
                wavg = nbs.get_weighted_average()
                print(f"Step {nbs.step_count:3d}  CONVERGED  "
                      f"hop={h:6.0f}  best={best:6.0f}  wavg={wavg:6.0f}  "
                      f"metric={metric_str}  max_w={np.max(nbs.weights):.4f}")
                print(f"  → estimate = {best:.0f} Hz  "
                      f"(error = {abs(best - optimal):.0f} Hz,  "
                      f"steps = {nbs.step_count})")
                return nbs

            if i % 10 == 0 or i == steps - 1:
                best = nbs.get_best_hoprate()
                wavg = nbs.get_weighted_average()
                max_w = np.max(nbs.weights)
                print(f"Step {nbs.step_count:3d}  hop={h:6.0f}  BER={ber:.4f}  "
                      f"best={best:6.0f}  wavg={wavg:6.0f}  "
                      f"metric={metric_str}  max_w={max_w:.4f}")

        best = nbs.get_best_hoprate()
        wavg = nbs.get_weighted_average()
        print(f"Did not converge in {steps} steps.")
        print(f"  best={best:.0f} Hz  wavg={wavg:.0f} Hz  "
              f"error={abs(best - optimal):.0f} Hz")
        return nbs

    # Test derivative-based MWU mapping
    run_test("Derivative-based (ΔBER%/Δhoprate ≤ threshold → RIGHT)",
             optimal=350.0, threshold=-0.002)
