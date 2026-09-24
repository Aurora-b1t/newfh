"""
FHSS 抗干扰仿真中的干扰机（jammer）实现。

本模块提供三类可叠加到接收信号上的干扰源，供 fh_env.py 的 FHSSQPSKEnv 调用；
所有数值参数均由构造参数传入，配置集中在 settings.py 的 JAMMER_CONFIG。

一、三类干扰源的职责
  1. ReactiveJammer —— 反应式干扰机
     基于能量检测理论（Urkowitz, "Energy Detection of Unknown Deterministic
     Signals", Proc. IEEE 55(4), 1967），以 detection_time（默认 1 ms）为基本
     时隙执行 scan → detect → jam 状态机：扫描某信道后做一次能量检测判决，
     判为"有信号"则下一时隙对该信道输出压制波形，随后回到**同一信道**重新扫描。
     - 检测门限 V_T 由虚警概率 p_fa 与噪声方差反解：H0（只有噪声）下检测统计量
       服从中心 chi2 分布 χ²(2γ)，γ = T·W 为时间-带宽积；
     - 检测概率 P_D 对 Rayleigh 衰落做平均：H1（存在信号）下统计量服从非中心
       chi2 分布 χ²(2γ, λ)，非中心参量 λ = 2·SNR·γ，期望用 Gauss-Laguerre
       数值积分求出（见 _fading_averaged_P_D）。
     每个时隙只在起点检测一次、且压制时隙内不再检测，因此该干扰机能跟上慢跳频，
     但跟不上快跳频：当每跳驻留时间接近或短于检测时隙时，检测结果已不能代表
     当前跳的信道，压制会大量落在错误信道上。

  2. IndiscriminateJammer —— 无差别（非反应式）干扰机
     不感知发射机，按固定 RF 时钟周期性压制频段，支持两种模式及其组合
     （mode = 'sweep' / 'comb' / 'both'）：
     - sweep 扫频：以 step 为步进、dwell_time 为驻留时间逐个频点扫描，
       默认 20 个频点 × 4 ms 驻留 = 80 ms 一个完整扫频周期；
     - comb 梳状：两组信道（channels_phase0 / channels_phase1）按
       switch_interval 交替压制；switch_interval 必须是 1 ms 的正整数倍
       （COMB_TIME_QUANTUM = 0.001 s），保证相位切换与 1 ms 基本时隙对齐。
     支持两条信号生成路径：precompute() 预生成 + get_composite_signal() 取切片
     （加速路径），以及 generate_samples() 动态生成（缓存的确定性载波 +
     现场抽取的新鲜基带噪声）。

  3. FastNoiseSource —— 快速带限噪声源
     预生成长段带限高斯噪声缓冲，供动态路径按需切片，避免每次调用都做
     FFT/IFFT。

二、共享带限噪声变体池（baseband variant pool）
  预生成路径下，相同带宽的干扰（reactive / sweep / comb）共享同一个
  BandLimitedNoiseVariantPool：池内保存 baseband_variant_count（默认 4）条
  相互独立、各自归一化到单位标准差的带限噪声基带波形，由 JammerVariantSelector
  随机**有放回**地选择（相邻周期可能抽到同一条变体）。抽选粒度与干扰的"自然
  周期"对齐，使同一周期内波形保持一致、只有跨周期才发生跳变：
    - sweep 按完整扫频周期（samples_per_dwell × num_steps）选变体；
    - comb 按 phase0 + phase1 完整周期（2 × comb_switch_samples）选变体；
    - reactive 按每次压制时隙（samples_per_ms）现场抽选。
  选择结果按 (jammer_kind, cycle_idx) 记忆化，因此对同一周期重复取切片会得到
  完全相同的波形——这既是"任意起点、任意长度"切片式取用能保持时间一致性的
  前提，也是分 block 并行仿真下结果可复现的来源。

三、预生成加速路径的随机性语义
  - 载波（cos 项）只依赖 RF 布局（Startfre/Endfre/step/dwell/切换间隔/信道组），
    是确定性的，由 _ensure_carrier_cache 缓存，动态路径与预生成路径共用同一份；
  - 基带噪声是唯一的随机来源：预生成路径在池内做有放回选择，动态路径则由调用方
    传入的 rng 现场抽取（见 _noise_source_slice），不消耗全局随机状态；
  - 因此预生成路径在给定 selector 种子下完全可复现，且与动态路径共享同一载波
    相位与周期结构，两者只在"基带噪声是否跨周期复用"上有差别。

四、与 fh_env.py / settings.py 的关系
  - fh_env.py 的 _build_jammer_variant_resources 负责按带宽构造
    BandLimitedNoiseVariantPool 与唯一的 JammerVariantSelector，并把池分发给
    ReactiveJammer / IndiscriminateJammer；episode 重置时调用 selector.reset()，
    每 block 以 jammer_ptr 作为连续 RF 采样时钟调用 get_composite_signal /
    generate_samples；
  - 本模块不导入 settings.py：所有阈值、带宽、功率、时隙参数都由构造参数给出，
    settings.JAMMER_CONFIG 只是这些参数的默认来源与说明文档。
"""

import math

import numpy as np
from scipy.stats import chi2, ncx2
from scipy.special import roots_laguerre


def _rng_randint(rng, low, high=None):
    """从新旧两种 NumPy 随机 API 中抽取一个整数。

    兼容层：legacy 的 RandomState / 全局 np.random 提供 randint，新式 Generator
    提供 integers；rng=None 时统一退化到全局 np.random。上层代码因此可以同时
    接受环境传入的 RandomState（训练可复现）与 Generator（新式 API），
    而无需在各调用点分支。

    Args:
        rng: 随机数生成器对象；None 表示使用全局 np.random。
        low: 下界（含）。
        high: 上界（不含）；None 时按 NumPy 语义把 low 当作上界、下界取 0。

    Returns:
        int: 抽取到的 Python 原生整数。
    """
    rng = np.random if rng is None else rng
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def _rng_random(rng):
    """从新旧两种 NumPy 随机 API 中抽取一个 [0, 1) 均匀分布随机数。

    两个 API 的对应方法同名（random），此函数的主要作用是统一 rng=None 的退化
    行为并保证返回 Python float，避免各调用点重复判断。

    Args:
        rng: 随机数生成器对象；None 表示使用全局 np.random。

    Returns:
        float: [0.0, 1.0) 区间内的随机数。
    """
    rng = np.random if rng is None else rng
    return float(rng.random())


def _rng_standard_normal(rng, size):
    """从新旧两种 NumPy 随机 API 中抽取标准正态样本。

    方法名与实现都按 API 分支：legacy 的全局 np.random 用 randn（legacy gauss
    算法），新式 Generator 用 standard_normal（ziggurat 算法）。两者算法不同、
    随机流也不可互换，因此不能简单地把 randn 换成 standard_normal，否则历史
    结果（离线 replay 的波形）会发生变化。

    Args:
        rng: 随机数生成器对象；None 表示使用全局 np.random。
        size: 采样点数（输出为一维）。

    Returns:
        ndarray: 形状 (size,) 的标准正态样本。
    """
    rng = np.random if rng is None else rng
    if rng is np.random:
        return rng.randn(size)
    return rng.standard_normal(size)


def _generate_band_limited_noise(Fs, bandwidth, length, rng=None):
    """生成一条归一化的带限高斯噪声序列（时域实采样）。

    实现思路：先在时域生成白高斯噪声，做 rfft 后在频域把高于截止频率的分量
    置零，再 irfft 回时域。频域硬截断等价于理想砖墙低通滤波器（过渡带为 0），
    频带外泄漏只取决于 FFT 长度，因此长缓冲比短缓冲更"干净"。
    最后除以样本标准差归一化到单位功率，使输出与调用方配置的干扰功率解耦
    （功率缩放由各干扰机自行施加），从而不同带宽的噪声源可以直接互换。

    Args:
        Fs: 采样率 (Hz)，决定频率轴刻度。
        bandwidth: 单边带宽 (Hz)；<= 0 表示不截断（全带宽白噪声）。
        length: 输出采样点数，必须 > 0。
        rng: 随机数生成器；None 表示使用全局 np.random。

    Returns:
        ndarray: float32，形状 (length,)，样本标准差约等于 1
        （bandwidth <= 0 或频谱被完全清空等退化情形除外）。

    Raises:
        ValueError: length <= 0。
    """
    length = int(length)
    if length <= 0:
        raise ValueError("Band-limited noise length must be positive.")

    n = np.asarray(_rng_standard_normal(rng, length), dtype=np.float32)
    spec = np.fft.rfft(n)
    if bandwidth > 0:
        # rfft 频点 f_k = k·Fs/length，故 cutoff_idx = floor(bandwidth·length/(2·Fs))
        # 是最后一个需要保留的频点下标（单边带宽 → 除以 2·Fs）。
        cutoff_idx = int(np.floor(float(bandwidth) * length / (2.0 * float(Fs))))
        # 仅当确有频点需要清除时才切片赋值；若带宽已覆盖到 Nyquist，
        # 保留全部频点，避免把频谱整体清空导致输出退化为全零。
        if cutoff_idx + 1 < len(spec):
            spec[cutoff_idx + 1:] = 0

    n_lp = np.fft.irfft(spec, n=length)
    # 归一化到单位标准差；std 极小时（零序列/常数序列）跳过，防止除零产生 NaN。
    std_val = np.std(n_lp)
    if std_val > 1e-12:
        n_lp /= std_val
    return np.asarray(n_lp, dtype=np.float32)


def _noise_source_slice(source, num_samples, rng=None):
    """按需把 rng 透传给噪声源，同时兼容旧式自定义噪声源。

    只有 FastNoiseSource 支持 rng 关键字参数；调用方（含外部脚本）可能传入任意
    实现了 get_noise(num_samples) 的旧式对象，因此这里用 isinstance 白名单判断
    后再决定是否传递 rng，避免破坏鸭子类型的旧接口。

    Args:
        source: 噪声源对象，需实现 get_noise(num_samples[, rng])。
        num_samples: 需要的采样点数。
        rng: 随机数生成器；None 或旧式噪声源时被忽略。

    Returns:
        ndarray: 长度为 num_samples 的噪声片段。
    """
    if rng is None or not isinstance(source, FastNoiseSource):
        return source.get_noise(num_samples)
    return source.get_noise(num_samples, rng=rng)


def validate_baseband_variant_count(value):
    """校验并归一化"基带噪声变体数量"配置项。

    该数值同时决定 BandLimitedNoiseVariantPool 的池大小与 JammerVariantSelector
    的可选下标范围，两者必须严格一致，因此在构造入口处集中校验，避免出现
    "选择器抽到池中不存在的下标"这类越界错误。
    特意排除 bool：Python 中 True/False 也是 int 子类，但把 True 当作"1 条变体"
    几乎总是配置书写错误，故直接拒绝而不是静默接受。

    Args:
        value: 待校验取值，必须是 int 或 np.integer 且 > 0。

    Returns:
        int: 校验通过的 Python 原生正整数。

    Raises:
        ValueError: 不是整数类型（含 bool）或取值 <= 0。
    """
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("JAMMER_CONFIG['baseband_variant_count'] must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError("JAMMER_CONFIG['baseband_variant_count'] must be a positive integer.")
    return value


class BandLimitedNoiseVariantPool:
    """某一带宽下、相互独立的若干条归一化带限噪声基带波形。

    设计意图：预生成路径需要"可复用的随机波形"，但若把同一条噪声无限重复，干扰
    在长时尺度上会退化成确定性周期信号（可被学习算法完全预测）。折中做法是预生成
    num_variants 条统计独立、功率归一的波形，运行期按周期有放回抽取——既省掉
    FFT/IFFT 开销，又保留"不同周期波形不同"的随机性。

    不变量：
      - 池内每条变体长度相同（length），且各自归一化到单位标准差；
      - 所有变体使用同一 (Fs, bandwidth)，因此可被带宽匹配的干扰机共享
        （fh_env.py 按 bandwidth 建池，再分发给 reactive / sweep / comb）。

    属性：
        Fs (float): 采样率 (Hz)。
        bandwidth (float): 单边带宽 (Hz)。
        num_variants (int): 变体条数（已经过 validate_baseband_variant_count）。
        length (int): 每条变体的采样点数。
        variants (ndarray): float32，形状 (num_variants, length)，每行一条变体。
    """

    def __init__(self, Fs, bandwidth, num_variants, length, rng=None):
        """一次性生成整池变体。

        Args:
            Fs: 采样率 (Hz)。
            bandwidth: 单边带宽 (Hz)。
            num_variants: 变体条数（正整数）。
            length: 每条变体的采样点数（>0）。应不小于最长的单周期切片需求，
                fh_env.py 会取 reactive 时隙 / sweep 周期 / comb 周期三者的最大值；
                不足时 get_variant 仍能工作（按周期环绕），只是会引入重复。
            rng: 随机数生成器；None 表示使用全局 np.random。

        Raises:
            ValueError: num_variants 非法（非正整数）或 length <= 0。
        """
        self.Fs = float(Fs)
        self.bandwidth = float(bandwidth)
        self.num_variants = validate_baseband_variant_count(num_variants)
        self.length = int(length)
        if self.length <= 0:
            raise ValueError("Band-limited noise variant length must be positive.")

        self.variants = np.empty(
            (self.num_variants, self.length),
            dtype=np.float32,
        )
        # 逐条独立生成：每条变体各自消耗 rng，彼此统计独立（互不相关），
        # 这正是"跨周期换变体"能带来波形跳变的物理基础。
        for variant_idx in range(self.num_variants):
            self.variants[variant_idx] = _generate_band_limited_noise(
                self.Fs,
                self.bandwidth,
                self.length,
                rng=rng,
            )

    def get_variant(self, variant_idx, num_samples=None, start_sample_idx=0):
        """按周期环绕地截取指定变体的一段连续采样。

        变体被当作无限循环序列使用：越出 length 后回到头部继续取，因此任意
        (start_sample_idx, num_samples) 都能得到连续切片，调用方无需关心缓冲长度
        是否整除请求长度（reactive 的 1 ms 时隙、sweep 的整周期、comb 的相位段
        都可以直接取用）。

        Args:
            variant_idx: 变体下标，范围 [0, num_variants)。
            num_samples: 需要的采样点数；None 表示取整条变体（length）。
            start_sample_idx: 起始采样点下标（内部对 length 取模后使用）。

        Returns:
            ndarray: float32，形状 (num_samples,)。

        Raises:
            IndexError: variant_idx 越界。
            ValueError: num_samples 或 start_sample_idx 为负。
        """
        variant_idx = int(variant_idx)
        if variant_idx < 0 or variant_idx >= self.num_variants:
            raise IndexError("variant_idx is outside the configured baseband pool.")

        if num_samples is None:
            num_samples = self.length
        num_samples = int(num_samples)
        start_sample_idx = int(start_sample_idx)
        if num_samples < 0 or start_sample_idx < 0:
            raise ValueError("Noise slice indices must be non-negative.")
        if num_samples == 0:
            return np.zeros(0, dtype=np.float32)

        source = self.variants[variant_idx]
        result = np.empty(num_samples, dtype=np.float32)
        source_idx = start_sample_idx % self.length
        result_idx = 0
        # 逐段拷贝直到填满请求长度：每轮取"本轮剩余"与"本次需求剩余"的较小者，
        # 越出变体末尾后 source_idx 归零回卷，实现周期延拓。
        while result_idx < num_samples:
            take = min(self.length - source_idx, num_samples - result_idx)
            result[result_idx:result_idx + take] = source[source_idx:source_idx + take]
            result_idx += take
            source_idx = 0
        return result


class JammerVariantSelector:
    """被所有干扰机共享的、可复现的"有放回"变体选择器。

    设计意图：reactive / sweep / comb 共享同一个基带池；若各自持有随机状态，
    抽选顺序会依赖"谁先被调用"（训练时 block 的调度顺序、是否启用 reactive 等
    都会改变调用序），导致同一实验配置下波形不可复现。这里把随机源收敛到一个
    对象，并把结果按 (jammer_kind, cycle_idx) 记忆化：
      - 同一周期内重复取切片 → 命中缓存 → 波形完全一致；
      - 同一周期在不同 block / 不同 worker 中被访问 → 结果稳定，与访问顺序无关；
      - reset() 把序列恢复到种子初始状态，供 episode 重置使用。

    注意这是"有放回"抽样（各次 draw 相互独立），相邻周期抽到同一条变体是正常
    现象——刻意保留这种随机性，使周期边界上的波形变化本身也是随机的。

    属性：
        num_variants (int): 可选变体数，必须与共享池的 num_variants 一致
            （IndiscriminateJammer._validate_variant_pools 会校验）。
        seed (int): 实际使用的种子；构造时若未指定则从全局 np.random 抽取，
            记录在此以便复现。
        _rng: 独立的 RandomState，不复用全局随机流，避免污染训练随机数序列。
        _cycle_choices (dict): {(jammer_kind, cycle_idx): variant_idx} 记忆化表。
    """

    def __init__(self, num_variants, seed=None):
        """初始化选择器并固定随机序列。

        Args:
            num_variants: 可选变体数（正整数）。
            seed: 随机种子；None 时从全局 np.random 抽取 [0, int32.max) 的整数，
                并把实际取值保存在 self.seed 中，便于事后复现。

        Raises:
            ValueError: num_variants 非法（非正整数）。
        """
        self.num_variants = validate_baseband_variant_count(num_variants)
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self._cycle_choices = {}

    def reset(self):
        """把随机序列与记忆化表恢复为初始状态（episode 重置时调用）。

        重置后相同的访问序列会得到与首次运行完全相同的变体选择，这是预生成
        路径可复现的关键。
        """
        self._rng = np.random.RandomState(self.seed)
        self._cycle_choices.clear()

    def draw(self):
        """独立地抽一个变体下标（有放回，各次抽取互不影响）。

        Returns:
            int: [0, num_variants) 内的下标。
        """
        return int(self._rng.randint(0, self.num_variants))

    def choice_for_cycle(self, jammer_kind, cycle_idx):
        """取得某干扰机在指定周期上使用的变体下标，带记忆化。

        首次访问某 (jammer_kind, cycle_idx) 时抽取并记录，之后恒定返回同一值；
        因此"周期"既是抽选粒度也是缓存粒度。

        Args:
            jammer_kind: 干扰机标识（如 "sweep" / "comb"），参与缓存键，
                使不同干扰机的抽选序列彼此独立。
            cycle_idx: 周期序号（>=0）：sweep 以完整扫频周期为单位，
                comb 以 phase0 + phase1 完整周期为单位。

        Returns:
            int: 变体下标。

        Raises:
            ValueError: cycle_idx 为负（负下标会与"从末尾倒数"的语义混淆，故拒绝）。
        """
        cycle_idx = int(cycle_idx)
        if cycle_idx < 0:
            raise ValueError("cycle_idx must be non-negative.")
        key = (str(jammer_kind), cycle_idx)
        if key not in self._cycle_choices:
            self._cycle_choices[key] = self.draw()
        return self._cycle_choices[key]

# -----------------------------
# 快速噪声源
# -----------------------------
class FastNoiseSource:
    """
    预生成长段带限噪声的缓冲区，避免反复计算 FFT/IFFT。

    动态路径（use_pregen=False 或未提供变体池）每次需要噪声时若都现场生成，
    每条 1 ms 时隙都要付一次 FFT/IFFT 的代价；本类改为在构造时生成一段
    duration 秒的噪声，运行期只做切片/平铺，把代价摊到整个 episode。
    与 BandLimitedNoiseVariantPool 的区别：本类只保存**一条**波形且不参与
    变体抽选，随机性完全由调用方传入的 rng 决定（见 get_noise）。

    属性：
        length (int): 缓冲采样点数 = int(Fs · duration)。
        noise (ndarray): float32，形状 (length,)，已归一化到单位标准差。
    """
    def __init__(self, Fs, bandwidth, duration=1.0, rng=None):
        """生成缓冲。

        Args:
            Fs: 采样率 (Hz)。
            bandwidth: 单边带宽 (Hz)。
            duration: 缓冲时长 (s)，默认 1 s；实际点数取 int(Fs·duration)。
            rng: 随机数生成器；None 表示使用全局 np.random。

        Raises:
            ValueError: int(Fs·duration) <= 0。
        """
        self.length = int(Fs * duration)
        self.noise = _generate_band_limited_noise(
            Fs,
            bandwidth,
            self.length,
            rng=rng,
        )

    def get_noise(self, num_samples, rng=None):
        """返回 num_samples 个噪声采样点。

        两条分支的语义不同，注意取舍：
          - num_samples >= length：整段循环平铺后截断，结果确定、与 rng 无关，
            因此长请求不消耗随机数（大请求下随机性退化为固定波形）；
          - num_samples < length：随机选起始位置后连续截取，起点均匀分布在
            [0, length - num_samples]（randint 上界不含），两次调用之间不保证
            相位连续，但能保证每次取到的都是"新鲜"的一段。

        Args:
            num_samples: 需要的采样点数。
            rng: 随机数生成器；None 表示使用全局 np.random。

        Returns:
            ndarray: float32，长度为 num_samples 的噪声片段。
        """
        if num_samples >= self.length:
            # 这种情况下直接返回全部并循环填充
            tile_count = (num_samples // self.length) + 1
            return np.tile(self.noise, tile_count)[:num_samples]
            
        # 随机起始下标：上界为 length - num_samples（不含），恰好覆盖最后一个
        # 能完整截取 num_samples 点的起点，不会越界。
        start = _rng_randint(rng, 0, self.length - num_samples)
        return self.noise[start : start + num_samples]


# -----------------------------
# 干扰机
# -----------------------------
class ReactiveJammer:
    """
    基于能量检测理论的反应式干扰机。

    参考文献：
      Urkowitz, "Energy Detection of Unknown Deterministic Signals",
      Proc. IEEE, vol. 55, no. 4, pp. 523–531, Apr. 1967.

    以 detection_time 为**基本时隙**（默认 1 ms）运行：

    * SCAN 模式 —— 每个时隙扫描一个信道，用能量检测判决该信道是否存在信号：
      - H0（只有噪声）：  V ~ χ²(2γ)        （中心 chi-square）
      - H1（存在信号）：  V ~ χ²(2γ, λ)     （非中心，λ = 2·SNR·γ）
      其中 γ = T·W 为时间-带宽积（T = detection_time，W = sub_interval）。
      门限 V_T 由 p_fa 反解，P_D 对 Rayleigh 衰落平均（见 _fading_averaged_P_D）。

    * JAM 模式 —— 对刚检测到的信道压制一个时隙，随后在下一个时隙对**同一信道**
      重新扫描：若目标仍在该信道就能连续压制；若目标已跳走，则本次压制浪费在
      空信道上，且下一次判决只有 p_fa 的误判概率能把干扰"追回来"。

    信道扫描顺序：0 → 1 → … → (num_channels-1) → 0 → …（循环）。

    压制信号按"信道 × 基带变体"预生成一次并缓存，运行期只做查表与拷贝；
    变体在每次进入 JAM 时现场抽取（有放回）。

    时序取舍：每个时隙只在起点做一次检测，且压制时隙内不再检测，因此该干扰机
    能跟上慢跳频，但跟不上快跳频——当每跳驻留时间接近或短于检测时隙时，
    _get_tx_channel 的"时隙起点采样"已不能代表整段时隙的信道，压制会大量落在
    错误信道上（这是本环境要考察的抗干扰/反侦察现象，不是实现缺陷）。

    关键属性：
        Fs (float): 采样率 (Hz)。
        num_channels (int): 信道数。
        sub_interval (float): 信道间隔 (Hz)，同时用作能量检测带宽 W。
        detection_time (float): 检测/压制时隙时长 T (s)。
        samples_per_ms (int): 每个时隙的采样点数 = int(Fs · detection_time)；
            名字沿用历史（默认 detection_time=1 ms 时名副其实），detection_time
            取其他值（如 0.5 ms）时它表示"每检测时隙"的采样点数。
        TW (float): 时间-带宽积 γ = T·W。
        dof (float): chi2 分布的自由度 2γ。
        V_T (float): 能量检测门限（中心 chi2 的 1-p_fa 分位数）。
        P_D (float): Rayleigh 衰落平均后的检测概率。
        snr_avg_linear / snr_avg_dB: 衰落前的平均 SNR（线性/分贝）。
        power (float): 压制波形幅度系数，作用在归一化基带上。
        _jam_cache (dict): {信道号: float32 数组 (num_variants, samples_per_ms)}，
            已乘功率与载波。
        current_channel (int): 当前扫描/压制信道。
        state (str): 'scan' 或 'jam'。
    """

    def __init__(self, Fs, num_channels=20, sub_interval=50000.0,
                 detection_time=0.001, p_fa=0.1,
                 power=0.8, bandwidth=50000.0, Startfre=3e6,
                 noise_source=None, speed=None,
                 noise_std=0.1, signal_power=None, Baud=25000,
                 variant_pool=None, variant_selector=None):
        """构造干扰机，推导检测门限/P_D，并预生成各信道的压制波形。

        构造顺序：参数归一化 → 时间-带宽积与自由度 → 干扰机处 SNR 估计 →
        门限 V_T（由 p_fa 反解）→ Rayleigh 衰落平均的 P_D（Gauss-Laguerre 积分）
        → 选择噪声来源并预生成压制波形 → 状态机复位。

        Args:
            Fs: 采样率 (Hz)。
            num_channels: 信道数（也是扫描循环的长度）。
            sub_interval: 信道间隔 (Hz)，兼作检测带宽 W 与载波中心频率的偏移步长。
            detection_time: 检测/压制时隙时长 T (s)，默认 1 ms。
            p_fa: 能量检测虚警概率，用于反解门限 V_T（越大越激进、越容易误判）。
            power: 压制波形幅度系数。
            bandwidth: 压制噪声的单边带宽 (Hz)，提供 variant_pool 时两者必须一致。
            Startfre: 频段起始频率 (Hz)；第 k 个信道中心为
                Startfre + k·sub_interval + 0.5·sub_interval。
            noise_source: 动态噪声源（FastNoiseSource 或兼容对象）；
                仅在未提供 variant_pool 时使用。
            speed: 仅为向后兼容旧 API 保留，新逻辑不使用（None 记为 +inf）。
            noise_std: 接收端噪声标准差 σ，用于推导干扰机处的 SNR 与门限，
                与 fh_env 的噪声保持一致。
            signal_power: 单采样点信号功率（衰落前）；None 时用理论值 Baud/Fs。
            Baud: 符号率 (Hz)，仅在 signal_power 为 None 时参与信号功率推导。
            variant_pool: 共享带限噪声变体池；提供时压制波形从池中取变体。
            variant_selector: 共享变体选择器；每次进入 JAM 时抽一个变体下标。

        Raises:
            ValueError: variant_pool 的带宽与本机 bandwidth 不一致。
        """
        # ---- 基础参数 ----
        self.Fs = float(Fs)
        self.num_channels = int(num_channels)
        self.sub_interval = float(sub_interval)
        self.detection_time = float(detection_time)
        self.p_fa = float(p_fa)
        self.power = float(power)
        self.bandwidth = float(bandwidth)
        self.Startfre = float(Startfre)
        self.noise_std = float(noise_std)
        # speed 仅为向后兼容旧 API 保留，新逻辑不再使用（None → +inf，
        # 语义上表示"无扫描速度上限"，不影响任何计算路径）。
        self.speed = float(speed) if speed is not None else float('inf')

        # ---- 能量检测：时间-带宽积与自由度 ----
        # γ = T·W：T 为观测时长 (s)，W 为检测带宽 (Hz)。注意这里 W 取信道间隔
        # sub_interval 而非 Fs/2，与"每个 50 kHz 信道单独做能量检测"的物理设定一致。
        self.TW = self.detection_time * self.sub_interval          # γ = T·W
        self.dof = 2.0 * self.TW                                   # 2γ

        # ---- 信号功率（实 RF、单采样点、衰落前） ----
        if signal_power is not None:
            self.signal_power = float(signal_power)
        else:
            # 理论默认值：符号率与采样率之比，即单位功率 QPSK 信号在
            # Fs 采样下的每采样点平均功率量级。
            self.signal_power = float(Baud) / self.Fs

        # ---- 检测带宽 W = Sub_interval 内的噪声功率 ----
        # N0（单边 PSD）  = 2 · σ² / Fs：实高斯噪声功率 σ² 摊到 [0, Fs/2] 上的谱密度
        # P_n = N0 · W = 2 · σ² · Sub_interval / Fs
        noise_power_in_band = 2.0 * self.noise_std**2 * self.sub_interval / self.Fs

        # ---- 平均 SNR（衰落前） ----
        # 下限 1e-30 仅用于避免 log10(0) / 除零，不改变正常取值。
        self.snr_avg_linear = self.signal_power / max(noise_power_in_band, 1e-30)
        self.snr_avg_dB = 10.0 * np.log10(max(self.snr_avg_linear, 1e-30))

        # ---- 由虚警概率反解门限 V_T（H0 下为中心 χ²） ----
        # P(V > V_T | H0) = p_fa  ⇒  V_T = F⁻¹_{χ²(2γ)}(1 - p_fa)
        self.V_T = chi2.ppf(1.0 - self.p_fa, self.dof)

        # ---- 检测概率 P_D（对 Rayleigh 衰落平均） ----
        # 瞬时 SNR：γ = SNR_avg · r²，其中 r² ~ Exp(1)
        # P_D_faded = ∫₀^∞ [1 − F_{ncχ²}(V_T | dof, 2·SNR_avg·x·TW)] · e⁻ˣ dx
        # 用 Gauss–Laguerre 求积计算（见 _fading_averaged_P_D）。
        self.P_D = self._fading_averaged_P_D(n_laguerre=50)

        # ---- 噪声来源与预生成的压制信号 ----
        # 优先级：共享变体池 > 外部 noise_source > 自建 FastNoiseSource。
        # 池模式与动态模式互斥（池存在时置 noise_source=None），避免两条随机
        # 来源同时存在导致波形不可复现。
        self.variant_pool = variant_pool
        self.variant_selector = variant_selector
        if self.variant_pool is not None:
            # 带宽不匹配说明池与干扰参数错配（会得到错误的干扰带宽），直接报错。
            if not np.isclose(self.variant_pool.bandwidth, self.bandwidth):
                raise ValueError("Reactive jammer variant-pool bandwidth does not match.")
            self.noise_source = None
        elif noise_source is None:
            self.noise_source = FastNoiseSource(Fs, bandwidth)
        else:
            self.noise_source = noise_source

        # 时隙采样点数：max(1, ...) 保证极小 Fs/detection_time 下仍有一个采样点，
        # 避免后续整除与切片出现零长度时隙。
        self.samples_per_ms = max(1, int(self.Fs * self.detection_time))
        self._pregenerate_jam_signals()

        # ---- 状态机 ----
        self.reset()

    # ------------------------------------------------------------------
    def _fading_averaged_P_D(self, n_laguerre=50):
        """
        计算对 Rayleigh 衰落平均后的检测概率 P_D。

        衰落模型：Rayleigh 幅度 r 归一化为 E[r²] = 1（等价于 scale = √2/2），
        故瞬时功率增益 x := r² ~ Exp(1)（概率密度 e⁻ˣ），瞬时 SNR = SNR_avg · x。

        对瞬时 SNR 求期望即得衰落平均检测概率：
            P_D_faded = ∫₀^∞ P_D(x · SNR_avg) · e⁻ˣ dx
        其中单点检测概率由非中心 chi2 的互补累积分布给出：
            P_D(λ) = 1 − F_{ncχ²}(V_T | dof, λ)，  λ = 2 · SNR_inst · TW

        用 Gauss–Laguerre 数值积分求该期望（被积函数天然带 e⁻ˣ 权重）：
            ∫₀^∞ f(x)·e⁻ˣ dx ≈ Σ w_i · f(x_i)
        n 点 Gauss–Laguerre 对形如 x^α·e⁻ˣ·(多项式/指数衰减) 的被积函数精度极高，
        这里 50 点足以把积分误差压到远小于 1e-6，且只需在构造时算一次。

        Args:
            n_laguerre: Gauss–Laguerre 节点数。节点越多尾部积分越准，但每次都要
                调用 scipy 求根；默认 50 是精度与一次性开销之间的折中。

        Returns:
            float: 截断到 [0, 1] 的衰落平均检测概率
            （数值误差可能给出极小负值或略大于 1 的值，故显式 clip）。
        """
        nodes, weights = roots_laguerre(n_laguerre)
        p_d = 0.0
        for x_i, w_i in zip(nodes, weights):
            snr_inst = self.snr_avg_linear * x_i          # 该积分节点处的瞬时 SNR
            lam = 2.0 * snr_inst * self.TW                # ncx2 非中心参量 λ = 2·SNR·γ
            p_d_inst = 1.0 - ncx2.cdf(self.V_T, self.dof, lam)
            p_d += w_i * p_d_inst                         # 按 Laguerre 权重加权求和
        return float(np.clip(p_d, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _pregenerate_jam_signals(self):
        """
        为每个"信道 × 基带变体"预生成一段压制波形。

        压制信号 = 带限噪声基带 × cos(2π·f_c·t)：载波只由信道中心频率决定、与变体
        无关，因此先把基带整理成 (num_variants, samples_per_ms) 矩阵，再与一维载波
        做广播外积，一次性得到整张缓存表（避免 num_channels × num_variants 次循环
        内的重复乘法）。

        缓存形状：{信道号: float32 数组 (num_variants, samples_per_ms)}。
        基带来源：有变体池时按变体下标取一条时隙长度的连续切片（不足则按周期环绕），
        无池时退回动态噪声源，此时只有变体 0（num_variants=1）。

        预生成的动机：运行期每个压制时隙都要输出一段波形，若现场合成则每次调用都
        需要噪声切片 + 载波乘法；预生成把代价前移到构造阶段，使训练内循环只剩
        数组拷贝。
        """
        self._jam_cache = {}
        # 时隙内时间轴 (s)，用 float64 避免 cos 参数在 f_c 达 MHz 量级时损失精度。
        t_1ms = np.arange(self.samples_per_ms, dtype=np.float64) / self.Fs
        num_variants = (
            self.variant_pool.num_variants
            if self.variant_pool is not None
            else 1
        )
        baseband_variants = np.empty(
            (num_variants, self.samples_per_ms),
            dtype=np.float32,
        )
        for variant_idx in range(num_variants):
            if self.variant_pool is not None:
                noise_1ms = self.variant_pool.get_variant(
                    variant_idx,
                    self.samples_per_ms,
                )
            else:
                noise_1ms = self.noise_source.get_noise(self.samples_per_ms)
            # 功率在预生成阶段就乘进基带，运行期不再缩放。
            baseband_variants[variant_idx] = (
                np.asarray(noise_1ms, dtype=np.float32) * self.power
            )

        for k in range(self.num_channels):
            # 第 k 个信道的中心频率 (Hz)：起始频率 + k 个信道间隔 + 半个信道间隔。
            f_c = self.Startfre + k * self.sub_interval + 0.5 * self.sub_interval
            carrier = np.cos(2.0 * np.pi * f_c * t_1ms)
            # 广播相乘：每条变体各乘同一载波，得到该信道的全部变体波形。
            self._jam_cache[k] = (
                baseband_variants * carrier[np.newaxis, :]
            ).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        """重置状态机：从信道 0 开始扫描。"""
        self.current_channel = 0
        self.state = 'scan'          # 'scan' | 'jam'

    # ------------------------------------------------------------------
    def _get_tx_channel(self, sample_pos, hop_seq, hoprate):
        """
        返回给定采样位置处发射机所处的信道号。

        位置→跳的映射：每跳持续 Fs/hoprate 个采样点，故
        hop_idx = floor(sample_pos / (Fs/hoprate))，再查 hop_seq[hop_idx]。

        约定与边界：
          - hop_seq 是"每跳一个信道号"的序列，且假定第 0 跳从 sample_pos=0 对齐
            （fh_env.py 每个 block 重新生成本 block 的 hop_seq，并以块内相对位置
            调用本函数，与 hop_carrier 的取法一致）；
          - 超出序列末尾时按最后一跳的信道保持，避免索引越界；
          - 空序列返回 -1：-1 不等于任何被扫描信道，等价于"当前信道上无信号"。

        Args:
            sample_pos: 块内采样位置（非全局采样时钟）。
            hop_seq: 每跳信道号序列。
            hoprate: 跳速 (hops/s)。

        Returns:
            int: 信道号；hop_seq 为空时返回 -1。
        """
        if len(hop_seq) == 0:
            return -1
        s_per_hop = self.Fs / float(hoprate)
        if s_per_hop <= 0.0:
            return int(hop_seq[0])
        hop_idx = int(sample_pos / s_per_hop)
        hop_idx = min(hop_idx, len(hop_seq) - 1)
        return int(hop_seq[hop_idx])

    # ------------------------------------------------------------------
    def _energy_detect(self, signal_present, rng=None):
        """
        模拟一次能量检测判决（伯努利抽样）。

        检测统计量在 H0/H1 下的完整分布已被门限 V_T 与概率对 (p_fa, P_D) 概括，
        因此这里不再重建 chi2 统计量，而是直接按对应概率抽一次伯努利：
        信号存在时以 P_D 判为"检测到"，不存在时以 p_fa 误判为"检测到"。
        这样既保持与理论一致，又让每次判决只消耗一个均匀随机数（性能友好）。

        Args:
            signal_present: 发射机当前是否在被扫描的信道上。
            rng: 随机数生成器；None 表示使用全局 np.random。

        Returns:
            bool: True 表示检测器报告"检测到信号"。
        """
        p = self.P_D if signal_present else self.p_fa
        return _rng_random(rng) < p

    # ------------------------------------------------------------------
    def generate_samples(
        self,
        num_samples,
        hop_seq,
        Startfre,
        Sub_interval,
        hoprate,
        rng=None,
    ):
        """
        生成一个 block 的反应式干扰波形（时隙状态机）。

        block 被切分为长度 samples_per_ms 的时隙，每个时隙由内部状态机决定动作：
          - 'scan'：对 current_channel 做一次能量检测；判到信号 → 下一时隙转入
            'jam'（压制同一信道）；未判到 → current_channel 前进一格继续扫描
            （按 num_channels 循环回绕）；
          - 'jam'：把 current_channel 的预生成波形写入本时隙，随后回到 'scan'
            并停留在同一信道重新检测（这就是"反应式"的核心：追着刚检测到的
            信道打，而不是继续轮询）。

        状态机跨调用保持（不随 block 重置），因此跨 block 的扫描/压制时序连续；
        时隙起点才采样发射机信道，时隙内假定信道不变（跳速远快于时隙时是近似，
        也是该干扰机"跟不上快跳频"的建模来源）。

        Args:
            num_samples: 本 block 采样点数。
            hop_seq: 本 block 每跳的信道号序列，用于判定被扫描信道是否有信号。
            Startfre: 频段起始频率 (Hz)，保留参数，当前逻辑不使用。
            Sub_interval: 信道间隔 (Hz)，保留参数，当前逻辑不使用。
            hoprate: 跳速 (hops/s)。
            rng: 随机数生成器，同时用于能量检测判决与压制变体抽取。

        Returns:
            jam: ndarray，float32，长度 num_samples；未被压制的采样点为 0。
            active: bool，本 block 内是否至少压制了一个时隙
                （供统计干扰占用率/渲染判决使用）。
        """
        N = int(num_samples)
        if N == 0:
            return np.zeros(0, dtype=np.float32), False

        jam = np.zeros(N, dtype=np.float32)
        any_jam = False

        pos = 0
        # 逐时隙推进；时隙在 block 末尾可能被截断（seg_len < samples_per_ms），
        # 此时状态机仍按完整时隙推进：波形被截断，时序不做补偿（下一 block 从
        # 新时隙继续），保证状态机与"1 ms 时隙网格"始终对齐。
        while pos < N:
            seg_end = min(pos + self.samples_per_ms, N)
            seg_len = seg_end - pos

            # --- 本时隙内发射机所在信道（只在时隙起点采样一次） ---
            tx_channel = self._get_tx_channel(pos, hop_seq, hoprate)

            if self.state == 'scan':
                # 能量检测：只有发射机恰好在本扫描信道上时才处于 H1，
                # 否则是 H0（此时仍可能以 p_fa 误判为"检测到"）。
                signal_present = (tx_channel == self.current_channel)
                if self._energy_detect(signal_present, rng=rng):
                    # 检测到 → 下一时隙压制同一信道
                    self.state = 'jam'
                else:
                    # 未检测到 → 扫描信道前进一格，继续 SCAN
                    self.current_channel = (self.current_channel + 1) % self.num_channels

            elif self.state == 'jam':
                # 输出当前信道的预生成压制波形
                ch = self.current_channel % self.num_channels
                num_variants = self._jam_cache[ch].shape[0]
                # 每次压制独立抽变体（有放回）；无池或单变体时固定用下标 0。
                if self.variant_selector is not None and num_variants > 1:
                    variant_idx = self.variant_selector.draw()
                else:
                    variant_idx = 0
                cached = self._jam_cache[ch][variant_idx]
                if seg_len >= len(cached):
                    jam[pos:pos + len(cached)] = cached
                else:
                    jam[pos:seg_end] = cached[:seg_len]
                any_jam = True
                # 压制结束 → 回到同一信道重新扫描（继续追踪该信道）
                self.state = 'scan'

            pos = seg_end

        return jam, any_jam

    def generate(self, t, hop_seq, Startfre, Sub_interval, hoprate, rng=None):
        """兼容旧接口：接受历史的时间向量 t，内部按 len(t) 转调 generate_samples。"""
        return self.generate_samples(
            len(t),
            hop_seq,
            Startfre,
            Sub_interval,
            hoprate,
            rng=rng,
        )


class IndiscriminateJammer:
    """无差别（非反应式）干扰机：不感知发射机，按固定 RF 时钟周期性压制频段。

    三种模式（mode）：
      - 'sweep'：扫频。以 s_step 为步进逐个频点压制，每个频点驻留 s_dwell 秒，
        扫完 [Startfre, Endfre) 后回卷，形成固定周期的扫频图案；
      - 'comb'：梳状。两组信道（channels_phase0 / channels_phase1）交替压制，
        每组在 comb_switch_interval 内同时输出多个 tone（中心频率
        Startfre + k·50 kHz + 25 kHz），两组交替构成 2×switch_interval 的完整周期；
      - 'both'：两种信号在时域直接相加（功率会叠加，配置时需自行留足余量）。

    两条生成路径共享同一份确定性载波缓存（_ensure_carrier_cache）：
      - 加速路径：precompute() 预生成"一个自然周期"的波形，运行期由
        get_composite_signal(start_sample_idx, num_samples) 按全局采样时钟取切片，
        不做任何 FFT 与随机抽样；
      - 动态路径：generate_samples() 用缓存载波 + 现场抽取的基带噪声合成，
        适合 use_pregen=False 或需要每 block 新鲜噪声的场景。

    时序约定：所有相位/频点位置都由"全局 RF 采样时钟"（start_sample_idx，即
    fh_env 的 jammer_ptr）推导，而不是由调用次数推导，因此可以任意起点、任意长度
    调用而结果仍与连续时间轴一致——这是分 block 并行仿真下保持一致性的前提。
    干扰机自身不保存时间状态（reset() 为空操作）。

    关键属性：
        Fs (float): 采样率 (Hz)。
        mode (str): 'sweep' / 'comb' / 'both'。
        s_step/s_dwell/s_power/s_bw: 扫频步进 (Hz)、驻留时间 (s)、功率、噪声带宽 (Hz)。
        c_step/c_power/c_bw: 梳状配置项（注意 c_step 目前不参与频率计算）。
        comb_switch_interval (float): 梳状相位切换周期 (s)，必须是 1 ms 的整数倍。
        comb_switch_samples (int): 单个相位的采样点数 = round(interval·Fs)。
        comb_period_samples (int): 梳状完整周期采样点数 = 2×comb_switch_samples。
        comb_channels (tuple): (phase0 信道组, phase1 信道组)。
        sweep_period_samples (int): 一个完整扫频周期的采样点数（precompute 后有效）。
        pre_buffer_sweep / pre_buffer_comb0 / pre_buffer_comb1: 预生成波形，
            形状 (num_variants, 周期采样点数)，每行对应一个基带变体。
        _carrier_sweep / _carrier_comb0 / _carrier_comb1: 确定性载波缓存（float32），
            只由 RF 布局决定，动态路径与预生成路径共用。
    """
    # comb 相位切换的最小时间量子 (s)：switch_interval 必须是它的正整数倍，
    # 以保证相位切换点落在 1 ms 基本时隙边界上（见 _validate_comb_switch_interval）。
    COMB_TIME_QUANTUM = 0.001

    def __init__(self, Fs, sweep_config=None, comb_config=None, 
                 noise_source=None, mode='sweep',
                 sweep_variant_pool=None, comb_variant_pool=None,
                 variant_selector=None, defer_dynamic_noise=False):
        """构造无差别干扰机并缓存配置参数。

        Args:
            Fs: 采样率 (Hz)。
            sweep_config: 扫频配置 dict，键 step/power/dwell_time/bandwidth；
                缺省（None 或空 dict）时使用内置默认值。
            comb_config: 梳状配置 dict，键 step/power/bandwidth/switch_interval/
                channels_phase0/channels_phase1。
            noise_source: 外部动态噪声源；提供时 sweep 与 comb 共用同一个源。
            mode: 'sweep' / 'comb' / 'both'。
            sweep_variant_pool: 扫频用的共享基带变体池（带宽须等于 s_bw）。
            comb_variant_pool: 梳状用的共享基带变体池（带宽须等于 c_bw）。
            variant_selector: 共享变体选择器，其 num_variants 须与池一致。
            defer_dynamic_noise: True 时不立即构造动态噪声源（预生成路径下用不到），
                推迟到 _dynamic_noise_source 首次被调用时再建，省掉一份 FFT 预生成开销。

        Raises:
            ValueError: mode 非法、comb switch_interval 非法，或变体池/选择器不匹配。
        """
        self.Fs = float(Fs)
        if mode not in {'sweep', 'comb', 'both'}:
            raise ValueError("mode must be 'sweep', 'comb', or 'both'.")
        self.mode = mode
        
        # 默认配置：空/缺省时退化为内置默认值（不修改传入的 dict，避免调用方配置被写坏）
        self.sweep_config = sweep_config if sweep_config else {}
        self.comb_config = comb_config if comb_config else {}
        
        # --- 扫频参数 ---
        # s_step: 频点步进 (Hz)；s_dwell: 每个频点的驻留时间 (s)；
        # s_bw: 压制噪声的单边带宽 (Hz)，须与 sweep_variant_pool 的带宽一致。
        self.s_step = float(self.sweep_config.get('step', 125000.0))
        self.s_power = float(self.sweep_config.get('power', 0.8))
        self.s_dwell = float(self.sweep_config.get('dwell_time', 0.004))
        self.s_bw = float(self.sweep_config.get('bandwidth', 30000.0))
        
        # --- 梳状参数 ---
        # c_step 目前不参与频率合成（_comb_frequencies 固定按 50 kHz 信道间隔取
        # tone 中心），保留该字段只为与 JAMMER_CONFIG 的键保持一一对应。
        self.c_step = float(self.comb_config.get('step', 100000.0))
        self.c_power = float(self.comb_config.get('power', 0.5))
        self.c_bw = float(self.comb_config.get('bandwidth', 30000.0))
        self.comb_switch_interval = float(
            self.comb_config.get('switch_interval', 0.05)
        )
        self._validate_comb_switch_interval()
        # 相位持续采样点数：round 把秒级配置换算到采样网格；因为 switch_interval
        # 已被约束为 1 ms 的整数倍，在整数 Fs 下 round 基本不会引入误差。
        self.comb_switch_samples = int(round(self.comb_switch_interval * self.Fs))
        self.comb_period_samples = 2 * self.comb_switch_samples

        # 两个交替梳状组所压制的信道序号。
        # 可通过 JAMMER_CONFIG["comb"]["channels_phase0/1"] 配置；
        # 默认值保持历史上的偶/奇 8 信道分组。序号 k 对应第 k 个 50 kHz 信道中心
        # （Startfre + k·50 kHz + 25 kHz）；两组长度可以不同、允许重叠。
        self.comb_channels = (
            list(self.comb_config.get('channels_phase0',
                                      [0, 2, 4, 6, 8, 10, 12, 14])),
            list(self.comb_config.get('channels_phase1',
                                      [1, 3, 5, 7, 9, 11, 13, 15])),
        )
        
        self.sweep_variant_pool = sweep_variant_pool
        self.comb_variant_pool = comb_variant_pool
        self.variant_selector = variant_selector

        # --- 动态噪声源 ---
        # 三种来源优先级：外部 noise_source > defer_dynamic_noise（延迟到首次使用）
        # > 现场自建。带宽相同时 sweep/comb 复用同一个 FastNoiseSource，
        # 避免重复预生成一段带限噪声缓冲。
        if noise_source:
             self.ns_sweep = noise_source
             self.ns_comb = noise_source
        elif defer_dynamic_noise:
             self.ns_sweep = None
             self.ns_comb = None
        else:
             self.ns_sweep = FastNoiseSource(Fs, self.s_bw)
             # 带宽相同则复用同一个噪声源，省掉一份重复的带限噪声缓冲。
             if self.c_bw == self.s_bw:
                 self.ns_comb = self.ns_sweep
             else:
                 self.ns_comb = FastNoiseSource(Fs, self.c_bw)

        # 预生成缓冲（precompute 前均为 None，取用时会报错而不是静默返回零波形）
        self.pre_buffer_sweep = None
        self.pre_buffer_comb0 = None
        self.pre_buffer_comb1 = None
        self.sweep_period_samples = 0

        # 确定性 RF 载波与动态路径的"新鲜"基带噪声相互独立。
        # 这里缓存一个自然时序周期的载波，预生成与运行期生成共用同一份，
        # 保证两条路径的相位结构完全一致。
        self._carrier_cache_key = None
        self._carrier_sweep = None
        self._carrier_comb0 = None
        self._carrier_comb1 = None

        self._validate_variant_pools()

    def set_mode(self, mode):
        """切换工作模式（'sweep' / 'comb' / 'both'）。

        只改模式标记：_carrier_key 把 mode 纳入指纹，因此下一次生成时会自动
        重建载波缓存，不会残留旧模式下的波形。

        Raises:
            ValueError: mode 非法。
        """
        if mode not in {'sweep', 'comb', 'both'}:
            raise ValueError("mode must be 'sweep', 'comb', or 'both'.")
        self.mode = mode

    def _carrier_key(self, Startfre, Endfre):
        """返回决定载波波形的 RF 布局指纹，用于判断缓存是否仍然有效。

        指纹包含 mode（决定哪些载波会被用到）、频段范围、扫频步进与驻留时间、
        梳状切换点数以及两组信道：任一项变化都会改变载波序列，必须重建。

        Returns:
            tuple: 可哈希的布局指纹（信道组转 tuple 以便哈希）。
        """
        return (
            self.mode,
            float(Startfre),
            float(Endfre),
            self.s_step,
            self.s_dwell,
            self.comb_switch_samples,
            tuple(self.comb_channels[0]),
            tuple(self.comb_channels[1]),
        )

    def _ensure_carrier_cache(self, Startfre, Endfre):
        """按 RF 布局构建/复用确定性的 sweep 与 comb 载波。

        载波只由频率与采样率决定（与随机基带无关），因此可跨 block、跨动态/
        预生成两条路径共享；指纹命中时直接返回，不做任何计算。fh_env 在首次
        step 前也会显式调用一次，避免多 worker 首次并发触发懒加载。

        生成的载波（均为 float32）：
          - _carrier_sweep: 一个完整扫频周期的余弦序列，长度
            samples_per_dwell × num_steps；每个 dwell 段都用 n = 0… 重新计数的
            cos(2π·f·n/Fs)，即 dwell 之间相位不连续——等价于每个频点独立起振，
            这样才能把整周期表示成一段可拼接的连续数组；
          - _carrier_comb0/_carrier_comb1: 一个相位区间（comb_switch_samples）
            的多 tone 叠加，并乘以 1/√len(freqs)，使不同信道组长度下输出功率
            大致可比（功率只随 tone 数变化，与具体频率无关）。
        """
        key = self._carrier_key(Startfre, Endfre)
        if key == self._carrier_cache_key:
            return

        self._carrier_sweep = None
        self._carrier_comb0 = None
        self._carrier_comb1 = None
        # 相位步进因子：cos(2π·f·n/Fs) = cos((2π/Fs · f) · n)，把 2π/Fs 提到循环外，
        # 每个采样点只需一次乘法。
        phase_k = 2.0 * np.pi / self.Fs

        if self.sweep_enabled:
            bw_total, samples_per_dwell, num_steps = self._sweep_layout(
                Startfre,
                Endfre,
            )
            sweep_carrier = np.empty(
                samples_per_dwell * num_steps,
                dtype=np.float32,
            )
            # 每个 dwell 复用同一段 n = 0…samples_per_dwell-1 的时间轴，
            # 因此各 dwell 的载波相位都从 0 开始（相位在 dwell 边界处不连续）。
            dwell_samples = np.arange(samples_per_dwell, dtype=np.float64)
            for sweep_idx in range(num_steps):
                # 取信道中心频率：起始频率 + 步进序号×步进 + 半步进
                f = Startfre + sweep_idx * self.s_step + self.s_step / 2.0
                if f >= Endfre:
                    # 回卷保护：num_steps = floor(bw_total/step) 时正常不会触发，
                    # 这里仅防御浮点误差导致的越界频点。
                    f = Startfre + np.mod(f - Startfre, bw_total)
                start = sweep_idx * samples_per_dwell
                sweep_carrier[start:start + samples_per_dwell] = np.cos(
                    (phase_k * f) * dwell_samples
                ).astype(np.float32)
            self._carrier_sweep = sweep_carrier

        if self.comb_enabled:
            phase_samples = np.arange(
                self.comb_switch_samples,
                dtype=np.float64,
            )
            comb_carriers = []
            for phase in (0, 1):
                freqs = self._comb_frequencies(phase, Startfre, Endfre)
                carrier = np.zeros(
                    self.comb_switch_samples,
                    dtype=np.float64,
                )
                # 多 tone 直接线性叠加（相位都从 0 起，故叠加是相干求和）；
                # 再按 tone 数做 1/√N 归一化，使不同信道组长度下功率量级一致。
                for f in freqs:
                    carrier += np.cos((phase_k * f) * phase_samples)
                if len(freqs) > 0:
                    carrier *= 1.0 / np.sqrt(len(freqs))
                comb_carriers.append(carrier.astype(np.float32))
            self._carrier_comb0, self._carrier_comb1 = comb_carriers

        self._carrier_cache_key = key

    def _validate_variant_pools(self):
        """校验变体池带宽、池间变体数一致性，以及选择器与池的匹配关系。

        约束来源：变体下标由共享选择器统一抽取，若 sweep/comb 池大小不同，同一
        下标在两侧的含义就不一致；因此要求两侧池大小相同且等于选择器的
        num_variants。带宽不匹配则说明池与干扰参数错配（会得到错误的干扰带宽），
        直接报错而不是静默降级。

        Raises:
            ValueError: 池带宽不匹配、池大小不一致，或选择器与池数量不一致。
        """
        if (
            self.sweep_variant_pool is not None
            and not np.isclose(self.sweep_variant_pool.bandwidth, self.s_bw)
        ):
            raise ValueError("Sweep jammer variant-pool bandwidth does not match.")
        if (
            self.comb_variant_pool is not None
            and not np.isclose(self.comb_variant_pool.bandwidth, self.c_bw)
        ):
            raise ValueError("Comb jammer variant-pool bandwidth does not match.")

        counts = {
            pool.num_variants
            for pool in (self.sweep_variant_pool, self.comb_variant_pool)
            if pool is not None
        }
        if len(counts) > 1:
            raise ValueError("Sweep and comb variant pools must use the same count.")
        if counts and self.variant_selector is not None:
            count = next(iter(counts))
            if self.variant_selector.num_variants != count:
                raise ValueError("Jammer selector and baseband pool counts do not match.")

    def set_variant_sources(
        self,
        sweep_variant_pool=None,
        comb_variant_pool=None,
        variant_selector=None,
    ):
        """替换变体池/选择器并立即重新校验（供环境在构造后注入资源）。

        Raises:
            ValueError: 同 _validate_variant_pools。
        """
        self.sweep_variant_pool = sweep_variant_pool
        self.comb_variant_pool = comb_variant_pool
        self.variant_selector = variant_selector
        self._validate_variant_pools()

    def _dynamic_noise_source(self, jammer_kind):
        """惰性取得（必要时构造）指定干扰的动态噪声源。

        - defer_dynamic_noise=True 时构造期不建源，这里首次访问才创建；
        - sweep 与 comb 带宽相同时复用同一个 FastNoiseSource，避免重复预生成
          一段带限噪声缓冲（两份缓冲既占内存又浪费一次 FFT）。

        Args:
            jammer_kind: "sweep" 或 "comb"。

        Returns:
            FastNoiseSource: 对应干扰的噪声源。

        Raises:
            ValueError: jammer_kind 不是 "sweep"/"comb"。
        """
        if jammer_kind == "sweep":
            if self.ns_sweep is None:
                self.ns_sweep = FastNoiseSource(self.Fs, self.s_bw)
                if self.c_bw == self.s_bw and self.ns_comb is None:
                    self.ns_comb = self.ns_sweep
            return self.ns_sweep
        if jammer_kind == "comb":
            if self.ns_comb is None:
                if self.c_bw == self.s_bw and self.ns_sweep is not None:
                    self.ns_comb = self.ns_sweep
                else:
                    self.ns_comb = FastNoiseSource(self.Fs, self.c_bw)
            return self.ns_comb
        raise ValueError(f"Unsupported jammer kind: {jammer_kind}")

    def _validate_comb_switch_interval(self):
        """校验 comb 相位切换周期是 1 ms 的正整数倍。

        为什么必须量化到 1 ms：切换时刻决定"哪些采样点被哪个信道组压制"。若切换点
        落在 1 ms 基本时隙内部，同一时隙会横跨两个相位，压制信道组在中途改变，
        从而破坏与 reactive 干扰的时隙对齐关系，并使离线 replay 难以按 1 ms 网格
        复现。这里用 interval / COMB_TIME_QUANTUM 是否为整数判断（atol=1e-9 容忍
        浮点除法误差），同时拒绝 NaN/inf 与非正值。

        Raises:
            ValueError: 非有限值、非正值，或不是 1 ms 的整数倍。
        """
        interval = self.comb_switch_interval
        interval_units = interval / self.COMB_TIME_QUANTUM
        if (
            not np.isfinite(interval)
            or interval <= 0.0
            or not np.isclose(
                interval_units,
                round(interval_units),
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "JAMMER_CONFIG['comb']['switch_interval'] must be a finite "
                "positive multiple of 0.001 seconds (1 ms)."
            )

    def _sweep_layout(self, Startfre, Endfre):
        """把频段范围换算成扫频的"采样点级布局"。

        计算规则：
          - bw_total = max(Endfre - Startfre, 1.0)：频段宽度 (Hz)，下限 1.0 用于
            防止退化配置（Endfre <= Startfre）导致后续除零/取模出错；
          - samples_per_dwell = max(1, round(s_dwell · Fs))：每频点驻留采样点数；
          - num_steps = max(1, floor(bw_total / s_step))：完整频点个数。
            注意 floor 会丢掉不足一个 step 的尾部频段，因此"扫频周期"通常略小于
            频段宽度——这是刻意的（保证每个频点都完整落在频段内）。

        Args:
            Startfre: 频段起始频率 (Hz)。
            Endfre: 频段结束频率 (Hz)。

        Returns:
            tuple: (bw_total, samples_per_dwell, num_steps)。

        Raises:
            ValueError: s_step 或 s_dwell 非有限/非正。
        """
        if not np.isfinite(self.s_step) or self.s_step <= 0.0:
            raise ValueError("Sweep step must be finite and positive.")
        if not np.isfinite(self.s_dwell) or self.s_dwell <= 0.0:
            raise ValueError("Sweep dwell_time must be finite and positive.")

        bw_total = max(float(Endfre) - float(Startfre), 1.0)
        samples_per_dwell = max(1, int(round(self.s_dwell * self.Fs)))
        num_steps = max(1, int(np.floor(bw_total / self.s_step)))
        return bw_total, samples_per_dwell, num_steps

    @property
    def comb_enabled(self):
        """当前 mode 是否启用梳状干扰（'comb' 或 'both'）。"""
        return self.mode in {'comb', 'both'}

    @property
    def sweep_enabled(self):
        """当前 mode 是否启用扫频干扰（'sweep' 或 'both'）。"""
        return self.mode in {'sweep', 'both'}

    def comb_phase_at(self, sample_idx):
        """返回全局采样下标处的梳状相位（0 或 1）。

        相位完全由全局采样时钟决定：每 comb_switch_samples 个采样点翻转一次。
        因为是无状态的纯函数，任何调用者（含并行 worker、离线数据生成脚本）
        只要传入同一个 start_sample_idx，算出的相位就完全一致。

        Args:
            sample_idx: 全局 RF 采样下标（>=0）。

        Returns:
            int: 0 或 1。

        Raises:
            ValueError: sample_idx 为负（负下标会与 Python 倒序索引语义混淆）。
        """
        sample_idx = int(sample_idx)
        if sample_idx < 0:
            raise ValueError("sample_idx must be non-negative.")
        return (sample_idx // self.comb_switch_samples) % 2

    def reset(self):
        """把干扰机的连续时间轴复位到确定性原点（当前实现为空操作）。

        扫频与梳状的位置都由环境采样时钟（start_sample_idx）推导，干扰机自身
        不保存时间状态，因此无需真正复位；保留这个显式方法是为了集中 reset
        语义——fh_env 在 episode 重置时统一调用各干扰机的 reset，不必区分
        哪些干扰机内部是有状态的（reactive 需要，本类不需要）。
        """
        # 扫频与梳状的位置都由环境采样时钟推导，本类没有需要清零的内部计时状态；
        # 保留该显式方法是为了让 reset 语义集中在一处（调用方无需区分干扰机类型）。
        return None

    @staticmethod
    def _as_variant_buffer(buffer):
        """把预生成缓冲规范成二维 (num_variants, period_samples) 视图。

        兼容"未使用变体池"时保存的一维数组：补一行成二维后，后续切片逻辑
        无需再分支判断维数。

        Args:
            buffer: 预生成缓冲，一维或二维 ndarray。

        Returns:
            ndarray: 二维视图（一维输入返回 np.newaxis 视图，不复制数据）。

        Raises:
            RuntimeError: buffer 为 None 或空，说明尚未调用 precompute。
            ValueError: 维数不是 1 或 2。
        """
        if buffer is None or np.size(buffer) == 0:
            raise RuntimeError("Requested jammer buffer has not been pre-computed.")
        buffer = np.asarray(buffer)
        if buffer.ndim == 1:
            return buffer[np.newaxis, :]
        if buffer.ndim != 2:
            raise ValueError("Pre-computed jammer buffers must be one- or two-dimensional.")
        return buffer

    def _variant_for_cycle(self, jammer_kind, cycle_idx, num_variants):
        """取该周期应使用的变体下标；单变体或无选择器时恒为 0。

        抽选粒度就是"周期"：sweep 以完整扫频周期为单位、comb 以 phase0 + phase1
        完整周期为单位，因此同一周期内波形恒定，只有跨周期才换变体。
        变体下标从共享选择器取（记忆化），从而与调用顺序无关。

        Args:
            jammer_kind: "sweep" / "comb"，参与选择器的缓存键。
            cycle_idx: 周期序号。
            num_variants: 缓冲中实际可用的变体数。

        Returns:
            int: 变体下标。
        """
        if num_variants <= 1 or self.variant_selector is None:
            return 0
        return self.variant_selector.choice_for_cycle(jammer_kind, cycle_idx)

    def _sweep_buffer_slice(self, start_sample_idx, num_samples):
        """从预生成扫频缓冲中取任意起点、任意长度的连续切片。

        缓冲按"一个完整扫频周期"组织，本方法把请求区间按周期边界切开：每段用
        cycle_idx = global_idx // period_samples 决定变体，再按 cycle_offset 在周期内
        取偏移，因此跨周期切片会自动切换到该周期对应的变体（同周期内变体恒定）。

        Args:
            start_sample_idx: 全局 RF 采样下标（>=0）。
            num_samples: 需要的采样点数。

        Returns:
            ndarray: float32，长度 num_samples。

        Raises:
            RuntimeError: 扫频缓冲尚未 precompute。
        """
        buffers = self._as_variant_buffer(self.pre_buffer_sweep)
        period_samples = buffers.shape[1]
        result = np.empty(num_samples, dtype=np.float32)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            # 周期序号与周期内偏移都只由全局采样下标推导，与调用次数无关，
            # 因此任意起点/长度的切片都与连续时间轴一致。
            cycle_idx = global_idx // period_samples
            cycle_offset = global_idx % period_samples
            variant_idx = self._variant_for_cycle(
                "sweep",
                cycle_idx,
                buffers.shape[0],
            )
            take = min(period_samples - cycle_offset, num_samples - result_idx)
            result[result_idx:result_idx + take] = buffers[
                variant_idx,
                cycle_offset:cycle_offset + take,
            ]
            global_idx += take
            result_idx += take
        return result

    def _comb_buffer_slice(self, start_sample_idx, num_samples):
        """从预生成梳状缓冲中取连续切片，并按相位选择 phase0/phase1 缓冲。

        与扫频版本的区别：一个"完整周期" = phase0 + phase1（2×switch_samples），
        变体按完整周期抽取（保证相位 0/1 两个半周期来自同一条基带变体，相位切换
        处基带连续），而当前相位由周期内偏移是否越过 comb_switch_samples 决定。

        Args:
            start_sample_idx: 全局 RF 采样下标（>=0）。
            num_samples: 需要的采样点数。

        Returns:
            ndarray: float32，长度 num_samples。

        Raises:
            RuntimeError: 梳状缓冲尚未 precompute，或两个相位缓冲形状不一致
                （数据被外部破坏时的防御性检查）。
        """
        buffers0 = self._as_variant_buffer(self.pre_buffer_comb0)
        buffers1 = self._as_variant_buffer(self.pre_buffer_comb1)
        if buffers0.shape != buffers1.shape:
            raise RuntimeError("Comb phase buffers must have matching shapes.")
        result = np.empty(num_samples, dtype=np.float32)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            # 变体按"完整相位周期"选（两个半周期共享同一变体），相位则按
            # 周期内偏移落在前/后半段判定。
            cycle_idx = global_idx // self.comb_period_samples
            cycle_offset = global_idx % self.comb_period_samples
            phase = 0 if cycle_offset < self.comb_switch_samples else 1
            phase_offset = cycle_offset % self.comb_switch_samples
            variant_idx = self._variant_for_cycle(
                "comb",
                cycle_idx,
                buffers0.shape[0],
            )
            take = min(
                self.comb_switch_samples - phase_offset,
                num_samples - result_idx,
            )
            buffer = buffers0 if phase == 0 else buffers1
            result[result_idx:result_idx + take] = buffer[
                variant_idx,
                phase_offset:phase_offset + take,
            ]
            global_idx += take
            result_idx += take

        return result

    def precompute(self, Startfre=3e6, Endfre=4e6):
        """预生成"一个自然周期"的射频缓冲，每个基带变体一份。

        预生成内容：
          - sweep：长度 sweep_period_samples = samples_per_dwell × num_steps，
            基带取一条整周期的连续切片 × 功率 × 载波；
          - comb：两个相位各 comb_switch_samples 点；基带取
            2×comb_switch_samples 的连续切片，并按相位一分为二，再分别乘上各自的
            载波——保证相位切换处基带连续（只有载波频率组改变），避免人为引入
            时域冲激（冲激会污染 PSD 与 BER）。
        缓冲统一为 (num_variants, 周期采样点数)，无池时只有 1 行。

        为什么只缓存一个周期：波形按周期重复（同周期内变体恒定、跨周期才换变体），
        因此任意长度请求都能由"周期缓冲 + 逐周期变体"重建，无需缓存整条轨迹，
        内存占用与仿真时长解耦。

        Args:
            Startfre: 频段起始频率 (Hz)。
            Endfre: 频段结束频率 (Hz)。

        Raises:
            ValueError: 扫频参数非法（由 _sweep_layout 抛出）。
        """
        print(f"Pre-computing Jammers (Mode: {self.mode})...")
        # 先清空旧缓冲：RF 布局或模式变化后若残留旧波形，切片长度与相位都会错配。
        self.pre_buffer_sweep = None
        self.pre_buffer_comb0 = None
        self.pre_buffer_comb1 = None
        self.sweep_period_samples = 0
        self._ensure_carrier_cache(Startfre, Endfre)

        if self.sweep_enabled:
            _, samples_per_dwell, num_steps = self._sweep_layout(Startfre, Endfre)
            self.sweep_period_samples = samples_per_dwell * num_steps
            num_variants = (
                self.sweep_variant_pool.num_variants
                if self.sweep_variant_pool is not None
                else 1
            )
            self.pre_buffer_sweep = np.empty(
                (num_variants, self.sweep_period_samples),
                dtype=np.float32,
            )
            for variant_idx in range(num_variants):
                # 每个变体都取"整周期长度"的连续基带，保证一个扫频周期内基带连续
                # （不重复、不接缝）；无池时退回动态噪声源（只有变体 0）。
                baseband = (
                    self.sweep_variant_pool.get_variant(
                        variant_idx,
                        self.sweep_period_samples,
                    )
                    if self.sweep_variant_pool is not None
                    else self._dynamic_noise_source("sweep").get_noise(
                        self.sweep_period_samples
                    )
                )
                self.pre_buffer_sweep[variant_idx] = (
                    np.asarray(baseband, dtype=np.float32)
                    * np.float32(self.s_power)
                    * self._carrier_sweep
                )

        if self.comb_enabled:
            num_variants = (
                self.comb_variant_pool.num_variants
                if self.comb_variant_pool is not None
                else 1
            )
            self.pre_buffer_comb0 = np.empty(
                (num_variants, self.comb_switch_samples),
                dtype=np.float32,
            )
            self.pre_buffer_comb1 = np.empty_like(self.pre_buffer_comb0)
            for variant_idx in range(num_variants):
                baseband = (
                    self.comb_variant_pool.get_variant(
                        variant_idx,
                        self.comb_period_samples,
                    )
                    if self.comb_variant_pool is not None
                    else self._dynamic_noise_source("comb").get_noise(
                        self.comb_period_samples
                    )
                )
                baseband = (
                    np.asarray(baseband, dtype=np.float32)
                    * np.float32(self.c_power)
                )
                # 同一段连续基带的前半段配 phase0 载波、后半段配 phase1 载波：
                # 相位切换处基带无缝衔接，只有 tone 组合发生变化。
                self.pre_buffer_comb0[variant_idx] = (
                    baseband[:self.comb_switch_samples]
                    * self._carrier_comb0
                )
                self.pre_buffer_comb1[variant_idx] = (
                    baseband[self.comb_switch_samples:]
                    * self._carrier_comb1
                )

        print("Jammer Pre-computation complete.")

    def precomputed_period_samples(self):
        """返回各干扰时序周期的 LCM（时序对齐周期，不是波形数值周期）。

        用途：给外部（数据预生成/缓存窗口规划）一个"多久之后时序图案完全重复"的
        采样点数。注意由于变体是按周期随机抽取的，真实波形的重复周期还要乘上
        变体序列的周期（取决于选择器状态），因此这里只是时序上的最小公共周期。

        Returns:
            int: sweep 周期与 comb 周期的 LCM；两者都未启用时返回 1。

        Raises:
            RuntimeError: 启用了 sweep 但尚未 precompute（sweep_period_samples <= 0）。
        """
        periods = []
        if self.sweep_enabled:
            if self.sweep_period_samples <= 0:
                raise RuntimeError("Sweep jammer has not been pre-computed.")
            periods.append(self.sweep_period_samples)
        if self.comb_enabled:
            periods.append(self.comb_period_samples)
        if not periods:
            return 1
        return math.lcm(*periods)

    def get_composite_signal(self, start_sample_idx, num_samples):
        """
        按全局采样时钟取当前模式下所有预生成干扰的叠加切片。

        加速路径入口：只做缓冲切片与相加，不产生随机数、不做 FFT，因此可以在
        block 级并行中被安全调用——变体选择按 (jammer_kind, cycle_idx) 记忆化，
        结果与调用顺序、调用次数都无关。

        Args:
            start_sample_idx: 全局 RF 采样下标（>=0），fh_env 用 jammer_ptr 维护。
            num_samples: 需要的采样点数。

        Returns:
            ndarray: float32，长度 num_samples；'both' 模式为 sweep + comb 之和。

        Raises:
            ValueError: start_sample_idx 或 num_samples 为负。
            RuntimeError: 需要的缓冲尚未 precompute。
        """
        start_sample_idx = int(start_sample_idx)
        num_samples = int(num_samples)
        if start_sample_idx < 0:
            raise ValueError("start_sample_idx must be non-negative.")
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        jam_total = np.zeros(num_samples, dtype=np.float32)

        if self.sweep_enabled:
            jam_total += self._sweep_buffer_slice(
                start_sample_idx,
                num_samples,
            )
        if self.comb_enabled:
            jam_total += self._comb_buffer_slice(start_sample_idx, num_samples)

        return jam_total

    def _comb_frequencies(self, phase, Startfre, Endfre):
        """返回某相位下落在频段内的干扰 tone 中心频率数组 (Hz)。

        信道序号 k 的中心频率固定为 Startfre + k·50 kHz + 25 kHz（50 kHz 是信道
        间隔常量；注意这里不使用 c_step，配置里的 comb step 目前不参与频率计算）。
        末尾用 [Startfre, Endfre) 过滤：允许配置的信道组序号超出频段范围而不报错，
        超出的 tone 直接被丢弃。

        Args:
            phase: 0 或 1，选择 comb_channels 中对应的信道组。
            Startfre: 频段起始频率 (Hz)。
            Endfre: 频段结束频率 (Hz)。

        Returns:
            ndarray: float64，该相位实际使用的中心频率（可能为空数组）。
        """
        sub_interval = 50000.0
        target_indices = np.asarray(self.comb_channels[phase])
        freqs = Startfre + target_indices * sub_interval + 0.5 * sub_interval
        return freqs[(freqs >= Startfre) & (freqs < Endfre)]

    def _generate_comb_signal(
        self,
        num_samples,
        start_sample_idx,
        Startfre,
        Endfre,
        fixed_phase=None,
        baseband_noise=None,
        rng=None,
    ):
        """动态生成一段梳状干扰（缓存载波 × 现场基带噪声）。

        时序：以 comb_switch_samples 为粒度把请求区间切段，段内相位由
        start_sample_idx（或 fixed_phase 强制指定）决定，载波从
        _carrier_comb0/_carrier_comb1 的相位内偏移处截取，因此切段边界与全局
        时间轴严格对齐（与预生成路径同相位）。

        细节与取舍：
          - 基带噪声按整段请求一次性抽取（连续、不分段），只有载波随相位切换，
            避免相位边界处基带出现人为接缝；
          - 若某相位在频段内没有可用 tone（freqs 为空），该段输出保持 0，
            但基带噪声仍按整段长度消耗（噪声推进与是否真正发射无关）；
          - freqs_used 按段追加，同一相位被切成多段时频率会重复出现；该列表仅用于
            日志/可视化，不保证去重。

        Args:
            num_samples: 生成采样点数。
            start_sample_idx: 全局 RF 采样下标，决定起始相位与载波偏移。
            Startfre: 频段起始频率 (Hz)。
            Endfre: 频段结束频率 (Hz)。
            fixed_phase: 强制使用的相位（0/1）；None 时按时间轴自动判定。
            baseband_noise: 外部提供的基带噪声；None 时现场抽取。
            rng: 随机数生成器，仅在需要现场抽取基带噪声时使用。

        Returns:
            jam: ndarray，float32，长度 num_samples。
            freqs_used: list，本次使用的 tone 中心频率 (Hz) 列表（可能含重复）。

        Raises:
            ValueError: 外部基带噪声长度与 num_samples 不一致。
        """
        num_samples = int(num_samples)
        jam = np.zeros(num_samples, dtype=np.float32)
        freqs_used = []
        if num_samples == 0:
            return jam, freqs_used

        self._ensure_carrier_cache(Startfre, Endfre)
        if baseband_noise is None:
            baseband_noise = _noise_source_slice(
                self._dynamic_noise_source("comb"),
                num_samples,
                rng=rng,
            )
        baseband_noise = np.asarray(baseband_noise, dtype=np.float32)
        if len(baseband_noise) != num_samples:
            raise ValueError("Comb baseband noise length does not match num_samples.")
        # 功率在合成前一次性乘到基带上（与预生成路径的乘功率位置语义一致）。
        baseband_noise = baseband_noise * np.float32(self.c_power)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            # phase_offset 为当前相位区间内的偏移；take 取到相位边界或请求末尾为止，
            # 因此循环每轮最多跨越一个相位切换点。
            phase_offset = global_idx % self.comb_switch_samples
            phase = (
                int(fixed_phase)
                if fixed_phase is not None
                else self.comb_phase_at(global_idx)
            )
            take = min(
                self.comb_switch_samples - phase_offset,
                num_samples - result_idx,
            )
            freqs = self._comb_frequencies(phase, Startfre, Endfre)
            if len(freqs) > 0:
                combined_carrier = (
                    self._carrier_comb0 if phase == 0 else self._carrier_comb1
                )[phase_offset:phase_offset + take]
                freqs_used.extend(freqs.tolist())
                jam[result_idx:result_idx + take] = (
                    baseband_noise[result_idx:result_idx + take] * combined_carrier
                )

            global_idx += take
            result_idx += take

        return jam, freqs_used

    def _generate_sweep_signal(
        self,
        num_samples,
        start_sample_idx,
        Startfre,
        Endfre,
        baseband_noise=None,
        rng=None,
    ):
        """动态生成一段扫频干扰（缓存载波 × 现场基带噪声）。

        时序：以 samples_per_dwell 为粒度切段，段内频点下标
        sweep_idx = (global_idx // samples_per_dwell) % num_steps，每 num_steps 段
        回卷一次，形成与预生成路径一致的循环扫频图案；频点中心频率
        f = Startfre + sweep_idx·s_step + s_step/2（取信道中心），载波从
        _carrier_sweep 的 dwell 内偏移处截取。

        细节：基带噪声整段连续抽取，只有载波随频点切换；每个切段向 freqs_used
        追加一个 f（一段对应一个频点）。

        Args:
            num_samples: 生成采样点数。
            start_sample_idx: 全局 RF 采样下标，决定起始频点与载波偏移。
            Startfre: 频段起始频率 (Hz)。
            Endfre: 频段结束频率 (Hz)。
            baseband_noise: 外部提供的基带噪声；None 时现场抽取。
            rng: 随机数生成器，仅在需要现场抽取基带噪声时使用。

        Returns:
            jam: ndarray，float32，长度 num_samples。
            freqs_used: list，每个切段使用的频点中心频率 (Hz)。

        Raises:
            ValueError: 外部基带噪声长度与 num_samples 不一致。
        """
        num_samples = int(num_samples)
        jam = np.zeros(num_samples, dtype=np.float32)
        freqs_used = []
        if num_samples == 0:
            return jam, freqs_used

        self._ensure_carrier_cache(Startfre, Endfre)
        bw_total, samples_per_dwell, num_steps = self._sweep_layout(
            Startfre,
            Endfre,
        )
        if baseband_noise is None:
            baseband_noise = _noise_source_slice(
                self._dynamic_noise_source("sweep"),
                num_samples,
                rng=rng,
            )
        baseband_noise = np.asarray(baseband_noise, dtype=np.float32)
        if len(baseband_noise) != num_samples:
            raise ValueError("Sweep baseband noise length does not match num_samples.")
        baseband_noise = baseband_noise * np.float32(self.s_power)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            # 段内偏移 + 频点序号（对 num_steps 取模即扫频回卷）；take 取到 dwell
            # 边界或请求末尾，因此循环每轮最多跨越一个频点切换点。
            dwell_offset = global_idx % samples_per_dwell
            sweep_idx = (global_idx // samples_per_dwell) % num_steps
            take = min(
                samples_per_dwell - dwell_offset,
                num_samples - result_idx,
            )
            f = Startfre + sweep_idx * self.s_step + self.s_step / 2.0
            if f >= Endfre:
                # 与载波缓存中的回卷保护一致：正常情况下不会触发，仅防御浮点误差。
                f = Startfre + np.mod(f - Startfre, bw_total)
            # 载波缓存里第 sweep_idx 个 dwell 从 sweep_idx·samples_per_dwell 开始，
            # 再加上段内偏移即得该采样点的载波位置。
            carrier_start = sweep_idx * samples_per_dwell + dwell_offset
            carrier = self._carrier_sweep[
                carrier_start:carrier_start + take
            ]
            jam[result_idx:result_idx + take] = (
                baseband_noise[result_idx:result_idx + take] * carrier
            )
            freqs_used.append(f)
            global_idx += take
            result_idx += take

        return jam, freqs_used

    def generate_samples(
        self,
        num_samples,
        Startfre,
        Endfre,
        start_sample_idx=0,
        rng=None,
    ):
        """动态生成一段干扰（缓存载波 + 新鲜基带噪声），返回波形与实际频点。

        与 get_composite_signal 的区别：基带噪声由 rng 现场抽取而非复用预生成变体，
        因此每次调用都是"新鲜"的（适合 use_pregen=False 的训练路径）；载波与相位
        仍严格由 start_sample_idx 决定，保持与预生成路径相同的时序结构。

        'both' 模式下按 comb → sweep 的固定顺序生成，且各自独立抽取一次噪声。
        这个顺序被刻意保留（历史行为）：它决定了 rng 的消耗次序，改动会改变
        随机流并破坏与既有实验/离线 replay 的可比性。

        Args:
            num_samples: 生成采样点数。
            Startfre: 频段起始频率 (Hz)。
            Endfre: 频段结束频率 (Hz)。
            start_sample_idx: 全局 RF 采样下标。
            rng: 随机数生成器；None 时噪声源退回全局 np.random。

        Returns:
            jam: ndarray，float32，长度 num_samples。
            freqs_used: list，本次使用的频率列表（comb 在前、sweep 在后）。
        """
        N = int(num_samples)
        if N == 0:
            return np.zeros(0, dtype=np.float32), []

        jam = np.zeros(N, dtype=np.float32)
        freqs_used = []

        # 保留 'both' 模式下历史的抽取/生成顺序：先 comb、后 sweep，
        # 且对 rng 做两次独立抽取。
        if self.comb_enabled:
            comb_jam, comb_freqs = self._generate_comb_signal(
                N,
                start_sample_idx,
                Startfre,
                Endfre,
                rng=rng,
            )
            jam += comb_jam
            freqs_used.extend(comb_freqs)

        if self.sweep_enabled:
            sweep_jam, sweep_freqs = self._generate_sweep_signal(
                N,
                start_sample_idx,
                Startfre,
                Endfre,
                rng=rng,
            )
            jam += sweep_jam
            freqs_used.extend(sweep_freqs)

        return jam, freqs_used

    def generate(self, t, Startfre, Endfre, start_sample_idx=0, rng=None):
        """兼容旧接口：接受历史的时间向量 t，内部按 len(t) 转调 generate_samples。"""
        return self.generate_samples(
            len(t),
            Startfre,
            Endfre,
            start_sample_idx=start_sample_idx,
            rng=rng,
        )
