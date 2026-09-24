"""
train_speed_derivative.py — 基于导数 NBS 的跳速阈值搜索入口
==========================================================

用 DerivativeNoisyBinarySearch（导数版噪声二分搜索，算法实现与详细原理见
noisy_binary_search_derivative.py）定位反应式干扰机的跳速跟踪阈值：低
hoprate 时干扰机能跟上跳频并压制通信（高 BER），hoprate 足够高时干扰机
跟不上（低 BER）。搜索目标就是 BER–hoprate 关系发生转折的那个 hoprate。

与基础版 NBS 的区别：方向决策不用"BER 升/降"的二元信号，而用基于 BER 与
hoprate 的*导数*（变化率）指标，且梯度参考点取最近一次与当前 hoprate
**不同**的观测（不一定是紧邻的上一步）。

本脚本的定位
------------
只做"跳速搜索"这一件事：不训练 SAC，也不学习 offset（offset 学习不在本
脚本范围内），因此每个环境 step 的 offset 用随机采样代替策略输出——使 BER
反馈只由 hoprate（以及环境固有随机性）决定，不引入策略漂移这一额外变量。

单步因果流程
------------
对第 i 个环境 step（共 --steps 步）：

1. **NBS 提议 hoprate**：上一轮 NBS 更新返回的 next_hoprate 即本步待测
   跳速；第 1 步的值来自 nbs.reset()（均匀权重下的加权中位数）。
2. **随机 offset 采样**：np.random.randint(0, n_channels, size=10) 生成
   10 个信道偏移，对应环境一个 step 内的 10 个 block（NUM_BLOCKS=10，
   每 block 100 ms）。每个 offset 是该 block 跳频图案的循环信道位移。
3. **环境执行**：env.step({"hoprate": h, "offsets": offsets}) 在给定跳速
   下跑完 10 个 block。环境内部会把 hoprate 量化到 10 Hz 网格并裁剪到
   [hoprate_min, hoprate_max]，实际生效值经 info["hoprate_used"] 回传，
   故本步真正测试的跳速与提议值可能相差最多数 Hz。
4. **BER 反馈**：取 info["ber_blocks"]（10 个 block 各自的 BER）的均值
   mean_ber，作为该 hoprate 下的一次带噪观测。
5. **NBS 更新**：nbs.step(mean_ber) 用导数指标更新 MWU 权重并返回下一个
   待测 hoprate。指标定义为

       delta_ber_percent = (BER_curr − BER_ref) × 100
       delta_hoprate     = hoprate_curr − hoprate_ref
       metric            = delta_ber_percent / delta_hoprate

   其中参考点 (hoprate_ref, BER_ref) 是最近一次与当前 hoprate 不同的
   观测。由于参考点机制保证 delta_hoprate 恒非零，**不存在除零问题**，
   无需对 delta_hoprate == 0 做任何兜底（基础版曾需要该兜底，本变体从
   参考点机制上结构性避免了这一情形）；仅当还没有"不同 hoprate"的观测
   可作参考时，NBS 内部走 _force_move 强制移动一次，以便尽快采到第二个
   不同 hoprate。
6. **判决规则**（阈值 --derivative_threshold，默认 -0.005）：
   - metric > threshold → 向左移动（支持更低 hoprate 的一侧）
   - metric ≤ threshold → 向右移动（支持更高 hoprate 的一侧）
7. **历史记录与落盘**：每步记录实际测试的 hoprate、mean_ber、NBS 的 MAP
   估计（get_best_hoprate）、加权平均（get_weighted_average）与本步导数
   指标；max(w) ≥ 1−δ 时日志打上收敛标记，但默认继续跑满 --steps（便于
   观察收敛后估计是否稳定）。

反应式干扰机在本脚本中被**强制打开**（enable_reactive=True）：只有它开启
时环境中才存在清晰的 BER–hoprate 阈值。无差别干扰机（sweep/comb）默认
关闭，可用 --enable_sweep 叠加。

输出产物（均位于 --output_dir）
-------------------------------
- training_log.txt：逐步日志与最终汇总（mode="w"，每次运行覆盖旧日志）。
- hoprate.png：测试 hoprate 轨迹 + NBS 的 MAP/加权平均估计。
- ber.png：每步 mean_ber 曲线。
- ber_vs_hoprate.png：BER–hoprate 散点（按 step 着色），阈值存在性最直观
  的证据。
- nbs_weights.png：最终权重分布（含 1−δ 收敛参考线）。
- derivative.png：导数指标随 step 的变化（**仅当至少一步产出有效指标时
  才生成**；图中阈值参考线固定为 -0.002 历史默认值，不随 CLI 参数变化）。
- nbs_distribution.npz：candidates / weights / hoprates_used / bers /
  derivatives（无效指标以 NaN 占位，保持与 step 一一对应）。

用法（Usage）
-------------

.. code-block:: bash

    # Quick test (default threshold -0.002)
    D:\\Anaconda\\envs\\rl_fhss\\python.exe train_speed_derivative.py --steps 60 --output_dir outputs/speed_test_derivative

    # With custom derivative threshold
    D:\\Anaconda\\envs\\rl_fhss\\python.exe train_speed_derivative.py --derivative_threshold -0.005 --steps 100

（上面示例注释里的 "-0.002" 为历史默认值；当前 CLI 默认
--derivative_threshold 是 -0.005，与默认输出目录名 .../-0.005 对应。）

参考论文
--------
Dereniowski et al., "Noisy (Binary) Searching: Simple, Fast and Correct"
(STACS 2025)。查询选择与 MWU 更新对应论文 Algorithm 3.1；项目级默认参数
集中在 settings.NBS_CONFIG。

可复现性说明
------------
__main__ 先用 settings.set_random_seeds() 固定全局种子（决定随机 offset 与
环境内部噪声），但 NBS 实例未传 seed（seed=None），其查询随机化 RNG 不可
复现，因此同一命令的逐 step 轨迹不保证完全一致；阈值估计与整体趋势稳定。
"""

import argparse
import os
import time
import numpy as np
import logging
import matplotlib.pyplot as plt

from fh_env import FHSSQPSKEnv
from noisy_binary_search_derivative import DerivativeNoisyBinarySearch
import settings


def setup_logger(log_file):
    """配置并返回根 logger，日志同时写入文件与控制台。

    设计说明：
    - 文件 handler 用 mode="w"（每次运行覆盖旧日志）并显式指定 UTF-8，
      避免 Windows 默认 GBK 编码下日志中的中文/特殊字符写盘失败。
    - force=True 会先移除已存在的 root handler，因此重复调用（或与其它
      模块的 logging.basicConfig 共存）时以本配置为准，不会出现重复输出。
    - 级别固定 INFO、格式与项目其它入口脚本一致，便于跨实验比对日志。

    Args:
        log_file: 日志文件路径；调用方需保证其父目录已创建。

    Returns:
        logging.Logger: 根 logger，后续直接用 logger.info 写日志。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
        force=True,
    )
    return logging.getLogger()


def search(args):
    """执行完整的导数 NBS 跳速阈值搜索，并落盘全部产物。

    流程：建输出目录与 logger → 构造环境（强制 enable_reactive=True）→
    构造 DerivativeNoisyBinarySearch（候选网格直接取 env.hoprate_min/max，
    保证提议值不会越界）→ 循环 args.steps 个环境 step（NBS 提议 hoprate、
    随机 offset、env.step、mean BER 反馈、NBS 更新、记录与日志）→ 打印
    汇总 → 保存 nbs_distribution.npz → 绘制诊断图。

与 train_speed_sweep.py 的确定性网格不同，这里由 MWU 权重分布自适应
选点：每次观测都会把权重按 2(1−p)（兼容侧）/ 2p（不兼容侧）的乘子重新
加权，使质量向阈值附近指数级集中，因此通常比全网格扫描省 step；代价是
采样点非均匀、轨迹依赖 NBS 内部随机化，单次运行的估计存在波动。

    Args:
        args: parse_args() 返回的命名空间；字段含义与默认值见 parse_args
            的 docstring。关键字段为 steps / output_dir / log_file /
            enable_sweep 与 NBS 四项超参。

    副作用：
        在 --output_dir 下写 training_log.txt、nbs_distribution.npz 与
        若干 PNG（见模块 docstring 的"输出产物"）；向根 logger 输出 INFO。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Algorithm: Derivative-based Noisy Binary Search")

    # ---- 环境（为发现阈值，强制打开反应式干扰机）-------------------------------
    # 先浅拷贝 ENV_CONFIG 再覆盖字段，避免污染 settings 里的全局配置。
    # enable_reactive 硬编码为 True：只有反应式干扰机具备"检测—跟踪—压制"
    # 时延（detection_time，默认 0.5 ms，即默认输出目录名中的 0.5ms），
    # BER 才会随 hoprate 上升出现清晰的阈值；无差别干扰机（sweep/comb）
    # 由 CLI 决定是否叠加。
    env_config = dict(settings.ENV_CONFIG)
    env_config["enable_reactive"] = True
    env_config["enable_sweep"] = args.enable_sweep

    env = FHSSQPSKEnv(**env_config)
    n_channels = env.num_channels
    logger.info(f"Environment: {n_channels} channels, "
                f"hoprate ∈ [{env.hoprate_min:.0f}, {env.hoprate_max:.0f}] Hz, "
                f"reactive={env_config['enable_reactive']}, "
                f"sweep={env_config['enable_sweep']}")

    # ---- 导数版噪声二分搜索 ----------------------------------------------------
    # 候选网格与环境的合法 hoprate 区间完全一致（env.hoprate_min/max），
    # 因此 NBS 提议的网格点不会被环境侧裁剪；hoprate_step 默认 10 Hz，
    # 与环境把 hoprate 量化到 10 Hz 网格的行为对齐，避免"提议值"与"实际
    # 生效值"系统性错位。
    nbs = DerivativeNoisyBinarySearch(
        hoprate_min=env.hoprate_min,
        hoprate_max=env.hoprate_max,
        hoprate_step=args.nbs_step,
        p=args.nbs_p,
        delta=args.nbs_delta,
        derivative_threshold=args.derivative_threshold,
    )
    logger.info(f"NBS: p={nbs.p}, δ={nbs.delta}, "
                f"step={nbs.hoprate_step} Hz, "
                f"derivative_threshold={nbs.derivative_threshold}, "
                f"candidates={nbs.n_candidates}")

    # ---- 逐步追踪（供日志、npz 与绘图共用）------------------------------------
    # 五个列表长度恒等于实际完成的 step 数且一一对应，是绘图与 npz 的对齐
    # 不变量；hoprates_used 记录**实际生效**的跳速（info["hoprate_used"]，
    # 已量化/裁剪），bers 记录 10 个 block BER 的均值（np.mean，非中位数）。
    hoprates_used = []       # 每步实际生效的 hoprate
    bers = []                # 每步的 mean BER（10 个 block 的均值）
    nbs_best_history = []    # NBS 的 MAP 估计随时间的变化
    nbs_wavg_history = []    # NBS 加权平均估计随时间的变化
    derivatives = []         # 每步导数指标，用于诊断

    # ---- 初始化 ----------------------------------------------------------------
    # 环境 reset 只为取得初始观测（本脚本不使用观测输入）；NBS reset 返回
    # 首个查询点（均匀权重下的加权中位数，约在区间中部），并把内部状态
    # 清空、权重复位为均匀分布。
    _state_img, info = env.reset()
    hoprate = float(nbs.reset())
    logger.info(f"Initial NBS hoprate: {hoprate:.0f} Hz")

    start_time = time.time()

    for step_idx in range(1, args.steps + 1):
        step_start = time.time()

        # --- 1. 环境 step：随机 offset（不训练 SAC，offset 学习不在范围内）----
        # 10 个 offset 对应本 step 内 10 个 block，各自是该 block 跳频图案的
        # 循环信道位移，取值 [0, n_channels-1]。用全局 np.random 而非局部
        # RNG，其种子由 __main__ 的 settings.set_random_seeds() 固定。
        offsets = np.random.randint(0, n_channels, size=10).astype(np.float32)
        _obs, _rew, terminated, truncated, info = env.step(
            {"hoprate": hoprate, "offsets": offsets}
        )

        # --- 2. BER：取 10 个 block 的均值作为本步观测 --------------------------
        # 单 block 的 BER 噪声较大，取均值可压低观测噪声（NBS 的噪声模型 p
        # 只用于覆盖剩余的不确定性）；ber_blocks 缺失时回退 0.0。
        # hoprate_used 是环境量化/裁剪后的真实生效值，与提议值可能不同。
        ber_blocks = info.get("ber_blocks", [])
        mean_ber = float(np.mean(ber_blocks)) if ber_blocks else 0.0
        hoprate_used = float(info.get("hoprate_used", hoprate))

        # --- 3. NBS 更新：喂入 mean BER，取回下一个待测 hoprate -----------------
        # 导数指标在 NBS 内部计算（相对最近一次不同 hoprate 的观测），此处
        # 不关心指标本身，只完成"观测 → 权重更新 → 下一查询点"的闭环。
        next_hoprate = nbs.step(mean_ber)

        # --- 4. 记录：先推进 hoprate，再记录本步实际用量与 NBS 当前估计 ---------
        # 注意次序：hoprate 在此处被赋值为下一步待测值，而 hoprates_used 记录
        # 的是本步**实际测试**的 hoprate_used，两者语义不同不可混用。
        hoprate = next_hoprate

        hoprates_used.append(hoprate_used)
        bers.append(mean_ber)
        nbs_best_history.append(nbs.get_best_hoprate())
        nbs_wavg_history.append(nbs.get_weighted_average())
        derivatives.append(nbs.get_last_derivative())

        step_duration = time.time() - step_start

        # --- 5. 日志：单行汇总本步关键量，并附收敛标记 --------------------------
        # 第 1 步尚无"不同 hoprate"的参考点，导数指标为 None，此时省略
        # metric 字段（而不是打印 0，避免误读为"梯度为零"）；max_w 达到
        # 1−δ 即 NBS 判定收敛，但循环不因此提前退出，便于观察收敛后的稳定性。
        nbs_max_w = np.max(nbs.weights)
        conv_flag = " ✓CONVERGED" if nbs.is_converged() else ""

        metric = derivatives[-1]
        metric_str = f"metric={metric:10.4f} | " if metric is not None else ""

        logger.info(
            f"Step {step_idx:4d}/{args.steps} | "
            f"Hop={hoprate_used:6.0f} Hz | "
            f"BER={mean_ber:.4f} | "
            f"{metric_str}"
            f"NBS_best={nbs_best_history[-1]:6.0f} | "
            f"wavg={nbs_wavg_history[-1]:6.0f} | "
            f"max_w={nbs_max_w:.4f} | "
            f"T={step_duration:.2f}s{conv_flag}"
        )

        if nbs.is_converged():
            logger.info(f"NBS converged at step {step_idx} — "
                        f"threshold ≈ {nbs.get_best_hoprate():.0f} Hz")

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    # ---- 汇总 ------------------------------------------------------------------
    # 同时给出两种阈值估计：MAP（最大权重网格点，硬估计）与加权平均（软估计）。
    # 二者相差较大说明权重分布仍分散、结果可信度低；最终 max(w) 与收敛标志
    # 一起构成这次搜索是否可信的自检信息。
    elapsed = time.time() - start_time
    nbs_best = nbs.get_best_hoprate()
    nbs_wavg = nbs.get_weighted_average()

    logger.info(f"{'='*60}")
    logger.info(f"Search complete.  {len(hoprates_used)} steps in {elapsed:.1f}s")
    logger.info(f"Algorithm: Derivative-based NBS")
    logger.info(f"Derivative threshold: {nbs.derivative_threshold}")
    logger.info(f"NBS best estimate:  {nbs_best:.0f} Hz")
    logger.info(f"NBS weighted avg:   {nbs_wavg:.0f} Hz")
    logger.info(f"Converged:          {nbs.is_converged()}")
    logger.info(f"Final max weight:   {np.max(nbs.weights):.4f}")
    if len(bers) > 0:
        logger.info(f"Final BER:          {bers[-1]:.4f}")
        logger.info(f"Mean BER all steps: {np.mean(bers):.4f}")

    # ---- 保存分布与逐步历史 ----------------------------------------------------
    # 无效导数（早期无参考点时为 None）写成 NaN 而不是丢弃，保证 npz 中
    # 各数组与 step 严格一一对应，便于外部按 step 对齐分析。
    nbs_candidates, nbs_weights = nbs.get_distribution()
    np.savez(
        os.path.join(args.output_dir, "nbs_distribution.npz"),
        candidates=nbs_candidates,
        weights=nbs_weights,
        hoprates_used=np.array(hoprates_used),
        bers=np.array(bers),
        derivatives=np.array([d if d is not None else np.nan for d in derivatives]),
    )

    # ---- 绘图（见 _plot_results 的 docstring）---------------------------------
    _plot_results(args.output_dir, hoprates_used, bers,
                  nbs_best_history, nbs_wavg_history,
                  nbs_candidates, nbs_weights, derivatives)
    logger.info(f"Plots saved to {args.output_dir}.")


def _plot_results(output_dir, hoprates, bers, best_hist, wavg_hist,
                  nbs_cand, nbs_w, derivatives):
    """绘制五类诊断图（数据不足的图自动跳过）。

    各图用途与读法：

    1. hoprate.png：测试 hoprate 轨迹 + NBS 的 MAP/加权平均估计。MAP 与
       加权平均趋同、且围绕某条水平线小幅波动，说明权重已集中、估计可信。
    2. ber.png：每步 mean_ber。用于判断观测噪声水平与阈值转折是否明显。
    3. ber_vs_hoprate.png：BER–hoprate 散点按 step 着色。低 hoprate 高 BER
       平台 → 高 hoprate 低 BER 平台，是阈值存在性最直观的证据；颜色可
       看出搜索后期是否集中在阈值附近。
    4. nbs_weights.png：最终权重分布；红虚线为收敛阈值 1−δ（取自
       settings.NBS_CONFIG["delta"]，不随 --nbs_delta 变化），柱高越过该线
       即 NBS 报告收敛。柱宽取前两个候选点的间距，候选只有一个时回退 10 Hz。
    5. derivative.png：导数指标随 step 变化（**仅当至少一步有有效指标时
       生成**）。注意图中阈值参考线硬编码为 -0.002 历史默认值，并不反映
       本次实际使用的 --derivative_threshold，只作视觉参考。

    Args:
        output_dir: 图片保存目录（须已存在）。
        hoprates: 每步实际生效的 hoprate 序列。
        bers: 每步 mean BER 序列。
        best_hist: 每步 NBS MAP 估计序列。
        wavg_hist: 每步 NBS 加权平均估计序列。
        nbs_cand: 候选 hoprate 网格（第 4 张图用）。
        nbs_w: 最终权重（与 nbs_cand 等长，和为 1）。
        derivatives: 每步导数指标序列，元素可为 None。

    副作用：
        在 output_dir 下写出 4~5 个 PNG；每个 figure 用完即 close，避免
        长步数运行累积内存。
    """
    steps = np.arange(1, len(hoprates) + 1)

    # 1. hoprate 轨迹 + NBS 估计
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, hoprates, ".-", color="steelblue", alpha=0.7, label="tested hoprate")
    ax.plot(steps, best_hist, "--", color="darkorange", label="NBS best (MAP)")
    ax.plot(steps, wavg_hist, ":", color="darkgreen", label="NBS weighted avg")
    ax.set_title("Hoprate Trajectory & NBS Estimates (Derivative-based)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Hoprate (Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "hoprate.png"))
    plt.close(fig)

    # 2. 每步 mean BER
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, bers, ".-", color="crimson", alpha=0.7)
    ax.set_title("Mean BER per Step (Derivative-based NBS)")
    ax.set_xlabel("Step")
    ax.set_ylabel("BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber.png"))
    plt.close(fig)

    # 3. BER–hoprate 散点（阈值所在位置一目了然）
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hoprates, bers, c=steps, cmap="viridis", alpha=0.7, edgecolors="k", linewidth=0.3)
    fig.colorbar(ax.collections[0], ax=ax, label="Step")
    ax.set_title("BER vs Hoprate (Derivative-based NBS)")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber_vs_hoprate.png"))
    plt.close(fig)

    # 4. NBS 最终权重分布（红虚线为收敛阈值 1−δ）
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = nbs_cand[1] - nbs_cand[0] if len(nbs_cand) > 1 else 10
    ax.bar(nbs_cand, nbs_w, width=bar_width, color="steelblue", alpha=0.8)
    ax.axhline(y=1.0 - settings.NBS_CONFIG["delta"], color="red", linestyle="--",
               label=f"convergence threshold (1−δ={1-settings.NBS_CONFIG['delta']:.2f})")
    ax.set_title("NBS Final Weight Distribution (Derivative-based)")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "nbs_weights.png"))
    plt.close(fig)

    # 5. 导数指标随时间变化（导数版 NBS 专属诊断；无有效指标时整张图跳过，
    #    因此不保证每次运行都有 derivative.png）
    metric_valid = [m for m in derivatives if m is not None]
    if len(metric_valid) > 0:
        fig, ax = plt.subplots(figsize=(10, 4))
        metric_np = np.array([m if m is not None else np.nan for m in derivatives])
        ax.plot(steps, metric_np, ".-", color="purple", alpha=0.7)
        ax.axhline(y=-0.002, color="red", linestyle="--", alpha=0.5, label="threshold (-0.002)")
        ax.set_title("Metric (ΔBER%/Δhoprate) over Steps")
        ax.set_xlabel("Step")
        ax.set_ylabel("Metric (% BER per Hz)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.savefig(os.path.join(output_dir, "derivative.png"))
        plt.close(fig)


# ------------------------------------------------------------------
# 命令行入口
# ------------------------------------------------------------------
def parse_args():
    """解析命令行参数。

    默认值分两类，避免"算法库默认"与"实验默认"混淆：
    - NBS 三项（--nbs_p/--nbs_delta/--nbs_step）直接取 settings.NBS_CONFIG，
      保证与算法实现的项目级默认一致；
    - --derivative_threshold 默认 -0.005：这是**本次实验**的导数判决阈值，
      与默认输出目录名 .../-0.005 对应（类内默认 -0.002 仅作算法库兜底）。
    默认 --output_dir 中的 "0.5ms" 指反应式干扰机的 detection_time=0.5 ms。

    Returns:
        argparse.Namespace: steps/output_dir/log_file/enable_sweep 与 NBS
        四项超参，供 search() 使用。

    说明：
        --enable_sweep 用内联 lambda 解析布尔字符串（"true"/"1"/"yes"，
        大小写不敏感），与 train_speed_sweep.py 的 str_to_bool 不同——这里
        既不接受 "y"/"on"，也不对非法值报错（静默按 False 处理）；默认
        False 表示只用反应式干扰机做阈值发现。
    """
    p = argparse.ArgumentParser(
        description="Derivative-based NBS hoprate threshold search (random offsets, reactive jammer ON)"
    )

    # ---- 主循环 ---------------------------------------------------------------
    p.add_argument("--steps", type=int, default=400,
                   help="Number of environment steps (default: 400)")
    p.add_argument("--output_dir", type=str, default="outputs/speed_derivative/0.5ms/-0.005")
    p.add_argument("--log_file", type=str, default="training_log.txt")

    # ---- 环境覆盖项 -----------------------------------------------------------
    p.add_argument("--enable_sweep", type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=False,
                   help="Enable sweep/comb jammer alongside reactive (default: false)")

    # ---- NBS 超参 -------------------------------------------------------------
    p.add_argument("--nbs_p", type=float, default=settings.NBS_CONFIG["p"],
                   help="NBS noise probability 0 <= p < 0.5 (default: %(default)s)")
    p.add_argument("--nbs_delta", type=float, default=settings.NBS_CONFIG["delta"],
                   help="NBS confidence threshold (default: %(default)s)")
    p.add_argument("--nbs_step", type=float, default=settings.NBS_CONFIG["hoprate_step"],
                   help="NBS candidate step in Hz (default: %(default)s)")
    p.add_argument("--derivative_threshold", type=float, default=-0.005,
                   help="Derivative decision threshold (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    # 先固定全局种子（影响随机 offset 与环境内部噪声）再开始搜索；NBS 自身
    # 的查询随机化未设种子，故仅"环境侧"随机性可复现（见模块 docstring）。
    settings.set_random_seeds()
    search(parse_args())
