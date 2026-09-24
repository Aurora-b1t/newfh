"""
train_speed_sweep.py — hoprate 网格扫描评测（导数 NBS 搜索的确定性对照）
======================================================================

按升序遍历 hoprate 网格（默认 10 → 1000 Hz、步长 10 Hz），在每个 hoprate
上执行固定数量的环境 step（默认每档 4 步），记录 BER/reward 诊断，得到
完整、确定性的 BER–hoprate 曲线。每个环境 step 内部执行 10 个 block
（100 ms/block），与 FHSSQPSKEnv 的 step 语义一致。

脚本定位：本脚本是导数 NBS 搜索入口（train_speed_derivative.py）的确定性
对照。NBS 用 MWU 权重分布自适应选点、通常更省 step，但采样点非均匀且轨迹
依赖收敛过程；本脚本均匀遍历全部候选点、不做任何自适应决策，因此

  - 可作为 NBS 阈值估计的参照基准（阈值落在哪一档），
  - 可完整暴露 BER 曲线的平台/转折与噪声水平，为 NBS 的 p、导数阈值等
    超参选择提供依据。

代价是 step 数 = 网格点数 × steps_per_hoprate（默认约 100 档 × 4 步 = 400
步），比 NBS 昂贵。默认启用反应式干扰机（--enable_reactive true）：只有它
开启时环境中才存在清晰的 BER–hoprate 阈值。

设计要点
--------
- 候选点按 --hoprate_step 生成，默认落在 10 Hz 栅格上，与环境把 hoprate
  量化到 10 Hz 网格的行为对齐，避免"目标值"与"实际生效值"系统性错位。
- offset 由 --offset_mode 控制：random 表示每步随机信道偏移（与导数 NBS
  入口的 offset 采样方式一致，便于横向对照），zeros 表示全零偏移（关闭
  offset 维度、只保留 m 序列基础跳频图案，作为确定性基线）。
- 随机性来自两处：本脚本用 RandomState(--seed) 采样 offset；环境内部
  （白噪声、Rayleigh、观测噪声等）用 settings.set_random_seeds(--seed)
  固定的全局种子。两者共同决定结果可复现性。

输出产物（均位于 --output_dir）
-------------------------------
- training_log.txt：逐 step 日志与每档汇总（mode="w"，每次运行覆盖）。
- hoprate_sweep_steps.csv：逐 step 明细（global_step、目标/实际 hoprate、
  local_step、mean_ber、reward、offsets 列表、耗时）。
- hoprate_sweep_summary.csv：逐 hoprate 汇总（均值/标准差、blocks、耗时）。
- hoprate_sweep_results.npz：hoprates/ber_mean/ber_std/reward_mean/
  reward_std 数组（仅在有汇总记录时写出）。
- ber_vs_hoprate.png、reward_vs_hoprate.png（带标准差误差棒）与
  step_ber_hoprate.png（BER 与 hoprate 双 y 轴轨迹）；--no_plots 可关闭
  绘图（CSV/NPZ 仍会保存）。

用法（Usage）
-------------

.. code-block:: bash

    D:\\Anaconda\\envs\\rl_fhss\\python.exe train_speed_sweep.py --output_dir outputs/speed_sweep

Quick smoke test:

.. code-block:: bash

    D:\\Anaconda\\envs\\rl_fhss\\python.exe train_speed_sweep.py --hoprate_max 20 --steps_per_hoprate 1
"""

import argparse
import csv
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from fh_env import FHSSQPSKEnv
import settings


def setup_logger(log_file):
    """配置并返回根 logger，日志同时写入文件与控制台。

    与 train_speed_derivative.py 中的同名函数是刻意的重复实现：两个入口都是
    可独立运行的脚本，不引入共享模块可避免"改一处影响另一个实验"的耦合。

    设计说明：
    - 文件 handler 用 mode="w"（每次运行覆盖旧日志）并显式指定 UTF-8，
      避免 Windows 默认 GBK 编码下中文/特殊字符写盘失败。
    - force=True 先移除已有 root handler，重复调用不会产生重复输出。
    - 级别固定 INFO、格式与其它入口脚本一致，便于跨实验比对日志。

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


def str_to_bool(value):
    """把命令行布尔字符串解析为 bool（argparse 的 type= 回调）。

    与 train_speed_derivative.py 的内联 lambda 不同：这里显式支持更多写法
    （含 "y"/"on"），且对无法识别的输入抛错而不是静默按 False 处理——布尔
    开关写错时应当尽早暴露，而不是让实验在错误配置下跑完。

    Args:
        value: bool 或字符串；字符串大小写不敏感。

    Returns:
        bool: 解析结果（bool 直传时原样返回）。

    Raises:
        argparse.ArgumentTypeError: 字符串不在可识别集合内时抛出；argparse
            会据此打印用法并以退出码 2 结束。
    """
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_hoprate_grid(hoprate_min, hoprate_max, hoprate_step):
    """生成升序 hoprate 候选网格（dtype=float32）。

    从 hoprate_min 起按 hoprate_step 递增，取
    count = floor((max − min) / step) + 1 个点；仅当 max 恰好是下一个栅格点
    （剩余量 ≈ step，用 np.isclose 判断）时才把 max 追加为末点，否则**不
    追加**——保证所有候选点都落在 step 栅格上（默认即 10 Hz 栅格），与环境
    对 hoprate 的 10 Hz 量化对齐，使"目标值"与 info["hoprate_used"] 一致，
    避免出现"设 1005 Hz 实际跑 1010 Hz"这类错位。

    返回 float32 以匹配环境 action 的浮点约定；默认 10 Hz 栅格上的整数值在
    float32 中可精确表示，不引入额外表示误差。

    Args:
        hoprate_min: 网格起点（Hz）。
        hoprate_max: 网格上界（Hz）；须 ≥ hoprate_min。
        hoprate_step: 步长（Hz）；须 > 0。

    Returns:
        np.ndarray: 升序候选 hoprate，dtype=float32，长度 ≥ 1。

    Raises:
        ValueError: hoprate_step ≤ 0，或 hoprate_max < hoprate_min。
    """
    if hoprate_step <= 0:
        raise ValueError("hoprate_step must be positive")
    if hoprate_max < hoprate_min:
        raise ValueError("hoprate_max must be >= hoprate_min")

    count = int(np.floor((hoprate_max - hoprate_min) / hoprate_step)) + 1
    grid = hoprate_min + np.arange(count, dtype=np.float64) * hoprate_step
    if grid[-1] < hoprate_max and np.isclose((hoprate_max - grid[-1]), hoprate_step):
        grid = np.append(grid, hoprate_max)
    return grid.astype(np.float32)


def make_offsets(n_channels, mode, rng):
    """按 mode 生成一次 env.step 所需的 10 个信道偏移（float32）。

    offset 是 env.step 中对应 block 跳频图案的循环信道位移（见 fh_env 的
    _get_block_hopseq），取值范围 [0, n_channels − 1]。长度硬编码为 10，对应
    FHSSQPSKEnv 的 NUM_BLOCKS=10（一个 step 10 个 block）；长度不匹配时环境
    会直接抛 ValueError。

    Args:
        n_channels: 环境信道数；random 模式下作为采样上界（不含）。
        mode: "random"（均匀随机偏移，与导数 NBS 入口的采样方式一致，便于
            对照）或 "zeros"（全零偏移：不做额外位移，只保留 m 序列基础
            跳频图案，作为确定性基线）。
        rng: np.random.RandomState；显式传入以保证 --seed 可复现，且不与
            全局 np.random 的其它调用相互干扰（对比 train_speed_derivative.py
            直接使用全局 np.random）。

    Returns:
        np.ndarray: shape=(10,)、dtype=float32 的整数信道偏移。

    Raises:
        ValueError: mode 不是 "random" 或 "zeros"。
    """
    if mode == "random":
        return rng.randint(0, n_channels, size=10).astype(np.float32)
    if mode == "zeros":
        return np.zeros(10, dtype=np.float32)
    raise ValueError(f"Unsupported offset_mode: {mode}")


def sweep(args):
    """按升序遍历 hoprate 网格并逐档评测，落盘 CSV/NPZ/PNG。

    外层循环遍历 build_hoprate_grid 生成的候选点，内层在每个 hoprate 上跑
    args.steps_per_hoprate 个环境 step；每步生成 offset、env.step、取
    info["ber_blocks"] 均值与 reward，写入逐 step 明细与逐 hoprate 汇总。

    与导数 NBS 入口（train_speed_derivative.py）的差异：这里不做任何自适应
    选点，hoprate 序列完全由命令行网格决定，因此结果确定、可复现，适合作为
    阈值估计与 BER 曲线形状的对照基准；代价是 step 数随网格线性增长。

    提前结束语义：环境报告 episode 结束（terminated/truncated）时跳出内层
    循环，并因该档未跑满而整体终止扫描——后续档位大概率同样立即结束，继续
    扫下去没有信息量，还会让 CSV 混入大量不完整档位。

    Args:
        args: parse_args() 返回的命名空间；字段含义见 parse_args 的 docstring。

    副作用：
        在 --output_dir 下写 training_log.txt、两个 CSV、一个 NPZ 与若干 PNG
        （见模块 docstring 的"输出产物"）；向根 logger 输出 INFO 日志。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

    # ---- 环境：干扰机开关来自 CLI，其余沿用 ENV_CONFIG -------------------------
    # 先浅拷贝再覆盖，避免污染 settings 的全局配置；默认只开反应式干扰机，
    # 因为扫描的目的是刻画它对跳速的跟踪阈值。
    env_config = dict(settings.ENV_CONFIG)
    env_config["enable_reactive"] = args.enable_reactive
    env_config["enable_sweep"] = args.enable_sweep

    env = FHSSQPSKEnv(**env_config)
    n_channels = env.num_channels
    logger.info(
        f"Environment: {n_channels} channels, "
        f"hoprate ∈ [{env.hoprate_min:.0f}, {env.hoprate_max:.0f}] Hz, "
        f"reactive={env_config['enable_reactive']}, "
        f"sweep={env_config['enable_sweep']}, "
        f"offset_mode={args.offset_mode}"
    )

    # ---- 候选网格：升序、落 10 Hz 栅格（生成规则见 build_hoprate_grid）---------
    hoprates = build_hoprate_grid(args.hoprate_min, args.hoprate_max, args.hoprate_step)
    logger.info(
        f"Hoprate sweep: {len(hoprates)} values, "
        f"{args.steps_per_hoprate} env steps per value "
        f"({args.steps_per_hoprate * 10} blocks per value)"
    )

    # ---- 可复现性：offset 用独立 RandomState，环境侧随机性由全局种子决定 -------
    # 环境 reset 只为取得初始观测（本脚本不消费观测）；RandomState 单独实例化
    # 使 offset 序列与环境中 np.random 的调用互不干扰，同一种子下可复现。
    rng = np.random.RandomState(args.seed)
    _state_img, _info = env.reset()

    # 两类记录：step_records 逐 step 明细（CSV/NPZ/第 3 张图），
    # summary_records 逐 hoprate 归约（误差棒与最优档位判定）。
    step_records = []
    summary_records = []
    start_time = time.time()
    global_step = 0

    for hop_idx, hoprate in enumerate(hoprates, start=1):
        # 逐档容器：只累积本档的 step 级结果，档末归约为一条汇总记录。
        # 汇总里的 std 用 np.std 的总体标准差（ddof=0）——这里把本档的若干
        # step 视为该档的全部观测，而不是从更大总体中抽出的样本。
        hoprate_bers = []
        hoprate_rewards = []
        hoprate_used_values = []
        hop_start = time.time()

        for local_step in range(1, args.steps_per_hoprate + 1):
            global_step += 1
            step_start = time.time()

            # --- 单步：offset 采样 → env.step → BER/reward 观测 ---
            # 环境会把目标 hoprate 量化到 10 Hz 网格并裁剪到合法区间，
            # hoprate_used 是真实生效值（与 hoprate_target 一并记录，便于
            # 核对网格是否与量化栅格对齐）；ber_blocks 缺失时 mean_ber 回退 0。
            offsets = make_offsets(n_channels, args.offset_mode, rng)
            _obs, reward, terminated, truncated, info = env.step(
                {"hoprate": float(hoprate), "offsets": offsets}
            )

            ber_blocks = info.get("ber_blocks", [])
            mean_ber = float(np.mean(ber_blocks)) if ber_blocks else 0.0
            hoprate_used = float(info.get("hoprate_used", hoprate))
            mean_reward = float(reward)
            step_duration = time.time() - step_start

            hoprate_bers.append(mean_ber)
            hoprate_rewards.append(mean_reward)
            hoprate_used_values.append(hoprate_used)

            # 逐 step 明细：key 必须与 _save_outputs 中 DictWriter 的 fieldnames
            # 一一对应（缺 key 会写空、多 key 会报错）；offsets 以 Python list
            # 形式序列化进单元格（用 ast.literal_eval 即可读回）。
            record = {
                "global_step": global_step,
                "hoprate_target": float(hoprate),
                "hoprate_used": hoprate_used,
                "local_step": local_step,
                "mean_ber": mean_ber,
                "reward": mean_reward,
                "offsets": offsets.astype(int).tolist(),
                "duration_sec": step_duration,
            }
            step_records.append(record)

            logger.info(
                f"Hop {hop_idx:3d}/{len(hoprates)} | "
                f"Step {local_step}/{args.steps_per_hoprate} | "
                f"Target={float(hoprate):6.0f} Hz | "
                f"Used={hoprate_used:6.0f} Hz | "
                f"BER={mean_ber:.4f} | "
                f"Reward={mean_reward:.4f} | "
                f"Offsets={offsets.astype(int).tolist()} | "
                f"T={step_duration:.2f}s"
            )

            if terminated or truncated:
                logger.info("Episode terminated early.")
                break

        # 档末归约：ber_mean 是本档 steps_per_hoprate 个 step 的 BER 均值
        # （每个 step 本身已是 10 个 block 的均值），blocks 按每 step 10 个
        # block 折算，便于与单 block 口径的实验对照。
        summary = {
            "hoprate_target": float(hoprate),
            "hoprate_used_mean": float(np.mean(hoprate_used_values)) if hoprate_used_values else float(hoprate),
            "steps": len(hoprate_bers),
            "blocks": len(hoprate_bers) * 10,
            "ber_mean": float(np.mean(hoprate_bers)) if hoprate_bers else 0.0,
            "ber_std": float(np.std(hoprate_bers)) if hoprate_bers else 0.0,
            "reward_mean": float(np.mean(hoprate_rewards)) if hoprate_rewards else 0.0,
            "reward_std": float(np.std(hoprate_rewards)) if hoprate_rewards else 0.0,
            "duration_sec": time.time() - hop_start,
        }
        summary_records.append(summary)

        logger.info(
            f"Summary target={summary['hoprate_target']:.0f} Hz | "
            f"BER={summary['ber_mean']:.4f}±{summary['ber_std']:.4f} | "
            f"Reward={summary['reward_mean']:.4f}±{summary['reward_std']:.4f} | "
            f"Blocks={summary['blocks']} | "
            f"T={summary['duration_sec']:.2f}s"
        )

        # 本档未跑满 ⇒ 内层因 episode 结束提前退出：终止整个扫描，不再扫
        # 更高档位（原因见 sweep 的 docstring"提前结束语义"）。
        if len(hoprate_bers) < args.steps_per_hoprate:
            break

    # 落盘：两个 CSV 必写；NPZ 与 PNG 依赖是否有汇总记录，PNG 还受 --no_plots 控制
    _save_outputs(args.output_dir, step_records, summary_records, save_plots=not args.no_plots)

    elapsed = time.time() - start_time
    logger.info(f"Sweep complete. {len(step_records)} env steps in {elapsed:.1f}s")
    if summary_records:
        # 最低平均 BER 的档位：通常落在干扰机完全跟不上的一侧，可作为
        # "阈值上界"的粗参照（它不是阈值本身，只是曲线的最低平台）。
        best_ber = min(summary_records, key=lambda item: item["ber_mean"])
        logger.info(
            f"Lowest mean BER: hoprate={best_ber['hoprate_target']:.0f} Hz, "
            f"BER={best_ber['ber_mean']:.4f}, "
            f"reward={best_ber['reward_mean']:.4f}"
        )


def _save_outputs(output_dir, step_records, summary_records, save_plots=False):
    """把逐 step 明细与逐 hoprate 汇总写成 CSV，并按需保存 NPZ/PNG。

    写出顺序与依赖：两个 CSV 无条件写出（表头固定，必须与记录字典的 key
    一致）；NPZ 与 PNG 需要 summary_records 非空；PNG 还需 save_plots=True。
    数据不足时（例如 --steps_per_hoprate 0，或环境第一步就结束）提前 return，
    避免写空数组、画空图这类无意义产物。

    注意 CSV 是"先写表头再写行"，DictWriter 遇到记录里缺失/多余的字段会
    分别写空值或抛 ValueError，因此 step/summary 的 key 集合是隐性契约。

    Args:
        output_dir: 输出目录（须已存在）。
        step_records: 逐 step 明细字典列表。
        summary_records: 逐 hoprate 汇总字典列表。
        save_plots: 是否生成 PNG（--no_plots 时为 False）。

    副作用：
        写 hoprate_sweep_steps.csv、hoprate_sweep_summary.csv、
        hoprate_sweep_results.npz，并在 save_plots 时写 3 张 PNG。
    """
    step_csv = os.path.join(output_dir, "hoprate_sweep_steps.csv")
    summary_csv = os.path.join(output_dir, "hoprate_sweep_summary.csv")

    with open(step_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "global_step",
                "hoprate_target",
                "hoprate_used",
                "local_step",
                "mean_ber",
                "reward",
                "offsets",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(step_records)

    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "hoprate_target",
                "hoprate_used_mean",
                "steps",
                "blocks",
                "ber_mean",
                "ber_std",
                "reward_mean",
                "reward_std",
                "duration_sec",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_records)

    if not summary_records:
        return

    hoprates = np.array([r["hoprate_target"] for r in summary_records], dtype=np.float64)
    ber_mean = np.array([r["ber_mean"] for r in summary_records], dtype=np.float64)
    ber_std = np.array([r["ber_std"] for r in summary_records], dtype=np.float64)
    reward_mean = np.array([r["reward_mean"] for r in summary_records], dtype=np.float64)
    reward_std = np.array([r["reward_std"] for r in summary_records], dtype=np.float64)

    np.savez(
        os.path.join(output_dir, "hoprate_sweep_results.npz"),
        hoprates=hoprates,
        ber_mean=ber_mean,
        ber_std=ber_std,
        reward_mean=reward_mean,
        reward_std=reward_std,
    )

    if not save_plots:
        return

    _plot_results(output_dir, hoprates, ber_mean, ber_std, reward_mean, reward_std, step_records)


def _plot_results(output_dir, hoprates, ber_mean, ber_std, reward_mean, reward_std, step_records):
    """绘制 3 张 PNG 诊断图，matplotlib 风格与导数 NBS 入口保持一致。

    1. ber_vs_hoprate.png：BER 均值 ± 标准差随 hoprate 变化。误差棒来自各档
       内 step 之间的波动，反映单档观测噪声；默认每档仅 4 步，误差棒偏大属
       正常，加 --steps_per_hoprate 可收紧。
    2. reward_vs_hoprate.png：reward 均值 ± 标准差。reward 含 BER 惩罚项
       （见 settings.REWARD_CONFIG），故曲线形状应与 BER 大致相反。
    3. step_ber_hoprate.png：全局 step 轴上 BER（左轴，红）与 hoprate（右轴，
       蓝）的双轴轨迹，用于核对网格是否严格升序推进、每档是否跑满；无
       step_records 时跳过。

    Args:
        output_dir: 图片保存目录（须已存在）。
        hoprates: 各档目标 hoprate 数组（x 轴）。
        ber_mean / ber_std: 各档 BER 均值与标准差。
        reward_mean / reward_std: 各档 reward 均值与标准差。
        step_records: 逐 step 明细（第 3 张图用）。

    副作用：
        在 output_dir 下写出最多 3 个 PNG；每个 figure 用完即 close，避免
        长扫描累积内存。
    """
    # 1. BER 随 hoprate（误差棒 = 档内 step 间标准差）
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(hoprates, ber_mean, yerr=ber_std, fmt=".-", capsize=3,
                color="crimson", alpha=0.8)
    ax.set_title("Mean BER vs Hoprate")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("Mean BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber_vs_hoprate.png"))
    plt.close(fig)

    # 2. Reward 随 hoprate（形状应与 BER 相反）
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(hoprates, reward_mean, yerr=reward_std, fmt=".-", capsize=3,
                color="steelblue", alpha=0.8)
    ax.set_title("Mean Reward vs Hoprate")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "reward_vs_hoprate.png"))
    plt.close(fig)

    # 3. 逐 step 的 BER 与 hoprate 轨迹（双 y 轴，核对扫描是否按预期推进）
    if step_records:
        step_x = np.array([r["global_step"] for r in step_records], dtype=np.int64)
        step_ber = np.array([r["mean_ber"] for r in step_records], dtype=np.float64)
        step_hoprate = np.array([r["hoprate_used"] for r in step_records], dtype=np.float64)

        fig, ax1 = plt.subplots(figsize=(10, 4))
        ax1.plot(step_x, step_ber, ".-", color="crimson", alpha=0.8, label="Mean BER")
        ax1.set_xlabel("Global env step")
        ax1.set_ylabel("Mean BER", color="crimson")
        ax1.tick_params(axis="y", labelcolor="crimson")
        ax1.grid(True, alpha=0.3)

        ax2 = ax1.twinx()
        ax2.plot(step_x, step_hoprate, "--", color="steelblue", alpha=0.7, label="Hoprate")
        ax2.set_ylabel("Hoprate (Hz)", color="steelblue")
        ax2.tick_params(axis="y", labelcolor="steelblue")

        ax1.set_title("Step BER and Hoprate Trajectory")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "step_ber_hoprate.png"))
        plt.close(fig)


def parse_args():
    """解析命令行参数。

    默认值构成一次完整对照实验：10 → 1000 Hz、步长 10 Hz（约 100 档）、每档
    4 个 step（= 40 个 block/档），反应式干扰机开启、无差别干扰机关闭，offset
    随机、种子取 settings.RANDOM_SEED。

    Returns:
        argparse.Namespace: 网格（hoprate_min/max/step）、每档步数、offset
        模式、干扰机开关、输出目录与日志/绘图选项。

    说明：
        - description 与 --steps_per_hoprate 的 help 文案里写的 "2 env steps
          per hoprate" 是历史遗留，实际默认 steps_per_hoprate=4（字符串字面量
          按要求不修改，因此 help 中的 "(default: 2)" 不可信）。
        - --no_plots 是 store_true：默认生成 PNG，加该开关只跳过绘图，CSV/NPZ
          照常写出。
        - --enable_reactive/--enable_sweep 走 str_to_bool，非法取值直接报错。
    """
    parser = argparse.ArgumentParser(
        description="Sweep hoprate from 10 to 1000 Hz; run 2 env steps per hoprate by default."
    )
    parser.add_argument("--hoprate_min", type=float, default=10.0,
                        help="Minimum target hoprate in Hz (default: 10)")
    parser.add_argument("--hoprate_max", type=float, default=1000.0,
                        help="Maximum target hoprate in Hz (default: 1000)")
    parser.add_argument("--hoprate_step", type=float, default=10.0,
                        help="Hoprate increment in Hz (default: 10)")
    parser.add_argument("--steps_per_hoprate", type=int, default=4,
                        help="Environment steps per hoprate; each step has 10 blocks (default: 2)")
    parser.add_argument("--offset_mode", choices=["random", "zeros"], default="random",
                        help="Offset sequence mode for each env step (default: random)")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED,
                        help="Random seed for random offsets (default: settings.RANDOM_SEED)")
    parser.add_argument("--enable_reactive", type=str_to_bool, default=True,
                        help="Enable reactive jammer during sweep (default: true)")
    parser.add_argument("--enable_sweep", type=str_to_bool, default=False,
                        help="Enable sweep/comb jammer during sweep (default: false)")
    parser.add_argument("--output_dir", type=str, default="outputs/speed_sweep/0.5ms",
                        help="Directory for logs, CSV, NPZ, and plots")
    parser.add_argument("--no_plots", action="store_true",
                        help="Do not save PNG plots; CSV and NPZ outputs are still saved.")
    parser.add_argument("--log_file", type=str, default="training_log.txt",
                        help="Log filename inside output_dir")
    return parser.parse_args()


if __name__ == "__main__":
    # 先解析参数（--seed 可覆盖默认种子）再固定全局种子，最后开始扫描；
    # 种子影响环境内部噪声与 offset 序列，决定结果可复现性。
    args = parse_args()
    settings.set_random_seeds(args.seed)
    sweep(args)
