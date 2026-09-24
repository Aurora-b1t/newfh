"""FHSS 抗干扰训练的 MBPO 变体：step-level 多头 SAC + 奖励模型 MBPO 训练入口。

与 baseline（train_offsets.py）的关系：
    - 复用 baseline 的 v3 step-level transition 契约：一条 transition 为
      (当前 PSD 观测图, 当前 hoprate, 10 维 offsets, 10 维 block reward,
      标量 step reward, 下一 PSD 观测图, 下一 hoprate, done)；
      并直接复用 build_agent_and_env / setup_logger 等基础设施，
      保证两套训练在环境、动作空间与奖励定义上严格可比。
    - 核心差异在于数据来源与更新方式：baseline 只用真实环境 replay，
      本脚本额外训练一个奖励模型（r_predict_model.StepRewardEnsemble），
      按 MBPO 思路用模型合成 transition 扩充 SAC 的训练数据。

奖励模型（StepRewardEnsemble）：
    - 输入为当前 PSD 观测图、实际 hoprate 与完整 10 维 offsets，
      预测 10 维 block reward（multi-member ensemble + elite 筛选）。
    - 只预测即时 reward，不预测下一 PSD；rollout 生成的合成 transition
      复用真实 replay 起点的外生 next state / next hoprate / done——
      环境动力学与终止信号对 reward-only MBPO 是外生的，既无需也
      无法由奖励模型给出。
    - reward 经潜变量（logit）空间回归，采样结果由 sigmoid 映射回
      由 BER 端点与奖励配置导出的 [lower, upper] 有界区间，天然保证
      合成 reward 不会越过物理可达范围。

训练循环的因果顺序（每个真实环境 step）：
    1. 真实交互：agent 在固定 hoprate 下采样 10 维 offsets，env.step
       产出 v3 transition 并写入 real replay。
    2. 奖励模型拟合：每 model_train_freq（当前为 1）个真实 step 触发
       一次，用全部真实 replay 从头重新拟合（train/holdout 切分、
       holdout 早停，并记录每次拟合内逐 epoch 的 holdout MSE 与
       训练损失曲线）。
    3. Rollout 合成：每次拟合后立即以 real replay 中的状态为起点，
       SAC policy 采样 offsets，elite ensemble 在潜变量空间采样并经
       sigmoid 得到有界 block reward，生成一批合成 transition 追加到
       model replay（容量满后 FIFO 淘汰最旧样本）。
    4. SAC 混合更新：按目标真实样本比例 real_ratio 把 batch 拆成
       real / model 两部分（resolve_mixed_counts），各自经
       TensorDataset + DataLoader 采样后拼接成完整 batch，每个真实
       step 执行 update_iters_per_step 次梯度更新（batch 2048 适合
       A800 量级的显存与吞吐）。

离线 replay：默认严格拒绝与环境/干扰/奖励 metadata 不匹配的 v3
replay（防止把不同配置下的数据混入拟合与更新）；--allow_replay_
config_mismatch 可显式放宽到跨配置复用。

计时诊断：settings.TIMING_ENABLED 控制整体阶段计时（拟合内逐 epoch
耗时与 rollout 各子阶段耗时，见 mbpo_timing_suffix）；settings.
BACKWARD_TIMING_ENABLED 控制奖励模型训练内每 batch 反向传播的
CUDA event 计时（日志标签 [MBPO-BWD]，实现在 r_predict_model.model，
CUDA event 测的是真实 GPU kernel 时间，代价是每 batch 一次同步）。
bfloat16 / torch.compile / TF32 等运行时加速开关集中在
settings.MBPO_CONFIG 与 parse_args 的 --model_* 参数，只影响速度、
不改变训练预算。

输出产物（全部写入 args.output_dir）：
    - training_log.txt：主日志与逐 step hop sequence 记录；
    - reward.png / ber.png / loss.png / model_reward.png /
      model_disagreement.png / model_target_saturation_fraction.png：
      训练诊断曲线；
    - holdout_curves.npz / train_curves.npz：历次拟合的逐 epoch
      曲线（跨拟合长度不齐，以 NaN padding 对齐）；
    - holdout_curves/ 与 train_curves/ 目录（可选，--save_model_
      curve_figures）：每次拟合一张 PNG；
    - figures/：指定 step 的观测 PSD 瀑布图与逐 block PSD 图；
    - sac_inference.pt 与 reward_model_inference.pt：推理 checkpoint
      （后者仅在奖励模型至少完成过一次拟合时保存）。
"""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from SAC import (
    ReplayBuffer,
    SAC_CHECKPOINT_FORMAT_VERSION,
    SAC_POLICY_ARCHITECTURE,
    load_sac_inference_checkpoint,
    save_sac_inference_checkpoint,
)
from fh_env import save_waterfall_figure
from offline_replay import environment_metadata, load_replay_into_buffer
from r_predict_model import StepRewardEnsemble
from r_predict_model.mbpo_adapter import (
    REPLAY_FIELDS,
    replay_tensor_dataset,
    resolve_mixed_counts,
    rollout_reward_model,
    train_reward_model_from_replay,
)
import settings
from train_offsets import (
    build_agent_and_env,
    parse_optional_replay_path,
    replay_ready,
    setup_logger,
)

def _figure_steps(args, env, logger):
    """解析 settings.PLOT_CONFIG["figure_save_steps"] 并向环境注册按 step 抓图。

    只接受位于 [1, args.steps_per_episode] 内的整数（与 "Step i/N" 日志
    的 1-based 序号一致）；布尔值被显式排除，因为 bool 是 int 的子类，
    不排除的话 True 会被误当成 1。非法条目只告警并跳过，不影响训练。

    Args:
        args: 已解析的命令行参数（使用 output_dir 与 steps_per_episode）。
        env: FHSS 环境实例，需提供 enable_step_figure_capture。
        logger: 主 logger，用于输出忽略告警与最终生效的抓图 step 列表。

    Returns:
        (save_steps, figures_dir)：合法的抓图 step 集合与图片输出目录
        output_dir/figures；集合为空表示本次运行不抓图。
    """
    figures_dir = os.path.join(args.output_dir, "figures")
    save_steps = set()
    for value in settings.PLOT_CONFIG.get("figure_save_steps", []):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            logger.warning("figure_save_steps entry %r is not an integer; ignored.", value)
        elif not 1 <= int(value) <= args.steps_per_episode:
            logger.warning(
                "figure_save_steps entry %d is outside [1, %d]; ignored.",
                int(value),
                args.steps_per_episode,
            )
        else:
            save_steps.add(int(value))
    if save_steps:
        env.enable_step_figure_capture(sorted(save_steps), figures_dir)
        logger.info(
            "Figure saving enabled for steps %s -> %s",
            sorted(save_steps),
            figures_dir,
        )
    return save_steps, figures_dir


def _validate_args(args):
    """集中校验 MBPO 训练参数，任一约束被破坏即抛 ValueError（快速失败）。

    约束与原因：
        - 正整数参数（步数、replay/batch 容量、更新迭代数、模型训练
          batch、rollout batch、model replay 容量、ensemble 规模、隐藏层
          宽度、最大 epoch 数）：这些量直接决定数组形状、循环边界或
          replay 语义，非正值会在深处产生难定位的异常，故在入口拦截。
        - data_loader_workers >= 0：0 表示主进程内采样（默认，规避
          Windows 上 worker 进程 spawn 的开销），负数无意义。
        - num_elites <= num_networks：elite 是 ensemble 成员按 holdout
          排序选出的子集，超出成员总数则筛选无定义。
        - real_ratio ∈ [0, 1]：0 表示全用合成样本、1 表示退化为纯真实
          replay 的 baseline 式更新，区间外无法解释为比例。
        - model_holdout_ratio ∈ (0, 1) 取开区间：0 会让早停失去评判
          依据，1 则没有训练数据。
        - 早停 patience / min_improvement 非负、model_lr 为正、
          weight_decay 非负：与优化器和早停判据的数学假设一致。
        - settings.MBPO_CONFIG["rollout_length"] 必须为 1：本实现的
          奖励模型只做一步 reward 预测，且合成样本复用真实起点的
          外生 next state；多步 rollout 需要环境动力学模型，不在
          本模块范围内。

    Args:
        args: parse_args() 返回的命名空间。

    Raises:
        ValueError: 任一约束被违反时。
    """
    positive_names = (
        "steps_per_episode",
        "replay_size",
        "batch_size",
        "update_iters_per_step",
        "model_train_freq",
        "model_train_batch_size",
        "rollout_batch_size",
        "model_replay_size",
        "num_networks",
        "num_elites",
        "pred_hidden_size",
        "model_max_epochs",
    )
    for name in positive_names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if int(getattr(args, "data_loader_workers", 0)) < 0:
        raise ValueError("data_loader_workers cannot be negative.")
    if args.num_elites > args.num_networks:
        raise ValueError("num_elites cannot exceed num_networks.")
    if not 0.0 <= args.real_ratio <= 1.0:
        raise ValueError("real_ratio must be in [0, 1].")
    if not 0.0 < args.model_holdout_ratio < 1.0:
        raise ValueError("model_holdout_ratio must be between zero and one.")
    if args.model_patience < 0 or args.model_min_improvement < 0.0:
        raise ValueError("Model early-stop settings cannot be negative.")
    if args.model_lr <= 0.0 or args.model_weight_decay < 0.0:
        raise ValueError("Model learning rate/weight decay settings are invalid.")
    if int(settings.MBPO_CONFIG.get("rollout_length", 1)) != 1:
        raise ValueError("This reward-only MBPO implementation requires rollout_length=1.")


def _configure_torch_runtime(args):
    """按 --model_fast_math 开关启用可选的 GPU 数学加速，不改变训练预算。

    TF32、cuDNN benchmark autotune 与 float32 matmul "high" 档位只改变
    kernel 的数值路径与调度速度，不触及任何超参数、replay 切分或早停
    设置；默认关闭以保持严格的 float32 确定性语义。
    """
    if not bool(getattr(args, "model_fast_math", False)):
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # set_random_seeds() 会启用 deterministic cuDNN，而 deterministic
    # 模式会整体禁用 benchmark autotune；fast-math 运行必须显式清除
    # 该标志，否则拿不到自动调优后的 kernel。
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def reward_model_ready(real_buffer, model_train_batch_size):
    """判断 real replay 当前规模是否足以支撑奖励模型的 train/holdout 切分。

    阈值取 max(2, model_train_batch_size)：一方面至少 2 条 transition
    才能同时切出非空的训练与 holdout 子集；另一方面沿用模型训练
    batch 的规模作为最小可信样本量，避免用零星样本拟合出误导性的
    holdout 指标（也用于日志中的 ModelWarmup=x/y 显示）。
    """
    return real_buffer.size() >= max(2, int(model_train_batch_size))


def mbpo_timing_suffix(model_stats, rollout_stats):
    """把奖励模型拟合与 rollout 阶段的计时摘要格式化为可拼进日志的后缀。

    settings.TIMING_ENABLED 为 False 时返回空串，调用方无需感知计时
    开关即可拼接。有计时时输出形如（各段以 " | " 连接）：
        " | epoch_avg=..s(min=..,max=..,n=..) | fit=..s | "
        "rollout sample=..s policy=..s predict=..s add=..s total=..s"
    字段含义：
        - epoch_avg：本次拟合内逐 epoch 耗时的均值，min/max/n 反映
          早停截断带来的耗时波动与实际 epoch 数；
        - fit：整次拟合的 wall clock（含全部 epoch 与 holdout 评估）；
        - rollout 四个子阶段：sample（从 real replay 采样起点）、
          policy（SAC 批量采样 offsets）、predict（ensemble 前向 +
          潜变量采样）、add（写回 FIFO model replay），total 为四者
          之和（CPU 侧 wall clock，不含 CUDA 异步排队的隐藏时间）。

    Args:
        model_stats: train_reward_model_from_replay 返回的拟合统计，
            读取 epoch_times 与 fit_time_sec（未开计时时为空/None）。
        rollout_stats: rollout_reward_model 返回的统计，读取 timing
            （未开计时时为 None）。

    Returns:
        以 " | " 引导的计时后缀；无任何可用计时数据时返回空串。
    """
    if not settings.TIMING_ENABLED:
        return ""
    parts = []
    epoch_times = model_stats.get("epoch_times", [])
    if epoch_times:
        epoch_times = [float(value) for value in epoch_times]
        parts.append(
            f"epoch_avg={np.mean(epoch_times):.3f}s"
            f"(min={np.min(epoch_times):.3f},max={np.max(epoch_times):.3f},"
            f"n={len(epoch_times)})"
        )
    fit_time = model_stats.get("fit_time_sec")
    if fit_time is not None:
        parts.append(f"fit={float(fit_time):.2f}s")
    timing = rollout_stats.get("timing")
    if timing:
        parts.append(
            f"rollout sample={timing['sample_s']:.3f}s "
            f"policy={timing['policy_s']:.3f}s "
            f"predict={timing['predict_s']:.3f}s "
            f"add={timing['add_s']:.3f}s total={timing['total_s']:.3f}s"
        )
    return " | " + " | ".join(parts) if parts else ""


def should_train_reward_model(
    step_idx, real_buffer, model_train_batch_size, model_train_freq
):
    """应用"每 model_train_freq 个真实 step 拟合一次奖励模型"的调度。

    触发条件 = replay 规模越过 warm-up 阈值（reward_model_ready）且
    step_idx 是 model_train_freq 的整数倍。注意每次拟合用的都是全部
    真实 replay 而非最近 freq 条增量，因此该频率只控制拟合触发的
    开销节奏，不影响单次拟合的数据量与"从头重拟合"的语义。

    Args:
        step_idx: 当前真实环境 step 序号（1-based）。
        real_buffer: 真实 replay。
        model_train_batch_size: 模型训练 batch，兼作 warm-up 阈值。
        model_train_freq: 拟合触发周期（真实 step 数，当前为 1）。

    Returns:
        本 step 是否应执行奖励模型拟合（与随后的 rollout）。

    Raises:
        ValueError: model_train_freq 非正时。
    """
    if int(model_train_freq) <= 0:
        raise ValueError("model_train_freq must be positive.")
    return reward_model_ready(real_buffer, model_train_batch_size) and (
        int(step_idx) % int(model_train_freq) == 0
    )


def train(args):
    """MBPO+SAC 主训练循环：真实交互 → 奖励模型拟合 → rollout 合成 → 混合更新。

    总览（按执行顺序）：
        0. 参数校验（_validate_args）与 torch 运行时配置
           （_configure_torch_runtime）；建立输出目录与双通道日志
           （logger 走 console+文件，hop_logger 仅文件，二者共用同一
           FileHandler 写 training_log.txt）。
        1. 构建：环境 / SAC agent / real replay（复用 baseline 的
           build_agent_and_env，CUDA-only）；同构的 model replay
           （v3 契约、FIFO 容量 model_replay_size）；按 step 抓图注册；
           离线 v3 replay 加载（默认严格校验 transition 形状与
           环境/干扰/奖励 metadata，不匹配即拒绝）。
        2. 交互循环（step_idx = 1..steps_per_episode）：
           a. agent 在固定 hoprate 下采样 10 维 offsets，env.step 产出
              block rewards / BER / hop sequences，v3 transition 写入
              real replay；
           b. 若触发拟合调度（should_train_reward_model）：用全部真实
              replay 从头重新拟合奖励模型（holdout 早停 + 逐 epoch
              曲线记录），随即 rollout 一批合成 transition 追加进
              model replay（FIFO），记录分歧/饱和诊断与计时；
           c. 若 real replay 够一个完整 batch（replay_ready）：按
              real_ratio 解析 real/model 两侧 batch 数量，各自建
              DataLoader，本 step 执行 update_iters_per_step 次
              混合 batch 的 SAC 更新；
           d. 逐步记录指标与日志；terminated/truncated 提前结束。
        3. 收尾：诊断曲线与 holdout/train 曲线 NPZ 落盘，保存
           sac_inference.pt 与（若拟合过）reward_model_inference.pt
           推理 checkpoint，输出总耗时与平均 block reward。

    Args:
        args: parse_args() 返回的命名空间，字段语义见 parse_args 与
            _validate_args。
    """
    # ---- 0. 参数校验与运行时配置 ------------------------------------------
    # 先校验再训练：参数越界必须在构建昂贵的 ensemble / CUDA 上下文之前失败。
    _validate_args(args)
    # fast-math 只调整 kernel 数值路径与调度，需在首次大矩阵运算前生效。
    _configure_torch_runtime(args)
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger, hop_logger = setup_logger(log_path)
    logger.info("Output directory: %s", args.output_dir)
    logger.info("Log file: %s", log_path)

    env, agent, real_buffer, device, n_actions = build_agent_and_env(args)
    model_buffer = ReplayBuffer(
        capacity=args.model_replay_size,
        num_heads=env.num_blocks,
        n_actions=n_actions,
    )
    save_steps, figures_dir = _figure_steps(args, env, logger)
    state_img, _reset_info = env.reset()
    current_metadata = environment_metadata(
        settings.ENV_CONFIG,
        settings.JAMMER_CONFIG,
        settings.REWARD_CONFIG,
    )

    if args.offline_replay_path is None:
        logger.info(
            "Offline replay disabled; reward-model warm-up requires %d real steps.",
            max(2, args.model_train_batch_size),
        )
    else:
        loaded_count, replay_metadata = load_replay_into_buffer(
            args.offline_replay_path,
            real_buffer,
            expected_observation_shape=np.asarray(state_img).shape,
            expected_num_actions=n_actions,
            expected_num_blocks=env.num_blocks,
            current_environment_metadata=current_metadata,
            strict_environment_metadata=not args.allow_replay_config_mismatch,
            logger=logger,
        )
        logger.info(
            "Loaded %d offline step transitions from %s (mode=%s)",
            loaded_count,
            args.offline_replay_path,
            replay_metadata.get("hoprate_mode", "unknown"),
        )
    reward_model = StepRewardEnsemble(
        network_size=args.num_networks,
        elite_size=args.num_elites,
        num_heads=env.num_blocks,
        n_actions=n_actions,
        reward_config=settings.REWARD_CONFIG,
        hoprate_min=env.hoprate_min,
        hoprate_max=env.hoprate_max,
        hidden_size=args.pred_hidden_size,
        learning_rate=args.model_lr,
        weight_decay=args.model_weight_decay,
        device=device,
        precision=getattr(args, "model_precision", "float32"),
        compile_model=getattr(args, "model_compile", False),
        extra_pool=getattr(args, "model_extra_pool", False),
    )
    fixed_hoprate = float(
        int(
            round(
                np.clip(
                    settings.TRAIN_CONFIG["fixed_hoprate"],
                    env.hoprate_min,
                    env.hoprate_max,
                )
                / 10.0
            )
        )
        * 10
    )
    logger.info(
        "MBPO reward ensemble: models=%d elites=%d hidden=%d full-fit-frequency=%d",
        args.num_networks,
        args.num_elites,
        args.pred_hidden_size,
        args.model_train_freq,
    )
    logger.info(
        "Reward runtime: precision=%s compile=%s fast_math=%s",
        getattr(args, "model_precision", "float32"),
        getattr(args, "model_compile", False),
        getattr(args, "model_fast_math", False),
    )
    logger.info(
        "SAC Replay DataLoader: workers=%d pin_memory=%s batch=%d",
        args.data_loader_workers,
        args.data_loader_pin_memory,
        args.batch_size,
    )
    logger.info(
        "Model replay uses persistent FIFO retention with capacity=%d.",
        model_buffer.capacity,
    )
    logger.info(
        "Start MBPO+SAC training for %d environment steps at hoprate %.1f.",
        args.steps_per_episode,
        fixed_hoprate,
    )

    start_time = time.time()
    ep_block_rewards = []
    metrics = {
        "rewards": [],
        "bers": [],
        "actor_losses": [],
        "critic_losses": [],
        "model_rewards": [],
        "disagreements": [],
        "target_saturation_fractions": [],
    }
    holdout_curve_steps = []
    holdout_curve_history = []
    train_curve_steps = []
    train_curve_history = []
    last_model_stats = {}
    last_rollout_stats = {}

    for step_idx in range(1, args.steps_per_episode + 1):
        step_start = time.time()
        offsets = agent.take_action(state_img, fixed_hoprate)
        if step_idx in save_steps:
            save_waterfall_figure(
                np.asarray(state_img),
                os.path.join(figures_dir, f"step_{step_idx:03d}_obs.png"),
                title=f"Step {step_idx} - Pre-action Observation (100 ms)",
            )

        next_state_img, step_reward, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )
        block_rewards = np.asarray(info.get("block_rewards", []), dtype=np.float32)
        if block_rewards.shape != (env.num_blocks,):
            raise RuntimeError("Environment returned an invalid block reward vector.")
        if not np.isclose(
            float(step_reward),
            float(np.mean(block_rewards)),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("Environment reward is inconsistent with block rewards.")
        ber_blocks = np.asarray(info.get("ber_blocks", []), dtype=np.float32)
        mean_step_ber = float(np.mean(ber_blocks)) if ber_blocks.size else 0.0
        done = bool(terminated or truncated)
        used_hoprate = float(info.get("hoprate_used", fixed_hoprate))
        real_buffer.add(
            state_img,
            used_hoprate,
            offsets,
            block_rewards,
            next_state_img,
            fixed_hoprate,
            done,
        )
        ep_block_rewards.extend(block_rewards.tolist())
        state_img = next_state_img

        model_fit_time = 0.0
        model_ready = reward_model_ready(
            real_buffer, args.model_train_batch_size
        )
        if should_train_reward_model(
            step_idx,
            real_buffer,
            args.model_train_batch_size,
            args.model_train_freq,
        ):
            model_start = time.time()
            last_model_stats = train_reward_model_from_replay(
                reward_model,
                real_buffer,
                batch_size=args.model_train_batch_size,
                holdout_ratio=args.model_holdout_ratio,
                patience=args.model_patience,
                max_epochs=args.model_max_epochs,
                min_improvement=args.model_min_improvement,
            )
            last_rollout_stats = rollout_reward_model(
                reward_model,
                agent,
                real_buffer,
                model_buffer,
                args.rollout_batch_size,
                settings.REWARD_CONFIG,
                deterministic_model=args.deterministic_model_rollout,
            )
            model_fit_time = time.time() - model_start
            holdout_curve_steps.append(step_idx)
            holdout_curve_history.append(last_model_stats["holdout_curves"])
            if args.save_model_curve_figures:
                _save_holdout_curves_figure(
                    args.output_dir, step_idx, last_model_stats["holdout_curves"]
                )
            train_curve_steps.append(step_idx)
            train_curve_history.append(last_model_stats["train_curves"])
            if args.save_model_curve_figures:
                _save_train_curves_figure(
                    args.output_dir, step_idx, last_model_stats["train_curves"]
                )
            logger.info(
                "Reward model | holdout=%.6f | elites=%s | epochs=%s | "
                "rollout=%d | model_buf=%d->%d/%d | fifo_evicted=%d | "
                "disagreement=%.6f(p95=%.6f) | "
                "target_sat=%.2f%% | T=%.2fs%s",
                last_model_stats["holdout_loss_mean"],
                last_model_stats["elite_model_idxes"],
                last_model_stats["epochs"],
                last_rollout_stats["generated"],
                last_rollout_stats["model_buffer_size_before"],
                last_rollout_stats["model_buffer_size_after"],
                last_rollout_stats["model_buffer_capacity"],
                last_rollout_stats["fifo_evicted"],
                last_rollout_stats["disagreement_mean"],
                last_rollout_stats["disagreement_p95"],
                100.0 * last_model_stats["target_saturation_fraction"],
                model_fit_time,
                mbpo_timing_suffix(last_model_stats, last_rollout_stats),
            )

        train_stats = {}
        if replay_ready(real_buffer, args.batch_size):
            real_count, model_count = resolve_mixed_counts(
                real_buffer.size(),
                model_buffer.size(),
                args.batch_size,
                args.real_ratio,
            )
            loader_kwargs = {
                "shuffle": True,
                "drop_last": True,
                "num_workers": args.data_loader_workers,
                "pin_memory": args.data_loader_pin_memory,
                "persistent_workers": bool(args.data_loader_workers > 0),
            }
            real_loader = (
                DataLoader(
                    replay_tensor_dataset(real_buffer),
                    batch_size=real_count,
                    **loader_kwargs,
                )
                if real_count
                else None
            )
            model_loader = (
                DataLoader(
                    replay_tensor_dataset(model_buffer),
                    batch_size=model_count,
                    **loader_kwargs,
                )
                if model_count
                else None
            )
            real_iterator = iter(real_loader) if real_loader is not None else None
            model_iterator = iter(model_loader) if model_loader is not None else None
            last_update_index = args.update_iters_per_step - 1
            for update_index in range(args.update_iters_per_step):
                batch_parts = []
                if real_iterator is not None:
                    try:
                        batch_parts.append(next(real_iterator))
                    except StopIteration:
                        real_iterator = iter(real_loader)
                        batch_parts.append(next(real_iterator))
                if model_iterator is not None:
                    try:
                        batch_parts.append(next(model_iterator))
                    except StopIteration:
                        model_iterator = iter(model_loader)
                        batch_parts.append(next(model_iterator))
                batch = {
                    field: torch.cat(
                        [part[field_index] for part in batch_parts], dim=0
                    )
                    for field_index, field in enumerate(REPLAY_FIELDS)
                }
                train_stats = agent.update(
                    batch,
                    return_stats=update_index == last_update_index,
                )

        metrics["rewards"].append(float(step_reward))
        metrics["bers"].append(mean_step_ber)
        metrics["actor_losses"].append(train_stats.get("actor_loss", np.nan))
        metrics["critic_losses"].append(train_stats.get("critic1_loss", np.nan))
        metrics["model_rewards"].append(
            last_rollout_stats.get("reward_mean", np.nan)
        )
        metrics["disagreements"].append(
            last_rollout_stats.get("disagreement_mean", np.nan)
        )
        metrics["target_saturation_fractions"].append(
            last_model_stats.get("target_saturation_fraction", np.nan)
        )

        hop_sequences = info.get("hop_sequences", [])
        first_channels = [sequence[0] for sequence in hop_sequences if sequence]
        hop_logger.info("Step %d HopSequences: %s", step_idx, hop_sequences)
        log_message = (
            f"Step {step_idx}/{args.steps_per_episode} | "
            f"Offsets: {offsets.astype(int).tolist()} | FirstCh: {first_channels} | "
            f"Rew: {float(step_reward):.4f} | BER: {mean_step_ber:.4f} | "
            f"RealBuf={real_buffer.size()} | ModelBuf={model_buffer.size()}"
        )
        if train_stats:
            log_message += (
                f" | Loss: A={train_stats['actor_loss']:.3f}, "
                f"C={train_stats['critic1_loss']:.3f}, "
                f"Alpha={train_stats['alpha']:.5f}"
            )
        else:
            log_message += f" | SACWarmup={real_buffer.size()}/{args.batch_size}"
        if not model_ready:
            log_message += (
                f" | ModelWarmup={real_buffer.size()}/"
                f"{max(2, args.model_train_batch_size)}"
            )
        log_message += f" | T={time.time() - step_start:.2f}s"
        logger.info(log_message)

        if done:
            logger.info("Episode terminated early.")
            break

    save_plots(args.output_dir, metrics, logger)
    _save_holdout_curves_npz(
        args.output_dir, holdout_curve_steps, holdout_curve_history, logger
    )
    _save_train_curves_npz(
        args.output_dir, train_curve_steps, train_curve_history, logger
    )
    checkpoint_metadata = dict(current_metadata)
    checkpoint_metadata.update(
        {
            "num_actions": n_actions,
            "num_blocks": env.num_blocks,
            "fixed_hoprate": fixed_hoprate,
        }
    )
    sac_checkpoint = os.path.join(args.output_dir, "sac_inference.pt")
    save_sac_inference_checkpoint(
        agent,
        sac_checkpoint,
        np.asarray(state_img).shape,
        env.hoprate_min,
        env.hoprate_max,
        checkpoint_metadata,
    )
    logger.info("Saved SAC inference checkpoint to %s", sac_checkpoint)
    if reward_model.is_fitted:
        reward_checkpoint = os.path.join(
            args.output_dir, "reward_model_inference.pt"
        )
        reward_model.save_checkpoint(reward_checkpoint, checkpoint_metadata)
        logger.info("Saved reward-model inference checkpoint to %s", reward_checkpoint)
    else:
        logger.warning(
            "Reward model never left warm-up; no reward-model checkpoint was saved."
        )

    mean_episode_reward = (
        float(np.mean(ep_block_rewards)) if ep_block_rewards else 0.0
    )
    logger.info(
        "Total Time: %.2fs | Mean Ep Reward: %.4f",
        time.time() - start_time,
        mean_episode_reward,
    )


def _save_holdout_curves_figure(output_dir, step_idx, holdout_curves):
    """为单次奖励模型拟合保存逐 epoch 的 holdout MSE 曲线图。

    ``holdout_curves`` 为 ``[num_members][num_epochs]`` 嵌套列表，每个成员
    一条曲线（含训练前的初始评估点），写入 ``holdout_curves/`` 子目录，
    文件名按训练 step 编号。
    """
    curves_dir = os.path.join(output_dir, "holdout_curves")
    os.makedirs(curves_dir, exist_ok=True)
    plt.figure()
    for member_idx, curve in enumerate(holdout_curves):
        plt.plot(
            range(len(curve)),
            curve,
            marker=".",
            label=f"Member {member_idx}",
        )
    plt.title(f"Reward-Model Holdout MSE (Step {step_idx})")
    plt.xlabel("Epoch")
    plt.ylabel("Holdout MSE")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(curves_dir, f"holdout_step_{step_idx:04d}.png"))
    plt.close()


def _save_holdout_curves_npz(output_dir, fit_steps, holdout_curve_history, logger):
    """把历次拟合的 holdout 曲线汇总保存为 ``holdout_curves.npz``。

    各次拟合的 epoch 数可能不同（早停），因此统一用 NaN 右填充到
    ``[num_fits, num_members, max_epochs]``；同时保存对应的 step 索引
    ``fit_steps``。保存失败只记错误日志，不中断已完成训练。
    """
    if not holdout_curve_history:
        logger.info("No reward-model fits ran; holdout_curves.npz was not saved.")
        return
    try:
        num_fits = len(holdout_curve_history)
        num_members = len(holdout_curve_history[0])
        max_len = max(
            len(curve) for curves in holdout_curve_history for curve in curves
        )
        padded = np.full((num_fits, num_members, max_len), np.nan, dtype=np.float64)
        for fit_idx, curves in enumerate(holdout_curve_history):
            for member_idx, curve in enumerate(curves):
                padded[fit_idx, member_idx, : len(curve)] = curve
        np.savez(
            os.path.join(output_dir, "holdout_curves.npz"),
            holdout_curves=padded,
            fit_steps=np.asarray(fit_steps, dtype=np.int64),
        )
        logger.info(
            "Saved %d per-fit holdout curves to %s.",
            num_fits,
            os.path.join(output_dir, "holdout_curves.npz"),
        )
    except Exception as exc:
        logger.error("Saving holdout curves failed: %s", exc)


def _save_train_curves_figure(output_dir, step_idx, train_curves):
    """为单次奖励模型拟合保存逐 epoch 的训练 NLL 曲线图（每成员一条）。

    ``train_curves`` 为 ``[num_members][num_epochs]`` 嵌套列表，写入
    ``train_curves/`` 子目录，文件名按训练 step 编号。
    """
    curves_dir = os.path.join(output_dir, "train_curves")
    os.makedirs(curves_dir, exist_ok=True)
    plt.figure()
    for member_idx, curve in enumerate(train_curves):
        plt.plot(
            range(1, len(curve) + 1),
            curve,
            marker=".",
            label=f"Member {member_idx}",
        )
    plt.title(f"Reward-Model Train NLL (Step {step_idx})")
    plt.xlabel("Epoch")
    plt.ylabel("Train NLL")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(curves_dir, f"train_step_{step_idx:04d}.png"))
    plt.close()


def _save_train_curves_npz(output_dir, fit_steps, train_curve_history, logger):
    """把历次拟合的训练 NLL 曲线汇总保存为 ``train_curves.npz``。

    与 holdout 版本相同：不同拟合的 epoch 数可能不同，统一 NaN 右填充为
    ``[num_fits, num_members, max_epochs]``，并保存 step 索引 ``fit_steps``。
    """
    if not train_curve_history:
        logger.info("No reward-model fits ran; train_curves.npz was not saved.")
        return
    try:
        num_fits = len(train_curve_history)
        num_members = len(train_curve_history[0])
        max_len = max(
            len(curve) for curves in train_curve_history for curve in curves
        )
        padded = np.full((num_fits, num_members, max_len), np.nan, dtype=np.float64)
        for fit_idx, curves in enumerate(train_curve_history):
            for member_idx, curve in enumerate(curves):
                padded[fit_idx, member_idx, : len(curve)] = curve
        np.savez(
            os.path.join(output_dir, "train_curves.npz"),
            train_curves=padded,
            fit_steps=np.asarray(fit_steps, dtype=np.int64),
        )
        logger.info(
            "Saved %d per-fit training curves to %s.",
            num_fits,
            os.path.join(output_dir, "train_curves.npz"),
        )
    except Exception as exc:
        logger.error("Saving training curves failed: %s", exc)


def _save_curve(output_dir, values, filename, title, ylabel, color=None):
    """把一条逐 step 序列画成单曲线 PNG 并保存到 ``output_dir/filename``。"""
    plt.figure()
    plt.plot(values, color=color)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_plots(output_dir, metrics, logger):
    """保存全部训练诊断曲线（reward/BER/loss/模型分歧/目标饱和比例）。

    绘图失败只记日志，不让一次 I/O 异常使已完成的训练作废。
    """
    try:
        _save_curve(
            output_dir, metrics["rewards"], "reward.png", "Mean Step Reward", "Reward"
        )
        _save_curve(
            output_dir, metrics["bers"], "ber.png", "Mean Step BER", "BER", "r"
        )
        plt.figure()
        plt.plot(metrics["actor_losses"], label="Actor Loss", alpha=0.7)
        plt.plot(metrics["critic_losses"], label="Critic Loss", alpha=0.7)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()
        _save_curve(
            output_dir,
            metrics["model_rewards"],
            "model_reward.png",
            "Synthetic Reward Mean",
            "Reward",
        )
        _save_curve(
            output_dir,
            metrics["disagreements"],
            "model_disagreement.png",
            "Elite Ensemble Disagreement",
            "Mean Std",
        )
        _save_curve(
            output_dir,
            metrics["target_saturation_fractions"],
            "model_target_saturation_fraction.png",
            "Reward-Model Target Saturation Fraction",
            "Fraction",
        )
        logger.info("Plots saved to %s.", output_dir)
    except Exception as exc:
        logger.error("Plotting failed: %s", exc)


def parse_args():
    """解析命令行参数；SAC/MBPO/离线 replay 默认值均取自 settings.py。

    MBPO 专属参数（--model_*、--num_networks/--num_elites、--real_ratio、
    --rollout_batch_size、--model_replay_size 等）覆盖 MBPO_CONFIG；
    --offline_replay_path 支持 none 表示纯在线；布尔开关（--model_compile、
    --model_extra_pool、--deterministic_model_rollout）为
    argparse.BooleanOptionalAction，用 --no-xxx 关闭。
    """
    parser = argparse.ArgumentParser(
        description="Step-level multi-head SAC with an MBPO reward ensemble"
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=settings.TRAIN_CONFIG["steps_per_episode"],
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/mbpo/comb/0"
    )
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    parser.add_argument("--actor_lr", type=float, default=settings.SAC_CONFIG["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=settings.SAC_CONFIG["critic_lr"])
    parser.add_argument("--alpha_lr", type=float, default=settings.SAC_CONFIG["alpha_lr"])
    parser.add_argument("--tau", type=float, default=settings.SAC_CONFIG["tau"])
    parser.add_argument("--gamma", type=float, default=settings.SAC_CONFIG["gamma"])
    parser.add_argument(
        "--replay_size", type=int, default=settings.BUFFER_CONFIG["capacity"]
    )
    parser.add_argument(
        "--batch_size", type=int, default=settings.BUFFER_CONFIG["batch_size"]
    )
    parser.add_argument(
        "--update_iters_per_step",
        type=int,
        default=settings.TRAIN_CONFIG["update_iters_per_step"],
    )
    parser.add_argument(
        "--offline_replay_path",
        type=parse_optional_replay_path,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Offline v3 replay path, or 'none' for online-only warm-up.",
    )
    parser.add_argument(
        "--allow_replay_config_mismatch",
        action="store_true",
        help="Allow v3 replay generated from different env/jammer/reward settings.",
    )

    parser.add_argument(
        "--real_ratio", type=float, default=settings.MBPO_CONFIG["real_ratio"]
    )
    parser.add_argument(
        "--model_train_freq",
        type=int,
        default=settings.MBPO_CONFIG["model_train_freq"],
    )
    parser.add_argument(
        "--rollout_batch_size",
        type=int,
        default=settings.MBPO_CONFIG["rollout_batch_size"],
    )
    parser.add_argument(
        "--model_replay_size",
        type=int,
        default=settings.MBPO_CONFIG["model_replay_size"],
    )
    parser.add_argument(
        "--num_networks",
        type=int,
        default=settings.MBPO_CONFIG["num_networks"],
    )
    parser.add_argument(
        "--num_elites", type=int, default=settings.MBPO_CONFIG["num_elites"]
    )
    parser.add_argument(
        "--pred_hidden_size",
        type=int,
        default=settings.MBPO_CONFIG["hidden_size"],
    )
    parser.add_argument(
        "--model_lr",
        type=float,
        default=settings.MBPO_CONFIG["learning_rate"],
    )
    parser.add_argument(
        "--model_weight_decay",
        type=float,
        default=settings.MBPO_CONFIG["weight_decay"],
    )
    parser.add_argument(
        "--model_train_batch_size",
        type=int,
        default=settings.MBPO_CONFIG["model_train_batch_size"],
    )
    parser.add_argument(
        "--model_holdout_ratio",
        type=float,
        default=settings.MBPO_CONFIG["holdout_ratio"],
    )
    parser.add_argument(
        "--model_patience",
        type=int,
        default=settings.MBPO_CONFIG["early_stop_patience"],
    )
    parser.add_argument(
        "--model_max_epochs",
        type=int,
        default=settings.MBPO_CONFIG["max_epochs"],
    )
    parser.add_argument(
        "--model_min_improvement",
        type=float,
        default=settings.MBPO_CONFIG["min_improvement"],
    )
    parser.add_argument(
        "--data_loader_workers",
        type=int,
        default=settings.MBPO_CONFIG.get("data_loader_workers", 0),
        help="DataLoader workers for SAC replay snapshots.",
    )
    parser.add_argument(
        "--data_loader_pin_memory",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG.get("data_loader_pin_memory", False),
        help="Pin CPU DataLoader batches before asynchronous H2D copies.",
    )
    parser.add_argument(
        "--model_precision",
        choices=("float32", "bfloat16", "float16"),
        default=settings.MBPO_CONFIG.get("model_precision", "float32"),
        help="Reward-model autocast precision; does not change training budgets.",
    )
    parser.add_argument(
        "--model_fast_math",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG.get("model_fast_math", False),
        help="Enable TF32 and cuDNN autotuning on CUDA.",
    )
    parser.add_argument(
        "--model_compile",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG.get("model_compile", False),
        help="Compile the persistent reward-model graph with torch.compile.",
    )
    parser.add_argument(
        "--model_extra_pool",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG.get("model_extra_pool", False),
        help="Add a second 2x2 pool before conv_fc (4x fewer parameters, faster epochs).",
    )
    parser.add_argument(
        "--save_model_curve_figures",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG["save_curve_figures"],
        help="Write a PNG pair after every reward-model fit.",
    )
    parser.add_argument(
        "--deterministic_model_rollout", action="store_true"
    )
    return parser.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    train(parse_args())
