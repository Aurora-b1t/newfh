"""
Baseline 训练入口：十头离散 SAC 直接在真实 FHSS 环境上学习 10 维 offsets。

【baseline 语义】
    本脚本是 offsets-only baseline：与 FHSSQPSKEnv 直接交互，一次 env.step()
    在环境内部执行 10 个 100 ms 通信 block；动作是 10 维 offset，每维取
    [0, num_channels-1] 的离散信道索引（叠加在 m-sequence 基础跳频序列上，
    实际信道 = (基础信道 + offset) % num_channels）；奖励逐 block 计算：
    base_reward - ber_penalty*BER - hoprate_penalty*hoprate（系数见
    settings.REWARD_CONFIG）；观测为 100×100 的 PSD waterfall。SAC 的
    actor/critic 均为十头结构（num_heads = num_blocks = 10），ReplayBuffer
    以 step 为单位存完整转移（一条 = 10 个 offsets + 10 个 block rewards），
    梯度更新只用真实交互数据。hoprate 固定为
    settings.TRAIN_CONFIG["fixed_hoprate"]，不是被学习的动作维。

【与 MBPO 入口（train_mbpo.py）的差异】
    - 不训练 r_predict_model.StepRewardEnsemble 奖励模型、不做 model
      rollout，也不维护 model replay；
    - 离线 replay 的 metadata 校验策略不同：baseline 调用
      load_replay_into_buffer 时保持 strict_environment_metadata=False
      （该函数默认值），档案里记录的环境/干扰/奖励配置与当前 settings
      不一致时只打 warning、不拒绝加载；MBPO 入口显式开启严格模式，
      不匹配直接抛 ValueError。

【离线 replay 加载行为】
    在训练循环开始（即首次梯度更新可能发生）之前，从
    args.offline_replay_path 加载 v3 格式的 .npz 档案（由
    generate_offline_replay.py 生成，metadata 记录生成时的
    env/jammer/reward 配置与 observation 形状、block 数等）。加载成功后
    buffer 立即可采样，第 1 个 step 起就能做梯度更新；传入
    --offline_replay_path none（或 null/空串，见 parse_optional_replay_path）
    则跳过加载，纯在线收集，直到 buffer 攒够 batch_size 条才开始更新
    （日志中显示为 Warmup）。

【输出产物】（均在 --output_dir 下）
    - <log_file>（默认 training_log.txt）：完整训练日志，含逐 step 的完整
      hop sequences（仅写文件，不刷控制台）；
    - figures/：settings.PLOT_CONFIG["figure_save_steps"] 指定的 step 处，
      保存动作前观测 waterfall（step_XXX_obs.png）与环境逐 block 的 PSD
      图（step_XXX_block_YY.png）；
    - reward.png / ber.png / loss.png：训练曲线；
    - sac_inference.pt：仅含推理权重（actor）的 checkpoint，附带
      environment metadata，供脱离训练环境的部署/评估加载。
"""

import argparse
import os
import time
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from fh_env import FHSSQPSKEnv, save_waterfall_figure
from SAC import SAC, ReplayBuffer, save_sac_inference_checkpoint
from offline_replay import (
    environment_metadata,
    load_replay_into_buffer,
)
import settings

def setup_logger(log_file):
    """配置日志系统：根 logger 同时输出控制台与文件，hop 序列只写文件。

    返回 ``(logger, hop_logger)``：
    - ``logger``：根 logger，控制台 + 文件双 handler，用于常规训练日志；
    - ``hop_logger``（名称 ``fh.hop_sequences``）：关闭 propagate、只挂文件
      handler，用于逐 step 的完整 10 block × 10 hop 信道序列——避免刷屏，
      但完整数据仍落在同一个 training_log.txt 里。
    """
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    hop_logger = logging.getLogger("fh.hop_sequences")
    hop_logger.setLevel(logging.INFO)
    hop_logger.propagate = False
    hop_logger.handlers.clear()
    hop_logger.addHandler(file_handler)

    return logger, hop_logger


def log_active_comb_channels(env, logger=None):
    """comb 干扰激活时，把两组相位信道配置各记录一次（控制台+日志文件）。

    逐 step 日志不重复输出该配置，因此在启动时打印一次便于实验溯源。
    返回是否实际打印。
    """
    if not (
        env.enable_sweep
        and env.sweep is not None
        and env.sweep.comb_enabled
    ):
        return False

    logger = logger or logging.getLogger()
    channels_phase0, channels_phase1 = env.sweep.comb_channels
    logger.info(
        "Comb channels_phase0=%s | channels_phase1=%s",
        channels_phase0,
        channels_phase1,
    )
    return True

def build_agent_and_env(args, env_config=None, jammer_config=None):
    """构建环境、SAC agent 与 replay buffer（baseline 训练的全部组件）。

    Args:
        args: 命令行参数（读取 actor_lr/critic_lr/alpha_lr/tau/gamma/replay_size）。
        env_config: 环境配置；None 时使用 ``settings.ENV_CONFIG``。
        jammer_config: 干扰机配置；None 时由 FHSSQPSKEnv 使用 settings 默认值。

    Returns:
        ``(env, agent, buffer, device, n_actions)``；device 固定为 CUDA，
        n_actions = 信道数（每个 offset 头的离散动作数）。

    说明：目标熵取 ``log(n_actions) * target_entropy_ratio``——离散动作空间
    用 log(|A|) 作参考尺度，ratio 控制探索强度；num_heads = 环境 block 数。
    """
    # -------------------------------------------------------------------------
    # 1. 设备配置（仅 CUDA：本入口不做 CPU 回退）
    # -------------------------------------------------------------------------
    device = torch.device("cuda")
    logging.info(f"Training Device: GPU ({torch.cuda.get_device_name(0)})")

    # -------------------------------------------------------------------------
    # 2. 环境初始化
    # -------------------------------------------------------------------------
    # 配置来自 settings（或调用方显式传入的覆盖副本）
    env_config = settings.ENV_CONFIG if env_config is None else env_config
    env = FHSSQPSKEnv(**env_config, jammer_config=jammer_config)
    log_active_comb_channels(env)

    # 离散动作数 = 信道数
    n_actions = env.num_channels
    num_blocks = env.num_blocks
    logging.info(
        "Environment Initialized. Num Channels/Actions: %d, Offset Heads: %d",
        n_actions,
        num_blocks,
    )

    # -------------------------------------------------------------------------
    # 3. 构建 SAC agent
    # -------------------------------------------------------------------------
    # 目标熵：连续动作空间通常取 -dim(A)，离散空间取 log(|A|) 的比例
    target_entropy = np.log(n_actions) * settings.SAC_CONFIG["target_entropy_ratio"]
    
    agent = SAC(
        n_actions=n_actions,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        alpha_lr=args.alpha_lr,
        target_entropy=target_entropy,
        tau=args.tau,
        gamma=args.gamma,
        device=device,
        num_heads=num_blocks,
        hoprate_min=env.hoprate_min,
        hoprate_max=env.hoprate_max,
    )

    # -------------------------------------------------------------------------
    # 4. Replay buffer（step-level：一条 = 10 offsets + 10 block rewards）
    # -------------------------------------------------------------------------
    buffer = ReplayBuffer(
        capacity=args.replay_size,
        num_heads=num_blocks,
        n_actions=n_actions,
    )

    return env, agent, buffer, device, n_actions


def replay_ready(buffer, batch_size):
    """replay 中的完整 step 数量是否够采样一个 batch（够则开始梯度更新）。"""
    return buffer.size() >= int(batch_size)


def train(args):
    """baseline 训练主流程。

    顺序：建 logger → 建环境/agent/replay → 配置 figure capture →
    （可选）加载离线 replay → 逐步交互训练 → 落盘曲线与推理 checkpoint。

    每个 step：actor 一次前向采样 10 个 offset → env.step 执行 10 个 block →
    校验 block reward 一致性并写入 replay → 若 replay 够 batch 则做
    ``update_iters_per_step`` 次梯度更新 → 记录日志/曲线。
    """
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger, hop_logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

    # 构建训练组件
    env, agent, buffer, device, n_actions = build_agent_and_env(args)

    # -------------------------------------------------------------------------
    # step 触发的图片保存（动作前观测 + 逐 block PSD）
    # -------------------------------------------------------------------------
    figures_dir = os.path.join(args.output_dir, "figures")
    save_steps = set()
    for s in settings.PLOT_CONFIG.get("figure_save_steps", []):
        if isinstance(s, bool) or not isinstance(s, (int, np.integer)):
            logger.warning("figure_save_steps entry %r is not an integer; ignored.", s)
        elif not 1 <= int(s) <= args.steps_per_episode:
            logger.warning(
                "figure_save_steps entry %d is outside [1, %d]; ignored.",
                int(s), args.steps_per_episode,
            )
        else:
            save_steps.add(int(s))
    if save_steps:
        env.enable_step_figure_capture(sorted(save_steps), figures_dir)
        logger.info(
            "Figure saving enabled for steps %s -> %s",
            sorted(save_steps), figures_dir,
        )

    state_img, info = env.reset()
    if args.offline_replay_path is None:
        logger.info(
            "Offline replay disabled; collecting online transitions until "
            "the buffer reaches batch_size=%d.",
            args.batch_size,
        )
    else:
        # baseline 保持 warn-only 兼容行为：metadata 不一致只警告不拒绝
        # （strict_environment_metadata 使用默认 False）。
        loaded_count, replay_metadata = load_replay_into_buffer(
            args.offline_replay_path,
            buffer,
            expected_observation_shape=np.asarray(state_img).shape,
            expected_num_actions=n_actions,
            expected_num_blocks=env.num_blocks,
            current_environment_metadata=environment_metadata(
                settings.ENV_CONFIG,
                settings.JAMMER_CONFIG,
                settings.REWARD_CONFIG,
            ),
            logger=logging.getLogger(),
        )
        logger.info(
            "Loaded %d offline step transitions from %s (mode=%s)",
            loaded_count,
            args.offline_replay_path,
            replay_metadata.get("hoprate_mode", "unknown"),
        )

    # 在线训练使用固定跳速（与既有实验保持一致），hoprate 不是被学习的动作
    fixed_hoprate = settings.TRAIN_CONFIG["fixed_hoprate"]

    logger.info(f"Start Training for 1 episode with {args.steps_per_episode} steps...")
    logger.info(f"Batch Size: {args.batch_size}, Updates per step: {args.update_iters_per_step}")
    
    start_time = time.time()
    total_steps = 0
    episode = 1
    
    ep_start_time = time.time()
    
    ep_block_rewards = []
    
    # Tracking for plots
    plot_rewards = []
    plot_losses_actor = []
    plot_losses_critic = []
    plot_bers = []
    
    logger.info(f"--- Episode {episode} Start ---")

    # 主循环
    for step_idx in range(1, args.steps_per_episode + 1):
        step_start_time = time.time()
        
        # 一次策略前向同时采样全部十个 categorical offset 头
        offsets = agent.take_action(state_img, fixed_hoprate)

        # 在配置的 step 保存"动作前"观测（agent 决策所见的输入状态），
        # 必须赶在 state_img 被 env.step() 的返回值覆盖之前。
        if step_idx in save_steps:
            save_waterfall_figure(
                np.asarray(state_img),
                os.path.join(figures_dir, f"step_{step_idx:03d}_obs.png"),
                title=f"Step {step_idx} - Pre-action Observation (100 ms)",
            )

        # -------------------------------------------------------
        # 2. 环境 step
        # -------------------------------------------------------
        # 一次性执行这 10 个 offset（环境内部对应 10 个 100 ms block）
        next_state_img, reward_total, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )

        # -------------------------------------------------------
        # 3. 奖励校验与写入 replay
        # -------------------------------------------------------
        ber_blocks = info.get("ber_blocks", [])
        block_rewards = np.asarray(info.get("block_rewards", []), dtype=np.float32)
        if block_rewards.shape != (env.num_blocks,):
            raise RuntimeError("Environment returned an invalid block reward vector.")
        # step reward 必须等于 block reward 均值，否则数据契约被破坏
        if not np.isclose(
            float(reward_total),
            float(np.mean(block_rewards)),
            rtol=1e-5,
            atol=1e-6,
        ):
            raise RuntimeError("Environment reward is inconsistent with block rewards.")
        ep_block_rewards.extend(block_rewards.tolist())
        
        mean_step_ber = np.mean(ber_blocks) if len(ber_blocks) > 0 else 0.0
        mean_step_reward = float(reward_total)
        done = bool(terminated or truncated)
        # 一条 transition = 完整环境 step：10 offsets + 10 block rewards；
        # next_hoprate 与当前相同（固定跳速），供 critic 构造下一状态输入。
        buffer.add(
            state_img,
            info.get("hoprate_used", fixed_hoprate),
            offsets,
            block_rewards,
            next_state_img,
            fixed_hoprate,
            done,
        )

        # 推进到下一状态
        state_img = next_state_img
        total_steps += 1

        # -------------------------------------------------------
        # 4. 梯度更新
        # -------------------------------------------------------
        train_stats = {}
        if replay_ready(buffer, args.batch_size):
            for _ in range(args.update_iters_per_step):
                batch = buffer.sample(args.batch_size)
                train_stats = agent.update(batch)

        step_duration = time.time() - step_start_time
        
        # 曲线记录
        plot_rewards.append(mean_step_reward)
        plot_bers.append(mean_step_ber)
        plot_losses_actor.append(train_stats.get('actor_loss', 0) if train_stats else 0)
        plot_losses_critic.append(train_stats.get('critic1_loss', 0) if train_stats else 0)

        # 每个 block 实际使用的首个跳频信道：(基础 m 序列 + offset) % 信道数
        hop_sequences = info.get("hop_sequences", [])
        first_channels = [seq[0] for seq in hop_sequences if len(seq) > 0]
        # 完整 hop 序列只写日志文件（10 blocks × 10 hops），不刷控制台
        hop_logger.info("Step %d HopSequences: %s", step_idx, hop_sequences)

        log_msg = (f"Step {step_idx}/{args.steps_per_episode} | "
                   f"Offsets: {offsets.astype(int).tolist()} | "
                   f"FirstCh: {first_channels} | "
                   f"Rew: {mean_step_reward:.4f} | "
                   f"BER: {mean_step_ber:.4f} | "
                   f"Replay: {buffer.size()}")
        
        if train_stats:
             log_msg += (f" | Loss: A={train_stats.get('actor_loss', 0):.3f}, "
                         f"C={train_stats.get('critic1_loss', 0):.3f}, "
                         f"Alpha={train_stats.get('alpha', 0):.5f}")
        else:
             # replay 尚未攒够一个 batch，本 step 只采集不更新
             log_msg += f" | Warmup: {buffer.size()}/{args.batch_size}"
        
        log_msg += f" | T: {step_duration:.2f}s"
        logger.info(log_msg)

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    # -------------------------------------------------------
    # Episode 结束
    # -------------------------------------------------------
    ep_duration = time.time() - ep_start_time
    mean_ep_reward = float(np.mean(ep_block_rewards)) if len(ep_block_rewards) > 0 else 0.0
    
    logger.info(f"--- Episode {episode} Finished ---")

    # 曲线绘制
    try:
        # 1. Reward
        plt.figure()
        plt.plot(plot_rewards)
        plt.title("Mean Step Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "reward.png"))
        plt.close()

        # 2. BER
        plt.figure()
        plt.plot(plot_bers, color='r')
        plt.title("Mean Step BER")
        plt.xlabel("Step")
        plt.ylabel("BER")
        plt.grid(True)
        plt.savefig(os.path.join(args.output_dir, "ber.png"))
        plt.close()

        # 3. Loss（自动缩放 Y 轴）
        plt.figure()
        plt.plot(plot_losses_actor, label="Actor Loss", alpha=0.7)
        plt.plot(plot_losses_critic, label="Critic Loss", alpha=0.7)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        # 用 1%~99% 分位数缩放 Y 轴，忽略最初的 loss 尖峰
        skip = max(5, int(len(plot_losses_critic) * 0.05))
        if len(plot_losses_critic) > skip:
            valid_vals = plot_losses_actor[skip:] + plot_losses_critic[skip:]
            if valid_vals:
                y_min, y_max = np.percentile(valid_vals, [1, 99])
                yr = y_max - y_min if y_max != y_min else 1.0
                plt.ylim(y_min - yr * 0.1, y_max + yr * 0.1)

        plt.savefig(os.path.join(args.output_dir, "loss.png"))
        plt.close()
        logger.info(f"Plots saved to {args.output_dir}.")
        
    except Exception as e:
        logger.error(f"Plotting failed: {e}")

    # 推理 checkpoint：只存 actor 权重 + 环境维度 + 配置 metadata，
    # 不含 optimizer/replay/RNG 状态，便于脱离训练环境加载评估。
    checkpoint_metadata = environment_metadata(
        settings.ENV_CONFIG,
        settings.JAMMER_CONFIG,
        settings.REWARD_CONFIG,
    )
    checkpoint_metadata.update(
        {
            "num_actions": n_actions,
            "num_blocks": env.num_blocks,
            "fixed_hoprate": float(fixed_hoprate),
        }
    )
    checkpoint_path = os.path.join(args.output_dir, "sac_inference.pt")
    save_sac_inference_checkpoint(
        agent,
        checkpoint_path,
        np.asarray(state_img).shape,
        env.hoprate_min,
        env.hoprate_max,
        checkpoint_metadata,
    )
    logger.info("Saved SAC inference checkpoint to %s", checkpoint_path)

    total_duration = time.time() - start_time
    logger.info(f"Total Time: {total_duration:.2f}s | Mean Ep Reward: {mean_ep_reward:.4f}")


def parse_args():
    """解析命令行参数；默认值全部取自 settings.py，便于集中调参。

    ``--offline_replay_path`` 使用 ``parse_optional_replay_path`` 做类型转换，
    因此可传 ``none``/``null``/空串表示纯在线 warm-up。
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_episode", type=int, default=settings.TRAIN_CONFIG["steps_per_episode"])
    parser.add_argument("--output_dir", type=str, default="outputs/offsets/comb/0")
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    # Agent 超参
    parser.add_argument("--actor_lr", type=float, default=settings.SAC_CONFIG["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=settings.SAC_CONFIG["critic_lr"])
    parser.add_argument("--alpha_lr", type=float, default=settings.SAC_CONFIG["alpha_lr"])
    parser.add_argument("--tau", type=float, default=settings.SAC_CONFIG["tau"])
    parser.add_argument("--gamma", type=float, default=settings.SAC_CONFIG["gamma"])

    # Replay buffer 参数
    parser.add_argument("--replay_size", type=int, default=settings.BUFFER_CONFIG["capacity"])
    parser.add_argument("--batch_size", type=int, default=settings.BUFFER_CONFIG["batch_size"])
    parser.add_argument("--update_iters_per_step", type=int, default=settings.TRAIN_CONFIG["update_iters_per_step"])

    parser.add_argument(
        "--offline_replay_path",
        type=parse_optional_replay_path,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Offline v3 replay path, or 'none' for online-only warm-up.",
    )
    return parser.parse_args()


def parse_optional_replay_path(value):
    """把 ``--offline_replay_path`` 的值转为路径或 None。

    ``none``/``null``/空串（不区分大小写、允许空白）统一解析为 None，
    表示跳过离线 replay、纯在线收集。
    """
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in {"none", "null", ""} else value


if __name__ == "__main__":
    settings.set_random_seeds()
    args = parse_args()
    train(args)
