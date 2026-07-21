"""Baseline training entry point for multi-head offset SAC."""

import argparse
import os
import time
import numpy as np
import torch
import logging
import matplotlib.pyplot as plt

from fh_env import FHSSQPSKEnv, save_waterfall_figure
from SAC import SAC, ReplayBuffer
from offline_replay import (
    environment_metadata,
    load_replay_into_buffer,
)
import settings

def setup_logger(log_file):
    """
    Configure the root logger (console + file) and a file-only logger for
    verbose per-step records. Both share a single FileHandler so the full
    hop sequences land in the same training_log.txt without console spam.
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

def build_agent_and_env(args):
    # -------------------------------------------------------------------------
    # 1. Device Configuration (GPU/CPU)
    # -------------------------------------------------------------------------
    if torch.cuda.is_available() and not args.cpu_only:
        device = torch.device("cuda")
        logging.info(f"Training Device: GPU ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device("cpu")
        logging.info("Training Device: CPU")

    # -------------------------------------------------------------------------
    # 2. Environment Initialization
    # -------------------------------------------------------------------------
    # Pass configuration from settings
    env = FHSSQPSKEnv(**settings.ENV_CONFIG)

    # Number of discrete actions matches number of channels
    n_actions = env.num_channels
    num_blocks = env.num_blocks
    logging.info(
        "Environment Initialized. Num Channels/Actions: %d, Offset Heads: %d",
        n_actions,
        num_blocks,
    )

    # -------------------------------------------------------------------------
    # 3. Build SAC Agent
    # -------------------------------------------------------------------------
    # Target entropy is typically -dim(A) for continuous, or relative to log(|A|) for discrete
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
    # 4. Replay Buffer
    # -------------------------------------------------------------------------
    buffer = ReplayBuffer(
        capacity=args.replay_size,
        num_heads=num_blocks,
        n_actions=n_actions,
    )

    return env, agent, buffer, device, n_actions


def replay_ready(buffer, batch_size):
    """Return whether replay contains enough complete steps for one update."""
    return buffer.size() >= int(batch_size)


def train(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger, hop_logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

    # Build components
    env, agent, buffer, device, n_actions = build_agent_and_env(args)

    # -------------------------------------------------------------------------
    # Step-triggered figure saving (obs + per-block PSD)
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

    # Use fixed hoprate for online training, matching the existing experiment.
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

    # Main Loop
    for step_idx in range(1, args.steps_per_episode + 1):
        step_start_time = time.time()
        
        # One policy pass samples all ten categorical offset heads.
        offsets = agent.take_action(state_img, fixed_hoprate)

        # Save the pre-action observation (the agent's input state) at
        # configured steps, before state_img is replaced by env.step().
        if step_idx in save_steps:
            save_waterfall_figure(
                np.asarray(state_img),
                os.path.join(figures_dir, f"step_{step_idx:03d}_obs.png"),
                title=f"Step {step_idx} - Pre-action Observation (100 ms)",
            )

        # -------------------------------------------------------
        # 2. Environment Step
        # -------------------------------------------------------
        # Execute the sequence of 10 offsets
        next_state_img, reward_total, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )

        # -------------------------------------------------------
        # 3. Reward Calculation & Storage
        # -------------------------------------------------------
        ber_blocks = info.get("ber_blocks", [])
        block_rewards = np.asarray(info.get("block_rewards", []), dtype=np.float32)
        if block_rewards.shape != (env.num_blocks,):
            raise RuntimeError("Environment returned an invalid block reward vector.")
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
        buffer.add(
            state_img,
            info.get("hoprate_used", fixed_hoprate),
            offsets,
            block_rewards,
            next_state_img,
            fixed_hoprate,
            done,
        )

        # Move to next state
        state_img = next_state_img
        total_steps += 1

        # -------------------------------------------------------
        # 4. Training Update
        # -------------------------------------------------------
        train_stats = {}
        if replay_ready(buffer, args.batch_size):
            for _ in range(args.update_iters_per_step):
                batch = buffer.sample(args.batch_size)
                train_stats = agent.update(batch)

        step_duration = time.time() - step_start_time
        
        # Logging Data
        plot_rewards.append(mean_step_reward)
        plot_bers.append(mean_step_ber)
        plot_losses_actor.append(train_stats.get('actor_loss', 0) if train_stats else 0)
        plot_losses_critic.append(train_stats.get('critic1_loss', 0) if train_stats else 0)

        # Actual channels used per block: (base m-sequence + offset) % num_channels
        hop_sequences = info.get("hop_sequences", [])
        first_channels = [seq[0] for seq in hop_sequences if len(seq) > 0]
        # Full hop sequences go to the log file only (10 blocks x 10 hops).
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
             log_msg += f" | Warmup: {buffer.size()}/{args.batch_size}"
        
        log_msg += f" | T: {step_duration:.2f}s"
        logger.info(log_msg)

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    # -------------------------------------------------------
    # End of Episode
    # -------------------------------------------------------
    ep_duration = time.time() - ep_start_time
    mean_ep_reward = float(np.mean(ep_block_rewards)) if len(ep_block_rewards) > 0 else 0.0
    
    logger.info(f"--- Episode {episode} Finished ---")

    # Plotting
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

        # 3. Loss (Auto-scaled)
        plt.figure()
        plt.plot(plot_losses_actor, label="Actor Loss", alpha=0.7)
        plt.plot(plot_losses_critic, label="Critic Loss", alpha=0.7)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)

        # Scale Y-axis to ignore initial spikes
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

    total_duration = time.time() - start_time
    logger.info(f"Total Time: {total_duration:.2f}s | Mean Ep Reward: {mean_ep_reward:.4f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps_per_episode", type=int, default=settings.TRAIN_CONFIG["steps_per_episode"])
    parser.add_argument("--output_dir", type=str, default="outputs/offsets/pre50000/comb/512_start")
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    # Agent Params
    parser.add_argument("--actor_lr", type=float, default=settings.SAC_CONFIG["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=settings.SAC_CONFIG["critic_lr"])
    parser.add_argument("--alpha_lr", type=float, default=settings.SAC_CONFIG["alpha_lr"])
    parser.add_argument("--tau", type=float, default=settings.SAC_CONFIG["tau"])
    parser.add_argument("--gamma", type=float, default=settings.SAC_CONFIG["gamma"])

    # Buffer Params
    parser.add_argument("--replay_size", type=int, default=settings.BUFFER_CONFIG["capacity"])
    parser.add_argument("--batch_size", type=int, default=settings.BUFFER_CONFIG["batch_size"])
    parser.add_argument("--update_iters_per_step", type=int, default=settings.TRAIN_CONFIG["update_iters_per_step"])

    parser.add_argument(
        "--offline_replay_path",
        type=parse_optional_replay_path,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Offline v3 replay path, or 'none' for online-only warm-up.",
    )

    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    return parser.parse_args()


def parse_optional_replay_path(value):
    if value is None:
        return None
    value = str(value).strip()
    return None if value.lower() in {"none", "null", ""} else value


if __name__ == "__main__":
    settings.set_random_seeds()
    args = parse_args()
    train(args)
