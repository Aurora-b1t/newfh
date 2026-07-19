"""
FHSS anti-jamming training entry point with SAC and an MBPO reward model.

This script keeps SAC as the actor/critic optimizer, and adds a lightweight
model-based branch that predicts one-step rewards.  Unlike the original MBPO
formulation for continuous control, this implementation does not predict the
next PSD image.  The learned ensemble only maps

    (state image, hop rate, block index, offset action) -> block reward

and writes rollout_length=1 synthetic transitions into a separate replay buffer.
SAC is then updated with a configurable mixture of real and synthetic samples.
"""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from SAC import ReplayBuffer
from offline_replay import (
    environment_metadata,
    load_replay_into_buffer,
)
from fh_env import compute_block_rewards
from r_predict_model import EnsembleDynamicsModel
from r_predict_model.mbpo_adapter import (
    concat_transition_batches,
    rollout_reward_model,
    train_reward_model_from_replay,
)
import settings
from train_offsets import build_agent_and_env, setup_logger


MBPO_DISABLED_MESSAGE = (
    "train_mbpo.py is temporarily disabled: its reward model still uses the "
    "legacy block-level replay schema. Migrate it to the v3 step-level, "
    "ten-head SAC interface before running MBPO experiments."
)


def add_block_transitions(*_args, **_kwargs):
    """Reject calls into the legacy MBPO block-transition adapter."""
    raise RuntimeError(MBPO_DISABLED_MESSAGE)


def sample_mixed_batch(real_buffer, model_buffer, batch_size, real_ratio):
    """
    Sample one SAC update batch from real and model-generated replay buffers.

    Args:
        real_buffer: Replay buffer containing transitions collected from the
            real FHSS environment.
        model_buffer: Replay buffer containing reward-model synthetic
            transitions.
        batch_size: Target number of transitions for one SAC update.
        real_ratio: Fraction of the batch reserved for real transitions.  The
            value is clipped into [0, 1] so command-line mistakes do not create
            negative sample counts.

    Returns:
        A single replay sample dictionary with the same structure as
        ``ReplayBuffer.sample``.
    """
    real_ratio = float(np.clip(real_ratio, 0.0, 1.0))
    batches = []
    if model_buffer.size() > 0:
        real_batch_size = min(int(batch_size * real_ratio), real_buffer.size())
        model_batch_size = min(batch_size - real_batch_size, model_buffer.size())
        if real_batch_size > 0:
            batches.append(real_buffer.sample(real_batch_size))
        if model_batch_size > 0:
            batches.append(model_buffer.sample(model_batch_size))
    else:
        real_batch_size = min(batch_size, real_buffer.size())
        if real_batch_size > 0:
            batches.append(real_buffer.sample(real_batch_size))

    return concat_transition_batches(batches)


def store_real_transitions(
    buffer,
    state_img,
    next_state_img,
    fixed_hoprate,
    offsets,
    per_block_rewards,
):
    """
    Split one environment step into ten block-level replay transitions.

    The FHSS environment returns a full-step PSD observation and a vector of ten
    offset actions.  SAC, however, learns a block-level decision: for each block
    index i, choose the offset action for that block.  This helper converts one
    environment interaction into a ten-state sequence. Internal blocks keep
    the current PSD image and advance the block index; only block 9 transitions
    to the next environment PSD image and wraps the index to zero.
    """
    add_block_transitions(
        buffer,
        state_img,
        next_state_img,
        fixed_hoprate,
        offsets,
        per_block_rewards,
        next_hoprate=fixed_hoprate,
    )


def train(args):
    """
    Run SAC training augmented with a reward-only MBPO ensemble.

    The high-level loop is:
        1. choose one offset for each of the ten FHSS blocks;
        2. step the real environment once and store block-level transitions;
        3. periodically train the reward ensemble on all real replay data;
        4. generate one-step synthetic transitions into ``model_buffer``;
        5. update SAC from a real/model mixed replay batch;
        6. save diagnostic curves after training finishes.
    """
    raise RuntimeError(MBPO_DISABLED_MESSAGE)

    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

    env, agent, real_buffer, device, n_actions = build_agent_and_env(args)
    model_buffer = ReplayBuffer(capacity=args.model_replay_size)
    fixed_hoprate = settings.TRAIN_CONFIG["fixed_hoprate"]

    state_img, _info = env.reset()
    loaded_count, replay_metadata = load_replay_into_buffer(
        args.offline_replay_path,
        real_buffer,
        expected_observation_shape=np.asarray(state_img).shape,
        expected_num_actions=n_actions,
        current_environment_metadata=environment_metadata(
            settings.ENV_CONFIG,
            settings.JAMMER_CONFIG,
            settings.REWARD_CONFIG,
        ),
        logger=logger,
    )
    logger.info(
        "Loaded %d offline real transitions from %s (mode=%s)",
        loaded_count,
        args.offline_replay_path,
        replay_metadata.get("hoprate_mode", "unknown"),
    )
    reward_model = EnsembleDynamicsModel(
        network_size=args.num_networks,
        elite_size=args.num_elites,
        state_size=int(np.asarray(state_img).size + 1 + 1),
        action_size=1,
        reward_size=1,
        hidden_size=args.pred_hidden_size,
        use_decay=True,
        device=device,
    )

    logger.info(
        "MBPO reward model: networks=%d elites=%d hidden=%d rollout_length=1 "
        "real_ratio=%.2f",
        args.num_networks,
        args.num_elites,
        args.pred_hidden_size,
        args.real_ratio,
    )
    logger.info(f"Start MBPO+SAC training for {args.steps_per_episode} environment steps.")

    start_time = time.time()
    ep_block_rewards = []
    plot_rewards = []
    plot_bers = []
    plot_losses_actor = []
    plot_losses_critic = []
    plot_model_rewards = []
    last_model_stats = {}
    last_rollout_stats = {}

    for step_idx in range(1, args.steps_per_episode + 1):
        step_start = time.time()

        # A single environment step contains ten independently selected block
        # offsets.  The hop rate is fixed in this training configuration.
        offsets = np.zeros(10, dtype=np.float32)
        for i in range(10):
            action = agent.take_action(state_img, fixed_hoprate, i)
            action = int(np.clip(action, 0, n_actions - 1))
            offsets[i] = action

        next_state_img, _reward_total, terminated, truncated, info = env.step(
            {"hoprate": fixed_hoprate, "offsets": offsets}
        )

        ber_blocks = info.get("ber_blocks", [])
        per_block_rewards = compute_block_rewards(
            ber_blocks,
            info.get("hoprate_used", fixed_hoprate),
        )
        # Store real experience before any model training so the reward model
        # and SAC always train on data that has actually been observed.
        store_real_transitions(
            real_buffer,
            state_img,
            next_state_img,
            fixed_hoprate,
            offsets,
            per_block_rewards,
        )

        ep_block_rewards.extend(per_block_rewards)
        mean_step_ber = float(np.mean(ber_blocks)) if len(ber_blocks) > 0 else 0.0
        mean_step_reward = float(np.mean(per_block_rewards)) if len(per_block_rewards) > 0 else 0.0
        state_img = next_state_img

        if step_idx % args.model_train_freq == 0:
            # The ensemble is trained from real replay only.  Synthetic samples
            # are never fed back into the reward model, which avoids compounding
            # model bias in the supervised target set.
            last_model_stats = train_reward_model_from_replay(
                reward_model,
                real_buffer,
                args.model_train_batch_size,
            )
            last_rollout_stats = rollout_reward_model(
                reward_model,
                agent,
                real_buffer,
                model_buffer,
                args.rollout_batch_size,
                fixed_hoprate,
                n_actions,
            )
            logger.info(
                "Reward model | holdout_loss=%.6f | elites=%s | epochs=%d | "
                "model_rollout=%d | model_rew=%.4f+/-%.4f",
                last_model_stats.get("holdout_loss_mean", 0.0),
                last_model_stats.get("elite_model_idxes", []),
                last_model_stats.get("epochs", 0),
                last_rollout_stats.get("generated", 0),
                last_rollout_stats.get("reward_mean", 0.0),
                last_rollout_stats.get("reward_std", 0.0),
            )

        train_stats = {}
        for _ in range(args.update_iters_per_step):
            # SAC receives a mixed batch.  ``real_ratio`` controls how much
            # the learned model can influence policy updates.
            batch = sample_mixed_batch(
                real_buffer,
                model_buffer,
                args.batch_size,
                args.real_ratio,
            )
            train_stats = agent.update(batch)

        step_duration = time.time() - step_start
        plot_rewards.append(mean_step_reward)
        plot_bers.append(mean_step_ber)
        plot_losses_actor.append(train_stats.get("actor_loss", 0.0) if train_stats else 0.0)
        plot_losses_critic.append(train_stats.get("critic1_loss", 0.0) if train_stats else 0.0)
        plot_model_rewards.append(last_rollout_stats.get("reward_mean", 0.0))

        log_msg = (
            f"Step {step_idx}/{args.steps_per_episode} | "
            f"Offsets: {offsets.astype(int).tolist()} | "
            f"Rew: {mean_step_reward:.4f} | BER: {mean_step_ber:.4f} | "
            f"RealBuf={real_buffer.size()} | ModelBuf={model_buffer.size()}"
        )
        if last_model_stats:
            log_msg += (
                f" | ModelHoldout={last_model_stats.get('holdout_loss_mean', 0.0):.6f} "
                f"| Elites={last_model_stats.get('elite_model_idxes', [])}"
            )
        if train_stats:
            log_msg += (
                f" | Loss: A={train_stats.get('actor_loss', 0.0):.3f}, "
                f"C={train_stats.get('critic1_loss', 0.0):.3f}, "
                f"Alpha={train_stats.get('alpha', 0.0):.5f}"
            )
        log_msg += f" | T: {step_duration:.2f}s"
        logger.info(log_msg)

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    save_plots(
        args.output_dir,
        plot_rewards,
        plot_bers,
        plot_losses_actor,
        plot_losses_critic,
        plot_model_rewards,
        logger,
    )
    total_duration = time.time() - start_time
    mean_ep_reward = float(np.mean(ep_block_rewards)) if ep_block_rewards else 0.0
    logger.info(f"Total Time: {total_duration:.2f}s | Mean Ep Reward: {mean_ep_reward:.4f}")


def save_plots(
    output_dir,
    plot_rewards,
    plot_bers,
    plot_losses_actor,
    plot_losses_critic,
    plot_model_rewards,
    logger,
):
    """
    Save training diagnostics as PNG curves.

    Plotting is deliberately best-effort: failed plotting should be logged but
    should not invalidate a completed training run.
    """
    try:
        plt.figure()
        plt.plot(plot_rewards)
        plt.title("Mean Step Reward")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "reward.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_bers, color="r")
        plt.title("Mean Step BER")
        plt.xlabel("Step")
        plt.ylabel("BER")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "ber.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_losses_actor, label="Actor Loss", alpha=0.7)
        plt.plot(plot_losses_critic, label="Critic Loss", alpha=0.7)
        plt.title("Training Loss")
        plt.xlabel("Step")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "loss.png"))
        plt.close()

        plt.figure()
        plt.plot(plot_model_rewards)
        plt.title("Synthetic Reward Mean")
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "model_reward.png"))
        plt.close()
        logger.info(f"Plots saved to {output_dir}.")
    except Exception as exc:
        logger.error(f"Plotting failed: {exc}")


def parse_args():
    """Parse command-line options, using ``settings.py`` as the default source."""
    parser = argparse.ArgumentParser(description="SAC + MBPO reward-model training")
    parser.add_argument("--steps_per_episode", type=int, default=settings.TRAIN_CONFIG["steps_per_episode"])
    parser.add_argument("--output_dir", type=str, default="outputs/mbpo/comb/pre50000")
    parser.add_argument("--log_file", type=str, default="training_log.txt")

    parser.add_argument("--actor_lr", type=float, default=settings.SAC_CONFIG["actor_lr"])
    parser.add_argument("--critic_lr", type=float, default=settings.SAC_CONFIG["critic_lr"])
    parser.add_argument("--alpha_lr", type=float, default=settings.SAC_CONFIG["alpha_lr"])
    parser.add_argument("--tau", type=float, default=settings.SAC_CONFIG["tau"])
    parser.add_argument("--gamma", type=float, default=settings.SAC_CONFIG["gamma"])

    parser.add_argument("--replay_size", type=int, default=settings.BUFFER_CONFIG["capacity"])
    parser.add_argument("--batch_size", type=int, default=settings.BUFFER_CONFIG["batch_size"])
    parser.add_argument("--update_iters_per_step", type=int, default=settings.TRAIN_CONFIG["update_iters_per_step"])

    parser.add_argument(
        "--offline_replay_path",
        type=str,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Offline real replay .npz file loaded before the first update.",
    )

    parser.add_argument("--real_ratio", type=float, default=settings.MBPO_CONFIG["real_ratio"])
    parser.add_argument("--model_train_freq", type=int, default=settings.MBPO_CONFIG["model_train_freq"])
    parser.add_argument("--rollout_batch_size", type=int, default=settings.MBPO_CONFIG["rollout_batch_size"])
    parser.add_argument("--model_replay_size", type=int, default=settings.MBPO_CONFIG["model_replay_size"])
    parser.add_argument("--num_networks", type=int, default=settings.MBPO_CONFIG["num_networks"])
    parser.add_argument("--num_elites", type=int, default=settings.MBPO_CONFIG["num_elites"])
    parser.add_argument("--pred_hidden_size", type=int, default=settings.MBPO_CONFIG["hidden_size"])
    parser.add_argument("--model_train_batch_size", type=int, default=settings.MBPO_CONFIG["model_train_batch_size"])

    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    return parser.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    torch.set_num_threads(max(1, torch.get_num_threads()))
    train(parse_args())
