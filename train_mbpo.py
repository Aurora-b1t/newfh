"""Step-level FHSS training with multi-head SAC and reward-model MBPO."""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from SAC import PolicyNet, ReplayBuffer, SAC_POLICY_ARCHITECTURE
from fh_env import save_waterfall_figure
from offline_replay import environment_metadata, load_replay_into_buffer
from r_predict_model import StepRewardEnsemble
from r_predict_model.mbpo_adapter import (
    rollout_reward_model,
    sample_mixed_batch,
    train_reward_model_from_replay,
)
import settings
from train_offsets import (
    build_agent_and_env,
    parse_optional_replay_path,
    replay_ready,
    setup_logger,
)


SAC_CHECKPOINT_FORMAT_VERSION = 2


def _checkpoint_images(observation_shape, device):
    shape = tuple(int(value) for value in observation_shape)
    if len(shape) == 2:
        return torch.zeros((1, 1, *shape), dtype=torch.float32, device=device)
    if len(shape) == 3:
        return torch.zeros((1, *shape), dtype=torch.float32, device=device)
    raise ValueError("observation_shape must have two or three axes.")


def save_sac_inference_checkpoint(
    agent,
    path,
    observation_shape,
    hoprate_min,
    hoprate_max,
    metadata=None,
):
    """Save the policy-only portion of SAC for deterministic inference."""
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    torch.save(
        {
            "format_version": SAC_CHECKPOINT_FORMAT_VERSION,
            "model_type": "MultiHeadSACPolicy",
            "architecture": SAC_POLICY_ARCHITECTURE,
            "config": {
                "n_actions": agent.n_actions,
                "num_heads": agent.num_heads,
                "hoprate_min": float(hoprate_min),
                "hoprate_max": float(hoprate_max),
            },
            "observation_shape": list(observation_shape),
            "actor_state_dict": agent.actor.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_sac_inference_checkpoint(
    path,
    device="cpu",
    expected_num_heads=None,
    expected_n_actions=None,
    expected_observation_shape=None,
):
    """Load a policy checkpoint and validate its environment-facing shape."""
    payload = torch.load(path, map_location=device, weights_only=True)
    format_version = payload.get("format_version")
    if format_version == 1:
        raise ValueError(
            "SAC inference checkpoint format v1 uses the old BatchNorm policy "
            "architecture and cannot be loaded safely; retrain SAC to create a "
            "GroupNorm v2 checkpoint."
        )
    if format_version != SAC_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported SAC inference checkpoint format.")
    if payload.get("model_type") != "MultiHeadSACPolicy":
        raise ValueError("Checkpoint does not contain a multi-head SAC policy.")
    if payload.get("architecture") != SAC_POLICY_ARCHITECTURE:
        raise ValueError(
            "SAC checkpoint policy architecture does not match "
            f"{SAC_POLICY_ARCHITECTURE!r}."
        )
    config = dict(payload["config"])
    observation_shape = tuple(payload["observation_shape"])
    if expected_num_heads is not None and int(expected_num_heads) != int(
        config["num_heads"]
    ):
        raise ValueError("SAC checkpoint block count does not match.")
    if expected_n_actions is not None and int(expected_n_actions) != int(
        config["n_actions"]
    ):
        raise ValueError("SAC checkpoint action count does not match.")
    if (
        expected_observation_shape is not None
        and tuple(expected_observation_shape) != observation_shape
    ):
        raise ValueError("SAC checkpoint observation shape does not match.")

    policy = PolicyNet(**config).to(device)
    policy.eval()
    with torch.no_grad():
        policy(
            _checkpoint_images(observation_shape, device),
            torch.full(
                (1, 1),
                (config["hoprate_min"] + config["hoprate_max"]) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
    policy.load_state_dict(payload["actor_state_dict"])
    policy.eval()
    return policy, dict(payload.get("metadata", {}))


def _figure_steps(args, env, logger):
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


def reward_model_ready(real_buffer, model_train_batch_size):
    """Return whether real replay can support a train/holdout split."""
    return real_buffer.size() >= max(2, int(model_train_batch_size))


def should_train_reward_model(
    step_idx, real_buffer, model_train_batch_size, model_train_freq
):
    """Apply the configured post-environment-step reward-model schedule."""
    if int(model_train_freq) <= 0:
        raise ValueError("model_train_freq must be positive.")
    return reward_model_ready(real_buffer, model_train_batch_size) and (
        int(step_idx) % int(model_train_freq) == 0
    )


def train(args):
    """Train step-level SAC from a mixture of real and reward-model replay."""
    _validate_args(args)
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
        "holdout_losses": [],
        "disagreements": [],
        "target_saturation_fractions": [],
    }
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
            logger.info(
                "Reward model | holdout=%.6f | elites=%s | epochs=%s | "
                "rollout=%d | model_buf=%d->%d/%d | fifo_evicted=%d | "
                "disagreement=%.6f(p95=%.6f) | "
                "target_sat=%.2f%% | T=%.2fs",
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
            )

        train_stats = {}
        if replay_ready(real_buffer, args.batch_size):
            for _ in range(args.update_iters_per_step):
                batch = sample_mixed_batch(
                    real_buffer,
                    model_buffer,
                    args.batch_size,
                    args.real_ratio,
                )
                train_stats = agent.update(batch)

        metrics["rewards"].append(float(step_reward))
        metrics["bers"].append(mean_step_ber)
        metrics["actor_losses"].append(train_stats.get("actor_loss", np.nan))
        metrics["critic_losses"].append(train_stats.get("critic1_loss", np.nan))
        metrics["model_rewards"].append(
            last_rollout_stats.get("reward_mean", np.nan)
        )
        metrics["holdout_losses"].append(
            last_model_stats.get("holdout_loss_mean", np.nan)
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


def _save_curve(output_dir, values, filename, title, ylabel, color=None):
    plt.figure()
    plt.plot(values, color=color)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()


def save_plots(output_dir, metrics, logger):
    """Save training diagnostics without invalidating a completed run."""
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
            metrics["holdout_losses"],
            "model_holdout.png",
            "Reward Model Holdout MSE",
            "MSE",
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
    parser = argparse.ArgumentParser(
        description="Step-level multi-head SAC with an MBPO reward ensemble"
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=settings.TRAIN_CONFIG["steps_per_episode"],
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/mbpo/comb/pre50000"
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
        "--deterministic_model_rollout", action="store_true"
    )
    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    return parser.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    train(parse_args())
