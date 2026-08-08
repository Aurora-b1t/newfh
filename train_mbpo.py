"""Step-level FHSS training with multi-head SAC and reward-model MBPO."""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from SAC import (
    ReplayBuffer,
    SAC_CHECKPOINT_FORMAT_VERSION,
    SAC_POLICY_ARCHITECTURE,
    load_sac_inference_checkpoint,
    save_sac_inference_checkpoint,
)
from fh_env import save_waterfall_figure
from mbpo_dataloader import (
    ReplayBufferDataset,
    build_mixed_replay_loader,
)
from offline_replay import environment_metadata, load_replay_into_buffer
from r_predict_model import RewardReplayDataset, StepRewardEnsemble
from r_predict_model.mbpo_adapter import (
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
    """Enable opt-in GPU math choices without changing training budgets."""
    if not bool(getattr(args, "model_fast_math", False)) or not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    # set_random_seeds() enables deterministic cuDNN, which disables benchmark
    # mode entirely; fast-math runs must clear it to get autotuned kernels.
    torch.backends.cudnn.deterministic = False
    torch.set_float32_matmul_precision("high")


def reward_model_ready(real_buffer, model_train_batch_size):
    """Return whether real replay can support a train/holdout split."""
    return real_buffer.size() >= max(2, int(model_train_batch_size))


def mbpo_timing_suffix(model_stats, rollout_stats):
    """Return a timing summary string for reward-model fit/rollout stages.

    Returns an empty string when settings.TIMING_ENABLED is False.
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
    """Apply the configured post-environment-step reward-model schedule."""
    if int(model_train_freq) <= 0:
        raise ValueError("model_train_freq must be positive.")
    return reward_model_ready(real_buffer, model_train_batch_size) and (
        int(step_idx) % int(model_train_freq) == 0
    )


def train(args):
    """Train step-level SAC from a mixture of real and reward-model replay."""
    _validate_args(args)
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
    real_replay_dataset = ReplayBufferDataset(
        real_buffer,
        device=device,
        cache_on_device=args.cache_replay_on_device,
    )
    model_replay_dataset = ReplayBufferDataset(
        model_buffer,
        device=device,
        cache_on_device=args.cache_replay_on_device,
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
    if real_buffer.size():
        real_replay_dataset.sync()

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
    )
    reward_dataset_cache = RewardReplayDataset(
        device=device,
        cache_on_device=args.cache_model_dataset,
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
        "Reward runtime: cache=%s precision=%s compile=%s fast_math=%s",
        args.cache_model_dataset,
        getattr(args, "model_precision", "float32"),
        getattr(args, "model_compile", False),
        getattr(args, "model_fast_math", False),
    )
    logger.info(
        "Replay DataLoader: cache=%s workers=%d pin_memory=%s batch=%d",
        args.cache_replay_on_device,
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
        real_replay_dataset.sync()
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
                cache_dataset_on_device=args.cache_model_dataset,
                dataset_cache=reward_dataset_cache,
                data_loader_workers=args.data_loader_workers,
                data_loader_pin_memory=args.data_loader_pin_memory,
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
            model_replay_dataset.sync()
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
            sac_loader = build_mixed_replay_loader(
                real_replay_dataset,
                model_replay_dataset,
                args.batch_size,
                args.real_ratio,
                args.update_iters_per_step,
                num_workers=args.data_loader_workers,
                pin_memory=args.data_loader_pin_memory,
            )
            last_update_index = args.update_iters_per_step - 1
            for update_index, batch in enumerate(sac_loader):
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
    """Save one per-epoch holdout-loss figure for a single reward-model fit."""
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
    """Persist all per-fit holdout curves with NaN padding for ragged epochs."""
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
    """Save one per-epoch training-loss figure for a single reward-model fit."""
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
    """Persist all per-fit training curves with NaN padding for ragged epochs."""
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
        "--cache_model_dataset",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG["cache_dataset_on_device"],
        help="Keep the reward-model replay tensors on the training device.",
    )
    parser.add_argument(
        "--cache_replay_on_device",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG.get("cache_replay_on_device", True),
        help="Keep SAC real/model replay tensors on the training device.",
    )
    parser.add_argument(
        "--data_loader_workers",
        type=int,
        default=settings.MBPO_CONFIG.get("data_loader_workers", 0),
        help="DataLoader workers; CUDA-resident datasets require zero.",
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
        "--save_model_curve_figures",
        action=argparse.BooleanOptionalAction,
        default=settings.MBPO_CONFIG["save_curve_figures"],
        help="Write a PNG pair after every reward-model fit.",
    )
    parser.add_argument(
        "--deterministic_model_rollout", action="store_true"
    )
    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    return parser.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    train(parse_args())
