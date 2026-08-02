"""Joint derivative-NBS hoprate control with multi-head offset SAC."""

import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from fh_env import save_waterfall_figure
from joint_training import (
    add_derivative_nbs_args,
    add_environment_override_args,
    build_derivative_nbs,
    configure_figure_capture,
    execute_joint_step,
    load_joint_replay,
    nbs_metadata,
    new_nbs_history,
    record_nbs_step,
    resolve_environment_configs,
    save_nbs_artifacts,
    validate_joint_args,
)
from offline_replay import environment_metadata
import settings
from train_mbpo import save_sac_inference_checkpoint
from train_offsets import (
    build_agent_and_env,
    parse_optional_replay_path,
    replay_ready,
    setup_logger,
)


def _save_curve(output_dir, values, filename, title, ylabel, color=None):
    fig, ax = plt.subplots()
    kwargs = {"color": color} if color is not None else {}
    ax.plot(values, **kwargs)
    ax.set(title=title, xlabel="Step", ylabel=ylabel)
    ax.grid(True)
    fig.savefig(os.path.join(output_dir, filename))
    plt.close(fig)


def save_plots(output_dir, metrics, logger):
    """Save baseline SAC diagnostics alongside the shared NBS plots."""
    try:
        _save_curve(
            output_dir,
            metrics["rewards"],
            "reward.png",
            "Mean Step Reward",
            "Reward",
        )
        _save_curve(
            output_dir,
            metrics["bers"],
            "ber.png",
            "Mean Step BER",
            "BER",
            color="crimson",
        )
        fig, ax = plt.subplots()
        ax.plot(metrics["actor_losses"], label="Actor Loss", alpha=0.8)
        ax.plot(metrics["critic_losses"], label="Critic Loss", alpha=0.8)
        ax.set(title="Training Loss", xlabel="Step", ylabel="Loss")
        ax.grid(True)
        ax.legend()
        fig.savefig(os.path.join(output_dir, "loss.png"))
        plt.close(fig)
        logger.info("Saved SAC plots to %s.", output_dir)
    except Exception as exc:
        logger.error("Plotting failed: %s", exc)


def train(args):
    """Train offset SAC while derivative NBS chooses each environment hoprate."""
    validate_joint_args(args)
    settings.set_random_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger, hop_logger = setup_logger(os.path.join(args.output_dir, args.log_file))

    env_config, jammer_config = resolve_environment_configs(args)
    env, agent, buffer, _device, n_actions = build_agent_and_env(
        args,
        env_config=env_config,
        jammer_config=jammer_config,
    )
    save_steps, figures_dir = configure_figure_capture(args, env, logger)
    state_img, _reset_info = env.reset()
    current_metadata = environment_metadata(
        env_config, jammer_config, settings.REWARD_CONFIG
    )

    if args.offline_replay_path is None:
        logger.info(
            "Offline replay disabled; collecting online transitions until "
            "batch_size=%d.",
            args.batch_size,
        )
    else:
        loaded_count, replay_metadata = load_joint_replay(
            args.offline_replay_path,
            buffer,
            state_img,
            env,
            current_metadata,
            allow_config_mismatch=args.allow_replay_config_mismatch,
            logger=logger,
        )
        logger.info(
            "Loaded %d offline transitions from %s (mode=%s).",
            loaded_count,
            args.offline_replay_path,
            replay_metadata.get("hoprate_mode", "unknown"),
        )

    nbs = build_derivative_nbs(env, args)
    current_hoprate = float(nbs.reset())
    nbs_history = new_nbs_history()
    metrics = {
        "rewards": [],
        "bers": [],
        "actor_losses": [],
        "critic_losses": [],
    }
    ep_block_rewards = []

    logger.info(
        "Start joint SAC training: steps=%d reactive=%s indiscriminate=%s "
        "jammer_mode=%s initial_hoprate=%.1f.",
        args.steps_per_episode,
        env_config["enable_reactive"],
        env_config["enable_sweep"],
        jammer_config["mode"],
        current_hoprate,
    )
    logger.info(
        "Derivative NBS: p=%.4f delta=%.4f step=%.1f threshold=%g.",
        nbs.p,
        nbs.delta,
        nbs.hoprate_step,
        nbs.derivative_threshold,
    )

    start_time = time.time()
    for step_idx in range(1, args.steps_per_episode + 1):
        step_start = time.time()
        if step_idx in save_steps:
            save_waterfall_figure(
                np.asarray(state_img),
                os.path.join(figures_dir, f"step_{step_idx:03d}_obs.png"),
                title=(
                    f"Step {step_idx} - Pre-action Observation "
                    f"(Hoprate {current_hoprate:.0f} Hz)"
                ),
            )

        result = execute_joint_step(
            state_img,
            current_hoprate,
            agent,
            env,
            nbs,
            buffer,
        )
        record_nbs_step(nbs_history, nbs, result)
        state_img = result.next_state_img
        current_hoprate = result.next_hoprate
        ep_block_rewards.extend(result.block_rewards.tolist())

        train_stats = {}
        if replay_ready(buffer, args.batch_size):
            for _ in range(args.update_iters_per_step):
                train_stats = agent.update(buffer.sample(args.batch_size))

        metrics["rewards"].append(result.step_reward)
        metrics["bers"].append(result.mean_ber)
        metrics["actor_losses"].append(
            train_stats.get("actor_loss", np.nan)
        )
        metrics["critic_losses"].append(
            train_stats.get("critic1_loss", np.nan)
        )

        hop_sequences = result.info.get("hop_sequences", [])
        first_channels = [sequence[0] for sequence in hop_sequences if sequence]
        hop_logger.info("Step %d HopSequences: %s", step_idx, hop_sequences)
        derivative = nbs.get_last_derivative()
        derivative_text = "n/a" if derivative is None else f"{derivative:.6f}"
        log_message = (
            f"Step {step_idx}/{args.steps_per_episode} | "
            f"Hop={result.used_hoprate:.0f}->Next={result.next_hoprate:.0f} | "
            f"Offsets={result.offsets.astype(int).tolist()} | "
            f"FirstCh={first_channels} | Rew={result.step_reward:.4f} | "
            f"BER={result.mean_ber:.4f} | Derivative={derivative_text} | "
            f"NBSBest={nbs.get_best_hoprate():.0f} | "
            f"NBSWAvg={nbs.get_weighted_average():.1f} | "
            f"MaxW={np.max(nbs.weights):.5f} | Replay={buffer.size()}"
        )
        if train_stats:
            log_message += (
                f" | Loss:A={train_stats['actor_loss']:.3f},"
                f"C={train_stats['critic1_loss']:.3f},"
                f"Alpha={train_stats['alpha']:.5f}"
            )
        else:
            log_message += f" | Warmup={buffer.size()}/{args.batch_size}"
        if nbs.is_converged():
            log_message += " | NBS=CONVERGED_CONTINUING"
        log_message += f" | T={time.time() - step_start:.2f}s"
        logger.info(log_message)

        if result.done:
            logger.info("Episode terminated early.")
            break

    save_plots(args.output_dir, metrics, logger)
    save_nbs_artifacts(args.output_dir, nbs_history, nbs, logger)

    checkpoint_metadata = dict(current_metadata)
    checkpoint_metadata.update(
        {
            "num_actions": n_actions,
            "num_blocks": env.num_blocks,
            "hoprate_controller": nbs_metadata(nbs),
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
    logger.info(
        "Training complete in %.2fs | Mean block reward=%.4f | "
        "Final NBS best=%.1f weighted=%.1f.",
        time.time() - start_time,
        float(np.mean(ep_block_rewards)) if ep_block_rewards else 0.0,
        nbs.get_best_hoprate(),
        nbs.get_weighted_average(),
    )
    env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Derivative-NBS hoprate control with multi-head offset SAC"
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=settings.TRAIN_CONFIG["steps_per_episode"],
    )
    parser.add_argument("--output_dir", default="outputs/joint_sac")
    parser.add_argument("--log_file", default="training_log.txt")
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
        default=settings.JOINT_OFFLINE_REPLAY_CONFIG["default_path"],
        help="Random-hoprate v3 replay path, or 'none' for online-only training.",
    )
    parser.add_argument(
        "--allow_replay_config_mismatch",
        action="store_true",
        help="Allow replay generated under different environment metadata.",
    )
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    add_environment_override_args(parser)
    add_derivative_nbs_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
