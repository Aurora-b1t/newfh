"""Joint derivative-NBS hoprate control with reward-model MBPO-SAC."""

import argparse
import os
import time

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
from r_predict_model import StepRewardEnsemble
from r_predict_model.mbpo_adapter import (
    rollout_reward_model,
    sample_mixed_batch,
    train_reward_model_from_replay,
)
import settings
from SAC import ReplayBuffer
from train_mbpo import (
    _validate_args,
    reward_model_ready,
    save_plots,
    save_sac_inference_checkpoint,
    should_train_reward_model,
)
from train_offsets import (
    build_agent_and_env,
    parse_optional_replay_path,
    replay_ready,
    setup_logger,
)


def train(args):
    """Train MBPO-SAC while derivative NBS chooses each environment hoprate."""
    _validate_args(args)
    validate_joint_args(args)
    settings.set_random_seeds(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    logger, hop_logger = setup_logger(os.path.join(args.output_dir, args.log_file))

    env_config, jammer_config = resolve_environment_configs(args)
    env, agent, real_buffer, device, n_actions = build_agent_and_env(
        args,
        env_config=env_config,
        jammer_config=jammer_config,
    )
    model_buffer = ReplayBuffer(
        capacity=args.model_replay_size,
        num_heads=env.num_blocks,
        n_actions=n_actions,
    )
    save_steps, figures_dir = configure_figure_capture(args, env, logger)
    state_img, _reset_info = env.reset()
    current_metadata = environment_metadata(
        env_config, jammer_config, settings.REWARD_CONFIG
    )

    if args.offline_replay_path is None:
        logger.info(
            "Offline replay disabled; reward-model warm-up requires %d real steps.",
            max(2, args.model_train_batch_size),
        )
    else:
        loaded_count, replay_metadata = load_joint_replay(
            args.offline_replay_path,
            real_buffer,
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
    nbs = build_derivative_nbs(env, args)
    current_hoprate = float(nbs.reset())
    nbs_history = new_nbs_history()

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
    ep_block_rewards = []
    last_model_stats = {}
    last_rollout_stats = {}

    logger.info(
        "Start joint MBPO-SAC training: steps=%d reactive=%s "
        "indiscriminate=%s jammer_mode=%s initial_hoprate=%.1f.",
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
    logger.info(
        "MBPO reward ensemble: models=%d elites=%d hidden=%d "
        "full-fit-frequency=%d.",
        args.num_networks,
        args.num_elites,
        args.pred_hidden_size,
        args.model_train_freq,
    )
    logger.info(
        "Model replay uses persistent FIFO retention with capacity=%d.",
        model_buffer.capacity,
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
            real_buffer,
        )
        record_nbs_step(nbs_history, nbs, result)
        state_img = result.next_state_img
        current_hoprate = result.next_hoprate
        ep_block_rewards.extend(result.block_rewards.tolist())

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

        metrics["rewards"].append(result.step_reward)
        metrics["bers"].append(result.mean_ber)
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
            f"MaxW={np.max(nbs.weights):.5f} | "
            f"RealBuf={real_buffer.size()} | ModelBuf={model_buffer.size()}"
        )
        if train_stats:
            log_message += (
                f" | Loss:A={train_stats['actor_loss']:.3f},"
                f"C={train_stats['critic1_loss']:.3f},"
                f"Alpha={train_stats['alpha']:.5f}"
            )
        else:
            log_message += f" | SACWarmup={real_buffer.size()}/{args.batch_size}"
        if not model_ready:
            log_message += (
                f" | ModelWarmup={real_buffer.size()}/"
                f"{max(2, args.model_train_batch_size)}"
            )
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
        logger.info("Saved reward-model checkpoint to %s", reward_checkpoint)
    else:
        logger.warning(
            "Reward model never left warm-up; no reward-model checkpoint was saved."
        )

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
        description="Derivative-NBS hoprate control with reward-model MBPO-SAC"
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=settings.TRAIN_CONFIG["steps_per_episode"],
    )
    parser.add_argument("--output_dir", default="outputs/joint_mbpo")
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
        "--model_lr", type=float, default=settings.MBPO_CONFIG["learning_rate"]
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
    parser.add_argument("--deterministic_model_rollout", action="store_true")
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    parser.add_argument("--cpu_only", action="store_true", default=settings.CPU_ONLY)
    add_environment_override_args(parser)
    add_derivative_nbs_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    train(parse_args())
