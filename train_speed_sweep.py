"""
train_speed_sweep.py — Hoprate Sweep Evaluation
===============================================

Sweeps hoprate from 10 Hz to 1000 Hz and runs a fixed number of environment
steps at each hoprate.  Each environment step internally executes 10 blocks,
matching the FHSSQPSKEnv step semantics used by the training scripts.

This script is a deterministic-grid counterpart to ``train_speed.py``: instead
of using Noisy Binary Search to choose the next hoprate, it evaluates every
candidate hoprate in ascending order and records BER/reward diagnostics.

Usage
-----

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
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y", "on"):
        return True
    if value in ("false", "0", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_hoprate_grid(hoprate_min, hoprate_max, hoprate_step):
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
    if mode == "random":
        return rng.randint(0, n_channels, size=10).astype(np.float32)
    if mode == "zeros":
        return np.zeros(10, dtype=np.float32)
    raise ValueError(f"Unsupported offset_mode: {mode}")


def sweep(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Log file: {log_path}")

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

    hoprates = build_hoprate_grid(args.hoprate_min, args.hoprate_max, args.hoprate_step)
    logger.info(
        f"Hoprate sweep: {len(hoprates)} values, "
        f"{args.steps_per_hoprate} env steps per value "
        f"({args.steps_per_hoprate * 10} blocks per value)"
    )

    rng = np.random.RandomState(args.seed)
    _state_img, _info = env.reset()

    step_records = []
    summary_records = []
    start_time = time.time()
    global_step = 0

    for hop_idx, hoprate in enumerate(hoprates, start=1):
        hoprate_bers = []
        hoprate_rewards = []
        hoprate_used_values = []
        hop_start = time.time()

        for local_step in range(1, args.steps_per_hoprate + 1):
            global_step += 1
            step_start = time.time()

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

        if len(hoprate_bers) < args.steps_per_hoprate:
            break

    _save_outputs(args.output_dir, step_records, summary_records, save_plots=not args.no_plots)

    elapsed = time.time() - start_time
    logger.info(f"Sweep complete. {len(step_records)} env steps in {elapsed:.1f}s")
    if summary_records:
        best_ber = min(summary_records, key=lambda item: item["ber_mean"])
        logger.info(
            f"Lowest mean BER: hoprate={best_ber['hoprate_target']:.0f} Hz, "
            f"BER={best_ber['ber_mean']:.4f}, "
            f"reward={best_ber['reward_mean']:.4f}"
        )


def _save_outputs(output_dir, step_records, summary_records, save_plots=False):
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
    """Generate PNG diagnostic plots, using the same matplotlib style as train_speed.py."""
    # 1. BER vs hoprate
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(hoprates, ber_mean, yerr=ber_std, fmt=".-", capsize=3,
                color="crimson", alpha=0.8)
    ax.set_title("Mean BER vs Hoprate")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("Mean BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber_vs_hoprate.png"))
    plt.close(fig)

    # 2. Reward vs hoprate
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.errorbar(hoprates, reward_mean, yerr=reward_std, fmt=".-", capsize=3,
                color="steelblue", alpha=0.8)
    ax.set_title("Mean Reward vs Hoprate")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("Mean Reward")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "reward_vs_hoprate.png"))
    plt.close(fig)

    # 3. Step BER and hoprate trajectory
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
    args = parse_args()
    settings.set_random_seeds(args.seed)
    sweep(args)
