"""
train_speed.py — Noisy Binary Search Hoprate Threshold Discovery
=================================================================

Uses the Noisy Binary Search (NBS) algorithm to find the hoprate threshold
where the reactive jammer transitions from "can fully jam" (low hoprate, high
BER) to "cannot follow" (high hoprate, low BER).

At each environment step:

1. NBS proposes a hoprate.
2. Random offsets are generated (no SAC — offset learning is out of scope).
3. The environment runs 10 blocks at the chosen hoprate.
4. The mean BER feeds back to NBS, which updates its weight distribution and
   returns the next hoprate to test.

The reactive jammer is **enabled** so that a clear BER-vs-hoprate threshold
exists in the environment.

Usage
-----

.. code-block:: bash

    # Quick test
    D:\\Anaconda\\envs\\rl_fhss\\python.exe train_speed.py --steps 60 --output_dir outputs/speed_test

    # With custom NBS parameters
    D:\\Anaconda\\envs\\rl_fhss\\python.exe train_speed.py --nbs_p 0.15 --nbs_delta 0.03 --steps 100
"""

import argparse
import os
import time
import numpy as np
import logging
import matplotlib.pyplot as plt

from fh_env import FHSSQPSKEnv
from noisy_binary_search import NoisyBinarySearch
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


def search(args):
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, args.log_file)
    logger = setup_logger(log_path)
    logger.info(f"Output directory: {args.output_dir}")

    # ---- environment (reactive jammer ON for threshold discovery) --------------
    env_config = dict(settings.ENV_CONFIG)
    env_config["enable_reactive"] = True
    env_config["enable_sweep"] = args.enable_sweep

    env = FHSSQPSKEnv(**env_config)
    n_channels = env.num_channels
    logger.info(f"Environment: {n_channels} channels, "
                f"hoprate ∈ [{env.hoprate_min:.0f}, {env.hoprate_max:.0f}] Hz, "
                f"reactive={env_config['enable_reactive']}, "
                f"sweep={env_config['enable_sweep']}")

    # ---- noisy binary search ---------------------------------------------------
    nbs = NoisyBinarySearch(
        hoprate_min=env.hoprate_min,
        hoprate_max=env.hoprate_max,
        hoprate_step=args.nbs_step,
        p=args.nbs_p,
        delta=args.nbs_delta,
    )
    logger.info(f"NBS: p={nbs.p}, δ={nbs.delta}, "
                f"step={nbs.hoprate_step} Hz, "
                f"candidates={nbs.n_candidates}")

    # ---- tracking ---------------------------------------------------------------
    hoprates_used = []       # actual hoprate each NBS step
    bers = []                # median BER each NBS step
    nbs_best_history = []    # NBS best estimate over time
    nbs_wavg_history = []    # NBS weighted average over time

    # ---- initialise ------------------------------------------------------------
    _state_img, info = env.reset()
    hoprate = float(nbs.reset())
    logger.info(f"Initial NBS hoprate: {hoprate:.0f} Hz")

    start_time = time.time()

    for step_idx in range(1, args.steps + 1):
        step_start = time.time()

        # --- 1. Environment step -----------------------------------------------
        offsets = np.random.randint(0, n_channels, size=10).astype(np.float32)
        _obs, _rew, terminated, truncated, info = env.step(
            {"hoprate": hoprate, "offsets": offsets}
        )

        # --- 2. BER ------------------------------------------------------------
        ber_blocks = info.get("ber_blocks", [])
        mean_ber = float(np.mean(ber_blocks)) if ber_blocks else 0.0
        hoprate_used = float(info.get("hoprate_used", hoprate))

        # --- 3. NBS update -----------------------------------------------------
        next_hoprate = nbs.step(mean_ber)

        # --- 4. Bookkeeping ----------------------------------------------------
        hoprate = next_hoprate

        hoprates_used.append(hoprate_used)
        bers.append(mean_ber)
        nbs_best_history.append(nbs.get_best_hoprate())
        nbs_wavg_history.append(nbs.get_weighted_average())

        step_duration = time.time() - step_start

        # --- 5. Logging --------------------------------------------------------
        nbs_max_w = np.max(nbs.weights)
        conv_flag = " ✓CONVERGED" if nbs.is_converged() else ""

        logger.info(
            f"Step {step_idx:4d}/{args.steps} | "
            f"Hop={hoprate_used:6.0f} Hz | "
            f"BER={mean_ber:.4f} | "
            f"NBS_best={nbs_best_history[-1]:6.0f} | "
            f"wavg={nbs_wavg_history[-1]:6.0f} | "
            f"max_w={nbs_max_w:.4f} | "
            f"T={step_duration:.2f}s{conv_flag}"
        )

        if nbs.is_converged():
            logger.info(f"NBS converged at step {step_idx} — "
                        f"threshold ≈ {nbs.get_best_hoprate():.0f} Hz")

        if terminated or truncated:
            logger.info("Episode terminated early.")
            break

    # ---- summary ---------------------------------------------------------------
    elapsed = time.time() - start_time
    nbs_best = nbs.get_best_hoprate()
    nbs_wavg = nbs.get_weighted_average()

    logger.info(f"{'='*60}")
    logger.info(f"Search complete.  {len(hoprates_used)} steps in {elapsed:.1f}s")
    logger.info(f"NBS best estimate:  {nbs_best:.0f} Hz")
    logger.info(f"NBS weighted avg:   {nbs_wavg:.0f} Hz")
    logger.info(f"Converged:          {nbs.is_converged()}")
    logger.info(f"Final max weight:   {np.max(nbs.weights):.4f}")
    if len(bers) > 0:
        logger.info(f"Final BER:          {bers[-1]:.4f}")
        logger.info(f"Mean BER all steps: {np.mean(bers):.4f}")

    # ---- save distributions ----------------------------------------------------
    nbs_candidates, nbs_weights = nbs.get_distribution()
    np.savez(
        os.path.join(args.output_dir, "nbs_distribution.npz"),
        candidates=nbs_candidates,
        weights=nbs_weights,
        hoprates_used=np.array(hoprates_used),
        bers=np.array(bers),
    )

    # ---- plotting --------------------------------------------------------------
    _plot_results(args.output_dir, hoprates_used, bers,
                  nbs_best_history, nbs_wavg_history,
                  nbs_candidates, nbs_weights)
    logger.info(f"Plots saved to {args.output_dir}.")


def _plot_results(output_dir, hoprates, bers, best_hist, wavg_hist,
                  nbs_cand, nbs_w):
    """Generate diagnostic plots."""
    steps = np.arange(1, len(hoprates) + 1)

    # 1. Hoprate trajectory + NBS estimates
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, hoprates, ".-", color="steelblue", alpha=0.7, label="tested hoprate")
    ax.plot(steps, best_hist, "--", color="darkorange", label="NBS best (MAP)")
    ax.plot(steps, wavg_hist, ":", color="darkgreen", label="NBS weighted avg")
    ax.set_title("Hoprate Trajectory & NBS Estimates")
    ax.set_xlabel("Step")
    ax.set_ylabel("Hoprate (Hz)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "hoprate.png"))
    plt.close(fig)

    # 2. BER over steps
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, bers, ".-", color="crimson", alpha=0.7)
    ax.set_title("Mean BER per Step")
    ax.set_xlabel("Step")
    ax.set_ylabel("BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber.png"))
    plt.close(fig)

    # 3. BER vs hoprate scatter (reveals the threshold)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(hoprates, bers, c=steps, cmap="viridis", alpha=0.7, edgecolors="k", linewidth=0.3)
    fig.colorbar(ax.collections[0], ax=ax, label="Step")
    ax.set_title("BER vs Hoprate (threshold visible as BER drop)")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber_vs_hoprate.png"))
    plt.close(fig)

    # 4. NBS final weight distribution
    fig, ax = plt.subplots(figsize=(10, 4))
    bar_width = nbs_cand[1] - nbs_cand[0] if len(nbs_cand) > 1 else 10
    ax.bar(nbs_cand, nbs_w, width=bar_width, color="steelblue", alpha=0.8)
    ax.axhline(y=1.0 - settings.NBS_CONFIG["delta"], color="red", linestyle="--",
               label=f"convergence threshold (1−δ={1-settings.NBS_CONFIG['delta']:.2f})")
    ax.set_title("NBS Final Weight Distribution")
    ax.set_xlabel("Hoprate (Hz)")
    ax.set_ylabel("Weight")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "nbs_weights.png"))
    plt.close(fig)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="NBS hoprate threshold search (random offsets, reactive jammer ON)"
    )

    # ---- main loop ------------------------------------------------------------
    p.add_argument("--steps", type=int, default=300,
                   help="Number of environment steps (default: 120)")
    p.add_argument("--output_dir", type=str, default="outputs/speed")
    p.add_argument("--log_file", type=str, default="training_log.txt")

    # ---- environment overrides ------------------------------------------------
    p.add_argument("--enable_sweep", type=lambda x: x.lower() in ("true", "1", "yes"),
                   default=False,
                   help="Enable sweep/comb jammer alongside reactive (default: true)")

    # ---- NBS ------------------------------------------------------------------
    p.add_argument("--nbs_p", type=float, default=settings.NBS_CONFIG["p"],
                   help="NBS noise probability 0 <= p < 0.5 (default: %(default)s)")
    p.add_argument("--nbs_delta", type=float, default=settings.NBS_CONFIG["delta"],
                   help="NBS confidence threshold (default: %(default)s)")
    p.add_argument("--nbs_step", type=float, default=settings.NBS_CONFIG["hoprate_step"],
                   help="NBS candidate step in Hz (default: %(default)s)")
    return p.parse_args()


if __name__ == "__main__":
    settings.set_random_seeds()
    search(parse_args())
