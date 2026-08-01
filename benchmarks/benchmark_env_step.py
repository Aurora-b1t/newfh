"""Warm FHSS environment step benchmark for serial and automatic workers."""

import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings  # noqa: E402
from fh_env import FHSSQPSKEnv  # noqa: E402


HISTORICAL_BASELINES = {
    True: 0.79,
    False: 1.73,
}
TARGET_MEDIANS = {
    True: 0.35,
    False: 0.45,
}
TARGET_SPEEDUPS = {
    True: 2.0,
    False: 3.0,
}


def run_case(use_pregen, block_workers, steps, warmup, seed, hoprate):
    settings.set_random_seeds(seed)
    config = dict(settings.ENV_CONFIG)
    config.update(
        use_pregen=use_pregen,
        block_workers=block_workers,
        debug_plot_psd=False,
        debug_log_hops=False,
    )
    env = FHSSQPSKEnv(**config)
    action = {
        "hoprate": float(hoprate),
        "offsets": np.zeros(env.num_blocks, dtype=np.int64),
    }
    try:
        env.reset(seed=seed)
        for _ in range(warmup):
            env.step(action)

        durations = []
        for _ in range(steps):
            start = time.perf_counter()
            env.step(action)
            durations.append(time.perf_counter() - start)
        return {
            "median": statistics.median(durations),
            "minimum": min(durations),
            "maximum": max(durations),
            "workers": env.block_workers,
        }
    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hoprate", type=float, default=100.0)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if an automatic-worker acceptance target is missed.",
    )
    args = parser.parse_args()
    if args.steps < 20:
        parser.error("--steps must be at least 20 for the acceptance benchmark.")
    if args.warmup < 1:
        parser.error("--warmup must be positive.")
    return args


def main():
    args = parse_args()
    failures = []
    print(
        "path      mode    workers  median(s)  min(s)  max(s)  "
        "vs-historical"
    )
    print("-" * 78)

    for use_pregen, path_name in ((True, "pregen"), (False, "dynamic")):
        results = {}
        for requested_workers, mode_name in ((1, "serial"), (None, "auto")):
            result = run_case(
                use_pregen,
                requested_workers,
                args.steps,
                args.warmup,
                args.seed,
                args.hoprate,
            )
            results[mode_name] = result
            historical_speedup = (
                HISTORICAL_BASELINES[use_pregen] / result["median"]
            )
            print(
                f"{path_name:<9} {mode_name:<7} {result['workers']:>7}  "
                f"{result['median']:>9.3f}  {result['minimum']:>6.3f}  "
                f"{result['maximum']:>6.3f}  {historical_speedup:>8.2f}x"
            )

        auto = results["auto"]
        serial_speedup = results["serial"]["median"] / auto["median"]
        historical_speedup = HISTORICAL_BASELINES[use_pregen] / auto["median"]
        print(
            f"  {path_name}: auto vs optimized serial {serial_speedup:.2f}x; "
            f"target median <= {TARGET_MEDIANS[use_pregen]:.2f}s, "
            f"historical speedup >= {TARGET_SPEEDUPS[use_pregen]:.1f}x"
        )
        if auto["median"] > TARGET_MEDIANS[use_pregen]:
            failures.append(
                f"{path_name} median {auto['median']:.3f}s exceeds target"
            )
        if historical_speedup < TARGET_SPEEDUPS[use_pregen]:
            failures.append(
                f"{path_name} speedup {historical_speedup:.2f}x misses target"
            )

    if failures:
        print("\nAcceptance misses:")
        for failure in failures:
            print(f"- {failure}")
        if args.check:
            raise SystemExit(1)
    else:
        print("\nAll automatic-worker acceptance targets passed.")


if __name__ == "__main__":
    main()
