"""FHSS 环境 step 耗时基准：预生成/动态路径 × 串行/自动并行。

用途：量化 `FHSSQPSKEnv` 单步耗时，验证两条加速手段的效果：
- `use_pregen=True`（复用 QPSK 基带与干扰波形）对比动态生成路径；
- `block_workers=None`（自动按 CPU 核数并行 10 个 block）对比串行执行。

`HISTORICAL_BASELINES` 记录开发机上的历史单步中位耗时（秒），用于打印
"相对历史基线的加速比"；`TARGET_MEDIANS`/`TARGET_SPEEDUPS` 是自动并行
路径的验收目标，`--check` 时未达标会以非零码退出（可接入 CI）。

用法（项目根目录或 benchmarks/ 下均可，脚本自带根目录路径注入）::

    python benchmarks/benchmark_env_step.py --steps 20 --check
"""

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


# 开发机历史基线：单步中位耗时（秒），按 use_pregen 区分
HISTORICAL_BASELINES = {
    True: 0.79,
    False: 1.73,
}
# 自动并行路径的单步中位耗时验收目标（秒）
TARGET_MEDIANS = {
    True: 0.35,
    False: 0.45,
}
# 相对历史基线的加速比验收目标
TARGET_SPEEDUPS = {
    True: 2.0,
    False: 3.0,
}


def run_case(use_pregen, block_workers, steps, warmup, seed, hoprate):
    """跑一组基准：固定 offset（全零）下测 ``steps`` 次环境 step 的耗时。

    Args:
        use_pregen: 是否使用预生成加速路径。
        block_workers: 1 表示串行；None 表示按 CPU 核数自动并行。
        steps: 计时步数（正式测量段）。
        warmup: 预热步数（不计时，避免首步建缓存/线程池的偏差）。
        seed: 随机种子。
        hoprate: 固定跳速 (Hz)。

    Returns:
        dict：median/minimum/maximum 单步耗时与 env.block_workers（实际
        生效的 worker 数）。环境在 finally 中关闭，保证线程池释放。
    """
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
    """解析基准参数；--steps 下限 20 保证中位数统计的稳定性。"""
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
    """遍历 2×2 组合（pregen/dynamic × serial/auto）打印耗时并做验收判定。"""
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

        # 自动并行相对串行、以及相对历史基线的加速比与验收目标
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
