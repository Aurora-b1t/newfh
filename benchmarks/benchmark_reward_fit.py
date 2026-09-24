"""奖励模型单次拟合的性能剖析基准（在准备好的 replay 档案上）。

用途：用 `StepRewardEnsemble.fit()` 对一份 v3 replay 做完整拟合，输出
墙钟耗时、逐 epoch 数、holdout 损失与 CUDA 峰值显存（JSON 一行），
用于对比不同精度（float32/bfloat16）、fast-math、torch.compile、
batch size 等运行时开关的性能收益。

说明：
- 需要 CUDA（模型 device 固定 "cuda"）；
- `--repeat` 可重复拟合多次（每次都从头重拟合），耗时统计取整体墙钟，
  单次统计取最后一次的返回值；
- 档案需包含 state_imgs / hoprates / actions / block_rewards 四个键。

用法（项目根目录或 benchmarks/ 下均可）::

    python benchmarks/benchmark_reward_fit.py --replay outputs/offline_replay/replay_5000_100_hoprate_v3.npz --precision bfloat16 --fast-math --compile
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import settings  # noqa: E402
from r_predict_model import StepRewardEnsemble  # noqa: E402


def load_reward_fields(path):
    """从 v3 replay ``.npz`` 读出奖励模型需要的四个字段。

    Returns:
        ``(state_imgs, hoprates, actions, block_rewards, state_shape, action_shape)``；
        前四项转为逐条 Python 列表（fit 接口按样本序列消费），后两项为
        原始数组形状（用于回填 num_heads / n_actions 与结果记录）。
    """
    with np.load(path) as payload:
        state_imgs = np.asarray(payload["state_imgs"], dtype=np.float32)
        hoprates = np.asarray(payload["hoprates"], dtype=np.float32)
        actions = np.asarray(payload["actions"], dtype=np.int64)
        block_rewards = np.asarray(payload["block_rewards"], dtype=np.float32)
    return (
        [state_imgs[index] for index in range(len(state_imgs))],
        [hoprates[index] for index in range(len(hoprates))],
        [actions[index] for index in range(len(actions))],
        [block_rewards[index] for index in range(len(block_rewards))],
        state_imgs.shape,
        actions.shape,
    )


def main():
    """解析参数 → 构建奖励模型 → 计时拟合 → 打印 JSON 性能报告。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--precision",
        choices=("float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--fast-math", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--repeat", type=int, default=1)
    args = parser.parse_args()

    settings.set_random_seeds(args.seed)
    fields = load_reward_fields(args.replay)
    state_imgs, hoprates, actions, block_rewards, state_shape, action_shape = fields
    if args.fast_math and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        # set_random_seeds() 打开了 deterministic cuDNN，这会禁用 benchmark
        # 模式；此处显式清除，让基准测到真实的 fast-math 内核速度。
        torch.backends.cudnn.deterministic = False
        torch.set_float32_matmul_precision("high")
    # 未显式指定的参数回落到 settings.MBPO_CONFIG，保证与训练配置同源
    batch_size = args.batch_size or settings.MBPO_CONFIG["model_train_batch_size"]
    max_epochs = args.epochs or settings.MBPO_CONFIG["max_epochs"]
    patience = (
        settings.MBPO_CONFIG["early_stop_patience"]
        if args.patience is None
        else args.patience
    )
    model = StepRewardEnsemble(
        network_size=settings.MBPO_CONFIG["num_networks"],
        elite_size=settings.MBPO_CONFIG["num_elites"],
        num_heads=action_shape[1],
        n_actions=int(np.max(actions)) + 1,
        reward_config=settings.REWARD_CONFIG,
        hoprate_min=settings.ENV_CONFIG["hoprate_min"],
        hoprate_max=settings.ENV_CONFIG["hoprate_max"],
        hidden_size=settings.MBPO_CONFIG["hidden_size"],
        learning_rate=settings.MBPO_CONFIG["learning_rate"],
        weight_decay=settings.MBPO_CONFIG["weight_decay"],
        device="cuda",
        precision=args.precision,
        compile_model=args.compile,
    )
    if args.repeat <= 0:
        raise ValueError("--repeat must be positive.")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    fit_start = time.perf_counter()
    fit_kwargs = {
        "batch_size": batch_size,
        "holdout_ratio": settings.MBPO_CONFIG["holdout_ratio"],
        "patience": patience,
        "max_epochs": max_epochs,
        "min_improvement": settings.MBPO_CONFIG["min_improvement"],
    }
    all_stats = []
    for _repeat in range(args.repeat):
        stats = model.fit(
            state_imgs,
            hoprates,
            actions,
            block_rewards,
            **fit_kwargs,
        )
        all_stats.append(stats)
    if torch.cuda.is_available():
        # 同步后再读墙钟，避免把异步 kernel 的时间漏算
        torch.cuda.synchronize()
    stats = all_stats[-1]
    wall_end = time.perf_counter()
    result = {
        "state_shape": list(state_shape),
        "count": len(state_imgs),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "precision": args.precision,
        "fast_math": args.fast_math,
        "compile": args.compile,
        "repeat": args.repeat,
        "fit_wall_sec": wall_end - fit_start,
        "fit_wall_sec_per_repeat": (wall_end - fit_start) / args.repeat,
        "all_epochs": [item["epochs"] for item in all_stats],
        "all_holdout_loss_mean": [
            item["holdout_loss_mean"] for item in all_stats
        ],
        "reported_fit_sec": stats.get("fit_time_sec"),
        "epochs": stats["epochs"],
        "holdout_loss_mean": stats["holdout_loss_mean"],
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated() / 2**20
            if torch.cuda.is_available()
            else None
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved() / 2**20
            if torch.cuda.is_available()
            else None
        ),
    }
    print(json.dumps(result, ensure_ascii=True))


if __name__ == "__main__":
    main()
