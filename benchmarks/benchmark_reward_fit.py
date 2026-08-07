"""Profile one step-level reward-model fit on a prepared replay archive."""

import argparse
from collections import deque
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
from r_predict_model import RewardReplayDataset, StepRewardEnsemble  # noqa: E402


class PreparedReplay:
    def __init__(self, states, hoprates, actions, rewards, capacity=20000):
        self.capacity = int(capacity)
        self.n_actions = int(np.max(actions)) + 1
        self.buffer = deque(maxlen=self.capacity)
        for index in range(len(states)):
            self.buffer.append(
                (
                    states[index],
                    float(hoprates[index]),
                    actions[index],
                    rewards[index],
                    float(np.mean(rewards[index])),
                    states[index],
                    float(hoprates[index]),
                    False,
                )
            )


def load_reward_fields(path):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", required=True)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--persistent-cache", action="store_true")
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
        torch.set_float32_matmul_precision("high")
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
    dataset_cache = None
    if args.persistent_cache:
        replay = PreparedReplay(
            np.asarray(state_imgs),
            np.asarray(hoprates),
            np.asarray(actions),
            np.asarray(block_rewards),
        )
        dataset_cache = RewardReplayDataset(
            device="cuda",
            cache_on_device=args.cache,
        )
        dataset_cache.sync(replay)

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
        "cache_dataset_on_device": args.cache,
    }
    all_stats = []
    for _repeat in range(args.repeat):
        if dataset_cache is None:
            stats = model.fit(
                state_imgs,
                hoprates,
                actions,
                block_rewards,
                **fit_kwargs,
            )
        else:
            dataset_cache.sync(replay)
            stats = model.fit(dataset=dataset_cache, **fit_kwargs)
        all_stats.append(stats)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    stats = all_stats[-1]
    wall_end = time.perf_counter()
    result = {
        "state_shape": list(state_shape),
        "count": len(state_imgs),
        "batch_size": batch_size,
        "max_epochs": max_epochs,
        "patience": patience,
        "cache": args.cache,
        "persistent_cache": args.persistent_cache,
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
