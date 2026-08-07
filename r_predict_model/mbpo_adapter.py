"""Adapters between step-level SAC replay and the MBPO reward ensemble."""

import time

import numpy as np

import settings
from .model import (
    DEFAULT_BER_MAX,
    DEFAULT_BER_MIN,
    RewardReplayDataset,
    reward_bounds_from_config,
)


def replay_fields_for_reward_model(replay_buffer):
    """Return replay-backed reward-model fields without an extraction copy."""
    transitions = tuple(replay_buffer.buffer)
    if not transitions:
        raise ValueError("Cannot train the reward model from an empty replay buffer.")
    return {
        "state_imgs": [transition[0] for transition in transitions],
        "hoprates": [transition[1] for transition in transitions],
        "actions": [transition[2] for transition in transitions],
        "block_rewards": [transition[3] for transition in transitions],
    }


def concat_transition_batches(batches, shuffle=True):
    """Concatenate complete SAC samples and optionally shuffle the result."""
    valid_batches = [batch for batch in batches if batch is not None]
    if not valid_batches:
        raise ValueError("Need at least one batch to concatenate.")
    keys = tuple(valid_batches[0].keys())
    if any(tuple(batch.keys()) != keys for batch in valid_batches[1:]):
        raise ValueError("Transition batches do not have the same schema.")
    combined = {
        key: np.concatenate([batch[key] for batch in valid_batches], axis=0)
        for key in keys
    }
    if shuffle and len(next(iter(combined.values()))) > 1:
        permutation = np.random.permutation(len(next(iter(combined.values()))))
        combined = {key: value[permutation] for key, value in combined.items()}
    return combined


def _target_real_count(batch_size, real_ratio):
    if not 0.0 <= real_ratio <= 1.0:
        raise ValueError("real_ratio must be in [0, 1].")
    real_count = int(round(batch_size * real_ratio))
    if batch_size >= 2 and 0.0 < real_ratio < 1.0:
        real_count = min(max(1, real_count), batch_size - 1)
    return real_count


def sample_mixed_batch(real_buffer, model_buffer, batch_size, real_ratio):
    """Sample exactly ``batch_size`` step transitions from real/model replay."""
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    real_ratio = float(real_ratio)
    total_available = real_buffer.size() + model_buffer.size()
    if total_available < batch_size:
        raise ValueError(
            f"Need {batch_size} total transitions, only {total_available} available."
        )

    if model_buffer.size() == 0:
        if real_buffer.size() < batch_size:
            raise ValueError("Real replay is too small for a fallback SAC batch.")
        return real_buffer.sample(batch_size)

    real_count = min(
        _target_real_count(batch_size, real_ratio), real_buffer.size()
    )
    model_count = min(batch_size - real_count, model_buffer.size())
    deficit = batch_size - real_count - model_count
    if deficit:
        additional_real = min(deficit, real_buffer.size() - real_count)
        real_count += additional_real
        deficit -= additional_real
    if deficit:
        additional_model = min(deficit, model_buffer.size() - model_count)
        model_count += additional_model
        deficit -= additional_model
    if deficit:
        raise ValueError("Replay buffers cannot supply a complete mixed batch.")

    batches = []
    if real_count:
        batches.append(real_buffer.sample(real_count))
    if model_count:
        batches.append(model_buffer.sample(model_count))
    # SAC uses GroupNorm rather than batch-statistics layers, so the two
    # independently randomized samples do not need a second full-array shuffle.
    return concat_transition_batches(batches, shuffle=False)


def train_reward_model_from_replay(
    reward_model, replay_buffer, dataset_cache=None, **fit_kwargs
):
    """Continue fitting the ensemble on the full current real replay."""
    if dataset_cache is not None:
        if not isinstance(dataset_cache, RewardReplayDataset):
            raise TypeError("dataset_cache must be a RewardReplayDataset.")
        dataset_cache.sync(replay_buffer)
        return reward_model.fit(dataset=dataset_cache, **fit_kwargs)
    fields = replay_fields_for_reward_model(replay_buffer)
    return reward_model.fit(**fields, **fit_kwargs)


def reward_bounds(
    hoprates,
    reward_config,
    ber_min=DEFAULT_BER_MIN,
    ber_max=DEFAULT_BER_MAX,
):
    """Return per-transition reward bounds implied by BER endpoints."""
    return reward_bounds_from_config(
        hoprates,
        reward_config,
        ber_min=ber_min,
        ber_max=ber_max,
    )


def rollout_reward_model(
    reward_model,
    agent,
    real_buffer,
    model_buffer,
    batch_size,
    reward_config,
    deterministic_model=False,
):
    """Append one-step synthetic transitions to the persistent FIFO replay."""
    if real_buffer.size() == 0:
        raise ValueError("Cannot roll out from an empty real replay buffer.")
    rollout_size = min(int(batch_size), real_buffer.size())
    if rollout_size <= 0:
        raise ValueError("rollout batch_size must be positive.")
    model_buffer_size_before = model_buffer.size()
    timing_enabled = bool(settings.TIMING_ENABLED)
    timing = {"sample_s": 0.0, "policy_s": 0.0, "predict_s": 0.0, "add_s": 0.0}
    stage_start = time.time() if timing_enabled else None

    starts = real_buffer.sample(rollout_size)
    if timing_enabled:
        timing["sample_s"] = time.time() - stage_start
        stage_start = time.time()
    state_imgs = starts["state_imgs"]
    hoprates = np.asarray(starts["hoprates"], dtype=np.float32)
    take_actions = getattr(agent, "take_actions", None)
    if callable(take_actions):
        actions = np.asarray(
            take_actions(state_imgs, hoprates),
            dtype=np.int64,
        )
    else:
        # Keep lightweight test agents and external SAC-compatible agents that
        # only expose the original single-state API working.
        actions = np.stack(
            [
                np.asarray(
                    agent.take_action(state_imgs[index], float(hoprates[index])),
                    dtype=np.int64,
                )
                for index in range(rollout_size)
            ]
        )
    if timing_enabled:
        timing["policy_s"] = time.time() - stage_start
        stage_start = time.time()
    expected_shape = (rollout_size, reward_model.num_heads)
    if actions.shape != expected_shape:
        raise ValueError(
            f"SAC policy returned actions with shape {actions.shape}, "
            f"expected {expected_shape}."
        )
    if np.any(actions < 0) or np.any(actions >= reward_model.n_actions):
        raise ValueError("SAC policy returned an action outside the environment range.")

    predicted_rewards, prediction_stats = reward_model.sample_rewards(
        state_imgs,
        hoprates,
        actions,
        deterministic=deterministic_model,
    )
    if timing_enabled:
        timing["predict_s"] = time.time() - stage_start
        stage_start = time.time()
    if predicted_rewards.shape != expected_shape or not np.all(
        np.isfinite(predicted_rewards)
    ):
        raise RuntimeError("Reward ensemble returned invalid block rewards.")
    lower_bounds, upper_bounds = reward_bounds(
        hoprates,
        reward_config,
        ber_min=float(getattr(reward_model, "ber_min", DEFAULT_BER_MIN)),
        ber_max=float(getattr(reward_model, "ber_max", DEFAULT_BER_MAX)),
    )
    tolerance = 1e-6 * np.maximum(1.0, upper_bounds - lower_bounds)
    if np.any(predicted_rewards < lower_bounds - tolerance) or np.any(
        predicted_rewards > upper_bounds + tolerance
    ):
        raise RuntimeError("Reward ensemble returned a reward outside its bounds.")
    predicted_rewards = predicted_rewards.astype(np.float32, copy=False)

    model_buffer.add_batch(
        state_imgs,
        hoprates,
        actions,
        predicted_rewards,
        starts["next_state_imgs"],
        starts["next_hoprates"],
        starts["dones"],
    )
    if timing_enabled:
        timing["add_s"] = time.time() - stage_start
        timing["total_s"] = sum(timing.values())

    model_buffer_size_after = model_buffer.size()
    fifo_evicted = max(
        0,
        model_buffer_size_before + rollout_size - model_buffer_size_after,
    )
    disagreement = np.asarray(
        prediction_stats["disagreement"], dtype=np.float32
    )
    return {
        "generated": rollout_size,
        "model_buffer_size_before": model_buffer_size_before,
        "model_buffer_size_after": model_buffer_size_after,
        "model_buffer_capacity": model_buffer.capacity,
        "fifo_evicted": fifo_evicted,
        "reward_mean": float(np.mean(predicted_rewards)),
        "reward_std": float(np.std(predicted_rewards)),
        "disagreement_mean": float(np.mean(disagreement)),
        "disagreement_p95": float(np.percentile(disagreement, 95)),
        "timing": (timing if timing_enabled else None),
    }
