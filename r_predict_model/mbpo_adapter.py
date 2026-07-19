"""
Adapters between the FHSS discrete SAC replay format and MBPO reward model.

The SAC replay buffer stores structured transition fields such as PSD images,
hop rates, block indices, and offset actions.  The reward ensemble expects a
flat supervised-learning matrix.  This module owns that conversion so the
training script and the model implementation stay independent of replay-buffer
layout details.
"""

import numpy as np


def encode_transition_inputs(state_imgs, hoprates, block_idxs, actions):
    """
    Flatten SAC transition fields into reward-model inputs.

    Output shape is [batch_size, flattened_state_size + 3], where the last
    three columns are hop rate, block index, and action.  Keeping these metadata
    columns explicit lets the reward model learn block- and action-dependent
    reward structure even when the PSD image is shared across block transitions.
    """
    state_flat = np.asarray(state_imgs, dtype=np.float32).reshape(len(state_imgs), -1)
    hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1, 1)
    block_idxs = np.asarray(block_idxs, dtype=np.float32).reshape(-1, 1)
    actions = np.asarray(actions, dtype=np.float32).reshape(-1, 1)
    return np.concatenate([state_flat, hoprates, block_idxs, actions], axis=1)


def replay_sample_to_model_data(sample):
    """
    Convert a SAC replay sample dictionary into supervised model data.

    Returns:
        inputs: Encoded transition matrix.
        labels: Reward column vector with shape [batch_size, 1].
    """
    inputs = encode_transition_inputs(
        sample["state_imgs"],
        sample["hoprates"],
        sample["block_idxs"],
        sample["actions"],
    )
    labels = np.asarray(sample["rewards"], dtype=np.float32).reshape(-1, 1)
    return inputs, labels


def concat_transition_batches(batches):
    """
    Concatenate non-empty SAC replay sample dictionaries.

    SAC update code expects a single replay sample dictionary.  This helper is
    used after separately sampling from real and synthetic buffers.
    """
    valid_batches = [batch for batch in batches if batch is not None]
    if not valid_batches:
        raise ValueError("Need at least one batch to concatenate.")
    keys = valid_batches[0].keys()
    return {
        key: np.concatenate([batch[key] for batch in valid_batches], axis=0)
        for key in keys
    }


def train_reward_model_from_replay(reward_model, replay_buffer, batch_size):
    """
    Train the reward model from all currently available real replay samples.

    Synthetic model-buffer samples are intentionally excluded so the supervised
    target distribution remains grounded in true environment feedback.
    """
    sample = replay_buffer.sample(replay_buffer.size())
    inputs, labels = replay_sample_to_model_data(sample)
    return reward_model.train(inputs, labels, batch_size=batch_size)


def rollout_reward_model(
    reward_model,
    agent,
    real_buffer,
    model_buffer,
    batch_size,
    fixed_hoprate,
    n_actions,
    deterministic_model=False,
):
    """
    Generate one-step synthetic transitions with the learned reward model.

    The rollout does not predict the next PSD image. It reuses the sampled real
    transition's ``next_state_img``: internal blocks keep the current PSD, while
    block 9 advances to the next environment PSD. This keeps model usage limited
    to reward augmentation, matching the assumptions in ``train_mbpo.py``.
    """
    starts = real_buffer.sample(batch_size)
    state_imgs = starts["state_imgs"]
    next_state_imgs = starts["next_state_imgs"]
    hoprates = np.asarray(
        starts.get("hoprates", np.full(len(state_imgs), float(fixed_hoprate))),
        dtype=np.float32,
    )
    next_hoprates = np.asarray(
        starts.get("next_hoprates", np.full(len(state_imgs), float(fixed_hoprate))),
        dtype=np.float32,
    )
    block_idxs = starts["block_idxs"]

    actions = np.zeros(len(state_imgs), dtype=np.int64)
    for i in range(len(state_imgs)):
        # Query the current SAC policy on real replay states, then clip to the
        # discrete offset-action range accepted by the environment.
        action = agent.take_action(state_imgs[i], float(hoprates[i]), block_idxs[i])
        actions[i] = int(np.clip(action, 0, n_actions - 1))

    model_inputs = encode_transition_inputs(state_imgs, hoprates, block_idxs, actions)
    rewards = reward_model.predict_reward(model_inputs, deterministic=deterministic_model)

    for i in range(len(state_imgs)):
        block_idx = int(np.clip(round(float(block_idxs[i])), 0, 9))
        next_block_idx = (block_idx + 1) % 10
        # Only the reward is synthetic here. The sequential next state is copied
        # from the sampled real transition because this model learns no dynamics.
        model_buffer.add(
            state_imgs[i],
            float(hoprates[i]),
            block_idx,
            int(actions[i]),
            float(rewards[i]),
            next_state_imgs[i].copy(),
            float(next_hoprates[i]),
            next_block_idx,
            False,
        )

    return {
        "generated": int(len(state_imgs)),
        "reward_mean": float(np.mean(rewards)) if len(rewards) else 0.0,
        "reward_std": float(np.std(rewards)) if len(rewards) else 0.0,
    }
