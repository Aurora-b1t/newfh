"""Serialization and validation for step-level FHSS replay data."""

import json
import logging
import os

import numpy as np


FORMAT_VERSION = 3
REPLAY_KEYS = (
    "state_imgs",
    "hoprates",
    "actions",
    "block_rewards",
    "step_rewards",
    "next_state_imgs",
    "next_hoprates",
    "dones",
)


def _as_jsonable(value):
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def environment_metadata(env_config, jammer_config, reward_config):
    """Return the configuration snapshot stored with generated replay data."""
    return {
        "env_config": _as_jsonable(env_config),
        "jammer_config": _as_jsonable(jammer_config),
        "reward_config": _as_jsonable(reward_config),
    }


def _buffer_to_arrays(buffer):
    if buffer.size() == 0:
        raise ValueError("Cannot save an empty replay buffer.")
    arrays = buffer.get_all()
    return {key: np.asarray(arrays[key]) for key in REPLAY_KEYS}


def save_replay_buffer(path, buffer, metadata=None):
    """Save a step-level ReplayBuffer in compressed NumPy format v3."""
    arrays = _buffer_to_arrays(buffer)
    num_step_transitions, num_blocks = arrays["actions"].shape
    expected_step_rewards = arrays["block_rewards"].mean(axis=1)
    if not np.allclose(
        arrays["step_rewards"], expected_step_rewards, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("step_rewards must equal mean(block_rewards) before saving.")

    metadata = dict(metadata or {})
    metadata.update(
        {
            "format_version": FORMAT_VERSION,
            "num_step_transitions": int(num_step_transitions),
            "observation_shape": list(arrays["state_imgs"].shape[1:]),
            "num_actions_observed": int(np.max(arrays["actions"])) + 1,
            "num_blocks": int(num_blocks),
        }
    )
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.savez_compressed(
        path,
        **arrays,
        metadata=np.asarray(json.dumps(_as_jsonable(metadata), sort_keys=True)),
    )


def _validate_format_version(metadata):
    file_version = metadata.get("format_version")
    if file_version is None:
        raise ValueError(
            "Offline replay metadata is missing format_version; regenerate it "
            "with generate_offline_replay.py using format v3."
        )
    try:
        parsed_version = int(file_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Offline replay format_version is invalid: {file_version!r}."
        ) from exc
    if parsed_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported offline replay format version: {parsed_version}; "
            "block-level v1/v2 data cannot be loaded by the step-level SAC. "
            "Regenerate the dataset with generate_offline_replay.py."
        )


def _load_arrays(path):
    try:
        archive = np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Offline replay file does not exist: {path}. Generate a v3 file with "
            "generate_offline_replay.py or use --offline_replay_path none."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read offline replay file '{path}': {exc}") from exc

    with archive:
        metadata_value = (
            archive["metadata"] if "metadata" in archive else np.asarray("{}")
        )
        try:
            metadata = json.loads(str(metadata_value.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Offline replay metadata is not valid JSON.") from exc
        _validate_format_version(metadata)

        missing = [key for key in REPLAY_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Offline replay is missing fields: {missing}")
        arrays = {key: np.asarray(archive[key]) for key in REPLAY_KEYS}
    return arrays, metadata


def _validate_array_shapes(arrays, expected_num_blocks):
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Offline replay fields have inconsistent lengths: {lengths}")
    count = next(iter(lengths.values()))
    if count == 0:
        raise ValueError("Offline replay contains no transitions.")

    vector_keys = ("hoprates", "step_rewards", "next_hoprates", "dones")
    for key in vector_keys:
        if arrays[key].shape != (count,):
            raise ValueError(
                f"Offline replay {key} must have shape ({count},), "
                f"got {arrays[key].shape}."
            )

    matrix_shape = (count, expected_num_blocks)
    for key in ("actions", "block_rewards"):
        if arrays[key].shape != matrix_shape:
            raise ValueError(
                f"Offline replay {key} must have shape {matrix_shape}, "
                f"got {arrays[key].shape}."
            )
    return count


def load_replay_into_buffer(
    path,
    buffer,
    expected_observation_shape=None,
    expected_num_actions=None,
    expected_num_blocks=10,
    current_environment_metadata=None,
    logger=None,
):
    """Validate a v3 offline replay file and append it to a ReplayBuffer."""
    arrays, metadata = _load_arrays(path)
    expected_num_blocks = int(expected_num_blocks)
    count = _validate_array_shapes(arrays, expected_num_blocks)

    state_shape = tuple(arrays["state_imgs"].shape[1:])
    next_state_shape = tuple(arrays["next_state_imgs"].shape[1:])
    if state_shape != next_state_shape:
        raise ValueError(
            f"State and next-state shapes differ: {state_shape} vs {next_state_shape}."
        )
    if (
        expected_observation_shape is not None
        and state_shape != tuple(expected_observation_shape)
    ):
        raise ValueError(
            "Offline replay observation shape does not match the current environment: "
            f"file={state_shape}, environment={tuple(expected_observation_shape)}."
        )

    for key in (
        "state_imgs",
        "hoprates",
        "block_rewards",
        "step_rewards",
        "next_state_imgs",
        "next_hoprates",
        "dones",
    ):
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Offline replay contains non-finite values in {key}.")

    raw_actions = arrays["actions"]
    rounded_actions = np.rint(raw_actions)
    if not np.allclose(raw_actions, rounded_actions):
        raise ValueError("Offline replay actions must be integer-valued.")
    actions = rounded_actions.astype(np.int64)
    if np.any(actions < 0):
        raise ValueError("Offline replay actions must be non-negative.")
    if expected_num_actions is not None:
        expected_num_actions = int(expected_num_actions)
        if np.any(actions >= expected_num_actions):
            raise ValueError(
                "Offline replay contains actions outside "
                f"[0, {expected_num_actions - 1}]."
            )
        file_num_actions = metadata.get("num_actions")
        if (
            file_num_actions is not None
            and int(file_num_actions) != expected_num_actions
        ):
            raise ValueError(
                "Offline replay action-space size does not match the environment: "
                f"file={file_num_actions}, environment={expected_num_actions}."
            )

    expected_step_rewards = arrays["block_rewards"].mean(axis=1)
    if not np.allclose(
        arrays["step_rewards"], expected_step_rewards, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Offline replay step_rewards must equal mean(block_rewards).")

    if not np.all(np.isin(arrays["dones"], (0, 1, False, True))):
        raise ValueError("Offline replay dones must contain only boolean values.")
    if int(metadata.get("num_blocks", expected_num_blocks)) != expected_num_blocks:
        raise ValueError("Offline replay was generated with a different block count.")
    metadata_count = metadata.get("num_step_transitions")
    if metadata_count is not None and int(metadata_count) != count:
        raise ValueError(
            "Offline replay transition count does not match metadata: "
            f"arrays={count}, metadata={metadata_count}."
        )
    if count > buffer.capacity:
        raise ValueError(
            f"Offline replay has {count} transitions but replay capacity is only "
            f"{buffer.capacity}."
        )
    if getattr(buffer, "num_heads", expected_num_blocks) != expected_num_blocks:
        raise ValueError("Replay buffer and offline file use different block counts.")

    for idx in range(count):
        buffer.add(
            arrays["state_imgs"][idx],
            arrays["hoprates"][idx],
            actions[idx],
            arrays["block_rewards"][idx],
            arrays["next_state_imgs"][idx],
            arrays["next_hoprates"][idx],
            bool(arrays["dones"][idx]),
        )

    if current_environment_metadata is not None:
        stored_config = {
            key: metadata.get(key)
            for key in ("env_config", "jammer_config", "reward_config")
        }
        current_config = _as_jsonable(current_environment_metadata)
        if any(
            stored_config.get(key) is not None
            and current_config.get(key) != stored_config.get(key)
            for key in current_config
        ):
            (logger or logging.getLogger(__name__)).warning(
                "Offline replay environment configuration differs from the current "
                "settings; use a dataset generated for the intended environment."
            )

    return count, metadata
