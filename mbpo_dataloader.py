"""GPU-resident replay datasets and DataLoader utilities for standalone MBPO."""

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate


def _collate_replay_batch(batch):
    """Collate either a vectorized dataset batch or ordinary samples."""
    if isinstance(batch, dict):
        return {
            key: value
            if torch.is_tensor(value)
            else torch.as_tensor(value)
            for key, value in batch.items()
        }
    if isinstance(batch, list) and batch and isinstance(batch[0], dict):
        return default_collate(batch)
    return default_collate(batch)


class ReplayBufferDataset(Dataset):
    """Mutation-aware replay view backed by CPU or CUDA ring storage.

    The source ``ReplayBuffer`` remains the owner of the public transition
    deque. This dataset only mirrors the fields needed by SAC, updating new
    append-only entries and normal one-entry FIFO evictions in place.
    """

    def __init__(self, replay_buffer, device=None, cache_on_device=True):
        self.replay_buffer = replay_buffer
        self.device = torch.device(device or "cpu")
        self.cache_on_device = bool(
            cache_on_device and self.device.type == "cuda"
        )
        self.capacity = None
        self.size = 0
        self._start = 0
        self._transition_refs = None
        self._state_shape = None
        self._normalized_state_shape = None
        self._state_storage = None
        self._next_state_storage = None
        self._hoprate_storage = None
        self._actions_storage = None
        self._rewards_storage = None
        self._step_rewards_storage = None
        self._next_hoprate_storage = None
        self._dones_storage = None

    @property
    def is_cached(self):
        return self.cache_on_device

    def __len__(self):
        return self.size

    @staticmethod
    def _state_array(state, expected_shape=None):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim not in (2, 3):
            raise ValueError("Each state image must have shape [H,W] or [C,H,W].")
        if expected_shape is not None and tuple(state_array.shape) != tuple(
            expected_shape
        ):
            raise ValueError(
                "State image shape differs from replay dataset: "
                f"expected {expected_shape}, got {state_array.shape}."
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("State images must contain only finite values.")
        return state_array

    def _allocate_storage(self, capacity, state_shape):
        self.capacity = int(capacity)
        self._state_shape = tuple(int(value) for value in state_shape)
        if len(self._state_shape) == 2:
            self._normalized_state_shape = (1, *self._state_shape)
        else:
            self._normalized_state_shape = self._state_shape

        if self.cache_on_device:
            storage_factory = lambda shape, dtype: torch.empty(
                shape, dtype=dtype, device=self.device
            )
            float_dtype = torch.float32
            long_dtype = torch.long
        else:
            storage_factory = lambda shape, dtype: np.empty(shape, dtype=dtype)
            float_dtype = np.float32
            long_dtype = np.int64

        self._state_storage = storage_factory(
            (self.capacity, *self._normalized_state_shape), float_dtype
        )
        self._next_state_storage = storage_factory(
            (self.capacity, *self._normalized_state_shape), float_dtype
        )
        self._hoprate_storage = storage_factory((self.capacity,), float_dtype)
        self._actions_storage = storage_factory(
            (self.capacity, self.replay_buffer.num_heads), long_dtype
        )
        self._rewards_storage = storage_factory(
            (self.capacity, self.replay_buffer.num_heads), float_dtype
        )
        self._step_rewards_storage = storage_factory((self.capacity,), float_dtype)
        self._next_hoprate_storage = storage_factory((self.capacity,), float_dtype)
        self._dones_storage = storage_factory((self.capacity,), float_dtype)

    def _transition_arrays(self, transitions):
        states = np.stack(
            [self._state_array(transition[0], self._state_shape) for transition in transitions]
        )
        next_states = np.stack(
            [
                self._state_array(transition[5], self._state_shape)
                for transition in transitions
            ]
        )
        if states.ndim == 3:
            states = states[:, np.newaxis, ...]
            next_states = next_states[:, np.newaxis, ...]
        hoprates = np.asarray(
            [float(transition[1]) for transition in transitions], dtype=np.float32
        )
        actions = np.asarray(
            [transition[2] for transition in transitions], dtype=np.int64
        )
        rewards = np.asarray(
            [transition[3] for transition in transitions], dtype=np.float32
        )
        step_rewards = np.asarray(
            [float(transition[4]) for transition in transitions], dtype=np.float32
        )
        next_hoprates = np.asarray(
            [float(transition[6]) for transition in transitions], dtype=np.float32
        )
        dones = np.asarray(
            [float(bool(transition[7])) for transition in transitions],
            dtype=np.float32,
        )
        return (
            states,
            hoprates,
            actions,
            rewards,
            step_rewards,
            next_states,
            next_hoprates,
            dones,
        )

    def _write_arrays(self, slots, arrays):
        slots = np.asarray(slots, dtype=np.int64).reshape(-1)
        if not len(slots):
            return
        if self.cache_on_device:
            index = torch.as_tensor(slots, dtype=torch.long, device=self.device)
            storages = (
                self._state_storage,
                self._hoprate_storage,
                self._actions_storage,
                self._rewards_storage,
                self._step_rewards_storage,
                self._next_state_storage,
                self._next_hoprate_storage,
                self._dones_storage,
            )
            dtypes = (
                torch.float32,
                torch.float32,
                torch.long,
                torch.float32,
                torch.float32,
                torch.float32,
                torch.float32,
                torch.float32,
            )
            for storage, values, dtype in zip(storages, arrays, dtypes):
                storage.index_copy_(
                    0,
                    index,
                    torch.as_tensor(values, dtype=dtype, device=self.device),
                )
            return

        storages = (
            self._state_storage,
            self._hoprate_storage,
            self._actions_storage,
            self._rewards_storage,
            self._step_rewards_storage,
            self._next_state_storage,
            self._next_hoprate_storage,
            self._dones_storage,
        )
        for storage, values in zip(storages, arrays):
            storage[slots] = values

    def _write_transitions(self, slots, transitions):
        self._write_arrays(slots, self._transition_arrays(transitions))

    def _rebuild(self, transitions):
        if not transitions:
            raise ValueError("Cannot build a replay dataset from an empty buffer.")
        first_state = self._state_array(transitions[0][0])
        self._allocate_storage(self.replay_buffer.capacity, first_state.shape)
        self.size = len(transitions)
        self._start = 0
        self._write_transitions(np.arange(self.size, dtype=np.int64), transitions)
        self._transition_refs = tuple(transitions)

    @staticmethod
    def _same_refs(left, right):
        return len(left) == len(right) and all(
            previous is current for previous, current in zip(left, right)
        )

    def sync(self, replay_buffer=None):
        """Synchronize the mirror after replay append/FIFO mutations."""
        if replay_buffer is not None:
            self.replay_buffer = replay_buffer
        transitions = tuple(self.replay_buffer.buffer)
        if not transitions:
            self.size = 0
            self._start = 0
            self._transition_refs = ()
            return
        if self._transition_refs is None or self.capacity is None:
            self._rebuild(transitions)
            return

        previous = self._transition_refs
        if self._same_refs(previous, transitions):
            return

        if (
            len(transitions) > len(previous)
            and len(transitions) <= self.capacity
            and self._same_refs(previous, transitions[: len(previous)])
        ):
            new_transitions = transitions[len(previous) :]
            slots = (
                self._start + np.arange(self.size, self.size + len(new_transitions))
            ) % self.capacity
            self._write_transitions(slots, new_transitions)
            self.size += len(new_transitions)
            self._transition_refs = transitions
            return

        if (
            len(transitions) == len(previous) == self.capacity
            and len(transitions) > 1
            and self._same_refs(previous[1:], transitions[:-1])
        ):
            self._write_transitions([self._start], [transitions[-1]])
            self._start = (self._start + 1) % self.capacity
            self._transition_refs = transitions
            return

        if (
            len(transitions) == len(previous) == self.capacity == 1
            and transitions[0] is not previous[0]
        ):
            self._write_transitions([self._start], [transitions[0]])
            self._transition_refs = transitions
            return

        self._rebuild(transitions)

    def _physical_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Replay dataset index is outside the active range.")
        return (indices + self._start) % self.capacity

    def _batch_dict(self, indices):
        logical_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        physical_indices = self._physical_indices(logical_indices)
        if self.cache_on_device:
            index = torch.as_tensor(
                physical_indices, dtype=torch.long, device=self.device
            )
            return {
                "state_imgs": self._state_storage.index_select(0, index),
                "hoprates": self._hoprate_storage.index_select(0, index),
                "actions": self._actions_storage.index_select(0, index),
                "block_rewards": self._rewards_storage.index_select(0, index),
                "step_rewards": self._step_rewards_storage.index_select(0, index),
                "next_state_imgs": self._next_state_storage.index_select(0, index),
                "next_hoprates": self._next_hoprate_storage.index_select(0, index),
                "dones": self._dones_storage.index_select(0, index),
            }

        return {
            "state_imgs": self._state_storage[physical_indices],
            "hoprates": self._hoprate_storage[physical_indices],
            "actions": self._actions_storage[physical_indices],
            "block_rewards": self._rewards_storage[physical_indices],
            "step_rewards": self._step_rewards_storage[physical_indices],
            "next_state_imgs": self._next_state_storage[physical_indices],
            "next_hoprates": self._next_hoprate_storage[physical_indices],
            "dones": self._dones_storage[physical_indices],
        }

    def __getitem__(self, index):
        batch = self._batch_dict([int(index)])
        return {key: value[0] for key, value in batch.items()}

    def __getitems__(self, indices):
        return self._batch_dict(indices)


def _target_real_count(batch_size, real_ratio):
    if not 0.0 <= float(real_ratio) <= 1.0:
        raise ValueError("real_ratio must be in [0, 1].")
    real_count = int(round(int(batch_size) * float(real_ratio)))
    if batch_size >= 2 and 0.0 < real_ratio < 1.0:
        real_count = min(max(1, real_count), batch_size - 1)
    return real_count


def _resolve_mixed_counts(real_size, model_size, batch_size, real_ratio):
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if real_size + model_size < batch_size:
        raise ValueError(
            f"Need {batch_size} total transitions, only "
            f"{real_size + model_size} available."
        )
    if model_size == 0:
        if real_size < batch_size:
            raise ValueError("Real replay is too small for a fallback SAC batch.")
        return batch_size, 0

    real_count = min(_target_real_count(batch_size, real_ratio), real_size)
    model_count = min(batch_size - real_count, model_size)
    deficit = batch_size - real_count - model_count
    if deficit:
        additional_real = min(deficit, real_size - real_count)
        real_count += additional_real
        deficit -= additional_real
    if deficit:
        additional_model = min(deficit, model_size - model_count)
        model_count += additional_model
        deficit -= additional_model
    if deficit:
        raise ValueError("Replay buffers cannot supply a complete mixed batch.")
    return real_count, model_count


class MixedReplayDataset(Dataset):
    """Concatenate two replay datasets while preserving source batch groups."""

    def __init__(self, real_dataset, model_dataset):
        self.real_dataset = real_dataset
        self.model_dataset = model_dataset
        self.real_size = len(real_dataset)

    def __len__(self):
        return self.real_size + len(self.model_dataset)

    def _merge(self, batches):
        if len(batches) == 1:
            return batches[0]
        merged = {}
        for key in batches[0]:
            values = [batch[key] for batch in batches]
            if all(torch.is_tensor(value) for value in values):
                merged[key] = torch.cat(values, dim=0)
            else:
                merged[key] = np.concatenate(values, axis=0)
        return merged

    def __getitem__(self, index):
        index = int(index)
        if index < self.real_size:
            return self.real_dataset[index]
        return self.model_dataset[index - self.real_size]

    def __getitems__(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        real_indices = indices[indices < self.real_size]
        model_indices = indices[indices >= self.real_size] - self.real_size
        batches = []
        if len(real_indices):
            batches.append(self.real_dataset.__getitems__(real_indices))
        if len(model_indices):
            batches.append(self.model_dataset.__getitems__(model_indices))
        if not batches:
            raise ValueError("Cannot build an empty mixed replay batch.")
        return self._merge(batches)


class MixedReplayBatchSampler(Sampler):
    """Yield exact-size, no-replacement real/model index batches."""

    def __init__(
        self,
        real_size,
        model_size,
        batch_size,
        real_ratio,
        num_batches,
    ):
        self.real_size = int(real_size)
        self.model_size = int(model_size)
        self.batch_size = int(batch_size)
        self.real_count, self.model_count = _resolve_mixed_counts(
            self.real_size,
            self.model_size,
            self.batch_size,
            real_ratio,
        )
        self.num_batches = int(num_batches)
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive.")

    def __iter__(self):
        for _ in range(self.num_batches):
            indices = []
            if self.real_count:
                indices.extend(
                    np.random.choice(
                        self.real_size, self.real_count, replace=False
                    ).tolist()
                )
            if self.model_count:
                indices.extend(
                    (
                        self.real_size
                        + np.random.choice(
                            self.model_size, self.model_count, replace=False
                        )
                    ).tolist()
                )
            yield indices

    def __len__(self):
        return self.num_batches


class RandomReplayBatchSampler(Sampler):
    """Yield a finite number of random replay batches for rollout seeding."""

    def __init__(self, size, batch_size, num_batches=1):
        self.size = int(size)
        self.batch_size = min(int(batch_size), self.size)
        self.num_batches = int(num_batches)
        if self.size <= 0 or self.batch_size <= 0:
            raise ValueError("Replay rollout sampler requires non-empty data.")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive.")

    def __iter__(self):
        for _ in range(self.num_batches):
            yield np.random.choice(self.size, self.batch_size, replace=False).tolist()

    def __len__(self):
        return self.num_batches


def _loader_kwargs(dataset, num_workers=0, pin_memory=None):
    num_workers = int(num_workers)
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")
    if dataset.is_cached and num_workers:
        raise ValueError("CUDA replay datasets require num_workers=0.")
    if pin_memory is None:
        pin_memory = bool(dataset.device.type == "cuda" and not dataset.is_cached)
    if dataset.is_cached:
        pin_memory = False
    return {
        "num_workers": num_workers,
        "pin_memory": bool(pin_memory),
        "persistent_workers": bool(num_workers > 0),
        "collate_fn": _collate_replay_batch,
    }


def build_mixed_replay_loader(
    real_dataset,
    model_dataset,
    batch_size,
    real_ratio,
    num_batches,
    num_workers=0,
    pin_memory=None,
):
    """Build a finite DataLoader for SAC's mixed replay updates."""
    dataset = MixedReplayDataset(real_dataset, model_dataset)
    sampler = MixedReplayBatchSampler(
        len(real_dataset),
        len(model_dataset),
        batch_size,
        real_ratio,
        num_batches,
    )
    kwargs = _loader_kwargs(real_dataset, num_workers, pin_memory)
    if model_dataset.is_cached and num_workers:
        raise ValueError("CUDA replay datasets require num_workers=0.")
    return DataLoader(dataset, batch_sampler=sampler, **kwargs)


def build_replay_loader(
    dataset,
    batch_size,
    num_batches=1,
    num_workers=0,
    pin_memory=None,
):
    """Build a finite DataLoader for random replay batches."""
    sampler = RandomReplayBatchSampler(len(dataset), batch_size, num_batches)
    return DataLoader(
        dataset,
        batch_sampler=sampler,
        **_loader_kwargs(dataset, num_workers, pin_memory),
    )
