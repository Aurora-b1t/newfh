"""Step-level probabilistic reward ensemble for FHSS MBPO.

The ensemble is implemented as a single "matrix" network whose parameters
carry a leading member dimension (every weight tensor has shape
``[num_members, out, in]`` or ``[num_members, ...]``).  All members therefore
run in one parallel forward/backward pass and are updated together by a single
shared Adam optimizer.  The model consumes one complete SAC decision and
predicts one reward per offset head:

    (PSD image, hoprate, offsets[num_heads]) -> block_rewards[num_heads]

Only rewards are modelled.  The MBPO adapter is responsible for pairing these
predictions with the exogenous next observation stored in real replay.
"""

import copy
import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data._utils.collate import default_collate

import settings
from SAC import (
    HOPRATE_FEATURE_DIM,
    PSD_FEATURE_DIM,
    STATE_FEATURE_DIM,
    normalize_hoprate,
)


REWARD_CHECKPOINT_FORMAT_VERSION = 3
REWARD_MODEL_ARCHITECTURE = (
    "cnn2_groupnorm_hop_mlp1_state_action_fusion1_matrix_v3"
)
DEFAULT_BER_MIN = 0.0
DEFAULT_BER_MAX = 0.5
DEFAULT_LOGIT_EPSILON = 1e-4


def _collate_reward_batch(batch):
    """Collate vectorized reward batches without an extra stack/copy."""
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


class _RewardIndexSampler(Sampler):
    """Sampler for a fixed split that reshuffles only when iterated."""

    def __init__(self, indices, shuffle):
        self.indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        self.shuffle = bool(shuffle)

    def __iter__(self):
        indices = (
            np.random.permutation(self.indices)
            if self.shuffle
            else self.indices
        )
        return iter(indices.tolist())

    def __len__(self):
        return len(self.indices)


def _canonical_reward_config(reward_config):
    required_keys = ("base_reward", "ber_penalty", "hoprate_penalty")
    if reward_config is None or any(key not in reward_config for key in required_keys):
        raise ValueError(
            "reward_config must define base_reward, ber_penalty, and "
            "hoprate_penalty."
        )
    config = {key: float(reward_config[key]) for key in required_keys}
    if not np.all(np.isfinite(list(config.values()))):
        raise ValueError("reward_config values must be finite.")
    if config["ber_penalty"] == 0.0:
        raise ValueError("ber_penalty must be non-zero for a bounded reward model.")
    return config


def reward_bounds_from_config(
    hoprates,
    reward_config,
    ber_min=DEFAULT_BER_MIN,
    ber_max=DEFAULT_BER_MAX,
):
    """Return reward bounds implied by BER endpoints for each hoprate."""
    config = _canonical_reward_config(reward_config)
    ber_min = float(ber_min)
    ber_max = float(ber_max)
    if not np.isfinite(ber_min) or not np.isfinite(ber_max) or ber_max <= ber_min:
        raise ValueError("BER bounds must be finite and satisfy ber_max > ber_min.")
    hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1, 1)
    if not np.all(np.isfinite(hoprates)):
        raise ValueError("Reward bounds require finite hoprates.")

    base = config["base_reward"] - config["hoprate_penalty"] * hoprates
    reward_at_ber_min = base - config["ber_penalty"] * ber_min
    reward_at_ber_max = base - config["ber_penalty"] * ber_max
    return (
        np.minimum(reward_at_ber_min, reward_at_ber_max).astype(
            np.float32, copy=False
        ),
        np.maximum(reward_at_ber_min, reward_at_ber_max).astype(
            np.float32, copy=False
        ),
    )


def _stable_sigmoid(values):
    values = np.asarray(values)
    return np.exp(-np.logaddexp(0.0, -values))


class _MatrixConv2d(nn.Module):
    """Grouped convolution whose weights carry a leading member dimension.

    The input is replicated once per member along the channel axis and the
    per-member kernels are placed into consecutive output-channel groups, so a
    single ``F.conv2d`` call evaluates every member in parallel.
    """

    def __init__(self, num_members, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.num_members = int(num_members)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.in_channels = None
        self.weight = None
        self.bias = None

    def _materialize(self, in_channels, device):
        weight = torch.empty(
            self.num_members,
            self.out_channels,
            in_channels,
            self.kernel_size,
            self.kernel_size,
            device=device,
        )
        nn.init.normal_(weight, 0.0, 0.1)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(
            torch.zeros(self.num_members, self.out_channels, device=device)
        )
        self.in_channels = int(in_channels)

    def forward(self, images):
        if images.ndim == 5:
            batch_size, members, channels, height, width = images.shape
            if members != self.num_members:
                raise ValueError(
                    "Matrix conv input member count does not match the network."
                )
            repeated = images.reshape(
                batch_size, members * channels, height, width
            )
        elif images.ndim == 4:
            batch_size, channels, height, width = images.shape
            repeated = images.repeat_interleave(self.num_members, dim=1)
        else:
            raise ValueError(
                "Matrix conv input must have shape [B,C,H,W] or [B,M,C,H,W]."
            )
        if self.weight is None or self.in_channels != channels:
            self._materialize(channels, images.device)
        weight = self.weight.reshape(
            self.num_members * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        convolved = F.conv2d(
            repeated,
            weight,
            self.bias.reshape(-1),
            stride=self.stride,
            padding=self.padding,
            groups=self.num_members,
        )
        return convolved.reshape(
            batch_size, self.num_members, self.out_channels, height, width
        )


class _MatrixGroupNorm(nn.Module):
    """GroupNorm applied independently to every matrix member."""

    def __init__(self, num_members, num_groups, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(int(num_groups), int(num_channels))

    def forward(self, features):
        batch_size, members, channels, height, width = features.shape
        reshaped = features.reshape(batch_size * members, channels, height, width)
        normalized = self.norm(reshaped)
        return normalized.reshape(batch_size, members, channels, height, width)


class _MatrixLinear(nn.Module):
    """Linear layer whose weight carries a leading member dimension."""

    def __init__(self, num_members, in_features, out_features):
        super().__init__()
        self.num_members = int(num_members)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        std = 1.0 / (2.0 * np.sqrt(max(1, self.in_features)))
        weight = torch.empty(self.num_members, self.out_features, self.in_features)
        nn.init.trunc_normal_(weight, std=std, a=-2.0 * std, b=2.0 * std)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.num_members, self.out_features))

    def forward(self, features):
        return (
            torch.matmul(features, self.weight.transpose(1, 2))
            + self.bias.unsqueeze(1)
        )


class MatrixStepRewardMember(nn.Module):
    """Every ensemble member stored in one network with member-dim weights.

    The state encoder uses a two-layer CNN (16 -> 32 channels with one max
    pool), a single-layer hoprate embedding (1 -> ``HOPRATE_FEATURE_DIM``) and
    a single-layer fusion to ``STATE_FEATURE_DIM``.  After the action encoder,
    one hidden layer feeds the latent mean/log-variance output heads.
    """

    def __init__(
        self,
        num_members,
        num_heads,
        n_actions,
        hoprate_min,
        hoprate_max,
        hidden_size=200,
    ):
        super().__init__()
        self.num_members = int(num_members)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions)
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.hidden_size = int(hidden_size)

        self.conv1 = _MatrixConv2d(self.num_members, 16)
        self.norm1 = _MatrixGroupNorm(self.num_members, 4, 16)
        self.conv2 = _MatrixConv2d(self.num_members, 32)
        self.norm2 = _MatrixGroupNorm(self.num_members, 8, 32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv_fc = None
        self.hoprate_embedding = _MatrixLinear(
            self.num_members, 1, HOPRATE_FEATURE_DIM
        )
        self.fusion = _MatrixLinear(
            self.num_members, PSD_FEATURE_DIM + HOPRATE_FEATURE_DIM, STATE_FEATURE_DIM
        )
        self.action_encoder = _MatrixLinear(
            self.num_members, self.num_heads * self.n_actions, self.hidden_size
        )
        self.fusion_head = _MatrixLinear(
            self.num_members, STATE_FEATURE_DIM + self.hidden_size, self.hidden_size
        )
        self.latent_mean_head = _MatrixLinear(
            self.num_members, self.hidden_size, self.num_heads
        )
        self.latent_logvar_head = _MatrixLinear(
            self.num_members, self.hidden_size, self.num_heads
        )
        self.register_buffer(
            "max_logvar", torch.full((self.num_members, 1, self.num_heads), 0.5)
        )
        self.register_buffer(
            "min_logvar", torch.full((self.num_members, 1, self.num_heads), -10.0)
        )

    def _ensure_conv_fc(self, feature_size, device):
        if self.conv_fc is None or self.conv_fc.in_features != feature_size:
            self.conv_fc = _MatrixLinear(
                self.num_members, feature_size, PSD_FEATURE_DIM
            ).to(device)

    def forward(
        self,
        images,
        hoprates,
        actions,
        return_logvar=False,
        validate_actions=True,
        return_mean_only=False,
    ):
        if actions.ndim != 2 or actions.shape[1] != self.num_heads:
            raise ValueError(
                "actions must have shape [batch_size, "
                f"{self.num_heads}], got {tuple(actions.shape)}."
            )
        if return_mean_only and return_logvar:
            raise ValueError("return_mean_only and return_logvar cannot both be true.")
        if validate_actions and (
            torch.any(actions < 0) or torch.any(actions >= self.n_actions)
        ):
            raise ValueError("actions are outside the configured action range.")

        batch_size = images.shape[0]
        image_features = F.relu(self.norm1(self.conv1(images)))
        image_features = F.relu(self.norm2(self.conv2(image_features)))
        members, channels, height, width = image_features.shape[1:]
        image_features = image_features.reshape(
            batch_size * members, channels, height, width
        )
        image_features = self.pool(image_features)
        image_features = image_features.reshape(batch_size, members, -1)
        self._ensure_conv_fc(image_features.shape[-1], images.device)
        image_features = F.relu(
            self.conv_fc(image_features.permute(1, 0, 2).contiguous())
        )

        normalized_hoprates = normalize_hoprate(
            hoprates, self.hoprate_min, self.hoprate_max
        )
        hoprate_features = F.relu(
            self.hoprate_embedding(
                normalized_hoprates.unsqueeze(0).expand(
                    self.num_members, -1, -1
                )
            )
        )
        state_features = F.relu(
            self.fusion(torch.cat([image_features, hoprate_features], dim=2))
        )

        one_hot_actions = F.one_hot(
            actions.long(), num_classes=self.n_actions
        ).to(dtype=images.dtype)
        action_features = F.relu(
            self.action_encoder(
                one_hot_actions.flatten(start_dim=1)
                .unsqueeze(0)
                .expand(self.num_members, -1, -1)
            )
        )
        fused = F.silu(
            self.fusion_head(torch.cat([state_features, action_features], dim=2))
        )
        mean = self.latent_mean_head(fused)
        if return_mean_only:
            return mean
        raw_logvar = self.latent_logvar_head(fused)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if return_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)


class _RewardDataset(Dataset):
    """Compact array-backed dataset used for repeated model epochs."""

    def __init__(self, state_imgs, hoprates, actions, block_rewards):
        try:
            self.state_imgs = np.asarray(state_imgs, dtype=np.float32)
            self.hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1)
            self.actions = np.asarray(actions)
            self.block_rewards = np.asarray(block_rewards, dtype=np.float32)
        except (TypeError, ValueError) as exc:
            raise ValueError("Reward-model fields must be rectangular arrays.") from exc

        lengths = {
            len(self.state_imgs),
            len(self.hoprates),
            len(self.actions),
            len(self.block_rewards),
        }
        if len(lengths) != 1:
            raise ValueError("Reward-model fields have inconsistent lengths.")
        self.size = lengths.pop()
        if self.size < 2:
            raise ValueError("Need at least two real transitions to fit the ensemble.")
        self._tensor_cache = None
        self._tensor_cache_device = None
        self._latent_target_cache = None
        self._bounded_target_cache = None

    def __len__(self):
        return self.size

    def validate(self, num_heads, n_actions):
        if self.state_imgs.ndim not in (3, 4):
            raise ValueError("state_imgs must have shape [B,H,W] or [B,C,H,W].")
        if self.hoprates.shape != (self.size,):
            raise ValueError(f"hoprates must have shape ({self.size},).")
        if not np.all(np.isfinite(self.hoprates)):
            raise ValueError("hoprates must contain only finite values.")
        if self.actions.shape != (self.size, int(num_heads)):
            raise ValueError(
                "actions must have shape "
                f"({self.size}, {int(num_heads)}), got {self.actions.shape}."
            )
        if not np.all(np.isfinite(self.actions)):
            raise ValueError("actions must contain only finite values.")
        rounded_actions = np.rint(self.actions)
        if not np.allclose(self.actions, rounded_actions):
            raise ValueError("actions must be integer-valued.")
        self.actions = rounded_actions.astype(np.int64, copy=False)
        if np.any(self.actions < 0) or np.any(self.actions >= int(n_actions)):
            raise ValueError("actions are outside the configured action range.")
        expected_rewards_shape = (self.size, int(num_heads))
        if self.block_rewards.shape != expected_rewards_shape:
            raise ValueError(
                "block_rewards must have shape "
                f"{expected_rewards_shape}, got {self.block_rewards.shape}."
            )
        if not np.all(np.isfinite(self.block_rewards)):
            raise ValueError("block_rewards must contain only finite values.")

    def cache_on_device(self, device):
        """Materialize the current real replay once on the training device."""
        images = self.state_imgs
        if images.ndim == 3:
            images = images[:, np.newaxis, ...]
        images = np.ascontiguousarray(images, dtype=np.float32)
        self._tensor_cache = (
            torch.as_tensor(images, dtype=torch.float32, device=device),
            torch.as_tensor(self.hoprates, dtype=torch.float32, device=device).view(-1, 1),
            torch.as_tensor(self.actions, dtype=torch.long, device=device),
            torch.as_tensor(self.block_rewards, dtype=torch.float32, device=device),
        )
        self._tensor_cache_device = torch.device(device)
        self._latent_target_cache = None
        self._bounded_target_cache = None
        # The replay still owns the original per-transition arrays. Release the
        # temporary contiguous image copy after its device transfer.
        self.state_imgs = None

    @property
    def is_cached(self):
        return self._tensor_cache is not None

    def tensor_batch(self, indices):
        if self._tensor_cache is None:
            raise RuntimeError("Reward dataset has not been cached on a device.")
        index_tensor = torch.as_tensor(
            np.asarray(indices, dtype=np.int64),
            dtype=torch.long,
            device=self._tensor_cache_device,
        )
        return tuple(
            values.index_select(0, index_tensor) for values in self._tensor_cache
        )

    def _batch_dict(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self.is_cached:
            tensors = self.tensor_batch(indices)
        else:
            tensors = self.batch(indices)
        batch = dict(
            zip(
                ("state_imgs", "hoprates", "actions", "block_rewards"),
                tensors,
            )
        )
        if self.has_prepared_targets:
            latent_targets, bounded_targets = self.target_batch(indices)
            batch["latent_targets"] = latent_targets
            batch["bounded_targets"] = bounded_targets
        return batch

    def __getitem__(self, index):
        batch = self._batch_dict([int(index)])
        return {key: value[0] for key, value in batch.items()}

    def __getitems__(self, indices):
        return self._batch_dict(indices)

    @property
    def first_state(self):
        if self.state_imgs is None:
            raise RuntimeError("Cached datasets do not retain CPU state images.")
        return self.state_imgs[0]

    def stats_fields(self):
        return self.hoprates, self.block_rewards

    def prepare_targets(self, reward_model):
        """Precompute deterministic reward targets for all cached samples."""
        if not self.is_cached:
            return
        self._latent_target_cache, self._bounded_target_cache = (
            reward_model._targets_to_latent(
                self._tensor_cache[3], self._tensor_cache[1]
            )
        )

    @property
    def has_prepared_targets(self):
        return (
            self._latent_target_cache is not None
            and self._bounded_target_cache is not None
        )

    def target_batch(self, indices):
        if not self.has_prepared_targets:
            raise RuntimeError("Reward targets have not been prepared.")
        index_tensor = torch.as_tensor(
            np.asarray(indices, dtype=np.int64),
            dtype=torch.long,
            device=self._tensor_cache_device,
        )
        return (
            self._latent_target_cache.index_select(0, index_tensor),
            self._bounded_target_cache.index_select(0, index_tensor),
        )

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return (
            self.state_imgs[indices],
            self.hoprates[indices],
            self.actions[indices],
            self.block_rewards[indices],
        )


class RewardReplayDataset(Dataset):
    """Mutation-aware ring cache for the reward fields of a replay buffer.

    The model is still retrained from scratch for every fit.  This class only
    avoids rebuilding and copying unchanged replay observations between fits.
    """

    def __init__(self, device=None, cache_on_device=True):
        self.device = torch.device(device or "cpu")
        self.cache_on_device = bool(cache_on_device and self.device.type == "cuda")
        self.capacity = None
        self.size = 0
        self._start = 0
        self._transition_refs = None
        self._state_shape = None
        self._normalized_state_shape = None
        self._first_state = None
        self._state_storage = None
        self._hoprate_storage = None
        self._action_storage = None
        self._reward_storage = None
        self._latent_target_cache = None
        self._bounded_target_cache = None

    def __len__(self):
        return self.size

    @staticmethod
    def _same_refs(left, right):
        return len(left) == len(right) and all(
            previous is current for previous, current in zip(left, right)
        )

    @staticmethod
    def _state_array(state, expected_shape=None):
        state_array = np.asarray(state, dtype=np.float32)
        if state_array.ndim not in (2, 3):
            raise ValueError("Each state image must have shape [H,W] or [C,H,W].")
        if expected_shape is not None and tuple(state_array.shape) != tuple(expected_shape):
            raise ValueError(
                "State image shape differs from the cached replay dataset: "
                f"expected {expected_shape}, got {state_array.shape}."
            )
        if not np.all(np.isfinite(state_array)):
            raise ValueError("State images must contain only finite values.")
        return state_array

    def _validate_transition(self, transition):
        state = self._state_array(transition[0], self._state_shape)
        hoprate = float(transition[1])
        if not np.isfinite(hoprate):
            raise ValueError("Hoprate must be finite.")
        actions = np.asarray(transition[2])
        if actions.shape != (self._num_heads,):
            raise ValueError(
                f"Actions must have shape ({self._num_heads},), got {actions.shape}."
            )
        if not np.all(np.isfinite(actions)):
            raise ValueError("Actions must contain only finite values.")
        rounded_actions = np.rint(actions)
        if not np.allclose(actions, rounded_actions):
            raise ValueError("Actions must be integer-valued.")
        actions = rounded_actions.astype(np.int64, copy=False)
        if np.any(actions < 0) or np.any(actions >= self._n_actions):
            raise ValueError("Actions are outside the configured action range.")
        rewards = np.asarray(transition[3], dtype=np.float32)
        if rewards.shape != (self._num_heads,):
            raise ValueError(
                "Block rewards must have shape "
                f"({self._num_heads},), got {rewards.shape}."
            )
        if not np.all(np.isfinite(rewards)):
            raise ValueError("Block rewards must contain only finite values.")
        return state, hoprate, actions, rewards

    def _physical_indices_numpy(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= self.size):
            raise IndexError("Reward dataset index is outside the active range.")
        return (indices + self._start) % self.capacity

    def _physical_indices_tensor(self, indices):
        if isinstance(indices, torch.Tensor):
            index_tensor = indices.to(device=self.device, dtype=torch.long)
        else:
            index_tensor = torch.as_tensor(
                np.asarray(indices, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )
        if index_tensor.numel() and (
            torch.any(index_tensor < 0) or torch.any(index_tensor >= self.size)
        ):
            raise IndexError("Reward dataset index is outside the active range.")
        return (index_tensor + self._start) % self.capacity

    def _allocate_storage(self, capacity, state_shape):
        self.capacity = int(capacity)
        self._state_shape = tuple(int(value) for value in state_shape)
        if len(self._state_shape) == 2:
            self._normalized_state_shape = (1, *self._state_shape)
        else:
            self._normalized_state_shape = self._state_shape
        self._num_heads = int(self._num_heads)
        if self.cache_on_device:
            self._state_storage = torch.empty(
                (self.capacity, *self._normalized_state_shape),
                dtype=torch.float32,
                device=self.device,
            )
            self._hoprate_storage = torch.empty(
                self.capacity, 1, dtype=torch.float32, device=self.device
            )
            self._action_storage = torch.empty(
                self.capacity,
                self._num_heads,
                dtype=torch.long,
                device=self.device,
            )
            self._reward_storage = torch.empty(
                self.capacity,
                self._num_heads,
                dtype=torch.float32,
                device=self.device,
            )
        else:
            self._state_storage = np.empty(
                (self.capacity, *self._normalized_state_shape), dtype=np.float32
            )
            self._hoprate_storage = np.empty(self.capacity, dtype=np.float32)
            self._action_storage = np.empty(
                (self.capacity, self._num_heads), dtype=np.int64
            )
            self._reward_storage = np.empty(
                (self.capacity, self._num_heads), dtype=np.float32
            )

    def _write_transition(self, slot, transition):
        state, hoprate, actions, rewards = self._validate_transition(transition)
        normalized_state = state if state.ndim == 3 else state[np.newaxis, ...]
        if self.cache_on_device:
            self._state_storage[slot].copy_(
                torch.as_tensor(normalized_state, dtype=torch.float32, device=self.device)
            )
            self._hoprate_storage[slot, 0] = hoprate
            self._action_storage[slot].copy_(
                torch.as_tensor(actions, dtype=torch.long, device=self.device)
            )
            self._reward_storage[slot].copy_(
                torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            )
        else:
            np.copyto(self._state_storage[slot], normalized_state)
            self._hoprate_storage[slot] = hoprate
            np.copyto(self._action_storage[slot], actions)
            np.copyto(self._reward_storage[slot], rewards)

    def _rebuild(self, transitions, replay_capacity, configured_n_actions=None):
        if not transitions:
            raise ValueError("Cannot cache an empty reward replay.")
        first_state = self._state_array(transitions[0][0])
        self._state_shape = tuple(first_state.shape)
        self._num_heads = int(np.asarray(transitions[0][2]).reshape(-1).size)
        observed_n_actions = max(
            int(np.max(np.asarray(transition[2]))) for transition in transitions
        ) + 1
        self._n_actions = max(
            observed_n_actions,
            int(configured_n_actions or observed_n_actions),
        )
        self._allocate_storage(replay_capacity, self._state_shape)
        self.size = len(transitions)
        self._start = 0
        if self.cache_on_device:
            states = np.stack(
                [self._state_array(transition[0]) for transition in transitions]
            )
            if states.ndim == 3:
                states = states[:, np.newaxis, ...]
            self._state_storage[: self.size].copy_(
                torch.as_tensor(states, dtype=torch.float32, device=self.device)
            )
            hoprates = np.asarray(
                [float(transition[1]) for transition in transitions], dtype=np.float32
            )
            actions = np.asarray(
                [transition[2] for transition in transitions], dtype=np.int64
            )
            rewards = np.asarray(
                [transition[3] for transition in transitions], dtype=np.float32
            )
            self._hoprate_storage[: self.size, 0].copy_(
                torch.as_tensor(hoprates, dtype=torch.float32, device=self.device)
            )
            self._action_storage[: self.size].copy_(
                torch.as_tensor(actions, dtype=torch.long, device=self.device)
            )
            self._reward_storage[: self.size].copy_(
                torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
            )
        else:
            for slot, transition in enumerate(transitions):
                self._write_transition(slot, transition)
        self._first_state = first_state.copy()
        self._transition_refs = tuple(transitions)
        self._latent_target_cache = None
        self._bounded_target_cache = None

    def sync(self, replay_buffer):
        """Synchronize only newly appended or FIFO-overwritten transitions."""
        transitions = tuple(replay_buffer.buffer)
        if not transitions:
            raise ValueError("Cannot cache an empty reward replay.")
        if self._transition_refs is None:
            self._rebuild(
                transitions,
                replay_buffer.capacity,
                replay_buffer.n_actions,
            )
            return

        previous = self._transition_refs
        current = transitions
        if self._same_refs(previous, current):
            return

        if (
            len(current) > len(previous)
            and len(current) <= self.capacity
            and self._same_refs(previous, current[: len(previous)])
        ):
            for transition in current[len(previous) :]:
                slot = (self._start + self.size) % self.capacity
                self._write_transition(slot, transition)
                self.size += 1
            self._transition_refs = current
            self._latent_target_cache = None
            self._bounded_target_cache = None
            return

        if (
            len(current) == len(previous) == self.capacity
            and len(current) > 1
            and self._same_refs(previous[1:], current[:-1])
        ):
            self._write_transition(self._start, current[-1])
            self._start = (self._start + 1) % self.capacity
            self._first_state = self._state_array(current[0][0]).copy()
            self._transition_refs = current
            self._latent_target_cache = None
            self._bounded_target_cache = None
            return

        # A caller may have mutated the buffer in a way that is not the normal
        # one-transition append path. Rebuild only in that exceptional case.
        self._rebuild(current, replay_buffer.capacity, replay_buffer.n_actions)

    @property
    def is_cached(self):
        return self.cache_on_device

    @property
    def first_state(self):
        return self._first_state

    def validate(self, num_heads, n_actions):
        if self.size < 2:
            raise ValueError("Need at least two real transitions to fit the ensemble.")
        if self._state_shape is None or len(self._state_shape) not in (2, 3):
            raise ValueError("state images must have shape [B,H,W] or [B,C,H,W].")
        if self._num_heads != int(num_heads):
            raise ValueError("Cached action head count does not match the ensemble.")
        if self._n_actions > int(n_actions):
            raise ValueError("Cached actions exceed the configured action range.")

    def stats_fields(self):
        indices = (np.arange(self.size, dtype=np.int64) + self._start) % self.capacity
        if self.cache_on_device:
            hoprates = self._hoprate_storage[indices, 0].cpu().numpy()
            rewards = self._reward_storage[indices].cpu().numpy()
        else:
            hoprates = self._hoprate_storage[indices]
            rewards = self._reward_storage[indices]
        return hoprates, rewards

    def prepare_targets(self, reward_model):
        if not self.is_cached:
            return
        indices = torch.arange(self.size, device=self.device, dtype=torch.long)
        physical = self._physical_indices_tensor(indices)
        latent_targets, bounded_rewards = reward_model._targets_to_latent(
            self._reward_storage.index_select(0, physical),
            self._hoprate_storage.index_select(0, physical),
        )
        self._latent_target_cache = latent_targets
        self._bounded_target_cache = bounded_rewards

    @property
    def has_prepared_targets(self):
        return (
            self._latent_target_cache is not None
            and self._bounded_target_cache is not None
        )

    def tensor_batch(self, indices):
        if not self.is_cached:
            raise RuntimeError("CPU reward datasets do not expose tensor batches.")
        physical = self._physical_indices_tensor(indices)
        return (
            self._state_storage.index_select(0, physical),
            self._hoprate_storage.index_select(0, physical),
            self._action_storage.index_select(0, physical),
            self._reward_storage.index_select(0, physical),
        )

    def _batch_dict(self, indices):
        indices = np.asarray(indices, dtype=np.int64).reshape(-1)
        if self.is_cached:
            tensors = self.tensor_batch(indices)
        else:
            tensors = self.batch(indices)
        batch = dict(
            zip(
                ("state_imgs", "hoprates", "actions", "block_rewards"),
                tensors,
            )
        )
        if self.has_prepared_targets:
            latent_targets, bounded_targets = self.target_batch(indices)
            batch["latent_targets"] = latent_targets
            batch["bounded_targets"] = bounded_targets
        return batch

    def __getitem__(self, index):
        batch = self._batch_dict([int(index)])
        return {key: value[0] for key, value in batch.items()}

    def __getitems__(self, indices):
        return self._batch_dict(indices)

    def target_batch(self, indices):
        if not self.has_prepared_targets:
            raise RuntimeError("Reward targets have not been prepared.")
        if isinstance(indices, torch.Tensor):
            logical = indices.to(device=self.device, dtype=torch.long)
        else:
            logical = torch.as_tensor(
                np.asarray(indices, dtype=np.int64),
                dtype=torch.long,
                device=self.device,
            )
        return (
            self._latent_target_cache.index_select(0, logical),
            self._bounded_target_cache.index_select(0, logical),
        )

    def batch(self, indices):
        if self.is_cached:
            raise RuntimeError("GPU reward datasets do not expose CPU batches.")
        physical = self._physical_indices_numpy(indices)
        states = self._state_storage[physical]
        if len(self._state_shape) == 2:
            states = states[:, 0]
        return (
            states,
            self._hoprate_storage[physical],
            self._action_storage[physical],
            self._reward_storage[physical],
        )


class StepRewardEnsemble(nn.Module):
    """Matrix CNN ensemble for complete FHSS step reward prediction."""

    def __init__(
        self,
        network_size,
        elite_size,
        num_heads,
        n_actions,
        reward_config,
        hoprate_min=10.0,
        hoprate_max=1000.0,
        ber_min=DEFAULT_BER_MIN,
        ber_max=DEFAULT_BER_MAX,
        logit_epsilon=DEFAULT_LOGIT_EPSILON,
        hidden_size=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device=None,
        precision="float32",
        compile_model=False,
    ):
        super().__init__()
        if network_size <= 0:
            raise ValueError("network_size must be positive.")
        if elite_size <= 0 or elite_size > network_size:
            raise ValueError("elite_size must be in [1, network_size].")
        if num_heads <= 0 or n_actions <= 0:
            raise ValueError("num_heads and n_actions must be positive.")
        if hidden_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("Invalid reward-model optimizer or hidden-size settings.")
        if hoprate_max <= hoprate_min:
            raise ValueError("hoprate_max must be greater than hoprate_min.")
        if not np.isfinite(ber_min) or not np.isfinite(ber_max) or ber_max <= ber_min:
            raise ValueError("BER bounds must be finite and satisfy ber_max > ber_min.")
        if not 0.0 < float(logit_epsilon) < 0.5:
            raise ValueError("logit_epsilon must be between zero and 0.5.")

        self.network_size = int(network_size)
        self.elite_size = int(elite_size)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions)
        self.reward_config = _canonical_reward_config(reward_config)
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.ber_min = float(ber_min)
        self.ber_max = float(ber_max)
        self.logit_epsilon = float(logit_epsilon)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        precision = str(precision).lower()
        if precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                "precision must be one of 'float32', 'bfloat16', or 'float16'."
            )
        if self.device.type != "cuda" and precision != "float32":
            raise ValueError("Mixed precision reward training requires CUDA.")
        self.precision = precision
        self.compile_model = bool(compile_model)
        if self.compile_model and not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile.")
        self.member = None
        self.optimizer = None
        self._member_observation_shape = None
        object.__setattr__(self, "_compiled_member", None)
        self.elite_model_idxes = list(range(self.elite_size))
        self.observation_shape = None
        self.is_fitted = False
        self.last_train_stats = {}

    def _autocast_context(self):
        if self.device.type != "cuda" or self.precision == "float32":
            return nullcontext()
        dtype = (
            torch.bfloat16 if self.precision == "bfloat16" else torch.float16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _new_grad_scaler(self):
        enabled = self.device.type == "cuda" and self.precision == "float16"
        return torch.amp.GradScaler("cuda", enabled=enabled)

    def _compute_member(self):
        return self._compiled_member or self.member

    def _dummy_tensors(self, observation_shape):
        shape = tuple(int(value) for value in observation_shape)
        if len(shape) == 2:
            images = torch.zeros((1, 1, *shape), device=self.device)
        elif len(shape) == 3:
            images = torch.zeros((1, *shape), device=self.device)
        else:
            raise ValueError("observation_shape must have two or three axes.")
        hoprates = torch.full(
            (1, 1), (self.hoprate_min + self.hoprate_max) / 2.0, device=self.device
        )
        actions = torch.zeros(
            (1, self.num_heads), dtype=torch.long, device=self.device
        )
        return images, hoprates, actions

    def _rebuild_member_and_optimizer(self, observation_shape):
        """Reset parameters/Adam without reallocating a stable module graph."""
        observation_shape = tuple(int(value) for value in observation_shape)
        if (
            self.member is None
            or self._member_observation_shape != observation_shape
        ):
            self.member = MatrixStepRewardMember(
                self.network_size,
                self.num_heads,
                self.n_actions,
                self.hoprate_min,
                self.hoprate_max,
                self.hidden_size,
            ).to(self.device)
            with torch.no_grad():
                self.member(*self._dummy_tensors(observation_shape))
            self.optimizer = torch.optim.Adam(
                self.member.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self._member_observation_shape = observation_shape
            if self.compile_model:
                object.__setattr__(
                    self,
                    "_compiled_member",
                    torch.compile(self.member, fullgraph=False),
                )
            return

        with torch.no_grad():
            for module in self.member.modules():
                if isinstance(module, _MatrixConv2d):
                    nn.init.normal_(module.weight, 0.0, 0.1)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, _MatrixLinear):
                    std = 1.0 / (2.0 * np.sqrt(max(1, module.in_features)))
                    nn.init.trunc_normal_(
                        module.weight,
                        std=std,
                        a=-2.0 * std,
                        b=2.0 * std,
                    )
                    nn.init.zeros_(module.bias)
                elif isinstance(module, _MatrixGroupNorm):
                    module.norm.reset_parameters()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()

    def reward_bounds(self, hoprates):
        return reward_bounds_from_config(
            hoprates,
            self.reward_config,
            ber_min=self.ber_min,
            ber_max=self.ber_max,
        )

    def _reward_bounds_tensor(self, hoprates):
        base = (
            self.reward_config["base_reward"]
            - self.reward_config["hoprate_penalty"] * hoprates
        )
        reward_at_ber_min = (
            base - self.reward_config["ber_penalty"] * self.ber_min
        )
        reward_at_ber_max = (
            base - self.reward_config["ber_penalty"] * self.ber_max
        )
        return (
            torch.minimum(reward_at_ber_min, reward_at_ber_max),
            torch.maximum(reward_at_ber_min, reward_at_ber_max),
        )

    def _bound_rewards_tensor(self, rewards, hoprates):
        lower, upper = self._reward_bounds_tensor(hoprates)
        return torch.minimum(torch.maximum(rewards, lower), upper)

    def _targets_to_latent(self, rewards, hoprates):
        lower, upper = self._reward_bounds_tensor(hoprates)
        bounded_rewards = torch.minimum(torch.maximum(rewards, lower), upper)
        normalized = (bounded_rewards - lower) / (upper - lower)
        normalized = normalized.clamp(
            self.logit_epsilon, 1.0 - self.logit_epsilon
        )
        latent_targets = torch.log(normalized) - torch.log1p(-normalized)
        return latent_targets, bounded_rewards

    def _latent_to_rewards_tensor(self, latent_values, hoprates):
        lower, upper = self._reward_bounds_tensor(hoprates)
        return lower + (upper - lower) * torch.sigmoid(latent_values)

    def _latent_to_rewards_numpy(self, latent_values, hoprates):
        latent_values = np.asarray(latent_values)
        lower, upper = self.reward_bounds(hoprates)
        while lower.ndim < latent_values.ndim:
            lower = lower[np.newaxis, ...]
            upper = upper[np.newaxis, ...]
        return lower + (upper - lower) * _stable_sigmoid(latent_values)

    def _target_saturation_fraction(self, hoprates, block_rewards):
        rewards = np.asarray(block_rewards, dtype=np.float32)
        expected_shape = (len(hoprates), self.num_heads)
        if rewards.shape != expected_shape:
            raise ValueError(
                f"block_rewards must have shape {expected_shape}, got {rewards.shape}."
            )
        lower, upper = self.reward_bounds(hoprates)
        return float(np.mean((rewards < lower) | (rewards > upper)))

    def _to_device_tensor(self, values, dtype):
        if torch.is_tensor(values):
            return values.to(
                device=self.device,
                dtype=dtype,
                non_blocking=True,
            )
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def _images_tensor(self, images):
        tensor = self._to_device_tensor(images, torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(
                "state images must have shape [B,H,W] or [B,C,H,W], got "
                f"{tuple(tensor.shape)}."
            )
        return tensor

    def _batch_tensors(self, batch, include_rewards=True):
        images, hoprates, actions, *remaining = batch
        images_t = self._images_tensor(images)
        hoprates_array = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        if hoprates_array.shape != (len(images),):
            raise ValueError(f"hoprates must have shape ({len(images)},).")
        if not np.all(np.isfinite(hoprates_array)):
            raise ValueError("hoprates must contain only finite values.")
        hoprates_t = torch.as_tensor(
            hoprates_array, dtype=torch.float32, device=self.device
        ).view(-1, 1)
        raw_actions = np.asarray(actions)
        if raw_actions.shape != (len(images), self.num_heads):
            raise ValueError(
                "actions must have shape "
                f"({len(images)}, {self.num_heads}), got {raw_actions.shape}."
            )
        rounded_actions = np.rint(raw_actions)
        if not np.allclose(raw_actions, rounded_actions):
            raise ValueError("actions must be integer-valued.")
        actions_t = torch.as_tensor(
            rounded_actions, dtype=torch.long, device=self.device
        )
        if np.any(rounded_actions < 0) or np.any(rounded_actions >= self.n_actions):
            raise ValueError("actions are outside the configured action range.")
        if not include_rewards:
            return images_t, hoprates_t, actions_t

        rewards = np.asarray(remaining[0], dtype=np.float32)
        expected_shape = (len(images), self.num_heads)
        if rewards.shape != expected_shape:
            raise ValueError(
                f"block_rewards must have shape {expected_shape}, got {rewards.shape}."
            )
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        return images_t, hoprates_t, actions_t, rewards_t

    def _loader_batch_tensors(self, batch):
        """Validate and move one DataLoader batch without NumPy round-trips."""
        if not isinstance(batch, dict):
            raise TypeError("Reward DataLoader batches must be dictionaries.")
        images_t = self._images_tensor(batch["state_imgs"])
        batch_size = images_t.shape[0]
        hoprates_t = self._to_device_tensor(
            batch["hoprates"], torch.float32
        ).view(-1, 1)
        if tuple(hoprates_t.shape) != (batch_size, 1):
            raise ValueError(f"hoprates must have shape ({batch_size},).")
        if not torch.all(torch.isfinite(hoprates_t)):
            raise ValueError("hoprates must contain only finite values.")

        actions_t = self._to_device_tensor(batch["actions"], torch.long)
        expected_shape = (batch_size, self.num_heads)
        if tuple(actions_t.shape) != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {tuple(actions_t.shape)}."
            )
        if torch.any(actions_t < 0) or torch.any(actions_t >= self.n_actions):
            raise ValueError("actions are outside the configured action range.")

        rewards_t = self._to_device_tensor(batch["block_rewards"], torch.float32)
        if tuple(rewards_t.shape) != expected_shape:
            raise ValueError(
                f"block_rewards must have shape {expected_shape}, "
                f"got {tuple(rewards_t.shape)}."
            )
        latent_targets = batch.get("latent_targets")
        bounded_targets = batch.get("bounded_targets")
        if latent_targets is not None:
            latent_targets = self._to_device_tensor(latent_targets, torch.float32)
        if bounded_targets is not None:
            bounded_targets = self._to_device_tensor(bounded_targets, torch.float32)
        return (
            images_t,
            hoprates_t,
            actions_t,
            rewards_t,
            latent_targets,
            bounded_targets,
        )

    @staticmethod
    def _probabilistic_loss(mean, logvar, targets):
        return torch.mean(
            torch.square(mean - targets.unsqueeze(0)) * torch.exp(-logvar) + logvar
        )

    def _make_reward_loader(
        self,
        dataset,
        indices,
        batch_size,
        shuffle,
        num_workers=0,
        pin_memory=None,
    ):
        num_workers = int(num_workers)
        if num_workers < 0:
            raise ValueError("data_loader_workers cannot be negative.")
        if dataset.is_cached and num_workers:
            raise ValueError("CUDA reward datasets require data_loader_workers=0.")
        if pin_memory is None:
            pin_memory = bool(self.device.type == "cuda" and not dataset.is_cached)
        if dataset.is_cached:
            pin_memory = False
        return DataLoader(
            dataset,
            batch_size=int(batch_size),
            sampler=_RewardIndexSampler(indices, shuffle=shuffle),
            drop_last=False,
            num_workers=num_workers,
            pin_memory=bool(pin_memory),
            persistent_workers=bool(num_workers > 0),
            collate_fn=_collate_reward_batch,
        )

    def _evaluate_ensemble(self, dataset, indices, batch_size):
        """Return the per-member holdout MSE with a single parallel pass."""
        loader = (
            indices
            if isinstance(indices, DataLoader)
            else self._make_reward_loader(
                dataset,
                indices,
                batch_size,
                shuffle=False,
            )
        )
        self.member.eval()
        squared_error = torch.zeros(
            self.network_size, dtype=torch.float32, device=self.device
        )
        value_count = 0
        with torch.inference_mode():
            for batch in loader:
                tensors = self._loader_batch_tensors(batch)
                images_t, hoprates_t, actions_t, rewards_t, _, bounded_targets = tensors
                with self._autocast_context():
                    latent_mean = self._compute_member()(
                        images_t,
                        hoprates_t,
                        actions_t,
                        validate_actions=False,
                        return_mean_only=True,
                    )
                reward_prediction = self._latent_to_rewards_tensor(
                    latent_mean.float(), hoprates_t.float()
                )
                bounded_targets = (
                    bounded_targets
                    if bounded_targets is not None
                    else self._bound_rewards_tensor(rewards_t, hoprates_t)
                )
                squared_error += torch.sum(
                    torch.square(reward_prediction - bounded_targets.unsqueeze(0)),
                    dim=(1, 2),
                )
                value_count += rewards_t.numel()
        return (squared_error / max(1, value_count)).cpu().numpy()

    def _dataset_batch_tensors(self, dataset, indices):
        tensors = (
            dataset.tensor_batch(indices)
            if dataset.is_cached
            else self._batch_tensors(dataset.batch(indices))
        )
        if getattr(dataset, "has_prepared_targets", False):
            return (*tensors, *dataset.target_batch(indices))
        return tensors

    def fit(
        self,
        state_imgs=None,
        hoprates=None,
        actions=None,
        block_rewards=None,
        batch_size=256,
        holdout_ratio=0.2,
        patience=5,
        max_epochs=100,
        min_improvement=0.01,
        cache_dataset_on_device=None,
        dataset=None,
        data_loader_workers=0,
        data_loader_pin_memory=None,
    ):
        """Retrain every member in parallel on the current real-replay split.

        All matrix weights and the shared Adam optimizer are re-initialized at
        the start of every call; training never continues from a previous fit.
        Members share one train/holdout split and one data order per epoch and
        are updated by a single optimizer step.  Global early stopping is
        driven by the best member's holdout loss; when it stops improving for
        ``patience`` epochs the whole ensemble stops and every member is
        restored to its state at the global best epoch.
        """
        if batch_size <= 0 or patience < 0 or max_epochs <= 0:
            raise ValueError("Invalid ensemble training limits.")
        if not 0.0 < holdout_ratio < 1.0:
            raise ValueError("holdout_ratio must be between zero and one.")
        if min_improvement < 0.0:
            raise ValueError("min_improvement must be non-negative.")
        if int(data_loader_workers) < 0:
            raise ValueError("data_loader_workers cannot be negative.")

        if dataset is None:
            if any(
                value is None
                for value in (state_imgs, hoprates, actions, block_rewards)
            ):
                raise ValueError(
                    "state_imgs, hoprates, actions, and block_rewards are required "
                    "when dataset is not provided."
                )
            dataset = _RewardDataset(
                state_imgs, hoprates, actions, block_rewards
            )
        dataset.validate(self.num_heads, self.n_actions)
        first_state = np.asarray(dataset.first_state, dtype=np.float32)
        if first_state.ndim not in (2, 3):
            raise ValueError("Each state image must have shape [H,W] or [C,H,W].")
        observation_shape = tuple(first_state.shape)
        if self.observation_shape is None:
            self.observation_shape = observation_shape
        elif observation_shape != self.observation_shape:
            raise ValueError(
                "Observation shape differs from the fitted reward model: "
                f"expected {self.observation_shape}, got {observation_shape}."
            )
        stats_hoprates, stats_rewards = dataset.stats_fields()
        target_saturation_fraction = self._target_saturation_fraction(
            stats_hoprates, stats_rewards
        )

        if cache_dataset_on_device is None:
            cache_dataset_on_device = (
                self.device.type == "cuda"
                and bool(settings.MBPO_CONFIG.get("cache_dataset_on_device", True))
            )
        self._rebuild_member_and_optimizer(observation_shape)
        if dataset is not None and isinstance(dataset, _RewardDataset):
            if cache_dataset_on_device:
                dataset.cache_on_device(self.device)
        dataset.prepare_targets(self)
        permutation = np.random.permutation(dataset.size)
        holdout_size = min(
            max(1, int(dataset.size * holdout_ratio)), dataset.size - 1
        )
        holdout_indices = permutation[:holdout_size]
        train_indices = permutation[holdout_size:]
        train_loader = self._make_reward_loader(
            dataset,
            train_indices,
            batch_size,
            shuffle=True,
            num_workers=data_loader_workers,
            pin_memory=data_loader_pin_memory,
        )
        holdout_loader = self._make_reward_loader(
            dataset,
            holdout_indices,
            batch_size,
            shuffle=False,
            num_workers=data_loader_workers,
            pin_memory=data_loader_pin_memory,
        )
        holdout_curves = []
        train_curves = []
        timing_enabled = bool(settings.TIMING_ENABLED)
        epoch_times = []
        fit_start = time.time()
        grad_scaler = self._new_grad_scaler()

        initial_holdout = self._evaluate_ensemble(
            dataset, holdout_loader, batch_size
        )
        if not np.all(np.isfinite(initial_holdout)):
            raise RuntimeError("Reward-model holdout loss became non-finite.")
        best_global_loss = float(np.min(initial_holdout))
        best_state = copy.deepcopy(self.member.state_dict())
        best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
        stale_epochs = 0
        epochs_run = 0
        for member_idx, value in enumerate(initial_holdout):
            holdout_curves.append([float(value)])
        train_curves = [[] for _ in range(self.network_size)]

        for epoch in range(max_epochs):
            epoch_start = time.time() if timing_enabled else None
            self.member.train()
            epoch_loss_total = None
            epoch_batch_count = 0
            for batch in train_loader:
                (
                    images_t,
                    hoprates_t,
                    actions_t,
                    rewards_t,
                    latent_targets,
                    _bounded_targets,
                ) = self._loader_batch_tensors(batch)
                with self._autocast_context():
                    mean, logvar = self._compute_member()(
                        images_t,
                        hoprates_t,
                        actions_t,
                        return_logvar=True,
                        validate_actions=False,
                    )
                if latent_targets is None:
                    latent_targets = self._targets_to_latent(
                        rewards_t, hoprates_t
                    )[0]
                loss = self._probabilistic_loss(
                    mean.float(), logvar.float(), latent_targets.float()
                )
                self.optimizer.zero_grad(set_to_none=True)
                if grad_scaler.is_enabled():
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                else:
                    loss.backward()
                    self.optimizer.step()
                detached_loss = loss.detach()
                epoch_loss_total = (
                    detached_loss
                    if epoch_loss_total is None
                    else epoch_loss_total + detached_loss
                )
                epoch_batch_count += 1
            mean_train_loss = float(
                (epoch_loss_total / max(1, epoch_batch_count)).item()
            )
            for member_idx in range(self.network_size):
                train_curves[member_idx].append(mean_train_loss)

            epoch_holdout = self._evaluate_ensemble(
                dataset, holdout_loader, batch_size
            )
            if not np.all(np.isfinite(epoch_holdout)):
                raise RuntimeError("Reward-model holdout loss became non-finite.")
            for member_idx, value in enumerate(epoch_holdout):
                holdout_curves[member_idx].append(float(value))

            epoch_best_loss = float(np.min(epoch_holdout))
            relative_improvement = (
                (best_global_loss - epoch_best_loss)
                / max(abs(best_global_loss), 1e-12)
            )
            if relative_improvement > min_improvement:
                best_global_loss = epoch_best_loss
                best_state = copy.deepcopy(self.member.state_dict())
                best_optimizer_state = copy.deepcopy(self.optimizer.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1
            epochs_run = epoch + 1
            if timing_enabled:
                epoch_times.append(time.time() - epoch_start)
            if stale_epochs >= patience:
                break

        self.member.load_state_dict(best_state)
        self.optimizer.load_state_dict(best_optimizer_state)
        self.member.eval()

        holdout_losses = self._evaluate_ensemble(
            dataset, holdout_loader, batch_size
        )
        self.elite_model_idxes = np.argsort(holdout_losses)[
            : self.elite_size
        ].tolist()
        self.is_fitted = True
        self.last_train_stats = {
            "epochs": [epochs_run] * self.network_size,
            "holdout_curves": holdout_curves,
            "train_curves": train_curves,
            "holdout_losses": holdout_losses,
            "holdout_loss_mean": float(np.mean(holdout_losses)),
            "elite_model_idxes": list(self.elite_model_idxes),
            "train_size": int(len(train_indices)),
            "holdout_size": int(len(holdout_indices)),
            "target_saturation_fraction": target_saturation_fraction,
            "epoch_times": epoch_times,
            "fit_time_sec": (time.time() - fit_start) if timing_enabled else None,
        }
        return self.last_train_stats

    def _validate_prediction_inputs(self, state_imgs, hoprates, actions):
        state_imgs = np.asarray(state_imgs, dtype=np.float32)
        hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions)
        if state_imgs.ndim not in (3, 4):
            raise ValueError("state_imgs must have shape [B,H,W] or [B,C,H,W].")
        batch_size = len(state_imgs)
        if batch_size == 0:
            raise ValueError("Prediction inputs cannot be empty.")
        if hoprates.shape != (batch_size,):
            raise ValueError(f"hoprates must have shape ({batch_size},).")
        if actions.shape != (batch_size, self.num_heads):
            raise ValueError(
                "actions must have shape "
                f"({batch_size}, {self.num_heads}), got {actions.shape}."
            )
        if not np.all(np.isfinite(state_imgs)) or not np.all(np.isfinite(hoprates)):
            raise ValueError("Prediction inputs must contain only finite values.")
        return state_imgs, hoprates, actions

    def _predict_latent(self, state_imgs, hoprates, actions, batch_size=1024):
        """Return latent Gaussian parameters with shape [models, batch, heads]."""
        if not self.is_fitted:
            raise RuntimeError("StepRewardEnsemble must be fitted before prediction.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        state_imgs, hoprates, actions = self._validate_prediction_inputs(
            state_imgs, hoprates, actions
        )
        self.member.eval()
        ensemble_means = []
        ensemble_variances = []
        with torch.inference_mode():
            for start in range(0, len(state_imgs), batch_size):
                stop = min(start + batch_size, len(state_imgs))
                tensors = self._batch_tensors(
                    (
                        state_imgs[start:stop],
                        hoprates[start:stop],
                        actions[start:stop],
                    ),
                    include_rewards=False,
                )
                with self._autocast_context():
                    means, variances = self.member(
                        *tensors,
                        validate_actions=False,
                    )
                ensemble_means.append(means.float().cpu().numpy())
                ensemble_variances.append(variances.float().cpu().numpy())
        return (
            np.concatenate(ensemble_means, axis=1).astype(np.float32, copy=False),
            np.concatenate(ensemble_variances, axis=1).astype(np.float32, copy=False),
        )

    def predict(self, state_imgs, hoprates, actions, batch_size=1024):
        """Return bounded reward locations and approximate reward variances."""
        latent_means, latent_variances = self._predict_latent(
            state_imgs, hoprates, actions, batch_size=batch_size
        )
        reward_locations = self._latent_to_rewards_numpy(
            latent_means, hoprates
        )
        lower, upper = self.reward_bounds(hoprates)
        lower = lower[np.newaxis, ...]
        upper = upper[np.newaxis, ...]
        sigmoid_values = _stable_sigmoid(latent_means)
        local_slopes = (upper - lower) * sigmoid_values * (1.0 - sigmoid_values)
        reward_variances = np.square(local_slopes) * latent_variances
        return (
            reward_locations.astype(np.float32, copy=False),
            reward_variances.astype(np.float32, copy=False),
        )

    def sample_rewards(
        self,
        state_imgs,
        hoprates,
        actions,
        deterministic=False,
        batch_size=1024,
    ):
        """Sample a complete reward vector from one elite per transition."""
        latent_means, latent_variances = self._predict_latent(
            state_imgs, hoprates, actions, batch_size=batch_size
        )
        reward_locations = self._latent_to_rewards_numpy(
            latent_means, hoprates
        )
        elite_indices = np.asarray(self.elite_model_idxes, dtype=np.int64)
        elite_reward_locations = reward_locations[elite_indices]
        disagreement = np.mean(
            np.std(elite_reward_locations, axis=0), axis=1
        )
        item_indices = np.arange(latent_means.shape[1])

        if deterministic:
            rewards = np.mean(elite_reward_locations, axis=0)
            selected_indices = np.full(
                latent_means.shape[1], -1, dtype=np.int64
            )
        else:
            selected_indices = np.random.choice(
                elite_indices, size=latent_means.shape[1]
            )
            selected_means = latent_means[selected_indices, item_indices]
            selected_variances = latent_variances[
                selected_indices, item_indices
            ]
            latent_samples = selected_means + np.random.normal(
                size=selected_means.shape
            ) * np.sqrt(np.maximum(selected_variances, 1e-12))
            rewards = self._latent_to_rewards_numpy(latent_samples, hoprates)

        return rewards.astype(np.float32, copy=False), {
            "selected_model_idxes": selected_indices,
            "disagreement": disagreement.astype(np.float32, copy=False),
        }

    def _config(self):
        return {
            "network_size": self.network_size,
            "elite_size": self.elite_size,
            "num_heads": self.num_heads,
            "n_actions": self.n_actions,
            "reward_config": dict(self.reward_config),
            "hoprate_min": self.hoprate_min,
            "hoprate_max": self.hoprate_max,
            "ber_min": self.ber_min,
            "ber_max": self.ber_max,
            "logit_epsilon": self.logit_epsilon,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
        }

    def save_checkpoint(self, path, metadata=None):
        if not self.is_fitted or self.observation_shape is None:
            raise RuntimeError("Cannot save an unfitted reward model.")
        output_dir = os.path.dirname(os.path.abspath(path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "format_version": REWARD_CHECKPOINT_FORMAT_VERSION,
                "model_type": "StepRewardEnsemble",
                "architecture": REWARD_MODEL_ARCHITECTURE,
                "config": self._config(),
                "observation_shape": list(self.observation_shape),
                "elite_model_idxes": list(self.elite_model_idxes),
                "model_state_dict": self.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    def _materialize(self, observation_shape):
        shape = tuple(int(value) for value in observation_shape)
        images, hoprates, actions = self._dummy_tensors(shape)
        if self.member is None:
            self.member = MatrixStepRewardMember(
                self.network_size,
                self.num_heads,
                self.n_actions,
                self.hoprate_min,
                self.hoprate_max,
                self.hidden_size,
            ).to(self.device)
        with torch.no_grad():
            self.member(images, hoprates, actions)
        self.observation_shape = shape

    @classmethod
    def load_checkpoint(
        cls,
        path,
        device="cpu",
        expected_num_heads=None,
        expected_n_actions=None,
        expected_observation_shape=None,
    ):
        payload = torch.load(path, map_location=device, weights_only=True)
        format_version = payload.get("format_version")
        if format_version == 1:
            raise ValueError(
                "Reward-model checkpoint format v1 uses the old shallow state "
                "encoder and two-layer state-action fusion; retrain the reward "
                "model to create a v3 checkpoint."
            )
        if format_version == 2:
            raise ValueError(
                "Reward-model checkpoint format v2 uses the old three-layer CNN, "
                "two-layer hoprate MLP and per-member networks; retrain the "
                "reward model to create a v3 checkpoint."
            )
        if format_version != REWARD_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported reward-model checkpoint format.")
        if payload.get("model_type") != "StepRewardEnsemble":
            raise ValueError("Checkpoint does not contain a StepRewardEnsemble.")
        if payload.get("architecture") != REWARD_MODEL_ARCHITECTURE:
            raise ValueError(
                "Reward-model checkpoint architecture does not match "
                f"{REWARD_MODEL_ARCHITECTURE!r}; retrain with the current architecture."
            )
        config = dict(payload["config"])
        if expected_num_heads is not None and int(expected_num_heads) != int(
            config["num_heads"]
        ):
            raise ValueError("Reward-model checkpoint block count does not match.")
        if expected_n_actions is not None and int(expected_n_actions) != int(
            config["n_actions"]
        ):
            raise ValueError("Reward-model checkpoint action count does not match.")
        observation_shape = tuple(payload["observation_shape"])
        if (
            expected_observation_shape is not None
            and tuple(expected_observation_shape) != observation_shape
        ):
            raise ValueError("Reward-model checkpoint observation shape does not match.")

        model = cls(**config, device=device)
        model._materialize(observation_shape)
        model.load_state_dict(payload["model_state_dict"])
        model.elite_model_idxes = [
            int(index) for index in payload["elite_model_idxes"]
        ]
        model.is_fitted = True
        model.eval()
        return model, dict(payload.get("metadata", {}))
