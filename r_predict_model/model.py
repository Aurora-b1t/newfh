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

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

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


class _RewardDataset:
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

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return (
            self.state_imgs[indices],
            self.hoprates[indices],
            self.actions[indices],
            self.block_rewards[indices],
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
        self.member = None
        self.optimizer = None
        self.elite_model_idxes = list(range(self.elite_size))
        self.observation_shape = None
        self.is_fitted = False
        self.last_train_stats = {}

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
        """Re-initialize the matrix member and its shared Adam optimizer."""
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

    def _images_tensor(self, images):
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self.device)
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

    @staticmethod
    def _probabilistic_loss(mean, logvar, targets):
        return torch.mean(
            torch.square(mean - targets.unsqueeze(0)) * torch.exp(-logvar) + logvar
        )

    def _evaluate_ensemble(self, dataset, indices, batch_size):
        """Return the per-member holdout MSE with a single parallel pass."""
        self.member.eval()
        squared_error = torch.zeros(
            self.network_size, dtype=torch.float32, device=self.device
        )
        value_count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                tensors = self._dataset_batch_tensors(dataset, batch_indices)
                latent_mean = self.member(
                    *tensors[:3],
                    validate_actions=False,
                    return_mean_only=True,
                )
                reward_prediction = self._latent_to_rewards_tensor(
                    latent_mean, tensors[1]
                )
                bounded_targets = self._bound_rewards_tensor(
                    tensors[3], tensors[1]
                )
                squared_error += torch.sum(
                    torch.square(reward_prediction - bounded_targets.unsqueeze(0)),
                    dim=(1, 2),
                )
                value_count += tensors[3].numel()
        return (squared_error / max(1, value_count)).cpu().numpy()

    def _dataset_batch_tensors(self, dataset, indices):
        if dataset.is_cached:
            return dataset.tensor_batch(indices)
        return self._batch_tensors(dataset.batch(indices))

    def fit(
        self,
        state_imgs,
        hoprates,
        actions,
        block_rewards,
        batch_size=256,
        holdout_ratio=0.2,
        patience=5,
        max_epochs=100,
        min_improvement=0.01,
        cache_dataset_on_device=None,
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

        dataset = _RewardDataset(state_imgs, hoprates, actions, block_rewards)
        dataset.validate(self.num_heads, self.n_actions)
        first_state = np.asarray(state_imgs[0], dtype=np.float32)
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
        target_saturation_fraction = self._target_saturation_fraction(
            hoprates, block_rewards
        )

        if cache_dataset_on_device is None:
            cache_dataset_on_device = (
                self.device.type == "cuda"
                and bool(settings.MBPO_CONFIG.get("cache_dataset_on_device", True))
            )
        self._rebuild_member_and_optimizer(observation_shape)
        if cache_dataset_on_device:
            dataset.cache_on_device(self.device)
        permutation = np.random.permutation(dataset.size)
        holdout_size = min(
            max(1, int(dataset.size * holdout_ratio)), dataset.size - 1
        )
        holdout_indices = permutation[:holdout_size]
        train_indices = permutation[holdout_size:]
        holdout_curves = []
        train_curves = []
        timing_enabled = bool(settings.TIMING_ENABLED)
        epoch_times = []
        fit_start = time.time()

        initial_holdout = self._evaluate_ensemble(
            dataset, holdout_indices, batch_size
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
            shuffled_indices = np.random.permutation(train_indices)
            epoch_loss_total = None
            epoch_batch_count = 0
            t1=time.time()
            for start in range(0, len(shuffled_indices), batch_size):
                batch_indices = shuffled_indices[start : start + batch_size]
                images_t, hoprates_t, actions_t, rewards_t = self._dataset_batch_tensors(
                    dataset, batch_indices
                )
                mean, logvar = self.member(
                    images_t,
                    hoprates_t,
                    actions_t,
                    return_logvar=True,
                    validate_actions=False,
                )
                latent_targets, _bounded_rewards = self._targets_to_latent(
                    rewards_t, hoprates_t
                )
                loss = self._probabilistic_loss(mean, logvar, latent_targets)
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.optimizer.step()
                detached_loss = loss.detach()
                epoch_loss_total = (
                    detached_loss
                    if epoch_loss_total is None
                    else epoch_loss_total + detached_loss
                )
                epoch_batch_count += 1
            t2=time.time()
            print(f"Epoch {epoch+1}/{max_epochs} training time: {t2-t1:.2f} seconds")
            mean_train_loss = float(
                (epoch_loss_total / max(1, epoch_batch_count)).item()
            )
            for member_idx in range(self.network_size):
                train_curves[member_idx].append(mean_train_loss)

            epoch_holdout = self._evaluate_ensemble(
                dataset, holdout_indices, batch_size
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
            dataset, holdout_indices, batch_size
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
        with torch.no_grad():
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
                means, variances = self.member(
                    *tensors,
                    validate_actions=False,
                )
                ensemble_means.append(means.cpu().numpy())
                ensemble_variances.append(variances.cpu().numpy())
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
