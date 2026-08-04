"""Step-level probabilistic reward ensemble for FHSS MBPO.

Each ensemble member owns an independent convolutional state encoder.  The
model consumes one complete SAC decision and predicts one reward per offset
head:

    (PSD image, hoprate, offsets[num_heads]) -> block_rewards[num_heads]

Only rewards are modelled.  The MBPO adapter is responsible for pairing these
predictions with the exogenous next observation stored in real replay.
"""

import copy
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from SAC import STATE_FEATURE_DIM, StateEncoder


REWARD_CHECKPOINT_FORMAT_VERSION = 2
REWARD_MODEL_ARCHITECTURE = (
    "cnn3_groupnorm_hop_mlp2_state_action_fusion3_v2"
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


def _init_linear(module):
    if isinstance(module, nn.Linear):
        std = 1.0 / (2.0 * np.sqrt(max(1, module.in_features)))
        nn.init.trunc_normal_(
            module.weight,
            std=std,
            a=-2.0 * std,
            b=2.0 * std,
        )
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class StepRewardMember(nn.Module):
    """One independent CNN member of the probabilistic reward ensemble."""

    def __init__(
        self,
        num_heads,
        n_actions,
        hoprate_min,
        hoprate_max,
        hidden_size=200,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions)
        self.state_encoder = StateEncoder(hoprate_min, hoprate_max)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.num_heads * self.n_actions, hidden_size),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(STATE_FEATURE_DIM + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.latent_mean_head = nn.Linear(hidden_size, self.num_heads)
        self.latent_logvar_head = nn.Linear(hidden_size, self.num_heads)
        self.register_buffer("max_logvar", torch.full((1, self.num_heads), 0.5))
        self.register_buffer("min_logvar", torch.full((1, self.num_heads), -10.0))
        self.action_encoder.apply(_init_linear)
        self.fusion.apply(_init_linear)
        self.latent_mean_head.apply(_init_linear)
        self.latent_logvar_head.apply(_init_linear)

    def forward(self, images, hoprates, actions, return_logvar=False):
        if actions.ndim != 2 or actions.shape[1] != self.num_heads:
            raise ValueError(
                "actions must have shape [batch_size, "
                f"{self.num_heads}], got {tuple(actions.shape)}."
            )
        if torch.any(actions < 0) or torch.any(actions >= self.n_actions):
            raise ValueError("actions are outside the configured action range.")

        state_features = self.state_encoder(images, hoprates)
        one_hot_actions = F.one_hot(
            actions.long(), num_classes=self.n_actions
        ).to(dtype=images.dtype)
        action_features = self.action_encoder(one_hot_actions.flatten(start_dim=1))
        fused = self.fusion(torch.cat([state_features, action_features], dim=1))
        mean = self.latent_mean_head(fused)
        raw_logvar = self.latent_logvar_head(fused)
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if return_logvar:
            return mean, logvar
        return mean, torch.exp(logvar)


class _RewardDataset:
    """Reference-based dataset that avoids copying a full image replay."""

    def __init__(self, state_imgs, hoprates, actions, block_rewards):
        lengths = {
            len(state_imgs),
            len(hoprates),
            len(actions),
            len(block_rewards),
        }
        if len(lengths) != 1:
            raise ValueError("Reward-model fields have inconsistent lengths.")
        self.size = lengths.pop()
        if self.size < 2:
            raise ValueError("Need at least two real transitions to fit the ensemble.")
        self.state_imgs = state_imgs
        self.hoprates = hoprates
        self.actions = actions
        self.block_rewards = block_rewards

    def batch(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return (
            np.stack([self.state_imgs[i] for i in indices]).astype(
                np.float32, copy=False
            ),
            np.asarray([self.hoprates[i] for i in indices], dtype=np.float32),
            np.stack([self.actions[i] for i in indices]).astype(
                np.int64, copy=False
            ),
            np.stack([self.block_rewards[i] for i in indices]).astype(
                np.float32, copy=False
            ),
        )


class StepRewardEnsemble(nn.Module):
    """Independent CNN ensemble for complete FHSS step reward prediction."""

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
        self.members = nn.ModuleList()
        self.optimizers = []
        self._reset_members_and_optimizers()
        self.elite_model_idxes = list(range(self.elite_size))
        self.observation_shape = None
        self.is_fitted = False
        self.last_train_stats = {}

    def _reset_members_and_optimizers(self):
        """Re-initialize every member network and Adam optimizer from scratch."""
        self.members = nn.ModuleList(
            StepRewardMember(
                self.num_heads,
                self.n_actions,
                self.hoprate_min,
                self.hoprate_max,
                self.hidden_size,
            )
            for _ in range(self.network_size)
        )
        self.to(self.device)
        self.optimizers = [
            torch.optim.Adam(
                member.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            for member in self.members
        ]

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
        if torch.any(actions_t < 0) or torch.any(actions_t >= self.n_actions):
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
        return torch.mean(torch.square(mean - targets) * torch.exp(-logvar) + logvar)

    def _evaluate_member(self, member, dataset, indices, batch_size):
        member.eval()
        squared_error = 0.0
        value_count = 0
        with torch.no_grad():
            for start in range(0, len(indices), batch_size):
                batch_indices = indices[start : start + batch_size]
                tensors = self._batch_tensors(dataset.batch(batch_indices))
                latent_mean, _logvar = member(
                    *tensors[:3], return_logvar=True
                )
                reward_prediction = self._latent_to_rewards_tensor(
                    latent_mean, tensors[1]
                )
                bounded_targets = self._bound_rewards_tensor(
                    tensors[3], tensors[1]
                )
                squared_error += torch.sum(
                    torch.square(reward_prediction - bounded_targets)
                ).item()
                value_count += tensors[3].numel()
        return squared_error / max(1, value_count)

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
    ):
        """Retrain every member from scratch on the current real-replay split.

        All member weights and Adam optimizer states are re-initialized at the
        start of every call; training never continues from a previous fit.
        """
        if batch_size <= 0 or patience < 0 or max_epochs <= 0:
            raise ValueError("Invalid ensemble training limits.")
        if not 0.0 < holdout_ratio < 1.0:
            raise ValueError("holdout_ratio must be between zero and one.")
        if min_improvement < 0.0:
            raise ValueError("min_improvement must be non-negative.")

        dataset = _RewardDataset(state_imgs, hoprates, actions, block_rewards)
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

        self._reset_members_and_optimizers()
        permutation = np.random.permutation(dataset.size)
        holdout_size = min(
            max(1, int(dataset.size * holdout_ratio)), dataset.size - 1
        )
        holdout_indices = permutation[:holdout_size]
        train_indices = permutation[holdout_size:]
        epoch_counts = []
        holdout_curves = []

        for member, optimizer in zip(self.members, self.optimizers):
            best_loss = self._evaluate_member(
                member, dataset, holdout_indices, batch_size
            )
            if not np.isfinite(best_loss):
                raise RuntimeError("Reward-model holdout loss became non-finite.")
            member_curve = [float(best_loss)]
            best_state = copy.deepcopy(member.state_dict())
            best_optimizer_state = copy.deepcopy(optimizer.state_dict())
            stale_epochs = 0
            epochs_run = 0

            for epoch in range(max_epochs):
                member.train()
                shuffled_indices = np.random.permutation(train_indices)
                for start in range(0, len(shuffled_indices), batch_size):
                    batch_indices = shuffled_indices[start : start + batch_size]
                    images_t, hoprates_t, actions_t, rewards_t = self._batch_tensors(
                        dataset.batch(batch_indices)
                    )
                    mean, logvar = member(
                        images_t, hoprates_t, actions_t, return_logvar=True
                    )
                    latent_targets, _bounded_rewards = self._targets_to_latent(
                        rewards_t, hoprates_t
                    )
                    loss = self._probabilistic_loss(
                        mean, logvar, latent_targets
                    )
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                holdout_loss = self._evaluate_member(
                    member, dataset, holdout_indices, batch_size
                )
                if not np.isfinite(holdout_loss):
                    raise RuntimeError("Reward-model holdout loss became non-finite.")
                member_curve.append(float(holdout_loss))
                relative_improvement = (
                    (best_loss - holdout_loss) / max(abs(best_loss), 1e-12)
                )
                if relative_improvement > min_improvement:
                    best_loss = holdout_loss
                    best_state = copy.deepcopy(member.state_dict())
                    best_optimizer_state = copy.deepcopy(optimizer.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                epochs_run = epoch + 1
                if stale_epochs >= patience:
                    break

            member.load_state_dict(best_state)
            optimizer.load_state_dict(best_optimizer_state)
            member.eval()
            epoch_counts.append(epochs_run)
            holdout_curves.append(member_curve)

        holdout_losses = np.asarray(
            [
                self._evaluate_member(member, dataset, holdout_indices, batch_size)
                for member in self.members
            ],
            dtype=np.float64,
        )
        self.elite_model_idxes = np.argsort(holdout_losses)[
            : self.elite_size
        ].tolist()
        self.is_fitted = True
        self.last_train_stats = {
            "epochs": epoch_counts,
            "holdout_curves": holdout_curves,
            "holdout_losses": holdout_losses,
            "holdout_loss_mean": float(np.mean(holdout_losses)),
            "elite_model_idxes": list(self.elite_model_idxes),
            "train_size": int(len(train_indices)),
            "holdout_size": int(len(holdout_indices)),
            "target_saturation_fraction": target_saturation_fraction,
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
        ensemble_means = []
        ensemble_variances = []
        for member in self.members:
            member.eval()
            member_means = []
            member_variances = []
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
                    means, variances = member(*tensors)
                    member_means.append(means.cpu().numpy())
                    member_variances.append(variances.cpu().numpy())
            ensemble_means.append(np.concatenate(member_means, axis=0))
            ensemble_variances.append(np.concatenate(member_variances, axis=0))
        return (
            np.stack(ensemble_means).astype(np.float32, copy=False),
            np.stack(ensemble_variances).astype(np.float32, copy=False),
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
        if len(shape) == 2:
            images = torch.zeros((1, 1, *shape), device=self.device)
        elif len(shape) == 3:
            images = torch.zeros((1, *shape), device=self.device)
        else:
            raise ValueError("Checkpoint observation_shape must have two or three axes.")
        hoprates = torch.full(
            (1, 1), (self.hoprate_min + self.hoprate_max) / 2.0, device=self.device
        )
        actions = torch.zeros((1, self.num_heads), dtype=torch.long, device=self.device)
        for member in self.members:
            member.eval()
            with torch.no_grad():
                member(images, hoprates, actions)
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
                "model to create a v2 checkpoint."
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
