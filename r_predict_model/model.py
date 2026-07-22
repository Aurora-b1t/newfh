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

from SAC import StateEncoder


CHECKPOINT_FORMAT_VERSION = 1


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
            nn.Linear(256 + hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.output = nn.Linear(hidden_size, 2 * self.num_heads)
        self.register_buffer("max_logvar", torch.full((1, self.num_heads), 0.5))
        self.register_buffer("min_logvar", torch.full((1, self.num_heads), -10.0))
        self.action_encoder.apply(_init_linear)
        self.fusion.apply(_init_linear)
        self.output.apply(_init_linear)

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
        output = self.output(fused)
        mean, raw_logvar = torch.chunk(output, 2, dim=-1)
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
        hoprate_min=10.0,
        hoprate_max=1000.0,
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

        self.network_size = int(network_size)
        self.elite_size = int(elite_size)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions)
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
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
        self.elite_model_idxes = list(range(self.elite_size))
        self.observation_shape = None
        self.is_fitted = False
        self.last_train_stats = {}

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
        hoprates_t = torch.as_tensor(
            hoprates, dtype=torch.float32, device=self.device
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
                mean, _variance = member(*tensors[:3])
                squared_error += torch.sum(torch.square(mean - tensors[3])).item()
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
        """Fit every member on the full current real-replay training split."""
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

        permutation = np.random.permutation(dataset.size)
        holdout_size = min(
            max(1, int(dataset.size * holdout_ratio)), dataset.size - 1
        )
        holdout_indices = permutation[:holdout_size]
        train_indices = permutation[holdout_size:]
        epoch_counts = []

        for member, optimizer in zip(self.members, self.optimizers):
            best_loss = float("inf")
            best_state = None
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
                    loss = self._probabilistic_loss(mean, logvar, rewards_t)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                holdout_loss = self._evaluate_member(
                    member, dataset, holdout_indices, batch_size
                )
                if not np.isfinite(holdout_loss):
                    raise RuntimeError("Reward-model holdout loss became non-finite.")
                relative_improvement = (
                    float("inf")
                    if not np.isfinite(best_loss)
                    else (best_loss - holdout_loss) / max(abs(best_loss), 1e-12)
                )
                if relative_improvement > min_improvement:
                    best_loss = holdout_loss
                    best_state = copy.deepcopy(member.state_dict())
                    stale_epochs = 0
                else:
                    stale_epochs += 1
                epochs_run = epoch + 1
                if stale_epochs >= patience:
                    break

            if best_state is None:
                raise RuntimeError("Reward-model member did not produce a valid checkpoint.")
            member.load_state_dict(best_state)
            member.eval()
            epoch_counts.append(epochs_run)

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
            "holdout_losses": holdout_losses,
            "holdout_loss_mean": float(np.mean(holdout_losses)),
            "elite_model_idxes": list(self.elite_model_idxes),
            "train_size": int(len(train_indices)),
            "holdout_size": int(len(holdout_indices)),
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

    def predict(self, state_imgs, hoprates, actions, batch_size=1024):
        """Return means and variances with shape [models, batch, heads]."""
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

    def sample_rewards(
        self,
        state_imgs,
        hoprates,
        actions,
        deterministic=False,
        batch_size=1024,
    ):
        """Sample a complete reward vector from one elite per transition."""
        means, variances = self.predict(
            state_imgs, hoprates, actions, batch_size=batch_size
        )
        elite_indices = np.asarray(self.elite_model_idxes, dtype=np.int64)
        elite_means = means[elite_indices]
        disagreement = np.mean(np.std(elite_means, axis=0), axis=1)
        item_indices = np.arange(means.shape[1])

        if deterministic:
            rewards = np.mean(elite_means, axis=0)
            selected_indices = np.full(means.shape[1], -1, dtype=np.int64)
        else:
            selected_indices = np.random.choice(
                elite_indices, size=means.shape[1]
            )
            selected_means = means[selected_indices, item_indices]
            selected_variances = variances[selected_indices, item_indices]
            rewards = selected_means + np.random.normal(
                size=selected_means.shape
            ) * np.sqrt(np.maximum(selected_variances, 1e-12))

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
            "hoprate_min": self.hoprate_min,
            "hoprate_max": self.hoprate_max,
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
                "format_version": CHECKPOINT_FORMAT_VERSION,
                "model_type": "StepRewardEnsemble",
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
        if payload.get("format_version") != CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported reward-model checkpoint format.")
        if payload.get("model_type") != "StepRewardEnsemble":
            raise ValueError("Checkpoint does not contain a StepRewardEnsemble.")
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
