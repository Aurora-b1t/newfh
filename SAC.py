"""Factorized multi-head discrete Soft Actor-Critic for FHSS offsets.

One environment transition contains ten categorical offset decisions. The
policy and both critics therefore expose one head per block while sharing an
image/hoprate encoder inside each network.
"""

import collections
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter


HOPRATE_INPUT_SCALE = 10.0
PSD_FEATURE_DIM = 512
HOPRATE_FEATURE_DIM = 64
STATE_FEATURE_DIM = 256
SAC_CHECKPOINT_FORMAT_VERSION = 4
SAC_POLICY_ARCHITECTURE = "cnn2_groupnorm_hop_mlp1_fusion_mlp1_v4"


def normalize_hoprate(hoprate, hoprate_min, hoprate_max):
    """Map a hoprate tensor to the bounded interval [-10, 10]."""
    if hoprate.ndim == 1:
        hoprate = hoprate.unsqueeze(-1)
    if hoprate.shape[-1] != 1:
        raise ValueError(
            f"Expected hoprate features with size 1, got {hoprate.shape[-1]}."
        )
    if hoprate_max <= hoprate_min:
        raise ValueError("hoprate_max must be greater than hoprate_min.")

    normalized = 2.0 * (
        (hoprate - float(hoprate_min)) / float(hoprate_max - hoprate_min)
    ) - 1.0
    return normalized.clamp(-1.0, 1.0) * HOPRATE_INPUT_SCALE


class ReplayBuffer:
    """Replay buffer storing one complete ten-offset environment transition."""

    def __init__(self, capacity, num_heads=10, n_actions=None):
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if n_actions is not None and n_actions <= 0:
            raise ValueError("n_actions must be positive when provided.")

        self.capacity = int(capacity)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions) if n_actions is not None else None
        self.observation_shape = None
        self.buffer = collections.deque(maxlen=self.capacity)

    def _validate_actions(self, actions):
        raw_actions = np.asarray(actions)
        if raw_actions.shape != (self.num_heads,):
            raise ValueError(
                f"actions must have shape ({self.num_heads},), got {raw_actions.shape}."
            )
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("actions contains non-finite values.")
        rounded = np.rint(raw_actions)
        if not np.allclose(raw_actions, rounded):
            raise ValueError("actions must contain integer-valued offsets.")

        actions_array = rounded.astype(np.int64)
        if np.any(actions_array < 0):
            raise ValueError("actions must be non-negative.")
        if self.n_actions is not None and np.any(actions_array >= self.n_actions):
            raise ValueError(
                f"actions must be in [0, {self.n_actions - 1}]."
            )
        return actions_array

    def _validate_block_rewards(self, block_rewards):
        rewards_array = np.asarray(block_rewards, dtype=np.float32)
        if rewards_array.shape != (self.num_heads,):
            raise ValueError(
                "block_rewards must have shape "
                f"({self.num_heads},), got {rewards_array.shape}."
            )
        if not np.all(np.isfinite(rewards_array)):
            raise ValueError("block_rewards contains non-finite values.")
        return rewards_array

    def add(
        self,
        state_img,
        hoprate,
        actions,
        block_rewards,
        next_state_img,
        next_hoprate,
        done,
    ):
        state_array = np.asarray(state_img, dtype=np.float32)
        next_state_array = np.asarray(next_state_img, dtype=np.float32)
        if state_array.shape != next_state_array.shape:
            raise ValueError(
                "state_img and next_state_img must have the same shape, got "
                f"{state_array.shape} and {next_state_array.shape}."
            )
        if state_array.size == 0:
            raise ValueError("state images cannot be empty.")
        if not np.all(np.isfinite(state_array)) or not np.all(
            np.isfinite(next_state_array)
        ):
            raise ValueError("state images must contain only finite values.")
        if self.observation_shape is None:
            self.observation_shape = state_array.shape
        elif state_array.shape != self.observation_shape:
            raise ValueError(
                "state image shape differs from existing replay data: "
                f"expected {self.observation_shape}, got {state_array.shape}."
            )

        hoprate = float(hoprate)
        next_hoprate = float(next_hoprate)
        if not np.isfinite(hoprate) or not np.isfinite(next_hoprate):
            raise ValueError("hoprates must be finite.")

        actions_array = self._validate_actions(actions)
        rewards_array = self._validate_block_rewards(block_rewards)
        step_reward = float(np.mean(rewards_array))

        self.buffer.append(
            (
                state_array.copy(),
                hoprate,
                actions_array.copy(),
                rewards_array.copy(),
                step_reward,
                next_state_array.copy(),
                next_hoprate,
                bool(done),
            )
        )

    def add_batch(
        self,
        state_imgs,
        hoprates,
        actions,
        block_rewards,
        next_state_imgs,
        next_hoprates,
        dones,
    ):
        """Append a batch of transitions with validation performed once."""
        state_array = np.asarray(state_imgs, dtype=np.float32)
        next_state_array = np.asarray(next_state_imgs, dtype=np.float32)
        if state_array.ndim not in (3, 4) or next_state_array.ndim != state_array.ndim:
            raise ValueError(
                "state images must have shape [B,H,W] or [B,C,H,W]."
            )
        if state_array.shape != next_state_array.shape:
            raise ValueError(
                "state_imgs and next_state_imgs must have the same shape, got "
                f"{state_array.shape} and {next_state_array.shape}."
            )
        batch_size = state_array.shape[0]
        if batch_size == 0:
            return
        if state_array.size == 0 or not np.all(np.isfinite(state_array)) or not np.all(
            np.isfinite(next_state_array)
        ):
            raise ValueError("state images must contain only finite values.")

        if self.observation_shape is None:
            self.observation_shape = state_array.shape[1:]
        elif tuple(state_array.shape[1:]) != tuple(self.observation_shape):
            raise ValueError(
                "state image shape differs from existing replay data: "
                f"expected {self.observation_shape}, got {state_array.shape[1:]}."
            )

        hoprate_array = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        next_hoprate_array = np.asarray(next_hoprates, dtype=np.float32).reshape(-1)
        if hoprate_array.shape != (batch_size,) or next_hoprate_array.shape != (
            batch_size,
        ):
            raise ValueError("hoprates must contain one value per transition.")
        if not np.all(np.isfinite(hoprate_array)) or not np.all(
            np.isfinite(next_hoprate_array)
        ):
            raise ValueError("hoprates must be finite.")

        raw_actions = np.asarray(actions)
        if raw_actions.shape != (batch_size, self.num_heads):
            raise ValueError(
                "actions must have shape "
                f"({batch_size}, {self.num_heads}), got {raw_actions.shape}."
            )
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("actions contains non-finite values.")
        rounded_actions = np.rint(raw_actions)
        if not np.allclose(raw_actions, rounded_actions):
            raise ValueError("actions must contain integer-valued offsets.")
        actions_array = rounded_actions.astype(np.int64)
        if np.any(actions_array < 0):
            raise ValueError("actions must be non-negative.")
        if self.n_actions is not None and np.any(actions_array >= self.n_actions):
            raise ValueError(
                f"actions must be in [0, {self.n_actions - 1}]."
            )

        rewards_array = np.asarray(block_rewards, dtype=np.float32)
        if rewards_array.shape != (batch_size, self.num_heads):
            raise ValueError(
                "block_rewards must have shape "
                f"({batch_size}, {self.num_heads}), got {rewards_array.shape}."
            )
        if not np.all(np.isfinite(rewards_array)):
            raise ValueError("block_rewards contains non-finite values.")

        dones_array = np.asarray(dones).reshape(-1)
        if dones_array.shape != (batch_size,):
            raise ValueError("dones must contain one value per transition.")

        step_rewards = np.mean(rewards_array, axis=1)
        for index in range(batch_size):
            self.buffer.append(
                (
                    state_array[index].copy(),
                    float(hoprate_array[index]),
                    actions_array[index].copy(),
                    rewards_array[index].copy(),
                    float(step_rewards[index]),
                    next_state_array[index].copy(),
                    float(next_hoprate_array[index]),
                    bool(dones_array[index]),
                )
            )

    @staticmethod
    def _batch_from_transitions(transitions):
        (
            state_imgs,
            hoprates,
            actions,
            block_rewards,
            step_rewards,
            next_state_imgs,
            next_hoprates,
            dones,
        ) = zip(*transitions)
        return {
            "state_imgs": np.asarray(state_imgs, dtype=np.float32),
            "hoprates": np.asarray(hoprates, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "block_rewards": np.asarray(block_rewards, dtype=np.float32),
            "step_rewards": np.asarray(step_rewards, dtype=np.float32),
            "next_state_imgs": np.asarray(next_state_imgs, dtype=np.float32),
            "next_hoprates": np.asarray(next_hoprates, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def sample(self, batch_size):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > self.size():
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {self.size()}."
            )
        return self._batch_from_transitions(random.sample(self.buffer, batch_size))

    def get_all(self):
        if not self.buffer:
            raise ValueError("Cannot read transitions from an empty replay buffer.")
        return self._batch_from_transitions(list(self.buffer))

    def clear(self):
        """Remove all transitions and reset shape inference for future samples."""
        self.buffer.clear()
        self.observation_shape = None

    def size(self):
        return len(self.buffer)


def _init_weights(module):
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)):
        if isinstance(getattr(module, "weight", None), UninitializedParameter):
            return
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight, 0, 0.1)
        if (
            getattr(module, "bias", None) is not None
            and not isinstance(module.bias, UninitializedParameter)
        ):
            nn.init.zeros_(module.bias)


class StateEncoder(nn.Module):
    """Encode a PSD image and scalar hoprate into one shared feature vector."""

    def __init__(self, hoprate_min, hoprate_max):
        super().__init__()
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)

        self.conv1 = nn.LazyConv2d(
            out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(8, 32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.conv_fc = nn.LazyLinear(PSD_FEATURE_DIM)
        self.hoprate_embedding = nn.Sequential(
            nn.Linear(1, HOPRATE_FEATURE_DIM),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(PSD_FEATURE_DIM + HOPRATE_FEATURE_DIM, STATE_FEATURE_DIM),
            nn.ReLU(),
        )
        self.apply(_init_weights)

    def forward(self, img, hoprate):
        hoprate = normalize_hoprate(
            hoprate, self.hoprate_min, self.hoprate_max
        )
        image_features = F.relu(self.norm1(self.conv1(img)))
        image_features = F.relu(self.norm2(self.conv2(image_features)))
        image_features = self.pool(image_features)
        image_features = self.flatten(image_features)
        image_features = F.relu(self.conv_fc(image_features))
        hoprate_features = self.hoprate_embedding(hoprate)
        return self.fusion(torch.cat([image_features, hoprate_features], dim=1))


class PolicyNet(nn.Module):
    def __init__(self, n_actions, num_heads=10, hoprate_min=10.0, hoprate_max=1000.0):
        super().__init__()
        self.n_actions = int(n_actions)
        self.num_heads = int(num_heads)
        self.encoder = StateEncoder(hoprate_min, hoprate_max)
        self.action_heads = nn.ModuleList(
            nn.Linear(STATE_FEATURE_DIM, self.n_actions)
            for _ in range(self.num_heads)
        )
        for head in self.action_heads:
            nn.init.uniform_(head.weight, -0.003, 0.003)
            nn.init.zeros_(head.bias)

    def forward(self, img, hoprate):
        features = self.encoder(img, hoprate)
        return torch.stack([head(features) for head in self.action_heads], dim=1)


class ValueNet(nn.Module):
    def __init__(self, n_actions, num_heads=10, hoprate_min=10.0, hoprate_max=1000.0):
        super().__init__()
        self.n_actions = int(n_actions)
        self.num_heads = int(num_heads)
        self.encoder = StateEncoder(hoprate_min, hoprate_max)
        self.q_heads = nn.ModuleList(
            nn.Linear(STATE_FEATURE_DIM, self.n_actions)
            for _ in range(self.num_heads)
        )
        self.q_heads.apply(_init_weights)

    def forward(self, img, hoprate):
        features = self.encoder(img, hoprate)
        return torch.stack([head(features) for head in self.q_heads], dim=1)


def _checkpoint_images(observation_shape, device):
    shape = tuple(int(value) for value in observation_shape)
    if len(shape) == 2:
        return torch.zeros((1, 1, *shape), dtype=torch.float32, device=device)
    if len(shape) == 3:
        return torch.zeros((1, *shape), dtype=torch.float32, device=device)
    raise ValueError("observation_shape must have two or three axes.")


def save_sac_inference_checkpoint(
    agent,
    path,
    observation_shape,
    hoprate_min,
    hoprate_max,
    metadata=None,
):
    """Save the policy-only portion of SAC for deterministic inference."""
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    midpoint_hoprate = (float(hoprate_min) + float(hoprate_max)) / 2.0
    with torch.no_grad():
        agent.actor(
            _checkpoint_images(observation_shape, agent.device),
            torch.full(
                (1, 1),
                midpoint_hoprate,
                dtype=torch.float32,
                device=agent.device,
            ),
        )

    torch.save(
        {
            "format_version": SAC_CHECKPOINT_FORMAT_VERSION,
            "model_type": "MultiHeadSACPolicy",
            "architecture": SAC_POLICY_ARCHITECTURE,
            "config": {
                "n_actions": agent.n_actions,
                "num_heads": agent.num_heads,
                "hoprate_min": float(hoprate_min),
                "hoprate_max": float(hoprate_max),
            },
            "observation_shape": list(observation_shape),
            "actor_state_dict": agent.actor.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_sac_inference_checkpoint(
    path,
    device="cpu",
    expected_num_heads=None,
    expected_n_actions=None,
    expected_observation_shape=None,
):
    """Load a policy checkpoint and validate its architecture and dimensions."""
    payload = torch.load(path, map_location=device, weights_only=True)
    format_version = payload.get("format_version")
    if format_version == 1:
        raise ValueError(
            "SAC inference checkpoint format v1 uses the old BatchNorm policy "
            "architecture and cannot be loaded safely; retrain SAC to create a "
            "v4 checkpoint."
        )
    if format_version == 2:
        raise ValueError(
            "SAC inference checkpoint format v2 uses the old shallow GroupNorm "
            "policy architecture; retrain SAC to create a v4 checkpoint."
        )
    if format_version == 3:
        raise ValueError(
            "SAC inference checkpoint format v3 uses the old three-layer CNN "
            "policy architecture; retrain SAC to create a v4 checkpoint."
        )
    if format_version != SAC_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported SAC inference checkpoint format.")
    if payload.get("model_type") != "MultiHeadSACPolicy":
        raise ValueError("Checkpoint does not contain a multi-head SAC policy.")
    if payload.get("architecture") != SAC_POLICY_ARCHITECTURE:
        raise ValueError(
            "SAC checkpoint policy architecture does not match "
            f"{SAC_POLICY_ARCHITECTURE!r}; retrain SAC with the current architecture."
        )

    config = dict(payload["config"])
    observation_shape = tuple(payload["observation_shape"])
    if expected_num_heads is not None and int(expected_num_heads) != int(
        config["num_heads"]
    ):
        raise ValueError("SAC checkpoint block count does not match.")
    if expected_n_actions is not None and int(expected_n_actions) != int(
        config["n_actions"]
    ):
        raise ValueError("SAC checkpoint action count does not match.")
    if (
        expected_observation_shape is not None
        and tuple(expected_observation_shape) != observation_shape
    ):
        raise ValueError("SAC checkpoint observation shape does not match.")

    policy = PolicyNet(**config).to(device)
    policy.eval()
    with torch.no_grad():
        policy(
            _checkpoint_images(observation_shape, device),
            torch.full(
                (1, 1),
                (config["hoprate_min"] + config["hoprate_max"]) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
    policy.load_state_dict(payload["actor_state_dict"])
    policy.eval()
    return policy, dict(payload.get("metadata", {}))


class SAC:
    def __init__(
        self,
        n_actions,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
        num_heads=10,
        hoprate_min=10.0,
        hoprate_max=1000.0,
    ):
        self.n_actions = int(n_actions)
        self.num_heads = int(num_heads)
        self.device = torch.device(device)

        network_args = {
            "n_actions": self.n_actions,
            "num_heads": self.num_heads,
            "hoprate_min": hoprate_min,
            "hoprate_max": hoprate_max,
        }
        self.critic_1 = ValueNet(**network_args).to(self.device)
        self.critic_2 = ValueNet(**network_args).to(self.device)
        self.actor = PolicyNet(**network_args).to(self.device)
        self.target_critic_1 = ValueNet(**network_args).to(self.device)
        self.target_critic_2 = ValueNet(**network_args).to(self.device)
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self._target_critics_initialized = False

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        self.log_alpha = torch.tensor(
            np.log(0.01),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr
        )

        self.target_entropy = float(target_entropy)
        self.gamma = float(gamma)
        self.tau = float(tau)

    @staticmethod
    def _single_image_tensor(state_img, device):
        img = torch.as_tensor(state_img, dtype=torch.float32, device=device)
        if img.ndim == 2:
            return img.unsqueeze(0).unsqueeze(0)
        if img.ndim == 3:
            return img.unsqueeze(0)
        raise ValueError(
            f"state_img must have shape [H,W] or [C,H,W], got {tuple(img.shape)}."
        )

    def take_action(self, state_img, hoprate, deterministic=False):
        state_array = np.asarray(state_img)
        if state_array.ndim not in (2, 3):
            raise ValueError(
                f"state_img must have shape [H,W] or [C,H,W], got {state_array.shape}."
            )
        return self.take_actions(
            state_array[np.newaxis, ...],
            np.asarray([float(hoprate)], dtype=np.float32),
            deterministic=deterministic,
        )[0]

    def take_actions(self, state_imgs, hoprates, deterministic=False):
        """Sample a complete batch of multi-head actions in one policy pass."""
        imgs = self._batch_images(state_imgs)
        hoprate_tensor = self._batch_hoprates(hoprates)
        if imgs.shape[0] != hoprate_tensor.shape[0]:
            raise ValueError(
                "state_imgs and hoprates must contain the same number of samples."
            )

        # LazyConv2d may materialize parameters on this first call; no_grad
        # keeps those parameters usable by the later autograd training pass.
        with torch.no_grad():
            logits = self.actor(imgs, hoprate_tensor)
            if deterministic:
                actions = logits.argmax(dim=-1)
            else:
                actions = torch.distributions.Categorical(logits=logits).sample()
        return actions.cpu().numpy().astype(np.int64, copy=False)

    def _batch_images(self, images):
        tensor = torch.as_tensor(images, dtype=torch.float32, device=self.device)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(
                "Replay images must have shape [B,H,W] or [B,C,H,W], got "
                f"{tuple(tensor.shape)}."
            )
        return tensor

    def _batch_hoprates(self, hoprates):
        return torch.as_tensor(
            hoprates, dtype=torch.float32, device=self.device
        ).view(-1, 1)

    def _ensure_target_critics_initialized(self, imgs, hoprates):
        if self._target_critics_initialized:
            return

        with torch.no_grad():
            self.critic_1(imgs[:1], hoprates[:1])
            self.critic_2(imgs[:1], hoprates[:1])

        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self._target_critics_initialized = True

    def calc_target(self, block_rewards, next_imgs, next_hoprates, dones):
        self._ensure_target_critics_initialized(next_imgs, next_hoprates)
        with torch.no_grad():
            next_logits = self.actor(next_imgs, next_hoprates)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            q1 = self.target_critic_1(next_imgs, next_hoprates)
            q2 = self.target_critic_2(next_imgs, next_hoprates)
            min_q = torch.minimum(q1, q2)
            head_values = torch.sum(
                next_probs * (min_q - self.log_alpha.exp() * next_log_probs),
                dim=-1,
            )
            global_next_value = head_values.mean(dim=1, keepdim=True)
        return block_rewards + self.gamma * global_next_value * (1.0 - dones)

    def soft_update(self, net, target_net):
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        for target_buffer, buffer in zip(target_net.buffers(), net.buffers()):
            target_buffer.data.copy_(buffer.data)

    def update(self, transition_dict, return_stats=True):
        imgs = self._batch_images(transition_dict["state_imgs"])
        next_imgs = self._batch_images(transition_dict["next_state_imgs"])
        hoprates = self._batch_hoprates(transition_dict["hoprates"])
        next_hoprates = self._batch_hoprates(
            transition_dict["next_hoprates"]
        )

        actions_array = np.asarray(transition_dict["actions"])
        expected_shape = (imgs.shape[0], self.num_heads)
        if tuple(actions_array.shape) != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {actions_array.shape}."
            )
        if np.any(actions_array < 0) or np.any(actions_array >= self.n_actions):
            raise ValueError("Replay actions are outside the configured action range.")
        actions = torch.as_tensor(
            actions_array, dtype=torch.long, device=self.device
        )
        block_rewards = torch.as_tensor(
            transition_dict["block_rewards"],
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(
            transition_dict["dones"], dtype=torch.float32, device=self.device
        ).view(-1, 1)

        if tuple(block_rewards.shape) != expected_shape:
            raise ValueError(
                "block_rewards must have shape "
                f"{expected_shape}, got {tuple(block_rewards.shape)}."
            )

        td_target = self.calc_target(
            block_rewards, next_imgs, next_hoprates, dones
        )
        q1_pred = self.critic_1(imgs, hoprates).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        q2_pred = self.critic_2(imgs, hoprates).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        critic_1_loss = F.mse_loss(q1_pred, td_target)
        critic_2_loss = F.mse_loss(q2_pred, td_target)

        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        logits = self.actor(imgs, hoprates)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        entropy = -torch.sum(probs * log_probs, dim=-1)

        with torch.no_grad():
            min_q = torch.minimum(
                self.critic_1(imgs, hoprates),
                self.critic_2(imgs, hoprates),
            )
        actor_loss = torch.sum(
            probs * (self.log_alpha.exp().detach() * log_probs - min_q),
            dim=-1,
        ).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (
            self.log_alpha * (entropy.detach() - self.target_entropy)
        ).mean()
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        if not return_stats:
            return {}

        return {
            "critic1_loss": float(critic_1_loss.item()),
            "critic2_loss": float(critic_2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "entropy": float(entropy.mean().item()),
        }
