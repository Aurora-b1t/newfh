import math
import unittest

import numpy as np
import torch
from torch import nn

from SAC import PolicyNet, SAC, ValueNet, normalize_hoprate


class FixedActor(nn.Module):
    def __init__(self, num_heads, n_actions):
        super().__init__()
        self.num_heads = num_heads
        self.n_actions = n_actions

    def forward(self, images, hoprates):
        return torch.zeros(
            images.shape[0],
            self.num_heads,
            self.n_actions,
            dtype=images.dtype,
            device=images.device,
        )


class FixedCritic(nn.Module):
    def __init__(self, num_heads, n_actions):
        super().__init__()
        values = torch.arange(num_heads, dtype=torch.float32)
        self.register_buffer(
            "values", values.view(1, num_heads, 1).repeat(1, 1, n_actions)
        )

    def forward(self, images, hoprates):
        return self.values.repeat(images.shape[0], 1, 1)


def make_agent(n_actions=5, gamma=0.95):
    return SAC(
        n_actions=n_actions,
        num_heads=10,
        hoprate_min=10.0,
        hoprate_max=1000.0,
        actor_lr=1e-4,
        critic_lr=1e-4,
        alpha_lr=1e-4,
        target_entropy=math.log(n_actions) * 0.1,
        tau=0.005,
        gamma=gamma,
        device="cpu",
    )


class MultiHeadNetworkTests(unittest.TestCase):
    def test_hoprate_normalization_keeps_requested_scale(self):
        values = torch.tensor([[10.0], [505.0], [1000.0], [2000.0]])
        normalized = normalize_hoprate(values, 10.0, 1000.0)
        np.testing.assert_allclose(
            normalized.numpy().ravel(), [-10.0, 0.0, 10.0, 10.0], atol=1e-6
        )

    def test_actor_and_critic_shapes(self):
        images = torch.randn(3, 1, 16, 16)
        hoprates = torch.full((3, 1), 100.0)
        actor = PolicyNet(5, num_heads=10)
        critic = ValueNet(5, num_heads=10)

        logits = actor(images, hoprates)
        q_values = critic(images, hoprates)

        self.assertEqual((3, 10, 5), tuple(logits.shape))
        self.assertEqual((3, 10, 5), tuple(q_values.shape))
        self.assertIsNot(actor.action_heads[0].weight, actor.action_heads[1].weight)
        self.assertIsNot(critic.q_heads[0].weight, critic.q_heads[1].weight)

    def test_take_action_returns_ten_valid_offsets(self):
        agent = make_agent()
        state = np.random.randn(16, 16).astype(np.float32)

        sampled = agent.take_action(state, 100.0)
        deterministic = agent.take_action(state, 100.0, deterministic=True)

        self.assertEqual((10,), sampled.shape)
        self.assertEqual((10,), deterministic.shape)
        self.assertEqual(np.int64, sampled.dtype)
        self.assertTrue(np.all((sampled >= 0) & (sampled < 5)))
        self.assertTrue(np.all((deterministic >= 0) & (deterministic < 5)))


class MultiHeadSACTests(unittest.TestCase):
    def test_td_target_uses_global_head_mean(self):
        agent = make_agent(n_actions=2, gamma=1.0)
        agent.actor = FixedActor(10, 2)
        agent.target_critic_1 = FixedCritic(10, 2)
        agent.target_critic_2 = FixedCritic(10, 2)
        agent._target_critics_initialized = True

        images = torch.zeros(2, 1, 4, 4)
        hoprates = torch.full((2, 1), 100.0)
        rewards = torch.zeros(2, 10)
        dones = torch.zeros(2, 1)
        targets = agent.calc_target(rewards, images, hoprates, dones)

        expected = 4.5 + 0.01 * math.log(2.0)
        np.testing.assert_allclose(
            targets.numpy(), np.full((2, 10), expected), rtol=1e-6, atol=1e-6
        )

    def test_update_returns_finite_stats(self):
        rng = np.random.default_rng(7)
        agent = make_agent()
        batch_size = 4
        states = rng.normal(size=(batch_size, 16, 16)).astype(np.float32)
        block_rewards = rng.normal(size=(batch_size, 10)).astype(np.float32)
        batch = {
            "state_imgs": states,
            "hoprates": np.full(batch_size, 100.0, dtype=np.float32),
            "actions": rng.integers(0, 5, size=(batch_size, 10), dtype=np.int64),
            "block_rewards": block_rewards,
            "step_rewards": block_rewards.mean(axis=1),
            "next_state_imgs": states + 0.1,
            "next_hoprates": np.full(batch_size, 110.0, dtype=np.float32),
            "dones": np.zeros(batch_size, dtype=np.float32),
        }

        stats = agent.update(batch)

        for key in (
            "critic1_loss",
            "critic2_loss",
            "actor_loss",
            "alpha_loss",
            "alpha",
            "entropy",
        ):
            self.assertTrue(np.isfinite(stats[key]), key)
        self.assertEqual(0, agent.log_alpha.ndim)
        self.assertTrue(agent._target_critics_initialized)


if __name__ == "__main__":
    unittest.main()
