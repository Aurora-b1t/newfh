import math
import unittest

import numpy as np
import torch
from torch import nn

from SAC import (
    FUSION_HIDDEN_DIM,
    HOPRATE_FEATURE_DIM,
    PSD_FEATURE_DIM,
    STATE_FEATURE_DIM,
    PolicyNet,
    SAC,
    StateEncoder,
    ValueNet,
    normalize_hoprate,
)


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


def make_transition_batch(batch_size=4, n_actions=5, seed=7):
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(batch_size, 16, 16)).astype(np.float32)
    block_rewards = rng.normal(size=(batch_size, 10)).astype(np.float32)
    return {
        "state_imgs": states,
        "hoprates": np.full(batch_size, 100.0, dtype=np.float32),
        "actions": rng.integers(
            0, n_actions, size=(batch_size, 10), dtype=np.int64
        ),
        "block_rewards": block_rewards,
        "step_rewards": block_rewards.mean(axis=1),
        "next_state_imgs": states + 0.1,
        "next_hoprates": np.full(batch_size, 110.0, dtype=np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
    }


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

    def test_actor_and_critic_use_group_norm_without_batch_norm(self):
        networks = (PolicyNet(5, num_heads=10), ValueNet(5, num_heads=10))
        modules = [module for network in networks for module in network.modules()]

        self.assertFalse(
            any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                for module in modules
            )
        )
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in modules))
        self.assertEqual(6, sum(isinstance(module, nn.GroupNorm) for module in modules))

    def test_state_encoder_has_requested_deep_architecture(self):
        encoder = StateEncoder(10.0, 1000.0)

        self.assertEqual((16, 32, 64), (
            encoder.conv1.out_channels,
            encoder.conv2.out_channels,
            encoder.conv3.out_channels,
        ))
        self.assertEqual((4, 8, 16), (
            encoder.norm1.num_groups,
            encoder.norm2.num_groups,
            encoder.norm3.num_groups,
        ))
        self.assertEqual(PSD_FEATURE_DIM, encoder.conv_fc.out_features)

        hoprate_linears = [
            module
            for module in encoder.hoprate_embedding
            if isinstance(module, nn.Linear)
        ]
        self.assertEqual([(1, 64), (64, HOPRATE_FEATURE_DIM)], [
            (module.in_features, module.out_features)
            for module in hoprate_linears
        ])
        fusion_linears = [
            module for module in encoder.fusion if isinstance(module, nn.Linear)
        ]
        self.assertEqual(
            [
                (PSD_FEATURE_DIM + HOPRATE_FEATURE_DIM, FUSION_HIDDEN_DIM),
                (FUSION_HIDDEN_DIM, STATE_FEATURE_DIM),
            ],
            [(module.in_features, module.out_features) for module in fusion_linears],
        )

        pool_calls = []
        handle = encoder.pool.register_forward_hook(
            lambda _module, _inputs, output: pool_calls.append(tuple(output.shape))
        )
        try:
            output = encoder(torch.randn(2, 1, 16, 16), torch.full((2, 1), 100.0))
        finally:
            handle.remove()
        self.assertEqual((2, STATE_FEATURE_DIM), tuple(output.shape))
        self.assertEqual([(2, 32, 8, 8), (2, 64, 4, 4)], pool_calls)

    def test_state_encoder_supports_test_and_production_psd_sizes(self):
        for size in (8, 16, 100):
            with self.subTest(size=size):
                encoder = StateEncoder(10.0, 1000.0)
                output = encoder(
                    torch.zeros(1, 1, size, size),
                    torch.full((1, 1), 100.0),
                )
                self.assertEqual((1, STATE_FEATURE_DIM), tuple(output.shape))
                if size == 100:
                    self.assertEqual(
                        20_972_000,
                        sum(parameter.numel() for parameter in encoder.parameters()),
                    )

    def test_train_and_eval_outputs_are_identical(self):
        images = torch.randn(3, 1, 16, 16)
        hoprates = torch.full((3, 1), 100.0)

        for network in (PolicyNet(5, num_heads=10), ValueNet(5, num_heads=10)):
            network.train()
            with torch.no_grad():
                training_output = network(images, hoprates)
            network.eval()
            with torch.no_grad():
                inference_output = network(images, hoprates)
            torch.testing.assert_close(
                training_output, inference_output, rtol=0.0, atol=0.0
            )

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

    def test_take_action_does_not_temporarily_flip_actor_mode(self):
        agent = make_agent()
        state = np.random.randn(16, 16).astype(np.float32)
        for requested_mode in (True, False):
            forward_modes = []
            handle = agent.actor.register_forward_pre_hook(
                lambda module, _inputs: forward_modes.append(module.training)
            )
            try:
                agent.actor.train(requested_mode)
                agent.take_action(state, 100.0, deterministic=True)
            finally:
                handle.remove()

            self.assertEqual([requested_mode], forward_modes)
            self.assertEqual(requested_mode, agent.actor.training)


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
        agent = make_agent()
        stats = agent.update(make_transition_batch())

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

    def test_update_keeps_online_network_forwards_in_training_mode(self):
        agent = make_agent()
        observed_modes = {"actor": [], "critic_1": [], "critic_2": []}
        handles = []
        for name in observed_modes:
            network = getattr(agent, name)
            handles.append(
                network.register_forward_pre_hook(
                    lambda module, _inputs, key=name: observed_modes[key].append(
                        module.training
                    )
                )
            )
        try:
            agent.update(make_transition_batch(seed=9))
        finally:
            for handle in handles:
                handle.remove()

        for name, modes in observed_modes.items():
            self.assertTrue(modes, name)
            self.assertTrue(all(modes), f"{name} modes: {modes}")


if __name__ == "__main__":
    unittest.main()
