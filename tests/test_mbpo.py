import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from SAC import ReplayBuffer, SAC
from r_predict_model import StepRewardEnsemble
from r_predict_model.mbpo_adapter import (
    reward_bounds,
    rollout_reward_model,
    sample_mixed_batch,
    train_reward_model_from_replay,
)
from train_mbpo import (
    load_sac_inference_checkpoint,
    reward_model_ready,
    save_sac_inference_checkpoint,
    should_train_reward_model,
)


def make_buffer(count=4, num_heads=3, n_actions=4, reward_value=0.0):
    buffer = ReplayBuffer(16, num_heads=num_heads, n_actions=n_actions)
    for index in range(count):
        state = np.full((8, 8), index, dtype=np.float32)
        rewards = np.full(num_heads, reward_value, dtype=np.float32)
        buffer.add(
            state,
            100.0 + index * 10.0,
            np.arange(num_heads, dtype=np.int64) % n_actions,
            rewards,
            state + 0.5,
            110.0 + index * 10.0,
            index == count - 1,
        )
    return buffer


class FixedAgent:
    def __init__(self, actions):
        self.actions = np.asarray(actions, dtype=np.int64)

    def take_action(self, state_img, hoprate):
        return self.actions.copy()


class FixedRewardModel:
    def __init__(self, rewards, n_actions=4):
        self.rewards = np.asarray(rewards, dtype=np.float32)
        self.num_heads = len(self.rewards)
        self.n_actions = int(n_actions)

    def sample_rewards(
        self, state_imgs, hoprates, actions, deterministic=False, batch_size=1024
    ):
        count = len(state_imgs)
        return np.tile(self.rewards, (count, 1)), {
            "selected_model_idxes": np.zeros(count, dtype=np.int64),
            "disagreement": np.linspace(0.1, 0.2, count, dtype=np.float32),
        }


class StepRewardEnsembleTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.states = self.rng.normal(size=(8, 8, 8)).astype(np.float32)
        self.hoprates = np.linspace(100.0, 170.0, 8, dtype=np.float32)
        self.actions = self.rng.integers(0, 4, size=(8, 3), dtype=np.int64)
        self.rewards = self.rng.normal(size=(8, 3)).astype(np.float32)

    def make_model(self):
        return StepRewardEnsemble(
            network_size=2,
            elite_size=1,
            num_heads=3,
            n_actions=4,
            hidden_size=16,
            device="cpu",
        )

    def test_members_have_independent_cnn_parameters(self):
        model = self.make_model()
        self.assertIsNot(
            model.members[0].state_encoder,
            model.members[1].state_encoder,
        )
        self.assertIsNot(
            model.members[0].state_encoder.conv1.weight,
            model.members[1].state_encoder.conv1.weight,
        )

    def test_fit_predict_and_sample_dynamic_heads(self):
        model = self.make_model()
        stats = model.fit(
            self.states,
            self.hoprates,
            self.actions,
            self.rewards,
            batch_size=2,
            max_epochs=2,
            patience=1,
        )
        means, variances = model.predict(
            self.states[:2], self.hoprates[:2], self.actions[:2]
        )
        sampled, diagnostics = model.sample_rewards(
            self.states[:2], self.hoprates[:2], self.actions[:2]
        )

        self.assertEqual((2, 2, 3), means.shape)
        self.assertEqual((2, 2, 3), variances.shape)
        self.assertEqual((2, 3), sampled.shape)
        self.assertEqual((2,), diagnostics["disagreement"].shape)
        self.assertTrue(np.all(np.isfinite(means)))
        self.assertTrue(np.all(variances > 0.0))
        self.assertTrue(all(epoch <= 2 for epoch in stats["epochs"]))
        self.assertEqual(1, len(stats["elite_model_idxes"]))

    def test_rejects_out_of_range_actions(self):
        model = self.make_model()
        invalid_actions = self.actions.copy()
        invalid_actions[0, 0] = 4
        with self.assertRaisesRegex(ValueError, "action range"):
            model.fit(
                self.states,
                self.hoprates,
                invalid_actions,
                self.rewards,
                batch_size=2,
                max_epochs=1,
                patience=0,
            )

    def test_reward_model_checkpoint_round_trip(self):
        model = self.make_model()
        model.fit(
            self.states,
            self.hoprates,
            self.actions,
            self.rewards,
            batch_size=2,
            max_epochs=1,
            patience=0,
        )
        before, _ = model.sample_rewards(
            self.states[:2],
            self.hoprates[:2],
            self.actions[:2],
            deterministic=True,
        )
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "reward.pt")
            model.save_checkpoint(path, {"experiment": "test"})
            loaded, metadata = StepRewardEnsemble.load_checkpoint(
                path,
                device="cpu",
                expected_num_heads=3,
                expected_n_actions=4,
                expected_observation_shape=(8, 8),
            )
            after, _ = loaded.sample_rewards(
                self.states[:2],
                self.hoprates[:2],
                self.actions[:2],
                deterministic=True,
            )
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
        self.assertEqual("test", metadata["experiment"])

    def test_one_elite_is_used_for_each_complete_reward_vector(self):
        model = self.make_model()
        model.elite_model_idxes = [0, 1]
        model.is_fitted = True
        means = np.stack(
            [
                np.full((4, 3), 1.0, dtype=np.float32),
                np.full((4, 3), 5.0, dtype=np.float32),
            ]
        )
        variances = np.zeros_like(means)
        with mock.patch.object(model, "predict", return_value=(means, variances)):
            with mock.patch(
                "r_predict_model.model.np.random.normal",
                return_value=np.zeros((4, 3), dtype=np.float32),
            ):
                sampled, diagnostics = model.sample_rewards(
                    self.states[:4], self.hoprates[:4], self.actions[:4]
                )
        for index, selected_model in enumerate(
            diagnostics["selected_model_idxes"]
        ):
            expected = 1.0 if selected_model == 0 else 5.0
            np.testing.assert_allclose(sampled[index], expected)


class AdapterTests(unittest.TestCase):
    def test_mixed_batch_is_complete_and_respects_ratio(self):
        real_buffer = make_buffer(count=4, reward_value=0.0)
        model_buffer = make_buffer(count=4, reward_value=10.0)
        batch = sample_mixed_batch(real_buffer, model_buffer, 6, 0.5)

        self.assertEqual((6, 3), batch["actions"].shape)
        self.assertEqual(3, int(np.sum(batch["step_rewards"] == 0.0)))
        self.assertEqual(3, int(np.sum(batch["step_rewards"] == 10.0)))

    def test_mixed_batch_falls_back_to_real_replay(self):
        real_buffer = make_buffer(count=4)
        model_buffer = ReplayBuffer(4, num_heads=3, n_actions=4)
        batch = sample_mixed_batch(real_buffer, model_buffer, 4, 0.2)
        self.assertEqual((4, 3), batch["block_rewards"].shape)

    def test_rollout_clips_rewards_and_copies_real_successors(self):
        real_buffer = make_buffer(count=2)
        model_buffer = ReplayBuffer(8, num_heads=3, n_actions=4)
        stats = rollout_reward_model(
            FixedRewardModel([-10.0, 10.0, 0.0]),
            FixedAgent([0, 1, 2]),
            real_buffer,
            model_buffer,
            batch_size=8,
            reward_config={
                "base_reward": 1.0,
                "ber_penalty": 8.0,
                "hoprate_penalty": 0.0,
            },
        )
        generated = model_buffer.get_all()

        self.assertEqual(2, stats["generated"])
        self.assertAlmostEqual(2.0 / 3.0, stats["clipped_fraction"], places=6)
        np.testing.assert_allclose(
            generated["block_rewards"],
            np.tile([-7.0, 1.0, 0.0], (2, 1)),
        )
        np.testing.assert_array_equal(
            generated["actions"], np.tile([0, 1, 2], (2, 1))
        )
        for index, state in enumerate(generated["state_imgs"]):
            state_id = int(round(float(state[0, 0])))
            np.testing.assert_allclose(
                generated["next_state_imgs"][index], state + 0.5
            )
            self.assertEqual(state_id == 1, bool(generated["dones"][index]))

    def test_model_replay_retains_fifo_capacity_across_rollouts(self):
        real_buffer = make_buffer(count=2)
        model_buffer = ReplayBuffer(2, num_heads=3, n_actions=4)
        for _ in range(2):
            rollout_reward_model(
                FixedRewardModel([0.0, 0.0, 0.0]),
                FixedAgent([0, 1, 2]),
                real_buffer,
                model_buffer,
                batch_size=2,
                reward_config={
                    "base_reward": 1.0,
                    "ber_penalty": 8.0,
                    "hoprate_penalty": 0.0,
                },
            )
        self.assertEqual(2, model_buffer.size())

    def test_reward_bounds_support_penalty_signs(self):
        lower, upper = reward_bounds(
            [100.0],
            {"base_reward": 1.0, "ber_penalty": -2.0, "hoprate_penalty": 0.01},
        )
        np.testing.assert_allclose(lower, [[0.0]])
        np.testing.assert_allclose(upper, [[2.0]])

    def test_full_replay_fields_train_model(self):
        buffer = make_buffer(count=4)
        model = StepRewardEnsemble(1, 1, 3, 4, hidden_size=8, device="cpu")
        stats = train_reward_model_from_replay(
            model,
            buffer,
            batch_size=2,
            max_epochs=1,
            patience=0,
        )
        self.assertEqual(4, stats["train_size"] + stats["holdout_size"])


class TrainingAndCheckpointTests(unittest.TestCase):
    def test_reward_model_schedule_uses_post_step_count(self):
        buffer = ReplayBuffer(4, num_heads=3, n_actions=4)
        self.assertFalse(reward_model_ready(buffer, 2))
        state = np.zeros((8, 8), dtype=np.float32)
        for _ in range(2):
            buffer.add(state, 100, [0, 1, 2], [0, 0, 0], state, 100, False)
        self.assertTrue(reward_model_ready(buffer, 2))
        self.assertTrue(should_train_reward_model(2, buffer, 2, 1))
        self.assertFalse(should_train_reward_model(3, buffer, 2, 2))

    def test_sac_policy_checkpoint_round_trip(self):
        agent = SAC(
            n_actions=4,
            num_heads=3,
            hoprate_min=10.0,
            hoprate_max=1000.0,
            actor_lr=1e-4,
            critic_lr=1e-4,
            alpha_lr=1e-4,
            target_entropy=math.log(4) * 0.1,
            tau=0.005,
            gamma=0.95,
            device="cpu",
        )
        state = np.random.randn(8, 8).astype(np.float32)
        agent.take_action(state, 100.0, deterministic=True)
        images = torch.from_numpy(state).view(1, 1, 8, 8)
        hoprates = torch.tensor([[100.0]])
        agent.actor.eval()
        with torch.no_grad():
            before = agent.actor(images, hoprates).numpy()

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "sac.pt")
            save_sac_inference_checkpoint(
                agent, path, (8, 8), 10.0, 1000.0, {"experiment": "test"}
            )
            loaded, metadata = load_sac_inference_checkpoint(
                path,
                device="cpu",
                expected_num_heads=3,
                expected_n_actions=4,
                expected_observation_shape=(8, 8),
            )
            with torch.no_grad():
                after = loaded(images, hoprates).numpy()
        np.testing.assert_allclose(before, after, rtol=1e-6, atol=1e-6)
        self.assertEqual("test", metadata["experiment"])


if __name__ == "__main__":
    unittest.main()
