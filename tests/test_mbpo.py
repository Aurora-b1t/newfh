import copy
import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np
import torch

from SAC import ReplayBuffer, SAC, SAC_POLICY_ARCHITECTURE
from r_predict_model import StepRewardEnsemble
from r_predict_model.mbpo_adapter import (
    reward_bounds,
    rollout_reward_model,
    sample_mixed_batch,
    train_reward_model_from_replay,
)
from train_mbpo import (
    SAC_CHECKPOINT_FORMAT_VERSION,
    load_sac_inference_checkpoint,
    reward_model_ready,
    save_sac_inference_checkpoint,
    should_train_reward_model,
)


TEST_REWARD_CONFIG = {
    "base_reward": 1.0,
    "ber_penalty": 2.0,
    "hoprate_penalty": 0.0,
}


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
            reward_config=TEST_REWARD_CONFIG,
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
        self.assertTrue(np.all((means >= 0.0) & (means <= 1.0)))
        self.assertTrue(np.all(np.isfinite(variances)))
        self.assertTrue(np.all(variances >= 0.0))
        self.assertTrue(np.all((sampled >= 0.0) & (sampled <= 1.0)))
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
        self.assertEqual(TEST_REWARD_CONFIG, loaded.reward_config)
        self.assertEqual(0.0, loaded.ber_min)
        self.assertEqual(0.5, loaded.ber_max)
        self.assertEqual(1e-4, loaded.logit_epsilon)

    def test_one_elite_is_used_for_each_complete_reward_vector(self):
        model = self.make_model()
        model.elite_model_idxes = [0, 1]
        model.is_fitted = True
        latent_means = np.stack(
            [
                np.full((4, 3), -2.0, dtype=np.float32),
                np.full((4, 3), 2.0, dtype=np.float32),
            ]
        )
        variances = np.zeros_like(latent_means)
        with mock.patch.object(
            model,
            "_predict_latent",
            return_value=(latent_means, variances),
        ):
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
            latent_value = -2.0 if selected_model == 0 else 2.0
            expected = 1.0 / (1.0 + np.exp(-latent_value))
            np.testing.assert_allclose(sampled[index], expected)

    def test_extreme_latent_samples_remain_bounded(self):
        model = self.make_model()
        model.elite_model_idxes = [0]
        model.is_fitted = True
        latent_means = np.array(
            [[[-1000.0, 0.0, 1000.0], [1000.0, -1000.0, 0.0]]],
            dtype=np.float32,
        )
        latent_variances = np.full_like(latent_means, 100.0)
        with mock.patch.object(
            model,
            "_predict_latent",
            return_value=(latent_means, latent_variances),
        ):
            sampled, _diagnostics = model.sample_rewards(
                self.states[:2], self.hoprates[:2], self.actions[:2]
            )
        self.assertTrue(np.all(np.isfinite(sampled)))
        self.assertTrue(np.all((sampled >= 0.0) & (sampled <= 1.0)))

    def test_fit_saturates_only_reward_model_targets(self):
        model = self.make_model()
        rewards = np.tile([-1.0, 0.5, 2.0], (len(self.states), 1)).astype(
            np.float32
        )
        original_rewards = rewards.copy()
        stats = model.fit(
            self.states,
            self.hoprates,
            self.actions,
            rewards,
            batch_size=2,
            max_epochs=1,
            patience=0,
        )
        self.assertAlmostEqual(
            2.0 / 3.0, stats["target_saturation_fraction"], places=6
        )
        np.testing.assert_array_equal(rewards, original_rewards)
        predictions, _variances = model.predict(
            self.states[:2], self.hoprates[:2], self.actions[:2]
        )
        self.assertTrue(np.all((predictions >= 0.0) & (predictions <= 1.0)))

    def test_early_stopping_restores_model_and_optimizer_state(self):
        model = StepRewardEnsemble(
            network_size=1,
            elite_size=1,
            num_heads=3,
            n_actions=4,
            reward_config=TEST_REWARD_CONFIG,
            hidden_size=16,
            device="cpu",
        )
        captured = {}
        evaluation_count = 0

        def fake_evaluate(member, _dataset, _indices, _batch_size):
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count == 1:
                return 10.0
            if evaluation_count == 2:
                captured["model"] = copy.deepcopy(member.state_dict())
                captured["optimizer"] = copy.deepcopy(
                    model.optimizers[0].state_dict()
                )
                return 5.0
            if evaluation_count == 3:
                return 6.0
            return 5.0

        with mock.patch.object(model, "_evaluate_member", side_effect=fake_evaluate):
            model.fit(
                self.states,
                self.hoprates,
                self.actions,
                np.full_like(self.rewards, 0.5),
                batch_size=2,
                holdout_ratio=0.25,
                max_epochs=2,
                patience=1,
                min_improvement=0.0,
            )

        restored_model = model.members[0].state_dict()
        for key, expected in captured["model"].items():
            torch.testing.assert_close(restored_model[key], expected)

        restored_optimizer = model.optimizers[0].state_dict()
        self.assertEqual(
            captured["optimizer"]["param_groups"],
            restored_optimizer["param_groups"],
        )
        for parameter_id, expected_state in captured["optimizer"]["state"].items():
            actual_state = restored_optimizer["state"][parameter_id]
            for key, expected in expected_state.items():
                if torch.is_tensor(expected):
                    torch.testing.assert_close(actual_state[key], expected)
                else:
                    self.assertEqual(actual_state[key], expected)


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

    def test_rollout_uses_bounded_rewards_and_copies_real_successors(self):
        real_buffer = make_buffer(count=2)
        model_buffer = ReplayBuffer(8, num_heads=3, n_actions=4)
        stats = rollout_reward_model(
            FixedRewardModel([0.0, 1.0, 0.5]),
            FixedAgent([0, 1, 2]),
            real_buffer,
            model_buffer,
            batch_size=8,
            reward_config=TEST_REWARD_CONFIG,
        )
        generated = model_buffer.get_all()

        self.assertEqual(2, stats["generated"])
        np.testing.assert_allclose(
            generated["block_rewards"],
            np.tile([0.0, 1.0, 0.5], (2, 1)),
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

    def test_rollout_rejects_out_of_bound_rewards(self):
        with self.assertRaisesRegex(RuntimeError, "outside its bounds"):
            rollout_reward_model(
                FixedRewardModel([-0.1, 1.1, 0.5]),
                FixedAgent([0, 1, 2]),
                make_buffer(count=2),
                ReplayBuffer(8, num_heads=3, n_actions=4),
                batch_size=2,
                reward_config=TEST_REWARD_CONFIG,
            )

    def test_model_replay_accumulates_and_evicts_oldest_entries_fifo(self):
        real_buffer = make_buffer(count=2)
        model_buffer = ReplayBuffer(3, num_heads=3, n_actions=4)
        first_stats = rollout_reward_model(
            FixedRewardModel([0.0, 0.25, 0.5]),
            FixedAgent([0, 1, 2]),
            real_buffer,
            model_buffer,
            batch_size=2,
            reward_config=TEST_REWARD_CONFIG,
        )
        self.assertEqual(2, model_buffer.size())
        self.assertEqual(0, first_stats["model_buffer_size_before"])
        self.assertEqual(2, first_stats["model_buffer_size_after"])
        self.assertEqual(0, first_stats["fifo_evicted"])

        second_stats = rollout_reward_model(
            FixedRewardModel([1.0, 0.75, 0.5]),
            FixedAgent([0, 1, 2]),
            real_buffer,
            model_buffer,
            batch_size=2,
            reward_config=TEST_REWARD_CONFIG,
        )
        generated = model_buffer.get_all()
        self.assertEqual(2, second_stats["model_buffer_size_before"])
        self.assertEqual(3, second_stats["model_buffer_size_after"])
        self.assertEqual(3, second_stats["model_buffer_capacity"])
        self.assertEqual(1, second_stats["fifo_evicted"])
        np.testing.assert_allclose(
            generated["block_rewards"],
            np.asarray(
                [
                    [0.0, 0.25, 0.5],
                    [1.0, 0.75, 0.5],
                    [1.0, 0.75, 0.5],
                ],
                dtype=np.float32,
            ),
        )

    def test_reward_bounds_support_penalty_signs(self):
        lower, upper = reward_bounds(
            [100.0],
            {"base_reward": 1.0, "ber_penalty": -2.0, "hoprate_penalty": 0.01},
        )
        np.testing.assert_allclose(lower, [[0.0]])
        np.testing.assert_allclose(upper, [[1.0]])

    def test_reward_bounds_use_half_ber_and_dynamic_hoprate_penalty(self):
        lower, upper = reward_bounds(
            [100.0, 200.0],
            {"base_reward": 10.0, "ber_penalty": 80.0, "hoprate_penalty": 0.01},
        )
        np.testing.assert_allclose(lower, [[-31.0], [-32.0]])
        np.testing.assert_allclose(upper, [[9.0], [8.0]])

        default_lower, default_upper = reward_bounds(
            [100.0],
            {"base_reward": 10.0, "ber_penalty": 80.0, "hoprate_penalty": 0.0},
        )
        np.testing.assert_allclose(default_lower, [[-30.0]])
        np.testing.assert_allclose(default_upper, [[10.0]])

    def test_reward_bounds_reject_zero_ber_penalty(self):
        with self.assertRaisesRegex(ValueError, "ber_penalty must be non-zero"):
            reward_bounds(
                [100.0],
                {
                    "base_reward": 10.0,
                    "ber_penalty": 0.0,
                    "hoprate_penalty": 0.0,
                },
            )

    def test_full_replay_fields_train_model(self):
        buffer = make_buffer(count=4)
        model = StepRewardEnsemble(
            1,
            1,
            3,
            4,
            reward_config=TEST_REWARD_CONFIG,
            hidden_size=8,
            device="cpu",
        )
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
        agent.actor.train()
        with torch.no_grad():
            before = agent.actor(images, hoprates).numpy()

        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "sac.pt")
            save_sac_inference_checkpoint(
                agent, path, (8, 8), 10.0, 1000.0, {"experiment": "test"}
            )
            payload = torch.load(path, map_location="cpu", weights_only=True)
            self.assertEqual(
                SAC_CHECKPOINT_FORMAT_VERSION, payload["format_version"]
            )
            self.assertEqual(SAC_POLICY_ARCHITECTURE, payload["architecture"])
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

    def test_sac_policy_checkpoint_rejects_legacy_batchnorm_v1(self):
        with tempfile.TemporaryDirectory() as tempdir:
            path = os.path.join(tempdir, "legacy_sac.pt")
            torch.save(
                {
                    "format_version": 1,
                    "model_type": "MultiHeadSACPolicy",
                },
                path,
            )
            with self.assertRaisesRegex(ValueError, "BatchNorm.*retrain"):
                load_sac_inference_checkpoint(path, device="cpu")


if __name__ == "__main__":
    unittest.main()
