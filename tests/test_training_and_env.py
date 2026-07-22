import os
import sys
import unittest
from unittest import mock

import numpy as np
from gymnasium import spaces

import settings
from fh_env import FHSSQPSKEnv, compute_block_rewards
from SAC import ReplayBuffer
from train_mbpo import _validate_args, parse_args as parse_mbpo_args
from train_offsets import parse_optional_replay_path, replay_ready


class TrainingHelperTests(unittest.TestCase):
    def test_online_only_path_and_warmup(self):
        self.assertIsNone(parse_optional_replay_path("none"))
        self.assertIsNone(parse_optional_replay_path(" NULL "))
        self.assertEqual("data.npz", parse_optional_replay_path("data.npz"))

        buffer = ReplayBuffer(4, num_heads=10, n_actions=20)
        self.assertFalse(replay_ready(buffer, 1))
        state = np.zeros((4, 4), dtype=np.float32)
        buffer.add(state, 100, np.zeros(10), np.zeros(10), state, 100, False)
        self.assertTrue(replay_ready(buffer, 1))

    def test_mbpo_entry_point_accepts_online_only_mode(self):
        with mock.patch.object(
            sys,
            "argv",
            ["train_mbpo.py", "--offline_replay_path", "none"],
        ):
            args = parse_mbpo_args()
        self.assertIsNone(args.offline_replay_path)
        _validate_args(args)


class EnvironmentInterfaceTests(unittest.TestCase):
    def test_reward_formula(self):
        ber = np.linspace(0.0, 0.9, 10)
        rewards = compute_block_rewards(ber, 100.0)
        expected = (
            settings.REWARD_CONFIG["base_reward"]
            - settings.REWARD_CONFIG["ber_penalty"] * ber
            - settings.REWARD_CONFIG["hoprate_penalty"] * 100.0
        )
        np.testing.assert_allclose(rewards, expected, rtol=1e-6, atol=1e-6)

    def test_offset_space_is_multidiscrete(self):
        env = FHSSQPSKEnv(
            use_pregen=False,
            enable_sweep=False,
            enable_reactive=False,
            enable_rayleigh=False,
        )
        offset_space = env.action_space["offsets"]
        self.assertIsInstance(offset_space, spaces.MultiDiscrete)
        self.assertEqual((10,), offset_space.nvec.shape)
        np.testing.assert_array_equal(
            offset_space.nvec, np.full(10, env.num_channels)
        )
        with self.assertRaisesRegex(ValueError, "integer-valued"):
            env.step({"hoprate": 100.0, "offsets": np.full(10, 0.5)})
        with self.assertRaisesRegex(ValueError, "must be in"):
            env.step(
                {"hoprate": 100.0, "offsets": np.full(10, env.num_channels)}
            )

    @unittest.skipUnless(
        os.environ.get("FHSS_RUN_ENV_SMOKE") == "1",
        "Set FHSS_RUN_ENV_SMOKE=1 to run the real RF environment smoke test.",
    )
    def test_real_environment_step_is_one_replay_transition(self):
        settings.set_random_seeds(42)
        env = FHSSQPSKEnv(**settings.ENV_CONFIG)
        state, _ = env.reset()
        actions = np.zeros(env.num_blocks, dtype=np.int64)
        next_state, reward, terminated, truncated, info = env.step(
            {"hoprate": 100.0, "offsets": actions}
        )

        self.assertEqual((100, 100), state.shape)
        self.assertEqual((10,), np.asarray(info["block_rewards"]).shape)
        self.assertAlmostEqual(reward, np.mean(info["block_rewards"]), places=6)

        buffer = ReplayBuffer(4, env.num_blocks, env.num_channels)
        buffer.add(
            state,
            info["hoprate_used"],
            actions,
            info["block_rewards"],
            next_state,
            100.0,
            terminated or truncated,
        )
        self.assertEqual(1, buffer.size())


if __name__ == "__main__":
    unittest.main()
