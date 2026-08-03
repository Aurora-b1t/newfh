import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
from gymnasium import spaces

import settings
import train_offsets
from fh_env import FHSSQPSKEnv, compute_block_rewards
from SAC import ReplayBuffer
from train_mbpo import _validate_args, parse_args as parse_mbpo_args
from train_offsets import (
    log_active_comb_channels,
    parse_optional_replay_path,
    replay_ready,
)


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

    def test_baseline_entrypoint_saves_inference_checkpoint_with_metadata(self):
        class FakeEnv:
            num_blocks = 3
            num_channels = 4
            hoprate_min = 10.0
            hoprate_max = 1000.0

            def reset(self):
                return np.zeros((8, 8), dtype=np.float32), {}

            def step(self, action):
                self.last_action = action
                block_rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
                return (
                    np.ones((8, 8), dtype=np.float32),
                    float(block_rewards.mean()),
                    True,
                    False,
                    {
                        "ber_blocks": [0.1, 0.2, 0.3],
                        "block_rewards": block_rewards,
                        "hoprate_used": float(action["hoprate"]),
                        "hop_sequences": [[0], [1], [2]],
                    },
                )

        class FakeAgent:
            n_actions = 4
            num_heads = 3

            def take_action(self, state_img, hoprate):
                return np.array([0, 1, 2], dtype=np.int64)

        env = FakeEnv()
        agent = FakeAgent()
        buffer = ReplayBuffer(4, num_heads=3, n_actions=4)
        logger = mock.Mock()
        hop_logger = mock.Mock()

        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            train_offsets,
            "build_agent_and_env",
            return_value=(env, agent, buffer, "cpu", env.num_channels),
        ), mock.patch.object(
            train_offsets,
            "setup_logger",
            return_value=(logger, hop_logger),
        ), mock.patch.object(
            train_offsets,
            "save_sac_inference_checkpoint",
        ) as save_checkpoint, mock.patch.dict(
            settings.PLOT_CONFIG,
            {"figure_save_steps": []},
        ):
            args = SimpleNamespace(
                output_dir=output_dir,
                log_file="training_log.txt",
                offline_replay_path=None,
                steps_per_episode=1,
                batch_size=2,
                update_iters_per_step=1,
            )
            train_offsets.train(args)

            save_checkpoint.assert_called_once()
            call_args = save_checkpoint.call_args.args
            self.assertIs(agent, call_args[0])
            self.assertEqual(
                os.path.join(output_dir, "sac_inference.pt"),
                call_args[1],
            )
            self.assertEqual((8, 8), call_args[2])
            self.assertEqual(env.hoprate_min, call_args[3])
            self.assertEqual(env.hoprate_max, call_args[4])
            metadata = call_args[5]
            self.assertEqual(env.num_channels, metadata["num_actions"])
            self.assertEqual(env.num_blocks, metadata["num_blocks"])
            self.assertEqual(
                settings.TRAIN_CONFIG["fixed_hoprate"],
                metadata["fixed_hoprate"],
            )
            self.assertIn("env_config", metadata)
            self.assertIn("jammer_config", metadata)
            self.assertIn("reward_config", metadata)


class EnvironmentInterfaceTests(unittest.TestCase):
    @staticmethod
    def make_small_environment(use_pregen, enable_rayleigh):
        return FHSSQPSKEnv(
            Startfre=0.0,
            Endfre=100.0,
            Fs=1000,
            Sub_interval=50.0,
            Hoprate=100,
            hoprate_min=10.0,
            hoprate_max=100.0,
            Baud=100,
            dt=0.01,
            df=10.0,
            enable_reactive=False,
            enable_sweep=False,
            enable_rayleigh=enable_rayleigh,
            rayleigh_coherence=4,
            use_pregen=use_pregen,
            noise_std=0.5,
            signal_power=0.1,
        )

    def test_reward_formula(self):
        ber = np.linspace(0.0, 0.9, 10)
        rewards = compute_block_rewards(ber, 100.0)
        expected = (
            settings.REWARD_CONFIG["base_reward"]
            - settings.REWARD_CONFIG["ber_penalty"] * ber
            - settings.REWARD_CONFIG["hoprate_penalty"] * 100.0
        )
        np.testing.assert_allclose(rewards, expected, rtol=1e-6, atol=1e-6)

    def test_pregenerated_blocks_use_fresh_awgn(self):
        env = self.make_small_environment(use_pregen=True, enable_rayleigh=False)
        zeros = np.zeros(env.pregen_data.block_len, dtype=np.float32)
        ones = np.ones(env.pregen_data.block_len, dtype=np.float32)
        hop_seq = np.zeros(10, dtype=np.int64)

        with mock.patch(
            "fh_env.np.random.randn",
            side_effect=[zeros, ones],
        ):
            first, first_carrier, _ = env.pregen_data.get_block(hop_seq)
            second, second_carrier, _ = env.pregen_data.get_block(hop_seq)

        np.testing.assert_array_equal(first_carrier, second_carrier)
        np.testing.assert_allclose(
            second - first,
            np.full(env.pregen_data.block_len, env.noise_std),
            rtol=0.0,
            atol=1e-6,
        )

    def test_rayleigh_is_generated_independently_for_each_hop(self):
        env = self.make_small_environment(use_pregen=False, enable_rayleigh=True)
        I_pulse = np.ones(30, dtype=np.float32)
        Q_pulse = np.zeros(30, dtype=np.float32)
        hop_seq = np.array([0, 1, 0], dtype=np.int64)
        call_values = []

        def fake_rayleigh(length):
            value = len(call_values) + 1
            call_values.append((length, value))
            return np.full(length, value, dtype=np.float32)

        env.noise_std = 0.0
        with mock.patch.object(
            env,
            "_generate_rayleigh",
            side_effect=fake_rayleigh,
        ):
            first, _ = env._assemble_signal_block(I_pulse, Q_pulse, hop_seq)
            second, _ = env._assemble_signal_block(I_pulse, Q_pulse, hop_seq)

        self.assertEqual([(10, 1), (10, 2), (10, 3), (10, 4), (10, 5), (10, 6)], call_values)
        self.assertFalse(np.array_equal(first, second))

    def test_dynamic_and_pregenerated_paths_share_block_assembler(self):
        action = {
            "hoprate": 10.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }
        for use_pregen in (False, True):
            with self.subTest(use_pregen=use_pregen):
                env = self.make_small_environment(
                    use_pregen=use_pregen,
                    enable_rayleigh=True,
                )
                env.reset()
                with mock.patch.object(
                    env,
                    "_assemble_signal_block",
                    wraps=env._assemble_signal_block,
                ) as assemble_mock:
                    env.step(action)
                self.assertEqual(10, assemble_mock.call_count)

    def test_comb_channel_groups_are_logged_only_when_active(self):
        logger = mock.Mock()
        active_env = SimpleNamespace(
            enable_sweep=True,
            sweep=SimpleNamespace(
                comb_enabled=True,
                comb_channels=([0, 2], [1, 3]),
            ),
        )
        self.assertTrue(log_active_comb_channels(active_env, logger=logger))
        logger.info.assert_called_once_with(
            "Comb channels_phase0=%s | channels_phase1=%s",
            [0, 2],
            [1, 3],
        )

        for inactive_env in (
            SimpleNamespace(enable_sweep=False, sweep=None),
            SimpleNamespace(
                enable_sweep=True,
                sweep=SimpleNamespace(comb_enabled=False),
            ),
        ):
            inactive_logger = mock.Mock()
            self.assertFalse(
                log_active_comb_channels(inactive_env, logger=inactive_logger)
            )
            inactive_logger.info.assert_not_called()

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
