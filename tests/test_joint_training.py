import copy
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from fh_env import FHSSQPSKEnv
from generate_offline_replay import make_hoprate_sampler
from joint_training import (
    build_derivative_nbs,
    execute_joint_step,
    load_joint_replay,
    resolve_environment_configs,
)
from SAC import ReplayBuffer
import settings
import train_joint_mbpo
import train_joint_sac
from train_joint_mbpo import parse_args as parse_joint_mbpo_args
from train_joint_sac import parse_args as parse_joint_sac_args


class _FakeAgent:
    def __init__(self, events):
        self.events = events
        self.calls = []

    def take_action(self, state_img, hoprate):
        self.events.append("agent")
        self.calls.append((np.asarray(state_img).copy(), float(hoprate)))
        return np.array([0, 1, 2], dtype=np.int64)


class _FakeNBS:
    def __init__(self, events):
        self.events = events
        self.observations = []

    def step(self, mean_ber):
        self.events.append("nbs")
        self.observations.append(float(mean_ber))
        return 130.0


class _TrainingNBS(_FakeNBS):
    def __init__(self, events):
        super().__init__(events)
        self.weights = np.array([0.5, 0.5], dtype=np.float64)
        self.p = 0.3
        self.delta = 0.01
        self.hoprate_step = 10.0
        self.derivative_threshold = -0.005

    def reset(self):
        return 120.0

    def get_last_derivative(self):
        return -0.004

    def get_best_hoprate(self):
        return 120.0

    def get_weighted_average(self):
        return 125.0

    def is_converged(self):
        return False


class _FakeEnv:
    num_blocks = 3
    num_channels = 4

    def __init__(self, events):
        self.events = events
        self.actions = []
        self.hoprate_min = 10.0
        self.hoprate_max = 1000.0
        self.closed = False

    def reset(self):
        return np.zeros((4, 4), dtype=np.float32), {}

    def step(self, action):
        self.events.append("env")
        self.actions.append(action)
        rewards = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        info = {
            "hoprate_used": 120.0,
            "ber_blocks": [0.1, 0.2, 0.3],
            "block_rewards": rewards.tolist(),
            "hop_sequences": [[0], [1], [2]],
        }
        return np.ones((4, 4), dtype=np.float32), 2.0, False, False, info

    def close(self):
        self.closed = True


class JointStepTests(unittest.TestCase):
    def test_joint_step_order_and_replay_next_hoprate(self):
        events = []
        state = np.zeros((4, 4), dtype=np.float32)
        agent = _FakeAgent(events)
        env = _FakeEnv(events)
        nbs = _FakeNBS(events)
        buffer = ReplayBuffer(4, num_heads=3, n_actions=4)

        result = execute_joint_step(state, 120.0, agent, env, nbs, buffer)

        self.assertEqual(["agent", "env", "nbs"], events)
        np.testing.assert_array_equal(agent.calls[0][0], state)
        self.assertEqual(120.0, agent.calls[0][1])
        self.assertEqual(120.0, env.actions[0]["hoprate"])
        np.testing.assert_array_equal(env.actions[0]["offsets"], [0, 1, 2])
        self.assertAlmostEqual(0.2, nbs.observations[0], places=6)
        self.assertEqual(130.0, result.next_hoprate)

        replay = buffer.get_all()
        np.testing.assert_allclose(replay["hoprates"], [120.0])
        np.testing.assert_allclose(replay["next_hoprates"], [130.0])
        np.testing.assert_array_equal(replay["actions"], [[0, 1, 2]])

    def test_joint_step_rejects_unexpected_environment_quantisation(self):
        class QuantisingEnv(_FakeEnv):
            def step(self, action):
                values = list(super().step(action))
                values[-1] = dict(values[-1], hoprate_used=110.0)
                return tuple(values)

        events = []
        with self.assertRaisesRegex(RuntimeError, "quantized"):
            execute_joint_step(
                np.zeros((4, 4), dtype=np.float32),
                120.0,
                _FakeAgent(events),
                QuantisingEnv(events),
                _FakeNBS(events),
                ReplayBuffer(4, num_heads=3, n_actions=4),
            )
        self.assertEqual(["agent", "env"], events)


class JointConfigurationTests(unittest.TestCase):
    def test_effective_configs_do_not_mutate_settings(self):
        env_before = copy.deepcopy(settings.ENV_CONFIG)
        jammer_before = copy.deepcopy(settings.JAMMER_CONFIG)
        args = SimpleNamespace(
            enable_reactive=True,
            enable_sweep=True,
            jammer_mode="both",
        )

        env_config, jammer_config = resolve_environment_configs(args)

        self.assertTrue(env_config["enable_reactive"])
        self.assertTrue(env_config["enable_sweep"])
        self.assertEqual("both", jammer_config["mode"])
        self.assertEqual(env_before, settings.ENV_CONFIG)
        self.assertEqual(jammer_before, settings.JAMMER_CONFIG)

    def test_environment_owns_a_copy_of_jammer_config(self):
        jammer_config = copy.deepcopy(settings.JAMMER_CONFIG)
        jammer_config["mode"] = "both"
        env = FHSSQPSKEnv(
            Startfre=0.0,
            Endfre=100.0,
            Fs=1000,
            Sub_interval=50.0,
            Baud=100,
            enable_reactive=False,
            enable_sweep=False,
            enable_rayleigh=False,
            use_pregen=False,
            jammer_config=jammer_config,
        )
        try:
            self.assertEqual("both", env.jammer_config["mode"])
            self.assertIsNot(jammer_config, env.jammer_config)
            jammer_config["mode"] = "comb"
            self.assertEqual("both", env.jammer_config["mode"])
        finally:
            env.close()

    def test_nbs_candidates_must_match_environment_grid(self):
        env = SimpleNamespace(hoprate_min=10.0, hoprate_max=1000.0)
        args = SimpleNamespace(
            nbs_step=10.0,
            nbs_p=0.3,
            nbs_delta=0.01,
            derivative_threshold=-0.005,
            seed=42,
        )
        nbs = build_derivative_nbs(env, args)
        np.testing.assert_allclose(nbs.candidates % 10.0, 0.0)

        args.nbs_step = 15.0
        with self.assertRaisesRegex(ValueError, "10 Hz"):
            build_derivative_nbs(env, args)

    def test_joint_entrypoint_defaults_and_cli_overrides(self):
        for parse_args in (parse_joint_sac_args, parse_joint_mbpo_args):
            defaults = parse_args([])
            self.assertEqual(
                settings.JOINT_OFFLINE_REPLAY_CONFIG["default_path"],
                defaults.offline_replay_path,
            )
            self.assertFalse(defaults.allow_replay_config_mismatch)
            self.assertEqual(-0.005, defaults.derivative_threshold)

            overridden = parse_args(
                [
                    "--enable_reactive",
                    "true",
                    "--enable_sweep",
                    "true",
                    "--jammer_mode",
                    "comb",
                    "--offline_replay_path",
                    "none",
                ]
            )
            self.assertTrue(overridden.enable_reactive)
            self.assertTrue(overridden.enable_sweep)
            self.assertEqual("comb", overridden.jammer_mode)
            self.assertIsNone(overridden.offline_replay_path)

    def test_joint_replay_is_strict_unless_explicitly_overridden(self):
        buffer = mock.Mock()
        env = SimpleNamespace(num_channels=4, num_blocks=3)
        state = np.zeros((4, 4), dtype=np.float32)
        metadata = {"env_config": {}, "jammer_config": {}, "reward_config": {}}

        with mock.patch(
            "joint_training.load_replay_into_buffer", return_value=(1, {})
        ) as loader:
            load_joint_replay(
                "replay.npz", buffer, state, env, metadata, logger=mock.Mock()
            )
            self.assertTrue(loader.call_args.kwargs["strict_environment_metadata"])

        with mock.patch(
            "joint_training.load_replay_into_buffer", return_value=(1, {})
        ) as loader:
            load_joint_replay(
                "replay.npz",
                buffer,
                state,
                env,
                metadata,
                allow_config_mismatch=True,
                logger=mock.Mock(),
            )
            self.assertFalse(loader.call_args.kwargs["strict_environment_metadata"])


class JointEntrypointLoopTests(unittest.TestCase):
    @staticmethod
    def _fake_components():
        events = []
        env = _FakeEnv(events)
        agent = _FakeAgent(events)
        agent.n_actions = env.num_channels
        agent.num_heads = env.num_blocks
        agent.update = mock.Mock(return_value={})
        buffer = ReplayBuffer(8, num_heads=env.num_blocks, n_actions=env.num_channels)
        nbs = _TrainingNBS(events)
        return events, env, agent, buffer, nbs

    def test_joint_sac_entrypoint_runs_one_causal_step(self):
        events, env, agent, buffer, nbs = self._fake_components()
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            train_joint_sac,
            "build_agent_and_env",
            return_value=(env, agent, buffer, "cpu", env.num_channels),
        ), mock.patch.object(
            train_joint_sac, "build_derivative_nbs", return_value=nbs
        ), mock.patch.object(
            train_joint_sac, "configure_figure_capture", return_value=(set(), output_dir)
        ), mock.patch.object(
            train_joint_sac, "setup_logger", return_value=(mock.Mock(), mock.Mock())
        ), mock.patch.object(train_joint_sac, "save_plots"), mock.patch.object(
            train_joint_sac, "save_nbs_artifacts"
        ), mock.patch.object(train_joint_sac, "save_sac_inference_checkpoint"):
            args = parse_joint_sac_args(
                [
                    "--steps_per_episode",
                    "1",
                    "--batch_size",
                    "2",
                    "--offline_replay_path",
                    "none",
                    "--output_dir",
                    output_dir,
                ]
            )
            train_joint_sac.train(args)

        self.assertEqual(["agent", "env", "nbs"], events)
        self.assertEqual(1, buffer.size())
        self.assertTrue(env.closed)

    def test_joint_mbpo_entrypoint_runs_one_causal_step(self):
        events, env, agent, buffer, nbs = self._fake_components()
        reward_model = SimpleNamespace(is_fitted=False)
        with tempfile.TemporaryDirectory() as output_dir, mock.patch.object(
            train_joint_mbpo,
            "build_agent_and_env",
            return_value=(env, agent, buffer, "cpu", env.num_channels),
        ), mock.patch.object(
            train_joint_mbpo, "build_derivative_nbs", return_value=nbs
        ), mock.patch.object(
            train_joint_mbpo,
            "configure_figure_capture",
            return_value=(set(), output_dir),
        ), mock.patch.object(
            train_joint_mbpo, "setup_logger", return_value=(mock.Mock(), mock.Mock())
        ), mock.patch.object(
            train_joint_mbpo, "StepRewardEnsemble", return_value=reward_model
        ), mock.patch.object(train_joint_mbpo, "save_plots"), mock.patch.object(
            train_joint_mbpo, "save_nbs_artifacts"
        ), mock.patch.object(train_joint_mbpo, "save_sac_inference_checkpoint"):
            args = parse_joint_mbpo_args(
                [
                    "--steps_per_episode",
                    "1",
                    "--batch_size",
                    "2",
                    "--model_train_batch_size",
                    "2",
                    "--offline_replay_path",
                    "none",
                    "--output_dir",
                    output_dir,
                ]
            )
            train_joint_mbpo.train(args)

        self.assertEqual(["agent", "env", "nbs"], events)
        self.assertEqual(1, buffer.size())
        self.assertTrue(env.closed)


class RandomHoprateReplayTests(unittest.TestCase):
    def test_random_sampler_uses_valid_uniform_grid(self):
        env = SimpleNamespace(hoprate_min=10.0, hoprate_max=40.0)
        sampler = make_hoprate_sampler(
            "random", 100.0, env, np.random.default_rng(12)
        )
        values = np.asarray([sampler() for _ in range(200)])
        self.assertTrue(set(values).issubset({10.0, 20.0, 30.0, 40.0}))
        self.assertGreater(len(set(values)), 1)

    def test_random_sampler_rejects_unknown_mode(self):
        env = SimpleNamespace(hoprate_min=10.0, hoprate_max=40.0)
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            make_hoprate_sampler(
                "trajectory", 100.0, env, np.random.default_rng(12)
            )


if __name__ == "__main__":
    unittest.main()
