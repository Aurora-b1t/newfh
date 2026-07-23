import unittest
from unittest import mock

import numpy as np

import settings
from fh_env import FHSSQPSKEnv
from jammers import IndiscriminateJammer


class ConstantNoise:
    def get_noise(self, num_samples):
        return np.ones(num_samples, dtype=np.float32)


def jammer_config(mode):
    return {
        "mode": mode,
        "sweep": {
            "step": 5.0,
            "power": 0.8,
            "dwell_time": 0.004,
            "bandwidth": 10.0,
        },
        "comb": {
            "power": 0.8,
            "bandwidth": 10.0,
            "switch_interval": 0.3,
            "channels_phase0": [0],
            "channels_phase1": [1],
        },
        "reactive": {
            "power": 1.0,
            "bandwidth": 10.0,
            "p_fa": 0.1,
            "detection_time": 0.001,
        },
    }


def make_jammer(mode, switch_interval=0.3):
    config = jammer_config(mode)
    config["sweep"]["step"] = 50.0
    config["comb"]["switch_interval"] = switch_interval
    return IndiscriminateJammer(
        Fs=1000,
        sweep_config=config["sweep"],
        comb_config=config["comb"],
        noise_source=ConstantNoise(),
        mode=mode,
    )


def make_environment(mode="comb", use_pregen=False, enable_sweep=True):
    config = jammer_config(mode)
    patcher = mock.patch.object(settings, "JAMMER_CONFIG", config)
    patcher.start()
    try:
        env = FHSSQPSKEnv(
            Startfre=0.0,
            Endfre=100.0,
            Fs=1000,
            Sub_interval=50.0,
            Hoprate=10,
            hoprate_min=10.0,
            hoprate_max=100.0,
            Baud=100,
            dt=0.01,
            df=10.0,
            enable_reactive=False,
            enable_sweep=enable_sweep,
            enable_rayleigh=False,
            use_pregen=use_pregen,
            noise_std=0.01,
            signal_power=0.1,
        )
    finally:
        patcher.stop()
    return env


class IndiscriminateJammerTimingTests(unittest.TestCase):
    def test_rejects_invalid_comb_switch_intervals(self):
        for interval in (0.0, -0.1, np.nan, np.inf, 0.25):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "positive multiple of 0.1"):
                    make_jammer("comb", switch_interval=interval)

    def test_precomputes_only_active_natural_periods(self):
        expected = {
            "sweep": (80, None, 80),
            "comb": (None, 300, 600),
            "both": (80, 300, 1200),
        }
        for mode, (sweep_len, comb_len, total_period) in expected.items():
            with self.subTest(mode=mode):
                jammer = make_jammer(mode)
                jammer.precompute(Startfre=0.0, Endfre=1000.0)
                actual_sweep_len = (
                    None
                    if jammer.pre_buffer_sweep is None
                    else len(jammer.pre_buffer_sweep)
                )
                actual_comb_len = (
                    None
                    if jammer.pre_buffer_comb0 is None
                    else len(jammer.pre_buffer_comb0)
                )
                self.assertEqual(sweep_len, actual_sweep_len)
                self.assertEqual(comb_len, actual_comb_len)
                actual_comb1_len = (
                    None
                    if jammer.pre_buffer_comb1 is None
                    else len(jammer.pre_buffer_comb1)
                )
                self.assertEqual(comb_len, actual_comb1_len)
                self.assertEqual(total_period, jammer.precomputed_period_samples())

    def test_dynamic_generation_selects_phase_channels_from_sample_clock(self):
        jammer = make_jammer("comb")
        t = np.arange(100, dtype=np.float64) / jammer.Fs

        _, phase0_freqs = jammer.generate(
            t,
            Startfre=0.0,
            Endfre=100000.0,
            start_sample_idx=0,
        )
        _, phase1_freqs = jammer.generate(
            t,
            Startfre=0.0,
            Endfre=100000.0,
            start_sample_idx=300,
        )
        self.assertEqual({25000.0}, set(phase0_freqs))
        self.assertEqual({75000.0}, set(phase1_freqs))

    def test_comb_slice_hard_switches_and_wraps_multiple_times(self):
        jammer = make_jammer("comb")
        jammer.pre_buffer_comb0 = np.zeros(300, dtype=np.float32)
        jammer.pre_buffer_comb1 = np.ones(300, dtype=np.float32)

        actual = jammer.get_composite_signal(250, 800)
        expected = np.concatenate(
            [
                np.zeros(50, dtype=np.float32),
                np.ones(300, dtype=np.float32),
                np.zeros(300, dtype=np.float32),
                np.ones(150, dtype=np.float32),
            ]
        )
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual([0, 0, 1, 1, 0], [
            jammer.comb_phase_at(index)
            for index in (0, 299, 300, 599, 600)
        ])

    def test_sweep_slice_supports_more_than_one_wrap(self):
        jammer = make_jammer("sweep")
        jammer.pre_buffer_sweep = np.arange(4, dtype=np.float32)
        actual = jammer.get_composite_signal(3, 10)
        np.testing.assert_array_equal(
            np.array([3, 0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.float32),
            actual,
        )

    def test_both_mode_adds_independent_periodic_slices(self):
        jammer = make_jammer("both")
        jammer.pre_buffer_sweep = np.array([1, 2], dtype=np.float32)
        jammer.pre_buffer_comb0 = np.full(300, 10, dtype=np.float32)
        jammer.pre_buffer_comb1 = np.full(300, 20, dtype=np.float32)
        actual = jammer.get_composite_signal(298, 6)
        np.testing.assert_array_equal(
            np.array([11, 12, 21, 22, 21, 22], dtype=np.float32),
            actual,
        )


class EnvironmentJammerTimingTests(unittest.TestCase):
    def test_observation_cache_uses_minimum_mode_period(self):
        expected_steps = {"sweep": 4, "comb": 6, "both": 12}
        for mode, expected in expected_steps.items():
            with self.subTest(mode=mode):
                env = make_environment(mode=mode, use_pregen=True)
                self.assertEqual(expected, env.pregen_data.num_steps)
                np.testing.assert_array_equal(
                    env.pregen_data.get_observation(0),
                    env.pregen_data.get_observation(expected),
                )

        env = make_environment(use_pregen=True, enable_sweep=False)
        self.assertEqual(1, env.pregen_data.num_steps)

    def test_first_step_phases_match_for_dynamic_and_pregenerated_paths(self):
        expected = [0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
        action = {
            "hoprate": 10.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }

        for use_pregen in (False, True):
            with self.subTest(use_pregen=use_pregen):
                env = make_environment(use_pregen=use_pregen)
                env.reset()
                self.assertEqual(100, env.jammer_ptr)
                _, _, _, _, info = env.step(action)
                self.assertEqual(expected, info["comb_phases"])
                self.assertEqual(1200, env.jammer_ptr)

                env.reset()
                _, _, _, _, reset_info = env.step(action)
                self.assertEqual(expected, reset_info["comb_phases"])

    def test_comb_phases_is_empty_when_comb_is_inactive(self):
        action = {
            "hoprate": 10.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }
        for mode, enable_sweep in (("sweep", True), ("comb", False)):
            with self.subTest(mode=mode, enable_sweep=enable_sweep):
                env = make_environment(
                    mode=mode,
                    use_pregen=False,
                    enable_sweep=enable_sweep,
                )
                env.reset()
                _, _, _, _, info = env.step(action)
                self.assertEqual([], info["comb_phases"])


if __name__ == "__main__":
    unittest.main()
