import unittest
from unittest import mock

import numpy as np

import settings
from fh_env import FHSSQPSKEnv
from jammers import (
    BandLimitedNoiseVariantPool,
    IndiscriminateJammer,
    JammerVariantSelector,
    ReactiveJammer,
)


class ConstantNoise:
    def get_noise(self, num_samples):
        return np.ones(num_samples, dtype=np.float32)


class ArrayVariantPool:
    def __init__(self, bandwidth, variants):
        self.bandwidth = float(bandwidth)
        self.variants = np.asarray(variants, dtype=np.float32)
        self.num_variants, self.length = self.variants.shape

    def get_variant(self, variant_idx, num_samples=None, start_sample_idx=0):
        if num_samples is None:
            num_samples = self.length
        source = self.variants[int(variant_idx)]
        indices = (
            np.arange(int(num_samples), dtype=np.int64) + int(start_sample_idx)
        ) % self.length
        return source[indices]


class MappingSelector:
    def __init__(self, num_variants, cycle_choices=None, draws=None):
        self.num_variants = int(num_variants)
        self.cycle_choices = dict(cycle_choices or {})
        self.draws = list(draws or [0])
        self.draw_count = 0

    def choice_for_cycle(self, jammer_kind, cycle_idx):
        return int(self.cycle_choices[(jammer_kind, int(cycle_idx))])

    def draw(self):
        value = self.draws[self.draw_count % len(self.draws)]
        self.draw_count += 1
        return int(value)


def jammer_config(mode):
    return {
        "mode": mode,
        "baseband_variant_count": 4,
        "sweep": {
            "step": 5.0,
            "power": 0.8,
            "dwell_time": 0.004,
            "bandwidth": 10.0,
        },
        "comb": {
            "power": 0.8,
            "bandwidth": 10.0,
            "switch_interval": 0.05,
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


def make_jammer(mode, switch_interval=0.05, fs=1000):
    config = jammer_config(mode)
    config["sweep"]["step"] = 50.0
    config["comb"]["switch_interval"] = switch_interval
    return IndiscriminateJammer(
        Fs=fs,
        sweep_config=config["sweep"],
        comb_config=config["comb"],
        noise_source=ConstantNoise(),
        mode=mode,
    )


def expected_comb_phases(start_sample_idx, num_samples, switch_samples):
    return [
        int((sample_idx // switch_samples) % 2)
        for sample_idx in range(
            int(start_sample_idx),
            int(start_sample_idx) + int(num_samples),
        )
    ]


def make_environment(
    mode="comb",
    use_pregen=False,
    enable_sweep=True,
    enable_reactive=False,
    config=None,
):
    config = jammer_config(mode) if config is None else config
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
            enable_reactive=enable_reactive,
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
    def test_accepts_1ms_comb_switch_intervals(self):
        for interval, expected_samples in (
            (0.001, 1),
            (0.007, 7),
            (0.05, 50),
            (0.055, 55),
            (0.073, 73),
            (0.15, 150),
        ):
            with self.subTest(interval=interval):
                jammer = make_jammer("comb", switch_interval=interval)
                self.assertEqual(expected_samples, jammer.comb_switch_samples)
                self.assertEqual(2 * expected_samples, jammer.comb_period_samples)

    def test_73ms_interval_is_exact_at_production_sample_rate(self):
        jammer = make_jammer("comb", switch_interval=0.073, fs=10_000_000)
        self.assertEqual(730_000, jammer.comb_switch_samples)
        self.assertEqual(1_460_000, jammer.comb_period_samples)

    def test_rejects_invalid_comb_switch_intervals(self):
        for interval in (0.0, -0.001, np.nan, np.inf, 0.0005, 0.0735):
            with self.subTest(interval=interval):
                with self.assertRaisesRegex(ValueError, "positive multiple of 0.001"):
                    make_jammer("comb", switch_interval=interval)

    def test_precomputes_only_active_natural_periods(self):
        expected = {
            "sweep": (80, None, 80),
            "comb": (None, 50, 100),
            "both": (80, 50, 400),
        }
        for mode, (sweep_len, comb_len, total_period) in expected.items():
            with self.subTest(mode=mode):
                jammer = make_jammer(mode)
                jammer.precompute(Startfre=0.0, Endfre=1000.0)
                actual_sweep_len = (
                    None
                    if jammer.pre_buffer_sweep is None
                    else jammer.pre_buffer_sweep.shape[-1]
                )
                actual_comb_len = (
                    None
                    if jammer.pre_buffer_comb0 is None
                    else jammer.pre_buffer_comb0.shape[-1]
                )
                self.assertEqual(sweep_len, actual_sweep_len)
                self.assertEqual(comb_len, actual_comb_len)
                actual_comb1_len = (
                    None
                    if jammer.pre_buffer_comb1 is None
                    else jammer.pre_buffer_comb1.shape[-1]
                )
                self.assertEqual(comb_len, actual_comb1_len)
                self.assertEqual(total_period, jammer.precomputed_period_samples())

    def test_dynamic_generation_selects_phase_channels_from_sample_clock(self):
        cases = (
            (0.001, 0, 4, [25000.0, 75000.0, 25000.0, 75000.0]),
            (0.007, 3, 30, [25000.0, 75000.0, 25000.0, 75000.0, 25000.0]),
            (0.05, 0, 100, [25000.0, 75000.0]),
            (0.15, 100, 100, [25000.0, 75000.0]),
        )
        for interval, start_sample_idx, num_samples, expected in cases:
            with self.subTest(interval=interval):
                jammer = make_jammer("comb", switch_interval=interval)
                t = np.arange(num_samples, dtype=np.float64) / jammer.Fs
                _, frequencies = jammer.generate(
                    t,
                    Startfre=0.0,
                    Endfre=100000.0,
                    start_sample_idx=start_sample_idx,
                )
                self.assertEqual(expected, frequencies)

    def test_comb_slice_hard_switches_and_wraps_multiple_times(self):
        jammer = make_jammer("comb")
        jammer.pre_buffer_comb0 = np.zeros(50, dtype=np.float32)
        jammer.pre_buffer_comb1 = np.ones(50, dtype=np.float32)

        actual = jammer.get_composite_signal(25, 250)
        expected = np.concatenate(
            [
                np.zeros(25, dtype=np.float32),
                np.ones(50, dtype=np.float32),
                np.zeros(50, dtype=np.float32),
                np.ones(50, dtype=np.float32),
                np.zeros(50, dtype=np.float32),
                np.ones(25, dtype=np.float32),
            ]
        )
        np.testing.assert_array_equal(expected, actual)
        self.assertEqual([0, 0, 1, 1, 0], [
            jammer.comb_phase_at(index)
            for index in (0, 49, 50, 99, 100)
        ])

    def test_comb_slice_switches_at_millisecond_boundaries(self):
        for interval, start_sample_idx in (
            (0.001, 0),
            (0.007, 3),
            (0.05, 0),
            (0.15, 100),
        ):
            with self.subTest(interval=interval):
                jammer = make_jammer("comb", switch_interval=interval)
                jammer.pre_buffer_comb0 = np.zeros(
                    jammer.comb_switch_samples,
                    dtype=np.float32,
                )
                jammer.pre_buffer_comb1 = np.ones(
                    jammer.comb_switch_samples,
                    dtype=np.float32,
                )
                actual = jammer.get_composite_signal(start_sample_idx, 100)
                expected = np.asarray(
                    expected_comb_phases(
                        start_sample_idx,
                        100,
                        jammer.comb_switch_samples,
                    ),
                    dtype=np.float32,
                )
                np.testing.assert_array_equal(expected, actual)

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
        jammer.pre_buffer_comb0 = np.full(50, 10, dtype=np.float32)
        jammer.pre_buffer_comb1 = np.full(50, 20, dtype=np.float32)
        actual = jammer.get_composite_signal(48, 6)
        np.testing.assert_array_equal(
            np.array([11, 12, 21, 22, 21, 22], dtype=np.float32),
            actual,
        )

    def test_variant_pool_validates_count_and_generates_independent_float32_rows(self):
        for invalid in (0, -1, 1.5, True):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    BandLimitedNoiseVariantPool(1000, 100, invalid, 256)

        np.random.seed(7)
        pool = BandLimitedNoiseVariantPool(1000, 100, 4, 256)
        self.assertEqual((4, 256), pool.variants.shape)
        self.assertEqual(np.float32, pool.variants.dtype)
        for left in range(pool.num_variants):
            self.assertAlmostEqual(float(np.std(pool.variants[left])), 1.0, places=5)
            for right in range(left + 1, pool.num_variants):
                self.assertFalse(np.array_equal(pool.variants[left], pool.variants[right]))

    def test_shared_selector_replays_choices_after_reset(self):
        selector = JammerVariantSelector(4, seed=123)
        first = [
            selector.choice_for_cycle("sweep", 0),
            selector.choice_for_cycle("sweep", 0),
            selector.draw(),
            selector.choice_for_cycle("comb", 0),
        ]
        self.assertEqual(first[0], first[1])

        selector.reset()
        second = [
            selector.choice_for_cycle("sweep", 0),
            selector.choice_for_cycle("sweep", 0),
            selector.draw(),
            selector.choice_for_cycle("comb", 0),
        ]
        self.assertEqual(first, second)
        self.assertTrue(all(0 <= value < 4 for value in first))

    def test_sweep_selects_one_variant_for_each_complete_cycle(self):
        jammer = make_jammer("sweep")
        jammer.pre_buffer_sweep = np.vstack(
            [
                np.zeros(4, dtype=np.float32),
                np.full(4, 10, dtype=np.float32),
            ]
        )
        jammer.variant_selector = MappingSelector(
            2,
            cycle_choices={("sweep", 0): 1, ("sweep", 1): 0},
        )
        actual = jammer.get_composite_signal(2, 4)
        np.testing.assert_array_equal(
            np.array([10, 10, 0, 0], dtype=np.float32),
            actual,
        )

    def test_comb_uses_same_variant_across_both_phases(self):
        jammer = make_jammer("comb")
        jammer.pre_buffer_comb0 = np.vstack(
            [
                np.zeros(50, dtype=np.float32),
                np.full(50, 10, dtype=np.float32),
            ]
        )
        jammer.pre_buffer_comb1 = np.vstack(
            [
                np.ones(50, dtype=np.float32),
                np.full(50, 11, dtype=np.float32),
            ]
        )
        jammer.variant_selector = MappingSelector(
            2,
            cycle_choices={("comb", 0): 1, ("comb", 1): 0},
        )
        actual = jammer.get_composite_signal(48, 54)
        expected = np.concatenate(
            [
                np.full(2, 10, dtype=np.float32),
                np.full(50, 11, dtype=np.float32),
                np.zeros(2, dtype=np.float32),
            ]
        )
        np.testing.assert_array_equal(expected, actual)

    def test_comb_precompute_keeps_baseband_continuous_across_phase_switch(self):
        jammer = make_jammer("comb")
        jammer.c_power = 1.0
        baseband = np.arange(jammer.comb_period_samples, dtype=np.float32)
        pool = ArrayVariantPool(jammer.c_bw, baseband[np.newaxis, :])
        jammer.set_variant_sources(
            comb_variant_pool=pool,
            variant_selector=MappingSelector(1),
        )
        jammer.precompute(Startfre=0.0, Endfre=100000.0)

        np.testing.assert_allclose(
            baseband[:jammer.comb_switch_samples],
            jammer.pre_buffer_comb0[0],
            rtol=0.0,
            atol=1e-4,
        )
        np.testing.assert_allclose(
            baseband[jammer.comb_switch_samples:],
            jammer.pre_buffer_comb1[0],
            rtol=0.0,
            atol=1e-4,
        )

    def test_reactive_jammer_draws_a_variant_for_every_jam_slot(self):
        pool = ArrayVariantPool(
            10.0,
            np.array([[1.0], [2.0]], dtype=np.float32),
        )
        selector = MappingSelector(2, draws=[0, 1])
        jammer = ReactiveJammer(
            Fs=1000,
            num_channels=1,
            sub_interval=1000.0,
            detection_time=0.001,
            p_fa=0.1,
            power=1.0,
            bandwidth=10.0,
            Startfre=0.0,
            noise_std=0.1,
            signal_power=0.1,
            Baud=100,
            variant_pool=pool,
            variant_selector=selector,
        )
        t = np.arange(1, dtype=np.float64) / jammer.Fs

        outputs = []
        for _ in range(2):
            jammer.state = "jam"
            jammer.current_channel = 0
            signal, active = jammer.generate(t, [0], 0.0, 1000.0, 10.0)
            self.assertTrue(active)
            outputs.append(float(signal[0]))

        self.assertEqual([1.0, 2.0], outputs)
        self.assertEqual(2, selector.draw_count)


class EnvironmentJammerTimingTests(unittest.TestCase):
    def test_environment_uses_1ms_comb_interval_quantum(self):
        config = jammer_config("comb")
        config["comb"]["switch_interval"] = 0.007
        env = make_environment(use_pregen=False, config=config)
        self.assertEqual(7, env.sweep.comb_switch_samples)

        invalid_config = jammer_config("comb")
        invalid_config["comb"]["switch_interval"] = 0.0075
        with self.assertRaisesRegex(ValueError, "positive multiple of 0.001"):
            make_environment(use_pregen=False, config=invalid_config)

    def test_equal_bandwidth_jammers_share_pool_and_selector(self):
        config = jammer_config("both")
        shared_bandwidth = 10.0
        config["reactive"]["bandwidth"] = shared_bandwidth
        config["sweep"]["bandwidth"] = shared_bandwidth
        config["comb"]["bandwidth"] = shared_bandwidth
        env = make_environment(
            mode="both",
            use_pregen=True,
            enable_reactive=True,
            config=config,
        )

        self.assertEqual(1, len(env.jammer_variant_pools))
        pool = env.jammer_variant_pools[shared_bandwidth]
        self.assertIs(pool, env.reactive.variant_pool)
        self.assertIs(pool, env.sweep.sweep_variant_pool)
        self.assertIs(pool, env.sweep.comb_variant_pool)
        self.assertIs(env.jammer_variant_selector, env.reactive.variant_selector)
        self.assertIs(env.jammer_variant_selector, env.sweep.variant_selector)
        self.assertEqual(4, pool.num_variants)

    def test_different_bandwidths_create_distinct_variant_pools(self):
        config = jammer_config("both")
        config["reactive"]["bandwidth"] = 10.0
        config["sweep"]["bandwidth"] = 20.0
        config["comb"]["bandwidth"] = 30.0
        env = make_environment(
            mode="both",
            use_pregen=True,
            enable_reactive=True,
            config=config,
        )
        self.assertEqual({10.0, 20.0, 30.0}, set(env.jammer_variant_pools))
        self.assertIsNot(env.reactive.variant_pool, env.sweep.sweep_variant_pool)
        self.assertIsNot(env.sweep.sweep_variant_pool, env.sweep.comb_variant_pool)

    def test_environment_rejects_invalid_variant_count(self):
        for invalid in (0, -1, 1.5, True):
            config = jammer_config("comb")
            config["baseband_variant_count"] = invalid
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    make_environment(
                        use_pregen=False,
                        enable_sweep=False,
                        config=config,
                    )

    def test_pregenerated_observation_recomputes_with_fresh_noise(self):
        env = make_environment(use_pregen=True, enable_sweep=False)
        self.assertFalse(hasattr(env.pregen_data, "obs_buffer"))
        captured_signals = []

        def capture_waterfall(signal, *args, **kwargs):
            captured_signals.append(np.asarray(signal).copy())
            return np.zeros((10, 10), dtype=np.float32)

        noise_samples = [
            np.zeros(100, dtype=np.float32),
            np.ones(100, dtype=np.float32),
        ]
        with mock.patch(
            "fh_env.np.random.randn",
            side_effect=noise_samples,
        ), mock.patch(
            "fh_env.compute_psd_waterfall",
            side_effect=capture_waterfall,
        ) as psd_mock:
            env._observe_100ms()
            env._observe_100ms()

        self.assertEqual(2, psd_mock.call_count)
        np.testing.assert_array_equal(captured_signals[0], np.zeros(100))
        np.testing.assert_allclose(
            captured_signals[1],
            np.full(100, 0.01),
            rtol=0.0,
            atol=1e-8,
        )

    def test_environment_reset_replays_pregenerated_variant_choices(self):
        env = make_environment(use_pregen=True)
        action = {
            "hoprate": 10.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }

        env.reset()
        env.step(action)
        first_choices = dict(env.jammer_variant_selector._cycle_choices)

        env.reset()
        env.step(action)
        second_choices = dict(env.jammer_variant_selector._cycle_choices)
        self.assertEqual(first_choices, second_choices)

    def test_comb_phase_slots_match_for_dynamic_and_pregenerated_paths(self):
        action = {
            "hoprate": 10.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }

        for interval in (0.007, 0.05, 0.15):
            config = jammer_config("comb")
            config["comb"]["switch_interval"] = interval
            for use_pregen in (False, True):
                with self.subTest(
                    interval=interval,
                    use_pregen=use_pregen,
                ):
                    env = make_environment(
                        use_pregen=use_pregen,
                        config=config,
                    )
                    _, reset_info = env.reset()
                    self.assertEqual([], reset_info["comb_phases"])
                    self.assertEqual(100, env.jammer_ptr)
                    _, _, _, _, info = env.step(action)
                    expected = [
                        expected_comb_phases(
                            100 + block_idx * 100,
                            100,
                            env.sweep.comb_switch_samples,
                        )
                        for block_idx in range(10)
                    ]
                    self.assertEqual((10, 100), np.asarray(info["comb_phases"]).shape)
                    self.assertEqual(expected, info["comb_phases"])
                    self.assertEqual(1200, env.jammer_ptr)

                    env.reset()
                    _, _, _, _, replayed_info = env.step(action)
                    self.assertEqual(expected, replayed_info["comb_phases"])

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
