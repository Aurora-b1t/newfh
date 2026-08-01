import pickle
import unittest
from unittest import mock

import numpy as np

import settings
from fh_env import FHSSQPSKEnv, _get_psd_plan, compute_psd_waterfall
from jammers import FastNoiseSource, IndiscriminateJammer


def acceleration_jammer_config(mode="both"):
    return {
        "mode": mode,
        "baseband_variant_count": 3,
        "sweep": {
            "step": 50_000.0,
            "power": 0.8,
            "dwell_time": 0.004,
            "bandwidth": 1_000.0,
        },
        "comb": {
            "power": 0.7,
            "bandwidth": 1_000.0,
            "switch_interval": 0.007,
            "channels_phase0": [0, 2],
            "channels_phase1": [1, 3],
        },
        "reactive": {
            "power": 1.0,
            "bandwidth": 1_000.0,
            "p_fa": 0.1,
            "detection_time": 0.001,
        },
    }


def make_acceleration_env(
    use_pregen,
    block_workers,
    *,
    mode="both",
    enable_rayleigh=True,
    enable_reactive=True,
):
    config = acceleration_jammer_config(mode)
    with mock.patch.object(settings, "JAMMER_CONFIG", config):
        return FHSSQPSKEnv(
            Startfre=0.0,
            Endfre=200_000.0,
            Fs=10_000,
            Sub_interval=50_000.0,
            Hoprate=100.0,
            hoprate_min=10.0,
            hoprate_max=1_000.0,
            Baud=1_000,
            dt=0.01,
            df=20_000.0,
            enable_reactive=enable_reactive,
            enable_sweep=True,
            enable_rayleigh=enable_rayleigh,
            rayleigh_coherence=7,
            use_pregen=use_pregen,
            noise_std=0.05,
            signal_power=0.1,
            block_workers=block_workers,
        )


def reference_psd_waterfall(
    signal,
    fs,
    f_start,
    f_end,
    dt,
    df,
    max_duration=0.1,
    window="hann",
):
    if max_duration is not None and max_duration > 0:
        signal = signal[:int(max_duration * fs)]
    nwin = int(dt * fs)
    n_bins = max(1, int(np.floor((f_end - f_start) / df)))
    if len(signal) < nwin:
        return np.zeros((0, n_bins))
    if window == "hann":
        win = np.hanning(nwin)
    elif window == "hamming":
        win = np.hamming(nwin)
    else:
        win = np.ones(nwin)
    nfft = int(2 ** np.ceil(np.log2(nwin)))
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    edges = f_start + np.arange(n_bins + 1) * df
    edge_indices = np.searchsorted(freqs, edges)
    norm = 1.0 / (fs * np.sum(win ** 2) / nwin)
    rows = []
    for start in range(0, len(signal) - nwin + 1, nwin):
        spec = np.fft.rfft(signal[start:start + nwin] * win, n=nfft)
        psd = np.abs(spec) ** 2 * norm
        row = np.zeros(n_bins)
        for bin_idx in range(n_bins):
            left = edge_indices[bin_idx]
            right = edge_indices[bin_idx + 1]
            if right > left:
                row[bin_idx] = np.sum(psd[left:right])
        rows.append(row)
    return 10.0 * np.log10(np.asarray(rows) + 1e-12)


def direct_fhss_carrier(env, length, hop_seq):
    carrier = np.empty(length, dtype=np.complex64)
    segment_lengths = env._hop_segment_lengths(length, len(hop_seq))
    phase_k = 2.0 * np.pi / env.Fs
    pos = 0
    cumsum_end = 0.0
    for hop_idx, segment_length in enumerate(segment_lengths):
        channel_idx = int(hop_seq[hop_idx]) % env.num_channels
        frequency = (
            env.Startfre
            + channel_idx * env.Sub_interval
            + 0.5 * env.Sub_interval
        )
        local_samples = np.arange(1, segment_length + 1, dtype=np.float64)
        phase = phase_k * (cumsum_end + frequency * local_samples)
        carrier[pos:pos + segment_length] = np.exp(1j * phase).astype(
            np.complex64
        )
        cumsum_end += frequency * segment_length
        pos += segment_length
    return carrier


def direct_comb(jammer, baseband, start_sample_idx, start_frequency, end_frequency):
    baseband = np.asarray(baseband) * jammer.c_power
    result = np.zeros(len(baseband), dtype=np.float64)
    phase_k = 2.0 * np.pi / jammer.Fs
    global_idx = int(start_sample_idx)
    result_idx = 0
    while result_idx < len(baseband):
        phase_offset = global_idx % jammer.comb_switch_samples
        phase = jammer.comb_phase_at(global_idx)
        take = min(
            jammer.comb_switch_samples - phase_offset,
            len(baseband) - result_idx,
        )
        frequencies = jammer._comb_frequencies(
            phase,
            start_frequency,
            end_frequency,
        )
        if len(frequencies):
            local_samples = np.arange(phase_offset, phase_offset + take)
            carrier = np.zeros(take, dtype=np.float64)
            for frequency in frequencies:
                carrier += np.cos((phase_k * frequency) * local_samples)
            carrier *= 1.0 / np.sqrt(len(frequencies))
            result[result_idx:result_idx + take] = (
                baseband[result_idx:result_idx + take] * carrier
            )
        global_idx += take
        result_idx += take
    return result


def direct_sweep(jammer, baseband, start_sample_idx, start_frequency, end_frequency):
    baseband = np.asarray(baseband) * jammer.s_power
    result = np.zeros(len(baseband), dtype=np.float64)
    bandwidth, dwell_samples, num_steps = jammer._sweep_layout(
        start_frequency,
        end_frequency,
    )
    phase_k = 2.0 * np.pi / jammer.Fs
    global_idx = int(start_sample_idx)
    result_idx = 0
    while result_idx < len(baseband):
        dwell_offset = global_idx % dwell_samples
        sweep_idx = (global_idx // dwell_samples) % num_steps
        take = min(dwell_samples - dwell_offset, len(baseband) - result_idx)
        frequency = (
            start_frequency + sweep_idx * jammer.s_step + jammer.s_step / 2.0
        )
        if frequency >= end_frequency:
            frequency = start_frequency + np.mod(
                frequency - start_frequency,
                bandwidth,
            )
        local_samples = np.arange(dwell_offset, dwell_offset + take)
        carrier = np.cos((phase_k * frequency) * local_samples)
        result[result_idx:result_idx + take] = (
            baseband[result_idx:result_idx + take] * carrier
        )
        global_idx += take
        result_idx += take
    return result


class BatchedPsdTests(unittest.TestCase):
    def test_batched_psd_matches_frame_loop_and_reuses_plan(self):
        signal = np.random.RandomState(91).standard_normal(1_234)
        kwargs = dict(
            fs=1_000.0,
            f_start=0.0,
            f_end=500.0,
            dt=0.1,
            df=50.0,
            max_duration=1.0,
            window="hann",
        )
        expected = reference_psd_waterfall(signal, **kwargs)
        before = _get_psd_plan.cache_info().hits
        actual = compute_psd_waterfall(signal, **kwargs)
        compute_psd_waterfall(signal, **kwargs)
        after = _get_psd_plan.cache_info().hits
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=3e-12)
        self.assertGreaterEqual(after - before, 1)


class CarrierCacheTests(unittest.TestCase):
    def test_cached_fhss_carriers_match_direct_formula_at_all_hoprates(self):
        env = make_acceleration_env(
            use_pregen=False,
            block_workers=1,
            mode="comb",
            enable_rayleigh=False,
            enable_reactive=False,
        )
        env.noise_std = 0.0
        try:
            rng = np.random.RandomState(7)
            i_pulse = rng.standard_normal(env._block_len).astype(np.float32)
            q_pulse = rng.standard_normal(env._block_len).astype(np.float32)
            baseband = (i_pulse + 1j * q_pulse).astype(np.complex64)
            for hoprate in (10.0, 100.0, 1_000.0):
                with self.subTest(hoprate=hoprate):
                    env._apply_hoprate(hoprate)
                    hop_count = max(1, int(round(hoprate * 0.1)))
                    hop_seq = np.arange(hop_count) % env.num_channels
                    received, carrier = env._assemble_signal_block(
                        i_pulse,
                        q_pulse,
                        hop_seq,
                        rng=np.random.RandomState(11),
                    )
                    expected_carrier = direct_fhss_carrier(
                        env,
                        env._block_len,
                        hop_seq,
                    )
                    expected_received = np.real(baseband * expected_carrier)
                    np.testing.assert_allclose(
                        carrier,
                        expected_carrier,
                        rtol=2e-5,
                        atol=2e-5,
                    )
                    np.testing.assert_allclose(
                        received,
                        expected_received,
                        rtol=3e-5,
                        atol=3e-5,
                    )
                    self.assertEqual(np.complex64, carrier.dtype)
        finally:
            env.close()

    def test_carrier_cache_has_no_configured_memory_ceiling(self):
        env = make_acceleration_env(
            use_pregen=False,
            block_workers=1,
            mode="comb",
            enable_rayleigh=False,
            enable_reactive=False,
        )
        try:
            self.assertFalse(
                hasattr(FHSSQPSKEnv, "CARRIER_TEMPLATE_CACHE_LIMIT")
            )
            env._apply_hoprate(10.0)
            hop_seq = np.array([2], dtype=np.int64)
            zeros = np.zeros(env._block_len, dtype=np.float32)
            _, carrier = env._assemble_signal_block(zeros, zeros, hop_seq)
            self.assertIsNotNone(env._carrier_templates)
            self.assertEqual(
                (env.num_channels, env._block_len),
                env._carrier_templates.shape,
            )
            np.testing.assert_allclose(
                carrier,
                direct_fhss_carrier(env, env._block_len, hop_seq),
                rtol=2e-5,
                atol=2e-5,
            )
        finally:
            env.close()


class JammerCarrierCacheTests(unittest.TestCase):
    def make_jammer(self, mode="both", noise_source=None):
        config = acceleration_jammer_config(mode)
        return IndiscriminateJammer(
            Fs=200_000,
            sweep_config=config["sweep"],
            comb_config=config["comb"],
            noise_source=noise_source,
            mode=mode,
        )

    def test_cached_comb_and_sweep_match_direct_formulas_across_cycles(self):
        jammer = self.make_jammer("both")
        baseband = np.random.RandomState(19).standard_normal(4_321).astype(
            np.float32
        )
        start_sample_idx = jammer.comb_switch_samples - 37
        actual_comb, _ = jammer._generate_comb_signal(
            len(baseband),
            start_sample_idx,
            0.0,
            200_000.0,
            baseband_noise=baseband,
        )
        actual_sweep, _ = jammer._generate_sweep_signal(
            len(baseband),
            start_sample_idx,
            0.0,
            200_000.0,
            baseband_noise=baseband,
        )
        np.testing.assert_allclose(
            actual_comb,
            direct_comb(jammer, baseband, start_sample_idx, 0.0, 200_000.0),
            rtol=2e-5,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            actual_sweep,
            direct_sweep(jammer, baseband, start_sample_idx, 0.0, 200_000.0),
            rtol=2e-5,
            atol=2e-5,
        )
        cache_ids = (
            id(jammer._carrier_comb0),
            id(jammer._carrier_comb1),
            id(jammer._carrier_sweep),
        )
        jammer._generate_comb_signal(
            20,
            3,
            0.0,
            200_000.0,
            baseband_noise=np.ones(20, dtype=np.float32),
        )
        self.assertEqual(
            cache_ids,
            (
                id(jammer._carrier_comb0),
                id(jammer._carrier_comb1),
                id(jammer._carrier_sweep),
            ),
        )

    def test_both_mode_draws_comb_then_sweep(self):
        calls = []

        class RecordingNoise:
            def __init__(self, label):
                self.label = label

            def get_noise(self, num_samples):
                calls.append(self.label)
                return np.ones(num_samples, dtype=np.float32)

        jammer = self.make_jammer("both", noise_source=RecordingNoise("unused"))
        jammer.ns_comb = RecordingNoise("comb")
        jammer.ns_sweep = RecordingNoise("sweep")
        jammer.generate_samples(100, 0.0, 200_000.0, rng=np.random.RandomState(3))
        self.assertEqual(["comb", "sweep"], calls)

    def test_dynamic_noise_slices_are_reproducible_and_statistically_fresh(self):
        source = FastNoiseSource(20_000, 2_000, duration=2.0, rng=np.random.RandomState(5))
        first = source.get_noise(4_000, rng=np.random.RandomState(17))
        replay = source.get_noise(4_000, rng=np.random.RandomState(17))
        second = source.get_noise(4_000, rng=np.random.RandomState(18))
        np.testing.assert_array_equal(first, replay)
        self.assertFalse(np.array_equal(first, second))
        self.assertLess(abs(float(np.mean(first))), 0.2)
        self.assertLess(abs(float(np.std(first)) - 1.0), 0.2)


class ParallelEnvironmentTests(unittest.TestCase):
    def assert_step_equal(self, serial_result, threaded_result):
        np.testing.assert_array_equal(serial_result[0], threaded_result[0])
        self.assertEqual(serial_result[1:], threaded_result[1:])

    def test_serial_and_threaded_paths_are_schedule_independent(self):
        actions = [
            {
                "hoprate": hoprate,
                "offsets": np.arange(10, dtype=np.int64) % 4,
            }
            for hoprate in (10.0, 100.0, 1_000.0)
        ]
        for use_pregen in (False, True):
            with self.subTest(use_pregen=use_pregen):
                np.random.seed(31415)
                serial = make_acceleration_env(use_pregen, 1)
                np.random.seed(31415)
                threaded = make_acceleration_env(use_pregen, 4)
                try:
                    serial_reset = serial.reset(seed=2026)
                    threaded_reset = threaded.reset(seed=2026)
                    np.testing.assert_array_equal(
                        serial_reset[0],
                        threaded_reset[0],
                    )
                    self.assertEqual(serial_reset[1], threaded_reset[1])
                    for action in actions:
                        serial_result = serial.step(action)
                        threaded_result = threaded.step(action)
                        self.assert_step_equal(serial_result, threaded_result)
                        self.assertEqual(serial.jammer_ptr, threaded.jammer_ptr)
                        self.assertEqual(
                            serial.reactive.state,
                            threaded.reactive.state,
                        )
                        self.assertEqual(
                            serial.reactive.current_channel,
                            threaded.reactive.current_channel,
                        )
                        self.assertEqual(
                            serial.jammer_variant_selector._cycle_choices,
                            threaded.jammer_variant_selector._cycle_choices,
                        )
                finally:
                    serial.close()
                    threaded.close()

    def test_rayleigh_and_reactive_feature_switches_preserve_determinism(self):
        action = {
            "hoprate": 100.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }
        for enable_rayleigh, enable_reactive in (
            (False, False),
            (True, False),
            (False, True),
            (True, True),
        ):
            with self.subTest(
                enable_rayleigh=enable_rayleigh,
                enable_reactive=enable_reactive,
            ):
                np.random.seed(99)
                serial = make_acceleration_env(
                    False,
                    1,
                    enable_rayleigh=enable_rayleigh,
                    enable_reactive=enable_reactive,
                )
                np.random.seed(99)
                threaded = make_acceleration_env(
                    False,
                    3,
                    enable_rayleigh=enable_rayleigh,
                    enable_reactive=enable_reactive,
                )
                try:
                    serial.reset(seed=123)
                    threaded.reset(seed=123)
                    self.assert_step_equal(serial.step(action), threaded.step(action))
                finally:
                    serial.close()
                    threaded.close()

    def test_worker_validation_auto_serial_and_pickle_lifecycle(self):
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_acceleration_env(False, 0)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            make_acceleration_env(False, True)

        auto = make_acceleration_env(False, None)
        self.assertEqual(1, auto.block_workers)
        auto.close()

        env = make_acceleration_env(True, 2)
        action = {
            "hoprate": 100.0,
            "offsets": np.zeros(10, dtype=np.int64),
        }
        try:
            env.reset(seed=11)
            env.step(action)
            self.assertIsNotNone(env._executor)
            payload = pickle.dumps(env)
            restored = pickle.loads(payload)
            try:
                self.assertIsNone(restored._executor)
                self.assertIs(restored, restored.pregen_data.env)
                restored.step(action)
                self.assertIsNotNone(restored._executor)
                restored.close()
                restored.close()
                self.assertIsNone(restored._executor)
            finally:
                restored.close()
        finally:
            env.close()
            env.close()


class RandomSubstreamTests(unittest.TestCase):
    def test_bits_awgn_and_rayleigh_substreams_are_independent_and_well_scaled(self):
        env = make_acceleration_env(
            use_pregen=False,
            block_workers=1,
            mode="comb",
            enable_rayleigh=True,
            enable_reactive=False,
        )
        try:
            seeds = env._draw_substream_seeds((10, 3))
            self.assertEqual(30, len(set(int(value) for value in seeds.ravel())))

            bits = np.vstack([
                env.modem.generate_bits(20_000, rng=np.random.RandomState(seed))
                for seed in seeds[:, 0]
            ])
            self.assertTrue(np.all(np.abs(np.mean(bits, axis=1) - 0.5) < 0.02))
            self.assertTrue(all(
                not np.array_equal(bits[left], bits[right])
                for left in range(len(bits))
                for right in range(left + 1, len(bits))
            ))

            awgn = np.vstack([
                np.random.RandomState(seed).standard_normal(20_000)
                for seed in seeds[:, 1]
            ])
            self.assertTrue(np.all(np.abs(np.mean(awgn, axis=1)) < 0.03))
            self.assertTrue(np.all(np.abs(np.std(awgn, axis=1) - 1.0) < 0.03))
            correlations = np.corrcoef(awgn)
            off_diagonal = correlations[~np.eye(len(awgn), dtype=bool)]
            self.assertLess(float(np.max(np.abs(off_diagonal))), 0.04)

            rayleigh = np.vstack([
                env._generate_rayleigh(20_000, rng=np.random.RandomState(seed))
                for seed in seeds[:, 2]
            ])
            expected_mean = np.sqrt(np.pi) / 2.0
            self.assertTrue(
                np.all(np.abs(np.mean(rayleigh, axis=1) - expected_mean) < 0.04)
            )
            self.assertTrue(
                np.all(np.abs(np.mean(rayleigh ** 2, axis=1) - 1.0) < 0.06)
            )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
