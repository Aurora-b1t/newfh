"""
Jammer implementations for FHSS anti-jamming simulation.

Includes a pre-generated band-limited noise source, an energy-detection based
reactive jammer, and an indiscriminate sweep/comb jammer with optional
pre-computed signal buffers.
"""

import math

import numpy as np
from scipy.stats import chi2, ncx2
from scipy.special import roots_laguerre


def _rng_randint(rng, low, high=None):
    """Draw one integer from either NumPy's legacy or Generator API."""
    rng = np.random if rng is None else rng
    if hasattr(rng, "integers"):
        return int(rng.integers(low, high))
    return int(rng.randint(low, high))


def _rng_random(rng):
    """Draw one uniform variate from either NumPy RNG API."""
    rng = np.random if rng is None else rng
    return float(rng.random())


def _rng_standard_normal(rng, size):
    """Draw standard-normal samples from either NumPy RNG API."""
    rng = np.random if rng is None else rng
    if rng is np.random:
        return rng.randn(size)
    return rng.standard_normal(size)


def _generate_band_limited_noise(Fs, bandwidth, length, rng=None):
    """Generate one normalized band-limited Gaussian sequence."""
    length = int(length)
    if length <= 0:
        raise ValueError("Band-limited noise length must be positive.")

    n = np.asarray(_rng_standard_normal(rng, length), dtype=np.float32)
    spec = np.fft.rfft(n)
    if bandwidth > 0:
        cutoff_idx = int(np.floor(float(bandwidth) * length / (2.0 * float(Fs))))
        if cutoff_idx + 1 < len(spec):
            spec[cutoff_idx + 1:] = 0

    n_lp = np.fft.irfft(spec, n=length)
    std_val = np.std(n_lp)
    if std_val > 1e-12:
        n_lp /= std_val
    return np.asarray(n_lp, dtype=np.float32)


def _noise_source_slice(source, num_samples, rng=None):
    """Use RNG-aware built-ins without breaking legacy custom sources."""
    if rng is None or not isinstance(source, FastNoiseSource):
        return source.get_noise(num_samples)
    return source.get_noise(num_samples, rng=rng)


def validate_baseband_variant_count(value):
    """Return a validated positive integer jammer baseband variant count."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError("JAMMER_CONFIG['baseband_variant_count'] must be a positive integer.")
    value = int(value)
    if value <= 0:
        raise ValueError("JAMMER_CONFIG['baseband_variant_count'] must be a positive integer.")
    return value


class BandLimitedNoiseVariantPool:
    """Independent normalized baseband-noise variants for one bandwidth."""

    def __init__(self, Fs, bandwidth, num_variants, length, rng=None):
        self.Fs = float(Fs)
        self.bandwidth = float(bandwidth)
        self.num_variants = validate_baseband_variant_count(num_variants)
        self.length = int(length)
        if self.length <= 0:
            raise ValueError("Band-limited noise variant length must be positive.")

        self.variants = np.empty(
            (self.num_variants, self.length),
            dtype=np.float32,
        )
        for variant_idx in range(self.num_variants):
            self.variants[variant_idx] = _generate_band_limited_noise(
                self.Fs,
                self.bandwidth,
                self.length,
                rng=rng,
            )

    def get_variant(self, variant_idx, num_samples=None, start_sample_idx=0):
        """Return a periodic slice from one stored variant."""
        variant_idx = int(variant_idx)
        if variant_idx < 0 or variant_idx >= self.num_variants:
            raise IndexError("variant_idx is outside the configured baseband pool.")

        if num_samples is None:
            num_samples = self.length
        num_samples = int(num_samples)
        start_sample_idx = int(start_sample_idx)
        if num_samples < 0 or start_sample_idx < 0:
            raise ValueError("Noise slice indices must be non-negative.")
        if num_samples == 0:
            return np.zeros(0, dtype=np.float32)

        source = self.variants[variant_idx]
        result = np.empty(num_samples, dtype=np.float32)
        source_idx = start_sample_idx % self.length
        result_idx = 0
        while result_idx < num_samples:
            take = min(self.length - source_idx, num_samples - result_idx)
            result[result_idx:result_idx + take] = source[source_idx:source_idx + take]
            result_idx += take
            source_idx = 0
        return result


class JammerVariantSelector:
    """One reproducible random-with-replacement selector shared by jammers."""

    def __init__(self, num_variants, seed=None):
        self.num_variants = validate_baseband_variant_count(num_variants)
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        self.seed = int(seed)
        self._rng = np.random.RandomState(self.seed)
        self._cycle_choices = {}

    def reset(self):
        self._rng = np.random.RandomState(self.seed)
        self._cycle_choices.clear()

    def draw(self):
        return int(self._rng.randint(0, self.num_variants))

    def choice_for_cycle(self, jammer_kind, cycle_idx):
        cycle_idx = int(cycle_idx)
        if cycle_idx < 0:
            raise ValueError("cycle_idx must be non-negative.")
        key = (str(jammer_kind), cycle_idx)
        if key not in self._cycle_choices:
            self._cycle_choices[key] = self.draw()
        return self._cycle_choices[key]

# -----------------------------
# 快速噪声源
# -----------------------------
class FastNoiseSource:
    """
    预生成长段带限噪声的缓冲区，避免反复计算 FFT/IFFT。
    """
    def __init__(self, Fs, bandwidth, duration=1.0, rng=None):
        self.length = int(Fs * duration)
        self.noise = _generate_band_limited_noise(
            Fs,
            bandwidth,
            self.length,
            rng=rng,
        )

    def get_noise(self, num_samples, rng=None):
        if num_samples >= self.length:
            # 这种情况下直接返回全部并循环填充
            tile_count = (num_samples // self.length) + 1
            return np.tile(self.noise, tile_count)[:num_samples]
            
        start = _rng_randint(rng, 0, self.length - num_samples)
        return self.noise[start : start + num_samples]


# -----------------------------
# 干扰机
# -----------------------------
class ReactiveJammer:
    """
    Reactive jammer based on energy detection theory.

    Reference:
      Urkowitz, "Energy Detection of Unknown Deterministic Signals",
      Proc. IEEE, vol. 55, no. 4, pp. 523–531, Apr. 1967.

    Operates on a **1 ms fundamental time unit**:

    * SCAN mode — scans one channel per 1 ms slot.  Energy detection
      decides whether a signal is present:
      - H0 (noise only):     V ~ χ²(2γ)        (central chi-square)
      - H1 (signal present): V ~ χ²(2γ, λ)     (non-central, λ = 2·SNR·γ)
      where γ = T·W is the time-bandwidth product.

    * JAM mode — jams the detected channel for 1 ms, then re-scans
      the **same** channel in the next slot.

    Channel scan order: 0 → 1 → … → (num_channels-1) → 0 → …  (cyclic).

    Jamming signals are pre-generated once per channel and baseband variant.
    """

    def __init__(self, Fs, num_channels=20, sub_interval=50000.0,
                 detection_time=0.001, p_fa=0.1,
                 power=0.8, bandwidth=50000.0, Startfre=3e6,
                 noise_source=None, speed=None,
                 noise_std=0.1, signal_power=None, Baud=25000,
                 variant_pool=None, variant_selector=None):
        # ---- basic parameters ----
        self.Fs = float(Fs)
        self.num_channels = int(num_channels)
        self.sub_interval = float(sub_interval)
        self.detection_time = float(detection_time)
        self.p_fa = float(p_fa)
        self.power = float(power)
        self.bandwidth = float(bandwidth)
        self.Startfre = float(Startfre)
        self.noise_std = float(noise_std)
        # speed is kept for backward API compatibility; unused in new logic
        self.speed = float(speed) if speed is not None else float('inf')

        # ---- energy detection: time-bandwidth product & degrees of freedom ----
        self.TW = self.detection_time * self.sub_interval          # γ = T·W
        self.dof = 2.0 * self.TW                                   # 2γ

        # ---- signal power (real RF, per sample, PRE-fading) ----
        if signal_power is not None:
            self.signal_power = float(signal_power)
        else:
            # theoretical default: Baud / Fs
            self.signal_power = float(Baud) / self.Fs

        # ---- noise power in detection bandwidth W = Sub_interval ----
        # N0 (one-sided PSD)  = 2 · σ² / Fs
        # P_n = N0 · W = 2 · σ² · Sub_interval / Fs
        noise_power_in_band = 2.0 * self.noise_std**2 * self.sub_interval / self.Fs

        # ---- average SNR (PRE-fading) ----
        self.snr_avg_linear = self.signal_power / max(noise_power_in_band, 1e-30)
        self.snr_avg_dB = 10.0 * np.log10(max(self.snr_avg_linear, 1e-30))

        # ---- threshold V_T from false-alarm probability (central χ²) ----
        self.V_T = chi2.ppf(1.0 - self.p_fa, self.dof)

        # ---- detection probability P_D, averaged over Rayleigh fading ----
        # Instantaneous SNR: γ = SNR_avg · r²  where r² ~ Exp(1)
        # P_D_faded = ∫₀^∞ [1 − F_{ncχ²}(V_T | dof, 2·SNR_avg·x·TW)] · e⁻ˣ dx
        # Evaluated via Gauss–Laguerre quadrature.
        self.P_D = self._fading_averaged_P_D(n_laguerre=50)

        # ---- noise source & pre-generated jamming signals ----
        self.variant_pool = variant_pool
        self.variant_selector = variant_selector
        if self.variant_pool is not None:
            if not np.isclose(self.variant_pool.bandwidth, self.bandwidth):
                raise ValueError("Reactive jammer variant-pool bandwidth does not match.")
            self.noise_source = None
        elif noise_source is None:
            self.noise_source = FastNoiseSource(Fs, bandwidth)
        else:
            self.noise_source = noise_source

        self.samples_per_ms = max(1, int(self.Fs * self.detection_time))
        self._pregenerate_jam_signals()

        # ---- state machine ----
        self.reset()

    # ------------------------------------------------------------------
    def _fading_averaged_P_D(self, n_laguerre=50):
        """
        Compute P_D averaged over Rayleigh fading.

        Rayleigh magnitude r has E[r²] = 1 (scale = √2/2).
        r² ~ Exp(1), so the instantaneous SNR = SNR_avg · r² where r² ~ Exp(1).

        P_D_faded = ∫₀^∞ P_D(x · SNR_avg) · e⁻ˣ dx

        Evaluated with Gauss–Laguerre quadrature:
            ∫₀^∞ f(x)·e⁻ˣ dx ≈ Σ w_i · f(x_i)
        """
        nodes, weights = roots_laguerre(n_laguerre)
        p_d = 0.0
        for x_i, w_i in zip(nodes, weights):
            snr_inst = self.snr_avg_linear * x_i
            lam = 2.0 * snr_inst * self.TW
            p_d_inst = 1.0 - ncx2.cdf(self.V_T, self.dof, lam)
            p_d += w_i * p_d_inst
        return float(np.clip(p_d, 0.0, 1.0))

    # ------------------------------------------------------------------
    def _pregenerate_jam_signals(self):
        """
        Pre-generate one jamming segment per channel and baseband variant.

        Jamming signal = band-limited noise × cos(2π·f_c·t). Each cache entry
        has shape ``[num_variants, samples_per_slot]``.
        """
        self._jam_cache = {}
        t_1ms = np.arange(self.samples_per_ms, dtype=np.float64) / self.Fs
        num_variants = (
            self.variant_pool.num_variants
            if self.variant_pool is not None
            else 1
        )
        baseband_variants = np.empty(
            (num_variants, self.samples_per_ms),
            dtype=np.float32,
        )
        for variant_idx in range(num_variants):
            if self.variant_pool is not None:
                noise_1ms = self.variant_pool.get_variant(
                    variant_idx,
                    self.samples_per_ms,
                )
            else:
                noise_1ms = self.noise_source.get_noise(self.samples_per_ms)
            baseband_variants[variant_idx] = (
                np.asarray(noise_1ms, dtype=np.float32) * self.power
            )

        for k in range(self.num_channels):
            f_c = self.Startfre + k * self.sub_interval + 0.5 * self.sub_interval
            carrier = np.cos(2.0 * np.pi * f_c * t_1ms)
            self._jam_cache[k] = (
                baseband_variants * carrier[np.newaxis, :]
            ).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self):
        """Reset state machine: start scanning channel 0."""
        self.current_channel = 0
        self.state = 'scan'          # 'scan' | 'jam'

    # ------------------------------------------------------------------
    def _get_tx_channel(self, sample_pos, hop_seq, hoprate):
        """
        Return the transmitter's channel index at a given sample position.

        Uses the hopping sequence and per-hop duration to locate the
        active hop, then reads the channel from ``hop_seq``.
        """
        if len(hop_seq) == 0:
            return -1
        s_per_hop = self.Fs / float(hoprate)
        if s_per_hop <= 0.0:
            return int(hop_seq[0])
        hop_idx = int(sample_pos / s_per_hop)
        hop_idx = min(hop_idx, len(hop_seq) - 1)
        return int(hop_seq[hop_idx])

    # ------------------------------------------------------------------
    def _energy_detect(self, signal_present, rng=None):
        """
        Simulate one energy-detection trial.

        Parameters
        ----------
        signal_present : bool
            ``True`` when the transmitter is on the channel being scanned.

        Returns
        -------
        bool
            ``True`` if the energy detector reports a detection.
        """
        p = self.P_D if signal_present else self.p_fa
        return _rng_random(rng) < p

    # ------------------------------------------------------------------
    def generate_samples(
        self,
        num_samples,
        hop_seq,
        Startfre,
        Sub_interval,
        hoprate,
        rng=None,
    ):
        """
        Generate reactive jamming for one block.

        The block is divided into 1-ms slots.  In each slot the internal
        state machine decides whether to scan (and possibly transition to
        jam) or to jam (and then return to scan on the same channel).

        Returns
        -------
        jam : ndarray (float32, same length as *t*)
            Jamming waveform (zero where no jamming occurred).
        active : bool
            ``True`` if at least one 1-ms slot in this block was jammed.
        """
        N = int(num_samples)
        if N == 0:
            return np.zeros(0, dtype=np.float32), False

        jam = np.zeros(N, dtype=np.float32)
        any_jam = False

        pos = 0
        while pos < N:
            seg_end = min(pos + self.samples_per_ms, N)
            seg_len = seg_end - pos

            # --- transmitter channel during this 1-ms slot ---
            tx_channel = self._get_tx_channel(pos, hop_seq, hoprate)

            if self.state == 'scan':
                # energy detection on the current scanning channel
                signal_present = (tx_channel == self.current_channel)
                if self._energy_detect(signal_present, rng=rng):
                    # detected → jam the same channel next slot
                    self.state = 'jam'
                else:
                    # not detected → advance to next channel, keep scanning
                    self.current_channel = (self.current_channel + 1) % self.num_channels

            elif self.state == 'jam':
                # output pre-generated jamming signal for current channel
                ch = self.current_channel % self.num_channels
                num_variants = self._jam_cache[ch].shape[0]
                if self.variant_selector is not None and num_variants > 1:
                    variant_idx = self.variant_selector.draw()
                else:
                    variant_idx = 0
                cached = self._jam_cache[ch][variant_idx]
                if seg_len >= len(cached):
                    jam[pos:pos + len(cached)] = cached
                else:
                    jam[pos:seg_end] = cached[:seg_len]
                any_jam = True
                # after jamming, re-scan the same channel
                self.state = 'scan'

            pos = seg_end

        return jam, any_jam

    def generate(self, t, hop_seq, Startfre, Sub_interval, hoprate, rng=None):
        """Compatibility wrapper accepting the historical time vector."""
        return self.generate_samples(
            len(t),
            hop_seq,
            Startfre,
            Sub_interval,
            hoprate,
            rng=rng,
        )


class IndiscriminateJammer:
    COMB_TIME_QUANTUM = 0.001

    def __init__(self, Fs, sweep_config=None, comb_config=None, 
                 noise_source=None, mode='sweep',
                 sweep_variant_pool=None, comb_variant_pool=None,
                 variant_selector=None, defer_dynamic_noise=False):
        self.Fs = float(Fs)
        if mode not in {'sweep', 'comb', 'both'}:
            raise ValueError("mode must be 'sweep', 'comb', or 'both'.")
        self.mode = mode
        
        # Default Configs
        self.sweep_config = sweep_config if sweep_config else {}
        self.comb_config = comb_config if comb_config else {}
        
        # --- Sweep Params ---
        self.s_step = float(self.sweep_config.get('step', 125000.0))
        self.s_power = float(self.sweep_config.get('power', 0.8))
        self.s_dwell = float(self.sweep_config.get('dwell_time', 0.004))
        self.s_bw = float(self.sweep_config.get('bandwidth', 30000.0))
        
        # --- Comb Params ---
        self.c_step = float(self.comb_config.get('step', 100000.0))
        self.c_power = float(self.comb_config.get('power', 0.5))
        self.c_bw = float(self.comb_config.get('bandwidth', 30000.0))
        self.comb_switch_interval = float(
            self.comb_config.get('switch_interval', 0.05)
        )
        self._validate_comb_switch_interval()
        self.comb_switch_samples = int(round(self.comb_switch_interval * self.Fs))
        self.comb_period_samples = 2 * self.comb_switch_samples

        # Jammed channel indices for the two alternating comb groups.
        # Configurable via JAMMER_CONFIG["comb"]["channels_phase0/1"];
        # defaults preserve the legacy even/odd 8-channel groups.
        self.comb_channels = (
            list(self.comb_config.get('channels_phase0',
                                      [0, 2, 4, 6, 8, 10, 12, 14])),
            list(self.comb_config.get('channels_phase1',
                                      [1, 3, 5, 7, 9, 11, 13, 15])),
        )
        
        self.sweep_variant_pool = sweep_variant_pool
        self.comb_variant_pool = comb_variant_pool
        self.variant_selector = variant_selector

        # --- Dynamic Noise Sources ---
        if noise_source:
             self.ns_sweep = noise_source
             self.ns_comb = noise_source
        elif defer_dynamic_noise:
             self.ns_sweep = None
             self.ns_comb = None
        else:
             self.ns_sweep = FastNoiseSource(Fs, self.s_bw)
             # Reuse if bandwidth same
             if self.c_bw == self.s_bw:
                 self.ns_comb = self.ns_sweep
             else:
                 self.ns_comb = FastNoiseSource(Fs, self.c_bw)

        # Pre-computed buffers
        self.pre_buffer_sweep = None
        self.pre_buffer_comb0 = None
        self.pre_buffer_comb1 = None
        self.sweep_period_samples = 0

        # Deterministic RF carriers are independent of the fresh baseband
        # noise used by the dynamic path.  Cache one natural timing cycle and
        # reuse it for both pre-computation and run-time generation.
        self._carrier_cache_key = None
        self._carrier_sweep = None
        self._carrier_comb0 = None
        self._carrier_comb1 = None

        self._validate_variant_pools()

    def set_mode(self, mode):
        if mode not in {'sweep', 'comb', 'both'}:
            raise ValueError("mode must be 'sweep', 'comb', or 'both'.")
        self.mode = mode

    def _carrier_key(self, Startfre, Endfre):
        return (
            self.mode,
            float(Startfre),
            float(Endfre),
            self.s_step,
            self.s_dwell,
            self.comb_switch_samples,
            tuple(self.comb_channels[0]),
            tuple(self.comb_channels[1]),
        )

    def _ensure_carrier_cache(self, Startfre, Endfre):
        """Build deterministic sweep/comb carriers once per RF layout."""
        key = self._carrier_key(Startfre, Endfre)
        if key == self._carrier_cache_key:
            return

        self._carrier_sweep = None
        self._carrier_comb0 = None
        self._carrier_comb1 = None
        phase_k = 2.0 * np.pi / self.Fs

        if self.sweep_enabled:
            bw_total, samples_per_dwell, num_steps = self._sweep_layout(
                Startfre,
                Endfre,
            )
            sweep_carrier = np.empty(
                samples_per_dwell * num_steps,
                dtype=np.float32,
            )
            dwell_samples = np.arange(samples_per_dwell, dtype=np.float64)
            for sweep_idx in range(num_steps):
                f = Startfre + sweep_idx * self.s_step + self.s_step / 2.0
                if f >= Endfre:
                    f = Startfre + np.mod(f - Startfre, bw_total)
                start = sweep_idx * samples_per_dwell
                sweep_carrier[start:start + samples_per_dwell] = np.cos(
                    (phase_k * f) * dwell_samples
                ).astype(np.float32)
            self._carrier_sweep = sweep_carrier

        if self.comb_enabled:
            phase_samples = np.arange(
                self.comb_switch_samples,
                dtype=np.float64,
            )
            comb_carriers = []
            for phase in (0, 1):
                freqs = self._comb_frequencies(phase, Startfre, Endfre)
                carrier = np.zeros(
                    self.comb_switch_samples,
                    dtype=np.float64,
                )
                for f in freqs:
                    carrier += np.cos((phase_k * f) * phase_samples)
                if len(freqs) > 0:
                    carrier *= 1.0 / np.sqrt(len(freqs))
                comb_carriers.append(carrier.astype(np.float32))
            self._carrier_comb0, self._carrier_comb1 = comb_carriers

        self._carrier_cache_key = key

    def _validate_variant_pools(self):
        if (
            self.sweep_variant_pool is not None
            and not np.isclose(self.sweep_variant_pool.bandwidth, self.s_bw)
        ):
            raise ValueError("Sweep jammer variant-pool bandwidth does not match.")
        if (
            self.comb_variant_pool is not None
            and not np.isclose(self.comb_variant_pool.bandwidth, self.c_bw)
        ):
            raise ValueError("Comb jammer variant-pool bandwidth does not match.")

        counts = {
            pool.num_variants
            for pool in (self.sweep_variant_pool, self.comb_variant_pool)
            if pool is not None
        }
        if len(counts) > 1:
            raise ValueError("Sweep and comb variant pools must use the same count.")
        if counts and self.variant_selector is not None:
            count = next(iter(counts))
            if self.variant_selector.num_variants != count:
                raise ValueError("Jammer selector and baseband pool counts do not match.")

    def set_variant_sources(
        self,
        sweep_variant_pool=None,
        comb_variant_pool=None,
        variant_selector=None,
    ):
        self.sweep_variant_pool = sweep_variant_pool
        self.comb_variant_pool = comb_variant_pool
        self.variant_selector = variant_selector
        self._validate_variant_pools()

    def _dynamic_noise_source(self, jammer_kind):
        if jammer_kind == "sweep":
            if self.ns_sweep is None:
                self.ns_sweep = FastNoiseSource(self.Fs, self.s_bw)
                if self.c_bw == self.s_bw and self.ns_comb is None:
                    self.ns_comb = self.ns_sweep
            return self.ns_sweep
        if jammer_kind == "comb":
            if self.ns_comb is None:
                if self.c_bw == self.s_bw and self.ns_sweep is not None:
                    self.ns_comb = self.ns_sweep
                else:
                    self.ns_comb = FastNoiseSource(self.Fs, self.c_bw)
            return self.ns_comb
        raise ValueError(f"Unsupported jammer kind: {jammer_kind}")

    def _validate_comb_switch_interval(self):
        interval = self.comb_switch_interval
        interval_units = interval / self.COMB_TIME_QUANTUM
        if (
            not np.isfinite(interval)
            or interval <= 0.0
            or not np.isclose(
                interval_units,
                round(interval_units),
                rtol=0.0,
                atol=1e-9,
            )
        ):
            raise ValueError(
                "JAMMER_CONFIG['comb']['switch_interval'] must be a finite "
                "positive multiple of 0.001 seconds (1 ms)."
            )

    def _sweep_layout(self, Startfre, Endfre):
        if not np.isfinite(self.s_step) or self.s_step <= 0.0:
            raise ValueError("Sweep step must be finite and positive.")
        if not np.isfinite(self.s_dwell) or self.s_dwell <= 0.0:
            raise ValueError("Sweep dwell_time must be finite and positive.")

        bw_total = max(float(Endfre) - float(Startfre), 1.0)
        samples_per_dwell = max(1, int(round(self.s_dwell * self.Fs)))
        num_steps = max(1, int(np.floor(bw_total / self.s_step)))
        return bw_total, samples_per_dwell, num_steps

    @property
    def comb_enabled(self):
        return self.mode in {'comb', 'both'}

    @property
    def sweep_enabled(self):
        return self.mode in {'sweep', 'both'}

    def comb_phase_at(self, sample_idx):
        sample_idx = int(sample_idx)
        if sample_idx < 0:
            raise ValueError("sample_idx must be non-negative.")
        return (sample_idx // self.comb_switch_samples) % 2

    def reset(self):
        """Reset the jammer's continuous timeline to its deterministic origin."""
        # Sweep and comb positions are derived from the environment sample clock.
        # The method remains explicit so reset semantics stay centralized.
        return None

    @staticmethod
    def _as_variant_buffer(buffer):
        if buffer is None or np.size(buffer) == 0:
            raise RuntimeError("Requested jammer buffer has not been pre-computed.")
        buffer = np.asarray(buffer)
        if buffer.ndim == 1:
            return buffer[np.newaxis, :]
        if buffer.ndim != 2:
            raise ValueError("Pre-computed jammer buffers must be one- or two-dimensional.")
        return buffer

    def _variant_for_cycle(self, jammer_kind, cycle_idx, num_variants):
        if num_variants <= 1 or self.variant_selector is None:
            return 0
        return self.variant_selector.choice_for_cycle(jammer_kind, cycle_idx)

    def _sweep_buffer_slice(self, start_sample_idx, num_samples):
        buffers = self._as_variant_buffer(self.pre_buffer_sweep)
        period_samples = buffers.shape[1]
        result = np.empty(num_samples, dtype=np.float32)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            cycle_idx = global_idx // period_samples
            cycle_offset = global_idx % period_samples
            variant_idx = self._variant_for_cycle(
                "sweep",
                cycle_idx,
                buffers.shape[0],
            )
            take = min(period_samples - cycle_offset, num_samples - result_idx)
            result[result_idx:result_idx + take] = buffers[
                variant_idx,
                cycle_offset:cycle_offset + take,
            ]
            global_idx += take
            result_idx += take
        return result

    def _comb_buffer_slice(self, start_sample_idx, num_samples):
        buffers0 = self._as_variant_buffer(self.pre_buffer_comb0)
        buffers1 = self._as_variant_buffer(self.pre_buffer_comb1)
        if buffers0.shape != buffers1.shape:
            raise RuntimeError("Comb phase buffers must have matching shapes.")
        result = np.empty(num_samples, dtype=np.float32)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            cycle_idx = global_idx // self.comb_period_samples
            cycle_offset = global_idx % self.comb_period_samples
            phase = 0 if cycle_offset < self.comb_switch_samples else 1
            phase_offset = cycle_offset % self.comb_switch_samples
            variant_idx = self._variant_for_cycle(
                "comb",
                cycle_idx,
                buffers0.shape[0],
            )
            take = min(
                self.comb_switch_samples - phase_offset,
                num_samples - result_idx,
            )
            buffer = buffers0 if phase == 0 else buffers1
            result[result_idx:result_idx + take] = buffer[
                variant_idx,
                phase_offset:phase_offset + take,
            ]
            global_idx += take
            result_idx += take

        return result

    def precompute(self, Startfre=3e6, Endfre=4e6):
        """Pre-compute one natural-cycle RF buffer per baseband variant."""
        print(f"Pre-computing Jammers (Mode: {self.mode})...")
        self.pre_buffer_sweep = None
        self.pre_buffer_comb0 = None
        self.pre_buffer_comb1 = None
        self.sweep_period_samples = 0
        self._ensure_carrier_cache(Startfre, Endfre)

        if self.sweep_enabled:
            _, samples_per_dwell, num_steps = self._sweep_layout(Startfre, Endfre)
            self.sweep_period_samples = samples_per_dwell * num_steps
            num_variants = (
                self.sweep_variant_pool.num_variants
                if self.sweep_variant_pool is not None
                else 1
            )
            self.pre_buffer_sweep = np.empty(
                (num_variants, self.sweep_period_samples),
                dtype=np.float32,
            )
            for variant_idx in range(num_variants):
                baseband = (
                    self.sweep_variant_pool.get_variant(
                        variant_idx,
                        self.sweep_period_samples,
                    )
                    if self.sweep_variant_pool is not None
                    else self._dynamic_noise_source("sweep").get_noise(
                        self.sweep_period_samples
                    )
                )
                self.pre_buffer_sweep[variant_idx] = (
                    np.asarray(baseband, dtype=np.float32)
                    * np.float32(self.s_power)
                    * self._carrier_sweep
                )

        if self.comb_enabled:
            num_variants = (
                self.comb_variant_pool.num_variants
                if self.comb_variant_pool is not None
                else 1
            )
            self.pre_buffer_comb0 = np.empty(
                (num_variants, self.comb_switch_samples),
                dtype=np.float32,
            )
            self.pre_buffer_comb1 = np.empty_like(self.pre_buffer_comb0)
            for variant_idx in range(num_variants):
                baseband = (
                    self.comb_variant_pool.get_variant(
                        variant_idx,
                        self.comb_period_samples,
                    )
                    if self.comb_variant_pool is not None
                    else self._dynamic_noise_source("comb").get_noise(
                        self.comb_period_samples
                    )
                )
                baseband = (
                    np.asarray(baseband, dtype=np.float32)
                    * np.float32(self.c_power)
                )
                self.pre_buffer_comb0[variant_idx] = (
                    baseband[:self.comb_switch_samples]
                    * self._carrier_comb0
                )
                self.pre_buffer_comb1[variant_idx] = (
                    baseband[self.comb_switch_samples:]
                    * self._carrier_comb1
                )

        print("Jammer Pre-computation complete.")

    def precomputed_period_samples(self):
        """Return the LCM of timing cycles, not a numerical waveform period."""
        periods = []
        if self.sweep_enabled:
            if self.sweep_period_samples <= 0:
                raise RuntimeError("Sweep jammer has not been pre-computed.")
            periods.append(self.sweep_period_samples)
        if self.comb_enabled:
            periods.append(self.comb_period_samples)
        if not periods:
            return 1
        return math.lcm(*periods)

    def get_composite_signal(self, start_sample_idx, num_samples):
        """
        Retrieve a continuous-time slice of the active pre-computed jammers.
        """
        start_sample_idx = int(start_sample_idx)
        num_samples = int(num_samples)
        if start_sample_idx < 0:
            raise ValueError("start_sample_idx must be non-negative.")
        if num_samples < 0:
            raise ValueError("num_samples must be non-negative.")
        jam_total = np.zeros(num_samples, dtype=np.float32)

        if self.sweep_enabled:
            jam_total += self._sweep_buffer_slice(
                start_sample_idx,
                num_samples,
            )
        if self.comb_enabled:
            jam_total += self._comb_buffer_slice(start_sample_idx, num_samples)

        return jam_total

    def _comb_frequencies(self, phase, Startfre, Endfre):
        sub_interval = 50000.0
        target_indices = np.asarray(self.comb_channels[phase])
        freqs = Startfre + target_indices * sub_interval + 0.5 * sub_interval
        return freqs[(freqs >= Startfre) & (freqs < Endfre)]

    def _generate_comb_signal(
        self,
        num_samples,
        start_sample_idx,
        Startfre,
        Endfre,
        fixed_phase=None,
        baseband_noise=None,
        rng=None,
    ):
        num_samples = int(num_samples)
        jam = np.zeros(num_samples, dtype=np.float32)
        freqs_used = []
        if num_samples == 0:
            return jam, freqs_used

        self._ensure_carrier_cache(Startfre, Endfre)
        if baseband_noise is None:
            baseband_noise = _noise_source_slice(
                self._dynamic_noise_source("comb"),
                num_samples,
                rng=rng,
            )
        baseband_noise = np.asarray(baseband_noise, dtype=np.float32)
        if len(baseband_noise) != num_samples:
            raise ValueError("Comb baseband noise length does not match num_samples.")
        baseband_noise = baseband_noise * np.float32(self.c_power)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            phase_offset = global_idx % self.comb_switch_samples
            phase = (
                int(fixed_phase)
                if fixed_phase is not None
                else self.comb_phase_at(global_idx)
            )
            take = min(
                self.comb_switch_samples - phase_offset,
                num_samples - result_idx,
            )
            freqs = self._comb_frequencies(phase, Startfre, Endfre)
            if len(freqs) > 0:
                combined_carrier = (
                    self._carrier_comb0 if phase == 0 else self._carrier_comb1
                )[phase_offset:phase_offset + take]
                freqs_used.extend(freqs.tolist())
                jam[result_idx:result_idx + take] = (
                    baseband_noise[result_idx:result_idx + take] * combined_carrier
                )

            global_idx += take
            result_idx += take

        return jam, freqs_used

    def _generate_sweep_signal(
        self,
        num_samples,
        start_sample_idx,
        Startfre,
        Endfre,
        baseband_noise=None,
        rng=None,
    ):
        num_samples = int(num_samples)
        jam = np.zeros(num_samples, dtype=np.float32)
        freqs_used = []
        if num_samples == 0:
            return jam, freqs_used

        self._ensure_carrier_cache(Startfre, Endfre)
        bw_total, samples_per_dwell, num_steps = self._sweep_layout(
            Startfre,
            Endfre,
        )
        if baseband_noise is None:
            baseband_noise = _noise_source_slice(
                self._dynamic_noise_source("sweep"),
                num_samples,
                rng=rng,
            )
        baseband_noise = np.asarray(baseband_noise, dtype=np.float32)
        if len(baseband_noise) != num_samples:
            raise ValueError("Sweep baseband noise length does not match num_samples.")
        baseband_noise = baseband_noise * np.float32(self.s_power)
        global_idx = int(start_sample_idx)
        result_idx = 0

        while result_idx < num_samples:
            dwell_offset = global_idx % samples_per_dwell
            sweep_idx = (global_idx // samples_per_dwell) % num_steps
            take = min(
                samples_per_dwell - dwell_offset,
                num_samples - result_idx,
            )
            f = Startfre + sweep_idx * self.s_step + self.s_step / 2.0
            if f >= Endfre:
                f = Startfre + np.mod(f - Startfre, bw_total)
            carrier_start = sweep_idx * samples_per_dwell + dwell_offset
            carrier = self._carrier_sweep[
                carrier_start:carrier_start + take
            ]
            jam[result_idx:result_idx + take] = (
                baseband_noise[result_idx:result_idx + take] * carrier
            )
            freqs_used.append(f)
            global_idx += take
            result_idx += take

        return jam, freqs_used

    def generate_samples(
        self,
        num_samples,
        Startfre,
        Endfre,
        start_sample_idx=0,
        rng=None,
    ):
        """Generate a dynamic slice using cached carriers and fresh noise."""
        N = int(num_samples)
        if N == 0:
            return np.zeros(0, dtype=np.float32), []

        jam = np.zeros(N, dtype=np.float32)
        freqs_used = []

        # Preserve the historical draw/generation order for ``both`` mode:
        # comb first, then sweep, with two independent draws from *rng*.
        if self.comb_enabled:
            comb_jam, comb_freqs = self._generate_comb_signal(
                N,
                start_sample_idx,
                Startfre,
                Endfre,
                rng=rng,
            )
            jam += comb_jam
            freqs_used.extend(comb_freqs)

        if self.sweep_enabled:
            sweep_jam, sweep_freqs = self._generate_sweep_signal(
                N,
                start_sample_idx,
                Startfre,
                Endfre,
                rng=rng,
            )
            jam += sweep_jam
            freqs_used.extend(sweep_freqs)

        return jam, freqs_used

    def generate(self, t, Startfre, Endfre, start_sample_idx=0, rng=None):
        """Compatibility wrapper accepting the historical time vector."""
        return self.generate_samples(
            len(t),
            Startfre,
            Endfre,
            start_sample_idx=start_sample_idx,
            rng=rng,
        )
