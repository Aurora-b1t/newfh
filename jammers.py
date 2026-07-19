"""
Jammer implementations for FHSS anti-jamming simulation.

Includes a pre-generated band-limited noise source, an energy-detection based
reactive jammer, and an indiscriminate sweep/comb jammer with optional
pre-computed signal buffers.
"""

import numpy as np
from scipy.stats import chi2, ncx2
from scipy.special import roots_laguerre

# -----------------------------
# 快速噪声源
# -----------------------------
class FastNoiseSource:
    """
    预生成长段带限噪声的缓冲区，避免反复计算 FFT/IFFT。
    """
    def __init__(self, Fs, bandwidth, duration=1.0):
        self.length = int(Fs * duration)
        # 1. Generate Gaussian noise
        n = np.random.randn(self.length).astype(np.float32)
        # 2. FFT
        spec = np.fft.rfft(n)
        # 3. Low-pass
        if bandwidth > 0:
            cutoff_idx = int(np.floor(bandwidth * self.length / (2.0 * Fs)))
            if cutoff_idx + 1 < len(spec):
                spec[cutoff_idx + 1:] = 0
                
        # 4. IFFT
        n_lp = np.fft.irfft(spec, n=self.length)
        # 5. Normalize
        std_val = np.std(n_lp)
        if std_val > 1e-12:
            n_lp /= std_val
        self.noise = n_lp

    def get_noise(self, num_samples):
        if num_samples >= self.length:
            # 这种情况下直接返回全部并循环填充
            tile_count = (num_samples // self.length) + 1
            return np.tile(self.noise, tile_count)[:num_samples]
            
        start = np.random.randint(0, self.length - num_samples)
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

    Jamming signals are pre-generated once per channel (20 possibilities).
    """

    def __init__(self, Fs, num_channels=20, sub_interval=50000.0,
                 detection_time=0.001, p_fa=0.1,
                 power=0.8, bandwidth=50000.0, Startfre=3e6,
                 noise_source=None, speed=None,
                 noise_std=0.1, signal_power=None, Baud=25000):
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
        if noise_source is None:
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
        Pre-generate one 1-ms jamming segment per channel.

        Jamming signal = band-limited noise × cos(2π·f_c·t).
        Stored as float32 arrays keyed by channel index.
        """
        self._jam_cache = {}
        t_1ms = np.arange(self.samples_per_ms, dtype=np.float64) / self.Fs
        noise_1ms = self.noise_source.get_noise(self.samples_per_ms) * self.power

        for k in range(self.num_channels):
            f_c = self.Startfre + k * self.sub_interval + 0.5 * self.sub_interval
            carrier = np.cos(2.0 * np.pi * f_c * t_1ms)
            self._jam_cache[k] = (noise_1ms * carrier).astype(np.float32)

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
    def _energy_detect(self, signal_present):
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
        return np.random.random() < p

    # ------------------------------------------------------------------
    def generate(self, t, hop_seq, Startfre, Sub_interval, hoprate):
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
        N = len(t)
        if N == 0:
            return np.zeros_like(t, dtype=np.float32), False

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
                if self._energy_detect(signal_present):
                    # detected → jam the same channel next slot
                    self.state = 'jam'
                else:
                    # not detected → advance to next channel, keep scanning
                    self.current_channel = (self.current_channel + 1) % self.num_channels

            elif self.state == 'jam':
                # output pre-generated jamming signal for current channel
                ch = self.current_channel % self.num_channels
                cached = self._jam_cache[ch]
                if seg_len >= len(cached):
                    jam[pos:pos + len(cached)] = cached
                else:
                    jam[pos:seg_end] = cached[:seg_len]
                any_jam = True
                # after jamming, re-scan the same channel
                self.state = 'scan'

            pos = seg_end

        return jam, any_jam


class IndiscriminateJammer:
    def __init__(self, Fs, sweep_config=None, comb_config=None, 
                 noise_source=None, mode='sweep'):
        self.Fs = float(Fs)
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
        
        self._sweep_idx = 0
        
        # --- Noise Sources ---
        if noise_source:
             self.ns_sweep = noise_source
             self.ns_comb = noise_source
        else:
             self.ns_sweep = FastNoiseSource(Fs, self.s_bw)
             # Reuse if bandwidth same
             if self.c_bw == self.s_bw:
                 self.ns_comb = self.ns_sweep
             else:
                 self.ns_comb = FastNoiseSource(Fs, self.c_bw)

        # State for changing Comb frequencies
        self.comb_phase = 0  # 0 or 1
        
        # Pre-computed buffers
        self.pre_buffer_sweep = None
        self.pre_buffer_comb0 = None
        self.pre_buffer_comb1 = None
        self.pre_len = 0
        self.pre_ptr = 0 # Not used internally, caller must manage or we use internal ptr?
                         # Usually caller manages because reset() logic is in Env.

    def set_mode(self, mode):
        if mode in ['sweep', 'comb', 'both']:
            self.mode = mode

    def step_comb(self):
        """Toggle the comb jamming frequency group."""
        self.comb_phase = 1 - self.comb_phase

    def reset_comb(self):
        self.comb_phase = 0

    def precompute(self, duration=4.4, Startfre=3e6, Endfre=4e6):
        """
        Pre-compute sweep and comb jamming signals for a fixed duration.
        """
        print(f"Pre-computing Jammers (Duration: {duration}s)...")
        num_samples = int(duration * self.Fs)
        self.pre_len = num_samples
        t = np.arange(num_samples) / self.Fs # Absolute time for continuity?
        # Actually generate() uses t just for length usually, BUT carrier phase
        # continuity across calls requires care.
        # Here we generate one long coherent block.
        
        # 1. Sweep Buffer
        # Temporarily save mode/state
        old_mode = self.mode
        old_idx = self._sweep_idx
        
        self.mode = 'sweep'
        self._sweep_idx = 0 # Start from beginning of sweep cycle
        # Note: generate() expects t relative to 0 for carrier phase if we call it naively.
        # But we want carrier phase continuous across the whole duration.
        # My current generate() implementation uses t_seg_local = np.arange(length) inside the loop,
        # resets phase at every dwell segment? No.
        # `carrier = np.cos((phase_k * f) * t_seg_local)`
        # Yes, it resets phase at every segment boundary in `generate`. 
        # But since we call generate ONCE for the whole duration, it will be consistent within that duration.
        
        # However, generate() chunks by dwell time. 
        # Inside generate loop: `t_seg_local` starts at 0 for each dwell segment.
        # This implies phase discontinuity at frequency hops. This is physically acceptable for a jammer (oscillator retuning).
        
        jam_s, _ = self.generate(t, Startfre, Endfre)
        self.pre_buffer_sweep = jam_s.astype(np.float32)
        
        # 2. Comb Buffer 0
        self.mode = 'comb'
        self.comb_phase = 0
        jam_c0, _ = self.generate(t, Startfre, Endfre)
        self.pre_buffer_comb0 = jam_c0.astype(np.float32)
        
        # 3. Comb Buffer 1
        self.mode = 'comb'
        self.comb_phase = 1
        jam_c1, _ = self.generate(t, Startfre, Endfre)
        self.pre_buffer_comb1 = jam_c1.astype(np.float32)
        
        # Restore state
        self.mode = old_mode
        self._sweep_idx = old_idx
        print("Jammer Pre-computation complete.")

    def get_composite_signal(self, start_sample_idx, num_samples):
        """
        Retrieve a slice of the pre-computed jamming signal based on current mode and comb phase.
        Auto-wraps if exceeding buffer duration.
        """
        if self.pre_buffer_sweep is None:
            # Fallback (should not happen if precompute called)
            return np.zeros(num_samples)
            
        # Determine slices (handling wrap-around)
        idx_start = start_sample_idx % self.pre_len
        idx_end = idx_start + num_samples
        
        # Helper to get slice from a buffer
        def get_slice(buf):
            if idx_end <= self.pre_len:
                return buf[idx_start:idx_end]
            else:
                # Wrap around
                part1 = buf[idx_start:]
                rem = idx_end - self.pre_len
                part2 = buf[:rem]
                return np.concatenate([part1, part2])

        jam_total = np.zeros(num_samples, dtype=np.float32)
        
        if self.mode == 'sweep' or self.mode == 'both':
            jam_total += get_slice(self.pre_buffer_sweep)
            
        if self.mode == 'comb' or self.mode == 'both':
            if self.comb_phase == 0:
                jam_total += get_slice(self.pre_buffer_comb0)
            else:
                jam_total += get_slice(self.pre_buffer_comb1)
                
        return jam_total

    def generate(self, t, Startfre, Endfre):
        N = len(t)
        if N == 0:
            return np.zeros_like(t), []
            
        jam = np.zeros_like(t)
        freqs_used = []
        phase_k = 2 * np.pi / self.Fs
        bw_total = max(Endfre - Startfre, 1.0)

        # -----------------------------
        # Comb Jamming
        # -----------------------------
        if self.mode == 'comb' or self.mode == 'both':
            # Use Comb Noise Source
            raw_noise_c = self.ns_comb.get_noise(N)
            baseband_noise_c = raw_noise_c * self.c_power
            
            # --- Optimized for 8 fixed points aligned to 50kHz channels ---
            sub_interval = 50000.0
            
            # Hardcoded selection for two alternating 8-channel comb groups.
            if self.comb_phase == 0:
                # Group 0: even channel indices
                target_indices = np.array([0, 2, 4, 6, 8, 10, 12, 14])
            else:
                # Group 1: odd channel indices
                target_indices = np.array([1, 3, 5, 7, 9, 11, 13, 15])
            
            # Map to frequencies: Start + k*sub + 0.5*sub
            freqs = Startfre + target_indices * sub_interval + 0.5 * sub_interval
            freqs = freqs[(freqs >= Startfre) & (freqs < Endfre)]
            
            if len(freqs) > 0:
                combined_carrier = np.zeros(N)
                t_local = np.arange(N)
                
                # Normalize power so total power remains roughly consistent or per-tone?
                # Usually comb power is defined per tone or total? 
                # Code originally: norm_factor = 1.0 / sqrt(len). 
                # This keeps TOTAL power = baseband power.
                norm_factor = 1.0 / np.sqrt(len(freqs))
                
                for f in freqs:
                    combined_carrier += np.cos((phase_k * f) * t_local)
                    freqs_used.append(f)
                
                combined_carrier *= norm_factor
                jam += baseband_noise_c * combined_carrier

        # -----------------------------
        # Sweep Jamming
        # -----------------------------
        if self.mode == 'sweep' or self.mode == 'both':
            # Use Sweep Noise Source
            raw_noise_s = self.ns_sweep.get_noise(N)
            baseband_noise_s = raw_noise_s * self.s_power
            
            # Sweep Params
            step = self.s_step
            dwell = self.s_dwell
            
            samples_per_dwell = max(1, int(round(dwell * self.Fs)))
            num_segments = int(np.ceil(len(t) / samples_per_dwell))
            num_steps = max(1, int(np.floor(bw_total / step)))
            
            idx = self._sweep_idx
            
            for seg in range(num_segments):
                f = Startfre + (idx % num_steps) * step + step / 2.0
                if f >= Endfre:
                    f = Startfre + np.mod(f - Startfre, bw_total)
                
                s = seg * samples_per_dwell
                e = min(len(t), (seg + 1) * samples_per_dwell)
                length = e - s
                
                t_seg_local = np.arange(length)
                carrier = np.cos((phase_k * f) * t_seg_local)
                jam[s:e] += baseband_noise_s[s:e] * carrier
                
                freqs_used.append(f)
                idx += 1
            self._sweep_idx = idx

        return jam, freqs_used
