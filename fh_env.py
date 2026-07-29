"""
FHSS/QPSK anti-jamming Gymnasium environment.

Provides the communication simulation chain: QPSK modulation/demodulation,
FHSS channel with Rayleigh fading, pre-generated data acceleration, PSD
waterfall observations, and support for reactive energy-detection and
indiscriminate sweep/comb jammers. The environment exposes a 10-offset
sequential decision interface for RL agents.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from commpy.filters import rrcosfilter
from scipy.signal import upfirdn
import gymnasium as gym
from gymnasium import spaces
import time
from jammers import (
    BandLimitedNoiseVariantPool,
    IndiscriminateJammer,
    JammerVariantSelector,
    ReactiveJammer,
    validate_baseband_variant_count,
)
import settings


NUM_BLOCKS = 10


def compute_block_rewards(ber_blocks, hoprate, reward_config=None):
    """Compute one configured reward for each FHSS block."""
    reward_config = reward_config or settings.REWARD_CONFIG
    ber_values = np.asarray(ber_blocks, dtype=np.float64)
    if ber_values.shape != (NUM_BLOCKS,):
        raise ValueError(
            f"ber_blocks must have shape ({NUM_BLOCKS},), got {ber_values.shape}."
        )
    if not np.all(np.isfinite(ber_values)) or not np.isfinite(hoprate):
        raise ValueError("BER values and hoprate must be finite.")

    base_reward = float(reward_config["base_reward"])
    ber_penalty = float(reward_config["ber_penalty"])
    hoprate_penalty = float(reward_config["hoprate_penalty"])
    return (
        base_reward
        - ber_penalty * ber_values
        - hoprate_penalty * float(hoprate)
    ).astype(np.float32)

# -----------------------------
# 基础函数
# -----------------------------
def rcosdesign_srv(rolloff, span, sps):
    """
    Root raised cosine filter normalized to unit energy.
    """
    # Ensure odd length (span*sps + 1) so that group delay is integer and aligned with sps
    rrc_filter = rrcosfilter(span * sps + 1, rolloff, 1, sps)[1]
    rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter ** 2) + 1e-12)
    return rrc_filter


def compute_psd_waterfall(signal, fs, f_start, f_end,
                          dt=0.001,      # 1 ms time resolution
                          df=10000.0,    # 10 kHz frequency resolution
                          max_duration=0.1,  # analyze first 100 ms
                          window='hann',
                          plot=False,
                          plot_title=""):
    """
    Optimized Compute PSD waterfall.
    """
    # ---- trim to first max_duration seconds ----
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * fs)
        signal = signal[:max_samples]

    Nwin = int(dt * fs)
    if Nwin <= 0:
        raise ValueError("Nwin must be positive. Check dt and fs.")
    if len(signal) < Nwin:
        return np.zeros((0, max(1, int(np.floor((f_end - f_start) / df)))) )

    step = Nwin

    if window == 'hann':
        win = np.hanning(Nwin)
    elif window == 'hamming':
        win = np.hamming(Nwin)
    else:
        win = np.ones(Nwin)
        
    win_sq_sum = np.sum(win ** 2)

    Nfft = int(2 ** np.ceil(np.log2(Nwin)))
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / fs)

    n_bins = int(np.floor((f_end - f_start) / df))
    n_bins = max(1, n_bins)
    f_bin_edges = f_start + np.arange(n_bins + 1) * df

    # Vectorized bin search
    # Find insertion points of edges in freqs
    # freqs is sorted.
    edge_indices = np.searchsorted(freqs, f_bin_edges)
    
    valid_bins = []
    for k in range(n_bins):
        start_idx = edge_indices[k]
        end_idx = edge_indices[k+1]
        if end_idx > start_idx:
            valid_bins.append((k, start_idx, end_idx))

    waterfall = []
    num_frames = (len(signal) - Nwin) // step + 1
    
    # Pre-calculate constants
    norm_factor = 1.0 / (fs * win_sq_sum / Nwin)

    for i in range(num_frames):
        start = i * step
        frame = signal[start:start + Nwin]
        if len(frame) < Nwin:
            break

        frame_win = frame * win
        spec = np.fft.rfft(frame_win, n=Nfft)
        # Compute PSD directly
        psd = (np.abs(spec) ** 2) * norm_factor

        band_powers = np.zeros(n_bins)
        
        # Optimize loop using slicing
        for k, s_idx, e_idx in valid_bins:
            # sum is much faster on slice than np.where
            band_powers[k] = np.sum(psd[s_idx:e_idx])

        waterfall.append(band_powers)

    waterfall = np.array(waterfall)  # [time, freq_bin]
    eps = 1e-12
    waterfall_db = 10 * np.log10(waterfall + eps)

    if plot and waterfall_db.size > 0:
        plt.figure(figsize=(8, 4))
        plt.imshow(waterfall_db.T, origin="lower", aspect="auto", cmap="jet")
        plt.colorbar(label='PSD (dB)')
        plt.xlabel('Time bin')
        plt.ylabel('Freq bin')
        title_str = 'PSD Waterfall ({:.0f} ms)'.format(max_duration * 1e3)
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    return waterfall_db


def save_waterfall_figure(waterfall_db, path, title=""):
    """
    Save a PSD waterfall array as a PNG figure.

    Uses the same imshow style as the debug plots in this module
    (jet colormap, lower origin, colorbar), but writes to *path*
    instead of opening an interactive window.
    """
    waterfall_db = np.asarray(waterfall_db)
    fig = plt.figure(figsize=(8, 4))
    plt.imshow(waterfall_db.T, origin="lower", aspect="auto", cmap="jet")
    plt.colorbar(label='PSD (dB)')
    plt.xlabel('Time bin')
    plt.ylabel('Freq bin')
    if title:
        plt.title(title)
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# -----------------------------
# M序列（LFSR）生成
# -----------------------------
def generate_mseq_states(n_bits=10, length=1000, taps=(10, 7), seed=1):
    if seed == 0:
        seed = 1
    mask = (1 << n_bits) - 1
    state = seed & mask
    seq = []
    for _ in range(length):
        seq.append(state)
        fb = 0
        for t in taps:
            fb ^= (state >> (t - 1)) & 1
        state = ((state << 1) & mask) | fb
        if state == 0:
            state = 1
    return np.array(seq, dtype=np.int64)





# -----------------------------
# 预生成数据管理器
# -----------------------------
class PreGeneratedData:
    def __init__(self, env, dtype=np.float32):
        print("Initializing PreGeneratedData (Reusable QPSK Baseband Mode)...")
        self.env = env
        self.dtype = dtype
        self.num_syms_block = int(round(env.Baud * 0.1))

        # 1. Baseband Bits (Ground Truth for BER) - Shared for all channels
        # Note: We use the same bits for all channels in this prototype optimization
        self.bits_len_per_block = 2 * self.num_syms_block
        self.common_bits = env.modem.generate_bits(self.bits_len_per_block).astype(np.int8)
        
        # 2. Baseband I/Q - Shared
        # Generate baseband signal once
        self.common_I, self.common_Q = env.modem.pulse_shape(self.common_bits)
        self.common_I = self.common_I.astype(dtype)
        self.common_Q = self.common_Q.astype(dtype)
        self.block_len = len(self.common_I)
        print(f"Reusable QPSK block length: {self.block_len}")
        print("Pre-generation complete.")

    def get_block(self, hop_seq):
        """Assemble one block from reusable I/Q and fresh channel randomness."""
        rx_assembled, carrier_assembled = self.env._assemble_signal_block(
            self.common_I,
            self.common_Q,
            hop_seq,
        )
        return rx_assembled, carrier_assembled, self.common_bits


# -----------------------------
# 调制与信道
# -----------------------------
class QPSKModem:
    def __init__(self, Baud, Fs, Ns, Nh):
        self.Baud = Baud
        self.Fs = Fs
        self.Ns = Ns
        self.Nh = Nh
        self.LBF = rcosdesign_srv(0.5, 16, Ns)

    def generate_bits(self, Bitrate):
        return np.random.binomial(n=1, p=0.5, size=Bitrate)

    def pulse_shape(self, bits):
        bits = np.asarray(bits).astype(np.int8)
        if len(bits) % 2 != 0:
            bits = np.concatenate([bits, np.array([0], dtype=np.int8)])

        I_bits = 2 * bits[0::2] - 1
        Q_bits = 2 * bits[1::2] - 1
        n_syms = len(I_bits)

        I_f = upfirdn(self.LBF, I_bits, up=self.Ns)
        Q_f = upfirdn(self.LBF, Q_bits, up=self.Ns)
        gd = (len(self.LBF) - 1) // 2
        I_pulse = I_f[gd:gd + n_syms * self.Ns]
        Q_pulse = Q_f[gd:gd + n_syms * self.Ns]
        return I_pulse, Q_pulse


class FHSSChannel:
    def __init__(self, Startfre, Sub_interval, Hoprate, Fs):
        self.Startfre = Startfre
        self.Sub_interval = Sub_interval
        self.Hoprate = Hoprate
        self.Fs = Fs

    def hop_carrier(self, t, hop_seq):
        """
        Return complex carrier directly: exp(j*phi)
        """
        N = len(t)
        if N == 0:
            return np.zeros(0, dtype=complex)

        s_per_hop = self.Fs / float(self.Hoprate)
        hop_idx = np.empty(N, dtype=int)
        pos = 0
        k = 0
        while pos < N and k < len(hop_seq):
            next_pos = int(round((k + 1) * s_per_hop))
            next_pos = min(N, max(pos + 1, next_pos))
            hop_idx[pos:next_pos] = hop_seq[k]
            pos = next_pos
            k += 1
        if pos < N:
            last_idx = max(0, min(k - 1, len(hop_seq) - 1))
            hop_idx[pos:] = hop_seq[last_idx]

        hop_fre = self.Startfre + hop_idx * self.Sub_interval + 0.5 * self.Sub_interval
        phase = 2 * np.pi * np.cumsum(hop_fre) / self.Fs
        # Optimized: calculate complex exponential directly
        carrier_complex = np.exp(1j * phase)
        return carrier_complex

    def transmit(self, I_pulse, Q_pulse, carrier_complex, noise_std=0.1):
        # carrier_complex is exp(j*phi)
        baseband = I_pulse + 1j * Q_pulse
        rf_complex = baseband * carrier_complex
        noise = noise_std * np.random.randn(len(rf_complex))
        return rf_complex, noise


class QPSKReceiver:
    def __init__(self, Ns):
        self.Ns = Ns
        self.MF = rcosdesign_srv(0.5, 16, Ns)

    def demodulate(self, modu_signal, bits, carrier_complex):
        # demodulate multiplying by conj(carrier)
        demod_complex = modu_signal * np.conj(carrier_complex)

        y_complex = upfirdn(self.MF, demod_complex, up=1, down=self.Ns)

        gd = (len(self.MF) - 1) // 2
        start_idx = gd // self.Ns
        
        y_complex = y_complex[start_idx:]

        y_i_end = np.real(y_complex)
        y_q_end = np.imag(y_complex)

        y_i_end = (y_i_end >= 0).astype(int)
        y_q_end = (y_q_end >= 0).astype(int)

        num_syms = len(bits) // 2
        min_len = min(len(y_i_end), len(y_q_end), num_syms)
        y_i_end = y_i_end[:min_len]
        y_q_end = y_q_end[:min_len]

        receive_data = np.zeros(2 * min_len, dtype=int)
        receive_data[::2] = y_i_end
        receive_data[1::2] = y_q_end

        bit_error = np.mean(receive_data != bits[:2 * min_len]) if min_len > 0 else 0.0
        return receive_data, bit_error





# -----------------------------
# Gym 环境封装
# -----------------------------
class FHSSQPSKEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self,
                 Startfre=3e6,
                 Endfre=4e6,
                 Fs=1e7,
                 Sub_interval=50000,
                 Hoprate=100,
                 hoprate_min=10.0,
                 hoprate_max=1000.0,
                 Baud=25000,
                 dt=0.001,
                 df=10000.0,
                 enable_reactive=False,
                 reactive_speed=160.0,
                 reactive_power=0.5,
                 reactive_bandwidth=50000.0,
                 enable_sweep=True,
                 sweep_step=125000,
                 sweep_power=0.8,
                 sweep_dwell=0.004,
                 sweep_bandwidth=30000.0,
                 sweep_mode='comb',
                 enable_rayleigh=False,
                 rayleigh_coherence=800,
                 mseq_length=1023,
                 mseq_nbits=10,
                 mseq_taps=(10, 7),
                 mseq_seed=46,
                 debug_plot_psd=False,
                 debug_log_hops=False,
                 reset_mseq_each_step=True,
                 use_pregen=True,
                 noise_std=0.1,
                 signal_power=0.0025):
        super().__init__()

        self.Startfre = float(Startfre)
        self.Endfre = float(Endfre)
        self.Fs = int(Fs)
        self.Sub_interval = float(Sub_interval)
        self.Baud = int(Baud)
        self.Tb = 1.0 / self.Baud
        self.Ns = int(self.Fs / self.Baud)
        self.dt = float(dt)
        self.df = float(df)

        self.num_channels = int(round((self.Endfre - self.Startfre) / self.Sub_interval))
        self.num_channels = max(1, self.num_channels)
        self.num_blocks = NUM_BLOCKS

        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max) if hoprate_max is not None else float(Baud)
        self.base_hoprate = float(Hoprate)

        self.Nh = 200 if (200 % 2 == 0) else 202
        self.current_hoprate = float(Hoprate)
        self.modem = QPSKModem(self.Baud, self.Fs, self.Ns, self.Nh)
        self.channel = FHSSChannel(self.Startfre, self.Sub_interval, self.current_hoprate, self.Fs)
        self.receiver = QPSKReceiver(self.Ns)
        
        self.enable_reactive = bool(enable_reactive)
        self.enable_sweep = bool(enable_sweep)
        self.use_pregen = bool(use_pregen)

        # 加载配置
        j_conf = settings.JAMMER_CONFIG
        self.noise_std = float(noise_std)
        self.signal_power = float(signal_power)

        self.baseband_variant_count = validate_baseband_variant_count(
            j_conf.get('baseband_variant_count', 4)
        )
        (
            self.jammer_variant_pools,
            self.jammer_variant_selector,
        ) = self._build_jammer_variant_resources(j_conf)

        if self.enable_reactive:
            r_conf = j_conf['reactive']
            self.reactive = ReactiveJammer(Fs=self.Fs,
                                           num_channels=self.num_channels,
                                           sub_interval=self.Sub_interval,
                                           detection_time=r_conf.get('detection_time', 0.001),
                                           p_fa=r_conf.get('p_fa', 0.1),
                                           power=r_conf['power'],
                                           bandwidth=r_conf['bandwidth'],
                                           Startfre=self.Startfre,
                                           noise_source=None,
                                           speed=r_conf.get('speed', None),
                                           noise_std=self.noise_std,
                                           signal_power=self.signal_power,
                                           Baud=self.Baud,
                                           variant_pool=self.jammer_variant_pools[
                                               float(r_conf['bandwidth'])
                                           ],
                                           variant_selector=self.jammer_variant_selector)
        else:
            self.reactive = None
                                       
        if self.enable_sweep:
            s_mode = j_conf.get('mode', 'sweep')
            self._validate_comb_channels(j_conf.get('comb', {}))
            self.sweep = IndiscriminateJammer(Fs=self.Fs,
                                              sweep_config=j_conf['sweep'],
                                              comb_config=j_conf['comb'],
                                              noise_source=None,
                                              mode=s_mode,
                                              sweep_variant_pool=(
                                                  self.jammer_variant_pools.get(
                                                      float(j_conf['sweep']['bandwidth'])
                                                  )
                                                  if self.use_pregen and s_mode in {'sweep', 'both'}
                                                  else None
                                              ),
                                              comb_variant_pool=(
                                                  self.jammer_variant_pools.get(
                                                      float(j_conf['comb']['bandwidth'])
                                                  )
                                                  if self.use_pregen and s_mode in {'comb', 'both'}
                                                  else None
                                              ),
                                              variant_selector=(
                                                  self.jammer_variant_selector
                                                  if self.use_pregen
                                                  else None
                                              ),
                                              defer_dynamic_noise=self.use_pregen)
        else:
            self.sweep = None

        self.enable_rayleigh = bool(enable_rayleigh)
        self.rayleigh_coherence = float(rayleigh_coherence)

        self.mseq_states = generate_mseq_states(n_bits=mseq_nbits,
                                                length=mseq_length,
                                                taps=mseq_taps,
                                                seed=mseq_seed)
        self.mseq_channels = (self.mseq_states % self.num_channels).astype(int)
        self._mseq_ptr = 0
        self.reset_mseq_each_step = bool(reset_mseq_each_step)

        # Init Pre-generator
        if self.use_pregen and self.enable_sweep and self.sweep is not None:
             self.sweep.precompute(Startfre=self.Startfre, Endfre=self.Endfre)

        self.pregen_data = None
        if self.use_pregen:
            self.pregen_data = PreGeneratedData(self)
        
        self.jammer_ptr = 0  # Continuous RF sample clock for indiscriminate jammers


        self.action_space = spaces.Dict({
            "hoprate": spaces.Box(
                low=np.array([self.hoprate_min], dtype=np.float32),
                high=np.array([self.hoprate_max], dtype=np.float32),
                dtype=np.float32
            ),
            "offsets": spaces.MultiDiscrete(
                np.full(self.num_blocks, self.num_channels, dtype=np.int64)
            ),
        })

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32
        )

        self.state = None
        self.last_info = {}

        self.debug_plot_psd = bool(debug_plot_psd)
        self.debug_log_hops = bool(debug_log_hops)
        self.current_step = 0

        # Step-triggered figure capture (see enable_step_figure_capture).
        self._fig_save_steps = set()
        self._fig_save_dir = None

        self._apply_hoprate(self.base_hoprate)

    def _build_jammer_variant_resources(self, jammer_config):
        requirements = {}

        def require_pool(bandwidth, min_samples):
            bandwidth = float(bandwidth)
            min_samples = max(1, int(min_samples))
            requirements[bandwidth] = max(
                requirements.get(bandwidth, 0),
                min_samples,
            )

        if self.enable_reactive:
            reactive_config = jammer_config['reactive']
            reactive_samples = max(
                1,
                int(
                    self.Fs
                    * float(reactive_config.get('detection_time', 0.001))
                ),
            )
            require_pool(reactive_config['bandwidth'], reactive_samples)

        if self.use_pregen and self.enable_sweep:
            mode = jammer_config.get('mode', 'sweep')
            if mode not in {'sweep', 'comb', 'both'}:
                raise ValueError("mode must be 'sweep', 'comb', or 'both'.")

            if mode in {'sweep', 'both'}:
                sweep_config = jammer_config['sweep']
                sweep_step = float(sweep_config.get('step', 125000.0))
                sweep_dwell = float(sweep_config.get('dwell_time', 0.004))
                if not np.isfinite(sweep_step) or sweep_step <= 0.0:
                    raise ValueError("Sweep step must be finite and positive.")
                if not np.isfinite(sweep_dwell) or sweep_dwell <= 0.0:
                    raise ValueError("Sweep dwell_time must be finite and positive.")
                bandwidth_span = max(self.Endfre - self.Startfre, 1.0)
                samples_per_dwell = max(1, int(round(sweep_dwell * self.Fs)))
                num_steps = max(1, int(np.floor(bandwidth_span / sweep_step)))
                require_pool(
                    sweep_config['bandwidth'],
                    samples_per_dwell * num_steps,
                )

            if mode in {'comb', 'both'}:
                comb_config = jammer_config['comb']
                switch_interval = float(comb_config.get('switch_interval', 0.3))
                interval_units = (
                    switch_interval / IndiscriminateJammer.COMB_TIME_QUANTUM
                )
                if (
                    not np.isfinite(switch_interval)
                    or switch_interval <= 0.0
                    or not np.isclose(
                        interval_units,
                        round(interval_units),
                        rtol=0.0,
                        atol=1e-9,
                    )
                ):
                    raise ValueError(
                        "JAMMER_CONFIG['comb']['switch_interval'] must be a finite "
                        "positive multiple of 0.1 seconds."
                    )
                require_pool(
                    comb_config['bandwidth'],
                    2 * int(round(switch_interval * self.Fs)),
                )

        if not requirements:
            return {}, None

        selector = JammerVariantSelector(self.baseband_variant_count)
        pools = {
            bandwidth: BandLimitedNoiseVariantPool(
                self.Fs,
                bandwidth,
                self.baseband_variant_count,
                requirements[bandwidth],
            )
            for bandwidth in sorted(requirements)
        }
        return pools, selector

    def _validate_comb_channels(self, comb_config):
        """
        Strictly validate the configurable comb channel index groups.

        Every entry of ``channels_phase0`` / ``channels_phase1`` must be an
        integer in ``[0, num_channels - 1]``; anything else raises ValueError.
        Group lengths may differ and groups may overlap. Missing keys fall
        back to the jammer's built-in defaults and are not checked here.
        """
        for key in ("channels_phase0", "channels_phase1"):
            channels = comb_config.get(key)
            if channels is None:
                continue
            if not isinstance(channels, (list, tuple)):
                raise ValueError(
                    f"JAMMER_CONFIG['comb']['{key}'] must be a list of "
                    f"channel indices, got {type(channels).__name__}."
                )
            for idx in channels:
                if isinstance(idx, bool) or not isinstance(idx, (int, np.integer)):
                    raise ValueError(
                        f"JAMMER_CONFIG['comb']['{key}'] entries must be "
                        f"integers, got {idx!r}."
                    )
                if idx < 0 or idx >= self.num_channels:
                    raise ValueError(
                        f"JAMMER_CONFIG['comb']['{key}'] entry {idx} is out "
                        f"of range [0, {self.num_channels - 1}]."
                    )

    def enable_step_figure_capture(self, save_steps, save_dir):
        """
        Enable per-block PSD figure saving at selected training steps.

        Args:
            save_steps: iterable of 1-based training step indices (matching
                the ``Step i/N`` training log index, i.e. the value of
                ``self.current_step + 1`` during ``step()``).
            save_dir: directory where ``step_XXX_block_YY.png`` files are
                written; created if missing.
        """
        self._fig_save_steps = set(int(s) for s in save_steps)
        self._fig_save_dir = str(save_dir)
        os.makedirs(self._fig_save_dir, exist_ok=True)

    def seed(self, seed=None):
        np.random.seed(seed)

    def _apply_hoprate(self, hoprate_target):
        hoprate_clip = float(np.clip(hoprate_target, self.hoprate_min, self.hoprate_max))
        hoprate_used = float(int(round(hoprate_clip / 10.0)) * 10)

        Nh = max(2, int(round(self.Baud / max(hoprate_used, 1e-9))))
        if Nh % 2 != 0:
            Nh += 1

        self.Nh = Nh
        self.modem.Nh = Nh
        self.current_hoprate = hoprate_used
        self.channel.Hoprate = hoprate_used

        return {
            "hoprate_action": hoprate_target,
            "hoprate_used": hoprate_used,
            "Nh_used": Nh
        }

    def _generate_rayleigh(self, length):
        if (not self.enable_rayleigh) or length <= 0:
            return None
        coh = max(1, int(self.rayleigh_coherence))
        num_seg = int(np.ceil(length / coh))
        mags = np.random.rayleigh(scale=np.sqrt(2) / 2, size=num_seg)
        mag_seq = np.repeat(mags, coh)[:length]
        return mag_seq

    def _assemble_signal_block(self, I_pulse, Q_pulse, hop_seq):
        """Apply fresh per-block AWGN and fresh per-hop Rayleigh fading."""
        I_pulse = np.asarray(I_pulse, dtype=np.float32)
        Q_pulse = np.asarray(Q_pulse, dtype=np.float32)
        if I_pulse.shape != Q_pulse.shape:
            raise ValueError("I_pulse and Q_pulse must have matching shapes.")

        baseband = (I_pulse + 1j * Q_pulse).astype(np.complex64)
        N = len(baseband)
        noise = (
            self.noise_std * np.random.randn(N)
        ).astype(np.float32)
        rx_assembled = np.empty(N, dtype=np.float32)
        carrier_assembled = np.empty(N, dtype=np.complex64)

        n_hops = len(hop_seq)
        if N == 0:
            return rx_assembled, carrier_assembled
        if n_hops == 0:
            rx_assembled[:] = noise
            carrier_assembled[:] = 0.0
            return rx_assembled, carrier_assembled

        s_per_hop_float = self.Fs / float(self.current_hoprate)
        two_pi_over_Fs = 2.0 * np.pi / self.Fs
        pos = 0
        cumsum_end = 0.0

        for hop_idx in range(n_hops):
            if hop_idx == n_hops - 1:
                next_pos = N
            else:
                next_pos = int(round((hop_idx + 1) * s_per_hop_float))
                next_pos = min(N, max(pos + 1, next_pos))
            length = next_pos - pos
            if length <= 0:
                continue

            ch_idx = int(hop_seq[hop_idx]) % self.num_channels
            f_c = (
                self.Startfre
                + ch_idx * self.Sub_interval
                + 0.5 * self.Sub_interval
            )
            t_local = np.arange(1, length + 1, dtype=np.float64)
            phase = two_pi_over_Fs * (cumsum_end + f_c * t_local)
            carrier_hop = np.exp(1j * phase).astype(np.complex64)
            cumsum_end += f_c * length

            rayleigh_mag = self._generate_rayleigh(length)
            if rayleigh_mag is None:
                rayleigh_mag = np.ones(length, dtype=np.float32)
            else:
                rayleigh_mag = np.asarray(rayleigh_mag, dtype=np.float32)

            faded_baseband = (
                baseband[pos:next_pos] * rayleigh_mag.astype(np.complex64)
            )
            rx_assembled[pos:next_pos] = (
                np.real(faded_baseband * carrier_hop) + noise[pos:next_pos]
            )
            carrier_assembled[pos:next_pos] = carrier_hop
            pos = next_pos
            if pos >= N:
                break

        if pos < N:
            rx_assembled[pos:] = noise[pos:]
            carrier_assembled[pos:] = 0.0
        return rx_assembled, carrier_assembled

    def _observe_100ms(self, block_id=None):
        N_obs = int(0.1 * self.Fs)
        sweep_jam = np.zeros(N_obs, dtype=np.float32)
        if self.enable_sweep and self.sweep is not None:
            if self.use_pregen:
                sweep_jam = self.sweep.get_composite_signal(
                    self.jammer_ptr,
                    N_obs,
                )
            else:
                t_obs = np.arange(N_obs) / self.Fs
                sweep_jam, _ = self.sweep.generate(
                    t_obs,
                    self.Startfre,
                    self.Endfre,
                    start_sample_idx=self.jammer_ptr,
                )
            self.jammer_ptr += N_obs

        noise = (
            self.noise_std * np.random.randn(N_obs)
        ).astype(np.float32)
        obs_signal = sweep_jam + noise

        plot_title = ""
        do_plot = self.debug_plot_psd
        if block_id is not None:
            plot_title = f"Step {self.current_step} - Block {block_id}"

        waterfall_db = compute_psd_waterfall(
            obs_signal,
            fs=self.Fs,
            f_start=self.Startfre,
            f_end=self.Endfre,
            dt=self.dt,
            df=self.df,
            max_duration=0.1,
            plot=do_plot,
            plot_title=plot_title
        )
        return waterfall_db

    def _get_block_hopseq(self, hops_per_block, offset):
        if hops_per_block <= 0:
            return np.array([], dtype=int)

        end_ptr = self._mseq_ptr + hops_per_block
        if end_ptr <= len(self.mseq_channels):
            base = self.mseq_channels[self._mseq_ptr:end_ptr]
        else:
            part1 = self.mseq_channels[self._mseq_ptr:]
            part2 = self.mseq_channels[:(end_ptr - len(self.mseq_channels))]
            base = np.concatenate([part1, part2])
        
        off_int = int(np.round(offset)) % self.num_channels
        hop_seq = (base + off_int) % self.num_channels
        return hop_seq

    def reset(self):
        self.current_step = 0
        self.jammer_ptr = 0 # Reset jammer pointer
        _ = self._apply_hoprate(self.base_hoprate)

        if self.jammer_variant_selector is not None:
            self.jammer_variant_selector.reset()
        
        if self.enable_sweep and self.sweep is not None:
             self.sweep.reset()

        # Reset Reactive Jammer state machine
        if self.enable_reactive and self.reactive is not None:
            self.reactive.reset()

        obs = self._observe_100ms(block_id=0)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
        )
        self.state = obs.astype(np.float32)
        self.last_info = {
            "ber_blocks": [],
            "hoprate_used": self.current_hoprate,
            "comb_phases": [],
        }
        return self.state, self.last_info

    def step(self, action=None):
        
        if action is None:
            hoprate_action = self.base_hoprate
            offsets_action = np.zeros(self.num_blocks, dtype=np.int64)
        else:
            if isinstance(action, dict):
                hoprate_action = float(action.get("hoprate", self.base_hoprate))
                offsets_action = np.array(
                    action.get("offsets", np.zeros(self.num_blocks)),
                    dtype=np.float32,
                )
            elif isinstance(action, (list, tuple)) and len(action) == 2:
                hoprate_action = float(action[0])
                offsets_action = np.array(action[1], dtype=np.float32)
            else:
                hoprate_action = float(action)
                offsets_action = np.zeros(self.num_blocks, dtype=np.float32)

        offsets_action = np.array(offsets_action, dtype=np.float32)
        if offsets_action.shape != (self.num_blocks,):
            raise ValueError(
                f"offsets must have shape ({self.num_blocks},), "
                f"got {offsets_action.shape}."
            )
        if not np.all(np.isfinite(offsets_action)):
            raise ValueError("offsets must contain only finite values.")
        rounded_offsets = np.rint(offsets_action)
        if not np.allclose(offsets_action, rounded_offsets):
            raise ValueError("offsets must contain integer-valued channel indices.")
        offsets_action = rounded_offsets.astype(np.int64)
        if np.any(offsets_action < 0) or np.any(offsets_action >= self.num_channels):
            raise ValueError(
                f"offsets must be in [0, {self.num_channels - 1}]."
            )

        ainfo = self._apply_hoprate(hoprate_action)

        hops_per_block = int(round(self.current_hoprate * 0.1))
        hops_per_block = max(1, hops_per_block)

        ber_blocks = []
        reactive_active_blocks = []
        hop_sequences = []
        comb_phases = []
        capture_figures = (
            (self.current_step + 1) in self._fig_save_steps
            and self._fig_save_dir is not None
        )

        for b in range(self.num_blocks):
            if (
                self.enable_sweep
                and self.sweep is not None
                and self.sweep.comb_enabled
            ):
                comb_phases.append(self.sweep.comb_phase_at(self.jammer_ptr))

            # ----------------------------------------------------------------
            # 2. Hopping Sequence (Dynamic)
            # ----------------------------------------------------------------
            hop_seq_block = self._get_block_hopseq(hops_per_block, offsets_action[b])
            self._mseq_ptr = (self._mseq_ptr + len(hop_seq_block)) % len(self.mseq_channels)
            hop_sequences.append(hop_seq_block.astype(int).tolist())
            
            # ----------------------------------------------------------------
            # 1. Obtain Baseband / Noise / Interference
            # ----------------------------------------------------------------
            use_pre = (self.use_pregen and self.pregen_data is not None)
            
            if use_pre:
                # NEW OPTIMIZED PATH: Assemble from Channel Pool
                # Returns fully assembled RX signal (Baseband*Carrier*Rayleigh + Noise)
                # But wait, we need separate components for Reactive Jammer logic?
                # Reactive Jammer needs Pure Carrier (to sync phase) or just frequency knowledge.
                # pregen_data.get_block returns (rx_signal_static, pure_carrier_for_rx, bits)
                rx_static, carrier_complex, bits_block = self.pregen_data.get_block(hop_seq_block)

                # t_block is needed for Reactive/Jammer
                if b == 0:
                     self._t_block_cache = np.arange(len(rx_static)) / self.Fs
                t_block = self._t_block_cache
                
                if len(t_block) != len(rx_static): # Safety for hoprate change?
                    t_block = np.arange(len(rx_static)) / self.Fs
                
                # Sweep Jammer?
                # If using pre-gen, we assumed Sweep is baked in? 
                # No, in new Code, PreGeneratedData does NOT bake in Sweep/Comb to the pool signal.
                # Because Sweep/Comb evolves over time/steps.
                # So we calculate it dynamically here.
                # Is that slow? IndiscriminateJammer.generate() uses cos/sin.
                # If we want to optimize that too, we should pre-gen 'comb' and 'sweep' buffers...
                # User asked to save "Carrier time".
                sweep_jam = np.zeros(len(rx_static))
                if self.enable_sweep and self.sweep is not None:
                     sweep_jam = self.sweep.get_composite_signal(
                         self.jammer_ptr,
                         len(rx_static),
                     )
                     self.jammer_ptr += len(rx_static)
                     
                rx_real = rx_static + sweep_jam
                
            else:
                # ----------------------------------------------------------------
                # OLD DYNAMIC PATH
                # ----------------------------------------------------------------
                num_syms_block = int(round(self.Baud * 0.1))
                bits_block = self.modem.generate_bits(2 * num_syms_block)
                I_pulse, Q_pulse = self.modem.pulse_shape(bits_block)

                t_block = np.arange(len(I_pulse)) / self.Fs
                rx_static, carrier_complex = self._assemble_signal_block(
                    I_pulse,
                    Q_pulse,
                    hop_seq_block,
                )
                    
                sweep_jam = np.zeros(len(I_pulse))
                if self.enable_sweep and self.sweep is not None:
                    sweep_jam, _ = self.sweep.generate(
                        t_block,
                        self.Startfre,
                        self.Endfre,
                        start_sample_idx=self.jammer_ptr,
                    )
                    self.jammer_ptr += len(I_pulse)
                
                rx_real = rx_static + sweep_jam

            # ----------------------------------------------------------------
            # 4. Reactive Jammer (Dynamic) - Independent of path
            # ----------------------------------------------------------------
            reactive_jam = np.zeros(len(rx_real))
            reactive_active = False
            if self.enable_reactive and self.reactive is not None:
                reactive_jam, reactive_active = self.reactive.generate(
                    t_block, hop_seq_block, self.Startfre, self.Sub_interval, self.current_hoprate
                )
            
            # Final RX mixing
            rx_real += reactive_jam

            # Step-triggered per-block PSD figure capture
            if capture_figures:
                wf_db = compute_psd_waterfall(
                    rx_real,
                    fs=self.Fs,
                    f_start=self.Startfre,
                    f_end=self.Endfre,
                    dt=self.dt,
                    df=self.df,
                    max_duration=0.1,
                    plot=False,
                )
                fig_name = f"step_{self.current_step + 1:03d}_block_{b + 1:02d}.png"
                save_waterfall_figure(
                    wf_db,
                    os.path.join(self._fig_save_dir, fig_name),
                    title=f"Step {self.current_step + 1} - Block {b + 1} PSD (100 ms)",
                )

            # ----------------------------------------------------------------
            # 5. Receive
            # ----------------------------------------------------------------
            # RX = (Signal * Rayleigh) + Reactive + Sweep + Noise

            _, ber = self.receiver.demodulate(rx_real, bits_block, carrier_complex)
            ber_blocks.append(float(ber))
            reactive_active_blocks.append(bool(reactive_active))

            if self.debug_plot_psd:
                compute_psd_waterfall(
                    rx_real,
                    fs=self.Fs,
                    f_start=self.Startfre,
                    f_end=self.Endfre,
                    dt=self.dt,
                    df=self.df,
                    max_duration=0.1,
                    plot=True,
                    plot_title=f"Step {self.current_step} - Block {b+1}"
                )
            if self.debug_log_hops:
                print(f"Block {b+1}: Hop Seq (with offset) = {hop_seq_block.tolist()}")

        
        self.current_step += 1
        
        obs = self._observe_100ms(block_id=0)
        self.state = obs.astype(np.float32)

        if self.reset_mseq_each_step:
            self._mseq_ptr = 0

        mean_ber = float(np.mean(ber_blocks)) if len(ber_blocks) > 0 else 0.0
        block_rewards = compute_block_rewards(
            ber_blocks,
            ainfo["hoprate_used"],
            settings.REWARD_CONFIG,
        )
        reward = float(np.mean(block_rewards))

        self.last_info = {
            "ber_blocks": ber_blocks,
            "block_rewards": block_rewards.tolist(),
            "mean_ber": mean_ber,
            "hoprate_used": ainfo["hoprate_used"],
            "hops_per_block": hops_per_block,
            "reactive_active_blocks": reactive_active_blocks,
            "hop_sequences": hop_sequences,
            "comb_phases": comb_phases,
        }

        terminated = False
        truncated = False
        return self.state, reward, terminated, truncated, self.last_info

    def render(self, mode="human"):
        if self.state is None:
            return
        plt.figure(figsize=(8, 4))
        plt.imshow(self.state.T, origin="lower", aspect="auto", cmap="jet")
        plt.colorbar(label='PSD (dB)')
        plt.xlabel('Time bin')
        plt.ylabel('Freq bin')
        plt.title('PSD Waterfall (100 ms observation)')
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    pr_start = time.time()
    # Note: Jammer configuration is now loaded from settings.py
    env = FHSSQPSKEnv(enable_reactive=False,
                      enable_sweep=False,
                      enable_rayleigh=True,
                      debug_plot_psd=False,
                      debug_log_hops=False,
                      use_pregen=True)
    pr_end = time.time()
    print(f"Environment Initialization Time: {pr_end - pr_start:.4f} s")
    obs, info = env.reset()

    offsets = np.zeros(env.num_blocks, dtype=np.int64)
    action = {"hoprate": 200.0, "offsets": offsets}
    for i in range(env.num_blocks):
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: Reward: {reward}, Mean BER: {info['mean_ber']}")
