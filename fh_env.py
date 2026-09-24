"""
FHSS/QPSK 抗干扰 Gymnasium 环境（项目通信仿真核心）。

本模块是强化学习实验的"仿真世界"：把 QPSK 收发链路、跳频（m 序列 LFSR 驱动）、
Rayleigh 衰落、三类干扰机（反应式能量检测 / 无差别扫频 / 无差别梳状）与 PSD
waterfall 观测封装为 Gymnasium 环境 FHSSQPSKEnv，供 SAC.py 的十头离散 SAC 学习
"每个 block 选择哪个起始信道 offset"的跳频抗干扰策略。

一、单 block 的通信仿真链路（_assemble_signal_block → _process_block_task）
    bits(2*num_syms_block)                    随机比特，作为 BER 的地面真值
    → QPSK 映射 + RRC 脉冲成型（上采样 Ns 倍）  I_pulse/Q_pulse，实数样本
    → 乘复载波 exp(j*phi)                     逐 hop 换频点（跳频）
    → 乘 Rayleigh 衰落幅度（逐 hop 独立）       enable_rayleigh 控制
    → 加干扰（sweep/comb/reactive）与 AWGN     rx_real = 有用信号 + 干扰 + 噪声
    → 乘共轭载波下变频 → 匹配滤波降采样 → 硬判决 → BER
    注意：接收端只做"理想同步"解调（用发送端载波与本地 RRC 匹配滤波），
    不含定时/频偏估计，因此 BER 完全由干扰、噪声与衰落决定。

二、关键概念与语义
    block：100 ms 通信片段，样本数 block_len = num_syms_block * Ns
        （num_syms_block = round(Baud*0.1)，Baud=25 kHz 时为 2500 符号）。
    step：一次 env.step() 按顺序执行 NUM_BLOCKS=10 个 block（合计 1 s 通信），
        动作给出 10 个 offset（每 block 一个）；奖励逐 block 计算后取均值返回。
    offset：block 内跳频序列的信道偏移量。第 k 个 hop 的信道索引为
        (mseq_channels[ptr+k] + offset) % num_channels，等价于对 m 序列基序列
        做循环平移；offset ∈ [0, num_channels-1] 为离散动作取值。
    PSD waterfall 观测：对 100 ms 观测窗（只含干扰 + AWGN，不含有用信号，
        因为策略需要"听"干扰环境）分帧 FFT，得到形状 (num_frames, n_bins)
        的功率矩阵；默认 dt=1 ms、df=10 kHz → (100, 100) 的 dB 瀑布图。
    hoprate 量化：动作 hoprate 先 clip 到 [hoprate_min, hoprate_max]，再量化到
        10 Hz 网格（int(round(x/10))*10），故实际取值只能是 10 Hz 的整数倍；
        量化值决定每 block 的 hop 数（round(hoprate*0.1)），并同步得到每跳符号数
        Nh = round(Baud/hoprate)（取偶数；本文件内仅作记录/诊断，不参与信号合成）。
    m 序列：n_bits 级 LFSR（默认 10 级、taps=(10,7)，周期 2^10-1=1023）生成
        跳频基序列，再对 num_channels 取模得到 mseq_channels；
        reset_mseq_each_step=True 时每个 step 结束把指针归零，使各 step 的
        基序列一致（保证不同 episode/step 之间动作可比）。
    use_pregen 加速路径：QPSK 比特、成型 I/Q（common_baseband）与干扰波形池
        被预生成并跨 block/step 复用；但每个 block 的 AWGN、每个 hop 的
        Rayleigh 衰落、每次观测的 AWGN 都重新生成，故仍是一次独立的信道实现。
    reset 语义：重置连续 RF 时钟（jammer_ptr）、干扰变体选择流、扫频/梳状
        位置与反应式干扰机状态机；但通信侧的随机流（self._rng 及其派生子流
        种子）不回退，因此连续两次 reset 得到的观测与 BER 不会完全相同。

三、随机性边界（可复现性约定）
    - 所有随机源都由 self._rng（np.random.RandomState）派生整数种子；只有主
      线程推进 self._rng，worker 线程各自用 RandomState(seed) 局部生成，
      不触碰共享随机状态，故多线程不改变结果分布。
    - 因此给定 seed（reset(seed=...) / seed()）后：观测噪声、每 block AWGN、
      每 hop Rayleigh、干扰变体选择可复现；但预生成干扰波形池本身在
      __init__ 阶段生成（来自全局 np.random），不受后续 seed() 影响。

四、与其它模块的关系
    jammers.py：ReactiveJammer / IndiscriminateJammer 负责干扰波形与相位，
        本模块只负责按时间线推进它们并把波形叠加到接收信号上。
    settings.py：REWARD_CONFIG（奖励系数）、JAMMER_CONFIG（干扰机参数）、
        ENV_CONFIG（本环境的构造参数默认值）。
    SAC.py：消费本环境返回的观测与逐 block 奖励；动作用
        {"hoprate": float, "offsets": int[num_blocks]} 传回本环境。

五、Gymnasium 接口
    observation_space：Box(-inf, inf, shape=(num_frames, n_bins)) float32，
        默认 (100, 100)，数值单位为 dB（10*log10(PSD)）；reset() 时按实际
        观测形状重建（因此形状取决于 dt/df/Fs 配置）。
    action_space：Dict{
        "hoprate": Box(low=[hoprate_min], high=[hoprate_max], dtype=float32)，
        "offsets": MultiDiscrete([num_channels] * num_blocks)}。
    step() 返回 (obs, reward, terminated, truncated, info)；本任务无终止条件，
        terminated/truncated 恒为 False；info 携带逐 block BER、逐 block 奖励、
        hoprate_used、hops_per_block、hop_sequences、comb_phases 等诊断字段。
"""

import copy
import os
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

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


# 一次 env.step() 内按顺序执行的 100 ms 通信 block 数（合计 1 s 仿真时长）。
# 与动作维度、奖励维度、_draw_substream_seeds((num_blocks, 5)) 的子流数量绑定，
# 改动它会同时改变动作空间与 info["block_rewards"] 的长度。
NUM_BLOCKS = 10


def compute_block_rewards(ber_blocks, hoprate, reward_config=None):
    """
    按统一奖励公式逐 block 计算奖励。

    公式（与 settings.REWARD_CONFIG 注释、train_offsets.py 的 baseline 一致）：
        reward_i = base_reward - ber_penalty * BER_i - hoprate_penalty * hoprate

    其中 hoprate 项在同一个 step 内对 10 个 block 完全相同（跳速是 step 级全局
    动作），因此 block 之间的奖励差异只来自 BER_i；环境返回的标量 step reward
    是这 10 个值的算术平均（见 FHSSQPSKEnv.step）。

    Args:
        ber_blocks: array-like，形状 (NUM_BLOCKS,)=(10,)，逐 block 误码率，
            取值 [0, 1]，元素必须为有限值。
        hoprate: float，本 step 实际使用的跳速（Hz）。应传入 _apply_hoprate
            量化后的值（10 Hz 的整数倍），而不是原始动作值，否则奖励尺度与
            环境其它部分不一致。
        reward_config: dict 或 None。None 时用 settings.REWARD_CONFIG；需包含
            键 base_reward、ber_penalty、hoprate_penalty。

    Returns:
        np.ndarray，形状 (10,)，dtype=float32，逐 block 奖励。

    Raises:
        ValueError: ber_blocks 形状不是 (10,) 时；BER 或 hoprate 含非有限值时。
    """
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
    生成单位能量的根升余弦（RRC）滤波器系数。

    Args:
        rolloff: float，滚降系数 β（commpy 的 alpha 参数），典型 0.5。
        span: int，滤波器每侧的符号跨度；总抽头数为 span*sps+1。
        sps: int，每符号采样数（本项目的 Ns = Fs / Baud）。

    Returns:
        np.ndarray，形状 (span*sps+1,)，dtype=float64，已按 sqrt(sum(h^2)) 归一化
        到单位能量（分母加 1e-12 仅作数值兜底）。

    设计要点：
        - 取奇数长度 span*sps+1，使群延迟 (len-1)//2 为整数且可被 sps 整除，
          这样 pulse_shape 里按 gd 截取能精确对齐符号起点（否则成型信号会有
          亚采样点的定时偏移）。
        - 单位能量归一化让成型/匹配滤波不改变符号能量，从而 signal_power 与
          noise_std 之间的 SNR 关系可直接解析（反应式干扰机的检测门限依赖该
          一致性）。
        - rrcosfilter 的 Ts=1、Fs=sps 表示采样间隔为 1/sps 个符号周期，即每符号
          sps 个采样点。
    """
    # 取奇数长度 span*sps+1，保证群延迟为整数且与 sps 对齐
    rrc_filter = rrcosfilter(span * sps + 1, rolloff, 1, sps)[1]
    rrc_filter = rrc_filter / np.sqrt(np.sum(rrc_filter ** 2) + 1e-12)
    return rrc_filter


@lru_cache(maxsize=32)
def _get_psd_plan(fs, f_start, f_end, dt, df, window):
    """
    返回 PSD 计算所需的只读 FFT/窗/频带几何参数（按配置缓存）。

    该函数被 lru_cache 以 (fs, f_start, f_end, dt, df, window) 为键缓存，避免每个
    block、每次观测都重复计算窗函数与频带边界索引（这些量只与配置有关，与信号
    无关）。返回的数组全部 setflags(write=False)：它们是跨线程、跨调用共享的对象，
    置为只读可防止下游误改导致缓存被污染。

    几何定义：
        Nwin  = int(dt * fs)                 每帧样本数（时间分辨率 dt）
        Nfft  = 2^ceil(log2(Nwin))           零填充到 2 的幂，加速 rfft
        freqs = np.fft.rfftfreq(Nfft, 1/fs)  rfft 频率轴（Hz），长度 Nfft//2+1
        n_bins = floor((f_end-f_start)/df)   覆盖 [f_start, f_end) 的频率 bin 数
        f_bin_edges = f_start + arange(n_bins+1)*df
        starts/ends = searchsorted(freqs, f_bin_edges) 的相邻切片，给出每个 bin
                 在 rfft 频率轴上的下标区间 [start, end)；空 bin 满足 end<=start
        valid_bins = 下标非空、即真正有 FFT 谱线落入的 bin 序号（其余 bin 保持 0）
        norm_factor = 1 / (fs * sum(win^2) / Nwin)

    Args:
        fs: float，采样率（Hz）。
        f_start: float，分析频带下限（Hz）。
        f_end: float，分析频带上限（Hz）。
        dt: float，每帧时长（s），决定时间分辨率与帧数。
        df: float，频率 bin 宽度（Hz），决定频率分辨率与 bin 数。
        window: str，'hann' / 'hamming' / 其它（其它值退化为矩形窗，即全 1）。

    Returns:
        tuple:
            Nwin (int), Nfft (int), n_bins (int),
            win (np.ndarray, (Nwin,), float64),
            freqs (np.ndarray, (Nfft//2+1,), float64),
            starts (np.ndarray, (n_bins,), intp),
            ends (np.ndarray, (n_bins,), intp),
            valid_bins (np.ndarray, (<=n_bins,), intp),
            norm_factor (float)

        关于 norm_factor：它等价于矩形窗下的 1/fs，把 |FFT|^2 折算成"每 Hz 功率"
        量级，使不同窗型/帧长下的 dB 数值可横向比较。注意这里未做单边谱的 ×2
        修正，也未除以 bin 宽度，故 dB 是相对量级——训练只需要相对差异。

    Raises:
        ValueError: Nwin <= 0（dt 或 fs 配置不合法）时。
    """
    fs = float(fs)
    f_start = float(f_start)
    f_end = float(f_end)
    dt = float(dt)
    df = float(df)
    Nwin = int(dt * fs)
    if Nwin <= 0:
        raise ValueError("Nwin must be positive. Check dt and fs.")

    if window == 'hann':
        win = np.hanning(Nwin)
    elif window == 'hamming':
        win = np.hamming(Nwin)
    else:
        win = np.ones(Nwin)   # 未识别的窗名退化为矩形窗
    win = np.asarray(win, dtype=np.float64)

    # 零填充到 2 的幂：rfft 在长度为 2 的幂时最快，代价是频率轴变密（不影响分 bin）
    Nfft = int(2 ** np.ceil(np.log2(Nwin)))
    freqs = np.fft.rfftfreq(Nfft, d=1.0 / fs)
    n_bins = max(1, int(np.floor((f_end - f_start) / df)))
    f_bin_edges = f_start + np.arange(n_bins + 1) * df
    # searchsorted 得到每个 bin 边界在频率轴上的位置，相邻两项即该 bin 的谱线区间
    edge_indices = np.searchsorted(freqs, f_bin_edges)
    starts = edge_indices[:-1].astype(np.intp, copy=False)
    ends = edge_indices[1:].astype(np.intp, copy=False)
    # 只保留有谱线落入的 bin（空 bin 保持 0，避免 log 后出现 -inf 之外的无意义值）
    valid_bins = np.flatnonzero(ends > starts).astype(np.intp, copy=False)
    # 窗能量归一化：等价于矩形窗下的 1/fs（见 docstring 对量级的说明）
    norm_factor = 1.0 / (fs * np.sum(win ** 2) / Nwin)

    # 共享对象置只读：lru_cache 的返回会被多次复用，禁止下游原地修改
    for array in (win, freqs, starts, ends, valid_bins):
        array.setflags(write=False)
    return Nwin, Nfft, n_bins, win, freqs, starts, ends, valid_bins, norm_factor


def compute_psd_waterfall(signal, fs, f_start, f_end,
                          dt=0.001,      # 时间分辨率 1 ms → 每帧 Nwin = dt*Fs 个样本
                          df=10000.0,    # 频率分辨率 10 kHz → 100 个频率 bin
                          max_duration=0.1,  # 只分析前 100 ms（与 block 时长一致）
                          window='hann',
                          plot=False,
                          plot_title=""):
    """
    用一次批量 FFT 计算 PSD waterfall（保持与旧实现逐点一致的结果）。

    实现要点（为什么这样写）：
        - 只分析前 max_duration 秒（默认 100 ms）：观测窗固定为 100 ms，与
          block 时长一致，保证观测与决策时间尺度对齐。
        - 分帧：把信号截成 num_frames = len(signal)//Nwin 帧，堆成
          (num_frames, Nwin) 矩阵后一次性 rfft(axis=1)，避免逐帧 Python 循环
          （这是本函数的主要加速点）。
        - 频带聚合：用 cumsum 前缀和求任意 [start, end) 频带的功率和，把每个
          bin 的一次切片求和变成两次索引相减，整体 O(num_frames * Nfft)。
        - 输出 dB：10*log10(P + 1e-12)，1e-12 防止空 bin 的 log(0) = -inf 破坏
          下游归一化。

    Args:
        signal: array-like，实数时域样本，1-D，长度应 >= Nwin 才有输出。
        fs: float，采样率（Hz）。
        f_start: float，分析频带下限（Hz）。
        f_end: float，分析频带上限（Hz）。
        dt: float，时间分辨率（s），默认 1 ms → 每帧 Nwin = dt*fs 个样本。
        df: float，频率分辨率（Hz），默认 10 kHz。
        max_duration: float 或 None，只分析前 max_duration 秒；<=0 或 None 表示
            使用全部样本。
        window: str，窗函数名（'hann' / 'hamming' / 其它 → 矩形窗）。
        plot: bool，True 时用 matplotlib 弹窗显示（仅调试用，训练路径为 False）。
        plot_title: str，plot=True 时的图标题。

    Returns:
        np.ndarray，形状 (num_frames, n_bins)，dtype=float64，单位 dB。
        默认配置（Fs=10 MHz, dt=1 ms, df=10 kHz, max_duration=0.1 s）下为
        (100, 100)，即（时间 bin, 频率 bin）——绘图时才转置成横轴时间、纵轴频率。
        若信号长度不足一帧则返回形状 (0, n_bins) 的空矩阵，调用方需容忍空结果。
    """
    if max_duration is not None and max_duration > 0:
        max_samples = int(max_duration * fs)
        signal = signal[:max_samples]

    (
        Nwin,
        Nfft,
        n_bins,
        win,
        _freqs,
        starts,
        ends,
        valid_bins,
        norm_factor,
    ) = _get_psd_plan(fs, f_start, f_end, dt, df, window)

    if len(signal) < Nwin:
        return np.zeros((0, n_bins))

    num_frames = len(signal) // Nwin
    frame_samples = np.asarray(signal[:num_frames * Nwin])
    frames = frame_samples.reshape(num_frames, Nwin)
    # 单次批量 FFT：一次算完所有帧（axis=1），避免逐帧 Python 循环
    spec = np.fft.rfft(frames * win[np.newaxis, :], n=Nfft, axis=1)
    # |X|^2：复数的实部平方 + 虚部平方（比 np.abs(spec)**2 少一次开方/平方）
    psd = np.square(spec.real)
    psd += np.square(spec.imag)
    psd *= norm_factor

    waterfall = np.zeros((num_frames, n_bins), dtype=np.float64)
    if valid_bins.size:
        # 前缀和技巧：band_powers = prefix[end-1] - prefix[start-1]，把每个 bin 的
        # 区间求和降为 O(1) 的两次索引，整体复杂度与 bin 数近似无关
        prefix = np.cumsum(psd, axis=1)
        valid_starts = starts[valid_bins]
        valid_ends = ends[valid_bins]
        band_powers = prefix[:, valid_ends - 1].copy()
        nonzero_start = valid_starts > 0
        if np.any(nonzero_start):
            band_powers[:, nonzero_start] -= prefix[
                :,
                valid_starts[nonzero_start] - 1,
            ]
        waterfall[:, valid_bins] = band_powers

    # 转 dB：+eps 防止空 bin 的 log10(0) = -inf 破坏下游归一化
    eps = 1e-12
    waterfall_db = 10 * np.log10(waterfall + eps)

    if plot and waterfall_db.size > 0:
        plt.figure(figsize=(8, 4))
        plt.imshow(waterfall_db.T, origin="lower", aspect="auto", cmap="jet")
        plt.colorbar(label='PSD (dB)')
        plt.xlabel('Time bin')
        plt.ylabel('Freq bin')
        title_str = plot_title or 'PSD Waterfall ({:.0f} ms)'.format(
            max_duration * 1e3
        )
        plt.title(title_str)
        plt.tight_layout()
        plt.show()

    return waterfall_db


def save_waterfall_figure(waterfall_db, path, title=""):
    """
    把已算好的 PSD waterfall 数组保存为 PNG 图片。

    绘图风格与模块内其它调试图完全一致（jet colormap、origin="lower"、带
    colorbar），区别是写入 *path* 而不弹窗——训练中批量出图时不能阻塞主线程。

    Args:
        waterfall_db: array-like，形状 (num_frames, n_bins)，单位 dB。内部会转置
            显示（.T），使横轴为时间 bin、纵轴为频率 bin。
        path: str，输出文件路径（父目录需已存在，本函数不创建目录）。
        title: str，非空时作为图标题。

    副作用：创建并关闭独立的 Figure，避免训练循环里反复出图导致 matplotlib
    句柄泄漏。
    """
    waterfall_db = np.asarray(waterfall_db)
    fig = plt.figure(figsize=(8, 4))
    # .T 转置：输入为 (时间 bin, 频率 bin)，显示时需要横轴时间、纵轴频率
    plt.imshow(waterfall_db.T, origin="lower", aspect="auto", cmap="jet")
    plt.colorbar(label='PSD (dB)')
    plt.xlabel('Time bin')
    plt.ylabel('Freq bin')
    if title:
        plt.title(title)
    plt.tight_layout()
    fig.savefig(path)
    # 显式关闭，避免训练中反复出图累积 Figure 句柄
    plt.close(fig)


def show_waterfall_figure(waterfall_db, title=""):
    """
    在主线程渲染并弹窗显示已算好的 waterfall。

    之所以强调"主线程"：matplotlib 的 pyplot 不是线程安全的，而 step() 里的
    block 计算可能跑在 ThreadPoolExecutor 的 worker 上，因此 worker 只返回
    数值数组（见 _process_block_task），所有绘图都回到主线程执行。

    Args:
        waterfall_db: array-like，形状 (num_frames, n_bins)，单位 dB（显示时转置）。
        title: str，非空时作为图标题。
    """
    waterfall_db = np.asarray(waterfall_db)
    plt.figure(figsize=(8, 4))
    # 同 save_waterfall_figure：转置后横轴时间、纵轴频率
    plt.imshow(waterfall_db.T, origin="lower", aspect="auto", cmap="jet")
    plt.colorbar(label='PSD (dB)')
    plt.xlabel('Time bin')
    plt.ylabel('Freq bin')
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# -----------------------------
# M序列（LFSR）生成
# -----------------------------
def generate_mseq_states(n_bits=10, length=1000, taps=(10, 7), seed=1):
    """
    用线性反馈移位寄存器（LFSR，SSRG/Fibonacci 结构）生成 m 序列的状态序列
    （跳频基序列的来源）。

    与"输出比特序列"不同，这里返回的是 LFSR 的完整状态字：每个状态的低位即
    该时刻的 m 序列输出，环境随后对它取模映射成信道索引（见 mseq_channels）。
    这样一次生成即可同时保留比特与状态信息。

    实现细节与约束：
        - seed == 0 时强制改为 1：全零状态是 LFSR 的吸收态，一旦进入将永远输出 0，
          必须避免（循环内的 state == 0 兜底同理）。
        - taps 为 1-based 抽头位置，反馈比特由这些位置的异或得到，再移入最低位
          （左移一位后 | fb）；因此状态字的低 n_bits 位始终有效（mask 保证）。
        - n_bits=10、taps=(10,7) 是本原多项式配置，状态序列以 2^10-1=1023 为周期
          遍历所有非零状态，即 length=1023 时恰好一个完整周期。

    Args:
        n_bits: int，LFSR 级数（寄存器位数），决定状态空间 2^n_bits - 1。
        length: int，生成的样本个数（状态数），可超过一个周期。
        taps: tuple[int, ...]，1-based 抽头位置，如 (10, 7)。
        seed: int，初始状态（0 会被替换为 1）。

    Returns:
        np.ndarray，形状 (length,)，dtype=int64，元素为 [1, 2^n_bits - 1] 内的
        状态字。调用方通常再对 num_channels 取模得到信道索引。
    """
    if seed == 0:
        seed = 1
    mask = (1 << n_bits) - 1
    state = seed & mask
    seq = []
    for _ in range(length):
        seq.append(state)
        fb = 0
        for t in taps:
            # 1-based 抽头 → 0-based 位移；异或得到反馈比特
            fb ^= (state >> (t - 1)) & 1
        state = ((state << 1) & mask) | fb
        if state == 0:
            # 兜底：LFSR 全零为吸收态，强制回到非零状态以维持序列长度
            state = 1
    return np.array(seq, dtype=np.int64)





# -----------------------------
# 预生成数据管理器
# -----------------------------
class PreGeneratedData:
    """
    预生成数据管理器：use_pregen=True 时的加速路径（复用 QPSK 基带与比特）。

    设计意图：QPSK 调制（比特生成 + RRC 成型）在每个 block 的输入都是同一套计算，
    真正随机的只有信道侧（AWGN / Rayleigh / 干扰）。因此这里把"每 block 恒定的
    部分"提前算好并跨 block、跨 step 复用，只把随机信道实现留给
    _assemble_signal_block，换来数倍的仿真吞吐（离线 replay 生成与训练都依赖它）。

    不变量（重要）：
        - common_bits / common_I / common_Q / common_baseband 在环境生命周期内
          不再改变：所有 block、所有 step 的发送比特完全相同，因此 BER 的差异
          纯粹来自干扰/噪声/衰落，而不是数据内容。
        - 这是"原型优化"的取舍：牺牲数据多样性（BER 统计只反映单一比特图案，
          对特定干扰的符号级影响可能被低估），换取吞吐。
        - common_I/common_Q 为 float32 实数样本，common_baseband 为 complex64
          解析信号，形状均为 (block_len,)。
        - block_len 与环境自身的 _block_len 在本配置下相等；step() 用
          pregen_data.block_len 作为实际 block 长度（见 step 中的 use_pre 分支），
          因此若两者不一致，以预生成值为准。

    关键属性：
        env: FHSSQPSKEnv，反向引用，用于调用 modem 与 _assemble_signal_block。
        num_syms_block: int，每 block 符号数 = round(Baud * 0.1)。
        bits_len_per_block: int，每 block 比特数 = 2 * num_syms_block（QPSK 每符号 2 bit）。
        common_bits: np.ndarray (bits_len_per_block,), int8，BER 的地面真值。
        common_I / common_Q: np.ndarray (block_len,), float32。
        common_baseband: np.ndarray (block_len,), complex64。
        block_len: int，成型后每 block 的样本数。
        dtype: np.dtype，I/Q 存储类型（默认 float32）。
    """

    def __init__(self, env, dtype=np.float32):
        """
        预生成共享比特与成型波形。

        Args:
            env: FHSSQPSKEnv，需已初始化 modem（Baud/Fs/Ns 可用）。
            dtype: np.dtype，I/Q 波形存储类型，默认 float32（省内存、与下游一致）。

        副作用：打印 3 行进度与 block 长度信息；比特取自全局 np.random（未传 rng），
        因此不受 env.seed() 控制——见模块 docstring 的"随机性边界"说明。
        """
        print("Initializing PreGeneratedData (Reusable QPSK Baseband Mode)...")
        self.env = env
        self.dtype = dtype
        self.num_syms_block = int(round(env.Baud * 0.1))

        # 1. 基带比特（BER 的地面真值）——所有信道/所有 block 共用
        # 注意：这是原型优化的取舍——共用同一段比特，牺牲数据多样性换吞吐；
        # 若要恢复多样性，应在此处为每个 block 生成独立比特。
        self.bits_len_per_block = 2 * self.num_syms_block
        self.common_bits = env.modem.generate_bits(self.bits_len_per_block).astype(np.int8)
        
        # 2. 基带 I/Q——同样全 block 共用
        # 成型只做一次（含上采样与 RRC 卷积，是每 block 最贵的固定开销）
        self.common_I, self.common_Q = env.modem.pulse_shape(self.common_bits)
        self.common_I = self.common_I.astype(dtype)
        self.common_Q = self.common_Q.astype(dtype)
        # 预拼 complex64 解析信号，供 _assemble_signal_block 免去每 block 的复数构造
        self.common_baseband = (
            self.common_I.astype(np.complex64)
            + 1j * self.common_Q.astype(np.complex64)
        ).astype(np.complex64, copy=False)
        self.block_len = len(self.common_I)
        print(f"Reusable QPSK block length: {self.block_len}")
        print("Pre-generation complete.")

    def get_block(self, hop_seq, rng=None, rayleigh_rng=None):
        """
        取一个 block 的接收信号与载波：复用预生成 I/Q，只重新生成信道随机量。

        Args:
            hop_seq: array-like[int]，本 block 的跳频信道索引序列（已含 offset 平移），
                长度 = hops_per_block。
            rng: np.random.RandomState 或 None，用于本 block 的 AWGN
                （None → 全局 np.random；多线程路径必须传入）。
            rayleigh_rng: np.random.RandomState 或 None，用于每 hop 的 Rayleigh 幅度；
                None 时回退到 rng（见 _assemble_signal_block）。

        Returns:
            tuple:
                rx_assembled: np.ndarray (block_len,), float32，接收实数带通信号
                    （有用信号 + AWGN，尚未叠加干扰）。
                carrier_assembled: np.ndarray (block_len,), complex64，逐样本载波
                    exp(j*phi)，供接收端共轭下变频使用。
                common_bits: np.ndarray (bits_len_per_block,), int8，BER 地面真值
                    （注意是共享对象，调用方不得原地修改）。
        """
        rx_assembled, carrier_assembled = self.env._assemble_signal_block(
            self.common_I,
            self.common_Q,
            hop_seq,
            rng=rng,
            rayleigh_rng=rayleigh_rng,
            baseband=self.common_baseband,
        )
        return rx_assembled, carrier_assembled, self.common_bits


# -----------------------------
# 调制与信道
# -----------------------------
class QPSKModem:
    """
    QPSK 调制器：比特流 → 成型后的 I/Q 基带波形。

    QPSK 映射按"双路 BPSK"实现：偶数位 → I 路、奇数位 → Q 路，比特 b 映射为
    2b-1（0 → -1、1 → +1），每符号携带 2 bit。成型使用 16 符号跨度、β=0.5 的 RRC
    滤波器，接收端使用同一滤波器做匹配滤波，级联后为升余弦响应、满足无 ISI 条件。

    关键属性：
        Baud: int，符号率（Hz），如 25 kHz。
        Fs: int，采样率（Hz），如 10 MHz。
        Ns: int，每符号采样数 = Fs // Baud，如 400。
        Nh: int，每跳符号数（Baud / hoprate，由 _apply_hoprate 更新）；本文件中
            仅作记录/诊断，不参与信号合成。
        LBF: np.ndarray，发送端 RRC 系数，长度 16*Ns+1，单位能量。

    注意：调制器本身不引入随机性——随机性只来自 generate_bits 与信道侧。
    """

    def __init__(self, Baud, Fs, Ns, Nh):
        """
        Args:
            Baud: int，符号率（Hz）。
            Fs: int，采样率（Hz）。
            Ns: int，每符号采样数（上采样因子，也是接收端降采样因子）。
            Nh: int，每跳符号数（仅记录）。
        """
        self.Baud = Baud
        self.Fs = Fs
        self.Ns = Ns
        self.Nh = Nh
        self.LBF = rcosdesign_srv(0.5, 16, Ns)

    def generate_bits(self, Bitrate, rng=None):
        """
        生成 0/1 等概率随机比特。

        Args:
            Bitrate: int，比特数（名字沿用"比特率"，语义实为长度）。
            rng: np.random.RandomState 或 None；None → 全局 np.random。
                训练主线程一律传入由种子派生的局部 rng，保证多线程可复现。

        Returns:
            np.ndarray，形状 (Bitrate,)，0/1 等概率（二项分布 p=0.5）。
        """
        rng = np.random if rng is None else rng
        return rng.binomial(n=1, p=0.5, size=Bitrate)

    def pulse_shape(self, bits):
        """
        把比特流映射成 QPSK 的 I/Q 基带波形（上采样 + RRC 成型）。

        流程：奇偶拆分 → ±1 映射 → upfirdn 上采样 Ns 倍并卷积 RRC → 按群延迟 gd
        截取 n_syms*Ns 个样本，使波形起点与第一个符号对齐（这正是 rcosdesign_srv
        要求滤波器长度为奇数、群延迟可被 Ns 整除的原因）。

        Args:
            bits: array-like[int]，比特序列；长度为奇数时末尾补一个 0，
                以保证 I/Q 两路符号数相同。

        Returns:
            tuple:
                I_pulse: np.ndarray，形状 (n_syms*Ns,)，float64，同相分量。
                Q_pulse: np.ndarray，形状 (n_syms*Ns,)，float64，正交分量。
                其中 n_syms = ceil(len(bits)/2)。
        """
        bits = np.asarray(bits).astype(np.int8)
        if len(bits) % 2 != 0:
            bits = np.concatenate([bits, np.array([0], dtype=np.int8)])

        I_bits = 2 * bits[0::2] - 1
        Q_bits = 2 * bits[1::2] - 1
        n_syms = len(I_bits)

        I_f = upfirdn(self.LBF, I_bits, up=self.Ns)
        Q_f = upfirdn(self.LBF, Q_bits, up=self.Ns)
        # 群延迟补偿：截掉滤波器拖尾，只保留 n_syms*Ns 个与符号对齐的样本
        gd = (len(self.LBF) - 1) // 2
        I_pulse = I_f[gd:gd + n_syms * self.Ns]
        Q_pulse = Q_f[gd:gd + n_syms * self.Ns]
        return I_pulse, Q_pulse


class FHSSChannel:
    """
    跳频信道模型：按 hop 序列生成复载波，并叠加 AWGN（可选 Rayleigh 衰落由环境侧完成）。

    频点约定：信道 k 的中心频率为 Startfre + k*Sub_interval + 0.5*Sub_interval，
    即把 [Startfre, Endfre] 等分成 num_channels 个 Sub_interval 宽的子信道并取各自
    中心。收发两端使用同一 hop_seq，因此跳频本身不引入相位/频率误差。

    相位连续性（关键设计）：载波相位按 cumsum(hop_fre)/Fs 累积（相位累加器），
    而不是 2π*f_c*t。这样跳频瞬间相位连续，不会产生人为的宽带频谱泄漏——否则
    每次换频都会在 PSD 观测上留下冲激，干扰策略会学到伪特征。

    关键属性：
        Startfre: float，频带下限（Hz）。
        Sub_interval: float，子信道间隔（Hz）。
        Hoprate: float，跳速（Hz）；由 FHSSQPSKEnv._apply_hoprate 同步更新。
        Fs: float，采样率（Hz）。

    注意：transmit() 提供"基带×载波 + AWGN"的最小参考实现；实际训练路径
    （FHSSQPSKEnv._assemble_signal_block）为性能内联了同样的运算并额外支持
    Rayleigh 衰落与载波模板缓存，两者数学上等价。
    """

    def __init__(self, Startfre, Sub_interval, Hoprate, Fs):
        """
        Args:
            Startfre: float，频带下限（Hz）。
            Sub_interval: float，子信道间隔（Hz），如 50 kHz。
            Hoprate: float，跳速（Hz）；之后由 _apply_hoprate 按动作更新。
            Fs: float，采样率（Hz）。
        """
        self.Startfre = Startfre
        self.Sub_interval = Sub_interval
        self.Hoprate = Hoprate
        self.Fs = Fs

    def hop_carrier(self, t, hop_seq):
        """
        按 hop_seq 生成逐样本的复载波 exp(j*phi)（相位连续）。

        Args:
            t: array-like，时间轴；实现上只用 len(t) 作为样本数（相位由内部累加器
                给出，不读取 t 的具体数值）。
            hop_seq: array-like[int]，hop 序号序列（信道索引，不含 Startfre 偏移）。

        Returns:
            np.ndarray，形状 (N,)，dtype=complex128，单位幅度载波
            exp(j*2π*Σf/Fs)；hop_seq 为空时返回空数组。

        实现说明：每个 hop 占 round(Fs/Hoprate) 个样本，逐段填充 hop_idx（每段至少
        推进 1 个样本以避免死循环，最后一段补满剩余样本）；hop_seq 用尽后沿用最后
        一个频点，保证整段信号都有载波。
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

        # 信道 k 的中心频率 = 下限 + k*子信道间隔 + 半个子信道（与接收端约定一致）
        hop_fre = self.Startfre + hop_idx * self.Sub_interval + 0.5 * self.Sub_interval
        # 相位累加器：cumsum 而非 f_c*t，保证换频时相位连续（无频谱泄漏）
        phase = 2 * np.pi * np.cumsum(hop_fre) / self.Fs
        # 直接合成复指数（比 cos/sin 分开算快，且与接收端共轭相乘天然配对）
        carrier_complex = np.exp(1j * phase)
        return carrier_complex

    def transmit(
        self,
        I_pulse,
        Q_pulse,
        carrier_complex,
        noise_std=0.1,
        rng=None,
    ):
        """
        发送一个 block：基带乘复载波得到带通信号，并生成同长度 AWGN。

        Args:
            I_pulse: array-like，基带同相波形。
            Q_pulse: array-like，基带正交波形（与 I 等长）。
            carrier_complex: array-like[complex]，逐样本载波 exp(j*phi)，
                必须与基带等长（来自 hop_carrier）。
            noise_std: float，AWGN 标准差（噪声功率 = noise_std^2）；默认 0.1 对应
                噪声功率 0.01，与 settings.ENV_CONFIG 的 signal_power=0.0025
                （Baud/Fs 的理论值）一起决定 SNR ≈ -6 dB。
            rng: np.random.RandomState 或 None；None → 全局 np.random。

        Returns:
            tuple:
                rf_complex: np.ndarray (N,)，dtype=complex，baseband*carrier；取实部
                    即为物理带通信号（本函数不取实部，交由调用方决定）。
                noise: np.ndarray (N,)，实数 AWGN 样本（未缩放之外的任何处理）。

        注意：噪声不在这里相加——调用方在逐 hop 叠加 Rayleigh 衰落之后再加噪声，
        以保证每个 block 的噪声都是一次独立实现。
        """
        # carrier_complex 为 exp(j*phi)，直接复数相乘即完成上变频
        baseband = I_pulse + 1j * Q_pulse
        rf_complex = baseband * carrier_complex
        if rng is None:
            noise = noise_std * np.random.randn(len(rf_complex))
        else:
            noise = noise_std * rng.standard_normal(len(rf_complex))
        return rf_complex, noise


class QPSKReceiver:
    """
    QPSK 接收机（理想同步）：共轭下变频 + RRC 匹配滤波 + 降采样 + 硬判决。

    假设收发共享同一跳频序列，载波完全对齐（不含频偏/相偏估计与定时恢复），
    匹配滤波器与发送成型滤波器同为 RRC，级联后为升余弦、无 ISI。因此 BER 完全
    由干扰功率、AWGN 与 Rayleigh 衰落决定，是"抗干扰策略好坏"的干净度量。

    关键属性：
        Ns: int，每符号采样数（匹配滤波后的降采样因子）。
        MF: np.ndarray，匹配滤波 RRC 系数，长度 16*Ns+1，单位能量。

    BER 定义：逐 bit 硬判决与发送比特的 Hamming 距离均值，取值 [0, 1]，计算前
    按双方最短长度截断。
    """

    def __init__(self, Ns):
        """
        Args:
            Ns: int，每符号采样数；接收端用同一 Ns 做 upfirdn 降采样。
        """
        self.Ns = Ns
        self.MF = rcosdesign_srv(0.5, 16, Ns)

    def demodulate(self, modu_signal, bits, carrier_complex):
        """
        解调一个 block，返回判决比特与 BER。

        Args:
            modu_signal: array-like，接收实数带通信号（有用信号 + 干扰 + AWGN），长度 N。
            bits: array-like[int]，本 block 发送比特（BER 地面真值）。
            carrier_complex: array-like[complex]，发送端逐样本载波 exp(j*phi)，
                与 modu_signal 等长。

        Returns:
            tuple:
                receive_data: np.ndarray，形状 (2*min_len,)，dtype=int，判决比特
                    （偶数下标 → I 路，奇数下标 → Q 路）。
                bit_error: float，BER = mean(receive_data != bits[:2*min_len])；
                    min_len = min(I 判决数, Q 判决数, len(bits)//2)，min_len==0 时
                    返回 0.0（例如 hop_seq 为空导致没有载波）。

        实现说明：upfirdn(..., up=1, down=Ns) 一次完成匹配滤波与 Ns 倍降采样；
        抽样起点按群延迟换算 start_idx = gd // Ns，与发送端 gd 补偿互为逆操作；
        硬判决阈值为 0（>=0 → 1），与 2b-1 映射一致。
        """
        # 下变频：乘发送载波的共轭（理想同步，无相位误差）
        demod_complex = modu_signal * np.conj(carrier_complex)

        # 匹配滤波 + Ns 倍降采样
        y_complex = upfirdn(self.MF, demod_complex, up=1, down=self.Ns)

        # 匹配滤波群延迟换算成降采样后的符号起点偏移
        gd = (len(self.MF) - 1) // 2
        start_idx = gd // self.Ns
        
        y_complex = y_complex[start_idx:]

        y_i_end = np.real(y_complex)
        y_q_end = np.imag(y_complex)

        # 硬判决：>=0 → 1（阈值为 0，与发送端 2b-1 的 ±1 映射一致）
        y_i_end = (y_i_end >= 0).astype(int)
        y_q_end = (y_q_end >= 0).astype(int)

        # 按最短长度对齐，避免 hop 段不足或滤波拖尾造成的长度不匹配
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
    """
    FHSS/QPSK 抗干扰环境（Gymnasium 接口）。

    任务形态：一次 env.step() 顺序执行 NUM_BLOCKS=10 个 100 ms 的通信 block（合计
    1 s），动作给出 10 个"起始信道 offset"（每个 block 一个，可学习跳速也可由
    baseline 固定）。奖励逐 block 计算后取均值，因此 agent 学的是"在给定的干扰
    时频结构下，如何把跳频图案平移到最空闲的频段"。

    观测（observation）：100 ms 观测窗（只含干扰 + AWGN，不含有用信号）的 PSD
    waterfall，形状默认 (100, 100)（时间 bin × 频率 bin，1 ms × 10 kHz），单位 dB；
    observation_space 在 reset() 时按实际形状重建。

    动作（action）：dict {"hoprate": float, "offsets": int[num_blocks]}；也兼容
    (hoprate, offsets) 元组或标量（此时 offsets 取全 0）。offsets 必须是
    [0, num_channels-1] 的整数值，否则 step() 抛 ValueError。

    奖励（reward）：reward_i = base_reward - ber_penalty*BER_i - hoprate_penalty*hoprate
    （系数见 settings.REWARD_CONFIG），step 返回其均值；terminated/truncated 恒为
    False（持续性任务，由外部训练循环控制 episode 长度）。

    关键属性：
        num_channels: int，子信道数 = round((Endfre-Startfre)/Sub_interval)。
        num_blocks: int，= NUM_BLOCKS = 10。
        _block_len: int，每 block 样本数 = _num_syms_block * Ns（use_pregen 时以
            pregen_data.block_len 为准）。
        current_hoprate: float，本 step 实际生效的跳速（已量化到 10 Hz 网格）。
        mseq_channels: np.ndarray (mseq_length,), int，m 序列状态对 num_channels
            取模后的跳频基序列。
        _mseq_ptr: int，下一个 block 从基序列的哪个位置开始取 hop。
        jammer_ptr: int，干扰机的连续 RF 采样时钟（跨 block/step 累加，reset 归零）；
            扫频/梳状干扰的位置与相位都由它决定。
        _rng: np.random.RandomState，环境级主随机流，只由主线程推进，用于派生
            每 block / 每次观测的子流种子（见 _draw_substream_seeds）。
        pregen_data: PreGeneratedData 或 None（use_pregen=False 时）。
        reactive / sweep: ReactiveJammer / IndiscriminateJammer 或 None。
        _carrier_templates: 载波模板缓存 (num_channels, max_hop_length) complex64，
            只读、可重建，因此不参与序列化。

    不变量：
        - 每个 block 的 AWGN、每个 hop 的 Rayleigh 衰落、每次观测的 AWGN 都必须
          重新生成；只有 QPSK 基带（use_pregen）与干扰波形池可复用。
        - 所有会推进随机/时间状态的操作（jammer_ptr、_mseq_ptr、_rng）只在主线程
          执行，worker 线程只做无状态计算（见 _process_block_task）。
        - self.state 始终是最近一次观测的 float32 副本，供 render() 与外部读取。
    """

    metadata = {"render.modes": ["human"]}
    AUTO_THREAD_MIN_BLOCK_SAMPLES = 100_000

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
                 signal_power=0.0025,
                 block_workers=None,
                 jammer_config=None):
        """
        构造环境：建立收发链路、干扰机、跳频基序列与动作/观测空间。

        参数分组说明（全部有默认值，正常训练由 settings.ENV_CONFIG 传入）：
            射频/带宽：Startfre、Endfre（Hz）、Sub_interval（子信道间隔 Hz）、Fs（采样率 Hz）。
                子信道数 num_channels = round((Endfre-Startfre)/Sub_interval)，信道 k 的
                中心频率为 Startfre + k*Sub_interval + 0.5*Sub_interval。
            调制/跳速：Baud（符号率 Hz）、Hoprate（基准跳速 Hz，reset 时恢复）、
                hoprate_min/hoprate_max（动作 hoprate 的裁剪区间，hoprate_max=None → Baud）。
            观测分辨率：dt（s）、df（Hz），共同决定观测形状 (0.1/dt, (Endfre-Startfre)/df)。
            干扰机：enable_reactive 与 reactive_*（反应式，能量检测）、enable_sweep 与
                sweep_*/sweep_mode（无差别扫频/梳状/both）；其中部分参数会被
                jammer_config['reactive'] / ['sweep'] 覆盖（见 jammers.py）。
            衰落：enable_rayleigh、rayleigh_coherence（相干长度，样本数）。
            跳频序列：mseq_length、mseq_nbits、mseq_taps、mseq_seed（LFSR 配置）。
            加速/调试：use_pregen、debug_plot_psd、debug_log_hops、reset_mseq_each_step。
            数值：noise_std（AWGN 标准差）、signal_power（衰落前理论信号功率，供反应式
                干扰机推导检测 SNR，应等于 Baud/Fs）。
            并行：block_workers（None → 自动判定，见 _resolve_block_workers）。
            配置覆盖：jammer_config（None → settings.JAMMER_CONFIG，内部深拷贝）。

        副作用：构造过程中会预生成干扰波形池（use_pregen 且 enable_sweep 时）、
        PreGeneratedData，并打印若干初始化信息；不启动训练循环。

        Raises:
            TypeError: jammer_config 不是 dict 时。
            ValueError: 干扰机配置不合法时（如 sweep step/dwell 非正、comb 的
                switch_interval 不是 1 ms 的整数倍、comb 信道索引越界）。
        """
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

        # 子信道数：把 [Startfre, Endfre] 按 Sub_interval 均分；max(1, ...) 兜底，
        # 避免配置过窄时得到 0 个信道（会让动作空间与取模运算失效）
        self.num_channels = int(round((self.Endfre - self.Startfre) / self.Sub_interval))
        self.num_channels = max(1, self.num_channels)
        self.num_blocks = NUM_BLOCKS
        # 100 ms block 的符号数与样本数；use_pregen 时实际长度以 pregen_data.block_len 为准
        self._num_syms_block = int(round(self.Baud * 0.1))
        self._block_len = self._num_syms_block * self.Ns

        # block 并行：requested 保留原始请求值（序列化后按它重建），executor 惰性创建
        self._block_workers_requested = block_workers
        self.block_workers = self._resolve_block_workers(block_workers)
        self._executor = None

        # 载波模板缓存（按信道 × 跳长索引）；属可重建的运行时缓存，序列化时丢弃
        self._carrier_templates = None
        self._carrier_template_key = None
        self._carrier_template_fallback_key = None

        # 从 NumPy 全局流中抽取一个环境级主随机流种子。只有主线程推进它；
        # worker 只接收整数种子、各自新建 RandomState，绝不触碰共享随机状态，
        # 因此并行计算不改变结果分布。
        self._random_seed = int(
            np.random.randint(0, np.iinfo(np.int32).max)
        )
        self._rng = np.random.RandomState(self._random_seed)

        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max) if hoprate_max is not None else float(Baud)
        self.base_hoprate = float(Hoprate)

        # Nh 在此只是占位（每跳符号数），__init__ 末尾的 _apply_hoprate 会按基准跳速重算
        self.Nh = 200 if (200 % 2 == 0) else 202
        self.current_hoprate = float(Hoprate)
        self.modem = QPSKModem(self.Baud, self.Fs, self.Ns, self.Nh)
        self.channel = FHSSChannel(self.Startfre, self.Sub_interval, self.current_hoprate, self.Fs)
        self.receiver = QPSKReceiver(self.Ns)
        
        self.enable_reactive = bool(enable_reactive)
        self.enable_sweep = bool(enable_sweep)
        self.use_pregen = bool(use_pregen)

        # 加载配置
        # 干扰机配置在本环境实例内深拷贝一份：命令行实验可覆盖参数而不污染
        # settings 的全局配置（同一进程内可并存多个不同配置的环境实例）。
        j_conf = copy.deepcopy(
            settings.JAMMER_CONFIG if jammer_config is None else jammer_config
        )
        if not isinstance(j_conf, dict):
            raise TypeError("jammer_config must be a mapping when provided.")
        self.jammer_config = j_conf
        self.noise_std = float(noise_std)
        self.signal_power = float(signal_power)

        # 带限噪声基带变体数量：相同带宽的 reactive/sweep/comb 共享同一波形池，
        # 按变体随机有放回选择，使预生成干扰波形本身也具备多样性
        self.baseband_variant_count = validate_baseband_variant_count(
            j_conf.get('baseband_variant_count', 4)
        )
        (
            self.jammer_variant_pools,
            self.jammer_variant_selector,
        ) = self._build_jammer_variant_resources(j_conf)

        # 反应式干扰机：以能量检测（Urkowitz）建模，按 1 ms 时隙 scan→detect→jam。
        # variant_pool 以带宽为键取自 jammer_variant_pools，故与同带宽的扫频/梳状
        # 干扰共享波形池；noise_source=None 表示不外部注入噪声源，由内部生成。
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
                                       
        # 无差别干扰机（扫频/梳状/both）。构造前先严格校验 comb 信道索引，
        # 让配置错误在环境创建时立刻暴露，而不是训练中途才崩。
        # 波形池只在 use_pregen 且对应模式启用时传入；defer_dynamic_noise=use_pregen
        # 表示预生成路径下把动态噪声的生成推迟到取用时刻（避免白算）。
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

        # m 序列：先由 LFSR 生成状态序列，再对 num_channels 取模得到跳频基序列。
        # 基序列本身是确定性的（固定 mseq_seed），随机性只体现在 offset 动作上。
        self.mseq_states = generate_mseq_states(n_bits=mseq_nbits,
                                                length=mseq_length,
                                                taps=mseq_taps,
                                                seed=mseq_seed)
        self.mseq_channels = (self.mseq_states % self.num_channels).astype(int)
        self._mseq_ptr = 0
        self.reset_mseq_each_step = bool(reset_mseq_each_step)

        # 初始化预生成器：干扰波形池必须在使用前 precompute（get_composite_signal
        # 依赖它，缺失会抛 RuntimeError）
        if self.use_pregen and self.enable_sweep and self.sweep is not None:
             self.sweep.precompute(Startfre=self.Startfre, Endfre=self.Endfre)

        self.pregen_data = None
        if self.use_pregen:
            self.pregen_data = PreGeneratedData(self)
        
        self.jammer_ptr = 0  # 无差别干扰机的连续 RF 采样时钟（reset 时归零）

        # 动作空间：hoprate 为 1 维连续 Box（实际会被量化到 10 Hz 网格），
        # offsets 为 10 维 MultiDiscrete，每维 num_channels 个取值
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

        # 观测空间先给占位形状 (1, 1)，reset() 会按真实 waterfall 形状重建
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(1, 1), dtype=np.float32
        )

        self.state = None
        self.last_info = {}

        self.debug_plot_psd = bool(debug_plot_psd)
        self.debug_log_hops = bool(debug_log_hops)
        self.current_step = 0

        # 按训练 step 触发出图（见 enable_step_figure_capture）
        self._fig_save_steps = set()
        self._fig_save_dir = None

        # 按基准跳速初始化跳速相关状态（Nh / current_hoprate / channel.Hoprate）
        self._apply_hoprate(self.base_hoprate)

    def _resolve_block_workers(self, block_workers):
        """
        解析 block 并行度：None → 自动判定，显式值 → 校验后原样使用。

        自动判定逻辑（为什么这样选）：
            - 只有 1 个 block，或单 block 样本数小于 AUTO_THREAD_MIN_BLOCK_SAMPLES
              （100k 样本 ≈ 10 ms @10 MHz）时退化为单线程：任务太小的话线程调度与
              种子开销超过收益。
            - 否则取 min(10, CPU 核数, num_blocks)：任务数就是 block 数，多开无益；
              上限 10 是经验值，避免与外部训练进程/数据加载争抢 CPU。
            - 之所以能安全用线程：block 计算几乎全在 numpy 的 FFT/upfirdn 中（释放
              GIL），且每个 block 使用独立的 RandomState，无共享可变状态。

        Args:
            block_workers: None 或正整数。

        Returns:
            int，>= 1 的 worker 数。

        Raises:
            ValueError: block_workers 为 bool、非整数或 <= 0 时。
        """
        if block_workers is None:
            if (
                self.num_blocks < 2
                or self._block_len < self.AUTO_THREAD_MIN_BLOCK_SAMPLES
            ):
                return 1
            return max(1, min(10, os.cpu_count() or 1, self.num_blocks))
        if (
            isinstance(block_workers, bool)
            or not isinstance(block_workers, (int, np.integer))
            or int(block_workers) <= 0
        ):
            raise ValueError("block_workers must be None or a positive integer.")
        return int(block_workers)

    def _get_executor(self):
        """
        惰性获取 block 线程池；单线程配置返回 None（调用方走串行路径）。

        线程池只在第一次真正并行时创建，避免"构造了环境却没用"的场景白占线程；
        返回 None 让 _run_block_tasks 无需分支判断配置，只判断返回值。
        """
        if self.block_workers <= 1:
            return None
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=self.block_workers,
                thread_name_prefix="fhss-block",
            )
        return self._executor

    def close(self):
        """
        释放惰性创建的 block 线程池（可重复调用）。

        先取出并置空引用再 shutdown，保证重入安全（__del__ 与显式 close 可能都会
        触发）；wait=True 等待在途 block 计算结束，避免线程在解释器退出时被强杀。
        最后调用父类 close 以维持 Gymnasium 生命周期约定。
        """
        executor = getattr(self, "_executor", None)
        self._executor = None
        if executor is not None:
            executor.shutdown(wait=True)
        super().close()

    def __getstate__(self):
        """
        序列化前的状态整理：剔除不可 pickle 与可重建的大对象。

        为什么需要它：ThreadPoolExecutor 内部持有锁与线程句柄，无法（也不应）
        pickle；载波模板是纯派生缓存且尺寸为 num_channels × max_hop_length 的
        complex64，写入 checkpoint 会显著膨胀文件。二者都在 __setstate__ 中重建，
        因此丢弃它们不改变任何行为。
        """
        state = self.__dict__.copy()
        state["_executor"] = None
        # 载波模板是可重建的运行时缓存且可能很大，故不写入序列化状态
        state["_carrier_templates"] = None
        state["_carrier_template_key"] = None
        state["_carrier_template_fallback_key"] = None
        return state

    def __setstate__(self, state):
        """
        反序列化：重建运行时缓存，并兼容旧 checkpoint 缺失字段的情况。

        兼容性处理（为什么有 hasattr 分支）：
            - 旧 checkpoint 可能没有 _block_workers_requested（并行度字段是后加的），
              此时按单线程恢复，保证能加载而不是直接崩。
            - 旧 checkpoint 可能没有 _rng（环境级随机流是后加的），此时重新从全局流
              抽一个种子建 RandomState：旧环境的随机流无法复原，只能重新播种。
        """
        self.__dict__.update(state)
        self._executor = None
        self._carrier_templates = None
        self._carrier_template_key = None
        self._carrier_template_fallback_key = None
        if not hasattr(self, "_block_workers_requested"):
            self._block_workers_requested = 1
        self.block_workers = self._resolve_block_workers(
            self._block_workers_requested
        )
        if not hasattr(self, "_rng"):
            self._random_seed = int(
                np.random.randint(0, np.iinfo(np.int32).max)
            )
            self._rng = np.random.RandomState(self._random_seed)

    def __del__(self):
        """析构兜底：释放线程池且不抛异常（解释器退出期安全）。"""
        # 解释器退出阶段可能已开始回收模块属性，故吞掉一切异常
        try:
            self.close()
        except Exception:
            pass

    def _draw_substream_seeds(self, shape):
        """
        从环境级主随机流派生整数子流种子（约定：只能由主线程调用）。

        每个 block/每次观测都拿到互不相同的种子，再各自新建 RandomState，这样
        并行执行顺序不影响结果——这正是"可复现"的关键机制。

        Args:
            shape: int 或 tuple，需要的种子数量/形状。

        Returns:
            np.ndarray 或 np.int64，dtype=int64，取值在 [0, 2^31-1)，
            可直接作为 np.random.RandomState 的种子。
        """
        return self._rng.randint(
            0,
            np.iinfo(np.int32).max,
            size=shape,
        )

    def _build_jammer_variant_resources(self, jammer_config):
        """
        按配置为每种干扰带宽创建一个带限噪声变体池，并创建一个变体选择器。

        为什么需要"池 + 需求长度"：预生成路径下干扰波形是整段拼接好的缓冲
        （get_composite_signal 按采样索引切片），因此池中每条波形必须至少覆盖干扰
        自身的一个完整周期，否则切片会越界或出现不连续跳变：
            - 反应式干扰：一个检测/压制时隙 detection_time * Fs 个样本。
            - 扫频干扰：samples_per_dwell * num_steps，即扫完整个带宽所需的样本数
              （num_steps = floor((Endfre-Startfre)/step)）。
            - 梳状干扰：2 * switch_interval * Fs，即两个相位各一个切换周期。
        同一带宽若被多个干扰机使用，取需求最大值（require_pool 的 max 语义），
        这样多个干扰机可共享同一条波形缓冲。

        Args:
            jammer_config: dict，本环境实例的干扰机配置（已深拷贝）。

        Returns:
            tuple:
                pools: dict {bandwidth(float): BandLimitedNoiseVariantPool}；
                    无任何需求时返回空 dict。
                selector: JammerVariantSelector 或 None；无需求时为 None，
                    此时所有干扰机都只用 0 号变体（等效于无变体随机性）。

        Raises:
            ValueError: mode 非 'sweep'/'comb'/'both'；sweep 的 step/dwell_time 非有限
                正数；comb 的 switch_interval 非 1 ms（COMB_TIME_QUANTUM）的整数倍。
        """
        requirements = {}

        def require_pool(bandwidth, min_samples):
            """
            登记某带宽所需的波形池样本数（同带宽取最大值）。

            Args:
                bandwidth: float，干扰带宽（Hz），作为波形池的键。
                min_samples: int，该干扰机一个完整周期所需的样本数（至少 1）。
            """
            # 同带宽取最大需求：多个干扰机共享同一波形池时必须满足最长的那个
            bandwidth = float(bandwidth)
            min_samples = max(1, int(min_samples))
            requirements[bandwidth] = max(
                requirements.get(bandwidth, 0),
                min_samples,
            )

        if self.enable_reactive:
            reactive_config = jammer_config['reactive']
            # 反应式干扰按 1 ms 基本时隙循环，池至少覆盖一个时隙
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
                # 池长度 = 扫完整个带宽（num_steps 个频点、每点 dwell）的样本数
                require_pool(
                    sweep_config['bandwidth'],
                    samples_per_dwell * num_steps,
                )

            if mode in {'comb', 'both'}:
                comb_config = jammer_config['comb']
                switch_interval = float(comb_config.get('switch_interval', 0.05))
                # 相位切换必须对齐到 1 ms 时隙（COMB_TIME_QUANTUM）：否则相位切换
                # 会落在时隙内部，预生成缓冲与 comb_phase_at 的相位推算将不一致
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
                        "positive multiple of 0.001 seconds (1 ms)."
                    )
                # 池长度 = 两个相位各一个切换周期（相位 0/1 交替占用同一缓冲）
                require_pool(
                    comb_config['bandwidth'],
                    2 * int(round(switch_interval * self.Fs)),
                )

        if not requirements:
            # 未启用任何干扰机：返回空池与 None 选择器，调用方据此跳过预生成
            return {}, None

        # 单一选择器被所有干扰机共享，使变体切换流在同一时间线上可复现
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
        严格校验可配置的梳状干扰信道索引分组。

        ``channels_phase0`` / ``channels_phase1`` 的每个元素都必须是
        ``[0, num_channels - 1]`` 内的整数，否则抛 ValueError（含 bool——bool 是 int
        的子类，必须显式排除，否则 True 会被当成信道 1）。两组长度可以不同、允许
        重叠；缺省键回退到干扰机内置默认值，此处不校验。

        之所以在环境构造期就校验：信道索引越界只会在干扰波形生成时才暴露，若不在
        启动时拦截，训练可能跑很久之后才崩，且难以定位。

        Args:
            comb_config: dict，JAMMER_CONFIG['comb']（或等价覆盖配置）。

        Raises:
            ValueError: 分组不是 list/tuple、元素非整数、或索引越界时。
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
        开启"在指定训练 step 保存逐 block PSD 图"的调试功能。

        触发条件在 step() 内判定：capture_figures = (current_step + 1) in save_steps，
        即用 1-based 的 step 序号匹配训练日志里的 "Step i/N"。启用后该 step 会额外
        计算每个 block 的 waterfall（compute_block_waterfalls），代价是 10 次额外
        FFT，因此只在少数 step 上开启。

        Args:
            save_steps: 可迭代对象，1-based 的训练 step 序号集合（与日志中
                ``Step i/N`` 一致，即 step() 内的 ``self.current_step + 1``）。
            save_dir: str，图片输出目录；不存在时自动创建。文件名为
                ``step_XXX_block_YY.png``（XXX 为 3 位 step、YY 为 2 位 block）。
        """
        self._fig_save_steps = set(int(s) for s in save_steps)
        self._fig_save_dir = str(save_dir)
        os.makedirs(self._fig_save_dir, exist_ok=True)

    def seed(self, seed=None):
        """
        播种随机性（Gymnasium 旧接口风格，reset(seed=...) 会转调此处）。

        同时设置 NumPy 全局种子与环境的 _rng：前者影响未显式传入 rng 的路径
        （如干扰机内部默认噪声、PreGeneratedData 的比特），后者影响每 block / 每次
        观测派生的子流。注意预生成干扰波形池在 __init__ 时已生成，播种不会重造它们。

        Args:
            seed: int 或 None；None 时从全局流随机取一个。

        Returns:
            list[int]，实际使用的种子（长度 1），便于日志记录与复现。
        """
        if seed is None:
            seed = int(np.random.randint(0, np.iinfo(np.int32).max))
        seed = int(seed)
        np.random.seed(seed)
        self._random_seed = seed
        self._rng = np.random.RandomState(seed)
        return [seed]

    def _apply_hoprate(self, hoprate_target):
        """
        把动作 hoprate 落地为环境状态：裁剪 → 量化 → 更新 Nh / current_hoprate。

        两级处理（为什么）：
            1. 裁剪到 [hoprate_min, hoprate_max]：策略输出可能越界，裁剪保证物理可行。
            2. 量化到 10 Hz 网格 int(round(x/10))*10：跳速只能取 10 Hz 的整数倍，
               使不同实现（真实环境、离线 replay、hoprate sweep）对同一动作得到完全
               相同的仿真，避免浮点抖动导致的数据/策略不一致。
        由此 hops_per_block = round(current_hoprate * 0.1) 恒为整数（10 Hz 网格的
        直接收益：每个 100 ms block 的 hop 数不出现半跳）。

        Nh = round(Baud / hoprate) 为每跳符号数，取偶数（奇数则 +1）：保持 I/Q 成对
        的偶数符号数，避免下游按符号对分析/规划时出现半符号边界；本文件内 Nh 只同步
        到 modem.Nh 与 info，不参与信号合成。

        Args:
            hoprate_target: float，原始 hoprate 动作值（Hz）。

        Returns:
            dict: {"hoprate_action": 原始值, "hoprate_used": 量化后的值,
                   "Nh_used": 每跳符号数}；hoprate_used 用于奖励计算与 info 记录，
                   必须是量化后的值而非原始动作值。
        """
        hoprate_clip = float(np.clip(hoprate_target, self.hoprate_min, self.hoprate_max))
        # 量化到 10 Hz 网格：保证 100 ms block 的 hop 数为整数，且跨实现完全一致
        hoprate_used = float(int(round(hoprate_clip / 10.0)) * 10)

        # 每跳符号数取偶数（便于按 I/Q 符号对划分）
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

    def _generate_rayleigh(self, length, rng=None):
        """
        生成逐样本的 Rayleigh 衰落幅度序列（分块衰落模型）。

        模型：每 rayleigh_coherence 个样本为一个相干块，块内幅度恒定、块间独立，
        服从 Rayleigh(scale=sqrt(2)/2)。该 scale 使 E[R^2] = 2*scale^2 = 1，即衰落
        平均功率为 1，因此 Rayleigh 只引入幅度起伏而不改变平均 SNR——这样奖励的
        尺度不会随 enable_rayleigh 开关而漂移。

        Args:
            length: int，需要的样本数。
            rng: np.random.RandomState 或 None；None → 全局 np.random。
                注意本方法不做"每 hop 重新生成"的划分，hop 级独立性由调用方
                （_assemble_signal_block 逐 hop 调用并传同一 rayleigh_rng）保证。

        Returns:
            np.ndarray，形状 (length,)，float（长度不足一个相干块时按块截断），
            或 None（enable_rayleigh=False 或 length<=0，调用方据此跳过衰落）。
        """
        if (not self.enable_rayleigh) or length <= 0:
            return None
        coh = max(1, int(self.rayleigh_coherence))
        num_seg = int(np.ceil(length / coh))
        rng = np.random if rng is None else rng
        # scale=sqrt(2)/2 → E[R^2]=2*scale^2=1：衰落平均功率归一，不改变平均 SNR
        mags = rng.rayleigh(scale=np.sqrt(2) / 2, size=num_seg)
        # 块内复制实现"相干块"衰落：块间独立、块内恒定（长度不足时截断）
        mag_seq = np.repeat(mags, coh)[:length]
        return mag_seq

    def _hop_segment_lengths(self, length, n_hops):
        """
        把 length 个样本按当前跳速切成 n_hops 段的样本数列表（逐段至少 1 个样本）。

        约定（必须与 FHSSChannel.hop_carrier 的分段方式一致，否则收发载波会错位）：
            第 k 段结束于 round((k+1) * Fs/hoprate)，最后一段直接吃掉剩余全部样本。
        用"四舍五入到绝对位置"而不是"每段固定长度"可以避免逐段累计的取整误差；
        max(pos+1, ...) 保证任何配置下都不会产生 0 长度段（否则该 hop 的载波为空）。

        Args:
            length: int，block 样本数。
            n_hops: int，本 block 的 hop 数（= round(hoprate * 0.1)）。

        Returns:
            list[int]，各 hop 的样本数，sum(segments) <= length（当 hop 数多于样本数时
            可能提前终止）；length<=0 或 n_hops<=0 时返回 []。
        """
        if length <= 0 or n_hops <= 0:
            return []
        segments = []
        s_per_hop_float = self.Fs / float(self.current_hoprate)
        pos = 0
        for hop_idx in range(n_hops):
            if hop_idx == n_hops - 1:
                next_pos = length
            else:
                next_pos = int(round((hop_idx + 1) * s_per_hop_float))
                next_pos = min(length, max(pos + 1, next_pos))
            if next_pos > pos:
                segments.append(next_pos - pos)
            pos = next_pos
            if pos >= length:
                break
        return segments

    def _ensure_carrier_templates(self, max_hop_length):
        """
        缓存"每信道 × 最长 hop"的 complex64 载波模板，避免逐 hop 重算 exp。

        为什么可以缓存：载波模板只依赖 (max_hop_length, Fs, Startfre, Sub_interval,
        num_channels)，与随机性无关；同一跳速下每个 block 的 hop 长度分布相同，
        因此模板可跨 block、跨 step 复用。这消除了每 hop 一次 exp（最贵的逐样本
        运算），是 use_pregen 路径之外的另一处主要加速。

        相位约定：模板第 n 个元素（n 从 0 起）对应"块内相对样本序号 n+1"的相位，
        即 exp(j*2π*f_c*(n+1)/Fs)；段起点非零时由调用方乘一个相位偏移因子
        exp(j*2π*cumsum_end/Fs) 拼接，从而与 cumsum 相位定义严格等价。

        缓存键与失效：
            - 命中 _carrier_template_key → 返回缓存数组（只读）。
            - 命中 _carrier_template_fallback_key → 直接返回 None：该键此前已确定
              无法建模板（长度非法或 MemoryError），无需重试，避免反复失败的开销。
            - 键变化（如 hoprate 变化导致 max_hop_length 变化）→ 清空重建。

        Args:
            max_hop_length: int，本 step 中最长 hop 的样本数。

        Returns:
            np.ndarray (num_channels, max_hop_length) complex64 且 write=False，
            或 None（max_hop_length<=0 或内存不足时，调用方退化为逐 hop 现算）。
        """
        max_hop_length = int(max_hop_length)
        key = (
            max_hop_length,
            self.Fs,
            self.Startfre,
            self.Sub_interval,
            self.num_channels,
        )
        if key == self._carrier_template_key:
            return self._carrier_templates
        # 该键此前失败过（长度非法/内存不足），直接走退化路径，不重复尝试
        if key == self._carrier_template_fallback_key:
            return None

        self._carrier_templates = None
        self._carrier_template_key = None
        self._carrier_template_fallback_key = None
        if max_hop_length <= 0:
            self._carrier_template_fallback_key = key
            return None

        try:
            # 内存上限 O(num_channels × max_hop_length) 个 complex64（约 8 字节/点）
            templates = np.empty(
                (self.num_channels, max_hop_length),
                dtype=np.complex64,
            )
            sample_idx = np.arange(
                1,
                max_hop_length + 1,
                dtype=np.float64,
            )
            phase_k = 2.0 * np.pi / self.Fs
            for channel_idx in range(self.num_channels):
                f_c = (
                    self.Startfre
                    + channel_idx * self.Sub_interval
                    + 0.5 * self.Sub_interval
                )
                templates[channel_idx] = np.exp(
                    1j * (phase_k * f_c) * sample_idx
                ).astype(np.complex64)
        except MemoryError:
            self._carrier_template_fallback_key = key
            return None

        templates.setflags(write=False)
        self._carrier_templates = templates
        self._carrier_template_key = key
        return templates

    def _assemble_signal_block(
        self,
        I_pulse,
        Q_pulse,
        hop_seq,
        rng=None,
        rayleigh_rng=None,
        baseband=None,
    ):
        """
        组装一个 block 的接收信号：逐 hop 上变频 + Rayleigh 衰落 + AWGN。

        随机性边界（关键不变量）：本方法每次都重新生成 AWGN（整 block 一次）与
        Rayleigh 幅度（逐 hop 一次），因此即使 I/Q 与载波模板被复用，信道实现仍是
        独立的。传入 rng/rayleigh_rng 时用局部 RandomState，保证多线程可复现。

        逐 hop 处理的内容：
            1. 载波：优先用 _ensure_carrier_templates 的缓存模板；段起点非零时乘
               相位偏移 exp(j*2π*cumsum_end/Fs)，与 FHSSChannel 的 cumsum 相位严格
               等价（模板起点是段内第 1 个样本，见 _ensure_carrier_templates）。
            2. 衰落：逐 hop 调用 _generate_rayleigh（同一 rayleigh_rng 顺序取数），
               因此 hop 间衰落独立；enable_rayleigh=False 时跳过。
            3. 加噪声：rx = Re(baseband * rayleigh * carrier) + noise，噪声是整 block
               预先抽好的实数序列，逐 hop 切片相加。

        Args:
            I_pulse: array-like，基带 I 波形（形状需与 Q 相同）。
            Q_pulse: array-like，基带 Q 波形。
            hop_seq: array-like[int]，本 block 的 hop 信道索引序列；空序列表示无跳频
                载波（此时只输出噪声，载波全 0）。
            rng: np.random.RandomState 或 None，AWGN 来源（None → 全局 np.random）。
            rayleigh_rng: np.random.RandomState 或 None，Rayleigh 来源；None 时回退
                到 rng（用同一个流，顺序消耗）。
            baseband: 可选的预拼 complex64 解析信号（use_pregen 路径传入
                common_baseband）；形状必须与 I_pulse 一致。

        Returns:
            tuple:
                rx_assembled: np.ndarray (N,), float32，接收实数带通信号
                    （= 有用信号 + AWGN；干扰由调用方随后叠加）。
                carrier_assembled: np.ndarray (N,), complex64，逐样本载波；
                    无 hop 或无载波的尾部区域填 0，使接收端下变频结果为 0。

        Raises:
            ValueError: I/Q 形状不一致，或 baseband 形状与 I/Q 不匹配时。
        """
        I_pulse = np.asarray(I_pulse, dtype=np.float32)
        Q_pulse = np.asarray(Q_pulse, dtype=np.float32)
        if I_pulse.shape != Q_pulse.shape:
            raise ValueError("I_pulse and Q_pulse must have matching shapes.")

        if baseband is None:
            baseband = (I_pulse + 1j * Q_pulse).astype(np.complex64)
        else:
            baseband = np.asarray(baseband, dtype=np.complex64)
            if baseband.shape != I_pulse.shape:
                raise ValueError("Cached baseband shape does not match I/Q pulses.")
        N = len(baseband)
        # 整 block 一次抽 AWGN（独立实现，不复用），逐 hop 切片相加
        if rng is None:
            noise_samples = np.random.randn(N)
        else:
            noise_samples = rng.standard_normal(N)
        noise = (self.noise_std * noise_samples).astype(np.float32)
        rx_assembled = np.empty(N, dtype=np.float32)
        carrier_assembled = np.empty(N, dtype=np.complex64)

        n_hops = len(hop_seq)
        if N == 0:
            return rx_assembled, carrier_assembled
        if n_hops == 0:
            # 无 hop 序列：无载波也无有用信号，只剩噪声（载波置 0 → 解调输出 0）
            rx_assembled[:] = noise
            carrier_assembled[:] = 0.0
            return rx_assembled, carrier_assembled

        segment_lengths = self._hop_segment_lengths(N, n_hops)
        max_hop_length = max(segment_lengths, default=0)
        carrier_templates = self._ensure_carrier_templates(max_hop_length)
        two_pi_over_Fs = 2.0 * np.pi / self.Fs
        pos = 0
        # 相位累加器的"上一段末尾累计值"Σ f_c*length（单位 Hz·样本），
        # 用于保证跨 hop 相位连续，与 FHSSChannel.hop_carrier 的定义一致
        cumsum_end = 0.0

        for hop_idx, length in enumerate(segment_lengths):
            next_pos = pos + length
            # 取模防御：offset 已归一到 [0, num_channels)，此处再兜底防止越界索引
            ch_idx = int(hop_seq[hop_idx]) % self.num_channels
            f_c = (
                self.Startfre
                + ch_idx * self.Sub_interval
                + 0.5 * self.Sub_interval
            )
            if carrier_templates is None:
                # 退化路径：无模板缓存（长度非法/内存不足）时逐 hop 现算相位
                t_local = np.arange(1, length + 1, dtype=np.float64)
                phase = two_pi_over_Fs * (cumsum_end + f_c * t_local)
                carrier_hop = np.exp(1j * phase).astype(np.complex64)
            elif cumsum_end == 0.0:
                # 首段（或累计相位恰为 0）：模板即最终载波，无需相位偏移
                carrier_hop = carrier_templates[ch_idx, :length]
            else:
                # 非首段：模板是"从零相位起算"的，需乘段起点相位偏移
                phase_start = np.exp(1j * two_pi_over_Fs * cumsum_end)
                carrier_hop = (
                    carrier_templates[ch_idx, :length]
                    * np.complex64(phase_start)
                )
            cumsum_end += f_c * length

            # 逐 hop 取 Rayleigh 幅度（顺序消耗 rayleigh_rng，故 hop 间独立）
            effective_rayleigh_rng = (
                rayleigh_rng if rayleigh_rng is not None else rng
            )
            if effective_rayleigh_rng is None:
                rayleigh_mag = self._generate_rayleigh(length)
            else:
                rayleigh_mag = self._generate_rayleigh(
                    length,
                    rng=effective_rayleigh_rng,
                )
            if rayleigh_mag is None:
                modulated = np.real(
                    baseband[pos:next_pos] * carrier_hop
                )
            else:
                # 衰落加在基带上（等效于乘到载波上），再取实部得到物理带通信号
                rayleigh_mag = np.asarray(rayleigh_mag, dtype=np.float32)
                faded_baseband = baseband[pos:next_pos] * rayleigh_mag
                modulated = np.real(faded_baseband * carrier_hop)
            rx_assembled[pos:next_pos] = (
                modulated + noise[pos:next_pos]
            )
            carrier_assembled[pos:next_pos] = carrier_hop
            pos = next_pos
            if pos >= N:
                break

        # 尾部兜底：hop 段不足以覆盖整 block 时，剩余样本只含噪声、载波置 0
        if pos < N:
            rx_assembled[pos:] = noise[pos:]
            carrier_assembled[pos:] = 0.0
        return rx_assembled, carrier_assembled

    def _observe_100ms(self, block_id=None, rng=None, jammer_rng=None):
        """
        生成 100 ms 观测窗并返回其 PSD waterfall（策略看到的唯一输入）。

        观测内容 = 干扰 + AWGN（不含有用信号）：这是"监听频谱"的建模，agent 需要
        从干扰的时频结构推断哪些信道/时刻可用。观测窗与 block 等长（100 ms），
        因此每个决策的观测与它要决策的时间段尺度一致。

        时间线一致性（重要）：观测窗同样推进 jammer_ptr，且预生成路径用
        get_composite_signal(jammer_ptr, N_obs) 取连续切片，因此观测到的干扰与
        随后的通信 block 所经历的干扰在时间上是衔接的（干扰机是持续发射的）。

        Args:
            block_id: int 或 None，仅用于调试图标题（debug_plot_psd=True 时）。
            rng: np.random.RandomState 或 None，观测窗 AWGN 来源（None → 全局流）。
            jammer_rng: np.random.RandomState 或 None，非预生成路径下干扰波形生成
                所需（None 时回退到 rng）。

        Returns:
            np.ndarray，形状 (num_frames, n_bins)，默认 (100, 100)，dtype=float64，
            单位 dB；reset() 用它重建 observation_space。

        副作用：推进 self.jammer_ptr（+N_obs）；debug_plot_psd 时弹窗绘图。
        """
        N_obs = int(0.1 * self.Fs)
        sweep_jam = np.zeros(N_obs, dtype=np.float32)
        if self.enable_sweep and self.sweep is not None:
            if self.use_pregen:
                # 预生成路径：按连续采样索引切片，保证与 block 时间线无缝衔接
                sweep_jam = self.sweep.get_composite_signal(
                    self.jammer_ptr,
                    N_obs,
                )
            else:
                # 动态路径：现算干扰波形（代价高，仅调试/对照时使用）
                effective_jammer_rng = (
                    jammer_rng if jammer_rng is not None else rng
                )
                sweep_jam, _ = self.sweep.generate_samples(
                    N_obs,
                    self.Startfre,
                    self.Endfre,
                    start_sample_idx=self.jammer_ptr,
                    rng=effective_jammer_rng,
                )
            self.jammer_ptr += N_obs

        # 观测噪声每次重新生成（与 block 的 AWGN 独立）
        if rng is None:
            noise_samples = np.random.randn(N_obs)
        else:
            noise_samples = rng.standard_normal(N_obs)
        noise = (self.noise_std * noise_samples).astype(np.float32)
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

    def _process_block_task(self, task):
        """
        执行单个 block 任务中"无状态、CPU 密集"的部分（可安全并行）。

        并行安全的前提（不变量）：本方法只读取 self 的配置与只读缓存
        （modem/receiver/载波模板/干扰波形池），不推进任何时间线指针、不触碰
        self._rng；所有随机性来自 task 里的整数种子，各自新建局部 RandomState。
        因此任务之间无共享可变状态，worker 线程的执行顺序不影响结果。

        处理顺序：先合成有用信号（预生成或现算），再叠加扫频/梳状干扰、最后叠加
        反应式干扰，然后解调算 BER；仅当 compute_waterfall 时才额外算接收端 PSD
        （默认关闭，因为它会带来 10 倍 FFT 开销）。

        Args:
            task: dict，字段含义见 step() 中构造处：
                block_idx (int)、hop_seq (np.ndarray)、use_pre (bool)、
                sweep_jam (np.ndarray 或 None，预生成干扰切片)、
                dynamic_sweep (bool，是否在 worker 内现算扫频干扰)、
                jammer_start (int，现算干扰时的起始采样索引)、
                reactive_jam (np.ndarray 或 None)、
                bits_seed / noise_seed / rayleigh_seed / jammer_seed (int)、
                compute_waterfall (bool)。

        Returns:
            tuple: (ber (float), waterfall_db (np.ndarray (num_frames, n_bins) 或 None))。
            注意返回的 waterfall 由调用方（主线程）负责绘图——matplotlib 非线程安全。
        """
        # 每个子流独立播种：bits/noise/rayleigh/jammer 互不干扰，便于单独复现
        bits_rng = np.random.RandomState(int(task["bits_seed"]))
        noise_rng = np.random.RandomState(int(task["noise_seed"]))
        rayleigh_rng = np.random.RandomState(int(task["rayleigh_seed"]))
        jammer_rng = np.random.RandomState(int(task["jammer_seed"]))
        hop_seq = task["hop_seq"]

        if task["use_pre"]:
            rx_static, carrier_complex, bits_block = self.pregen_data.get_block(
                hop_seq,
                rng=noise_rng,
                rayleigh_rng=rayleigh_rng,
            )
        else:
            bits_block = self.modem.generate_bits(
                2 * self._num_syms_block,
                rng=bits_rng,
            )
            I_pulse, Q_pulse = self.modem.pulse_shape(bits_block)
            rx_static, carrier_complex = self._assemble_signal_block(
                I_pulse,
                Q_pulse,
                hop_seq,
                rng=noise_rng,
                rayleigh_rng=rayleigh_rng,
            )

        rx_real = rx_static
        # 扫频/梳状干扰：预生成路径直接切片相加；动态路径在 worker 内现算
        if task["dynamic_sweep"]:
            sweep_jam, _ = self.sweep.generate_samples(
                len(rx_real),
                self.Startfre,
                self.Endfre,
                start_sample_idx=task["jammer_start"],
                rng=jammer_rng,
            )
            rx_real += sweep_jam
        elif task["sweep_jam"] is not None:
            rx_real += task["sweep_jam"]

        # 反应式干扰在主线程按 hop_seq 生成（依赖时间线状态），此处只做叠加
        reactive_jam = task["reactive_jam"]
        if reactive_jam is not None:
            rx_real += reactive_jam

        # 解调只需要接收信号、发送比特与载波；返回的判决比特此处不使用
        _, ber = self.receiver.demodulate(
            rx_real,
            bits_block,
            carrier_complex,
        )

        waterfall_db = None
        if task["compute_waterfall"]:
            # 只在需要出图/调试时计算接收端 PSD（10 倍 FFT 开销），plot 恒为 False
            waterfall_db = compute_psd_waterfall(
                rx_real,
                fs=self.Fs,
                f_start=self.Startfre,
                f_end=self.Endfre,
                dt=self.dt,
                df=self.df,
                max_duration=0.1,
                plot=False,
            )
        return float(ber), waterfall_db

    def _run_block_tasks(self, tasks):
        """
        按顺序执行（或并行）block 任务列表，返回与输入同序的结果列表。

        单线程时直接串行列表推导；多线程时用 executor.map，其返回值顺序与输入任务
        顺序一致——这是关键：结果按 block 顺序对齐，BER/奖励的语义不依赖完成先后。

        Args:
            tasks: list[dict]，由 step() 构造，元素顺序即 block 顺序。

        Returns:
            list[tuple]，第 i 项对应 tasks[i] 的 (ber, waterfall_db)。
        """
        if self.block_workers <= 1:
            return [self._process_block_task(task) for task in tasks]
        executor = self._get_executor()
        return list(executor.map(self._process_block_task, tasks))

    def _get_block_hopseq(self, hops_per_block, offset):
        """
        取一个 block 的跳频序列：从 m 序列基序列连续取 hops_per_block 个 hop，
        再整体循环平移 offset 个信道。

        语义要点：
            - 平移是在"信道索引"上取模，等价于把整个跳频图案搬到另一段频谱，
              因此 offset 相同则图案形状相同、频率整体平移（这正是 agent 要学的
              抗干扰自由度：避开被压制的频段而不改变图案结构）。
            - 基序列按 _mseq_ptr 顺序推进，所以同一 step 内 10 个 block 共享一段
              连续的 m 序列（不会重复同一段图案）。
            - 越界时用"首尾拼接"实现环形缓冲，避免每次取序列都做一次 np.roll。

        Args:
            hops_per_block: int，本 block 的 hop 数（= round(current_hoprate * 0.1)）。
            offset: 任意实数形式的 offset（内部四舍五入为整数并取模）。

        Returns:
            np.ndarray，形状 (hops_per_block,)，dtype=int，取值 [0, num_channels-1]；
            hops_per_block<=0 时返回空数组。
        """
        if hops_per_block <= 0:
            return np.array([], dtype=int)

        # 环形读取：跨越序列末尾时拆成"尾部 + 头部"两段拼接
        end_ptr = self._mseq_ptr + hops_per_block
        if end_ptr <= len(self.mseq_channels):
            base = self.mseq_channels[self._mseq_ptr:end_ptr]
        else:
            part1 = self.mseq_channels[self._mseq_ptr:]
            part2 = self.mseq_channels[:(end_ptr - len(self.mseq_channels))]
            base = np.concatenate([part1, part2])
        
        # 四舍五入 + 取模：容忍传入浮点 offset，并保证结果落在合法信道范围
        off_int = int(np.round(offset)) % self.num_channels
        hop_seq = (base + off_int) % self.num_channels
        return hop_seq

    def reset(self, *, seed=None, options=None):
        """
        重置环境并返回初始观测。

        重置内容与理由：
            - current_step=0、jammer_ptr=0：把连续 RF 采样时钟归零，使扫频/梳状
              干扰的位置与相位回到确定性起点（它们都由采样索引唯一决定）。
            - _apply_hoprate(base_hoprate)：跳速恢复为基准值，清除上一个 episode 的
              动作残留（否则载波模板与 hops_per_block 会沿用旧跳速）。
            - 干扰变体选择器、扫频/梳状干扰、反应式干扰机状态机复位：使变体切换流
              与检测状态从零开始，保证 episode 内可复现。
            - 通信/观测的随机流不回退（见模块 docstring 的"随机性边界"）：
              _draw_substream_seeds 从当前 _rng 继续取种子，故连续两次 reset 的初始
              观测不同；需要完全复现时用 reset(seed=...)，它会重置 np.random 全局
              种子并重建 _rng。

        Args:
            seed: int 或 None；非 None 时先调用 seed(seed) 再重置（Gymnasium 约定）。
            options: dict 或 None；本环境未使用，仅为接口兼容保留。

        Returns:
            tuple:
                obs: np.ndarray，形状 (num_frames, n_bins)（默认 (100, 100)），
                    float32，dB 单位的 PSD waterfall。
                info: dict，含 ber_blocks=[]、hoprate_used、comb_phases=[] 占位字段，
                    与 step() 返回的 info 键对齐，便于训练脚本统一读取。

        副作用：按实际观测形状重建 observation_space；更新 self.state 与 self.last_info。
        """
        if seed is not None:
            self.seed(seed)
        super().reset(seed=seed)
        self.current_step = 0
        self.jammer_ptr = 0 # 干扰机的连续采样时钟归零
        _ = self._apply_hoprate(self.base_hoprate)

        # 变体选择流复位：让"第几个干扰周期用哪个波形变体"从头开始
        if self.jammer_variant_selector is not None:
            self.jammer_variant_selector.reset()
        
        if self.enable_sweep and self.sweep is not None:
             self.sweep.reset()

        # 复位反应式干扰机的状态机（检测/压制循环）
        if self.enable_reactive and self.reactive is not None:
            self.reactive.reset()

        # 初始观测也消耗两个子流（观测噪声、干扰动态噪声），并推进 jammer_ptr
        obs_seeds = self._draw_substream_seeds(2)
        obs = self._observe_100ms(
            block_id=0,
            rng=np.random.RandomState(int(obs_seeds[0])),
            jammer_rng=np.random.RandomState(int(obs_seeds[1])),
        )
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
        """
        执行一个 step：顺序通信 10 个 100 ms block，并生成下一次观测。

        执行流程（顺序即物理时间线，全部状态推进都在主线程完成）：
            1. 解析动作（dict / (hoprate, offsets) 元组 / 标量），并严格校验 offsets。
            2. _apply_hoprate 落地跳速（裁剪 + 10 Hz 量化），得到 hops_per_block。
            3. 按 block 顺序推进时间线：记录梳状相位、取 hop 序列并推进 _mseq_ptr、
               按 jammer_ptr 取/算扫频干扰并推进 jammer_ptr、生成反应式干扰波形。
            4. 构造 10 个任务（各自带子流种子）交 _run_block_tasks 串行/并行执行。
            5. 主线程负责出图与调试日志，然后生成下一次观测（观测同样推进 jammer_ptr）。
            6. 逐 block 计算奖励并取均值，组装 info。

        关键设计：
            - 时间线只在主线程推进（jammer_ptr / _mseq_ptr / _rng），worker 只做无状态
              计算，因此并行度不影响仿真结果。
            - 观测在通信之后生成，对应"决策 → 通信 → 观察下一段干扰"的因果顺序；
              观测窗同样占用 100 ms 的干扰时间线，故干扰机是连续发射的。
            - reset_mseq_each_step=True 时把 _mseq_ptr 归零，使各 step 的 hop 基序列
              相同，从而不同 step 的 offset 动作效果可比（消融实验的前提）。

        Args:
            action: None | dict | tuple | 标量。
                None → hoprate=base_hoprate 且 offsets 全 0（baseline/对照路径）。
                dict → {"hoprate": float, "offsets": array-like[num_blocks]}，
                    缺失键回退到默认值。
                (hoprate, offsets) 二元组 → 分别解析。
                其它标量 → 只给 hoprate，offsets 全 0。

        Returns:
            tuple:
                obs: np.ndarray (num_frames, n_bins), float32, dB，下一次观测。
                reward: float，10 个 block 奖励的均值。
                terminated: bool，恒为 False（本任务无自然终止状态）。
                truncated: bool，恒为 False（不设时间上限，由外部训练循环控制）。
                info: dict，字段：
                    ber_blocks (list[float], 长度 10)、
                    block_rewards (list[float], 长度 10)、
                    mean_ber (float)、hoprate_used (float, 量化后)、
                    hops_per_block (int)、reactive_active_blocks (list[bool], 长度 10)、
                    hop_sequences (list[list[int]], 长度 10)、
                    comb_phases (list[list[int]]，每 block 内各 1 ms 时隙的梳状相位)。

        Raises:
            ValueError: offsets 形状不是 (num_blocks,)、含非有限值、非整数值或越界时。
        """
        # 解析动作：三种调用形式（dict / 二元组 / 标量）统一成 hoprate + offsets
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

        # 校验 offsets：允许浮点表示但必须是整数值，避免静默取整造成"动作与奖励
        # 记录不一致"（训练/离线 replay 都按整数信道索引解释动作）
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

        # 跳速必须在 block 循环之前落地：它决定 hops_per_block 与载波模板长度，
        # 而这两者都会影响本 step 全部 10 个 block 的合成
        ainfo = self._apply_hoprate(hoprate_action)

        # 每个 100 ms block 的 hop 数（10 Hz 网格保证其为整数）；至少 1 跳
        hops_per_block = int(round(self.current_hoprate * 0.1))
        hops_per_block = max(1, hops_per_block)

        reactive_active_blocks = []
        hop_sequences = []
        comb_phases = []
        comb_active = (
            self.enable_sweep
            and self.sweep is not None
            and self.sweep.comb_enabled
        )
        comb_phase_sample_offsets = ()
        if comb_active:
            # 按 1 ms 时隙采样梳状相位（100 ms / 1 ms = 100 个时隙）：记录每个 block
            # 内干扰在两相位组之间的切换时刻，离线 replay 与调试分析都需要该元数据
            comb_phase_slot_count = int(round(
                0.1 / IndiscriminateJammer.COMB_TIME_QUANTUM
            ))
            comb_phase_sample_offsets = tuple(
                int(round(
                    slot_idx
                    * IndiscriminateJammer.COMB_TIME_QUANTUM
                    * self.Fs
                ))
                for slot_idx in range(comb_phase_slot_count)
            )
        # 出图触发：训练 step 命中保存列表时，本 step 需要额外算每个 block 的 PSD
        capture_figures = (
            (self.current_step + 1) in self._fig_save_steps
            and self._fig_save_dir is not None
        )
        compute_block_waterfalls = capture_figures or self.debug_plot_psd
        # 预生成路径可用性：开关打开且 pregen_data 确实构建成功
        use_pre = self.use_pregen and self.pregen_data is not None
        # block 长度以预生成值为准（正常配置下与 _block_len 相同）
        block_len = (
            self.pregen_data.block_len if use_pre else self._block_len
        )

        # 预先在主线程算好 hop 段长并预热载波模板：worker 首次并发进入时会竞争惰性
        # 缓存，主线程先建好可避免竞态与重复计算
        segment_lengths = self._hop_segment_lengths(
            block_len,
            hops_per_block,
        )
        self._ensure_carrier_templates(max(segment_lengths, default=0))
        if (
            not use_pre
            and self.enable_sweep
            and self.sweep is not None
        ):
            # 同理预热干扰机载波缓存，避免首个 step 并发时的惰性缓存竞态
            self.sweep._ensure_carrier_cache(self.Startfre, self.Endfre)

        # 子流种子：每 block 5 个（bits / noise / rayleigh / jammer / reactive），
        # 观测再单独 2 个（观测噪声 / 干扰动态噪声）
        block_seeds = self._draw_substream_seeds((self.num_blocks, 5))
        obs_seeds = self._draw_substream_seeds(2)
        tasks = []

        # 所有有状态的时间线都在主线程按 block 顺序推进（保证与并行度无关）
        for b in range(self.num_blocks):
            if comb_active:
                # 相位由"block 起始采样索引 + 时隙偏移"推算，与干扰波形切片共用同一
                # 采样时钟，因此记录值与实际发射的相位一致
                comb_phases.append([
                    self.sweep.comb_phase_at(
                        self.jammer_ptr + sample_offset
                    )
                    for sample_offset in comb_phase_sample_offsets
                ])

            # ----------------------------------------------------------------
            # 2. 跳频序列（逐 block 动态生成）
            # ----------------------------------------------------------------
            hop_seq_block = self._get_block_hopseq(hops_per_block, offsets_action[b])
            # 推进 m 序列指针（环形）：使同一 step 内 10 个 block 使用不重叠的图案段
            self._mseq_ptr = (self._mseq_ptr + len(hop_seq_block)) % len(self.mseq_channels)
            hop_sequences.append(hop_seq_block.astype(int).tolist())
            # 记录本 block 的干扰起始采样索引，再按 block 长度推进全局采样时钟
            jammer_start = self.jammer_ptr
            sweep_jam = None
            dynamic_sweep = False
            if self.enable_sweep and self.sweep is not None:
                if use_pre:
                    # 预生成路径：按采样索引切片，与观测窗共享同一连续时间线
                    sweep_jam = self.sweep.get_composite_signal(
                        jammer_start,
                        block_len,
                    )
                else:
                    # 动态路径：留给 worker 现算（需要 jammer_seed 子流）
                    dynamic_sweep = True
                self.jammer_ptr += block_len

            reactive_jam = None
            reactive_active = False
            if self.enable_reactive and self.reactive is not None:
                # 反应式干扰依赖 hop_seq 与内部检测状态机，必须在主线程按顺序生成，
                # 之后把波形交给 worker 叠加（worker 不接触该状态机）
                reactive_rng = np.random.RandomState(
                    int(block_seeds[b, 4])
                )
                reactive_jam, reactive_active = self.reactive.generate_samples(
                    block_len,
                    hop_seq_block,
                    self.Startfre,
                    self.Sub_interval,
                    self.current_hoprate,
                    rng=reactive_rng,
                )
            reactive_active_blocks.append(bool(reactive_active))

            tasks.append({
                "block_idx": b,
                "hop_seq": hop_seq_block,
                "use_pre": use_pre,
                "sweep_jam": sweep_jam,
                "dynamic_sweep": dynamic_sweep,
                "jammer_start": jammer_start,
                "reactive_jam": reactive_jam,
                "bits_seed": block_seeds[b, 0],
                "noise_seed": block_seeds[b, 1],
                "rayleigh_seed": block_seeds[b, 2],
                "jammer_seed": block_seeds[b, 3],
                "compute_waterfall": compute_block_waterfalls,
            })

        # 并行/串行执行；返回顺序与 tasks 一致，故 ber_blocks[b] 对应第 b 个 block
        block_results = self._run_block_tasks(tasks)
        ber_blocks = [result[0] for result in block_results]

        # matplotlib 必须留在主线程：worker 只在需要出图/调试时返回数值 waterfall
        for b, (_, waterfall_db) in enumerate(block_results):
            if capture_figures:
                fig_name = (
                    f"step_{self.current_step + 1:03d}_block_{b + 1:02d}.png"
                )
                save_waterfall_figure(
                    waterfall_db,
                    os.path.join(self._fig_save_dir, fig_name),
                    title=f"Step {self.current_step + 1} - Block {b + 1} PSD (100 ms)",
                )
            if self.debug_plot_psd:
                show_waterfall_figure(
                    waterfall_db,
                    title=f"Step {self.current_step} - Block {b + 1}",
                )
            if self.debug_log_hops:
                print(
                    f"Block {b + 1}: Hop Seq (with offset) = "
                    f"{tasks[b]['hop_seq'].tolist()}"
                )

        
        self.current_step += 1
        
        # 观测在通信之后生成：对应"决策 → 通信 1 s → 观察下一段干扰"的因果顺序，
        # 且观测窗也消耗 100 ms 的 jammer 时间线（干扰机连续发射）
        obs = self._observe_100ms(
            block_id=0,
            rng=np.random.RandomState(int(obs_seeds[0])),
            jammer_rng=np.random.RandomState(int(obs_seeds[1])),
        )
        self.state = obs.astype(np.float32)

        # 可选：每个 step 结束把 m 序列指针归零，使各 step 的 hop 基序列一致，
        # 从而不同 step 的 offset 动作效果可比（对比实验的前提）
        if self.reset_mseq_each_step:
            self._mseq_ptr = 0

        # 逐 block 奖励 → step 标量奖励（均值）；hoprate 用量化后的实际值
        mean_ber = float(np.mean(ber_blocks)) if len(ber_blocks) > 0 else 0.0
        block_rewards = compute_block_rewards(
            ber_blocks,
            ainfo["hoprate_used"],
            settings.REWARD_CONFIG,
        )
        reward = float(np.mean(block_rewards))

        # info 同时服务训练日志与离线 replay 元数据（hop_sequences/comb_phases 用于分析）
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
        """
        以 imshow 弹窗显示最近一次观测（Gymnasium 兼容的可视化接口）。

        仅用于人工调试：训练循环不应调用它（会阻塞在 plt.show）。self.state 未初始化
        （reset 之前）时直接返回，避免报错。
        """
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
    # 自检入口：只用最轻的配置跑通"构造 → reset → 10 次 step"，
    # 注意干扰机配置统一来自 settings.py（此处不传 jammer_config）
    pr_start = time.time()
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
