"""
FHSS 抗干扰强化学习项目的全局配置中心。

集中定义：随机种子、环境参数（ENV_CONFIG，传给 FHSSQPSKEnv）、
干扰机参数（JAMMER_CONFIG）、SAC 超参（SAC_CONFIG）、replay buffer
（BUFFER_CONFIG）、离线 replay（OFFLINE_REPLAY_CONFIG）、MBPO 奖励模型
（MBPO_CONFIG）、噪声二分搜索（NBS_CONFIG）、训练循环（TRAIN_CONFIG）、
图片保存（PLOT_CONFIG）与奖励系数（REWARD_CONFIG）。

各训练/搜索脚本的输出目录不在此配置，由每个入口自带默认 --output_dir。
修改环境/干扰/奖励相关配置后，已生成的离线 replay metadata 将不再匹配：
baseline 只警告，MBPO 等入口默认直接拒绝加载（详见 OFFLINE_REPLAY.md），
此时需重新运行 generate_offline_replay.py 或显式允许配置不匹配。
"""

import random
import numpy as np

# 全局随机种子：保证 Python/NumPy/PyTorch 全链路可复现。
RANDOM_SEED = 42

# 计时剖析总开关。
# True 时，MBPO 奖励模型训练会测量每个 epoch 的墙钟时间、rollout 各阶段
# 耗时并写入训练日志；False 时关闭全部计时测量。
TIMING_ENABLED = True

# 每 batch 反向传播计时开关（仅 MBPO 奖励模型训练）。
# True 时用 CUDA event 测量每个 batch 反向传播的真实 GPU 耗时，并以
# [MBPO-BWD] 标签逐 batch 打印到控制台；该测量需要一次 per-batch 的
# 流同步，会轻微拖慢训练。False 时关闭逐 batch 打印与同步。
BACKWARD_TIMING_ENABLED = True


def set_random_seeds(seed=None):
    """统一设置所有随机数生成器的种子，保证实验可复现。

    Args:
        seed: 随机种子。None 时使用 settings.RANDOM_SEED。

    说明：同时固定 cudnn.deterministic=True、benchmark=False，
    保证 CUDA 卷积结果确定；代价是部分算子可能变慢。
    """
    if seed is None:
        seed = RANDOM_SEED

    # Python 标准库
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch（局部导入，避免无 torch 环境下的循环依赖）
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 保证 CUDA 卷积确定性
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# ============================== 环境配置 ==============================
# 原样传给 FHSSQPSKEnv(**ENV_CONFIG)，字段含义见 fh_env.py。
ENV_CONFIG = {
    "Startfre": 3e6,          # FHSS 工作频段下界 (Hz)
    "Endfre": 4e6,            # FHSS 工作频段上界 (Hz)
    "Sub_interval": 50000,     # 子信道间隔 (Hz)，1 MHz 频段共 20 个 50 kHz 信道
    "Fs": 1e7,                # 系统采样率 (Hz)，1 个 100 ms block 含 1e6 个采样点
    "Baud": 25000,            # 码元速率 (Baud)，每 block 2500 个码元
    "Hoprate": 100,           # 基础跳速 (Hz)，环境以 10 Hz 网格量化实际生效值
    "hoprate_min": 10.0,      # 合法跳速下界 (Hz)
    "hoprate_max": 1000.0,    # 合法跳速上界 (Hz)
    "enable_reactive": False,  # 是否启用反应式（能量检测）干扰机
    "enable_sweep": True,     # 是否启用无差别（扫频/梳状）干扰机
    "enable_rayleigh": True,  # 是否启用 Rayleigh 衰落信道
    "debug_plot_psd": False,  # 调试：绘制 PSD（会拖慢训练，仅排查用）
    "debug_log_hops": False,  # 调试：记录 hop 序列日志
    "use_pregen": True,       # 预生成加速路径：复用 QPSK bits/IQ 与干扰波形池，
                              # 但每 block 白噪声、每 hop Rayleigh、每次观测噪声都重新生成
    "noise_std": 0.1,         # 接收端热噪声标准差（同时被反应式干扰机用于推导检测 SNR）
    # 驱动基础跳频图案的 m 序列（LFSR）参数。更换 m 序列的方法：
    #   - 改 mseq_seed  -> 同一条 m 序列的不同相位（循环移位），最简单；
    #   - 改 mseq_taps  -> 得到真正不同的 m 序列；抽头必须对应本原多项式
    #                      （如 10 级的 (10, 7)、(10, 3)），否则周期骤减；
    #   - 改 mseq_nbits -> 改变周期（2^n - 1），同时 mseq_taps 必须换成
    #                      对应级数的本原抽头。
    # 注意：更换 m 序列后需重新生成离线 replay（或用 --offline_replay_path none）。
    "mseq_seed": 46,
    "mseq_taps": (10, 7),
    "mseq_nbits": 10,
    "mseq_length": 1023,      # 10 级 m 序列完整周期 2^10 - 1
    # 每 sampling 点的射频信号功率（衰落前）。理论值 Baud / Fs = 25000/1e7 = 0.0025。
    # 反应式干扰机把它与 noise_std 一起推导检测 SNR，并经 Gauss-Laguerre
    # 数值积分对 Rayleigh 衰落取平均，保证与接收端噪声设置一致。
    "signal_power": 0.0025,
}

# ============================== 干扰机配置 ==============================
# 传给 jammers.py 的各干扰机；启用开关在 ENV_CONFIG 中。
JAMMER_CONFIG = {
    # 无差别干扰模式：'sweep'（扫频）、'comb'（梳状）或 'both'（同时叠加）
    "mode": "comb",
    # 预生成干扰使用的独立带限噪声基带数量；相同带宽的 reactive/sweep/comb
    # 共享同一波形池，随机有放回选择。
    "baseband_variant_count": 4,

    # 扫频干扰：默认 20 个频点 × 4 ms 驻留 = 80 ms 一个完整扫频周期
    "sweep": {
        "step": 50000,        # 扫频步进 (Hz)，即逐个扫过 50 kHz 信道
        "power": 0.8,         # 干扰功率
        "dwell_time": 0.004,  # 每个频点驻留时间 (s)
        "bandwidth": 50000.0, # 噪声带宽 (Hz)
    },

    # 梳状干扰：两组干扰信道按固定周期交替压制
    "comb": {
        "power": 0.8,         # 干扰功率
        "bandwidth": 50000.0, # 每个 tone 的噪声带宽 (Hz)
        "switch_interval": 0.057, # 相位切换周期 (s)；必须为 1 ms 的正整数倍
        # 两组交替的干扰信道序号。序号 k 对应第 k 个 50 kHz 信道的中心
        # （Startfre + k*50kHz + 25kHz）；必须是 [0, num_channels-1] 内的
        # 整数，否则环境启动时直接 ValueError。两组长度可不同、允许重叠。
        # 注意：修改后需重新生成离线 replay。
        "channels_phase0": [0, 2, 4, 6, 8, 10, 12, 14],   # 相位 0：偶数信道组
        "channels_phase1": [1, 3, 5, 7, 9, 11, 13, 15],  # 相位 1：奇数信道组
    },

    # 反应式干扰：基于能量检测理论（Urkowitz 1967），
    # 按 1 ms 基本时隙 scan → detect → jam 循环。
    # 干扰机处的 SNR 由 ENV_CONFIG["noise_std"] 与理论信号功率
    # （Baud / Fs）推导，保证与接收端噪声一致。
    "reactive": {
        "power": 1.5,              # 干扰功率
        "bandwidth": 50000.0,      # 噪声带宽 (Hz)
        "p_fa": 0.1,               # 能量检测虚警概率（用于推导检测门限）
        "detection_time": 0.0005, # 检测/压制时隙时长 (s)，即 0.5 ms
    }
}

# ============================== SAC 超参 ==============================
# 十头离散 SAC（SAC.py）的优化超参。
SAC_CONFIG = {
    "actor_lr": 1e-5,            # actor 学习率
    "critic_lr": 1e-4,           # critic 学习率
    "alpha_lr": 1e-4,            # 温度系数 alpha 的学习率
    "tau": 0.005,                # target 网络软更新系数
    "gamma": 0.95,               # 折扣因子
    "target_entropy_ratio": 0.1, # 目标熵 = ratio * (-log(1/n_actions))，
                                 # 控制探索强度（对十头每头相同）
}

# ============================== Replay Buffer ==============================
# SAC 训练用 step-level replay buffer。
BUFFER_CONFIG = {
    "capacity": 20000,
    # A800 训练用大 batch 保持卷积编码器满载。
    "batch_size": 256,
}

# ============================== 离线真实 replay ==============================
# 一条 transition 表示一个完整环境 step：10 个 offset 动作 + 10 个 block
# reward。default_path 是训练入口（train_offsets.py / train_mbpo.py）默认
# 加载的文件；num_step_transitions 是 generate_offline_replay.py 不带参数
# 运行时的生成条数默认值——两者相互独立，注意直接运行生成脚本会按
# num_step_transitions 覆写 default_path 指向的文件。
OFFLINE_REPLAY_CONFIG = {
    "num_step_transitions": 512,
    "default_path": "outputs/offline_replay/replay_5000_100_hoprate_v3.npz",
    "hoprate_mode": "fixed",
    "fixed_hoprate": 100.0,
}

# 联合训练（derivative NBS + offset 策略）使用的随机跳速 replay 配置。
# 注意：联合训练入口（train_joint_sac.py / train_joint_mbpo.py /
# joint_training.py）已从当前代码库移除，如需恢复请从 git 历史找回；
# 本配置块保留供恢复后使用，不影响现存入口。
JOINT_OFFLINE_REPLAY_CONFIG = {
    "num_step_transitions": 5000,
    "default_path": "outputs/offline_replay/replay_5000_random_hoprate_v3.npz",
    "hoprate_mode": "random",
}

# ============================== MBPO 奖励模型 ==============================
# StepRewardEnsemble（r_predict_model/model.py）与 train_mbpo.py 的配置，
# 设计细节见 MBPO_MODULE.md。
MBPO_CONFIG = {
    "num_networks": 5,       # ensemble 成员数量
    "num_elites": 3,         # holdout MSE 最优的 elite 数量
    "hidden_size": 200,      # offsets one-hot 编码与融合层宽度
    "learning_rate": 1e-3,   # 共享 Adam 优化器学习率
    "weight_decay": 1e-5,    # 权重衰减
    # 每隔多少个真实环境 step 用全部真实 replay 从头重新拟合一次奖励模型，
    # 代替每个 step 阻塞拟合。
    "model_train_freq": 1,
    "model_train_batch_size": 512,
    "holdout_ratio": 0.2,    # holdout 集占比（用于 elite 选择与早停）
    "early_stop_patience": 10,
    "max_epochs": 150,
    "min_improvement": 0.01,
    "rollout_batch_size": 1024,  # 每次合成 rollout 的 batch 大小
    "rollout_length": 1,         # 只做一步 rollout（奖励模型不递推状态）
    "real_ratio": 0.2,           # SAC 更新时真实样本的目标占比
    "model_replay_size": 4000,   # model replay 容量，满后 FIFO 淘汰最旧样本
    # 奖励模型始终在 CUDA 上训练，replay 常驻训练设备，不做逐 epoch 的
    # H2D 拷贝；SAC replay 快照可选择 worker 进程与 pinned batch。
    "data_loader_workers": 0,
    "data_loader_pin_memory": False,
    # 运行时加速开关：不改变奖励模型架构、optimizer 预算、replay 划分
    # 或早停设置。
    "model_precision": "bfloat16",
    "model_fast_math": True,
    "model_compile": True,
    # PNG 曲线属于诊断 I/O，正常训练时仍会生成（False 可关闭以省 I/O）。
    "save_curve_figures": True,
    # 可选的第二层 2x2 池化（conv_fc 前）。注意：这会改变奖励模型架构：
    # 每成员 conv_fc 输入从 80000 降到 20000，参数量从约 205M 降到约 51M，
    # 单 batch 耗时约降 30%。默认关闭以保持既有模型兼容。
    "model_extra_pool": False,
}

# ============================== 噪声二分搜索（NBS） ==============================
# 基于 MWU 的噪声二分搜索跳速调节参数（noisy_binary_search_derivative.py）。
# 参考：Dereniowski et al. "Noisy (Binary) Searching: Simple, Fast and
# Correct" (STACS 2025)。
NBS_CONFIG = {
    "p": 0.3,                 # 假设的噪声概率，0 ≤ p < 0.5。
                              # p 越大探索越强（BER 平坦区需要）。
    "delta": 0.01,            # 置信阈值，0 < δ ≤ 1；最大权重 ≥ 1-δ 判定收敛。
    "hoprate_step": 10.0,     # 候选 hoprate 离散步长 (Hz)，与环境 10 Hz 量化一致。
    "derivative_threshold": -0.005,  # 导数判决阈值：metric > 阈值 → 向左移动
}

# ============================== 训练循环 ==============================
TRAIN_CONFIG = {
    "steps_per_episode": 80,      # 每次训练运行的环境 step 总数
    "update_iters_per_step": 10,  # 每个环境 step 后的梯度更新次数
    "fixed_hoprate": 100.0,       # offset 训练使用的固定跳速 (Hz)
}

# ============================== 图片保存配置 ==============================
# train_offsets.py 与 train_mbpo.py 使用。在列出的训练 step（1-based，
# 与日志中 "Step i/N" 一致）保存：动作前观测 figures/step_XXX_obs.png，
# 以及该 step 内 10 个 block 各自的接收 PSD 图 step_XXX_block_YY.png，
# 均位于对应 --output_dir 的 figures/ 子目录下。可填多个 step；
# 超出 [1, steps_per_episode] 的值会被忽略并告警；空列表关闭该功能。
PLOT_CONFIG = {
    "figure_save_steps": [49, 50],
}

# ============================== 奖励系数 ==============================
# 逐 block 奖励公式，与 FHSSQPSKEnv.step() 一致：
#   reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
# 环境为每个 block 计算上式（info["block_rewards"]），其均值作为 env.step()
# 返回的 step reward。baseline、离线生成器与 hoprate sweep 共享同一语义。
REWARD_CONFIG = {
    "base_reward": 10.0,
    "ber_penalty": 80.0,
    "hoprate_penalty": 0,
}
