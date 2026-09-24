"""FHSS MBPO 的 step-level 概率奖励集成模型（StepRewardEnsemble）。

本模块实现 MBPO 风格奖励增强所需的奖励预测器，只建模一步 block reward，
不预测下一状态。模型消费一条完整的 SAC 决策上下文，为每个 offset 头
输出一个 block reward：

    (PSD image, hoprate, offsets[num_heads]) -> block_rewards[num_heads]

只建模 reward 本身；把这些预测与真实 replay 中存储的外生下一观测
（next_state_imgs / next_hoprates）配对、组装合成 transition 的工作由
MBPO 适配层（r_predict_model/mbpo_adapter.py）完成。

矩阵式 ensemble 的设计动机：
    network_size 个成员被组织进单个"矩阵"网络 MatrixStepRewardMember。
    每个权重张量都带一个成员领先维（Linear 权重 ``[num_members, out, in]``，
    CNN 权重 ``[num_members, C_out, C_in, k, k]``），成员之间不共享参数。
    相比实例化 N 个独立网络再在 Python 层逐个 forward/backward，这种布局
    让全部成员在同一次 forward/backward 中并行计算（一次 kernel 启动、
    一次反传），并由单个共享 Adam 优化器一次性更新。CNN 部分靠分组卷积
    技巧实现并行：输入沿通道维复制 num_members 份，各成员的卷积核放入
    相邻的输出通道组，一次 F.conv2d 即可算完所有成员。

数据契约（shape 记 B=批大小、M=成员数、num_heads=offset 头数）：
    - state_imgs: float32，[B, H, W] 或 [B, C, H, W]（本项目为 100x100
      单通道 PSD 灰度图）；
    - hoprates: float32，[B]（跳频速率，网络内部先经 normalize_hoprate
      归一化到 [-10, 10]）；
    - actions（即 offsets）: 整数，[B, num_heads]，取值 [0, n_actions)，
      网络内部做 one-hot 编码；
    - block_rewards: float32，[B, num_heads]，逐分块的物理 reward。

潜变量与有界 reward 的双向变换：
    物理 reward 满足 reward(hoprate, BER) = base_reward
    - hoprate_penalty * hoprate - ber_penalty * BER，BER 被约束在
    [ber_min, ber_max]（默认 [0, 0.5]）内，因此每条样本存在由 hoprate
    决定的物理上下界。训练时目标先饱和（clamp）到边界内、归一化到
    (0, 1)，再避开 sigmoid 端点（logit_epsilon）后做 logit 变换进入
    潜变量空间；网络在该空间输出 Logistic-Normal 潜变量的 mean/logvar，
    训练损失为全体成员 latent Gaussian NLL 的均值。推理时在潜变量空间
    采样，经 sigmoid 映射回有界 reward，保证预测永远落在物理边界内。

elite 选择与重拟合：
    holdout 评估在真实 reward 单位下比较 sigmoid 有界预测与有界目标的
    MSE，按该指标选出 elite_size 个最优成员；fit() 每次都从全部真实
    replay 从头拟合——成员权重与 Adam 优化器状态全部重新初始化，
    绝不延续上一次拟合。

精度与 checkpoint：
    CUDA 上支持 bfloat16/float16 autocast（float16 另配 GradScaler），
    可选 torch.compile。checkpoint 为 reward v3 格式，架构标识在
    extra_pool 开启时带 "_pool2" 后缀（第二层 2x2 池化的轻量变体，
    conv_fc 参数量约从 205M 降到 51M）。

相关文件：
    - train_mbpo.py：训练入口，按 settings.MBPO_CONFIG 等配置构造本模型；
    - r_predict_model/mbpo_adapter.py：适配层，从真实 replay 提取字段
      调用 fit()，并调用 sample_rewards() 生成单步合成 transition。
"""

import os
import time
from contextlib import nullcontext

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, default_collate

import settings
from SAC import (
    HOPRATE_FEATURE_DIM,
    PSD_FEATURE_DIM,
    STATE_FEATURE_DIM,
    normalize_hoprate,
)


# fit() 缓存在设备上的五个张量字段（最后一个是派生的潜变量目标，并非
# replay 原始字段）；_DeviceBatchDataset 按位置与之对应。
REWARD_FIELD_NAMES = ("state_imgs", "hoprates", "actions", "block_rewards",
                      "latent_targets")


def _collate_reward_batch(batch):
    """DataLoader collate：让已堆叠好的设备端 batch 直接透传，不做多余的 stack/拷贝。

    ``_DeviceBatchDataset.__getitems__`` 产出的已是整 batch 的张量字典
    （各字段常驻 GPU），若走 default_collate 会再做一次逐元素 stack 与
    设备间拷贝。对 dict 输入逐字段原样返回（非张量值仅做一次
    as_tensor 转换）；其余情况回退到 default_collate。
    """
    if isinstance(batch, dict):
        return {
            key: value if torch.is_tensor(value) else torch.as_tensor(value)
            for key, value in batch.items()
        }
    return default_collate(batch)


class _DeviceBatchDataset(Dataset):
    """常驻设备的 dataset：每个 batch 的每个字段用一次 index_select 聚合，
    而不是每条 transition 各触发一个微型 GPU 索引 kernel。

    ``torch.utils.data.DataLoader`` 在 dataset 定义了 ``__getitems__`` 时
    会按 batch 整体调用它，因此一个 512 条的 batch 只入队少量 kernel 而
    不是数千个，保持 CUDA launch 队列浅，避免调度开销吞掉吞吐。
    """

    def __init__(self, tensors, indices):
        # tensors: 与 REWARD_FIELD_NAMES 位置对应的常驻设备张量；
        # indices: 本子集（train 或 holdout）在原数组中的行号。
        self.tensors = tensors
        self.indices = torch.as_tensor(
            np.asarray(indices, dtype=np.int64), dtype=torch.long,
            device=tensors[0].device,
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {
            name: tensor[index] for name, tensor in zip(REWARD_FIELD_NAMES, self.tensors)
        }

    def __getitems__(self, indices):
        # DataLoader 的整 batch 取数路径：先把 batch 内的局部索引映射到
        # 构造时传入的样本行号，再对每个字段各做一次 index_select——
        # 整批 gather 只需少量 kernel，而非每条样本一次索引。
        index_tensor = torch.as_tensor(
            np.asarray(indices, dtype=np.int64), dtype=torch.long,
            device=self.tensors[0].device,
        )
        index_tensor = self.indices.index_select(0, index_tensor)
        return {
            name: tensor.index_select(0, index_tensor)
            for name, tensor in zip(REWARD_FIELD_NAMES, self.tensors)
        }


# checkpoint 格式版本与架构标识：两者共同决定旧 checkpoint 是否可加载。
# v3 = 矩阵式 ensemble + 两层 CNN + 单层 hoprate MLP + state-action fusion；
# 任何改变 state_dict 结构的改动都必须同时升版本或改架构标识，否则
# load_checkpoint 会（也应该）拒绝加载。
REWARD_CHECKPOINT_FORMAT_VERSION = 3
REWARD_MODEL_ARCHITECTURE = (
    "cnn2_groupnorm_hop_mlp1_state_action_fusion1_matrix_v3"
)
# BER 取值范围与 logit 端点避让系数：默认值同时被 settings.MBPO_CONFIG 与
# mbpo_adapter.reward_bounds 复用，改动会影响奖励边界与目标饱和程度。
DEFAULT_BER_MIN = 0.0
DEFAULT_BER_MAX = 0.5
DEFAULT_LOGIT_EPSILON = 1e-4


def _canonical_reward_config(reward_config):
    """校验并规范化物理 reward 公式的三个必需系数。

    物理模型为 reward = base_reward - hoprate_penalty * hoprate
    - ber_penalty * BER，三个系数缺一不可：缺 base_reward 无法定位量级；
    缺 hoprate_penalty 无法由 hoprate 推出每条样本的边界；缺 ber_penalty
    则 BER 端点不再产生区间宽度，本模块赖以成立的"有界 reward + logit
    变换"前提失效。因此 ber_penalty == 0 被显式拒绝，而不是静默容忍。

    Args:
        reward_config: 映射，必须含 base_reward、ber_penalty、hoprate_penalty。

    Returns:
        dict: 只含上述三个键、值均为 float 的规范化副本。故意不保留调用方
        的额外键，避免下游按未经验证的字段构造模型。

    Raises:
        ValueError: 缺键、任一值非有限（NaN/Inf）或 ber_penalty == 0 时。
    """
    required_keys = ("base_reward", "ber_penalty", "hoprate_penalty")
    if reward_config is None or any(key not in reward_config for key in required_keys):
        raise ValueError(
            "reward_config must define base_reward, ber_penalty, and "
            "hoprate_penalty."
        )
    config = {key: float(reward_config[key]) for key in required_keys}
    if not np.all(np.isfinite(list(config.values()))):
        raise ValueError("reward_config values must be finite.")
    if config["ber_penalty"] == 0.0:
        raise ValueError("ber_penalty must be non-zero for a bounded reward model.")
    return config


def reward_bounds_from_config(
    hoprates,
    reward_config,
    ber_min=DEFAULT_BER_MIN,
    ber_max=DEFAULT_BER_MAX,
):
    """由 BER 端点推导每条 transition 的物理 reward 上下界。

    reward 关于 BER 是线性的，故 BER 取端点时 reward 取到该样本的极值；但
    ber_penalty 的符号不受约束，BER 上下界未必对应 reward 上下界，所以这里
    对两个端点值取 min/max 而不是假设方向。hoprates 先 reshape 成 [B, 1]
    以便广播，返回值也是 [B, 1]，可直接夹逼 [B, num_heads] 的 reward。

    Args:
        hoprates: array-like，形状 [B]（内部 reshape 为 [B, 1]），单位 Hz。
        reward_config: 物理 reward 系数，见 _canonical_reward_config。
        ber_min: BER 下界，默认 0.0。
        ber_max: BER 上界，默认 0.5（随机猜测的物理上限）。

    Returns:
        (np.ndarray, np.ndarray): (lower, upper)，均为 float32、形状 [B, 1]。

    Raises:
        ValueError: reward_config 非法、BER 界非有限或 ber_max <= ber_min、
            hoprate 含非有限值时。
    """
    config = _canonical_reward_config(reward_config)
    ber_min = float(ber_min)
    ber_max = float(ber_max)
    if not np.isfinite(ber_min) or not np.isfinite(ber_max) or ber_max <= ber_min:
        raise ValueError("BER bounds must be finite and satisfy ber_max > ber_min.")
    hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1, 1)
    if not np.all(np.isfinite(hoprates)):
        raise ValueError("Reward bounds require finite hoprates.")

    # hoprate 惩罚项：base 随跳速线性下降，BER 项再从中扣减。
    base = config["base_reward"] - config["hoprate_penalty"] * hoprates
    reward_at_ber_min = base - config["ber_penalty"] * ber_min
    reward_at_ber_max = base - config["ber_penalty"] * ber_max
    return (
        # 不假定 ber_penalty 的符号，用 min/max 保证 lower <= upper。
        np.minimum(reward_at_ber_min, reward_at_ber_max).astype(
            np.float32, copy=False
        ),
        np.maximum(reward_at_ber_min, reward_at_ber_max).astype(
            np.float32, copy=False
        ),
    )


def _stable_sigmoid(values):
    """数值稳定的 sigmoid，等价于 1 / (1 + exp(-x)) 但不做危险中间量。

    直接 exp(-x) 在 x 为大负数时溢出为 inf，再除会得到 0/inf 甚至 nan；
    exp(-logaddexp(0, -x)) 把两个指数项先在对数域合并，因此对任意有限输入
    都返回 (0, 1) 内的有限值。推理端在 numpy 上完成 latent -> reward 映射，
    用它替代 torch.sigmoid，保证与训练端一致的端点行为，不让 inf/nan 污染
    合成 replay。

    Args:
        values: array-like，任意形状的潜变量。

    Returns:
        np.ndarray: 与输入同形状、取值在 (0, 1) 的值。
    """
    values = np.asarray(values)
    return np.exp(-np.logaddexp(0.0, -values))


class _MatrixConv2d(nn.Module):
    """权重带领先成员维的分组卷积层（矩阵式 ensemble 的卷积构件）。

    权重布局为 [M, out_channels, in_channels, k, k]（M=num_members），成员
    之间完全独立、不共享参数。单次 F.conv2d 并行算完所有成员的做法：

    1. 输入复制：把 [B, C, H, W] 沿通道维复制成 [B, M*C, H, W]（repeat_interleave，
       首层单通道 PSD 输入下通道 j 恰为成员 j 的副本）；上游若已是矩阵布局
       [B, M, C, H, W]，则直接 reshape 成 [B, M*C, H, W]，此时第 m 个成员的
       C 个通道连续占据区间 [m*C, (m+1)*C)；
    2. 权重物化：把 [M, C_out, C_in, k, k] reshape 成
       [M*C_out, C_in, k, k]，第 m 个成员的全部卷积核落在输出通道区间
       [m*C_out, (m+1)*C_out)；
    3. 分组卷积：groups=M 时 F.conv2d 只把第 m 个输入通道组与第 m 段输出
       通道（即第 m 个成员的核）相连，数学上完全等价于 M 次独立卷积；
    4. 输出 reshape 回 [B, M, C_out, H', W']，把成员维显式拆出来。

    相比"实例化 M 个 nn.Conv2d 再逐个调用"，这里只有一次 kernel 启动、
    一次反传，且能被 torch.compile 整体捕获；代价是成员维必须在层间显式
    维护（下游 _MatrixGroupNorm/_MatrixLinear 都按此约定）。

    惰性初始化（UninitializedParameter 机制的轻量替代）：in_channels 由
    输入决定，首次 forward 时才在输入所在 device 上分配 weight/bias 并
    初始化；之后输入通道数变化会重新物化。这样模型在 observation_shape
    未知时也能先构造（load_checkpoint 在 _materialize 之前正是这种状态），
    又不必引入 LazyModuleMixin 的 hooks 与参数替换流程。

    约束：4D 输入走 repeat_interleave（通道序为 c*M+m），只有当 C == 1 时
    第 m 个成员的通道才恰好落在 [m, m+1) 而与 groups=M 对齐——本层只用于
    首层单通道 PSD 卷积，满足该前提；多通道输入必须走 5D 路径（成员通道
    连续），例如第二层卷积接收上一层的 [B,M,C,H,W] 输出。
    """

    def __init__(self, num_members, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.num_members = int(num_members)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.stride = int(stride)
        self.padding = int(padding)
        self.in_channels = None
        self.weight = None
        self.bias = None

    def _materialize(self, in_channels, device):
        """按输入通道数分配并初始化权重/偏置（惰性初始化）。

        权重取 N(0, 0.1^2)、偏置置零，与 SAC 主干编码器（_init_weights）
        的尺度一致，保证 reward 模型与策略网络的图像特征分布可比。仅在
        首次 forward 或输入通道数变化时调用。
        """
        weight = torch.empty(
            self.num_members,
            self.out_channels,
            in_channels,
            self.kernel_size,
            self.kernel_size,
            device=device,
        )
        nn.init.normal_(weight, 0.0, 0.1)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(
            torch.zeros(self.num_members, self.out_channels, device=device)
        )
        self.in_channels = int(in_channels)

    def forward(self, images):
        # 两种合法布局：CNN 特征流为 [B,C,H,W]（需复制成员份）；上游若已是
        # 矩阵布局 [B,M,C,H,W]（例如复用本模块输出），则直接展平通道维。
        if images.ndim == 5:
            batch_size, members, channels, height, width = images.shape
            if members != self.num_members:
                raise ValueError(
                    "Matrix conv input member count does not match the network."
                )
            # [B,M,C,H,W] -> [B,M*C,H,W]：第 m 段通道即第 m 个成员的输入。
            repeated = images.reshape(
                batch_size, members * channels, height, width
            )
        elif images.ndim == 4:
            batch_size, channels, height, width = images.shape
            # 每个成员各一份输入副本（成员差异只来自权重）。repeat_interleave
            # 的通道序是 c*M+m，C=1 时即"通道 j = 成员 j 的副本"，与 groups=M
            # 的输入分组一致；多通道输入请走 5D 路径。
            repeated = images.repeat_interleave(self.num_members, dim=1)
        else:
            raise ValueError(
                "Matrix conv input must have shape [B,C,H,W] or [B,M,C,H,W]."
            )
        if self.weight is None or self.in_channels != channels:
            # 惰性物化：首帧或输入通道数变化时（重新）分配参数。
            self._materialize(channels, images.device)
        # [M,C_out,C_in,k,k] -> [M*C_out,C_in,k,k]，成员核按输出通道相邻排布，
        # 才能与 groups=M 的通道分组对齐。
        weight = self.weight.reshape(
            self.num_members * self.out_channels,
            self.in_channels,
            self.kernel_size,
            self.kernel_size,
        )
        # groups=M：第 m 个输入通道组只与第 m 段输出通道（第 m 个成员的核）卷积。
        convolved = F.conv2d(
            repeated,
            weight,
            self.bias.reshape(-1),
            stride=self.stride,
            padding=self.padding,
            groups=self.num_members,
        )
        # [B,M*C_out,H',W'] -> [B,M,C_out,H',W']，把成员维显式拆回。这里沿用
        # 输入的空间尺寸，因此本层实际按 3x3/pad1/stride1 的等尺寸卷积使用。
        return convolved.reshape(
            batch_size, self.num_members, self.out_channels, height, width
        )


class _MatrixGroupNorm(nn.Module):
    """对矩阵布局 [B, M, C, H, W] 的每个成员独立归一化的 GroupNorm 包装层。

    实现上把成员维折进 batch 维（[B*M, C, H, W]）交给单个 nn.GroupNorm：
    归一化统计量按 (样本, 成员) 各自计算，而仿射参数 gamma/beta 在成员间
    共享。这是有意的取舍——成员间的差异由卷积/线性权重承担，归一化层共享
    既省参数，又只需一次 kernel 调用；又因为各成员看到的是同一张 PSD
    图像，共享仿射参数不会造成跨成员的信息泄漏或成员坍缩。

    形状：输入/输出均为 [B, M, C, H, W]。
    """

    def __init__(self, num_members, num_groups, num_channels):
        super().__init__()
        self.norm = nn.GroupNorm(int(num_groups), int(num_channels))

    def forward(self, features):
        batch_size, members, channels, height, width = features.shape
        # 成员维折进 batch 维：GroupNorm 把每个 (样本, 成员) 当作独立样本
        # 统计均值/方差，成员之间互不影响。
        reshaped = features.reshape(batch_size * members, channels, height, width)
        normalized = self.norm(reshaped)
        return normalized.reshape(batch_size, members, channels, height, width)


class _MatrixLinear(nn.Module):
    """权重带领先成员维的全连接层（矩阵式 ensemble 的线性构件）。

    权重 [M, out_features, in_features]、偏置 [M, out_features]，成员之间
    不共享参数。forward 用广播式批量矩阵乘一次算完所有成员：
    features [M, B, in] 与 weight.transpose(1, 2) [M, in, out] 相乘再加
    bias.unsqueeze(1) 广播的 [M, 1, out]，得到 [M, B, out]。

    因此本层约定输入特征的第一维是成员维、第二维是 batch 维——把"成员"
    当作批量维来摊平，M 次小 kernel 被合并成一次 batched GEMM。
    初始化沿用 trunc_normal_(std = 1/(2*sqrt(in_features))，截断 ±2σ)，
    尺度比 SAC 主干的 N(0, 0.1^2) 更小，避免成员输入在融合层前方差过大。
    """

    def __init__(self, num_members, in_features, out_features):
        super().__init__()
        self.num_members = int(num_members)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        std = 1.0 / (2.0 * np.sqrt(max(1, self.in_features)))
        weight = torch.empty(self.num_members, self.out_features, self.in_features)
        nn.init.trunc_normal_(weight, std=std, a=-2.0 * std, b=2.0 * std)
        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.zeros(self.num_members, self.out_features))

    def forward(self, features):
        # [M,B,in] @ [M,in,out] + [M,1,out] -> [M,B,out]，成员维即批量维。
        return (
            torch.matmul(features, self.weight.transpose(1, 2))
            + self.bias.unsqueeze(1)
        )


class MatrixStepRewardMember(nn.Module):
    """把全部 ensemble 成员装进单个"矩阵"网络的 step reward 预测器。

    输出 Logistic-Normal 潜变量的 mean/logvar，不预测下一状态。记 B=batch、
    M=num_members、num_heads=offset 头数（H*A 表示 num_heads*n_actions）：

    PSD 分支（成员各自独立，权重不共享）：
        [B,1,100,100]
        -> conv1(1->16, 3x3/pad1) + GroupNorm(4) + ReLU
        -> conv2(16->32, 3x3/pad1) + GroupNorm(8) + ReLU -> [B,M,32,100,100]
        -> MaxPool2d(2) -> [B,M,32,50,50]（extra_pool 时再池化 -> [B,M,32,25,25]）
        -> flatten -> [B,M,80000]（extra_pool 时 20000）
        -> conv_fc（输入维惰性确定）-> ReLU -> [M,B,PSD_FEATURE_DIM=512]
    hoprate 分支：
        [B,1] 经 normalize_hoprate 归一化到 [-10,10] 后广播为 [M,B,1]
        -> _MatrixLinear(1->HOPRATE_FEATURE_DIM=64) + ReLU -> [M,B,64]
    state fusion：
        cat([512, 64], dim=2) -> _MatrixLinear(576->STATE_FEATURE_DIM=256)
        + ReLU -> [M,B,256]（图像与跳速唯一的交互位置）
    action 分支（offsets）：
        [B,num_heads] -> one-hot -> [B,H*A] -> 广播 [M,B,H*A]
        -> _MatrixLinear(H*A->hidden_size=200) + ReLU -> [M,B,200]
    state-action fusion：
        cat([256, 200], dim=2) -> _MatrixLinear(456->200) + SiLU -> [M,B,200]
    输出头：
        latent_mean_head(200->num_heads) / latent_logvar_head(200->num_heads)
        -> [M,B,num_heads]

    logvar 经 max/min 双重 softplus 夹逼（Dreamer 式），把方差限制在
    (exp(-10), exp(0.5)) 内，避免训练早期方差爆炸或塌缩到零；max_logvar/
    min_logvar 注册为 [M,1,num_heads] 的 buffer，以便沿 batch 维广播。

    成员维度在 _MatrixLinear 中必须位于首位、batch 维在第二位，而 CNN 内部
    用 [B,M,C,H,W]（由 _MatrixConv2d 生成），两处布局不同，转换点见 forward。
    """

    def __init__(
        self,
        num_members,
        num_heads,
        n_actions,
        hoprate_min,
        hoprate_max,
        hidden_size=200,
        extra_pool=False,
    ):
        super().__init__()
        self.num_members = int(num_members)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions)
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.hidden_size = int(hidden_size)

        # 与 SAC 主干编码器同构（16/32 通道、GroupNorm(4)/(8)），保证 reward
        # 模型看到的 PSD 特征与策略网络同一量级；每层权重带成员领先维。
        self.conv1 = _MatrixConv2d(self.num_members, 16)
        self.norm1 = _MatrixGroupNorm(self.num_members, 4, 16)
        self.conv2 = _MatrixConv2d(self.num_members, 32)
        self.norm2 = _MatrixGroupNorm(self.num_members, 8, 32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        # 可选的第二个 2x2 池化（flatten 之前）：把 conv_fc 的输入缩小 4 倍
        # （每成员 80000 -> 20000），该层参数量从约 205M 降到约 51M，代价是
        # 模型容量下降；它同时改变架构标识（见 _architecture_tag）。
        self.extra_pool = nn.MaxPool2d(kernel_size=2) if extra_pool else None
        # None 表示尚未物化：输入特征维取决于池化后的空间尺寸，只能等到
        # 首次 forward 才知道（见 _ensure_conv_fc）。
        self.conv_fc = None
        self.hoprate_embedding = _MatrixLinear(
            self.num_members, 1, HOPRATE_FEATURE_DIM
        )
        self.fusion = _MatrixLinear(
            self.num_members, PSD_FEATURE_DIM + HOPRATE_FEATURE_DIM, STATE_FEATURE_DIM
        )
        self.action_encoder = _MatrixLinear(
            self.num_members, self.num_heads * self.n_actions, self.hidden_size
        )
        self.fusion_head = _MatrixLinear(
            self.num_members, STATE_FEATURE_DIM + self.hidden_size, self.hidden_size
        )
        self.latent_mean_head = _MatrixLinear(
            self.num_members, self.hidden_size, self.num_heads
        )
        self.latent_logvar_head = _MatrixLinear(
            self.num_members, self.hidden_size, self.num_heads
        )
        self.register_buffer(
            "max_logvar", torch.full((self.num_members, 1, self.num_heads), 0.5)
        )
        self.register_buffer(
            "min_logvar", torch.full((self.num_members, 1, self.num_heads), -10.0)
        )

    def _ensure_conv_fc(self, feature_size, device):
        """惰性构造 conv_fc：输入特征维首次确定或改变时重建并搬到 device。"""
        if self.conv_fc is None or self.conv_fc.in_features != feature_size:
            self.conv_fc = _MatrixLinear(
                self.num_members, feature_size, PSD_FEATURE_DIM
            ).to(device)

    def forward(
        self,
        images,
        hoprates,
        actions,
        return_logvar=False,
        validate_actions=True,
        return_mean_only=False,
    ):
        """一次 forward 算出全部 M 个成员的潜变量分布参数。

        Args:
            images: PSD 图像 [B,C,H,W]（上游 _images_tensor 已把 [B,H,W]
                升维），成员维在 _MatrixConv2d 内部复制/展平得到。
            hoprates: [B,1] 的跳速（Hz），内部经 normalize_hoprate 归一到
                [-10,10]，与 SAC 编码器使用同一套预处理。
            actions: [B,num_heads] 的整数 offsets（即 SAC 动作），内部 one-hot。
            return_logvar: True 返回 (mean, logvar)，False 返回 (mean, var)。
            validate_actions: False 时跳过取值域校验——训练/推理热路径上
                actions 已由 _prepare_reward_arrays/_batch_tensors 校验过。
            return_mean_only: True 时只算 mean（holdout 评估不需要方差，
                可省掉 logvar 头与夹逼计算）。

        Returns:
            mean [M,B,num_heads]；var 或 logvar 同形状。

        Raises:
            ValueError: actions 形状不是 [B,num_heads]、同时要求
                return_mean_only 与 return_logvar，或 actions 越界时。
        """
        if actions.ndim != 2 or actions.shape[1] != self.num_heads:
            raise ValueError(
                "actions must have shape [batch_size, "
                f"{self.num_heads}], got {tuple(actions.shape)}."
            )
        if return_mean_only and return_logvar:
            raise ValueError("return_mean_only and return_logvar cannot both be true.")
        if validate_actions and (
            torch.any(actions < 0) or torch.any(actions >= self.n_actions)
        ):
            raise ValueError("actions are outside the configured action range.")

        batch_size = images.shape[0]
        # 卷积链：输入 [B,C,H,W] -> conv 后 [B,M,16,H,W] -> [B,M,32,H,W]，
        # 每个成员各有一份独立权重。
        image_features = F.relu(self.norm1(self.conv1(images)))
        image_features = F.relu(self.norm2(self.conv2(image_features)))
        members, channels, height, width = image_features.shape[1:]
        # 池化不需要成员维语义，先折进 batch 维：[B,M,C,H,W] -> [B*M,C,H,W]，
        # 这样一次 MaxPool2d 覆盖所有成员。
        image_features = image_features.reshape(
            batch_size * members, channels, height, width
        )
        image_features = self.pool(image_features)
        if self.extra_pool is not None:
            image_features = self.extra_pool(image_features)
        # 再展平成 [B,M,feat]，feat = C*H'*W'（默认 32*50*50=80000）。
        image_features = image_features.reshape(batch_size, members, -1)
        self._ensure_conv_fc(image_features.shape[-1], images.device)
        # _MatrixLinear 约定成员维在前：[B,M,feat] -> permute -> [M,B,feat]。
        image_features = F.relu(
            self.conv_fc(image_features.permute(1, 0, 2).contiguous())
        )

        # hoprate 归一化用与 SAC 完全相同的函数，避免两套缩放导致特征分布漂移。
        normalized_hoprates = normalize_hoprate(
            hoprates, self.hoprate_min, self.hoprate_max
        )
        # [B,1] -> [B] -> 广播成 [M,B,1]：各成员用各自的 embedding 权重，
        # 因此同一 hoprate 在不同成员中会得到不同特征。
        hoprate_features = F.relu(
            self.hoprate_embedding(
                normalized_hoprates.unsqueeze(0).expand(
                    self.num_members, -1, -1
                )
            )
        )
        # [M,B,512+64=576] -> [M,B,256]：图像与跳速唯一的信息交互点。
        state_features = F.relu(
            self.fusion(torch.cat([image_features, hoprate_features], dim=2))
        )

        # offsets -> one-hot -> [B, num_heads*n_actions]，再广播到成员维；
        # 成员维为 0 时 expand 不占额外显存。
        one_hot_actions = F.one_hot(
            actions.long(), num_classes=self.n_actions
        ).to(dtype=images.dtype)
        action_features = F.relu(
            self.action_encoder(
                one_hot_actions.flatten(start_dim=1)
                .unsqueeze(0)
                .expand(self.num_members, -1, -1)
            )
        )
        # [M,B,256+200=456] -> [M,B,200]，SiLU 比 ReLU 更平滑，利于潜变量回归。
        fused = F.silu(
            self.fusion_head(torch.cat([state_features, action_features], dim=2))
        )
        mean = self.latent_mean_head(fused)
        if return_mean_only:
            return mean
        raw_logvar = self.latent_logvar_head(fused)
        # 双重 softplus 夹逼：softplus 恒正，故第一式把 logvar 压到 max_logvar
        # 之下、第二式把 logvar 抬到 min_logvar 之上，得到有界方差。
        logvar = self.max_logvar - F.softplus(self.max_logvar - raw_logvar)
        logvar = self.min_logvar + F.softplus(logvar - self.min_logvar)
        if return_logvar:
            return mean, logvar
        # 默认返回方差（= 潜变量高斯采样所需）；调用方负责在需要时取 log。
        return mean, torch.exp(logvar)


def _prepare_reward_arrays(
    state_imgs, hoprates, actions, block_rewards, num_heads, n_actions
):
    """校验并把真实 replay 字段统一成 fit() 内部使用的 numpy 布局。

    这是奖励模型训练路径的数据入口守卫：形状、有限性、整数性、取值域都在
    这里一次性检查（失败即 ValueError），下游训练循环因此可以信任字段合法，
    不必在热路径上重复校验。约定：

    - state_imgs: float32 [B,H,W] 或 [B,C,H,W]；
    - hoprates: float32 [B]（上游任意布局先 reshape 成 1 维）；
    - actions: 先 rint 再判等，以接受浮点表示的整数 offsets，转 int64 [B,num_heads]；
    - block_rewards: float32 [B,num_heads]。

    要求至少 2 条真实 transition：fit() 需要切出非空的 holdout 子集，样本
    更少时无法同时保证训练集与 holdout 非空。

    Args:
        state_imgs/hoprates/actions/block_rewards: 真实 replay 的字段。
        num_heads: offset 头数，决定 actions/block_rewards 的第二维。
        n_actions: 每个头的动作数，用于校验 offsets 取值范围。

    Returns:
        tuple: (state_imgs, hoprates, actions, block_rewards)，已转换的 numpy
        数组（actions 为 int64，其余为 float32）。

    Raises:
        ValueError: 字段无法转成矩形数组、长度不一致、样本数不足、形状不符、
            含非有限值、actions 非整数或越界时。
    """
    try:
        state_imgs = np.asarray(state_imgs, dtype=np.float32)
        hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions)
        block_rewards = np.asarray(block_rewards, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("Reward-model fields must be rectangular arrays.") from exc

    lengths = {
        len(state_imgs),
        len(hoprates),
        len(actions),
        len(block_rewards),
    }
    if len(lengths) != 1:
        raise ValueError("Reward-model fields have inconsistent lengths.")
    size = lengths.pop()
    if size < 2:
        raise ValueError("Need at least two real transitions to fit the ensemble.")
    if state_imgs.ndim not in (3, 4):
        raise ValueError("state_imgs must have shape [B,H,W] or [B,C,H,W].")
    if not np.all(np.isfinite(hoprates)):
        raise ValueError("hoprates must contain only finite values.")
    if actions.shape != (size, int(num_heads)):
        raise ValueError(
            "actions must have shape "
            f"({size}, {int(num_heads)}), got {actions.shape}."
        )
    if not np.all(np.isfinite(actions)):
        raise ValueError("actions must contain only finite values.")
    rounded_actions = np.rint(actions)
    if not np.allclose(actions, rounded_actions):
        raise ValueError("actions must be integer-valued.")
    actions = rounded_actions.astype(np.int64, copy=False)
    if np.any(actions < 0) or np.any(actions >= int(n_actions)):
        raise ValueError("actions are outside the configured action range.")
    expected_rewards_shape = (size, int(num_heads))
    if block_rewards.shape != expected_rewards_shape:
        raise ValueError(
            "block_rewards must have shape "
            f"{expected_rewards_shape}, got {block_rewards.shape}."
        )
    if not np.all(np.isfinite(block_rewards)):
        raise ValueError("block_rewards must contain only finite values.")
    return state_imgs, hoprates, actions, block_rewards


class StepRewardEnsemble(nn.Module):
    """矩阵式 CNN 概率奖励集成：为整条 FHSS 决策上下文预测各 offset 的 step reward。

    只建模一步 block reward，不预测下一状态（状态推进由适配层用 replay 中
    真实存储的外生 next_state_imgs/next_hoprates 补全）。每次 fit() 都在
    全部真实 replay 上从头训练：成员权重与共享 Adam 优化器状态全部重新
    初始化，绝不 warm-start。所有成员共享同一 train/holdout 划分与同一
    数据顺序，由单次优化器 step 一起更新。

    holdout 指标为真实 reward 单位下的 MSE（sigmoid 有界预测 vs clamp 后的
    有界目标），据此选 elite_size 个最优成员；推理时每条 transition 随机挑
    一个 elite，在潜变量空间采样后经 sigmoid 映射回该样本的物理边界内，
    因此预测结构性地不会越界。

    训练与推理都在 CUDA 上（混合精度要求 CUDA），autocast 默认 bfloat16；
    checkpoint 只保存这个外层模块（member 子模块随之落盘），格式见
    save_checkpoint / load_checkpoint。
    """

    def __init__(
        self,
        network_size,
        elite_size,
        num_heads,
        n_actions,
        reward_config,
        hoprate_min=10.0,
        hoprate_max=1000.0,
        ber_min=DEFAULT_BER_MIN,
        ber_max=DEFAULT_BER_MAX,
        logit_epsilon=DEFAULT_LOGIT_EPSILON,
        hidden_size=200,
        learning_rate=1e-3,
        weight_decay=1e-5,
        device=None,
        precision="float32",
        compile_model=False,
        extra_pool=False,
    ):
        super().__init__()
        if network_size <= 0:
            raise ValueError("network_size must be positive.")
        if elite_size <= 0 or elite_size > network_size:
            raise ValueError("elite_size must be in [1, network_size].")
        if num_heads <= 0 or n_actions <= 0:
            raise ValueError("num_heads and n_actions must be positive.")
        if hidden_size <= 0 or learning_rate <= 0.0 or weight_decay < 0.0:
            raise ValueError("Invalid reward-model optimizer or hidden-size settings.")
        if hoprate_max <= hoprate_min:
            raise ValueError("hoprate_max must be greater than hoprate_min.")
        if not np.isfinite(ber_min) or not np.isfinite(ber_max) or ber_max <= ber_min:
            raise ValueError("BER bounds must be finite and satisfy ber_max > ber_min.")
        if not 0.0 < float(logit_epsilon) < 0.5:
            raise ValueError("logit_epsilon must be between zero and 0.5.")

        self.network_size = int(network_size)
        self.elite_size = int(elite_size)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions)
        self.reward_config = _canonical_reward_config(reward_config)
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)
        self.ber_min = float(ber_min)
        self.ber_max = float(ber_max)
        self.logit_epsilon = float(logit_epsilon)
        self.hidden_size = int(hidden_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.device = torch.device(device or "cuda")
        precision = str(precision).lower()
        if precision not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                "precision must be one of 'float32', 'bfloat16', or 'float16'."
            )
        if self.device.type != "cuda" and precision != "float32":
            raise ValueError("Mixed precision reward training requires CUDA.")
        self.precision = precision
        self.compile_model = bool(compile_model)
        self.extra_pool = bool(extra_pool)
        if self.compile_model and not hasattr(torch, "compile"):
            raise RuntimeError("This PyTorch build does not provide torch.compile.")
        # member/optimizer 由 _rebuild_member_and_optimizer 惰性创建：构造时
        # 还不知道 observation_shape，惰性层无法物化。
        self.member = None
        self.optimizer = None
        self._member_observation_shape = None
        # 用 object.__setattr__ 绕开 nn.Module 的 __setattr__，避免把编译后的
        # 模块（含优化器闭包等非参数对象）注册进 _modules 而破坏 state_dict。
        object.__setattr__(self, "_compiled_member", None)
        # fit 之前的占位 elite：仅表示"尚未排名"，一旦 fit 就被 holdout MSE
        # 升序选出的真实下标覆盖。
        self.elite_model_idxes = list(range(self.elite_size))
        self.observation_shape = None
        self.is_fitted = False
        self.last_train_stats = {}

    def _autocast_context(self):
        """按 precision 返回 autocast 上下文，float32 或 CPU 时为空上下文。

        统一返回上下文对象而不是让调用方写 if 分支，训练/评估/推理三条
        路径保持同一份代码，避免"开/不开 autocast"两份实现逐渐漂移。
        """
        if self.device.type != "cuda" or self.precision == "float32":
            return nullcontext()
        dtype = (
            torch.bfloat16 if self.precision == "bfloat16" else torch.float16
        )
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _new_grad_scaler(self):
        """为 float16 训练构造 GradScaler（bfloat16/float32 下为禁用的 no-op）。

        bfloat16 与 float32 的动态范围无需梯度缩放，只有 float16 存在下溢
        风险；每次 fit() 都新建一个，避免沿用上一轮 fit 的缩放状态。
        """
        enabled = self.device.type == "cuda" and self.precision == "float16"
        return torch.amp.GradScaler("cuda", enabled=enabled)

    def _compute_member(self):
        """返回当前生效的成员网络：启用 torch.compile 时优先编译版本。"""
        return self._compiled_member or self.member

    def _dummy_tensors(self, observation_shape):
        """构造一次前向所需的形状探针张量，用于物化惰性层。

        observation_shape 是单条观测的空间形状：两轴 [H,W] 视为单通道灰度
        图，三轴 [C,H,W] 原样使用。hoprate 取区间中点（归一化后恰为 0，不
        落在归一化边界上），actions 全零（合法取值）。这些张量的数值会被
        丢弃，只有形状/通道数参与权重物化。

        Args:
            observation_shape: 单条观测形状，[H,W] 或 [C,H,W]。

        Returns:
            (images [1,C,H,W], hoprates [1,1], actions [1,num_heads])

        Raises:
            ValueError: observation_shape 轴数不是 2 或 3 时。
        """
        shape = tuple(int(value) for value in observation_shape)
        if len(shape) == 2:
            images = torch.zeros((1, 1, *shape), device=self.device)
        elif len(shape) == 3:
            images = torch.zeros((1, *shape), device=self.device)
        else:
            raise ValueError("observation_shape must have two or three axes.")
        hoprates = torch.full(
            (1, 1), (self.hoprate_min + self.hoprate_max) / 2.0, device=self.device
        )
        actions = torch.zeros(
            (1, self.num_heads), dtype=torch.long, device=self.device
        )
        return images, hoprates, actions

    def _rebuild_member_and_optimizer(self, observation_shape):
        """把成员权重与共享 Adam 状态复位，为本次 fit() 从头训练做准备。

        分两条路径：
        - 首次调用或 observation_shape 变化：新建 MatrixStepRewardMember，
          用 _dummy_tensors 跑一次前向以物化全部惰性层（_MatrixConv2d 的
          权重、conv_fc 的输入维），再建共享 Adam；compile_model 时同时
          建立 torch.compile 版本。
        - 形状未变：保留模块对象（compile 缓存与已物化的惰性层仍然有效），
          只对 _MatrixConv2d/_MatrixLinear/_MatrixGroupNorm 原地重初始化，
          并清空优化器状态，等价于一个全新模型。

        逐类型原地重初始化而不是重建模块，是为了不让 torch.compile 因模块
        对象更换而失效；两条路径都保证 fit() 绝不 warm-start。
        """
        observation_shape = tuple(int(value) for value in observation_shape)
        if (
            self.member is None
            or self._member_observation_shape != observation_shape
        ):
            self.member = MatrixStepRewardMember(
                self.network_size,
                self.num_heads,
                self.n_actions,
                self.hoprate_min,
                self.hoprate_max,
                self.hidden_size,
                extra_pool=self.extra_pool,
            ).to(self.device)
            with torch.no_grad():
                self.member(*self._dummy_tensors(observation_shape))
            self.optimizer = torch.optim.Adam(
                self.member.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
            self._member_observation_shape = observation_shape
            if self.compile_model:
                object.__setattr__(
                    self,
                    "_compiled_member",
                    torch.compile(self.member, fullgraph=False),
                )
            return

        # 复用已有模块：逐类型重初始化参数，保持 torch.compile 的图缓存有效。
        with torch.no_grad():
            for module in self.member.modules():
                if isinstance(module, _MatrixConv2d):
                    # 与 _MatrixConv2d._materialize 保持同一初始化尺度。
                    nn.init.normal_(module.weight, 0.0, 0.1)
                    nn.init.zeros_(module.bias)
                elif isinstance(module, _MatrixLinear):
                    std = 1.0 / (2.0 * np.sqrt(max(1, module.in_features)))
                    nn.init.trunc_normal_(
                        module.weight,
                        std=std,
                        a=-2.0 * std,
                        b=2.0 * std,
                    )
                    nn.init.zeros_(module.bias)
                elif isinstance(module, _MatrixGroupNorm):
                    module.norm.reset_parameters()
        # 清空 Adam 的 exp_avg/exp_avg_sq/step，等价于新建优化器。
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer.state.clear()

    def reward_bounds(self, hoprates):
        """物理 reward 上下界（numpy，均为 [B,1]），供适配层与诊断脚本调用。"""
        return reward_bounds_from_config(
            hoprates,
            self.reward_config,
            ber_min=self.ber_min,
            ber_max=self.ber_max,
        )

    def _reward_bounds_tensor(self, hoprates):
        """reward_bounds 的 torch 版本，返回 ([B,1], [B,1]) 的 (lower, upper)。

        热路径上使用以避免 numpy<->torch 往返；[B,1] 的边界可直接广播到
        [B,num_heads] 的目标与 [M,B,num_heads] 的预测。
        """
        base = (
            # 与 reward_bounds_from_config 同一公式的 torch 版本。
            self.reward_config["base_reward"]
            - self.reward_config["hoprate_penalty"] * hoprates
        )
        reward_at_ber_min = (
            base - self.reward_config["ber_penalty"] * self.ber_min
        )
        reward_at_ber_max = (
            base - self.reward_config["ber_penalty"] * self.ber_max
        )
        # min/max 而非假定方向：ber_penalty 的符号不受约束。
        return (
            torch.minimum(reward_at_ber_min, reward_at_ber_max),
            torch.maximum(reward_at_ber_min, reward_at_ber_max),
        )

    def _bound_rewards_tensor(self, rewards, hoprates):
        """把真实 reward clamp 进物理边界，得到 holdout MSE 用的有界目标。

        reward_bounds 已保证 lower <= upper，因此先 max 后 min 即等价于
        clamp，不需要额外判序。
        """
        lower, upper = self._reward_bounds_tensor(hoprates)
        return torch.minimum(torch.maximum(rewards, lower), upper)

    def _targets_to_latent(self, rewards, hoprates):
        """真实 reward -> 潜变量目标：饱和、归一化、端点避让、logit 变换。

        四步（[B,num_heads] 的 reward 与 [B,1] 的边界广播）：
        1. clamp 到 [lower, upper]：物理上不可能的 reward（如 BER 越界）被
           饱和到边界，使训练目标落在推理时可达的集合内；
        2. 线性归一化到 (0, 1)：每条样本有各自的物理区间，只有归一化后
           不同 hoprate 的样本才处于同一尺度，NLL 才不会偏向幅值大的样本；
        3. clamp 到 [logit_epsilon, 1-logit_epsilon]：logit 在 0/1 处发散，
           不避让会得到 ±inf 目标；
        4. logit = log(p) - log1p(-p)：把有界目标映射到无界潜变量空间，
           与网络输出（无约束的 mean/logvar）同域。

        Returns:
            (latent_targets, bounded_rewards)：前者 [B,num_heads] 用于 NLL
            损失；后者 [B,num_heads] 为 clamp 后的目标，用于在真实 reward
            单位下计算 holdout MSE（elite 排名据此产生）。
        """
        lower, upper = self._reward_bounds_tensor(hoprates)
        bounded_rewards = torch.minimum(torch.maximum(rewards, lower), upper)
        # 归一化：把各样本不同的物理区间拉到统一的 (0,1) 尺度。
        normalized = (bounded_rewards - lower) / (upper - lower)
        normalized = normalized.clamp(
            self.logit_epsilon, 1.0 - self.logit_epsilon
        )
        # logit 变换：log(p/(1-p))，与 sigmoid 互逆。
        latent_targets = torch.log(normalized) - torch.log1p(-normalized)
        return latent_targets, bounded_rewards

    def _latent_to_rewards_tensor(self, latent_values, hoprates):
        """潜变量 -> 有界 reward（torch）：lower + (upper-lower)*sigmoid(z)。

        sigmoid 把 z 压进 (0,1) 再仿射回该样本的物理区间，因此预测值在结构
        上不可能越界。holdout 评估用它把 mean 映射到真实 reward 单位；
        [M,B,num_heads] 的 z 与 [B,1] 的边界按广播对齐。
        """
        lower, upper = self._reward_bounds_tensor(hoprates)
        return lower + (upper - lower) * torch.sigmoid(latent_values)

    def _latent_to_rewards_numpy(self, latent_values, hoprates):
        """潜变量 -> 有界 reward（numpy 版），推理与采样路径使用。

        reward_bounds 返回 [B,1]，当 latent_values 带额外的成员/头维度
        （[M,B,num_heads] 或 [B,num_heads]）时逐级前插新轴以对齐广播；
        sigmoid 用 _stable_sigmoid，保证大 |z| 下不产生 inf/nan。
        """
        latent_values = np.asarray(latent_values)
        lower, upper = self.reward_bounds(hoprates)
        while lower.ndim < latent_values.ndim:
            lower = lower[np.newaxis, ...]
            upper = upper[np.newaxis, ...]
        return lower + (upper - lower) * _stable_sigmoid(latent_values)

    def _target_saturation_fraction(self, hoprates, block_rewards):
        """统计真实 reward 落在物理边界外的比例（诊断量，不参与训练）。

        比例偏高意味着环境回报已饱和（BER 越界或 hoprate 惩罚过强），此时
        归一化目标会大量堆在 logit_epsilon 端点附近，潜变量空间的有效动态
        范围被压缩、梯度变弱。该值记入 last_train_stats 供训练诊断。

        Args:
            hoprates: [B] 的跳速。
            block_rewards: [B,num_heads] 的真实 reward。

        Returns:
            float: 越界元素占比，范围 [0, 1]。

        Raises:
            ValueError: block_rewards 形状不是 [B,num_heads] 时。
        """
        rewards = np.asarray(block_rewards, dtype=np.float32)
        expected_shape = (len(hoprates), self.num_heads)
        if rewards.shape != expected_shape:
            raise ValueError(
                f"block_rewards must have shape {expected_shape}, got {rewards.shape}."
            )
        lower, upper = self.reward_bounds(hoprates)
        return float(np.mean((rewards < lower) | (rewards > upper)))

    def _to_device_tensor(self, values, dtype):
        """把 numpy/张量输入搬到训练设备并统一 dtype（张量走非阻塞拷贝）。"""
        if torch.is_tensor(values):
            return values.to(
                device=self.device,
                dtype=dtype,
                non_blocking=True,
            )
        return torch.as_tensor(values, dtype=dtype, device=self.device)

    def _images_tensor(self, images):
        """把 [B,H,W] 观测升维成 [B,1,H,W]（本项目的 PSD 单通道灰度图）。"""
        tensor = self._to_device_tensor(images, torch.float32)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)
        if tensor.ndim != 4:
            raise ValueError(
                "state images must have shape [B,H,W] or [B,C,H,W], got "
                f"{tuple(tensor.shape)}."
            )
        return tensor

    def _batch_tensors(self, batch, include_rewards=True):
        """把一批原始字段转成设备上的张量元组，供 member.forward 使用。

        这是推理路径的数据守卫（训练路径已由 _prepare_reward_arrays 把关）：
        hoprates 统一成 [B,1]、actions 校验整数性与取值域后转 int64、
        block_rewards 校验形状。include_rewards=False 时只返回决策上下文，
        用于只需上下文、没有真实 reward 的采样场景。

        Args:
            batch: (state_imgs, hoprates, actions[, block_rewards]) 的序列。
            include_rewards: False 时不解析第 4 项。

        Returns:
            include_rewards=True 时返回
            (images [B,1,H,W], hoprates [B,1], actions [B,num_heads],
            rewards [B,num_heads])；否则只返回前三项。

        Raises:
            ValueError: hoprates/actions/rewards 形状不符、含非有限值、
                actions 非整数或越界时。
        """
        images, hoprates, actions, *remaining = batch
        images_t = self._images_tensor(images)
        hoprates_array = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        if hoprates_array.shape != (len(images),):
            raise ValueError(f"hoprates must have shape ({len(images)},).")
        if not np.all(np.isfinite(hoprates_array)):
            raise ValueError("hoprates must contain only finite values.")
        # 统一成 [B,1]：normalize_hoprate 与 reward 边界都按列向量广播。
        hoprates_t = torch.as_tensor(
            hoprates_array, dtype=torch.float32, device=self.device
        ).view(-1, 1)
        raw_actions = np.asarray(actions)
        if raw_actions.shape != (len(images), self.num_heads):
            raise ValueError(
                "actions must have shape "
                f"({len(images)}, {self.num_heads}), got {raw_actions.shape}."
            )
        # 接受浮点表示的整数 offsets（上游 replay 可能以 float 存储）。
        rounded_actions = np.rint(raw_actions)
        if not np.allclose(raw_actions, rounded_actions):
            raise ValueError("actions must be integer-valued.")
        actions_t = torch.as_tensor(
            rounded_actions, dtype=torch.long, device=self.device
        )
        if np.any(rounded_actions < 0) or np.any(rounded_actions >= self.n_actions):
            raise ValueError("actions are outside the configured action range.")
        if not include_rewards:
            return images_t, hoprates_t, actions_t

        rewards = np.asarray(remaining[0], dtype=np.float32)
        expected_shape = (len(images), self.num_heads)
        if rewards.shape != expected_shape:
            raise ValueError(
                f"block_rewards must have shape {expected_shape}, got {rewards.shape}."
            )
        rewards_t = torch.as_tensor(
            rewards, dtype=torch.float32, device=self.device
        )
        return images_t, hoprates_t, actions_t, rewards_t

    @staticmethod
    def _probabilistic_loss(mean, logvar, targets):
        """全体成员共享的 latent Gaussian NLL，逐元素平均后得到标量损失。

        NLL = (mean - target)^2 * exp(-logvar) + logvar：logvar 项充当方差
        正则，阻止网络把方差推向 0 来"白拿"精度。mean/logvar 为
        [M,B,num_heads]，targets 为 [B,num_heads]（unsqueeze 后广播到成员
        维），因此 M 个成员各自产生一份 NLL 并被一起平均——所有成员由同一次
        backward 更新，这正是矩阵式 ensemble 的收益所在。
        """
        return torch.mean(
            torch.square(mean - targets.unsqueeze(0)) * torch.exp(-logvar) + logvar
        )

    def _evaluate_ensemble(self, loader):
        """一次并行前向算出每个成员的 holdout MSE（真实 reward 单位）。

        矩阵网络让 M 个成员共享一次前向，因此这里没有逐成员的 Python 循环：
        squared_error 沿成员维（dim=1）累加、再除以元素总数，得到 [M] 的
        逐成员 MSE。用 return_mean_only 跳过 logvar 头，评估在
        inference_mode 下进行、不建图。比较对象是 sigmoid 有界预测与 clamp
        后的有界目标，数值与物理 reward 同量纲，elite 排名据此产生。

        Args:
            loader: holdout 的 DataLoader（batch 为字段字典）。

        Returns:
            np.ndarray: 形状 [network_size] 的逐成员 MSE（float32）。
        """
        self.member.eval()
        squared_error = torch.zeros(
            self.network_size, dtype=torch.float32, device=self.device
        )
        value_count = 0
        with torch.inference_mode():
            for batch in loader:
                images_t = batch["state_imgs"]
                hoprates_t = batch["hoprates"]
                actions_t = batch["actions"]
                rewards_t = batch["block_rewards"]
                with self._autocast_context():
                    latent_mean = self._compute_member()(
                        images_t,
                        hoprates_t,
                        actions_t,
                        validate_actions=False,
                        return_mean_only=True,
                    )
                # 转 float32 再映射：autocast 下 latent_mean 可能是 bf16，
                # 而边界是 float32，先升精度避免 MSE 累加时的精度损失。
                reward_prediction = self._latent_to_rewards_tensor(
                    latent_mean.float(), hoprates_t.float()
                )
                # 目标同样 clamp 到物理边界，保证比较双方在同一可达集合内。
                bounded_targets = self._bound_rewards_tensor(rewards_t, hoprates_t)
                # 沿 (batch, num_heads) 求和、成员维保留 -> 每个成员一个平方误差和；
                # bounded_targets.unsqueeze(0) 广播到成员维。
                squared_error += torch.sum(
                    torch.square(reward_prediction - bounded_targets.unsqueeze(0)),
                    dim=(1, 2),
                )
                # 除以元素总数（= B*num_heads 的累计），得到真实单位下的 MSE。
                value_count += rewards_t.numel()
        return (squared_error / max(1, value_count)).cpu().numpy()

    def fit(
        self,
        state_imgs=None,
        hoprates=None,
        actions=None,
        block_rewards=None,
        batch_size=256,
        holdout_ratio=0.2,
        patience=5,
        max_epochs=100,
        min_improvement=0.01,
    ):
        """在当前真实 replay 上从头并行重训全部成员。

        每次调用都从零开始：矩阵权重与共享 Adam 优化器状态全部重新初始化
        （见 _rebuild_member_and_optimizer），绝不延续上一次 fit 的结果。
        replay 张量一次性常驻训练设备（始终 CUDA），由主进程 DataLoader 经
        _DeviceBatchDataset 的整批索引流式取数，不做逐 batch 的 H2D 拷贝。
        所有成员共享同一 train/holdout 划分与每个 epoch 的同一数据顺序，由
        单次优化器 step 一起更新。全局早停由"最优成员的 holdout loss"驱动：
        连续 patience 个 epoch 的相对提升不超过 min_improvement 即整体停止。
        最终使用最后一个已训练 epoch 的权重（不保留 best-epoch 快照）；
        holdout_curves/train_curves 每个已训练 epoch 记录一项，且不做未训练
        权重的 fit 前预评估。

        settings.BACKWARD_TIMING_ENABLED 为 True 时，每个 batch 用 CUDA event
        打印真实反传 GPU 耗时，带 [MBPO-BWD] 前缀。

        Args:
            state_imgs/hoprates/actions/block_rewards: 真实 replay 字段，
                形状契约见 _prepare_reward_arrays，四项均必填。
            batch_size: 训练 mini-batch 大小。
            holdout_ratio: holdout 占比，用于早停与 elite 选择。
            patience: 早停容忍的连续无提升 epoch 数（0 表示不早停）。
            max_epochs: 训练 epoch 数上限。
            min_improvement: 相对提升阈值，低于它视为无提升。

        Returns:
            dict: last_train_stats，含逐 epoch 的 train/holdout 曲线、
            holdout_losses、elite_model_idxes、train/holdout 规模、
            target_saturation_fraction、epoch_times 与 fit_time_sec 等。

        Raises:
            ValueError: 训练超参非法、字段缺失/形状不符，或观测形状与已拟合
                模型不一致时。
            RuntimeError: holdout loss 出现非有限值时——训练发散要显式失败，
                而不是把 NaN 带进 elite 选择。
        """
        if batch_size <= 0 or patience < 0 or max_epochs <= 0:
            raise ValueError("Invalid ensemble training limits.")
        if not 0.0 < holdout_ratio < 1.0:
            raise ValueError("holdout_ratio must be between zero and one.")
        if min_improvement < 0.0:
            raise ValueError("min_improvement must be non-negative.")

        if any(
            value is None
            for value in (state_imgs, hoprates, actions, block_rewards)
        ):
            raise ValueError(
                "state_imgs, hoprates, actions, and block_rewards are required."
            )
        state_imgs, hoprates, actions, block_rewards = _prepare_reward_arrays(
            state_imgs,
            hoprates,
            actions,
            block_rewards,
            self.num_heads,
            self.n_actions,
        )
        first_state = state_imgs[0]
        observation_shape = tuple(first_state.shape)
        # 观测形状一旦确定就锁定：改变它会让已物化的惰性层与 checkpoint 失效，
        # 因此这里显式报错而不是静默重建。
        if self.observation_shape is None:
            self.observation_shape = observation_shape
        elif observation_shape != self.observation_shape:
            raise ValueError(
                "Observation shape differs from the fitted reward model: "
                f"expected {self.observation_shape}, got {observation_shape}."
            )
        # 先记录饱和比例（诊断量），不参与训练。
        target_saturation_fraction = self._target_saturation_fraction(
            hoprates, block_rewards
        )

        images_t = torch.as_tensor(state_imgs, dtype=torch.float32, device=self.device)
        if images_t.ndim == 3:
            # [B,H,W] -> [B,1,H,W]，与 SAC 编码器的单通道 PSD 约定一致。
            images_t = images_t.unsqueeze(1)
        hoprates_t = torch.as_tensor(
            hoprates, dtype=torch.float32, device=self.device
        ).view(-1, 1)
        actions_t = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.as_tensor(
            block_rewards, dtype=torch.float32, device=self.device
        )
        # 潜变量目标是数据的确定性函数（只依赖 reward 与 hoprate，与网络权重
        # 无关），因此每次 fit 只算一次并常驻设备，而不是每个 batch 重算。
        latent_all = self._targets_to_latent(rewards_t, hoprates_t)[0]
        tensors = (images_t, hoprates_t, actions_t, rewards_t, latent_all)
        dataset_size = images_t.shape[0]
        # 一次置换同时切出 holdout 与 train：两个子集互补且不重叠，早停/elite
        # 选择与训练数据严格隔离。
        permutation = np.random.permutation(dataset_size)
        holdout_size = min(
            max(1, int(dataset_size * holdout_ratio)), dataset_size - 1
        )
        holdout_indices = permutation[:holdout_size].astype(np.int64)
        train_indices = permutation[holdout_size:].astype(np.int64)
        batch_size = int(batch_size)
        train_dataset = _DeviceBatchDataset(tensors, train_indices)
        holdout_dataset = _DeviceBatchDataset(tensors, holdout_indices)
        # 训练集不小于一个 batch 时丢弃末尾不满的 batch：尾批会因形状不同
        # 触发一次 torch.compile 重编译，而它对优化预算的影响可忽略。
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=len(train_indices) >= batch_size,
            collate_fn=_collate_reward_batch,
        )
        holdout_loader = DataLoader(
            holdout_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=_collate_reward_batch,
        )
        # 权重与优化器在此处复位：fit 是"从零重训"，不是继续训练。
        self._rebuild_member_and_optimizer(observation_shape)
        holdout_curves = [[] for _ in range(self.network_size)]
        train_curves = []
        timing_enabled = bool(settings.TIMING_ENABLED)
        epoch_times = []
        fit_start = time.time()
        grad_scaler = self._new_grad_scaler()
        backward_timing_enabled = bool(settings.BACKWARD_TIMING_ENABLED)
        if backward_timing_enabled:
            # 每次 fit 只分配一对计时 event，之后每个 batch 在当前 CUDA 流上
            # 重新记录。训练始终在 CUDA（A800）上进行，故不设 CPU 计时回退。
            backward_start_event = torch.cuda.Event(enable_timing=True)
            backward_end_event = torch.cuda.Event(enable_timing=True)
        num_train_batches = len(train_loader)

        best_global_loss = None
        stale_epochs = 0
        epochs_run = 0
        train_curves = [[] for _ in range(self.network_size)]

        for epoch in range(max_epochs):
            epoch_start = time.time() if timing_enabled else None
            self.member.train()
            epoch_loss_total = None
            epoch_batch_count = 0
            for batch in train_loader:
                # 整批张量已在设备上，字段取用零拷贝（见 _collate_reward_batch）。
                images_b = batch["state_imgs"]
                hoprates_b = batch["hoprates"]
                actions_b = batch["actions"]
                latent_targets = batch["latent_targets"]
                with self._autocast_context():
                    # 一次前向得到全部 M 个成员的 mean/logvar（[M,B,num_heads]）。
                    mean, logvar = self._compute_member()(
                        images_b,
                        hoprates_b,
                        actions_b,
                        return_logvar=True,
                        validate_actions=False,
                    )
                # 损失在 float32 下计算：bf16 的 NLL 累加误差会掩盖成员间差异。
                loss = self._probabilistic_loss(
                    mean.float(), logvar.float(), latent_targets.float()
                )
                self.optimizer.zero_grad(set_to_none=True)
                # 只给本 batch 的反传计时。CUDA kernel 是异步的，用墙钟包住
                # backward() 只能测到下发时间；在流上打 CUDA event 才能精确
                # 框住反传 kernel，得到真实 GPU 耗时。
                if backward_timing_enabled:
                    backward_start_event.record()
                if grad_scaler.is_enabled():
                    # float16：先缩放损失再反传，避免小梯度下溢为 0。
                    grad_scaler.scale(loss).backward()
                else:
                    loss.backward()
                if backward_timing_enabled:
                    backward_end_event.record()
                    backward_end_event.synchronize()
                    backward_sec = (
                        backward_start_event.elapsed_time(backward_end_event)
                        / 1000.0
                    )
                    print(
                        f"[MBPO-BWD] epoch={epoch + 1}/{max_epochs} "
                        f"batch={epoch_batch_count + 1}/{num_train_batches} "
                        f"loss={loss.detach().item():.6f} "
                        f"backward={backward_sec * 1000.0:.3f} ms",
                        flush=True,
                    )
                if grad_scaler.is_enabled():
                    grad_scaler.step(self.optimizer)
                    grad_scaler.update()
                else:
                    self.optimizer.step()
                # 累积标量损失用于曲线记录：detach 以免把图留到 epoch 结束。
                detached_loss = loss.detach()
                epoch_loss_total = (
                    detached_loss
                    if epoch_loss_total is None
                    else epoch_loss_total + detached_loss
                )
                epoch_batch_count += 1
            # 全体成员共享同一个损失值，因此 train_curves 每条曲线相同；
            # 保留逐成员结构是为了与 holdout_curves 的索引语义对齐。
            mean_train_loss = float(
                (epoch_loss_total / max(1, epoch_batch_count)).item()
            )
            for member_idx in range(self.network_size):
                train_curves[member_idx].append(mean_train_loss)

            epoch_holdout = self._evaluate_ensemble(holdout_loader)
            if not np.all(np.isfinite(epoch_holdout)):
                raise RuntimeError("Reward-model holdout loss became non-finite.")
            for member_idx, value in enumerate(epoch_holdout):
                holdout_curves[member_idx].append(float(value))

            # 早停以"当轮最优成员"的 holdout MSE 为准：只要还有成员在进步，
            # 整个 ensemble 就继续训练。
            epoch_best_loss = float(np.min(epoch_holdout))
            if best_global_loss is None:
                # 第一个训练 epoch 建立早停基线：不对未训练权重做 fit 前
                # holdout 预评估。
                best_global_loss = epoch_best_loss
                stale_epochs = 0
            else:
                # 相对提升（分母取绝对值并加下限，避免除零）。
                relative_improvement = (
                    (best_global_loss - epoch_best_loss)
                    / max(abs(best_global_loss), 1e-12)
                )
                if relative_improvement > min_improvement:
                    best_global_loss = epoch_best_loss
                    stale_epochs = 0
                else:
                    stale_epochs += 1
            epochs_run = epoch + 1
            if timing_enabled:
                epoch_times.append(time.time() - epoch_start)
            if stale_epochs >= patience:
                break

        self.member.eval()

        # 最后一个 epoch 的 holdout 评估用的正是当前权重与 eval 模式，直接复用
        # 其结果，不再多跑一次前向。
        holdout_losses = epoch_holdout
        # elite = holdout MSE 升序前 elite_size 名（越小越好）；argsort 升序。
        self.elite_model_idxes = np.argsort(holdout_losses)[
            : self.elite_size
        ].tolist()
        self.is_fitted = True
        self.last_train_stats = {
            "epochs": [epochs_run] * self.network_size,
            "holdout_curves": holdout_curves,
            "train_curves": train_curves,
            "holdout_losses": holdout_losses,
            "holdout_loss_mean": float(np.mean(holdout_losses)),
            "elite_model_idxes": list(self.elite_model_idxes),
            "train_size": int(len(train_indices)),
            "holdout_size": int(len(holdout_indices)),
            "target_saturation_fraction": target_saturation_fraction,
            "epoch_times": epoch_times,
            "fit_time_sec": (time.time() - fit_start) if timing_enabled else None,
        }
        return self.last_train_stats

    def _validate_prediction_inputs(self, state_imgs, hoprates, actions):
        """推理入口的输入校验：形状、非空、有限性，不做任何裁剪或改写。

        与训练路径的 _prepare_reward_arrays 不同，这里允许 actions 以任意
        数值类型传入（仅由 _batch_tensors 做整数性校验），并且只在 numpy
        层检查，避免在推理热路径上多做一次设备同步。
        """
        state_imgs = np.asarray(state_imgs, dtype=np.float32)
        hoprates = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        actions = np.asarray(actions)
        if state_imgs.ndim not in (3, 4):
            raise ValueError("state_imgs must have shape [B,H,W] or [B,C,H,W].")
        batch_size = len(state_imgs)
        if batch_size == 0:
            raise ValueError("Prediction inputs cannot be empty.")
        if hoprates.shape != (batch_size,):
            raise ValueError(f"hoprates must have shape ({batch_size},).")
        if actions.shape != (batch_size, self.num_heads):
            raise ValueError(
                "actions must have shape "
                f"({batch_size}, {self.num_heads}), got {actions.shape}."
            )
        if not np.all(np.isfinite(state_imgs)) or not np.all(np.isfinite(hoprates)):
            raise ValueError("Prediction inputs must contain only finite values.")
        return state_imgs, hoprates, actions

    def _predict_latent(self, state_imgs, hoprates, actions, batch_size=1024):
        """返回全部成员的潜变量高斯参数，形状 [M, B, num_heads]。

        矩阵网络一次前向即可算出所有成员，因此这里按 batch_size 分块只为
        限制显存峰值，而不是逐成员循环；分块结果沿 axis=1 拼接回完整 batch。
        返回 (mean, var)——member.forward 默认已把 logvar 取 exp。

        Args:
            state_imgs: [B,H,W] 或 [B,C,H,W] 的 PSD 图像。
            hoprates: [B] 的跳速（Hz）。
            actions: [B,num_heads] 的整数 offsets。
            batch_size: 分块前向的块大小。

        Returns:
            (np.ndarray, np.ndarray): (mean, var)，均为 float32、[M,B,num_heads]。

        Raises:
            RuntimeError: 模型尚未 fit 时。
            ValueError: batch_size 非正或输入形状不合法时。
        """
        if not self.is_fitted:
            raise RuntimeError("StepRewardEnsemble must be fitted before prediction.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        state_imgs, hoprates, actions = self._validate_prediction_inputs(
            state_imgs, hoprates, actions
        )
        self.member.eval()
        ensemble_means = []
        ensemble_variances = []
        with torch.inference_mode():
            for start in range(0, len(state_imgs), batch_size):
                stop = min(start + batch_size, len(state_imgs))
                tensors = self._batch_tensors(
                    (
                        state_imgs[start:stop],
                        hoprates[start:stop],
                        actions[start:stop],
                    ),
                    include_rewards=False,
                )
                with self._autocast_context():
                    means, variances = self.member(
                        *tensors,
                        validate_actions=False,
                    )
                # 立刻转 float32 并搬回 CPU：分块之间不需要保留设备侧中间量。
                ensemble_means.append(means.float().cpu().numpy())
                ensemble_variances.append(variances.float().cpu().numpy())
        return (
            np.concatenate(ensemble_means, axis=1).astype(np.float32, copy=False),
            np.concatenate(ensemble_variances, axis=1).astype(np.float32, copy=False),
        )

    def predict(self, state_imgs, hoprates, actions, batch_size=1024):
        """返回全部成员的有界 reward 位置与近似方差（[M,B,num_heads]）。

        方差用 delta method 近似：把潜变量高斯经 sigmoid 映射后，
        Var[r] ≈ (dr/dz)^2 * Var[z]，其中 dr/dz = (upper-lower)*s*(1-s)，
        s = sigmoid(z)。严格来说变换后的分布并非高斯，这里给的是供下游
        参考的不确定度近似量，不参与训练。

        Returns:
            (locations, variances)：均为 float32、[M,B,num_heads]，locations
            已落在各样本的物理 reward 边界内。
        """
        latent_means, latent_variances = self._predict_latent(
            state_imgs, hoprates, actions, batch_size=batch_size
        )
        reward_locations = self._latent_to_rewards_numpy(
            latent_means, hoprates
        )
        # 前插成员/头维以对齐 latent 的 [M,B,num_heads] 广播。
        lower, upper = self.reward_bounds(hoprates)
        lower = lower[np.newaxis, ...]
        upper = upper[np.newaxis, ...]
        sigmoid_values = _stable_sigmoid(latent_means)
        # sigmoid 的导数 s*(1-s)，再乘仿射斜率 (upper-lower) 得到 dr/dz。
        local_slopes = (upper - lower) * sigmoid_values * (1.0 - sigmoid_values)
        reward_variances = np.square(local_slopes) * latent_variances
        return (
            reward_locations.astype(np.float32, copy=False),
            reward_variances.astype(np.float32, copy=False),
        )

    def sample_rewards(
        self,
        state_imgs,
        hoprates,
        actions,
        deterministic=False,
        batch_size=1024,
    ):
        """每条 transition 随机挑一个 elite 成员，采样出一条完整 reward 向量。

        deterministic=True 时退化为 elite 的均值预测（评估/调试用），
        selected_model_idxes 全为 -1 表示未做随机选择。非确定性路径：先为
        每条 transition 独立随机选一个 elite，再用该成员在潜变量空间的
        mean/var 采样 z，最后 sigmoid 映射回该样本的物理边界内。每次只取
        一个成员，因此同一条 transition 的 num_heads 个 reward 共享同一成员，
        保持输出向量内部的成员一致性（与 MBPO 的 ensemble rollout 语义一致）。

        Args:
            state_imgs/hoprates/actions: 决策上下文，契约见 _batch_tensors。
            deterministic: True 时用 elite 均值而不做潜变量采样。
            batch_size: 分块前向的块大小。

        Returns:
            (rewards [B,num_heads] float32, stats: dict)：
            stats["selected_model_idxes"] 为每条 transition 选中的成员下标
            （deterministic 时为 -1）；stats["disagreement"] 为 elite 之间
            预测标准差在 num_heads 维上的均值（[B]），是模型分歧度诊断量，
            供适配层记录/调参使用。
        """
        latent_means, latent_variances = self._predict_latent(
            state_imgs, hoprates, actions, batch_size=batch_size
        )
        reward_locations = self._latent_to_rewards_numpy(
            latent_means, hoprates
        )
        # elite 子集上的预测离散度：std 沿成员维（axis=0）后对头取均值。
        elite_indices = np.asarray(self.elite_model_idxes, dtype=np.int64)
        elite_reward_locations = reward_locations[elite_indices]
        disagreement = np.mean(
            np.std(elite_reward_locations, axis=0), axis=1
        )
        # 用于按 (成员, 样本) 成对取值的列下标。
        item_indices = np.arange(latent_means.shape[1])

        if deterministic:
            # 均值集成：不做成员随机选择，故无成员下标可报。
            rewards = np.mean(elite_reward_locations, axis=0)
            selected_indices = np.full(
                latent_means.shape[1], -1, dtype=np.int64
            )
        else:
            # 每条 transition 独立均匀抽取一个 elite（有放回）。
            selected_indices = np.random.choice(
                elite_indices, size=latent_means.shape[1]
            )
            selected_means = latent_means[selected_indices, item_indices]
            selected_variances = latent_variances[
                selected_indices, item_indices
            ]
            # 潜变量空间采样：z = mean + eps * sqrt(var)，var 下限 1e-12 防除零/零方差。
            latent_samples = selected_means + np.random.normal(
                size=selected_means.shape
            ) * np.sqrt(np.maximum(selected_variances, 1e-12))
            # sigmoid 映射回物理边界内，保证合成 replay 中的 reward 合法。
            rewards = self._latent_to_rewards_numpy(latent_samples, hoprates)

        return rewards.astype(np.float32, copy=False), {
            "selected_model_idxes": selected_indices,
            "disagreement": disagreement.astype(np.float32, copy=False),
        }

    def _architecture_tag(self):
        """当前架构标识：基名 + extra_pool 时的 "_pool2" 后缀。

        checkpoint 的架构校验依赖它：extra_pool 会改变 conv_fc 的输入维，
        两种变体的 state_dict 互不兼容，必须靠标签区分（见 load_checkpoint）。
        """
        return REWARD_MODEL_ARCHITECTURE + ("_pool2" if self.extra_pool else "")

    def _config(self):
        """落盘/重建所需的全部构造参数（不含 device、precision 等运行时选项）。

        load_checkpoint 直接用 ``cls(**config, device=device)`` 重建模型，
        因此这里的键必须与 __init__ 的构造参数一一对应；device、precision、
        compile_model 属于部署期选择，故意不写入 checkpoint，避免换机器或
        换精度时被迫重训。
        """
        return {
            "network_size": self.network_size,
            "elite_size": self.elite_size,
            "num_heads": self.num_heads,
            "n_actions": self.n_actions,
            "reward_config": dict(self.reward_config),
            "hoprate_min": self.hoprate_min,
            "hoprate_max": self.hoprate_max,
            "ber_min": self.ber_min,
            "ber_max": self.ber_max,
            "logit_epsilon": self.logit_epsilon,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "extra_pool": self.extra_pool,
        }

    def save_checkpoint(self, path, metadata=None):
        """保存 reward v3 checkpoint（含成员权重、elite 下标与元数据）。

        目录不存在时自动创建。payload 各键用途：format_version 供加载时
        区分/拒绝旧格式；architecture 用于拒绝架构不匹配的权重；
        config 供 load_checkpoint 重建模型；observation_shape 供惰性层
        物化；elite_model_idxes 保留 elite 选择结果（推理端随机选成员要用）；
        metadata 是调用方附加的自由字段（如训练步数、环境配置），加载时原样
        返回，不参与任何校验。

        Args:
            path: 输出路径。
            metadata: 可选的附加信息 dict。

        Raises:
            RuntimeError: 模型尚未 fit（权重与 observation_shape 未就绪）时。
        """
        if not self.is_fitted or self.observation_shape is None:
            raise RuntimeError("Cannot save an unfitted reward model.")
        output_dir = os.path.dirname(os.path.abspath(path))
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        torch.save(
            {
                "format_version": REWARD_CHECKPOINT_FORMAT_VERSION,
                "model_type": "StepRewardEnsemble",
                "architecture": self._architecture_tag(),
                "config": self._config(),
                "observation_shape": list(self.observation_shape),
                "elite_model_idxes": list(self.elite_model_idxes),
                "model_state_dict": self.state_dict(),
                "metadata": dict(metadata or {}),
            },
            path,
        )

    def _materialize(self, observation_shape):
        """按已知观测形状建好成员网络并物化全部惰性层。

        load_checkpoint 路径使用：先按 config 构造（此时 observation_shape
        未知，conv_fc 与 _MatrixConv2d 的权重尚未分配），再用 _dummy_tensors
        跑一次前向，让惰性层按真实特征维分配参数，之后才能 load_state_dict。
        member 已存在时（例如重复调用）只做前向物化，不重建模块。
        """
        shape = tuple(int(value) for value in observation_shape)
        images, hoprates, actions = self._dummy_tensors(shape)
        if self.member is None:
            self.member = MatrixStepRewardMember(
                self.network_size,
                self.num_heads,
                self.n_actions,
                self.hoprate_min,
                self.hoprate_max,
                self.hidden_size,
                extra_pool=self.extra_pool,
            ).to(self.device)
        with torch.no_grad():
            self.member(images, hoprates, actions)
        self.observation_shape = shape

    @classmethod
    def load_checkpoint(
        cls,
        path,
        device="cuda",
        expected_num_heads=None,
        expected_n_actions=None,
        expected_observation_shape=None,
    ):
        """加载 reward v3 checkpoint，返回 (model, metadata)。

        显式拒绝 v1/v2 旧格式并给出重训提示：v1 用的是浅层状态编码器与两层
        state-action fusion，v2 用的是三层 CNN、两层 hoprate MLP 且成员是各自
        独立的网络，两者都与当前矩阵架构不兼容，静默加载只会得到错误模型。
        v3 还需校验 model_type、architecture（含 extra_pool 后缀）以及可选的
        num_heads/n_actions/observation_shape 一致性，全部通过后才重建模型、
        物化惰性层并载入权重。

        Args:
            path: checkpoint 路径。
            device: 加载设备，默认 "cuda"（训练/推理均要求 CUDA）。
            expected_num_heads: 期望的 offset 头数，非 None 时比对。
            expected_n_actions: 期望的每头动作数，非 None 时比对。
            expected_observation_shape: 期望的观测形状，非 None 时比对。

        Returns:
            (StepRewardEnsemble, dict): 已物化、is_fitted=True 且处于 eval
            模式的模型，以及保存时写入的 metadata。

        Raises:
            ValueError: 格式版本过旧/不支持、model_type 不符、config 缺失
                （架构无法校验）、架构标签不符或期望值不一致时。
        """
        payload = torch.load(path, map_location=device, weights_only=True)
        format_version = payload.get("format_version")
        if format_version == 1:
            raise ValueError(
                "Reward-model checkpoint format v1 uses the old shallow state "
                "encoder and two-layer state-action fusion; retrain the reward "
                "model to create a v3 checkpoint."
            )
        if format_version == 2:
            raise ValueError(
                "Reward-model checkpoint format v2 uses the old three-layer CNN, "
                "two-layer hoprate MLP and per-member networks; retrain the "
                "reward model to create a v3 checkpoint."
            )
        if format_version != REWARD_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("Unsupported reward-model checkpoint format.")
        if payload.get("model_type") != "StepRewardEnsemble":
            raise ValueError("Checkpoint does not contain a StepRewardEnsemble.")
        # 缺少 config 字典的畸形 v3 payload 无法校验架构（extra_pool 后缀），
        # 这里显式拒绝，而不是让下面的字典取值抛出裸 KeyError。
        config = payload.get("config")
        if not isinstance(config, dict):
            raise ValueError(
                "Reward-model checkpoint has no config dictionary; its "
                "architecture cannot be verified, so retrain the reward "
                "model with the current architecture."
            )
        config = dict(config)
        # 由 config 内的 extra_pool 复原期望架构标签，再与落盘的 architecture
        # 比对——两处必须同时一致，避免只改一处的畸形 checkpoint 蒙混过关。
        expected_architecture = REWARD_MODEL_ARCHITECTURE + (
            "_pool2" if config.get("extra_pool") else ""
        )
        if payload.get("architecture") != expected_architecture:
            raise ValueError(
                "Reward-model checkpoint architecture does not match "
                f"{expected_architecture!r}; retrain with the current architecture."
            )
        if expected_num_heads is not None and int(expected_num_heads) != int(
            config["num_heads"]
        ):
            raise ValueError("Reward-model checkpoint block count does not match.")
        if expected_n_actions is not None and int(expected_n_actions) != int(
            config["n_actions"]
        ):
            raise ValueError("Reward-model checkpoint action count does not match.")
        observation_shape = tuple(payload["observation_shape"])
        if (
            expected_observation_shape is not None
            and tuple(expected_observation_shape) != observation_shape
        ):
            raise ValueError("Reward-model checkpoint observation shape does not match.")

        model = cls(**config, device=device)
        # 先按 observation_shape 物化惰性层，state_dict 的键/形状才能对上。
        model._materialize(observation_shape)
        model.load_state_dict(payload["model_state_dict"])
        # elite 选择结果随 checkpoint 一起恢复：推理时 sample_rewards 需要它。
        model.elite_model_idxes = [
            int(index) for index in payload["elite_model_idxes"]
        ]
        # 载入即视为已拟合，且固定 eval 模式（推理不更新任何统计量）。
        model.is_fitted = True
        model.eval()
        return model, dict(payload.get("metadata", {}))
