"""FHSS offset 策略的十头因子化离散 Soft Actor-Critic（SAC）。

一、动作空间：一次 step = 十个条件独立的 categorical 决策
-------------------------------------------------------------------------------
环境一次 step 需要为 num_heads（默认 10）个 frequency block 各选定一个起始信道
offset，动作因此是一条 10 维离散向量，每个分量取值 [0, n_actions)。本实现把它
建模为"给定状态后十头条件独立"的因子化分布族：PolicyNet 用一个共享编码器提取
状态特征，再由 10 个参数互不共享的线性头各输出 n_actions 个 logits，联合 log
概率 = 十头 log 概率之和（等价于联合概率 = 十头概率之积）。因子化的取舍：参数
量与采样复杂度从 n_actions**10（全联合动作空间）降到 O(num_heads * n_actions)，
代价是放弃块间相关性建模；由于各 block 的 PSD 观测与干扰近似独立，该近似可接受。
actor 前向只返回 raw logits、不做 softmax：训练统一用
torch.distributions.Categorical(logits=...)/log_softmax（数值更稳），确定性推理
则逐头 argmax。

二、为什么用离散 SAC 的"精确期望"而不是重参数化
-------------------------------------------------------------------------------
离散动作无法使用重参数化技巧，因此 critic 的下一状态 continuation 直接对每个头
枚举全部 n_actions 求期望：sum_a pi(a) * [min(Q1, Q2)(s', a) - alpha * log pi(a)]，
再对十头取均值得到整步的 global next value（见 SAC.calc_target）。"对十头取均值"
意味着十个头共享同一份 step 级回报信号，这与"十头共享一个温度 alpha"配套：一次
step 只有一个全局的探索-利用权衡，温度因此只需学一个标量，熵约束作用于十头的
平均熵。

三、为什么用 GroupNorm 而不是 BatchNorm
-------------------------------------------------------------------------------
GroupNorm 按 (N, G, C/G, H, W) 在单条样本内部统计，既与 batch 维无关、也没有
running statistics，因此 train 与 eval 模式输出逐位一致；BatchNorm 则在 train
模式用 batch 统计、eval 模式用滑动统计，两者不一致，且 batch 较小时统计噪声大。
本项目要求 (1) 在线前向与推理路径数值一致，(2) target critic 固定为 eval 模式后
不得再更新任何统计量；GroupNorm 一次性消除了这两个隐患（tests/test_sac.py 同时
冻结了"禁止 BatchNorm / Dropout"这一架构约束）。同理也需要确定性推理与训练走
同一条数值路径。

四、online / target 网络的模式与惰性初始化
-------------------------------------------------------------------------------
- 在线网络 actor、critic_1、critic_2 自创建起始终保持 nn.Module 默认的 train
  模式，训练全程不做 train/eval 切换：既避免模式差异带来的行为漂移，在 GroupNorm
  下两种模式输出又完全相同，因此 take_action 的推理前向与 update 的训练前向走
  同一条数值路径（update 中不会调用 net.eval()）。
- target_critic_1/2 显式 eval()，并且从不注册 optimizer；参数只由 soft_update 从
  在线 critic 拷贝，TD target 又在 torch.no_grad() 下计算，保证 target 对当前
  优化步是常量（半固定监督信号），避免"目标网络跟着自己跑"的发散。
- 惰性初始化：编码器用到 LazyConv2d/LazyLinear，参数形状要到首次 forward 才由
  PSD 尺寸确定，在此之前是 UninitializedParameter、无法拷贝。因此 target critic
  在第一次 calc_target 之前不物化，首次需要时才用一个样本走一遍前向把形状定下来
  并按在线 critic 复制（见 _ensure_target_critics_initialized），此后每个优化步
  软更新。这样既省掉启动时无意义的复制，又保证复制发生在形状就绪之后。

五、与 step-level replay 的关系
-------------------------------------------------------------------------------
ReplayBuffer 的一条经验 = 一次完整环境 step：
(state_img, hoprate, actions[num_heads], block_rewards[num_heads], step_reward,
 next_state_img, next_hoprate, done)。SAC.update 只把 block_rewards 当作逐头奖励
（TD target 形状 [B, num_heads]），step_reward = mean(block_rewards) 仅用于诊断
以及外部消费者（MBPO 奖励模型、离线 replay 校验），不参与梯度。采样结果以 numpy
返回，转 torch/device 的工作统一由 SAC._batch_images/_batch_hoprates 完成。

六、checkpoint 版本策略
-------------------------------------------------------------------------------
推理 checkpoint 只保存 actor（PolicyNet）state_dict 及 config、observation_shape、
metadata，并用 format_version 与 architecture 字符串双闸门校验。历史版本 v1 用
BatchNorm、v2 用浅层 GroupNorm、v3 用三层 CNN，它们的 state_dict 键名/形状与当前
架构都不兼容，静默加载要么报错要么给出错误推理结果，因此
load_sac_inference_checkpoint 对 v1/v2/v3 一律显式抛错并要求重新训练。当前版本号
为 SAC_CHECKPOINT_FORMAT_VERSION，架构标识为 SAC_POLICY_ARCHITECTURE。

顶层常量
-------------------------------------------------------------------------------
HOPRATE_INPUT_SCALE = 10.0      hoprate 归一化后的幅值上限（区间 [-10, 10]）
PSD_FEATURE_DIM = 512           PSD 卷积分支输出维度
HOPRATE_FEATURE_DIM = 64        hoprate 分支输出维度
STATE_FEATURE_DIM = 256         融合后的共享状态特征维度（各 actor/Q 头输入）
SAC_CHECKPOINT_FORMAT_VERSION / SAC_POLICY_ARCHITECTURE：推理 checkpoint 版本闸门
"""

import collections
import os
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.parameter import UninitializedParameter


HOPRATE_INPUT_SCALE = 10.0
PSD_FEATURE_DIM = 512
HOPRATE_FEATURE_DIM = 64
STATE_FEATURE_DIM = 256
SAC_CHECKPOINT_FORMAT_VERSION = 4
SAC_POLICY_ARCHITECTURE = "cnn2_groupnorm_hop_mlp1_fusion_mlp1_v4"


def normalize_hoprate(hoprate, hoprate_min, hoprate_max):
    """把 hoprate（Hz）线性映射并缩放到固定区间 [-10, 10]。

    先做 min-max 归一化到 [-1, 1]，再乘 HOPRATE_INPUT_SCALE(=10) 放大到 [-10, 10]。
    这样做的原因：hoprate 的量纲是 Hz（典型合法区间 [10, 1000]），而 PSD 图像特征
    经 GroupNorm 后幅值接近 O(1)，两者直接拼接会让 hoprate 分支主导融合层的输入
    尺度；固定常量缩放让 hoprate 与图像特征处于可比量级，同时把归一化写成环境
    上下界的纯函数（而不是可学习的标准化层），从而 actor、critic、独立的推理
    checkpoint 三处行为完全一致，推理端也无需携带额外统计量。

    Args:
        hoprate: hoprate 张量，形状 [..., 1]；一维 [B] 输入会被自动升维为 [B, 1]。
        hoprate_min: 环境允许的 hoprate 下界（Hz），必须严格小于 hoprate_max。
        hoprate_max: 环境允许的 hoprate 上界（Hz）。

    Returns:
        与输入同形状的浮点张量，取值范围 [-10, 10]（映射关系：hoprate_min -> -10，
        hoprate_max -> +10，外推值被 clamp 到边界）。

    Raises:
        ValueError: 最后一维不是 1，或 hoprate_max <= hoprate_min（区间非法，
            此时除法会退化或变号）。
    """
    if hoprate.ndim == 1:
        hoprate = hoprate.unsqueeze(-1)
    if hoprate.shape[-1] != 1:
        raise ValueError(
            f"Expected hoprate features with size 1, got {hoprate.shape[-1]}."
        )
    if hoprate_max <= hoprate_min:
        raise ValueError("hoprate_max must be greater than hoprate_min.")

    # 仿射到 [-1, 1]：转 float 是为了避免 numpy 标量/张量 dtype 混用时的类型歧义。
    normalized = 2.0 * (
        (hoprate - float(hoprate_min)) / float(hoprate_max - hoprate_min)
    ) - 1.0
    # clamp 兜住越界输入（例如评测时传入训练区间外的 hoprate），保证进入网络前
    # 的数值范围恒定，不会因为极端跳速撑爆 MLP 输出。
    return normalized.clamp(-1.0, 1.0) * HOPRATE_INPUT_SCALE


class ReplayBuffer:
    """step-level 经验回放池：一条经验 = 一次完整环境 step。

    与"每个 block 一条经验"的 flat 方案不同，这里以 step 为最小单位存储，因为一次
    step 的十个 offset 决策是同时产生、由同一个 step reward 共同评价的，只有
    step-level 记录才能保留"同一状态下十头动作联合采样"这一事实（也是 MBPO 奖励
    模型/离线 replay 的输入格式）。

    存储结构为 collections.deque(maxlen=capacity)：到达容量后自动 FIFO 丢弃最旧
    经验，因此不需要额外的读写指针，也没有环形缓冲区的索引计算。每条记录是一个
    定序元组：
        (state_img, hoprate, actions, block_rewards, step_reward,
         next_state_img, next_hoprate, done)
    该顺序是对外契约（r_predict_model/mbpo_adapter.py 按下标取字段），不得调整。
    其中状态/动作/reward 均以 .copy() 存入，避免与环境内部每步复用的数组发生别名。

    校验规则（在 add/add_batch 内集中执行，尽量早地暴露上游数据错误，而不是等到
    训练时出现 NaN）：
        - 图像必须是 [H, W] 或 [C, H, W]（batch 版为 [B, H, W]/[B, C, H, W]）、
          非空且全为有限值；首次写入推得 observation_shape，之后所有写入必须与之
          一致（防止环境切换 PSD 尺寸后新旧数据混池）。
        - actions 必须是整数值（np.rint 后可精确还原）、非负；若构造时提供了
          n_actions，还要求 < n_actions。
        - block_rewards 形状必须是 (num_heads,)，全为有限值。
        - hoprate / done 会被转成 python 标量（float/bool），保证 get_all 转数组
          时 dtype 稳定。

    Args:
        capacity: 最大存放的 step 数，必须为正（<= 0 抛 ValueError）。
        num_heads: 每步的 offset 决策数，须与环境的 block 数一致（默认 10）。
        n_actions: 每个头的离散取值数（可选）；给定时对动作做上界校验。
    """

    def __init__(self, capacity, num_heads=10, n_actions=None):
        """初始化空缓冲区；参数非法时立即抛 ValueError。

        Args:
            capacity: 最大 step 数（正整数，内部转 int）。
            num_heads: 每步 offset 决策数，须 > 0。
            n_actions: 每个头的动作数上界，可选；给定时须 > 0。
        """
        if capacity <= 0:
            raise ValueError("Replay capacity must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if n_actions is not None and n_actions <= 0:
            raise ValueError("n_actions must be positive when provided.")

        self.capacity = int(capacity)
        self.num_heads = int(num_heads)
        self.n_actions = int(n_actions) if n_actions is not None else None
        # 首次写入时由数据推得；同时也是后续写入的形状一致性基准。
        self.observation_shape = None
        # deque(maxlen=...) 自带 FIFO 淘汰语义：超出容量时自动丢弃最旧记录。
        self.buffer = collections.deque(maxlen=self.capacity)

    def _validate_actions(self, actions):
        """校验并规范化单步动作向量。

        允许浮点输入但要求逐元素为整数值（np.rint 后 allclose 通过），以便直接
        接收环境/离线文件里以 float32 存储的 offset；返回 int64 数组。

        Args:
            actions: 形状 (num_heads,) 的标量序列。

        Returns:
            np.ndarray，dtype=int64，形状 (num_heads,)。

        Raises:
            ValueError: 形状不符、含非有限值、含非整数值、有负值，或（配置了
                n_actions 时）超出 [0, n_actions - 1]。
        """
        raw_actions = np.asarray(actions)
        if raw_actions.shape != (self.num_heads,):
            raise ValueError(
                f"actions must have shape ({self.num_heads},), got {raw_actions.shape}."
            )
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("actions contains non-finite values.")
        rounded = np.rint(raw_actions)
        if not np.allclose(raw_actions, rounded):
            raise ValueError("actions must contain integer-valued offsets.")

        actions_array = rounded.astype(np.int64)
        if np.any(actions_array < 0):
            raise ValueError("actions must be non-negative.")
        if self.n_actions is not None and np.any(actions_array >= self.n_actions):
            raise ValueError(
                f"actions must be in [0, {self.n_actions - 1}]."
            )
        return actions_array

    def _validate_block_rewards(self, block_rewards):
        """校验单步逐 block reward 向量，返回 float32 副本。

        Args:
            block_rewards: 形状 (num_heads,) 的标量序列。

        Returns:
            np.ndarray，dtype=float32，形状 (num_heads,)。

        Raises:
            ValueError: 形状不符或含非有限值。
        """
        rewards_array = np.asarray(block_rewards, dtype=np.float32)
        if rewards_array.shape != (self.num_heads,):
            raise ValueError(
                "block_rewards must have shape "
                f"({self.num_heads},), got {rewards_array.shape}."
            )
        if not np.all(np.isfinite(rewards_array)):
            raise ValueError("block_rewards contains non-finite values.")
        return rewards_array

    def add(
        self,
        state_img,
        hoprate,
        actions,
        block_rewards,
        next_state_img,
        next_hoprate,
        done,
    ):
        """追加一条 step-level 经验（单样本入口，训练主循环每步调用一次）。

        写入前完成全部校验，并按 observation_shape 做跨样本形状一致性检查；数据
        以 .copy() 深拷贝存入，防止外部在本步之后复用/覆写同一块内存。

        Args:
            state_img: 当前 PSD 图像，[H, W] 或 [C, H, W]，float32，有限值。
            hoprate: 当前跳速（Hz）；标量，会被 float() 转换并检查有限性。
            actions: 十个 offset，形状 (num_heads,)；允许整数值浮点，见
                _validate_actions。
            block_rewards: 逐 block reward，形状 (num_heads,)。
            next_state_img: 下一 PSD 图像，形状须与 state_img 完全一致。
            next_hoprate: 下一跳速（Hz），标量。
            done: 该 step 后回合是否终止（写入前转 bool）。

        Returns:
            None。

        Raises:
            ValueError: 图像形状不一致/为空/含非有限值、hoprate 非有限、或
                actions / block_rewards 未通过校验。

        副作用：
            - 首次调用时设定 self.observation_shape；
            - 超出 capacity 时由 deque 自动淘汰最旧经验（FIFO）。
        """
        state_array = np.asarray(state_img, dtype=np.float32)
        next_state_array = np.asarray(next_state_img, dtype=np.float32)
        if state_array.shape != next_state_array.shape:
            raise ValueError(
                "state_img and next_state_img must have the same shape, got "
                f"{state_array.shape} and {next_state_array.shape}."
            )
        if state_array.size == 0:
            raise ValueError("state images cannot be empty.")
        if not np.all(np.isfinite(state_array)) or not np.all(
            np.isfinite(next_state_array)
        ):
            raise ValueError("state images must contain only finite values.")
        if self.observation_shape is None:
            # 首次写入即锁定观测形状；后续与之不符一律报错。
            self.observation_shape = state_array.shape
        elif state_array.shape != self.observation_shape:
            raise ValueError(
                "state image shape differs from existing replay data: "
                f"expected {self.observation_shape}, got {state_array.shape}."
            )

        # hoprate 以 python float 存储，避免 numpy 标量在拼接 batch 时被广播成
        # 意外形状。
        hoprate = float(hoprate)
        next_hoprate = float(next_hoprate)
        if not np.isfinite(hoprate) or not np.isfinite(next_hoprate):
            raise ValueError("hoprates must be finite.")

        actions_array = self._validate_actions(actions)
        rewards_array = self._validate_block_rewards(block_rewards)
        # step reward 是十头 reward 的均值，只用于诊断与外部（MBPO/离线 replay）
        # 消费；SAC.update 使用 block_rewards 做逐头 TD target，不受此字段影响。
        step_reward = float(np.mean(rewards_array))

        self.buffer.append(
            (
                state_array.copy(),
                hoprate,
                actions_array.copy(),
                rewards_array.copy(),
                step_reward,
                next_state_array.copy(),
                next_hoprate,
                bool(done),
            )
        )

    def add_batch(
        self,
        state_imgs,
        hoprates,
        actions,
        block_rewards,
        next_state_imgs,
        next_hoprates,
        dones,
    ):
        """按批追加经验：校验做一次，然后逐条展开写入。

        主要用于离线 replay 载入与 MBPO 模型 rollout 批量灌入（逐条调用 add 会
        重复做同样的形状/有限性检查，开销明显）。校验规则与 add 完全一致，唯一
        差别是这里的输入多一维 batch，且 observation_shape 取自 [1:]（去掉 batch
        维），因此与 add 写入的数据可以混在同一个池子里。

        Args:
            state_imgs: [B, H, W] 或 [B, C, H, W] 的 PSD 图像批，全为有限值。
            hoprates: 每步一个 hoprate，形状可广播为 (B,)（内部 reshape(-1)）。
            actions: (B, num_heads) 的 offset，整数值。
            block_rewards: (B, num_heads) 的逐 block reward。
            next_state_imgs: 与 state_imgs 同形状的下一状态批。
            next_hoprates: (B,) 的下一跳速。
            dones: (B,) 的终止标志。

        Returns:
            None。

        Raises:
            ValueError: 图像维度/形状不符或含非有限值、hoprates/dones 数量与 batch
                不符、actions 或 block_rewards 形状/取值范围非法。

        说明：
            B == 0 时直接返回（无操作），便于上游传空 batch 而不必特判。
        """
        state_array = np.asarray(state_imgs, dtype=np.float32)
        next_state_array = np.asarray(next_state_imgs, dtype=np.float32)
        if state_array.ndim not in (3, 4) or next_state_array.ndim != state_array.ndim:
            raise ValueError(
                "state images must have shape [B,H,W] or [B,C,H,W]."
            )
        if state_array.shape != next_state_array.shape:
            raise ValueError(
                "state_imgs and next_state_imgs must have the same shape, got "
                f"{state_array.shape} and {next_state_array.shape}."
            )
        batch_size = state_array.shape[0]
        if batch_size == 0:
            return  # 空 batch：静默无操作，避免上游为零样本场景写特判
        if state_array.size == 0 or not np.all(np.isfinite(state_array)) or not np.all(
            np.isfinite(next_state_array)
        ):
            raise ValueError("state images must contain only finite values.")

        if self.observation_shape is None:
            # 单样本视角的形状（去掉 batch 维），保证与 add 推得的形状同一语义。
            self.observation_shape = state_array.shape[1:]
        elif tuple(state_array.shape[1:]) != tuple(self.observation_shape):
            raise ValueError(
                "state image shape differs from existing replay data: "
                f"expected {self.observation_shape}, got {state_array.shape[1:]}."
            )

        # reshape(-1) 容许上游传 (B,)、(B,1) 或标量数组等多种布局。
        hoprate_array = np.asarray(hoprates, dtype=np.float32).reshape(-1)
        next_hoprate_array = np.asarray(next_hoprates, dtype=np.float32).reshape(-1)
        if hoprate_array.shape != (batch_size,) or next_hoprate_array.shape != (
            batch_size,
        ):
            raise ValueError("hoprates must contain one value per transition.")
        if not np.all(np.isfinite(hoprate_array)) or not np.all(
            np.isfinite(next_hoprate_array)
        ):
            raise ValueError("hoprates must be finite.")

        raw_actions = np.asarray(actions)
        if raw_actions.shape != (batch_size, self.num_heads):
            raise ValueError(
                "actions must have shape "
                f"({batch_size}, {self.num_heads}), got {raw_actions.shape}."
            )
        if not np.all(np.isfinite(raw_actions)):
            raise ValueError("actions contains non-finite values.")
        rounded_actions = np.rint(raw_actions)
        if not np.allclose(raw_actions, rounded_actions):
            raise ValueError("actions must contain integer-valued offsets.")
        actions_array = rounded_actions.astype(np.int64)
        if np.any(actions_array < 0):
            raise ValueError("actions must be non-negative.")
        if self.n_actions is not None and np.any(actions_array >= self.n_actions):
            raise ValueError(
                f"actions must be in [0, {self.n_actions - 1}]."
            )

        rewards_array = np.asarray(block_rewards, dtype=np.float32)
        if rewards_array.shape != (batch_size, self.num_heads):
            raise ValueError(
                "block_rewards must have shape "
                f"({batch_size}, {self.num_heads}), got {rewards_array.shape}."
            )
        if not np.all(np.isfinite(rewards_array)):
            raise ValueError("block_rewards contains non-finite values.")

        dones_array = np.asarray(dones).reshape(-1)
        if dones_array.shape != (batch_size,):
            raise ValueError("dones must contain one value per transition.")

        # 逐条求均值：与 add 的 step_reward 定义保持一致（而非整批求一个均值）。
        step_rewards = np.mean(rewards_array, axis=1)
        for index in range(batch_size):
            self.buffer.append(
                (
                    state_array[index].copy(),
                    float(hoprate_array[index]),
                    actions_array[index].copy(),
                    rewards_array[index].copy(),
                    float(step_rewards[index]),
                    next_state_array[index].copy(),
                    float(next_hoprate_array[index]),
                    bool(dones_array[index]),
                )
            )

    @staticmethod
    def _batch_from_transitions(transitions):
        """把定序元组序列转成按字段命名的 numpy batch 字典。

        用 zip(*transitions) 按列拆包，再统一转成训练需要的 dtype：图像/奖励/
        hoprate 用 float32，动作用 int64，done 用 float32（便于直接与 gamma 相乘
        做 (1 - done) 掩码）。

        Args:
            transitions: 可迭代的元组集合，每个元组形如 add() 中 append 的记录，
                长度固定为 8、字段顺序一致。

        Returns:
            dict，键为 state_imgs / hoprates / actions / block_rewards /
            step_rewards / next_state_imgs / next_hoprates / dones，值为 batch
            维在前的 numpy 数组；形状分别为
            [B, *obs] / [B] / [B, num_heads] / [B, num_heads] / [B] /
            [B, *obs] / [B] / [B]。其中 step_rewards 仅供诊断，SAC.update 不读取。
        """
        (
            state_imgs,
            hoprates,
            actions,
            block_rewards,
            step_rewards,
            next_state_imgs,
            next_hoprates,
            dones,
        ) = zip(*transitions)
        return {
            "state_imgs": np.asarray(state_imgs, dtype=np.float32),
            "hoprates": np.asarray(hoprates, dtype=np.float32),
            "actions": np.asarray(actions, dtype=np.int64),
            "block_rewards": np.asarray(block_rewards, dtype=np.float32),
            "step_rewards": np.asarray(step_rewards, dtype=np.float32),
            "next_state_imgs": np.asarray(next_state_imgs, dtype=np.float32),
            "next_hoprates": np.asarray(next_hoprates, dtype=np.float32),
            "dones": np.asarray(dones, dtype=np.float32),
        }

    def sample(self, batch_size):
        """无放回随机抽取 batch_size 条 step 经验。

        用 random.sample 而非 numpy 索引：deque 不支持随机下标访问，random.sample
        对序列按下标取值，正好绕开 O(n) 的逐元素遍历。无放回保证同一 batch 内不
        出现重复样本（离散小容量池下可避免过度加权同一条经验）。

        Args:
            batch_size: 抽取条数，必须 > 0 且 <= size()。

        Returns:
            dict，字段与形状同 _batch_from_transitions。

        Raises:
            ValueError: batch_size <= 0，或 batch_size 超过当前经验数。
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > self.size():
            raise ValueError(
                f"Cannot sample {batch_size} transitions from a buffer of size {self.size()}."
            )
        return self._batch_from_transitions(random.sample(self.buffer, batch_size))

    def get_all(self):
        """按插入顺序返回全池快照（不打乱），用于离线/模型 rollout 训练。

        典型用途：MBPO 的 model buffer 快照、离线 replay 导出与单元测试逐条比对。
        与 sample 不同，这里保留时间顺序，便于复现与排查数据问题。

        Returns:
            dict，字段与形状同 _batch_from_transitions，batch 维 = size()。

        Raises:
            ValueError: 缓冲区为空（没有可复现的字段形状，zip(*) 也无法拆包）。
        """
        if not self.buffer:
            raise ValueError("Cannot read transitions from an empty replay buffer.")
        return self._batch_from_transitions(list(self.buffer))

    def clear(self):
        """清空全部经验，并重置 observation_shape。

        重置形状推断让同一个 buffer 实例可以换用不同 PSD 尺寸的环境（例如先跑
        16x16 的快速测试、再切到生产尺寸），而不必重新构造对象。
        """
        self.buffer.clear()
        self.observation_shape = None

    def size(self):
        """返回当前存放的 step 数（<= capacity）。"""
        return len(self.buffer)


def _init_weights(module):
    """统一的权重初始化回调，供 nn.Module.apply() 递归应用到每个子模块。

    规则：weight ~ N(0, 0.1^2)，bias 置零。0.1 的小标准差让网络初始输出幅值接近
    0：PSD 图像经过归一化层后幅值在 O(1)，若用 PyTorch 默认的 Kaiming 初始化，
    两层卷积 + 池化后的 flatten 输入（生产 PSD 尺寸下高达 80000 维）会让 FC 层
    输出爆炸，进而使 Q 值量级过大、TD target 初期剧烈震荡。

    惰性模块保护：LazyConv2d/LazyLinear 在首次 forward 之前 weight/bias 是
    UninitializedParameter，对它们调用 nn.init.* 会抛异常，因此这里显式跳过；
    这类参数在惰性初始化完成时使用 PyTorch 默认初始化（apply 只在构造期遍历一次，
    无法覆盖之后的物化过程）。动作头/融合层等非惰性层则不受影响。

    Args:
        module: 被 apply 遍历到的子模块；只有 Linear/Conv2d/LazyLinear/LazyConv2d
            会被处理，其余模块（GroupNorm、ReLU、MaxPool2d、Flatten 等）原样跳过。

    Returns:
        None（原地修改）。
    """
    if isinstance(module, (nn.Linear, nn.Conv2d, nn.LazyLinear, nn.LazyConv2d)):
        # 尚未物化的真实形状未知，跳过以免触发异常。
        if isinstance(getattr(module, "weight", None), UninitializedParameter):
            return
        if getattr(module, "weight", None) is not None:
            nn.init.normal_(module.weight, 0, 0.1)
        if (
            getattr(module, "bias", None) is not None
            and not isinstance(module.bias, UninitializedParameter)
        ):
            nn.init.zeros_(module.bias)


class StateEncoder(nn.Module):
    """把 PSD 图像与标量 hoprate 编码成同一个共享状态特征向量。

    actor 与两个 critic 各自持有独立的 StateEncoder（参数不共享），但结构完全相同。

    结构（输入 img [B, 1, H, W] 或 [B, C, H, W]，hoprate [B, 1]）：
        conv1 = LazyConv2d(out=16, k=3, s=1, p=1)          -> [B, 16, H, W]
        norm1 = GroupNorm(4, 16) + ReLU
        conv2 = Conv2d(16 -> 32, k=3, s=1, p=1)            -> [B, 32, H, W]
        norm2 = GroupNorm(8, 32) + ReLU
        pool  = MaxPool2d(2)                               -> [B, 32, H/2, W/2]
        flatten -> LazyLinear(PSD_FEATURE_DIM=512) + ReLU  -> [B, 512]
        hoprate_embedding = Linear(1 -> HOPRATE_FEATURE_DIM=64) + ReLU -> [B, 64]
        fusion = cat([512, 64], dim=1) -> Linear(576 -> STATE_FEATURE_DIM=256) + ReLU
                 -> [B, 256]

    各层作用与设计取舍：
    - 两层 3x3 卷积都用 padding=1，保持空间尺寸不变，只在 conv2 之后做一次 2x2
      池化（空间减半）。选择"两层 16/32 通道"的轻量 CNN 而非更深的骨干：PSD 图像
      语义简单（能量分布/干扰形态），深网络带来的收益有限，却会直接抬高每个环境
      step 的决策延迟——离散十头策略要求推理足够快才能跟上跳频节奏。
    - 首层与 flatten 之后的线性层都是 Lazy 版本，自动适配输入尺寸：测试用 8/16，
      生产用 100（见 tests/test_sac.py 对两种尺寸的断言），避免把 H/W 写死。代价是
      首次 forward 之前参数未物化（UninitializedParameter），因此加载 state_dict
      前必须先做一次 dry-run 前向（checkpoint 函数与 target critic 初始化都据此）。
    - 归一化层用 GroupNorm（4 组对 16 通道、8 组对 32 通道，即每组 4 个通道），
      在单条样本内部统计，与 batch 维无关、无 running statistics：train/eval 输出
      逐位一致，target critic 固定 eval 后也不会再有统计量被隐式更新。
    - hoprate 不用裸标量直接拼，而是先用一个小 MLP（1 -> 64）学出非线性嵌入：
      跳速与可对抗的干扰样式强相关，先升维再融合能表达这种非线性关系。
    - 融合层（576 -> 256）是唯一的图像-跳速交互点，输出 256 维即 STATE_FEATURE_DIM，
      之后被 10 个 actor/Q 头共享。

    参数量：与 PSD 面积近似成正比的主要项是 flatten 后的 LazyLinear（生产尺寸
    100x100 时为 512 * 32 * 50 * 50 ≈ 4.10e7，占编码器 4.11e7 总参数的绝对多数），
    两个卷积层合计仅约 4.8e3 参数（外加 GroupNorm 的 96 个可学习参数）。这解释了
    训练侧为什么要用大 batch 让卷积/FC 满载。

    Args:
        hoprate_min: hoprate 归一化下界（Hz），转 float 保存。
        hoprate_max: hoprate 归一化上界（Hz），转 float 保存。
    """

    def __init__(self, hoprate_min, hoprate_max):
        """按上述结构搭建编码器，并对非惰性层应用 _init_weights。

        Args:
            hoprate_min: hoprate 归一化下界（Hz）。
            hoprate_max: hoprate 归一化上界（Hz）。
        """
        super().__init__()
        # 保存为普通 float 属性（非 buffer）：随 checkpoint 的 config 一起显式传递，
        # 不进入 state_dict，避免推理端额外依赖。
        self.hoprate_min = float(hoprate_min)
        self.hoprate_max = float(hoprate_max)

        # Lazy 卷积：in_channels 由首次 forward 推得，兼容 [B,1,H,W] 与多通道 PSD。
        self.conv1 = nn.LazyConv2d(
            out_channels=16, kernel_size=3, stride=1, padding=1
        )
        self.norm1 = nn.GroupNorm(4, 16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(8, 32)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.conv_fc = nn.LazyLinear(PSD_FEATURE_DIM)
        self.hoprate_embedding = nn.Sequential(
            nn.Linear(1, HOPRATE_FEATURE_DIM),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(PSD_FEATURE_DIM + HOPRATE_FEATURE_DIM, STATE_FEATURE_DIM),
            nn.ReLU(),
        )
        # apply 递归初始化：Lazy 层此时未物化会被 _init_weights 跳过，
        # hoprate_embedding / fusion 的 Linear 则按 N(0, 0.1^2) 初始化。
        self.apply(_init_weights)

    def forward(self, img, hoprate):
        """编码单批观测。

        Args:
            img: PSD 图像张量，[B, 1, H, W] 或 [B, C, H, W]（float32）；H/W 可以是
                任意尺寸（测试 8/16、生产 100），Lazy 层首次调用后即固定。
            hoprate: [B, 1] 的跳速（Hz），内部会先经 normalize_hoprate 归一到 [-10, 10]。

        Returns:
            [B, STATE_FEATURE_DIM] 的共享状态特征（float32），供各 actor/Q 头使用。
        """
        # 归一化在此处完成，使所有网络分支（actor / critic / target critic）以及
        # 推理 checkpoint 使用完全相同的 hoprate 预处理，不存在两套缩放。
        hoprate = normalize_hoprate(
            hoprate, self.hoprate_min, self.hoprate_max
        )
        # conv -> GroupNorm -> ReLU：GroupNorm 放在卷积后、激活前（标准 pre-activation
        # 顺序），逐样本归一化保证 train/eval 数值一致。
        image_features = F.relu(self.norm1(self.conv1(img)))
        image_features = F.relu(self.norm2(self.conv2(image_features)))
        # 单次 2x2 池化：把空间尺寸减半，降低后续 FC 的输入维度与参数量。
        image_features = self.pool(image_features)
        image_features = self.flatten(image_features)
        image_features = F.relu(self.conv_fc(image_features))
        hoprate_features = self.hoprate_embedding(hoprate)
        # 在通道维拼接 [B, 576] 后融合到 256 维；这是图像与跳速唯一的交互位置。
        return self.fusion(torch.cat([image_features, hoprate_features], dim=1))


class PolicyNet(nn.Module):
    """十头因子化 actor：共享编码器 + num_heads 个互不共享参数的动作头。

    输出 raw logits（不做 softmax/采样），形状 [B, num_heads, n_actions]：
        - 训练：交给 torch.distributions.Categorical(logits=...)/log_softmax，
          由框架内部做数值稳定的归一化；
        - 确定性推理：逐头 argmax（take_actions 中实现）。
    返回 logits 而非概率/动作，使"训练用随机采样、评测用 argmax"共用一次前向，
    也便于离散 SAC 的期望计算（sum_a pi(a) * (...)）一次拿到全部动作的 Q 输入。

    十头之间只共享 256 维状态特征、不共享头参数：各 block 的干扰环境不同，独立
    头允许每个 block 学到不同的 offset 偏好（tests 断言各 head.weight 不是同一对象）。

    参数量 ≈ StateEncoder 的参数量 + num_heads * (256 * n_actions + n_actions)；
    头参数在十头中占比很小，主要开销仍在编码器。

    Args:
        n_actions: 每个头的离散动作数（合法 offset 数）。
        num_heads: 头数，须与环境 block 数一致（默认 10）。
        hoprate_min: hoprate 归一化下界（Hz）。
        hoprate_max: hoprate 归一化上界（Hz）。
    """

    def __init__(self, n_actions, num_heads=10, hoprate_min=10.0, hoprate_max=1000.0):
        """搭建编码器与动作头，并对头做小尺度初始化。

        Args:
            n_actions: 每个头的动作数。
            num_heads: 头数（默认 10）。
            hoprate_min: hoprate 归一化下界（Hz，默认 10.0）。
            hoprate_max: hoprate 归一化上界（Hz，默认 1000.0）。
        """
        super().__init__()
        self.n_actions = int(n_actions)
        self.num_heads = int(num_heads)
        self.encoder = StateEncoder(hoprate_min, hoprate_max)
        # ModuleList 而非普通 list：保证头参数被注册进 state_dict 并随 .to(device)
        # 一起迁移；每个头是独立的 Linear，参数不绑定。
        self.action_heads = nn.ModuleList(
            nn.Linear(STATE_FEATURE_DIM, self.n_actions)
            for _ in range(self.num_heads)
        )
        for head in self.action_heads:
            # 头权重刻意用极小的均匀分布（±0.003）覆盖 _init_weights 的 N(0, 0.1)：
            # 初始 logits 接近 0 => 初始策略近似均匀分布，避免训练早期策略过早
            # 收敛到少数 offset 而失去探索；bias 置零保持十头对称起点。
            nn.init.uniform_(head.weight, -0.003, 0.003)
            nn.init.zeros_(head.bias)

    def forward(self, img, hoprate):
        """前向计算十头 raw logits。

        Args:
            img: PSD 图像，[B, 1, H, W] 或 [B, C, H, W]。
            hoprate: [B, 1] 跳速（Hz）。

        Returns:
            [B, num_heads, n_actions] 的 logits（未归一化）。dim=1 为头（block）维，
            dim=-1 为动作维；数值上未做 softmax，调用方自行 log_softmax/采样。
        """
        features = self.encoder(img, hoprate)
        # stack 到 dim=1，把 [B, n_actions] 的十份输出拼成 [B, num_heads, n_actions]，
        # 与 critic 输出、replay 中的动作/奖励布局完全对齐。
        return torch.stack([head(features) for head in self.action_heads], dim=1)


class ValueNet(nn.Module):
    """十头 Q 网络（critic）：共享编码器 + num_heads 个互不共享参数的 Q 头。

    输出 [B, num_heads, n_actions] 的 Q(s, a) 估计，与 actor 的 logits 逐头逐动作
    对齐；训练时用 gather 取出实际执行动作的 Q 值，计算 TD target 时对动作维做
    离散期望。网络中不使用 BatchNorm/Dropout，任何模式下前向都是确定性映射，
    因此 target critic 可以安全地长期停在 eval 模式。

    SAC 用两个结构相同的 ValueNet（critic_1/critic_2）做 clipped double-Q：取
    二者最小值抑制 Q 值高估，target 侧同样取 min 后再做期望。

    Args:
        n_actions: 每个头的动作数。
        num_heads: 头数（默认 10）。
        hoprate_min: hoprate 归一化下界（Hz）。
        hoprate_max: hoprate 归一化上界（Hz）。
    """

    def __init__(self, n_actions, num_heads=10, hoprate_min=10.0, hoprate_max=1000.0):
        """搭建编码器与 Q 头，并用 _init_weights（N(0, 0.1^2)、bias=0）初始化头部。

        Args:
            n_actions: 每个头的动作数。
            num_heads: 头数（默认 10）。
            hoprate_min: hoprate 归一化下界（Hz，默认 10.0）。
            hoprate_max: hoprate 归一化上界（Hz，默认 1000.0）。
        """
        super().__init__()
        self.n_actions = int(n_actions)
        self.num_heads = int(num_heads)
        self.encoder = StateEncoder(hoprate_min, hoprate_max)
        self.q_heads = nn.ModuleList(
            nn.Linear(STATE_FEATURE_DIM, self.n_actions)
            for _ in range(self.num_heads)
        )
        # Q 头不做 actor 那样的极小尺度初始化：critic 需要足够的初始幅值来区分
        # 不同动作，统一用 N(0, 0.1^2) 即可。
        self.q_heads.apply(_init_weights)

    def forward(self, img, hoprate):
        """前向计算十头 Q 值。

        Args:
            img: PSD 图像，[B, 1, H, W] 或 [B, C, H, W]。
            hoprate: [B, 1] 跳速（Hz）。

        Returns:
            [B, num_heads, n_actions] 的 Q 值张量；dim=1 为头维、dim=-1 为动作维，
            与 PolicyNet 的 logits 布局一致，可直接做期望/argmax/gather。
        """
        features = self.encoder(img, hoprate)
        # 与 actor 相同的堆叠方式，保证 actor logits 与 Q 在同一 (head, action) 网格上对位。
        return torch.stack([head(features) for head in self.q_heads], dim=1)


def _checkpoint_images(observation_shape, device):
    """构造一个形状正确、全零的 PSD 占位图像，用于触发 Lazy 层的形状推断。

    保存时用固定 hoprate 中点 + 该占位图跑一次 dry-run 前向，把 LazyConv2d/
    LazyLinear 的 UninitializedParameter 物化成真实形状；否则 state_dict 里只有
    "未初始化" 条目，保存下来无法被推理端 load_state_dict 还原，加载时也无法在
    load 前确定网络形状。用全零输入（而不是随机噪声）可以让该 dry-run 不依赖随机
    性、结果确定，便于复现。

    Args:
        observation_shape: 观测形状；长度 2 表示 (H, W)（单通道，补成 [1,1,H,W]），
            长度 3 表示 (C, H, W)（补成 [1,C,H,W]）。
        device: 生成张量所在设备，须与网络一致（Lazy 层会按首个输入的 device 物化）。

    Returns:
        [1, 1, H, W] 或 [1, C, H, W] 的 float32 全零张量，batch=1。

    Raises:
        ValueError: observation_shape 的轴数不是 2 或 3（无法区分单/多通道布局）。
    """
    shape = tuple(int(value) for value in observation_shape)
    if len(shape) == 2:
        return torch.zeros((1, 1, *shape), dtype=torch.float32, device=device)
    if len(shape) == 3:
        return torch.zeros((1, *shape), dtype=torch.float32, device=device)
    raise ValueError("observation_shape must have two or three axes.")


def save_sac_inference_checkpoint(
    agent,
    path,
    observation_shape,
    hoprate_min,
    hoprate_max,
    metadata=None,
):
    """保存 SAC 的 actor（纯策略部分）为可独立部署的推理 checkpoint。

    只保存 actor 而不保存 critics/optimizer：推理只需要确定性策略前向，critic 的
    Q 头、target 网络与优化器状态既无用又会让文件体积翻数倍。checkpoint 中记录
    足以完整重建 PolicyNet 的信息：
        format_version      版本闸门（当前 SAC_CHECKPOINT_FORMAT_VERSION）
        model_type          固定字符串，标识这是多头 SAC 策略
        architecture        架构标识（SAC_POLICY_ARCHITECTURE），防止结构变更后误加载
        config              n_actions / num_heads / hoprate_min / hoprate_max
        observation_shape   训练时的观测形状，供推理端 dry-run 前向确定 Lazy 层形状
        actor_state_dict    actor 的权重
        metadata            调用方附加信息（训练轮数、超参等），可为空

    保存前先在 agent.device 上用 observation_shape 对应的全零输入做一次前向：
    这会物化 LazyConv2d/LazyLinear 的参数，保证 actor_state_dict 里是真实形状的
    权重（否则存下来的是未初始化参数，加载端无法使用）。整体在 torch.no_grad()
    下进行，不建图、不产生任何副作用到网络状态。

    Args:
        agent: SAC 实例；只读取 agent.actor / agent.device / agent.n_actions /
            agent.num_heads。
        path: 输出文件路径；其父目录会被自动创建。
        observation_shape: 训练时 PSD 观测形状，(H, W) 或 (C, H, W)。
        hoprate_min: hoprate 归一化下界（Hz），写入 config 供推理端重建编码器。
        hoprate_max: hoprate 归一化上界（Hz）。
        metadata: 可选 dict，原样存入；None 时写空 dict。

    Returns:
        None。

    Raises:
        与 _checkpoint_images 相同（observation_shape 轴数非法）以及 torch.save
        自身的 IO 异常。
    """
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # 用区间中点作为 dry-run 的 hoprate：任何合法 hoprate 都能触发形状推断，选中点
    # 是为了让这一步不依赖具体训练配置。
    midpoint_hoprate = (float(hoprate_min) + float(hoprate_max)) / 2.0
    with torch.no_grad():
        # 纯形状物化前向，结果丢弃；不加 no_grad 会残留计算图并污染后续反传。
        agent.actor(
            _checkpoint_images(observation_shape, agent.device),
            torch.full(
                (1, 1),
                midpoint_hoprate,
                dtype=torch.float32,
                device=agent.device,
            ),
        )

    torch.save(
        {
            "format_version": SAC_CHECKPOINT_FORMAT_VERSION,
            "model_type": "MultiHeadSACPolicy",
            "architecture": SAC_POLICY_ARCHITECTURE,
            "config": {
                # 只存重建 PolicyNet 所需的构造参数；hoprate 归一化区间必须一起存，
                # 否则推理端用默认区间会让 hoprate 分支的输入尺度错位。
                "n_actions": agent.n_actions,
                "num_heads": agent.num_heads,
                "hoprate_min": float(hoprate_min),
                "hoprate_max": float(hoprate_max),
            },
            "observation_shape": list(observation_shape),
            "actor_state_dict": agent.actor.state_dict(),
            "metadata": dict(metadata or {}),
        },
        path,
    )


def load_sac_inference_checkpoint(
    path,
    device="cuda",
    expected_num_heads=None,
    expected_n_actions=None,
    expected_observation_shape=None,
):
    """加载推理 checkpoint，校验版本/架构/维度后返回可用的确定性策略。

    版本策略：v1（BatchNorm 策略）、v2（浅层 GroupNorm）、v3（三层 CNN）都被显式
    拒绝。原因不是"不兼容格式"，而是它们的 state_dict 键名与张量形状和当前架构
    (SAC_POLICY_ARCHITECTURE) 不同，静默 load 要么直接报错、要么（在部分键匹配时）
    只加载一部分权重并留下随机初始化的层，产生看似正常但错误的推理结果——对 FHSS
    抗干扰这种以 BER 为最终指标的任务，这种静默错误比直接失败更危险，因此宁可让
    调用方重新训练。除版本号外还有 architecture 字符串这道独立闸门，任何一侧
    不一致都拒绝加载。

    加载流程：校验 -> 用 config 重建 PolicyNet -> eval() -> 用 observation_shape
    对应的占位输入 dry-run 一次（物化 Lazy 层）-> load_state_dict -> 再次 eval()。

    Args:
        path: checkpoint 文件路径。
        device: 加载到的设备（默认 "cuda"）；dry-run 前向与返回的策略都在这台设备上。
        expected_num_heads: 期望的头数；给定且与 checkpoint 不符时抛错。
        expected_n_actions: 期望的每头动作数；给定且不符时抛错。
        expected_observation_shape: 期望的观测形状；给定且不符时抛错。

    Returns:
        (policy, metadata)：policy 是 eval 模式、已载权重、位于 device 上的
        PolicyNet；metadata 是保存时写入的 dict（缺失时为空 dict）。

    Raises:
        ValueError: 版本为 v1/v2/v3、版本号未知、model_type 不是多头策略、
            architecture 不匹配，或 num_heads / n_actions / observation_shape
            与期望值不一致。
    """
    # weights_only=True：只反序列化张量与基础容器，避免 pickle 任意对象执行，
    # 这是加载外部 checkpoint 的安全默认。
    payload = torch.load(path, map_location=device, weights_only=True)
    format_version = payload.get("format_version")
    # 下面三个分支必须在通用版本比较之前：它们给出的是"重新训练"这一可操作建议，
    # 而不是笼统的版本不支持。
    if format_version == 1:
        raise ValueError(
            "SAC inference checkpoint format v1 uses the old BatchNorm policy "
            "architecture and cannot be loaded safely; retrain SAC to create a "
            "v4 checkpoint."
        )
    if format_version == 2:
        raise ValueError(
            "SAC inference checkpoint format v2 uses the old shallow GroupNorm "
            "policy architecture; retrain SAC to create a v4 checkpoint."
        )
    if format_version == 3:
        raise ValueError(
            "SAC inference checkpoint format v3 uses the old three-layer CNN "
            "policy architecture; retrain SAC to create a v4 checkpoint."
        )
    if format_version != SAC_CHECKPOINT_FORMAT_VERSION:
        raise ValueError("Unsupported SAC inference checkpoint format.")
    if payload.get("model_type") != "MultiHeadSACPolicy":
        raise ValueError("Checkpoint does not contain a multi-head SAC policy.")
    # 第二道闸门：即使版本号相同，架构标识不一致也说明 state_dict 布局不同。
    if payload.get("architecture") != SAC_POLICY_ARCHITECTURE:
        raise ValueError(
            "SAC checkpoint policy architecture does not match "
            f"{SAC_POLICY_ARCHITECTURE!r}; retrain SAC with the current architecture."
        )

    config = dict(payload["config"])
    observation_shape = tuple(payload["observation_shape"])
    # 三个 expected_* 都是可选的外部一致性检查：部署脚本可据此确认 checkpoint 与
    # 当前环境配置（block 数、候选信道数、PSD 分辨率）匹配。
    if expected_num_heads is not None and int(expected_num_heads) != int(
        config["num_heads"]
    ):
        raise ValueError("SAC checkpoint block count does not match.")
    if expected_n_actions is not None and int(expected_n_actions) != int(
        config["n_actions"]
    ):
        raise ValueError("SAC checkpoint action count does not match.")
    if (
        expected_observation_shape is not None
        and tuple(expected_observation_shape) != observation_shape
    ):
        raise ValueError("SAC checkpoint observation shape does not match.")

    policy = PolicyNet(**config).to(device)
    policy.eval()
    with torch.no_grad():
        # dry-run：Lazy 层此刻仍未物化，必须先按观测形状前向一次才能接收 state_dict，
        # 否则会因键/形状不匹配而 load_state_dict 失败（或需要 assign=True 等分支逻辑）。
        policy(
            _checkpoint_images(observation_shape, device),
            torch.full(
                (1, 1),
                (config["hoprate_min"] + config["hoprate_max"]) / 2.0,
                dtype=torch.float32,
                device=device,
            ),
        )
    policy.load_state_dict(payload["actor_state_dict"])
    # load_state_dict 默认 copy_ 到已有参数，不会改 training 标志；这里显式再 eval()
    # 一次，确保返回的对象在任何构建路径下都是确定性推理模式。
    policy.eval()
    return policy, dict(payload.get("metadata", {}))


class SAC:
    """十头因子化离散 SAC 智能体：actor + 双 critic + lazy target + 自适应温度。

    组件构成（五个网络结构相同、参数各自独立，每个网络内部都带一套完整的
    StateEncoder，即 actor 与两个 critic 的编码器不共享参数）：
        actor              PolicyNet    十头 logits，策略更新 / 采样
        critic_1, critic_2 ValueNet     两套独立 Q，clipped double-Q 抑制高估
        target_critic_1/2  ValueNet     与 online 同构，eval 模式，只用于 TD target

    优化器：actor / critic_1 / critic_2 各一个 Adam，外加一个只优化标量 log_alpha
    的 Adam。target 网络没有优化器——它们只通过 soft_update 被写入。

    温度自适应：log_alpha 是可学习标量（0 维 tensor，requires_grad=True，单独放入
    alpha 优化器，初值 log(0.01) 即 alpha≈0.01）。十头共享它，因此 alpha_loss 对
    十头熵取平均来与 target_entropy 对齐。用 log 参数化而非直接学 alpha，是为了
    保证 alpha 恒正（无需投影/截断），且梯度尺度在数量级变化时更稳定。

    lazy target 机制：target_critic_1/2 在构造时只建结构、不复制参数，因为此时
    Lazy 层的形状尚未确定（UninitializedParameter 无法拷贝）。首次 calc_target 时
    才用一个样本把 online critic 物化，再把参数/缓冲复制到 target 上
    （见 _ensure_target_critics_initialized）。此前 target 网络不参与任何计算，因此
    "未初始化" 期间没有数值风险；这样也避免了启动时一次无意义的全量复制。

    软更新：每个优化步后 target <- (1 - tau) * target + tau * online（含 buffer
    同步，如 GroupNorm 无 running stats 时 buffer 为空，该分支自然空转），使 TD
    target 缓慢跟随在线 critic，兼顾稳定与时效。

    Args:
        n_actions: 每个头的候选 offset 数。
        actor_lr: actor 的 Adam 学习率。
        critic_lr: 两个在线 critic 的 Adam 学习率。
        alpha_lr: 温度 log_alpha 的 Adam 学习率。
        target_entropy: 目标熵（通常取 ratio * log(n_actions)，远小于最大熵
            log(n_actions)，即鼓励偏低熵但仍保留探索）。
        tau: 软更新系数（0 < tau <= 1）。
        gamma: 折扣因子。
        device: 网络所在设备（字符串或 torch.device）。
        num_heads: 头数，须与环境 block 数一致（默认 10）。
        hoprate_min: hoprate 归一化下界（Hz，默认 10.0）。
        hoprate_max: hoprate 归一化上界（Hz，默认 1000.0）。
    """

    def __init__(
        self,
        n_actions,
        actor_lr,
        critic_lr,
        alpha_lr,
        target_entropy,
        tau,
        gamma,
        device,
        num_heads=10,
        hoprate_min=10.0,
        hoprate_max=1000.0,
    ):
        """构建五个网络、四个优化器与温度变量，并完成全部超参的初始化。

        网络的 dtype/设备此处一次性确定（.to(self.device)），训练与推理不再发生
        隐式迁移；所有网络保持 PyTorch 默认的 train 模式（target 除外，见类文档）。

        Args:
            n_actions, actor_lr, critic_lr, alpha_lr, target_entropy, tau, gamma,
            device, num_heads, hoprate_min, hoprate_max: 见类 docstring。
        """
        self.n_actions = int(n_actions)
        self.num_heads = int(num_heads)
        self.device = torch.device(device)

        # 五个网络共用同一份构造参数，保证结构严格一致（target 与 online 必须同构
        # 才能 load_state_dict）；这份 dict 的字段与推理 checkpoint 的 config 对应。
        network_args = {
            "n_actions": self.n_actions,
            "num_heads": self.num_heads,
            "hoprate_min": hoprate_min,
            "hoprate_max": hoprate_max,
        }
        self.critic_1 = ValueNet(**network_args).to(self.device)
        self.critic_2 = ValueNet(**network_args).to(self.device)
        self.actor = PolicyNet(**network_args).to(self.device)
        # target 网络先只建结构：此刻 Lazy 层未物化，参数无法复制，留待首次
        # calc_target 时惰性初始化。
        self.target_critic_1 = ValueNet(**network_args).to(self.device)
        self.target_critic_2 = ValueNet(**network_args).to(self.device)
        # 固定 eval：target 不得再有任何"训练态"行为（本项目用 GroupNorm，本就
        # train/eval 一致，这里是显式约束，防止后续引入 Dropout 等层时被误用）。
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        # 惰性初始化标记：False 时 target 参数尚未从 online 复制，不可用于计算。
        self._target_critics_initialized = False

        # 三个优化器各自只持有自己网络的参数，互不影响（target 无优化器）。
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_1_optimizer = torch.optim.Adam(
            self.critic_1.parameters(), lr=critic_lr
        )
        self.critic_2_optimizer = torch.optim.Adam(
            self.critic_2.parameters(), lr=critic_lr
        )

        # 温度用 log 空间参数化：初值 log(0.01) 对应 alpha=0.01（熵项权重较小，
        # 训练早期以回报为主）。它是 0 维 tensor 而非 nn.Parameter，因此需要显式
        # requires_grad=True 并手动交给优化器；0 维形状让 alpha 能直接与 [B, H]
        # 的张量广播相乘而不引入额外维度。
        self.log_alpha = torch.tensor(
            np.log(0.01),
            dtype=torch.float32,
            device=self.device,
            requires_grad=True,
        )
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr
        )

        self.target_entropy = float(target_entropy)
        self.gamma = float(gamma)
        self.tau = float(tau)

    @staticmethod
    def _single_image_tensor(state_img, device):
        """把单张 PSD 图转成 batch=1 的张量，形状 [1, 1, H, W] 或 [1, C, H, W]。

        Args:
            state_img: 单张图，[H, W]（视为单通道）或 [C, H, W]。
            device: 目标设备。

        Returns:
            [1, 1, H, W] 或 [1, C, H, W] 的 float32 张量。

        Raises:
            ValueError: 维度不是 2 或 3（多张图或未知布局无法作为单样本处理）。
        """
        img = torch.as_tensor(state_img, dtype=torch.float32, device=device)
        if img.ndim == 2:
            return img.unsqueeze(0).unsqueeze(0)
        if img.ndim == 3:
            return img.unsqueeze(0)
        raise ValueError(
            f"state_img must have shape [H,W] or [C,H,W], got {tuple(img.shape)}."
        )

    def take_action(self, state_img, hoprate, deterministic=False):
        """为单个状态生成一条完整的十头动作向量。

        这是训练/评测主循环最常用的入口（每个环境 step 调用一次）：把单样本包装成
        batch=1 后复用 take_actions，避免重复实现采样逻辑，同时保证单样本与批量
        路径的数值行为完全一致。

        Args:
            state_img: 单张 PSD 图像，[H, W] 或 [C, H, W]。
            hoprate: 当前跳速（Hz），标量。
            deterministic: True 时逐头 argmax（评测/复现用），False 时按 categorical
                采样（训练探索用）。

        Returns:
            np.ndarray，dtype=int64，形状 (num_heads,)，元素范围 [0, n_actions)。

        Raises:
            ValueError: state_img 维度不是 2 或 3。
        """
        state_array = np.asarray(state_img)
        if state_array.ndim not in (2, 3):
            raise ValueError(
                f"state_img must have shape [H,W] or [C,H,W], got {state_array.shape}."
            )
        # 加 batch 维后走批量实现；hoprate 同步包成 (1,) 保持批次配对。
        return self.take_actions(
            state_array[np.newaxis, ...],
            np.asarray([float(hoprate)], dtype=np.float32),
            deterministic=deterministic,
        )[0]

    def take_actions(self, state_imgs, hoprates, deterministic=False):
        """批量生成各状态的十头动作（一次 actor 前向）。

        raw logits 语义：actor 输出 [B, num_heads, n_actions] 的未归一化 logits，
        随机路径把它交给 Categorical（内部 log_softmax，等价于逐头独立 categorical
        采样），确定性路径逐头在 dim=-1 上 argmax；两条路径共用同一次前向，因此同一
        组输入下 argmax 结果与采样分布严格对应同一策略。

        模式语义：本方法不切换 actor 的 train/eval（不调用 actor.eval()），推理与
        训练共用同一条数值路径；配合 GroupNorm，两种模式结果一致，而网络上层的
        调用方即使正处于训练的 train 模式也不会被本方法悄然改变状态。

        Args:
            state_imgs: [B, H, W] 或 [B, C, H, W] 的 PSD 图像批（numpy 或 tensor）。
            hoprates: 长度 B 的跳速（Hz；numpy、列表或 tensor）。
            deterministic: True 逐头 argmax，False 按 categorical 采样。

        Returns:
            np.ndarray，dtype=int64，形状 (B, num_heads)，元素范围 [0, n_actions)。

        Raises:
            ValueError: 图像维度非法，或 state_imgs 与 hoprates 的 batch 不一致。
        """
        imgs = self._batch_images(state_imgs)
        hoprate_tensor = self._batch_hoprates(hoprates)
        if imgs.shape[0] != hoprate_tensor.shape[0]:
            raise ValueError(
                "state_imgs and hoprates must contain the same number of samples."
            )

        # LazyConv2d 可能在此首次调用中物化参数；no_grad 保证这些新参数不带
        # requires_grad 的中间图，之后训练前向仍能正常参与 autograd。
        with torch.no_grad():
            logits = self.actor(imgs, hoprate_tensor)
            if deterministic:
                # 逐头取最大 logits 对应的 offset：推理端完全确定、可复现。
                actions = logits.argmax(dim=-1)
            else:
                # 离散 SAC 的探索来自这个 categorical 采样；十头各自独立采样，
                # 每次调用得到一条完整的联合动作。
                actions = torch.distributions.Categorical(logits=logits).sample()
        # 转 numpy 交给环境（环境侧为 numpy 接口）；astype(copy=False) 在已是 int64
        # 时避免额外拷贝。
        return actions.cpu().numpy().astype(np.int64, copy=False)

    def _batch_images(self, images):
        """把 replay/环境给的图像统一成 [B, C, H, W] 的 float32 张量并搬到设备。

        同时兼容 numpy 与 torch 输入：[B, H, W] 会被补上通道维变成 [B, 1, H, W]，
        使单通道 PSD 与多通道输入共享同一套下游代码。非块拷贝（non_blocking）在
        数据来自 pinned memory 时可与计算重叠。

        Args:
            images: [B, H, W] 或 [B, C, H, W] 的 numpy 数组或 torch 张量。

        Returns:
            [B, C, H, W] 的 float32 张量（位于 self.device）。

        Raises:
            ValueError: 维度不是 3 或 4。
        """
        if torch.is_tensor(images):
            tensor = images.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        else:
            tensor = torch.as_tensor(
                images, dtype=torch.float32, device=self.device
            )
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)  # [B,H,W] -> [B,1,H,W]
        if tensor.ndim != 4:
            raise ValueError(
                "Replay images must have shape [B,H,W] or [B,C,H,W], got "
                f"{tuple(tensor.shape)}."
            )
        return tensor

    def _batch_hoprates(self, hoprates):
        """把 hoprate 统一成 [B, 1] 的 float32 张量并搬到设备。

        view(-1, 1) 兼容 (B,)、(B,1) 甚至标量数组等各种上游布局：normalize_hoprate
        要求最后一维为 1，这里保证该约定始终成立。

        Args:
            hoprates: 长度 B 的跳速（Hz），numpy/列表/tensor 均可。

        Returns:
            [B, 1] 的 float32 张量（位于 self.device）。
        """
        if torch.is_tensor(hoprates):
            return hoprates.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ).view(-1, 1)
        return torch.as_tensor(
            hoprates, dtype=torch.float32, device=self.device
        ).view(-1, 1)

    def _ensure_target_critics_initialized(self, imgs, hoprates):
        """首次使用前把 target critic 物化并从 online critic 复制参数。

        动机（两重约束）：
        1. 编码器含 LazyConv2d/LazyLinear，构造后参数是 UninitializedParameter，
           形状未知时 load_state_dict 无法完成；
        2. 未初始化的 target 网络不能参与任何前向，否则会输出错误 Q 值并污染 TD
           target。
        因此把"物化 + 复制"推迟到第一次真正需要 target 的时刻（calc_target 开头），
        用真实 batch 的一行（imgs[:1], hoprates[:1]）做 dry-run 触发形状推断——
        只需一个样本即可，取切片是为了避免额外的整批卷积开销。此后
        _target_critics_initialized 为 True，本方法变成空操作，参数交由 soft_update
        持续跟踪在线 critic。

        Args:
            imgs: [B, 1, H, W]/[B, C, H, W] 的当前/下一状态图像批（只用第 0 条）。
            hoprates: [B, 1] 的对应跳速（只用第 0 条）。

        Returns:
            None。

        副作用：
            - 物化 online critic 的 Lazy 参数（这也会被 _init_weights 之外的默认
              初始化填充，随后被下面的 load_state_dict 覆盖）；
            - 把两个 target critic 置为 eval 并置初始化标记。
        """
        if self._target_critics_initialized:
            return

        with torch.no_grad():
            # 只用于触发 Lazy 层形状推断；no_grad 避免为这次临时前向建图。
            self.critic_1(imgs[:1], hoprates[:1])
            self.critic_2(imgs[:1], hoprates[:1])

        # state_dict 中包含编码器与十个 Q 头的全部参数，逐个键复制保证 target 与
        # online 完全同构；这是"惰性初始化"的实质内容。
        self.target_critic_1.load_state_dict(self.critic_1.state_dict())
        self.target_critic_2.load_state_dict(self.critic_2.state_dict())
        # 再显式 eval 一次：load_state_dict 不改模式，但保证任何构建路径下 target
        # 都处于确定性推理模式。
        self.target_critic_1.eval()
        self.target_critic_2.eval()
        self._target_critics_initialized = True

    def calc_target(self, block_rewards, next_imgs, next_hoprates, dones):
        """计算十头 TD target（离散 SAC 的 bootstrap 项）。

        数学形式（逐头 h 独立，再跨头共享一份 step 级价值）：
            V(s') = mean_h sum_a pi_h(a|s') * [ min(Q1_h, Q2_h)(s', a)
                                              - alpha * log pi_h(a|s') ]
            y_h   = r_h + gamma * (1 - done) * V(s')
        要点：
        - 对动作维做"精确期望"而非采样：离散动作可直接枚举全部 n_actions，方差比
          单样本 bootstrap 更低；概率来自 actor 的 log_softmax，因此期望项同时
          包含熵正则（-alpha * log pi）。
        - 用两个 target critic 的逐元素最小值（clipped double-Q）抑制 Q 高估。
        - mean_h 把十个头压缩成一个全局价值再广播回十个头：一次 step 的十个 offset
          共同决定该 step 的回报，故十头共享同一 bootstrap 信号；这也是与"十头共享
          一个 alpha"、step-level replay 相一致的建模选择。
        - 全程 torch.no_grad()：TD target 是监督信号而非被优化的量，必须与计算图
          断开，否则会经 target critic/actor 反向传播导致目标漂移。
        - (1 - done) 掩码：终止状态之后不再 bootstrap，只保留即时奖励。

        Args:
            block_rewards: [B, num_heads] 的逐 block 即时奖励。
            next_imgs: [B, 1, H, W] 或 [B, C, H, W] 的下一状态图像。
            next_hoprates: [B, 1] 的下一状态跳速（Hz）。
            dones: [B, 1] 的终止标志（float，1.0 表示终止）。

        Returns:
            [B, num_heads] 的 TD target；第 1 维的十列为该 step 的全局 bootstrap
            值广播到各头，便于与 Q 预测逐头做 MSE。

        副作用：
            首次调用会触发 target critic 的惰性初始化。
        """
        # 惰性初始化必须在 no_grad 块外先完成（内部会物化 online critic）。
        self._ensure_target_critics_initialized(next_imgs, next_hoprates)
        with torch.no_grad():
            # actor 只提供分布（logits），不参与梯度；这里用 target critic 估值，
            # actor 自身为在线网络，因此 bootstrap 项对 actor 的更新是"给定分布下的
            # 期望 Q"，与 actor loss 中的 min(Q_online) 一起构成策略改进方向。
            next_logits = self.actor(next_imgs, next_hoprates)
            next_log_probs = F.log_softmax(next_logits, dim=-1)
            next_probs = next_log_probs.exp()
            q1 = self.target_critic_1(next_imgs, next_hoprates)
            q2 = self.target_critic_2(next_imgs, next_hoprates)
            # 逐元素 min：对同一 (head, action) 取两个 critic 的较小估计。
            min_q = torch.minimum(q1, q2)
            # sum_a 在动作维上做期望：pi * (Q - alpha * log pi)，
            # 结果形状 [B, num_heads]（每个头一个 soft value）。
            head_values = torch.sum(
                next_probs * (min_q - self.log_alpha.exp() * next_log_probs),
                dim=-1,
            )
            # 十头取均值得到 step 级全局价值，keepdim 保持 [B, 1] 以便广播。
            global_next_value = head_values.mean(dim=1, keepdim=True)
        # 广播到十头：每个头共享同一 bootstrap 项，叠加各自的即时 block reward；
        # (1 - dones) 在终止处截断 bootstrap。
        return block_rewards + self.gamma * global_next_value * (1.0 - dones)

    def soft_update(self, net, target_net):
        """按 tau 对 target 网络做指数滑动平均（Polyak 更新）。

        target <- (1 - tau) * target + tau * online。相比周期性硬拷贝，软更新让 TD
        target 平滑跟随在线网络，避免目标突变引起的 Q 值震荡；tau 很小（默认 0.005）
        时 target 等价于在线网络最近约 1/tau 步的滑动平均。

        实现细节：所有参数共用同一个 tau（不做每层差异化）；用 .data.copy_ 原地写入，
        避免 autograd 记录这一步（优化器状态与图都不受影响，且 target 的
        requires_grad 语义不变）。

        Args:
            net: 在线网络（参数来源），如 self.critic_1。
            target_net: 目标网络，必须与 net 参数/buffer 顺序一致（同构网络）。

        Returns:
            None（原地修改 target_net）。
        """
        for target_param, param in zip(target_net.parameters(), net.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        # buffer 同步是为将来引入 running statistics 类层（如 BatchNorm）预留的
        # 通路；当前架构（GroupNorm，无 running stats）该循环为空操作。
        for target_buffer, buffer in zip(target_net.buffers(), net.buffers()):
            target_buffer.data.copy_(buffer.data)

    def update(self, transition_dict, return_stats=True):
        """执行一次完整梯度步：critic -> actor -> alpha -> target 软更新。

        数据统一来自 ReplayBuffer.sample/_batch_from_transitions 的 step-level 字典，
        本方法负责把 numpy 字段搬到设备并按需转 dtype。

        损失与梯度路径（p = softmax(logits) 为十头策略，逐头独立）：
        1) critic loss：td_target = calc_target(...)（no_grad，形状 [B, num_heads]），
           预测值 q_pred = gather(Q(s, a_taken), action)，两条分支分别对 td_target 做
           MSE：critic 目标是逐头奖励 + 全局 bootstrap，二者形状 [B, num_heads]；
           F.mse_loss 默认对 batch 维与头维一起取平均，因此单头样本的权重相同。
           两个 critic 各自 backward/step，互不共享梯度。
        2) actor loss：对动作维枚举求期望——sum_a p * (alpha * log p - min Q(s,a))，
           再对维度取平均。alpha 在此处被 .detach()，即策略梯度不回头去影响温度
           （温度由 3) 单独学）；min Q 在 no_grad 下计算，避免 actor 的梯度经 critic
           参数传播（critic 只走 1) 的路径）。最终 .mean() 同时平均 batch 维与十头：
           十个头的策略梯度等权相加（每头贡献被 1/(B*num_heads) 缩放），与"十头共享
           温度、共享全局 bootstrap"的设定一致。
        3) alpha loss：log_alpha * (H.detach() - target_entropy)，对 batch 与十头取
           平均。梯度 d(loss)/d(log_alpha) = H̄ - target_entropy：熵低于目标时梯度为
           负，Adam 抬高 log_alpha（增大熵项权重，鼓励探索），反之降低，形成对目标熵
           的自适应追踪；entropy 被 detach，使温度的学习不干扰 actor 参数。
        4) soft update：两个 target critic 向在线 critic 滑动平均，使下一轮 TD target
           平滑跟随。

        模式约定：全程不调用 actor/critic 的 eval()，在线网络保持 train 模式前向
        （GroupNorm 下与 eval 数值一致，无需切换）；target critic 已在初始化时固定
        为 eval。

        Args:
            transition_dict: 批字段字典，至少包含
                state_imgs / next_state_imgs（[B,H,W] 或 [B,C,H,W]）、
                hoprates / next_hoprates（[B] 或 [B,1]）、
                actions（[B, num_heads]，int）、block_rewards（[B, num_heads]）、
                dones（[B] 或 [B,1]，float）。step_rewards 字段被读取方忽略。
                numpy 数组与 torch 张量均可。
            return_stats: True 返回统计量字典；False 返回空 dict，用于跳过
                .item() 同步（高频更新时可省下 GPU->CPU 同步开销）。

        Returns:
            dict；return_stats=True 时包含
            critic1_loss / critic2_loss / actor_loss / alpha_loss / alpha /
            entropy（均为 python float，entropy 是十头平均熵），
            否则为空 dict。

        Raises:
            ValueError: actions/block_rewards 形状不等于 (B, num_heads)，或动作
                超出 [0, n_actions) 范围。
        """
        # --- 观测与条件输入：统一成 [B,C,H,W] / [B,1] 并放到 self.device ---
        imgs = self._batch_images(transition_dict["state_imgs"])
        next_imgs = self._batch_images(transition_dict["next_state_imgs"])
        hoprates = self._batch_hoprates(transition_dict["hoprates"])
        next_hoprates = self._batch_hoprates(
            transition_dict["next_hoprates"]
        )

        # 期望形状 (B, num_heads)：动作与逐头奖励都必须严格是这个形状，防止
        # 上游广播出静默错误的 (B, 1) 或 (1, H) 张量。
        expected_shape = (imgs.shape[0], self.num_heads)
        raw_actions = transition_dict["actions"]
        if torch.is_tensor(raw_actions):
            # 张量分支：先转到目标设备再校验，避免在错误设备上做比较。
            actions = raw_actions.to(
                device=self.device,
                dtype=torch.long,
                non_blocking=True,
            )
            actions_shape = tuple(actions.shape)
            invalid_actions = torch.any(actions < 0) or torch.any(
                actions >= self.n_actions
            )
        else:
            # numpy 分支：在 host 上校验（省一次设备同步），再拷贝到设备。
            actions_array = np.asarray(raw_actions)
            actions_shape = tuple(actions_array.shape)
            invalid_actions = np.any(actions_array < 0) or np.any(
                actions_array >= self.n_actions
            )
            actions = torch.as_tensor(
                actions_array, dtype=torch.long, device=self.device
            )
        if actions_shape != expected_shape:
            raise ValueError(
                f"actions must have shape {expected_shape}, got {actions_shape}."
            )
        if invalid_actions:
            raise ValueError("Replay actions are outside the configured action range.")
        raw_block_rewards = transition_dict["block_rewards"]
        if torch.is_tensor(raw_block_rewards):
            block_rewards = raw_block_rewards.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            )
        else:
            block_rewards = torch.as_tensor(
                raw_block_rewards,
                dtype=torch.float32,
                device=self.device,
            )
        raw_dones = transition_dict["dones"]
        # done 转成 float 并 view(-1, 1)：直接参与 (1 - done) 的乘法掩码，形状
        # [B,1] 可广播到 [B, num_heads]。
        if torch.is_tensor(raw_dones):
            dones = raw_dones.to(
                device=self.device,
                dtype=torch.float32,
                non_blocking=True,
            ).view(-1, 1)
        else:
            dones = torch.as_tensor(
                raw_dones, dtype=torch.float32, device=self.device
            ).view(-1, 1)

        # --- 1) critic 更新：TD 误差，两个 critic 独立反向 ---
        if tuple(block_rewards.shape) != expected_shape:
            raise ValueError(
                "block_rewards must have shape "
                f"{expected_shape}, got {tuple(block_rewards.shape)}."
            )

        td_target = self.calc_target(
            block_rewards, next_imgs, next_hoprates, dones
        )
        # gather(-1, a)：从 [B, num_heads, n_actions] 中取出实际执行动作对应的 Q，
        # unsqueeze 只是为了让 gather 的 index 与最后一维对齐，squeeze 还原为
        # [B, num_heads]，与 td_target 逐头对应。
        q1_pred = self.critic_1(imgs, hoprates).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        q2_pred = self.critic_2(imgs, hoprates).gather(
            -1, actions.unsqueeze(-1)
        ).squeeze(-1)
        critic_1_loss = F.mse_loss(q1_pred, td_target)
        critic_2_loss = F.mse_loss(q2_pred, td_target)

        # set_to_none=True 释放梯度缓存（比置零更省显存），backward 后立即 step，
        # 两个 critic 的顺序更新互不影响（各自的优化器只持有自己的参数）。
        self.critic_1_optimizer.zero_grad(set_to_none=True)
        critic_1_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.zero_grad(set_to_none=True)
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # --- 2) actor 更新：最小化 E_a[alpha * log pi(a) - Q(s,a)] ---
        logits = self.actor(imgs, hoprates)
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()
        # 逐头逐样本熵 [B, num_heads]：既用于 alpha loss，也作为训练监控指标
        # （离散分布最大熵为 log(n_actions)）。
        entropy = -torch.sum(probs * log_probs, dim=-1)

        with torch.no_grad():
            # 用在线 critic 的 min 作为策略改进目标；no_grad 保证 actor 的梯度不回传
            # 进 critic（critic 只由上面的 MSE 训练）。
            min_q = torch.minimum(
                self.critic_1(imgs, hoprates),
                self.critic_2(imgs, hoprates),
            )
        # 动作维期望后 .mean()：一次性对 batch 与十头取平均，即每个头的策略梯度
        # 等权（缩放 1/(B*num_heads)）。alpha 用 .detach()：温度是"给定常数"，
        # 其自身的学习交给 alpha loss。
        actor_loss = torch.sum(
            probs * (self.log_alpha.exp().detach() * log_probs - min_q),
            dim=-1,
        ).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        # --- 3) 温度自适应：把十头平均熵拉向 target_entropy ---
        # entropy.detach()：避免 alpha 的学习反传到 actor；梯度符号由
        # (H - target_entropy) 决定，熵过高/过低分别减小/增大 alpha。
        alpha_loss = (
            self.log_alpha * (entropy.detach() - self.target_entropy)
        ).mean()
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # --- 4) 目标网络软更新：让下一轮的 TD target 平滑跟随在线 critic ---
        self.soft_update(self.critic_1, self.target_critic_1)
        self.soft_update(self.critic_2, self.target_critic_2)

        if not return_stats:
            return {}

        # 转成 python float 便于日志/写盘；这一步会触发 GPU 同步，因此提供了
        # return_stats=False 的旁路。
        return {
            "critic1_loss": float(critic_1_loss.item()),
            "critic2_loss": float(critic_2_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.log_alpha.exp().item()),
            "entropy": float(entropy.mean().item()),
        }
