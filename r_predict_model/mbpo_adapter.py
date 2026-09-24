"""step-level SAC replay 与 MBPO 奖励集成模型之间的适配层。

为什么需要这一层
----------------
SAC.ReplayBuffer 以"环境 step"为单位存放 8 元定序元组，StepRewardEnsemble
（r_predict_model/model.py）面向设备端张量 batch 训练，两者在数据结构、批大小
语义与随机性来源上都不一致。本模块只做数据搬运与形状/范围校验，不改变任何
一方的算法，把耦合限制在三个方向上：

    1. replay → 奖励模型训练数据：
       ``replay_fields_for_reward_model`` 按定序元组下标提取奖励模型真正需要的
       4 个字段（state_imgs / hoprates / actions / block_rewards），经
       ``train_reward_model_from_replay`` 交给 ``StepRewardEnsemble.fit``
       （fit 每次都用全部真实 replay 从头重拟合，不做增量续训）。
    2. replay → SAC 训练 batch：
       ``replay_tensor_dataset`` 把 buffer 全池快照包成 TensorDataset，
       ``resolve_mixed_counts`` 解析 real_ratio 下 real/model 各自应取的条数，
       ``sample_mixed_batch`` 演示"两个 DataLoader 各取一半再拼接"的用法；
       train_mbpo.py 复用前两个函数自建持久 DataLoader（避免每 step 重建）。
    3. 奖励模型 → 合成 transition：
       ``rollout_reward_model`` 从 real replay 采样起点、用当前 SAC policy 采
       offsets、由 elite ensemble 在潜变量空间采样并 sigmoid 映射到有界
       block reward，复用起点的外生 next_state_imgs / next_hoprates / dones，
       把一步合成 transition 追加进 model replay（容量满时 FIFO 淘汰最旧）。

关键不变量
----------
- ``REPLAY_FIELDS`` 的顺序是 SAC 侧的位置契约：``_batch_to_dict`` 用它把
  TensorDataset 的按位张量还原成命名字段，train_mbpo.py 也用它按位拼接
  real/model 两半 batch，任何重排都会静默错位；
- 合成 transition 的状态与终止信号来自真实 replay（环境动力学对 reward-only
  MBPO 是外生的，奖励模型不预测 next state），只有 block_rewards 是模型输出；
- 采样出的 block reward 必须落在由 BER 端点与奖励系数导出的 [lower, upper]
  内，rollout 对此显式校验，越界即视为模型或配置不一致（而不是静默裁剪）。

相关文件：
    - SAC.py：ReplayBuffer 的定序元组契约、add_batch、get_all、FIFO 容量；
    - r_predict_model/model.py：StepRewardEnsemble.fit / sample_rewards /
      reward_bounds_from_config；
    - train_mbpo.py：唯一生产调用方（混合 batch 与 rollout 的调度与日志）。
"""

import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import settings
from .model import (
    DEFAULT_BER_MAX,
    DEFAULT_BER_MIN,
    reward_bounds_from_config,
)


# SAC 侧张量字段的位置契约（顺序不可调整）：既用于 TensorDataset 的按位构造，
# 也用于把按位张量还原成命名字段、以及训练侧拼接 real/model 两半 batch。
# 字段语义与形状（B 为批大小）：
#   state_imgs [B,H,W]、hoprates [B]、actions [B,num_heads]、
#   block_rewards [B,num_heads]、step_rewards [B]（仅诊断，SAC.update 不读）、
#   next_state_imgs [B,H,W]、next_hoprates [B]、dones [B]。
REPLAY_FIELDS = (
    "state_imgs",       # [B, H, W] float32，动作前 PSD 观测
    "hoprates",         # [B] float32，本 step 生效的跳速
    "actions",          # [B, num_heads] int64，起始信道 offset
    "block_rewards",    # [B, num_heads] float32，逐 block 奖励
    "step_rewards",     # [B] float32，仅诊断；SAC.update 不读取
    "next_state_imgs",  # [B, H, W] float32，动作后 PSD 观测
    "next_hoprates",    # [B] float32，下一 step 的策略输入 hoprate
    "dones",            # [B] float32，1.0 表示该 step 终止
)


def replay_fields_for_reward_model(replay_buffer):
    """直接按定序元组下标提取奖励模型训练所需的 4 个字段（不做整池数组拷贝）。

    刻意绕过 ``buffer.get_all()``：那会把 8 个字段全部堆成 float32 大数组
    （state_imgs 是最贵的），而 ``fit`` 只需要 4 个字段。这里直接遍历
    ``buffer.buffer``（deque 中每条都是 SAC 契约的定序元组），只按下标取
    state/hoprate/actions/block_rewards 并保持原始逐条数组引用，把拷贝推迟到
    ``fit`` 内部的张量化阶段。

    依赖 SAC.ReplayBuffer 的元组下标契约：(0) state_img、(1) hoprate、
    (2) actions、(3) block_rewards、(4) step_reward、(5) next_state_img、
    (6) next_hoprate、(7) done。该顺序由 SAC.py 固定，不得调整。

    Args:
        replay_buffer: SAC.ReplayBuffer（需提供 ``buffer`` 与 ``size()``）。

    Returns:
        dict，键为 ``state_imgs`` / ``hoprates`` / ``actions`` / ``block_rewards``，
        值为按插入顺序排列的逐条数据列表（长度均为 N）。

    Raises:
        ValueError: replay 为空（空池无法拟合，且 fit 也会因缺数据报错）。
    """
    transitions = tuple(replay_buffer.buffer)
    if not transitions:
        raise ValueError("Cannot train the reward model from an empty replay buffer.")
    # 元组快照避免在遍历期间 deque 被并发写入；下标对应 SAC 的定序契约。
    return {
        "state_imgs": [transition[0] for transition in transitions],
        "hoprates": [transition[1] for transition in transitions],
        "actions": [transition[2] for transition in transitions],
        "block_rewards": [transition[3] for transition in transitions],
    }


def concat_transition_batches(batches, shuffle=True):
    """把多个 SAC 风格 batch 沿 batch 维（axis=0）拼接，可选再整体打乱。

    用途是把 real 半批与 model 半批合成一个完整 batch 交给 ``agent.update``。
    要求所有非 None 的 batch 拥有**完全相同的字段集合与顺序**——字段错位会让
    reward 与状态张量错配且不报错，因此这里显式校验 schema，宁可早失败。
    None 表示该侧样本数为 0（resolve_mixed_counts 可能返回 0），直接跳过。

    打乱用全局 numpy RNG（``np.random.permutation``），因此训练入口调用
    ``settings.set_random_seeds`` 后行为可复现；当只有 1 条样本时不打乱
    （permutation 无意义，且避免无谓分配）。

    Args:
        batches: batch 字典的可迭代对象，元素可为 None（表示该侧为空）。
        shuffle: True 时对拼接结果整体做一次行置换。

    Returns:
        dict，键与输入 batch 一致，值为拼接后的 numpy 数组，batch 维长度为各
        输入长度之和。

    Raises:
        ValueError: 全部 batch 都是 None，或各 batch 字段集合/顺序不一致。
    """
    valid_batches = [batch for batch in batches if batch is not None]
    if not valid_batches:
        raise ValueError("Need at least one batch to concatenate.")
    # 以第一个 batch 的键序为基准（SAC 侧字段顺序本身就是契约）。
    keys = tuple(valid_batches[0].keys())
    if any(tuple(batch.keys()) != keys for batch in valid_batches[1:]):
        raise ValueError("Transition batches do not have the same schema.")
    combined = {
        key: np.concatenate([batch[key] for batch in valid_batches], axis=0)
        for key in keys
    }
    if shuffle and len(next(iter(combined.values()))) > 1:
        permutation = np.random.permutation(len(next(iter(combined.values()))))
        combined = {key: value[permutation] for key, value in combined.items()}
    return combined


def _target_real_count(batch_size, real_ratio):
    """把 real_ratio 换算成目标真实样本条数（不含两侧可用量的裁剪）。

    规则：
        - ``real_ratio`` 必须落在 [0, 1]，否则抛 ValueError（配置错误早暴露）；
        - 四舍五入 ``batch_size * real_ratio``（例如 batch=2048、ratio=0.2 → 410）；
        - 当 batch_size >= 2 且 ratio 严格位于 (0, 1) 之间时，把结果夹到
          [1, batch_size - 1]：此时用户要的是"混合 batch"，四舍五入可能得到
          0 或 batch_size（小 batch 或极端比例下），夹取可让目标条数保留混合
          语义（实际能否混合还取决于两侧可用量，见 ``resolve_mixed_counts``）；
          ratio 为 0/1 或 batch_size < 2 时不做夹取，返回纯边界值。

    Args:
        batch_size: 目标 batch 总条数。
        real_ratio: 真实样本目标占比。

    Returns:
        int：目标 real 条数（0 <= 返回值 <= batch_size）。

    Raises:
        ValueError: real_ratio 不在 [0, 1] 内。
    """
    if not 0.0 <= real_ratio <= 1.0:
        raise ValueError("real_ratio must be in [0, 1].")
    real_count = int(round(batch_size * real_ratio))
    # 保证"混合"语义：严格中间比例下两侧各至少 1 条。
    if batch_size >= 2 and 0.0 < real_ratio < 1.0:
        real_count = min(max(1, real_count), batch_size - 1)
    return real_count


def resolve_mixed_counts(real_size, model_size, batch_size, real_ratio):
    """解析一个 SAC batch 中 real / model 各自的**精确**条数（二者之和 == batch_size）。

    解析规则（按顺序）：
        1. 参数合法性：``batch_size`` 必须为正；两侧可用量之和必须 >= batch_size，
           否则无论怎么分配都凑不满一个 batch，直接报错；
        2. ``model_size == 0``（尚未生成任何合成样本，例如 warm-up 早期）：退化为
           纯 real batch，返回 ``(batch_size, 0)``；此时若 real 侧也不够则报错；
        3. 否则先按 ``_target_real_count`` 求目标 real 条数，再受两侧可用量裁剪：
           ``real_count = min(目标, real_size)``，
           ``model_count = min(batch_size - real_count, model_size)``；
        4. 若因裁剪出现缺口（deficit），先用 real 侧余量补，再用 model 侧余量补；
           仍补不齐说明两侧可用量之和不足（与第 1 步结论矛盾），抛错兜底。

    边界语义：目标比例只是"首选"，可用量不足时实际比例会偏离 real_ratio——
    例如 model 侧只有 3 条时，batch 里多出来的位置全部由 real 填充，而不会
    重复采样同一批 model 样本。因此返回值可能退化为单边 batch（某侧为 0）。

    Args:
        real_size: real replay 当前条数。
        model_size: model replay 当前条数。
        batch_size: 目标 batch 总条数。
        real_ratio: 真实样本目标占比，须在 [0, 1]。

    Returns:
        tuple(int, int)：(real_count, model_count)，满足
        ``real_count + model_count == batch_size``、``real_count <= real_size``、
        ``model_count <= model_size``。

    Raises:
        ValueError: batch_size 非正、real_ratio 越界，或两侧可用量不足 batch_size。
    """
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    real_ratio = float(real_ratio)
    real_size = int(real_size)
    model_size = int(model_size)
    total_available = real_size + model_size
    if total_available < batch_size:
        raise ValueError(
            f"Need {batch_size} total transitions, only {total_available} available."
        )

    if model_size == 0:
        # warm-up 早期：还没做过 rollout，model replay 为空，只能整批取自 real。
        if real_size < batch_size:
            raise ValueError("Real replay is too small for a fallback SAC batch.")
        return batch_size, 0

    # 先按目标比例取 real，再让 model 填满剩余额度；两侧都不能超出各自可用量。
    real_count = min(_target_real_count(batch_size, real_ratio), real_size)
    model_count = min(batch_size - real_count, model_size)
    deficit = batch_size - real_count - model_count
    # 缺口优先用 real 的剩余样本补，再用 model 的剩余样本补（不做有放回重复采样）。
    if deficit:
        additional_real = min(deficit, real_size - real_count)
        real_count += additional_real
        deficit -= additional_real
    if deficit:
        additional_model = min(deficit, model_size - model_count)
        model_count += additional_model
        deficit -= additional_model
    if deficit:
        raise ValueError("Replay buffers cannot supply a complete mixed batch.")

    return real_count, model_count


def replay_tensor_dataset(replay_buffer):
    """把某个 replay buffer 的全池快照包成标准 TensorDataset（按 REPLAY_FIELDS 顺序）。

    ``get_all()`` 返回的是"按插入顺序"的整池数组快照，因此本函数相当于对当前
    池内容做一次时点拷贝：之后 buffer 继续 add / FIFO 淘汰都不会影响这个
    dataset。``torch.as_tensor`` 直接复用 numpy 内存（不做 dtype 提升），
    dtype 沿用 SAC 契约（图像/奖励/hoprate/done 为 float32，动作为 int64）。

    该 dataset 的每个样本是 8 个张量的定序元组，配合默认 collate 得到按位
    batch，因此需要 ``_batch_to_dict`` 还原字段名；train_mbpo.py 则按
    ``REPLAY_FIELDS`` 的位置直接拼接 real/model 两半。

    Args:
        replay_buffer: SAC.ReplayBuffer（需提供 get_all()）。

    Returns:
        TensorDataset：长度为池内条数，第 i 个样本对应 buffer 中第 i 条 transition。

    Raises:
        ValueError: replay 为空（由 get_all() 抛出）。
    """
    replay = replay_buffer.get_all()
    # 生成器按 REPLAY_FIELDS 顺序展开，位置即字段身份，顺序不可调整。
    return TensorDataset(
        *(
            torch.as_tensor(replay[field])
            for field in REPLAY_FIELDS
        )
    )


def _batch_to_dict(batch):
    """把 TensorDataset 产出的按位 batch 还原成以 REPLAY_FIELDS 命名的字典。

    与 ``replay_tensor_dataset`` 的位置顺序严格互逆；依赖二者顺序一致，
    因此不能在 REPLAY_FIELDS 之外单独调整任何一侧。

    Args:
        batch: 长度等于 ``len(REPLAY_FIELDS)`` 的序列（通常是张量元组）。

    Returns:
        dict：字段名 → 张量（或任意批次元素），供 ``concat_transition_batches``
        与 ``agent.update`` 使用。
    """
    return dict(zip(REPLAY_FIELDS, batch))


def sample_mixed_batch(real_buffer, model_buffer, batch_size, real_ratio):
    """用两个标准 DataLoader 各取一段，拼成一个比例精确的混合 batch。

    流程：``resolve_mixed_counts`` 定条数 → 两侧各自建 TensorDataset（整池快照）
    → 各自建 ``shuffle=True`` 的 DataLoader，batch_size 恰为目标条数 → 各取
    第一批 → 拼接。某侧条数为 0 时跳过该侧的 loader（也是 resolve 可能给出的
    合法退化情形）。

    刻意不做二次打乱（``shuffle=False``）：两个 loader 已各自独立随机取样，
    SAC 的 update 不要求 batch 内顺序随机。

    注意：本函数每次调用都会重建 loader，属于便捷封装（测试与一次性取批用）；
    训练热路径（train_mbpo.py）自行持有持久 loader 与迭代器，避免每个 step 重建。

    Args:
        real_buffer: real replay buffer。
        model_buffer: model（合成）replay buffer。
        batch_size: 目标 batch 总条数。
        real_ratio: 真实样本目标占比，须在 [0, 1]。

    Returns:
        dict：键为 REPLAY_FIELDS，batch 维长度为 batch_size（real/model 两段
        按 real 在前、model 在后拼接）。

    Raises:
        ValueError: 数量解析失败（见 ``resolve_mixed_counts``），或拼接时 schema
            不一致。
    """
    real_count, model_count = resolve_mixed_counts(
        real_buffer.size(), model_buffer.size(), batch_size, real_ratio
    )

    batches = []
    if real_count:
        # batch_size=real_count 保证第一批恰好取满，无需 drop_last 或多次迭代。
        real_loader = DataLoader(
            replay_tensor_dataset(real_buffer),
            batch_size=real_count,
            shuffle=True,
        )
        batches.append(_batch_to_dict(next(iter(real_loader))))
    if model_count:
        model_loader = DataLoader(
            replay_tensor_dataset(model_buffer),
            batch_size=model_count,
            shuffle=True,
        )
        batches.append(_batch_to_dict(next(iter(model_loader))))

    # 两个 loader 已各自独立随机取样，SAC update 不要求再对整批做一次打乱。
    return concat_transition_batches(batches, shuffle=False)


def train_reward_model_from_replay(reward_model, replay_buffer, **fit_kwargs):
    """用**当前全部** real replay 重新拟合奖励 ensemble（fit 的薄封装）。

    每次调用都是"从头拟合"：``StepRewardEnsemble.fit`` 会重建成员权重与 Adam
    状态，不做增量续训，因此本函数可以在训练循环里按 model_train_freq 反复调用
    而不会累积旧拟合的偏差。

    Args:
        reward_model: StepRewardEnsemble 实例。
        replay_buffer: real replay buffer（非空）；内容由
            ``replay_fields_for_reward_model`` 提取。
        **fit_kwargs: 透传给 ``fit`` 的训练超参（batch_size、holdout_ratio、
            patience、max_epochs、min_improvement 等）。

    Returns:
        dict：``fit`` 返回的拟合统计（holdout 曲线、elite 索引、epoch 数、
        target_saturation_fraction 等）。

    Raises:
        ValueError: replay 为空，或 fit_kwargs 触发 fit 内部校验失败。
    """
    fields = replay_fields_for_reward_model(replay_buffer)
    return reward_model.fit(**fields, **fit_kwargs)


def reward_bounds(
    hoprates,
    reward_config,
    ber_min=DEFAULT_BER_MIN,
    ber_max=DEFAULT_BER_MAX,
):
    """按 BER 端点与奖励系数求每条 transition 的 reward 物理上下界。

    物理 reward = base_reward - hoprate_penalty * hoprate - ber_penalty * BER，
    在 BER ∈ [ber_min, ber_max] 上取两个端点即得 [lower, upper]（ber_penalty
    为负时顺序自动交换，由 ``reward_bounds_from_config`` 的 min/max 保证）。
    因此下界随 hoprate 单调变化，每条 transition 的区间不同。

    本函数是 ``reward_bounds_from_config`` 的薄封装，只为把默认 BER 端点与
    适配层统一（rollout 会用 ensemble 自身的 ber_min/ber_max 覆盖默认值）。

    Args:
        hoprates: 形状 [B] 的跳速序列（或可转 float32 的标量序列）。
        reward_config: 含 base_reward / ber_penalty / hoprate_penalty 的奖励配置。
        ber_min: BER 下界，默认 ``DEFAULT_BER_MIN``（0.0）。
        ber_max: BER 上界，默认 ``DEFAULT_BER_MAX``（0.5）。

    Returns:
        tuple(np.ndarray, np.ndarray)：逐 transition 的 (lower, upper)，形状均为
        [B, 1] float32。

    Raises:
        ValueError: reward_config 缺键/非有限值/ber_penalty 为 0，BER 端点非法，
            或 hoprates 含非有限值（均由 ``reward_bounds_from_config`` 抛出）。
    """
    return reward_bounds_from_config(
        hoprates,
        reward_config,
        ber_min=ber_min,
        ber_max=ber_max,
    )


def rollout_reward_model(
    reward_model,
    agent,
    real_buffer,
    model_buffer,
    batch_size,
    reward_config,
    deterministic_model=False,
):
    """用奖励模型生成一批**一步**合成 transition 并追加到持久 FIFO model replay。

    完整构造流程（rollout_length 恒为 1，奖励模型不递推状态）：
        1. 起点采样：从 real replay **无放回**随机抽取 ``rollout_size`` 条起点
           （``ReplayBuffer.sample`` 用 ``random.sample``，因此 rollout_size 必须
           先截断到池容量以内），只使用其中的 state_imgs / hoprates /
           next_state_imgs / next_hoprates / dones；起点自带的 actions 与
           block_rewards 被丢弃——合成样本表达的是"当前 SAC 策略在这些真实状态上
           会怎么决策、奖励模型认为会拿到什么奖励"；
        2. 策略采样 offsets：优先调用批量接口 ``agent.take_actions(state_imgs,
           hoprates)``（一次前向出全部头，采样态而非 argmax）；仅暴露单条
           ``take_action`` 的轻量测试 agent / 外部 SAC 兼容实现走逐条回退路径；
           随后校验动作形状恰为 (rollout_size, num_heads) 且落在 [0, n_actions)；
        3. 奖励预测：``reward_model.sample_rewards(...)`` 在潜变量空间工作——
           对每条 transition 从 elite 成员中随机挑一个成员，取该成员的
           Logistic-Normal mean/logvar 采样潜变量，再经 sigmoid 映射回由 BER
           端点与奖励配置导出的 [lower, upper] 区间（``deterministic_model=True``
           时改为取 elite 均值、不注入采样噪声，便于可复现评估）；返回值同时带
           elite 成员间的 disagreement（模型不确定性诊断）；
        4. 有界性复核：预测 reward 必须形状正确、全为有限值且落在
           ``reward_bounds(hoprates, reward_config, ...)`` 给出的区间内
           （容差 ``1e-6 * max(1, upper - lower)``，吸收 sigmoid 浮点误差）；
           越界抛 RuntimeError，视为模型/配置不一致而非静默裁剪；
        5. 追加写入：``model_buffer.add_batch`` 把 (state, hoprate, 策略 offsets,
           预测 block_rewards, **外生** next_state_imgs / next_hoprates / dones)
           一次性写入；next 观测与终止信号直接复用真实 replay 起点——环境动力学
           与终止对 reward-only MBPO 是外生的，奖励模型不预测它们；buffer 满时
           deque 自动 FIFO 淘汰最旧样本，淘汰条数由前后 size 差反推并随统计返回；
        6. 计时：``settings.TIMING_ENABLED`` 为 True 时分别累计
           sample / policy / predict / add 四阶段墙钟耗时，否则 timing 为 None
           （不产生额外 time.time() 调用）。

    Args:
        reward_model: StepRewardEnsemble；需提供 num_heads、n_actions、
            sample_rewards，以及可选的 ber_min / ber_max 属性。
        agent: SAC agent；需提供 take_actions（批量）或 take_action（单条回退）。
        real_buffer: real replay buffer，提供合成样本的状态与终止信号来源。
        model_buffer: model replay buffer（持久、定容 FIFO），合成样本写入目标。
        batch_size: 本次期望生成的合成条数；实际取
            ``min(batch_size, real_buffer.size())``（不会重复采样超过池容量）。
        reward_config: 奖励系数配置，用于计算 reward 物理上下界。
        deterministic_model: True 时奖励取 elite 均值（不采样潜变量噪声）。

    Returns:
        dict：本次 rollout 的统计量——
            generated（实际生成条数）、model_buffer_size_before/after/capacity、
            fifo_evicted（被 FIFO 淘汰的旧样本数）、
            reward_mean/reward_std（合成 reward 的均值与标准差）、
            disagreement_mean/disagreement_p95（elite 间分歧）、
            timing（TIMING_ENABLED 为 False 时是 None）。

    Raises:
        ValueError: real replay 为空、rollout_size <= 0，或策略返回的动作形状/
            取值范围非法。
        RuntimeError: 奖励 ensemble 返回了非有限值或越出物理上下界的 reward。
    """
    if real_buffer.size() == 0:
        raise ValueError("Cannot roll out from an empty real replay buffer.")
    # sample 为无放回抽样，故先截断到池容量；同时这也是 batch_size 的合法上界。
    rollout_size = min(int(batch_size), real_buffer.size())
    if rollout_size <= 0:
        raise ValueError("rollout batch_size must be positive.")
    model_buffer_size_before = model_buffer.size()
    # 计时开关来自全局配置：关闭时 stage_start 恒为 None，各 time.time() 调用被跳过。
    timing_enabled = bool(settings.TIMING_ENABLED)
    timing = {"sample_s": 0.0, "policy_s": 0.0, "predict_s": 0.0, "add_s": 0.0}
    stage_start = time.time() if timing_enabled else None

    # 起点提供状态与外生的 next_state/next_hoprate/done；旧 action/reward 不使用。
    starts = real_buffer.sample(rollout_size)
    if timing_enabled:
        timing["sample_s"] = time.time() - stage_start
        stage_start = time.time()
    state_imgs = starts["state_imgs"]
    hoprates = np.asarray(starts["hoprates"], dtype=np.float32)
    # 优先走批量接口：一次前向出全部头的 offsets（采样态），避免逐条调用。
    take_actions = getattr(agent, "take_actions", None)
    if callable(take_actions):
        actions = np.asarray(
            take_actions(state_imgs, hoprates),
            dtype=np.int64,
        )
    else:
        # 回退路径：兼容只暴露原始单条 API 的轻量测试 agent 与外部 SAC 实现。
        actions = np.stack(
            [
                np.asarray(
                    agent.take_action(state_imgs[index], float(hoprates[index])),
                    dtype=np.int64,
                )
                for index in range(rollout_size)
            ]
        )
    if timing_enabled:
        timing["policy_s"] = time.time() - stage_start
        stage_start = time.time()
    # 合成 transition 必须与 v3 契约同形：(rollout_size, num_heads) 且动作在范围内，
    # 否则写入 model replay 会污染 SAC 更新（且错误更晚才暴露）。
    expected_shape = (rollout_size, reward_model.num_heads)
    if actions.shape != expected_shape:
        raise ValueError(
            f"SAC policy returned actions with shape {actions.shape}, "
            f"expected {expected_shape}."
        )
    if np.any(actions < 0) or np.any(actions >= reward_model.n_actions):
        raise ValueError("SAC policy returned an action outside the environment range.")

    # 潜变量采样 + sigmoid 有界映射在 ensemble 内部完成；这里拿到 block reward
    # 与 elite 间 disagreement（模型不确定性诊断）。
    predicted_rewards, prediction_stats = reward_model.sample_rewards(
        state_imgs,
        hoprates,
        actions,
        deterministic=deterministic_model,
    )
    if timing_enabled:
        timing["predict_s"] = time.time() - stage_start
        stage_start = time.time()
    if predicted_rewards.shape != expected_shape or not np.all(
        np.isfinite(predicted_rewards)
    ):
        raise RuntimeError("Reward ensemble returned invalid block rewards.")
    # 用 ensemble 自身的 BER 端点（若存在）复算物理上下界；容差吸收 sigmoid 浮点误差。
    lower_bounds, upper_bounds = reward_bounds(
        hoprates,
        reward_config,
        ber_min=float(getattr(reward_model, "ber_min", DEFAULT_BER_MIN)),
        ber_max=float(getattr(reward_model, "ber_max", DEFAULT_BER_MAX)),
    )
    tolerance = 1e-6 * np.maximum(1.0, upper_bounds - lower_bounds)
    if np.any(predicted_rewards < lower_bounds - tolerance) or np.any(
        predicted_rewards > upper_bounds + tolerance
    ):
        raise RuntimeError("Reward ensemble returned a reward outside its bounds.")
    # 统一为 float32，与 real replay / SAC 契约一致（已在界内，无需再裁剪）。
    predicted_rewards = predicted_rewards.astype(np.float32, copy=False)

    # 一次性批量写入 model replay：next_* 与 dones 直接复用起点（外生），
    # 只有 block_rewards 是模型产物；deque(maxlen=capacity) 自动 FIFO 淘汰。
    model_buffer.add_batch(
        state_imgs,
        hoprates,
        actions,
        predicted_rewards,
        starts["next_state_imgs"],
        starts["next_hoprates"],
        starts["dones"],
    )
    if timing_enabled:
        timing["add_s"] = time.time() - stage_start
        timing["total_s"] = sum(timing.values())

    # 由写入前后 size 差反推被淘汰的旧样本数（capacity 足够时为 0）。
    model_buffer_size_after = model_buffer.size()
    fifo_evicted = max(
        0,
        model_buffer_size_before + rollout_size - model_buffer_size_after,
    )
    disagreement = np.asarray(
        prediction_stats["disagreement"], dtype=np.float32
    )
    # 统计量只用于日志/曲线，不参与训练逻辑；timing 在关闭计时时为 None。
    return {
        "generated": rollout_size,
        "model_buffer_size_before": model_buffer_size_before,
        "model_buffer_size_after": model_buffer_size_after,
        "model_buffer_capacity": model_buffer.capacity,
        "fifo_evicted": fifo_evicted,
        "reward_mean": float(np.mean(predicted_rewards)),
        "reward_std": float(np.std(predicted_rewards)),
        "disagreement_mean": float(np.mean(disagreement)),
        "disagreement_p95": float(np.percentile(disagreement, 95)),
        "timing": (timing if timing_enabled else None),
    }
