"""v3 step-level 离线 replay 的序列化、校验与加载工具。

格式契约（FORMAT_VERSION = 3）
-----------------------------
一条 transition 表示一次**完整环境 step**（而非"一个 block 的一次决策"），
文件为压缩 ``.npz``，含 8 个数组字段：

    state_imgs      [N, H, W]  动作前的 PSD waterfall 观测（100 ms 窗口）
    hoprates        [N]        本 step 实际生效的跳速（Hz，已按 10 Hz 网格量化）
    actions         [N, 10]    10 个 block 各自的起始信道 offset
    block_rewards   [N, 10]    10 个 block 的逐块奖励
    step_rewards    [N]        = mean(block_rewards[n])，冗余字段，便于诊断/校验
    next_state_imgs [N, H, W]  动作后的 PSD 观测
    next_hoprates   [N]        下一 step 交给 offset 策略的 hoprate
    dones           [N]        本 step 是否 terminated/truncated

保存与加载两侧共同强制的不变量：
    - 所有字段第一维长度一致且 > 0（无空文件、无字段错位）；
    - step_rewards 必须等于 mean(block_rewards, axis=1)（rtol=1e-5、atol=1e-6）；
    - state_imgs 与 next_state_imgs 形状相同，且与当前环境的观测形状一致；
    - actions 必须为整数值、非负、且落在 [0, n_actions)；
    - 除 actions 外的数值字段必须全为有限值，dones 只允许布尔/0/1；
    - metadata 中的 num_step_transitions / num_blocks / num_actions 必须与
      数组内容及当前环境一致；总条数不得超过目标 ReplayBuffer 的容量。

随机采样下的因果一致性：采集非终止 transition 时，next_hoprates[n] 就是下一
step 真正喂给策略的 hoprate（即 hoprates[n+1]），所以每条 transition 内部
(state, hoprate, action, reward, next_state, next_hoprate, done) 自洽；训练侧
逐条独立采样，不依赖跨条相邻性。

为什么要校验 metadata：replay 本质是"配置快照"。环境参数、干扰机时序/模式、
奖励系数任一变化，都会让旧数据在物理含义上失效（同一张 PSD 图对应的 BER 与
reward 都会不同）。因此文件内固化 env_config / jammer_config / reward_config
三份快照，训练入口据此选择直接拒绝（strict）还是仅告警（兼容模式）。修改
settings.py 中相关配置后应重新运行 generate_offline_replay.py，而不是复用旧文件。

版本策略：v1/v2 是 block-level 格式（每条 transition 带 block_idx，一次决策一条），
无法表达"同一状态下十头动作联合采样"这一 step-level 事实，因此**直接拒绝而不做
隐式转换**；缺少 format_version 的文件同样拒绝，并提示用 v3 重新生成。

相关文件：
    - generate_offline_replay.py：以随机 offset 策略采集并写出 v3 文件；
    - subset_offline_replay.py：从已有 v3 文件抽取前 N 条或种子随机子集；
    - train_offsets.py / train_mbpo.py：经 load_replay_into_buffer 冷启动 real
      replay（MBPO 默认 strict，``--allow_replay_config_mismatch`` 可放宽）；
    - SAC.ReplayBuffer：加载目标，单条记录为 8 元定序元组。
"""

import json
import logging
import os

import numpy as np


# 当前支持的 replay 格式版本；3 = step-level。v1/v2 为 block-level，加载时拒绝。
FORMAT_VERSION = 3
# 文件必须包含的字段名集合（也是加载时的读取顺序与校验范围）。
# 注意：字段顺序是 SAC.ReplayBuffer._batch_from_transitions 的输出键序，
# 数组第一维始终是 transition 索引 N。
REPLAY_KEYS = (
    "state_imgs",       # [N, H, W] float32，动作前 PSD 观测
    "hoprates",         # [N] float32，本 step 生效的跳速（Hz）
    "actions",          # [N, 10] int64，10 个 block 的起始信道 offset
    "block_rewards",    # [N, 10] float32，逐 block 奖励
    "step_rewards",     # [N] float32，= mean(block_rewards, axis=1)
    "next_state_imgs",  # [N, H, W] float32，动作后 PSD 观测
    "next_hoprates",    # [N] float32，下一 step 的策略输入 hoprate
    "dones",            # [N] bool/0-1，本 step 是否终止
)


def _as_jsonable(value):
    """把任意嵌套的配置对象递归转成可被 ``json.dumps`` 序列化的纯 Python 类型。

    settings.py 的配置里混有 numpy 标量（如 ``np.float64``）与 ndarray，直接
    ``json.dumps`` 会抛 TypeError。这里统一降级：dict → 键转 str 的 dict、
    list/tuple → list、ndarray → list、numpy 标量 → Python 标量；其余类型原样
    返回（假定本身可序列化，如 int/float/str/bool/None）。

    不变量：转换只改变容器/标量类型，不改变数值，因此"当前配置"与"文件内
    metadata 快照"两边用同一函数处理后可直接做 ``==`` 比较（metadata 一致性
    校验依赖这一点）。

    Args:
        value: 任意配置值（通常是 dict 或嵌套结构）。

    Returns:
        与输入结构等价的 JSON 友好对象。
    """
    if isinstance(value, dict):
        return {str(key): _as_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_as_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def environment_metadata(env_config, jammer_config, reward_config):
    """打包环境/干扰机/奖励三份配置快照，作为 replay 文件的一致性凭据。

    三个键名是 metadata 契约的一部分：加载侧（``load_replay_into_buffer``）按
    固定键名取出并与当前运行配置逐项比较，因此这里不做重命名或裁剪；配置内容
    经 ``_as_jsonable`` 归一化，保证与运行时配置可直接比较。

    Args:
        env_config: 环境配置（settings.ENV_CONFIG 或实例级覆盖后的副本）。
        jammer_config: 干扰机配置（settings.JAMMER_CONFIG 或覆盖后的副本）。
        reward_config: 奖励系数配置（settings.REWARD_CONFIG）。

    Returns:
        dict，含 ``env_config`` / ``jammer_config`` / ``reward_config`` 三个键，
        值为 JSON 友好的配置快照。
    """
    return {
        "env_config": _as_jsonable(env_config),
        "jammer_config": _as_jsonable(jammer_config),
        "reward_config": _as_jsonable(reward_config),
    }


def _buffer_to_arrays(buffer):
    """把 ReplayBuffer 的全池快照按 REPLAY_KEYS 转成 numpy 数组字典。

    只取 ``REPLAY_KEYS`` 中的 8 个字段并丢弃其余键，保证落盘字段集与加载侧
    期望严格一致；空 buffer 无法表达任何 transition，直接报错而不是写出空文件。

    Args:
        buffer: SAC.ReplayBuffer（或提供 size()/get_all() 的同构实现）。

    Returns:
        dict，键为 REPLAY_KEYS，值为 numpy 数组（batch 维在前，长度均为 N）。

    Raises:
        ValueError: buffer 为空（size() == 0）。
    """
    if buffer.size() == 0:
        raise ValueError("Cannot save an empty replay buffer.")
    arrays = buffer.get_all()
    return {key: np.asarray(arrays[key]) for key in REPLAY_KEYS}


def save_replay_buffer(path, buffer, metadata=None):
    """把 step-level ReplayBuffer 序列化为 v3 压缩 ``.npz`` 文件。

    写出前先做一次"保存侧自检"，把错误尽量拦在生成阶段：
        - 由 ``actions`` 的二维形状推得 ``N`` 与 ``num_blocks``；
        - 强制 ``step_rewards == mean(block_rewards, axis=1)``（与加载侧同一容差），
          避免写出训练时才发现不一致的冗余字段。

    随后在调用方 metadata 之上补齐由数据推得的描述性字段（调用方同名键会被
    覆盖，因为这些值必须与数组一致）：``format_version``、``num_step_transitions``、
    ``observation_shape``（不含 batch 维）、``num_actions_observed``（数据中出现过的
    最大动作值 + 1，仅供审计；加载侧读的是生成脚本显式写入的 ``num_actions``）、
    ``num_blocks``。metadata 以 JSON 字符串（键排序）存入同一 npz 的 ``metadata`` 数组。

    Args:
        path: 输出路径；父目录不存在时自动创建。
        buffer: 待保存的 ReplayBuffer（非空）。
        metadata: 额外 metadata 字典，如 environment_metadata(...) 的结果，可选。

    Returns:
        None。

    Raises:
        ValueError: buffer 为空，或 step_rewards 与 block_rewards 的均值不一致。
    """
    arrays = _buffer_to_arrays(buffer)
    # actions 是 [N, num_blocks] 二维数组，两个维度分别给出 transition 数与 block 数。
    num_step_transitions, num_blocks = arrays["actions"].shape
    expected_step_rewards = arrays["block_rewards"].mean(axis=1)
    if not np.allclose(
        arrays["step_rewards"], expected_step_rewards, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("step_rewards must equal mean(block_rewards) before saving.")

    # 复制一份，避免原地修改调用方传入的字典；下列派生字段一律以数组为准。
    metadata = dict(metadata or {})
    metadata.update(
        {
            "format_version": FORMAT_VERSION,
            "num_step_transitions": int(num_step_transitions),
            "observation_shape": list(arrays["state_imgs"].shape[1:]),
            "num_actions_observed": int(np.max(arrays["actions"])) + 1,
            "num_blocks": int(num_blocks),
        }
    )
    # np.savez_compressed 不会创建父目录，这里显式补上（dirname 可能为空串）。
    output_dir = os.path.dirname(os.path.abspath(path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # metadata 以单个 JSON 字符串数组落盘（键排序保证同内容文件字节可比），
    # 与 8 个数据字段共存于同一 npz；加载时用 allow_pickle=False 读取。
    np.savez_compressed(
        path,
        **arrays,
        metadata=np.asarray(json.dumps(_as_jsonable(metadata), sort_keys=True)),
    )


def _validate_format_version(metadata):
    """校验文件 metadata 的 ``format_version``，只接受当前 v3。

    三条规则（任一不满足即抛 ValueError，且错误信息都指向"用
    generate_offline_replay.py 重新生成"这一修复路径）：
        1. 键缺失 → 视为未知/过旧文件，拒绝（不猜测版本）；
        2. 值不能转成 int（如字符串垃圾值）→ 拒绝；
        3. 可解析但不等于 FORMAT_VERSION → 拒绝：v1/v2 是 block-level 格式，
           其 transition 语义与 step-level SAC 不兼容，不做隐式转换。

    用 ``int()`` 解析是刻意的：JSON 中版本号可能被写成 ``3.0`` 或 ``"3"``，
    这些等价写法都应接受；非整数值（如 ``3.5``）在 int() 后被截断的极端情况
    只可能来自手工改写的文件，不影响正常数据。

    Args:
        metadata: 已解析的 metadata dict。

    Returns:
        None（校验通过即正常返回）。

    Raises:
        ValueError: format_version 缺失、无法解析，或不是当前支持的版本。
    """
    file_version = metadata.get("format_version")
    if file_version is None:
        raise ValueError(
            "Offline replay metadata is missing format_version; regenerate it "
            "with generate_offline_replay.py using format v3."
        )
    try:
        parsed_version = int(file_version)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Offline replay format_version is invalid: {file_version!r}."
        ) from exc
    if parsed_version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported offline replay format version: {parsed_version}; "
            "block-level v1/v2 data cannot be loaded by the step-level SAC. "
            "Regenerate the dataset with generate_offline_replay.py."
        )


def _load_arrays(path):
    """读取 v3 ``.npz`` 文件，返回（字段数组字典, metadata dict）。

    只做"文件级"读取与校验，不涉及当前环境配置（形状/动作范围/容量等由
    ``_validate_array_shapes`` 与 ``load_replay_into_buffer`` 负责）：

        - 以 ``allow_pickle=False`` 打开：本格式只含数值数组与一个 JSON 字符串，
          禁止 pickle 可避免执行不可信文件中的对象；
        - metadata 数组缺失时按 ``"{}"`` 处理，交给 ``_validate_format_version``
          报出"缺少 format_version"这一更明确的错误；
        - metadata 必须是合法 JSON（用 ``str(...item())`` 兼容 0 维字符串数组）；
        - 先校验 format_version，再检查 REPLAY_KEYS 是否齐全——顺序是刻意的：
          版本不对时应给出"重新生成"的提示，而不是先报字段缺失；
        - 读取完立即关闭 archive（``with`` 块），返回值是已实体化的数组副本。

    Args:
        path: v3 replay ``.npz`` 文件路径。

    Returns:
        tuple(dict, dict)：``{字段名: np.ndarray}``（仅 REPLAY_KEYS 中的键）与
        解析后的 metadata。

    Raises:
        FileNotFoundError: 文件不存在（错误信息提示改用 ``--offline_replay_path none``）。
        ValueError: 文件无法读取、metadata 不是合法 JSON、format_version 不受支持，
            或缺少任一必需字段。
    """
    try:
        archive = np.load(path, allow_pickle=False)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Offline replay file does not exist: {path}. Generate a v3 file with "
            "generate_offline_replay.py or use --offline_replay_path none."
        ) from exc
    except Exception as exc:
        raise ValueError(f"Could not read offline replay file '{path}': {exc}") from exc

    with archive:
        # 缺失 metadata 时用 "{}" 占位，让版本校验报出明确的"缺少 format_version"。
        metadata_value = (
            archive["metadata"] if "metadata" in archive else np.asarray("{}")
        )
        try:
            metadata = json.loads(str(metadata_value.item()))
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError("Offline replay metadata is not valid JSON.") from exc
        # 先定版本，再看字段：v1/v2 文件缺字段属预期，应给出重新生成提示。
        _validate_format_version(metadata)

        missing = [key for key in REPLAY_KEYS if key not in archive]
        if missing:
            raise ValueError(f"Offline replay is missing fields: {missing}")
        # 只取 REPLAY_KEYS，忽略文件里的其他键（如 metadata 之外的历史字段）。
        arrays = {key: np.asarray(archive[key]) for key in REPLAY_KEYS}
    return arrays, metadata


def _validate_array_shapes(arrays, expected_num_blocks):
    """校验字段间的形状自洽性（不涉及当前环境配置），返回 transition 条数 N。

    规则：
        1. 所有字段第一维长度必须一致——v3 契约要求逐条对齐，长度不同意味着
           文件被拼接/截断或字段写错，继续训练会产生错位的 (s, a, r, s') 组合；
        2. 长度不能为 0（空 replay 无法冷启动）；
        3. 向量字段 ``hoprates`` / ``step_rewards`` / ``next_hoprates`` / ``dones``
           必须恰为 ``(N,)``（防止把 [N,1] 之类的列向量当成标量序列）；
        4. 矩阵字段 ``actions`` / ``block_rewards`` 必须恰为
           ``(N, expected_num_blocks)``——第二个维度即每步的 offset 头数，必须与
           当前环境的 block 数一致；
        5. 图像字段 ``state_imgs`` / ``next_state_imgs`` 的形状留给调用方校验
           （需要与当前环境观测形状比对），此处只通过长度一致性间接约束。

    Args:
        arrays: ``{字段名: np.ndarray}``（来自 ``_load_arrays``）。
        expected_num_blocks: 当前环境的 block 数（int 或可转 int）。

    Returns:
        int：transition 条数 N。

    Raises:
        ValueError: 字段长度不一致、条数为 0，或任一字段形状不符合上述规则。
    """
    lengths = {key: len(value) for key, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"Offline replay fields have inconsistent lengths: {lengths}")
    count = next(iter(lengths.values()))
    if count == 0:
        raise ValueError("Offline replay contains no transitions.")

    # 向量字段：每个 transition 一个标量，形状必须严格等于 (N,)。
    vector_keys = ("hoprates", "step_rewards", "next_hoprates", "dones")
    for key in vector_keys:
        if arrays[key].shape != (count,):
            raise ValueError(
                f"Offline replay {key} must have shape ({count},), "
                f"got {arrays[key].shape}."
            )

    # 矩阵字段：每个 transition 一组 per-block 值，形状为 (N, num_blocks)。
    matrix_shape = (count, expected_num_blocks)
    for key in ("actions", "block_rewards"):
        if arrays[key].shape != matrix_shape:
            raise ValueError(
                f"Offline replay {key} must have shape {matrix_shape}, "
                f"got {arrays[key].shape}."
            )
    return count


def load_replay_into_buffer(
    path,
    buffer,
    expected_observation_shape=None,
    expected_num_actions=None,
    expected_num_blocks=10,
    current_environment_metadata=None,
    logger=None,
    strict_environment_metadata=False,
):
    """校验 v3 离线 replay 文件并逐条追加到 ReplayBuffer。

    ``strict_environment_metadata`` 是显式 opt-in：baseline（train_offsets.py）不传，
    保持"配置不一致只告警"的历史兼容行为；MBPO 入口显式传 True，把不一致直接
    升级为拒绝（``--allow_replay_config_mismatch`` 可再放宽）。

    校验规则（顺序即失败优先级，全部通过后才会写入 buffer，避免"写一半失败"
    留下部分加载的脏 replay）：
        1. 文件级：存在、可读、metadata 为合法 JSON、format_version == 3、
           REPLAY_KEYS 齐全（见 ``_load_arrays``）；
        2. 形状自洽：字段长度一致且非空、向量/矩阵字段形状正确
           （见 ``_validate_array_shapes``）；
        3. 观测形状：state_imgs 与 next_state_imgs 的空间形状相同；给定
           ``expected_observation_shape`` 时还须与环境当前观测形状一致
           （不含 batch 维）；
        4. 有限值：除 actions 外的数值字段（图像、hoprate、block/step reward、
           next_*）必须全为有限值，防止 NaN/Inf 污染 replay 与后续拟合；
        5. 动作：整数值（``np.rint`` 后 allclose）、非负；给定
           ``expected_num_actions`` 时须 < n_actions，且文件 metadata 的
           ``num_actions``（若存在）须与之一致；
        6. 奖励冗余：step_rewards == mean(block_rewards, axis=1)（同一容差）；
        7. dones 只允许布尔/0/1；
        8. metadata 一致性：``num_blocks`` 须等于 ``expected_num_blocks``；
           ``num_step_transitions``（若存在）须等于实际条数；
        9. 容量：条数不得超过 ``buffer.capacity``，且
           ``buffer.num_heads`` 须与 block 数一致（防止十头 buffer 装入
           其他头数的数据）；
       10. 配置快照：``current_environment_metadata`` 给定时，比较文件内
           env_config / jammer_config / reward_config 与当前配置——
           strict 模式下任一键缺失、为 None 或取值不同即拒绝；
           非 strict 模式下仅当文件侧存在该键且取值不同时打 warning。

    通过后按文件顺序逐条调用 ``buffer.add``（buffer 自身会再做一次逐条校验并
    copy 数据，因此这里不需要额外防御性拷贝）；返回条数与 metadata 供调用方
    记录日志（如 ``hoprate_mode``）。

    Args:
        path: v3 replay ``.npz`` 路径。
        buffer: 目标 SAC.ReplayBuffer（需提供 capacity / num_heads / add）。
        expected_observation_shape: 当前环境观测形状（不含 batch 维），可选；
            给定时校验文件观测形状。
        expected_num_actions: 当前环境每个头的动作数，可选；给定时校验动作上界
            与 metadata ``num_actions``。
        expected_num_blocks: 当前环境 block 数（默认 10），同时用于矩阵字段形状
            与 metadata ``num_blocks`` 校验。
        current_environment_metadata: ``environment_metadata(...)`` 的当前配置快照，
            可选；给定时才做配置一致性校验。
        logger: 可选 logger；非 strict 模式下的不一致 warning 用它输出，
            缺省时回退到本模块 logger。
        strict_environment_metadata: True 时配置不一致直接抛错；默认 False
            保持告警行为。

    Returns:
        tuple(int, dict)：(实际加载的 transition 条数, 文件 metadata)。

    Raises:
        FileNotFoundError: 文件不存在。
        ValueError: 上述任一校验规则失败。
    """
    arrays, metadata = _load_arrays(path)
    expected_num_blocks = int(expected_num_blocks)
    count = _validate_array_shapes(arrays, expected_num_blocks)

    state_shape = tuple(arrays["state_imgs"].shape[1:])
    next_state_shape = tuple(arrays["next_state_imgs"].shape[1:])
    if state_shape != next_state_shape:
        raise ValueError(
            f"State and next-state shapes differ: {state_shape} vs {next_state_shape}."
        )
    if (
        expected_observation_shape is not None
        and state_shape != tuple(expected_observation_shape)
    ):
        raise ValueError(
            "Offline replay observation shape does not match the current environment: "
            f"file={state_shape}, environment={tuple(expected_observation_shape)}."
        )

    for key in (
        "state_imgs",
        "hoprates",
        "block_rewards",
        "step_rewards",
        "next_state_imgs",
        "next_hoprates",
        "dones",
    ):
        # actions 单独处理（下面会先取整再查上界），其余数值字段一律要求有限。
        if not np.all(np.isfinite(arrays[key])):
            raise ValueError(f"Offline replay contains non-finite values in {key}.")

    # 允许以 float 存储的 offset（生成侧就是这样写的），但必须是精确整数值。
    raw_actions = arrays["actions"]
    rounded_actions = np.rint(raw_actions)
    if not np.allclose(raw_actions, rounded_actions):
        raise ValueError("Offline replay actions must be integer-valued.")
    actions = rounded_actions.astype(np.int64)
    if np.any(actions < 0):
        raise ValueError("Offline replay actions must be non-negative.")
    if expected_num_actions is not None:
        # 双重校验：数组取值必须在 [0, n_actions)，metadata 记录的动作空间大小
        # 也必须与当前环境一致（前者防越界，后者防"同范围不同语义"的错配）。
        expected_num_actions = int(expected_num_actions)
        if np.any(actions >= expected_num_actions):
            raise ValueError(
                "Offline replay contains actions outside "
                f"[0, {expected_num_actions - 1}]."
            )
        file_num_actions = metadata.get("num_actions")
        if (
            file_num_actions is not None
            and int(file_num_actions) != expected_num_actions
        ):
            raise ValueError(
                "Offline replay action-space size does not match the environment: "
                f"file={file_num_actions}, environment={expected_num_actions}."
            )

    # step_rewards 是冗余字段：必须与 block_rewards 的逐行均值一致（保存侧同规则）。
    expected_step_rewards = arrays["block_rewards"].mean(axis=1)
    if not np.allclose(
        arrays["step_rewards"], expected_step_rewards, rtol=1e-5, atol=1e-6
    ):
        raise ValueError("Offline replay step_rewards must equal mean(block_rewards).")

    # dones 在 v3 里是布尔语义；用 isin 同时接受 bool 与 0/1（含 float32 的 0.0/1.0）。
    if not np.all(np.isin(arrays["dones"], (0, 1, False, True))):
        raise ValueError("Offline replay dones must contain only boolean values.")
    if int(metadata.get("num_blocks", expected_num_blocks)) != expected_num_blocks:
        raise ValueError("Offline replay was generated with a different block count.")
    metadata_count = metadata.get("num_step_transitions")
    if metadata_count is not None and int(metadata_count) != count:
        raise ValueError(
            "Offline replay transition count does not match metadata: "
            f"arrays={count}, metadata={metadata_count}."
        )
    # 容量校验放在写入之前：ReplayBuffer 是 FIFO，超容量会静默丢最旧数据，
    # 那样"加载了 N 条"的假设就不成立，必须显式报错。
    if count > buffer.capacity:
        raise ValueError(
            f"Offline replay has {count} transitions but replay capacity is only "
            f"{buffer.capacity}."
        )
    if getattr(buffer, "num_heads", expected_num_blocks) != expected_num_blocks:
        raise ValueError("Replay buffer and offline file use different block counts.")

    if current_environment_metadata is not None:
        # 只比较三个固定配置键；文件里的其他 metadata（生成参数、subset_* 等）不参与校验。
        stored_config = {
            key: metadata.get(key)
            for key in ("env_config", "jammer_config", "reward_config")
        }
        current_config = _as_jsonable(current_environment_metadata)
        if strict_environment_metadata:
            # strict：任一配置键缺失/为 None/取值不同都拒绝（跨配置复用必须显式放开）。
            mismatched_keys = [
                key
                for key in current_config
                if key not in stored_config
                or stored_config.get(key) is None
                or current_config.get(key) != stored_config.get(key)
            ]
            if mismatched_keys:
                raise ValueError(
                    "Offline replay environment metadata does not match the "
                    f"current settings for: {mismatched_keys}."
                )
        elif any(
            # 非 strict（baseline 兼容路径）：文件侧有记录且与当前配置不同才告警。
            stored_config.get(key) is not None
            and current_config.get(key) != stored_config.get(key)
            for key in current_config
        ):
            (logger or logging.getLogger(__name__)).warning(
                "Offline replay environment configuration differs from the current "
                "settings; use a dataset generated for the intended environment."
            )

    # 逐条写入：保持文件内的原始顺序（get_all 也按插入顺序返回），便于复现；
    # buffer.add 内部会做逐条校验并 copy，无需在此重复防御。
    for idx in range(count):
        buffer.add(
            arrays["state_imgs"][idx],
            arrays["hoprates"][idx],
            actions[idx],
            arrays["block_rewards"][idx],
            arrays["next_state_imgs"][idx],
            arrays["next_hoprates"][idx],
            bool(arrays["dones"][idx]),
        )

    return count, metadata
