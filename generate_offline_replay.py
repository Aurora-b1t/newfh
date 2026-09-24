"""离线真实环境 replay 生成脚本（v3 step-level 格式）。

职责
----
在真实 FHSS/QPSK 环境中用 **随机 offset 策略**（不训练任何 agent）滚动采集
数据，把每条完整环境 step 序列化为 v3 step-level replay ``.npz`` 文件，供
``train_offsets.py`` / ``train_mbpo.py`` 在首次梯度更新前冷启动（warm start）。

一条 transition 的语义（与训练脚本完全一致）::

    state_img    : 100 ms PSD waterfall 观测（动作前），形状 [100, 100]
    hoprate      : 本 step 实际生效的跳速（Hz，已按 10 Hz 网格量化）
    actions      : 10 个 block 各自的起始信道 offset，形状 [10]
    block_rewards: 10 个 block 的逐块奖励，形状 [10]
    next_state_img / next_hoprate : 下一 step 的观测与跳速
    done         : 本 step 是否触发 terminated/truncated

hoprate 采样支持两种模式：
- ``fixed``  : 全程固定一个 hoprate（默认 100 Hz，与 baseline 训练一致）；
- ``random`` : 每条 step 从环境的合法 10 Hz 网格中均匀随机抽取。

生成的文件带有环境 / 干扰机 / 奖励 metadata；训练入口默认会严格校验
metadata 与当前配置一致，跨配置实验需显式 ``--allow_replay_config_mismatch``。
因此修改 ``settings.py`` 中的环境或干扰配置后，应重新生成本文件。

用法示例::

    python generate_offline_replay.py --num_step_transitions 5000 \
        --hoprate_mode fixed --fixed_hoprate 100 \
        --output_path outputs/offline_replay/replay_5000_100_hoprate_v3.npz

注意：本文件原先从已删除的 ``joint_training.py`` 导入环境覆盖参数工具，
现已在本地内联实现（``str_to_bool`` / ``add_environment_override_args`` /
``resolve_environment_configs``），保持行为不变。
"""

import argparse
import copy
import logging

import numpy as np

from fh_env import FHSSQPSKEnv
from SAC import ReplayBuffer
import settings
from offline_replay import environment_metadata, save_replay_buffer

# 无差别干扰机（sweep/comb）支持的模式，与 jammers.IndiscriminateJammer 一致。
JAMMER_MODES = ("comb", "sweep", "both")


def str_to_bool(value):
    """把命令行字符串解析为布尔值。

    接受 ``true/1/yes/on``（不区分大小写）及其否定形式；传入已经是
    bool 的值则原样返回，便于代码内部复用。非法取值抛 ``ValueError``。
    """
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def add_environment_override_args(parser):
    """向 argparse parser 追加实例级环境/干扰机覆盖参数。

    包括 ``--enable_reactive``、``--enable_sweep`` 和 ``--jammer_mode``，
    默认值取自 ``settings.ENV_CONFIG`` / ``settings.JAMMER_CONFIG``，即不传参
    时与直接读取 settings 的行为完全一致。
    """
    parser.add_argument(
        "--enable_reactive",
        type=str_to_bool,
        default=settings.ENV_CONFIG["enable_reactive"],
        help="Enable the reactive jammer (default: settings.ENV_CONFIG).",
    )
    parser.add_argument(
        "--enable_sweep",
        type=str_to_bool,
        default=settings.ENV_CONFIG["enable_sweep"],
        help="Enable the indiscriminate sweep/comb jammer.",
    )
    parser.add_argument(
        "--jammer_mode",
        choices=JAMMER_MODES,
        default=settings.JAMMER_CONFIG["mode"],
        help="Indiscriminate jammer mode when --enable_sweep is true.",
    )
    return parser


def resolve_environment_configs(args):
    """根据命令行覆盖项返回生效的环境/干扰机配置副本。

    深拷贝 ``settings.ENV_CONFIG`` 与 ``settings.JAMMER_CONFIG`` 后应用
    ``--enable_reactive`` / ``--enable_sweep`` / ``--jammer_mode`` 覆盖，
    不污染全局 settings，因此多次构建环境时互不影响。
    """
    env_config = copy.deepcopy(settings.ENV_CONFIG)
    jammer_config = copy.deepcopy(settings.JAMMER_CONFIG)
    env_config["enable_reactive"] = bool(args.enable_reactive)
    env_config["enable_sweep"] = bool(args.enable_sweep)
    jammer_config["mode"] = str(args.jammer_mode)
    return env_config, jammer_config


def quantize_hoprate(hoprate, env):
    """把任意 hoprate 量化到环境支持的 10 Hz 网格并夹到合法范围。

    与 ``FHSSQPSKEnv._apply_hoprate()`` 的量化规则一致：先按
    ``[hoprate_min, hoprate_max]`` 截断，再四舍五入到最近的 10 Hz 倍数。
    """
    clipped = np.clip(hoprate, env.hoprate_min, env.hoprate_max)
    return float(int(round(clipped / 10.0)) * 10)


def make_hoprate_sampler(mode, fixed_hoprate, env, rng):
    """构造每条 step 的 hoprate 采样器。

    Args:
        mode: ``"fixed"`` 固定跳速；``"random"`` 在合法 10 Hz 网格内均匀随机。
        fixed_hoprate: ``mode="fixed"`` 时使用的跳速（会被量化）。
        env: 用于读取 ``hoprate_min`` / ``hoprate_max`` 的环境实例。
        rng: ``numpy.random.Generator``，random 模式下的随机源。

    Returns:
        无参 callable，每次调用返回一个 float hoprate（Hz）。

    Raises:
        ValueError: mode 非法，或环境跳速范围内没有任何合法 10 Hz 值。
    """
    if mode == "fixed":
        quantized = quantize_hoprate(float(fixed_hoprate), env)
        return lambda: quantized

    if mode != "random":
        raise ValueError(f"Unsupported hoprate_mode: {mode!r}.")

    min_step = int(np.ceil(env.hoprate_min / 10.0))
    max_step = int(np.floor(env.hoprate_max / 10.0))
    if min_step > max_step:
        raise ValueError("Environment hoprate range has no valid 10 Hz value.")
    valid_rates = np.arange(min_step, max_step + 1, dtype=np.int32) * 10
    return lambda: float(rng.choice(valid_rates))


def generate(args):
    """主流程：在真实环境中随机滚动并保存 v3 step-level replay。

    每次循环对应一条完整环境 step：随机抽 10 个 offset，环境执行
    10 个通信 block，校验返回的 block 数量与均值奖励一致性后写入
    ``SAC.ReplayBuffer``，最后经 ``offline_replay.save_replay_buffer``
    序列化为带 metadata 的 ``.npz``。episode 结束（done）时自动 reset。
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    settings.set_random_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    env_config, jammer_config = resolve_environment_configs(args)
    env = FHSSQPSKEnv(**env_config, jammer_config=jammer_config)
    state_img, _info = env.reset()
    n_actions = env.num_channels
    num_blocks = env.num_blocks
    buffer = ReplayBuffer(
        capacity=args.num_step_transitions,
        num_heads=num_blocks,
        n_actions=n_actions,
    )
    hoprate_sampler = make_hoprate_sampler(
        args.hoprate_mode, args.fixed_hoprate, env, rng
    )

    current_hoprate = hoprate_sampler()
    while buffer.size() < args.num_step_transitions:
        # 随机 offset 策略：每个 block 从 [0, num_channels) 均匀抽取起始信道。
        offsets = rng.integers(
            0, n_actions, size=num_blocks, dtype=np.int64
        )
        next_state_img, step_reward, terminated, truncated, info = env.step(
            {"hoprate": current_hoprate, "offsets": offsets}
        )
        used_hoprate = float(info.get("hoprate_used", current_hoprate))
        # 下一 step 的 hoprate 与动作无关（外生变量），直接预采样并写入本条
        # transition 的 next_hoprate，训练时据此重建 (state, hoprate) 输入。
        next_hoprate = hoprate_sampler()
        block_rewards = np.asarray(info.get("block_rewards", []), dtype=np.float32)
        if block_rewards.shape != (num_blocks,):
            raise RuntimeError(
                "Environment did not return one reward for each offset block."
            )
        if not np.isclose(
            float(step_reward), float(np.mean(block_rewards)), rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("Environment step reward is inconsistent with block rewards.")

        done = bool(terminated or truncated)
        buffer.add(
            state_img,
            used_hoprate,
            offsets,
            block_rewards,
            next_state_img,
            next_hoprate,
            done,
        )
        state_img = next_state_img
        current_hoprate = next_hoprate

        if done:
            state_img, _info = env.reset()
            current_hoprate = hoprate_sampler()

        if buffer.size() % 100 == 0 or buffer.size() == args.num_step_transitions:
            logging.info(
                "Collected %d/%d step transitions",
                buffer.size(),
                args.num_step_transitions,
            )

    # metadata 记录环境/干扰/奖励配置，训练入口据此做严格一致性校验。
    metadata = environment_metadata(
        env_config,
        jammer_config,
        settings.REWARD_CONFIG,
    )
    metadata.update(
        {
            "generator": "generate_offline_replay.py",
            "seed": args.seed,
            "hoprate_mode": args.hoprate_mode,
            "fixed_hoprate": (
                args.fixed_hoprate if args.hoprate_mode == "fixed" else None
            ),
            "num_env_steps": buffer.size(),
            "num_actions": n_actions,
            "num_blocks": num_blocks,
            "hoprate_grid_step": 10.0,
        }
    )
    save_replay_buffer(args.output_path, buffer, metadata)
    logging.info(
        "Saved %d step transitions to %s",
        buffer.size(),
        args.output_path,
    )
    env.close()


def parse_args(argv=None):
    """解析命令行参数；``argv=None`` 时读取 ``sys.argv``（便于测试注入）。"""
    parser = argparse.ArgumentParser(
        description="Generate real FHSS step-level replay with random offsets."
    )
    parser.add_argument(
        "--num_step_transitions",
        type=int,
        default=settings.OFFLINE_REPLAY_CONFIG["num_step_transitions"],
        help="Number of complete environment-step transitions to save.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Output v3 .npz path.",
    )
    parser.add_argument(
        "--hoprate_mode",
        choices=("random", "fixed"),
        default=settings.OFFLINE_REPLAY_CONFIG["hoprate_mode"],
        help="Use uniformly random valid hoprates or one fixed hoprate.",
    )
    parser.add_argument(
        "--fixed_hoprate",
        type=float,
        default=settings.TRAIN_CONFIG["fixed_hoprate"],
        help="Hoprate used when --hoprate_mode=fixed.",
    )
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    add_environment_override_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate(parse_args())
