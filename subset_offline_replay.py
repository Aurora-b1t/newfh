"""从已有 v3 离线 replay 文件抽取一个更小的子集文件。

用途
----
训练入口通过 ``--offline_replay_path`` 指定的文件会被**整份**加载进 real replay
（见 offline_replay.load_replay_into_buffer），而 smoke test、消融或小规模快速
实验并不需要全部数据。本脚本在不改动任何训练代码的前提下，从现有 v3 文件抽出
前 N 条或种子随机子集，写成同样合法的 v3 文件，训练时换一个路径即可。

抽取语义
--------
- 默认取**前 N 条**（``--count``）：保持原文件采集顺序，输出是原文件的前缀；
- ``--shuffle`` 时按 ``--seed`` 做**无放回**均匀随机抽样，并对抽出的行号排序，
  使输出仍是原文件顺序的子序列（便于与其他子集逐行对齐）；``--seed`` 缺省取
  settings.RANDOM_SEED，因此"同一输入 + 同一种子"可复现；
- 抽取以**整条 transition 为最小单位**：state/action/block_rewards/next_state
  等字段成套保留，单条内部的因果自洽不被破坏；随机子集会让跨条相邻性
  （next_hoprates[n] 曾是 hoprates[n+1]）不再成立，但 v3 训练逐条独立采样，
  不依赖跨条相邻性，因此这不影响正确性。

metadata
--------
输出文件继承输入文件的全部 metadata（``format_version`` 保持 v3，加载侧校验规则
完全不变），并：
    - 覆盖 ``num_step_transitions`` 为实际抽取条数——必须更新，否则加载侧会因
      "数组条数与 metadata 不符"直接拒绝；
    - 追加 ``subset_source``：源文件路径；
    - 追加 ``subset_indices``：被选中的原始行号列表（升序）。
后两者不参与加载侧校验（加载只读 format_version / num_actions / num_blocks /
num_step_transitions / env_config / jammer_config / reward_config），仅用于追溯。

与训练入口的配合
----------------
子集文件仍携带原始的环境/干扰机/奖励配置快照，因此 MBPO 入口默认的 strict
metadata 校验照常生效——抽子集**不会**绕过配置一致性检查；环境或奖励配置变更后
仍应重新运行 generate_offline_replay.py 生成完整数据集：

    python train_offsets.py --offline_replay_path outputs/offline_replay/replay_1000.npz
    python train_mbpo.py --offline_replay_path outputs/offline_replay/replay_1000.npz

Example:
    python subset_offline_replay.py outputs/offline_replay/replay_20000.npz \
        --count 10000 --output outputs/offline_replay/replay_10000.npz
"""

import argparse
import json

import numpy as np

from offline_replay import REPLAY_KEYS, _load_arrays


def main():
    """命令行入口：解析参数 → 加载并校验源文件 → 抽取子集 → 写出新 v3 文件。

    校验与写入策略：先确认请求条数不超过源文件条数，再抽取；写出前把 metadata
    的 ``num_step_transitions`` 同步为实际条数，保证输出文件能通过
    load_replay_into_buffer 的 metadata 一致性校验。

    Raises:
        ValueError: ``--count`` 超过源文件条数，或源文件不满足 v3 契约
            （由 ``_load_arrays`` 抛出：版本不符、字段缺失、metadata 非 JSON）。
    """
    # description=__doc__：直接把模块 docstring 作为 --help 文本，避免两处维护。
    parser = argparse.ArgumentParser(description=__doc__)
    # 位置参数：待抽取的源 v3 文件；其父目录不必存在，但文件必须已存在。
    parser.add_argument("input", help="Path to the existing v3 replay .npz file.")
    # 保留条数；必须为正且不超过源文件条数（下方显式校验）。
    parser.add_argument(
        "--count",
        type=int,
        required=True,
        help="Number of transitions to keep.",
    )
    # 输出路径；源路径会被写进 metadata["subset_source"] 以便追溯。
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the subset replay .npz file.",
    )
    # 开关式参数：不加 --shuffle 即取前 N 条（顺序前缀）。
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Take a uniformly random subset instead of the first N transitions.",
    )
    # 仅在 --shuffle 下使用；缺省 None 时回退到 settings.RANDOM_SEED（见下）。
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed used with --shuffle (default: settings.RANDOM_SEED).",
    )
    args = parser.parse_args()

    # 复用离线加载器：顺带完成 format_version、字段齐全性与 metadata JSON 校验，
    # 因此本脚本不需要（也不应该）重复实现一遍格式检查。
    arrays, metadata = _load_arrays(args.input)
    # v3 契约保证所有字段第一维长度一致，任取一个字段即可得到总条数 N。
    total = len(arrays["state_imgs"])
    if args.count > total:
        # 提前报错：np 的高级索引在越界时会直接抛 IndexError，且无法给出这种
        # "请求条数 vs 文件条数"的明确提示。
        raise ValueError(
            f"Requested {args.count} transitions but the file only has {total}."
        )

    if args.shuffle:
        if args.seed is None:
            # 延迟导入 settings：非 shuffle 路径不必加载全局配置，也便于本脚本
            # 在最小依赖环境下被导入（例如只调用 main() 的测试）。
            import settings

            args.seed = settings.RANDOM_SEED
        # 独立 Generator：不复用/扰动全局 numpy 随机状态，抽样结果只由 seed 决定。
        rng = np.random.default_rng(args.seed)
        # replace=False 为无放回抽样；排序后输出仍是源文件顺序的子序列。
        indices = np.sort(rng.choice(total, size=args.count, replace=False))
    else:
        # 前 N 条：等价于取前缀，保留原始采集顺序。
        indices = np.arange(args.count)

    # 按行抽取：每条 transition 的全部字段共享同一组行号，保证整条成套。
    subset = {key: arrays[key][indices] for key in REPLAY_KEYS}

    # 复制后改写：不修改 _load_arrays 返回的 metadata 对象。
    metadata = dict(metadata)
    # 必须同步为实际条数，否则加载侧 "arrays=N, metadata=M" 校验会失败。
    metadata["num_step_transitions"] = int(args.count)
    # 审计字段：记录来源与选中行号，便于回溯"这个子集是怎么来的"；
    # 加载侧不读取它们，也不影响 format_version 与配置快照校验。
    metadata["subset_source"] = str(args.input)
    metadata["subset_indices"] = indices.tolist()

    # metadata 来自已解析的 JSON，类型天然可序列化，无需 _as_jsonable；
    # 键排序与 save_replay_buffer 保持一致，便于同内容文件做字节级比较。
    np.savez_compressed(
        args.output,
        **subset,
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    print(
        f"Wrote {args.count}/{total} transitions "
        f"({'random' if args.shuffle else 'first-N'}) to {args.output}"
    )


if __name__ == "__main__":
    main()
