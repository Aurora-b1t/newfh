# MBPO 算法模块说明

> **当前暂停使用。** 根目录 SAC 和 replay 已升级为十头 step-level v3 格式，本文描述的 MBPO 奖励模型仍依赖 `block_idx` 和单动作 block transition。`train_mbpo.py` 会在创建环境或训练模型前抛出明确错误。下文保留为 legacy 设计记录，待奖励模型改为 `(state, hoprate, actions[10]) -> block_rewards[10]` 后再恢复。

本文档说明本项目中 MBPO（Model-Based Policy Optimization）相关代码的设计、数据流、训练流程和关键参数。当前实现是面向 FHSS 抗干扰离散 offset 决策任务的工程化版本，并非直接照搬连续控制任务中的完整 dynamics MBPO。

## 1. 模块定位

本项目的主强化学习算法仍然是离散动作 SAC。MBPO 模块的作用是额外训练一个奖励预测模型，用它生成少量合成 replay 样本，辅助 SAC 更新策略和 Q 网络。

当前 MBPO 模块只预测一步 reward：

```text
flatten(state_img, hoprate, block_idx, action) -> reward
```

它不预测下一帧 PSD 图像，也不做多步状态递推。因此，本项目中的 MBPO 更准确地说是 reward-model augmented SAC：使用模型扩充奖励样本，而不是完整学习环境动力学。

## 2. 相关文件

| 文件 | 职责 |
| --- | --- |
| `train_mbpo.py` | FHSS + SAC + MBPO 的训练入口，负责真实环境交互、真实 replay 写入、奖励模型训练、合成 replay 生成、SAC 混合更新和曲线输出。 |
| `r_predict_model/model.py` | 奖励预测 ensemble 模型实现，包括标准化器、ensemble 全连接层、概率模型训练、holdout 验证和 elite 模型选择。 |
| `r_predict_model/mbpo_adapter.py` | SAC replay 与奖励模型输入格式之间的适配层，负责 flatten 特征、提取监督学习标签、执行模型 rollout 并写入 model replay。 |
| `r_predict_model/replay_memory.py` | 通用 MBPO 模板中的简单环形 replay memory。当前 FHSS 训练入口主要使用 `SAC.py` 中的 `ReplayBuffer`。 |
| `r_predict_model/main.py` | 通用 MBPO 训练模板，保留为参考实现；它依赖外部提供 `SAC`、`env`、`env_sampler` 等对象，不是当前 FHSS 训练的主入口。 |
| `r_predict_model/__init__.py` | 暴露 `EnsembleDynamicsModel`，便于 `from r_predict_model import EnsembleDynamicsModel`。 |

## 3. 与标准 MBPO 的区别

标准 MBPO 通常训练一个 ensemble dynamics model：

```text
(state, action) -> reward, state_delta
```

然后从真实 replay 的状态出发，短长度 rollout 生成虚拟轨迹，再用真实和虚拟样本混合训练策略。

本项目的实现有三个重要差异：

1. 只预测 reward，不预测 `next_state_img`。
2. 合成样本的 rollout 长度固定为 1。
3. 合成样本复用真实 replay 已编码的顺序 next state：block 0～8 保持当前 PSD/hoprate 并前进 block index，block 9 才进入下一环境 PSD/hoprate 并回到 block 0；只有 reward 被模型预测值替换。

这样做的原因是 FHSS 状态包含 PSD waterfall 图像，直接学习下一帧图像动力学成本高且误差容易累积；而当前训练目标主要需要为 SAC 提供更多 block-level reward 监督信号。

## 4. SAC replay 数据结构

`SAC.py` 中的 `ReplayBuffer` 存储如下字段：

```text
state_img
hoprate
block_idx
action
reward
next_state_img
next_hoprate
next_block_idx
done
```

其中：

- `state_img`：环境返回的 PSD 图像观测。
- `hoprate`：当前跳频速率。当前 `train_mbpo.py` 使用固定 hoprate。
- `block_idx`：当前环境 step 内部第几个 block，范围通常是 `0..9`。
- `action`：当前 block 选择的 offset 离散动作。
- `reward`：由 BER 和 hoprate 计算得到的 block-level reward。
- `next_state_img`：block 0～8 为当前 PSD，block 9 为下一环境 step 的 PSD。
- `next_block_idx`：下一 block 索引，当前实现为 `(block_idx + 1) % 10`，即第 10 个 block 结束后 s_(t+1) 的 block_idx 回到 0（新一轮 block 的起点）。
- `done`：终止标记。

一个环境 step 会产生 10 个 offset 动作，因此 `train_mbpo.py` 会把一次环境交互拆成 10 条 block-level replay transition。

## 5. 奖励模型输入与标签

`r_predict_model/mbpo_adapter.py` 中的 `encode_transition_inputs()` 将结构化 replay 字段转换为二维矩阵：

```text
input = concat(
    flatten(state_img),
    hoprate,
    block_idx,
    action
)
```

标签为：

```text
label = reward
```

如果 batch size 为 `B`，PSD 图像 flatten 后长度为 `S`，则：

```text
inputs.shape = [B, S + 3]
labels.shape = [B, 1]
```

其中额外的 3 个标量特征分别是 `hoprate`、`block_idx` 和 `action`。

## 6. Ensemble 奖励模型

`EnsembleDynamicsModel` 是对底层 `EnsembleModel` 的封装。它负责：

1. 对输入特征做标准化。
2. 划分训练集和 holdout 验证集。
3. 并行训练多个 ensemble 成员。
4. 根据 holdout MSE 选择 elite 模型。
5. 提供 deterministic 或 stochastic reward 预测。

底层网络由多个 `EnsembleFC` 层组成。每个 ensemble 成员有独立参数，但通过 batched matrix multiplication 并行前向计算：

```text
input:  [network_size, batch_size, input_dim]
weight: [network_size, input_dim, output_dim]
output: [network_size, batch_size, output_dim]
```

最终输出包含 reward 均值和方差：

```text
mean, variance = model(input)
```

训练时使用带不确定性的概率损失。holdout 阶段使用普通 MSE 对每个 ensemble 成员排序，并选择 loss 最低的 `num_elites` 个模型作为 elite 模型。

## 7. 训练主流程

`train_mbpo.py` 的核心循环如下：

```text
初始化环境、SAC agent、真实 replay、模型 replay、奖励 ensemble

for step_idx in 1..steps_per_episode:
    1. SAC actor 依次为 10 个 block 选择 offset
    2. 环境执行一次 step，返回 next_state_img 和每个 block 的 BER
    3. 根据 BER 计算每个 block 的 reward
    4. 将一次环境交互拆成 10 条真实 replay transition
    5. 当真实 replay 足够且达到 model_train_freq：
        a. 用全部真实 replay 训练奖励 ensemble
        b. 从真实 replay 采样起点
        c. 用当前 SAC actor 选 action
        d. 用奖励 ensemble 预测 synthetic reward
        e. 写入 model replay
    6. 当真实 replay 足够：
        a. 从 real replay 和 model replay 按 real_ratio 混合采样
        b. 更新 SAC actor / critic / alpha
    7. 记录 reward、BER、loss 和模型 reward 曲线
```

训练结束后输出：

```text
reward.png
ber.png
loss.png
model_reward.png
training_log.txt
```

默认输出目录为：

```text
outputs/mbpo/comb/pre50000
```

## 8. 合成 replay 的生成方式

`rollout_reward_model()` 的合成样本流程为：

1. 从真实 replay 中采样 `rollout_batch_size` 条 transition。
2. 使用当前 SAC actor 为每个采样状态重新选择 action。
3. 将 `state_img + hoprate + block_idx + action` 编码为模型输入。
4. 使用 elite ensemble 预测 reward。
5. 写入 `model_buffer`。

写入的 transition 结构为：

```text
state_img       = sampled state_img
hoprate         = fixed_hoprate
block_idx       = sampled block_idx
action          = actor selected action
reward          = model predicted reward
next_state_img  = sampled next_state_img.copy()   # 真实 replay 的顺序 next state
next_hoprate    = sampled next_hoprate
next_block_idx  = (block_idx + 1) % 10
done            = False
```

注意：这里的 `next_state_img` 并不是模型预测出来的，而是复用真实 replay 中该 transition 已编码的顺序 next state。block 0～8 的 next image 与当前 image 相同，block 9 才使用下一环境 PSD。奖励部分仍由模型预测，这是当前 reward-only MBPO 设计的核心近似。

## 9. 真实样本与合成样本混合

`sample_mixed_batch()` 根据 `real_ratio` 控制 SAC 更新 batch 中真实样本和合成样本的比例。

例如：

```text
batch_size = 256
real_ratio = 0.5
```

理想情况下，每次 SAC 更新使用：

```text
128 条真实 replay
128 条模型 replay
```

如果 `model_buffer` 为空，则退化为只使用真实 replay。实际采样数量还会受到两个 replay buffer 当前大小限制。

## 10. 关键配置

MBPO 相关默认配置位于 `settings.py` 的 `MBPO_CONFIG`：

```python
MBPO_CONFIG = {
    "num_networks": 5,
    "num_elites": 3,
    "hidden_size": 200,
    "model_train_freq": 1,
    "model_train_batch_size": 256,
    "rollout_batch_size": 2000,
    "rollout_length": 1,
    "real_ratio": 0.2,
    "model_replay_size": 30000,
}
```

参数含义：

| 参数 | 含义 |
| --- | --- |
| `num_networks` | ensemble 中模型总数。数量越大，不确定性估计越稳，但训练成本越高。 |
| `num_elites` | 用于 rollout 的 elite 模型数量，根据 holdout loss 选择。 |
| `hidden_size` | 奖励模型隐藏层宽度。 |
| `model_train_freq` | 每隔多少个环境 step 训练一次奖励模型并生成合成样本。 |
| `model_train_batch_size` | 奖励模型训练 batch size。 |
| `rollout_batch_size` | 每次模型 rollout 从真实 replay 采样多少个起点。 |
| `rollout_length` | 当前设计固定为 1；配置保留用于表达 MBPO 语义。 |
| `real_ratio` | SAC 更新 batch 中真实样本占比。 |
| `model_replay_size` | 合成 replay buffer 容量。 |

训练相关配置位于 `TRAIN_CONFIG`：

```python
TRAIN_CONFIG = {
    "steps_per_episode": 150,
    "update_iters_per_step": 10,
    "fixed_hoprate": 100.0,
}
```

Offline real replay is loaded before the first SAC and reward-model update, so this training entry point has no replay warm-up gate.

## 11. 运行方式

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --help
```

短 smoke test：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --steps_per_episode 5 --batch_size 20 --model_train_freq 2 --rollout_batch_size 20 --offline_replay_path outputs/offline_replay/replay_50000_random_hoprate_v2.npz --output_dir outputs/mbpo_smoke
```

默认训练：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py
```

指定 CPU：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --cpu_only
```

指定输出目录：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --output_dir outputs/mbpo_experiment
```

## 12. 日志解读

训练日志中常见字段：

| 字段 | 含义 |
| --- | --- |
| `Offsets` | 当前环境 step 中 10 个 block 的 offset 动作。 |
| `Rew` | 当前 step 的平均 block reward。 |
| `BER` | 当前 step 的平均 BER。 |
| `RealBuf` | 真实 replay 当前样本数。 |
| `ModelBuf` | 合成 replay 当前样本数。 |
| `ModelHoldout` | 奖励 ensemble 在 holdout 数据上的平均 MSE。 |
| `Elites` | 当前被选中的 elite ensemble 成员索引。 |
| `Loss: A` | SAC actor loss。 |
| `Loss: C` | SAC critic loss。 |
| `Alpha` | SAC entropy temperature。 |
| `T` | 当前环境 step 耗时。 |

如果 `ModelHoldout` 长期很高，说明奖励模型没有很好拟合真实 replay；可以考虑增加真实样本、降低模型使用比例、调整 reward 尺度或增大 ensemble 容量。

## 13. 当前实现的假设与限制

1. 合成样本只对 reward 建模，不建模 PSD 图像动态。
2. `next_state_img` 复用真实 transition 的顺序 next state，因此奖励模型本身不学习状态动力学。
3. 当前 rollout 长度实际为 1，避免多步模型误差累积。
4. 奖励模型只使用真实 replay 训练，不使用合成 replay 反复自训练。
5. 10 个 block 被拆成 10 条 transition，这是一种工程近似，不是严格的 sequence-level MDP 建模。
6. `real_ratio` 过低时，SAC 更新会更依赖模型样本，可能放大模型偏差。
7. 奖励模型性能依赖 replay 覆盖度；早期数据不足时，合成 reward 可能噪声较大。

## 14. 调参建议

优先建议从保守配置开始：

```text
real_ratio >= 0.5
rollout_length = 1
num_networks = 5
num_elites = 3
model_train_freq = 1
```

注：当前 `settings.MBPO_CONFIG` 默认 `real_ratio=0.2`、`model_train_freq=1`，偏向更激进地使用合成样本；若训练不稳定，建议通过命令行临时调高 `real_ratio`。

如果模型样本帮助不明显：

- 增大 `model_replay_size`，保留更多合成样本。
- 增大 `rollout_batch_size`，每次模型训练后生成更多样本。
- 适当降低 `real_ratio`，提高合成样本参与比例。

如果训练不稳定：

- 提高 `real_ratio`，让 SAC 更依赖真实 replay。
- Select a larger or environment-specific offline replay file with `--offline_replay_path` when more real-data coverage is needed.
- 降低 `rollout_batch_size`，减少早期低质量合成样本。
- 检查 `ModelHoldout` 和 `model_reward.png`，确认模型 reward 没有明显漂移。

如果奖励模型欠拟合：

- 增大 `hidden_size`。
- 增大 `num_networks`。
- 检查 reward 尺度是否过大或过小。
- 增加真实环境采样步数。

## 15. 后续可改进方向

1. 学习完整 dynamics：将目标扩展为 `reward + next_state_delta`，但需要处理 PSD 图像维度高、误差累积和计算成本问题。
2. 引入 termination model：对真实 done 进行建模，而不是合成样本固定 `done=False`。
3. 改成 latent dynamics：先用 encoder 将 PSD 图像压缩到 latent，再在 latent 空间预测下一状态。
4. 使用 sequence policy：把 10 个 offset 作为序列整体建模，减少 block-level replay 拆分近似。
5. 加入不确定性过滤：当 ensemble disagreement 过高时丢弃合成样本，降低模型偏差。
6. 增加模型评估脚本：单独可视化真实 reward 与预测 reward 的散点图、误差分布和 elite 模型差异。
