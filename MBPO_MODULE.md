# MBPO 奖励模型模块

本文档描述当前 FHSS 环境使用的 step-level reward-model augmented SAC。训练入口为 `train_mbpo.py`，数据格式与根目录十头离散 SAC 及 v3 replay 完全一致。

## 1. 算法边界

标准 MBPO 通常学习完整环境动力学：

```text
(state, action) -> reward, next_state
```

本项目只学习一步 block reward：

```text
(PSD image, hoprate, offsets[num_blocks]) -> block_rewards[num_blocks]
```

模型不预测下一张 PSD，也不进行多步状态递推。当前环境的 observation 只包含外部干扰和噪声，下一 observation 随干扰时序变化，不受本步 offsets 影响，因此合成样本可以复用真实 replay 中对应的 `next_state_img`、`next_hoprate` 和 `done`。

如果未来 observation 引入本步发射信号或其他 action-dependent 状态，该假设必须重新验证。

## 2. Step-Level 数据契约

真实 replay 和 model replay 都使用 `SAC.py` 的 `ReplayBuffer`。一条 transition 表示一个完整环境 step：

```text
state_imgs       [B, H, W] 或 [B, C, H, W]
hoprates         [B]
actions          [B, num_blocks]
block_rewards    [B, num_blocks]
step_rewards     [B] = mean(block_rewards, axis=1)
next_state_imgs  与 state_imgs 同形状
next_hoprates    [B]
dones            [B]
```

不存在 `block_idx`、单 offset action 或人为构造的 block 间 transition。block 数量和离散 action 数量均从环境读取，不在 MBPO 中硬编码。

## 3. 独立 CNN Ensemble

`r_predict_model/model.py` 提供 `StepRewardEnsemble`。默认包含 5 个模型并选择 3 个 elite。

每个成员拥有完全独立的：

- CNN PSD 编码器；
- hoprate embedding；
- 完整 offsets one-hot 编码器；
- fusion MLP；
- 十维 Logistic-Normal latent mean/log-variance 输出。

成员之间不共享 CNN 或其他权重。实现按成员顺序训练和推理，以控制 100×100 PSD 在 GPU 上的峰值显存。

每个成员在潜变量空间输出对角高斯分布。训练 reward 先按每条样本的 hoprate 和 `BER∈[0,0.5]` 推导边界，超界目标仅在奖励模型内部饱和，再归一化、避开 sigmoid 端点并做 logit 变换。训练损失为 latent Gaussian NLL；holdout 排名则在真实 reward 单位下比较 sigmoid 有界预测与有界目标的平均 MSE。

## 4. 模型训练

奖励模型只使用真实 replay，不使用合成样本自训练。默认流程是：

1. 执行一个真实环境 step 并写入一条完整 transition。
2. 当真实 replay 数量至少达到 `model_train_batch_size` 时，按 `model_train_freq` 判断是否拟合。
3. 对当前 real buffer 中全部 transition 做一次随机 train/holdout 划分。
4. 公共 holdout 默认占 20%；各成员遍历完整 train split，但使用独立随机顺序。
5. 网络参数在不同环境 step 之间连续训练，不重新初始化。
6. holdout 连续 5 个 epoch 没有至少 1% 改善时早停，最多 100 epoch。
7. 恢复各成员本次拟合期间的最佳权重及对应 Adam 状态，再选择 holdout MSE 最低的 elite。

即使预载了离线 replay，也不会在第一个在线 step 前单独初训。默认 `model_train_freq=1`，因此每个在线 step 后都会遍历当时完整的真实 replay。对 5,000 条 100×100 transition 而言仍有较高成本；需要更快实验时应显式调大训练间隔或降低 epoch 上限。

## 5. 一步合成 Rollout

每次奖励模型拟合后：

1. 从真实 replay 均匀采样最多 `rollout_batch_size` 个完整 step。
2. SAC 根据每条样本自己的 PSD 和 hoprate 一次生成完整 offsets 向量。
3. 每条 transition 随机选择一个 elite。
4. 从该 elite 的完整 latent Gaussian 中采样一条向量。
5. 对每条样本计算 `reward(BER=0)` 与 `reward(BER=0.5)`，取其最小值和最大值为边界，再用仿射 sigmoid 将 latent 样本映射为合法 reward；rollout 不做 reward 裁剪，若模型违反边界则直接报错。
6. 将预测 reward 与真实 transition 的 next state、next hoprate、done 组合后追加到现有 model replay；不再清空上一版本经验，容量满后由固定容量队列按 FIFO 淘汰最旧 transition。

同一 transition 的所有 block reward 来自同一个 elite，避免逐 head 选择不同模型。日志同时记录：

- 合成 reward 均值和标准差；
- elite mean disagreement 的均值和 P95；
- 真实训练目标在奖励模型内部被物理边界饱和的比例。
- rollout 前后 model replay 大小、容量以及本轮 FIFO 淘汰数量。

model replay 使用固定容量的持续 FIFO。它会保留旧奖励模型版本生成的样本，直到后续 rollout 将其逐步淘汰；这能避免频繁重建造成的数据量骤降，但也意味着模型快速变化时缓冲区会短暂混合多个版本的合成奖励。

## 6. SAC 混合更新

默认 `real_ratio=0.2`。每个 SAC batch 按目标比例分别从真实和合成 replay 采样，然后合并并打乱。

- model replay 未就绪时，使用纯真实 batch。
- 一侧样本不足时，由另一侧补齐。
- 返回 batch 始终保持 v3 schema 和请求的完整 batch size。
- SAC actor、critic、alpha 和 TD target 算法本身不因 MBPO 改变。

## 7. 离线 Replay

MBPO 只接受 v3 step-level replay。默认严格比较以下 metadata：

- `env_config`；
- `jammer_config`；
- `reward_config`。

缺失或不匹配会在写入 buffer 前报错。确实需要跨配置实验时可显式传入：

```bash
--allow_replay_config_mismatch
```

纯在线 warm-up 使用：

```bash
--offline_replay_path none
```

## 8. 默认配置

`settings.py` 中的主要参数：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `num_networks` | 5 | 独立 CNN 成员数 |
| `num_elites` | 3 | rollout 可选择的 elite 数 |
| `hidden_size` | 200 | action/fusion MLP 宽度 |
| `learning_rate` | 1e-3 | 奖励模型学习率 |
| `weight_decay` | 1e-5 | 奖励模型权重衰减 |
| `model_train_freq` | 1 | 每隔多少真实环境 step 全量继续拟合 |
| `model_train_batch_size` | 256 | 模型训练 batch 及在线 warm-up 门槛 |
| `holdout_ratio` | 0.2 | holdout 比例 |
| `early_stop_patience` | 5 | 早停 patience |
| `max_epochs` | 100 | 单次全量拟合最大 epoch |
| `min_improvement` | 0.01 | holdout 相对改善阈值 |
| `rollout_batch_size` | 500 | 每次生成的最大合成 step 数 |
| `rollout_length` | 1 | 固定为一步 |
| `real_ratio` | 0.2 | SAC batch 中真实样本目标比例 |
| `model_replay_size` | 4000 | 合成 replay FIFO 容量 |

所有训练预算参数均有对应 CLI 选项，可通过 `train_mbpo.py --help` 查看。

## 9. 运行方式

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --help
```

使用默认 v3 离线 replay：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py
```

不依赖离线文件的缩小 smoke run：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --offline_replay_path none --steps_per_episode 2 --batch_size 2 --model_train_batch_size 2 --num_networks 2 --num_elites 1 --pred_hidden_size 16 --model_max_epochs 1 --model_patience 0 --rollout_batch_size 2 --update_iters_per_step 1 --output_dir outputs/mbpo_smoke
```

## 10. 输出与 Checkpoint

默认输出目录为 `outputs/mbpo/comb/pre50000`，包括：

```text
training_log.txt
reward.png
ber.png
loss.png
model_reward.png
model_holdout.png
model_disagreement.png
model_target_saturation_fraction.png
sac_inference.pt
reward_model_inference.pt
figures/
```

`sac_inference.pt` 保存 actor 权重和网络/环境维度；其格式升级为 v2，并写入 `groupnorm_v2` 架构标识。旧 SAC v1 使用 BatchNorm，训练与单样本推理的统计口径不同，加载器会明确拒绝，必须重新训练。`reward_model_inference.pt` 保存全部 ensemble 权重、elite 索引、reward 配置、BER 边界、logit epsilon 和元数据，其独立 checkpoint 格式仍为 v1；旧的单一无界输出头与新的 latent mean/log-variance 双头权重不兼容，不提供迁移。两类 checkpoint 都不包含 optimizer、replay 或 RNG 状态，只用于推理和评估，不能精确恢复训练。

若纯在线运行始终未达到奖励模型 warm-up 门槛，只保存 SAC checkpoint，并在日志中明确说明奖励模型尚未拟合。

## 11. 已知限制

1. 这不是完整 dynamics MBPO，无法在模型内部递推 PSD 状态。
2. 合成 next state 来自真实 replay，依赖 observation transition 与 action 无关的环境假设。
3. 默认每 step 全量训练独立 CNN ensemble，计算成本很高。
4. ensemble 继续使用每次拟合时重新随机划分的公共 holdout，且各成员遍历同一训练集合；holdout 与分歧可能偏乐观。
5. 低 `real_ratio` 会放大奖励模型偏差；需结合 holdout、disagreement 和目标饱和率诊断。
6. 当前完整动作编码保留跨 block 表达能力，但 factorized SAC 在 reactive 模式下无法完整表示任意跨 block action 耦合。
7. v1/v2 block-level replay、SAC BatchNorm checkpoint v1 与旧的无界奖励模型 checkpoint 均不兼容，必须重新生成数据和训练模型。
8. 持续 FIFO 会在奖励模型版本切换后保留一段旧合成经验；若模型非平稳性很强，需要结合容量、rollout 频率和 `real_ratio` 调节其滞后程度。
