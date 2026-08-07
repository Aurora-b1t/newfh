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

## 3. 矩阵式 CNN Ensemble

`r_predict_model/model.py` 提供 `StepRewardEnsemble`。默认包含 5 个模型并选择 3 个 elite。

所有成员被组织进**单个矩阵网络** `MatrixStepRewardMember`：每个权重张量带成员领先维（Linear 权重 shape `[num_networks, out, in]`，CNN 权重 shape `[num_networks, C_out, C_in, k, k]`），成员之间不共享任何参数。CNN 部分用分组卷积技巧（输入沿通道维复制 `num_networks` 份，各组权重放入相邻输出通道组），一次 forward/backward 并行计算全部成员，并由**一个共享 Adam 优化器**一次 step 更新所有成员。

每个成员拥有完全独立的：

- 两层 CNN PSD 编码器；
- 单层 hoprate embedding；
- 完整 offsets one-hot 编码器；
- state fusion 与 state-action fusion（各一层）；
- 十维 Logistic-Normal latent mean/log-variance 输出。

共享 state encoder 的固定结构为：PSD 分支 `Conv 1→16 + GN4 + ReLU → Conv 16→32 + GN8 + ReLU → Pool → FC512 + ReLU`；连续 hoprate 归一化后经过一层 `1→64` ReLU MLP；两者拼接后经过一层 `576→256` ReLU fusion。随后完整 offsets one-hot 先编码到 `hidden_size`，再与 256 维 state 特征进入一层同宽 SiLU state-action fusion，最后接 latent mean/log-variance 输出头。100×100 PSD 下单个成员约 2097 万参数（五成员矩阵共约 1.05 亿参数）。

训练和推理都按成员维整体并行，不再逐成员循环。每个成员在潜变量空间输出对角高斯分布。训练 reward 先按每条样本的 hoprate 和 `BER∈[0,0.5]` 推导边界，超界目标仅在奖励模型内部饱和，再归一化、避开 sigmoid 端点并做 logit 变换。训练损失为全体成员的 latent Gaussian NLL 均值；holdout 排名则在真实 reward 单位下比较 sigmoid 有界预测与有界目标的平均 MSE。

## 4. 模型训练

奖励模型只使用真实 replay，不使用合成样本自训练。默认流程是：

1. 执行一个真实环境 step 并写入一条完整 transition。
2. 当真实 replay 数量至少达到 `model_train_batch_size` 时，按 `model_train_freq` 判断是否拟合。
3. 对当前 real buffer 中全部 transition 做一次随机 train/holdout 划分，全部成员共用该划分。
4. 公共 holdout 默认占 20%；每个 epoch 全体成员用同一批序并行训练，一次反向传播同时更新所有成员，最大 `max_epochs` 轮。
5. 每次拟合都从头重新训练：拟合开始时重新随机初始化矩阵成员全部权重并清空共享 Adam 状态，不在上一版本模型基础上继续训练；实现上可复用已物化的模块和优化器对象。
6. **全局早停**：每个 epoch 并行评估全体成员的 holdout MSE，以最优成员（holdout 最低者）为准；该成员连续 `patience` 个 epoch 没有至少 `min_improvement` 相对改善时，全体成员一起停止。
7. 停止时全体成员恢复到全局最佳 epoch（最优成员达到最佳 holdout 的那个 epoch）的各自权重，共享优化器状态同步恢复。
8. 按最终 holdout MSE 排序选择 elite。
9. 记录每个成员在本次拟合内逐 epoch 的 holdout MSE 曲线（含训练前的初始评估作为 epoch 0）；全部曲线以 NaN 填充对齐后汇总到 `holdout_curves.npz`。传入 `--save_model_curve_figures` 时才额外为每次拟合保存独立 PNG（`holdout_curves/holdout_step_XXXX.png`）。不再生成跨 step 的 holdout 汇总曲线。全体成员的 `epochs` 相同。

即使预载了离线 replay，也不会在第一个在线 step 前单独初训。默认每 10 个在线 step 对全部真实 replay 重训一次，可通过 `--model_train_freq` 调整。对 20,000 条 100×100 transition 而言仍有较高成本；GPU 训练路径会把 reward 字段放入持久化 ring cache，后续 fit 只更新新增或 FIFO 覆盖的 transition，避免重复构造和 CPU→GPU 拷贝；模型权重和 Adam 状态仍在每次 fit 开始时重新初始化。资源不足时可使用 `--no-cache_model_dataset`。

在 CUDA 服务器上，`train_mbpo.py` 支持不改变训练预算的运行时优化选项：

- `--model_compile`：复用 `torch.compile` 的矩阵 ensemble 图；首次 fit 有编译开销，后续 fit 更快。
- `--model_precision bfloat16` 或 `float16`：启用 reward-model autocast；需要用 holdout 和最终策略结果验证数值容差。
- `--model_fast_math`：启用 TF32 和 cuDNN autotuning；可能改变收敛轨迹，默认关闭。

训练数据现在统一通过 PyTorch `DataLoader` 处理。reward-model 的 train/holdout split 使用可复用的 Dataset，SAC 的真实/模型 replay 使用 GPU 常驻 ring Dataset 和固定数量的混合 batch sampler；每个 batch 保持原有真实/模型比例和 batch 内无放回采样。A800 CUDA 路径默认 `num_workers=0`，因为 CUDA tensor 不应交给多进程 worker；关闭 GPU cache 时可使用 `--data_loader_workers` 和 `--data_loader_pin_memory` 启用 CPU streaming fallback。

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

默认 `real_ratio=0.2`。每个 SAC batch 按目标比例分别从真实和合成 replay 采样，然后合并；两侧已经独立随机采样，因此不再做一次额外的整批 shuffle。

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
| `batch_size` | 2048 | SAC 混合 replay batch |
| `model_train_freq` | 10 | 每隔多少真实环境 step 全量继续拟合 |
| `model_train_batch_size` | 2048 | 模型训练 batch 及在线 warm-up 门槛 |
| `holdout_ratio` | 0.2 | holdout 比例 |
| `early_stop_patience` | 5 | 全局早停 patience（以最优成员为准） |
| `max_epochs` | 100 | 单次全量拟合最大 epoch |
| `min_improvement` | 0.01 | 最优成员 holdout 相对改善阈值 |
| `rollout_batch_size` | 2048 | 每次生成的最大合成 step 数 |
| `rollout_length` | 1 | 固定为一步 |
| `real_ratio` | 0.2 | SAC batch 中真实样本目标比例 |
| `model_replay_size` | 4000 | 合成 replay FIFO 容量 |
| `cache_dataset_on_device` | true | 是否把当前 reward-model 数据集缓存到训练设备 |
| `cache_replay_on_device` | true | 是否把 SAC real/model replay 缓存到训练设备 |
| `data_loader_workers` | 0 | DataLoader worker 数；CUDA cache 时必须为 0 |
| `data_loader_pin_memory` | false | CPU fallback 是否固定 DataLoader batch |
| `model_precision` | float32 | reward-model 运行精度，不改变训练预算 |
| `model_compile` | false | 是否复用 `torch.compile` 图 |
| `model_fast_math` | true | 是否启用 TF32/cuDNN autotuning |
| `save_curve_figures` | true | 是否每次拟合额外写入两张 PNG 曲线图 |

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

默认输出目录为 `outputs/mbpo/comb/0`，包括：

```text
training_log.txt
reward.png
ber.png
loss.png
model_reward.png
model_disagreement.png
model_target_saturation_fraction.png
holdout_curves.npz
holdout_curves/             # 仅 --save_model_curve_figures 时生成
train_curves/               # 仅 --save_model_curve_figures 时生成
sac_inference.pt
reward_model_inference.pt
figures/
```

`sac_inference.pt` 保存 actor 权重和网络/环境维度；其格式升级为 v4，并写入 `cnn2_groupnorm_hop_mlp1_fusion_mlp1_v4` 架构标识。旧 SAC v1 BatchNorm、v2 浅层 GroupNorm、v3 三层 CNN 或其他架构会被明确拒绝。`reward_model_inference.pt` 保存矩阵 ensemble 权重、elite 索引、reward 配置、BER 边界、logit epsilon 和元数据；其格式升级为 v3，并写入 `cnn2_groupnorm_hop_mlp1_state_action_fusion1_matrix_v3`。旧 reward v1/v2 不做部分迁移。两类 checkpoint 都不包含 optimizer、replay 或 RNG 状态，只用于推理和评估，不能精确恢复训练。

若纯在线运行始终未达到奖励模型 warm-up 门槛，只保存 SAC checkpoint，并在日志中明确说明奖励模型尚未拟合。

## 11. 已知限制

1. 这不是完整 dynamics MBPO，无法在模型内部递推 PSD 状态。
2. 合成 next state 来自真实 replay，依赖 observation transition 与 action 无关的环境假设。
3. 每次 reward-model fit 仍然从头全量训练矩阵 ensemble，单次计算成本很高；默认每 10 个环境 step fit 一次，并行更新会同时持有全部成员的激活，峰值显存约为逐成员训练的 `num_networks` 倍。
4. ensemble 继续使用每次拟合时重新随机划分的公共 holdout，且全体成员遍历同一训练集和同一批序；holdout 与分歧可能偏乐观，成员多样性仅来自随机初始化。
5. 低 `real_ratio` 会放大奖励模型偏差；需结合 holdout、disagreement 和目标饱和率诊断。
6. 当前完整动作编码保留跨 block 表达能力，但 factorized SAC 在 reactive 模式下无法完整表示任意跨 block action 耦合。
7. v1/v2 block-level replay、SAC inference v1/v2 与 reward-model v1 checkpoint 均不兼容；旧模型必须重新训练，但现有 step-level v3 replay 可继续使用。
8. 持续 FIFO 会在奖励模型版本切换后保留一段旧合成经验；若模型非平稳性很强，需要结合容量、rollout 频率和 `real_ratio` 调节其滞后程度。
