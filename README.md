# 强化学习跳频抗干扰系统（进行中ing）

> **状态：项目进行中，尚未完成。** 环境与多个算法入口可运行，但训练效果验证、算法稳定性仍在推进。本文描述的是当前代码快照的真实状态，而非最终目标。

本项目是一个面向跳频扩频（FHSS, Frequency Hopping Spread Spectrum）抗干扰研究的 Python 实验环境，当前包含：

- **FHSS/QPSK 通信仿真环境**（`fh_env.py`）：QPSK 收发链路、m 序列跳频、Rayleigh 衰落、PSD waterfall 观测、反应式/扫频/梳状干扰机叠加、预生成加速路径。
- **baseline 十头离散 SAC 训练**（`SAC.py` + `train_offsets.py`）：一次前向并行生成 10 个 categorical offset，一次环境 step 对应一条完整 replay transition。
- **step-level MBPO 奖励增强**（`r_predict_model/` + `train_mbpo.py`）：矩阵式 CNN ensemble 联合预测完整十维 block reward，生成与 v3 replay 同构的一步合成样本辅助 SAC（详见 [MBPO_MODULE.md](MBPO_MODULE.md)）。
- **导数 NBS 跳速阈值搜索**（`noisy_binary_search_derivative.py` + `train_speed_derivative.py`）：基于 MWU 的噪声二分搜索导数变体，寻找反应式干扰机的跳速跟踪/失效边界。
- **hoprate sweep 评估**（`train_speed_sweep.py`）：确定性网格遍历所有候选 hoprate，作为 NBS 搜索的对照基线。
- **离线 replay 工具链**（`offline_replay.py` / `generate_offline_replay.py` / `subset_offline_replay.py`）：生成/校验/加载/抽子集 v3 step-level 真实环境 replay（详见 [OFFLINE_REPLAY.md](OFFLINE_REPLAY.md)）。

项目根目录采用平铺结构，入口清晰、文件职责清晰。每个训练/搜索脚本自带默认输出目录，互不覆盖。

---

## 1. 项目结构

```
newfh/
├── settings.py                     # 全局配置中心（环境/干扰机/SAC/MBPO/NBS/训练循环/奖励）
├── fh_env.py                       # FHSS/QPSK Gymnasium 环境（核心仿真）
├── jammers.py                      # 干扰机实现（反应式能量检测 + 扫频/梳状）
├── SAC.py                          # 十头离散 SAC + step-level ReplayBuffer + v3 推理 checkpoint
├── r_predict_model/                # MBPO 奖励模型子包
│   ├── model.py                    #   StepRewardEnsemble：矩阵式 CNN 奖励集成（5 成员选 3 elite）
│   ├── mbpo_adapter.py             #   replay→奖励模型数据适配、混合采样、合成 rollout
│   └── __init__.py                 #   暴露 StepRewardEnsemble
├── train_offsets.py                # 训练入口 1：baseline 十头 SAC offset 训练
├── train_mbpo.py                   # 训练入口 2：SAC + 奖励模型 MBPO 训练
├── train_speed_derivative.py       # 搜索入口：导数 NBS 跳速阈值搜索（随机 offset，不训练 SAC）
├── train_speed_sweep.py            # 搜索入口：hoprate 确定性网格扫描（NBS 的对照）
├── noisy_binary_search_derivative.py  # 导数版 MWU 噪声二分搜索算法实现
├── offline_replay.py               # v3 step-level replay 序列化、严格校验与加载
├── generate_offline_replay.py      # 离线真实 replay 生成脚本（输出 .npz）
├── subset_offline_replay.py        # 从已有 replay 抽取前 N 条或随机子集
├── benchmarks/                     # 性能基准（环境 step 耗时、奖励模型拟合耗时）
├── tests/                          # unittest/pytest 测试套件
├── environment.yml                 # conda 环境定义（版本与本机验证环境一致）
├── requirements.txt                # pip 依赖清单（含 GPU/CPU torch 安装说明）
├── README.md                       # 本文档
├── MBPO_MODULE.md                  # MBPO 奖励模型模块设计文档
└── OFFLINE_REPLAY.md               # v3 离线 replay 工作流文档
```

各文件详细职责：

| 文件 | 作用 |
| --- | --- |
| [settings.py](settings.py) | 统一配置环境参数、干扰机参数、SAC/MBPO/NBS 超参、训练循环和奖励系数，每个配置块有详细中文注释。 |
| [fh_env.py](fh_env.py) | FHSS/QPSK Gymnasium 环境：预生成加速、m 序列跳频、QPSK 收发、干扰/衰落叠加、BER 与 reward 计算、PSD waterfall 观测。 |
| [jammers.py](jammers.py) | 干扰机实现：快速带限噪声源与共享基带变体池、基于能量检测的反应式干扰机（Urkowitz 1967）、扫频/梳状宽带干扰机。 |
| [SAC.py](SAC.py) | 十头离散 SAC：step-level ReplayBuffer、独立参数的 actor/critic、逐 block 输出头、共享温度系数、软更新与 v3 推理 checkpoint。 |
| [train_offsets.py](train_offsets.py) | baseline 训练入口：构建环境、SAC agent、replay buffer、训练循环、日志和曲线输出。 |
| [train_mbpo.py](train_mbpo.py) | step-level SAC + MBPO 入口：真实交互、奖励模型周期性全量重拟合、合成 replay、真实/合成混合更新、诊断与推理 checkpoint。 |
| [train_speed_derivative.py](train_speed_derivative.py) | 导数 NBS 跳速阈值搜索入口（随机 offset + 反应式干扰机）。 |
| [train_speed_sweep.py](train_speed_sweep.py) | hoprate 网格扫描评估入口，NBS 搜索的确定性对照。 |
| [noisy_binary_search_derivative.py](noisy_binary_search_derivative.py) | 导数版 MWU 噪声二分搜索算法，参考 Dereniowski et al. STACS 2025。 |
| [offline_replay.py](offline_replay.py) | v3 step-level replay 序列化、严格校验与加载工具。 |
| [generate_offline_replay.py](generate_offline_replay.py) | 离线真实 replay 生成脚本，输出带 metadata 的 `.npz` 文件。 |
| [subset_offline_replay.py](subset_offline_replay.py) | 从现有 replay 抽取子集（前 N 条或种子随机），方便小规模快速实验。 |
| `benchmarks/` | [benchmark_env_step.py](benchmarks/benchmark_env_step.py) 环境 step 耗时基准；[benchmark_reward_fit.py](benchmarks/benchmark_reward_fit.py) 奖励模型单次拟合剖析。 |
| `tests/` | 标准库 unittest 测试（兼容 pytest），覆盖多头 SAC、replay 校验、奖励模型、训练入口、环境加速路径与干扰机时序。 |

> **历史文件说明**：旧入口/参考文件（`train_joint_sac.py`、`train_joint_mbpo.py`、`joint_training.py`、`train_speed.py`、`noisy_binary_search.py`、`example/`、`pdf/`、`special_hopping_test/`、`tests/test_joint_training.py`）已从当前代码库移除，如需查看可通过 git 历史找回。`settings.py` 中的 `JOINT_OFFLINE_REPLAY_CONFIG` 为已移除的联合训练入口保留的配置块。

## 2. 环境搭建（新机器移交指南）

### 2.1 依赖概览

| 依赖 | 验证版本 | 说明 |
| --- | --- | --- |
| Python | 3.11（3.11.14） | 推荐 Anaconda/Miniconda 管理 |
| numpy | 1.26.4 | |
| scipy | 1.16.3 | |
| matplotlib | 3.10.8 | |
| gymnasium | 1.2.3 | Gymnasium 环境接口 |
| scikit-commpy | 0.8.0 | 脉冲成形滤波（rrcosfilter） |
| torch | 2.9.1（本机为 2.9.1+cu130） | GPU/CPU 版本按机器选择，见下文 |

torchvision **未被本项目使用**，无需安装。运行测试另需 pytest（9.x，可选，也可用标准库 unittest）。

### 2.2 创建环境

方式一（推荐，conda，版本与开发机完全一致）：

```bash
conda env create -f environment.yml
conda activate rl_fhss
```

方式二（纯 pip，在任意 Python 3.11 环境）：

```bash
pip install numpy==1.26.4 scipy==1.16.3 matplotlib==3.10.8 gymnasium==1.2.3 scikit-commpy==0.8.0
```

### 2.3 安装 PyTorch（重要）

`environment.yml` / PyPI 中的 `torch==2.9.1` 在 Windows 上是 **CPU 版**。训练入口默认使用 CUDA（奖励模型要求 CUDA），需要 GPU 时请用 PyTorch 官方源覆盖安装：

```bash
# 先确认本机 NVIDIA 驱动支持的 CUDA 版本（nvidia-smi 查看）
# 本项目开发机验证版本为 cu130：
pip install torch==2.9.1+cu130 --index-url https://download.pytorch.org/whl/cu130
# 驱动不支持 CUDA 13.0 时，换成官方支持的版本号，例如：
# pip install torch==2.9.1+cu126 --index-url https://download.pytorch.org/whl/cu126
```

只跑单元测试/短冒烟时 CPU 版即可，但完整训练会非常慢。

### 2.4 验证安装

```bash
# 在项目根目录执行
python -m pytest tests -q            # 方式一：pytest（推荐）
python -m unittest discover -s tests -v   # 方式二：标准库 unittest
```

预期结果：101 个测试通过、1 个跳过（跳过的用例需要显式开启环境变量，见下）。其中"真实 RF 环境单步"冒烟测试由环境变量门控，需要时手动开启：

```bash
# Git Bash / Linux
FHSS_RUN_ENV_SMOKE=1 python -m pytest tests -q
# PowerShell
$env:FHSS_RUN_ENV_SMOKE='1'; python -m pytest tests -q
```

再做一个训练入口冒烟（约几十秒，验证环境+依赖完整）：

```bash
python train_offsets.py --steps_per_episode 3 --batch_size 20 --offline_replay_path none --output_dir outputs/smoke
```

### 2.5 常见问题

- **命令里的 `python`**：下文所有示例假设已激活上述 conda 环境；开发机上等价于 `D:\Anaconda\envs\rl_fhss\python.exe`。
- **所有命令默认在项目根目录运行**（`benchmarks/` 脚本已自带根目录路径注入，也可直接运行）。
- **GPU 要求**：两个训练入口（`train_offsets.py` / `train_mbpo.py`）内部硬编码 `torch.device("cuda")`，**必须有可用 CUDA GPU**；单元测试、两个跳速搜索脚本（`train_speed_*.py`，纯 NumPy）与离线 replay 生成脚本在 CPU 上即可运行。
- **`torch.compile` 相关报错**：Windows 首次运行会在 `__pycache__` 下生成编译缓存，属正常现象；MBPO 的编译开关可用 `--no-model_compile` 关闭。
- **显存不足**：MBPO 奖励模型默认约 2.06 亿参数（每成员约 4125 万），小显存机器建议减小 `--model_train_batch_size`，或用 `--model_extra_pool` 切换参数量约 5260 万的池化变体架构（注意：会改变模型架构，checkpoint 不通用）。

## 3. 快速开始

### 3.1 Baseline：离散 SAC offset 训练

```bash
# 查看全部参数
python train_offsets.py --help
# 默认训练（默认输出目录 outputs/offsets/comb/0）
python train_offsets.py
# 冒烟测试
python train_offsets.py --steps_per_episode 3 --batch_size 20 --offline_replay_path none --output_dir outputs/smoke
# 指定输出目录 / 跳过离线 replay 纯在线
python train_offsets.py --output_dir outputs/baseline
python train_offsets.py --offline_replay_path none
```

训练结束后在 `--output_dir` 下保存 `sac_inference.pt`：只含 actor 推理权重、环境维度和配置元数据，不含 optimizer、replay 或 RNG 状态。

### 3.2 MBPO 奖励模型训练

`train_mbpo.py` 使用与十头 SAC 完全相同的 step-level v3 transition。奖励模型输入当前 PSD、实际 hoprate 和完整 offsets，为每个 block 输出 Logistic-Normal 潜变量，经 sigmoid 映射到由 `BER∈[0,0.5]`、hoprate 和奖励公式共同确定的 reward 区间；模型不预测下一 PSD，合成样本复用真实 replay 的外生 next state/next hoprate/done。默认 5 个参数完全独立的 CNN 成员选 3 个 holdout MSE 最优 elite；每隔 `model_train_freq` 个真实 step 用全部真实 replay 从头重拟合，合成样本追加到 model replay（满后 FIFO 淘汰）。完整设计与参数见 [MBPO_MODULE.md](MBPO_MODULE.md)。

```bash
python train_mbpo.py --help
python train_mbpo.py
# 纯在线缩小 smoke
python train_mbpo.py --offline_replay_path none --steps_per_episode 2 --batch_size 2 --model_train_batch_size 2 --num_networks 2 --num_elites 1 --pred_hidden_size 16 --model_max_epochs 1 --model_patience 0 --rollout_batch_size 2 --update_iters_per_step 1 --output_dir outputs/mbpo_smoke
```

MBPO 默认**严格拒绝**环境/干扰/reward metadata 不匹配的离线 replay；跨配置实验必须显式加 `--allow_replay_config_mismatch`。baseline（`train_offsets.py`）对不匹配只警告不拒绝。

### 3.3 导数 NBS 跳速阈值搜索

`train_speed_derivative.py` 使用 `noisy_binary_search_derivative.py` 的导数版 NBS，在反应式干扰机开启时搜索跳速阈值。每个环境 step：

1. NBS 给出一个待测试 hoprate；
2. 脚本随机生成 10 个 offset（不训练 SAC）；
3. 环境执行 10 个 block 并返回平均 BER；
4. NBS 用导数指标更新候选跳速权重分布：

```text
metric = ΔBER_percent / Δhoprate     （Δ 的参考点是最近一次 hoprate 不同的观测）
metric > threshold  → 向左移动（支持更小 hoprate）
metric ≤ threshold  → 向右移动（支持更大 hoprate）
```

```bash
python train_speed_derivative.py --steps 60 --output_dir outputs/speed_test_derivative
python train_speed_derivative.py --derivative_threshold -0.005 --steps 100
# 默认：--steps 400，输出目录 outputs/speed_derivative/0.5ms/-0.005
```

### 3.4 hoprate 网格扫描评估

`train_speed_sweep.py` 是 NBS 搜索的确定性网格对照：不用 NBS 选下一跳速，而是按升序遍历每个候选 hoprate（默认 10→1000 Hz、步长 10 Hz、每档 4 个 step），记录 BER/reward 诊断。每个 step 内部同样执行 10 个 block。

```bash
python train_speed_sweep.py --output_dir outputs/speed_sweep
# 快速冒烟
python train_speed_sweep.py --hoprate_max 20 --steps_per_hoprate 1
# 默认输出目录 outputs/speed_sweep/0.5ms
```

### 3.5 离线 replay：生成 / 抽子集 / 加载

一条 v3 transition 表示一个完整环境 step（10 个 offset + 10 个 block reward）。当前默认数据文件：

- baseline（固定 100 Hz）：`outputs/offline_replay/replay_5000_100_hoprate_v3.npz`
- 随机跳速版（为已移除的联合训练入口准备）：`outputs/offline_replay/replay_5000_random_hoprate_v3.npz`

生成（reactive + comb、随机 hoprate 版）：

```bash
python generate_offline_replay.py --num_step_transitions 5000 --hoprate_mode random --enable_reactive true --enable_sweep true --jammer_mode comb --output_path outputs/offline_replay/replay_5000_random_hoprate_v3.npz
# 固定 hoprate 版
python generate_offline_replay.py --num_step_transitions 5000 --hoprate_mode fixed --fixed_hoprate 100 --output_path outputs/offline_replay/replay_5000_100_hoprate_v3.npz
```

> ⚠ 注意：不带参数直接运行 `generate_offline_replay.py` 时，生成条数取 `settings.OFFLINE_REPLAY_CONFIG["num_step_transitions"]`（当前为 512），而输出路径取 `default_path`——即会**用 512 条数据覆写** `replay_5000_100_hoprate_v3.npz`。务必显式传 `--num_step_transitions` 与 `--output_path`。

从现有 replay 抽取子集做小规模实验：

```bash
python subset_offline_replay.py outputs/offline_replay/replay_5000_100_hoprate_v3.npz --count 1000 --output outputs/offline_replay/replay_1000.npz
# 或种子随机抽取（默认用 settings.RANDOM_SEED）
python subset_offline_replay.py <输入.npz> --count 1000 --shuffle --seed 0 --output <输出.npz>
```

指定 replay 给训练入口：

```bash
python train_offsets.py --offline_replay_path outputs/offline_replay/replay_5000_100_hoprate_v3.npz
# 显式纯在线
python train_offsets.py --offline_replay_path none
```

**修改 comb `switch_interval`、相位信道组、m 序列、`baseband_variant_count` 或其他干扰/环境配置后，必须重新生成离线 replay**（metadata 会记录新配置）：baseline 会警告不匹配；MBPO 默认直接拒绝。完整校验规则见 [OFFLINE_REPLAY.md](OFFLINE_REPLAY.md)。

## 4. 训练语义（一步的完整因果流程）

当前 offset 训练（baseline / MBPO 共享）流程如下：

1. 环境 `reset()` 返回 100 ms PSD waterfall observation（100×100）。
2. 训练脚本固定 hoprate（baseline；MBPO 同样固定，见 `TRAIN_CONFIG["fixed_hoprate"]`）。
3. SAC actor 输入当前 observation 与 hoprate，一次前向输出 `[10, num_channels]` raw logits，并从十个条件独立的 categorical head 采样 offset。
4. 环境一次性执行这 10 个 offset，对应 10 个 100 ms block。
5. 环境返回下一个 observation、`info["ber_blocks"]`、`info["block_rewards"]`、`info["comb_phases"]` 及其均值 reward；`comb_phases` 形状为 10 block × 100 个 1 ms 槽，表示各半开时间槽起点生效的 phase。
6. 训练脚本将整组 `actions[10]` 与 `block_rewards[10]` 写为一条 replay transition。

逐 block reward 公式（`settings.REWARD_CONFIG`，环境与离线生成器、sweep 共享同一语义）：

```text
reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
```

当启用 comb 或 both 模式时，启动日志会在控制台和 `training_log.txt` 中各记录一次实际 `channels_phase0` / `channels_phase1` 配置。

critic 同样输出十个离散 Q 头，每个头使用对应的即时 block reward；下一状态 continuation 先对每头做离散 SAC 期望，再对十头求均值并广播，从而学习平均 block return，而不构造虚假的 block 间状态转移。

## 5. SAC 实现要点（SAC.py）

- **actor、两个 online critic 与两个 target critic 参数完全独立**，各自拥有完整的 `StateEncoder`（类定义复用，不共享权重）。
- **不输入 block index**：hoprate 按环境上下界归一化到 `[-10,10]`（`HOPRATE_INPUT_SCALE`），十个固定位置的输出头分别对应 block 0～9。
- **actor 只返回 raw logits**：`take_action()` 单次返回十维整数动作；训练采样用 `Categorical(logits=...)`，确定性推理用逐头 `argmax`。
- **共享温度系数**：十头共用一个 alpha，actor、critic 和 entropy loss 都对 batch/head 维取平均。
- **PSD 编码器（两层卷积）**：`LazyConv2d 1→16 + GroupNorm(4) + ReLU → Conv 16→32 + GroupNorm(8) + ReLU → MaxPool2d(2) → flatten → LazyLinear(512) + ReLU`。GroupNorm 不维护 batch running stats（train/eval 数值一致，避免 target 值分布漂移）；参数量主要由展平后的 `flatten→512` 全连接层决定（100×100 观测下单编码器约 4100 万参数），卷积层本身只占几千参数。
- **hoprate 分支与融合**：hoprate 归一化后经过一层 `1→64` ReLU；与 512 维 PSD 特征拼接后经过 `576→256` ReLU fusion，输出 256 维共享状态特征，actor/Q 头直接 `256→n_actions`。
- **online 网络不临时切换模式**：正常 `update()` 中 actor 与两个 online critic 的 forward 均处于 train 模式；target critic 固定 eval 模式、不参与优化。
- **Lazy target 延迟初始化**：首次 TD target 计算前物化 target critic 并完整复制参数，之后做 soft update。
- **`calc_target` 在 `torch.no_grad()` 下计算**：每头先按离散 SAC 期望求 V(s′)，再对十头取均值并广播，配合 `1-done` 掩码。
- **推理 checkpoint 为 v4**（架构标识 `cnn2_groupnorm_hop_mlp1_fusion_mlp1_v4`）：旧 BatchNorm v1、浅层 GroupNorm v2、三层 CNN v3 或其他架构均被明确拒绝，必须重新训练。
- **step-level replay**：一条经验保存十维动作、十维 block reward、仅用于诊断的均值 step reward，以及真实下一环境状态；梯度只用 `block_rewards`，`step_rewards` 仅作诊断。

## 6. 输出目录与产物

输出目录不在 `settings.py` 集中配置，由各入口自带默认 `--output_dir`（均可命令行覆盖）：

| 脚本 | 默认 `--output_dir` |
| --- | --- |
| `train_offsets.py` | `outputs/offsets/comb/0` |
| `train_mbpo.py` | `outputs/mbpo/comb/0` |
| `train_speed_derivative.py` | `outputs/speed_derivative/0.5ms/-0.005` |
| `train_speed_sweep.py` | `outputs/speed_sweep/0.5ms` |

每个脚本还支持 `--log_file`（默认 `training_log.txt`，位于对应 `--output_dir` 内）。常见输出文件：

- `training_log.txt`：训练/搜索日志。offset 训练中每个 step 的日志行包含 offsets 和每个 block 实际使用的首 hop 信道（`FirstCh`）；完整的 10 block × 10 hop 真实信道序列以 `HopSequences` 行写入同一文件（仅写文件，不在终端显示）。
- `reward.png` / `ber.png` / `loss.png`：平均 step reward、平均 BER、actor/critic loss 曲线（offset/MBPO 训练）。
- `model_reward.png`：奖励模型预测曲线（MBPO）。
- `holdout_curves/`、`holdout_curves.npz`：MBPO 奖励模型每次拟合内逐 epoch 的 holdout MSE 曲线（每成员一条，含训练前初始评估）。
- `train_curves/`、`train_curves.npz`：同上，为训练损失曲线。
- `model_disagreement.png`、`model_target_saturation_fraction.png`：elite 分歧与训练目标触及物理边界的比例（MBPO）。
- `sac_inference.pt`、`reward_model_inference.pt`：推理 checkpoint（SAC v4 / reward v3），不含 optimizer、replay 或 RNG 状态；旧版本架构会被明确拒绝，必须重新训练。
- `hoprate.png`、`ber.png`、`derivative.png`、`ber_vs_hoprate.png`、`nbs_weights.png`、`nbs_distribution.npz`：NBS 搜索诊断（候选集合、权重、测试 hoprate、BER、导数、MAP/加权估计与收敛状态）。
- `hoprate_sweep.csv`、`hoprate_sweep.npz`：sweep 评估数据。
- `figures/`：由 `PLOT_CONFIG["figure_save_steps"]` 指定的 step 保存动作前 observation 图（`step_XXX_obs.png`）与 10 个 block 的接收 PSD 图（`step_XXX_block_YY.png`）。

`outputs/` 已加入 [.gitignore](.gitignore)，训练产物默认不进入版本控制。

## 7. 配置说明（settings.py）

主要配置集中在 [settings.py](settings.py)，每个配置块带详细中文注释。要点速览：

### 设备与可复现性

- 两个训练入口内部硬编码 CUDA 设备（无 CPU 回退开关）；单元测试与 `train_speed_*.py` 搜索脚本、离线 replay 生成脚本均可在 CPU 运行。
- `set_random_seeds()`：统一设置 Python/NumPy/PyTorch 随机种子，并设 `cudnn.deterministic=True, benchmark=False` 保证可复现。
- `TIMING_ENABLED` / `BACKWARD_TIMING_ENABLED`：MBPO 训练计时剖析开关（后者逐 batch 用 CUDA event 测反向传播，会轻微拖慢训练）。

### `ENV_CONFIG`（传给 `FHSSQPSKEnv(**ENV_CONFIG)`）

- `Startfre`/`Endfre`：FHSS 工作频段（默认 3–4 MHz）。
- `Sub_interval`：子信道间隔（50 kHz，共 20 个信道）。
- `Fs`/`Baud`：采样率 10 MHz、码元率 25 kBaud。
- `Hoprate`、`hoprate_min`、`hoprate_max`：基础跳速与合法范围（10–1000 Hz，按 10 Hz 网格量化）。
- `enable_reactive` / `enable_sweep` / `enable_rayleigh`：各类干扰与衰落的开关。
- `use_pregen`：预生成加速路径。复用 QPSK bits/IQ 和多组干扰波形，但每个 block 的白噪声、每个 hop 的 Rayleigh 衰落与每次 observation 的白噪声都重新生成；observation 每次重新计算 PSD。
- `noise_std`、`signal_power`：接收端噪声与理论信号功率（Baud/Fs），反应式干扰机据此推导检测 SNR 并对 Rayleigh 衰落做 Gauss-Laguerre 平均。
- `mseq_seed` / `mseq_taps` / `mseq_nbits` / `mseq_length`：驱动跳频图案的 m 序列（LFSR）参数。换 m 序列的方法与约束（抽头必须对应本原多项式）见 settings.py 注释；**更换后需重新生成离线 replay**。

### `JAMMER_CONFIG`

- `mode`：`sweep` / `comb` / `both`。
- `baseband_variant_count`：预生成干扰共享的独立带限噪声基带数量（默认 4）；相同带宽的 reactive/sweep/comb 共用同一波形池，随机有放回选择。
- `sweep`：步进 50 kHz、功率 0.8、驻留 4 ms、带宽 50 kHz；默认 20 个频点 × 4 ms = 80 ms 一轮周期。
- `comb`：功率、单 tone 带宽、相位切换周期 `switch_interval`（秒，必须为 1 ms 的正整数倍，默认 0.057 s），以及两组交替干扰信道 `channels_phase0`（偶数信道）/`channels_phase1`（奇数信道）。相位按连续 RF 时钟切换，可在单个 100 ms block 内发生切换；信道序号必须在 `[0, num_channels-1]` 内否则启动即 `ValueError`。**修改后需重新生成离线 replay**。
- `reactive`：功率、带宽、虚警概率 `p_fa`（0.1，用于推导能量检测门限）、检测时隙 `detection_time`（0.5 ms）；按 1 ms 基本时隙 scan→detect→jam，每次实际压制随机选择一个缓存基带变体。

### `SAC_CONFIG` / `BUFFER_CONFIG`

- SAC：`actor_lr 1e-5`、`critic_lr 1e-4`、`alpha_lr 1e-4`、`tau 0.005`、`gamma 0.95`、`target_entropy_ratio 0.1`。
- Buffer：`capacity 20000`、`batch_size 256`（A800 用大 batch 保持卷积编码器满载）。

### `MBPO_CONFIG`

ensemble/elite 数量（5/3）、学习率、拟合频率 `model_train_freq`（每 1 个真实 step 全量重拟合）、`model_train_batch_size 512`、holdout/早停、`rollout_batch_size 1024`、真实样本比例 `real_ratio 0.2`、`model_replay_size 4000`（FIFO）、精度与运行时加速（bfloat16 / fast_math / compile）、曲线图开关与 `model_extra_pool` 架构变体（默认关闭）。详见 [MBPO_MODULE.md](MBPO_MODULE.md)。

### `NBS_CONFIG`

- `p 0.3`：MWU 假设的噪声概率（0 ≤ p < 0.5），越大探索越强。
- `delta 0.01`：收敛阈值，最大权重 ≥ 1−δ 判定收敛。
- `hoprate_step 10.0`：候选 hoprate 离散步长，与环境量化一致。
- `derivative_threshold -0.005`：导数判决阈值。

### `TRAIN_CONFIG`

- `steps_per_episode 80`：单次训练运行的环境 step 数。
- `update_iters_per_step 10`：每个环境 step 后的梯度更新次数。
- `fixed_hoprate 100.0`：offset 训练固定跳速。
- 离线 replay 在首次梯度更新前加载，路径由 `OFFLINE_REPLAY_CONFIG["default_path"]` 或 `--offline_replay_path` 指定。

### `PLOT_CONFIG`

- `figure_save_steps`：需要保存图片的训练 step 序号（1-based，与日志 `Step i/N` 一致，可多个；默认 `[49, 50]`）。命中的 step 保存 `figures/step_XXX_obs.png` 与 `step_XXX_block_01.png`~`step_XXX_block_10.png`；超出 `[1, steps_per_episode]` 的值忽略并告警，空列表关闭该功能。

### `REWARD_CONFIG`

```text
reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
```

默认 `base_reward 10.0`、`ber_penalty 80.0`、`hoprate_penalty 0`。

## 8. 通信环境设计取舍

- `fh_env.py` 中 `PreGeneratedData.common_bits` 及其脉冲成形 I/Q 一次生成并复用；通信 AWGN 每个 block 全新，Rayleigh 衰落每个 hop 独立生成。
- `reset_mseq_each_step=True` 的固定模板训练行为保留。
- `use_pregen=True` 默认复用 QPSK 基带和干扰 RF 变体；每次 observation 加入全新时域白噪声并精确重算 PSD waterfall。
- 相同带宽的干扰共享 `baseband_variant_count` 条基带变体和一个随机选择流：sweep 按完整扫频周期选择，comb 按 phase0+phase1 完整周期选择，reactive 按每次压制选择；随机有放回，相邻周期可能偶然使用同一变体。
- comb 使用两组固定信道交替；连续 RF 时钟包含 observation 和发送 block；默认 57 ms 下一个 100 ms block 内相位会发生切换。
- `env.reset()` 将连续 RF 时钟、comb phase、sweep 位置和干扰变体选择流重置到可复现起点；通信 AWGN/Rayleigh 随机流不回退复用。

如需更严谨的通信对照实验，建议单独比较：`use_pregen=True/False`、固定/连续 m 序列、不同 bits 随机化策略、不同 reward 权重、是否启用 Rayleigh/反应式/扫频干扰组合等。

## 9. 测试

```bash
# 快速套件（默认跳过真实 RF 冒烟）
python -m pytest tests -q
# 包含真实 RF 环境单步冒烟
FHSS_RUN_ENV_SMOKE=1 python -m pytest tests -q        # Git Bash
$env:FHSS_RUN_ENV_SMOKE='1'; python -m pytest tests -q  # PowerShell
```

| 测试文件 | 覆盖内容 |
| --- | --- |
| `tests/test_sac.py` | 多头 SAC 网络、归一化、replay 校验、checkpoint 拒绝旧架构。 |
| `tests/test_offline_replay.py` | v3 replay 序列化/加载/校验规则与 metadata 拒绝行为。 |
| `tests/test_mbpo.py` | 奖励集成模型、混合采样数量解析、rollout 与 checkpoint。 |
| `tests/test_training_and_env.py` | 训练入口参数、训练循环与环境接口（含环境变量门控的 RF 冒烟）。 |
| `tests/test_env_acceleration.py` | 预生成加速路径与逐步环境的等价性。 |
| `tests/test_jammer_timing.py` | comb 相位切换、sweep 周期与反应式时隙时序。 |

## 10. 已知限制和后续建议

- **项目尚未完成**：训练效果验证、算法稳定性仍在推进中。
- 十个动作头采用条件独立的因子化策略，无法直接表达 offset 之间的联合相关性；需要时应另行设计联合 critic 或自回归策略。
- 训练效果对 reward 权重、alpha 初值、target entropy、batch size 等参数敏感。
- MBPO 是 reward-only 一步增强，不是完整 dynamics MBPO；其 next-state 复用依赖"环境 observation transition 与本步 offsets 无关"这一假设。
- MBPO 默认每 1 个在线 step 就用完整真实 replay 重拟合 5 个独立 CNN，计算成本高；快速实验应调大 `model_train_freq`、减小 `max_epochs` 或开启 `model_extra_pool`。
- NBS 跳速搜索依赖 BER-vs-hoprate 的可辨识趋势；若多种强干扰叠加或随机 offset 方差大，需增加步数、调大 `p` 或做多次重复评估。
- v1/v2 block-level replay 与当前 step-level v3 replay 不兼容；SAC inference v1/v2/v3 与 reward-model v1/v2 checkpoint 不能加载到新网络（会被明确拒绝），需重新训练，现有 step-level v3 replay 无需重新生成。
- 若后续要进一步规范工程结构，可做第二阶段重构：拆分 `env/`、`algos/`、`train/` 子包（当前平铺结构是有意保留的，入口清晰）。
