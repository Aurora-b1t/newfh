# 强化学习跳频抗干扰系统（进行中）

> **状态：项目进行中，尚未完成。** 环境与多个算法入口可运行，但训练效果验证、算法稳定性、部分模块联调仍在推进。下列文档描述的是当前代码快照的真实状态，而非最终目标。

本项目是一个面向跳频扩频（FHSS, Frequency Hopping Spread Spectrum）抗干扰研究的 Python 实验环境。当前代码快照包含：

- **FHSS/QPSK 通信仿真环境**：QPSK 收发链路、跳频信道、PSD waterfall 观测、Rayleigh 衰落、扫频/梳状/反应式干扰机，支持预生成加速。
- **baseline 十头离散 SAC 训练**：一次前向并行生成 10 个 categorical offset，一次环境 step 对应一条完整 replay transition。
- **step-level MBPO 奖励增强**：独立 CNN ensemble 联合预测完整十维 block reward，生成与 v3 replay 同构的一步合成样本辅助 SAC（详见 [MBPO_MODULE.md](MBPO_MODULE.md)）。
- **Noisy Binary Search 跳速阈值搜索**：MWU-based noisy binary search 寻找反应式干扰机的跳速跟踪/失效边界，含 derivative 变体。
- **hoprate sweep 评估**：确定性网格遍历所有候选 hoprate，作为 NBS 搜索的对照基线。
- **离线 replay 生成与加载**：生成/复用 v3 step-level 真实环境 replay，供 baseline offset SAC 冷启动（详见 [OFFLINE_REPLAY.md](OFFLINE_REPLAY.md)）。
- **special hopping pattern 隔离测试**：独立子目录环境/干扰机/训练脚本，用于验证固定 comb 干扰下的可解 offset 模式。

项目根目录采用平铺结构，入口清晰、文件职责清晰。每个训练/搜索脚本自带默认输出目录，互不覆盖。

## 项目文件说明

### 根目录

| 文件 | 作用 |
| --- | --- |
| [settings.py](settings.py) | 统一配置环境参数、干扰机参数、SAC/MBPO/NBS 超参数、训练循环和奖励系数。 |
| [fh_env.py](fh_env.py) | FHSS/QPSK Gymnasium 环境：预生成加速、跳频序列、QPSK 收发、干扰/衰落叠加、BER 与 reward 计算、PSD waterfall 观测。 |
| [jammers.py](jammers.py) | 干扰机实现：快速带限噪声源、基于能量检测的反应式干扰机、扫频/梳状宽带干扰机。 |
| [SAC.py](SAC.py) | 十头离散 SAC：step-level ReplayBuffer、共享编码器实现、独立 actor/Q 输出头、温度系数自适应和软更新。 |
| [train_offsets.py](train_offsets.py) | baseline 训练入口：构建环境、SAC agent、replay buffer、训练循环、日志和曲线输出。 |
| [train_mbpo.py](train_mbpo.py) | step-level SAC + MBPO 入口：真实交互、CNN 奖励 ensemble、合成 replay、混合更新、诊断与推理 checkpoint。 |
| [train_speed.py](train_speed.py) | Noisy Binary Search 跳速阈值搜索入口（随机 offset + 反应式干扰机）。 |
| [train_speed_derivative.py](train_speed_derivative.py) | derivative-based NBS 变体入口，用 BER-hoprate 导数指标做方向决策。 |
| [train_speed_sweep.py](train_speed_sweep.py) | hoprate 网格扫描评估入口，NBS 搜索的确定性对照。 |
| [noisy_binary_search.py](noisy_binary_search.py) | MWU-based noisy binary search 算法实现，参考 Dereniowski et al. STACS 2025。 |
| [noisy_binary_search_derivative.py](noisy_binary_search_derivative.py) | derivative-based NBS 算法实现。 |
| [offline_replay.py](offline_replay.py) | v3 step-level replay 序列化、严格校验与加载工具。 |
| [generate_offline_replay.py](generate_offline_replay.py) | 离线真实 replay 生成脚本，输出 `.npz` 文件。 |
| [environment.yml](environment.yml) | Python 3.11 核心运行依赖；CUDA 版 Torch 需按机器环境选择安装方式。 |
| `tests/` | 标准库 `unittest` 测试，覆盖多头 SAC、replay、环境接口和兼容性边界。 |

### `r_predict_model/` 子包

| 文件 | 作用 |
| --- | --- |
| [r_predict_model/model.py](r_predict_model/model.py) | `StepRewardEnsemble`：成员参数完全独立的 CNN 概率奖励模型、全量拟合、holdout/elite 选择与 checkpoint。 |
| [r_predict_model/mbpo_adapter.py](r_predict_model/mbpo_adapter.py) | v3 replay 适配、完整动作 rollout、reward 裁界及真实/合成 batch 混合。 |
| [r_predict_model/__init__.py](r_predict_model/__init__.py) | 暴露 `StepRewardEnsemble`。 |

### `special_hopping_test/` 子目录

独立隔离测试套件：固定 comb 干扰下两组交替信道相位 + 特殊跳频模式，用于验证 agent 能否学到可解的 10-offset 模式。使用 20 信道、不同 reward 配置，与主目录环境互不依赖。

| 文件 | 作用 |
| --- | --- |
| [special_hopping_test/SAC_test.py](special_hopping_test/SAC_test.py) | 隔离测试专用 SAC 实现（与根目录 `SAC.py` 结构一致）。 |
| [special_hopping_test/fh_env_test.py](special_hopping_test/fh_env_test.py) | 隔离测试专用 FHSS 环境。 |
| [special_hopping_test/jammers_test.py](special_hopping_test/jammers_test.py) | 隔离测试专用干扰机实现。 |
| [special_hopping_test/test_settings.py](special_hopping_test/test_settings.py) | 隔离测试专用配置。 |
| [special_hopping_test/train_test.py](special_hopping_test/train_test.py) | 隔离测试训练入口。 |
| [special_hopping_test/validate_pattern.py](special_hopping_test/validate_pattern.py) | 不跑 RF 仿真的纯逻辑校验：验证特殊跳频/comb 构造的碰撞数与期望 offset。 |
| [special_hopping_test/validate_psd.py](special_hopping_test/validate_psd.py) | 用真实 RF 配置生成并校验 comb 干扰 PSD。 |

### 其它

| 文件 | 作用 |
| --- | --- |
| [example/main.py](example/main.py) | 通用 SAC + MBPO 参考训练模板（非 FHSS 入口）。 |
| [example/model.py](example/model.py)、[example/replay_memory.py](example/replay_memory.py) | 参考 MBPO 模型与 replay memory。 |
| `pdf/` | 参考文献 PDF：Dereniowski et al. STACS 2025、Urkowitz 能量检测。 |

历史文件说明：旧训练/启动脚本（如 `run.py`、`speedDQN.py`、`Jamming Strategy.md` 等）已不作为当前维护入口，如需查看可通过 git 历史找回。

## 运行环境

已验证的本机解释器为：

```bash
D:\Anaconda\envs\rl_fhss\python.exe
```

也可以先创建项目环境：

```bash
conda env create -f environment.yml
conda activate rl_fhss
```

代码中主要使用：

- `numpy` / `matplotlib` / `scipy` / `gymnasium` / `torch` / `scikit-commpy`

`environment.yml` 中的 Torch 是通用版本声明。需要 GPU 时，请根据本机驱动与 CUDA 版本使用 PyTorch 官方安装命令替换；本项目当前验证版本为 `torch 2.9.1+cu130`。

运行快速单元测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe -m unittest discover -s tests -v
```

运行包含真实 RF 环境单步的测试：

```powershell
$env:FHSS_RUN_ENV_SMOKE='1'
D:\Anaconda\envs\rl_fhss\python.exe -m unittest discover -s tests -v
```

建议在项目根目录运行所有命令。以下示例均使用上述 Python 解释器，可替换为你环境中的等价路径。

## Baseline：离散 SAC offset 训练

查看参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --help
```

短训练冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --steps_per_episode 3 --batch_size 20 --offline_replay_path outputs/offline_replay/replay_50000_random_hoprate_v3.npz --output_dir outputs/smoke
```

默认训练（默认输出目录 `outputs/offsets/pre50000/comb/512_start`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py
```

强制 CPU：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --cpu_only
```

指定输出目录：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --output_dir outputs/baseline
```

显式跳过离线 replay，从在线数据 warm-up：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --offline_replay_path none
```

## MBPO 奖励模型

[train_mbpo.py](train_mbpo.py) 使用与十头 SAC 完全相同的 step-level v3 transition。奖励模型输入当前 PSD、实际 hoprate 和完整 offsets 向量，联合输出每个 block 的 reward 均值与方差；它不预测下一 PSD，而是复用真实 replay 的外生 next state、next hoprate 和 done。

默认模型包含 5 个参数完全独立的 CNN 成员，并选择 3 个 holdout MSE 最低的 elite。每条合成 transition 随机选择一个 elite 采样完整 reward 向量，再按当前 reward 公式的物理范围裁剪。默认每个真实环境 step 后使用当前全部真实 replay 继续拟合，因此 50,000 条离线数据下计算成本较高。

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --help
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py
```

纯在线缩小 smoke：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_mbpo.py --offline_replay_path none --steps_per_episode 2 --batch_size 2 --model_train_batch_size 2 --num_networks 2 --num_elites 1 --pred_hidden_size 16 --model_max_epochs 1 --model_patience 0 --rollout_batch_size 2 --update_iters_per_step 1 --output_dir outputs/mbpo_smoke
```

MBPO 默认严格拒绝环境、干扰或 reward metadata 不匹配的离线 replay；跨配置实验必须显式使用 `--allow_replay_config_mismatch`。完整设计与参数见 [MBPO_MODULE.md](MBPO_MODULE.md)。

## 跳速阈值搜索（Noisy Binary Search）

[train_speed.py](train_speed.py) 使用 [noisy_binary_search.py](noisy_binary_search.py) 中的 MWU-based noisy binary search 算法，在反应式干扰机开启时搜索跳速阈值。每个环境 step 中：

1. NBS 给出一个待测试 hoprate。
2. 训练脚本随机生成 10 个 offset（不训练 SAC）。
3. 环境执行 10 个 block 并返回 BER。
4. NBS 根据当前 BER 与上一次 BER 的变化更新候选跳速权重分布。

运行示例（默认输出目录 `outputs/speed`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed.py --steps 60 --output_dir outputs/speed_test
```

自定义 NBS 参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed.py --nbs_p 0.15 --nbs_delta 0.03 --nbs_step 10 --steps 100 --output_dir outputs/speed_custom
```

### derivative NBS 变体

[train_speed_derivative.py](train_speed_derivative.py) 使用 [noisy_binary_search_derivative.py](noisy_binary_search_derivative.py) 的 derivative-based NBS 算法，用 BER-hoprate 导数指标代替单纯 BER 升降做方向决策：

```
metric = ΔBER_percent / Δhoprate_clamped
metric > threshold  → LEFT move（支持更小 hoprate）
metric ≤ threshold  → RIGHT move（支持更大 hoprate）
```

运行示例（默认输出目录 `outputs/speed_derivative/0.5ms/-0.005`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_derivative.py --steps 60 --output_dir outputs/speed_test_derivative
D:\Anaconda\envs\rl_fhss\python.exe train_speed_derivative.py --derivative_threshold -0.005 --steps 100
```

### hoprate 网格扫描评估

[train_speed_sweep.py](train_speed_sweep.py) 是 [train_speed.py](train_speed.py) 的确定性网格对照：不使用 NBS 选择下一个 hoprate，而是按升序遍历每个候选 hoprate，记录 BER/reward 诊断。每个环境 step 内部执行 10 个 block，与训练脚本的环境 step 语义一致。

运行示例（默认输出目录 `outputs/speed_sweep/0.5ms`）：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_sweep.py --output_dir outputs/speed_sweep
```

冒烟测试：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_speed_sweep.py --hoprate_max 20 --steps_per_hoprate 1
```

### 离线 replay 生成与加载

offset baseline 在首次梯度更新前加载真实 v3 replay。默认数据包含 50,000 条完整环境 step transition，每条都带 10 个 offset 与 10 个 block reward：

修改 comb `switch_interval`、相位信道组、`baseband_variant_count` 或其他干扰配置后，必须重新生成离线 replay。metadata 会记录新配置；baseline 会提示不匹配，MBPO 默认拒绝加载不匹配的数据（仍可显式使用 `--allow_replay_config_mismatch` 做跨配置实验）。

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py
```

固定 hoprate 版本：

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --hoprate_mode fixed --fixed_hoprate 100 --output_path outputs/offline_replay/replay_50000_fixed_100_v3.npz
```

指定 replay 文件给训练入口：

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --offline_replay_path outputs/offline_replay/replay_50000_fixed_100_v3.npz
```

生成少量冒烟数据时使用新的数量参数：

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --num_step_transitions 2 --output_path outputs/offline_replay/smoke_v3.npz
```

### special hopping pattern 隔离测试

在 `special_hopping_test/` 目录下独立运行（它有独立的 settings/env/jammers）：

```bash
cd special_hopping_test
D:\Anaconda\envs\rl_fhss\python.exe train_test.py --steps_per_episode 250 --output_dir results
```

纯逻辑校验（不跑 RF 仿真）：

```bash
cd special_hopping_test
D:\Anaconda\envs\rl_fhss\python.exe validate_pattern.py
```

PSD 校验：

```bash
cd special_hopping_test
D:\Anaconda\envs\rl_fhss\python.exe validate_psd.py
```

### NBS / 搜索主要输出

- `training_log.txt`：搜索日志。
- `hoprate.png`：实际测试 hoprate 与 NBS 估计轨迹。
- `ber.png`：每 step 平均 BER。
- `ber_vs_hoprate.png`：BER 与 hoprate 散点图。
- `nbs_weights.png`：最终候选 hoprate 权重分布。
- `nbs_distribution.npz`：候选集合、权重、测试 hoprate、BER 的 numpy 数据。
- `hoprate_sweep.csv` / `hoprate_sweep.npz`：sweep 评估的 CSV 和 numpy 数据。

相关参考：

- [Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf](<pdf/Dereniowski 等 - 2025 - Noisy (Binary) Searching Simple, Fast and Correct.pdf>)
- [Energy_detection_of_unknown_deterministic_signals.pdf](pdf/Energy_detection_of_unknown_deterministic_signals.pdf)

## 输出目录

输出目录不再集中在 [settings.py](settings.py)，而是由各训练/搜索脚本自带默认 `--output_dir`，均可通过命令行覆盖：

| 脚本 | 默认 `--output_dir` |
| --- | --- |
| [train_offsets.py](train_offsets.py) | `outputs/offsets/pre50000/comb/512_start` |
| [train_mbpo.py](train_mbpo.py) | `outputs/mbpo/comb/pre50000` |
| [train_speed.py](train_speed.py) | `outputs/speed` |
| [train_speed_derivative.py](train_speed_derivative.py) | `outputs/speed_derivative/0.5ms/-0.005` |
| [train_speed_sweep.py](train_speed_sweep.py) | `outputs/speed_sweep/0.5ms` |
| [special_hopping_test/train_test.py](special_hopping_test/train_test.py) | `special_hopping_test/results` |

每个脚本还支持 `--log_file`（默认 `training_log.txt`，位于对应 `--output_dir` 内）。常见输出文件包括：

- `training_log.txt`：训练或搜索日志。offset 训练中每个 step 的日志行包含 offsets 和每个 block 实际使用的首 hop 信道（`FirstCh`），完整的 10 block × 10 hop 真实信道序列也以 `HopSequences` 行写入同一文件（仅写文件，不在终端显示）。
- `reward.png`：平均 step reward 曲线（offset/MBPO 训练）。
- `ber.png`：平均 step BER 曲线。
- `loss.png`：actor/critic loss 曲线（offset/MBPO 训练）。
- `model_reward.png`：奖励模型预测曲线（MBPO 训练）。
- `model_holdout.png`、`model_disagreement.png`、`model_clipped_fraction.png`：MBPO 奖励模型拟合质量、elite 分歧和物理边界裁剪率。
- `sac_inference.pt`、`reward_model_inference.pt`：MBPO 推理 checkpoint，不含 optimizer、replay 或 RNG 状态。
- `hoprate.png`、`ber_vs_hoprate.png`、`nbs_weights.png`：NBS 搜索诊断图。
- `hoprate_sweep.csv`、`hoprate_sweep.npz`：sweep 评估数据。
- `figures/`：由 `PLOT_CONFIG["figure_save_steps"]` 指定的 step 保存的动作前 observation 图与 10 个 block PSD 图（offset/MBPO 训练）。
- PSD capture 图：指定 step 的观测与 10 个 block PSD（special hopping 测试）。

`outputs/` 已加入 [.gitignore](.gitignore)，训练产物默认不进入版本控制。

## 配置说明

主要配置集中在 [settings.py](settings.py)。

### 设备

- `CPU_ONLY`：是否强制使用 CPU。
- `set_random_seeds()`：统一设置 Python/NumPy/PyTorch 随机种子，并设置 `cudnn.deterministic=True, benchmark=False` 以保证可复现性。各训练入口不再单独覆盖此设置。

### `ENV_CONFIG`

传给 `FHSSQPSKEnv(**ENV_CONFIG)` 的环境参数，主要包括：

- `Startfre` / `Endfre`：FHSS 工作频段。
- `Sub_interval`：子信道间隔。
- `Fs`：采样率。
- `Baud`：码元率。
- `Hoprate`、`hoprate_min`、`hoprate_max`：基础跳速和跳速范围。
- `enable_reactive`：是否启用反应式干扰机。
- `enable_sweep`：是否启用扫频/梳状干扰机。
- `enable_rayleigh`：是否启用 Rayleigh 衰落。
- `use_pregen`：是否使用预生成加速路径。该路径复用 QPSK bits/IQ 和多组干扰波形，但每个通信 block 的白噪声、每个 hop 的 Rayleigh 衰落以及每次 observation 的白噪声都会重新生成；observation 会重新计算 PSD，不再复用周期 waterfall 图。
- `noise_std`、`signal_power`：接收端噪声和反应式干扰检测所需信号功率参数。
- `mseq_seed`、`mseq_taps`、`mseq_nbits`、`mseq_length`：驱动基础跳频图案的 m 序列（LFSR）参数。换一份 m 序列的方法：
  - 改 `mseq_seed`（初始状态）：同一 m 序列的不同相位（循环移位），最简单；
  - 改 `mseq_taps`（反馈抽头）：得到真正不同的 m 序列，抽头必须对应本原多项式（如 10 级的 `(10, 7)`、`(10, 3)`），否则周期骤减；
  - 改 `mseq_nbits`（寄存器级数）：改变周期（2ⁿ−1），同时 `mseq_taps` 必须换成对应级数的本原抽头。
  - 注意：更换 m 序列后需重新生成离线 replay 数据（或用 `--offline_replay_path none`）；baseline 会警告 metadata 差异，MBPO 默认直接拒绝。

### `JAMMER_CONFIG`

干扰机配置：

- `mode`：`sweep`、`comb` 或 `both`。
- `baseband_variant_count`：预生成干扰使用的独立带限噪声基带数量，必须为正整数，默认 4；相同带宽的 reactive / sweep / comb 共享同一波形池。
- `sweep`：扫频干扰的步进、功率、驻留时间、噪声带宽。预生成路径为每个基带变体缓存一轮自然扫频周期；每轮周期随机有放回选择一个变体。默认 20 个频点 × 4 ms，即 80 ms。
- `comb`：梳状干扰的功率、单 tone 带宽、相位切换周期 `switch_interval`，以及两组交替的干扰信道序号 `channels_phase0` / `channels_phase1`（默认为偶数/奇数各 8 个 50 kHz 对齐信道）。一个基带变体连续贯穿 phase 0 和 phase 1，并在下一个完整周期重新随机选择。`switch_interval` 使用秒，必须是有限正数且为 10 ms 的整数倍；默认 0.05 秒。相位按连续 RF 时钟切换，因此可以在一个 100 ms block 内发生切换。信道序号必须是 `[0, num_channels-1]` 范围内的整数，否则环境启动时直接报 `ValueError`；两组长度可以不同、允许重叠。修改后同样需重新生成离线 replay 数据。
- `reactive`：反应式干扰机的功率、带宽、虚警概率、检测时长等。检测逻辑基于能量检测理论，按 slot 扫描/检测/压制；每次实际压制都会随机选择一个缓存基带变体。

### `SAC_CONFIG`

baseline SAC 超参数：`actor_lr`、`critic_lr`、`alpha_lr`、`tau`、`gamma`、`target_entropy_ratio`。

### `MBPO_CONFIG`

MBPO 奖励模型配置包括 ensemble/elite 数量、CNN 后端 MLP 宽度、学习率、权重衰减、全量拟合频率、holdout/早停、rollout batch、真实样本比例和 model replay 容量，详见 [MBPO_MODULE.md](MBPO_MODULE.md)。

### `BUFFER_CONFIG`

普通 replay buffer 配置：`capacity`、`batch_size`。

### `NBS_CONFIG`

Noisy Binary Search 配置：

- `p`：假设的噪声概率，需满足 `0 <= p < 0.5`。
- `delta`：收敛阈值；当最大权重 `>= 1 - delta` 时认为收敛。
- `hoprate_step`：候选 hoprate 离散步长，默认 10 Hz，与环境 `_apply_hoprate()` 的量化一致。

### `TRAIN_CONFIG`

训练循环配置：

- `steps_per_episode`：单次训练运行的环境 step 数。
- `update_iters_per_step`：每个环境 step 后的梯度更新次数。
- `fixed_hoprate`：offset 训练时使用的固定 hoprate。
- 离线 replay 在首次梯度更新前加载，文件路径由 `OFFLINE_REPLAY_CONFIG` 或 `--offline_replay_path` 配置。

### `PLOT_CONFIG`

指定 step 图片保存配置（`train_offsets.py` 与 `train_mbpo.py` 使用）：

- `figure_save_steps`：需要保存图片的训练 step 序号列表（1-based，与日志中 `Step i/N` 一致），可填多个。在命中的 step 保存：
  - `figures/step_XXX_obs.png`：该 step 动作前的 observation（即 agent 决策所见的 100 ms PSD waterfall）；
  - `figures/step_XXX_block_01.png` ~ `step_XXX_block_10.png`：该 step 内 10 个 block 各自的接收信号 PSD 图。
  - 图片保存在对应 `--output_dir` 的 `figures/` 子目录下；超出 `[1, steps_per_episode]` 的值会被忽略并告警，空列表则完全关闭该功能。

### `REWARD_CONFIG`

训练脚本中逐 block reward 的计算参数：

```text
reward = base_reward - ber_penalty * BER - hoprate_penalty * hoprate
```

环境为每个 block 使用上式生成 `info["block_rewards"]`，并将其均值作为 `env.step()` 返回的 step reward。baseline、离线生成器和 hoprate sweep 因而共享同一奖励语义。

## 训练语义

当前 offset 训练流程如下：

1. 环境 `reset()` 返回 100 ms PSD waterfall observation。
2. 训练脚本固定 hoprate。
3. SAC actor 输入当前 observation 与 hoprate，一次前向输出 `[10, num_channels]` raw logits，并从十个条件独立的 categorical head 采样 offset。
4. 环境一次性执行这 10 个 offset，对应 10 个 100 ms block。
5. 环境返回下一个 observation、`info["ber_blocks"]`、`info["block_rewards"]`、`info["comb_phases"]` 及其均值 reward；`comb_phases` 的形状为 10 个 block × 10 个 10 ms 槽，每个值表示对应半开时间槽起点生效的 phase。
6. 训练脚本将整组 `actions[10]` 与 `block_rewards[10]` 写为一条 replay transition。

当 SAC/MBPO 实际启用 comb 或 both 模式时，启动日志会在控制台和 `training_log.txt` 中各记录一次实际 `channels_phase0` / `channels_phase1` 配置；逐 step 日志仍不重复输出该配置。

critic 同样输出十个离散 Q 头。每个头使用对应的即时 block reward；下一状态 continuation 先对每个头做离散 SAC 期望，再对十头求均值并广播，从而学习平均 block return，而不再构造虚假的 block 间状态转移。

## SAC 实现要点

当前 [SAC.py](SAC.py) / [SAC_test.py](special_hopping_test/SAC_test.py) 的实现要点（近期已修复若干问题）：

- **actor、两个 online critic 与两个 target critic 参数完全独立**；它们只复用 `StateEncoder` 类定义，不共享权重。
- **不再输入 block index**：hoprate 按环境上下界归一化到 `[-10,10]`，十个固定位置的输出头分别对应 block 0～9。
- **actor 只返回 raw logits**：`take_action()` 单次返回十维整数动作；训练采样用 `Categorical(logits=...)`，确定性推理使用逐头 `argmax`。
- **共享温度系数**：十头共用一个 alpha，actor、critic 和 entropy loss 都对 batch/head 维取平均。
- **目标网络固定为 eval 模式**：`target_critic_1/2` 始终用 BatchNorm running stats，不随 batch 抖动。
- **Lazy target 延迟初始化**：首次 TD target 计算前先物化 online critics，再完整复制参数和 BatchNorm buffers；后续只做 soft update。
- **`soft_update` 同步 BatchNorm buffers**（running_mean/running_var/num_batches_tracked）到目标网络。
- **`calc_target` 在 `torch.no_grad()` 下、actor 切 eval** 计算 next-state 目标，避免建图/污染 BN stats。
- **actor loss 计算时 critic 切 eval**，事后恢复 train，让策略梯度基于稳定的 running stats。
- **step-level replay**：一条经验保存十维动作、十维 block reward、仅用于诊断的均值 step reward，以及真实下一环境状态。

## 通信环境设计取舍

- [fh_env.py](fh_env.py) 中 `PreGeneratedData.common_bits` 及其脉冲成形 I/Q 生成一次并复用；通信 AWGN 每个 block 全新，Rayleigh 衰落每个 hop 独立生成并遵守 `rayleigh_coherence`。
- `reset_mseq_each_step=True` 的固定模板训练行为保留。
- `use_pregen=True` 默认复用 QPSK 基带和干扰 RF 变体；每次 observation 都加入全新时域白噪声并精确重算 PSD waterfall。
- 相同带宽的干扰共享 4 条默认基带变体和一个随机选择流。sweep 按完整扫频周期选择，comb 按 phase0+phase1 完整周期选择，reactive 按每次压制选择；随机有放回，因此相邻周期可能偶然使用同一变体。
- comb 干扰使用两组固定 8 信道频点交替：phase 0 使用偶数信道组，phase 1 使用奇数信道组。连续 RF 时钟包含 observation 和发送 block；默认 50 ms 下首个 step 的每个 block 都在前五个 10 ms 槽使用 phase 0、后五个槽使用 phase 1，即 `[0,0,0,0,0,1,1,1,1,1]`。
- `env.reset()` 将连续 RF 时钟、comb phase、sweep 位置和干扰变体选择流重置到可复现起点；通信 AWGN/Rayleigh 随机流不会因此回退复用。

如需做更严谨的通信环境对照实验，建议后续单独比较：`use_pregen=True/False`、固定/连续 m-sequence、不同 bits 随机化策略、不同 reward 权重、是否启用 Rayleigh/反应式/扫频干扰组合等。

## 已知限制和后续建议

- **项目尚未完成**：训练效果验证、算法稳定性、多模块联调仍在推进中。
- 十个动作头采用条件独立的因子化策略，无法直接表达 offset 之间的联合动作相关性；需要这类能力时应另行设计联合 critic 或自回归策略。
- 训练效果对 reward 权重、alpha 初值、target entropy、batch size 等参数敏感。
- MBPO 是 reward-only 一步增强，不是完整 dynamics MBPO；其 next-state 复用依赖环境 observation transition 与 offsets 无关。
- MBPO 默认每个在线 step 都用完整真实 replay 继续训练 5 个独立 CNN，计算成本高；需要快速实验时应调大 `model_train_freq` 或降低 `model_max_epochs`。
- NBS 跳速搜索依赖 BER-vs-hoprate 的可辨识趋势；若同时启用多种强干扰或随机 offset 方差很大，可能需要增加步数、调大 `p` 或做多次重复评估。
- v1/v2 block-level replay 和旧 SAC checkpoint 与当前网络拓扑不兼容，必须重新生成数据并重新训练。
- 若后续要进一步规范工程结构，可以再做第二阶段重构：拆分 `env/`、`algos/`、`train/` 子包。
