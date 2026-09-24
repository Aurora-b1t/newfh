"""FHSS step-level MBPO 奖励模型子包的公开接口。

本包只暴露奖励模型本体与其 checkpoint 契约常量；训练循环所需的数据适配
（replay → TensorDataset、real/model 混合采样、合成 rollout）刻意不在此 re-export，
需要时显式 ``from r_predict_model.mbpo_adapter import ...``，避免入口脚本
误以为 ``r_predict_model`` 就是训练 API：

    - ``StepRewardEnsemble``：矩阵式概率奖励集成（network_size 个成员、按
      holdout MSE 选 elite_size 个 elite），只预测一步 block reward，不预测
      next state；训练/推理细节见 r_predict_model/model.py；
    - ``REWARD_CHECKPOINT_FORMAT_VERSION``：奖励模型 checkpoint 的格式版本
      （当前 3），加载旧 checkpoint 时用于版本校验；
    - ``REWARD_MODEL_ARCHITECTURE``：架构标识字符串，checkpoint 用它确认
      网络结构与当前代码一致（extra_pool 变体会附加后缀）。

包的内部模块（不对外承诺稳定接口）：
    - model.py：StepRewardEnsemble 及矩阵式成员网络、有界 reward 变换；
    - mbpo_adapter.py：SAC replay 与奖励模型之间的适配层。
"""

# re-export：仅转发 model.py 中的三个公开符号（模型类 + 两个 checkpoint 契约常量），
# 使 train_mbpo.py 可以 ``from r_predict_model import StepRewardEnsemble``。
from .model import (
    REWARD_CHECKPOINT_FORMAT_VERSION,
    REWARD_MODEL_ARCHITECTURE,
    StepRewardEnsemble,
)

# 显式声明公开接口，同时约束 ``from r_predict_model import *`` 的范围；
# 新增公开符号时需同步更新这里。
__all__ = [
    "REWARD_CHECKPOINT_FORMAT_VERSION",
    "REWARD_MODEL_ARCHITECTURE",
    "StepRewardEnsemble",
]
