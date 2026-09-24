"""SAC.py 十头离散 SAC 的单元测试。

覆盖内容：
- StateEncoder / PolicyNet / ValueNet 的结构约定：hoprate 归一化、
  (batch, num_heads, n_actions) 输出形状、逐头独立参数、以 GroupNorm
  替代 BatchNorm 且无 Dropout、紧凑两层 CNN 与多种 PSD 输入尺寸
  （含生产尺寸 100 的参数量冻结断言）；
- take_action / take_actions 的合法动作采样，以及确定性推理不临时
  翻转 actor 训练模式的保证；
- SAC 训练逻辑：TD target 对十个头取全局均值、update() 返回有限统计
  量、在线网络在 update 过程中保持 training 模式。

测试策略：纯 CPU 单元测试，无需 GPU，也无环境变量门控；
FixedActor / FixedCritic 提供输出可解析计算的常数替身，配合
forward hook 与固定随机种子验证数值和模式行为。

单独运行（在项目根目录）：
    "D:\\Anaconda\\envs\\rl_fhss\\python.exe" -m pytest tests/test_sac.py -q
    "D:\\Anaconda\\envs\\rl_fhss\\python.exe" -m unittest discover -s tests -p test_sac.py -v
"""

import math
import unittest

import numpy as np
import torch
from torch import nn

from SAC import (
    HOPRATE_FEATURE_DIM,
    PSD_FEATURE_DIM,
    STATE_FEATURE_DIM,
    PolicyNet,
    SAC,
    StateEncoder,
    ValueNet,
    normalize_hoprate,
)


class FixedActor(nn.Module):
    """输出全零 logits 的 actor 替身，使策略为均匀分布，便于解析计算。"""

    def __init__(self, num_heads, n_actions):
        super().__init__()
        self.num_heads = num_heads
        self.n_actions = n_actions

    def forward(self, images, hoprates):
        return torch.zeros(
            images.shape[0],
            self.num_heads,
            self.n_actions,
            dtype=images.dtype,
            device=images.device,
        )


class FixedCritic(nn.Module):
    """按头编号取常数 Q 值的 critic 替身：第 k 个头的 Q 恒为 k。"""

    def __init__(self, num_heads, n_actions):
        super().__init__()
        values = torch.arange(num_heads, dtype=torch.float32)
        self.register_buffer(
            "values", values.view(1, num_heads, 1).repeat(1, 1, n_actions)
        )

    def forward(self, images, hoprates):
        return self.values.repeat(images.shape[0], 1, 1)


def make_agent(n_actions=5, gamma=0.95):
    """构造 CPU 上的十头 SAC 测试实例（默认 5 个动作）。"""
    return SAC(
        n_actions=n_actions,
        num_heads=10,
        hoprate_min=10.0,
        hoprate_max=1000.0,
        actor_lr=1e-4,
        critic_lr=1e-4,
        alpha_lr=1e-4,
        target_entropy=math.log(n_actions) * 0.1,
        tau=0.005,
        gamma=gamma,
        device="cpu",
    )


def make_transition_batch(batch_size=4, n_actions=5, seed=7):
    """用固定种子生成一个完整 transition batch，step_rewards 为逐块奖励均值。"""
    rng = np.random.default_rng(seed)
    states = rng.normal(size=(batch_size, 16, 16)).astype(np.float32)
    block_rewards = rng.normal(size=(batch_size, 10)).astype(np.float32)
    return {
        "state_imgs": states,
        "hoprates": np.full(batch_size, 100.0, dtype=np.float32),
        "actions": rng.integers(
            0, n_actions, size=(batch_size, 10), dtype=np.int64
        ),
        "block_rewards": block_rewards,
        "step_rewards": block_rewards.mean(axis=1),
        "next_state_imgs": states + 0.1,
        "next_hoprates": np.full(batch_size, 110.0, dtype=np.float32),
        "dones": np.zeros(batch_size, dtype=np.float32),
    }


class MultiHeadNetworkTests(unittest.TestCase):
    """网络结构单元测试：StateEncoder / PolicyNet / ValueNet。

    验证 hoprate 归一化的缩放与越界裁剪、actor/critic 输出形状与逐头
    独立参数、GroupNorm（禁 BatchNorm/Dropout）的架构约束、StateEncoder
    的紧凑结构约定与多种 PSD 尺寸（含生产尺寸 100 的参数量断言）、
    train/eval 输出逐位一致，以及 take_action / take_actions 的合法
    采样与 actor 模式不被临时翻转。
    """

    def test_hoprate_normalization_keeps_requested_scale(self):
        """归一化把 [hoprate_min, hoprate_max] 映射到 [-10, 10]，越界值裁剪到边界。"""
        values = torch.tensor([[10.0], [505.0], [1000.0], [2000.0]])
        normalized = normalize_hoprate(values, 10.0, 1000.0)
        np.testing.assert_allclose(
            normalized.numpy().ravel(), [-10.0, 0.0, 10.0, 10.0], atol=1e-6
        )

    def test_actor_and_critic_shapes(self):
        images = torch.randn(3, 1, 16, 16)
        hoprates = torch.full((3, 1), 100.0)
        actor = PolicyNet(5, num_heads=10)
        critic = ValueNet(5, num_heads=10)

        logits = actor(images, hoprates)
        q_values = critic(images, hoprates)

        self.assertEqual((3, 10, 5), tuple(logits.shape))
        self.assertEqual((3, 10, 5), tuple(q_values.shape))
        self.assertIsNot(actor.action_heads[0].weight, actor.action_heads[1].weight)
        self.assertIsNot(critic.q_heads[0].weight, critic.q_heads[1].weight)

    def test_actor_and_critic_use_group_norm_without_batch_norm(self):
        networks = (PolicyNet(5, num_heads=10), ValueNet(5, num_heads=10))
        modules = [module for network in networks for module in network.modules()]

        self.assertFalse(
            any(
                isinstance(module, nn.modules.batchnorm._BatchNorm)
                for module in modules
            )
        )
        self.assertFalse(any(isinstance(module, nn.Dropout) for module in modules))
        self.assertEqual(4, sum(isinstance(module, nn.GroupNorm) for module in modules))

    def test_state_encoder_has_requested_compact_architecture(self):
        """冻结 encoder 紧凑结构（16/32 通道、无 conv3、特征维度常量），并用 forward hook 断言池化输出形状为 (2, 32, 8, 8)。"""
        encoder = StateEncoder(10.0, 1000.0)

        self.assertEqual((16, 32), (
            encoder.conv1.out_channels,
            encoder.conv2.out_channels,
        ))
        self.assertEqual((4, 8), (
            encoder.norm1.num_groups,
            encoder.norm2.num_groups,
        ))
        self.assertFalse(hasattr(encoder, "conv3"))
        self.assertEqual(PSD_FEATURE_DIM, encoder.conv_fc.out_features)

        hoprate_linears = [
            module
            for module in encoder.hoprate_embedding
            if isinstance(module, nn.Linear)
        ]
        self.assertEqual([(1, HOPRATE_FEATURE_DIM)], [
            (module.in_features, module.out_features)
            for module in hoprate_linears
        ])
        fusion_linears = [
            module for module in encoder.fusion if isinstance(module, nn.Linear)
        ]
        self.assertEqual(
            [
                (PSD_FEATURE_DIM + HOPRATE_FEATURE_DIM, STATE_FEATURE_DIM),
            ],
            [(module.in_features, module.out_features) for module in fusion_linears],
        )

        pool_calls = []
        handle = encoder.pool.register_forward_hook(
            lambda _module, _inputs, output: pool_calls.append(tuple(output.shape))
        )
        try:
            output = encoder(torch.randn(2, 1, 16, 16), torch.full((2, 1), 100.0))
        finally:
            handle.remove()
        self.assertEqual((2, STATE_FEATURE_DIM), tuple(output.shape))
        self.assertEqual([(2, 32, 8, 8)], pool_calls)

    def test_state_encoder_supports_test_and_production_psd_sizes(self):
        """encoder 需兼容测试小尺寸（8/16）与生产 PSD 尺寸 100，并冻结生产尺寸下的总参数量 41_113_248。"""
        for size in (8, 16, 100):
            with self.subTest(size=size):
                encoder = StateEncoder(10.0, 1000.0)
                output = encoder(
                    torch.zeros(1, 1, size, size),
                    torch.full((1, 1), 100.0),
                )
                self.assertEqual((1, STATE_FEATURE_DIM), tuple(output.shape))
                if size == 100:
                    self.assertEqual(
                        41_113_248,
                        sum(parameter.numel() for parameter in encoder.parameters()),
                    )

    def test_train_and_eval_outputs_are_identical(self):
        images = torch.randn(3, 1, 16, 16)
        hoprates = torch.full((3, 1), 100.0)

        for network in (PolicyNet(5, num_heads=10), ValueNet(5, num_heads=10)):
            network.train()
            with torch.no_grad():
                training_output = network(images, hoprates)
            network.eval()
            with torch.no_grad():
                inference_output = network(images, hoprates)
            torch.testing.assert_close(
                training_output, inference_output, rtol=0.0, atol=0.0
            )

    def test_take_action_returns_ten_valid_offsets(self):
        agent = make_agent()
        state = np.random.randn(16, 16).astype(np.float32)

        sampled = agent.take_action(state, 100.0)
        deterministic = agent.take_action(state, 100.0, deterministic=True)

        self.assertEqual((10,), sampled.shape)
        self.assertEqual((10,), deterministic.shape)
        self.assertEqual(np.int64, sampled.dtype)
        self.assertTrue(np.all((sampled >= 0) & (sampled < 5)))
        self.assertTrue(np.all((deterministic >= 0) & (deterministic < 5)))

    def test_take_actions_returns_one_action_vector_per_state(self):
        agent = make_agent()
        states = np.random.randn(4, 16, 16).astype(np.float32)
        actions = agent.take_actions(states, np.full(4, 100.0))

        self.assertEqual((4, 10), actions.shape)
        self.assertEqual(np.int64, actions.dtype)
        self.assertTrue(np.all((actions >= 0) & (actions < 5)))

    def test_take_action_does_not_temporarily_flip_actor_mode(self):
        """用 forward pre-hook 记录 actor.training，确保确定性推理不会临时切换 actor 的 train/eval 模式。"""
        agent = make_agent()
        state = np.random.randn(16, 16).astype(np.float32)
        for requested_mode in (True, False):
            forward_modes = []
            handle = agent.actor.register_forward_pre_hook(
                lambda module, _inputs: forward_modes.append(module.training)
            )
            try:
                agent.actor.train(requested_mode)
                agent.take_action(state, 100.0, deterministic=True)
            finally:
                handle.remove()

            self.assertEqual([requested_mode], forward_modes)
            self.assertEqual(requested_mode, agent.actor.training)


class MultiHeadSACTests(unittest.TestCase):
    """SAC 训练逻辑单元测试。

    用 FixedActor / FixedCritic 常数替身推导 TD target 的解析期望值，
    验证 target 对十个头取全局均值而非逐头独立；并验证 update() 返回
    的统计量全部有限、log_alpha 保持标量、目标 critic 完成惰性初始化，
    以及 update 过程中在线 actor/critic 始终以 training 模式前向。
    """

    def test_td_target_uses_global_head_mean(self):
        """常数 critic 逐头 Q（0..9）的全局均值为 4.5，加上均匀策略熵 log 2 与系数 0.01 得到解析期望。"""
        agent = make_agent(n_actions=2, gamma=1.0)
        agent.actor = FixedActor(10, 2)
        agent.target_critic_1 = FixedCritic(10, 2)
        agent.target_critic_2 = FixedCritic(10, 2)
        agent._target_critics_initialized = True

        images = torch.zeros(2, 1, 4, 4)
        hoprates = torch.full((2, 1), 100.0)
        rewards = torch.zeros(2, 10)
        dones = torch.zeros(2, 1)
        targets = agent.calc_target(rewards, images, hoprates, dones)

        expected = 4.5 + 0.01 * math.log(2.0)
        np.testing.assert_allclose(
            targets.numpy(), np.full((2, 10), expected), rtol=1e-6, atol=1e-6
        )

    def test_update_returns_finite_stats(self):
        agent = make_agent()
        stats = agent.update(make_transition_batch())

        for key in (
            "critic1_loss",
            "critic2_loss",
            "actor_loss",
            "alpha_loss",
            "alpha",
            "entropy",
        ):
            self.assertTrue(np.isfinite(stats[key]), key)
        self.assertEqual(0, agent.log_alpha.ndim)
        self.assertTrue(agent._target_critics_initialized)

    def test_update_keeps_online_network_forwards_in_training_mode(self):
        agent = make_agent()
        observed_modes = {"actor": [], "critic_1": [], "critic_2": []}
        handles = []
        for name in observed_modes:
            network = getattr(agent, name)
            handles.append(
                network.register_forward_pre_hook(
                    lambda module, _inputs, key=name: observed_modes[key].append(
                        module.training
                    )
                )
            )
        try:
            agent.update(make_transition_batch(seed=9))
        finally:
            for handle in handles:
                handle.remove()

        for name, modes in observed_modes.items():
            self.assertTrue(modes, name)
            self.assertTrue(all(modes), f"{name} modes: {modes}")


if __name__ == "__main__":
    unittest.main()
