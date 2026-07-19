import argparse
import time
import gym
import torch
import numpy as np
from itertools import count
import logging

from model import EnsembleDynamicsModel
from replay_memory import ReplayMemory
#参数
def readParser():
    parser = argparse.ArgumentParser(description='MBPO')
    parser.add_argument('--seed', type=int, default=123456, metavar='N',
                        help='random seed (default: 123456)')
    parser.add_argument('--use_decay', type=bool, default=True, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    return parser.parse_args()

def train_predict_model(args, env_pool, predict_model):
    # Get all samples from environment
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))  #将环境经验池所有经验取出
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
    predict_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)

def _get_logprob( x, means, variances):
    k = x.shape[-1]
    # [ num_networks, batch_size ]
    log_prob = -1 / 2 * (k * np.log(2 * np.pi) + np.log(variances).sum(-1) + (np.power(x - means, 2) / variances).sum(-1))
    # [ batch_size ]
    prob = np.exp(log_prob).sum(0)
    # [ batch_size ]
    log_prob = np.log(prob)
    stds = np.std(means, 0).mean(-1)
    return log_prob, stds

def step(predict_model, obs, act, next_obs, deterministic=False):
    if len(obs.shape) == 1:
        obs = obs[None]
        act = act[None]
        next_obs=next_obs[None]
        return_single = True
    else:
        return_single = False

    inputs = np.concatenate((obs, act), axis=-1)
    ensemble_model_means, ensemble_model_vars = predict_model.predict(inputs)
    ensemble_model_stds = np.sqrt(ensemble_model_vars)

    if deterministic:
        ensemble_samples = ensemble_model_means
    else:
        ensemble_samples = ensemble_model_means + np.random.normal(
            size=ensemble_model_means.shape) * ensemble_model_stds

    num_models, batch_size, _ = ensemble_model_means.shape
    # 随机从精英model里选择一个mode进行预测
    model_idxes = np.random.choice(predict_model.elite_model_idxes, size=batch_size)
    batch_idxes = np.arange(0, batch_size)

    samples = ensemble_samples[model_idxes, batch_idxes]
    model_means = ensemble_model_means[model_idxes, batch_idxes]
    model_stds = ensemble_model_stds[model_idxes, batch_idxes]

    log_prob, dev = _get_logprob(samples, ensemble_model_means, ensemble_model_vars)
    rewards= samples[:, :1]
    batch_size = model_means.shape[0]
    terminals = np.zeros((batch_size, 1)) #表示done 是否为环境状态终止，我们算法应该一直是0
    return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
    return_stds = np.concatenate((model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]), axis=-1)
    if return_single:
        next_obs = next_obs[0]
        return_means = return_means[0]
        return_stds = return_stds[0]
        rewards = rewards[0]
        terminals = terminals[0]
    info = {'mean': return_means, 'std': return_stds, 'log_prob': log_prob, 'dev': dev}
    return next_obs, rewards, terminals, info

def rollout_model(args, predict_model, agent, model_pool, env_pool, rollout_length):
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        #Get a batch of actions
        action = agent.select_action(state)
        # 对state及reward进行预测
        next_states, rewards, done, info = step(predict_model, state, action, next_state)
        # Push a batch of samples
        model_pool.push_batch([(state[j], action[j], rewards[j], next_states[j], done[j]) for j in range(state.shape[0])])
def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    # 环境里走n_steps步，才训练一次
    if total_step % args.train_every_n_steps > 0:
        return 0
    # 防止策略网络更新次数（train_step）相对于环境交互次数（total_step）过高。
    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat): #决定虚拟训练多少步
        #虚拟训练采用env和prediction model的混合经验进行训练
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size
        # 从环境经验池抽取env_batch_size条经验
        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(int(env_batch_size))

        if model_batch_size > 0 and len(model_pool) > 0:
            # 从预测经验池抽取model_batch_size条经验
            model_state, model_action, model_reward, model_next_state, model_done = model_pool.sample_all_batch(int(model_batch_size))
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = np.concatenate((env_state, model_state), axis=0), \
                                                                                    np.concatenate((env_action, model_action),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward), axis=0), \
                                                                                    np.concatenate((env_next_state, model_next_state),
                                                                                                   axis=0), np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done), axis=0)
        else:
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = env_state, env_action, env_reward, env_next_state, env_done
        #把多余的维度挤掉，变回(256, )
        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        #.astype(int) 把布尔值转成 1 和 0。
        batch_done = (~batch_done).astype(int)
        # 执行sac网络训练
        agent.update_parameters((batch_state, batch_action, batch_reward, batch_next_state, batch_done), args.policy_train_batch_size, i)
    return args.num_train_repeat


def train(args, env_sampler, predict_model, agent, env_pool, model_pool):
    total_step = 0
    reward_sum = 0
    #填充env_pool
    exploration_before_start(args, env_sampler, env_pool, agent)  #初始阶段 收集环境经验
    start_step = total_step
    train_policy_steps = 0
    for i in count():
        cur_step = total_step - start_step
        if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
            break
        if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
            train_predict_model(args, env_pool, predict_model)
            #进行推演并存经验
            rollout_model(args, predict_model, agent, model_pool, env_pool, args.rollout_length)
        cur_state, action, next_state, reward, done= env_sampler.sample(agent) #真实环境里进行一次交互，得到（s,a,s',r,done）
        env_pool.push(cur_state, action, reward, next_state, done) #存经验

        if len(env_pool) > args.min_pool_size:
            # 真实环境虽然只走了 1 步，但这里通常会连续对策略网络执行多次梯度更新
            train_policy_steps += train_policy_repeats(args, total_step, train_policy_steps, cur_step, env_pool, model_pool, agent)
        total_step += 1


def main(args=None):
    if args is None:
        args = readParser()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args) #sac agent

    # Initial ensemble model
    predict_model = EnsembleDynamicsModel(args.num_networks, args.num_elites, args.state_size, args.action_size, args.reward_size, args.pred_hidden_size,
                                          use_decay=args.use_decay)
    # 环境经验池
    env_pool = ReplayMemory(args.replay_size)
    # 预测经验池
    model_pool = ReplayMemory(args.replay_size)
    #env_sampler是他自己写的环境，换成自己的就行
    train(args, env_sampler, predict_model, agent, env_pool, model_pool)