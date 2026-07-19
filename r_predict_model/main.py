"""
Generic MBPO training template.

This module is not the FHSS-specific entry point; ``train_mbpo.py`` is the
project-specific script used by the current SAC anti-jamming environment.  The
functions here document the standard MBPO workflow:

    1. collect real environment transitions into ``env_pool``;
    2. train an ensemble dynamics model on real replay;
    3. roll out short synthetic trajectories into ``model_pool``;
    4. update SAC from a mixture of real and synthetic transitions.

Several symbols used by ``main`` (for example ``SAC``, ``env``, and
``env_sampler``) are expected to be supplied by the application integrating this
template.
"""

import argparse
from itertools import count
import logging
import time

import gym
import numpy as np
import torch

from model import EnsembleDynamicsModel
from replay_memory import ReplayMemory


def readParser():
    """
    Parse baseline SAC/MBPO hyperparameters.

    The parser only contains the common options that existed in the original
    template.  A complete application may add rollout length, model ensemble
    size, replay capacity, and environment-specific dimensions before calling
    ``train``.
    """
    parser = argparse.ArgumentParser(description="MBPO")
    parser.add_argument(
        "--seed",
        type=int,
        default=123456,
        metavar="N",
        help="random seed (default: 123456)",
    )
    parser.add_argument(
        "--use_decay",
        type=bool,
        default=True,
        metavar="G",
        help="whether to apply model weight decay (default: True)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor for reward (default: 0.99)",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        metavar="G",
        help="target smoothing coefficient (default: 0.005)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.2,
        metavar="G",
        help=(
            "temperature parameter controlling the entropy term against "
            "the reward (default: 0.2)"
        ),
    )
    parser.add_argument(
        "--policy",
        default="Gaussian",
        help="policy type: Gaussian | Deterministic (default: Gaussian)",
    )
    parser.add_argument(
        "--target_update_interval",
        type=int,
        default=1,
        metavar="N",
        help="value target update interval in optimizer steps (default: 1)",
    )
    parser.add_argument(
        "--automatic_entropy_tuning",
        type=bool,
        default=False,
        metavar="G",
        help="automatically adjust alpha (default: False)",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=256,
        metavar="N",
        help="hidden size (default: 256)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0003,
        metavar="G",
        help="learning rate (default: 0.0003)",
    )
    return parser.parse_args()


def train_predict_model(args, env_pool, predict_model):
    """
    Train the ensemble dynamics model from all real environment replay.

    Inputs are concatenated ``[state, action]`` vectors.  Labels are
    ``[reward, next_state - state]`` so the learned model predicts immediate
    reward and state delta, which is the standard MBPO dynamics-model target.
    """
    state, action, reward, next_state, done = env_pool.sample(len(env_pool))
    delta_state = next_state - state
    inputs = np.concatenate((state, action), axis=-1)
    labels = np.concatenate((np.reshape(reward, (reward.shape[0], -1)), delta_state), axis=-1)
    predict_model.train(inputs, labels, batch_size=256, holdout_ratio=0.2)


def _get_logprob(x, means, variances):
    """
    Estimate ensemble mixture log probability and model disagreement.

    Args:
        x: Sampled predictions with shape [batch_size, output_dim].
        means: Ensemble means with shape [num_networks, batch_size, output_dim].
        variances: Ensemble variances with the same shape as ``means``.

    Returns:
        log_prob: Log probability of each sample under the ensemble mixture.
        stds: Mean standard deviation of ensemble means, used as an uncertainty
            diagnostic.
    """
    k = x.shape[-1]
    log_prob = -1 / 2 * (
        k * np.log(2 * np.pi)
        + np.log(variances).sum(-1)
        + (np.power(x - means, 2) / variances).sum(-1)
    )
    prob = np.exp(log_prob).sum(0)
    log_prob = np.log(prob)
    stds = np.std(means, 0).mean(-1)
    return log_prob, stds


def step(predict_model, obs, act, next_obs, deterministic=False):
    """
    Use the learned dynamics model to produce one synthetic transition step.

    The model output layout is ``[reward, delta_state...]``.  This function
    samples one elite model per batch item, extracts the predicted reward, and
    returns the provided ``next_obs`` together with diagnostic uncertainty
    information.  Terminal flags are forced to zero because this template does
    not include an environment-specific termination model.
    """
    if len(obs.shape) == 1:
        obs = obs[None]
        act = act[None]
        next_obs = next_obs[None]
        return_single = True
    else:
        return_single = False

    inputs = np.concatenate((obs, act), axis=-1)
    ensemble_model_means, ensemble_model_vars = predict_model.predict(inputs)
    ensemble_model_stds = np.sqrt(ensemble_model_vars)

    if deterministic:
        ensemble_samples = ensemble_model_means
    else:
        ensemble_samples = (
            ensemble_model_means
            + np.random.normal(size=ensemble_model_means.shape) * ensemble_model_stds
        )

    num_models, batch_size, _ = ensemble_model_means.shape
    # Randomly choose one elite model for each batch item, following MBPO's
    # ensemble sampling strategy.
    model_idxes = np.random.choice(predict_model.elite_model_idxes, size=batch_size)
    batch_idxes = np.arange(0, batch_size)

    samples = ensemble_samples[model_idxes, batch_idxes]
    model_means = ensemble_model_means[model_idxes, batch_idxes]
    model_stds = ensemble_model_stds[model_idxes, batch_idxes]

    log_prob, dev = _get_logprob(samples, ensemble_model_means, ensemble_model_vars)
    rewards = samples[:, :1]
    batch_size = model_means.shape[0]
    terminals = np.zeros((batch_size, 1))
    return_means = np.concatenate((model_means[:, :1], terminals, model_means[:, 1:]), axis=-1)
    return_stds = np.concatenate(
        (model_stds[:, :1], np.zeros((batch_size, 1)), model_stds[:, 1:]),
        axis=-1,
    )
    if return_single:
        next_obs = next_obs[0]
        return_means = return_means[0]
        return_stds = return_stds[0]
        rewards = rewards[0]
        terminals = terminals[0]
    info = {"mean": return_means, "std": return_stds, "log_prob": log_prob, "dev": dev}
    return next_obs, rewards, terminals, info


def rollout_model(args, predict_model, agent, model_pool, env_pool, rollout_length):
    """
    Generate short synthetic rollouts and append them to ``model_pool``.

    Start states are sampled from real replay.  At every synthetic step, the
    current policy chooses actions and the learned model supplies predicted
    rewards and terminal flags.
    """
    state, action, reward, next_state, done = env_pool.sample_all_batch(args.rollout_batch_size)
    for i in range(rollout_length):
        action = agent.select_action(state)
        next_states, rewards, done, info = step(predict_model, state, action, next_state)
        model_pool.push_batch(
            [
                (state[j], action[j], rewards[j], next_states[j], done[j])
                for j in range(state.shape[0])
            ]
        )


def train_policy_repeats(args, total_step, train_step, cur_step, env_pool, model_pool, agent):
    """
    Run repeated SAC updates from mixed real/model replay.

    The two guards at the top control update frequency and prevent policy
    optimization from running too far ahead of real environment interaction.
    """
    if total_step % args.train_every_n_steps > 0:
        return 0
    if train_step > args.max_train_repeat_per_step * total_step:
        return 0

    for i in range(args.num_train_repeat):
        env_batch_size = int(args.policy_train_batch_size * args.real_ratio)
        model_batch_size = args.policy_train_batch_size - env_batch_size
        env_state, env_action, env_reward, env_next_state, env_done = env_pool.sample(
            int(env_batch_size)
        )

        if model_batch_size > 0 and len(model_pool) > 0:
            model_state, model_action, model_reward, model_next_state, model_done = (
                model_pool.sample_all_batch(int(model_batch_size))
            )
            batch_state = np.concatenate((env_state, model_state), axis=0)
            batch_action = np.concatenate((env_action, model_action), axis=0)
            batch_reward = np.concatenate(
                (np.reshape(env_reward, (env_reward.shape[0], -1)), model_reward),
                axis=0,
            )
            batch_next_state = np.concatenate((env_next_state, model_next_state), axis=0)
            batch_done = np.concatenate(
                (np.reshape(env_done, (env_done.shape[0], -1)), model_done),
                axis=0,
            )
        else:
            batch_state = env_state
            batch_action = env_action
            batch_reward = env_reward
            batch_next_state = env_next_state
            batch_done = env_done

        # Flatten reward/done to match the SAC implementation's expected batch
        # format, then convert done into the mask convention used by SAC.
        batch_reward, batch_done = np.squeeze(batch_reward), np.squeeze(batch_done)
        batch_done = (~batch_done).astype(int)
        agent.update_parameters(
            (batch_state, batch_action, batch_reward, batch_next_state, batch_done),
            args.policy_train_batch_size,
            i,
        )
    return args.num_train_repeat


def train(args, env_sampler, predict_model, agent, env_pool, model_pool):
    """
    Coordinate real data collection, model training, rollout, and policy updates.

    ``env_sampler`` is expected to expose ``sample(agent)`` and return
    ``(state, action, next_state, reward, done)`` for one real environment
    interaction.
    """
    total_step = 0
    reward_sum = 0
    exploration_before_start(args, env_sampler, env_pool, agent)
    start_step = total_step
    train_policy_steps = 0
    for i in count():
        cur_step = total_step - start_step
        if cur_step >= args.epoch_length and len(env_pool) > args.min_pool_size:
            break
        if cur_step > 0 and cur_step % args.model_train_freq == 0 and args.real_ratio < 1.0:
            train_predict_model(args, env_pool, predict_model)
            rollout_model(args, predict_model, agent, model_pool, env_pool, args.rollout_length)
        cur_state, action, next_state, reward, done = env_sampler.sample(agent)
        env_pool.push(cur_state, action, reward, next_state, done)

        if len(env_pool) > args.min_pool_size:
            train_policy_steps += train_policy_repeats(
                args,
                total_step,
                train_policy_steps,
                cur_step,
                env_pool,
                model_pool,
                agent,
            )
        total_step += 1


def main(args=None):
    """
    Build the template components and launch training.

    This function requires the host project to provide ``env``, ``SAC``, and
    ``env_sampler`` symbols.  It is retained as an integration example rather
    than as the FHSS project's runnable training command.
    """
    if args is None:
        args = readParser()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    agent = SAC(env.observation_space.shape[0], env.action_space, args)
    predict_model = EnsembleDynamicsModel(
        args.num_networks,
        args.num_elites,
        args.state_size,
        args.action_size,
        args.reward_size,
        args.pred_hidden_size,
        use_decay=args.use_decay,
    )
    env_pool = ReplayMemory(args.replay_size)
    model_pool = ReplayMemory(args.replay_size)
    train(args, env_sampler, predict_model, agent, env_pool, model_pool)
