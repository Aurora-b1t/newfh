"""Generate real-environment step-level replay data for offset SAC."""

import argparse
import logging

import numpy as np

from fh_env import FHSSQPSKEnv
from joint_training import (
    add_environment_override_args,
    resolve_environment_configs,
)
from SAC import ReplayBuffer
import settings
from offline_replay import environment_metadata, save_replay_buffer


def quantize_hoprate(hoprate, env):
    clipped = np.clip(hoprate, env.hoprate_min, env.hoprate_max)
    return float(int(round(clipped / 10.0)) * 10)


def make_hoprate_sampler(mode, fixed_hoprate, env, rng):
    if mode == "fixed":
        quantized = quantize_hoprate(float(fixed_hoprate), env)
        return lambda: quantized

    if mode != "random":
        raise ValueError(f"Unsupported hoprate_mode: {mode!r}.")

    min_step = int(np.ceil(env.hoprate_min / 10.0))
    max_step = int(np.floor(env.hoprate_max / 10.0))
    if min_step > max_step:
        raise ValueError("Environment hoprate range has no valid 10 Hz value.")
    valid_rates = np.arange(min_step, max_step + 1, dtype=np.int32) * 10
    return lambda: float(rng.choice(valid_rates))


def generate(args):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    settings.set_random_seeds(args.seed)
    rng = np.random.default_rng(args.seed)
    env_config, jammer_config = resolve_environment_configs(args)
    env = FHSSQPSKEnv(**env_config, jammer_config=jammer_config)
    state_img, _info = env.reset()
    n_actions = env.num_channels
    num_blocks = env.num_blocks
    buffer = ReplayBuffer(
        capacity=args.num_step_transitions,
        num_heads=num_blocks,
        n_actions=n_actions,
    )
    hoprate_sampler = make_hoprate_sampler(
        args.hoprate_mode, args.fixed_hoprate, env, rng
    )

    current_hoprate = hoprate_sampler()
    while buffer.size() < args.num_step_transitions:
        offsets = rng.integers(
            0, n_actions, size=num_blocks, dtype=np.int64
        )
        next_state_img, step_reward, terminated, truncated, info = env.step(
            {"hoprate": current_hoprate, "offsets": offsets}
        )
        used_hoprate = float(info.get("hoprate_used", current_hoprate))
        next_hoprate = hoprate_sampler()
        block_rewards = np.asarray(info.get("block_rewards", []), dtype=np.float32)
        if block_rewards.shape != (num_blocks,):
            raise RuntimeError(
                "Environment did not return one reward for each offset block."
            )
        if not np.isclose(
            float(step_reward), float(np.mean(block_rewards)), rtol=1e-5, atol=1e-6
        ):
            raise RuntimeError("Environment step reward is inconsistent with block rewards.")

        done = bool(terminated or truncated)
        buffer.add(
            state_img,
            used_hoprate,
            offsets,
            block_rewards,
            next_state_img,
            next_hoprate,
            done,
        )
        state_img = next_state_img
        current_hoprate = next_hoprate

        if done:
            state_img, _info = env.reset()
            current_hoprate = hoprate_sampler()

        if buffer.size() % 100 == 0 or buffer.size() == args.num_step_transitions:
            logging.info(
                "Collected %d/%d step transitions",
                buffer.size(),
                args.num_step_transitions,
            )

    metadata = environment_metadata(
        env_config,
        jammer_config,
        settings.REWARD_CONFIG,
    )
    metadata.update(
        {
            "generator": "generate_offline_replay.py",
            "seed": args.seed,
            "hoprate_mode": args.hoprate_mode,
            "fixed_hoprate": (
                args.fixed_hoprate if args.hoprate_mode == "fixed" else None
            ),
            "num_env_steps": buffer.size(),
            "num_actions": n_actions,
            "num_blocks": num_blocks,
            "hoprate_grid_step": 10.0,
        }
    )
    save_replay_buffer(args.output_path, buffer, metadata)
    logging.info(
        "Saved %d step transitions to %s",
        buffer.size(),
        args.output_path,
    )
    env.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate real FHSS step-level replay with random offsets."
    )
    parser.add_argument(
        "--num_step_transitions",
        type=int,
        default=settings.OFFLINE_REPLAY_CONFIG["num_step_transitions"],
        help="Number of complete environment-step transitions to save.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=settings.OFFLINE_REPLAY_CONFIG["default_path"],
        help="Output v3 .npz path.",
    )
    parser.add_argument(
        "--hoprate_mode",
        choices=("random", "fixed"),
        default=settings.OFFLINE_REPLAY_CONFIG["hoprate_mode"],
        help="Use uniformly random valid hoprates or one fixed hoprate.",
    )
    parser.add_argument(
        "--fixed_hoprate",
        type=float,
        default=settings.TRAIN_CONFIG["fixed_hoprate"],
        help="Hoprate used when --hoprate_mode=fixed.",
    )
    parser.add_argument("--seed", type=int, default=settings.RANDOM_SEED)
    add_environment_override_args(parser)
    return parser.parse_args(argv)


if __name__ == "__main__":
    generate(parse_args())
