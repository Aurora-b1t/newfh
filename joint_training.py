"""Shared utilities for derivative-NBS plus offset-policy training."""

from __future__ import annotations

import copy
import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from noisy_binary_search_derivative import DerivativeNoisyBinarySearch
from offline_replay import load_replay_into_buffer
import settings


JAMMER_MODES = ("comb", "sweep", "both")


def str_to_bool(value):
    """Parse an explicit true/false command-line value."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "on"}:
        return True
    if normalized in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"Expected a boolean value, got {value!r}.")


def add_environment_override_args(parser):
    """Add instance-local jammer/environment overrides to an argument parser."""
    parser.add_argument(
        "--enable_reactive",
        type=str_to_bool,
        default=settings.ENV_CONFIG["enable_reactive"],
        help="Enable the reactive jammer (default: settings.ENV_CONFIG).",
    )
    parser.add_argument(
        "--enable_sweep",
        type=str_to_bool,
        default=settings.ENV_CONFIG["enable_sweep"],
        help="Enable the indiscriminate sweep/comb jammer.",
    )
    parser.add_argument(
        "--jammer_mode",
        choices=JAMMER_MODES,
        default=settings.JAMMER_CONFIG["mode"],
        help="Indiscriminate jammer mode when --enable_sweep is true.",
    )
    return parser


def add_derivative_nbs_args(parser):
    """Add the derivative-NBS parameters shared by both joint trainers."""
    parser.add_argument("--nbs_p", type=float, default=settings.NBS_CONFIG["p"])
    parser.add_argument(
        "--nbs_delta", type=float, default=settings.NBS_CONFIG["delta"]
    )
    parser.add_argument(
        "--nbs_step",
        type=float,
        default=settings.NBS_CONFIG["hoprate_step"],
    )
    parser.add_argument(
        "--derivative_threshold",
        type=float,
        default=settings.NBS_CONFIG["derivative_threshold"],
    )
    return parser


def resolve_environment_configs(args):
    """Return effective environment and jammer configs without global mutation."""
    env_config = copy.deepcopy(settings.ENV_CONFIG)
    jammer_config = copy.deepcopy(settings.JAMMER_CONFIG)
    env_config["enable_reactive"] = bool(args.enable_reactive)
    env_config["enable_sweep"] = bool(args.enable_sweep)
    jammer_config["mode"] = str(args.jammer_mode)
    return env_config, jammer_config


def build_derivative_nbs(env, args):
    """Build an NBS controller whose candidates match environment quantisation."""
    nbs = DerivativeNoisyBinarySearch(
        hoprate_min=env.hoprate_min,
        hoprate_max=env.hoprate_max,
        hoprate_step=args.nbs_step,
        p=args.nbs_p,
        delta=args.nbs_delta,
        derivative_threshold=args.derivative_threshold,
        seed=args.seed,
    )
    quantized = np.rint(nbs.candidates / 10.0) * 10.0
    if not np.allclose(nbs.candidates, quantized, rtol=0.0, atol=1e-8):
        raise ValueError(
            "Derivative-NBS candidates must lie on the environment's 10 Hz "
            "hoprate grid; choose a compatible --nbs_step."
        )
    if len(nbs.candidates) > 1 and not np.allclose(
        np.diff(nbs.candidates), args.nbs_step, rtol=0.0, atol=1e-8
    ):
        raise ValueError(
            "--nbs_step must divide the configured hoprate range exactly."
        )
    return nbs


def validate_joint_args(args):
    """Validate arguments common to the two joint training entry points."""
    for name in (
        "steps_per_episode",
        "replay_size",
        "batch_size",
        "update_iters_per_step",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"{name} must be positive.")
    if float(args.nbs_step) <= 0.0:
        raise ValueError("nbs_step must be positive.")
    if not np.isfinite(float(args.derivative_threshold)):
        raise ValueError("derivative_threshold must be finite.")


def configure_figure_capture(args, env, logger):
    """Enable the existing pre-action and block figure capture schedule."""
    figures_dir = os.path.join(args.output_dir, "figures")
    save_steps = set()
    for value in settings.PLOT_CONFIG.get("figure_save_steps", []):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
            logger.warning("figure_save_steps entry %r is not an integer; ignored.", value)
        elif not 1 <= int(value) <= args.steps_per_episode:
            logger.warning(
                "figure_save_steps entry %d is outside [1, %d]; ignored.",
                int(value),
                args.steps_per_episode,
            )
        else:
            save_steps.add(int(value))
    if save_steps:
        env.enable_step_figure_capture(sorted(save_steps), figures_dir)
        logger.info(
            "Figure saving enabled for steps %s -> %s",
            sorted(save_steps),
            figures_dir,
        )
    return save_steps, figures_dir


def load_joint_replay(
    path,
    buffer,
    state_img,
    env,
    metadata,
    allow_config_mismatch=False,
    logger=None,
):
    """Load v3 replay with strict metadata validation by default."""
    return load_replay_into_buffer(
        path,
        buffer,
        expected_observation_shape=np.asarray(state_img).shape,
        expected_num_actions=env.num_channels,
        expected_num_blocks=env.num_blocks,
        current_environment_metadata=metadata,
        strict_environment_metadata=not allow_config_mismatch,
        logger=logger,
    )


@dataclass(frozen=True)
class JointStepResult:
    offsets: np.ndarray
    next_state_img: np.ndarray
    step_reward: float
    block_rewards: np.ndarray
    ber_blocks: np.ndarray
    mean_ber: float
    used_hoprate: float
    next_hoprate: float
    done: bool
    info: dict


def execute_joint_step(state_img, current_hoprate, agent, env, nbs, buffer):
    """Execute the causal NBS -> offset -> environment -> NBS transition."""
    current_hoprate = float(current_hoprate)
    offsets = np.asarray(
        agent.take_action(state_img, current_hoprate), dtype=np.int64
    )
    next_state_img, step_reward, terminated, truncated, info = env.step(
        {"hoprate": current_hoprate, "offsets": offsets}
    )

    block_rewards = np.asarray(info.get("block_rewards", []), dtype=np.float32)
    ber_blocks = np.asarray(info.get("ber_blocks", []), dtype=np.float32)
    expected_shape = (env.num_blocks,)
    if block_rewards.shape != expected_shape:
        raise RuntimeError("Environment returned an invalid block reward vector.")
    if ber_blocks.shape != expected_shape:
        raise RuntimeError("Environment returned an invalid block BER vector.")
    if not np.isclose(
        float(step_reward),
        float(np.mean(block_rewards)),
        rtol=1e-5,
        atol=1e-6,
    ):
        raise RuntimeError("Environment reward is inconsistent with block rewards.")

    used_hoprate = float(info.get("hoprate_used", current_hoprate))
    if not np.isclose(used_hoprate, current_hoprate, rtol=0.0, atol=1e-8):
        raise RuntimeError(
            "Environment quantized the NBS hoprate unexpectedly: "
            f"requested={current_hoprate}, used={used_hoprate}."
        )

    mean_ber = float(np.mean(ber_blocks))
    next_hoprate = float(nbs.step(mean_ber))
    if not np.isfinite(next_hoprate):
        raise RuntimeError("Derivative NBS returned a non-finite next hoprate.")

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
    return JointStepResult(
        offsets=offsets,
        next_state_img=np.asarray(next_state_img, dtype=np.float32),
        step_reward=float(step_reward),
        block_rewards=block_rewards,
        ber_blocks=ber_blocks,
        mean_ber=mean_ber,
        used_hoprate=used_hoprate,
        next_hoprate=next_hoprate,
        done=done,
        info=info,
    )


def new_nbs_history():
    return {
        "hoprates_used": [],
        "bers": [],
        "derivatives": [],
        "best_hoprates": [],
        "weighted_hoprates": [],
        "max_weights": [],
        "converged": [],
    }


def record_nbs_step(history, nbs, result):
    """Append post-update derivative-NBS diagnostics for one environment step."""
    derivative = nbs.get_last_derivative()
    history["hoprates_used"].append(result.used_hoprate)
    history["bers"].append(result.mean_ber)
    history["derivatives"].append(
        np.nan if derivative is None else float(derivative)
    )
    history["best_hoprates"].append(float(nbs.get_best_hoprate()))
    history["weighted_hoprates"].append(float(nbs.get_weighted_average()))
    history["max_weights"].append(float(np.max(nbs.weights)))
    history["converged"].append(bool(nbs.is_converged()))


def nbs_metadata(nbs):
    """Return compact controller metadata for inference checkpoints."""
    return {
        "controller": "DerivativeNoisyBinarySearch",
        "p": float(nbs.p),
        "delta": float(nbs.delta),
        "hoprate_step": float(nbs.hoprate_step),
        "derivative_threshold": float(nbs.derivative_threshold),
        "final_best_hoprate": float(nbs.get_best_hoprate()),
        "final_weighted_hoprate": float(nbs.get_weighted_average()),
        "converged": bool(nbs.is_converged()),
    }


def save_nbs_artifacts(output_dir, history, nbs, logger=None):
    """Save the complete controller history, final distribution, and plots."""
    os.makedirs(output_dir, exist_ok=True)
    candidates, weights = nbs.get_distribution()
    arrays = {
        key: np.asarray(values) for key, values in history.items()
    }
    np.savez(
        os.path.join(output_dir, "nbs_distribution.npz"),
        candidates=candidates,
        weights=weights,
        **arrays,
    )
    if not history["hoprates_used"]:
        return

    steps = np.arange(1, len(history["hoprates_used"]) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["hoprates_used"], ".-", label="tested hoprate")
    ax.plot(steps, history["best_hoprates"], "--", label="NBS best (MAP)")
    ax.plot(
        steps,
        history["weighted_hoprates"],
        ":",
        label="NBS weighted average",
    )
    ax.set(title="Joint Hoprate Trajectory", xlabel="Step", ylabel="Hoprate (Hz)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "hoprate.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    scatter = ax.scatter(
        history["hoprates_used"],
        history["bers"],
        c=steps,
        cmap="viridis",
        edgecolors="k",
        linewidth=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="Step")
    ax.set(title="BER vs Hoprate", xlabel="Hoprate (Hz)", ylabel="Mean BER")
    ax.grid(True, alpha=0.3)
    fig.savefig(os.path.join(output_dir, "ber_vs_hoprate.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(steps, history["derivatives"], ".-", label="BER derivative")
    ax.axhline(
        nbs.derivative_threshold,
        color="red",
        linestyle="--",
        label=f"threshold={nbs.derivative_threshold:g}",
    )
    ax.set(title="Derivative-NBS Metric", xlabel="Step", ylabel="Metric")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "derivative.png"))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    width = candidates[1] - candidates[0] if len(candidates) > 1 else 10.0
    ax.bar(candidates, weights, width=width, alpha=0.8)
    ax.axhline(
        1.0 - nbs.delta,
        color="red",
        linestyle="--",
        label=f"convergence threshold={1.0 - nbs.delta:.3f}",
    )
    ax.set(
        title="Final Derivative-NBS Weight Distribution",
        xlabel="Hoprate (Hz)",
        ylabel="Weight",
    )
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(os.path.join(output_dir, "nbs_weights.png"))
    plt.close(fig)

    if logger is not None:
        logger.info("Saved derivative-NBS diagnostics to %s.", output_dir)
