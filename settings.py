"""
Central configuration for the FHSS RL anti-jamming training project.

Defines environment parameters, jammer settings, SAC and step-level MBPO
hyperparameters, replay buffer sizes, noisy binary search settings, training
loop options, and reward coefficients. Output directories are configured
per-script via each training entry point's ``--output_dir`` argument.
"""

import random
import numpy as np

# Device Configuration
CPU_ONLY = False # Set to True to force CPU usage

# Global Random Seed for full reproducibility
RANDOM_SEED = 42

# Timing Profiling Switch
# When True, MBPO reward-model training measures per-epoch wall-clock times
# and rollout (experience generation) per-stage times, and the training logs
# include them. Set to False to disable all timing measurements.
TIMING_ENABLED = True


def set_random_seeds(seed=None):
    """
    Set random seeds for all random number generators to ensure reproducibility.

    Args:
        seed: Random seed. If None, uses settings.RANDOM_SEED.
    """
    if seed is None:
        seed = RANDOM_SEED

    # Python standard library
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch (import locally to avoid circular dependency)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Ensure CUDA convolution determinism
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass

# Environment Configuration
# Passed to FHSSQPSKEnv(**ENV_CONFIG)
ENV_CONFIG = {
    "Startfre": 3e6,
    "Endfre": 4e6,
    "Sub_interval": 50000,
    "Fs": 1e7,
    "Baud": 25000,
    "Hoprate": 100,             # Base hoprate
    "hoprate_min": 10.0,
    "hoprate_max": 1000.0,
    "enable_reactive": False,   # Reactive Jammer
    "enable_sweep": True,       # Indiscriminate Jammer (Swipe/Comb)
    "enable_rayleigh": True,    # Fading Channel
    "debug_plot_psd": False,
    "debug_log_hops": False,
    "use_pregen": True,
    "noise_std": 0.1,           # Thermal noise std at receiver (also used by reactive jammer)
    # M-sequence (LFSR) driving the base hopping pattern.
    # To switch to a different m-sequence:
    #   - change mseq_seed  -> same m-sequence, different phase (cyclic shift);
    #   - change mseq_taps  -> a genuinely different m-sequence; taps MUST form a
    #                          primitive polynomial (e.g. 10-stage: (10, 7), (10, 3)),
    #                          otherwise the period collapses below 2^n - 1;
    #   - change mseq_nbits -> changes the period (2^n - 1); taps must be replaced
    #                          with a primitive set of that degree accordingly.
    # NOTE: after changing these, regenerate the offline replay dataset.
    "mseq_seed": 46,
    "mseq_taps": (10, 7),
    "mseq_nbits": 10,
    "mseq_length": 1023,
    # Real RF signal power per sample (before fading).
    # Theoretical: Baud / Fs = 25000 / 1e7 = 0.0025.
    # The reactive jammer uses this together with noise_std to derive its
    # detection SNR, averaged over Rayleigh fading via Gauss-Laguerre quadrature.
    "signal_power": 0.0025,
}

# Jammer Configuration
JAMMER_CONFIG = {
    # Global Jamming Mode: 'sweep', 'comb', or 'both'
    "mode": "comb",
    # Independent band-limited baseband realizations used by pre-generated
    # reactive/sweep/comb waveforms. Equal bandwidths share the same pool.
    "baseband_variant_count": 4,
    
    # Sweep Jamming Configuration
    "sweep": {
        "step": 50000,       # Frequency step for sweep
        "power": 0.8,        # Jamming power
        "dwell_time": 0.004, # Dwell time per step
        "bandwidth": 50000.0,# Noise bandwidth
    },
    
    # Comb Jamming Configuration
    "comb": {
        "power": 0.8,        # Total power or per-tone power factor
        "bandwidth": 50000.0,# Noise bandwidth per tone
        "switch_interval": 0.057, # Seconds; positive multiple of 1 ms
        # Jammed channel indices for the two alternating comb groups.
        # Each index k maps to the centre of 50kHz channel k
        # (Startfre + k*50kHz + 25kHz); indices must be integers in
        # [0, num_channels - 1] or the environment raises ValueError at startup.
        # Group lengths may differ and groups may overlap.
        # NOTE: after changing these, regenerate the offline replay dataset.
        "channels_phase0": [0, 2, 4, 6, 8, 10, 12, 14],   # Group 0 (even channels)
        "channels_phase1": [1, 3, 5, 7, 9, 11, 13, 15],  # Group 1 (odd channels)
    },
    
    # Reactive Jamming Configuration
    # Based on energy detection theory (Urkowitz 1967).
    # Operates on 1 ms fundamental time slots: scan → detect → jam → re-scan.
    # SNR at the jammer is derived from ENV_CONFIG["noise_std"] and the
    # theoretical signal power (Baud / Fs), ensuring consistency with the receiver.
    "reactive": {
        "power": 1.5,              # Jamming power factor
        "bandwidth": 50000.0,      # Noise bandwidth (Hz)
        "p_fa": 0.1,               # False-alarm probability for energy detection
        "detection_time": 0.0005,   # Detection / jamming slot duration (s) — 1 ms
    }
}

# SAC Agent Configuration
SAC_CONFIG = {
    "actor_lr": 1e-5,
    "critic_lr": 1e-4,
    "alpha_lr": 1e-4,
    "tau": 0.005,
    "gamma": 0.95,
    "target_entropy_ratio": 0.1,
}

# Replay Buffer Configuration
BUFFER_CONFIG = {
    "capacity": 20000,
    # A800 training uses large batches to keep the convolutional encoders busy.
    "batch_size": 2048,
}

# Offline real-environment replay configuration. Each transition represents one
# complete environment step with ten offset actions and ten block rewards.
OFFLINE_REPLAY_CONFIG = {
    "num_step_transitions": 20000,
    "default_path": "outputs/offline_replay/replay_5000_100_hoprate_v3.npz",
    "hoprate_mode": "fixed",
    "fixed_hoprate": 100.0,
}

# Joint derivative-NBS + offset training uses a broad hoprate replay without
# changing the fixed-hoprate baseline defaults above.
JOINT_OFFLINE_REPLAY_CONFIG = {
    "num_step_transitions": 5000,
    "default_path": "outputs/offline_replay/replay_20000_random_hoprate_v3.npz",
    "hoprate_mode": "random",
}

# MBPO Reward-Model Configuration
MBPO_CONFIG = {
    "num_networks": 5,
    "num_elites": 3,
    "hidden_size": 200,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    # Refit the full real replay periodically instead of blocking every env step.
    "model_train_freq": 10,
    "model_train_batch_size": 2048,
    "holdout_ratio": 0.2,
    "early_stop_patience": 5,
    "max_epochs": 100,
    "min_improvement": 0.01,
    "rollout_batch_size": 2048,
    "rollout_length": 1,
    "real_ratio": 0.2,
    "model_replay_size": 4000,
    # The server training path has enough device memory to avoid repeated H2D
    # copies for every reward-model epoch. Use --no-cache_model_dataset when it
    # is not available.
    "cache_dataset_on_device": True,
    # SAC replay is also mirrored on CUDA for DataLoader-driven updates.
    "cache_replay_on_device": True,
    # CUDA-resident datasets must use the main process. CPU fallback can opt in
    # to workers and pinned batches through the CLI.
    "data_loader_workers": 0,
    "data_loader_pin_memory": False,
    # Runtime-only acceleration knobs. These do not change the reward-model
    # architecture, optimizer budget, replay split, or early-stop settings.
    "model_precision": "float32",
    "model_fast_math": True,
    "model_compile": False,
    # PNG generation is diagnostic I/O and is disabled during normal training.
    "save_curve_figures": True,
}

# Noisy Binary Search Configuration
# Parameters for the MWU-based noisy binary search hoprate adjustment.
# Reference: Dereniowski et al. "Noisy (Binary) Searching: Simple, Fast and Correct" (STACS 2025)
NBS_CONFIG = {
    "p": 0.3,                 # Assumed noise probability, 0 ≤ p < 0.5.
                               # Higher p → more exploration (needed for flat BER regions).
    "delta": 0.01,            # Confidence threshold, 0 < δ ≤ 1.
                               # Convergence when max weight ≥ 1 − δ.
    "hoprate_step": 10.0,     # Discretisation step (Hz). Matches _apply_hoprate quantisation.
    "derivative_threshold": -0.005,
}

# Training Loop Configuration
TRAIN_CONFIG = {
    "steps_per_episode": 80,       # Total environment steps per episode
    "update_iters_per_step": 10,      # Gradient updates per environment step
    "fixed_hoprate": 100.0,          # Fixed hopping rate for training
}

# Figure Saving Configuration (train_offsets.py and train_mbpo.py)
# At each listed training step (1-based, matching the "Step i/N" log index),
# save the pre-action observation (agent's input state) as step_XXX_obs.png and
# one PSD figure per block as step_XXX_block_YY.png under output_dir/figures/.
# Multiple steps may be listed; values outside [1, steps_per_episode] are
# ignored with a warning. Empty list disables figure saving.
PLOT_CONFIG = {
    "figure_save_steps": [49, 50],
}

# Reward Calculation Configuration
# Matches FHSSQPSKEnv.step(): base_reward - BER * ber_penalty - hoprate * hoprate_penalty
REWARD_CONFIG = {
    "base_reward": 10.0,
    "ber_penalty": 80.0,
    "hoprate_penalty": 0,
}
