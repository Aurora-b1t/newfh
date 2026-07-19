"""
Central configuration for the FHSS RL anti-jamming training project.

Defines environment parameters, jammer settings, SAC and legacy MBPO
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
    "pregen_steps": 44,         # Align with 4.4s cycle (0.1s step)
    "noise_std": 0.1,           # Thermal noise std at receiver (also used by reactive jammer)
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
    
    # Sweep Jamming Configuration
    "sweep": {
        "step": 50000,       # Frequency step for sweep
        "power": 0.8,        # Jamming power
        "dwell_time": 0.004, # Dwell time per step
        "bandwidth": 50000.0,# Noise bandwidth
    },
    
    # Comb Jamming Configuration
    "comb": {
        # Note: Frequency selection is HARDCODED in jammers.py to 8 fixed points (centers of 50kHz channels).
        "power": 0.8,        # Total power or per-tone power factor
        "bandwidth": 50000.0,# Noise bandwidth per tone
        # "frequencies": [] # Removed as it is now hardcoded in logic
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
    "capacity": 50000,
    "batch_size": 256,
}

# Offline real-environment replay configuration. Each transition represents one
# complete environment step with ten offset actions and ten block rewards.
OFFLINE_REPLAY_CONFIG = {
    "num_step_transitions": 50000,
    "default_path": "outputs/offline_replay/replay_50000_random_hoprate_v3.npz",
    "hoprate_mode": "random",
}

# MBPO Reward-Model Configuration
MBPO_CONFIG = {
    "num_networks": 5,
    "num_elites": 3,
    "hidden_size": 200,
    "model_train_freq": 1,
    "model_train_batch_size": 256,
    "rollout_batch_size": 2000,
    "rollout_length": 1,
    "real_ratio": 0.2,
    "model_replay_size": 30000,
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
}

# Training Loop Configuration
TRAIN_CONFIG = {
    "steps_per_episode": 150,       # Total environment steps per episode
    "update_iters_per_step": 10,      # Gradient updates per environment step
    "fixed_hoprate": 100.0,          # Fixed hopping rate for training
}

# Reward Calculation Configuration
# Matches FHSSQPSKEnv.step(): base_reward - BER * ber_penalty - hoprate * hoprate_penalty
REWARD_CONFIG = {
    "base_reward": 1.0,
    "ber_penalty": 8.0,
    "hoprate_penalty": 0,
}
