# Offline replay workflow (v3)

One v3 transition represents one complete environment step: one PSD state, the
selected hoprate, ten offset actions, ten block rewards, the next PSD state,
and the hoprate that will be supplied to the offset policy at the next step.

The fixed-hoprate baseline keeps its existing default dataset:

```text
outputs/offline_replay/replay_5000_100_hoprate_v3.npz
```

The derivative-NBS joint trainers use a separate 5,000-step random-hoprate
dataset by default:

```text
outputs/offline_replay/replay_5000_random_hoprate_v3.npz
```

Generate the joint reactive+comb dataset with uniformly random valid 10 Hz
hoprates and independently uniform ten-offset actions:

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --num_step_transitions 5000 --hoprate_mode random --enable_reactive true --enable_sweep true --jammer_mode comb --output_path outputs/offline_replay/replay_5000_random_hoprate_v3.npz
```

Generate the two-transition smoke dataset used by the integration check:

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --num_step_transitions 2 --hoprate_mode random --enable_reactive true --enable_sweep true --jammer_mode comb --output_path outputs/offline_replay/smoke_random_reactive_comb_v3.npz
```

The original fixed-hoprate dataset can still be generated independently:

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --num_step_transitions 5000 --hoprate_mode fixed --fixed_hoprate 100 --output_path outputs/offline_replay/replay_5000_100_hoprate_v3.npz
```

Each v3 file contains:

```text
state_imgs       [N, H, W]
hoprates         [N]
actions          [N, 10]
block_rewards    [N, 10]
step_rewards     [N]
next_state_imgs  [N, H, W]
next_hoprates    [N]
dones            [N]
```

For randomly sampled nonterminal replay, `next_hoprates[n]` is sampled before
the next transition and becomes `hoprates[n+1]`, so the stored successor policy
input is causally consistent. `step_rewards[n]` must equal
`mean(block_rewards[n])`.

The loader validates observation shape, action range, block count, finite
values, metadata count and replay capacity. Both joint trainers reject
environment, jammer or reward metadata mismatches by default. Use
`--allow_replay_config_mismatch` only for an explicit cross-configuration
experiment. The original baseline keeps its warning-only compatibility
behavior, while the original MBPO entry remains strict by default.

Any change to jammer timing, comb channel groups, jammer mode,
`baseband_variant_count`, environment parameters or reward coefficients
invalidates the normal-use replay dataset and requires regeneration.

Select or disable replay explicitly when training:

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_joint_sac.py --offline_replay_path outputs/offline_replay/replay_5000_random_hoprate_v3.npz
D:\Anaconda\envs\rl_fhss\python.exe train_joint_mbpo.py --offline_replay_path outputs/offline_replay/replay_5000_random_hoprate_v3.npz
D:\Anaconda\envs\rl_fhss\python.exe train_joint_sac.py --offline_replay_path none
```

Formats v1 and v2 modeled each block as a separate transition with a
`block_idx`. They cannot represent the ten-head step transition and are
rejected rather than converted.
