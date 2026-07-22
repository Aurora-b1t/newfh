# Offline replay workflow (v3)

The baseline offset SAC loads complete environment-step transitions before its
first update. The default path is:

```text
outputs/offline_replay/replay_50000_random_hoprate_v3.npz
```

Generate the default 50,000 step transitions with random valid hoprates and
random ten-offset actions:

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py
```

Use a fixed hoprate or create a small smoke dataset:

```bash
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --hoprate_mode fixed --fixed_hoprate 100 --output_path outputs/offline_replay/replay_50000_fixed_100_v3.npz
D:\Anaconda\envs\rl_fhss\python.exe generate_offline_replay.py --num_step_transitions 2 --output_path outputs/offline_replay/smoke_v3.npz
```

Each v3 transition stores:

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

`step_rewards[n]` must equal `mean(block_rewards[n])`. The loader also validates
observation shape, action range, block count, finite values, metadata count and
buffer capacity. Baseline training warns about environment metadata differences;
MBPO rejects them by default and requires `--allow_replay_config_mismatch` to
override that check.

Select a dataset for baseline training:

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --offline_replay_path outputs/offline_replay/replay_50000_fixed_100_v3.npz
```

For explicit online-only collection, disable preload and wait for replay
warm-up:

```bash
D:\Anaconda\envs\rl_fhss\python.exe train_offsets.py --offline_replay_path none
```

Formats v1 and v2 modeled each block as a separate transition with a
`block_idx`. They cannot represent the new ten-head step transition and are
rejected rather than converted. Regenerate those datasets with the v3 generator.

The default now means 50,000 real environment steps, not 50,000 block rows.
Current measurements are roughly 15 seconds for initial pre-generation and
0.5 seconds per step, so full generation can take several hours. Both baseline
offset SAC and the step-level MBPO reward ensemble consume this v3 format.
