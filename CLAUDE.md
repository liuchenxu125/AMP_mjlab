# AMP_mjlab Project Notes

## Overview
AMP motion control project built on mjlab + rsl_rl. A single PPO policy learns both locomotion (walk/run) and recovery (fall-and-get-up) via AMP discriminator.

**GitHub**: https://github.com/liuchenxu125/AMP_mjlab (fork of unitree_rl_mjlab)
**Remote name**: `myfork`, push to `master:main`

## Supported Robots

| Robot | DOF | Tasks | Motion Data |
|---|---|---|---|
| Casbot Skeleton | 25 | `Casbot-Skeleton-AMP-Rough/Flat` | `src/assets/motions/casbot_skeledon/amp/` |
| Marathon 001 | 18 | `Marathon-001-AMP-Rough/Flat` | `src/assets/motions/marathon_001/amp/` |
| Unitree G1 | 23 | `Unitree-G1-AMP-Rough/Flat` | `src/assets/motions/g1/amp/` |

## Key Directories
- `src/tasks/amp_loco/config/` — robot env/rl configs (casbot_skeleton, marathon_001, g1)
- `src/tasks/amp_loco/mdp/events.py` — MotionResetManager (motion loading for reset)
- `src/tasks/amp_loco/ampmotion_loader.py` — NPZ motion file loader
- `rsl_rl/runners/amp_on_policy_runner.py` — AMP training runner
- `rsl_rl/algorithms/amp_ppo.py` — AMP PPO algorithm with discriminator
- `rsl_rl/utils/motion_loader.py` — AMPLoader (expert data for discriminator)
- `scripts/train.py` — training entry point
- `scripts/play.py` — evaluation entry point
- `RoboMimic_Deploy/` — deployment code (FSM, ROS2 nodes, ONNX policies)

## Two Motion Loading Pipelines
1. **Pipeline A (Env Reset)**: `mdp.events.MotionResetManager` — loads WalkandRun + Recovery NPZ for episode reset. Recovery motions used for `delay_reset_env_ratio=0.4` environments.
2. **Pipeline B (AMP Discriminator)**: `rsl_rl.utils.motion_loader.AMPLoader` — recursively loads ALL .npz files as expert demonstrations for the discriminator.

## Training
```bash
python scripts/train.py Casbot-Skeleton-AMP-Flat --env.scene.num-envs=4096
```
Logs: `logs/rsl_rl/casbot_skeleton_amp_locomotion/<timestamp>/`

## Known Issues for Reproducibility

### 1. warp-lang version incompatibility
mjlab==1.2.0 uses `wp.context` API but warp-lang >=1.14.0 removed it. Fix:
```bash
pip install 'warp-lang>=1.12.0,<1.14.0'
```

### 2. rsl_rl version conflict
mjlab requires `rsl-rl-lib==5.0.1` but local `rsl_rl/` is based on 2.3.1 with custom AMP modules. Fix: install local rsl_rl as editable, which overrides the mjlab version:
```bash
pip install -e rsl_rl/
```

### 3. mjlab patch
`ObservationGroupCfg` needs `history_ordering` parameter added by patch:
```bash
SITE_PKGS=$(python -c "import mjlab, os; print(os.path.dirname(mjlab.__file__))")
cp mjlab_patch/mjlab/managers/observation_manager.py $SITE_PKGS/managers/
```

### 4. Fork LFS restriction
This is a public fork — GitHub does NOT allow Git LFS on forks. Large files (>100 MB) must be excluded from the repo.

## Install Order (clean env)
```bash
conda create -n mjlab python=3.11 -y && conda activate mjlab
pip install mjlab==1.2.0 'warp-lang>=1.12.0,<1.14.0'
pip install -e rsl_rl/
pip install -e .
# then apply mjlab patch (see above)
```

## Gitignore Exclusions
- `logs.png`, `nohup_*` (removed from gitignore — training logs ARE tracked now)
- `RoboMimic_Deploy/wbc_fsm/hl_motion/` — compiled .so libraries
- `RoboMimic_Deploy/wbc_fsm/onnxruntime-linux-x64-1.22.0/`
- `RoboMimic_Deploy/unitree_sdk2_python/`
- `motion_data_csv/`

## Current Status (2026-06-26)
- Casbot has learned fall-and-get-up recovery
- XML collision bodies optimized — collection time reduced from 10s to 2s
- Recovery motion data: `src/assets/motions/casbot_skeledon/amp/Recovery/fallAndGetUp1_406_1950.npz`
- Walking shows slight veering — likely due to turning motions in WalkandRun training data mixing with forward walk
