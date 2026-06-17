# AMP_mjlab

[中文 README](README_zh.md)

Deployment integration code is in [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm), under `MJAmp State`.

AMP motion control project built on top of [mjlab](https://github.com/mujocolab/mjlab) + rsl_rl. A single policy learns both locomotion (walk/run) and recovery (fall-and-get-up) via AMP discriminators that regularize motion style and priors.

## Supported Robots

| Robot | DOF | Task IDs | Motion Data |
|---|---|---|---|
| **Casbot Skeleton** | 25 | `Casbot-Skeleton-AMP-Rough`, `Casbot-Skeleton-AMP-Flat` | `src/assets/motions/casbot_skeledon/amp/WalkandRun` |
| **Marathon 001** | 18 | `Marathon-001-AMP-Rough`, `Marathon-001-AMP-Flat` | `src/assets/motions/marathon_001/amp/WalkandRun` |
| **Unitree G1** | 23 | `Unitree-G1-AMP-Rough`, `Unitree-G1-AMP-Flat` | `src/assets/motions/g1/amp/WalkandRun`, `src/assets/motions/g1/amp/Recovery` |

## Core Idea

Instead of training separate policies for locomotion and recovery and switching between them, this project learns both capabilities in one unified policy.

Implementation highlights:

- Motion data split:
  - Walk/Run data: `src/assets/motions/{robot}/amp/WalkandRun`
  - Recovery data: `src/assets/motions/{robot}/amp/Recovery`
- Delayed termination/reset:
  - A subset of environments does not reset immediately after termination
  - These environments receive a recovery window and reset states sampled from recovery clips
- Unified AMP training:
  - One actor-critic + One AMP discriminator
  - Velocity tracking, perturbation robustness, and recovery are learned together

## Requirements

- Linux (Ubuntu 20.04/22.04 recommended)
- NVIDIA GPU (training requires CUDA; macOS supported for evaluation only)
- Python 3.11 (recommended; 3.8+ supported)
- Working MuJoCo and GPU driver setup

## Quick Start

### 1. Create conda environment

```bash
conda create -n mjlab python=3.11 -y
conda activate mjlab
```

### 2. Install dependencies

```bash
# Install mjlab from PyPI (pulls most dependencies: torch, mujoco, wandb, tyro, etc.)
pip install mjlab==1.2.0

# Install the local rsl_rl copy (contains custom AMP algorithm modifications)
cd AMP_mjlab
pip install -e rsl_rl/

# Install this project
pip install -e .
```

### 3. Apply mjlab Patch (Optional)

If you do not apply this patch, remove `history_ordering` configuration from the code.

What this patch does:

- It adds an option for how observation history is flattened: by time (`time`) or by term (`term`).
- Default mjlab behavior supports only `term` ordering.

Patch file:

- `mjlab_patch/mjlab/managers/observation_manager.py`

Example command (adjust path to your conda environment):

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  "$(dirname $(which python))/../lib/python3.11/site-packages/mjlab/managers/observation_manager.py"
```

Or use `pip show mjlab` to find the install location.

### 4. List Available Tasks

```bash
python scripts/list_envs.py --keyword AMP
```

Main tasks:

- `Casbot-Skeleton-AMP-Rough` / `Casbot-Skeleton-AMP-Flat`
- `Marathon-001-AMP-Rough` / `Marathon-001-AMP-Flat`
- `Unitree-G1-AMP-Rough` / `Unitree-G1-AMP-Flat`

## Training

```bash
# Train Casbot Skeleton on flat terrain
python scripts/train.py Casbot-Skeleton-AMP-Flat --env.scene.num-envs=4096

# Train Marathon 001 on rough terrain
python scripts/train.py Marathon-001-AMP-Rough --env.scene.num-envs=4096

# Train Unitree G1 on flat terrain
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

Logs are saved by default to:

- `logs/rsl_rl/casbot_skeleton_amp_locomotion/<timestamp>/`
- `logs/rsl_rl/marathon_001_amp_locomotion/<timestamp>/`
- `logs/rsl_rl/g1_amp_locomotion/<timestamp>/`

## Training Curve Note (Important)

- Around `2w` iterations (about 20k), the policy often suddenly learns fall-recovery behavior.
- As a result, multiple metrics in `logs` may show abrupt jumps. This is expected and not necessarily a training failure.

![Training log transition example](logs.png)

## Evaluation and Visualization

Replay with a trained checkpoint:

```bash
python scripts/play.py Casbot-Skeleton-AMP-Rough \
  --checkpoint-file logs/rsl_rl/casbot_skeleton_amp_locomotion/<run_dir>/model_<iter>.pt
```

Note: ONNX export is enabled by default in both training and play workflows.

## Motion Data Preparation

NPZ motion files for supported robots are already included under `src/assets/motions/`. If you have new motion data in CSV format, use the conversion script:

```bash
python scripts/csv_to_npz.py --help
```

Recommended data layout:

- Raw CSV: `motion_data_csv/amp`
- Converted NPZ: `src/assets/motions/{robot}/amp/WalkandRun` and `src/assets/motions/{robot}/amp/Recovery`

If valid NPZ files exist in these folders, training config loads them automatically.

## Repository Structure

- `src/tasks/amp_loco`: AMP locomotion/recovery task implementation
- `src/tasks/amp_loco/config/`: Robot configs (casbot_skeleton, marathon_001, g1)
- `src/tasks/amp_loco/mdp`: rewards, observations, events, termination logic
- `src/assets/motions/`: motion data (NPZ + CSV) for each robot
- `src/assets/robots/`: robot MJCF/URDF models, meshes, and constants
- `scripts/train.py`: training entry point
- `scripts/play.py`: playback entry point
- `scripts/csv_to_npz.py`: motion data conversion tool
- `scripts/replay_motion.py`: motion replay utility
- `scripts/list_envs.py`: list registered tasks
- `rsl_rl/`: local rsl_rl copy with custom AMP modules
- `mjlab_patch/`: required local patch for mjlab

## Highlights

- One policy unifies walk/run and recovery skills
- AMP + velocity objective jointly optimize style and task performance
- Delayed reset with recovery sampling explicitly improves recovery ability
- End-to-end pipeline supports ONNX export for deployment
- Multi-robot support: Casbot Skeleton, Marathon 001, Unitree G1

## Acknowledgements

- Thanks to [mujocolab/mjlab](https://github.com/mujocolab/mjlab) for the simulation and RL framework.
- Thanks to [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) for open-sourcing their work and inspiration.
- Thanks to [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab); the rsl_rl AMP part in this project references their implementation.
