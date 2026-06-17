# AMP_mjlab

[English README](README.md)

部署集成代码位于 [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm) 项目中的 `MJAmp State`。

基于 [mjlab](https://github.com/mujocolab/mjlab) + rsl_rl 的 AMP 运动控制项目。使用单一策略同时学习 locomotion（走/跑）与 recovery（跌倒恢复），通过 AMP 判别器约束动作风格与运动先验。

## 支持的机器人

| 机器人 | 自由度 | 任务 ID | 运动数据路径 |
|---|---|---|---|
| **Casbot Skeleton** | 25 | `Casbot-Skeleton-AMP-Rough`, `Casbot-Skeleton-AMP-Flat` | `src/assets/motions/casbot_skeledon/amp/WalkandRun` |
| **Marathon 001** | 18 | `Marathon-001-AMP-Rough`, `Marathon-001-AMP-Flat` | `src/assets/motions/marathon_001/amp/WalkandRun` |
| **Unitree G1** | 23 | `Unitree-G1-AMP-Rough`, `Unitree-G1-AMP-Flat` | `src/assets/motions/g1/amp/WalkandRun`, `src/assets/motions/g1/amp/Recovery` |

## 核心思路

传统做法常把"走跑策略"和"恢复策略"分开训练并做切换；本项目将两类能力放入一个策略中统一学习。

实现要点：

- 运动数据分组：
  - Walk/Run 数据目录：`src/assets/motions/{robot}/amp/WalkandRun`
  - Recovery 数据目录：`src/assets/motions/{robot}/amp/Recovery`
- 延迟重置机制（Delayed Termination）：
  - 一部分环境在触发终止后不立即 reset，而是给定恢复窗口
  - 该子集环境优先从 Recovery 片段采样 reset 状态
- 统一 AMP 训练：
  - 单一 actor-critic + 单一 AMP discriminator
  - 在同一训练过程中学习速度跟踪、抗扰动与恢复能力

## 环境要求

- Linux（推荐 Ubuntu 20.04/22.04）
- NVIDIA GPU（训练需要 CUDA；macOS 仅支持评估）
- Python 3.11（推荐；支持 3.8+）
- 已可用的 MuJoCo / GPU 驱动环境

## 快速开始

### 1. 创建 conda 环境

```bash
conda create -n mjlab python=3.11 -y
conda activate mjlab
```

### 2. 安装依赖

```bash
# 从 PyPI 安装 mjlab（会自动安装大部分依赖：torch, mujoco, wandb, tyro 等）
pip install mjlab==1.2.0

# 安装本地 rsl_rl（含自定义的 AMP 算法修改）
cd AMP_mjlab
pip install -e rsl_rl/

# 安装本项目
pip install -e .
```

### 3. 应用 mjlab 补丁（可选）

如果不打这个补丁，则需要在代码中去掉 `history_ordering` 配置。

补丁作用说明：

- 增加了历史观测的展开方式选项，可选择按时间维(`time`)或按观测项(`term`)展开。
- mjlab 默认仅支持按 `term` 展开。

补丁文件：

- `mjlab_patch/mjlab/managers/observation_manager.py`

示例覆盖命令（请根据实际 conda 环境路径调整）：

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  "$(dirname $(which python))/../lib/python3.11/site-packages/mjlab/managers/observation_manager.py"
```

也可以用 `pip show mjlab` 查看安装位置。

### 4. 查看可用任务

```bash
python scripts/list_envs.py --keyword AMP
```

主要任务：

- `Casbot-Skeleton-AMP-Rough` / `Casbot-Skeleton-AMP-Flat`
- `Marathon-001-AMP-Rough` / `Marathon-001-AMP-Flat`
- `Unitree-G1-AMP-Rough` / `Unitree-G1-AMP-Flat`

## 训练

```bash
# 训练 Casbot Skeleton（平地）
python scripts/train.py Casbot-Skeleton-AMP-Flat --env.scene.num-envs=4096

# 训练 Marathon 001（崎岖地形）
python scripts/train.py Marathon-001-AMP-Rough --env.scene.num-envs=4096

# 训练 Unitree G1（平地）
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

日志默认保存路径：

- `logs/rsl_rl/casbot_skeleton_amp_locomotion/<timestamp>/`
- `logs/rsl_rl/marathon_001_amp_locomotion/<timestamp>/`
- `logs/rsl_rl/g1_amp_locomotion/<timestamp>/`

## 训练曲线说明（重要）

- 在约 `2w` 轮（约 20k iterations）附近，策略通常会突然学会"跌倒后恢复"行为。
- 对应地，`logs` 中多个指标会出现明显突变（阶跃式变化），这是正常现象，不一定是训练异常。

![训练日志突变示例](logs.png)

## 评估与可视化

使用已训练权重回放：

```bash
python scripts/play.py Casbot-Skeleton-AMP-Rough \
  --checkpoint-file logs/rsl_rl/casbot_skeleton_amp_locomotion/<run_dir>/model_<iter>.pt
```

说明：训练与回放阶段都支持 ONNX 导出（默认开启）。

## 运动数据准备

各机器人的 NPZ 运动数据已经包含在 `src/assets/motions/` 目录下。如有新的 CSV 格式运动数据，可使用转换脚本：

```bash
python scripts/csv_to_npz.py --help
```

推荐目录组织：

- 原始 CSV：`motion_data_csv/amp`
- 转换后 NPZ：`src/assets/motions/{robot}/amp/WalkandRun` 与 `src/assets/motions/{robot}/amp/Recovery`

只要上述目录中存在可用 NPZ，训练配置会自动加载。

## 目录说明

- `src/tasks/amp_loco`：AMP locomotion/recovery 任务实现
- `src/tasks/amp_loco/config/`：各机器人配置（casbot_skeleton, marathon_001, g1）
- `src/tasks/amp_loco/mdp`：奖励、观测、事件、终止逻辑
- `src/assets/motions/`：各机器人的运动数据（NPZ + CSV）
- `src/assets/robots/`：机器人 MJCF/URDF 模型、mesh 文件和常量配置
- `scripts/train.py`：训练入口
- `scripts/play.py`：回放入口
- `scripts/csv_to_npz.py`：运动数据转换工具
- `scripts/replay_motion.py`：运动回放工具
- `scripts/list_envs.py`：列出已注册的任务
- `rsl_rl/`：本地 rsl_rl 副本（含自定义 AMP 模块）
- `mjlab_patch/`：依赖的 mjlab 本地补丁

## 项目亮点总结

- 单一策略统一覆盖走跑与跌倒恢复
- AMP + 速度任务联合优化，兼顾风格与任务性能
- 延迟重置与 recovery 采样机制，显式强化恢复能力
- 训练到部署链路完整，支持 ONNX 导出
- 多机器人支持：Casbot Skeleton、Marathon 001、Unitree G1

## 致谢

- 感谢 [mujocolab/mjlab](https://github.com/mujocolab/mjlab) 提供的仿真与强化学习框架。
- 感谢 [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) 项目的开源工作与启发。
- 感谢 [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab)，本项目在 rsl_rl 的 AMP 部分参考了该实现。
