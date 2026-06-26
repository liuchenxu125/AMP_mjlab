# AMP_mjlab — 人形机器人AMP拟人行走训练框架

## 项目概述

基于 Isaac Lab / RSL-RL 的 AMP (Adversarial Motion Priors) 人形机器人运动训练框架。通过对抗学习让机器人模仿参考动作数据的运动风格，同时完成速度指令跟踪任务。

## 目录结构

```
AMP_mjlab/
├── src/
│   ├── assets/
│   │   ├── robots/           # 机器人模型和电机配置
│   │   │   ├── unitree_g1/       # G1 (29DOF) — 参考机器人
│   │   │   ├── casbot_skeleton/  # Casbot (25DOF) — 主要训练机器人
│   │   │   └── marathon_001/     # Marathon (18DOF) — 无腰部无头部
│   │   └── motions/          # 参考动作数据 (CSV + NPZ)
│   │       ├── g1/amp/           # G1动作 (WalkandRun + Recovery)
│   │       ├── casbot_skeledon/amp/  # Casbot动作 (WalkandRun + Recovery)
│   │       └── marathon_001/amp/     # Marathon动作 (WalkandRun)
│   └── tasks/
│       └── amp_loco/         # AMP运动训练任务
│           ├── amp_env_cfg.py            # AMP基础环境配置 (工厂函数)
│           ├── ampmotion_loader.py       # 环境Reset用MotionLoader
│           ├── mdp/                      # 自定义MDP函数
│           └── config/                   # 各机器人专属配置
│               ├── g1/                   # G1 AMP配置
│               ├── casbot_skeleton/      # Casbot AMP配置
│               └── marathon_001/         # Marathon AMP配置
├── rsl_rl/                   # RSL-RL训练库 (AMP扩展)
│   ├── algorithms/amp_ppo.py             # AMP PPO算法
│   ├── runners/amp_on_policy_runner.py   # AMP训练Runner
│   ├── modules/discriminator.py          # AMP判别器
│   └── utils/motion_loader.py            # AMP判别器用AMPLoader
├── scripts/
│   ├── train.py              # 训练入口
│   ├── csv_to_npz.py         # 通用CSV→NPZ转换
│   ├── convert_casbot_motion.py  # Casbot专用CSV→NPZ转换
│   └── replay_motion.py      # NPZ动作可视化播放器
└── mjlab_patch/              # mjlab框架补丁
```

## 已配置的机器人

### 1. Casbot Skeleton (25DOF) — 主要训练对象
- 路径: `src/assets/robots/casbot_skeleton/`
- XML: `xmls/casbot_skeleton_25dof.xml` (碰撞胶囊体+IMU传感器)
- 电机: `casbot_constants.py` (真实参数) / `casbot_constants_ys.py` (G1参数对比)
- 关节: 12腿 + 1腰 + 2头 + 10臂 = 25 DOF
- anchor=waist_yaw_link, root=base_link
- 13个AMP观测body
- 任务ID: `Casbot-Skeleton-AMP-Flat`, `Casbot-Skeleton-AMP-Rough`

### 2. Marathon 001 (18DOF)
- 路径: `src/assets/robots/marathon_001/`
- 关节: 12腿 + 6臂 = 18 DOF (无腰、无头、无腕)
- 任务ID: `Marathon-001-AMP-Flat`, `Marathon-001-AMP-Rough`

### 3. Unitree G1 (29DOF) — 参考机器人
- 任务ID: `Unitree-G1-AMP-Flat`, `Unitree-G1-AMP-Rough`

## 核心概念

### AMP训练流程
```
参考动作NPZ → 判别器学习"真人运动风格"
策略 → 输出关节位置 → PD控制器 → 力矩 → 物理仿真
判别器判断策略产生的运动是否"自然" → style reward
总奖励 = (1-0.75)*风格奖励 + 0.75*任务奖励
```

### 关键参数
- 物理频率: 200Hz (timestep=0.005s), 控制频率: 50Hz (decimation=4)
- episode: 20s, 观测历史: actor=4帧, amp=1帧
- AMP: amp_reward_coef=0.1, amp_task_reward_lerp=0.75
- PPO: 5epochs, 4mini_batches, lr=1e-3

### 两种MotionLoader
1. MotionLoader (ampmotion_loader.py): 环境reset用，从NPZ随机采样帧
2. AMPLoader (rsl_rl/utils/motion_loader.py): 判别器训练用，计算body相对anchor的局部坐标

### 域随机化
- 观测噪声: base_ang_vel(±0.2), gravity(±0.05), joint_pos(±0.01), joint_vel(±0.5)
- 物理参数: 脚底摩擦(0.3~1.2), 编码器偏置(±0.015rad), 质心偏移(±2.5cm)
- 外力扰动: 每1~3秒随机推搡

### 动作空间
- 关节位置控制: 策略输出×action_scale → 关节角偏移(rad) → +default_joint_pos → PD目标

## CSV→NPZ转换

### Casbot (30fps输入)
```bash
python scripts/convert_casbot_motion.py --input-dir <csv_dir> --output-dir <npz_dir>
```

### 其他机器人
```bash
python scripts/csv_to_npz.py --robot <robot_name> --input-dir <csv_dir> --output-dir <npz_dir> --input-fps 30
```

CSV格式: base_pos(3) + base_quat_xyzw(4) + joint_pos(25) = 32列
NPZ输出: fps, joint_pos/vel, body_pos/quat/lin_vel/ang_vel_w

## 常用命令
```bash
# 训练
python scripts/train.py Casbot-Skeleton-AMP-Flat --env.scene.num-envs=4096

# NPZ动作回放
python scripts/replay_motion.py <npz> --robot casbot_skeleton [--speed 0.5] [--loop]

# Casbot电机配置验证
python -m src.assets.robots.casbot_skeleton.casbot_constants
```
