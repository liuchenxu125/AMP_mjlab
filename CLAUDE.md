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

---

# RoboMimic_Deploy — Casbot Skeleton AMP 策略部署项目

## 项目概述

Casbot Skeleton（25-DOF 人形机器人）的 AMP 行走策略 MuJoCo 仿真部署和真机部署代码。将训练好的 ONNX 策略模型加载到 MuJoCo 物理引擎中运行，支持 Xbox 手柄实时控制。

- **ONNX 模型**：输入 336 维（84维×4帧历史），输出 25 维动作
- **部署版本**：Python 单线程版、C++ 单进程版、C++ DDS 双进程版、ROS2 真机版
- **电机参数权威来源**：`casbot_skeleton/casbot_constants.py`（6组真实参数）

## RoboMimic_Deploy 目录结构

```
RoboMimic_Deploy/
├── deploy_mujoco/
│   ├── deploy_casbot.py              # Python 主入口 (Xbox手柄)
│   ├── deploy_casbot_keyboard.py     # Python 备份 (键盘)
│   └── config/mujoco.yaml            # G1仿真参数参考
├── policy/casbot_amp/
│   ├── CasbotAMP.py                  # 策略类 (观测构建+ONNX推理+动作缩放)
│   └── config/CasbotAMP.yaml         # 策略配置 (Kp/Kd/限速/default_pose)
├── casbot_skeleton/
│   ├── scene.xml                     # 场景 (include机器人+地板+光照)
│   ├── casbot_skeleton_25dof.xml     # 机器人模型 (25关节+执行器+IMU传感器)
│   ├── policy.onnx                   # 训练好的AMP策略
│   └── meshes/*.STL                  # 31个网格文件
├── common/
│   ├── joystick.py                   # Xbox手柄驱动 (pygame)
│   └── ctrlcomp.py                   # StateAndCmd / PolicyOutput 数据结构
├── wbc_fsm/
│   ├── casbot_deploy/                # ★ C++ 单进程MuJoCo部署 (零Python依赖)
│   │   ├── include/CasbotAmpDeploy.h
│   │   ├── src/CasbotAmpDeploy.cpp
│   │   ├── src/main.cpp
│   │   ├── config/casbot_amp.json
│   │   └── CMakeLists.txt
│   ├── casbot_dds/                   # ★ C++ DDS架构 (仿G1, 控制器-仿真器分离)
│   │   ├── include/FSM/State_CasbotAmp.h
│   │   ├── include/interface/IOSDK.h
│   │   ├── simulate/                 # DDS仿真器 (Python)
│   │   └── CMakeLists.txt
│   ├── casbot_ros2/                  # ★ ROS2真机部署 (编译依赖机器人上的crb_ros_msg)
│   ├── mujoco/                       # MuJoCo C库 (从pip包提取, 5.4MB)
│   ├── onnxruntime-linux-x64-1.22.0/ # ONNX Runtime C库 (22MB)
│   ├── unitree_mujoco/               # G1 DDS仿真器 (参考)
│   └── hl_motion/                    # Casbot真机SDK (ROS2 + EtherCAT, 参考)
└── environment.yml                   # Conda环境一键配置
```

## 仿真频率配置

```
物理步进: 500 Hz (dt=0.002s)
策略推理:  50 Hz (每10步, control_decimation=10)
PD控制:   500 Hz (每步, 与物理同步, 有力矩钳位)
渲染:     viewer.sync() 每帧调用 (降频到~60Hz可减少阻塞)
```

## 电机参数 (6组, 来自casbot_constants.py)

| 组 | 关节 | Kp | Kd | Effort | dof_action_scale |
|---|---|---|---|---|---|
| LEG_BIG | pelvic_pitch/roll, knee (×6) | 276.31 | 17.59 | 150 | 0.1357 |
| LEG_SMALL | pelvic_yaw, ankle_pitch/roll, waist (×7) | 156.31 | 9.95 | 60 | 0.0960 |
| ARM_MID | shoulder_pitch/roll, elbow (×6) | 130.20 | 8.29 | 75 | 0.1440 |
| ARM_SMALL | shoulder_yaw, wrist_yaw, head (×6) | 96.83 | 6.16 | 36 | 0.0930 |

- dof_action_scale = 0.25 × effort / Kp (来自 MJAMP 公式)
- 默认姿态：KNEES_BENT_KEYFRAME (微蹲)
- Python YAML 和 C++ JSON 配置必须保持同步

## 常见问题和注意事项

### C++ 编译
- MuJoCo C库 + ONNX Runtime C库已本地化在 `wbc_fsm/mujoco/` 和 `wbc_fsm/onnxruntime-linux-x64-1.22.0/`
- 编译命令: `cd wbc_fsm/casbot_deploy/build && cmake .. && make -j`
- 同事无需Python/conda, 只需 `apt install libeigen3-dev nlohmann-json3-dev libglfw3-dev cmake g++`

### C++ actuator_trnid 陷阱
- 关节ID在 `actuator_trnid[2*i]` (索引0), **不是** `[2*i+1]` (索引1=-1)
- 读取关节角度必须用 `jnt_qposadr[jid]`，不能直接用 `qpos[jid]`（自由基座占前面7个）

### C++ vsync
- `glfwSwapInterval(0)` 必须关掉，否则物理被锁在60fps（`glfwSwapInterval(1)` = 每帧等16ms显示器刷新）

### 仿真稳定性
- 高 Kp(276) 需要 dt≤0.002s
- PD 有力矩钳位 (`np.clip` / `std::clamp`)，防止力矩过大导致 MuJoCo 发散
- 如果改大 dt，需加隐式积分器 (`m.opt.integrator = mjINT_IMPLICITFAST`)

### Python viewer.sync 性能
- `viewer.sync()` 可能偶尔阻塞 5-70ms（GPU/vsync相关），不是代码问题
- 降频渲染 (`render_decimation=8`) 可缓解

### 配置同步
- `policy/casbot_amp/config/CasbotAMP.yaml` (Python) 和 `wbc_fsm/casbot_deploy/config/casbot_amp.json` (C++) 的速度限制、电机参数必须同步

### 关节顺序
- 策略输出顺序 = XML关节顺序 = MJLAB训练顺序: L腿(6) R腿(6) 腰(1) 头(2) L臂(5) R臂(5)

### C++ 文件路径
- 场景: `casbot_skeleton/scene.xml` → include `casbot_skeleton_25dof.xml`
- ONNX: `casbot_skeleton/policy.onnx` (Python直接读, C++通过符号链接在 `model/policy.onnx`)

### G1 与 Casbot 架构对比
- G1: DDS双进程(控制器50Hz↔仿真器200Hz+1000Hz DDS Bridge), FSM多状态, 真机/仿真同一套代码
- Casbot: 单进程直接集成(物理+渲染+控制全在一进程), 单策略无状态机, 简单直观
- Casbot DDS版: 仿G1架构, 控制器和仿真器分离, 代码已写好但缺DDS仿真器对端
