# RoboMimic_Deploy — Casbot Skeleton AMP 策略部署项目

## 项目概述

Casbot Skeleton（25-DOF 人形机器人）的 AMP 行走策略部署框架，包含 MuJoCo 仿真部署和真机部署。

- **策略**：MJLAB AMP 框架训练，ONNX 格式，输入 336 维（84维×4帧历史），输出 25 维动作
- **机器人**：25 关节（12腿 + 1腰 + 2头 + 10臂），模型在 `casbot_skeleton/`
- **电机参数**：`casbot_skeleton/casbot_constants.py` 是权威来源（6组真实参数）

## 关键文件地图

### Python MuJoCo 仿真部署
```
deploy_mujoco/deploy_casbot.py          ← 主入口 (Xbox手柄, 单线程)
deploy_mujoco/deploy_casbot_keyboard.py  ← 备份 (键盘控制)
policy/casbot_amp/CasbotAMP.py           ← 策略类 (观测构建+ONNX推理+动作缩放)
policy/casbot_amp/config/CasbotAMP.yaml  ← 策略配置 (Kp/Kd/限速/default_pose)
common/joystick.py                       ← Xbox手柄驱动 (pygame)
```

### C++ MuJoCo 仿真部署 (单进程)
```
wbc_fsm/casbot_deploy/
├── include/CasbotAmpDeploy.h           ← 策略类声明
├── src/CasbotAmpDeploy.cpp             ← 策略类实现 (与Python逻辑1:1对应)
├── src/main.cpp                        ← GLFW+MuJoCo主循环
├── config/casbot_amp.json              ← 策略配置
└── CMakeLists.txt                      ← 构建 (本地MuJoCo+ONNX Runtime, 零Python依赖)
```

### C++ DDS 架构部署 (控制器-仿真器分离, 仿G1)
```
wbc_fsm/casbot_dds/
├── include/FSM/State_CasbotAmp.h       ← FSM状态 (策略)
├── include/interface/IOSDK.h           ← DDS I/O层
├── src/FSM/State_CasbotAmp.cpp
└── simulate/                           ← DDS仿真器 (Python)
    ├── casbot_mujoco_sim.py
    └── dds_bridge.py
```

### ROS2 真机部署 (未完成编译)
```
wbc_fsm/casbot_ros2/
├── include/casbot_ros2/casbot_amp_node.hpp
└── src/casbot_amp_node.cpp
```

### G1 参考代码
```
wbc_fsm/                                ← G1 C++ FSM框架 (.cpp已加密)
deploy_mujoco/deploy_mujoco.py          ← G1 Python仿真 (FSM多状态)
unitree_mujoco/                         ← G1 DDS仿真器
hl_motion/                              ← Casbot真机SDK (ROS2 + EtherCAT)
```

## 仿真频率 (标准配置)
```
物理步进: 500 Hz (dt=0.002s)
策略推理:  50 Hz (每10步, control_decimation=10)
PD控制:   500 Hz (每步, 与物理同步, 有力矩钳位)
```

## 训练和电机参数
- 训练框架：与 G1 MJAMP 相同
- 电机参数在 `casbot_constants.py` 定义了 6 组：
  - LEG_BIG (pelvic_pitch/roll, knee): Kp=276.31, Kd=17.59, Effort=150
  - LEG_SMALL (pelvic_yaw, ankle_pitch/roll, waist): Kp=156.31, Kd=9.95, Effort=60
  - ARM_MID (shoulder_pitch/roll, elbow): Kp=130.20, Kd=8.29, Effort=75
  - ARM_SMALL (shoulder_yaw, wrist_yaw, head): Kp=96.83, Kd=6.16, Effort=36
- dof_action_scale = 0.25 × effort / Kp (来自 MJAMP 公式)
- 默认姿态：KNEES_BENT_KEYFRAME (微蹲)
- Python YAML 和 C++ JSON 配置必须保持同步

## 常见问题和注意事项

1. **C++ 编译**: 需要 MuJoCo C库 + ONNX Runtime + Eigen3 + GLFW。这些已本地化在 `wbc_fsm/mujoco/` 和 `wbc_fsm/onnxruntime-linux-x64-1.22.0/`，零Python依赖。

2. **仿真稳定性**: 高 Kp(276) 需要 dt≤0.002s。如果改 dt，必须同时用隐式积分器 (`mjINT_IMPLICITFAST`)。PD 有力矩钳位 (`std::clamp`)。

3. **C++ actuator_trnid 陷阱**: 关节ID在 `actuator_trnid[2*i]` (索引0)，不是 `[2*i+1]` (索引1=-1)。读取关节角度必须用 `jnt_qposadr[jid]` 不能直接用 `qpos[jid]`。

4. **C++ vsync**: `glfwSwapInterval(0)` 必须关掉，否则物理被锁在60fps。

5. **Python viewer.sync 慢**: viewer.sync() 可能阻塞 5-70ms。降频渲染 (render_decimation) 或独立线程可缓解，但最简单是单线程+降频。

6. **场景加载**: Python和C++都加载 `casbot_skeleton/scene.xml`，它 include `casbot_skeleton_25dof.xml`。后者只含机器人不含地板。

7. **速度限制同步**: Python YAML 和 C++ JSON 的 vx_limit_max 等参数要同步。

8. **关节顺序**: 策略输出顺序 = XML关节顺序 = MJLAB训练顺序 (L腿 R腿 腰 头 L臂 R臂)。
