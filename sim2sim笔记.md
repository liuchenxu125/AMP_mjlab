Casbot C++ MuJoCo 部署 — 逐行讲解

先备知识：C++ 和 Python 的核心区别

┌──────┬────────────────────┬──────────────────────────────────────────────────────────┐
│ 概念 │       Python       │                           C++                            │
├──────┼────────────────────┼──────────────────────────────────────────────────────────┤
│ 变量 │ x = 5 自动推断类型 │ int x = 5; 必须声明类型                                  │
├──────┼────────────────────┼──────────────────────────────────────────────────────────┤
│ 数组 │ [1,2,3] 动态       │ std::array<int,3> 固定大小,编译时确定                    │
├──────┼────────────────────┼──────────────────────────────────────────────────────────┤
│ 函数 │ def f(x):          │ int f(int x) { } 返回类型+参数类型都要写                 │
├──────┼────────────────────┼──────────────────────────────────────────────────────────┤
│ 类   │ class A:           │ class A { public: ... private: ... }; 分public/private区 │
├──────┼────────────────────┼──────────────────────────────────────────────────────────┤
│ 编译 │ 不需要             │ 必须 cmake + make 生成可执行文件                         │
└──────┴────────────────────┴──────────────────────────────────────────────────────────┘

---
文件 1：CasbotAmpDeploy.h（头文件 = Python 里的 import 声明）

第 22-23 行：头文件保护

#ifndef CASBOT_AMP_DEPLOY_H   // "如果还没定义过这个宏"
#define CASBOT_AMP_DEPLOY_H   // "那就定义它"

作用：防止同一个文件被 #include 两次。Python 不需要这个（自动处理）。

第 25-36 行：#include = Python 的 import

#include <onnxruntime_cxx_api.h>   // import onnxruntime
#include <array>                   // import array (固定大小数组)
#include <vector>                  // import list   (动态数组)
#include <string>                  // import str

第 38 行：#define = Python 的常量

#define CASBOT_NUM_DOF 25

等价于 Python 的 CASBOT_NUM_DOF = 25。但 #define 是文本替换——编译前把所有 CASBOT_NUM_DOF 替换成 25。比 Python 变量更快（零运行时开销）。

第 47-74 行：namespace = Python 的模块/命名空间

namespace CasbotMotor {                  // 类似 Python: class CasbotMotor:
    constexpr double ARMATURE_LEG_BIG = 0.06999046;
    //           ^^^^^^                    ^^^^^^^^^^
    //           类型                      值
    //
    // constexpr = 编译期常量, 类似 Python 里没法修改的模块级变量
    // double    = 64位浮点数 (Python 的 float)
}

这 6 组电机参数跟 casbot_constants.py 里的完全一致，只是翻译成了 C++ 语法。

第 92-107 行：类的声明

class CasbotAmpDeploy {         // Python: class CasbotAMP:
public:                         // public = 外部可以调用的方法
    struct StepResult {         //     类似 Python 的 dataclass / dict
        std::array<float, 25> actions{};  // float actions[25] 全初始化为 0
        std::array<float, 25> kps{};
        std::array<float, 25> kds{};
        bool terminated = false;
    };

    explicit CasbotAmpDeploy(const std::string &configPath);
    //     ^^^^^^^^^^^^^^^^
    //     构造函数 = Python 的 __init__(self, configPath)
    //     explicit 防止隐式类型转换 (安全措施)

Python 对照：
class CasbotAMP:
    def __init__(self, config_path: str):   # ← C++ 的构造函数
        ...

第 121-124 行：initBuffers 的参数类型

void initBuffers(
    const std::array<float, 4> &baseQuat,   // 4个float的数组, 只读引用
    //  ^^^^^                    ^
    //  只读(不会修改)            & = 引用传递 (类似 Python 的传引用, 避免拷贝)
    const std::array<float, 3> &angVel,
    const std::array<float, 25> &q,
    const std::array<float, 25> &dq
);

Python 对照：
def init_buffers(self, base_quat: np.ndarray,  # shape (4,)
                       ang_vel:   np.ndarray,  # shape (3,)
                       q:         np.ndarray,  # shape (25,)
                       dq:        np.ndarray): # shape (25,)

const & = "传引用但不准修改"，类似 Python 的默认行为（传对象引用），但 C++ 必须显式写出来，否则会拷贝整个数组（浪费性能）。

第 135-139 行：step() 返回类型

StepResult step(...);   // 返回 StepResult 结构体

Python 版返回 dict，C++ 版返回一个强类型的 struct。好处是编译时就能检查字段名是否正确（Python 的 result["actions"] 写错不会报错，运行时才发现）。

第 160-236 行：private: 区

private:                              // 外部无法访问 (Python 的 _前缀约定)
    std::unique_ptr<Ort::Session> _session;  // ONNX 推理会话
    //                ^^^^^^^^^^^
    // 智能指针 = Python 的对象引用 (自动管理内存, 不用手动 delete)

Python 里私有变量用 _ 前缀约定（self._obs_buffer），C++ 强制编译器检查。

第 211-215 行：成员变量

std::array<float, 25> _kps{};    // {} = 初始化为全0
std::array<float, 25> _kds{};
std::array<float, 25> _tauLimit{};

等价 Python：
self._kps = np.zeros(25, dtype=np.float32)

---
文件 2：CasbotAmpDeploy.cpp（实现文件 = Python 的 def 函数体）

构造函数（= Python 的 __init__）

CasbotAmpDeploy::CasbotAmpDeploy(const std::string &configPath)
// ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// 类名::方法名    类名(构造函数)   参数: 字符串引用
    : _env(ORT_LOGGING_LEVEL_WARNING, "CasbotAmp")
//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    初始化列表: 在进入函数体之前初始化 _env 成员
//    等价 Python: self._env = ort.Env(...)
{
    // ── 加载 JSON 配置 ──
    std::ifstream f(configPath);    // f = open(configPath, 'r')
    if (!f.is_open()) {             // if not f:
        throw std::runtime_error("...");  // raise RuntimeError(...)
    }
    json cfg = json::parse(f);      // cfg = yaml.load(f) 类似

读取配置值

_modelPath = cfg.value("model_path", "model/policy.onnx");
//           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//           JSON 取值, 如果 key 不存在用默认值 "model/policy.onnx"

等价 Python：
self._model_path = cfg.get("model_path", "model/policy.onnx")

设置每个关节的电机参数

// 左腿: [0]=BIG(骨盆pitch) [1]=BIG(骨盆roll) [2]=SMALL(骨盆yaw)
//       [3]=BIG(膝) [4]=SMALL(踝pitch) [5]=SMALL(踝roll)
_kps[0] = _kps[1] = _kps[3] = CasbotMotor::STIFFNESS_LEG_BIG;
//                                          ^^^^^^^^^^^^^^^^^^^
//                                          = 276.31

C++ 可以链式赋值（a = b = c = 值），全部等于同一个值。Python 也行但不太常用。

计算 dof_action_scale

for (int i = 0; i < CASBOT_NUM_DOF; ++i) {   // for i in range(25):
    _dofActionScale[i] = _actionScale * _tauLimit[i] / _kps[i];
}

跟 Python 版一模一样的公式：0.25 * effort / stiffness。

_loadPolicy() — 加载 ONNX 模型

void CasbotAmpDeploy::_loadPolicy() {
    _session = std::make_unique<Ort::Session>(_env, _modelPath.c_str(), _sessionOptions);
    //         ^^^^^^^^^^^^^^^^^^^^
    //         创建一个 Ort::Session 对象, 用智能指针管理
    //         _modelPath.c_str() = C++ string → C 字符串 (ONNX Runtime 是 C 库)
}

等价 Python：
self._session = ort.InferenceSession(self._model_path)

initBuffers() — 填充观测历史

void CasbotAmpDeploy::initBuffers(...) {
    _vCmdBodyPast = {0.0f, 0.0f, 0.0f};   // = np.zeros(3)
    _lastAction.fill(0.0f);                 // = np.zeros(25)
    std::fill(_obsBuffer.begin(), _obsBuffer.end(), 0.0f);  // = np.zeros(336)

    std::array<float, 3> zeroCmd = {0.0f, 0.0f, 0.0f};   // 零速度指令
    for (int i = 0; i < 4; ++i) {
        _observationsCompute(baseQuat, angVel, zeroCmd, q, dq);
    }
}

跟 Python 版逻辑完全一样：先全部清零，然后用当前状态跑 4 次观测构建填满滑动窗口。

_computeProjectedGravity() — 投影重力

std::array<float, 3> CasbotAmpDeploy::_computeProjectedGravity(
        const std::array<float, 4> &baseQuat)
{
    float qw = baseQuat[0], qx = baseQuat[1], qy = baseQuat[2], qz = baseQuat[3];
    std::array<float, 3> g;
    g[0] =  2.0f * (-qz * qx + qw * qy);
    g[1] = -2.0f * ( qz * qy + qw * qx);
    g[2] =  1.0f - 2.0f * (qw * qw + qz * qz);
    return g;
}

跟 Python 的 get_gravity_orientation 完全一样的公式。f 后缀表示 float（32 位），不加 f 默认是 double（64 位）。

_observationsCompute() — 构建观测

这是核心函数。逐块对照 Python：

// 1. 投影重力
auto projGravity = _computeProjectedGravity(baseQuat);  // auto = 自动推断类型

// 2. 角速度缩放
float angVelS[3];
for (int i = 0; i < 3; ++i) angVelS[i] = angVel[i] * _angVelScale[i];

// 3. 速度指令 (直通)
float cmd[3];
for (int i = 0; i < 3; ++i) cmd[i] = cmdVel[i];

// 4. 关节位置偏移 (q - default) × scale
float dofPosS[25];
for (int i = 0; i < 25; ++i) {
    int mi = _dofMapping[i];
    dofPosS[i] = (q[mi] - _defaultDofPos[mi]) * _dofPosScale;
}

// 5. 关节速度 × scale
float dofVelS[25];
for (int i = 0; i < 25; ++i) {
    int mi = _dofMapping[i];
    dofVelS[i] = dq[mi] * _dofVelScale;
}

拼接帧 — C++ 没有 NumPy 的 np.concatenate，用 std::copy 手动拼接：

std::vector<float> frame(84, 0.0f);  // 创建84个0的数组
int offset = 0;
std::copy(angVelS,    angVelS+3,    frame.data()+offset);  offset += 3;  // 拷贝3个角速度
std::copy(projGravity, projGravity+3, frame.data()+offset); offset += 3;  // 拷贝3个重力
std::copy(cmd,         cmd+3,         frame.data()+offset); offset += 3;  // 拷贝3个指令
std::copy(dofPosS,     dofPosS+25,    frame.data()+offset); offset += 25; // 拷贝25个位置
std::copy(dofVelS,     dofVelS+25,    frame.data()+offset); offset += 25; // 拷贝25个速度
std::copy(_lastAction,  _lastAction+25, frame.data()+offset);              // 拷贝25个上帧动作

std::copy(源起始, 源结束, 目标起始)= Python 的 target[offset:offset+n] = source[0:n]。

滑动窗口：

for (int i = 0; i < 336 - 84; ++i)        // 左移: buf[0:252] = buf[84:336]
    _obsBuffer[i] = _obsBuffer[i + 84];
std::copy(frame.begin(), frame.end(),      // 新帧放到末尾
          _obsBuffer.begin() + (336 - 84)); // buf[252:336] = frame

_actionCompute() — ONNX 推理

// 1. 裁剪观测
for (auto &v : obsClipped)
    v = std::clamp(v, -_clipObservations, _clipObservations);
//      ^^^^^^^^^^^ C++17 的 clamp (Python: np.clip)

// 2. 创建输入张量 [1, 336]
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    memInfo, obsClipped.data(), obsClipped.size(), shape.data(), shape.size());

// 3. 执行推理
auto outputs = _session->Run(Ort::RunOptions{nullptr},
    _inputNames.data(), &inputTensor, 1,     // 1个输入
    _outputNames.data(), 1);                  // 1个输出

// 4. 获取输出指针
float *actionData = outputs[0].GetTensorMutableData<float>();
//    ^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    指针     拿到ONNX输出数组的首地址 (类似 NumPy 的 .data)

// 5. 缩放到目标关节角度
for (int i = 0; i < 25; ++i) {
    int motorIdx = _dofMapping[i];
    result.actions[motorIdx] = actionData[i] * _dofActionScale[motorIdx]
                             + _defaultDofPos[motorIdx];
}

指针 (float *) 是 C++ 最核心也最容易晕的概念。简单理解：actionData 是一块内存的"门牌号"，actionData[i] 就是这块内存的第 i 个 float。等价于 Python 的 action_data[i]（NumPy 数组在底层也是指针）。

---
文件 3：main.cpp（主循环）

全局变量

static mjModel    *g_m = nullptr;   // MuJoCo 模型指针
static mjData     *g_d = nullptr;   // MuJoCo 数据指针
static mjvCamera   g_cam;           // 相机
static mjvScene    g_scn;           // 渲染场景
static GLFWwindow  *g_window = nullptr;  // 窗口

static = 全局变量（整个文件共享）。* 表示指针。nullptr = Python 的 None。

main() 函数 — 程序入口

int main(int argc, char **argv) {   // Python: if __name__ == "__main__":

① 加载 MuJoCo 模型：

g_m = mj_loadXML(scenePath, nullptr, error, sizeof(error));
//    相当于 Python: m = mujoco.MjModel.from_xml_path(scene_path)
g_d = mj_makeData(g_m);
//    相当于 Python: d = mujoco.MjData(m)

② 创建策略对象：

CasbotAmpDeploy policy(configPath);
//  相当于 Python: policy = CasbotAMP(config_path)

③ 读取初始状态 + 填充缓冲区：

readState(baseQuat, angVel, qj, dqj);    // 从MuJoCo读状态
policy.initBuffers(baseQuat, angVel, qj, dqj);  // 填充4帧历史

④ 打开窗口：

g_window = glfwCreateWindow(1200, 900, "Casbot Skeleton", nullptr, nullptr);
glfwMakeContextCurrent(g_window);

⑤ 初始化渲染：

mjv_defaultCamera(&g_cam);       // 默认相机
mjv_makeScene(g_m, &g_scn, 2000); // 创建渲染场景
mjr_makeContext(g_m, &g_con, 200);// 创建渲染上下文

主循环

while (g_running && !glfwWindowShouldClose(g_window)) {
    // ── ① 处理输入事件 ──
    glfwPollEvents();        // 鼠标、键盘、手柄事件

    // ── ② 读取手柄 ──
    float ly = -gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    float lx = -gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float rx = -gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];

    // ── ③ 策略推理 (每~21ms一次) ──
    if (elapsed >= ctrlDt) {     // ctrlDt = 0.003 × 7 = 0.021s
        readState(baseQuat, angVel, qj, dqj);
        auto result = policy.step(baseQuat, angVel, cmd, qj, dqj);
        //            ^^^^^^^^^^^ = CasbotAmpDeploy::StepResult
        policyActions = result.actions;  // 25维目标关节位置
        kps = result.kps;                // PD刚度
        kds = result.kds;                // PD阻尼
    }

    // ── ④ PD控制 + 物理步进 (每0.003s) ──
    pdControl(policyActions, kps, kds, policy.tauLimit());
    mj_step(g_m, g_d);   // MuJoCo 前向动力学一步

    // ── ⑤ 渲染 ──
    mjv_updateScene(g_m, g_d, &g_opt, nullptr, &g_cam, mjCAT_ALL, &g_scn);
    mjr_render(viewport, &g_scn, &g_con);
    glfwSwapBuffers(g_window);  // 交换前后缓冲区 (双缓冲渲染)
}

PD 控制函数

static void pdControl(...) {
    for (int i = 0; i < 25; ++i) {
        int jid  = g_m->actuator_trnid[2*i];    // 执行器→关节ID
        int qadr = g_m->jnt_qposadr[jid];       // 关节在 qpos 中的地址
        int vadr = g_m->jnt_dofadr[jid];         // 关节在 qvel 中的地址
        double q  = g_d->qpos[qadr];             // 当前角度
        double dq = g_d->qvel[vadr];             // 当前角速度
        double tau = (targetQ[i] - q) * kps[i] + (0.0 - dq) * kds[i];
        g_d->ctrl[i] = std::clamp(tau, -tauLimit[i], tauLimit[i]);
        //   ^^^^^^^^  MuJoCo 的控制输入 (力矩)
    }
}

为什么必须用 jnt_qposadr？

qpos 数组布局：
qpos[0] qpos[1] qpos[2] qpos[3..6] qpos[7] qpos[8] ...
  x       y       z    四元数(wxyz) 关节1   关节2   ...
└─ 自由基座(7个) ─┘└── 25个关节角度 ──────────────┘

如果直接用 qpos[1]（关节ID=1），读到的其实是基座的 Y 坐标，而不是关节角度！jnt_qposadr[1] 返回 7，所以 qpos[7] 才是第一个关节的真实角度。这就是之前"拧成麻花"bug 的根因。

---
总结：C++ 版和 Python 版的完整对应

┌────────────────────────────────┬──────────────────────────────────────────────┐
│             Python             │                     C++                      │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ policy = CasbotAMP(yaml)       │ CasbotAmpDeploy policy(json)                 │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ policy.init_buffers(...)       │ policy.initBuffers(...)                      │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ result = policy.step(...)      │ auto result = policy.step(...)               │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ result["actions"]              │ result.actions                               │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ d.qpos[7:]                     │ g_d->qpos[7] (需 jnt_qposadr 算偏移)         │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ tau = (tgt - q) * kp + ...     │ double tau = (targetQ[i] - q) * kps[i] + ... │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ d.ctrl[:] = tau                │ g_d->ctrl[i] = tau                           │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ np.clip(x, lo, hi)             │ std::clamp(x, lo, hi)                        │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ np.concatenate([a,b,c])        │ std::copy 逐段拼接                           │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ mujoco.viewer.launch_passive() │ GLFW + mjv_updateScene + mjr_render          │
└────────────────────────────────┴──────────────────────────────────────────────┘













 为了让我更好的理解C++代码,你先帮我从头到尾按照部署流程逐行解释python手柄仿真部署代码/home/csabot/amp_lab/AMP_mjlab/RoboMimic_Deploy/deploy_mujoco/deploy_casbot.py,涉及到的每一个文件每一行都不要漏,并且关键地方要重点解释,我对你写的python仿真代码也一点都不熟悉,越详细越好

  Thought for 16s, read 3 files

下面我从头到尾逐行讲解整个 Python 部署系统。

---
Casbot Python MuJoCo 部署 — 逐行完整讲解

---
第一部分：部署入口 deploy_casbot.py

第 1-19 行：文件头和文档字符串

#!/usr/bin/env python3
"""
deploy_casbot.py — MuJoCo deployment for Casbot Skeleton (25-DOF)
...
"""

#!/usr/bin/env python3 是 shebang，让 Linux 可以直接 ./deploy_casbot.py 运行（不用敲 python）。"""...""" 是 docstring，描述这个文件是干什么的。

第 21-24 行：路径设置

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

逐层解释：
- __file__ = 当前文件的路径 = /home/.../deploy_mujoco/deploy_casbot.py
- Path(__file__).parent = 当前文件所在目录 = deploy_mujoco/
- .parent 再来一次 = 上一级 = RoboMimic_Deploy/（项目根目录）
- .absolute() = 转成绝对路径
- str() = Path 对象 → 字符串
- sys.path.append(...) = 把项目根目录加到 Python 搜索路径里

为什么要这样？ 因为后面的 from policy.casbot_amp.CasbotAMP import CasbotAMP 需要从项目根目录开始找 policy/ 包。不加这行会 ModuleNotFoundError。

第 26-33 行：导入依赖

import time              # 计时 (time.time(), time.sleep())
import mujoco            # MuJoCo 物理引擎
import mujoco.viewer     # MuJoCo 渲染窗口
import numpy as np       # 数组运算
import os                # 路径操作

from policy.casbot_amp.CasbotAMP import CasbotAMP  # 策略类
from common.joystick import JoyStick, JoystickButton  # 手柄驱动

第 40-43 行：PD 控制器

def pd_control(target_q, q, kp, target_dq, dq, kd, tau_limit):
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    return np.clip(tau, -tau_limit, tau_limit)

这就是机器人力控的核心公式：

τ = Kp × (q_desired - q_actual)   ← 位置误差 → 回复力
  + Kd × (dq_desired - dq_actual) ← 速度误差 → 阻尼力

然后钳位到 [-tau_limit, tau_limit] 防止力矩过大导致仿真崩溃

物理直觉：
- 把每个关节想象成一根弹簧+阻尼器。Kp 越大弹簧越硬（偏离目标时回复力越强），Kd 越大阻尼越大（运动越快阻力越
#define CASBOT_AMP_DEPLOY_H   // "那就定义它"

作用：防止同一个文件被 #include 两次。Python 不需要这个（自动处理）。

第 25-36 行：#include = Python 的 import

#include <onnxruntime_cxx_api.h>   // import onnxruntime
#include <array>                   // import array (固定大小数组)
#include <vector>                  // import list   (动态数组)
#include <string>                  // import str

第 38 行：#define = Python 的常量

#define CASBOT_NUM_DOF 25

等价于 Python 的 CASBOT_NUM_DOF = 25。但 #define 是文本替换——编译前把所有 CASBOT_NUM_DOF 替换成 25。比 Python 变量更快（零运行时开销）。

第 47-74 行：namespace = Python 的模块/命名空间

namespace CasbotMotor {                  // 类似 Python: class CasbotMotor:
    constexpr double ARMATURE_LEG_BIG = 0.06999046;
    //           ^^^^^^                    ^^^^^^^^^^
    //           类型                      值
    //
    // constexpr = 编译期常量, 类似 Python 里没法修改的模块级变量
    // double    = 64位浮点数 (Python 的 float)
}

这 6 组电机参数跟 casbot_constants.py 里的完全一致，只是翻译成了 C++ 语法。

第 92-107 行：类的声明

class CasbotAmpDeploy {         // Python: class CasbotAMP:
public:                         // public = 外部可以调用的方法
    struct StepResult {         //     类似 Python 的 dataclass / dict
        std::array<float, 25> actions{};  // float actions[25] 全初始化为 0
        std::array<float, 25> kps{};
        std::array<float, 25> kds{};
        bool terminated = false;
    };

    explicit CasbotAmpDeploy(const std::string &configPath);
    //     ^^^^^^^^^^^^^^^^
    //     构造函数 = Python 的 __init__(self, configPath)
    //     explicit 防止隐式类型转换 (安全措施)

Python 对照：
class CasbotAMP:
    def __init__(self, config_path: str):   # ← C++ 的构造函数
        ...

第 121-124 行：initBuffers 的参数类型

void initBuffers(
    const std::array<float, 4> &baseQuat,   // 4个float的数组, 只读引用
    //  ^^^^^                    ^
    //  只读(不会修改)            & = 引用传递 (类似 Python 的传引用, 避免拷贝)
    const std::array<float, 3> &angVel,
    const std::array<float, 25> &q,
    const std::array<float, 25> &dq
);

Python 对照：
def init_buffers(self, base_quat: np.ndarray,  # shape (4,)
                       ang_vel:   np.ndarray,  # shape (3,)
                       q:         np.ndarray,  # shape (25,)
                       dq:        np.ndarray): # shape (25,)

const & = "传引用但不准修改"，类似 Python 的默认行为（传对象引用），但 C++ 必须显式写出来，否则会拷贝整个数组（浪费性能）。

第 135-139 行：step() 返回类型

StepResult step(...);   // 返回 StepResult 结构体

Python 版返回 dict，C++ 版返回一个强类型的 struct。好处是编译时就能检查字段名是否正确（Python 的 result["act发现）。

第 160-236 行：private: 区

private:                              // 外部无法访问 (Python 的 _前缀约定)
    std::unique_ptr<Ort::Session> _session;  // ONNX 推理会话
    //                ^^^^^^^^^^^
    // 智能指针 = Python 的对象引用 (自动管理内存, 不用手动 delete)

Python 里私有变量用 _ 前缀约定（self._obs_buffer），C++ 强制编译器检查。

第 211-215 行：成员变量

std::array<float, 25> _kps{};    // {} = 初始化为全0
std::array<float, 25> _kds{};
std::array<float, 25> _tauLimit{};

等价 Python：
self._kps = np.zeros(25, dtype=np.float32)

---
文件 2：CasbotAmpDeploy.cpp（实现文件 = Python 的 def 函数体）

构造函数（= Python 的 __init__）

CasbotAmpDeploy::CasbotAmpDeploy(const std::string &configPath)
// ^^^^^^^^^^^^^  ^^^^^^^^^^^^^^  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
// 类名::方法名    类名(构造函数)   参数: 字符串引用
    : _env(ORT_LOGGING_LEVEL_WARNING, "CasbotAmp")
//    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    初始化列表: 在进入函数体之前初始化 _env 成员
//    等价 Python: self._env = ort.Env(...)
{
    // ── 加载 JSON 配置 ──
    std::ifstream f(configPath);    // f = open(configPath, 'r')
    if (!f.is_open()) {             // if not f:
        throw std::runtime_error("...");  // raise RuntimeError(...)
    }
    json cfg = json::parse(f);      // cfg = yaml.load(f) 类似

读取配置值

_modelPath = cfg.value("model_path", "model/policy.onnx");
//           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//           JSON 取值, 如果 key 不存在用默认值 "model/policy.onnx"

等价 Python：
self._model_path = cfg.get("model_path", "model/policy.onnx")

设置每个关节的电机参数

// 左腿: [0]=BIG(骨盆pitch) [1]=BIG(骨盆roll) [2]=SMALL(骨盆yaw)
//       [3]=BIG(膝) [4]=SMALL(踝pitch) [5]=SMALL(踝roll)
_kps[0] = _kps[1] = _kps[3] = CasbotMotor::STIFFNESS_LEG_BIG;
//                                          ^^^^^^^^^^^^^^^^^^^
//                                          = 276.31

C++ 可以链式赋值（a = b = c = 值），全部等于同一个值。Python 也行但不太常用。

计算 dof_action_scale

for (int i = 0; i < CASBOT_NUM_DOF; ++i) {   // for i in range(25):
    _dofActionScale[i] = _actionScale * _tauLimit[i] / _kps[i];
}

跟 Python 版一模一样的公式：0.25 * effort / stiffness。

_loadPolicy() — 加载 ONNX 模型

void CasbotAmpDeploy::_loadPolicy() {
    _session = std::make_unique<Ort::Session>(_env, _modelPath.c_str(), _sessionOptions);
    //         ^^^^^^^^^^^^^^^^^^^^
    //         创建一个 Ort::Session 对象, 用智能指针管理
    //         _modelPath.c_str() = C++ string → C 字符串 (ONNX Runtime 是 C 库)
}

等价 Python：
self._session = ort.InferenceSession(self._model_path)

initBuffers() — 填充观测历史

void CasbotAmpDeploy::initBuffers(...) {
    _vCmdBodyPast = {0.0f, 0.0f, 0.0f};   // = np.zeros(3)
    _lastAction.fill(0.0f);                 // = np.zeros(25)
    std::fill(_obsBuffer.begin(), _obsBuffer.end(), 0.0f);  // = np.zeros(336)

    std::array<float, 3> zeroCmd = {0.0f, 0.0f, 0.0f};   // 零速度指令
    for (int i = 0; i < 4; ++i) {
        _observationsCompute(baseQuat, angVel, zeroCmd, q, dq);
    }
}

跟 Python 版逻辑完全一样：先全部清零，然后用当前状态跑 4 次观测构建填满滑动窗口。

_computeProjectedGravity() — 投影重力

std::array<float, 3> CasbotAmpDeploy::_computeProjectedGravity(
        const std::array<float, 4> &baseQuat)
{
    float qw = baseQuat[0], qx = baseQuat[1], qy = baseQuat[2], qz = baseQuat[3];
    std::array<float, 3> g;
    g[0] =  2.0f * (-qz * qx + qw * qy);
    g[1] = -2.0f * ( qz * qy + qw * qx);
    g[2] =  1.0f - 2.0f * (qw * qw + qz * qz);

}

跟 Python 的 get_gravity_orientation 完全一样的公式。f 后缀表示 float（32 位），不加 f 默认是 double（64 位）。

_observationsCompute() — 构建观测

这是核心函数。逐块对照 Python：

// 1. 投影重力
auto projGravity = _computeProjectedGravity(baseQuat);  // auto = 自动推断类型

// 2. 角速度缩放
float angVelS[3];
for (int i = 0; i < 3; ++i) angVelS[i] = angVel[i] * _angVelScale[i];

// 3. 速度指令 (直通)
float cmd[3];
for (int i = 0; i < 3; ++i) cmd[i] = cmdVel[i];

// 4. 关节位置偏移 (q - default) × scale
float dofPosS[25];
for (int i = 0; i < 25; ++i) {
    int mi = _dofMapping[i];
    dofPosS[i] = (q[mi] - _defaultDofPos[mi]) * _dofPosScale;
}

// 5. 关节速度 × scale
float dofVelS[25];
for (int i = 0; i < 25; ++i) {
    int mi = _dofMapping[i];
    dofVelS[i] = dq[mi] * _dofVelScale;
}

拼接帧 — C++ 没有 NumPy 的 np.concatenate，用 std::copy 手动拼接：

std::vector<float> frame(84, 0.0f);  // 创建84个0的数组
int offset = 0;
std::copy(angVelS,    angVelS+3,    frame.data()+offset);  offset += 3;  // 拷贝3个角速度
std::copy(projGravity, projGravity+3, frame.data()+offset); offset += 3;  // 拷贝3个重力
std::copy(cmd,         cmd+3,         frame.data()+offset); offset += 3;  // 拷贝3个指令
std::copy(dofPosS,     dofPosS+25,    frame.data()+offset); offset += 25; // 拷贝25个位置
std::copy(dofVelS,     dofVelS+25,    frame.data()+offset); offset += 25; // 拷贝25个速度
std::copy(_lastAction,  _lastAction+25, frame.data()+offset);              // 拷贝25个上帧动作

std::copy(源起始, 源结束, 目标起始)= Python 的 target[offset:offset+n] = source[0:n]。

滑动窗口：

for (int i = 0; i < 336 - 84; ++i)        // 左移: buf[0:252] = buf[84:336]
    _obsBuffer[i] = _obsBuffer[i + 84];
std::copy(frame.begin(), frame.end(),      // 新帧放到末尾
          _obsBuffer.begin() + (336 - 84)); // buf[252:336] = frame

_actionCompute() — ONNX 推理

// 1. 裁剪观测
for (auto &v : obsClipped)
    v = std::clamp(v, -_clipObservations, _clipObservations);
//      ^^^^^^^^^^^ C++17 的 clamp (Python: np.clip)

// 2. 创建输入张量 [1, 336]
Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
    memInfo, obsClipped.data(), obsClipped.size(), shape.data(), shape.size());

// 3. 执行推理
auto outputs = _session->Run(Ort::RunOptions{nullptr},
    _inputNames.data(), &inputTensor, 1,     // 1个输入
    _outputNames.data(), 1);                  // 1个输出

// 4. 获取输出指针
float *actionData = outputs[0].GetTensorMutableData<float>();
//    ^        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
//    指针     拿到ONNX输出数组的首地址 (类似 NumPy 的 .data)

// 5. 缩放到目标关节角度
for (int i = 0; i < 25; ++i) {
    int motorIdx = _dofMapping[i];
    result.actions[motorIdx] = actionData[i] * _dofActionScale[motorIdx]
                             + _defaultDofPos[motorIdx];
}

指针 (float *) 是 C++ 最核心也最容易晕的概念。简单理解：actionData 是一块内存的"门牌号"，actionData[i] 就是这块内存的第 i 个 float。等价于 Python 的 action_data[i]（NumPy 数组在底层也是指针）。

---
文件 3：main.cpp（主循环）

全局变量

static mjModel    *g_m = nullptr;   // MuJoCo 模型指针
static mjData     *g_d = nullptr;   // MuJoCo 数据指针
static mjvCamera   g_cam;           // 相机
static mjvScene    g_scn;           // 渲染场景
static GLFWwindow  *g_window = nullptr;  // 窗口

static = 全局变量（整个文件共享）。* 表示指针。nullptr = Python 的 None。

main() 函数 — 程序入口

int main(int argc, char **argv) {   // Python: if __name__ == "__main__":

① 加载 MuJoCo 模型：

g_m = mj_loadXML(scenePath, nullptr, error, sizeof(error));
//    相当于 Python: m = mujoco.MjModel.from_xml_path(scene_path)
g_d = mj_makeData(g_m);
//    相当于 Python: d = mujoco.MjData(m)

② 创建策略对象：

CasbotAmpDeploy policy(configPath);
//  相当于 Python: policy = CasbotAMP(config_path)

③ 读取初始状态 + 填充缓冲区：

readState(baseQuat, angVel, qj, dqj);    // 从MuJoCo读状态
policy.initBuffers(baseQuat, angVel, qj, dqj);  // 填充4帧历史

④ 打开窗口：

g_window = glfwCreateWindow(1200, 900, "Casbot Skeleton", nullptr, nullptr);
glfwMakeContextCurrent(g_window);

⑤ 初始化渲染：

mjv_defaultCamera(&g_cam);       // 默认相机
mjv_makeScene(g_m, &g_scn, 2000); // 创建渲染场景
mjr_makeContext(g_m, &g_con, 200);// 创建渲染上下文

主循环

while (g_running && !glfwWindowShouldClose(g_window)) {
    // ── ① 处理输入事件 ──
    glfwPollEvents();        // 鼠标、键盘、手柄事件

    // ── ② 读取手柄 ──
    float ly = -gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
    float lx = -gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
    float rx = -gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];

    // ── ③ 策略推理 (每~21ms一次) ──
    if (elapsed >= ctrlDt) {     // ctrlDt = 0.003 × 7 = 0.021s
        readState(baseQuat, angVel, qj, dqj);
        auto result = policy.step(baseQuat, angVel, cmd, qj, dqj);
        //            ^^^^^^^^^^^ = CasbotAmpDeploy::StepResult
        policyActions = result.actions;  // 25维目标关节位置
        kps = result.kps;                // PD刚度
        kds = result.kds;                // PD阻尼
    }

    // ── ④ PD控制 + 物理步进 (每0.003s) ──
    pdControl(policyActions, kps, kds, policy.tauLimit());
    mj_step(g_m, g_d);   // MuJoCo 前向动力学一步

    // ── ⑤ 渲染 ──
    mjv_updateScene(g_m, g_d, &g_opt, nullptr, &g_cam, mjCAT_ALL, &g_scn);
    mjr_render(viewport, &g_scn, &g_con);
    glfwSwapBuffers(g_window);  // 交换前后缓冲区 (双缓冲渲染)
}

PD 控制函数

static void pdControl(...) {
    for (int i = 0; i < 25; ++i) {
        int jid  = g_m->actuator_trnid[2*i];    // 执行器→关节ID
        int qadr = g_m->jnt_qposadr[jid];       // 关节在 qpos 中的地址
        int vadr = g_m->jnt_dofadr[jid];         // 关节在 qvel 中的地址
        double q  = g_d->qpos[qadr];             // 当前角度
        double dq = g_d->qvel[vadr];             // 当前角速度
        double tau = (targetQ[i] - q) * kps[i] + (0.0 - dq) * kds[i];
        g_d->ctrl[i] = std::clamp(tau, -tauLimit[i], tauLimit[i]);
        //   ^^^^^^^^  MuJoCo 的控制输入 (力矩)
    }
}

为什么必须用 jnt_qposadr？

qpos 数组布局：
qpos[0] qpos[1] qpos[2] qpos[3..6] qpos[7] qpos[8] ...
  x       y       z    四元数(wxyz) 关节1   关节2   ...
└─ 自由基座(7个) ─┘└── 25个关节角度 ──────────────┘

如果直接用 qpos[1]（关节ID=1），读到的其实是基座的 Y 坐标，而不是关节角度！jnt_qposadr[1] 返回 7，所以 qpos[7] 才是第一个关节的真实角度。这就是之前"拧成麻花"bug 的根因。

---
总结：C++ 版和 Python 版的完整对应

          │                     C++                      │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ policy = CasbotAMP(yaml)       │ CasbotAmpDeploy policy(json)                 │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ policy.init_buffers(...)       │ policy.initBuffers(...)                      │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ result = policy.step(...)      │ auto result = policy.step(...)               │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ result["actions"]              │ result.actions                               │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ d.qpos[7:]                     │ g_d->qpos[7] (需 jnt_qposadr 算偏移)         │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ tau = (tgt - q) * kp + ...     │ double tau = (targetQ[i] - q) * kps[i] + ... │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ d.ctrl[:] = tau                │ g_d->ctrl[i] = tau                           │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ np.clip(x, lo, hi)             │ std::clamp(x, lo, hi)                        │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ np.concatenate([a,b,c])        │ std::copy 逐段拼接                           │
├────────────────────────────────┼──────────────────────────────────────────────┤
│ mujoco.viewer.launch_passive() │ GLFW + mjv_updateScene + mjr_render          │
└────────────────────────────────┴──────────────────────────────────────────────┘

✻ Worked for 1m 29s

❯ 为了让我更好的理解C++代码,你先帮我从头到尾按照部署流程逐行解释python手柄仿真部署代码/home/csabot/amp_lab/AMP_mjlab/RoboMimic_Deploy/deploy_mujoco/deploy_casbot.py,涉及到的每一个文件每一行都不要漏,并且关键地方要重点解释,我对你写的python仿真代码也一点都不熟悉,越详细越好

  Thought for 16s, read 3 files

下面我从头到尾逐行讲解整个 Python 部署系统。

---
Casbot Python MuJoCo 部署 — 逐行完整讲解

---
第一部分：部署入口 deploy_casbot.py

第 1-19 行：文件头和文档字符串

#!/usr/bin/env python3
"""
deploy_casbot.py — MuJoCo deployment for Casbot Skeleton (25-DOF)
...
"""

#!/usr/bin/env python3 是 shebang，让 Linux 可以直接 ./deploy_casbot.py 运行（不用敲 python）。"""...""" 是 docstring，描述这个文件是干什么的。

第 21-24 行：路径设置

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

逐层解释：
- __file__ = 当前文件的路径 = /home/.../deploy_mujoco/deploy_casbot.py
- Path(__file__).parent = 当前文件所在目录 = deploy_mujoco/
- .parent 再来一次 = 上一级 = RoboMimic_Deploy/（项目根目录）
- .absolute() = 转成绝对路径
- str() = Path 对象 → 字符串
- sys.path.append(...) = 把项目根目录加到 Python 搜索路径里

为什么要这样？ 因为后面的 from policy.casbot_amp.CasbotAMP import CasbotAMP 需要从项目根目录开始找 policy/ 包。不加这行会 ModuleNotFoundError。

第 26-33 行：导入依赖

import time              # 计时 (time.time(), time.sleep())
import mujoco            # MuJoCo 物理引擎
import mujoco.viewer     # MuJoCo 渲染窗口
import numpy as np       # 数组运算
import os                # 路径操作

from policy.casbot_amp.CasbotAMP import CasbotAMP  # 策略类
from common.joystick import JoyStick, JoystickButton  # 手柄驱动

第 40-43 行：PD 控制器

def pd_control(target_q, q, kp, target_dq, dq, kd, tau_limit):
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    return np.clip(tau, -tau_limit, tau_limit)

这就是机器人力控的核心公式：

τ = Kp × (q_desired - q_actual)   ← 位置误差 → 回复力
  + Kd × (dq_desired - dq_actual) ← 速度误差 → 阻尼力

然后钳位到 [-tau_limit, tau_limit] 防止力矩过大导致仿真崩溃

物理直觉：
- 把每个关节想象成一根弹簧+阻尼器。Kp 越大弹簧越硬（偏离目标时回复力越强），Kd 越大阻尼越大（运动越快阻力越大）
- target_q = 策略告诉关节"你应该转到这个角度"
- q = 关节实际当前位置（从 MuJoCo 读取）
- 位置差 × Kp = 往目标方向拉的力矩
- target_dq 我们始终传 np.zeros_like(kps) = 全零，意思是"目标速度为零"——让关节尽快稳定到目标位置，不要震荡

为什么要钳位？ 更新真实电机参数后 Kp 增大了 7 倍（39→276），如果不限制力矩上限，PD 控制器会算出几千 Nm 的力 /Inf in QACC 报错）。

第 50 行：Python 的 main 入口

if __name__ == "__main__":

当直接运行 python deploy_casbot.py 时，__name__ 等于 "__main__"，条件成立。如果被 import 导入则不成立。等价于 C++ 的 int main()。

第 51-54 行：路径准备

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
xml_path = str(PROJECT_ROOT / "casbot_skeleton" / "scene.xml")

Path 对象用 / 拼接路径（运算符重载），等价于 os.path.join(PROJECT_ROOT, "casbot_skeleton", "scene.xml")。最终指向 /home/.../RoboMimic_Deploy/casbot_skeleton/scene.xml。

scene.xml 是什么？
<mujoco model="casbot scene">
  <include file="casbot_skeleton_25dof.xml"/>   ← 引入机器人模型
  <worldbody>
    <geom name="floor" type="plane" .../>         ← 地板
  </worldbody>
</mujoco>

scene.xml 只是个"包装"——引入机器人 + 加地板和光照。实际机器人定义在 casbot_skeleton_25dof.xml 里（25 个关节 31 个 STL 网格）。

第 58-59 行：仿真参数

simulation_dt = 0.003         # 物理步长 = 3ms = 333Hz
control_decimation = 7        # 每7个物理步跑1次策略 = 48Hz

为什么物理步长和策略频率要分开？
- MuJoCo 物理积分需要高频（333Hz）保证数值稳定——如果步长太大，碰撞检测不准确、积分器发散
- ONNX 推理有开销（~10ms），跑不到 333Hz。48Hz（每 21ms 一次）足够捕捉人体运动动态
- 物理步 7 次 × 0.003s = 0.021s ≈ 48Hz

第 62-67 行：加载 MuJoCo 模型

m = mujoco.MjModel.from_xml_path(xml_path)  # 解析XML, 构建物理模型
d = mujoco.MjData(m)                         # 分配运行时数据
m.opt.timestep = simulation_dt               # 设置物理步长
num_joints = m.nu                            # nu = 执行器数量 = 25

m 和 d 是什么？
- m（MjModel）= 静态数据：关节结构、质量惯性、几何形状。加载后不变，全局共享
- d（MjData）= 动态数据：当前关节角度、速度、力矩。每步都在变
- m.opt.timestep = 每次 mj_step() 前进的物理时间

m.nu 为什么是 25？ nu = number of actuators（执行器数量）。casbot 有 25 个 <motor>，每个对应一个关节。

第 70 行：创建策略对象

policy = CasbotAMP()

这会触发 CasbotAMP.__init__() —— 加载 YAML 配置 → 设置电机参数 → 加载 ONNX 模型。后面详细讲。

第 73-76 行：运行时缓冲区

policy_actions = np.zeros(num_joints, dtype=np.float32)  # 当前目标位置 [25]
kps = policy.kps.copy()   # PD刚度 [25], 从策略配置复制
kds = policy.kds.copy()   # PD阻尼 [25], 从策略配置复制
sim_counter = 0           # 步数计数器

policy_actions 存储策略最新输出的目标关节位置，在 PD 控制时用。初始化为零。

第 79-86 行：手柄初始化

try:
    joystick = JoyStick()  # 创建手柄对象
except RuntimeError as e:
    sys.exit(1)            # 没插手柄就退出

JoyStick() 构造函数在 common/joystick.py 里：

class JoyStick:
    def __init__(self):
        pygame.init()                # 初始化 pygame 库
        pygame.joystick.init()       # 初始化手柄子系统

        joystick_count = pygame.joystick.get_count()  # 检测插了几个手柄
        if joystick_count == 0:
            raise RuntimeError("No joystick connected!")  # 没插→报错

        self.joystick = pygame.joystick.Joystick(0)  # 取第0号手柄 (第一个)
        self.joystick.init()

        self.button_count = self.joystick.get_numbuttons()  # 按钮数量
        self.button_states = [False] * self.button_count     # 按钮当前状态
        self.axis_count = self.joystick.get_numaxes()        # 摇杆轴数量
        self.axis_states = [0.0] * self.axis_count           # 摇杆当前值

Xbox 手柄的按键编号（JoystickButton枚举）：
A=0, B=1, X=2, Y=3, LB=4, RB=5, SELECT=6, START=7, L3=8, R3=9

Xbox 手柄的轴编号：
轴0 = 左摇杆X (左右, -1~1)
轴1 = 左摇杆Y (上下, -1~1)
轴2 = LT 左扳机 (0~1)
轴3 = 右摇杆X (左右, -1~1)
轴4 = 右摇杆Y (上下, -1~1)
轴5 = RT 右扳机 (0~1)

第 91-97 行：初始化阶段

base_quat = d.qpos[3:7].copy()   # 基座姿态四元数 [qw,qx,qy,qz]
ang_vel   = d.qvel[3:6].copy()   # 基座角速度 [ωx,ωy,ωz]
qj  = d.qpos[7:].copy()          # 25个关节角度
dqj = d.qvel[6:].copy()          # 25个关节角速度

policy.init_buffers(base_quat, ang_vel, qj, dqj)
policy_actions = policy._target_pos.copy()

d.qpos 的布局（这是最重要的概念）：

d.qpos = [x, y, z, qw, qx, qy, qz, joint_0, joint_1, ..., joint_24]
          └─ 基座位姿 (7个) ─┘  └─────── 25个关节角度 ──────────┘
索引:      0  1  2   3   4   5   6    7     8          31

- d.qpos[3:7] = 四元数 [qw, qx, qy, qz] —— 描述机器人基座在空间中的姿态
- d.qpos[7:] = 25 个关节角度，按 XML 里的关节顺序排列

d.qvel 的布局：
d.qvel = [vx, vy, vz, ωx, ωy, ωz, dq_joint_0, ..., dq_joint_24]
          └─ 基座速度 (6个) ─┘  └────── 25个关节角速度 ──────┘
索引:      0   1   2   3   4   5    6            30

- d.qvel[3:6] = 基座角速度 [ωx, ωy, ωz] —— 陀螺仪数据
- d.qvel[6:] = 25 个关节角速度

init_buffers 做了什么？ 用当前静止状态跑 4 次观测构建，填满 336 维滑动窗口。因为一开始没有"上一帧动作"，4 帧都填相同的初始状态。

第 108 行：打开 MuJoCo 渲染窗口

with mujoco.viewer.launch_passive(m, d) as viewer:

launch_passive 创建一个独立的渲染线程，不阻塞主循环。返回一个 viewer 对象用于同步。with 语句保证退出时自动关闭窗口。

和 launch 的区别：
- launch(m, d) = 阻塞式：渲染接管主线程，你的代码在回调里跑
- launch_passive(m, d) = 非阻塞式：渲染在后台线程，你的代码在主线程跑主循环

第 109-183 行：主循环

while viewer.is_running() and running:
    step_start = time.time()    # 记录本帧开始时间

每一帧执行以下 5 个步骤：

步骤 1：手柄输入（第 113-147 行）

joystick.update()    # 刷新手柄状态

update() 方法（joystick.py 第 50-67 行）：
def update(self):
    pygame.event.pump()    # 读取操作系统的手柄事件

    self.button_released = [False] * self.button_count  # 清空"释放"标记

    for i in range(self.button_count):
        current_state = self.joystick.get_button(i) == 1  # 读取每个按钮
        # 检测"释放"事件：上一帧按着 + 这一帧没按 = 释放
        if self.button_states[i] and not current_state:
            self.button_released[i] = True
        self.button_states[i] = current_state  # 更新状态

    for i in range(self.axis_count):
        self.axis_states[i] = self.joystick.get_axis(i)  # 读取每个轴

按下的检测 (is_button_pressed)：当前帧按着就返回 True（持续按住一直为 True）

释放的检测 (is_button_released)：只有松开的那一帧返回 True（边沿触发）。用于 RESET 操作——避免按住 START 时每

# SELECT 按下 → 退出
if joystick.is_button_pressed(JoystickButton.SELECT):  # SELECT = 6
    running = False

# START 松开 → 重置姿态
if joystick.is_button_released(JoystickButton.START):  # START = 7
    d.qpos[7:] = policy.default_dof_pos.copy()   # 关节角 → 默认站立姿态
    d.qvel[6:] = 0.0                              # 关节速度 → 0
    policy.reset()                                 # 清空策略内部状态
    policy.init_buffers(...)                       # 重新填充观测历史

# RT 扳机 (轴5) > 0.3 → 高速模式
r2_axis = joystick.get_axis_value(5)   # Xbox 右扳机
high_speed = r2_axis > 0.3             # 阈值 0.3 (防误触)

速度指令（第 143-147 行）：
ly_raw = -joystick.get_axis_value(1)  # 左摇杆Y, 取反 (前推为正)
lx_raw = -joystick.get_axis_value(0)  # 左摇杆X, 取反 (右推为正)
rx_raw = -joystick.get_axis_value(3)  # 右摇杆X, 取反

cmd_vel = policy.get_user_cmd(ly_raw, lx_raw, rx_raw)

这些值传给 get_user_cmd()，内部做死区过滤 + 方向限速 + 高低速切换。

步骤 2：PD 控制 + 物理步进（第 150-160 行）

tau = pd_control(
    policy_actions,          # target_q: 策略输出的目标位置 [25]
    d.qpos[7:],              # q: 当前实际关节角度 [25]
    kps,                     # kp: 刚度 [25]
    np.zeros_like(kps),      # target_dq: 目标速度 = 全零
    d.qvel[6:],              # dq: 当前实际角速度 [25]
    kds,                     # kd: 阻尼 [25]
    policy.tau_limit,        # tau_limit: 力矩上限 [25]
)
d.ctrl[:] = tau              # 写入 MuJoCo 控制输入
mujoco.mj_step(m, d)         # 前向动力学一步 (0.003s)

d.ctrl 是 MuJoCo 的控制输入数组，长度 = 执行器数量（25）。每个元素是施加到对应执行器的力矩（Nm）。mj_step() 读取 d.ctrl，加上重力/接触力等，计算下一步的关节加速度、速度、位置。

每帧都跑 PD + 物理（333Hz），不管策略有没有更新。策略输出不变时，PD 控制器用"旧目标"继续执行——这叫零阶保持。

步骤 3：策略推理（第 163-177 行）—— 每 7 帧才跑一次

sim_counter += 1
if sim_counter % control_decimation == 0:    # 7, 14, 21, 28... 时触发
    base_quat = d.qpos[3:7].copy()           # 读基座姿态
    ang_vel   = d.qvel[3:6].copy()           # 读角速度
    qj  = d.qpos[7:].copy()                  # 读关节角度
    dqj = d.qvel[6:].copy()                  # 读关节角速度

    result = policy.step(base_quat, ang_vel, cmd_vel, qj, dqj)
    # result = {"actions": [25], "kps": [25], "kds": [25], "terminated": bool}

    policy_actions = result["actions"].copy()  # 更新目标位置
    kps = result["kps"].copy()                 # 更新刚度
    kds = result["kds"].copy()                 # 更新阻尼

sim_counter % 7 == 0 = 每 7 步才执行一次 ONNX 推理。策略更新频率 = 333Hz / 7 ≈ 48Hz。

步骤 4：渲染同步（第 180 行）

viewer.sync()

等待渲染线程完成当前帧的绘制。主循环（物理+推理）和渲染线程通过这个调用来同步。

步骤 5：帧率控制（第 181-183 行）

time_until_next = m.opt.timestep - (time.time() - step_start)
if time_until_next > 0:
    time.sleep(time_until_next)

如果本帧的物理+渲染耗时小于 0.003s，就睡眠补足到 0.003s，保证仿真实时运行（不加速不减速）。

---
第二部分：策略核心类 CasbotAMP (CasbotAMP.py)

构造函数 __init__ (第 29-110 行)

加载配置（第 36-42 行）：
current_dir = os.path.dirname(os.path.abspath(__file__))
# = /home/.../policy/casbot_amp/

config_path = os.path.join(current_dir, "config", "CasbotAMP.yaml")
# = /home/.../policy/casbot_amp/config/CasbotAMP.yaml

with open(config_path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

读取 YAML 配置（第 52-96 行）：逐一从 YAML 提取参数，存为类属性。例如：
self.num_actions = config["num_actions"]    # 25
self.num_obs = config["num_obs"]            # 336
self.history_length = config["history_length"]  # 4

关键计算：dof_action_scale（第 100 行）：
self.dof_action_scale = self.action_scale * self.tau_limit / self.kps

这是 NumPy 的向量化运算：25 个元素同时做乘除。结果：
腿大电机: 0.25 * 150 / 276.31 = 0.1357
腿小电机: 0.25 * 60  / 156.31 = 0.0960
臂中电机: 0.25 * 75  / 130.20 = 0.1440
臂小电机: 0.25 * 36  /  96.83 = 0.0930

物理含义：tau_limit/kps = 不超力矩限制下的最大位置偏移量。0.25 是全局缩放（训练时也用的这个值）。

初始化运行时状态（第 103-107 行）：
self._high_speed_mode = False                        # 默认低速
self._vCmdBodyPast = np.zeros(3, dtype=np.float32)  # 速度指令历史 = [0,0,0]
self._last_action = np.zeros(25, dtype=np.float32)  # 上一帧动作 = 全零
self.obs_buffer = np.zeros(336, dtype=np.float32)   # 观测缓冲区 = 全零
self._target_pos = np.zeros(25, dtype=np.float32)   # 当前目标位置 = 全零

_load_policy() (第 124-129 行) — 加载 ONNX 模型

self.ort_session = ort.InferenceSession(self.onnx_path)
# 等价于 C++: session = make_unique<Ort::Session>(env, path, opts)

self.input_name = self.ort_session.get_inputs()[0].name   # "obs"
self.output_name = self.ort_session.get_outputs()[0].name # "actions"

ONNX Runtime 的 Python API 比 C++ 简洁很多。get_inputs()[0] 取第一个输入节点，.name 取它的名称。

init_buffers() (第 143-164 行) — 初始化观测历史

def init_buffers(self, base_quat, ang_vel, q, dq):
    self._vCmdBodyPast = np.zeros(3, dtype=np.float32)   # 指令历史=零
    self._last_action = np.zeros(25, dtype=np.float32)    # 上帧动作=零
    self.obs_buffer = np.zeros(336, dtype=np.float32)     # 观测缓冲区=零

    cmd_vel = np.zeros(3, dtype=np.float32)  # 零指令

    for _ in range(self.history_length):     # 跑4次
        self._observations_compute(base_quat, ang_vel, cmd_vel, q, dq)

用当前状态 + 零指令 + 零动作，连续跑 4 次观测构建。结果是 obs_buffer 的 4 帧完全一样——机器人初始静止。

step() (第 166-203 行) — 运行一次策略推理

def step(self, base_quat, ang_vel, cmd_vel, q, dq):
    # 1. 构建观测
    observation = self._observations_compute(base_quat, ang_vel, cmd_vel, q, dq)

    # 2. ONNX 推理 + 动作缩放
    result = self._action_compute(observation)

    # 3. 安全检查：投影重力的 Z 分量应该接近 -1.0
    projected_gravity = self._compute_projected_gravity(base_quat)
    anchor_error = abs(projected_gravity[2] - (-1.0))  # 偏离多少
    terminated = anchor_error > self.safe_projgravity_threshold  # >2.6 → 摔倒

    return {
        "actions": result["actions"],      # 25维目标关节位置
        "kps": result["kps"],              # 25维刚度
        "kds": result["kds"],              # 25维阻尼
        "terminated": terminated,          # 是否触发安全终止
    }

安全检查原理：机器人直立时，世界重力 [0,0,-1] 投影到机器人坐标系也接近 [0,0,-1]（Z≈-1）。如果 |Z - (-1)| > 2.6，说明机器人严重倾斜（可能已摔倒），触发安全标志。

get_user_cmd() (第 205-249 行) — 手柄速度处理

vx_lim = self.vx_lim_slow if not self._high_speed_mode else self.vx_lim
# 低速: vx上限1.0   高速: vx上限2.5

# 死区过滤 (±0.2)
if ly < -self.dead_zone:       # 摇杆拉到底
    vx = ly * (-vx_lim[0])     # 后退: ly×0.8 m/s
elif ly > self.dead_zone:      # 摇杆推到底
    vx = ly * vx_lim[1]        # 前进: ly×2.5(高速) 或 ly×1.0(低速)
else:
    vx = 0.0                    # 死区内 → 不动

死区的物理意义：手柄摇杆即使在中位也会有微小漂移（比如 0.02），不加死区机器人会自己慢慢走。±0.2 过滤掉这些噪

方向性限速：后退限速 (0.8 m/s) 远小于前进 (2.5 m/s)——因为机器人后走更不稳定。

# 指数平滑
vCmdBody = vCmdBodyPast * cmd_smoothes + new_cmd * (1.0 - cmd_smoothes)

当前 cmd_smoothes = 0.0，所以平滑关闭——新指令直接生效。如果设为 0.9，指令会缓慢过渡（避免突变）。

_compute_projected_gravity() (第 263-271 行) — 投影重力

@staticmethod
def _compute_projected_gravity(base_quat):
    qw, qx, qy, qz = base_quat
    g = np.zeros(3, dtype=np.float32)
    g[0] = 2.0 * (-qz * qx + qw * qy)    # 重力在机器人X轴分量
    g[1] = -2.0 * (qz * qy + qw * qx)    # 重力在机器人Y轴分量
    g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)  # 重力在机器人Z轴分量
    return g

这是四元数旋转的简化公式。世界重力 = [0, 0, -1]。用基座姿态的逆旋转把世界重力转到机器人局部坐标系：
- 机器人直立：g ≈ [0, 0, -1] （重力完全朝下）
- 前倾 30°：g[0] > 0（重力在机器人前方有分量）
- 侧倾 30°：g[1] ≠ 0

为什么不用完整的旋转矩阵？ 这个简化版只算 3 个输出（不是 3×3 矩阵），快 3 倍。公式避免了 qw²+qx²+qy²+qz² 的归一化（MuJoCo 的四元数已经归一化）。

_observations_compute() (第 273-322 行) — 观测构建

这是整个部署最核心的函数：

# 1. 投影重力 (3维)
projected_gravity = self._compute_projected_gravity(base_quat)

# 2. 角速度 → 保底3维 (3维)
ang_vel = np.asarray(ang_vel, dtype=np.float32).flatten()[:3]

# 3. 速度指令 (3维)
vCmdBody = np.asarray(cmd_vel, dtype=np.float32).flatten()[:3]

# 4. 关节位置偏移 (25维)
dof_pos_scaled = (q - default_dof_pos) * dof_pos_scale
# 例: 膝关节当前 0.8rad, 默认 0.53rad → 偏移 0.27rad × 1.0 = 0.27

# 5. 关节速度 (25维)
dof_vel_scaled = dq * dof_vel_scale

# 6. 角速度缩放 (3维)
ang_vel_scaled = ang_vel * ang_vel_scale   # scale=1.0, 实际不变

拼接成一帧（84维）：
current_robot_state = np.concatenate([
    ang_vel_scaled,       # [0:3]   角速度
    projected_gravity,    # [3:6]   投影重力
    vCmdBody,             # [6:9]   手柄指令
    dof_pos_scaled,       # [9:34]  关节位置偏移 (25)
    dof_vel_scaled,       # [34:59] 关节速度 (25)
    self._last_action,    # [59:84] 上一帧动作 (25)
])
# 3+3+3+25+25+25 = 84

滑动窗口：
# 左移: buf[0:252] = buf[84:336]   (旧3帧前移)
self.obs_buffer[0:252] = self.obs_buffer[252:336]

# 新帧放末尾: buf[252:336] = current_frame
self.obs_buffer[252:336] = current_robot_state

图解：
obs_buffer 布局 (336维):
┌──────────┬──────────┬──────────┬──────────┐
│ frame_0  │ frame_1  │ frame_2  │ frame_3  │  ← 时间顺序: 0最旧, 3最新
│  (84维)  │  (84维)  │  (84维)  │  (84维)  │
└──────────┴──────────┴──────────┴──────────┘
  旧 ←───────────────────────────────→ 新

每次新的观测:
  ① frame_0 被丢弃
  ② frame_1→0, frame_2→1, frame_3→2  (左移)
  ③ 新帧放入 frame_3 位置

_action_compute() (第 324-363 行) — ONNX 推理

# 1. 整形 [336] → [1, 336] (ONNX 需要 batch 维度)
obs_tensor = observation.reshape(1, -1).astype(np.float32)

# 2. ONNX 前向传播
outputs = self.ort_session.run(
    [self.output_name],              # 要哪些输出: ["actions"]
    {self.input_name: obs_tensor}    # 输入字典: {"obs": 数据}
)
action_policy = outputs[0].squeeze()  # [1,25] → [25]

# 3. 裁剪动作到 [-100, 100]
action_policy = np.clip(action_policy, -self.clip_actions, self.clip_actions)

# 4. 缩放到目标关节角度
for policy_idx in range(25):
    motor_idx = self.dof_mapping[policy_idx]   # dof_mapping 是恒等映射 [0,1,2,...,24]
    target_pos_motor[motor_idx] = (
        action_policy[policy_idx] * self.dof_action_scale[motor_idx]
        + self.default_dof_pos[motor_idx]
    )

转换公式：
target_q[i] = action[i] × dof_action_scale[i] + default_dof_pos[i]

例: 策略输出 action[0] = 0.5 (左腿骨盆pitch想往前摆)
     dof_action_scale[0] = 0.1357
     default_dof_pos[0] = -0.32
     → target_q[0] = 0.5 × 0.1357 + (-0.32) = -0.252 rad

# 5. 保存上帧动作 (下次观测用)
self._last_action = action_policy.copy()
self._target_pos = target_pos_motor.copy()

_last_action 存的是原始的 ONNX 输出（25 维 float，未缩放），不是目标角度。这样策略能看到"我上一帧输出了多大动作"，形成闭环。

---
第三部分：完整数据流总结

每一帧 (333Hz):

① 手柄读取
   pygame → JoyStick.update() → ly, lx, rx → get_user_cmd() → cmd_vel[3]

② PD 控制 + 物理步进
   target_q[25] ─┐
   q_actual[25] ─┤→ pd_control() → tau[25] → d.ctrl[:] → mj_step()
   kp[25], kd[25]┘

③ 策略推理 (每7帧1次, 48Hz)
   d.qpos[3:7] → base_quat → _compute_projected_gravity → proj_grav[3]
   d.qvel[3:6] → ang_vel                                   → ang_vel[3]
   手柄         → cmd_vel                                   → cmd_vel[3]
   d.qpos[7:]   → q         → (q - default) × scale        → dof_pos[25]
   d.qvel[6:]   → dq        → dq × scale                   → dof_vel[25]
   上一帧       → _last_action                              → last_act[25]
                                └── concatenate ──→ frame[84]
                                └── slide window ─→ obs[336]
                                └── ONNX inference ─→ action[25]
                                └── action × scale + default ─→ target_q[25]




域随机化

我现在的训练框架有域随机化的部分吗,我训练casbot机器人时有没有用到

  Thought for 5s, searched for 1 pattern

有，而且正在用。 AMP 框架内置了 4 种域随机化 + 1 种扰动，都在 amp_env_cfg.py 的 events 里定义，casbot 全部继承：

观测噪声（actor 专属，每帧）

┌───────────────────┬────────────┬────────────┐
│       噪声        │    范围    │   作用于   │
├───────────────────┼────────────┼────────────┤
│ base_ang_vel      │ ±0.2 rad/s │ IMU 角速度 │
├───────────────────┼────────────┼────────────┤
│ projected_gravity │ ±0.05      │ 重力方向   │
├───────────────────┼────────────┼────────────┤
│ joint_pos         │ ±0.01 rad  │ 关节角度   │
├───────────────────┼────────────┼────────────┤
│ joint_vel         │ ±0.5 rad/s │ 关节角速度 │
└───────────────────┴────────────┴────────────┘

critic 无噪声，actor 有噪声——这是 sim-to-real 的关键设计：critic 看到真实状态来准确评估，actor 看到带噪声的观测来学习鲁棒性。

物理参数随机化（reset 时采样一次，持续整个 episode）

┌────────────┬────────────────────┬─────────────────────────────────┐
│    事件    │        函数        │              参数               │
├────────────┼────────────────────┼─────────────────────────────────┤
│ 脚底摩擦   │ dr.geom_friction   │ 0.3~1.2 随机，双脚共享同一值    │
├────────────┼────────────────────┼─────────────────────────────────┤
│ 编码器偏置 │ dr.encoder_bias    │ ±0.015 rad 关节角偏置           │
├────────────┼────────────────────┼─────────────────────────────────┤
│ 质心偏移   │ dr.body_com_offset │ x/y ±2.5cm, z ±3cm 躯干质心偏移 │
└────────────┴────────────────────┴─────────────────────────────────┘

外力扰动（interval 模式，每 1~3 秒随机推一次）

┌────────────┬──────────────┬─────────────┐
│    事件    │     方向     │    范围     │
├────────────┼──────────────┼─────────────┤
│ push_robot │ x 线速度     │ ±1.0 m/s    │
├────────────┼──────────────┼─────────────┤
│            │ y 线速度     │ ±0.5 m/s    │
├────────────┼──────────────┼─────────────┤
│            │ z 线速度     │ ±0.4 m/s    │
├────────────┼──────────────┼─────────────┤
│            │ roll 角速度  │ ±0.52 rad/s │
├────────────┼──────────────┼─────────────┤
│            │ pitch 角速度 │ ±0.52 rad/s │
├────────────┼──────────────┼─────────────┤
│            │ yaw 角速度   │ ±0.78 rad/s │
└────────────┴──────────────┴─────────────┘

哪几个真正生效于 casbot

全部。casbot 的 env_cfgs.py 没有删除或禁用任何一个，只在 play 模式下禁用了 push_robot 和观测 corruption。训练时这 4 种全部开启。
















一、Delay 期间的奖励调制
     
  核心实现在两个文件：

  ┌───────────────────────────────────────────┬────────────────────────────────────────────────────────────────────────────────────────┐
  │                   文件                    │                                          内容                                          │
  ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/tasks/amp_loco/mdp/rewards.py:27-68   │ 两个延迟调制函数：_apply_delay_env_reward_scaling 和 _apply_delay_env_reward_mask_only │
  ├───────────────────────────────────────────┼────────────────────────────────────────────────────────────────────────────────────────┤
  │ src/tasks/amp_loco/amp_env_cfg.py:305-368 │ 每个奖励项的 delay 参数配置                                                            │
  └───────────────────────────────────────────┴────────────────────────────────────────────────────────────────────────────────────────┘

  rewards.py 里的核心逻辑：

  # rewards.py:37-51 — 对 delay 环境把奖励×ratio（速度类 ratio=0 → 直接归零）
  def _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio):
      delay_env_mask = _get_delay_env_mask(env)  # delay 环境中计数器>0的
      if delay_env_mask is None:
          return reward
      return torch.where(delay_env_mask, reward * delay_env_rew_ratio, reward)

  # rewards.py:54-68 — 对非 delay 环境直接给0，只有 delay 环境才给（高度奖励用这个）
  def _apply_delay_env_reward_mask_only(env, reward, mask_delay, delay_env_rew_ratio):
      delay_env_mask = _get_delay_env_mask(env)
      return torch.where(delay_env_mask, reward * delay_env_rew_ratio, torch.zeros_like(reward))

  amp_env_cfg.py 里的参数配置：

  # 速度奖励 → 延迟期间归零
  "track_anchor_linear_velocity": RewardTermCfg(
      ..., params={"mask_delay": True, "delay_env_rew_ratio": 0.0}),  # ×0

  # 高度奖励 → 延迟期间×3.5，正常环境不生效
  "track_root_height": RewardTermCfg(
      ..., params={"mask_delay": True, "delay_env_rew_ratio": 3.5}),  # ×3.5

  # 终止惩罚 → 不受 delay 调制，始终 -200
  "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),

  ---
  二、Recovery Reset（从倒地姿势初始化）

  核心实现在两个文件：

  ┌───────────────────────────────────────────────────────┬─────────┬─────────────────────────────────────────────────────────────────────────┐
  │                         文件                          │  行号   │                                  内容                                   │
  ├───────────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────┤
  │ src/tasks/amp_loco/mdp/events.py                      │ 69-100  │ MotionResetManager.reset() — 区分 delay/normal 环境，分别采样不同动作帧 │
  ├───────────────────────────────────────────────────────┼─────────┼─────────────────────────────────────────────────────────────────────────┤
  │ src/tasks/amp_loco/config/casbot_skeleton/env_cfgs.py │ 107-119 │ 配置 recovery_dir + delay_reset_env_ratio + max_delay_steps             │
  └───────────────────────────────────────────────────────┴─────────┴─────────────────────────────────────────────────────────────────────────┘

  events.py:69-100 的核心逻辑：

  def reset(self, env, env_ids, motion_dir, ...):
      # 分离 delay 和 normal 环境
      delay_mask = self._get_delay_env_mask(env)
      delay_ids = env_ids[is_delay]
      normal_ids = env_ids[~is_delay]

      # 正常环境 → 从 WalkandRun 采站立帧
      self._write_reset_state(env, normal_ids, self.walk_run_frames[motion_dir])

      # 延迟环境 → 从 Recovery 动作采倒地/爬起帧
      recovery = self.recovery_frames.get(motion_dir)
      frames = recovery if recovery is not None else self.walk_run_frames[motion_dir]
      self._write_reset_state(env, delay_ids, frames)

  env_cfgs.py:107-119 的配置：

  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir


