/**
 * @file State_CasbotAmp.cpp — Casbot AMP policy state implementation.
 *
 * Ported from standalone CasbotAmpDeploy, wrapped in FSMState.
 * Observation building, ONNX inference, and action scaling are identical.
 */

#include "FSM/State_CasbotAmp.h"
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <cmath>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════
//  Constructor
// ═══════════════════════════════════════════════════════════════

State_CasbotAmp::State_CasbotAmp(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::CASBOT_AMP, "casbot_amp")
{
    // ── Load JSON config ──
    std::string configPath = std::string(PROJECT_ROOT_DIR) + "/casbot_dds/config/casbot_amp.json";
    std::ifstream f(configPath);
    if (!f.is_open()) throw std::runtime_error("[CasbotAmp] Cannot open: " + configPath);
    json cfg = json::parse(f);

    _modelPath        = cfg.value("model_path", "../model/policy.onnx");
    _actionScale      = cfg.value("action_scale", 0.25f);
    _clipObservations = cfg.value("clip_observations", 100.0f);
    _clipActions      = cfg.value("clip_actions", 100.0f);
    _deadZone         = cfg.value("dead_zone", 0.2f);
    _cmdSmoothes      = cfg.value("cmd_smoothes", 0.0);
    _dofPosScale      = cfg.value("dof_pos_scale", 1.0f);
    _dofVelScale      = cfg.value("dof_vel_scale", 1.0f);
    _safeProjGravThresh = cfg.value("safe_projgravity_threshold", 2.6f);

    _vxLim[0]     = cfg.value("vx_limit_min", -0.8f);  _vxLim[1]     = cfg.value("vx_limit_max", 2.5f);
    _vxLimSlow[0] = cfg.value("vx_limit_min_slow", -0.8f); _vxLimSlow[1] = cfg.value("vx_limit_max_slow", 1.0f);
    _vyLim[0]     = cfg.value("vy_limit_min", -1.0f);  _vyLim[1]     = cfg.value("vy_limit_max", 1.0f);
    _wyawLim[0]   = cfg.value("wyaw_limit_min", -3.14f); _wyawLim[1]  = cfg.value("wyaw_limit_max", 3.14f);

    // ── Resolve model path relative to config dir ──
    if (!_modelPath.empty() && _modelPath[0] != '/') {
        auto slash = configPath.find_last_of('/');
        if (slash != std::string::npos)
            _modelPath = configPath.substr(0, slash) + "/" + _modelPath;
    }

    // ── Set per-joint motor params (6 groups) ──
    // L leg: [0]=BIG [1]=BIG [2]=SMALL [3]=BIG [4]=SMALL [5]=SMALL
    _kps[0]=_kps[1]=_kps[3]=CasbotArmature::STIFFNESS_LEG_BIG;
    _kds[0]=_kds[1]=_kds[3]=CasbotArmature::DAMPING_LEG_BIG;
    _tauLimit[0]=_tauLimit[1]=_tauLimit[3]=CasbotArmature::EFFORT_LEG_BIG;
    _kps[2]=_kps[4]=_kps[5]=CasbotArmature::STIFFNESS_LEG_SMALL;
    _kds[2]=_kds[4]=_kds[5]=CasbotArmature::DAMPING_LEG_SMALL;
    _tauLimit[2]=_tauLimit[4]=_tauLimit[5]=CasbotArmature::EFFORT_LEG_SMALL;

    // R leg: [6]=BIG [7]=BIG [8]=SMALL [9]=BIG [10]=SMALL [11]=SMALL
    _kps[6]=_kps[7]=_kps[9]=CasbotArmature::STIFFNESS_LEG_BIG;
    _kds[6]=_kds[7]=_kds[9]=CasbotArmature::DAMPING_LEG_BIG;
    _tauLimit[6]=_tauLimit[7]=_tauLimit[9]=CasbotArmature::EFFORT_LEG_BIG;
    _kps[8]=_kps[10]=_kps[11]=CasbotArmature::STIFFNESS_LEG_SMALL;
    _kds[8]=_kds[10]=_kds[11]=CasbotArmature::DAMPING_LEG_SMALL;
    _tauLimit[8]=_tauLimit[10]=_tauLimit[11]=CasbotArmature::EFFORT_LEG_SMALL;

    // Waist [12]: LEG_SMALL
    _kps[12]=CasbotArmature::STIFFNESS_LEG_SMALL; _kds[12]=CasbotArmature::DAMPING_LEG_SMALL;
    _tauLimit[12]=CasbotArmature::EFFORT_LEG_SMALL;

    // Head [13][14]: ARM_SMALL
    _kps[13]=_kps[14]=CasbotArmature::STIFFNESS_ARM_SMALL;
    _kds[13]=_kds[14]=CasbotArmature::DAMPING_ARM_SMALL;
    _tauLimit[13]=_tauLimit[14]=CasbotArmature::EFFORT_ARM_SMALL;

    // L arm: [15]=MID [16]=MID [17]=SMALL [18]=MID [19]=SMALL
    _kps[15]=_kps[16]=_kps[18]=CasbotArmature::STIFFNESS_ARM_MID;
    _kds[15]=_kds[16]=_kds[18]=CasbotArmature::DAMPING_ARM_MID;
    _tauLimit[15]=_tauLimit[16]=_tauLimit[18]=CasbotArmature::EFFORT_ARM_MID;
    _kps[17]=_kps[19]=CasbotArmature::STIFFNESS_ARM_SMALL;
    _kds[17]=_kds[19]=CasbotArmature::DAMPING_ARM_SMALL;
    _tauLimit[17]=_tauLimit[19]=CasbotArmature::EFFORT_ARM_SMALL;

    // R arm: [20]=MID [21]=MID [22]=SMALL [23]=MID [24]=SMALL
    _kps[20]=_kps[21]=_kps[23]=CasbotArmature::STIFFNESS_ARM_MID;
    _kds[20]=_kds[21]=_kds[23]=CasbotArmature::DAMPING_ARM_MID;
    _tauLimit[20]=_tauLimit[21]=_tauLimit[23]=CasbotArmature::EFFORT_ARM_MID;
    _kps[22]=_kps[24]=CasbotArmature::STIFFNESS_ARM_SMALL;
    _kds[22]=_kds[24]=CasbotArmature::DAMPING_ARM_SMALL;
    _tauLimit[22]=_tauLimit[24]=CasbotArmature::EFFORT_ARM_SMALL;

    // ── Default pose (KNEES_BENT_KEYFRAME) ──
    _defaultDofPos = {
        -0.32f,0,0, 0.53f,-0.19f,0,  -0.32f,0,0, 0.53f,-0.19f,0,
        0, 0,0,  0.2f,0.3f,0,-0.35f,0,  0.2f,-0.3f,0,-0.35f,0
    };

    // ── dof_action_scale = action_scale × tau_limit / kps ──
    for (int i = 0; i < NUM_DOF; ++i)
        _dofActionScale[i] = _actionScale * _tauLimit[i] / _kps[i];

    _obsBuffer.resize(NUM_OBS, 0.0f);
    _loadPolicy();
}

// ═══════════════════════════════════════════════════════════════
//  ONNX
// ═══════════════════════════════════════════════════════════════

void State_CasbotAmp::_loadPolicy() {
    _session = std::make_unique<Ort::Session>(_env, _modelPath.c_str(), _sessionOpts);
    std::cout << "[CasbotAmp] ONNX loaded: " << _modelPath << std::endl;
}

// ═══════════════════════════════════════════════════════════════
//  FSMState interface
// ═══════════════════════════════════════════════════════════════

void State_CasbotAmp::enter() {
    std::cout << "[CasbotAmp] Enter" << std::endl;
    _highSpeedMode = false;
    _terminateFlag = false;
    _vCmdBodyPast  = {0,0,0};
    _lastAction.fill(0);
    _targetPos.fill(0);

    // Hold current position during transition
    for (int i = 0; i < NUM_DOF; ++i) {
        lowCmd()->motorCmd[i].q  = lowState()->motorState[i].q;
        lowCmd()->motorCmd[i].dq = 0;
        lowCmd()->motorCmd[i].Kp = _kps[i];
        lowCmd()->motorCmd[i].Kd = _kds[i];
        lowCmd()->motorCmd[i].tau = 0;
    }
    _initBuffers();
}

void State_CasbotAmp::run() {
    _observationsCompute();
    _actionCompute();
}

void State_CasbotAmp::exit() {
    std::cout << "[CasbotAmp] Exit" << std::endl;
    _lastAction.fill(0);
    std::fill(_obsBuffer.begin(), _obsBuffer.end(), 0.0f);
    _vCmdBodyPast = {0,0,0};
    _terminateFlag = false;
    _highSpeedMode = false;
}

FSMStateName State_CasbotAmp::checkChange() {
    // Priority: L2+B → PASSIVE (emergency)
    //           Terminate flag → PASSIVE
    //           UserCmd SELECT → PASSIVE
    //           START → FIXEDSTAND
    //           Default: stay in CASBOT_AMP
    if (_terminateFlag) {
        std::cout << "[CasbotAmp] Anchor terminated → PASSIVE" << std::endl;
        return FSMStateName::PASSIVE;
    }
    if (lowState()->userCmd == UserCommand::SELECT ||
        lowState()->userCmd == UserCommand::L2_B) {
        return FSMStateName::PASSIVE;
    }
    if (lowState()->userCmd == UserCommand::START) {
        return FSMStateName::FIXEDSTAND;
    }
    return FSMStateName::CASBOT_AMP;
}

// ═══════════════════════════════════════════════════════════════
//  Buffers
// ═══════════════════════════════════════════════════════════════

void State_CasbotAmp::_initBuffers() {
    _vCmdBodyPast = {0,0,0};
    _lastAction.fill(0);
    std::fill(_obsBuffer.begin(), _obsBuffer.end(), 0.0f);
    for (int i = 0; i < HISTORY_LENGTH; ++i)
        _observationsCompute();
    std::cout << "[CasbotAmp] Buffers initialized" << std::endl;
}

// ═══════════════════════════════════════════════════════════════
//  Projected gravity
// ═══════════════════════════════════════════════════════════════

std::array<float, 3> State_CasbotAmp::_projectedGravity(const std::array<float, 4> &q) {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    return {
         2.0f * (-qz * qx + qw * qy),
        -2.0f * ( qz * qy + qw * qx),
         1.0f - 2.0f * (qw * qw + qz * qz)
    };
}

// ═══════════════════════════════════════════════════════════════
//  User command
// ═══════════════════════════════════════════════════════════════

std::array<float, 3> State_CasbotAmp::_getUserCmd() {
    // Joystick axes from lowState userValue [ly, lx, rx]
    float ly = lowState()->userValue[0];
    float lx = lowState()->userValue[1];
    float rx = lowState()->userValue[2];

    const auto &vxLim = _highSpeedMode ? _vxLim : _vxLimSlow;

    ly = deadZone(ly, _deadZone); lx = deadZone(lx, _deadZone); rx = deadZone(rx, _deadZone);

    float vx = 0, vy = 0, wyaw = 0;
    if      (ly < 0) vx = ly * (-vxLim[0]);
    else if (ly > 0) vx = ly * vxLim[1];
    if      (lx < 0) vy = lx * (-_vyLim[0]);
    else if (lx > 0) vy = lx * _vyLim[1];
    if      (rx < 0) wyaw = rx * (-_wyawLim[0]);
    else if (rx > 0) wyaw = rx * _wyawLim[1];

    std::array<float, 3> cmd = {vx, vy, wyaw};
    for (int i = 0; i < 3; ++i)
        _vCmdBodyPast[i] = _vCmdBodyPast[i] * _cmdSmoothes + cmd[i] * (1.0f - _cmdSmoothes);
    return _vCmdBodyPast;
}

// ═══════════════════════════════════════════════════════════════
//  Observation
// ═══════════════════════════════════════════════════════════════

void State_CasbotAmp::_observationsCompute() {
    constexpr int SD = ROBOT_STATE_DIM;
    constexpr int ND = NUM_DOF;

    // Read state from lowState
    std::array<float, 4> quat;
    quat[0] = lowState()->imu.quaternion[0];
    quat[1] = lowState()->imu.quaternion[1];
    quat[2] = lowState()->imu.quaternion[2];
    quat[3] = lowState()->imu.quaternion[3];

    auto projGrav = _projectedGravity(quat);
    auto angVel   = lowState()->imu.getGyro();
    auto cmdVel   = _getUserCmd();

    // Build frame
    std::vector<float> frame(SD, 0.0f);
    int off = 0;
    for (int i = 0; i < 3; ++i) frame[off++] = angVel(i) * 1.0f;
    for (int i = 0; i < 3; ++i) frame[off++] = projGrav[i];
    for (int i = 0; i < 3; ++i) frame[off++] = cmdVel[i];
    for (int i = 0; i < ND; ++i) frame[off++] = (lowState()->motorState[i].q - _defaultDofPos[i]) * _dofPosScale;
    for (int i = 0; i < ND; ++i) frame[off++] = lowState()->motorState[i].dq * _dofVelScale;
    for (int i = 0; i < ND; ++i) frame[off++] = _lastAction[i];

    // Slide window
    for (int i = 0; i < NUM_OBS - SD; ++i)
        _obsBuffer[i] = _obsBuffer[i + SD];
    std::copy(frame.begin(), frame.end(), _obsBuffer.begin() + (NUM_OBS - SD));
}

// ═══════════════════════════════════════════════════════════════
//  Action
// ═══════════════════════════════════════════════════════════════

void State_CasbotAmp::_actionCompute() {
    try {
        // Clip observations
        std::vector<float> obsClipped = _obsBuffer;
        for (auto &v : obsClipped) v = clamp(v, -_clipObservations, _clipObservations);

        // ONNX inference
        Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::vector<int64_t> shape = {1, NUM_OBS};
        Ort::Value input = Ort::Value::CreateTensor<float>(mem, obsClipped.data(), obsClipped.size(), shape.data(), shape.size());
        auto outputs = _session->Run(Ort::RunOptions{nullptr}, _inputNames.data(), &input, 1, _outputNames.data(), 1);
        float *actionData = outputs[0].GetTensorMutableData<float>();

        // Clip actions
        int nOut = outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
        for (int i = 0; i < nOut; ++i)
            actionData[i] = clamp(actionData[i], -_clipActions, _clipActions);

        // Scale to motor targets & write to lowCmd
        for (int i = 0; i < NUM_DOF && i < nOut; ++i) {
            int mi = _dofMapping[i];
            _targetPos[mi] = actionData[i] * _dofActionScale[mi] + _defaultDofPos[mi];
            _lastAction[i]  = actionData[i];
        }

        for (int i = 0; i < NUM_DOF; ++i) {
            lowCmd()->motorCmd[i].q   = _targetPos[i];
            lowCmd()->motorCmd[i].dq  = 0;
            lowCmd()->motorCmd[i].Kp  = _kps[i];
            lowCmd()->motorCmd[i].Kd  = _kds[i];
            lowCmd()->motorCmd[i].tau = 0;
        }

        // Anchor termination check
        float pgZ = _obsBuffer[NUM_OBS - SD + 5];  // proj_gravity[2] of most recent frame
        float anchorErr = std::abs(pgZ - (-1.0f));
        _terminateFlag = (anchorErr > _safeProjGravThresh);

    } catch (const std::exception &e) {
        std::cerr << "[CasbotAmp] ONNX error: " << e.what() << std::endl;
        // Hold position
        for (int i = 0; i < NUM_DOF; ++i)
            lowCmd()->motorCmd[i].q = lowState()->motorState[i].q;
    }
}
