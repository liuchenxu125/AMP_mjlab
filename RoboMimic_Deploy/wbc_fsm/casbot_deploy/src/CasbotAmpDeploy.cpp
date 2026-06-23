/**
 * @file CasbotAmpDeploy.cpp
 * @brief Implementation of the Casbot AMP locomotion policy.
 *
 * Ported from the Python CasbotAMP with identical logic.
 * Uses ONNX Runtime C++ API for inference.
 */

#include "CasbotAmpDeploy.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// ═══════════════════════════════════════════════════════════════
//  Constructor — load config & ONNX model
// ═══════════════════════════════════════════════════════════════

CasbotAmpDeploy::CasbotAmpDeploy(const std::string &configPath)
    : _env(ORT_LOGGING_LEVEL_WARNING, "CasbotAmp")
{
    // ── Load JSON config ──
    std::ifstream f(configPath);
    if (!f.is_open()) {
        throw std::runtime_error("[CasbotAmp] Cannot open config: " + configPath);
    }
    json cfg = json::parse(f);

    _modelPath               = cfg.value("model_path", "model/policy.onnx");

    // Resolve relative model path against config file directory
    if (!_modelPath.empty() && _modelPath[0] != '/') {
        // Extract directory from config path
        std::string configDir = configPath;
        size_t lastSlash = configDir.find_last_of('/');
        if (lastSlash != std::string::npos) {
            configDir = configDir.substr(0, lastSlash);
            _modelPath = configDir + "/" + _modelPath;
        }
    }
    _actionScale             = cfg.value("action_scale", 0.25f);
    _clipObservations         = cfg.value("clip_observations", 100.0f);
    _clipActions              = cfg.value("clip_actions", 100.0f);
    _deadZone                 = cfg.value("dead_zone", 0.2f);
    _cmdSmoothes              = cfg.value("cmd_smoothes", 0.0);
    _dofPosScale              = cfg.value("dof_pos_scale", 1.0f);
    _dofVelScale              = cfg.value("dof_vel_scale", 1.0f);
    _safeProjGravityThreshold = cfg.value("safe_projgravity_threshold", 2.6f);

    // Velocity limits
    _vxLim[0]     = cfg.value("vx_limit_min",      -0.8f);
    _vxLim[1]     = cfg.value("vx_limit_max",       2.5f);
    _vxLimSlow[0] = cfg.value("vx_limit_min_slow", -0.8f);
    _vxLimSlow[1] = cfg.value("vx_limit_max_slow",  1.0f);
    _vyLim[0]     = cfg.value("vy_limit_min",      -1.0f);
    _vyLim[1]     = cfg.value("vy_limit_max",       1.0f);
    _wyawLim[0]   = cfg.value("wyaw_limit_min",    -3.14f);
    _wyawLim[1]   = cfg.value("wyaw_limit_max",     3.14f);

    // ── Set motor parameters per joint (REAL parameters, 6 groups) ──
    // Joint order: L leg(6), R leg(6), waist(1), head(2), L arm(5), R arm(5)
    //
    // LEG_BIG:   pelvic_pitch, pelvic_roll, knee_pitch
    // LEG_SMALL: pelvic_yaw, ankle_pitch, ankle_roll  (also waist_yaw)
    // ARM_MID:   shoulder_pitch, shoulder_roll, elbow_pitch
    // ARM_SMALL: shoulder_yaw, wrist_yaw  (also head_yaw, head_pitch)

    // Left leg: [0]=pelvic_pitch(BIG) [1]=pelvic_roll(BIG) [2]=pelvic_yaw(SMALL)
    //           [3]=knee_pitch(BIG) [4]=ankle_pitch(SMALL) [5]=ankle_roll(SMALL)
    _kps[0] = _kps[1] = _kps[3] = CasbotMotor::STIFFNESS_LEG_BIG;
    _kds[0] = _kds[1] = _kds[3] = CasbotMotor::DAMPING_LEG_BIG;
    _tauLimit[0] = _tauLimit[1] = _tauLimit[3] = CasbotMotor::EFFORT_LEG_BIG;

    _kps[2] = _kps[4] = _kps[5] = CasbotMotor::STIFFNESS_LEG_SMALL;
    _kds[2] = _kds[4] = _kds[5] = CasbotMotor::DAMPING_LEG_SMALL;
    _tauLimit[2] = _tauLimit[4] = _tauLimit[5] = CasbotMotor::EFFORT_LEG_SMALL;

    // Right leg: [6]=pelvic_pitch(BIG) [7]=pelvic_roll(BIG) [8]=pelvic_yaw(SMALL)
    //            [9]=knee_pitch(BIG) [10]=ankle_pitch(SMALL) [11]=ankle_roll(SMALL)
    _kps[6] = _kps[7] = _kps[9] = CasbotMotor::STIFFNESS_LEG_BIG;
    _kds[6] = _kds[7] = _kds[9] = CasbotMotor::DAMPING_LEG_BIG;
    _tauLimit[6] = _tauLimit[7] = _tauLimit[9] = CasbotMotor::EFFORT_LEG_BIG;

    _kps[8] = _kps[10] = _kps[11] = CasbotMotor::STIFFNESS_LEG_SMALL;
    _kds[8] = _kds[10] = _kds[11] = CasbotMotor::DAMPING_LEG_SMALL;
    _tauLimit[8] = _tauLimit[10] = _tauLimit[11] = CasbotMotor::EFFORT_LEG_SMALL;

    // Waist: [12]=waist_yaw → LEG_SMALL (same armature as leg small)
    _kps[12]      = CasbotMotor::STIFFNESS_LEG_SMALL;
    _kds[12]      = CasbotMotor::DAMPING_LEG_SMALL;
    _tauLimit[12] = CasbotMotor::EFFORT_LEG_SMALL;

    // Head: [13]=head_yaw [14]=head_pitch → ARM_SMALL
    _kps[13] = _kps[14] = CasbotMotor::STIFFNESS_ARM_SMALL;
    _kds[13] = _kds[14] = CasbotMotor::DAMPING_ARM_SMALL;
    _tauLimit[13] = _tauLimit[14] = CasbotMotor::EFFORT_ARM_SMALL;

    // Left arm: [15]=shoulder_pitch(MID) [16]=shoulder_roll(MID) [17]=shoulder_yaw(SMALL)
    //           [18]=elbow_pitch(MID) [19]=wrist_yaw(SMALL)
    _kps[15] = _kps[16] = _kps[18] = CasbotMotor::STIFFNESS_ARM_MID;
    _kds[15] = _kds[16] = _kds[18] = CasbotMotor::DAMPING_ARM_MID;
    _tauLimit[15] = _tauLimit[16] = _tauLimit[18] = CasbotMotor::EFFORT_ARM_MID;

    _kps[17] = _kps[19] = CasbotMotor::STIFFNESS_ARM_SMALL;
    _kds[17] = _kds[19] = CasbotMotor::DAMPING_ARM_SMALL;
    _tauLimit[17] = _tauLimit[19] = CasbotMotor::EFFORT_ARM_SMALL;

    // Right arm: [20]=shoulder_pitch(MID) [21]=shoulder_roll(MID) [22]=shoulder_yaw(SMALL)
    //            [23]=elbow_pitch(MID) [24]=wrist_yaw(SMALL)
    _kps[20] = _kps[21] = _kps[23] = CasbotMotor::STIFFNESS_ARM_MID;
    _kds[20] = _kds[21] = _kds[23] = CasbotMotor::DAMPING_ARM_MID;
    _tauLimit[20] = _tauLimit[21] = _tauLimit[23] = CasbotMotor::EFFORT_ARM_MID;

    _kps[22] = _kps[24] = CasbotMotor::STIFFNESS_ARM_SMALL;
    _kds[22] = _kds[24] = CasbotMotor::DAMPING_ARM_SMALL;
    _tauLimit[22] = _tauLimit[24] = CasbotMotor::EFFORT_ARM_SMALL;

    // ── Default joint positions (KNEES_BENT_KEYFRAME) ──
    // Left leg: pelvic_pitch=-0.32, rest=0, knee=0.53, ankle_pitch=-0.19, ankle_roll=0
    _defaultDofPos[0] = -0.32f;  _defaultDofPos[1] = 0.0f;  _defaultDofPos[2] = 0.0f;
    _defaultDofPos[3] =  0.53f;  _defaultDofPos[4] = -0.19f; _defaultDofPos[5] = 0.0f;
    // Right leg (same)
    _defaultDofPos[6] = -0.32f;  _defaultDofPos[7] = 0.0f;  _defaultDofPos[8] = 0.0f;
    _defaultDofPos[9] =  0.53f;  _defaultDofPos[10]= -0.19f; _defaultDofPos[11]= 0.0f;
    // Waist
    _defaultDofPos[12] = 0.0f;
    // Head
    _defaultDofPos[13] = 0.0f;  _defaultDofPos[14] = 0.0f;
    // Left arm: shoulder_pitch=0.2, shoulder_roll=0.3, shoulder_yaw=0, elbow=-0.35, wrist=0
    _defaultDofPos[15] = 0.2f;  _defaultDofPos[16] = 0.3f;  _defaultDofPos[17] = 0.0f;
    _defaultDofPos[18] = -0.35f;_defaultDofPos[19] = 0.0f;
    // Right arm: shoulder_pitch=0.2, shoulder_roll=-0.3, shoulder_yaw=0, elbow=-0.35, wrist=0
    _defaultDofPos[20] = 0.2f;  _defaultDofPos[21] = -0.3f; _defaultDofPos[22] = 0.0f;
    _defaultDofPos[23] = -0.35f;_defaultDofPos[24] = 0.0f;

    // ── Compute dof_action_scale = action_scale × tau_limit / kps ──
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        _dofActionScale[i] = _actionScale * _tauLimit[i] / _kps[i];
    }

    // ── Allocate observation buffer ──
    _obsBuffer.resize(CASBOT_NUM_OBS, 0.0f);

    // ── Load ONNX model ──
    _loadPolicy();

    std::cout << "[CasbotAmp] Policy initialized.\n"
              << "  Model: " << _modelPath << "\n"
              << "  Observation: " << CASBOT_NUM_OBS << " dims, Actions: " << CASBOT_NUM_DOF << "\n"
              << "  dof_action_scale (leg):  " << _dofActionScale[0]  << "\n"
              << "  dof_action_scale (waist): " << _dofActionScale[12] << "\n"
              << "  dof_action_scale (arm):  "  << _dofActionScale[15] << std::endl;
}

// ═══════════════════════════════════════════════════════════════
//  ONNX model loading
// ═══════════════════════════════════════════════════════════════

void CasbotAmpDeploy::_loadPolicy() {
    _session = std::make_unique<Ort::Session>(_env, _modelPath.c_str(), _sessionOptions);

    // Query input shape
    Ort::TypeInfo typeInfo = _session->GetInputTypeInfo(0);
    auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
    _inputShape = tensorInfo.GetShape();
    _obsSize = _inputShape[1];  // [batch, obs_dim]

    // Query output shape
    Ort::TypeInfo outTypeInfo = _session->GetOutputTypeInfo(0);
    auto outTensorInfo = outTypeInfo.GetTensorTypeAndShapeInfo();
    _outputShape = outTensorInfo.GetShape();
    _actionSize = _outputShape[1];  // [batch, action_dim]

    std::cout << "[CasbotAmp] ONNX loaded. Input: " << _obsSize
              << " (" << _inputShape[0] << "×" << _inputShape[1] << ")"
              << ", Output: " << _actionSize << std::endl;
}

// ═══════════════════════════════════════════════════════════════
//  Public API
// ═══════════════════════════════════════════════════════════════

void CasbotAmpDeploy::reset() {
    _highSpeedMode = false;
    _vCmdBodyPast  = {0.0f, 0.0f, 0.0f};
    _lastAction.fill(0.0f);
    _targetPos.fill(0.0f);
    std::fill(_obsBuffer.begin(), _obsBuffer.end(), 0.0f);
}

void CasbotAmpDeploy::initBuffers(
        const std::array<float, 4> &baseQuat,
        const std::array<float, 3> &angVel,
        const std::array<float, CASBOT_NUM_DOF> &q,
        const std::array<float, CASBOT_NUM_DOF> &dq)
{
    _vCmdBodyPast = {0.0f, 0.0f, 0.0f};
    _lastAction.fill(0.0f);
    std::fill(_obsBuffer.begin(), _obsBuffer.end(), 0.0f);

    std::array<float, 3> zeroCmd = {0.0f, 0.0f, 0.0f};
    for (int i = 0; i < CASBOT_HISTORY_LENGTH; ++i) {
        _observationsCompute(baseQuat, angVel, zeroCmd, q, dq);
    }
    std::cout << "[CasbotAmp] Buffers initialized (" << CASBOT_HISTORY_LENGTH
              << " frames)" << std::endl;
}

CasbotAmpDeploy::StepResult CasbotAmpDeploy::step(
        const std::array<float, 4> &baseQuat,
        const std::array<float, 3> &angVel,
        const std::array<float, 3> &cmdVel,
        const std::array<float, CASBOT_NUM_DOF> &q,
        const std::array<float, CASBOT_NUM_DOF> &dq)
{
    _observationsCompute(baseQuat, angVel, cmdVel, q, dq);
    return _actionCompute(_obsBuffer);
}

std::array<float, 3> CasbotAmpDeploy::getUserCmd(float ly, float lx, float rx) {
    const auto &vxLim = _highSpeedMode ? _vxLim : _vxLimSlow;

    auto dead = [this](float val) {
        return (val > -_deadZone && val < _deadZone) ? 0.0f : val;
    };

    float vx = 0.0f, vy = 0.0f, wyaw = 0.0f;

    // Forward velocity from ly
    float ly_d = dead(ly);
    if      (ly_d < 0.0f) vx = ly_d * (-vxLim[0]);
    else if (ly_d > 0.0f) vx = ly_d * vxLim[1];

    // Lateral velocity from lx
    float lx_d = dead(lx);
    if      (lx_d < 0.0f) vy = lx_d * (-_vyLim[0]);
    else if (lx_d > 0.0f) vy = lx_d * _vyLim[1];

    // Yaw rate from rx
    float rx_d = dead(rx);
    if      (rx_d < 0.0f) wyaw = rx_d * (-_wyawLim[0]);
    else if (rx_d > 0.0f) wyaw = rx_d * _wyawLim[1];

    std::array<float, 3> newCmd = {vx, vy, wyaw};

    // Exponential smoothing
    for (int i = 0; i < 3; ++i) {
        _vCmdBodyPast[i] = _vCmdBodyPast[i] * _cmdSmoothes
                         + newCmd[i] * (1.0f - _cmdSmoothes);
    }
    return _vCmdBodyPast;
}

// ═══════════════════════════════════════════════════════════════
//  Internal: Projected Gravity
// ═══════════════════════════════════════════════════════════════

std::array<float, 3> CasbotAmpDeploy::_computeProjectedGravity(
        const std::array<float, 4> &baseQuat)
{
    // Quaternion: [qw, qx, qy, qz]
    float qw = baseQuat[0], qx = baseQuat[1], qy = baseQuat[2], qz = baseQuat[3];
    // Equivalent to QuatRotateInverse([0,0,-1], quat)
    // = rotate gravity vector [0,0,-1] by inverse of base quaternion
    std::array<float, 3> g;
    g[0] =  2.0f * (-qz * qx + qw * qy);
    g[1] = -2.0f * ( qz * qy + qw * qx);
    g[2] =  1.0f - 2.0f * (qw * qw + qz * qz);
    return g;
}

// ═══════════════════════════════════════════════════════════════
//  Internal: Observation Computation
// ═══════════════════════════════════════════════════════════════

void CasbotAmpDeploy::_observationsCompute(
        const std::array<float, 4> &baseQuat,
        const std::array<float, 3> &angVel,
        const std::array<float, 3> &cmdVel,
        const std::array<float, CASBOT_NUM_DOF> &q,
        const std::array<float, CASBOT_NUM_DOF> &dq)
{
    constexpr int SD = CASBOT_ROBOT_STATE_DIM;  // 84
    constexpr int ND = CASBOT_NUM_DOF;           // 25

    // 1. Projected gravity
    std::array<float, 3> projGravity = _computeProjectedGravity(baseQuat);

    // 2. Scaled angular velocity
    float angVelS[3];
    for (int i = 0; i < 3; ++i) angVelS[i] = angVel[i] * _angVelScale[i];

    // 3. Command velocity (passthrough)
    float cmd[3];
    for (int i = 0; i < 3; ++i) cmd[i] = cmdVel[i];

    // 4. Joint positions — offset from default, scaled
    float dofPosS[ND];
    for (int i = 0; i < ND; ++i) {
        int mi = _dofMapping[i];
        dofPosS[i] = (q[mi] - _defaultDofPos[mi]) * _dofPosScale;
    }

    // 5. Joint velocities — scaled
    float dofVelS[ND];
    for (int i = 0; i < ND; ++i) {
        int mi = _dofMapping[i];
        dofVelS[i] = dq[mi] * _dofVelScale;
    }

    // 6. Build single frame [3 + 3 + 3 + 25 + 25 + 25] = 84
    std::vector<float> frame(SD, 0.0f);
    int offset = 0;
    std::copy(angVelS,    angVelS    + 3,  frame.data() + offset); offset += 3;
    std::copy(projGravity.data(), projGravity.data() + 3, frame.data() + offset); offset += 3;
    std::copy(cmd,         cmd         + 3,  frame.data() + offset); offset += 3;
    std::copy(dofPosS,     dofPosS     + ND, frame.data() + offset); offset += ND;
    std::copy(dofVelS,     dofVelS     + ND, frame.data() + offset); offset += ND;
    std::copy(_lastAction.data(), _lastAction.data() + ND, frame.data() + offset);

    // 7. Slide window: shift left, append new frame
    int total = CASBOT_NUM_OBS;  // 336
    for (int i = 0; i < total - SD; ++i) {
        _obsBuffer[i] = _obsBuffer[i + SD];
    }
    std::copy(frame.begin(), frame.end(), _obsBuffer.begin() + (total - SD));
}

// ═══════════════════════════════════════════════════════════════
//  Internal: ONNX Inference → Action Computation
// ═══════════════════════════════════════════════════════════════

CasbotAmpDeploy::StepResult CasbotAmpDeploy::_actionCompute(
        const std::vector<float> &observation)
{
    StepResult result;

    // Use kps/kds as defaults (hold on error)
    result.kps = _kps;
    result.kds = _kds;

    try {
        // ── Prepare input tensor ──
        Ort::MemoryInfo memInfo = Ort::MemoryInfo::CreateCpu(
            OrtArenaAllocator, OrtMemTypeDefault);

        std::vector<float> obsClipped = observation;
        for (auto &v : obsClipped) {
            v = std::clamp(v, -_clipObservations, _clipObservations);
        }

        std::vector<int64_t> inputShape = {1, CASBOT_NUM_OBS};
        Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
            memInfo, obsClipped.data(), obsClipped.size(),
            inputShape.data(), inputShape.size());

        // ── Run inference ──
        auto outputTensors = _session->Run(
            Ort::RunOptions{nullptr},
            _inputNames.data(), &inputTensor, 1,
            _outputNames.data(), 1);

        float *actionData = outputTensors[0].GetTensorMutableData<float>();
        int actionCount = outputTensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

        // ── Clip actions ──
        for (int i = 0; i < actionCount; ++i) {
            actionData[i] = std::clamp(actionData[i], -_clipActions, _clipActions);
        }

        // ── Scale to motor target positions ──
        // target[i] = action[i] × dof_action_scale[i] + default_dof_pos[i]
        for (int policyIdx = 0; policyIdx < CASBOT_NUM_DOF && policyIdx < actionCount; ++policyIdx) {
            int motorIdx = _dofMapping[policyIdx];
            result.actions[motorIdx] = actionData[policyIdx] * _dofActionScale[motorIdx]
                                     + _defaultDofPos[motorIdx];
        }

        // ── Store last action ──
        for (int i = 0; i < CASBOT_NUM_DOF && i < actionCount; ++i) {
            _lastAction[i] = actionData[i];
        }
        _targetPos = result.actions;

        // ── Anchor termination check ──
        // Projected gravity z should be close to -1.0
        // We compute it from the observation buffer (stored at offset 3 per frame)
        // Actually, we need the most recent frame's projected gravity
        // It's at obsBuffer[total-SD+3] through obsBuffer[total-SD+5]
        int pgOffset = CASBOT_NUM_OBS - CASBOT_ROBOT_STATE_DIM + 3;
        float pgZ = _obsBuffer[pgOffset + 2];  // projected_gravity[2]
        float anchorError = std::abs(pgZ - (-1.0f));
        result.terminated = (anchorError > _safeProjGravityThreshold);

        if (result.terminated) {
            std::cerr << "[CasbotAmp WARNING] Large anchor proj_gravity error: "
                      << anchorError << " (threshold: " << _safeProjGravityThreshold << ")"
                      << std::endl;
        }

    } catch (const std::exception &e) {
        std::cerr << "[CasbotAmp ERROR] ONNX inference failed: " << e.what() << std::endl;
        // Hold current position on error
        result.actions = _targetPos;
    }

    return result;
}
