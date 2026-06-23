/**
 * @file State_CasbotAmp.h — Casbot AMP locomotion policy state (DDS architecture).
 *
 * Wraps the same ONNX inference + observation logic as the standalone
 * CasbotAmpDeploy, but inherits from FSMState for the DDS FSM framework.
 *
 * Observation: 336 dims (84 × 4 frames)
 * Action:      25 dims
 * Motor params: real casbot_constants.py values (6 groups)
 */

#ifndef CASBOT_STATE_AMP_H
#define CASBOT_STATE_AMP_H

#include "FSM/FSMState.h"
#include "common/mathTypes.h"
#include "common/mathTools.h"
#include <onnxruntime_cxx_api.h>
#include <array>
#include <vector>
#include <memory>
#include <string>

class State_CasbotAmp : public FSMState {
public:
    State_CasbotAmp(CtrlComponents *ctrlComp);
    ~State_CasbotAmp() override = default;

    void enter() override;
    void run()   override;
    void exit()  override;
    FSMStateName checkChange() override;

private:
    // ── ONNX ──
    Ort::Env                          _env{ORT_LOGGING_LEVEL_WARNING, "CasbotAmp"};
    Ort::SessionOptions               _sessionOpts;
    std::unique_ptr<Ort::Session>     _session;
    std::vector<const char*>          _inputNames  = {"obs"};
    std::vector<const char*>          _outputNames = {"actions"};
    std::string                       _modelPath;

    // ── Configuration ──
    static constexpr int NUM_DOF           = CASBOT_NUM_DOF;   // 25
    static constexpr int ROBOT_STATE_DIM   = 84;   // 3+3+3+25+25+25
    static constexpr int HISTORY_LENGTH    = 4;
    static constexpr int NUM_OBS           = ROBOT_STATE_DIM * HISTORY_LENGTH;  // 336

    float _actionScale        = 0.25f;
    float _clipObservations   = 100.0f;
    float _clipActions        = 100.0f;
    float _deadZone           = 0.2f;
    float _cmdSmoothes        = 0.0f;
    float _dofPosScale        = 1.0f;
    float _dofVelScale        = 1.0f;
    float _safeProjGravThresh = 2.6f;

    std::array<float, 2> _vxLim     = {-0.8f, 2.5f};
    std::array<float, 2> _vxLimSlow = {-0.8f, 1.0f};
    std::array<float, 2> _vyLim     = {-1.0f, 1.0f};
    std::array<float, 2> _wyawLim   = {-3.14f, 3.14f};

    // ── Motor parameters (per-joint) ──
    std::array<float, NUM_DOF> _kps{};
    std::array<float, NUM_DOF> _kds{};
    std::array<float, NUM_DOF> _tauLimit{};
    std::array<float, NUM_DOF> _defaultDofPos{};
    std::array<float, NUM_DOF> _dofActionScale{};

    static constexpr int _dofMapping[NUM_DOF] = {
         0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,
        12,13,14,15,16,17,18,19,20,21,22,23,24
    };

    // ── Runtime state ──
    bool _highSpeedMode  = false;
    bool _terminateFlag  = false;
    std::array<float, 3>         _vCmdBodyPast{};
    std::array<float, NUM_DOF>   _lastAction{};
    std::array<float, NUM_DOF>   _targetPos{};
    std::vector<float>           _obsBuffer;    // size = NUM_OBS

    // ── Methods ──
    void _loadPolicy();
    void _initBuffers();
    void _observationsCompute();
    void _actionCompute();
    std::array<float, 3> _getUserCmd();
    static std::array<float, 3> _projectedGravity(const std::array<float, 4> &quat);
};

#endif  // CASBOT_STATE_AMP_H
