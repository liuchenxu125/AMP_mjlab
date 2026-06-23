/**
 * @file CasbotAmpDeploy.h
 * @brief Casbot Skeleton (25-DOF) AMP locomotion policy — C++ MuJoCo deployment.
 *
 * Standalone policy class (no FSM dependency). Adapted from G1 State_MJAMP
 * for the 25-joint casbot_skeleton robot trained with the same MJLAB AMP
 * framework and hyper-parameters.
 *
 * Observation layout (84 dims/frame × 4-frame history = 336 total):
 *   [ang_vel(3), proj_gravity(3), cmd_vel(3),
 *    dof_pos(25), dof_vel(25), last_action(25)]
 *
 * Joint order (matches MJCF XML and MJLAB training order):
 *   Left leg  (0-5):  pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
 *   Right leg (6-11): pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
 *   Waist     (12):   waist_yaw
 *   Head      (13-14):head_yaw, head_pitch
 *   Left arm  (15-19):shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw
 *   Right arm (20-24):shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw
 */

#ifndef CASBOT_AMP_DEPLOY_H
#define CASBOT_AMP_DEPLOY_H

#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

// ── Casbot joint count ──
#define CASBOT_NUM_DOF 25

// ── Observation dimensions ──
#define CASBOT_ROBOT_STATE_DIM 84   // 3+3+3+25+25+25
#define CASBOT_HISTORY_LENGTH 4
#define CASBOT_NUM_OBS 336          // 84 × 4

// ── Motor constants for casbot (from casbot_constants.py — REAL parameters) ──
// STIFFNESS = ARMATURE × (10×2π)²,  DAMPING = 2 × DAMPING_RATIO × ARMATURE × (10×2π)
namespace CasbotMotor {
    constexpr double NATURAL_FREQ   = 10.0 * 2.0 * 3.1415926535;  // 62.8319 rad/s
    constexpr double DAMPING_RATIO  = 2.0;

    // Leg big: pelvic_pitch, pelvic_roll, knee_pitch  (6 joints)
    constexpr double ARMATURE_LEG_BIG   = 0.06999046;
    constexpr double STIFFNESS_LEG_BIG  = ARMATURE_LEG_BIG * NATURAL_FREQ * NATURAL_FREQ;   // 276.31
    constexpr double DAMPING_LEG_BIG    = 2.0 * DAMPING_RATIO * ARMATURE_LEG_BIG * NATURAL_FREQ; // 17.59
    constexpr double EFFORT_LEG_BIG     = 150.0;

    // Leg small: pelvic_yaw, ankle_pitch, ankle_roll  (6 joints)
    constexpr double ARMATURE_LEG_SMALL   = 0.03959369;
    constexpr double STIFFNESS_LEG_SMALL  = ARMATURE_LEG_SMALL * NATURAL_FREQ * NATURAL_FREQ;  // 156.31
    constexpr double DAMPING_LEG_SMALL    = 2.0 * DAMPING_RATIO * ARMATURE_LEG_SMALL * NATURAL_FREQ; // 9.95
    constexpr double EFFORT_LEG_SMALL     = 60.0;

    // Arm mid: shoulder_pitch, shoulder_roll, elbow_pitch  (6 joints)
    constexpr double ARMATURE_ARM_MID   = 0.03298028;
    constexpr double STIFFNESS_ARM_MID  = ARMATURE_ARM_MID * NATURAL_FREQ * NATURAL_FREQ;    // 130.20
    constexpr double DAMPING_ARM_MID    = 2.0 * DAMPING_RATIO * ARMATURE_ARM_MID * NATURAL_FREQ; // 8.29
    constexpr double EFFORT_ARM_MID     = 75.0;

    // Arm small: shoulder_yaw, wrist_yaw  (4 joints) + head_yaw/pitch + waist_yaw
    constexpr double ARMATURE_ARM_SMALL   = 0.02452611;
    constexpr double STIFFNESS_ARM_SMALL  = ARMATURE_ARM_SMALL * NATURAL_FREQ * NATURAL_FREQ;  // 96.83
    constexpr double DAMPING_ARM_SMALL    = 2.0 * DAMPING_RATIO * ARMATURE_ARM_SMALL * NATURAL_FREQ; // 6.16
    constexpr double EFFORT_ARM_SMALL     = 36.0;
}  // namespace CasbotMotor


/**
 * @brief Standalone AMP locomotion policy for the Casbot Skeleton robot.
 *
 * Usage:
 *   CasbotAmpDeploy policy("config/casbot_amp.json");
 *   policy.initBuffers(baseQuat, angVel, jointPos, jointVel);
 *   while (running) {
 *       CasbotAmpDeploy::StepResult res = policy.step(
 *           baseQuat, angVel, cmdVel, jointPos, jointVel);
 *       // res.actions[25]  — target joint positions
 *       // res.kps[25]      — stiffness gains for PD control
 *       // res.kds[25]      — damping gains for PD control
 *       // res.terminated   — anchor safety triggered
 *   }
 */
class CasbotAmpDeploy {
public:
    /// Result returned by each step() call.
    struct StepResult {
        std::array<float, CASBOT_NUM_DOF> actions{};
        std::array<float, CASBOT_NUM_DOF> kps{};
        std::array<float, CASBOT_NUM_DOF> kds{};
        bool terminated = false;
    };

    /**
     * @brief Construct from a JSON config file.
     * @param configPath  Path to casbot_amp.json.
     */
    explicit CasbotAmpDeploy(const std::string &configPath);
    ~CasbotAmpDeploy() = default;

    // ── Public API ──────────────────────────────────────────────

    /// Reset runtime buffers (call once before the first step).
    void reset();

    /**
     * @brief Fill observation history buffer.
     * @param baseQuat  Base orientation quaternion [w, x, y, z].
     * @param angVel    Base angular velocity [3].
     * @param q         Joint positions [25].
     * @param dq        Joint velocities [25].
     */
    void initBuffers(const std::array<float, 4> &baseQuat,
                     const std::array<float, 3> &angVel,
                     const std::array<float, CASBOT_NUM_DOF> &q,
                     const std::array<float, CASBOT_NUM_DOF> &dq);

    /**
     * @brief Run one policy step.
     * @param baseQuat  Base orientation quaternion [w, x, y, z].
     * @param angVel    Base angular velocity [3].
     * @param cmdVel    Commanded velocity [vx, vy, wyaw].
     * @param q         Joint positions [25].
     * @param dq        Joint velocities [25].
     * @return StepResult with target positions, gains, and termination flag.
     */
    StepResult step(const std::array<float, 4> &baseQuat,
                    const std::array<float, 3> &angVel,
                    const std::array<float, 3> &cmdVel,
                    const std::array<float, CASBOT_NUM_DOF> &q,
                    const std::array<float, CASBOT_NUM_DOF> &dq);

    /**
     * @brief Process joystick-style velocity commands with dead zone & limits.
     * @param ly  Left stick Y  [-1..1] (forward / back).
     * @param lx  Left stick X  [-1..1] (lateral).
     * @param rx  Right stick X [-1..1] (yaw).
     * @return Processed velocity [vx, vy, wyaw].
     */
    std::array<float, 3> getUserCmd(float ly, float lx, float rx);

    // ── Getters / Setters ──────────────────────────────────────
    bool highSpeedMode() const { return _highSpeedMode; }
    void setHighSpeedMode(bool v) { _highSpeedMode = v; }

    const std::array<float, CASBOT_NUM_DOF> &defaultDofPos() const { return _defaultDofPos; }
    const std::array<float, CASBOT_NUM_DOF> &kps()          const { return _kps; }
    const std::array<float, CASBOT_NUM_DOF> &kds()          const { return _kds; }
    const std::array<float, CASBOT_NUM_DOF> &tauLimit()     const { return _tauLimit; }
    const std::array<float, CASBOT_NUM_DOF> &targetPos()    const { return _targetPos; }

private:
    // ── ONNX ───────────────────────────────────────────────────
    void _loadPolicy();

    Ort::Env                          _env;
    Ort::SessionOptions               _sessionOptions;
    std::unique_ptr<Ort::Session>     _session;
    Ort::AllocatorWithDefaultOptions  _allocator;

    std::vector<const char *> _inputNames  = {"obs"};
    std::vector<const char *> _outputNames = {"actions"};

    std::vector<int64_t> _inputShape;
    std::vector<int64_t> _outputShape;
    int64_t _obsSize    = CASBOT_NUM_OBS;
    int64_t _actionSize = CASBOT_NUM_DOF;

    std::string _modelPath;

    // ── Core pipeline ──────────────────────────────────────────
    void _observationsCompute(const std::array<float, 4> &baseQuat,
                              const std::array<float, 3> &angVel,
                              const std::array<float, 3> &cmdVel,
                              const std::array<float, CASBOT_NUM_DOF> &q,
                              const std::array<float, CASBOT_NUM_DOF> &dq);

    StepResult _actionCompute(const std::vector<float> &observation);

    /// Projected gravity: world gravity [0,0,-1] rotated into robot frame.
    static std::array<float, 3> _computeProjectedGravity(const std::array<float, 4> &baseQuat);

    // ── Configuration (loaded from JSON) ───────────────────────
    float _actionScale     = 0.25f;
    float _clipObservations = 100.0f;
    float _clipActions      = 100.0f;
    float _deadZone         = 0.2f;
    float _cmdSmoothes      = 0.0f;

    std::array<float, 3> _angVelScale  = {1.0f, 1.0f, 1.0f};
    float _dofPosScale = 1.0f;
    float _dofVelScale = 1.0f;

    // Velocity limits
    std::array<float, 2> _vxLim      = {-0.8f, 2.5f};
    std::array<float, 2> _vxLimSlow  = {-0.8f, 1.0f};
    std::array<float, 2> _vyLim      = {-1.0f, 1.0f};
    std::array<float, 2> _wyawLim    = {-3.14f, 3.14f};

    float _safeProjGravityThreshold = 2.6f;

    // ── Motor parameters ───────────────────────────────────────
    std::array<float, CASBOT_NUM_DOF> _kps{};
    std::array<float, CASBOT_NUM_DOF> _kds{};
    std::array<float, CASBOT_NUM_DOF> _tauLimit{};
    std::array<float, CASBOT_NUM_DOF> _defaultDofPos{};
    std::array<float, CASBOT_NUM_DOF> _dofActionScale{};

    /// Identity mapping: policy action index == motor index
    static constexpr int _dofMapping[CASBOT_NUM_DOF] = {
         0,  1,  2,  3,  4,  5,      // left leg
         6,  7,  8,  9, 10, 11,      // right leg
        12,                           // waist
        13, 14,                       // head
        15, 16, 17, 18, 19,          // left arm
        20, 21, 22, 23, 24           // right arm
    };

    /// Gravity in world frame
    static constexpr float _gravityVec[3] = {0.0f, 0.0f, -1.0f};

    // ── Runtime state ──────────────────────────────────────────
    bool _highSpeedMode = false;

    std::array<float, 3> _vCmdBodyPast{};
    std::array<float, CASBOT_NUM_DOF> _lastAction{};
    std::array<float, CASBOT_NUM_DOF> _targetPos{};
    std::vector<float> _obsBuffer;  // size = CASBOT_NUM_OBS
};

#endif  // CASBOT_AMP_DEPLOY_H
