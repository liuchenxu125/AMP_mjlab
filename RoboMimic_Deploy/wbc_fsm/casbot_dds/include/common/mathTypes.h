/**
 * @file mathTypes.h — Casbot DDS: Eigen typedefs & motor armature constants.
 * Adapted from G1 wbc_fsm, 25-DOF casbot with real motor parameters.
 */

#ifndef CASBOT_MATHTYPES_H
#define CASBOT_MATHTYPES_H

#include <eigen3/Eigen/Dense>

// ── Vector typedefs ──
using Vec2  = Eigen::Matrix<double, 2, 1>;
using Vec3  = Eigen::Matrix<double, 3, 1>;
using Vec4  = Eigen::Matrix<double, 4, 1>;
using Vec6  = Eigen::Matrix<double, 6, 1>;
using Vec25 = Eigen::Matrix<double, 25, 1>;
using VecX  = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Quat  = Eigen::Matrix<double, 4, 1>;
using RotMat = Eigen::Matrix<double, 3, 3>;
using Mat3  = Eigen::Matrix<double, 3, 3>;
using MatX  = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

#define I3 Eigen::MatrixXd::Identity(3, 3)

// ── Motor armature constants (from casbot_constants.py — REAL parameters) ──
namespace CasbotArmature {
    constexpr double NATURAL_FREQ  = 10.0 * 2.0 * 3.1415926535;  // 62.8319 rad/s
    constexpr double DAMPING_RATIO = 2.0;

    // Leg big: pelvic_pitch, pelvic_roll, knee_pitch
    constexpr double ARMATURE_LEG_BIG      = 0.06999046;
    constexpr double STIFFNESS_LEG_BIG     = ARMATURE_LEG_BIG * NATURAL_FREQ * NATURAL_FREQ;
    constexpr double DAMPING_LEG_BIG       = 2.0 * DAMPING_RATIO * ARMATURE_LEG_BIG * NATURAL_FREQ;
    constexpr double EFFORT_LEG_BIG        = 150.0;

    // Leg small: pelvic_yaw, ankle_pitch, ankle_roll
    constexpr double ARMATURE_LEG_SMALL    = 0.03959369;
    constexpr double STIFFNESS_LEG_SMALL   = ARMATURE_LEG_SMALL * NATURAL_FREQ * NATURAL_FREQ;
    constexpr double DAMPING_LEG_SMALL     = 2.0 * DAMPING_RATIO * ARMATURE_LEG_SMALL * NATURAL_FREQ;
    constexpr double EFFORT_LEG_SMALL      = 60.0;

    // Arm mid: shoulder_pitch, shoulder_roll, elbow_pitch
    constexpr double ARMATURE_ARM_MID      = 0.03298028;
    constexpr double STIFFNESS_ARM_MID     = ARMATURE_ARM_MID * NATURAL_FREQ * NATURAL_FREQ;
    constexpr double DAMPING_ARM_MID       = 2.0 * DAMPING_RATIO * ARMATURE_ARM_MID * NATURAL_FREQ;
    constexpr double EFFORT_ARM_MID        = 75.0;

    // Arm small: shoulder_yaw, wrist_yaw (also head yaw/pitch, waist yaw)
    constexpr double ARMATURE_ARM_SMALL    = 0.02452611;
    constexpr double STIFFNESS_ARM_SMALL   = ARMATURE_ARM_SMALL * NATURAL_FREQ * NATURAL_FREQ;
    constexpr double DAMPING_ARM_SMALL     = 2.0 * DAMPING_RATIO * ARMATURE_ARM_SMALL * NATURAL_FREQ;
    constexpr double EFFORT_ARM_SMALL      = 36.0;
}  // namespace CasbotArmature

#endif  // CASBOT_MATHTYPES_H
