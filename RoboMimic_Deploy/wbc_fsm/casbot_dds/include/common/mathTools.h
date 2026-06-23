/**
 * @file mathTools.h — Casbot DDS: quaternion math & rotation utilities.
 * Reuses G1 wbc_fsm mathTools.h patterns, adapted for casbot.
 */

#ifndef CASBOT_MATHTOOLS_H
#define CASBOT_MATHTOOLS_H

#include "common/mathTypes.h"
#include <cmath>
#include <vector>
#include <stdexcept>
#include <algorithm>

// ── Clamp ──
template<typename T>
inline T clamp(const T &a, const T &lo, const T &hi) {
    return std::max(lo, std::min(a, hi));
}

// ── Quaternion → rotation matrix ──
inline RotMat quatToRotMat(const Quat &q) {
    double e0 = q(0), e1 = q(1), e2 = q(2), e3 = q(3);
    RotMat R;
    R << 1 - 2*(e2*e2 + e3*e3), 2*(e1*e2 - e0*e3),     2*(e1*e3 + e0*e2),
         2*(e1*e2 + e0*e3),     1 - 2*(e1*e1 + e3*e3), 2*(e2*e3 - e0*e1),
         2*(e1*e3 - e0*e2),     2*(e2*e3 + e0*e1),     1 - 2*(e1*e1 + e2*e2);
    return R;
}

// ── Rotation matrix → RPY (radians) ──
inline Vec3 rotMatToRPY(const RotMat &R) {
    Vec3 rpy;
    rpy(0) = atan2(R(2,1), R(2,2));
    rpy(1) = asin(-R(2,0));
    rpy(2) = atan2(R(1,0), R(0,0));
    return rpy;
}

// ── QuatRotateInverse: rotate vector by inverse quaternion ──
// Projects world gravity [0,0,-1] into robot frame
inline std::vector<float> quatRotateInverse(const std::vector<float> &q,
                                             const std::vector<float> &v) {
    float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
    Eigen::Vector3f vv(v[0], v[1], v[2]);
    Eigen::Vector3f qv(qx, qy, qz);
    Eigen::Vector3f a = vv * (2.0f * qw * qw - 1.0f);
    Eigen::Vector3f b = qv.cross(vv) * qw * 2.0f;
    Eigen::Vector3f c = qv * (qv.dot(vv) * 2.0f);
    Eigen::Vector3f g = (a - b + c).eval();
    return {g(0), g(1), g(2)};
}

// ── Dead zone ──
inline float deadZone(float val, float threshold) {
    return (val > -threshold && val < threshold) ? 0.0f : val;
}

#endif  // CASBOT_MATHTOOLS_H
