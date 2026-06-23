/**
 * @file LowlevelState.h — Casbot DDS: 25-DOF motor state + IMU.
 */

#ifndef CASBOT_LOWLEVELSTATE_H
#define CASBOT_LOWLEVELSTATE_H

#include "common/mathTypes.h"
#include "common/mathTools.h"
#include "common/enumClass.h"

struct MotorState {
    unsigned int mode = 0;
    float q      = 0.0f;       // joint position
    float dq     = 0.0f;       // joint velocity
    float ddq    = 0.0f;       // joint acceleration
    float tauEst = 0.0f;       // estimated torque
};

struct IMU {
    float quaternion[4] = {1, 0, 0, 0};   // [w, x, y, z]
    float gyroscope[3]  = {0, 0, 0};
    float accelerometer[3] = {0, 0, 0};

    RotMat getRotMat() const {
        Quat q; q << quaternion[0], quaternion[1], quaternion[2], quaternion[3];
        return quatToRotMat(q);
    }

    Vec3 getGyro() const {
        return Vec3(gyroscope[0], gyroscope[1], gyroscope[2]);
    }

    Quat getQuat() const {
        Quat q; q << quaternion[0], quaternion[1], quaternion[2], quaternion[3];
        return q;
    }
};

struct LowlevelState {
    IMU imu;
    MotorState motorState[CASBOT_NUM_DOF];
    UserCommand userCmd  = UserCommand::NONE;
    float userValue[3]   = {0, 0, 0};  // [ly, lx, rx] joystick axes

    RotMat getRotMat() const { return imu.getRotMat(); }
    Vec3   getGyro()   const { return imu.getGyro(); }
    double getYaw()    const { return rotMatToRPY(getRotMat())(2); }
};

#endif  // CASBOT_LOWLEVELSTATE_H
