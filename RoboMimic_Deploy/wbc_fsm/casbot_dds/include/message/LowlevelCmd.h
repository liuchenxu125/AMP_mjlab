/**
 * @file LowlevelCmd.h — Casbot DDS: 25-DOF motor command structure.
 */

#ifndef CASBOT_LOWLEVELCMD_H
#define CASBOT_LOWLEVELCMD_H

#include "common/mathTypes.h"

#define CASBOT_NUM_DOF 25

struct MotorCmd {
    unsigned int mode = 0;   // 0=PR (series), 1=AB (parallel)
    float q  = 0.0f;         // target position
    float dq = 0.0f;         // target velocity
    float tau = 0.0f;        // feed-forward torque
    float Kp = 0.0f;         // stiffness
    float Kd = 0.0f;         // damping
};

struct LowlevelCmd {
    MotorCmd motorCmd[CASBOT_NUM_DOF];
};

#endif  // CASBOT_LOWLEVELCMD_H
