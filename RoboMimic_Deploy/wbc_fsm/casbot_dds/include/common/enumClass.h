/**
 * @file enumClass.h — Casbot DDS: platform & FSM state enums.
 */

#ifndef CASBOT_ENUMCLASS_H
#define CASBOT_ENUMCLASS_H

enum class CtrlPlatform {
    MUJOCO,
    REALROBOT,
};

enum class FSMMode {
    NORMAL,
    CHANGE
};

enum class FSMStateName {
    INVALID,
    PASSIVE,
    FIXEDSTAND,
    CASBOT_AMP,
};

enum class UserCommand {
    NONE,
    SELECT,     // exit
    START,      // fixed pose reset
    R2,         // toggle speed mode (press=low, release=high)
    L2_B,       // passive (emergency stop)
    R2_B,       // back to amp from passive
};

#endif  // CASBOT_ENUMCLASS_H
