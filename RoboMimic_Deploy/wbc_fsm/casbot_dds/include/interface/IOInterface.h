/**
 * @file IOInterface.h — Casbot DDS: abstract I/O layer.
 * Same pattern as G1: sendRecv() exchanges motor commands and state.
 * Implementations: IOSDK (DDS/Unitree), IOMujoco (sim), IOReal (real robot).
 */

#ifndef CASBOT_IOINTERFACE_H
#define CASBOT_IOINTERFACE_H

#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"

class IOInterface {
public:
    IOInterface() = default;
    virtual ~IOInterface() = default;

    /// Send motor commands, receive robot state. Blocking call per control cycle.
    virtual void sendRecv(const LowlevelCmd *cmd, LowlevelState *state) = 0;
};

#endif  // CASBOT_IOINTERFACE_H
