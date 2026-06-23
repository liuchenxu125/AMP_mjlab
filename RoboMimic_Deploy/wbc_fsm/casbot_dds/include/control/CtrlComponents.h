/**
 * @file CtrlComponents.h — Casbot DDS: shared control components container.
 */

#ifndef CASBOT_CTRLCOMPONENTS_H
#define CASBOT_CTRLCOMPONENTS_H

#include "message/LowlevelCmd.h"
#include "message/LowlevelState.h"
#include "interface/IOInterface.h"
#include "common/enumClass.h"

struct CtrlComponents {
    CtrlComponents(IOInterface *io) : ioInter(io) {
        lowCmd   = new LowlevelCmd();
        lowState = new LowlevelState();
    }
    ~CtrlComponents() {
        delete lowCmd;
        delete lowState;
        delete ioInter;
    }

    LowlevelCmd   *lowCmd;
    LowlevelState *lowState;
    IOInterface   *ioInter;
    CtrlPlatform   ctrlPlatform = CtrlPlatform::MUJOCO;
    double         dt           = 0.02;   // 50 Hz control
    bool           exitFlag     = false;
    bool           running      = true;

    void sendRecv() { ioInter->sendRecv(lowCmd, lowState); }
};

#endif  // CASBOT_CTRLCOMPONENTS_H
