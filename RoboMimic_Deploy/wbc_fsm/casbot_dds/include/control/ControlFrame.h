/**
 * @file ControlFrame.h — Casbot DDS: top-level control loop.
 */

#ifndef CASBOT_CONTROLFRAME_H
#define CASBOT_CONTROLFRAME_H

#include "FSM/FSM.h"
#include "control/CtrlComponents.h"

class ControlFrame {
public:
    ControlFrame(CtrlComponents *ctrlComp);
    ~ControlFrame() { delete _fsm; }

    /// Main control loop: sendRecv → FSM.run() → repeat
    void run();

private:
    FSM             *_fsm;
    CtrlComponents  *_ctrlComp;
};

#endif  // CASBOT_CONTROLFRAME_H
