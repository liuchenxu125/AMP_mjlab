/**
 * @file State_Passive.h — Damping protection mode (all joints limp).
 */

#ifndef CASBOT_STATE_PASSIVE_H
#define CASBOT_STATE_PASSIVE_H

#include "FSM/FSMState.h"

class State_Passive : public FSMState {
public:
    State_Passive(CtrlComponents *ctrlComp);
    void enter() override;
    void run()   override;
    void exit()  override;
    FSMStateName checkChange() override;

private:
    float _passiveKd = 8.0f;  // low damping for all joints
};

#endif
