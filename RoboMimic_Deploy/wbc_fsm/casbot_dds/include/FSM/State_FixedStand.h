/**
 * @file State_FixedStand.h — Smooth transition to default standing pose.
 */

#ifndef CASBOT_STATE_FIXEDSTAND_H
#define CASBOT_STATE_FIXEDSTAND_H

#include "FSM/FSMState.h"

class State_FixedStand : public FSMState {
public:
    State_FixedStand(CtrlComponents *ctrlComp);
    void enter() override;
    void run()   override;
    void exit()  override;
    FSMStateName checkChange() override;

private:
    float _duration    = 2.0f;   // seconds to reach target
    float _elapsed     = 0.0f;
    float _startPos[CASBOT_NUM_DOF]{};
    float _targetPos[CASBOT_NUM_DOF]{};  // default_dof_pos
};

#endif
