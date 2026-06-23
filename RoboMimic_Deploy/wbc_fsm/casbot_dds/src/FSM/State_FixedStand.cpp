#include "FSM/State_FixedStand.h"
#include <iostream>
#include <algorithm>

// Default pose (KNEES_BENT_KEYFRAME)
static const float DEFAULT_POSE[CASBOT_NUM_DOF] = {
    -0.32f,0,0, 0.53f,-0.19f,0,  -0.32f,0,0, 0.53f,-0.19f,0,
    0, 0,0,  0.2f,0.3f,0,-0.35f,0,  0.2f,-0.3f,0,-0.35f,0
};

State_FixedStand::State_FixedStand(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::FIXEDSTAND, "fixed_stand") {}

void State_FixedStand::enter() {
    std::cout << "[FixedStand] Enter — interpolating to default pose" << std::endl;
    _elapsed = 0;
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        _startPos[i]  = lowState()->motorState[i].q;
        _targetPos[i] = DEFAULT_POSE[i];
    }
}

void State_FixedStand::run() {
    _elapsed += _ctrlComp->dt;
    float t = std::min(_elapsed / _duration, 1.0f);

    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        float q = _startPos[i] + (_targetPos[i] - _startPos[i]) * t;
        lowCmd()->motorCmd[i].q   = q;
        lowCmd()->motorCmd[i].dq  = 0;
        lowCmd()->motorCmd[i].Kp  = 50;
        lowCmd()->motorCmd[i].Kd  = 3;
        lowCmd()->motorCmd[i].tau = 0;
    }
}

void State_FixedStand::exit() {}

FSMStateName State_FixedStand::checkChange() {
    if (_elapsed >= _duration)
        return FSMStateName::CASBOT_AMP;
    return FSMStateName::FIXEDSTAND;
}
