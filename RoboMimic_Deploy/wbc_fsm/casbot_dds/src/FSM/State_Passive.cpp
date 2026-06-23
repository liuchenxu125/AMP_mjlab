#include "FSM/State_Passive.h"
#include <iostream>

State_Passive::State_Passive(CtrlComponents *ctrlComp)
    : FSMState(ctrlComp, FSMStateName::PASSIVE, "passive") {}

void State_Passive::enter() {
    std::cout << "[Passive] Enter — damping mode" << std::endl;
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        lowCmd()->motorCmd[i].q   = lowState()->motorState[i].q;
        lowCmd()->motorCmd[i].dq  = 0;
        lowCmd()->motorCmd[i].Kp  = 0;
        lowCmd()->motorCmd[i].Kd  = _passiveKd;
        lowCmd()->motorCmd[i].tau = 0;
    }
}

void State_Passive::run() {
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        lowCmd()->motorCmd[i].q  = lowState()->motorState[i].q;
        lowCmd()->motorCmd[i].Kp = 0;
        lowCmd()->motorCmd[i].Kd = _passiveKd;
    }
}

void State_Passive::exit() {}

FSMStateName State_Passive::checkChange() {
    if (lowState()->userCmd == UserCommand::START)
        return FSMStateName::FIXEDSTAND;
    if (lowState()->userCmd == UserCommand::R2_B)
        return FSMStateName::CASBOT_AMP;
    return FSMStateName::PASSIVE;
}
