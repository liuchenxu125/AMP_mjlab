#include "FSM/FSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixedStand.h"
#include "FSM/State_CasbotAmp.h"
#include <iostream>

FSM::FSM(CtrlComponents *ctrlComp) : _ctrlComp(ctrlComp) {
    _createStates();
    _currentState = _states[FSMStateName::PASSIVE];
    _currentState->enter();
    _mode = FSMMode::NORMAL;
}

FSM::~FSM() {
    for (auto &kv : _states) delete kv.second;
}

void FSM::_createStates() {
    _states[FSMStateName::PASSIVE]    = new State_Passive(_ctrlComp);
    _states[FSMStateName::FIXEDSTAND] = new State_FixedStand(_ctrlComp);
    _states[FSMStateName::CASBOT_AMP] = new State_CasbotAmp(_ctrlComp);
}

FSMState *FSM::getState(FSMStateName name) {
    auto it = _states.find(name);
    return (it != _states.end()) ? it->second : nullptr;
}

void FSM::run() {
    if (_mode == FSMMode::NORMAL) {
        _currentState->run();
        FSMStateName next = _currentState->checkChange();
        if (next != _currentState->_stateName && next != FSMStateName::INVALID) {
            _mode = FSMMode::CHANGE;
            _currentState->exit();
            _currentState = _states[next];
            std::cout << "[FSM] Transition: → " << _currentState->_stateNameStr << std::endl;
            _currentState->enter();
            _mode = FSMMode::NORMAL;
        }
    }
}
