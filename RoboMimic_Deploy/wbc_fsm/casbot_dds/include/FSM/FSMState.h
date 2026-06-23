/**
 * @file FSMState.h — Casbot DDS: abstract FSM state base class.
 */

#ifndef CASBOT_FSMSTATE_H
#define CASBOT_FSMSTATE_H

#include <string>
#include "control/CtrlComponents.h"
#include "common/enumClass.h"

class FSMState {
public:
    FSMState(CtrlComponents *ctrlComp, FSMStateName name, std::string nameStr)
        : _ctrlComp(ctrlComp), _stateName(name), _stateNameStr(nameStr) {}

    virtual ~FSMState() = default;

    virtual void enter() = 0;
    virtual void run()   = 0;
    virtual void exit()  = 0;
    virtual FSMStateName checkChange() { return FSMStateName::INVALID; }

    FSMStateName  _stateName;
    std::string   _stateNameStr;

protected:
    CtrlComponents *_ctrlComp;
    FSMStateName    _nextStateName = FSMStateName::INVALID;

    // Convenience accessors
    LowlevelCmd   *lowCmd()   { return _ctrlComp->lowCmd; }
    LowlevelState *lowState() { return _ctrlComp->lowState; }
};

#endif  // CASBOT_FSMSTATE_H
