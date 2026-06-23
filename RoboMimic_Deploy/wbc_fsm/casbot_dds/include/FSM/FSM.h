/**
 * @file FSM.h — Casbot DDS: Finite State Machine controller.
 *
 * Manages state transitions. For casbot single-policy deployment,
 * the state list contains: PASSIVE, FIXEDSTAND, CASBOT_AMP.
 * Start state: PASSIVE (damping mode for safety).
 */

#ifndef CASBOT_FSM_H
#define CASBOT_FSM_H

#include <map>
#include "FSM/FSMState.h"
#include "common/enumClass.h"

// Forward declarations
class State_Passive;
class State_FixedStand;
class State_CasbotAmp;

class FSM {
public:
    FSM(CtrlComponents *ctrlComp);
    ~FSM();

    /// Run one control cycle: execute current state, check transitions.
    void run();

    /// Get a pointer to a specific state by name.
    FSMState *getState(FSMStateName name);

private:
    CtrlComponents                    *_ctrlComp;
    std::map<FSMStateName, FSMState*>  _states;
    FSMState                          *_currentState = nullptr;
    FSMMode                            _mode = FSMMode::NORMAL;

    void _createStates();
};

#endif  // CASBOT_FSM_H
