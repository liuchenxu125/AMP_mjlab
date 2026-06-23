/**
 * @file IOSDK.cpp — Unitree SDK2 DDS I/O for Casbot (25-DOF).
 *
 * Sends LowCmd_ (29-slot message, casbot uses first 25) to rt/lowcmd.
 * Receives LowState_ from rt/lowstate, unpacks into LowlevelState.
 */

#include "interface/IOSDK.h"
#include <iostream>
#include <thread>
#include <chrono>

IOSDK::IOSDK() {
    // Publisher: motor commands → rt/lowcmd
    unitree::robot::ChannelFactory::Instance()->Init(0);
    _lowcmdPublisher.reset(new ChannelPublisher<LowCmd_>(CASBOT_CMD_TOPIC));
    _lowcmdPublisher->InitChannel();

    // Subscriber: motor state ← rt/lowstate
    _lowstateSubscriber.reset(new ChannelSubscriber<LowState_>(CASBOT_STATE_TOPIC));
    _lowstateSubscriber->InitChannel([this](const void *msg) {
        _onLowState(msg);
    }, 1);

    std::cout << "[IOSDK] DDS channels: cmd→" << CASBOT_CMD_TOPIC
              << ", state←" << CASBOT_STATE_TOPIC << std::endl;
}

void IOSDK::_onLowState(const void *msg) {
    const LowState_ *ls = static_cast<const LowState_ *>(msg);

    // Unpack IMU
    _cachedState.imu.quaternion[0]    = ls->imu_state().quaternion()[0];  // w
    _cachedState.imu.quaternion[1]    = ls->imu_state().quaternion()[1];  // x
    _cachedState.imu.quaternion[2]    = ls->imu_state().quaternion()[2];  // y
    _cachedState.imu.quaternion[3]    = ls->imu_state().quaternion()[3];  // z
    _cachedState.imu.gyroscope[0]     = ls->imu_state().gyroscope()[0];
    _cachedState.imu.gyroscope[1]     = ls->imu_state().gyroscope()[1];
    _cachedState.imu.gyroscope[2]     = ls->imu_state().gyroscope()[2];
    _cachedState.imu.accelerometer[0] = ls->imu_state().accelerometer()[0];
    _cachedState.imu.accelerometer[1] = ls->imu_state().accelerometer()[1];
    _cachedState.imu.accelerometer[2] = ls->imu_state().accelerometer()[2];

    // Unpack 25 motor states (from 29-slot DDS message)
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        _cachedState.motorState[i].q      = ls->motor_state()[i].q();
        _cachedState.motorState[i].dq     = ls->motor_state()[i].dq();
        _cachedState.motorState[i].ddq    = ls->motor_state()[i].ddq();
        _cachedState.motorState[i].tauEst = ls->motor_state()[i].tau_est();
    }

    _stateReady = true;
}

void IOSDK::sendRecv(const LowlevelCmd *cmd, LowlevelState *state) {
    // ── Publish motor commands ──
    LowCmd_ ddsCmd;
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        ddsCmd.motor_cmd()[i].q()   = cmd->motorCmd[i].q;
        ddsCmd.motor_cmd()[i].dq()  = cmd->motorCmd[i].dq;
        ddsCmd.motor_cmd()[i].tau() = cmd->motorCmd[i].tau;
        ddsCmd.motor_cmd()[i].kp()  = cmd->motorCmd[i].Kp;
        ddsCmd.motor_cmd()[i].kd()  = cmd->motorCmd[i].Kd;
    }
    _lowcmdPublisher->Write(ddsCmd);

    // ── Wait for fresh state ──
    _stateReady = false;
    int timeout = 100;  // 100 × 0.5ms = 50ms max
    while (!_stateReady && timeout-- > 0)
        std::this_thread::sleep_for(std::chrono::microseconds(500));

    if (_stateReady)
        *state = _cachedState;
}
