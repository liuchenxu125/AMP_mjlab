/**
 * @file IOSDK.h — Casbot DDS: Unitree SDK2 DDS I/O implementation.
 *
 * Uses the same DDS topics as G1 (rt/lowcmd, rt/lowstate).
 * The 25-DOF Casbot motor commands are packed into the standard
 * Unitree HG LowCmd_ message (first 25 of 29 slots used).
 */

#ifndef CASBOT_IOSDK_H
#define CASBOT_IOSDK_H

#include "interface/IOInterface.h"
#include <string>

// Unitree SDK2 headers (same as G1)
#include <unitree/robot/channel/channel_publisher.hpp>
#include <unitree/robot/channel/channel_subscriber.hpp>
#include <unitree/idl/hg/LowCmd_.hpp>
#include <unitree/idl/hg/LowState_.hpp>

static const std::string CASBOT_CMD_TOPIC   = "rt/lowcmd";
static const std::string CASBOT_STATE_TOPIC = "rt/lowstate";
static const std::string CASBOT_IMU_TOPIC   = "rt/secondary_imu";

using namespace unitree::common;
using namespace unitree::robot;
using namespace unitree_hg::msg::dds_;

class IOSDK : public IOInterface {
public:
    IOSDK();
    ~IOSDK() override = default;

    void sendRecv(const LowlevelCmd *cmd, LowlevelState *state) override;

private:
    ChannelPublisherPtr<LowCmd_>   _lowcmdPublisher;
    ChannelSubscriberPtr<LowState_> _lowstateSubscriber;

    // Callback for incoming lowstate messages
    void _onLowState(const void *msg);
    LowlevelState _cachedState;
    bool _stateReady = false;
};

#endif  // CASBOT_IOSDK_H
