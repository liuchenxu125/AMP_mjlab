/**
 * @file casbot_amp_node.cpp
 * @brief Casbot AMP policy ROS 2 node — real-robot deployment.
 */

#include "casbot_ros2/casbot_amp_node.hpp"
#include <algorithm>
#include <iostream>

namespace casbot_ros2 {

// ═══════════════════════════════════════════════════════════════
//  Constructor
// ═══════════════════════════════════════════════════════════════

CasbotAmpNode::CasbotAmpNode(const std::string &configPath)
    : Node("casbot_amp_node"),
      _policy(configPath)
{
    _buildJointMap();

    // ── Subscriptions ──
    _jointStateSub = this->create_subscription<sensor_msgs::msg::JointState>(
        "/motion/joint_state", 10,
        std::bind(&CasbotAmpNode::_jointStateCb, this, std::placeholders::_1));

    _imuSub = this->create_subscription<sensor_msgs::msg::Imu>(
        "/motion/imu", 10,
        std::bind(&CasbotAmpNode::_imuCb, this, std::placeholders::_1));

    _cmdVelSub = this->create_subscription<geometry_msgs::msg::Twist>(
        "/motion/cmd_vel", 10,
        std::bind(&CasbotAmpNode::_cmdVelCb, this, std::placeholders::_1));

    // ── Publisher (UpperJointData — matches hardware driver type) ──
    _jointCmdPub = this->create_publisher<crb_ros_msg::msg::UpperJointData>(
        "/upper/joint_cmd", 10);

    // ── Policy timer: 50 Hz = 20ms ──
    _policyTimer = this->create_wall_timer(
        std::chrono::milliseconds(20),
        std::bind(&CasbotAmpNode::_policyLoop, this));

    RCLCPP_INFO(this->get_logger(),
        "CasbotAmpNode ready. Policy: %d DOF → %d real joints",
        POLICY_NUM_DOF, REAL_NUM_JOINTS);
}

// ═══════════════════════════════════════════════════════════════
//  Joint name → index map
// ═══════════════════════════════════════════════════════════════

void CasbotAmpNode::_buildJointMap() {
    for (int i = 0; i < POLICY_NUM_DOF; ++i)
        _jointNameToIdx[POLICY_TO_REAL[i]] = i;
    for (int i = 0; i < NUM_EXTRA; ++i)
        _jointNameToIdx[EXTRA_JOINTS[i]] = i;
}

int CasbotAmpNode::_jointIndex(const std::string &name) const {
    auto it = _jointNameToIdx.find(name);
    return (it != _jointNameToIdx.end()) ? it->second : -1;
}

// ═══════════════════════════════════════════════════════════════
//  Callbacks
// ═══════════════════════════════════════════════════════════════

void CasbotAmpNode::_jointStateCb(const sensor_msgs::msg::JointState::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(_mutex);
    for (size_t i = 0; i < msg->name.size(); ++i) {
        int idx = _jointIndex(msg->name[i]);
        if (idx >= 0 && idx < REAL_NUM_JOINTS) {
            if (i < msg->position.size()) _jointPos[idx] = msg->position[i];
            if (i < msg->velocity.size()) _jointVel[idx] = msg->velocity[i];
        }
    }
    _jointStateReady = true;
}

void CasbotAmpNode::_imuCb(const sensor_msgs::msg::Imu::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(_mutex);
    _imuQuat[0] = msg->orientation.w;
    _imuQuat[1] = msg->orientation.x;
    _imuQuat[2] = msg->orientation.y;
    _imuQuat[3] = msg->orientation.z;
    _imuGyro[0] = msg->angular_velocity.x;
    _imuGyro[1] = msg->angular_velocity.y;
    _imuGyro[2] = msg->angular_velocity.z;
    _imuReady = true;
}

void CasbotAmpNode::_cmdVelCb(const geometry_msgs::msg::Twist::SharedPtr msg) {
    std::lock_guard<std::mutex> lock(_mutex);
    _cmdVel[0] = msg->linear.x;
    _cmdVel[1] = msg->linear.y;
    _cmdVel[2] = msg->angular.z;
    _cmdVelReady = true;
}

// ═══════════════════════════════════════════════════════════════
//  Policy loop (50 Hz)
// ═══════════════════════════════════════════════════════════════

void CasbotAmpNode::_policyLoop() {
    std::lock_guard<std::mutex> lock(_mutex);

    if (!_jointStateReady || !_imuReady) return;

    // ── Map real-robot state → policy input (25 DOF) ──
    std::array<float, POLICY_NUM_DOF> q_policy, dq_policy;
    for (int p = 0; p < POLICY_NUM_DOF; ++p) {
        q_policy[p]  = _jointPos[p];   // first 25 joints: legs(12)+waist(1)+head(2)+arms(10)
        dq_policy[p] = _jointVel[p];
    }

    // ── First call: init buffers ──
    if (!_policyInitialized) {
        _policy.initBuffers(_imuQuat, _imuGyro, q_policy, dq_policy);
        _policyActions = _policy.targetPos();
        _kps = _policy.kps();
        _kds = _policy.kds();
        _policyInitialized = true;
        RCLCPP_INFO(this->get_logger(), "Policy buffers initialized");
    }

    // ── Run policy ──
    auto cmd = _cmdVelReady ? _cmdVel : std::array<float, 3>{0,0,0};
    auto result = _policy.step(_imuQuat, _imuGyro, cmd, q_policy, dq_policy);
    _policyActions = result.actions;

    // Override with real-robot PD gains (from motion_params.yaml)
    // Real motor driver runs PD @ 1000Hz with these gains
    static const float REAL_KP_LEG[6]  = {400, 500, 380, 440, 120, 80};
    static const float REAL_KD_LEG[6]  = {30,  19,  5,   25,  3,   3};
    static const float REAL_KP_ARM[5]  = {150, 150, 150, 100, 100};
    static const float REAL_KD_ARM[5]  = {10,  10,  10,  5,   5};
    static const float REAL_KP_WAIST   = 1000, REAL_KD_WAIST = 49;
    static const float REAL_KP_HEAD    = 30,   REAL_KD_HEAD  = 2;

    // Apply real gains: legs(0-11), waist(12), head(13-14), arms(15-24)
    for (int leg = 0; leg < 2; ++leg) {
        for (int j = 0; j < 6; ++j) {
            int idx = leg * 6 + j;
            _kps[idx] = REAL_KP_LEG[j];  _kds[idx] = REAL_KD_LEG[j];
        }
    }
    _kps[12] = REAL_KP_WAIST;  _kds[12] = REAL_KD_WAIST;
    _kps[13] = _kps[14] = REAL_KP_HEAD;  _kds[13] = _kds[14] = REAL_KD_HEAD;
    for (int arm = 0; arm < 2; ++arm) {
        for (int j = 0; j < 5; ++j) {
            int idx = 15 + arm * 5 + j;
            _kps[idx] = REAL_KP_ARM[j];  _kds[idx] = REAL_KD_ARM[j];
        }
    }

    if (result.terminated) {
        RCLCPP_WARN(this->get_logger(), "Anchor gravity threshold exceeded!");
    }

    // ── Publish ──
    _publishCmd();
}

// ═══════════════════════════════════════════════════════════════
//  Publish motor commands
// ═══════════════════════════════════════════════════════════════

void CasbotAmpNode::_publishCmd() {
    auto msg = crb_ros_msg::msg::UpperJointData();
    msg.header.stamp = this->now();
    msg.time_ref = 0.0f;
    msg.vel_scale = 1.0f;

    auto &joint = msg.joint;
    // Policy-controlled joints (25)
    for (int p = 0; p < POLICY_NUM_DOF; ++p) {
        joint.name.push_back(POLICY_TO_REAL[p]);
        joint.position.push_back(_policyActions[p]);
        joint.velocity.push_back(0.0);
        joint.effort.push_back(_kps[p]);    // Kp packed in effort
    }

    // Extra joints (arm_l6,l7, arm_r6,r7) — hold current position
    for (int e = 0; e < NUM_EXTRA; ++e) {
        joint.name.push_back(EXTRA_JOINTS[e]);
        int ri = POLICY_NUM_DOF + e;
        joint.position.push_back(_jointPos[ri]);
        joint.velocity.push_back(0.0);
        joint.effort.push_back(80.0f);
    }

    _jointCmdPub->publish(msg);
}

}  // namespace casbot_ros2
