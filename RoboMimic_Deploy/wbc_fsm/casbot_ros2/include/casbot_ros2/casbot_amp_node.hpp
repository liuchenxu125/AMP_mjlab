/**
 * @file casbot_amp_node.hpp
 * @brief ROS 2 node wrapping CasbotAmpDeploy policy for real-robot deployment.
 *
 * Architecture:
 *   SUBSCRIBE  /motion/joint_state  (sensor_msgs::JointState) → robot state
 *   SUBSCRIBE  /motion/imu          (sensor_msgs::Imu)        → IMU data
 *   SUBSCRIBE  /motion/cmd_vel      (geometry_msgs::Twist)    → velocity cmd
 *   PUBLISH    /upper/joint_cmd    (sensor_msgs::JointState) → motor targets
 *
 * Joint mapping (MuJoCo policy → real robot):
 *   Policy[0..5]  = L leg → leg_l1..l6_joint
 *   Policy[6..11] = R leg → leg_r1..r6_joint
 *   Policy[12]    = waist → waist_yaw_joint
 *   Policy[13..14]= head  → head_yaw/pitch_joint
 *   Policy[15..19]= L arm → arm_l1..l5_joint
 *   Policy[20..24]= R arm → arm_r1..r5_joint
 *
 * Extra real-robot joints (arm_l6/l7, arm_r6/r7) held at current position.
 */

#ifndef CASBOT_AMP_NODE_HPP
#define CASBOT_AMP_NODE_HPP

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <crb_ros_msg/msg/upper_joint_data.hpp>

#include <array>
#include <string>
#include <vector>
#include <mutex>
#include <memory>

#include "CasbotAmpDeploy.h"

namespace casbot_ros2 {

class CasbotAmpNode : public rclcpp::Node {
public:
    explicit CasbotAmpNode(const std::string &configPath);
    ~CasbotAmpNode() override = default;

private:
    // ── Timer callback (policy loop @ 50Hz) ──
    void _policyLoop();

    // ── Subscription callbacks ──
    void _jointStateCb(const sensor_msgs::msg::JointState::SharedPtr msg);
    void _imuCb(const sensor_msgs::msg::Imu::SharedPtr msg);
    void _cmdVelCb(const geometry_msgs::msg::Twist::SharedPtr msg);

    // ── Joint name → index helpers ──
    void _buildJointMap();
    int  _jointIndex(const std::string &name) const;

    // ── Publish motor commands ──
    void _publishCmd();

    // ═════════════════════════════════════════════════════
    //  Policy
    // ═════════════════════════════════════════════════════
    CasbotAmpDeploy _policy;

    // ═════════════════════════════════════════════════════
    //  State (protected by _mutex)
    // ═════════════════════════════════════════════════════
    std::mutex _mutex;

    // Robot state (29 joints on real robot, we use 25 from policy + 4 held)
    static constexpr int REAL_NUM_JOINTS = 29;
    static constexpr int POLICY_NUM_DOF   = CASBOT_NUM_DOF;  // 25

    std::array<float, REAL_NUM_JOINTS> _jointPos{};
    std::array<float, REAL_NUM_JOINTS> _jointVel{};
    bool _jointStateReady = false;

    // IMU
    std::array<float, 4> _imuQuat{1,0,0,0};
    std::array<float, 3> _imuGyro{};
    bool _imuReady = false;

    // Velocity command
    std::array<float, 3> _cmdVel{};
    bool _cmdVelReady = false;

    // Policy buffers
    bool _policyInitialized = false;
    std::array<float, POLICY_NUM_DOF> _policyActions{};
    std::array<float, POLICY_NUM_DOF> _kps{};
    std::array<float, POLICY_NUM_DOF> _kds{};

    // ── Joint name mapping ──
    // Policy joint order → real robot joint names
    static constexpr const char* POLICY_TO_REAL[POLICY_NUM_DOF] = {
        "leg_l1_joint","leg_l2_joint","leg_l3_joint","leg_l4_joint","leg_l5_joint","leg_l6_joint",
        "leg_r1_joint","leg_r2_joint","leg_r3_joint","leg_r4_joint","leg_r5_joint","leg_r6_joint",
        "waist_yaw_joint",
        "head_yaw_joint","head_pitch_joint",
        "arm_l1_joint","arm_l2_joint","arm_l3_joint","arm_l4_joint","arm_l5_joint",
        "arm_r1_joint","arm_r2_joint","arm_r3_joint","arm_r4_joint","arm_r5_joint",
    };

    // Extra real joints not controlled by policy (held at current pos)
    static constexpr const char* EXTRA_JOINTS[] = {
        "arm_l6_joint","arm_l7_joint","arm_r6_joint","arm_r7_joint"
    };
    static constexpr int NUM_EXTRA = 4;

    // Name→index lookup for all real robot joints
    std::map<std::string, int> _jointNameToIdx;

    // ═════════════════════════════════════════════════════
    //  ROS 2 interfaces
    // ═════════════════════════════════════════════════════
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr _jointStateSub;
    rclcpp::Subscription<sensor_msgs::msg::Imu>::SharedPtr        _imuSub;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr    _cmdVelSub;
    rclcpp::Publisher<crb_ros_msg::msg::UpperJointData>::SharedPtr _jointCmdPub;
    rclcpp::TimerBase::SharedPtr                                   _policyTimer;
};

}  // namespace casbot_ros2

#endif  // CASBOT_AMP_NODE_HPP
