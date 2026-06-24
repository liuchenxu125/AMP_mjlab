/**
 * @file main.cpp — Casbot AMP ROS 2 real-robot deployment entry point.
 *
 * Usage:
 *   ros2 run casbot_ros2 casbot_amp_node --ros-args -p config_path:=/path/to/config.yaml
 */

#include <rclcpp/rclcpp.hpp>
#include "casbot_ros2/casbot_amp_node.hpp"

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);

    auto node = std::make_shared<rclcpp::Node>("casbot_amp_runner");
    std::string configPath;
    node->declare_parameter("config_path", "");
    node->get_parameter("config_path", configPath);

    if (configPath.empty()) {
        RCLCPP_ERROR(node->get_logger(), "No config_path provided! Use --ros-args -p config_path:=...");
        rclcpp::shutdown();
        return 1;
    }

    auto ampNode = std::make_shared<casbot_ros2::CasbotAmpNode>(configPath);
    rclcpp::spin(ampNode);
    rclcpp::shutdown();
    return 0;
}
