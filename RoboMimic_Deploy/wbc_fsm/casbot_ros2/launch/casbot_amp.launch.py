"""Launch file for Casbot AMP real-robot deployment."""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_dir = get_package_share_directory('casbot_ros2')
    config_path = os.path.join(pkg_dir, 'config', 'casbot_amp.yaml')

    return LaunchDescription([
        Node(
            package='casbot_ros2',
            executable='casbot_amp_node',
            name='casbot_amp_node',
            parameters=[{'config_path': config_path}],
            output='screen',
        ),
    ])
