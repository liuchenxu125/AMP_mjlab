cmake_minimum_required(VERSION 3.14)
project(casbot_ros2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════════════════════════
#  ROS 2
# ═══════════════════════════════════════════════════════════════
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(crb_ros_msg REQUIRED)
find_package(Eigen3 REQUIRED)

# ═══════════════════════════════════════════════════════════════
#  ONNX Runtime (bundled)
# ═══════════════════════════════════════════════════════════════
set(ONNXRUNTIME_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../onnxruntime-linux-x64-1.22.0")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB     "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0")

# ═══════════════════════════════════════════════════════════════
#  CasbotAmpDeploy (reuse standalone lib)
# ═══════════════════════════════════════════════════════════════
set(CASBOT_DEPLOY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../casbot_deploy")

# ═══════════════════════════════════════════════════════════════
#  Include dirs
# ═══════════════════════════════════════════════════════════════
include_directories(
    include
    ${CASBOT_DEPLOY_DIR}/include
    ${ONNXRUNTIME_INCLUDE}
    ${EIGEN3_INCLUDE_DIR}
)

# ═══════════════════════════════════════════════════════════════
#  Sources
# ═══════════════════════════════════════════════════════════════
set(SOURCES
    src/main.cpp
    src/casbot_amp_node.cpp
    ${CASBOT_DEPLOY_DIR}/src/CasbotAmpDeploy.cpp
)

# ═══════════════════════════════════════════════════════════════
#  Executable
# ═══════════════════════════════════════════════════════════════
add_executable(casbot_amp_node ${SOURCES})

ament_target_dependencies(casbot_amp_node
    rclcpp sensor_msgs geometry_msgs std_msgs crb_ros_msg
)

target_link_libraries(casbot_amp_node
    ${ONNXRUNTIME_LIB}
    pthread dl
)

# ═══════════════════════════════════════════════════════════════
#  Install
# ═══════════════════════════════════════════════════════════════
install(TARGETS casbot_amp_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch model DESTINATION share/${PROJECT_NAME})

ament_package()
