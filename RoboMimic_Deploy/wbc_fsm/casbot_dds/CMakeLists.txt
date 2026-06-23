cmake_minimum_required(VERSION 3.14)
project(casbot_dds)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════════════════════════
#  Dependencies
# ═══════════════════════════════════════════════════════════════

# Unitree SDK2 (for DDS communication)
list(APPEND CMAKE_PREFIX_PATH "/opt/unitree_robotics")
find_package(unitree_sdk2 REQUIRED)

# ONNX Runtime (bundled)
set(ONNXRUNTIME_ROOT "${PROJECT_SOURCE_DIR}/../onnxruntime-linux-x64-1.22.0")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB     "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0")

# Eigen3
find_package(Eigen3 REQUIRED)

# nlohmann_json
find_package(nlohmann_json 3.7.3 REQUIRED)

# Boost (required by unitree_sdk2)
find_package(Boost REQUIRED)

# ═══════════════════════════════════════════════════════════════
#  Include directories
# ═══════════════════════════════════════════════════════════════

include_directories(
    include
    ${ONNXRUNTIME_INCLUDE}
    ${EIGEN3_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
)

# ═══════════════════════════════════════════════════════════════
#  Source files
# ═══════════════════════════════════════════════════════════════

file(GLOB SRC_LIST
    "src/*.cpp"
    "src/*/*.cpp"
)

# ═══════════════════════════════════════════════════════════════
#  Executable
# ═══════════════════════════════════════════════════════════════

add_executable(casbot_dds ${SRC_LIST})

target_compile_definitions(casbot_dds PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}/.."
)

target_link_libraries(casbot_dds PRIVATE
    unitree_sdk2
    ${ONNXRUNTIME_LIB}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    ${Boost_LIBRARIES}
    pthread
)

# Copy ONNX Runtime .so to build dir
add_custom_command(TARGET casbot_dds POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_LIB} $<TARGET_FILE_DIR:casbot_dds>/
    COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.so.1.22.0 $<TARGET_FILE_DIR:casbot_dds>/libonnxruntime.so.1
    COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.so.1 $<TARGET_FILE_DIR:casbot_dds>/libonnxruntime.so
    COMMENT "Copying libonnxruntime.so to build dir"
)
