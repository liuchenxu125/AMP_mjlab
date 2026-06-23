cmake_minimum_required(VERSION 3.14)
project(casbot_amp_deploy)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ═══════════════════════════════════════════════════════════════
#  All dependencies use LOCAL paths (no Python/conda required)
# ═══════════════════════════════════════════════════════════════

set(WBC_DIR "${PROJECT_SOURCE_DIR}/..")  # wbc_fsm/

# ── MuJoCo (bundled) ──
set(MUJOCO_ROOT      "${WBC_DIR}/mujoco")
set(MUJOCO_INCLUDE    "${MUJOCO_ROOT}/include")
set(MUJOCO_LIB        "${MUJOCO_ROOT}/lib/libmujoco.so.3.9.0")

# ── ONNX Runtime (bundled) ──
set(ONNXRUNTIME_ROOT  "${WBC_DIR}/onnxruntime-linux-x64-1.22.0")
set(ONNXRUNTIME_INCLUDE "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB    "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0")

# ── Eigen3 (system) ──
find_package(Eigen3 REQUIRED)

# ── nlohmann_json (system) ──
find_package(nlohmann_json 3.7.3 REQUIRED)

# ── GLFW (system) ──
find_package(glfw3 REQUIRED)

# ═══════════════════════════════════════════════════════════════
#  Include directories
# ═══════════════════════════════════════════════════════════════

include_directories(
    include
    ${MUJOCO_INCLUDE}
    ${ONNXRUNTIME_INCLUDE}
    ${EIGEN3_INCLUDE_DIR}
)

# ═══════════════════════════════════════════════════════════════
#  Source files
# ═══════════════════════════════════════════════════════════════

set(SRC_LIST
    src/main.cpp
    src/CasbotAmpDeploy.cpp
)

# ═══════════════════════════════════════════════════════════════
#  Executable
# ═══════════════════════════════════════════════════════════════

add_executable(casbot_amp_deploy ${SRC_LIST})

target_compile_definitions(casbot_amp_deploy PRIVATE
    PROJECT_ROOT_DIR="${PROJECT_SOURCE_DIR}/.."
)

target_link_libraries(casbot_amp_deploy PRIVATE
    ${MUJOCO_LIB}
    ${ONNXRUNTIME_LIB}
    Eigen3::Eigen
    nlohmann_json::nlohmann_json
    glfw
    pthread
    dl
)

# ═══════════════════════════════════════════════════════════════
#  RPATH: 可执行文件在同目录找 .so (方便拷贝给同事)
# ═══════════════════════════════════════════════════════════════

set_target_properties(casbot_amp_deploy PROPERTIES
    BUILD_RPATH "$ORIGIN"
    INSTALL_RPATH "$ORIGIN"
)

# ═══════════════════════════════════════════════════════════════
#  构建后自动把 .so 拷贝到可执行文件旁边
# ═══════════════════════════════════════════════════════════════

add_custom_command(TARGET casbot_amp_deploy POST_BUILD
    # MuJoCo
    COMMAND ${CMAKE_COMMAND} -E copy ${MUJOCO_ROOT}/lib/libmujoco.so.3.9.0 $<TARGET_FILE_DIR:casbot_amp_deploy>/
    # ONNX Runtime
    COMMAND ${CMAKE_COMMAND} -E copy ${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0 $<TARGET_FILE_DIR:casbot_amp_deploy>/
    COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.so.1.22.0 $<TARGET_FILE_DIR:casbot_amp_deploy>/libonnxruntime.so.1
    COMMAND ${CMAKE_COMMAND} -E create_symlink libonnxruntime.so.1 $<TARGET_FILE_DIR:casbot_amp_deploy>/libonnxruntime.so
    COMMENT "Copying .so files to build dir"
)
