/**
 * @file main.cpp
 * @brief MuJoCo + GLFW deployment for Casbot Skeleton (25-DOF).
 *
 * Loads the casbot scene, runs AMP ONNX policy, renders via MuJoCo/GLFW,
 * and supports Xbox gamepad control via GLFW's gamepad API.
 *
 * Controls (Xbox):
 *   Left Stick      — Move (forward/back, lateral)
 *   Right Stick X   — Yaw rotation
 *   RT (R2)         — HIGH speed mode (hold)
 *   START           — Reset robot pose
 *   BACK/SELECT     — Exit
 *
 * Build:
 *   mkdir build && cd build && cmake .. && make -j
 *   ./casbot_amp_deploy [scene.xml] [config.json]
 */

#include "CasbotAmpDeploy.h"

#include <mujoco/mujoco.h>
#include <GLFW/glfw3.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <thread>

// ═══════════════════════════════════════════════════════════════
//  Global state for the render loop
// ═══════════════════════════════════════════════════════════════

static mjModel    *g_m = nullptr;
static mjData     *g_d = nullptr;
static mjvCamera   g_cam;
static mjvOption   g_opt;
static mjvScene    g_scn;
static mjrContext  g_con;
static GLFWwindow *g_window = nullptr;

static bool g_running       = true;
static bool g_startPressed  = false;
static bool g_selectPressed = false;

// ── Mouse state ───────────────────────────────────────────────
static double g_lastMouseX = 0.0;
static double g_lastMouseY = 0.0;
static bool   g_buttonLeft   = false;
static bool   g_buttonRight  = false;
static bool   g_buttonMiddle = false;

// ── PD control ────────────────────────────────────────────────
static void pdControl(const std::array<float, CASBOT_NUM_DOF> &targetQ,
                      const std::array<float, CASBOT_NUM_DOF> &kps,
                      const std::array<float, CASBOT_NUM_DOF> &kds,
                      const std::array<float, CASBOT_NUM_DOF> &tauLimit)
{
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        int jid = g_m->actuator_trnid[2 * i];
        int qadr = g_m->jnt_qposadr[jid];
        int vadr = g_m->jnt_dofadr[jid];
        double q  = g_d->qpos[qadr];
        double dq = g_d->qvel[vadr];
        double tau = (targetQ[i] - q) * kps[i] + (0.0 - dq) * kds[i];
        g_d->ctrl[i] = std::clamp(tau, -double(tauLimit[i]), double(tauLimit[i]));
    }
}

// ── Read robot state from MuJoCo ──────────────────────────────
static void readState(std::array<float, 4> &baseQuat,
                      std::array<float, 3> &angVel,
                      std::array<float, CASBOT_NUM_DOF> &q,
                      std::array<float, CASBOT_NUM_DOF> &dq)
{
    // Base quaternion: qpos[3..6] (free joint: 3 pos + 4 quat)
    baseQuat[0] = g_d->qpos[3]; baseQuat[1] = g_d->qpos[4];
    baseQuat[2] = g_d->qpos[5]; baseQuat[3] = g_d->qpos[6];
    // Base angular velocity: qvel[3..5] (free joint: 3 lin + 3 ang)
    angVel[0] = g_d->qvel[3]; angVel[1] = g_d->qvel[4]; angVel[2] = g_d->qvel[5];
    // Joint positions & velocities — use MuJoCo address API
    for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
        int jid = g_m->actuator_trnid[2 * i];  // joint ID for actuator i
        int qadr = g_m->jnt_qposadr[jid];
        int vadr = g_m->jnt_dofadr[jid];
        q[i]  = g_d->qpos[qadr];
        dq[i] = g_d->qvel[vadr];
    }
}

// ── Keyboard callback ─────────────────────────────────────────
static void keyboardCallback(GLFWwindow *w, int key, int /*scancode*/, int act, int /*mods*/)
{
    if (act == GLFW_PRESS && key == GLFW_KEY_ESCAPE) {
        g_running = false;
    }
}

// ── Window resize ─────────────────────────────────────────────
static void framebufferSizeCallback(GLFWwindow *w, int width, int height) {
    mjrRect viewport = {0, 0, width, height};
    mjv_updateScene(g_m, g_d, &g_opt, nullptr, &g_cam, mjCAT_ALL, &g_scn);
}

// ── Mouse button callback ─────────────────────────────────────
static void mouseButtonCallback(GLFWwindow *w, int button, int action, int /*mods*/)
{
    bool pressed = (action == GLFW_PRESS);
    if (button == GLFW_MOUSE_BUTTON_LEFT)   g_buttonLeft   = pressed;
    if (button == GLFW_MOUSE_BUTTON_RIGHT)  g_buttonRight  = pressed;
    if (button == GLFW_MOUSE_BUTTON_MIDDLE) g_buttonMiddle = pressed;

    // Record cursor position on press
    glfwGetCursorPos(w, &g_lastMouseX, &g_lastMouseY);
}

// ── Scroll callback → zoom ────────────────────────────────────
static void scrollCallback(GLFWwindow * /*w*/, double /*xoffset*/, double yoffset)
{
    g_cam.distance *= (1.0 - yoffset * 0.1);
    if (g_cam.distance < 0.5)  g_cam.distance = 0.5;
    if (g_cam.distance > 20.0) g_cam.distance = 20.0;
}

// ── Cursor position callback → orbit / pan ────────────────────
static void cursorPosCallback(GLFWwindow *w, double xpos, double ypos)
{
    double dx = xpos - g_lastMouseX;
    double dy = ypos - g_lastMouseY;
    g_lastMouseX = xpos;
    g_lastMouseY = ypos;

    // No button held → nothing to do
    if (!g_buttonLeft && !g_buttonRight && !g_buttonMiddle) return;

    // ── Left / Middle: orbit ──
    if (g_buttonLeft || g_buttonMiddle) {
        g_cam.azimuth   += dx * 0.3;
        g_cam.elevation -= dy * 0.3;
        // Clamp elevation to avoid gimbal lock
        if (g_cam.elevation >  89.0) g_cam.elevation =  89.0;
        if (g_cam.elevation < -89.0) g_cam.elevation = -89.0;
    }

    // ── Right: pan ──
    if (g_buttonRight) {
        // Camera right & up vectors (approximate, matching MuJoCo viewer)
        double azimuth_rad   = g_cam.azimuth   * M_PI / 180.0;
        double elevation_rad = g_cam.elevation * M_PI / 180.0;

        double ca = cos(azimuth_rad), sa = sin(azimuth_rad);
        double ce = cos(elevation_rad), se = sin(elevation_rad);

        // Camera right  = cross(forward, world_up) ≈ [ca, sa, 0]    (rotated in azimuth)
        // Camera up     ≈ [-sa * se, ca * se, ce]   (rotated in elevation)
        double scale = g_cam.distance * 0.001;
        g_cam.lookat[0] += scale * (-ca * dx - sa * se * dy);
        g_cam.lookat[1] += scale * (-sa * dx + ca * se * dy);
        g_cam.lookat[2] += scale * (          ce * dy);
    }
}

// ═══════════════════════════════════════════════════════════════
//  Main
// ═══════════════════════════════════════════════════════════════

int main(int argc, char **argv)
{
    // ── Paths ──
    const char *projectRoot = PROJECT_ROOT_DIR;
    char scenePath[512], configPath[512];
    snprintf(scenePath,  sizeof(scenePath),  "%s/../casbot_skeleton/scene.xml", projectRoot);
    snprintf(configPath, sizeof(configPath), "%s/casbot_deploy/config/casbot_amp.json", projectRoot);

    if (argc >= 2) snprintf(scenePath,  sizeof(scenePath),  "%s", argv[1]);
    if (argc >= 3) snprintf(configPath, sizeof(configPath), "%s", argv[2]);

    std::cout << "[Main] Scene:  " << scenePath  << "\n"
              << "[Main] Config: " << configPath << std::endl;

    // ── Simulation parameters ──
    const double simDt     = 0.003;
    const int    ctrlDecim = 7;           // policy ~48 Hz
    const double ctrlDt    = simDt * ctrlDecim;

    // ── Load MuJoCo model ──
    char error[1000] = "";
    g_m = mj_loadXML(scenePath, nullptr, error, sizeof(error));
    if (!g_m) {
        std::cerr << "[Main] MuJoCo XML error: " << error << std::endl;
        return 1;
    }
    g_d = mj_makeData(g_m);
    g_m->opt.timestep = simDt;

    std::cout << "[Main] MuJoCo: " << g_m->nu << " actuators, "
              << g_m->nq << " qpos" << std::endl;

    // ── Init policy ──
    CasbotAmpDeploy policy(configPath);
    std::array<float, CASBOT_NUM_DOF> policyActions{};
    std::array<float, CASBOT_NUM_DOF> kps = policy.kps();
    std::array<float, CASBOT_NUM_DOF> kds = policy.kds();

    // ── Fill observation buffer ──
    std::array<float, 4> baseQuat{};
    std::array<float, 3> angVel{};
    std::array<float, CASBOT_NUM_DOF> qj{}, dqj{};
    readState(baseQuat, angVel, qj, dqj);
    policy.initBuffers(baseQuat, angVel, qj, dqj);
    policyActions = policy.targetPos();

    // ── Init GLFW ──
    if (!glfwInit()) {
        std::cerr << "[Main] GLFW init failed" << std::endl;
        mj_deleteData(g_d); mj_deleteModel(g_m);
        return 1;
    }

    // Create window
    g_window = glfwCreateWindow(1200, 900, "Casbot Skeleton — AMP Deploy", nullptr, nullptr);
    if (!g_window) {
        std::cerr << "[Main] GLFW window creation failed" << std::endl;
        glfwTerminate();
        mj_deleteData(g_d); mj_deleteModel(g_m);
        return 1;
    }
    glfwMakeContextCurrent(g_window);
    glfwSetKeyCallback(g_window, keyboardCallback);
    glfwSetFramebufferSizeCallback(g_window, framebufferSizeCallback);
    glfwSetMouseButtonCallback(g_window, mouseButtonCallback);
    glfwSetCursorPosCallback(g_window, cursorPosCallback);
    glfwSetScrollCallback(g_window, scrollCallback);
    glfwSwapInterval(1);

    // ── Init MuJoCo visualization ──
    mjv_defaultCamera(&g_cam);
    mjv_defaultOption(&g_opt);
    mjv_makeScene(g_m, &g_scn, 2000);
    mjr_defaultContext(&g_con);
    mjr_makeContext(g_m, &g_con, 200);

    // Position camera
    g_cam.distance = 3.0;
    g_cam.azimuth  = 140.0;
    g_cam.elevation = -20.0;
    g_cam.lookat[0] = 0.5;
    g_cam.lookat[1] = 0.0;
    g_cam.lookat[2] = 0.8;

    // ── Time tracking ──
    auto lastCtrlTime = std::chrono::steady_clock::now();
    GLFWgamepadstate gamepadState;
    bool startPrev = false, backPrev = false;

    std::cout << "[Main] Running. Controls:\n"
              << "  Left Stick=Move  Right Stick X=Rotate  RT=HighSpeed\n"
              << "  START=Reset  BACK=Exit  ESC=Exit\n";

    // ═══════════════════════════════════════════════════════════
    //  Main loop
    // ═══════════════════════════════════════════════════════════
    while (g_running && !glfwWindowShouldClose(g_window)) {
        auto frameStart = std::chrono::steady_clock::now();

        // ── GLFW events ──
        glfwPollEvents();

        // ── Gamepad ──
        float ly = 0.0f, lx = 0.0f, rx = 0.0f;
        bool rtHigh = false, startNow = false, backNow = false;

        if (glfwJoystickPresent(GLFW_JOYSTICK_1)) {
            if (glfwGetGamepadState(GLFW_JOYSTICK_1, &gamepadState)) {
                ly = -gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_Y];
                lx = -gamepadState.axes[GLFW_GAMEPAD_AXIS_LEFT_X];
                rx = -gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_X];
                rtHigh   = (gamepadState.axes[GLFW_GAMEPAD_AXIS_RIGHT_TRIGGER] > 0.3f);
                startNow = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_START];
                backNow  = gamepadState.buttons[GLFW_GAMEPAD_BUTTON_BACK];
            }
        }

        // ── Back → exit ──
        if (backNow && !backPrev) {
            std::cout << "[Main] BACK → exit" << std::endl;
            g_running = false;
        }
        backPrev = backNow;

        // ── START → reset pose ──
        if (startNow && !startPrev) {
            std::cout << "[Main] START → reset pose" << std::endl;
            for (int i = 0; i < CASBOT_NUM_DOF; ++i) {
                g_d->qpos[7 + i] = policy.defaultDofPos()[i];
                g_d->qvel[6 + i] = 0.0;
            }
            policy.reset();
            readState(baseQuat, angVel, qj, dqj);
            policy.initBuffers(baseQuat, angVel, qj, dqj);
            policyActions = policy.targetPos();
        }
        startPrev = startNow;

        // ── Speed toggle ──
        if (rtHigh != policy.highSpeedMode()) {
            policy.setHighSpeedMode(rtHigh);
            std::cout << "[Main] Speed: " << (rtHigh ? "HIGH" : "LOW") << std::endl;
        }

        // ── Policy step (decimated) ──
        auto now = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(now - lastCtrlTime).count();

        if (elapsed >= ctrlDt) {
            auto cmdVel = policy.getUserCmd(ly, lx, rx);
            std::array<float, 3> cv = {cmdVel[0], cmdVel[1], cmdVel[2]};

            readState(baseQuat, angVel, qj, dqj);
            auto result = policy.step(baseQuat, angVel, cv, qj, dqj);
            policyActions = result.actions;
            kps = result.kps;
            kds = result.kds;
            lastCtrlTime = now;
        }

        // ── PD control + physics step ──
        pdControl(policyActions, kps, kds, policy.tauLimit());
        mj_step(g_m, g_d);

        // ── Render ──
        int fbWidth, fbHeight;
        glfwGetFramebufferSize(g_window, &fbWidth, &fbHeight);
        mjrRect viewport = {0, 0, fbWidth, fbHeight};

        mjv_updateScene(g_m, g_d, &g_opt, nullptr, &g_cam, mjCAT_ALL, &g_scn);
        mjr_render(viewport, &g_scn, &g_con);
        glfwSwapBuffers(g_window);

        // ── Frame timing ──
        auto frameEnd = std::chrono::steady_clock::now();
        double frameElapsed = std::chrono::duration<double>(frameEnd - frameStart).count();
        if (frameElapsed < simDt) {
            std::this_thread::sleep_for(std::chrono::duration<double>(simDt - frameElapsed));
        }
    }

    // ── Clean up ──
    mjr_freeContext(&g_con);
    mjv_freeScene(&g_scn);
    glfwDestroyWindow(g_window);
    glfwTerminate();
    mj_deleteData(g_d);
    mj_deleteModel(g_m);

    std::cout << "[Main] Done." << std::endl;
    return 0;
}
