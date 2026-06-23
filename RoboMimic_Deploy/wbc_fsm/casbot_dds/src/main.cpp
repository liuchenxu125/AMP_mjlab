/**
 * @file main.cpp — Casbot DDS Deployment entry point.
 *
 * DDS architecture (same pattern as G1 wbc_fsm):
 *   Controller process ←→ DDS ←→ MuJoCo Simulator / Real Robot
 *
 * Usage:
 *   ./casbot_dds [config.json]
 *
 * CtrlPlatform is selected at compile time:
 *   MUJOCO    — DDS to unitree_mujoco simulator
 *   REALROBOT — DDS to real casbot robot
 */

#include "control/ControlFrame.h"
#include "interface/IOSDK.h"
#include <iostream>
#include <csignal>

static bool g_running = true;

void signalHandler(int) { g_running = false; }

int main(int argc, char **argv) {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    std::cout << "=== Casbot DDS Deployment (25-DOF) ===" << std::endl;

    // ── Create I/O layer ──
    IOInterface *io = new IOSDK();
    CtrlComponents ctrlComp(io);

#ifdef REALROBOT
    ctrlComp.ctrlPlatform = CtrlPlatform::REALROBOT;
    std::cout << "[Main] Platform: REALROBOT" << std::endl;
#else
    ctrlComp.ctrlPlatform = CtrlPlatform::MUJOCO;
    std::cout << "[Main] Platform: MUJOCO (DDS simulator)" << std::endl;
#endif

    ctrlComp.running = &g_running;

    // ── Run control loop ──
    ControlFrame frame(&ctrlComp);
    frame.run();

    std::cout << "[Main] Shutdown complete." << std::endl;
    return 0;
}
