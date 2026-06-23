#include "control/ControlFrame.h"
#include <iostream>
#include <chrono>
#include <thread>

ControlFrame::ControlFrame(CtrlComponents *ctrlComp) : _ctrlComp(ctrlComp) {
    _fsm = new FSM(ctrlComp);
}

void ControlFrame::run() {
    std::cout << "[ControlFrame] Starting control loop @ "
              << (1.0 / _ctrlComp->dt) << " Hz" << std::endl;

    while (_ctrlComp->running && !_ctrlComp->exitFlag) {
        auto t0 = std::chrono::steady_clock::now();

        // 1. Exchange data with robot/simulator via DDS
        _ctrlComp->sendRecv();

        // 2. Run FSM (policy inference)
        _fsm->run();

        // 3. Timing
        auto t1 = std::chrono::steady_clock::now();
        double elapsed = std::chrono::duration<double>(t1 - t0).count();
        double sleepTime = _ctrlComp->dt - elapsed;
        if (sleepTime > 0)
            std::this_thread::sleep_for(std::chrono::duration<double>(sleepTime));
    }
    std::cout << "[ControlFrame] Exiting" << std::endl;
}
