#!/usr/bin/env python3
"""Casbot MuJoCo DDS Simulator — standalone process for DDS architecture.

Architecture (same as G1 unitree_mujoco):
  ┌─ SimulationThread ─┐    ┌─ ViewerThread ─┐
  │ mj_step() @ 200Hz   │    │ viewer.sync()   │
  │ DDS bridge           │    │ @ 50Hz          │
  │  ← LowCmd (PD ctrl)  │    └────────────────┘
  │  → LowState (state)  │
  └─────────────────────┘

Usage:
    python casbot_mujoco_sim.py
"""

import time
import mujoco
import mujoco.viewer
from threading import Thread, Lock
import threading
import os
import sys

# Add project root for config import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from dds_bridge import CasbotDdsBridge

from unitree_sdk2py.core.channel import ChannelFactoryInitialize

locker = Lock()

# ── Load MuJoCo model ──
print(f"[Sim] Loading: {config.ROBOT_SCENE}")
mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)
mj_model.opt.timestep = config.SIMULATE_DT

print(f"[Sim] Model: {mj_model.nu} actuators, timestep={config.SIMULATE_DT}s")
time.sleep(0.2)


def SimulationThread():
    """Physics loop: mj_step() at SIMULATE_DT rate, with DDS bridge."""
    global mj_model, mj_data

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    bridge = CasbotDdsBridge(mj_model, mj_data)

    if config.USE_JOYSTICK:
        bridge.setupJoystick(device_id=0, js_type=config.JOYSTICK_TYPE)

    print("[Sim] Simulation thread running...")
    while viewer.is_running():
        step_start = time.perf_counter()

        locker.acquire()
        mujoco.mj_step(mj_model, mj_data)
        locker.release()

        elapsed = time.perf_counter() - step_start
        if elapsed < mj_model.opt.timestep:
            time.sleep(mj_model.opt.timestep - elapsed)


def ViewerThread():
    """Rendering loop: viewer.sync() at VIEWER_DT rate."""
    print("[Sim] Viewer thread running...")
    while viewer.is_running():
        locker.acquire()
        viewer.sync()
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)

    sim_thread   = Thread(target=SimulationThread, daemon=True)
    view_thread  = Thread(target=ViewerThread, daemon=True)

    sim_thread.start()
    view_thread.start()

    print("=" * 60)
    print("Casbot DDS Simulator — waiting for controller...")
    print(f"  DDS domain={config.DOMAIN_ID} interface={config.INTERFACE}")
    print(f"  Physics: {1.0/config.SIMULATE_DT:.0f} Hz")
    print(f"  Subscribing: {config.TOPIC_LOWCMD if hasattr(config, 'TOPIC_LOWCMD') else 'rt/lowcmd'}")
    print(f"  Publishing:  rt/lowstate")
    print("=" * 60)

    # Keep main thread alive
    try:
        while viewer.is_running():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\n[Sim] Shutting down...")
