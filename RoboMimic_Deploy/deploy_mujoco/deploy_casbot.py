#!/usr/bin/env python3
"""
deploy_casbot.py — MuJoCo deployment for Casbot Skeleton (25-DOF)
with CasbotAMP ONNX locomotion policy (Xbox Joystick).

Single-policy deployment (no FSM). Loads the Casbot Skeleton MJCF model,
runs the AMP locomotion policy, and supports Xbox joystick control.

Usage:
    python deploy_mujoco/deploy_casbot.py

Xbox Joystick Controls:
    Left Stick Y     — Forward / Backward
    Left Stick X     — Lateral left / right
    Right Stick X    — Yaw rotation
    R2 (hold)        — HIGH speed mode
    START            — Reset robot pose
    SELECT           — Exit
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import time
import mujoco
import mujoco.viewer
import numpy as np
import os

from policy.casbot_amp.CasbotAMP import CasbotAMP
from common.joystick import JoyStick, JoystickButton


# ══════════════════════════════════════════════════════════════════════
#  PD controller
# ══════════════════════════════════════════════════════════════════════

def pd_control(target_q, q, kp, target_dq, dq, kd, tau_limit):
    """Compute joint torques from position targets, clamped to effort limits."""
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    return np.clip(tau, -tau_limit, tau_limit)


# ══════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    # ── Paths ──
    xml_path = str(PROJECT_ROOT / "casbot_skeleton" / "scene.xml")
    print(f"[Deploy] Loading model: {xml_path}")

    # ── Simulation parameters (same as MJAMP MuJoCo deploy) ──
    simulation_dt = 0.003
    control_decimation = 7  # policy runs at ~48 Hz

    # ── Load MuJoCo model ──
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    num_joints = m.nu  # number of actuators (25)
    print(f"[Deploy] Model loaded: {num_joints} actuators, dt={simulation_dt}")

    # ── Initialize policy ──
    policy = CasbotAMP()

    # ── Runtime buffers ──
    policy_actions = np.zeros(num_joints, dtype=np.float32)
    kps = policy.kps.copy()
    kds = policy.kds.copy()
    sim_counter = 0

    # ── Joystick ──
    try:
        joystick = JoyStick()
        print(f"[Deploy] Joystick connected: {joystick.joystick.get_name()}")
        print(f"  Buttons: {joystick.button_count}, Axes: {joystick.axis_count}")
    except RuntimeError as e:
        print(f"[Deploy] ERROR: {e}")
        print("[Deploy] Please connect an Xbox controller and try again.")
        sys.exit(1)

    running = True

    # ── Initialization phase ──
    base_quat = d.qpos[3:7].copy()  # [qw, qx, qy, qz]
    ang_vel = d.qvel[3:6].copy()
    qj = d.qpos[7:].copy()
    dqj = d.qvel[6:].copy()

    policy.init_buffers(base_quat, ang_vel, qj, dqj)
    policy_actions = policy._target_pos.copy()

    print("[Deploy] Initialization complete. Starting simulation...")
    print("  Controls:")
    print("    Left Stick   — Move forward/back, lateral left/right")
    print("    Right Stick  — Yaw rotation")
    print("    R2 (hold)    — HIGH speed mode")
    print("    START        — Reset robot pose")
    print("    SELECT       — Exit")

    # ── Main simulation loop ──
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and running:
            step_start = time.time()

            # ── Joystick input ──
            joystick.update()

            # Exit: SELECT button
            if joystick.is_button_pressed(JoystickButton.SELECT):
                print("[Deploy] SELECT pressed — exiting")
                running = False

            # Reset pose: START button
            if joystick.is_button_released(JoystickButton.START):
                print("[Deploy] START pressed — resetting pose")
                d.qpos[7:] = policy.default_dof_pos.copy()
                d.qvel[6:] = 0.0
                policy.reset()
                base_quat = d.qpos[3:7].copy()
                ang_vel = d.qvel[3:6].copy()
                qj = d.qpos[7:].copy()
                dqj = d.qvel[6:].copy()
                policy.init_buffers(base_quat, ang_vel, qj, dqj)
                policy_actions = policy._target_pos.copy()

            # Speed mode: R2 (RT) held → high speed
            # On Xbox controllers, RT is axis 5 (value ~1.0 when pressed)
            r2_axis = joystick.get_axis_value(5)  # RT / R2 axis
            high_speed = r2_axis > 0.3
            if high_speed != policy.high_speed_mode:
                policy.high_speed_mode = high_speed
                mode_str = "HIGH" if high_speed else "LOW"
                print(f"[Deploy] Speed mode: {mode_str}")

            # ── Velocity commands (same convention as deploy_mujoco.py G1) ──
            ly_raw = -joystick.get_axis_value(1)   # left stick Y, inverted
            lx_raw = -joystick.get_axis_value(0)   # left stick X, inverted
            rx_raw = -joystick.get_axis_value(3)   # right stick X, inverted

            cmd_vel = policy.get_user_cmd(ly_raw, lx_raw, rx_raw)

            # ── PD control & physics step ──
            tau = pd_control(
                policy_actions,
                d.qpos[7:],
                kps,
                np.zeros_like(kps),
                d.qvel[6:],
                kds,
                policy.tau_limit,
            )
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)

            # ── Policy inference (decimated) ──
            sim_counter += 1
            if sim_counter % control_decimation == 0:
                base_quat = d.qpos[3:7].copy()
                ang_vel = d.qvel[3:6].copy()
                qj = d.qpos[7:].copy()
                dqj = d.qvel[6:].copy()

                result = policy.step(base_quat, ang_vel, cmd_vel, qj, dqj)

                policy_actions = result["actions"].copy()
                kps = result["kps"].copy()
                kds = result["kds"].copy()

                if result["terminated"]:
                    print("[Deploy] SAFETY: anchor gravity threshold exceeded!")

            # ── Sync ──
            viewer.sync()
            time_until_next = m.opt.timestep - (time.time() - step_start)
            if time_until_next > 0:
                time.sleep(time_until_next)

    print("[Deploy] Simulation ended.")
