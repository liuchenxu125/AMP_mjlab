#!/usr/bin/env python3
"""
deploy_casbot.py — MuJoCo deployment for Casbot Skeleton (25-DOF)
with CasbotAMP ONNX locomotion policy (Xbox Joystick).

Single-policy deployment (no FSM). Loads the Casbot Skeleton MJCF model,
runs the AMP locomotion policy, and supports Xbox joystick control.

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


def pd_control(target_q, q, kp, target_dq, dq, kd, tau_limit):
    """Compute joint torques from position targets, clamped to effort limits."""
    tau = (target_q - q) * kp + (target_dq - dq) * kd
    return np.clip(tau, -tau_limit, tau_limit)


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.absolute()

    # ── Paths ──
    xml_path = str(PROJECT_ROOT / "casbot_skeleton" / "scene.xml")
    print(f"[Deploy] Loading model: {xml_path}")

    # ── Simulation parameters ──
    simulation_dt = 0.002
    control_decimation = 10  # policy @ 50 Hz (500Hz / 10)

    # ── Load MuJoCo model ──
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt
    num_joints = m.nu
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
    except RuntimeError as e:
        print(f"[Deploy] ERROR: {e}")
        sys.exit(1)

    running = True

    # ── Initialization ──
    base_quat = d.qpos[3:7].copy()
    ang_vel = d.qvel[3:6].copy()
    qj = d.qpos[7:].copy()
    dqj = d.qvel[6:].copy()
    policy.init_buffers(base_quat, ang_vel, qj, dqj)
    policy_actions = policy._target_pos.copy()

    print("[Deploy] Starting simulation...")
    print("  Controls:")
    print("    Left Stick   — Move forward/back, lateral left/right")
    print("    Right Stick  — Yaw rotation")
    print("    R2 (hold)    — HIGH speed mode")
    print("    START        — Reset robot pose")
    print("    SELECT       — Exit")

    # ══════════════════════════════════════════════════════════
    #  Main loop
    # ══════════════════════════════════════════════════════════
    # Timing profiler
    t_joystick_total = 0.0; t_pd_total = 0.0; t_infer_total = 0.0
    t_sync_total = 0.0; t_sleep_total = 0.0; t_frame_total = 0.0
    t_frame_max = 0.0; infer_count = 0
    profile_interval = 50
    log_slow_frames = True

    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and running:
            step_start = time.perf_counter()

            # ── Joystick input ──
            t0 = time.perf_counter()
            joystick.update()

            if joystick.is_button_pressed(JoystickButton.SELECT):
                print("[Deploy] SELECT pressed — exiting")
                running = False

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

            r2_axis = joystick.get_axis_value(5)
            high_speed = r2_axis > 0.3
            if high_speed != policy.high_speed_mode:
                policy.high_speed_mode = high_speed
                print(f"[Deploy] Speed mode: {'HIGH' if high_speed else 'LOW'}")

            ly_raw = -joystick.get_axis_value(1)
            lx_raw = -joystick.get_axis_value(0)
            rx_raw = -joystick.get_axis_value(3)
            cmd_vel = policy.get_user_cmd(ly_raw, lx_raw, rx_raw)
            t_joystick = time.perf_counter() - t0

            # ── PD control & physics step ──
            t0 = time.perf_counter()
            tau = pd_control(policy_actions, d.qpos[7:], kps,
                           np.zeros_like(kps), d.qvel[6:], kds, policy.tau_limit)
            d.ctrl[:] = tau
            mujoco.mj_step(m, d)
            t_pd = time.perf_counter() - t0

            # ── Policy inference (decimated) ──
            sim_counter += 1
            t_infer = 0.0
            if sim_counter % control_decimation == 0:
                t0 = time.perf_counter()
                base_quat = d.qpos[3:7].copy()
                ang_vel = d.qvel[3:6].copy()
                qj = d.qpos[7:].copy()
                dqj = d.qvel[6:].copy()
                result = policy.step(base_quat, ang_vel, cmd_vel, qj, dqj)
                policy_actions = result["actions"].copy()
                kps = result["kps"].copy()
                kds = result["kds"].copy()
                t_infer = time.perf_counter() - t0
                infer_count += 1
                if result["terminated"]:
                    print("[Deploy] SAFETY: anchor gravity threshold exceeded!")

            # ── Sync ──
            t0 = time.perf_counter()
            viewer.sync()
            t_sync = time.perf_counter() - t0

            # ── Sleep ──
            time_until_next = m.opt.timestep - (time.perf_counter() - step_start)
            t_sleep = 0.0
            if time_until_next > 0:
                time.sleep(time_until_next)
                t_sleep = time_until_next

            # ── Accumulate timing ──
            frame_time = time.perf_counter() - step_start
            t_joystick_total += t_joystick
            t_pd_total += t_pd
            t_infer_total += t_infer
            t_sync_total += t_sync
            t_sleep_total += t_sleep
            t_frame_total += frame_time
            if frame_time > t_frame_max:
                t_frame_max = frame_time

            if log_slow_frames and frame_time > simulation_dt * 2.5:
                print(f"[TIMING SLOW] frame={sim_counter:5d}  total={frame_time*1000:6.2f}ms "
                      f"(expect {simulation_dt*1000:.1f}ms)  "
                      f"joystick={t_joystick*1000:.2f}ms  pd={t_pd*1000:.2f}ms  "
                      f"infer={t_infer*1000:.2f}ms  sync={t_sync*1000:.2f}ms")

            if sim_counter % profile_interval == 0:
                avg_frame = t_frame_total / profile_interval * 1000
                avg_pd = t_pd_total / profile_interval * 1000
                avg_sync = t_sync_total / profile_interval * 1000
                avg_infer = t_infer_total / max(infer_count, 1) * 1000 if infer_count > 0 else 0
                print(f"[TIMING] frames={sim_counter:5d}  "
                      f"avg_frame={avg_frame:5.2f}ms  avg_pd={avg_pd:5.2f}ms  "
                      f"avg_sync={avg_sync:5.2f}ms  avg_infer={avg_infer:5.2f}ms  "
                      f"max_frame={t_frame_max*1000:5.2f}ms  infer_count={infer_count}")
                t_joystick_total = 0.0; t_pd_total = 0.0; t_infer_total = 0.0
                t_sync_total = 0.0; t_sleep_total = 0.0; t_frame_total = 0.0
                t_frame_max = 0.0; infer_count = 0

    print("[Deploy] Simulation ended.")
