#!/usr/bin/env python3
"""
deploy_casbot.py — MuJoCo deployment for Casbot Skeleton (25-DOF)
with CasbotAMP ONNX locomotion policy.

Single-policy deployment (no FSM). Loads the Casbot Skeleton MJCF model,
runs the AMP locomotion policy, and supports keyboard velocity control.

Usage:
    python deploy_mujoco/deploy_casbot.py

Controls:
    WASD        — Move forward/back/left/right (hold Shift for velocity)
    Q/E         — Rotate left/right (hold Shift for velocity)
    Arrow keys  — Move forward/back/left/right (no Shift needed)
    R           — Reset robot pose
    F           — Toggle high/low speed mode
    ESC         — Exit
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.absolute()))

import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
import os

from policy.casbot_amp.CasbotAMP import CasbotAMP
from pynput import keyboard as pynput_keyboard


# ══════════════════════════════════════════════════════════════════════
#  Keyboard handler
# ══════════════════════════════════════════════════════════════════════

class Keyboard:
    """Keyboard input handler using pynput global listener."""

    def __init__(self):
        self.key_states = {}
        self.key_prev_states = {}
        self.key_pressed_events = {}
        self.key_released_events = {}
        self._listener = None
        self._lock = threading.Lock()

        self.key_map = {
            "W": "w", "A": "a", "S": "s", "D": "d",
            "Q": "q", "E": "e", "R": "r", "F": "f",
            "UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right",
            "LSHIFT": "shift", "RSHIFT": "shift",
            "ESCAPE": "esc",
            "SPACE": "space",
        }
        self._start_listener()

    def _start_listener(self):
        def on_press(key):
            try:
                if hasattr(key, "char") and key.char:
                    k = key.char.lower()
                else:
                    k = str(key).replace("Key.", "").lower()
                with self._lock:
                    self.key_states[k] = True
            except Exception:
                pass

        def on_release(key):
            try:
                if hasattr(key, "char") and key.char:
                    k = key.char.lower()
                else:
                    k = str(key).replace("Key.", "").lower()
                with self._lock:
                    self.key_states[k] = False
            except Exception:
                pass

        self._listener = pynput_keyboard.Listener(
            on_press=on_press, on_release=on_release
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self):
        if self._listener:
            self._listener.stop()

    def update(self):
        with self._lock:
            self.key_pressed_events.clear()
            self.key_released_events.clear()
            for key_name, key_char in self.key_map.items():
                cur = self.key_states.get(key_char, False)
                prev = self.key_prev_states.get(key_char, False)
                if cur and not prev:
                    self.key_pressed_events[key_name] = True
                if not cur and prev:
                    self.key_released_events[key_name] = True
                self.key_prev_states[key_char] = cur

    def is_key_pressed(self, key):
        key_char = self.key_map.get(key, key.lower())
        with self._lock:
            return self.key_states.get(key_char, False)

    def is_key_released(self, key):
        return self.key_released_events.get(key, False)

    def get_axis_from_keys(self, neg_key, pos_key):
        neg = self.is_key_pressed(neg_key)
        pos = self.is_key_pressed(pos_key)
        if neg and pos:
            return 0.0
        elif neg:
            return -1.0
        elif pos:
            return 1.0
        return 0.0


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
    mj_step_duration = simulation_dt * control_decimation

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

    # ── Keyboard ──
    keyboard = Keyboard()
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
    print("    WASD        — Move (hold Shift)")
    print("    Q/E         — Rotate (hold Shift)")
    print("    Arrow keys  — Move (no Shift needed)")
    print("    R           — Reset robot pose")
    print("    F           — Toggle speed mode (high/low)")
    print("    ESC         — Exit")

    # ── Main simulation loop ──
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running() and running:
            step_start = time.time()

            # ── Keyboard input ──
            keyboard.update()

            # Exit
            if keyboard.is_key_pressed("ESCAPE"):
                print("[Deploy] ESC pressed — exiting")
                running = False

            # Reset pose
            if keyboard.is_key_released("R"):
                print("[Deploy] R pressed — resetting pose")
                # Reset to default pose
                d.qpos[7:] = policy.default_dof_pos.copy()
                d.qvel[6:] = 0.0
                policy.reset()
                base_quat = d.qpos[3:7].copy()
                ang_vel = d.qvel[3:6].copy()
                qj = d.qpos[7:].copy()
                dqj = d.qvel[6:].copy()
                policy.init_buffers(base_quat, ang_vel, qj, dqj)
                policy_actions = policy._target_pos.copy()

            # Toggle speed mode
            if keyboard.is_key_released("F"):
                policy.high_speed_mode = not policy.high_speed_mode
                mode_str = "HIGH" if policy.high_speed_mode else "LOW"
                print(f"[Deploy] Speed mode: {mode_str}")

            # ── Velocity commands ──
            ly = 0.0  # forward/back
            lx = 0.0  # left/right
            rx = 0.0  # yaw

            shift_held = keyboard.is_key_pressed("LSHIFT") or keyboard.is_key_pressed(
                "RSHIFT"
            )

            # Shift + WASD/QE
            if shift_held:
                ly = keyboard.get_axis_from_keys("S", "W")
                lx = keyboard.get_axis_from_keys("A", "D")
                rx = keyboard.get_axis_from_keys("Q", "E")

            # Arrow keys (always active, overrides if shift not held)
            arrow_ly = keyboard.get_axis_from_keys("DOWN", "UP")
            arrow_lx = keyboard.get_axis_from_keys("LEFT", "RIGHT")

            if not shift_held:
                if arrow_ly != 0:
                    ly = arrow_ly
                if arrow_lx != 0:
                    lx = -arrow_lx  # left arrow = move left

            cmd_vel = policy.get_user_cmd(ly, lx, rx)

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

    keyboard.stop()
    print("[Deploy] Simulation ended.")
