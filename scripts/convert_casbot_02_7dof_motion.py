#!/usr/bin/env python3
"""Convert casbot_02_7dof raw motion data (.data CSV) to AMP-training NPZ format.

Input format (61 columns, comma-separated, 1000 Hz):
  Cols  1-3:   base position (m) — x, y, z
  Cols  4-6:   base euler angles (rad) — pitch, yaw, roll (ZYX order)
  Cols  7-9:   base linear velocity (m/s) — x, y, z
  Cols 10-12:  base angular velocity (rad/s) — pitch, yaw, roll
  Cols 13-18:  left leg joints (6) — pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
  Cols 19-24:  right leg joints (6) — same order
  Cols 25-31:  left arm joints (7) — shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw, wrist_pitch, wrist_roll
  Cols 32-38:  right arm joints (7) — same order
  Cols 39-48:  left hand (10) — skipped (passive)
  Cols 49-58:  right hand (10) — skipped (passive)
  Cols 59-60:  head (2) — skipped (passive)
  Col  61:     waist (1) — waist_yaw

Output NPZ (50 Hz):
  - fps: scalar
  - joint_pos: (T, 27) — positions for 27 controlled DOF
  - joint_vel: (T, 27)
  - body_pos_w: (T, N, 3)
  - body_quat_w: (T, N, 4)
  - body_lin_vel_w: (T, N, 3)
  - body_ang_vel_w: (T, N, 3)

Usage:
    python scripts/convert_casbot_02_7dof_motion.py \\
        --input-dir src/assets/motions/casbot_02_7dof/amp/ \\
        --output-dir src/assets/motions/casbot_02_7dof/amp/
"""

from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np


def euler_to_quat(pitch: float, yaw: float, roll: float) -> np.ndarray:
    """Convert ZYX euler angles (pitch, yaw, roll) to wxyz quaternion."""
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z])


# ── 27 controlled joint names (actuator order) ──
CONTROLLED_JOINT_NAMES = [
    # Left leg (6)
    "left_leg_pelvic_pitch_joint", "left_leg_pelvic_roll_joint", "left_leg_pelvic_yaw_joint",
    "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint", "left_leg_ankle_roll_joint",
    # Right leg (6)
    "right_leg_pelvic_pitch_joint", "right_leg_pelvic_roll_joint", "right_leg_pelvic_yaw_joint",
    "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint", "right_leg_ankle_roll_joint",
    # Waist (1)
    "waist_yaw_joint",
    # Head (2)
    "head_yaw_joint", "head_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]

# CSV column indices (0-based) → 29 controlled joints
# Layout: 12 base, 6Lleg, 6Rleg, 7Larm, 7Rarm, 10Lhand, 10Rhand, 2head, 1waist
_CSV_IDX = (
    list(range(12, 18))    # left leg: cols 13-18 → idx 12-17
    + list(range(18, 24))  # right leg: cols 19-24 → idx 18-23
    + [60]                  # waist: col 61 → idx 60
    + list(range(58, 60))  # head: cols 59-60 → idx 58-59
    + list(range(24, 31))  # left arm: cols 25-31 → idx 24-30
    + list(range(31, 38))  # right arm: cols 32-38 → idx 31-37
)

# Base state column indices (0-based)
_BASE_POS_IDX = (0, 1, 2)       # x, y, z (m)
_BASE_EULER_IDX = (3, 4, 5)     # pitch, yaw, roll (rad)
_BASE_LIN_VEL_IDX = (6, 7, 8)   # x, y, z (m/s) — used as reference


def load_data_file(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a .data CSV file and extract base state and joint positions.

    Returns:
        joint_pos: (T, 27) joint positions in rad
        root_pos: (T, 3) base position in world frame (m)
        root_quat: (T, 4) base quaternion wxyz
    """
    raw = np.loadtxt(path, delimiter=",", usecols=range(61))
    T = raw.shape[0]

    # Joint positions from CSV columns → 29 DOF
    joint_pos = np.zeros((T, 29), dtype=np.float32)
    for i, csv_idx in enumerate(_CSV_IDX):
        if csv_idx < raw.shape[1]:
            joint_pos[:, i] = raw[:, csv_idx].astype(np.float32)

    # Root position
    root_pos = np.zeros((T, 3), dtype=np.float32)
    root_pos[:, 0] = raw[:, 0].astype(np.float32)
    root_pos[:, 1] = raw[:, 1].astype(np.float32)
    root_pos[:, 2] = raw[:, 2].astype(np.float32)

    # Root quaternion from euler angles (cols 4-6 = yaw, pitch, roll)
    # The data uses Yaw/Pitch/Roll order, NOT Pitch/Yaw/Roll
    root_quat = np.zeros((T, 4), dtype=np.float32)
    for t in range(T):
        yaw, pitch, roll = float(raw[t, 3]), float(raw[t, 4]), float(raw[t, 5])
        q = euler_to_quat(pitch, yaw, roll)
        root_quat[t] = q.astype(np.float32)

    return joint_pos, root_pos, root_quat


def interpolate_data(data: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    """Linear interpolation from input_fps to output_fps."""
    if input_fps == output_fps:
        return data
    T_in = data.shape[0]
    duration = (T_in - 1) / input_fps
    T_out = int(duration * output_fps) + 1
    t_in = np.arange(T_in) / input_fps
    t_out = np.arange(T_out) / output_fps
    result = np.zeros((T_out,) + data.shape[1:], dtype=data.dtype)
    for d in range(data.shape[1]):
        result[:, d] = np.interp(t_out, t_in, data[:, d])
    return result


def slerp_quat(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two quaternions (wxyz)."""
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    if theta < 1e-6:
        return q0 * (1 - t) + q1 * t
    s0 = np.sin((1 - t) * theta) / np.sin(theta)
    s1 = np.sin(t * theta) / np.sin(theta)
    return s0 * q0 + s1 * q1


def interpolate_quat(quats: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    """SLERP quaternion interpolation."""
    if input_fps == output_fps:
        return quats
    T_in = quats.shape[0]
    duration = (T_in - 1) / input_fps
    T_out = int(duration * output_fps) + 1
    t_in = np.arange(T_in) / input_fps
    t_out = np.arange(T_out) / output_fps
    result = np.zeros((T_out, 4), dtype=quats.dtype)
    for i, t in enumerate(t_out):
        idx = np.searchsorted(t_in, t)
        if idx == 0:
            result[i] = quats[0]
        elif idx >= T_in:
            result[i] = quats[-1]
        else:
            alpha = (t - t_in[idx - 1]) / (t_in[idx] - t_in[idx - 1])
            result[i] = slerp_quat(quats[idx - 1], quats[idx], alpha)
    return result


def compute_velocities(positions: np.ndarray, dt: float) -> np.ndarray:
    """Compute velocities from positions using central difference."""
    T = positions.shape[0]
    vel = np.zeros_like(positions)
    if T < 3:
        return vel
    vel[1:-1] = (positions[2:] - positions[:-2]) / (2 * dt)
    vel[0] = (positions[1] - positions[0]) / dt
    vel[-1] = (positions[-1] - positions[-2]) / dt
    return vel


def convert_single_file(
    input_path: str,
    output_path: str,
    input_fps: float = 1000.0,
    output_fps: float = 50.0,
) -> None:
    """Convert a single .data file to NPZ."""
    print(f"[Convert] {os.path.basename(input_path)}")

    # Load data
    joint_pos, root_pos, root_quat = load_data_file(input_path)

    # Interpolate to output FPS
    joint_pos = interpolate_data(joint_pos, input_fps, output_fps)
    root_pos = interpolate_data(root_pos, input_fps, output_fps)
    root_quat = interpolate_quat(root_quat, input_fps, output_fps)

    T_out = joint_pos.shape[0]
    output_dt = 1.0 / output_fps

    # Compute velocities
    joint_vel = compute_velocities(joint_pos, output_dt)
    root_lin_vel = compute_velocities(root_pos, output_dt)

    # Compute angular velocity from quaternion differences
    root_ang_vel = np.zeros((T_out, 3), dtype=np.float32)
    for t in range(1, T_out - 1):
        q_prev = root_quat[t - 1]
        q_next = root_quat[t + 1]
        # Box-minus approximation
        dq = q_next - q_prev
        # Simplified: project to angular velocity
        w_prev = q_prev[1:]  # imaginary part
        root_ang_vel[t] = (dq[1:]) / (2 * output_dt)

    # ── Compute body positions via raw MuJoCo FK (no rendering deps) ──
    xml_path = str(Path(__file__).resolve().parent.parent / "src" / "assets" / "robots" / "casbot_02_7dof" / "xmls" / "casbot_02_7dof.xml")

    # Collect all body names from XML
    tree = ET.parse(xml_path)
    body_names = [b.get("name", "") for b in tree.getroot().iter("body") if b.get("name")]

    # Create MuJoCo model + data
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)

    # Resolve 27 controlled joint names → qpos indices
    joint_qpos_idx = []
    for jname in CONTROLLED_JOINT_NAMES:
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            raise ValueError(f"Joint '{jname}' not found in XML")
        joint_qpos_idx.append(m.jnt_qposadr[jid])

    # Resolve body names → MuJoCo body indices
    body_mj_ids = []
    for bname in body_names:
        bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, bname)
        body_mj_ids.append(bid)

    # Per-frame FK
    body_pos_w = np.zeros((T_out, len(body_names), 3), dtype=np.float32)
    body_quat_w = np.zeros((T_out, len(body_names), 4), dtype=np.float32)

    for t in range(T_out):
        d.qpos[0:3] = root_pos[t]       # x, y, z
        d.qpos[3:7] = root_quat[t]      # w, x, y, z
        for j, qi in enumerate(joint_qpos_idx):
            d.qpos[qi] = joint_pos[t, j]
        mujoco.mj_forward(m, d)
        for i, bid in enumerate(body_mj_ids):
            body_pos_w[t, i] = d.xpos[bid]
            body_quat_w[t, i] = d.xquat[bid]

    # Compute body velocities by finite difference
    body_lin_vel_w = compute_velocities(body_pos_w, output_dt)
    body_ang_vel_w = np.zeros_like(body_pos_w)
    for b in range(len(body_names)):
        body_ang_vel_w[:, b] = compute_velocities(body_quat_w[:, b, 1:], output_dt)

    # Save NPZ
    np.savez(
        output_path,
        fps=np.array([output_fps], dtype=np.float32),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_lin_vel_w=body_lin_vel_w,
        body_ang_vel_w=body_ang_vel_w,
    )
    print(f"  → Saved: {output_path} ({T_out} frames, {len(body_names)} bodies)")


def convert_directory(input_dir: str, output_dir: str, input_fps: float = 1000.0, output_fps: float = 50.0) -> None:
    """Recursively convert all .data files in a directory to .npz."""
    os.makedirs(output_dir, exist_ok=True)
    for root_dir, _dirs, files in os.walk(input_dir):
        for fname in sorted(files):
            if fname.endswith(".data"):
                in_path = os.path.join(root_dir, fname)
                rel = os.path.relpath(in_path, input_dir)
                out_name = os.path.splitext(rel)[0] + ".npz"
                out_path = os.path.join(output_dir, out_name)
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                convert_single_file(in_path, out_path, input_fps, output_fps)


if __name__ == "__main__":
    import tyro

    def main(
        input_dir: str = "",
        output_dir: str = "",
        input_fps: float = 1000.0,
        output_fps: float = 50.0,
    ):
        if not input_dir or not output_dir:
            print(__doc__)
            sys.exit(1)
        convert_directory(input_dir, output_dir, input_fps, output_fps)
        print("[Done] Conversion complete.")

    tyro.cli(main)
