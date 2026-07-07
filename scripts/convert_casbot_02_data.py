#!/usr/bin/env python3
"""Convert casbot_02_7dof .data motion data (1000Hz) to NPZ format (50Hz) for AMP training.

.data format: 61 columns at 1000Hz, comma-separated, trailing comma allowed.
Column layout (1-indexed):
  1-3:   base position x, y, z (m)
  4-6:   base rotation pitch, yaw, roll (rad) — Euler angles
  7-9:   base linear velocity x, y, z (m/s)
  10-12: base rotation pitch, yaw, roll (mrad) — redundant with 4-6, ignored
  13-18: left leg 6 joints
  19-24: right leg 6 joints
  25-31: left arm 7 joints
  32-38: right arm 7 joints
  39-48: left hand 10 fingers (ignored — not in AMP joint set)
  49-58: right hand 10 fingers (ignored — not in AMP joint set)
  59-60: head 2 joints (yaw, pitch)
  61:    waist 1 joint (yaw)

Output .npz format (same as convert_casbot_02_csv.py, compatible with AMP training):
  - fps: [50]
  - joint_pos:     (T, 29)  joint positions in MuJoCo order
  - joint_vel:     (T, 29)  joint velocities in MuJoCo order
  - body_pos_w:    (T, N, 3)  body link world positions
  - body_quat_w:   (T, N, 4)  body link world quaternions (wxyz)
  - body_lin_vel_w:(T, N, 3)  body link world linear velocities
  - body_ang_vel_w:(T, N, 3)  body link world angular velocities

Usage:
  python scripts/convert_casbot_02_data.py --input-file motion.data
  python scripts/convert_casbot_02_data.py --input-dir src/assets/motions/casbot_02_7dof/amp/WalkandRun
"""

import sys
from pathlib import Path

# Ensure our project's src is found before any conflicting packages
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import time
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
    axis_angle_from_quat,
    quat_conjugate,
    quat_mul,
    quat_slerp,
)

# 29 joints in XML / MuJoCo order — must match the order extracted from .data columns.
# The JOINT_NAMES list determines both the extraction order from .data and the
# order in which joint positions are written to the simulator.
JOINT_NAMES = [
    "left_leg_pelvic_pitch_joint",
    "left_leg_pelvic_roll_joint",
    "left_leg_pelvic_yaw_joint",
    "left_leg_knee_pitch_joint",
    "left_leg_ankle_pitch_joint",
    "left_leg_ankle_roll_joint",
    "right_leg_pelvic_pitch_joint",
    "right_leg_pelvic_roll_joint",
    "right_leg_pelvic_yaw_joint",
    "right_leg_knee_pitch_joint",
    "right_leg_ankle_pitch_joint",
    "right_leg_ankle_roll_joint",
    "waist_yaw_joint",
    "head_yaw_joint",
    "head_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint",
    "left_wrist_yaw_joint",
    "left_wrist_pitch_joint",
    "left_wrist_roll_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint",
    "right_wrist_yaw_joint",
    "right_wrist_pitch_joint",
    "right_wrist_roll_joint",
]

# Column indices (0-indexed) in .data file that map to JOINT_NAMES in order.
# Finger joints (cols 39-58) are excluded — AMP only uses the 29 main joints.
DATA_JOINT_COLUMNS = [
    12, 13, 14, 15, 16, 17,   # left leg:  pelvic pitch/roll/yaw, knee pitch, ankle pitch/roll
    18, 19, 20, 21, 22, 23,   # right leg: pelvic pitch/roll/yaw, knee pitch, ankle pitch/roll
    60,                         # waist yaw (col 61)
    58, 59,                     # head yaw, pitch (cols 59, 60)
    24, 25, 26, 27, 28, 29, 30,  # left arm: shoulder pitch/roll/yaw, elbow pitch, wrist yaw/pitch/roll
    31, 32, 33, 34, 35, 36, 37,  # right arm: shoulder pitch/roll/yaw, elbow pitch, wrist yaw/pitch/roll
]


def load_data_file(path: str) -> np.ndarray:
    """Load a .data file, handling trailing commas.

    Returns float32 array of shape (T, 61).
    """
    with open(path) as f:
        lines = [line.strip().rstrip(",") for line in f if line.strip()]
    data = np.array([list(map(float, line.split(","))) for line in lines], dtype=np.float32)
    assert data.shape[1] == 61, f"Expected 61 columns, got {data.shape[1]} in {path}"
    return data


# ── Coordinate frame mapping (verified against colleague's working code) ──
# .data file columns 4-6 (0-indexed 3-5): euler angles in rad
#   col 4 = pitch, col 5 = roll, col 6 = yaw
# Standing upright: pitch ≈ π/2 (offset), roll ≈ 0, yaw ≈ 0.
# Root pos: col 1=X(lateral), col 2=Y(forward), col 3=Z(up).
# MuJoCo frame: X=forward, Y=lateral, Z=up.
# Transform: pos_mujoco = [data_Y, -data_X, data_Z]

PITCH_STANDING_OFFSET = np.pi / 2.0


def euler_pyr_to_quat_xyzw(pitch, yaw, roll):
    """Convert (pitch, yaw, roll) ZYX intrinsic Euler to xyzw quaternion."""
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    quat = np.stack([x, y, z, w], axis=-1)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    return (quat / np.clip(norm, 1e-8, None)).astype(np.float32)


class MotionDataLoader:
    """Load .data file (1000Hz), interpolate to target FPS, compute velocities.

    Mirrors the MotionLoader in convert_casbot_02_csv.py but reads .data
    format (Euler angles → quaternion) instead of CSV format (quaternion).
    """

    def __init__(
        self,
        motion_file: str,
        input_fps: float,
        output_fps: float,
        device: torch.device | str,
    ):
        self.input_fps = input_fps
        self.output_fps = output_fps
        self.input_dt = 1.0 / input_fps
        self.output_dt = 1.0 / output_fps
        self.current_idx = 0
        self.device = device
        self._load_data(motion_file)
        self._interpolate()
        self._compute_velocities()

    def _load_data(self, path: str):
        data = load_data_file(path)

        # Base position: cols 1-3 (0-indexed: 0,1,2)
        #   data X = lateral (左右), data Y = forward (前后), data Z = up (上下)
        # MuJoCo: X = forward, Y = lateral, Z = up
        # → pos_mujoco = [data_Y, -data_X, data_Z]  (verified against colleague)
        pos_data = data[:, 0:3]
        base_pos = np.stack(
            [pos_data[:, 1], -pos_data[:, 0], pos_data[:, 2]], axis=1
        ).astype(np.float32)

        # Base orientation: cols 4-6 (0-indexed: 3,4,5)
        #   col 4 = pitch, col 5 = ROLL, col 6 = YAW (verified against colleague)
        # Standing upright: pitch ≈ π/2 → subtract offset
        pitch_raw = data[:, 3]
        pitch = pitch_raw - PITCH_STANDING_OFFSET
        roll = data[:, 4]
        yaw = -data[:, 5]  # negated (verified against colleague)

        # Euler (pitch, yaw, roll) ZYX intrinsic → quaternion xyzw
        base_quat_xyzw = euler_pyr_to_quat_xyzw(pitch, yaw, roll)

        # Extract 29 joints in JOINT_NAMES order
        joint_pos = data[:, DATA_JOINT_COLUMNS]  # (T, 29)
        assert joint_pos.shape[1] == 29, f"Expected 29 joints, got {joint_pos.shape[1]}"

        # Convert to torch tensors (float32, on target device)
        self.motion_base_poss_input = torch.from_numpy(base_pos).to(torch.float32).to(self.device)
        self.motion_base_rots_input = torch.from_numpy(base_quat_xyzw).to(torch.float32).to(self.device)
        # MuJoCo uses wxyz quaternion convention; scipy/spatial gives xyzw
        self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
        self.motion_dof_poss_input = torch.from_numpy(joint_pos).to(torch.float32).to(self.device)
        self.input_frames = data.shape[0]
        self.duration = (self.input_frames - 1) * self.input_dt

    def _interpolate(self):
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        idx0, idx1, blend = self._frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[idx0], self.motion_base_poss_input[idx1], blend.unsqueeze(1)
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[idx0], self.motion_base_rots_input[idx1], blend
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[idx0], self.motion_dof_poss_input[idx1], blend.unsqueeze(1)
        )
        print(f"  frames: {self.input_frames} → {self.output_frames} @ {self.output_fps} fps")

    def _frame_blend(self, times):
        phase = times / self.duration
        idx0 = (phase * (self.input_frames - 1)).floor().long()
        idx1 = torch.minimum(idx0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - idx0
        return idx0, idx1, blend

    def _lerp(self, a, b, blend):
        return a * (1 - blend) + b * blend

    def _slerp(self, a, b, blend):
        out = torch.zeros_like(a)
        for i in range(a.shape[0]):
            out[i] = quat_slerp(a[i], b[i], float(blend[i]))
        return out

    def _compute_velocities(self):
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        q_prev, q_next = self.motion_base_rots[:-2], self.motion_base_rots[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))
        omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
        self.motion_base_ang_vels = torch.cat([omega[:1], omega, omega[-1:]], dim=0)

    def get_next_state(self):
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset = self.current_idx >= self.output_frames
        if reset:
            self.current_idx = 0
        return state, reset


def build_casbot_02_scene(device: str) -> tuple[Scene, Simulation]:
    """Build minimal scene with casbot_02_7dof robot (1 env)."""
    from src.assets.robots import get_casbot_02_7dof_robot_cfg

    scene_cfg = SceneCfg(entities={"robot": get_casbot_02_7dof_robot_cfg()}, num_envs=1, extent=2.0)
    scene = Scene(scene_cfg, device=device)
    model = scene.compile()

    sim_cfg = SimulationCfg()
    sim_cfg.mujoco.timestep = 1.0 / 50.0
    sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
    scene.initialize(sim.mj_model, sim.model, sim.data)
    return scene, sim


def main(
    input_file: str | None = None,
    input_dir: str | None = None,
    output_dir: str | None = None,
    input_fps: float = 1000.0,
    output_fps: float = 50.0,
    device: str = "cpu",
):
    """Convert casbot_02_7dof .data motion files (1000Hz) to NPZ (50Hz).

    Usage:
      # Single file
      python scripts/convert_casbot_02_data.py --input-file motion.data

      # Batch directory
      python scripts/convert_casbot_02_data.py --input-dir src/assets/motions/casbot_02_7dof/amp/WalkandRun
    """
    if input_file is None and input_dir is None:
        raise ValueError("Must specify --input-file or --input-dir")

    # Collect files
    if input_dir:
        data_files = sorted(Path(input_dir).glob("*.data"))
        if not data_files:
            raise FileNotFoundError(f"No .data files in: {input_dir}")
        file_pairs = [(str(f), f.with_suffix(".npz").name) for f in data_files]
        print(f"Found {len(data_files)} .data file(s)")
    else:
        output_name = Path(input_file).with_suffix(".npz").name
        file_pairs = [(input_file, output_name)]

    if output_dir is None:
        if input_dir:
            output_dir = input_dir
        else:
            output_dir = str(Path(input_file).parent)

    # Build scene (shared across files for efficiency)
    print("Building scene...")
    scene, sim = build_casbot_02_scene(device)
    robot: Entity = scene["robot"]
    joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
    assert len(joint_indexes) == 29, f"Expected 29 joints, got {len(joint_indexes)}"

    # Process each file
    for i, (data_path, npz_name) in enumerate(file_pairs):
        print(f"\n{'=' * 60}")
        print(f"[{i + 1}/{len(file_pairs)}] {data_path}")
        print(f"{'=' * 60}")

        motion = MotionDataLoader(data_path, input_fps, output_fps, device)

        log: dict[str, Any] = {
            "fps": [output_fps],
            "joint_pos": [],
            "joint_vel": [],
            "body_pos_w": [],
            "body_quat_w": [],
            "body_lin_vel_w": [],
            "body_ang_vel_w": [],
        }

        scene.reset()
        pbar = tqdm(total=motion.output_frames, desc="Processing", unit="f", ncols=100)
        done = False

        while not done:
            (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), reset = (
                motion.get_next_state()
            )

            root_states = robot.data.default_root_state.clone()
            root_states[:, :3] = base_pos
            root_states[:, :2] += scene.env_origins[:, :2]
            root_states[:, 3:7] = base_rot
            root_states[:, 7:10] = base_lin_vel
            root_states[:, 10:] = base_ang_vel
            robot.write_root_state_to_sim(root_states)

            jp = robot.data.default_joint_pos.clone()
            jv = robot.data.default_joint_vel.clone()
            jp[:, joint_indexes] = dof_pos
            jv[:, joint_indexes] = dof_vel
            robot.write_joint_state_to_sim(jp, jv)

            sim.forward()
            scene.update(sim.mj_model.opt.timestep)

            log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

            pbar.update(1)

            if reset:
                done = True
                pbar.close()

        # Stack lists into numpy arrays
        for k in log:
            if k != "fps":
                log[k] = np.stack(log[k], axis=0)

        out_path = Path(output_dir) / npz_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(str(out_path), **log)
        npz_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"  Saved: {out_path}  ({log['joint_pos'].shape[0]} frames, {npz_size_mb:.1f} MB)")

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    tyro.cli(main)
