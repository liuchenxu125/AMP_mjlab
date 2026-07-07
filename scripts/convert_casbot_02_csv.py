#!/usr/bin/env python3
"""Convert casbot_02_7dof CSV motion data (30Hz) to NPZ format (50Hz) for AMP training.

CSV format: base_pos(3) + base_quat_xyzw(4) + joint_pos(29) = 36 columns
Joint order in CSV must match XML order (see JOINT_NAMES below).

Usage:
  python scripts/convert_casbot_02_csv.py --input-file test.csv
  python scripts/convert_casbot_02_csv.py --input-dir src/assets/motions/casbot_02_7dof/amp/WalkandRun
"""

import time
from pathlib import Path
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

# 29 joints in XML order — must match CSV column order
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


class MotionLoader:
  """Load CSV, interpolate to target FPS, compute velocities."""

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
    self._load_csv(motion_file)
    self._interpolate()
    self._compute_velocities()

  def _load_csv(self, path: str):
    motion = torch.from_numpy(np.loadtxt(path, delimiter=","))
    motion = motion.to(torch.float32).to(self.device)
    assert motion.shape[1] == 36, f"Expected 36 columns (3+4+29), got {motion.shape[1]}"

    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    # CSV quat is xyzw, MuJoCo wants wxyz
    self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, 7:]  # 29 columns
    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate(self):
    times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
    self.output_frames = times.shape[0]
    idx0, idx1, blend = self._frame_blend(times)
    self.motion_base_poss = self._lerp(self.motion_base_poss_input[idx0], self.motion_base_poss_input[idx1], blend.unsqueeze(1))
    self.motion_base_rots = self._slerp(self.motion_base_rots_input[idx0], self.motion_base_rots_input[idx1], blend)
    self.motion_dof_poss = self._lerp(self.motion_dof_poss_input[idx0], self.motion_dof_poss_input[idx1], blend.unsqueeze(1))
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
      self.motion_base_poss[self.current_idx:self.current_idx + 1],
      self.motion_base_rots[self.current_idx:self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx:self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx:self.current_idx + 1],
      self.motion_dof_poss[self.current_idx:self.current_idx + 1],
      self.motion_dof_vels[self.current_idx:self.current_idx + 1],
    )
    self.current_idx += 1
    reset = self.current_idx >= self.output_frames
    if reset:
      self.current_idx = 0
    return state, reset


def build_casbot_02_scene(device: str) -> tuple[Scene, Simulation]:
  """Build minimal scene with casbot_02_7dof robot."""
  from src.assets.robots import get_casbot_02_robot_cfg

  scene_cfg = SceneCfg(entities={"robot": get_casbot_02_robot_cfg()}, num_envs=1, extent=2.0)
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
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cpu",
):
  """Convert casbot_02_7dof CSV motion files (30Hz) to NPZ (50Hz).

  Usage:
    # Single file
    python scripts/convert_casbot_02_csv.py --input-file motion.csv

    # Batch directory
    python scripts/convert_casbot_02_csv.py --input-dir src/assets/motions/casbot_02_7dof/amp/WalkandRun
  """
  if input_file is None and input_dir is None:
    raise ValueError("Must specify --input-file or --input-dir")

  # Collect files
  if input_dir:
    csv_files = sorted(Path(input_dir).glob("*.csv"))
    if not csv_files:
      raise FileNotFoundError(f"No CSV files in: {input_dir}")
    file_pairs = [(str(f), f.with_suffix(".npz").name) for f in csv_files]
    print(f"Found {len(csv_files)} CSV file(s)")
  else:
    output_name = Path(input_file).with_suffix(".npz").name
    file_pairs = [(input_file, output_name)]

  if output_dir is None:
    if input_dir:
      output_dir = input_dir
    else:
      output_dir = str(Path(input_file).parent)

  # Build scene
  print("Building scene...")
  scene, sim = build_casbot_02_scene(device)
  robot: Entity = scene["robot"]
  joint_indexes = robot.find_joints(JOINT_NAMES, preserve_order=True)[0]
  assert len(joint_indexes) == 29, f"Expected 29 joints, got {len(joint_indexes)}"

  # Process each file
  for i, (csv_path, npz_name) in enumerate(file_pairs):
    print(f"\n{'=' * 60}")
    print(f"[{i + 1}/{len(file_pairs)}] {csv_path}")
    print(f"{'=' * 60}")

    motion = MotionLoader(csv_path, input_fps, output_fps, device)

    log: dict[str, Any] = {
      "fps": [output_fps],
      "joint_pos": [], "joint_vel": [],
      "body_pos_w": [], "body_quat_w": [],
      "body_lin_vel_w": [], "body_ang_vel_w": [],
    }

    scene.reset()
    pbar = tqdm(total=motion.output_frames, desc="Processing", unit="f", ncols=100)
    done = False

    while not done:
      (base_pos, base_rot, base_lin_vel, base_ang_vel, dof_pos, dof_vel), reset = motion.get_next_state()

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

    # Stack and save
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
