#!/usr/bin/env python3
"""Convert casbot_skeleton CSV motion data to NPZ format for AMP training.

Usage:
  # 单个文件转换
  python scripts/convert_casbot_motion.py --input B3_walk_forward_poses.csv

  # 批量转换整个目录
  python scripts/convert_casbot_motion.py --input-dir src/assets/motions/casbot_skeledon/amp/WalkandRun

  # 带渲染预览
  python scripts/convert_casbot_motion.py --input test.csv --render --render-backend window
"""

import time
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene, SceneCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

# ── Casbot joint names (must match CSV column order) ────────────────────────
CASBOT_JOINT_NAMES = [
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
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_pitch_joint",
  "right_wrist_yaw_joint",
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
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    # CSV quat is xyzw, MuJoCo wants wxyz
    self.motion_base_rots_input = self.motion_base_rots_input[:, [3, 0, 1, 2]]
    self.motion_dof_poss_input = motion[:, 7:]
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


def build_casbot_scene(device: str) -> tuple[Scene, Simulation]:
  """Build minimal scene with casbot robot for motion replay."""
  from src.assets.robots import get_casbot_robot_cfg

  scene_cfg = SceneCfg(entities={"robot": get_casbot_robot_cfg()}, num_envs=1, extent=2.0)
  scene = Scene(scene_cfg, device=device)
  model = scene.compile()

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / 50.0  # will be overridden
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)
  return scene, sim


def main(
  input_file: str | None = None,
  input_dir: str | None = None,
  output_dir: str = "./motion_data_npz",
  input_fps: float = 30.0,
  output_fps: float = 50.0,
  device: str = "cpu",
  render: bool = False,
  render_backend: str = "window",  # "window" or "offscreen"
  window_realtime: bool = False,
  window_realtime_scale: float = 1.0,
  video_output: str | None = None,
):
  """Convert casbot_skeleton CSV motion files to NPZ.

  Args:
    input_file: Single CSV file to convert.
    input_dir: Directory of CSV files for batch conversion.
    output_dir: Where to save NPZ files.
    input_fps: Frame rate of the CSV data (default 50).
    output_fps: Desired output frame rate (default 50).
    device: 'cpu' or 'cuda:0'.
    render: Show MuJoCo viewer during conversion.
    render_backend: 'window' or 'offscreen'.
    window_realtime: Sync playback to wall-clock time.
    window_realtime_scale: Speed multiplier for realtime playback.
    video_output: Save offscreen render to mp4.
  """
  if input_file is None and input_dir is None:
    raise ValueError("必须指定 --input-file 或 --input-dir")

  # ── Collect files ──
  if input_dir:
    csv_files = sorted(Path(input_dir).glob("*.csv"))
    if not csv_files:
      raise FileNotFoundError(f"目录中没有 CSV 文件: {input_dir}")
    file_pairs = [(str(f), f.with_suffix(".npz").name) for f in csv_files]
    print(f"找到 {len(csv_files)} 个 CSV 文件")
  else:
    output_name = Path(input_file).with_suffix(".npz").name
    file_pairs = [(input_file, output_name)]

  # ── Build scene ──
  scene, sim = build_casbot_scene(device)
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=scene.compile(), device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  joint_indexes = robot.find_joints(CASBOT_JOINT_NAMES, preserve_order=True)[0]

  # ── Renderer ──
  renderer = None
  if render and render_backend == "offscreen":
    viewer_cfg = ViewerConfig(
      height=480, width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      entity_name="robot", distance=2.0, elevation=-5.0, azimuth=20,
    )
    renderer = OffscreenRenderer(model=sim.mj_model, cfg=viewer_cfg, scene=scene)
    renderer.initialize()

  # ── Process each file ──
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

    frames = []
    scene.reset()

    pbar = tqdm(total=motion.output_frames, desc="Processing", unit="f", ncols=100)
    frame_count = 0
    wall_start = time.perf_counter()
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

      if render and renderer is not None:
        renderer.update(sim.data)
        frames.append(renderer.render())

      if render and render_backend == "window" and frame_count == 0:
        # Only launch viewer on first frame of first file
        pass

      log["joint_pos"].append(robot.data.joint_pos[0].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0].cpu().numpy().copy())
      log["body_lin_vel_w"].append(robot.data.body_link_lin_vel_w[0].cpu().numpy().copy())
      log["body_ang_vel_w"].append(robot.data.body_link_ang_vel_w[0].cpu().numpy().copy())

      frame_count += 1
      pbar.update(1)

      if reset:
        done = True
        pbar.close()

    # ── Stack and save ──
    for k in log:
      if k != "fps":
        log[k] = np.stack(log[k], axis=0)

    out_path = Path(output_dir) / npz_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(out_path), **log)
    print(f"  ✓ Saved: {out_path}  ({log['joint_pos'].shape[0]} frames)")

    # ── Video ──
    if render and renderer is not None and frames:
      mp4_path = Path(video_output) if video_output else out_path.with_suffix(".mp4")
      try:
        import imageio.v3 as iio
        iio.imwrite(str(mp4_path), np.stack(frames, axis=0), fps=output_fps)
        print(f"  ✓ Video: {mp4_path}")
      except ImportError:
        print("  ⚠ imageio not installed, skip video")

  print(f"\n完成! 输出目录: {output_dir}")


if __name__ == "__main__":
  tyro.cli(main)
