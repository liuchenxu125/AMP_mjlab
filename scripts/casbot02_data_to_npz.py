"""Convert CASBOT02 .data files to AMP .npz motion files."""

from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_conjugate,
  quat_mul,
)

from src.assets.robots import CASBOT02_23DOF_JOINT_NAMES
from src.tasks.amp_loco.config.casbot02.env_cfgs import casbot02_amp_flat_env_cfg


FPS_IN = 1000
FPS_OUT = 50
PITCH_STANDING_OFFSET = np.pi / 2.0


def euler_pyr_to_quat_xyzw(pyr: np.ndarray) -> np.ndarray:
  """Convert (pitch, yaw, roll) Euler angles to xyzw quaternions."""
  pitch = pyr[:, 0]
  yaw = pyr[:, 1]
  roll = pyr[:, 2]

  cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
  cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
  cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)

  w = cr * cp * cy + sr * sp * sy
  x = sr * cp * cy - cr * sp * sy
  y = cr * sp * cy + sr * cp * sy
  z = cr * cp * sy - sr * sp * cy

  quat = np.stack([x, y, z, w], axis=1)
  norm = np.linalg.norm(quat, axis=1, keepdims=True)
  return (quat / np.clip(norm, 1e-8, None)).astype(np.float32)


def _make_quat_continuous(quat_wxyz: torch.Tensor) -> torch.Tensor:
  quat = quat_wxyz.clone()
  for i in range(1, quat.shape[0]):
    if torch.dot(quat[i - 1], quat[i]) < 0.0:
      quat[i] = -quat[i]
  return quat


def _so3_derivative(rotations: torch.Tensor, dt: float) -> torch.Tensor:
  if rotations.shape[0] < 3:
    return torch.zeros(rotations.shape[0], 3, device=rotations.device)

  q_prev, q_next = rotations[:-2], rotations[2:]
  q_rel = quat_mul(q_next, quat_conjugate(q_prev))
  omega = axis_angle_from_quat(q_rel) / (2.0 * dt)
  return torch.cat([omega[:1], omega, omega[-1:]], dim=0)


def _load_raw_data(path: Path, skip_first_line: bool) -> np.ndarray:
  raw = np.genfromtxt(
    str(path),
    delimiter=",",
    skip_header=1 if skip_first_line else 0,
    invalid_raise=False,
  )
  if raw.ndim == 1:
    raw = raw[None, :]

  if raw.shape[1] >= 62 and np.all(np.isnan(raw[:, -1])):
    raw = raw[:, :-1]

  valid = ~np.any(np.isnan(raw), axis=1)
  raw = raw[valid]

  if raw.shape[1] < 61:
    raise ValueError(f"{path}: expected at least 61 columns, got {raw.shape[1]}")

  return raw[:, :61]


class Casbot02Motion:
  def __init__(
    self,
    input_file: str | Path,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    skip_first_line: bool,
  ):
    self.input_file = Path(input_file)
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.output_dt = 1.0 / output_fps
    self.device = device
    self.current_idx = 0

    raw = _load_raw_data(self.input_file, skip_first_line)
    if input_fps % output_fps != 0:
      raise ValueError(
        f"Expected integer downsample ratio, got {input_fps} -> {output_fps}"
      )
    stride = input_fps // output_fps
    raw = raw[::stride]

    self.output_frames = raw.shape[0]
    self.motion_base_poss = torch.tensor(
      self._root_pos(raw), dtype=torch.float32, device=device
    )
    self.motion_base_rots = torch.tensor(
      self._root_quat_wxyz(raw), dtype=torch.float32, device=device
    )
    self.motion_base_rots = _make_quat_continuous(self.motion_base_rots)
    self.motion_dof_poss = torch.tensor(
      self._dof_pos(raw), dtype=torch.float32, device=device
    )

    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = _so3_derivative(
      self.motion_base_rots, self.output_dt
    )

    print(
      f"Loaded {self.input_file}: {self.output_frames} frames @ {output_fps} Hz"
    )

  def _root_pos(self, raw: np.ndarray) -> np.ndarray:
    root_pos_data = raw[:, 0:3].astype(np.float32)
    return np.stack(
      [root_pos_data[:, 1], -root_pos_data[:, 0], root_pos_data[:, 2]],
      axis=1,
    ).astype(np.float32)

  def _root_quat_wxyz(self, raw: np.ndarray) -> np.ndarray:
    pyr = raw[:, 3:6].astype(np.float32)
    col4_mean = float(np.mean(pyr[:, 0]))
    if abs(col4_mean - PITCH_STANDING_OFFSET) < abs(col4_mean):
      pitch = pyr[:, 0] - PITCH_STANDING_OFFSET
      print(
        f"  [pitch] col4 mean={col4_mean:.4f} -> subtract pi/2"
      )
    else:
      pitch = pyr[:, 0].copy()
      print(f"  [pitch] col4 mean={col4_mean:.4f} -> no offset")

    roll = pyr[:, 1]
    yaw = -pyr[:, 2]
    quat_xyzw = euler_pyr_to_quat_xyzw(np.stack([pitch, yaw, roll], axis=1))
    return quat_xyzw[:, [3, 0, 1, 2]].astype(np.float32)

  def _dof_pos(self, raw: np.ndarray) -> np.ndarray:
    left_leg = raw[:, 12:18]
    right_leg = raw[:, 18:24]
    left_arm5 = raw[:, 24:29]
    right_arm5 = raw[:, 31:36]
    waist = raw[:, 60:61]
    dof_pos = np.concatenate(
      [left_leg, right_leg, waist, left_arm5, right_arm5],
      axis=1,
    ).astype(np.float32)
    if dof_pos.shape[1] != len(CASBOT02_23DOF_JOINT_NAMES):
      raise ValueError(f"Expected 23 dof columns, got {dof_pos.shape[1]}")
    return dof_pos

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def run_sim(
  sim: Simulation,
  scene: Scene,
  input_file: str | Path,
  input_fps: int,
  output_fps: int,
  output_name: str,
  output_dir: str | Path,
  skip_first_line: bool,
):
  motion = Casbot02Motion(
    input_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    skip_first_line=skip_first_line,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(
    CASBOT02_23DOF_JOINT_NAMES,
    preserve_order=True,
  )[0]

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
  pbar = tqdm(total=motion.output_frames, desc=Path(input_file).name, unit="frame")

  file_saved = False
  frame_count = 0
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
    log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

    frame_count += 1
    pbar.update(1)

    if reset_flag:
      file_saved = True
      pbar.close()

  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)

  output_dir_path = Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)
  out_path = output_dir_path / output_name
  np.savez(str(out_path), **log)  # type: ignore[arg-type]
  print(f"[OK] {input_file} -> {out_path} ({frame_count} frames)")


def main(
  input_file: str | None = None,
  input_dir: str | None = "casbot_data",
  output_name: str | None = None,
  output_dir: str = "src/assets/motions/casbot02/amp/WalkandRun",
  input_fps: int = FPS_IN,
  output_fps: int = FPS_OUT,
  device: str = "cuda:0",
  skip_first_line: bool = True,
):
  """Replay CASBOT02 .data files on the 23DOF model and save AMP .npz files."""
  if input_file is None and input_dir is None:
    raise ValueError("Either --input-file or --input-dir must be specified.")

  if input_file is not None:
    if output_name is None:
      output_name = Path(input_file).with_suffix(".npz").name
    file_pairs = [(input_file, output_name)]
  else:
    assert input_dir is not None
    data_files = sorted(Path(input_dir).glob("*.data"))
    if not data_files:
      raise FileNotFoundError(f"No .data files found in {input_dir}")
    file_pairs = [(str(f), f.with_suffix(".npz").name) for f in data_files]
    print(f"Found {len(data_files)} .data files in {input_dir}")

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(casbot02_amp_flat_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  for i, (cur_input_file, cur_output_name) in enumerate(file_pairs):
    if len(file_pairs) > 1:
      print(f"\nProcessing file {i + 1}/{len(file_pairs)}: {cur_input_file}")
    run_sim(
      sim=sim,
      scene=scene,
      input_file=cur_input_file,
      input_fps=input_fps,
      output_fps=output_fps,
      output_name=cur_output_name,
      output_dir=output_dir,
      skip_first_line=skip_first_line,
    )


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
