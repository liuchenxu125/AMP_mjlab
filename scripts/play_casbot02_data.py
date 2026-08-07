"""Replay CASBOT02 motion data in a native MuJoCo window."""

from pathlib import Path
import time

import mujoco
import mujoco.viewer as mj_viewer
import numpy as np
import torch
import tyro

import mjlab
from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg

from scripts.casbot02_data_to_npz import Casbot02Motion, FPS_IN, FPS_OUT
from src.assets.robots import CASBOT02_23DOF_JOINT_NAMES
from src.tasks.amp_loco.config.casbot02.env_cfgs import casbot02_amp_flat_env_cfg


class NpzMotion:
  def __init__(self, input_file: str | Path, output_fps: int, device: str):
    data = np.load(input_file)
    self.input_file = Path(input_file)
    self.output_fps = int(data["fps"][0]) if "fps" in data else output_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0

    self.motion_base_poss = torch.tensor(
      data["body_pos_w"][:, 0, :], dtype=torch.float32, device=device
    )
    self.motion_base_rots = torch.tensor(
      data["body_quat_w"][:, 0, :], dtype=torch.float32, device=device
    )
    self.motion_base_lin_vels = torch.tensor(
      data["body_lin_vel_w"][:, 0, :], dtype=torch.float32, device=device
    )
    self.motion_base_ang_vels = torch.tensor(
      data["body_ang_vel_w"][:, 0, :], dtype=torch.float32, device=device
    )
    self.motion_dof_poss = torch.tensor(
      data["joint_pos"], dtype=torch.float32, device=device
    )
    self.motion_dof_vels = torch.tensor(
      data["joint_vel"], dtype=torch.float32, device=device
    )
    self.output_frames = self.motion_dof_poss.shape[0]
    print(
      f"Loaded {self.input_file}: {self.output_frames} frames "
      f"@ {self.output_fps} Hz"
    )

  def get_next_state(self):
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx = (self.current_idx + 1) % self.output_frames
    return state


def _load_motion(
  input_file: str,
  input_fps: int,
  output_fps: int,
  device: str,
  skip_first_line: bool,
):
  suffix = Path(input_file).suffix.lower()
  if suffix == ".data":
    return Casbot02Motion(
      input_file=input_file,
      input_fps=input_fps,
      output_fps=output_fps,
      device=device,
      skip_first_line=skip_first_line,
    )
  if suffix == ".npz":
    return NpzMotion(input_file=input_file, output_fps=output_fps, device=device)
  raise ValueError(f"Unsupported motion file suffix: {suffix}")


def _write_state(
  sim: Simulation,
  scene: Scene,
  robot: Entity,
  robot_joint_indexes: list[int],
  motion,
) -> None:
  (
    motion_base_pos,
    motion_base_rot,
    motion_base_lin_vel,
    motion_base_ang_vel,
    motion_dof_pos,
    motion_dof_vel,
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


def _sync_window(sim: Simulation, window_viewer) -> None:
  sim.mj_data.qpos[:] = sim.data.qpos[0].cpu().numpy()
  sim.mj_data.qvel[:] = sim.data.qvel[0].cpu().numpy()
  if sim.mj_model.nmocap > 0:
    sim.mj_data.mocap_pos[:] = sim.data.mocap_pos[0].cpu().numpy()
    sim.mj_data.mocap_quat[:] = sim.data.mocap_quat[0].cpu().numpy()
  sim.mj_data.xfrc_applied[:] = sim.data.xfrc_applied[0].cpu().numpy()
  mujoco.mj_forward(sim.mj_model, sim.mj_data)
  window_viewer.sync()


def main(
  input_file: str = "src/assets/motions/casbot02/amp/WalkandRun_TurnBoost_v1/原地左转.npz",
  input_fps: int = FPS_IN,
  output_fps: int = FPS_OUT,
  device: str = "cuda:0",
  speed: float = 1.0,
  skip_first_line: bool = True,
):
  """Replay a CASBOT02 .npz or raw .data file."""
  input_path = Path(input_file)
  if not input_path.exists():
    raise FileNotFoundError(input_path)

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(casbot02_amp_flat_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  motion = _load_motion(
    input_file=str(input_path),
    input_fps=input_fps,
    output_fps=output_fps,
    device=device,
    skip_first_line=skip_first_line,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(
    CASBOT02_23DOF_JOINT_NAMES,
    preserve_order=True,
  )[0]

  scene.reset()
  frame_dt = 1.0 / getattr(motion, "output_fps", output_fps)
  speed = max(speed, 1e-6)
  print(f"Playing {input_path} at {speed:.2f}x. Close the MuJoCo window to stop.")

  with mj_viewer.launch_passive(sim.mj_model, sim.mj_data) as window_viewer:
    next_frame_time = time.perf_counter()
    while window_viewer.is_running():
      _write_state(sim, scene, robot, robot_joint_indexes, motion)
      _sync_window(sim, window_viewer)

      next_frame_time += frame_dt / speed
      sleep_s = next_frame_time - time.perf_counter()
      if sleep_s > 0:
        time.sleep(sleep_s)
      else:
        next_frame_time = time.perf_counter()


if __name__ == "__main__":
  tyro.cli(main, config=mjlab.TYRO_FLAGS)
