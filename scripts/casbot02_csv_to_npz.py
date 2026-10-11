"""Convert GMR CASBOT02 CSV motion to the current 22-joint AMP NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def joint_mapping(source: mujoco.MjModel, target: mujoco.MjModel):
  for model in (source, target):
    if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
      raise ValueError("Expected a free root joint at index zero")
    if not np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE):
      raise ValueError("Only hinge joints after the root are supported")
  if target.nq != 29 or target.nv != 28 or target.nbody != 31:
    raise ValueError("Expected the current 22-joint, 30-body CASBOT02 target")
  source_names = [source.joint(i).name for i in range(1, source.njnt)]
  target_names = [target.joint(i).name for i in range(1, target.njnt)]
  indices = np.array([source_names.index(name) for name in target_names])
  for target_index, source_index in enumerate(indices):
    if not np.allclose(
      source.jnt_axis[source_index + 1], target.jnt_axis[target_index + 1],
    ):
      raise ValueError(f"Joint axes differ for {target_names[target_index]}")
  dropped = [name for name in source_names if name not in target_names]
  return indices, dropped


def resample(raw, indices, input_fps, output_fps, quat_order):
  if not np.isfinite([input_fps, output_fps]).all() or min(input_fps, output_fps) <= 0:
    raise ValueError("Frame rates must be positive and finite")
  duration = (len(raw) - 1) / input_fps
  times = np.arange(0.0, duration, 1.0 / output_fps)
  if len(times) < 3:
    raise ValueError("At least three output frames are required")
  phase = times * input_fps
  first = np.floor(phase).astype(int)
  second = np.minimum(first + 1, len(raw) - 1)
  blend = (phase - first)[:, None]

  def interpolate(values):
    return values[first] * (1.0 - blend) + values[second] * blend

  root_pos = interpolate(raw[:, :3])
  joint_pos = interpolate(raw[:, 7 + indices])
  quat = raw[:, 3:7].copy()
  if quat_order == "wxyz":
    quat = quat[:, [1, 2, 3, 0]]
  if np.min(np.linalg.norm(quat, axis=-1)) < 1e-8:
    raise ValueError("Root quaternion has zero norm")
  rotations = Slerp(
    np.arange(len(raw)) / input_fps, Rotation.from_quat(quat),
  )(times)
  root_quat = rotations.as_quat()[:, [3, 0, 1, 2]]
  for i in range(1, len(root_quat)):
    if np.dot(root_quat[i - 1], root_quat[i]) < 0:
      root_quat[i] *= -1

  dt = 1.0 / output_fps
  root_lin_vel = np.gradient(root_pos, dt, axis=0)
  joint_vel = np.gradient(joint_pos, dt, axis=0)
  # Left quaternion difference is angular velocity in world coordinates.
  omega = (rotations[2:] * rotations[:-2].inv()).as_rotvec() / (2.0 * dt)
  root_ang_vel = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
  return root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel


def build_motion(model, states, output_fps):
  root_pos, root_quat, root_lin_vel, root_ang_vel, joint_pos, joint_vel = states
  frames = len(joint_pos)
  result = {
    "fps": np.asarray([output_fps], dtype=np.float64),
    "joint_pos": joint_pos.astype(np.float32),
    "joint_vel": joint_vel.astype(np.float32),
    "body_pos_w": np.empty((frames, 30, 3), dtype=np.float32),
    "body_quat_w": np.empty((frames, 30, 4), dtype=np.float32),
    "body_lin_vel_w": np.empty((frames, 30, 3), dtype=np.float32),
    "body_ang_vel_w": np.empty((frames, 30, 3), dtype=np.float32),
  }
  data = mujoco.MjData(model)
  velocity = np.zeros(6)
  for frame in range(frames):
    data.qpos[:3] = root_pos[frame]
    data.qpos[3:7] = root_quat[frame]
    data.qpos[7:] = joint_pos[frame]
    data.qvel[:3] = root_lin_vel[frame]
    # MuJoCo free-joint angular qvel is expressed in the root's local frame.
    root_rotation = Rotation.from_quat(root_quat[frame, [1, 2, 3, 0]])
    data.qvel[3:6] = root_rotation.inv().apply(root_ang_vel[frame])
    data.qvel[6:] = joint_vel[frame]
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_comVel(model, data)
    result["body_pos_w"][frame] = data.xpos[1:]
    result["body_quat_w"][frame] = data.xquat[1:]
    for body in range(1, model.nbody):
      # XBODY gives link-origin velocity, not the inertial COM velocity.
      mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
      result["body_ang_vel_w"][frame, body - 1] = velocity[:3]
      result["body_lin_vel_w"][frame, body - 1] = velocity[3:]
  for name, value in result.items():
    if not np.isfinite(value).all():
      raise ValueError(f"Non-finite output in {name}")
  if not np.allclose(np.linalg.norm(result["body_quat_w"], axis=-1), 1.0, atol=2e-6):
    raise ValueError("Output quaternions are not normalized")
  np.testing.assert_allclose(result["body_lin_vel_w"][:, 0], root_lin_vel, atol=2e-6)
  np.testing.assert_allclose(result["body_ang_vel_w"][:, 0], root_ang_vel, atol=2e-6)
  return result


def main():
  root = Path(__file__).resolve().parents[1]
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input-file", type=Path, required=True)
  parser.add_argument("--source-xml", type=Path, required=True)
  parser.add_argument("--input-fps", type=float, required=True)
  parser.add_argument("--output-fps", type=float, default=50.0)
  parser.add_argument("--quat-order", choices=("xyzw", "wxyz"), default="xyzw")
  parser.add_argument("--output-file", type=Path, required=True)
  parser.add_argument("--target-xml", type=Path, default=root / "CASBOT02_7dof/xml/CASBOT_02_23dof.xml")
  args = parser.parse_args()
  output = args.output_file.expanduser().resolve()
  if output.suffix != ".npz":
    raise ValueError("Output filename must end in .npz")
  if output.exists():
    raise FileExistsError(f"Refusing to overwrite {output}")
  raw = np.loadtxt(args.input_file.expanduser(), delimiter=",", ndmin=2)
  source = mujoco.MjModel.from_xml_path(str(args.source_xml.expanduser().resolve()))
  target = mujoco.MjModel.from_xml_path(str(args.target_xml.expanduser().resolve()))
  if raw.shape[0] < 3 or raw.shape[1] != source.nq or not np.isfinite(raw).all():
    raise ValueError(f"Expected >= 3 finite frames of {source.nq} CSV columns, got {raw.shape}")
  indices, dropped = joint_mapping(source, target)
  print(f"[mapping] keep source joint indices: {indices.tolist()}; drop: {dropped}")
  states = resample(raw, indices, args.input_fps, args.output_fps, args.quat_order)
  motion = build_motion(target, states, args.output_fps)
  for i in range(22):
    if target.jnt_limited[i + 1]:
      low, high = target.jnt_range[i + 1]
      q = motion["joint_pos"][:, i]
      excess = np.maximum(low - q, q - high)
      if float(excess.max()) > 1e-5:
        print(f"[warning] {target.joint(i + 1).name}: {int((excess > 1e-5).sum())} frames outside limits; max excess={float(excess.max()):.6f} rad (not clipped)")
  output.parent.mkdir(parents=True, exist_ok=True)
  with output.open("xb") as file:
    np.savez_compressed(file, **motion)
  with np.load(output, allow_pickle=False) as saved:
    for key, value in motion.items():
      np.testing.assert_array_equal(saved[key], value)
  print(f"[saved] {output}: {len(raw)} frames @ {args.input_fps:g} Hz -> {len(states[0])} frames @ {args.output_fps:g} Hz; {(len(states[0]) - 1) / args.output_fps:.3f}s; 22 joints, 30 bodies")


if __name__ == "__main__":
  main()
