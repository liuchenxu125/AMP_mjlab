"""Create offline sagittal mirrors of CASBOT02 AMP NPZ motion clips."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np


POLAR_SIGNS = np.array([1.0, -1.0, 1.0])
AXIAL_SIGNS = -POLAR_SIGNS
QUAT_SIGNS = np.array([1.0, -1.0, 1.0, -1.0])  # wxyz: R' = S R S.
FIELDS = {
  "fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
  "body_lin_vel_w", "body_ang_vel_w",
}


def partner_name(name: str) -> str:
  for left, right in (
    ("leg_l", "leg_r"),
    ("upper_left_", "upper_right_"),
    ("left_", "right_"),
  ):
    if name.startswith(left):
      return right + name[len(left):]
    if name.startswith(right):
      return left + name[len(right):]
  return name


def mirror_mapping(model: mujoco.MjModel):
  if model.nq != 29 or model.nv != 28 or model.nbody != 31:
    raise ValueError("Expected CASBOT02 with 22 hinge joints and 30 robot bodies")
  if model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
    raise ValueError("Expected the first joint to be the free root joint")
  if not np.all(model.jnt_type[1:] == mujoco.mjtJoint.mjJNT_HINGE):
    raise ValueError("Only hinge joints after the free root are supported")
  joints = [model.joint(i).name for i in range(1, model.njnt)]
  bodies = [model.body(i).name for i in range(1, model.nbody)]
  joint_perm = np.array([joints.index(partner_name(n)) for n in joints])
  body_perm = np.array([bodies.index(partner_name(n)) for n in bodies])
  joint_signs = []
  for i, partner in enumerate(joint_perm):
    reflected_axis = model.jnt_axis[partner + 1] * AXIAL_SIGNS
    sign = float(np.dot(model.jnt_axis[i + 1], reflected_axis))
    if not np.isclose(abs(sign), 1.0, atol=1e-7):
      raise ValueError(f"Unsupported mirrored axis for {joints[i]}")
    joint_signs.append(float(np.sign(sign)))
  joint_signs = np.array(joint_signs)
  if not np.array_equal(joint_perm[joint_perm], np.arange(len(joints))):
    raise ValueError("Joint mapping must be an involution")
  if not np.array_equal(body_perm[body_perm], np.arange(len(bodies))):
    raise ValueError("Body mapping must be an involution")
  if not np.array_equal(joint_signs * joint_signs[joint_perm], np.ones(len(joints))):
    raise ValueError("Joint signs must cancel after two reflections")
  return joint_perm, joint_signs, body_perm


def validate_motion(motion: dict[str, np.ndarray]) -> None:
  if set(motion) != FIELDS:
    raise ValueError(f"Unexpected NPZ fields: {set(motion) ^ FIELDS}")
  fps = motion["fps"]
  if fps.size != 1 or not np.isfinite(fps).all() or float(fps.item()) <= 0:
    raise ValueError("fps must contain a single positive finite number")
  q = motion["joint_pos"]
  if q.ndim != 2 or q.shape[1] != 22 or q.shape[0] < 2:
    raise ValueError("joint_pos must have shape (frames >= 2, 22)")
  frames = q.shape[0]
  shapes = {
    "joint_pos": (frames, 22), "joint_vel": (frames, 22),
    "body_pos_w": (frames, 30, 3), "body_quat_w": (frames, 30, 4),
    "body_lin_vel_w": (frames, 30, 3), "body_ang_vel_w": (frames, 30, 3),
  }
  for key, shape in shapes.items():
    value = motion[key]
    if value.shape != shape or not np.issubdtype(value.dtype, np.floating):
      raise ValueError(f"{key}: expected floating array of shape {shape}")
    if not np.isfinite(value).all():
      raise ValueError(f"{key} contains non-finite values")
  norms = np.linalg.norm(motion["body_quat_w"], axis=-1)
  if np.max(np.abs(norms - 1)) > 5e-4:
    raise ValueError("body_quat_w must contain unit wxyz quaternions")


def mirror_motion(motion, mapping):
  validate_motion(motion)
  joint_perm, joint_signs, body_perm = mapping
  mirrored = {"fps": motion["fps"].copy()}
  for key in ("joint_pos", "joint_vel"):
    value = motion[key]
    mirrored[key] = value[:, joint_perm] * joint_signs.astype(value.dtype)
  for key, signs in (
    ("body_pos_w", POLAR_SIGNS),
    ("body_quat_w", QUAT_SIGNS),
    ("body_lin_vel_w", POLAR_SIGNS),
    ("body_ang_vel_w", AXIAL_SIGNS),
  ):
    value = motion[key]
    mirrored[key] = value[:, body_perm] * signs.astype(value.dtype)
  validate_motion(mirrored)
  return mirrored


def check_kinematics(motion, model):
  """Verify saved body poses against root/joints, without stepping dynamics."""
  data = mujoco.MjData(model)
  max_pos_error = 0.0
  max_quat_error = 0.0
  for frame in range(motion["joint_pos"].shape[0]):
    root_quat = motion["body_quat_w"][frame, 0].astype(np.float64)
    root_quat /= np.linalg.norm(root_quat)
    data.qpos[:3] = motion["body_pos_w"][frame, 0]
    data.qpos[3:7] = root_quat
    data.qpos[7:] = motion["joint_pos"][frame]
    mujoco.mj_kinematics(model, data)
    max_pos_error = max(max_pos_error, float(np.max(np.linalg.norm(
      data.xpos[1:] - motion["body_pos_w"][frame], axis=-1,
    ))))
    saved = motion["body_quat_w"][frame].astype(np.float64)
    saved /= np.linalg.norm(saved, axis=-1, keepdims=True)
    errors = np.minimum(
      np.linalg.norm(data.xquat[1:] - saved, axis=-1),
      np.linalg.norm(data.xquat[1:] + saved, axis=-1),
    )
    max_quat_error = max(max_quat_error, float(np.max(errors)))
  if max_pos_error > 2e-4 or max_quat_error > 2e-4:
    raise ValueError(
      f"NPZ is inconsistent with XML: max position error={max_pos_error:.3g} m, "
      f"quaternion error={max_quat_error:.3g}"
    )
  return max_pos_error, max_quat_error


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("inputs", nargs="+", type=Path)
  parser.add_argument("--output-dir", type=Path)
  parser.add_argument(
    "--xml", type=Path,
    default=Path(__file__).resolve().parents[1]
    / "CASBOT02_7dof/xml/CASBOT_02_23dof.xml",
  )
  args = parser.parse_args()
  model = mujoco.MjModel.from_xml_path(str(args.xml.resolve()))
  mapping = mirror_mapping(model)
  outputs = []
  # Validate the entire request before creating any output file.
  for source in args.inputs:
    source = source.expanduser().resolve()
    if source.suffix != ".npz" or source.stem.endswith("_M"):
      raise ValueError(f"Expected an original NPZ clip, got {source}")
    parent = args.output_dir.expanduser().resolve() if args.output_dir else source.parent
    output = parent / f"{source.stem}_M.npz"
    if output.exists() or any(output == item[0] for item in outputs):
      raise FileExistsError(f"Refusing to overwrite {output}")
    with np.load(source, allow_pickle=False) as data:
      motion = {key: data[key].copy() for key in data.files}
    mirrored = mirror_motion(motion, mapping)
    recovered = mirror_motion(mirrored, mapping)
    for key in motion:
      if not np.array_equal(motion[key], recovered[key]):
        raise ValueError(f"Double mirror failed for {source.name}/{key}")
    source_errors = check_kinematics(motion, model)
    mirror_errors = check_kinematics(mirrored, model)
    print(
      f"[validated] {source.name}: {len(motion['joint_pos'])} frames "
      f"@ {float(motion['fps'].item()):g} Hz; FK max errors "
      f"source={source_errors}, mirror={mirror_errors}; double mirror exact"
    )
    outputs.append((output, mirrored))
  for output, motion in outputs:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as file:
      np.savez_compressed(file, **motion)
    with np.load(output, allow_pickle=False) as saved:
      for key, value in motion.items():
        if saved[key].dtype != value.dtype or not np.array_equal(saved[key], value):
          raise ValueError(f"Saved data mismatch for {output}/{key}")
    print(f"[saved] {output}")


if __name__ == "__main__":
  main()
