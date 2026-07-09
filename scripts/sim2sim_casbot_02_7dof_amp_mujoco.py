"""Casbot 02 7DOF (29-DOF) AMP sim2sim with ONNX policy in MuJoCo viewer."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure our project's src is found before any conflicting packages
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import argparse
from collections import deque
import re
import time

import mujoco
import mujoco.viewer
import numpy as np

from src.assets.robots import (
  CASBOT_02_7DOF_ACTION_SCALE,
  CASBOT_02_7DOF_JOINT_NAMES,
  get_casbot_02_7dof_robot_cfg,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INIT_MOTION = (
  REPO_ROOT
  / "src" / "assets" / "motions" / "casbot_02_7dof" / "amp"
  / "WalkandRun" / "CASBOT02_YKS_ZhiXing1_Base_NotRotion_1000HZ.npz"
)

# ── Configurable defaults ──
DEFAULT_ONNX_MODEL = ""
DEFAULT_COMMAND_X = 0.0
DEFAULT_COMMAND_Y = 0.0
DEFAULT_COMMAND_YAW = 0.0
DEFAULT_DURATION = 600.0
DEFAULT_VIEW_SPEED = 1.0
DEFAULT_DECIMATION = 4
DEFAULT_REALTIME = True
JOYSTICK_DEADZONE = 0.2
VEL_LIMIT_SLOW = (-0.8, 1.0)
VEL_LIMIT_FAST = (-0.8, 2.5)
VEL_LATERAL = (-1.0, 1.0)
VEL_YAW = (-1.57, 1.57)

SIM_TIMESTEP = 0.005
HISTORY_LENGTH = 4
NUM_ACTIONS = len(CASBOT_02_7DOF_JOINT_NAMES)  # 29
SINGLE_FRAME_OBS_SIZE = 3 + 3 + 3 + NUM_ACTIONS * 3  # 96
FULL_OBS_SIZE = SINGLE_FRAME_OBS_SIZE * HISTORY_LENGTH  # 384


def _onnx_model_step(path: Path) -> int:
  match = re.search(r"_model_(\d+)\.onnx$", path.name)
  return int(match.group(1)) if match else -1


def find_latest_onnx() -> Path:
  export_root = REPO_ROOT / "logs" / "rsl_rl" / "casbot_02_7dof_amp_locomotion"
  model_candidates = list(
    export_root.glob("*/export/Casbot02-7DOF-AMP-Flat_model_*.onnx")
  )
  if model_candidates:
    return max(
      model_candidates,
      key=lambda p: (_onnx_model_step(p), p.stat().st_mtime),
    )

  policy_candidates = sorted(
    list(export_root.glob("*/policy.onnx")),
    key=lambda p: p.stat().st_mtime,
  )
  if not policy_candidates:
    raise FileNotFoundError(
      "No casbot_02_7dof ONNX found. Train first, or pass model path."
    )
  return policy_candidates[-1]


# ── ONNX Policy ──────────────────────────────────────────────────────────────

class OnnxPolicy:
  def __init__(self, model_path: str | Path):
    try:
      import onnxruntime as ort
    except ModuleNotFoundError as exc:
      raise ModuleNotFoundError(
        "onnxruntime not installed. Run: pip install onnxruntime"
      ) from exc

    available = ort.get_available_providers()
    providers = [
      p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in available
    ] or available

    self.session = ort.InferenceSession(str(model_path), providers=providers)
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name
    print(
      f"[ONNX] {model_path}\n"
      f"       providers={self.session.get_providers()}\n"
      f"       input={self.input_name} {self.session.get_inputs()[0].shape}, "
      f"output={self.output_name} {self.session.get_outputs()[0].shape}"
    )

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    obs = np.ascontiguousarray(obs.astype(np.float32))
    action = self.session.run([self.output_name], {self.input_name: obs})[0]
    return np.asarray(action, dtype=np.float32).reshape(-1)


# ── Quaternion helpers ───────────────────────────────────────────────────────

def quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Rotate world-frame vector into body frame using quaternion (w,x,y,z)."""
  w, x, y, z = quat
  q_vec = np.array([-x, -y, -z], dtype=np.float64)
  t = 2.0 * np.cross(q_vec, vec)
  return vec + w * t + np.cross(q_vec, t)


# ── Model building ───────────────────────────────────────────────────────────

def build_model() -> mujoco.MjModel:
  from mjlab.entity import Entity
  entity = Entity(get_casbot_02_7dof_robot_cfg())
  spec = entity.spec
  ground = spec.worldbody.add_geom(
    name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE,
  )
  ground.size = [0.0, 0.0, 1.0]
  ground.condim = 4
  ground.friction = [0.9, 0.2, 0.2]
  model = spec.compile()
  model.opt.timestep = SIM_TIMESTEP
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  return model


# ── Observation building ─────────────────────────────────────────────────────

def get_obs_frame(
  data: mujoco.MjData,
  command: np.ndarray,
  last_action: np.ndarray,
  default_joint_pos: np.ndarray,
) -> np.ndarray:
  """Build a single 96-dim observation frame, matching training format."""
  base_ang_vel = np.asarray(
    data.sensor("angular-velocity").data, dtype=np.float64
  ).copy()
  quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
  projected_gravity = quat_apply_inverse_wxyz(
    quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float64),
  )
  joint_pos_rel = np.asarray(data.qpos[7:], dtype=np.float64) - default_joint_pos
  joint_vel_rel = np.asarray(data.qvel[6:], dtype=np.float64)

  obs = np.concatenate(
    [
      base_ang_vel,        # 3
      projected_gravity,   # 3
      command,             # 3
      joint_pos_rel,       # 29
      joint_vel_rel,       # 29
      last_action,         # 29
    ],
    axis=0,
  )
  if obs.shape != (SINGLE_FRAME_OBS_SIZE,):
    raise RuntimeError(
      f"Expected obs shape ({SINGLE_FRAME_OBS_SIZE},), got {obs.shape}"
    )
  return obs.astype(np.float32)


# ── Motion data ──────────────────────────────────────────────────────────────

def load_init_motion(path: str | Path) -> dict[str, np.ndarray]:
  motion = np.load(path)
  return {
    "root_pos": np.asarray(motion["body_pos_w"][0, 0], dtype=np.float64),
    "root_quat": np.asarray(motion["body_quat_w"][0, 0], dtype=np.float64),
    "root_lin_vel": np.asarray(motion["body_lin_vel_w"][0, 0], dtype=np.float64),
    "root_ang_vel": np.asarray(motion["body_ang_vel_w"][0, 0], dtype=np.float64),
    "joint_pos": np.asarray(motion["joint_pos"][0], dtype=np.float64),
    "joint_vel": np.asarray(motion["joint_vel"][0], dtype=np.float64),
  }


def reset_state(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  default_joint_pos: np.ndarray,
  init_motion: str | Path | None,
) -> None:
  mujoco.mj_resetData(model, data)
  if init_motion is not None:
    init = load_init_motion(init_motion)
    data.qpos[0:3] = init["root_pos"]
    data.qpos[3:7] = init["root_quat"]
    data.qpos[7:] = init["joint_pos"]
    data.qvel[0:3] = init["root_lin_vel"]
    data.qvel[3:6] = init["root_ang_vel"]
    data.qvel[6:] = init["joint_vel"]
  else:
    data.qpos[0:3] = [0.0, 0.0, 0.9]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = default_joint_pos
    data.qvel[:] = 0.0

  mujoco.mj_forward(model, data)


# ── Action helpers ───────────────────────────────────────────────────────────

def make_action_scale() -> np.ndarray:
  """Build per-joint action_scale array in JOINT_NAMES order."""
  scale = np.zeros(len(CASBOT_02_7DOF_JOINT_NAMES), dtype=np.float64)
  for i, name in enumerate(CASBOT_02_7DOF_JOINT_NAMES):
    for pattern, val in CASBOT_02_7DOF_ACTION_SCALE.items():
      if re.match(pattern, name):
        scale[i] = val
        break
  return scale


# ── Main loop ────────────────────────────────────────────────────────────────

def run(model_arg: str = "") -> None:
  model_str = model_arg or DEFAULT_ONNX_MODEL
  model_path = Path(model_str) if model_str else find_latest_onnx()
  if not model_path.exists():
    raise FileNotFoundError(model_path)

  init_motion = Path(DEFAULT_INIT_MOTION) if DEFAULT_INIT_MOTION else None
  if init_motion is not None and not init_motion.exists():
    print(f"[sim2sim] init motion not found: {init_motion}, using default pose")
    init_motion = None

  policy = OnnxPolicy(model_path)
  model = build_model()
  if model.nu != NUM_ACTIONS:
    raise RuntimeError(f"Expected {NUM_ACTIONS} actuators, got {model.nu}")

  data = mujoco.MjData(model)
  default_joint_pos = np.zeros(NUM_ACTIONS, dtype=np.float64)
  action_scale = make_action_scale()
  ctrl_lo = model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
  ctrl_hi = model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
  command = np.array(
    [DEFAULT_COMMAND_X, DEFAULT_COMMAND_Y, DEFAULT_COMMAND_YAW],
    dtype=np.float64,
  )

  reset_state(model, data, default_joint_pos, init_motion)

  decimation = int(DEFAULT_DECIMATION)
  policy_dt = model.opt.timestep * decimation
  total_steps = int(DEFAULT_DURATION / model.opt.timestep)
  history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
  last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
  target_pos = default_joint_pos.copy()

  print(
    f"[sim2sim] dt={model.opt.timestep}, decimation={decimation}, "
    f"policy_hz={1.0 / policy_dt:.1f}, duration={DEFAULT_DURATION}s"
  )
  print(
    f"[sim2sim] command=({DEFAULT_COMMAND_X:.3f}, {DEFAULT_COMMAND_Y:.3f}, "
    f"{DEFAULT_COMMAND_YAW:.3f})"
  )
  if init_motion is not None:
    print(f"[sim2sim] initialized from: {init_motion}")

  # ── Joystick ──
  try:
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT / "RoboMimic_Deploy"))
    from common.joystick import JoyStick, JoystickButton
    joystick = JoyStick()
    print(f"[Joystick] {joystick.joystick.get_name()} connected")
    print("[sim2sim] Controls: Left Stick=move, Right Stick=yaw, R2=boost, START=reset, SELECT=exit")
  except Exception:
    joystick = None
    print("[sim2sim] No joystick — robot will stand still. Pass command args to move.")

  # ── Viewer loop ──
  with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 4.0
    viewer.cam.azimuth = 80.0
    viewer.cam.elevation = -15.0
    viewer.cam.lookat[:] = (0.0, 0.0, 0.5)

    next_time = time.perf_counter()
    for step in range(total_steps):
      if not viewer.is_running():
        break

      # ── Joystick input ──
      if joystick is not None:
        joystick.update()
        if joystick.is_button_pressed(JoystickButton.SELECT):
          print("[sim2sim] SELECT pressed — exit")
          break
        if joystick.is_button_released(JoystickButton.START):
          print("[sim2sim] START pressed — reset pose")
          reset_state(model, data, default_joint_pos, init_motion)
          history.clear()
          last_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
          target_pos = default_joint_pos.copy()

        # Read axes
        ly = -joystick.get_axis_value(1)  # left stick Y (forward/back)
        lx = -joystick.get_axis_value(0)  # left stick X (lateral)
        rx = -joystick.get_axis_value(3)  # right stick X (yaw)
        r2 = joystick.get_axis_value(5)   # RT / R2

        # Deadzone
        if abs(ly) < JOYSTICK_DEADZONE: ly = 0.0
        if abs(lx) < JOYSTICK_DEADZONE: lx = 0.0
        if abs(rx) < JOYSTICK_DEADZONE: rx = 0.0

        # Speed mode
        vx_lim = VEL_LIMIT_FAST if r2 > 0.3 else VEL_LIMIT_SLOW

        command[0] = ly * (vx_lim[1] if ly > 0 else abs(vx_lim[0]))
        command[1] = lx * (VEL_LATERAL[1] if lx > 0 else abs(VEL_LATERAL[0]))
        command[2] = rx * (VEL_YAW[1] if rx > 0 else abs(VEL_YAW[0]))

      if step % decimation == 0:
        obs_frame = get_obs_frame(data, command, last_action, default_joint_pos)
        if not history:
          for _ in range(HISTORY_LENGTH):
            history.append(obs_frame)
        else:
          history.append(obs_frame)

        obs = np.concatenate(list(history), axis=0).reshape(1, -1)
        action = policy(obs)
        if action.shape != (NUM_ACTIONS,):
          raise RuntimeError(
            f"Expected ONNX action shape ({NUM_ACTIONS},), got {action.shape}"
          )

        last_action = action.astype(np.float32)
        target_pos = default_joint_pos + action.astype(np.float64) * action_scale
        target_pos = np.clip(target_pos, ctrl_lo, ctrl_hi)

        if step % int(1.0 / model.opt.timestep) == 0:
          print(
            f"t={step * model.opt.timestep:6.2f}s "
            f"base_z={data.qpos[2]:.3f} "
            f"cmd=[{command[0]:+.2f},{command[1]:+.2f},{command[2]:+.2f}] "
            f"act=[{action.min():+.2f},{action.max():+.2f}]"
          )

      data.ctrl[:] = target_pos
      mujoco.mj_step(model, data)
      viewer.sync()

      if DEFAULT_REALTIME:
        next_time += model.opt.timestep / max(DEFAULT_VIEW_SPEED, 1e-6)
        sleep_s = next_time - time.perf_counter()
        if sleep_s > 0:
          time.sleep(sleep_s)
        else:
          next_time = time.perf_counter()

  print("[sim2sim] finished.")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Casbot 02 7DOF AMP sim2sim with ONNX policy in MuJoCo viewer."
  )
  parser.add_argument(
    "model_path",
    nargs="?",
    type=str,
    default="",
    help="Optional ONNX model path. Empty = latest casbot_02_7dof ONNX under logs/.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args().model_path)
