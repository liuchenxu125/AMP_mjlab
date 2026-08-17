"""CASBOT02 leg-observation AMP sim2sim for the current 22-action policy."""

from __future__ import annotations

import argparse
from collections import deque
from pathlib import Path
import re
import time

import mujoco
import numpy as np

from mjlab.entity import Entity

from src.assets.robots import (
  CASBOT02_22DOF_NO_WAIST_ACTION_SCALE,
  CASBOT02_22DOF_NO_WAIST_JOINT_NAMES,
  CASBOT02_23DOF_JOINT_NAMES,
  CASBOT02_LEG_ONLY_JOINT_NAMES,
  get_casbot02_23dof_robot_cfg,
)
from src.assets.robots.casbot02.casbot02_constants import HOME_KEYFRAME


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INIT_MOTION_CANDIDATES = (
  REPO_ROOT
  / "src"
  / "assets"
  / "motions"
  / "casbot02"
  / "amp"
  / "WalkandRun_TurnBoost_v1"
  / "直行1步态.npz",
  REPO_ROOT
  / "src"
  / "assets"
  / "motions"
  / "casbot02"
  / "amp"
  / "WalkandRun_InPlaceTurnOnly"
  / "直行1步态.npz",
)
DEFAULT_INIT_MOTION = next(
  (path for path in DEFAULT_INIT_MOTION_CANDIDATES if path.exists()), None
)

# Edit these defaults when you want to change sim2sim behavior from this file.
DEFAULT_ONNX_MODEL = ""
DEFAULT_COMMAND_X = 0.0
DEFAULT_COMMAND_Y = 0.0
DEFAULT_COMMAND_YAW = 0.0
DEFAULT_DURATION = 6000.0
DEFAULT_VIEW_SPEED = 1.0
DEFAULT_DECIMATION = 4
DEFAULT_RENDER_DECIMATION = 4
DEFAULT_VIEWER_MODE = "window"
DEFAULT_VIEWER_WIDTH = 2560
DEFAULT_VIEWER_HEIGHT = 1440
DEFAULT_PRINT_EVERY = 1.0
DEFAULT_REALTIME = True
DEFAULT_ONNXRUNTIME_PROVIDER = "auto"

SIM_TIMESTEP = 0.005
HISTORY_LENGTH = 4
NUM_OBS_JOINTS = len(CASBOT02_LEG_ONLY_JOINT_NAMES)  # 12 leg-only actor obs joints
NUM_POLICY_ACTIONS = len(CASBOT02_22DOF_NO_WAIST_JOINT_NAMES)
NUM_FULL_JOINTS = len(CASBOT02_23DOF_JOINT_NAMES)  # 23 MuJoCo actuators, including waist
OBS_JOINT_INDICES = np.array(
  [CASBOT02_23DOF_JOINT_NAMES.index(n) for n in CASBOT02_LEG_ONLY_JOINT_NAMES],
  dtype=np.int64,
)
ACTION_JOINT_INDICES = np.array(
  [
    CASBOT02_23DOF_JOINT_NAMES.index(n)
    for n in CASBOT02_22DOF_NO_WAIST_JOINT_NAMES
  ],
  dtype=np.int64,
)
CURRENT_SINGLE_FRAME_OBS_SIZE = 45  # 3+3+3+12+12+12, no phase
COMMAND_X_RANGE = (-3.5, 5.0)
COMMAND_Y_RANGE = (-1.0, 1.0)
COMMAND_YAW_RANGE = (-3.14, 3.14 )
COMMAND_X_STEP = 0.1
COMMAND_Y_STEP = 0.1
COMMAND_YAW_STEP = 0.1


def make_default_joint_pos() -> np.ndarray:
  """Return full 23-dim default joint positions for target_pos mapping."""
  joint_pos = HOME_KEYFRAME.joint_pos
  if joint_pos is None:
    return np.zeros(NUM_FULL_JOINTS, dtype=np.float64)
  fallback = float(joint_pos.get(".*", 0.0))
  return np.array(
    [float(joint_pos.get(name, fallback)) for name in CASBOT02_23DOF_JOINT_NAMES],
    dtype=np.float64,
  )


def make_obs_joint_pos() -> np.ndarray:
  """Return default joint positions for the 12 actor-observed leg joints."""
  return make_default_joint_pos()[OBS_JOINT_INDICES]


def make_default_base_pos() -> np.ndarray:
  return np.asarray(HOME_KEYFRAME.pos, dtype=np.float64)


def _onnx_model_step(path: Path) -> int:
  match = re.search(r"_model_(\d+)\.onnx$", path.name)
  return int(match.group(1)) if match else -1


def find_latest_onnx() -> Path:
  export_root = REPO_ROOT / "logs" / "rsl_rl" / "casbot02_leg_amp_locomotion"
  run_dirs = sorted(
    [path for path in export_root.iterdir() if path.is_dir()],
    key=lambda p: (p.name, p.stat().st_mtime),
    reverse=True,
  )
  for run_dir in run_dirs:
    model_candidates = list(
      (run_dir / "export").glob("Casbot02-Leg-AMP-Flat_model_*.onnx")
    )
    if model_candidates:
      return max(
        model_candidates,
        key=lambda p: (_onnx_model_step(p), p.stat().st_mtime),
      )

    policy_path = run_dir / "policy.onnx"
    if policy_path.exists():
      return policy_path

  raise FileNotFoundError(
    "No CASBOT02 leg AMP ONNX found. Run play.py once, or pass an ONNX path."
  )


class OnnxPolicy:
  def __init__(self, model_path: str | Path, provider: str):
    try:
      import onnxruntime as ort
    except ModuleNotFoundError as exc:
      raise ModuleNotFoundError(
        "onnxruntime is not installed in this conda env. Install one of:\n"
        "  pip install onnxruntime\n"
        "  pip install onnxruntime-gpu"
      ) from exc

    available = ort.get_available_providers()
    if provider == "auto":
      providers = [
        p
        for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
        if p in available
      ] or available
    else:
      providers = [provider]

    self.session = ort.InferenceSession(str(model_path), providers=providers)
    self.input_name = self.session.get_inputs()[0].name
    self.output_name = self.session.get_outputs()[0].name
    input_shape = self.session.get_inputs()[0].shape
    output_shape = self.session.get_outputs()[0].shape
    self.input_dim = input_shape[1] if isinstance(input_shape[1], int) else None
    self.output_dim = output_shape[1] if isinstance(output_shape[1], int) else None
    print(
      f"[ONNX] loaded {model_path}\n"
      f"       providers={self.session.get_providers()}\n"
      f"       input={self.input_name}{input_shape}, "
      f"output={self.output_name}{output_shape}"
    )

  def __call__(self, obs: np.ndarray) -> np.ndarray:
    obs = np.ascontiguousarray(obs.astype(np.float32))
    action = self.session.run([self.output_name], {self.input_name: obs})[0]
    return np.asarray(action, dtype=np.float32).reshape(-1)


def quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
  """Rotate a world-frame vector into the body frame."""
  w, x, y, z = quat
  q_vec = -np.array([x, y, z], dtype=np.float64)
  t = 2.0 * np.cross(q_vec, vec)
  rotated = vec + w * t + np.cross(q_vec, t)
  return rotated


def build_model() -> mujoco.MjModel:
  entity = Entity(get_casbot02_23dof_robot_cfg())
  spec = entity.spec

  ground = spec.worldbody.add_geom(
    name="ground",
    type=mujoco.mjtGeom.mjGEOM_PLANE,
  )
  ground.size = [0.0, 0.0, 1.0]
  ground.condim = 4
  ground.friction = [0.9, 0.2, 0.2]
  ground.material = "matplane"

  model = spec.compile()
  model.opt.timestep = SIM_TIMESTEP
  model.opt.iterations = 10
  model.opt.ls_iterations = 20
  return model


def get_obs_frame(
  data: mujoco.MjData,
  command: np.ndarray,
  last_action: np.ndarray,
  default_joint_pos: np.ndarray,
) -> np.ndarray:
  base_ang_vel = np.asarray(
    data.sensor("angular-velocity").data, dtype=np.float64
  ).copy()
  quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
  projected_gravity = quat_apply_inverse_wxyz(
    quat_wxyz,
    np.array([0.0, 0.0, -1.0], dtype=np.float64),
  )
  joint_pos_rel = (
    np.asarray(data.qpos[7:], dtype=np.float64)[OBS_JOINT_INDICES]
    - default_joint_pos
  )
  joint_vel_rel = np.asarray(data.qvel[6:], dtype=np.float64)[OBS_JOINT_INDICES]
  obs_parts = [
    base_ang_vel,
    projected_gravity,
    command,
    joint_pos_rel,
    joint_vel_rel,
    last_action,
  ]
  obs = np.concatenate(obs_parts, axis=0)
  if obs.shape != (CURRENT_SINGLE_FRAME_OBS_SIZE,):
    raise RuntimeError(
      "Expected single-frame obs shape "
      f"({CURRENT_SINGLE_FRAME_OBS_SIZE},), got {obs.shape}"
    )
  return obs.astype(np.float32)


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
    data.qpos[0:3] = make_default_base_pos()
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = default_joint_pos
    data.qvel[:] = 0.0

  mujoco.mj_forward(model, data)


def ctrl_range(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
  lo = model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
  hi = model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
  return lo, hi


def make_action_scale() -> np.ndarray:
  """Return action scales in the current 20-joint policy order."""
  return np.array(
    [
      CASBOT02_22DOF_NO_WAIST_ACTION_SCALE[name]
      for name in CASBOT02_22DOF_NO_WAIST_JOINT_NAMES
    ],
    dtype=np.float64,
  )


def make_viewer(model: mujoco.MjModel, data: mujoco.MjData, mode: str):
  try:
    import mujoco_viewer
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "mujoco_viewer is not installed in this conda env. Install it with:\n"
      "  pip install mujoco-python-viewer\n"
      "or use the package name your RoboParty env used for `import mujoco_viewer`."
    ) from exc

  viewer = mujoco_viewer.MujocoViewer(
    model,
    data,
    mode=mode,
    width=DEFAULT_VIEWER_WIDTH,
    height=DEFAULT_VIEWER_HEIGHT,
  )
  viewer.cam.distance = 4.0
  viewer.cam.azimuth = 45.0
  viewer.cam.elevation = -20.0
  viewer.cam.lookat = [
    float(data.qpos[0]),
    float(data.qpos[1]),
    float(data.qpos[2]),
  ]
  return viewer


def install_safe_mouse_callback(viewer) -> None:
  import glfw

  original_mouse_callback = viewer._mouse_button_callback

  def mouse_button_callback(window, button, action, mods):
    try:
      original_mouse_callback(window, button, action, mods)
    except Exception as exc:
      print(f"[viewer] ignored mouse callback error: {exc}")
      if hasattr(viewer, "pert"):
        viewer.pert.active = 0

  glfw.set_mouse_button_callback(viewer.window, mouse_button_callback)


def install_command_controls(viewer, command: np.ndarray) -> None:
  import glfw

  def clamp(value: float, limits: tuple[float, float]) -> float:
    return float(np.clip(value, limits[0], limits[1]))

  def print_command() -> None:
    print(
      "[command] "
      f"vx={command[0]:+.2f}, vy={command[1]:+.2f}, yaw={command[2]:+.2f}"
    )

  command_keys = {
    glfw.KEY_UP,
    glfw.KEY_DOWN,
    glfw.KEY_LEFT,
    glfw.KEY_RIGHT,
    glfw.KEY_SPACE,
  }
  original_key_callback = viewer._key_callback

  def key_callback(window, key, scancode, action, mods):
    if key in command_keys and not (key == glfw.KEY_S and mods & glfw.MOD_CONTROL):
      if action in (glfw.PRESS, glfw.REPEAT):
        if key == glfw.KEY_UP:
          command[0] = clamp(command[0] + COMMAND_X_STEP, COMMAND_X_RANGE)
        elif key == glfw.KEY_DOWN:
          command[0] = clamp(command[0] - COMMAND_X_STEP, COMMAND_X_RANGE)
        elif key == glfw.KEY_LEFT:
          command[2] = clamp(command[2] + COMMAND_YAW_STEP, COMMAND_YAW_RANGE)
        elif key == glfw.KEY_RIGHT:
          command[2] = clamp(command[2] - COMMAND_YAW_STEP, COMMAND_YAW_RANGE)
        elif key == glfw.KEY_SPACE:
          command[:] = 0.0
        print_command()
      return

    original_key_callback(window, key, scancode, action, mods)

  glfw.set_key_callback(viewer.window, key_callback)

  original_create_overlay = viewer._create_overlay

  def create_overlay_with_command() -> None:
    original_create_overlay()
    gridpos = mujoco.mjtGridPos.mjGRID_TOPRIGHT
    if gridpos not in viewer._overlay:
      viewer._overlay[gridpos] = ["", ""]
    viewer._overlay[gridpos][0] += (
      "Command vx/vy/yaw\n"
      "Up/Down forward\n"
      "Left/Right yaw\n"
      "Space stop\n"
    )
    viewer._overlay[gridpos][1] += (
      f"{command[0]:+.2f}  {command[1]:+.2f}  {command[2]:+.2f}\n"
      f"+/- {COMMAND_X_STEP:.1f} m/s\n"
      f"+/- {COMMAND_YAW_STEP:.1f} rad/s\n"
      "zero command\n"
    )

  viewer._create_overlay = create_overlay_with_command
  print("[sim2sim] speed controls: Up/Down vx, Left/Right yaw, Space zero")


def run(model_arg: str = "") -> None:
  model_str = model_arg or DEFAULT_ONNX_MODEL
  if model_str:
    model_path = Path(model_str)
    if not model_path.is_absolute():
      model_path = REPO_ROOT / model_path
  else:
    model_path = find_latest_onnx()
  if not model_path.exists():
    raise FileNotFoundError(model_path)

  init_motion = Path(DEFAULT_INIT_MOTION) if DEFAULT_INIT_MOTION else None
  if init_motion is not None and not init_motion.exists():
    raise FileNotFoundError(init_motion)

  policy = OnnxPolicy(model_path, provider=DEFAULT_ONNXRUNTIME_PROVIDER)
  if policy.output_dim is not None and policy.output_dim != NUM_POLICY_ACTIONS:
    raise RuntimeError(
      f"Expected ONNX output dim {NUM_POLICY_ACTIONS}, got {policy.output_dim}. "
      "This sim2sim script expects the current 22-action no-waist policy."
    )
  input_dim = policy.input_dim or HISTORY_LENGTH * CURRENT_SINGLE_FRAME_OBS_SIZE
  if input_dim % CURRENT_SINGLE_FRAME_OBS_SIZE != 0:
    raise RuntimeError(
      "ONNX input dim must be a multiple of the current single-frame obs size. "
      f"Got input_dim={input_dim}, supported={CURRENT_SINGLE_FRAME_OBS_SIZE}."
    )
  single_frame_obs_size = CURRENT_SINGLE_FRAME_OBS_SIZE
  history_length = input_dim // single_frame_obs_size
  obs_layout = "leg_no_phase_12j_22a"
  model = build_model()
  if model.nu != NUM_FULL_JOINTS:
    raise RuntimeError(f"Expected {NUM_FULL_JOINTS} actuators, got {model.nu}")

  data = mujoco.MjData(model)
  default_joint_pos = make_default_joint_pos()          # 23-dim all MuJoCo joints
  default_obs_joint_pos = make_obs_joint_pos()          # 12-dim leg-only obs joints
  action_scale = make_action_scale()                    # 20-dim policy actions
  ctrl_lo, ctrl_hi = ctrl_range(model)
  command = np.array(
    [DEFAULT_COMMAND_X, DEFAULT_COMMAND_Y, DEFAULT_COMMAND_YAW],
    dtype=np.float64,
  )

  reset_state(model, data, default_joint_pos, init_motion)

  decimation = int(DEFAULT_DECIMATION)
  policy_dt = model.opt.timestep * decimation
  total_steps = int(DEFAULT_DURATION / model.opt.timestep)
  history: deque[np.ndarray] = deque(maxlen=history_length)
  last_action = np.zeros(NUM_OBS_JOINTS, dtype=np.float32)
  target_pos = default_joint_pos.copy()

  print(
    "[sim2sim] "
    f"dt={model.opt.timestep}, decimation={decimation}, "
    f"policy_hz={1.0 / policy_dt:.1f}, duration={DEFAULT_DURATION}s, "
    f"history_length={history_length}, single_frame={single_frame_obs_size}, "
    f"obs_layout={obs_layout}"
  )
  print(
    "[sim2sim] command="
    f"({DEFAULT_COMMAND_X:.3f}, {DEFAULT_COMMAND_Y:.3f}, {DEFAULT_COMMAND_YAW:.3f})"
  )
  if init_motion is not None:
    print(f"[sim2sim] initialized from: {init_motion}")

  viewer = make_viewer(model, data, DEFAULT_VIEWER_MODE)
  install_safe_mouse_callback(viewer)
  install_command_controls(viewer, command)
  next_time = time.perf_counter()
  try:
    for step in range(total_steps):
      if getattr(viewer, "is_alive", True) is False:
        break

      if step % decimation == 0:
        obs_frame = get_obs_frame(
          data,
          command,
          last_action,
          default_obs_joint_pos,
        )
        if not history:
          for _ in range(history_length):
            history.append(obs_frame)
        else:
          history.append(obs_frame)

        obs = np.concatenate(list(history), axis=0).reshape(1, -1)
        action = policy(obs)
        if action.shape != (NUM_POLICY_ACTIONS,):
          raise RuntimeError(
            f"Expected ONNX action shape ({NUM_POLICY_ACTIONS},), got {action.shape}"
          )

        last_action = action[:NUM_OBS_JOINTS].astype(np.float32)
        target_pos = default_joint_pos.copy()
        target_pos[ACTION_JOINT_INDICES] = (
          default_joint_pos[ACTION_JOINT_INDICES]
          + action.astype(np.float64) * action_scale
        )
        target_pos = np.clip(target_pos, ctrl_lo, ctrl_hi)

        viewer.cam.lookat = [
          float(data.qpos[0]),
          float(data.qpos[1]),
          float(data.qpos[2]),
        ]
        if (
          DEFAULT_PRINT_EVERY > 0
          and step % int(DEFAULT_PRINT_EVERY / model.opt.timestep) == 0
        ):
          print(
            f"t={step * model.opt.timestep:7.2f}s "
            f"base_z={data.qpos[2]:.3f} "
            f"action=[{action.min():+.3f},{action.max():+.3f}] "
            f"target=[{target_pos.min():+.3f},{target_pos.max():+.3f}]"
          )
          # 打印 23 个关节的实际位置（弧度），方便与真机对比
          actual_qpos = np.asarray(data.qpos[7:], dtype=np.float64)
          jp_str = "  ".join(
            f"{name}={actual_qpos[i]:+.4f}"
            for i, name in enumerate(CASBOT02_23DOF_JOINT_NAMES)
          )
          print(f"    joint_pos(rad): {jp_str}")

      data.ctrl[:] = target_pos
      mujoco.mj_step(model, data)
      if step % DEFAULT_RENDER_DECIMATION == 0:
        viewer.render()

      if DEFAULT_REALTIME:
        next_time += model.opt.timestep / max(DEFAULT_VIEW_SPEED, 1e-6)
        sleep_s = next_time - time.perf_counter()
        if sleep_s > 0:
          time.sleep(sleep_s)
        else:
          next_time = time.perf_counter()
  finally:
    viewer.close()

  print("[sim2sim] finished.")


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="CASBOT02 current lower-body AMP sim2sim with ONNX policy in MuJoCo viewer."
  )
  parser.add_argument(
    "model_path",
    nargs="?",
    type=str,
    default="",
    help="Optional ONNX model path. Empty uses the latest CASBOT02 leg AMP ONNX.",
  )
  return parser.parse_args()


if __name__ == "__main__":
  run(parse_args().model_path)
