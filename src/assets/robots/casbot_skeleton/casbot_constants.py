"""Casbot Skeleton (25 DOF) constants.

Joint structure:
  - 12 leg joints (pelvic_pitch/roll/yaw, knee_pitch, ankle_pitch/roll ×2)
  - 1 waist_yaw
  - 2 head (yaw, pitch)
  - 10 arm joints (shoulder_pitch/roll/yaw, elbow_pitch, wrist_yaw ×2)

Motor data from real URDF + user-provided armature values.
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

CASBOT_XML: Path = (
  SRC_PATH / "assets" / "robots" / "casbot_skeleton" / "xmls" / "casbot_skeleton_25dof.xml"
)
assert CASBOT_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, CASBOT_XML.parent.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(CASBOT_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config — real motor parameters.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

# ── Leg big: pelvic_pitch, pelvic_roll, knee_pitch ──
ARMATURE_LEG_BIG = 0.06999046
EFFORT_LEG_BIG = 150.0
VELOCITY_LEG_BIG = 14.0
STIFFNESS_LEG_BIG = ARMATURE_LEG_BIG * NATURAL_FREQ ** 2
DAMPING_LEG_BIG = 2.0 * DAMPING_RATIO * ARMATURE_LEG_BIG * NATURAL_FREQ

CASBOT_LEG_BIG_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_pelvic_pitch_joint",
    ".*_leg_pelvic_roll_joint",
    ".*_leg_knee_pitch_joint",
  ),
  stiffness=STIFFNESS_LEG_BIG,
  damping=DAMPING_LEG_BIG,
  effort_limit=EFFORT_LEG_BIG,
  armature=ARMATURE_LEG_BIG,
)

# ── Leg small: pelvic_yaw, ankle_pitch, ankle_roll ──
ARMATURE_LEG_SMALL = 0.03959369
EFFORT_LEG_SMALL = 60.0
VELOCITY_LEG_SMALL = 14.0
STIFFNESS_LEG_SMALL = ARMATURE_LEG_SMALL * NATURAL_FREQ ** 2
DAMPING_LEG_SMALL = 2.0 * DAMPING_RATIO * ARMATURE_LEG_SMALL * NATURAL_FREQ

CASBOT_LEG_SMALL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_pelvic_yaw_joint",
    ".*_leg_ankle_pitch_joint",
    ".*_leg_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_LEG_SMALL,
  damping=DAMPING_LEG_SMALL,
  effort_limit=EFFORT_LEG_SMALL,
  armature=ARMATURE_LEG_SMALL,
)

# ── Arm mid: shoulder_pitch, shoulder_roll, elbow_pitch ──
ARMATURE_ARM_MID = 0.03298028
EFFORT_ARM_MID = 75.0
VELOCITY_ARM_MID = 12.2
STIFFNESS_ARM_MID = ARMATURE_ARM_MID * NATURAL_FREQ ** 2
DAMPING_ARM_MID = 2.0 * DAMPING_RATIO * ARMATURE_ARM_MID * NATURAL_FREQ

CASBOT_ARM_MID_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_elbow_pitch_joint",
  ),
  stiffness=STIFFNESS_ARM_MID,
  damping=DAMPING_ARM_MID,
  effort_limit=EFFORT_ARM_MID,
  armature=ARMATURE_ARM_MID,
)

# ── Arm small: shoulder_yaw, wrist_yaw ──
ARMATURE_ARM_SMALL = 0.02452611
EFFORT_ARM_SMALL = 36.0
VELOCITY_ARM_SMALL = 9.3
STIFFNESS_ARM_SMALL = ARMATURE_ARM_SMALL * NATURAL_FREQ ** 2
DAMPING_ARM_SMALL = 2.0 * DAMPING_RATIO * ARMATURE_ARM_SMALL * NATURAL_FREQ

CASBOT_ARM_SMALL_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_yaw_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=STIFFNESS_ARM_SMALL,
  damping=DAMPING_ARM_SMALL,
  effort_limit=EFFORT_ARM_SMALL,
  armature=ARMATURE_ARM_SMALL,
)

# ── Waist: waist_yaw ──
CASBOT_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=STIFFNESS_LEG_SMALL,
  damping=DAMPING_LEG_SMALL,
  effort_limit=EFFORT_LEG_SMALL,
  armature=ARMATURE_LEG_SMALL,
)

# ── Head: head_yaw, head_pitch ──
CASBOT_HEAD_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_yaw_joint",
    "head_pitch_joint",
  ),
  stiffness=STIFFNESS_ARM_SMALL,
  damping=DAMPING_ARM_SMALL,
  effort_limit=EFFORT_ARM_SMALL,
  armature=ARMATURE_ARM_SMALL,
)

##
# Keyframe config.
##

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.87),
  joint_pos={
    ".*_leg_pelvic_pitch_joint": -0.15,
    ".*_leg_knee_pitch_joint": 0.3,
    ".*_leg_ankle_pitch_joint": -0.15,
    ".*_shoulder_pitch_joint": 0.2,
    ".*_elbow_pitch_joint": 0.5,
    "left_shoulder_roll_joint": 0.15,
    "right_shoulder_roll_joint": -0.15,
  },
  joint_vel={".*": 0.0},
)

KNEES_BENT_KEYFRAME = EntityCfg.InitialStateCfg(
  pos=(0, 0, 0.844),
  joint_pos={
    # 手臂
    ".*_elbow_pitch_joint": -0.35,
    "left_shoulder_roll_joint": 0.3,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.3,
    "right_shoulder_pitch_joint": 0.2,
    # 腿部微蹲
    ".*_leg_pelvic_pitch_joint": -0.32,
    ".*_leg_knee_pitch_joint": 0.53,
    ".*_leg_ankle_pitch_joint": -0.19,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

##修改xml碰撞体前的设置
# FULL_COLLISION = CollisionCfg(
#   geom_names_expr=(".*",),
#   condim={r"^(left|right)_foot_collision$": 3, ".*": 1},
#   priority={r"^(left|right)_foot_collision$": 1},
#   friction={r"^(left|right)_foot_collision$": (0.6,)},
# )



FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={r"^(left|right)_foot[1-7]_collision$": 3, ".*_collision": 1},
  priority={r"^(left|right)_foot[1-7]_collision$": 1},
  friction={r"^(left|right)_foot[1-7]_collision$": (0.6,)},
)

##
# Final config.
##

CASBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    CASBOT_LEG_BIG_ACTUATOR,
    CASBOT_LEG_SMALL_ACTUATOR,
    CASBOT_ARM_MID_ACTUATOR,
    CASBOT_ARM_SMALL_ACTUATOR,
    CASBOT_WAIST_ACTUATOR,
    CASBOT_HEAD_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_casbot_robot_cfg() -> EntityCfg:
  """Get a fresh Casbot Skeleton robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=CASBOT_ARTICULATION,
  )


CASBOT_ACTION_SCALE: dict[str, float] = {}
for a in CASBOT_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  if s == 0.0:
    continue  # skip passive joints (e.g. head)
  for n in names:
    CASBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_casbot_robot_cfg())

  # Print motor parameter summary
  print("=" * 65)
  print("Casbot Skeleton — Motor Configuration Summary")
  print("=" * 65)
  print(f"Natural Frequency: {NATURAL_FREQ:.2f} rad/s ({NATURAL_FREQ/(2*3.1415926535):.1f} Hz)")
  print(f"Damping Ratio:    {DAMPING_RATIO}")
  print()
  for label, cfg, arm, eff, vel in [
    ("Leg Big   ", CASBOT_LEG_BIG_ACTUATOR, ARMATURE_LEG_BIG, EFFORT_LEG_BIG, VELOCITY_LEG_BIG),
    ("Leg Small ", CASBOT_LEG_SMALL_ACTUATOR, ARMATURE_LEG_SMALL, EFFORT_LEG_SMALL, VELOCITY_LEG_SMALL),
    ("Arm Mid   ", CASBOT_ARM_MID_ACTUATOR, ARMATURE_ARM_MID, EFFORT_ARM_MID, VELOCITY_ARM_MID),
    ("Arm Small ", CASBOT_ARM_SMALL_ACTUATOR, ARMATURE_ARM_SMALL, EFFORT_ARM_SMALL, VELOCITY_ARM_SMALL),
    ("Waist     ", CASBOT_WAIST_ACTUATOR, ARMATURE_LEG_SMALL, EFFORT_LEG_SMALL, VELOCITY_LEG_SMALL),
    ("Head      ", CASBOT_HEAD_ACTUATOR, ARMATURE_ARM_SMALL, EFFORT_ARM_SMALL, VELOCITY_ARM_SMALL),
  ]:
    action_scale = 0.25 * eff / cfg.stiffness if cfg.stiffness > 0 else 0.0
    print(f"--- {label} ---")
    print(f"  armature:     {arm:.8f}")
    print(f"  stiffness:    {cfg.stiffness:.2f} Nm/rad")
    print(f"  damping:      {cfg.damping:.2f} Nm/(rad/s)")
    print(f"  effort_limit: {eff} Nm")
    print(f"  velocity:     {vel} rad/s")
    print(f"  action_scale: {action_scale:.4f} rad ({action_scale*180/3.14159:.1f}°)")
    print(f"  targets:      {cfg.target_names_expr}")
    print()

  viewer.launch(robot.spec.compile())
