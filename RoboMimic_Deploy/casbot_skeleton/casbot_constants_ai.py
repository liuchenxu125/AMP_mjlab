"""Casbot Skeleton (25 DOF) constants."""

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
# Actuator config.
#
# Casbot uses 25 actuators. Since we lack real motor datasheets, we use
# conservative estimates based on the XML default armature (0.01) and
# reasonable effort limits for each joint group.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

# Leg joints: 6 per leg (pitch, roll, yaw at pelvic; pitch at knee; pitch, roll at ankle).
# Higher armature and effort for weight-bearing joints.
ARMATURE_LEG = 0.01
STIFFNESS_LEG = ARMATURE_LEG * NATURAL_FREQ ** 2
DAMPING_LEG = 2.0 * DAMPING_RATIO * ARMATURE_LEG * NATURAL_FREQ
EFFORT_LEG = 80.0

CASBOT_LEG_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_pelvic_pitch_joint",
    ".*_leg_pelvic_roll_joint",
    ".*_leg_pelvic_yaw_joint",
    ".*_leg_knee_pitch_joint",
    ".*_leg_ankle_pitch_joint",
    ".*_leg_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_LEG,
  damping=DAMPING_LEG,
  effort_limit=EFFORT_LEG,
  armature=ARMATURE_LEG,
)

# Waist yaw: medium effort.
ARMATURE_WAIST = 0.005
STIFFNESS_WAIST = ARMATURE_WAIST * NATURAL_FREQ ** 2
DAMPING_WAIST = 2.0 * DAMPING_RATIO * ARMATURE_WAIST * NATURAL_FREQ
EFFORT_WAIST = 50.0

CASBOT_WAIST_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=("waist_yaw_joint",),
  stiffness=STIFFNESS_WAIST,
  damping=DAMPING_WAIST,
  effort_limit=EFFORT_WAIST,
  armature=ARMATURE_WAIST,
)

# Head + Arms: 2 head + 10 arm joints. Lower effort for upper body.
ARMATURE_ARM = 0.003
STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ ** 2
DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ
EFFORT_ARM = 20.0

CASBOT_ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_yaw_joint",
    "head_pitch_joint",
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
    ".*_wrist_yaw_joint",
  ),
  stiffness=STIFFNESS_ARM,
  damping=DAMPING_ARM,
  effort_limit=EFFORT_ARM,
  armature=ARMATURE_ARM,
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
    pos=(0, 0, 0.844),  # 
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
#
# Matches all geoms (visual meshes + named foot collision boxes).
# Foot collision geoms get condim=3 (full friction cone), everything
# else gets condim=1 (no friction).
##

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*",),
  condim={r"^(left|right)_foot_collision$": 3, ".*": 1},
  priority={r"^(left|right)_foot_collision$": 1},
  friction={r"^(left|right)_foot_collision$": (0.6,)},
)

##
# Final config.
##

CASBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    CASBOT_LEG_ACTUATOR,
    CASBOT_WAIST_ACTUATOR,
    CASBOT_ARM_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_casbot_robot_cfg() -> EntityCfg:
  """Get a fresh Casbot Skeleton robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
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
  for n in names:
    CASBOT_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_casbot_robot_cfg())

  viewer.launch(robot.spec.compile())
