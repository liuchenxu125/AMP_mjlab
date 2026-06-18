"""Marathon_001 (18 DOF) constants.

Joint structure:
  - 12 leg joints (pelvic_pitch/roll/yaw, knee_pitch, ankle_pitch/roll ×2)
  - 6 arm joints (shoulder_pitch/roll, elbow_pitch ×2)
  - No waist, no head, no wrist joints.
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

MARATHON_XML: Path = (
  SRC_PATH / "assets" / "robots" / "marathon_001" / "xmls" / "marathon_001.xml"
)
assert MARATHON_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, MARATHON_XML.parent.parent / "meshes", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(MARATHON_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

# Leg joints: 12 joints — higher effort for weight-bearing.
ARMATURE_LEG = 0.01
STIFFNESS_LEG = ARMATURE_LEG * NATURAL_FREQ ** 2
DAMPING_LEG = 2.0 * DAMPING_RATIO * ARMATURE_LEG * NATURAL_FREQ
EFFORT_LEG = 80.0

MARATHON_LEG_ACTUATOR = BuiltinPositionActuatorCfg(
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

# Arm joints: 6 joints — lower effort.
ARMATURE_ARM = 0.003
STIFFNESS_ARM = ARMATURE_ARM * NATURAL_FREQ ** 2
DAMPING_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ARM * NATURAL_FREQ
EFFORT_ARM = 20.0

MARATHON_ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_elbow_pitch_joint",
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
  pos=(0, 0, 0.82),
  joint_pos={
    ".*_leg_pelvic_pitch_joint": -0.35,
    ".*_leg_knee_pitch_joint": 0.7,
    ".*_leg_ankle_pitch_joint": -0.35,
    ".*_elbow_pitch_joint": 0.6,
    ".*_shoulder_pitch_joint": 0.2,
    "left_shoulder_roll_joint": 0.2,
    "right_shoulder_roll_joint": -0.2,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
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

MARATHON_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    MARATHON_LEG_ACTUATOR,
    MARATHON_ARM_ACTUATOR,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_marathon_robot_cfg() -> EntityCfg:
  """Get a fresh Marathon_001 robot configuration instance."""
  return EntityCfg(
    init_state=KNEES_BENT_KEYFRAME,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=MARATHON_ARTICULATION,
  )


MARATHON_ACTION_SCALE: dict[str, float] = {}
for a in MARATHON_ARTICULATION.actuators:
  assert isinstance(a, BuiltinPositionActuatorCfg)
  e = a.effort_limit
  s = a.stiffness
  names = a.target_names_expr
  assert e is not None
  for n in names:
    MARATHON_ACTION_SCALE[n] = 0.25 * e / s


if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_marathon_robot_cfg())

  viewer.launch(robot.spec.compile())
