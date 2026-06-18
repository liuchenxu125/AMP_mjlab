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
# Actuator config — real motor parameters.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

# ── pelvic_pitch: arm=0.19897, eff=255, vel=13.72 rad/s (131 rpm) ──
ARMATURE_PELVIC_PITCH = 0.1989722
EFFORT_PELVIC_PITCH = 255.0
STIFFNESS_PELVIC_PITCH = ARMATURE_PELVIC_PITCH * NATURAL_FREQ ** 2
DAMPING_PELVIC_PITCH = 2.0 * DAMPING_RATIO * ARMATURE_PELVIC_PITCH * NATURAL_FREQ

MARATHON_PELVIC_PITCH_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_pelvic_pitch_joint",),
  stiffness=STIFFNESS_PELVIC_PITCH,
  damping=DAMPING_PELVIC_PITCH,
  effort_limit=EFFORT_PELVIC_PITCH,
  armature=ARMATURE_PELVIC_PITCH,
)

# ── pelvic_roll + knee_pitch: arm=0.27722, eff=212, vel=6.28 rad/s (60 rpm) ──
ARMATURE_PELVIC_ROLL_KNEE = 0.277224
EFFORT_PELVIC_ROLL_KNEE = 212.0
STIFFNESS_PELVIC_ROLL_KNEE = ARMATURE_PELVIC_ROLL_KNEE * NATURAL_FREQ ** 2
DAMPING_PELVIC_ROLL_KNEE = 2.0 * DAMPING_RATIO * ARMATURE_PELVIC_ROLL_KNEE * NATURAL_FREQ

MARATHON_PELVIC_ROLL_KNEE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_pelvic_roll_joint",
    ".*_leg_knee_pitch_joint",
  ),
  stiffness=STIFFNESS_PELVIC_ROLL_KNEE,
  damping=DAMPING_PELVIC_ROLL_KNEE,
  effort_limit=EFFORT_PELVIC_ROLL_KNEE,
  armature=ARMATURE_PELVIC_ROLL_KNEE,
)

# ── pelvic_yaw: arm=0.06117, eff=110, vel=15.18 rad/s (145 rpm) ──
ARMATURE_PELVIC_YAW = 0.06117
EFFORT_PELVIC_YAW = 110.0
STIFFNESS_PELVIC_YAW = ARMATURE_PELVIC_YAW * NATURAL_FREQ ** 2
DAMPING_PELVIC_YAW = 2.0 * DAMPING_RATIO * ARMATURE_PELVIC_YAW * NATURAL_FREQ

MARATHON_PELVIC_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_leg_pelvic_yaw_joint",),
  stiffness=STIFFNESS_PELVIC_YAW,
  damping=DAMPING_PELVIC_YAW,
  effort_limit=EFFORT_PELVIC_YAW,
  armature=ARMATURE_PELVIC_YAW,
)

# ── ankle + arms: arm=0.03298, eff=72, vel=12.15 rad/s (116 rpm) ──
ARMATURE_ANKLE_ARM = 0.03298
EFFORT_ANKLE_ARM = 72.0
STIFFNESS_ANKLE_ARM = ARMATURE_ANKLE_ARM * NATURAL_FREQ ** 2
DAMPING_ANKLE_ARM = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE_ARM * NATURAL_FREQ

MARATHON_ANKLE_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_ankle_pitch_joint",
    ".*_leg_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_ANKLE_ARM,
  damping=DAMPING_ANKLE_ARM,
  effort_limit=EFFORT_ANKLE_ARM,
  armature=ARMATURE_ANKLE_ARM,
)

MARATHON_ARM_ACTUATOR = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_elbow_pitch_joint",
  ),
  stiffness=STIFFNESS_ANKLE_ARM,
  damping=DAMPING_ANKLE_ARM,
  effort_limit=EFFORT_ANKLE_ARM,
  armature=ARMATURE_ANKLE_ARM,
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
    MARATHON_PELVIC_PITCH_ACTUATOR,
    MARATHON_PELVIC_ROLL_KNEE_ACTUATOR,
    MARATHON_PELVIC_YAW_ACTUATOR,
    MARATHON_ANKLE_ACTUATOR,
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

  print("=" * 65)
  print("Marathon_001 — Motor Configuration Summary")
  print("=" * 65)
  print(f"Natural Frequency: {NATURAL_FREQ:.2f} rad/s ({NATURAL_FREQ/(2*3.1415926535):.1f} Hz)")
  print(f"Damping Ratio:    {DAMPING_RATIO}")
  print()

  configs = [
    ("Pelvic Pitch  ", MARATHON_PELVIC_PITCH_ACTUATOR, ARMATURE_PELVIC_PITCH, EFFORT_PELVIC_PITCH),
    ("Pelvic Roll+Knee", MARATHON_PELVIC_ROLL_KNEE_ACTUATOR, ARMATURE_PELVIC_ROLL_KNEE, EFFORT_PELVIC_ROLL_KNEE),
    ("Pelvic Yaw    ", MARATHON_PELVIC_YAW_ACTUATOR, ARMATURE_PELVIC_YAW, EFFORT_PELVIC_YAW),
    ("Ankle         ", MARATHON_ANKLE_ACTUATOR, ARMATURE_ANKLE_ARM, EFFORT_ANKLE_ARM),
    ("Arms          ", MARATHON_ARM_ACTUATOR, ARMATURE_ANKLE_ARM, EFFORT_ANKLE_ARM),
  ]

  for label, cfg, arm, eff in configs:
    action_scale = 0.25 * eff / cfg.stiffness
    print(f"--- {label} ---")
    print(f"  armature:     {arm:.8f}")
    print(f"  stiffness:    {cfg.stiffness:.2f} Nm/rad")
    print(f"  damping:      {cfg.damping:.2f} Nm/(rad/s)")
    print(f"  effort_limit: {eff} Nm")
    print(f"  action_scale: {action_scale:.4f} rad ({action_scale*180/3.14159:.1f} deg)")
    print()

  viewer.launch(robot.spec.compile())
