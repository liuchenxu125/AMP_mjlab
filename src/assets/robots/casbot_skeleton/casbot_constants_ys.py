"""Casbot Skeleton (25 DOF) constants — G1 motor parameters.

Uses the exact same stiffness/damping/effort/armature as Unitree G1,
mapped to casbot joint names. For comparison testing with real motor params.
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
# Actuator config — G1 parameters (for comparison testing).
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0

# ── G1 7520_22: hip_pitch, hip_roll, knee → casbot leg_pelvic_pitch, leg_pelvic_roll, leg_knee_pitch ──
ARMATURE_7520_22 = 0.02510192
EFFORT_7520_22 = 139.0
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ ** 2   # 99.10
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ

CASBOT_G1_LEG_BIG = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_pelvic_pitch_joint",
    ".*_leg_pelvic_roll_joint",
    ".*_leg_knee_pitch_joint",
  ),
  stiffness=STIFFNESS_7520_22,
  damping=DAMPING_7520_22,
  effort_limit=EFFORT_7520_22,
  armature=ARMATURE_7520_22,
)

# ── G1 7520_14: hip_yaw, waist_yaw → casbot leg_pelvic_yaw, waist_yaw ──
ARMATURE_7520_14 = 0.01017752
EFFORT_7520_14 = 88.0
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ ** 2    # 40.18
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ

CASBOT_G1_LEG_SMALL = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_pelvic_yaw_joint",
    "waist_yaw_joint",
  ),
  stiffness=STIFFNESS_7520_14,
  damping=DAMPING_7520_14,
  effort_limit=EFFORT_7520_14,
  armature=ARMATURE_7520_14,
)

# ── G1 5020x2 (ANKLE): ankle_pitch, ankle_roll → casbot leg_ankle_pitch, leg_ankle_roll ──
ARMATURE_5020 = 0.00360972
ARMATURE_ANKLE = ARMATURE_5020 * 2                      # 0.00722
EFFORT_ANKLE = 50.0
STIFFNESS_ANKLE = ARMATURE_ANKLE * NATURAL_FREQ ** 2     # 28.50
DAMPING_ANKLE = 2.0 * DAMPING_RATIO * ARMATURE_ANKLE * NATURAL_FREQ

CASBOT_G1_ANKLE = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_leg_ankle_pitch_joint",
    ".*_leg_ankle_roll_joint",
  ),
  stiffness=STIFFNESS_ANKLE,
  damping=DAMPING_ANKLE,
  effort_limit=EFFORT_ANKLE,
  armature=ARMATURE_ANKLE,
)

# ── G1 5020: elbow, shoulder_* → casbot shoulder_pitch/roll/yaw, elbow_pitch ──
EFFORT_5020 = 25.0
STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ ** 2       # 14.25
DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ

CASBOT_G1_ARM_MID = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*_shoulder_pitch_joint",
    ".*_shoulder_roll_joint",
    ".*_shoulder_yaw_joint",
    ".*_elbow_pitch_joint",
  ),
  stiffness=STIFFNESS_5020,
  damping=DAMPING_5020,
  effort_limit=EFFORT_5020,
  armature=ARMATURE_5020,
)

# ── G1 5010_16: wrist_* → casbot wrist_yaw ──
ARMATURE_5010_16 = 0.00218120
EFFORT_5010_16 = 10.0
STIFFNESS_5010_16 = ARMATURE_5010_16 * NATURAL_FREQ ** 2  # 8.61
DAMPING_5010_16 = 2.0 * DAMPING_RATIO * ARMATURE_5010_16 * NATURAL_FREQ

CASBOT_G1_WRIST = BuiltinPositionActuatorCfg(
  target_names_expr=(".*_wrist_yaw_joint",),
  stiffness=STIFFNESS_5010_16,
  damping=DAMPING_5010_16,
  effort_limit=EFFORT_5010_16,
  armature=ARMATURE_5010_16,
)

# ── G1 4010: (unused in G1 29dof, used for head in casbot) ──
ARMATURE_4010 = 0.00425000
EFFORT_4010 = 5.0
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ ** 2         # 16.78
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

CASBOT_G1_HEAD = BuiltinPositionActuatorCfg(
  target_names_expr=(
    "head_yaw_joint",
    "head_pitch_joint",
  ),
  stiffness=STIFFNESS_4010,
  damping=DAMPING_4010,
  effort_limit=EFFORT_4010,
  armature=ARMATURE_4010,
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
    ".*_elbow_pitch_joint": -0.35,
    "left_shoulder_roll_joint": 0.3,
    "left_shoulder_pitch_joint": 0.2,
    "right_shoulder_roll_joint": -0.3,
    "right_shoulder_pitch_joint": 0.2,
    ".*_leg_pelvic_pitch_joint": -0.32,
    ".*_leg_knee_pitch_joint": 0.53,
    ".*_leg_ankle_pitch_joint": -0.19,
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

CASBOT_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    CASBOT_G1_LEG_BIG,
    CASBOT_G1_LEG_SMALL,
    CASBOT_G1_ANKLE,
    CASBOT_G1_ARM_MID,
    CASBOT_G1_WRIST,
    CASBOT_G1_HEAD,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_casbot_robot_cfg() -> EntityCfg:
  """Get a fresh Casbot Skeleton robot config with G1 motor params."""
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

  print("=" * 65)
  print("Casbot Skeleton — G1 Motor Parameters (Testing)")
  print("=" * 65)
  print(f"Natural Frequency: {NATURAL_FREQ:.2f} rad/s ({NATURAL_FREQ/(2*3.1415926535):.1f} Hz)")
  print(f"Damping Ratio:    {DAMPING_RATIO}")
  print()

  configs = [
    ("7520_22  ", CASBOT_G1_LEG_BIG,   ARMATURE_7520_22,  EFFORT_7520_22,  "leg_pelvic_pitch/roll, knee"),
    ("7520_14  ", CASBOT_G1_LEG_SMALL, ARMATURE_7520_14,  EFFORT_7520_14,  "leg_pelvic_yaw, waist_yaw"),
    ("5020x2   ", CASBOT_G1_ANKLE,     ARMATURE_ANKLE,    EFFORT_ANKLE,    "leg_ankle_pitch/roll"),
    ("5020     ", CASBOT_G1_ARM_MID,   ARMATURE_5020,     EFFORT_5020,     "shoulder_*, elbow"),
    ("5010_16  ", CASBOT_G1_WRIST,     ARMATURE_5010_16,  EFFORT_5010_16,  "wrist_yaw"),
    ("4010     ", CASBOT_G1_HEAD,      ARMATURE_4010,     EFFORT_4010,     "head_yaw/pitch"),
  ]

  for label, cfg, arm, eff, joints in configs:
    action_scale = 0.25 * eff / cfg.stiffness
    print(f"--- G1 {label} -> {joints} ---")
    print(f"  armature:     {arm:.8f}")
    print(f"  stiffness:    {cfg.stiffness:.2f} Nm/rad")
    print(f"  damping:      {cfg.damping:.3f} Nm/(rad/s)")
    print(f"  effort_limit: {eff} Nm")
    print(f"  action_scale: {action_scale:.4f} rad ({action_scale*180/3.14159:.1f} deg)")
    print()

  viewer.launch(robot.spec.compile())
