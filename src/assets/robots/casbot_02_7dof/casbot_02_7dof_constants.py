"""Casbot 02 7DOF (27-DOF) robot constants.

27 controlled DOF: 12 legs + 1 waist + 14 arms (7 per arm).
Hand fingers and head are passive (not in actuator list).
"""

from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.os import update_assets

##
# MJCF and assets.
##

CASBOT_02_7DOF_XML: Path = (
    SRC_PATH / "assets" / "robots" / "casbot_02_7dof" / "xmls" / "casbot_02_7dof.xml"
)
assert CASBOT_02_7DOF_XML.exists(), f"XML not found: {CASBOT_02_7DOF_XML}"


def get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(
        assets,
        (CASBOT_02_7DOF_XML.parent / meshdir).resolve(),
        meshdir.rstrip("/"),
    )
    return assets


def get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(CASBOT_02_7DOF_XML))
    spec.assets = get_assets(spec.meshdir)

    # The source XML is a standalone MuJoCo scene. In mjlab the terrain
    # is provided by SceneCfg, so keep only the robot body tree.
    for geom in list(spec.geoms):
        if geom.name == "ground":
            spec.delete(geom)

    # Delete XML torque motors — mjlab creates position actuators below.
    for act in list(spec.actuators):
        spec.delete(act)

    return spec


##
# Actuator config — values verified against colleague's working deployment.
##

NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535  # 10 Hz
DAMPING_RATIO = 2.0


def _stiffness(armature: float) -> float:
    return armature * NATURAL_FREQ**2


def _damping(armature: float) -> float:
    return 2.0 * DAMPING_RATIO * armature * NATURAL_FREQ


# ── Leg heavy: pelvic_pitch, pelvic_roll, knee_pitch ──
CASBOT_LEG_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_leg_pelvic_pitch_joint",
        ".*_leg_pelvic_roll_joint",
        ".*_leg_knee_pitch_joint",
    ),
    stiffness=_stiffness(0.07),
    damping=_damping(0.07),
    effort_limit=120.0,
    armature=0.07,
    frictionloss=0.01,
)

# ── Leg light: pelvic_yaw, ankle_pitch, ankle_roll ──
CASBOT_LEG_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_leg_pelvic_yaw_joint",
        ".*_leg_ankle_pitch_joint",
        ".*_leg_ankle_roll_joint",
    ),
    stiffness=_stiffness(0.039),
    damping=_damping(0.039),
    effort_limit=80.0,
    armature=0.039,
    frictionloss=0.01,
)

# ── Waist: waist_yaw ──
CASBOT_WAIST_YAW_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_yaw_joint",),
    stiffness=_stiffness(0.0245),
    damping=_damping(0.0245),
    effort_limit=60.0,
    armature=0.0245,
    frictionloss=0.01,
)

# ── Arm heavy: shoulder_pitch, shoulder_roll, elbow_pitch ──
CASBOT_ARM_HEAVY_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_elbow_pitch_joint",
    ),
    stiffness=_stiffness(0.033),
    damping=_damping(0.033),
    effort_limit=40.0,
    armature=0.033,
    frictionloss=0.01,
)

# ── Arm light: shoulder_yaw, wrist_yaw, wrist_pitch, wrist_roll ──
CASBOT_ARM_LIGHT_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_shoulder_yaw_joint",
        ".*_wrist_yaw_joint",
        ".*_wrist_pitch_joint",
        ".*_wrist_roll_joint",
    ),
    stiffness=_stiffness(0.0245),
    damping=_damping(0.0245),
    effort_limit=30.0,
    armature=0.0245,
    frictionloss=0.01,
)

# ── Head: head_yaw, head_pitch (same as arm light) ──
CASBOT_HEAD_ACTUATOR = BuiltinPositionActuatorCfg(
    target_names_expr=(
        "head_yaw_joint",
        "head_pitch_joint",
    ),
    stiffness=_stiffness(0.0245),
    damping=_damping(0.0245),
    effort_limit=30.0,
    armature=0.0245,
    frictionloss=0.01,
)

##
# Keyframe config — matching colleague's HOME_KEYFRAME.
##

_HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.9),
    joint_pos={".*": 0.0},
    joint_vel={".*": 0.0},
)

##
# Final config.
##

CASBOT_02_7DOF_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        CASBOT_LEG_HEAVY_ACTUATOR,
        CASBOT_LEG_LIGHT_ACTUATOR,
        CASBOT_WAIST_YAW_ACTUATOR,
        CASBOT_ARM_HEAVY_ACTUATOR,
        CASBOT_ARM_LIGHT_ACTUATOR,
        CASBOT_HEAD_ACTUATOR,
    ),
    soft_joint_pos_limit_factor=0.9,
)


def get_casbot_02_7dof_robot_cfg() -> EntityCfg:
    """Get a fresh Casbot 02 7DOF robot configuration instance."""
    return EntityCfg(
        init_state=_HOME_KEYFRAME,
        spec_fn=get_spec,
        articulation=CASBOT_02_7DOF_ARTICULATION,
        sort_actuators=True,
    )


CASBOT_02_7DOF_ACTION_SCALE: dict[str, float] = {}
for a in CASBOT_02_7DOF_ARTICULATION.actuators:
    assert isinstance(a, BuiltinPositionActuatorCfg)
    e = a.effort_limit
    s = a.stiffness
    names = a.target_names_expr
    assert e is not None
    if s == 0.0:
        continue
    for n in names:
        CASBOT_02_7DOF_ACTION_SCALE[n] = 0.25 * e / s

# ── AMP body names (colleague-verified selection, 13 bodies) ──
CASBOT_02_7DOF_AMP_BODY_NAMES: tuple[str, ...] = (
    "base_link",
    "left_leg_pelvic_roll_link",
    "left_leg_knee_pitch_link",
    "left_leg_ankle_roll_link",
    "right_leg_pelvic_roll_link",
    "right_leg_knee_pitch_link",
    "right_leg_ankle_roll_link",
    "left_shoulder_roll_link",
    "left_elbow_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_pitch_link",
    "right_wrist_yaw_link",
)

# ── Joint name helpers (for CSV → NPZ conversion) ──
# 29 controlled DOF in XML actuator order
CASBOT_02_7DOF_JOINT_NAMES: list[str] = [
    # Left leg (6)
    "left_leg_pelvic_pitch_joint", "left_leg_pelvic_roll_joint", "left_leg_pelvic_yaw_joint",
    "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint", "left_leg_ankle_roll_joint",
    # Right leg (6)
    "right_leg_pelvic_pitch_joint", "right_leg_pelvic_roll_joint", "right_leg_pelvic_yaw_joint",
    "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint", "right_leg_ankle_roll_joint",
    # Waist (1)
    "waist_yaw_joint",
    # Head (2)
    "head_yaw_joint", "head_pitch_joint",
    # Left arm (7)
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_pitch_joint", "left_wrist_yaw_joint", "left_wrist_pitch_joint", "left_wrist_roll_joint",
    # Right arm (7)
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
    "right_elbow_pitch_joint", "right_wrist_yaw_joint", "right_wrist_pitch_joint", "right_wrist_roll_joint",
]


if __name__ == "__main__":
    print("=" * 60)
    print("Casbot 02 7DOF — Motor Configuration")
    print("=" * 60)
    print(f"Natural Frequency: {NATURAL_FREQ:.2f} rad/s")
    print(f"Damping Ratio:    {DAMPING_RATIO}")
    print(f"Controlled DOF:   {len(CASBOT_02_7DOF_JOINT_NAMES)}")
    print(f"Action scale entries: {len(CASBOT_02_7DOF_ACTION_SCALE)}")
    print()
    for label, cfg in [
        ("Leg Heavy", CASBOT_LEG_HEAVY_ACTUATOR),
        ("Leg Light", CASBOT_LEG_LIGHT_ACTUATOR),
        ("Waist    ", CASBOT_WAIST_YAW_ACTUATOR),
        ("Arm Heavy", CASBOT_ARM_HEAVY_ACTUATOR),
        ("Arm Light", CASBOT_ARM_LIGHT_ACTUATOR),
        ("Head     ", CASBOT_HEAD_ACTUATOR),
    ]:
        as_scale = 0.25 * cfg.effort_limit / cfg.stiffness
        print(f"--- {label}: stiffness={cfg.stiffness:.1f}, damping={cfg.damping:.2f}, "
              f"effort={cfg.effort_limit}, action_scale={as_scale:.4f}")
        print(f"    targets: {cfg.target_names_expr}")
