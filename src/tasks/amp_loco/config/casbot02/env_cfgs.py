"""CASBOT02 AMP Locomotion environment configurations."""

import math
import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp as velocity_mdp

from src.assets.robots import (
  CASBOT02_23DOF_ACTION_SCALE,
  CASBOT02_23DOF_AMP_BODY_NAMES,
  CASBOT02_FOOT_GEOM_NAMES,
  CASBOT02_FOOT_SITE_NAMES,
  CASBOT02_LEG_ONLY_JOINT_NAMES,
  get_casbot02_23dof_robot_cfg,
)
import src.tasks.amp_loco.mdp as amp_mdp
from src.tasks.amp_loco.amp_env_cfg import make_amp_env_cfg
from src.tasks.velocity.mdp import UniformVelocityCommandCfg

# 训练时躯干(waist_yaw_link)质心后偏量(米),正值=向后,对齐真机质心 gap。
# 真机站立后倾、sim 前倾,说明真机上半身(头+双臂挂在 waist_yaw_link)质心比模型更靠后。
# 把 sim 的 waist_yaw_link 质心固定后移 3cm,让策略在训练时就学会往前压应对。
WAIST_COM_BACKWARD_OFFSET = 0.02


def _leg_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
  )


def _torso_asset() -> SceneEntityCfg:
  return SceneEntityCfg("robot", body_names=("torso",))


def _feet_body_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", body_names=("leg_l6_link", "leg_r6_link"), preserve_order=True
  )


def _feet_site_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot", site_names=CASBOT02_FOOT_SITE_NAMES, preserve_order=True
  )


def _knee_body_asset() -> SceneEntityCfg:
  return SceneEntityCfg(
    "robot",
    body_names=("leg_l4_link", "leg_l3_link", "leg_r4_link", "leg_r3_link"),
    preserve_order=True,
  )


def _apply_casbot02_loco_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
  """Replace AMP task rewards with the HANDOFF Casbot02 loco-teacher stack.

  Velocity tracking uses mjlab body-frame root velocity (same formula as
  loco). Swing-height and pose terms are omitted so AMP style is not fought
  by an explicit gait/posture prior. Remaining foot terms still use the
  loco sole sites ``left_foot`` / ``right_foot``.
  """
  rewards = cfg.rewards
  for name in (
    "body_ang_vel_xy_l2",
    "is_terminated",
    "joint_acc_l2",
    "joint_pos_limits",
    "standing_feet_slip",
    "standing_foot_distance",
    "feet_air_time",
    "flat_orientation_l2",
    "undesired_contacts",
    "foot_clearance",
    "track_anchor_linear_velocity",
    "track_anchor_angular_velocity",
    "pose",
    "swing_height_curve",
    "foot_swing_height",
  ):
    rewards.pop(name, None)

  rewards["track_linear_velocity"] = RewardTermCfg(
    func=velocity_mdp.track_linear_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": 0.5, "asset_cfg": _torso_asset()},
  )
  rewards["track_angular_velocity"] = RewardTermCfg(
    func=velocity_mdp.track_angular_velocity,
    weight=2.0,
    params={"command_name": "twist", "std": 0.7071, "asset_cfg": _torso_asset()},
  )
  rewards["upright"] = RewardTermCfg(
    func=velocity_mdp.flat_orientation,
    weight=1.0,
    params={"std": math.sqrt(0.2), "asset_cfg": _torso_asset()},
  )
  rewards["body_ang_vel"] = RewardTermCfg(
    func=velocity_mdp.body_angular_velocity_penalty,
    weight=-0.05,
    params={"asset_cfg": _torso_asset()},
  )
  rewards["angular_momentum"] = RewardTermCfg(
    func=velocity_mdp.angular_momentum_penalty,
    weight=-0.02,
    params={"sensor_name": "robot/root_angmom"},
  )
  rewards["dof_pos_limits"] = RewardTermCfg(
    func=envs_mdp.joint_pos_limits,
    weight=-1.0,
    params={"asset_cfg": _leg_asset()},
  )
  rewards["action_rate_l2"].weight = -0.1

  rewards["air_time"] = RewardTermCfg(
    func=amp_mdp.command_conditioned_feet_air_time,
    weight=1.0,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "translation_threshold_max": 0.65,
      "turning_threshold_max": 0.42,
      "command_name": "twist",
      "command_threshold": 0.2,
      "turning_linear_threshold": 0.2,
      "turning_angular_threshold": 0.2,
    },
  )
  rewards["foot_slip"].func = velocity_mdp.feet_slip
  rewards["foot_slip"].weight = -2.0
  rewards["foot_slip"].params["asset_cfg"] = _feet_site_asset()
  rewards["foot_slip"].params["command_threshold"] = 0.05
  rewards["soft_landing"].func = velocity_mdp.soft_landing
  rewards["soft_landing"].weight = -6e-3
  rewards["soft_landing"].params["command_threshold"] = 0.05

  rewards["stand_pose"] = RewardTermCfg(
    func=amp_mdp.stand_pose,
    weight=-4.0,
    params={"command_name": "twist", "asset_cfg": _leg_asset()},
  )
  # rewards["feet_distance_lateral"] = RewardTermCfg(
  #   func=amp_mdp.feet_distance_lateral,
  #   weight=2.5,
  #   params={
  #     "asset_cfg": _feet_site_asset(),
  #     "min_distance": 0.266,
  #     "max_distance": 0.40,
  #   },
  # )
  # rewards["knee_distance_lateral"] = RewardTermCfg(
  #   func=amp_mdp.knee_distance_lateral,
  #   weight=2.5,
  #   params={
  #     "asset_cfg": _knee_body_asset(),
  #     "min_distance": 0.279,
  #     "max_distance": 0.32,
  #   },
  # )
  # rewards["flat_foot"] = RewardTermCfg(
  #   func=amp_mdp.flat_foot,
  #   weight=-0.5,
  #   params={
  #     "sensor_name": "feet_ground_contact",
  #     "asset_cfg": _feet_body_asset(),
  #   },
  # )
  rewards["self_collisions"] = RewardTermCfg(
    func=velocity_mdp.self_collision_cost,
    weight=-1.0,
    params={"sensor_name": "self_collision", "force_threshold": 10.0},
  )


def _apply_casbot02_twist(cfg: ManagerBasedRlEnvCfg) -> None:
  """HANDOFF loco teacher command sampling and velocity curriculum.

  AMP style already covers rest-to-walk, so the old 10% stand-then-go lane
  is folded into the dedicated forward cohort (0.2 → 0.3), matching loco.
  """
  base_twist_cmd = cfg.commands["twist"]
  assert isinstance(base_twist_cmd, UniformVelocityCommandCfg)
  vx, vy, wz = (-1.0, 1.0), (0.0, 0.0), (-1.0, 1.0)
  twist_cmd = amp_mdp.Casbot02VelocityCommandCfg(
    resampling_time_range=base_twist_cmd.resampling_time_range,
    debug_vis=base_twist_cmd.debug_vis,
    entity_name=base_twist_cmd.entity_name,
    heading_command=True,
    heading_control_stiffness=base_twist_cmd.heading_control_stiffness,
    rel_standing_envs=0.1,
    rel_turning_envs=0.2,
    rel_backward_envs=0.0,
    rel_stand_then_go_envs=0.0,
    rel_heading_envs=0.2,
    rel_world_envs=0.0,
    rel_forward_envs=0.3,
    init_velocity_prob=0.0,
    min_turning_ang_vel=0.2,
    ranges=amp_mdp.Casbot02VelocityCommandCfg.Ranges(
      lin_vel_x=vx,
      lin_vel_y=vy,
      ang_vel_z=wz,
      heading=(-math.pi, math.pi),
    ),
    viz=base_twist_cmd.viz,
  )
  twist_cmd.viz.z_offset = 1.15
  cfg.commands["twist"] = twist_cmd
  cfg.curriculum["command_vel"] = CurriculumTermCfg(
    func=velocity_mdp.commands_vel,
    params={
      "command_name": "twist",
      "velocity_stages": [
        {
          "step": 0,
          "lin_vel_x": (vx[0] * 0.5, vx[1] * 0.5),
          "lin_vel_y": vy,
          "ang_vel_z": (wz[0] * 0.5, wz[1] * 0.5),
        },
        {
          "step": 5000 * 24,
          "lin_vel_x": vx,
          "lin_vel_y": vy,
          "ang_vel_z": wz,
        },
      ],
    },
  )


def _apply_casbot02_push(cfg: ManagerBasedRlEnvCfg) -> None:
  """Match loco / mjlab default interval push (planar + vertical + tilt)."""
  push_robot = cfg.events["push_robot"]
  push_robot.interval_range_s = (1.0, 3.0)
  push_robot.params["velocity_range"] = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.4, 0.4),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
  }


def _apply_casbot02_reset(cfg: ManagerBasedRlEnvCfg) -> None:
  """Stand from default pose like the HANDOFF loco teacher, not motion frames."""
  cfg.events.pop("reset_from_motion", None)
  cfg.events["reset_base"] = EventTermCfg(
    func=envs_mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.01, 0.05),
        "yaw": (-3.14, 3.14),
      },
      "velocity_range": {},
    },
  )
  cfg.events["reset_robot_joints"] = EventTermCfg(
    func=envs_mdp.reset_joints_by_offset,
    mode="reset",
    params={
      "position_range": (0.0, 0.0),
      "velocity_range": (0.0, 0.0),
      "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
    },
  )


def _add_casbot02_phase_observation(group) -> None:
  terms = dict(group.terms)
  terms["phase"] = ObservationTermCfg(
    func=amp_mdp.phase,
    params={"period": 1.0, "command_name": "twist"},
  )
  order = ("base_ang_vel", "projected_gravity", "command", "phase")
  ordered_terms = {name: terms.pop(name) for name in order if name in terms}
  ordered_terms.update(terms)
  group.terms = ordered_terms


def casbot02_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create CASBOT02 rough terrain velocity configuration."""
  cfg = make_amp_env_cfg()

  cfg.observations["actor"].history_length = 4
  cfg.observations["critic"].history_length = 4
  _add_casbot02_phase_observation(cfg.observations["actor"])
  _add_casbot02_phase_observation(cfg.observations["critic"])

  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 64

  cfg.scene.entities = {"robot": get_casbot02_23dof_robot_cfg()}

  cfg.observations["actor"].terms["base_ang_vel"].params[
    "sensor_name"
  ] = "robot/angular-velocity"
  cfg.observations["critic"].terms["base_ang_vel"].params[
    "sensor_name"
  ] = "robot/angular-velocity"
  cfg.observations["critic"].terms["base_lin_vel"].params[
    "sensor_name"
  ] = "robot/linear-velocity"

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "torso"

  feet_body_pattern = r"^(leg_l6_link|leg_r6_link)$"
  anchor_name = "torso"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=feet_body_pattern,
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
  )

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="torso", entity="robot"),
    fields=("found", "force"),
    reduce="none",
    num_slots=1,
    history_length=4,
  )

  # Body-level (not subtree) so foot children of thigh/torso are not included.
  # Exclude ankle pitch (leg_*5) and ankle roll (leg_*6), matching RoboParty's
  # "anything with ankle in the name" filter.
  non_foot_ground_cfg = ContactSensorCfg(
    name="non_foot_ground_contact",
    primary=ContactMatch(
      mode="body",
      pattern=r".*",
      entity="robot",
      exclude=("leg_l5_link", "leg_r5_link", "leg_l6_link", "leg_r6_link"),
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="maxforce",
    num_slots=1,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
    non_foot_ground_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = CASBOT02_23DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  # Randomize only the two foot collision geoms. The corresponding MJCF geoms
  # are explicitly named so this selection remains stable if geom order changes.
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        geom_names=CASBOT02_FOOT_GEOM_NAMES,
        preserve_order=True,
      ),
      "operation": "abs",
      "ranges": (0.3, 1.3),
      "shared_random": True,
    },
  )
  cfg.events["actuator_gains"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.pd_gains,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        actuator_ids=[0, 1],  # 腿部 actuator 组（LEG_HEAVY + LEG_LIGHT）
      ),
      "kp_range": (0.85, 1.15),
      "kd_range": (0.85, 1.15),
      "operation": "scale",
      "distribution": "uniform",
    },
  )
  cfg.events["joint_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.joint_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        joint_names=CASBOT02_LEG_ONLY_JOINT_NAMES,
        preserve_order=True,
      ),
      "ranges": (0.5, 1.5),
      "operation": "scale",
      "distribution": "uniform",
    },
  )
  # cfg.events["joint_armature"] = EventTermCfg(
  #   mode="startup",
  #   func=envs_mdp.dr.joint_armature,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot"),
  #     "ranges": (0.95, 1.05),
  #     "operation": "scale",
  #     "distribution": "uniform",
  #     "shared_random": False,
  #   },
  # )
  cfg.events["base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      # The source URDF base_link is named torso in the MuJoCo model.
      "asset_cfg": SceneEntityCfg("robot", body_names=("torso",)),
      "ranges": (-4.0, 4.0),
      "operation": "add",
      "distribution": "uniform",
    },
  )
  cfg.events["non_base_mass"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.body_mass,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", body_names=(r"^(?!torso$).+$",)
      ),
      "ranges": (0.80, 1.20),
      "operation": "scale",
      "distribution": "uniform",
      "shared_random": False,
    },
  )
  # cfg.events["joint_default_pos"].params["ranges"] = (-0.02, 0.02)
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso",)
  # cfg.events["torso_mass"].params["asset_cfg"].body_names = ("waist_yaw_link",)
  # 躯干(waist_yaw_link)质心前后移,对齐真机 gap。承载头+双臂,是上半身质量大头。
  # cfg.events["waist_com_backward"] = EventTermCfg(
  #   mode="startup",
  #   func=envs_mdp.dr.body_com_offset,
  #   params={
  #     "asset_cfg": SceneEntityCfg("robot", body_names=("waist_yaw_link",)),
  #     "operation": "add",
  #     "ranges": {
  #       0: (-WAIST_COM_BACKWARD_OFFSET, -WAIST_COM_BACKWARD_OFFSET),  # 固定后偏
  #       1: (0.0, 0.0),
  #       2: (0.0, 0.0),
  #     },
  #     "distribution": "uniform",
  #   },
  # )

  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 0

  _motion_base = os.path.join(
    os.path.dirname(__file__),
    "..",
    "..",
    "..",
    "..",
    "assets",
    "motions",
    "casbot02",
    "amp",
  )
  _motion_dir = os.path.abspath(
    os.path.join(_motion_base, "WalkandRun")
  )

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = None

  _apply_casbot02_loco_rewards(cfg)
  _apply_casbot02_twist(cfg)
  _apply_casbot02_push(cfg)
  _apply_casbot02_reset(cfg)

  cfg.observations["critic"].terms["body_pos_b"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_pos_b"].params[
    "body_cfg"
  ].body_names = CASBOT02_23DOF_AMP_BODY_NAMES

  cfg.observations["critic"].terms["body_ori_b"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.observations["critic"].terms["body_ori_b"].params[
    "body_cfg"
  ].body_names = CASBOT02_23DOF_AMP_BODY_NAMES

  cfg.observations["amp"].terms["body_pos_b"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_pos_b"].params[
    "body_cfg"
  ].body_names = CASBOT02_23DOF_AMP_BODY_NAMES

  cfg.observations["amp"].terms["body_ori_b"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ori_b"].params[
    "body_cfg"
  ].body_names = CASBOT02_23DOF_AMP_BODY_NAMES

  cfg.observations["amp"].terms["body_lin_vel_b"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params[
    "body_cfg"
  ].body_names = CASBOT02_23DOF_AMP_BODY_NAMES

  cfg.observations["amp"].terms["body_ang_vel_b"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params[
    "body_cfg"
  ].body_names = CASBOT02_23DOF_AMP_BODY_NAMES

  if play:
    cfg.episode_length_s = int(1e9)

    cfg.observations["actor"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    cfg.curriculum = {}
    cfg.events["randomize_terrain"] = EventTermCfg(
      func=envs_mdp.randomize_terrain,
      mode="reset",
      params={},
    )

    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0

  return cfg


def casbot02_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create CASBOT02 flat terrain velocity configuration."""
  cfg = casbot02_amp_rough_env_cfg(play=play)

  cfg.sim.njmax = 640
  cfg.sim.mujoco.ccd_iterations = 50
  cfg.sim.contact_sensor_maxmatch = 256
  cfg.sim.nconmax = None

  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None

  cfg.scene.sensors = tuple(
    s for s in (cfg.scene.sensors or ()) if s.name != "terrain_scan"
  )

  return cfg
