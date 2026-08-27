"""CASBOT02 AMP Locomotion environment configurations."""

import os

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp

from src.assets.robots import (
  CASBOT02_23DOF_ACTION_SCALE,
  CASBOT02_23DOF_AMP_BODY_NAMES,
  CASBOT02_23DOF_JOINT_NAMES,
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

  site_names = ("left_force", "right_force")
  feet_body_pattern = r"^(leg_l6_link|leg_r6_link)$"
  anchor_name = "torso"
  root_name = "torso"

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

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    feet_ground_cfg,
    self_collision_cfg,
  )

  if cfg.scene.terrain is not None and cfg.scene.terrain.terrain_generator is not None:
    cfg.scene.terrain.terrain_generator.curriculum = True

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = CASBOT02_23DOF_ACTION_SCALE

  cfg.viewer.body_name = "torso"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15
  twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
  # twist_cmd.ang_vel_deadband = 0.3  # 去掉 deadband，让策略学习全范围转弯命令

  # Randomize only the two foot collision geoms. The corresponding MJCF geoms
  # are explicitly named so this selection remains stable if geom order changes.
  cfg.events["foot_friction"] = EventTermCfg(
    mode="startup",
    func=envs_mdp.dr.geom_friction,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        geom_names=("left_foot_collision", "right_foot_collision"),
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
  cfg.events["actuator_effort_limits"] = EventTermCfg(
    mode="startup",
    func=amp_mdp.effort_limits_with_delayed_actuators,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot",
        actuator_ids=[0, 1],  # 腿部 actuator 组（LEG_HEAVY + LEG_LIGHT）
      ),
      "effort_limit_range": (0.85, 1.15),
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
      "ranges": (-0.5, 0.5),
      "operation": "add",
      "distribution": "uniform",
    },
  )
  # cfg.events["non_base_mass"] = EventTermCfg(
  #   mode="startup",
  #   func=envs_mdp.dr.body_mass,
  #   params={
  #     "asset_cfg": SceneEntityCfg(
  #       "robot", body_names=(r"^(?!torso$).+$",)
  #     ),
  #     "ranges": (0.9, 1.1),
  #     "operation": "scale",
  #     "distribution": "uniform",
  #     "shared_random": False,
  #   },
  # )
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
  #       0: (-WAIST_COM_BACKWARD_OFFSET, WAIST_COM_BACKWARD_OFFSET),  # 固定后偏
  #       1: (0.0, 0.0),
  #       2: (-0.2, 0.2),
  #     },
  #     "distribution": "uniform",
  #   },
  # )

  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

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
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir
  cfg.events["reset_from_motion"].params["asset_cfg"] = SceneEntityCfg(
    "robot",
    joint_names=CASBOT02_23DOF_JOINT_NAMES,
    preserve_order=True,
  )

  cfg.rewards["track_anchor_linear_velocity"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.rewards["track_anchor_linear_velocity"].weight = 1.5#1.5
  cfg.rewards["track_anchor_linear_velocity"].params["std"] = 0.5
  cfg.rewards["track_anchor_angular_velocity"].params[
    "anchor_cfg"
  ].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].weight = 1.5
  cfg.rewards["track_anchor_angular_velocity"].params["std"] = 0.5
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["foot_slip"].params["asset_cfg"].preserve_order = True
  cfg.rewards["foot_slip"].params["command_threshold"] = 0.2
  cfg.rewards["foot_slip"].weight = -0.5
  cfg.rewards["feet_air_time"] = RewardTermCfg(
    func=mdp.feet_air_time,
    weight=0.3,
    params={
      "sensor_name": "feet_ground_contact",
      "threshold_min": 0.05,
      "threshold_max": 0.5,
      "command_name": "twist",
      "command_threshold": 0.2,
    },
  )
  cfg.rewards["standing_feet_slip"] = RewardTermCfg(
    func=amp_mdp.standing_feet_slip,
    weight=-2.0,#-2
    params={
      "sensor_name": "feet_ground_contact",
      "command_name": "twist",
      "command_threshold": 0.2,
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=site_names,
        preserve_order=True,
      ),
    },
  )
  cfg.rewards["standing_foot_distance"] = RewardTermCfg(
    func=amp_mdp.standing_foot_distance,
    weight=-10.0,#-10
    params={
      "command_name": "twist",
      "command_threshold": 0.2,
      "target_lateral_distance": 0.285,
      "target_fore_distance": 0.0,
      "asset_cfg": SceneEntityCfg(
        "robot",
        site_names=site_names,
        preserve_order=True,
      ),
    },
  )
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  # cfg.rewards["flat_orientation_l2"] = RewardTermCfg(
  #   func=envs_mdp.flat_orientation_l2,
  #   weight=-1,#0.2
  #   params={"asset_cfg": SceneEntityCfg("robot")},
  # )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  # 关节级奖励/DR 只作用腿部关节（手臂由摆臂公式控制，不经网络）。
  # joint_acc_l2 保留两个摆臂肩关节（upper_left_1/upper_right_1），其余手臂去掉。
  _leg_swing_joints = CASBOT02_LEG_ONLY_JOINT_NAMES + (
    "upper_left_1_joint",
    "upper_right_1_joint",
  )
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=_leg_swing_joints, preserve_order=True
  )
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
  )
  # 力矩惩罚：只惩罚腿部 actuator 组（LEG_HEAVY + LEG_LIGHT），抑制髋 roll 极大力矩。
  # cfg.rewards["joint_torques_l2"] = RewardTermCfg(
  #   func=envs_mdp.joint_torques_l2,
  #   weight=-1e-7,
  #   params={
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       actuator_ids=[0, 1],  # 腿部 actuator 组（LEG_HEAVY + LEG_LIGHT）
  #     ),
  #   },
  # )

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

    cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.0

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

  if play:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    twist_cmd.ranges.lin_vel_x = (-1.0, 1.0)
    twist_cmd.ranges.lin_vel_y = (-0.3, 0.3)
    twist_cmd.ranges.ang_vel_z = (-1.6, 1.6)

  return cfg
