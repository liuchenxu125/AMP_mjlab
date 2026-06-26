"""Casbot Skeleton AMP Locomotion — Domain Randomization enhanced.

DR additions over the base ``env_cfgs.py``:
  [NEW]  full_body_friction  — 全身物理材质随机化 (摩擦+恢复)
  [NEW]  joint_default_pos   — 关节零位偏移 (模拟装配误差)
  [NEW]  mass_inertia        — 质量+惯性随机化 (物理一致, e^{2α}: 0.55×~4.95×)
  [ENH]  encoder_bias        — 范围扩大 ±0.015→±0.02 rad
  [ENH]  base_com            — COM偏移扩大 y:±0.025→±0.05, z:±0.03→±0.05
  [ENH]  push_robot          — 更频繁/更强推搡
  [ENH]  observation noise   — 噪声幅度加大
  [KEPT] foot_friction       — 脚部摩擦 (shared_random保证左右对称)
"""

import os

from src.assets.robots import (
  CASBOT_ACTION_SCALE,
  get_casbot_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg, RayCastSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from src.tasks.amp_loco.amp_env_cfg_dr import make_amp_env_cfg_dr


def casbot_amp_rough_env_cfg_dr(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Casbot Skeleton rough terrain AMP config with aggressive DR."""
  cfg = make_amp_env_cfg_dr()

  cfg.sim.mujoco.ccd_iterations = 128
  cfg.sim.contact_sensor_maxmatch = 128
  cfg.sim.nconmax = 48

  cfg.scene.entities = {"robot": get_casbot_robot_cfg()}

  for sensor in cfg.scene.sensors or ():
    if sensor.name == "terrain_scan":
      assert isinstance(sensor, RayCastSensorCfg)
      sensor.frame.name = "base_link"

  site_names = ("left_foot", "right_foot")
  geom_names = tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  )
  # [NEW] 全身geom名单 (用于full_body_friction, 不包括视觉geom)
  all_body_geom_names = tuple(
    f"{prefix}_{suffix}_collision"
    for prefix in ("left", "right")
    for suffix in ("hip", "hip_roll", "thigh", "shin", "ankle",
                   "upper_arm", "shoulder_roll", "shoulder_yaw", "forearm", "hand")
  ) + tuple(
    f"{side}_foot{i}_collision" for side in ("left", "right") for i in range(1, 8)
  ) + (
    "base_link_collision", "torso_collision",
    "head_yaw_collision", "head_pitch_collision",
  )

  body_names = (
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
  anchor_name = "waist_yaw_link"
  root_name = "base_link"

  feet_ground_cfg = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(
      mode="subtree",
      pattern=r"^(left_leg_ankle_roll_link|right_leg_ankle_roll_link)$",
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
    primary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="base_link", entity="robot"),
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
  joint_pos_action.scale = CASBOT_ACTION_SCALE

  cfg.viewer.body_name = "waist_yaw_link"

  twist_cmd = cfg.commands["twist"]
  assert isinstance(twist_cmd, UniformVelocityCommandCfg)
  twist_cmd.viz.z_offset = 1.15

  # ── [NEW] 全身摩擦: 所有碰撞geom ──
  cfg.events["full_body_friction"].params["asset_cfg"].geom_names = all_body_geom_names
  # ── foot_friction: 脚部14个胶囊 ──
  cfg.events["foot_friction"].params["asset_cfg"].geom_names = geom_names
  # ── [NEW] mass_inertia: 只对躯干 ──
  cfg.events["mass_inertia"].params["asset_cfg"].body_names = ("waist_yaw_link", "base_link")
  # ── COM偏移 ──
  cfg.events["base_com"].params["asset_cfg"].body_names = ("waist_yaw_link",)

  cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.4
  cfg.events["init_motion_loader"].params["max_delay_steps"] = 250

  _motion_base = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "assets", "motions", "casbot_skeledon", "amp"
  )
  _motion_dir = os.path.abspath(os.path.join(_motion_base, "WalkandRun"))
  _recovery_dir = os.path.abspath(os.path.join(_motion_base, "Recovery"))

  cfg.events["init_motion_loader"].params["motion_dir"] = _motion_dir
  cfg.events["init_motion_loader"].params["recovery_dir"] = _recovery_dir
  cfg.events["reset_from_motion"].params["motion_dir"] = _motion_dir

  cfg.rewards["track_anchor_linear_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["track_anchor_angular_velocity"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.rewards["foot_slip"].params["asset_cfg"].site_names = site_names
  cfg.rewards["self_collisions"] = RewardTermCfg(
    func=mdp.self_collision_cost,
    weight=-0.1,
    params={"sensor_name": self_collision_cfg.name, "force_threshold": 10.0},
  )
  cfg.rewards["body_ang_vel_xy_l2"].params["body_cfg"].body_names = (root_name,)

  for group in ("critic", "amp"):
    cfg.observations[group].terms["body_pos_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations[group].terms["body_pos_b"].params["body_cfg"].body_names = body_names
    cfg.observations[group].terms["body_ori_b"].params["anchor_cfg"].body_names = (anchor_name,)
    cfg.observations[group].terms["body_ori_b"].params["body_cfg"].body_names = body_names

  cfg.observations["amp"].terms["body_lin_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_lin_vel_b"].params["body_cfg"].body_names = body_names
  cfg.observations["amp"].terms["body_ang_vel_b"].params["anchor_cfg"].body_names = (anchor_name,)
  cfg.observations["amp"].terms["body_ang_vel_b"].params["body_cfg"].body_names = body_names

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


def casbot_amp_flat_env_cfg_dr(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Casbot Skeleton flat terrain AMP config with aggressive DR."""
  cfg = casbot_amp_rough_env_cfg_dr(play=play)

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
    twist_cmd.ranges.lin_vel_x = (-1.5, 3.0)
    twist_cmd.ranges.lin_vel_y = (-1.0, 1.0)
    twist_cmd.ranges.ang_vel_z = (-3.14 / 2, 3.14 / 2)

  return cfg
