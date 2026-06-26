"""AMP task configuration — Domain Randomization enhanced version.

Adds over the base ``amp_env_cfg.py``:
  - Full-body friction + restitution randomization
  - Joint default position randomization (simulates assembly zero-offset errors)
  - Mass + inertia randomization via pseudo_inertia (physically consistent)
  - Aggressive push perturbation
  - Stronger observation noise
  - Wider COM offset ranges
"""

import math
from dataclasses import replace

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.terrains.config import ROUGH_TERRAINS_CFG
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

import src.tasks.amp_loco.mdp as mdp
from src.tasks.amp_loco.mdp.terrain import RANDOM_ROUGH_TERRAINS_CFG


def make_amp_env_cfg_dr() -> ManagerBasedRlEnvCfg:
  """Create AMP locomotion config with aggressive domain randomization."""

  ##
  # Sensors (same as base)
  ##

  terrain_scan = RayCastSensorCfg(
    name="terrain_scan",
    frame=ObjRef(type="body", name="", entity="robot"),
    ray_alignment="yaw",
    pattern=GridPatternCfg(size=(1.6, 1.0), resolution=0.1),
    max_distance=5.0,
    exclude_parent_body=True,
    debug_vis=True,
    viz=RayCastSensorCfg.VizCfg(show_normals=True),
  )

  ##
  # Observations — stronger noise for sim-to-real
  ##

  actor_terms = {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.3, n_max=0.3),    # ↑ ±0.2 → ±0.3
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.08, n_max=0.08),   # ↑ ±0.05 → ±0.08
    ),
    "command": ObservationTermCfg(
      func=mdp.generated_commands,
      params={"command_name": "twist"},
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.02, n_max=0.02),   # ↑ ±0.01 → ±0.02
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.0, n_max=1.0),     # ↑ ±0.5 → ±1.0
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }

  critic_terms = {
    **actor_terms,
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "body_pos_b": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_ori_b": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  amp_terms = {
    "body_pos_b": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_ori_b": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_lin_vel_b": ObservationTermCfg(
      func=mdp.robot_body_lin_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
    "body_ang_vel_b": ObservationTermCfg(
      func=mdp.robot_body_ang_vel_b,
      params={
        "anchor_cfg": SceneEntityCfg("robot", body_names=()),
        "body_cfg": SceneEntityCfg("robot", body_names=()),
      },
    ),
  }

  observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms, concatenate_terms=True,
      enable_corruption=True, history_length=4, history_ordering="time",
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms, concatenate_terms=True,
      enable_corruption=False, history_length=4, history_ordering="time",
    ),
    "amp": ObservationGroupCfg(
      terms=amp_terms, concatenate_terms=True,
      enable_corruption=False, history_length=1,
    ),
  }

  ##
  # Actions (same as base)
  ##

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.25,
      use_default_offset=True,
    )
  }

  ##
  # Commands (same as base)
  ##

  commands: dict[str, CommandTermCfg] = {
    "twist": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(3.0, 8.0),
      rel_standing_envs=0.05,
      rel_heading_envs=0.25,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=(-1.5, 3.0),
        lin_vel_y=(-1.0, 1.0),
        ang_vel_z=(-3.14 / 2, 3.14 / 2),
        heading=(-math.pi / 2, math.pi / 2),
      ),
    )
  }

  ##
  # Events — aggressive domain randomization
  ##

  events = {
    # ── Motion reset (same as base) ──
    "init_motion_loader": EventTermCfg(
      func=mdp.init_motion_loader,
      mode="startup",
      params={
        "motion_dir": "",
        "recovery_dir": None,
        "delay_reset_env_ratio": 0.0,
        "max_delay_steps": 0,
      },
    ),
    "reset_from_motion": EventTermCfg(
      func=mdp.reset_from_motion_data,
      mode="reset",
      params={
        "motion_dir": "",
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
      },
    ),

    # ── [NEW] 全身物理材质随机化 ──
    "full_body_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # per-robot fills
        "operation": "abs",
        "ranges": (0.3, 1.6),             # 0.3~1.6 (wider than feet-only)
        "shared_random": False,            # each body independent
      },
    ),
    # ── [NEW] 脚部摩擦保留 (shared_random保证对称) ──
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),  # per-robot fills
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    # ── [ENHANCED] 编码器偏置 (范围扩大) ──
    "encoder_bias": EventTermCfg(
      mode="startup",
      func=dr.encoder_bias,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "bias_range": (-0.02, 0.02),      # ↑ ±0.015 → ±0.02
      },
    ),
    # ── [NEW] 关节默认位置随机化 ──
    "joint_default_pos": EventTermCfg(
      mode="startup",
      func=dr.joint_default_pos,
      params={
        "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
        "ranges": (-0.015, 0.015),        # ±0.015 rad zero-offset error
        "distribution": "uniform",
        "operation": "add",
      },
    ),
    # ── [NEW] 质量+惯性随机化 (物理一致) ──
    "mass_inertia": EventTermCfg(
      mode="startup",
      func=dr.pseudo_inertia,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),  # per-robot fills
        "alpha_range": (-0.30, 0.80),     # mass scale e^{2α}: 0.55× ~ 4.95×
        "distribution": "uniform",
      },
    ),
    # ── [ENHANCED] COM偏移 (范围扩大) ──
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.04, 0.04),               # ↑ x: ±0.025 → ±0.04
          1: (-0.05, 0.05),               # ↑ y: ±0.025 → ±0.05
          2: (-0.05, 0.05),               # ↑ z: ±0.03  → ±0.05
        },
      },
    ),
    # ── [ENHANCED] 更激进的外力推搡 ──
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(0.8, 2.5),        # ↑ 更频繁: 1~3s → 0.8~2.5s
      params={
        "velocity_range": {
          "x": (-1.0, 1.0),
          "y": (-0.75, 0.75),             # ↑ ±0.5 → ±0.75
          "z": (-0.5, 0.5),               # ↑ ±0.4 → ±0.5
          "roll": (-0.78, 0.78),          # ↑ ±0.52 → ±0.78
          "pitch": (-0.78, 0.78),         # ↑ ±0.52 → ±0.78
          "yaw": (-1.17, 1.17),           # ↑ ±0.78 → ±1.17
        },
      },
    ),
  }

  ##
  # Rewards (same as base)
  ##

  rewards = {
    "track_anchor_linear_velocity": RewardTermCfg(
      func=mdp.track_anchor_linear_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": 1.0,
              "mask_delay": True, "delay_env_rew_ratio": 0.0,
              "anchor_cfg": SceneEntityCfg("robot", body_names=()),},
    ),
    "track_anchor_angular_velocity": RewardTermCfg(
      func=mdp.track_anchor_angular_velocity,
      weight=1.0,
      params={"command_name": "twist", "std": 3.14,
              "mask_delay": True, "delay_env_rew_ratio": 0.0,
              "anchor_cfg": SceneEntityCfg("robot", body_names=()),},
    ),
    "track_root_height": RewardTermCfg(
      func=mdp.track_root_height,
      weight=1.0,
      params={"std": 0.3, "mask_delay": True, "delay_env_rew_ratio": 3.5},
    ),
    "body_ang_vel_xy_l2": RewardTermCfg(
      func=mdp.body_ang_vel_xy_l2,
      weight=0.5,
      params={"std": 3.14, "mask_delay": True, "delay_env_rew_ratio": 0.0,
              "body_cfg": SceneEntityCfg("robot", body_names=("pelvis",)),},
    ),
    "is_terminated": RewardTermCfg(func=mdp.is_terminated, weight=-200.0),
    "joint_acc_l2": RewardTermCfg(func=mdp.joint_acc_l2, weight=-2.5e-7),
    "joint_pos_limits": RewardTermCfg(func=mdp.joint_pos_limits, weight=-10.0),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-0.01),
    "foot_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.25,
      params={"sensor_name": "feet_ground_contact", "command_name": "twist",
              "command_threshold": 0.1,
              "asset_cfg": SceneEntityCfg("robot", site_names=()),},
    ),
    "self_collisions": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=-0.1,
      params={"sensor_name": "self_collision", "force_threshold": 10.0},
    ),
  }

  ##
  # Terminations (same as base)
  ##

  terminations = {
    "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
    "bad_orientation": TerminationTermCfg(
      func=mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0)},
    ),
    "bad_base_height": TerminationTermCfg(
      func=mdp.root_height_below_minimum,
      params={"minimum_height": 0.5,},
    ),
  }

  ##
  # Assemble
  ##

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      terrain=TerrainEntityCfg(
        terrain_type="generator",
        terrain_generator=replace(ROUGH_TERRAINS_CFG),
        max_init_terrain_level=5,
      ),
      sensors=(terrain_scan,),
      num_envs=1,
      extent=2.0,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    rewards=rewards,
    terminations=terminations,
    curriculum={},
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot", body_name="",
      distance=3.0, elevation=-5.0, azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=35, njmax=1500,
      mujoco=MujocoCfg(timestep=0.005, iterations=10, ls_iterations=20),
    ),
    decimation=4,
    episode_length_s=20.0,
  )
