"""CASBOT02 leg-only AMP locomotion environment configurations.

These configs exclude arm joints and arm body kinematics from all observations.
The action space is reduced to 13 DOF (12 legs + 1 waist).
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots import (
  CASBOT02_LEG_AMP_BODY_NAMES,
  CASBOT02_LEG_JOINT_NAMES,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_flat_env_cfg as _base_flat_env_cfg,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_rough_env_cfg as _base_rough_env_cfg,
)


def _apply_leg_only_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Apply leg-only observation and action filtering on top of a base cfg."""
  LEG_ASSET = SceneEntityCfg("robot", joint_names=CASBOT02_LEG_JOINT_NAMES)
  LEG_BODY = SceneEntityCfg("robot", body_names=CASBOT02_LEG_AMP_BODY_NAMES)

  # ---- Action space: 13 DOF (legs + waist) ----
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = CASBOT02_LEG_JOINT_NAMES
  # Filter per-joint scale dict to leg joints only (otherwise it still references arm joints).
  if isinstance(joint_pos_action.scale, dict):
    joint_pos_action.scale = {
      n: joint_pos_action.scale[n] for n in CASBOT02_LEG_JOINT_NAMES
    }

  # ---- Actor observations: filter joint terms ----
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = LEG_ASSET

  # ---- Critic observations: filter joint terms + body terms ----
  cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["critic"].terms["body_pos_b"].params["body_cfg"] = LEG_BODY
  cfg.observations["critic"].terms["body_ori_b"].params["body_cfg"] = LEG_BODY

  # ---- AMP observations: filter body terms ----
  amp = cfg.observations["amp"]
  amp.terms["body_pos_b"].params["body_cfg"] = LEG_BODY
  amp.terms["body_ori_b"].params["body_cfg"] = LEG_BODY
  amp.terms["body_lin_vel_b"].params["body_cfg"] = LEG_BODY
  amp.terms["body_ang_vel_b"].params["body_cfg"] = LEG_BODY

  # ---- Rewards: filter joint-level penalties to leg joints only ----
  # NOTE: use independent copies to avoid shared-object resolution conflicts
  # with observation terms that also reference LEG_ASSET.
  cfg.rewards["joint_pos_limits"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_JOINT_NAMES
  )
  cfg.rewards["joint_acc_l2"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_JOINT_NAMES
  )

  # ---- reset_from_motion: filter joints ----
  cfg.events["reset_from_motion"].params["asset_cfg"] = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_JOINT_NAMES
  )

  return cfg


def casbot02_leg_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CASBOT02 rough terrain leg-only velocity configuration."""
  return _apply_leg_only_overrides(_base_rough_env_cfg(play=play))


def casbot02_leg_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CASBOT02 flat terrain leg-only velocity configuration."""
  return _apply_leg_only_overrides(_base_flat_env_cfg(play=play))
