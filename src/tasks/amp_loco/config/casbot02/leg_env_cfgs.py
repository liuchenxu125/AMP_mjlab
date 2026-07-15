"""CASBOT02 AMP locomotion with lower-body actor observations.

Actor observations expose only the legs and waist for deployment, while the
policy still outputs all 23 CASBOT02 joints. Critic and AMP body-kinematics
observations keep the same 13 key bodies as the full-body AMP task.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots import (
  CASBOT02_23DOF_AMP_BODY_NAMES,
  CASBOT02_LEG_JOINT_NAMES,
)
import src.tasks.amp_loco.mdp as amp_mdp
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_flat_env_cfg as _base_flat_env_cfg,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_rough_env_cfg as _base_rough_env_cfg,
)


def _apply_leg_only_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Use leg+waist actor state with full-body action, critic, and AMP style."""
  LEG_ASSET = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_JOINT_NAMES, preserve_order=True
  )
  FULL_AMP_BODY = SceneEntityCfg(
    "robot", body_names=CASBOT02_23DOF_AMP_BODY_NAMES, preserve_order=True
  )

  # ---- Actor observations: deployment-visible state only (legs + waist) ----
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["actor"].terms["actions"].func = amp_mdp.last_action_slice
  cfg.observations["actor"].terms["actions"].params = {
    "start": 0,
    "stop": len(CASBOT02_LEG_JOINT_NAMES),
  }

  # ---- Critic observations: same lower-body actor state plus full AMP bodies ----
  cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = LEG_ASSET
  cfg.observations["critic"].terms["actions"].func = amp_mdp.last_action_slice
  cfg.observations["critic"].terms["actions"].params = {
    "start": 0,
    "stop": len(CASBOT02_LEG_JOINT_NAMES),
  }
  cfg.observations["critic"].terms["body_pos_b"].params[
    "body_cfg"
  ] = FULL_AMP_BODY
  cfg.observations["critic"].terms["body_ori_b"].params[
    "body_cfg"
  ] = FULL_AMP_BODY

  # ---- AMP observations: keep full-body style discriminator input ----
  amp = cfg.observations["amp"]
  amp.terms["body_pos_b"].params["body_cfg"] = FULL_AMP_BODY
  amp.terms["body_ori_b"].params["body_cfg"] = FULL_AMP_BODY
  amp.terms["body_lin_vel_b"].params["body_cfg"] = FULL_AMP_BODY
  amp.terms["body_ang_vel_b"].params["body_cfg"] = FULL_AMP_BODY

  return cfg


def casbot02_leg_amp_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CASBOT02 rough terrain AMP with lower-body actor observations."""
  return _apply_leg_only_overrides(_base_rough_env_cfg(play=play))


def casbot02_leg_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CASBOT02 flat terrain AMP with lower-body actor observations."""
  return _apply_leg_only_overrides(_base_flat_env_cfg(play=play))
