"""CASBOT02 AMP locomotion with lower-body actor observations.

Actor and critic observations expose only the 12 leg joints and omit phase.
The policy controls all CASBOT02 joints except ``waist_yaw_joint``. AMP
discriminator observations keep the same full-body 13 key bodies as the
full-body AMP task.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots import (
  CASBOT02_22DOF_NO_WAIST_ACTION_SCALE,
  CASBOT02_22DOF_NO_WAIST_JOINT_NAMES,
  CASBOT02_23DOF_AMP_BODY_NAMES,
  CASBOT02_LEG_ONLY_JOINT_NAMES,
)
import src.tasks.amp_loco.mdp as amp_mdp
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_flat_env_cfg as _base_flat_env_cfg,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_rough_env_cfg as _base_rough_env_cfg,
)


def _remove_phase_observation(cfg: ManagerBasedRlEnvCfg) -> None:
  cfg.observations["actor"].terms.pop("phase", None)
  cfg.observations["critic"].terms.pop("phase", None)


def _apply_leg_only_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Use 12-leg actor/critic state, 22-DoF no-waist action, and full AMP style."""
  LEG_ONLY_ASSET = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
  )
  FULL_AMP_BODY = SceneEntityCfg(
    "robot", body_names=CASBOT02_23DOF_AMP_BODY_NAMES, preserve_order=True
  )

  # 保留 phase 观测（条件相位：前进/后退才激活，转弯置零）
  # _remove_phase_observation(cfg)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.actuator_names = CASBOT02_22DOF_NO_WAIST_JOINT_NAMES
  joint_pos_action.scale = CASBOT02_22DOF_NO_WAIST_ACTION_SCALE

  # ---- Actor observations: deployment-visible state only (12 leg joints) ----
  cfg.observations["actor"].terms["joint_pos"].params["asset_cfg"] = LEG_ONLY_ASSET
  cfg.observations["actor"].terms["joint_vel"].params["asset_cfg"] = LEG_ONLY_ASSET
  cfg.observations["actor"].terms["actions"].func = amp_mdp.last_action_slice
  cfg.observations["actor"].terms["actions"].params = {
    "start": 0,
    "stop": len(CASBOT02_LEG_ONLY_JOINT_NAMES),
  }

  # ---- Critic observations: same 12-leg state plus full AMP bodies ----
  cfg.observations["critic"].terms["joint_pos"].params["asset_cfg"] = LEG_ONLY_ASSET
  cfg.observations["critic"].terms["joint_vel"].params["asset_cfg"] = LEG_ONLY_ASSET
  cfg.observations["critic"].terms["actions"].func = amp_mdp.last_action_slice
  cfg.observations["critic"].terms["actions"].params = {
    "start": 0,
    "stop": len(CASBOT02_LEG_ONLY_JOINT_NAMES),
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
  """CASBOT02 rough terrain AMP with 12-leg actor observations."""
  return _apply_leg_only_overrides(_base_rough_env_cfg(play=play))


def casbot02_leg_amp_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """CASBOT02 flat terrain AMP with 12-leg actor observations."""
  return _apply_leg_only_overrides(_base_flat_env_cfg(play=play))
