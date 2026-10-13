"""CASBOT02 AMP locomotion with lower-body actor observations.

Actor and critic expose 12 leg joints plus four left/right gait-clock features.
The policy controls 12 leg joints; arms follow the existing knee-mapped swing.
AMP discriminator observations retain the seven configured key bodies.
"""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from src.assets.robots import (
  CASBOT02_LEG_ONLY_ACTION_SCALE,
  CASBOT02_LEG_AMP_BODY_NAMES,
  CASBOT02_LEG_ONLY_JOINT_NAMES,
)
import src.tasks.amp_loco.mdp as amp_mdp
from src.tasks.amp_loco.gait import (
  GAIT_OFFSET,
  GAIT_PERIOD,
  GAIT_STANCE_RATIO,
  GAIT_STANDING_THRESHOLD,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_flat_env_cfg as _base_flat_env_cfg,
)
from src.tasks.amp_loco.config.casbot02.env_cfgs import (
  casbot02_amp_rough_env_cfg as _base_rough_env_cfg,
)


def _add_leg_gait(cfg: ManagerBasedRlEnvCfg) -> None:
  phase_params = {
    "command_name": "twist",
    "gait_period": GAIT_PERIOD,
    "gait_offset": GAIT_OFFSET,
    "standing_threshold": GAIT_STANDING_THRESHOLD,
  }
  for group_name in ("actor", "critic"):
    group = cfg.observations[group_name]
    terms = {name: term for name, term in group.terms.items() if name != "phase"}
    group.terms = {}
    for name, term in terms.items():
      group.terms[name] = term
      if name == "actions":
        group.terms["phase"] = ObservationTermCfg(
          func=amp_mdp.loco_phase_features, params=dict(phase_params)
        )
  cfg.rewards["gait_phase_contact"] = RewardTermCfg(
    func=amp_mdp.gait_phase_contact,
    weight=0.5,
    params={
      **phase_params,
      "sensor_name": "feet_ground_contact",
      "stance_ratio": GAIT_STANCE_RATIO,
      "contact_force_threshold": 1.0,
    },
  )


def _apply_leg_only_overrides(cfg: ManagerBasedRlEnvCfg) -> ManagerBasedRlEnvCfg:
  """Use 12-leg actor/critic state, 12-leg action, and leg+waist AMP style."""
  LEG_ONLY_ASSET = SceneEntityCfg(
    "robot", joint_names=CASBOT02_LEG_ONLY_JOINT_NAMES, preserve_order=True
  )
  FULL_AMP_BODY = SceneEntityCfg(
    "robot", body_names=CASBOT02_LEG_AMP_BODY_NAMES, preserve_order=True
  )

  _add_leg_gait(cfg)

  # 12 腿 action + 手臂摆臂（手臂由摆臂公式控制，不经网络）。
  cfg.actions["joint_pos"] = amp_mdp.LegWithArmSwingActionCfg(
    entity_name="robot",
    actuator_names=CASBOT02_LEG_ONLY_JOINT_NAMES,
    scale=CASBOT02_LEG_ONLY_ACTION_SCALE,
    use_default_offset=True,
  )

  # Both ankle penalties cover pitch and roll; keep existing metric names.
  # cfg.rewards["ankle_pitch_action_rate_l2"] = RewardTermCfg(
  #   func=amp_mdp.joint_action_rate_l2,
  #   weight=-0.01,
  #   params={
  #     "action_name": "joint_pos",
  #     "joint_names": (
  #       "leg_l5_joint", "leg_l6_joint", "leg_r5_joint", "leg_r6_joint"
  #     ),
  #   },
  # )

  # cfg.rewards["ankle_pitch_joint_acc_l2"] = RewardTermCfg(
  #   func=envs_mdp.joint_acc_l2,
  #   weight=-2.5e-7,
  #   params={
  #     "asset_cfg": SceneEntityCfg(
  #       "robot",
  #       joint_names=(
  #         "leg_l5_joint", "leg_l6_joint", "leg_r5_joint", "leg_r6_joint"
  #       ),
  #       preserve_order=True,
  #     ),
  #   },
  # )

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
