"""Custom actions for CASBOT02 AMP tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class LegWithArmSwingAction(JointPositionAction):
  """12-leg position action with arm swing computed from knee angles.

  The arms are not controlled by the policy network. Instead the two shoulder
  pitch joints swing based on the knee-angle difference, mimicking human
  cross-body arm swing. All other arm joints are held at their HOME_KEYFRAME
  defaults.

  Joint indices follow CASBOT02_23DOF_JOINT_NAMES ordering (23 joints).
  """

  _LEG_L4_ID = 3
  _LEG_R4_ID = 9
  _ARM_IDS = (13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
  _LEFT_SHOULDER = 0   # arm_target index for upper_left_1
  _RIGHT_SHOULDER = 5  # arm_target index for upper_right_1

  def __init__(self, cfg: JointPositionActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)
    arm_ids = torch.tensor(self._ARM_IDS, device=self.device, dtype=torch.long)
    self._arm_ids = arm_ids
    self._arm_default = self._entity.data.default_joint_pos[:, arm_ids].clone()

  def apply_actions(self) -> None:
    # 先应用腿关节的 position target（父类逻辑）。
    super().apply_actions()

    leg_l4 = self._entity.data.joint_pos[:, self._LEG_L4_ID]
    leg_r4 = self._entity.data.joint_pos[:, self._LEG_R4_ID]

    arm_target = self._arm_default.clone()
    # 左臂肩：左膝弯（左腿迈步）时向后摆，配合交叉摆臂。
    arm_target[:, self._LEFT_SHOULDER] = (
      0.5 * (leg_l4 - leg_r4) + self._arm_default[:, self._LEFT_SHOULDER]
    )
    # 右臂肩：反向摆动。
    arm_target[:, self._RIGHT_SHOULDER] = (
      0.5 * (leg_r4 - leg_l4) + self._arm_default[:, self._RIGHT_SHOULDER]
    )
    self._entity.set_joint_position_target(arm_target, joint_ids=self._arm_ids)


@dataclass(kw_only=True)
class LegWithArmSwingActionCfg(JointPositionActionCfg):
  """Configuration for 12-leg position action with computed arm swing."""

  def build(self, env: ManagerBasedRlEnv) -> LegWithArmSwingAction:
    return LegWithArmSwingAction(self, env)
