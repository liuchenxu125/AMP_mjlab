from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from src.tasks.amp_loco.gait import (
  GAIT_OFFSET,
  GAIT_PERIOD,
  GAIT_STANDING_THRESHOLD,
  validate_gait_parameters,
)

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
  quat_apply_inverse,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def loco_leg_phase(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gait_period: float = GAIT_PERIOD,
  gait_offset: float = GAIT_OFFSET,
  standing_threshold: float = GAIT_STANDING_THRESHOLD,
) -> torch.Tensor:
  """Episode-clock phases, with both feet in stance for near-zero commands."""
  validate_gait_parameters(gait_period, gait_offset, standing_threshold)
  elapsed = env.episode_length_buf.to(torch.float32) * env.step_dt
  left = torch.remainder(elapsed, gait_period) / gait_period
  right = torch.remainder(left + gait_offset, 1.0)
  phases = torch.stack((left, right), dim=-1)
  command = env.command_manager.get_command(command_name)
  # Use the actual command so deployment does not need a hidden standing flag.
  standing = torch.linalg.vector_norm(command, dim=-1) < standing_threshold
  return torch.where(standing.unsqueeze(-1), torch.zeros_like(phases), phases)


def loco_phase_features(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  gait_period: float = GAIT_PERIOD,
  gait_offset: float = GAIT_OFFSET,
  standing_threshold: float = GAIT_STANDING_THRESHOLD,
) -> torch.Tensor:
  phases = loco_leg_phase(
    env, command_name, gait_period, gait_offset, standing_threshold
  )
  angles = 2.0 * torch.pi * phases
  return torch.stack(
    (angles[:, 0].sin(), angles[:, 0].cos(), angles[:, 1].sin(), angles[:, 1].cos()),
    dim=-1,
  )


def phase(
    env: ManagerBasedRlEnv,
    period: float,
    command_name: str,
) -> torch.Tensor:
  global_phase = (env.episode_length_buf * env.step_dt) % period / period
  phase = torch.zeros(env.num_envs, 2, device=env.device)
  phase[:, 0] = torch.sin(global_phase * torch.pi * 2.0)
  phase[:, 1] = torch.cos(global_phase * torch.pi * 2.0)
  command = env.command_manager.get_command(command_name)
  stand_mask = torch.linalg.norm(command, dim=1) < 0.1
  phase = torch.where(stand_mask.unsqueeze(1), torch.zeros_like(phase), phase)
  return phase


def last_action_slice(
    env: ManagerBasedRlEnv,
    start: int = 0,
    stop: int | None = None,
    action_name: str | None = None,
) -> torch.Tensor:
  action = (
    env.action_manager.action
    if action_name is None
    else env.action_manager.get_term(action_name).raw_action
  )
  return action[:, start:stop]


def robot_body_pos_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    pos_b, _ = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    return pos_b.reshape(env.num_envs, -1)

def robot_body_ori_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    anchor_pos_w = asset.data.body_link_pos_w[:, anchor_cfg.body_ids[0]]   # (num_envs, 3)
    anchor_quat_w = asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]  # (num_envs, 4)
    
    body_pos_w = asset.data.body_link_pos_w[:, body_cfg.body_ids]     # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 4)

    num_bodies = body_pos_w.shape[1]
    _, ori_b = subtract_frame_transforms(
        anchor_pos_w[:, None, :].expand(-1, num_bodies, -1),
        anchor_quat_w[:, None, :].expand(-1, num_bodies, -1),
        body_pos_w,
        body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)

def robot_body_lin_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_lin_vel_w = asset.data.body_link_lin_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_lin_vel_w.shape[1]

    body_lin_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_lin_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_lin_vel_b.reshape(env.num_envs, -1)

def robot_body_ang_vel_b(
    env: ManagerBasedRlEnv,
    anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
    body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
    asset: Entity = env.scene[anchor_cfg.name]
    
    body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids]   # (num_envs, num_bodies, 3)
    body_quat_w = asset.data.body_link_quat_w[:, body_cfg.body_ids]       # (num_envs, num_bodies, 4)

    num_bodies = body_ang_vel_w.shape[1]

    body_ang_vel_b = quat_apply_inverse(
        body_quat_w.reshape(-1, 4),
        body_ang_vel_w.reshape(-1, 3),
    ).reshape(env.num_envs, num_bodies, 3)

    return body_ang_vel_b.reshape(env.num_envs, -1)
