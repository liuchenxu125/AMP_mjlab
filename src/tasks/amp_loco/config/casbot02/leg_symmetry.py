"""Left-right symmetry transforms for the CASBOT02 leg AMP policy.

The actor observation is term-major and contains four history frames::

  base_ang_vel(3), projected_gravity(3), command(3),
  joint_pos(12), joint_vel(12), last_action(12)

The critic appends four frames of base linear velocity and the relative
positions/orientations of the seven AMP bodies. Reflection is about the
robot sagittal (x-z) plane.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


HISTORY_LENGTH = 4
NUM_ACTIONS = 12
NUM_AMP_BODIES = 7
ACTOR_OBS_DIM = 180
CRITIC_OBS_DIM = 444

# Per-leg order: hip pitch, hip roll, hip yaw, knee pitch, ankle pitch,
# ankle roll. Roll and yaw change sign under sagittal reflection.
_JOINT_REFLECTION_SIGN = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)

# AMP-body order: torso, three left-leg bodies, three right-leg bodies.
_BODY_MIRROR_INDEX = (0, 4, 5, 6, 1, 2, 3)


def _mirror_vector_history(
  value: torch.Tensor,
  sign: tuple[float, ...],
) -> torch.Tensor:
  """Reflect a term stored as ``[history, term_dim]`` on the last axis."""
  term_dim = len(sign)
  expected_dim = HISTORY_LENGTH * term_dim
  if value.shape[-1] != expected_dim:
    raise ValueError(
      f"Expected a {expected_dim}-D history term, got {value.shape[-1]}"
    )
  frames = value.reshape(*value.shape[:-1], HISTORY_LENGTH, term_dim)
  return (frames * value.new_tensor(sign)).reshape_as(value)


def mirror_leg_joints(value: torch.Tensor) -> torch.Tensor:
  """Swap the two six-DoF legs and reflect roll/yaw coordinates."""
  if value.shape[-1] != NUM_ACTIONS:
    raise ValueError(f"Expected {NUM_ACTIONS} leg values, got {value.shape[-1]}")
  sign = value.new_tensor(_JOINT_REFLECTION_SIGN)
  left = value[..., :6]
  right = value[..., 6:]
  return torch.cat((right * sign, left * sign), dim=-1)


def _mirror_joint_history(value: torch.Tensor) -> torch.Tensor:
  expected_dim = HISTORY_LENGTH * NUM_ACTIONS
  if value.shape[-1] != expected_dim:
    raise ValueError(
      f"Expected a {expected_dim}-D joint history term, got {value.shape[-1]}"
    )
  frames = value.reshape(*value.shape[:-1], HISTORY_LENGTH, NUM_ACTIONS)
  return mirror_leg_joints(frames).reshape_as(value)


def _mirror_body_history(
  value: torch.Tensor,
  component_sign: tuple[float, ...],
) -> torch.Tensor:
  """Swap left/right AMP bodies and reflect each body's components."""
  component_dim = len(component_sign)
  expected_dim = HISTORY_LENGTH * NUM_AMP_BODIES * component_dim
  if value.shape[-1] != expected_dim:
    raise ValueError(
      f"Expected a {expected_dim}-D body history term, got {value.shape[-1]}"
    )
  frames = value.reshape(
    *value.shape[:-1], HISTORY_LENGTH, NUM_AMP_BODIES, component_dim
  )
  body_index = torch.tensor(_BODY_MIRROR_INDEX, device=value.device)
  mirrored = frames.index_select(-2, body_index)
  mirrored = mirrored * value.new_tensor(component_sign)
  return mirrored.reshape_as(value)


def mirror_actor_observation(obs: torch.Tensor) -> torch.Tensor:
  """Mirror one flattened 180-D actor observation."""
  if obs.shape[-1] != ACTOR_OBS_DIM:
    raise ValueError(
      f"Expected a {ACTOR_OBS_DIM}-D actor observation, got {obs.shape[-1]}"
    )

  mirrored = obs.clone()
  offset = 0
  vector_width = HISTORY_LENGTH * 3

  # Angular velocity is an axial vector: x/z flip, y stays unchanged.
  mirrored[..., offset : offset + vector_width] = _mirror_vector_history(
    obs[..., offset : offset + vector_width], (-1.0, 1.0, -1.0)
  )
  offset += vector_width

  # Projected gravity is a polar vector: y flips.
  mirrored[..., offset : offset + vector_width] = _mirror_vector_history(
    obs[..., offset : offset + vector_width], (1.0, -1.0, 1.0)
  )
  offset += vector_width

  # Command is body-frame (vx, vy, yaw rate).
  mirrored[..., offset : offset + vector_width] = _mirror_vector_history(
    obs[..., offset : offset + vector_width], (1.0, -1.0, -1.0)
  )
  offset += vector_width

  joint_width = HISTORY_LENGTH * NUM_ACTIONS
  for _ in range(3):
    mirrored[..., offset : offset + joint_width] = _mirror_joint_history(
      obs[..., offset : offset + joint_width]
    )
    offset += joint_width

  if offset != ACTOR_OBS_DIM:
    raise RuntimeError(f"Actor symmetry consumed {offset} values")
  return mirrored


def mirror_critic_observation(obs: torch.Tensor) -> torch.Tensor:
  """Mirror one flattened 444-D critic observation."""
  if obs.shape[-1] != CRITIC_OBS_DIM:
    raise ValueError(
      f"Expected a {CRITIC_OBS_DIM}-D critic observation, got {obs.shape[-1]}"
    )

  mirrored = obs.clone()
  mirrored[..., :ACTOR_OBS_DIM] = mirror_actor_observation(
    obs[..., :ACTOR_OBS_DIM]
  )
  offset = ACTOR_OBS_DIM

  vector_width = HISTORY_LENGTH * 3
  mirrored[..., offset : offset + vector_width] = _mirror_vector_history(
    obs[..., offset : offset + vector_width], (1.0, -1.0, 1.0)
  )
  offset += vector_width

  body_pos_width = HISTORY_LENGTH * NUM_AMP_BODIES * 3
  mirrored[..., offset : offset + body_pos_width] = _mirror_body_history(
    obs[..., offset : offset + body_pos_width], (1.0, -1.0, 1.0)
  )
  offset += body_pos_width

  # 6D orientation is the first two rotation-matrix columns flattened as
  # [r00, r01, r10, r11, r20, r21]. For S=diag(1,-1,1), R' = S R S.
  body_ori_width = HISTORY_LENGTH * NUM_AMP_BODIES * 6
  mirrored[..., offset : offset + body_ori_width] = _mirror_body_history(
    obs[..., offset : offset + body_ori_width],
    (1.0, -1.0, -1.0, 1.0, 1.0, -1.0),
  )
  offset += body_ori_width

  if offset != CRITIC_OBS_DIM:
    raise RuntimeError(f"Critic symmetry consumed {offset} values")
  return mirrored


@torch.no_grad()
def compute_symmetric_states(
  env: ManagerBasedRlEnv,
  obs: torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
  obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Append one sagittally mirrored copy to an RSL-RL mini-batch."""
  del env

  obs_aug = None
  if obs is not None:
    if obs_type == "policy":
      mirrored_obs = mirror_actor_observation(obs)
    elif obs_type == "critic":
      mirrored_obs = mirror_critic_observation(obs)
    else:
      raise ValueError(f"Unsupported observation type: {obs_type!r}")
    obs_aug = torch.cat((obs, mirrored_obs), dim=0)

  actions_aug = None
  if actions is not None:
    actions_aug = torch.cat((actions, mirror_leg_joints(actions)), dim=0)

  return obs_aug, actions_aug


__all__ = [
  "ACTOR_OBS_DIM",
  "CRITIC_OBS_DIM",
  "HISTORY_LENGTH",
  "NUM_ACTIONS",
  "compute_symmetric_states",
  "mirror_actor_observation",
  "mirror_critic_observation",
  "mirror_leg_joints",
]
