"""Left-right symmetry transforms for CASBOT02 AMP (RSL-RL / AMPPPO).

AMP_mjlab stores actor/critic history as time-major (``history_ordering="time"``)::

  [frame_t0, frame_t1, frame_t2, frame_t3]

Each frame is the concatenated observation terms.  Reflection is about the
robot sagittal (x-z) plane.  AMP discriminator expert batches are not
mirrored here; AMPPPO only calls this on policy/critic rollouts.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

HISTORY_LENGTH = 4
NUM_LEG_ACTIONS = 12
NUM_FULL_ACTIONS = 22

LEG_ACTOR_OBS_DIM = 180
LEG_CRITIC_OBS_DIM = 444
FULL_ACTOR_OBS_DIM = 308
FULL_CRITIC_OBS_DIM = 788

# Joint order per leg: hip pitch, hip roll, hip yaw, knee pitch,
# ankle pitch, ankle roll.
_LEG_JOINT_SIGNS = (1.0, -1.0, -1.0, 1.0, 1.0, -1.0)

# Polar vector (lin vel, gravity, body pos): y flips.
_POLAR_SIGNS = (1.0, -1.0, 1.0)
# Axial vector (ang vel): x/z flip, y stays.
_AXIAL_SIGNS = (-1.0, 1.0, -1.0)
# Command = body-frame vx, vy, axial yaw rate wz.
_COMMAND_SIGNS = (1.0, -1.0, -1.0)
# 6D rotation (first two columns of R). Sagittal R' = S R S, S=diag(1,-1,1).
_ORI6_SIGNS = (1.0, -1.0, 1.0, -1.0, 1.0, -1.0)

# AMP key-body order: torso, L hip/knee/ankle, R hip/knee/ankle.
# Full-body critic also appends arm bodies; those stay unmirrored (loco-style).
_LEG_BODY_PERM = (0, 4, 5, 6, 1, 2, 3)
_NUM_LEG_BODIES = 7

_FRAME_TERMS: dict[int, tuple[tuple[str, int], ...]] = {
  45: (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("command", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
  ),
  111: (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("command", 3),
    ("joint_pos", 12),
    ("joint_vel", 12),
    ("actions", 12),
    ("base_lin_vel", 3),
    ("body_pos_b", 21),
    ("body_ori_b", 42),
  ),
  77: (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("command", 3),
    ("phase", 2),
    ("joint_pos", 22),
    ("joint_vel", 22),
    ("actions", 22),
  ),
  197: (
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("command", 3),
    ("phase", 2),
    ("joint_pos", 22),
    ("joint_vel", 22),
    ("actions", 22),
    ("base_lin_vel", 3),
    ("body_pos_b", 39),
    ("body_ori_b", 78),
  ),
}


def _scale_last(value: torch.Tensor, signs: tuple[float, ...]) -> torch.Tensor:
  if value.shape[-1] != len(signs):
    raise ValueError(f"Expected last dim {len(signs)}, got {value.shape[-1]}")
  return value * value.new_tensor(signs)


def mirror_leg_joints(value: torch.Tensor) -> torch.Tensor:
  """Swap the two six-DoF legs and reflect roll/yaw joint coordinates."""
  if value.shape[-1] != NUM_LEG_ACTIONS:
    raise ValueError(f"Expected {NUM_LEG_ACTIONS} leg values, got {value.shape[-1]}")
  sign = value.new_tensor(_LEG_JOINT_SIGNS)
  left = value[..., :6]
  right = value[..., 6:]
  return torch.cat((right * sign, left * sign), dim=-1)


def mirror_full_joints(value: torch.Tensor) -> torch.Tensor:
  """Mirror the 12 legs only; arm joints are left unchanged."""
  if value.shape[-1] != NUM_FULL_ACTIONS:
    raise ValueError(f"Expected {NUM_FULL_ACTIONS} joint values, got {value.shape[-1]}")
  legs = mirror_leg_joints(value[..., :NUM_LEG_ACTIONS])
  return torch.cat((legs, value[..., NUM_LEG_ACTIONS:]), dim=-1)


def mirror_actions(value: torch.Tensor) -> torch.Tensor:
  width = value.shape[-1]
  if width == NUM_LEG_ACTIONS:
    return mirror_leg_joints(value)
  if width == NUM_FULL_ACTIONS:
    return mirror_full_joints(value)
  raise ValueError(
    f"CASBOT02 symmetry expects 12 or 22 actions, got {width}"
  )


def _mirror_bodies(
  value: torch.Tensor,
  per_body: int,
  signs: tuple[float, ...],
) -> torch.Tensor:
  n_bodies = value.shape[-1] // per_body
  if n_bodies * per_body != value.shape[-1]:
    raise ValueError(
      f"Body term dim {value.shape[-1]} is not a multiple of {per_body}"
    )
  if n_bodies not in (_NUM_LEG_BODIES, 13):
    raise ValueError(f"Unsupported AMP body count {n_bodies}")
  bodies = value.reshape(*value.shape[:-1], n_bodies, per_body)
  legs = bodies[..., :_NUM_LEG_BODIES, :]
  legs = _scale_last(legs[..., list(_LEG_BODY_PERM), :], signs)
  if n_bodies == _NUM_LEG_BODIES:
    return legs.reshape_as(value)
  arms = bodies[..., _NUM_LEG_BODIES:, :]
  return torch.cat((legs, arms), dim=-2).reshape_as(value)


def _mirror_named_term(name: str, value: torch.Tensor) -> torch.Tensor:
  if name == "base_ang_vel":
    return _scale_last(value, _AXIAL_SIGNS)
  if name in ("projected_gravity", "base_lin_vel"):
    return _scale_last(value, _POLAR_SIGNS)
  if name == "command":
    return _scale_last(value, _COMMAND_SIGNS)
  if name == "phase":
    return value
  if name in ("joint_pos", "joint_vel", "actions"):
    return mirror_actions(value)
  if name == "body_pos_b":
    return _mirror_bodies(value, 3, _POLAR_SIGNS)
  if name == "body_ori_b":
    return _mirror_bodies(value, 6, _ORI6_SIGNS)
  raise ValueError(f"No CASBOT02 sagittal mirror for observation term '{name}'")


def _mirror_one_frame(
  frame: torch.Tensor, terms: tuple[tuple[str, int], ...]
) -> torch.Tensor:
  pieces: list[torch.Tensor] = []
  offset = 0
  for name, dim in terms:
    pieces.append(_mirror_named_term(name, frame[..., offset : offset + dim]))
    offset += dim
  if offset != frame.shape[-1]:
    raise RuntimeError(
      f"Frame term specs consumed {offset} values from {frame.shape[-1]}"
    )
  return torch.cat(pieces, dim=-1)


def _history_ordering(env: Any) -> str:
  if env is None:
    return "time"
  unwrapped = getattr(env, "unwrapped", env)
  try:
    return unwrapped.observation_manager.cfg["actor"].history_ordering
  except Exception:
    return "time"


def _terms_for_obs_dim(obs_dim: int) -> tuple[tuple[str, int], ...]:
  if obs_dim % HISTORY_LENGTH != 0:
    raise ValueError(
      f"CASBOT02 AMP obs dim {obs_dim} is not divisible by history={HISTORY_LENGTH}"
    )
  frame_dim = obs_dim // HISTORY_LENGTH
  terms = _FRAME_TERMS.get(frame_dim)
  if terms is None:
    raise ValueError(
      "CASBOT02 AMP symmetry expected actor/critic dims "
      f"{LEG_ACTOR_OBS_DIM}/{LEG_CRITIC_OBS_DIM} (leg) or "
      f"{FULL_ACTOR_OBS_DIM}/{FULL_CRITIC_OBS_DIM} (full-body), got {obs_dim}"
    )
  return terms


def mirror_amp_observation(
  obs: torch.Tensor, *, history_ordering: str = "time"
) -> torch.Tensor:
  """Mirror one flattened actor or critic observation batch."""
  terms = _terms_for_obs_dim(obs.shape[-1])
  frame_dim = obs.shape[-1] // HISTORY_LENGTH
  if history_ordering == "time":
    frames = obs.reshape(*obs.shape[:-1], HISTORY_LENGTH, frame_dim)
    return _mirror_one_frame(frames, terms).reshape_as(obs)

  # Term-major: each term's H frames are contiguous.
  mirrored = obs.clone()
  offset = 0
  for name, dim in terms:
    width = HISTORY_LENGTH * dim
    term = obs[..., offset : offset + width]
    frames = term.reshape(*term.shape[:-1], HISTORY_LENGTH, dim)
    mirrored[..., offset : offset + width] = _mirror_named_term(name, frames).reshape_as(
      term
    )
    offset += width
  if offset != obs.shape[-1]:
    raise RuntimeError(
      f"Term-major symmetry consumed {offset} values from {obs.shape[-1]}"
    )
  return mirrored


def _augment_tensor(value: torch.Tensor, mirrored: torch.Tensor) -> torch.Tensor:
  return torch.cat((value, mirrored), dim=0)


@torch.no_grad()
def compute_symmetric_states(
  env: ManagerBasedRlEnv | None = None,
  obs: torch.Tensor | None = None,
  actions: torch.Tensor | None = None,
  obs_type: str = "policy",
  **kwargs,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
  """Append the sagittally mirrored sample to an AMPPPO mini-batch.

  AMPPPO calls this with raw tensors (not TensorDict) and ``obs_type`` of
  ``"policy"`` or ``"critic"``.
  """
  del obs_type, kwargs
  ordering = _history_ordering(env)

  obs_aug = None
  if obs is not None:
    if torch.is_tensor(obs):
      obs_aug = _augment_tensor(obs, mirror_amp_observation(obs, history_ordering=ordering))
    else:
      batch_size = obs.batch_size[0]
      obs_aug = obs.repeat(2)
      for key, value in obs.items():
        obs_aug[key][:batch_size] = value
        obs_aug[key][batch_size:] = mirror_amp_observation(
          value, history_ordering=ordering
        )

  actions_aug = None
  if actions is not None:
    actions_aug = _augment_tensor(actions, mirror_actions(actions))

  return obs_aug, actions_aug


def amp_symmetry_cfg() -> dict[str, Any]:
  """RSL-RL ``symmetry_cfg`` matching HANDOFF loco (aug + mirror loss 0.1)."""
  return {
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 0.1,
    "data_augmentation_func": (
      "src.tasks.amp_loco.mdp.casbot02_symmetry:compute_symmetric_states"
    ),
  }


__all__ = [
  "FULL_ACTOR_OBS_DIM",
  "FULL_CRITIC_OBS_DIM",
  "HISTORY_LENGTH",
  "LEG_ACTOR_OBS_DIM",
  "LEG_CRITIC_OBS_DIM",
  "NUM_FULL_ACTIONS",
  "NUM_LEG_ACTIONS",
  "amp_symmetry_cfg",
  "compute_symmetric_states",
  "mirror_actions",
  "mirror_amp_observation",
  "mirror_full_joints",
  "mirror_leg_joints",
]
