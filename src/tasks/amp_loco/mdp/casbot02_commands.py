"""Casbot02 velocity command sampling, matching the HANDOFF loco teacher."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mjlab.utils.lab_api.math import quat_apply, wrap_to_pi

from src.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)


class Casbot02VelocityCommand(UniformVelocityCommand):
  """Uniform velocity command with mutually exclusive gait-mode cohorts."""

  cfg: Casbot02VelocityCommandCfg

  def __init__(self, cfg: Casbot02VelocityCommandCfg, env) -> None:
    super().__init__(cfg, env)
    self.is_turning_env = torch.zeros_like(self.is_standing_env)
    self.is_backward_env = torch.zeros_like(self.is_standing_env)
    self.is_forward_env = torch.zeros_like(self.is_standing_env)
    self.is_world_env = torch.zeros_like(self.is_standing_env)
    self.is_stand_then_go_env = torch.zeros_like(self.is_standing_env)
    self.vel_command_w = torch.zeros_like(self.vel_command_b)
    self.stand_then_go_vel = torch.zeros_like(self.vel_command_b)
    self.stand_then_go_walk_duration = torch.zeros(self.num_envs, device=self.device)

  def _sample_turning_yaw_rate(self, count: int) -> torch.Tensor:
    if count == 0:
      return torch.empty(0, device=self.device)
    lo, hi = self.cfg.ranges.ang_vel_z
    if not (lo < 0.0 < hi):
      raise ValueError(
        "Pure-turn sampling requires ang_vel_z to span both signs, "
        f"got {(lo, hi)}"
      )
    sign_positive = torch.rand(count, device=self.device) >= 0.5
    max_magnitude = torch.where(
      sign_positive,
      torch.full((count,), hi, device=self.device),
      torch.full((count,), -lo, device=self.device),
    )
    min_magnitude = torch.clamp(
      torch.full_like(max_magnitude, self.cfg.min_turning_ang_vel),
      max=max_magnitude,
    )
    magnitude = min_magnitude + torch.rand(count, device=self.device) * (
      max_magnitude - min_magnitude
    )
    return torch.where(sign_positive, magnitude, -magnitude)

  def _sample_straight_vx(self, count: int, *, backward: bool) -> torch.Tensor:
    """Sample |vx| within the current curriculum, vy=wz handled by caller."""
    if count == 0:
      return torch.empty(0, device=self.device)
    lo, hi = self.cfg.ranges.lin_vel_x
    min_speed = self.cfg.min_straight_speed
    if backward:
      if lo >= 0.0:
        raise ValueError("Backward sampling requires a negative lin_vel_x lower bound")
      upper = min(hi, -min(min_speed, -lo))
      return torch.empty(count, device=self.device).uniform_(lo, upper)
    lower = max(lo, min(min_speed, hi) if hi > 0.0 else hi)
    return torch.empty(count, device=self.device).uniform_(lower, hi)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    count = len(env_ids)
    if count == 0:
      return

    random = torch.empty(count, device=self.device)
    self.vel_command_b[env_ids, 0] = random.uniform_(*self.cfg.ranges.lin_vel_x)
    self.vel_command_b[env_ids, 1] = random.uniform_(*self.cfg.ranges.lin_vel_y)
    self.vel_command_b[env_ids, 2] = random.uniform_(*self.cfg.ranges.ang_vel_z)

    if self.cfg.heading_command:
      assert self.cfg.ranges.heading is not None
      self.heading_target[env_ids] = random.uniform_(*self.cfg.ranges.heading)

    mode = torch.rand(count, device=self.device)
    standing_end = self.cfg.rel_standing_envs
    turning_end = standing_end + self.cfg.rel_turning_envs
    forward_end = turning_end + self.cfg.rel_forward_envs
    backward_end = forward_end + self.cfg.rel_backward_envs
    stand_then_go_end = backward_end + self.cfg.rel_stand_then_go_envs
    heading_end = stand_then_go_end + (
      self.cfg.rel_heading_envs if self.cfg.heading_command else 0.0
    )

    standing = mode < standing_end
    turning = (mode >= standing_end) & (mode < turning_end)
    forward = (mode >= turning_end) & (mode < forward_end)
    backward = (mode >= forward_end) & (mode < backward_end)
    stand_then_go = (mode >= backward_end) & (mode < stand_then_go_end)
    heading = (mode >= stand_then_go_end) & (mode < heading_end)

    self.is_standing_env[env_ids] = standing | stand_then_go
    self.is_turning_env[env_ids] = turning
    self.is_forward_env[env_ids] = forward
    self.is_backward_env[env_ids] = backward
    self.is_stand_then_go_env[env_ids] = stand_then_go
    self.is_heading_env[env_ids] = heading
    self.stand_then_go_vel[env_ids] = 0.0
    self.stand_then_go_walk_duration[env_ids] = 0.0

    turning_ids = env_ids[turning]
    self.vel_command_b[turning_ids, :2] = 0.0
    self.vel_command_b[turning_ids, 2] = self._sample_turning_yaw_rate(
      len(turning_ids)
    )

    forward_ids = env_ids[forward]
    self.vel_command_b[forward_ids, 0] = (
      self.vel_command_b[forward_ids, 0].abs().clamp(min=self.cfg.min_straight_speed)
    )
    self.vel_command_b[forward_ids, 1:] = 0.0

    backward_ids = env_ids[backward]
    if len(backward_ids) > 0:
      self.vel_command_b[backward_ids, 0] = self._sample_straight_vx(
        len(backward_ids), backward=True
      )
      self.vel_command_b[backward_ids, 1:] = 0.0

    stg_ids = env_ids[stand_then_go]
    if len(stg_ids) > 0:
      stand_lo, stand_hi = self.cfg.stand_then_go_stand_time_range
      walk_lo, walk_hi = self.cfg.stand_then_go_walk_time_range
      stand_dur = torch.empty(len(stg_ids), device=self.device).uniform_(
        stand_lo, stand_hi
      )
      walk_dur = torch.empty(len(stg_ids), device=self.device).uniform_(
        walk_lo, walk_hi
      )
      go_backward = torch.rand(len(stg_ids), device=self.device) < 0.5
      vx = torch.empty(len(stg_ids), device=self.device)
      n_back = int(go_backward.sum().item())
      n_fwd = len(stg_ids) - n_back
      if n_back > 0:
        vx[go_backward] = self._sample_straight_vx(n_back, backward=True)
      if n_fwd > 0:
        vx[~go_backward] = self._sample_straight_vx(n_fwd, backward=False)
      pending = torch.zeros(len(stg_ids), 3, device=self.device)
      pending[:, 0] = vx
      self.stand_then_go_vel[stg_ids] = pending
      self.stand_then_go_walk_duration[stg_ids] = walk_dur
      self.time_left[stg_ids] = stand_dur + walk_dur
      self.vel_command_b[stg_ids] = 0.0

    self.is_world_env[env_ids] = (
      torch.rand(count, device=self.device) <= self.cfg.rel_world_envs
    ) & ~(standing | turning | forward | backward | stand_then_go | heading)
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

    init_velocity_mask = (
      torch.rand(count, device=self.device) < self.cfg.init_velocity_prob
    ) & ~stand_then_go
    init_velocity_env_ids = env_ids[init_velocity_mask]
    if len(init_velocity_env_ids) > 0:
      root_pos = self.robot.data.root_link_pos_w[init_velocity_env_ids]
      root_quat = self.robot.data.root_link_quat_w[init_velocity_env_ids]
      lin_vel_b = self.robot.data.root_link_lin_vel_b[init_velocity_env_ids].clone()
      lin_vel_b[:, :2] = self.vel_command_b[init_velocity_env_ids, :2]
      root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
      root_ang_vel_b = self.robot.data.root_link_ang_vel_b[
        init_velocity_env_ids
      ].clone()
      root_ang_vel_b[:, 2] = self.vel_command_b[init_velocity_env_ids, 2]
      root_state = torch.cat(
        [root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1
      )
      self.robot.write_root_state_to_sim(root_state, init_velocity_env_ids)

  def _update_stand_then_go(self) -> None:
    stg = self.is_stand_then_go_env
    if not stg.any():
      return
    still_standing = stg & (self.time_left > self.stand_then_go_walk_duration)
    walking = stg & ~still_standing
    self.is_standing_env = torch.where(stg, still_standing, self.is_standing_env)
    if walking.any():
      self.vel_command_b[walking] = self.stand_then_go_vel[walking]
      self.vel_command_w[walking] = self.stand_then_go_vel[walking]
      vx = self.stand_then_go_vel[:, 0]
      self.is_forward_env = torch.where(walking, vx > 0.0, self.is_forward_env)
      self.is_backward_env = torch.where(walking, vx < 0.0, self.is_backward_env)

  def _update_command(self) -> None:
    if self.cfg.heading_command:
      self.heading_error = wrap_to_pi(self.heading_target - self.robot.data.heading_w)
      env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
      self.vel_command_b[env_ids, 2] = torch.clip(
        self.cfg.heading_control_stiffness * self.heading_error[env_ids],
        min=self.cfg.ranges.ang_vel_z[0],
        max=self.cfg.ranges.ang_vel_z[1],
      )
    if self.is_world_env.any():
      w_ids = self.is_world_env.nonzero(as_tuple=False).flatten()
      heading = self.robot.data.heading_w[w_ids]
      cos_h = torch.cos(heading)
      sin_h = torch.sin(heading)
      vx_w = self.vel_command_w[w_ids, 0]
      vy_w = self.vel_command_w[w_ids, 1]
      self.vel_command_b[w_ids, 0] = cos_h * vx_w + sin_h * vy_w
      self.vel_command_b[w_ids, 1] = -sin_h * vx_w + cos_h * vy_w

    self._update_stand_then_go()

    standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
    self.vel_command_b[standing_env_ids, :] = 0.0
    self.vel_command_w[standing_env_ids, :] = 0.0


@dataclass(kw_only=True)
class Casbot02VelocityCommandCfg(UniformVelocityCommandCfg):
  """Configuration for :class:`Casbot02VelocityCommand`."""

  rel_turning_envs: float = 0.2
  rel_backward_envs: float = 0.0
  rel_forward_envs: float = 0.0
  rel_stand_then_go_envs: float = 0.0
  rel_world_envs: float = 0.0
  min_turning_ang_vel: float = 0.2
  min_straight_speed: float = 0.3
  stand_then_go_stand_time_range: tuple[float, float] = (2.0, 3.0)
  stand_then_go_walk_time_range: tuple[float, float] = (3.0, 4.0)

  def build(self, env) -> Casbot02VelocityCommand:
    return Casbot02VelocityCommand(self, env)

  def __post_init__(self) -> None:
    super().__post_init__()
    fractions = (
      self.rel_standing_envs,
      self.rel_turning_envs,
      self.rel_forward_envs,
      self.rel_backward_envs,
      self.rel_stand_then_go_envs,
      self.rel_heading_envs if self.heading_command else 0.0,
    )
    if any(value < 0.0 for value in fractions):
      raise ValueError(f"Command mode fractions must be non-negative: {fractions}")
    if sum(fractions) > 1.0 + 1.0e-8:
      raise ValueError(
        "Standing + turning + forward + backward + stand-then-go + heading "
        f"fractions must not exceed 1.0, got {sum(fractions):.3f}"
      )
    if self.min_turning_ang_vel <= 0.0:
      raise ValueError("min_turning_ang_vel must be positive")
    if self.min_straight_speed <= 0.0:
      raise ValueError("min_straight_speed must be positive")
    stand_lo, stand_hi = self.stand_then_go_stand_time_range
    walk_lo, walk_hi = self.stand_then_go_walk_time_range
    if stand_lo <= 0.0 or stand_hi < stand_lo:
      raise ValueError(
        f"Invalid stand_then_go_stand_time_range {self.stand_then_go_stand_time_range}"
      )
    if walk_lo <= 0.0 or walk_hi < walk_lo:
      raise ValueError(
        f"Invalid stand_then_go_walk_time_range {self.stand_then_go_walk_time_range}"
      )
    needs_backward = self.rel_backward_envs > 0.0 or self.rel_stand_then_go_envs > 0.0
    if needs_backward and self.ranges.lin_vel_x[0] >= 0.0:
      raise ValueError("Backward sampling requires a negative lin_vel_x lower bound")


__all__ = ["Casbot02VelocityCommand", "Casbot02VelocityCommandCfg"]
