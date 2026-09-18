from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.entity import Entity
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import BuiltinSensor, ContactSensor
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
  yaw_quat,
  quat_apply,
)
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _get_delay_env_mask(env: ManagerBasedRlEnv) -> torch.Tensor | None:
  """Get delaying env mask from DelayedTerminationManager if installed."""
  tm = env.termination_manager
  delay_env_mask = getattr(tm, "_delay_env_mask", None)
  delay_counters = getattr(tm, "_delay_counters", None)
  if isinstance(delay_env_mask, torch.Tensor) and isinstance(delay_counters, torch.Tensor):
    return delay_env_mask & (delay_counters > 0)
  return None


def _apply_delay_env_reward_scaling(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return reward

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return reward

  scaled_reward = reward * delay_env_rew_ratio
  return torch.where(delay_env_mask, scaled_reward, reward)


def _apply_delay_env_reward_mask_only(
  env: ManagerBasedRlEnv,
  reward: torch.Tensor,
  mask_delay: bool,
  delay_env_rew_ratio: float,
) -> torch.Tensor:
  if not mask_delay:
    return torch.zeros_like(reward)

  delay_env_mask = _get_delay_env_mask(env)
  if delay_env_mask is None:
    return torch.zeros_like(reward)

  scaled_reward = reward * delay_env_rew_ratio
  masked_reward = torch.where(delay_env_mask, scaled_reward, torch.zeros_like(reward))
  return masked_reward

def track_anchor_linear_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward for tracking the commanded anchor linear velocity.

  The commanded z velocity is assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  command_xyz_b = torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), dim=-1)
  command_xyz_w = quat_apply(
    yaw_quat(asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]]),
    command_xyz_b,
  )
  lin_vel_error = torch.sum(torch.square(command_xyz_w[:,:3] - asset.data.body_link_lin_vel_w[:, anchor_cfg.body_ids[0], :3]), dim=1)
  reward = torch.exp(-lin_vel_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)


def track_anchor_angular_velocity(
  env: ManagerBasedRlEnv,
  std: float,
  command_name: str,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  anchor_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None, f"Command '{command_name}' not found."

  anchor_ang_vel_w = asset.data.body_link_ang_vel_w[:, anchor_cfg.body_ids[0]]
  anchor_ang_z_vel_w = anchor_ang_vel_w[:, 2]
  command_ang_vel_w = command[:, 2]
  ang_vel_z_error = torch.square(command_ang_vel_w - anchor_ang_z_vel_w)

  anchor_ang_vel_b =  quat_apply_inverse(
    asset.data.body_link_quat_w[:, anchor_cfg.body_ids[0]],
    anchor_ang_vel_w,
  )
  ang_vel_xy_error = torch.sum(torch.square(anchor_ang_vel_b[:, :2]), dim=-1)

  total_error = ang_vel_z_error + ang_vel_xy_error

  reward = torch.exp(-total_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)

def body_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  body_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=()),
) -> torch.Tensor:
  """Reward heading error for heading-controlled envs, angular velocity for others.

  The commanded xy angular velocities are assumed to be zero.
  """
  asset: Entity = env.scene[body_cfg.name]
  body_ang_vel_w = asset.data.body_link_ang_vel_w[:, body_cfg.body_ids[0]]
  body_ang_vel_b = quat_apply_inverse(
    asset.data.body_link_quat_w[:, body_cfg.body_ids[0]],
    body_ang_vel_w,
  )
  body_ang_vel_xy_b = body_ang_vel_b[:, :2]
  ang_vel_xy_error = torch.sum(torch.square(body_ang_vel_xy_b), dim=-1)

  reward = torch.exp(-ang_vel_xy_error / std**2)
  return _apply_delay_env_reward_scaling(env, reward, mask_delay, delay_env_rew_ratio)

def track_root_height(
  env: ManagerBasedRlEnv,
  std: float,
  mask_delay: bool = False,
  delay_env_rew_ratio: float = 1.0,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Reward for tracking the commanded anchor height."""
  asset: Entity = env.scene[asset_cfg.name]

  desired_height = asset.data.default_root_state[:, 2]
  cur_root_height = asset.data.body_link_pos_w[:, 0, 2]
  height_error = torch.square(desired_height - cur_root_height)
  reward = torch.exp(-height_error / std**2)
  return _apply_delay_env_reward_mask_only(env, reward, mask_delay, delay_env_rew_ratio)

def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.01,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding (xy velocity while in contact)."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  active = (total_command > command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()  # [B, N]
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]  # [B, N, 2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)  # [B, N]
  vel_xy_norm_sq = torch.square(vel_xy_norm)  # [B, N]
  cost = torch.sum(vel_xy_norm_sq * in_contact, dim=1) * active
  num_in_contact = torch.sum(in_contact)
  mean_slip_vel = torch.sum(vel_xy_norm * in_contact) / torch.clamp(
    num_in_contact, min=1
  )
  env.extras["log"]["Metrics/slip_velocity_mean"] = mean_slip_vel
  return cost


def standing_feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str,
  command_threshold: float = 0.2,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize foot sliding only for standing or near-standing commands."""
  asset: Entity = env.scene[asset_cfg.name]
  contact_sensor: ContactSensor = env.scene[sensor_name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  standing = (total_command < command_threshold).float()
  assert contact_sensor.data.found is not None
  in_contact = (contact_sensor.data.found > 0).float()
  foot_vel_xy = asset.data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  vel_xy_norm = torch.norm(foot_vel_xy, dim=-1)
  cost = torch.sum(torch.square(vel_xy_norm) * in_contact, dim=1) * standing
  num_standing_contacts = torch.sum(in_contact * standing.unsqueeze(1))
  mean_standing_slip = torch.sum(
    vel_xy_norm * in_contact * standing.unsqueeze(1)
  ) / torch.clamp(num_standing_contacts, min=1)
  env.extras["log"]["Metrics/standing_slip_velocity_mean"] = mean_standing_slip
  return cost


def standing_foot_distance(
  env: ManagerBasedRlEnv,
  command_name: str,
  command_threshold: float,
  target_lateral_distance: float,
  target_fore_distance: float,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  """Penalize stance width/fore-aft drift only for standing commands."""
  asset: Entity = env.scene[asset_cfg.name]
  command = env.command_manager.get_command(command_name)
  assert command is not None
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  total_command = linear_norm + angular_norm
  standing = (total_command < command_threshold).float()
  foot_pos = asset.data.site_pos_w[:, asset_cfg.site_ids, :2]
  left_right_delta = foot_pos[:, 0] - foot_pos[:, 1]
  fore_distance = torch.abs(left_right_delta[:, 0])
  lateral_distance = torch.abs(left_right_delta[:, 1])
  fore_error = torch.square(fore_distance - target_fore_distance)
  lateral_error = torch.square(lateral_distance - target_lateral_distance)
  cost = (fore_error + lateral_error) * standing
  denom = torch.clamp(torch.sum(standing), min=1)
  env.extras["log"]["Metrics/standing_foot_fore_error"] = (
    torch.sum(torch.sqrt(fore_error) * standing) / denom
  )
  env.extras["log"]["Metrics/standing_foot_lateral_error"] = (
    torch.sum(torch.sqrt(lateral_error) * standing) / denom
  )
  return cost


def soft_landing(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  command_name: str | None = None,
  command_threshold: float = 0.05,
) -> torch.Tensor:
  """Penalize high impact forces at landing to encourage soft footfalls."""
  contact_sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = contact_sensor.data
  assert sensor_data.force is not None
  forces = sensor_data.force  # [B, N, 3]
  force_magnitude = torch.norm(forces, dim=-1)  # [B, N]
  first_contact = contact_sensor.compute_first_contact(dt=env.step_dt)  # [B, N]
  landing_impact = force_magnitude * first_contact.float()  # [B, N]
  cost = torch.sum(landing_impact, dim=1)  # [B]
  num_landings = torch.sum(first_contact.float())
  mean_landing_force = torch.sum(landing_impact) / torch.clamp(num_landings, min=1)
  env.extras["log"]["Metrics/landing_force_mean"] = mean_landing_force
  if command_name is not None:
    command = env.command_manager.get_command(command_name)
    if command is not None:
      linear_norm = torch.norm(command[:, :2], dim=1)
      angular_norm = torch.abs(command[:, 2])
      total_command = linear_norm + angular_norm
      active = (total_command > command_threshold).float()
      cost = cost * active
  return cost

def undesired_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 1.0,
) -> torch.Tensor:
  """Penalize non-foot bodies contacting the terrain.

  Returns the number of primary bodies whose contact force exceeds
  ``force_threshold``. Falls back to the instantaneous ``found`` count
  when force is unavailable.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force is not None:
    in_contact = torch.norm(data.force, dim=-1) > force_threshold
  else:
    assert data.found is not None
    in_contact = data.found > 0
  cost = in_contact.float().sum(dim=-1)
  env.extras["log"]["Metrics/undesired_contact_count"] = torch.mean(cost)
  return cost


def self_collision_cost(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  force_threshold: float = 10.0,
) -> torch.Tensor:
  """Penalize self-collisions.

  When the sensor provides force history (from ``history_length > 0``),
  counts substeps where any contact force exceeds *force_threshold*.
  Falls back to the instantaneous ``found`` count otherwise.
  """
  sensor: ContactSensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    # force_history: [B, N, H, 3]
    force_mag = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hit = (force_mag > force_threshold).any(dim=1)  # [B, H]
    return hit.sum(dim=-1).float()  # [B]
  assert data.found is not None
  return data.found.squeeze(-1)


def _get_robot(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> Entity:
  return env.scene[asset_cfg.name]


def _foot_site_heights(
  env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """World-Z of sole sites. On a z=0 plane this is foot clearance."""
  asset = _get_robot(env, asset_cfg)
  return asset.data.site_pos_w[:, asset_cfg.site_ids, 2]


def _expand_gravity_for_bodies(
  gravity_vec_w: torch.Tensor, *, num_envs: int, num_bodies: int
) -> torch.Tensor:
  """Broadcast gravity to `[num_envs, num_bodies, 3]` for body-wise projections."""
  if gravity_vec_w.ndim == 1:
    gravity_vec_w = gravity_vec_w.unsqueeze(0)
  if gravity_vec_w.shape[0] == 1 and num_envs != 1:
    gravity_vec_w = gravity_vec_w.expand(num_envs, -1)
  elif gravity_vec_w.shape[0] != num_envs:
    raise ValueError(
      f"Expected gravity_vec_w to have {num_envs} rows, got {gravity_vec_w.shape[0]}"
    )
  return gravity_vec_w.unsqueeze(1).expand(-1, num_bodies, -1)


def _positive_per_foot_target(
  value: float | tuple[float, ...],
  *,
  num_feet: int,
  device: torch.device | str,
  name: str,
) -> torch.Tensor:
  """Convert a scalar/per-foot target to a positive ``[1, num_feet]`` tensor."""
  target = torch.as_tensor(value, device=device, dtype=torch.float32)
  if target.ndim == 0:
    target = target.repeat(num_feet)
  else:
    target = target.flatten()
  if target.numel() != num_feet:
    raise ValueError(
      f"{name} must be scalar or contain {num_feet} values, got {target.numel()}"
    )
  if bool(torch.any(target <= 0.0)):
    raise ValueError(f"{name} values must all be positive, got {value}")
  return target.unsqueeze(0)


def command_turning_blend(
  command: torch.Tensor,
  *,
  turning_linear_threshold: float,
  turning_angular_threshold: float,
) -> torch.Tensor:
  """Return a smooth 0=translation, 1=in-place-turn command blend."""
  if turning_linear_threshold <= 0.0 or turning_angular_threshold <= 0.0:
    raise ValueError("Turning blend thresholds must be positive")
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  low_translation = torch.clamp(
    1.0 - linear_norm / turning_linear_threshold, min=0.0, max=1.0
  )
  active_turn = torch.clamp(
    angular_norm / turning_angular_threshold, min=0.0, max=1.0
  )
  return low_translation * active_turn


class command_conditioned_swing_height_curve:
  """Track separate translational and in-place-turn swing-height curves.

  Ported from HANDOFF loco teacher. Foot height is the world-Z of the sole
  sites (on a z=0 plane this equals clearance).
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    site_names = cfg.params["asset_cfg"].site_names
    if site_names is None:
      raise ValueError("command_conditioned_swing_height_curve requires foot sites")
    num_feet = len(site_names)
    self.translation_peak_height = _positive_per_foot_target(
      cfg.params["translation_peak_height"],
      num_feet=num_feet,
      device=env.device,
      name="translation_peak_height",
    )
    self.turning_peak_height = _positive_per_foot_target(
      cfg.params["turning_peak_height"],
      num_feet=num_feet,
      device=env.device,
      name="turning_peak_height",
    )
    self.translation_swing_time = _positive_per_foot_target(
      cfg.params["translation_swing_time"],
      num_feet=num_feet,
      device=env.device,
      name="translation_swing_time",
    )
    self.turning_swing_time = _positive_per_foot_target(
      cfg.params["turning_swing_time"],
      num_feet=num_feet,
      device=env.device,
      name="turning_swing_time",
    )

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    translation_peak_height: float | tuple[float, ...],
    turning_peak_height: float | tuple[float, ...],
    translation_swing_time: float | tuple[float, ...],
    turning_swing_time: float | tuple[float, ...],
    command_name: str = "twist",
    command_threshold: float = 0.05,
    turning_linear_threshold: float = 0.2,
    turning_angular_threshold: float = 0.2,
  ) -> torch.Tensor:
    del (
      translation_peak_height,
      turning_peak_height,
      translation_swing_time,
      turning_swing_time,
    )
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    assert contact_sensor.data.current_air_time is not None

    command = env.command_manager.get_command(command_name)
    assert command is not None
    turn_blend = command_turning_blend(
      command,
      turning_linear_threshold=turning_linear_threshold,
      turning_angular_threshold=turning_angular_threshold,
    ).unsqueeze(1)
    peak_height = torch.lerp(
      self.translation_peak_height, self.turning_peak_height, turn_blend
    )
    swing_time = torch.lerp(
      self.translation_swing_time, self.turning_swing_time, turn_blend
    )

    foot_heights = _foot_site_heights(env, asset_cfg)
    current_air_time = contact_sensor.data.current_air_time
    in_air = contact_sensor.data.found == 0
    assert foot_heights.shape == current_air_time.shape == in_air.shape, (
      "swing-height curve requires matching height/contact frames, "
      f"got {tuple(foot_heights.shape)}, {tuple(current_air_time.shape)}, "
      f"and {tuple(in_air.shape)}"
    )

    swing_progress = torch.clamp(current_air_time / swing_time, 0.0, 1.0)
    desired_height = peak_height * torch.sin(torch.pi * swing_progress)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm > command_threshold).float()
    swing_mask = in_air.float() * active.unsqueeze(1)
    normalized_error = (foot_heights - desired_height) / peak_height
    squared_error = torch.square(normalized_error) * swing_mask
    cost = torch.sum(squared_error, dim=1)

    num_swing_feet = torch.clamp(torch.sum(swing_mask), min=1.0)
    env.extras["log"]["Metrics/swing_foot_height_mean"] = (
      torch.sum(foot_heights * swing_mask) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_target_mean"] = (
      torch.sum(desired_height * swing_mask) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_curve_rmse"] = torch.sqrt(
      torch.sum(squared_error) / num_swing_feet
    )
    env.extras["log"]["Metrics/swing_height_turn_blend_mean"] = turn_blend.mean()
    return cost


class command_conditioned_feet_swing_height:
  """Penalize landing peak-height error with a lower pure-turn target."""

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    site_names = cfg.params["asset_cfg"].site_names
    if site_names is None:
      raise ValueError("command_conditioned_feet_swing_height requires foot sites")
    num_feet = len(site_names)
    self.peak_heights = torch.zeros(
      (env.num_envs, num_feet), device=env.device, dtype=torch.float32
    )
    self.step_dt = env.step_dt

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str,
    asset_cfg: SceneEntityCfg,
    translation_target_height: float,
    turning_target_height: float,
    command_name: str = "twist",
    command_threshold: float = 0.05,
    turning_linear_threshold: float = 0.2,
    turning_angular_threshold: float = 0.2,
  ) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene[sensor_name]
    assert contact_sensor.data.found is not None
    foot_heights = _foot_site_heights(env, asset_cfg)
    in_air = contact_sensor.data.found == 0
    self.peak_heights = torch.where(
      in_air, torch.maximum(self.peak_heights, foot_heights), self.peak_heights
    )

    command = env.command_manager.get_command(command_name)
    assert command is not None
    turn_blend = command_turning_blend(
      command,
      turning_linear_threshold=turning_linear_threshold,
      turning_angular_threshold=turning_angular_threshold,
    )
    target_height = torch.lerp(
      torch.full_like(turn_blend, translation_target_height),
      torch.full_like(turn_blend, turning_target_height),
      turn_blend,
    ).unsqueeze(1)
    first_contact = contact_sensor.compute_first_contact(dt=self.step_dt)
    linear_norm = torch.norm(command[:, :2], dim=1)
    angular_norm = torch.abs(command[:, 2])
    active = (linear_norm + angular_norm > command_threshold).float()
    error = self.peak_heights / target_height - 1.0
    cost = torch.sum(torch.square(error) * first_contact.float(), dim=1) * active

    landing_mask = first_contact.float() * active.unsqueeze(1)
    num_landings = torch.clamp(torch.sum(landing_mask), min=1.0)
    env.extras["log"]["Metrics/peak_height_mean"] = (
      torch.sum(self.peak_heights * landing_mask) / num_landings
    )
    env.extras["log"]["Metrics/peak_height_target_mean"] = (
      torch.sum(target_height * landing_mask) / num_landings
    )
    self.peak_heights = torch.where(
      first_contact, torch.zeros_like(self.peak_heights), self.peak_heights
    )
    return cost

  def reset(self, env_ids: torch.Tensor | slice | None) -> None:
    if env_ids is None:
      self.peak_heights.zero_()
    else:
      self.peak_heights[env_ids] = 0.0


def command_conditioned_feet_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  threshold_min: float = 0.05,
  translation_threshold_max: float = 0.60,
  turning_threshold_max: float = 0.42,
  command_name: str = "twist",
  command_threshold: float = 0.2,
  turning_linear_threshold: float = 0.2,
  turning_angular_threshold: float = 0.2,
) -> torch.Tensor:
  """Dense air-time reward with a shorter window for in-place turns."""
  sensor: ContactSensor = env.scene[sensor_name]
  current_air_time = sensor.data.current_air_time
  assert current_air_time is not None
  command = env.command_manager.get_command(command_name)
  assert command is not None
  turn_blend = command_turning_blend(
    command,
    turning_linear_threshold=turning_linear_threshold,
    turning_angular_threshold=turning_angular_threshold,
  )
  threshold_max = torch.lerp(
    torch.full_like(turn_blend, translation_threshold_max),
    torch.full_like(turn_blend, turning_threshold_max),
    turn_blend,
  ).unsqueeze(1)
  in_range = (current_air_time > threshold_min) & (current_air_time < threshold_max)
  reward = torch.sum(in_range.float(), dim=1)
  linear_norm = torch.norm(command[:, :2], dim=1)
  angular_norm = torch.abs(command[:, 2])
  active = (linear_norm + angular_norm > command_threshold).float()

  in_air = current_air_time > 0.0
  num_in_air = torch.clamp(torch.sum(in_air.float()), min=1.0)
  env.extras["log"]["Metrics/air_time_mean"] = (
    torch.sum(current_air_time * in_air.float()) / num_in_air
  )
  env.extras["log"]["Metrics/air_time_max_target_mean"] = threshold_max.mean()
  return reward * active


def stand_pose(
  env: ManagerBasedRlEnv,
  command_name: str = "twist",
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  default_joint_pos = asset.data.default_joint_pos
  joint_ids = asset_cfg.joint_ids
  error = torch.sum(
    torch.square(asset.data.joint_pos[:, joint_ids] - default_joint_pos[:, joint_ids]),
    dim=-1,
  )
  twist_cmd = env.command_manager.get_term(command_name)
  return error * twist_cmd.is_standing_env.float()


def flat_foot(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  contact_force_threshold: float = 1.0,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  sensor: ContactSensor = env.scene[sensor_name]
  force = sensor.data.force
  assert force is not None
  contact = (torch.norm(force[..., :3], dim=-1) > contact_force_threshold).float()
  foot_quat = asset.data.body_link_quat_w[:, asset_cfg.body_ids, :]
  gravity = _expand_gravity_for_bodies(
    asset.data.gravity_vec_w,
    num_envs=env.num_envs,
    num_bodies=foot_quat.shape[1],
  )
  projected = quat_apply_inverse(
    foot_quat.reshape(-1, 4), gravity.reshape(-1, 3)
  ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
  tilt_error = torch.sum(torch.square(projected[..., :2]), dim=-1)
  return torch.sum(tilt_error * contact, dim=-1)


def feet_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  """Reward zero inside a lateral foot-spacing band and negative outside it."""
  asset = _get_robot(env, asset_cfg)
  root_quat = asset.data.root_link_quat_w
  if asset_cfg.site_names is not None:
    foot_pos_w = asset.data.site_pos_w[:, asset_cfg.site_ids, :]
  else:
    foot_pos_w = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  if foot_pos_w.shape[1] != 2:
    raise ValueError(
      "feet_distance_lateral requires exactly two foot bodies or sites, "
      f"got {foot_pos_w.shape[1]}"
    )

  left_right_delta_w = foot_pos_w[:, 0] - foot_pos_w[:, 1]
  left_right_delta_b = quat_apply_inverse(root_quat, left_right_delta_w)
  lateral = torch.abs(left_right_delta_b[:, 1])
  too_close = torch.clamp(lateral - min_distance, max=0.0)
  too_far = torch.clamp(-lateral + max_distance, max=0.0)
  return too_close + too_far


def knee_distance_lateral(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  min_distance: float,
  max_distance: float,
) -> torch.Tensor:
  asset = _get_robot(env, asset_cfg)
  root_pos = asset.data.root_link_pos_w
  root_quat = asset.data.root_link_quat_w
  body_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :]
  delta = body_pos - root_pos.unsqueeze(1)
  body_pos_b = quat_apply_inverse(
    root_quat.unsqueeze(1).expand(-1, len(asset_cfg.body_ids), -1).reshape(-1, 4),
    delta.reshape(-1, 3),
  ).reshape(env.num_envs, len(asset_cfg.body_ids), 3)
  lateral = torch.abs(body_pos_b[:, 0, 1] - body_pos_b[:, 2, 1]) + torch.abs(
    body_pos_b[:, 1, 1] - body_pos_b[:, 3, 1]
  )
  too_close = torch.clamp(lateral - 2.0 * min_distance, max=0.0)
  too_far = torch.clamp(-lateral + 2.0 * max_distance, max=0.0)
  return too_close + too_far
