"""Headless mjlab evaluation of CASBOT02-Leg-AMP checkpoints on six axes.

Axes: velocity tracking, landing impact, foot slip, leg shape, standing
posture, standing stability.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import tyro

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends

from src.assets.robots import CASBOT02_LEG_ONLY_JOINT_NAMES
from src.tasks.velocity.mdp import UniformVelocityCommandCfg

TASK_CANDIDATES = ("Casbot02-Leg-AMP-Flat", "Mjlab-Casbot02-Leg-AMP-Flat")

DEFAULT_HEIGHT = 0.92
TARGET_FOOT_CLEARANCE = 0.06
HOME_LEG = {
  "leg_l1_joint": -0.185,
  "leg_l2_joint": 0.0,
  "leg_l3_joint": 0.0,
  "leg_l4_joint": 0.36,
  "leg_l5_joint": -0.175,
  "leg_l6_joint": 0.0,
  "leg_r1_joint": -0.185,
  "leg_r2_joint": 0.0,
  "leg_r3_joint": 0.0,
  "leg_r4_joint": 0.36,
  "leg_r5_joint": -0.175,
  "leg_r6_joint": 0.0,
}

# Domain randomization is disabled so every checkpoint sees identical physics.
_RANDOMIZATION_EVENTS = (
  "push_robot",
  "randomize_terrain",
  "encoder_bias",
  "base_com",
  "waist_com_backward",
  "joint_default_pos",
  "robot_friction",
  "actuator_gains",
)

# Sideways is omitted: CASBOT02 AMP trains lin_vel_y = (0, 0), so a 0.3 m/s
# lateral command just knocks every env over and stresses the contact solver.
SCENARIOS = (
  ("stand", (0.0, 0.0, 0.0)),
  ("walk_0.3", (0.3, 0.0, 0.0)),
  ("walk_0.6", (0.6, 0.0, 0.0)),
  ("walk_1.0", (1.0, 0.0, 0.0)),
  ("back_0.5", (-0.5, 0.0, 0.0)),
  ("turn_0.8", (0.0, 0.0, 0.8)),
)
LOCO_SCENARIOS = tuple(n for n, _ in SCENARIOS if n != "stand")
FWD_SCENARIOS = ("walk_0.3", "walk_0.6", "walk_1.0")


@dataclass(frozen=True)
class EvalConfig:
  run_dir: str = (
    "logs/rsl_rl/casbot02_leg_amp_locomotion/2026-08-25_11-28-46"
  )
  task_id: str = "Casbot02-Leg-AMP-Flat"
  num_envs: int = 24
  device: str = "cuda:0"
  seed: int = 12345
  stand_seconds: float = 8.0
  walk_seconds: float = 8.0
  warmup_seconds: float = 2.0
  ckpt_min: int = 1000
  ckpt_max: int = 999999
  ckpt_step: int = 1000
  out_prefix: str = "eval6"


def _prepare_env_cfg(task_id: str, num_envs: int, seed: int):
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  env_cfg.episode_length_s = 60.0
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.curriculum = {}
  for name in _RANDOMIZATION_EVENTS:
    env_cfg.events.pop(name, None)
  if "init_motion_loader" in env_cfg.events:
    env_cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
    env_cfg.events["init_motion_loader"].params["max_delay_steps"] = 0

  twist = env_cfg.commands["twist"]
  assert isinstance(twist, UniformVelocityCommandCfg)
  twist.heading_command = False
  twist.rel_standing_envs = 0.0
  twist.rel_heading_envs = 0.0
  twist.init_velocity_prob = 0.0
  twist.resampling_time_range = (1.0e6, 1.0e6)
  twist.ranges.heading = None
  return env_cfg


def _force_command(env: ManagerBasedRlEnv, vx: float, vy: float, wz: float) -> None:
  term = env.command_manager.get_term("twist")
  term.vel_command_b[:, 0] = vx
  term.vel_command_b[:, 1] = vy
  term.vel_command_b[:, 2] = wz
  term.is_standing_env[:] = False
  term.is_heading_env[:] = False
  term.time_left[:] = 1.0e6


def _mean(x: torch.Tensor, mask: torch.Tensor) -> float:
  """Mean of ``x`` over the envs selected by the 1-D boolean ``mask``."""
  if not bool(mask.any()):
    return float("nan")
  return float(x[mask].mean().item())


class Accumulator:
  """Running mean over steps, ignoring NaN contributions."""

  def __init__(self) -> None:
    self._sums: dict[str, float] = {}
    self._counts: dict[str, int] = {}

  def add(self, key: str, value: float) -> None:
    if not np.isfinite(value):
      return
    self._sums[key] = self._sums.get(key, 0.0) + value
    self._counts[key] = self._counts.get(key, 0) + 1

  def result(self) -> dict[str, float]:
    return {
      k: self._sums[k] / self._counts[k] if self._counts[k] else float("nan")
      for k in self._sums
    }


def run_scenario(
  env: ManagerBasedRlEnv,
  policy,
  command: tuple[float, float, float],
  duration_s: float,
  warmup_s: float,
  joint_ids: dict[str, int],
  foot_ids: tuple[int, int],
  body_weight: float,
  seed: int,
) -> dict[str, float]:
  robot = env.scene["robot"]
  contact = env.scene["feet_ground_contact"]
  vx, vy, wz = command
  device = env.device
  dt = env.step_dt

  torch.manual_seed(seed)
  np.random.seed(seed)
  obs, _ = env.reset()
  _force_command(env, vx, vy, wz)
  obs = env.observation_manager.compute()

  n = env.num_envs
  alive = torch.ones(n, dtype=torch.bool, device=device)
  fell = torch.zeros(n, dtype=torch.bool, device=device)
  total_steps = int(round(duration_s / dt))
  warmup_steps = int(round(warmup_s / dt))

  left_id, right_id = foot_ids
  idx = {k: joint_ids[k] for k in joint_ids}
  leg_idx = torch.tensor(
    [joint_ids[j] for j in CASBOT02_LEG_ONLY_JOINT_NAMES],
    device=device,
    dtype=torch.long,
  )
  home = torch.tensor(
    [HOME_LEG[j] for j in CASBOT02_LEG_ONLY_JOINT_NAMES],
    device=device,
    dtype=torch.float32,
  )
  hip_pitch = torch.tensor([idx["leg_l1_joint"], idx["leg_r1_joint"]], device=device)
  hip_roll = torch.tensor([idx["leg_l2_joint"], idx["leg_r2_joint"]], device=device)
  hip_yaw = torch.tensor([idx["leg_l3_joint"], idx["leg_r3_joint"]], device=device)
  knee = torch.tensor([idx["leg_l4_joint"], idx["leg_r4_joint"]], device=device)
  ankle_pitch = torch.tensor([idx["leg_l5_joint"], idx["leg_r5_joint"]], device=device)
  ankle_roll = torch.tensor([idx["leg_l6_joint"], idx["leg_r6_joint"]], device=device)

  acc = Accumulator()
  cmd_vec = torch.tensor([vx, vy, wz], device=device)
  landing_forces: list[float] = []
  touchdown_vz: list[float] = []
  slip_series: list[float] = []
  peak_force = 0.0
  swing_peaks: list[float] = []
  start_xy: torch.Tensor | None = None
  last_xy: torch.Tensor | None = None
  drift_ref_alive: torch.Tensor | None = None
  prev_foot_vz = torch.zeros(n, 2, device=device)
  air_time_samples: list[float] = []

  actor_obs = obs["actor"] if hasattr(obs, "keys") else obs
  for step in range(total_steps):
    _force_command(env, vx, vy, wz)
    with torch.inference_mode():
      actions = policy(actor_obs)
    obs, _, terminated, truncated, _ = env.step(actions)
    actor_obs = obs["actor"] if hasattr(obs, "keys") else obs
    term_b = terminated.to(dtype=torch.bool)
    done = term_b | truncated.to(dtype=torch.bool)
    fell |= alive & term_b

    foot_pos = robot.data.site_pos_w[:, [left_id, right_id], :]
    foot_vel = robot.data.site_lin_vel_w[:, [left_id, right_id], :]
    in_contact_b = contact.data.found > 0
    if in_contact_b.ndim == 3:
      in_contact_b = in_contact_b.squeeze(-1)
    in_contact_b = in_contact_b[:, :2]
    first_contact = contact.compute_first_contact(dt=dt)[:, :2]
    force_mag = torch.norm(contact.data.force[:, :2, :], dim=-1)

    if step < warmup_steps:
      prev_foot_vz = foot_vel[:, :, 2].clone()
      alive = alive & ~done
      continue

    mask = alive
    if not bool(mask.any()):
      break

    # --- Landing impact ---
    land = first_contact & mask.unsqueeze(1)
    if bool(land.any()):
      landing_forces.extend((force_mag[land] / body_weight).tolist())
      touchdown_vz.extend(prev_foot_vz[land].abs().tolist())
    contact_force = force_mag * mask.unsqueeze(1)
    peak_force = max(peak_force, float(contact_force.max().item()) / body_weight)
    acc.add("contact_force_mean", _mean(force_mag.sum(dim=1) / body_weight, mask))

    # --- Base state ---
    lin_b = robot.data.root_link_lin_vel_b
    ang_b = robot.data.root_link_ang_vel_b
    grav = robot.data.projected_gravity_b
    height = robot.data.root_link_pos_w[:, 2]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    acc.add("err_vx", _mean((lin_b[:, 0] - cmd_vec[0]).abs(), mask))
    acc.add("err_vy", _mean((lin_b[:, 1] - cmd_vec[1]).abs(), mask))
    acc.add("err_wz", _mean((ang_b[:, 2] - cmd_vec[2]).abs(), mask))
    acc.add("vx", _mean(lin_b[:, 0], mask))
    acc.add("vy", _mean(lin_b[:, 1], mask))
    acc.add("wz", _mean(ang_b[:, 2], mask))
    acc.add("vel_spread", float(lin_b[mask, 0].std().item()) if int(mask.sum()) > 1 else 0.0)
    acc.add("lin_vel_xy", _mean(torch.norm(lin_b[:, :2], dim=-1), mask))
    acc.add("lin_vel_z", _mean(lin_b[:, 2].abs(), mask))
    acc.add("ang_vel", _mean(torch.norm(ang_b, dim=-1), mask))
    acc.add("tilt", _mean(torch.norm(grav[:, :2], dim=-1), mask))
    acc.add("height", _mean(height, mask))

    # --- Foot slip ---
    slip = torch.norm(foot_vel[:, :, :2], dim=-1) * in_contact_b.float()
    acc.add("foot_slip", _mean(slip.mean(dim=1), mask))
    slip_series.append(float(slip[mask].max().item()) if bool(mask.any()) else 0.0)

    # --- Leg shape ---
    swing = (~in_contact_b).float()
    acc.add("stance_width", _mean((foot_pos[:, 0, 1] - foot_pos[:, 1, 1]).abs(), mask))
    acc.add("stance_fore", _mean((foot_pos[:, 0, 0] - foot_pos[:, 1, 0]).abs(), mask))
    acc.add("hip_roll_abs", _mean(joint_pos[:, hip_roll].abs().mean(dim=1), mask))
    acc.add("hip_yaw_abs", _mean(joint_pos[:, hip_yaw].abs().mean(dim=1), mask))
    acc.add("ankle_roll_abs", _mean(joint_pos[:, ankle_roll].abs().mean(dim=1), mask))
    acc.add("knee_mean", _mean(joint_pos[:, knee].mean(dim=1), mask))
    left = torch.stack(
      [
        joint_pos[:, hip_pitch[0]],
        joint_pos[:, hip_roll[0]],
        joint_pos[:, knee[0]],
        joint_pos[:, ankle_pitch[0]],
      ],
      dim=-1,
    )
    right = torch.stack(
      [
        joint_pos[:, hip_pitch[1]],
        -joint_pos[:, hip_roll[1]],
        joint_pos[:, knee[1]],
        joint_pos[:, ankle_pitch[1]],
      ],
      dim=-1,
    )
    acc.add("lr_asym", _mean(torch.norm(left - right, dim=-1), mask))

    torque = robot.data.actuator_force
    acc.add("hip_roll_torque", _mean(torque[:, hip_roll].abs().mean(dim=1), mask))
    acc.add("leg_torque", _mean(torque[:, leg_idx].abs().mean(dim=1), mask))

    swing_h = foot_pos[:, :, 2] * swing
    if bool((swing > 0).any()):
      swing_peaks.append(float(swing_h[(swing > 0) & mask.unsqueeze(1)].max().item()))
    air = contact.data.current_air_time
    if air is not None:
      air = air[:, :2]
      lifted = (~in_contact_b) & mask.unsqueeze(1)
      if bool(lifted.any()):
        air_time_samples.append(float(air[lifted].mean().item()))

    # --- Standing posture / stability ---
    acc.add("joint_still", _mean(torch.norm(joint_vel[:, leg_idx], dim=-1), mask))
    acc.add("joint_home_err", _mean(torch.norm(joint_pos[:, leg_idx] - home, dim=-1), mask))
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    acc.add("action_rate", _mean(torch.norm(action - prev_action, dim=-1), mask))

    xy = robot.data.root_link_pos_w[:, :2]
    if start_xy is None:
      start_xy = xy.clone()
      drift_ref_alive = mask.clone()
    last_xy = xy.clone()

    prev_foot_vz = foot_vel[:, :, 2].clone()
    alive = alive & ~done

  metrics = acc.result()
  for key in (
    "err_vx",
    "err_vy",
    "err_wz",
    "vx",
    "vy",
    "wz",
    "vel_spread",
    "lin_vel_xy",
    "lin_vel_z",
    "ang_vel",
    "tilt",
    "height",
    "foot_slip",
    "stance_width",
    "stance_fore",
    "hip_roll_abs",
    "hip_yaw_abs",
    "ankle_roll_abs",
    "knee_mean",
    "lr_asym",
    "hip_roll_torque",
    "leg_torque",
    "joint_still",
    "joint_home_err",
    "action_rate",
    "contact_force_mean",
  ):
    metrics.setdefault(key, float("nan"))
  metrics["fall_rate"] = float(fell.float().mean().item())
  metrics["alive_frac"] = float(alive.float().mean().item())
  metrics["landing_force_mean"] = float(np.mean(landing_forces)) if landing_forces else 0.0
  metrics["landing_force_p95"] = (
    float(np.percentile(landing_forces, 95)) if landing_forces else 0.0
  )
  metrics["landing_force_peak"] = peak_force
  metrics["touchdown_vz"] = float(np.mean(touchdown_vz)) if touchdown_vz else 0.0
  metrics["touchdown_vz_p95"] = (
    float(np.percentile(touchdown_vz, 95)) if touchdown_vz else 0.0
  )
  metrics["n_landings"] = float(len(landing_forces))
  metrics["foot_slip_p95"] = float(np.percentile(slip_series, 95)) if slip_series else 0.0
  metrics["peak_foot_clearance"] = float(np.mean(swing_peaks)) if swing_peaks else 0.0
  metrics["air_time"] = float(np.mean(air_time_samples)) if air_time_samples else 0.0
  if start_xy is not None and last_xy is not None and drift_ref_alive is not None:
    ref = drift_ref_alive & alive
    if bool(ref.any()):
      disp = torch.norm(last_xy - start_xy, dim=-1)
      metrics["xy_drift"] = float(disp[ref].mean().item())
    else:
      metrics["xy_drift"] = float("nan")
  else:
    metrics["xy_drift"] = float("nan")
  metrics["height_err"] = abs(metrics.get("height", DEFAULT_HEIGHT) - DEFAULT_HEIGHT)
  metrics["cmd"] = [vx, vy, wz]
  return metrics


def _rank(values: np.ndarray, higher_better: bool) -> np.ndarray:
  """Percentile rank in [0, 1]; ties share the average rank."""
  x = np.asarray(values, dtype=np.float64)
  out = np.full(x.shape, 0.5)
  finite = np.isfinite(x)
  if finite.sum() < 2:
    return out
  vals = x[finite] if higher_better else -x[finite]
  order = np.argsort(vals, kind="mergesort")
  raw = np.empty(len(vals))
  raw[order] = np.arange(len(vals), dtype=np.float64)
  # Average the positions of tied values so equal metrics get equal scores.
  uniq, inv = np.unique(vals, return_inverse=True)
  avg = np.zeros(len(uniq))
  for i in range(len(uniq)):
    avg[i] = raw[inv == i].mean()
  out[finite] = avg[inv] / max(len(vals) - 1, 1)
  return out


def _band(values: np.ndarray, target: float, width: float) -> np.ndarray:
  x = np.asarray(values, dtype=np.float64)
  return np.exp(-(((x - target) / width) ** 2))


def score_models(rows: list[dict], stance_target: float) -> list[dict]:
  def col(key: str, scenario: str) -> np.ndarray:
    return np.array(
      [r["scenarios"][scenario].get(key, np.nan) for r in rows], dtype=np.float64
    )

  def avg(key: str, scenarios) -> np.ndarray:
    return np.nanmean(np.stack([col(key, s) for s in scenarios]), axis=0)

  stand = "stand"
  loco = LOCO_SCENARIOS

  # 1. Velocity tracking.
  err_main = np.nanmean(
    np.stack(
      [col("err_vx", s) for s in FWD_SCENARIOS]
      + [col("err_vx", "back_0.5"), col("err_wz", "turn_0.8")]
    ),
    axis=0,
  )
  err_off = np.nanmean(
    np.stack(
      [col("err_vy", s) + 0.3 * col("err_wz", s) for s in FWD_SCENARIOS]
      + [col("err_vy", "back_0.5") + 0.3 * col("err_wz", "back_0.5")]
    ),
    axis=0,
  )
  track_score = (
    0.55 * _rank(err_main, False)
    + 0.25 * _rank(err_off, False)
    + 0.20 * _rank(avg("vel_spread", loco), False)
  )

  # 2. Landing impact.
  land_score = (
    0.35 * _rank(avg("landing_force_mean", loco), False)
    + 0.25 * _rank(avg("landing_force_p95", loco), False)
    + 0.25 * _rank(avg("touchdown_vz", loco), False)
    + 0.15 * _rank(avg("landing_force_peak", loco), False)
  )

  # 3. Foot slip.
  slip_score = (
    0.45 * _rank(avg("foot_slip", loco), False)
    + 0.25 * _rank(avg("foot_slip_p95", loco), False)
    + 0.30 * _rank(col("foot_slip", stand), False)
  )

  # 4. Leg shape.
  clearance = avg("peak_foot_clearance", FWD_SCENARIOS)
  leg_score = (
    0.24 * _rank(avg("hip_roll_abs", loco), False)
    + 0.14 * _rank(avg("hip_yaw_abs", loco), False)
    + 0.12 * _rank(avg("ankle_roll_abs", loco), False)
    + 0.18 * _rank(np.abs(avg("stance_width", loco) - stance_target), False)
    + 0.14 * _rank(avg("lr_asym", loco), False)
    + 0.10 * _rank(_band(clearance, TARGET_FOOT_CLEARANCE, 0.035), True)
    + 0.08 * _rank(avg("hip_roll_torque", loco), False)
  )

  # 5. Standing posture.
  posture_score = (
    0.30 * _rank(col("tilt", stand), False)
    + 0.25 * _rank(col("height_err", stand), False)
    + 0.25 * _rank(col("joint_home_err", stand), False)
    + 0.12 * _rank(np.abs(col("stance_width", stand) - stance_target), False)
    + 0.08 * _rank(col("stance_fore", stand), False)
  )

  # 6. Standing stability.
  stability_score = (
    0.25 * _rank(col("lin_vel_xy", stand), False)
    + 0.15 * _rank(col("ang_vel", stand), False)
    + 0.20 * _rank(col("joint_still", stand), False)
    + 0.15 * _rank(col("action_rate", stand), False)
    + 0.15 * _rank(col("xy_drift", stand), False)
    + 0.10 * _rank(col("fall_rate", stand), False)
  )

  weights = {
    "tracking": 0.20,
    "landing": 0.15,
    "slip": 0.15,
    "leg_shape": 0.15,
    "posture": 0.15,
    "stability": 0.20,
  }
  parts = {
    "tracking": track_score,
    "landing": land_score,
    "slip": slip_score,
    "leg_shape": leg_score,
    "posture": posture_score,
    "stability": stability_score,
  }
  total = sum(weights[k] * parts[k] for k in weights)
  fall_all = np.nanmean(np.stack([col("fall_rate", s) for s, _ in SCENARIOS]), axis=0)
  total = total * (1.0 - np.clip(fall_all, 0.0, 1.0) * 0.8)

  ranked = []
  for i, row in enumerate(rows):
    out = dict(row)
    out["scores"] = {k: float(v[i]) for k, v in parts.items()}
    out["scores"]["total"] = float(total[i])
    out["diagnostics"] = {
      "err_main": float(err_main[i]),
      "err_off": float(err_off[i]),
      "fall_rate_all": float(fall_all[i]),
    }
    ranked.append(out)
  ranked.sort(key=lambda r: r["scores"]["total"], reverse=True)
  for i, r in enumerate(ranked):
    r["rank"] = i + 1
  return ranked


def _resolve_task(task_id: str) -> str:
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  tasks = list_tasks()
  if task_id in tasks:
    return task_id
  for cand in TASK_CANDIDATES:
    if cand in tasks:
      return cand
  raise RuntimeError(f"Task not found. Available: {tasks}")


def _write_report(path: Path, ranked: list[dict], stance_target: float) -> None:
  best = ranked[0]
  lines = [
    "# CASBOT02-Leg-AMP checkpoint evaluation (6 axes)",
    "",
    f"**Best overall: `{best['ckpt']}`**",
    "",
    "Weights: tracking 0.20, landing 0.15, slip 0.15, leg shape 0.15, "
    "posture 0.15, stability 0.20.",
    f"Stance-width target {stance_target:.3f} m, height target "
    f"{DEFAULT_HEIGHT:.2f} m, clearance target {TARGET_FOOT_CLEARANCE:.2f} m.",
    "",
    "## Scores (0-1, percentile within this run)",
    "",
    "| Rank | Checkpoint | Total | Tracking | Landing | Slip | LegShape | Posture | Stability |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for r in ranked:
    s = r["scores"]
    lines.append(
      f"| {r['rank']} | `{r['ckpt']}` | {s['total']:.3f} | {s['tracking']:.3f} | "
      f"{s['landing']:.3f} | {s['slip']:.3f} | {s['leg_shape']:.3f} | "
      f"{s['posture']:.3f} | {s['stability']:.3f} |"
    )

  lines += [
    "",
    "## Raw metrics",
    "",
    "vel err = mean |commanded - measured| over the 6 locomotion commands; "
    "landing force in body weights; slip in m/s; angles in rad; drift in m.",
    "",
    "| Ckpt | VelErr | OffAxis | LandF | LandF95 | TdVz | Slip(walk) | Slip(stand) | "
    "HipRoll | StanceW | LRasym | Clear | Tilt | Height | HomeErr | Sway | JointVel | Drift | Fall |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for r in ranked:
    st = r["scenarios"]["stand"]
    loco = [r["scenarios"][s] for s in LOCO_SCENARIOS]
    fwd = [r["scenarios"][s] for s in FWD_SCENARIOS]

    def lm(key: str) -> float:
      return float(np.nanmean([m.get(key, np.nan) for m in loco]))

    def fm(key: str) -> float:
      return float(np.nanmean([m.get(key, np.nan) for m in fwd]))

    d = r["diagnostics"]
    lines.append(
      f"| `{r['ckpt']}` | {d['err_main']:.3f} | {d['err_off']:.3f} | "
      f"{lm('landing_force_mean'):.2f} | {lm('landing_force_p95'):.2f} | "
      f"{lm('touchdown_vz'):.3f} | {lm('foot_slip'):.3f} | {st['foot_slip']:.4f} | "
      f"{lm('hip_roll_abs'):.4f} | {lm('stance_width'):.3f} | {lm('lr_asym'):.3f} | "
      f"{fm('peak_foot_clearance'):.3f} | {st['tilt']:.4f} | {st['height']:.3f} | "
      f"{st['joint_home_err']:.3f} | {st['lin_vel_xy']:.4f} | "
      f"{st['joint_still']:.3f} | {st['xy_drift']:.3f} | {d['fall_rate_all']:.2f} |"
    )

  lines += ["", "## Per-scenario velocity tracking (measured vs commanded)", ""]
  header = "| Ckpt | " + " | ".join(LOCO_SCENARIOS) + " |"
  lines.append(header)
  lines.append("| --- | " + " | ".join("---:" for _ in LOCO_SCENARIOS) + " |")
  for r in ranked:
    cells = []
    for s in LOCO_SCENARIOS:
      m = r["scenarios"][s]
      if s == "turn_0.8":
        cells.append(f"{m['wz']:.2f}")
      elif s == "side_0.3":
        cells.append(f"{m['vy']:.2f}")
      else:
        cells.append(f"{m['vx']:.2f}")
    lines.append(f"| `{r['ckpt']}` | " + " | ".join(cells) + " |")

  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(cfg: EvalConfig) -> None:
  configure_torch_backends()
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id = _resolve_task(cfg.task_id)
  run_dir = Path(cfg.run_dir).resolve()
  selected = []
  for p in sorted(run_dir.glob("model_*.pt")):
    try:
      it = int(p.stem.split("_")[1])
    except (IndexError, ValueError):
      continue
    if cfg.ckpt_min <= it <= cfg.ckpt_max and it % cfg.ckpt_step == 0:
      selected.append((it, p))
  selected.sort()
  if not selected:
    raise FileNotFoundError(f"No checkpoints in {run_dir}")

  print(f"[INFO] task={task_id} envs={cfg.num_envs} ckpts={len(selected)}")
  env_cfg = _prepare_env_cfg(task_id, cfg.num_envs, cfg.seed)
  agent_cfg = load_rl_cfg(task_id)
  if hasattr(agent_cfg, "amp_num_preload_transitions"):
    agent_cfg.amp_num_preload_transitions = min(
      int(agent_cfg.amp_num_preload_transitions), 8000
    )
  device = cfg.device if torch.cuda.is_available() else "cpu"

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner = load_runner_cls(task_id)(env, asdict(agent_cfg), device=device)

  robot = raw_env.scene["robot"]
  names = list(robot.joint_names)
  joint_ids = {j: names.index(j) for j in CASBOT02_LEG_ONLY_JOINT_NAMES}
  ids, site_names = robot.find_sites(("left_force", "right_force"))
  site_map = {n: i for n, i in zip(site_names, ids)}
  foot_ids = (int(site_map["left_force"]), int(site_map["right_force"]))

  body_weight = float(np.sum(raw_env.sim.mj_model.body_mass)) * 9.81
  raw_env.reset()
  foot_pos = robot.data.site_pos_w[:, [foot_ids[0], foot_ids[1]], :]
  stance_target = float((foot_pos[:, 0, 1] - foot_pos[:, 1, 1]).abs().mean().item())
  print(
    f"[INFO] body_weight={body_weight:.1f} N  stance_target={stance_target:.3f} m  "
    f"feet={foot_ids}"
  )

  rows = []
  for it, path in selected:
    print(f"\n===== {path.name} =====", flush=True)
    runner.load(str(path), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    scenario_metrics = {}
    for name, cmd in SCENARIOS:
      dur = cfg.stand_seconds if name == "stand" else cfg.walk_seconds
      m = run_scenario(
        raw_env,
        policy,
        cmd,
        dur,
        cfg.warmup_seconds,
        joint_ids,
        foot_ids,
        body_weight,
        cfg.seed,
      )
      scenario_metrics[name] = m
      print(
        f"  {name:9s} vx={m['vx']:+.2f} vy={m['vy']:+.2f} wz={m['wz']:+.2f} "
        f"fall={m['fall_rate']:.2f} landF={m['landing_force_mean']:.2f} "
        f"slip={m['foot_slip']:.3f} hipRoll={m['hip_roll_abs']:.3f} "
        f"tilt={m['tilt']:.3f} h={m['height']:.3f}",
        flush=True,
      )
    rows.append({"ckpt": path.name, "iter": it, "scenarios": scenario_metrics})
    if torch.cuda.is_available():
      torch.cuda.empty_cache()

  ranked = score_models(rows, stance_target)
  out_json = run_dir / f"{cfg.out_prefix}_results.json"
  out_md = run_dir / f"{cfg.out_prefix}_ranking.md"
  out_json.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
  _write_report(out_md, ranked, stance_target)

  print("\n========== RANKING ==========")
  for r in ranked:
    s = r["scores"]
    print(
      f"{r['rank']:2d}. {r['ckpt']:16s} total={s['total']:.3f}  "
      f"track={s['tracking']:.3f} land={s['landing']:.3f} slip={s['slip']:.3f} "
      f"leg={s['leg_shape']:.3f} post={s['posture']:.3f} stab={s['stability']:.3f}"
    )
  print(f"\nBest: {ranked[0]['ckpt']}")
  print(f"Wrote {out_json}\nWrote {out_md}")
  raw_env.close()


if __name__ == "__main__":
  main(tyro.cli(EvalConfig))
