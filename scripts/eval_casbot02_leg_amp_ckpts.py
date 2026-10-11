"""Headless mjlab evaluation for CASBOT02-Leg-AMP checkpoints.

Priority:
  1. Standing sway, posture, still joint angles
  2. Walk foot clearance, foot slip, leg shape
  3. Walking ability and smoothness
"""

from __future__ import annotations

import json
import math
import sys
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


RUN_DIR = Path(
  "logs/rsl_rl/casbot02_leg_amp_locomotion/2026-08-25_22-59-57"
).resolve()
TASK_ID = "Mjlab-Casbot02-Leg-AMP-Flat"
# registry name used in this repo
TASK_CANDIDATES = (
  "Casbot02-Leg-AMP-Flat",
  "Mjlab-Casbot02-Leg-AMP-Flat",
)

DEFAULT_HEIGHT = 0.92
TARGET_STANCE_WIDTH = 0.285
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


@dataclass(frozen=True)
class EvalConfig:
  run_dir: str = str(RUN_DIR)
  task_id: str = "Casbot02-Leg-AMP-Flat"
  num_envs: int = 24
  device: str = "cuda:0"
  seed: int = 42
  stand_seconds: float = 6.0
  walk_seconds: float = 8.0
  warmup_seconds: float = 1.5
  ckpt_min: int = 4000
  ckpt_max: int = 999999
  ckpt_step: int = 1000


def _prepare_env_cfg(task_id: str, num_envs: int, seed: int):
  env_cfg = load_env_cfg(task_id, play=True)
  env_cfg.scene.num_envs = num_envs
  env_cfg.seed = seed
  env_cfg.episode_length_s = 30.0
  env_cfg.observations["actor"].enable_corruption = False
  env_cfg.events.pop("push_robot", None)
  env_cfg.curriculum = {}
  if "init_motion_loader" in env_cfg.events:
    env_cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 0.0
    env_cfg.events["init_motion_loader"].params["max_delay_steps"] = 0
  env_cfg.events.pop("reset_from_motion", None)
  env_cfg.events.pop("randomize_terrain", None)

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


def _joint_index_map(robot) -> dict[str, int]:
  names = list(robot.joint_names)
  return {name: names.index(name) for name in CASBOT02_LEG_ONLY_JOINT_NAMES}


def _site_ids(robot) -> tuple[int, int]:
  ids, names = robot.find_sites(("left_force", "right_force"))
  name_to_id = {n: i for n, i in zip(names, ids)}
  return int(name_to_id["left_force"]), int(name_to_id["right_force"])


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> float:
  if mask.ndim < x.ndim:
    mask = mask.reshape((-1,) + (1,) * (x.ndim - 1))
  valid = mask.expand_as(x)
  if not bool(valid.any()):
    return float("nan")
  return float(x[valid].mean().item())


def _masked_std(x: torch.Tensor, mask: torch.Tensor) -> float:
  if mask.ndim < x.ndim:
    mask = mask.reshape((-1,) + (1,) * (x.ndim - 1))
  valid = mask.expand_as(x)
  if int(valid.sum().item()) < 2:
    return float("nan")
  return float(x[valid].std().item())


def run_scenario(
  env: ManagerBasedRlEnv,
  policy,
  command: tuple[float, float, float],
  duration_s: float,
  warmup_s: float,
  joint_ids: dict[str, int],
  foot_ids: tuple[int, int],
) -> dict[str, float]:
  robot = env.scene["robot"]
  contact = env.scene["feet_ground_contact"]
  vx, vy, wz = command
  is_stand = abs(vx) + abs(vy) + abs(wz) < 1e-6

  obs, _ = env.reset()
  _force_command(env, vx, vy, wz)
  # Recompute actor obs so the first action sees the forced command.
  obs = env.observation_manager.compute()

  n = env.num_envs
  device = env.device
  alive = torch.ones(n, dtype=torch.bool, device=device)
  dt = env.step_dt
  total_steps = int(round(duration_s / dt))
  warmup_steps = int(round(warmup_s / dt))

  left_id, right_id = foot_ids
  l1, r1 = joint_ids["leg_l1_joint"], joint_ids["leg_r1_joint"]
  l2, r2 = joint_ids["leg_l2_joint"], joint_ids["leg_r2_joint"]
  l4, r4 = joint_ids["leg_l4_joint"], joint_ids["leg_r4_joint"]
  l5, r5 = joint_ids["leg_l5_joint"], joint_ids["leg_r5_joint"]
  leg_idx = torch.tensor(
    [joint_ids[n] for n in CASBOT02_LEG_ONLY_JOINT_NAMES],
    device=device,
    dtype=torch.long,
  )
  home = torch.tensor(
    [HOME_LEG[n] for n in CASBOT02_LEG_ONLY_JOINT_NAMES],
    device=device,
    dtype=torch.float32,
  )

  sums = {k: 0.0 for k in (
    "lin_vel_xy", "lin_vel_z", "ang_vel", "tilt", "height",
    "joint_still", "joint_home_err", "joint_lr_asym",
    "foot_slip", "stance_width", "stance_fore",
    "swing_height", "swing_knee", "stance_knee", "hip_roll_abs",
    "track_xy", "track_yaw", "action_rate", "joint_acc",
  )}
  count = 0
  swing_heights = []
  max_tilt = 0.0
  fell = torch.zeros(n, dtype=torch.bool, device=device)

  actor_obs = obs["actor"] if hasattr(obs, "keys") else obs
  for step in range(total_steps):
    _force_command(env, vx, vy, wz)
    with torch.inference_mode():
      actions = policy(actor_obs)
    obs, _, terminated, truncated, extras = env.step(actions)
    actor_obs = obs["actor"] if hasattr(obs, "keys") else obs
    done = (terminated | truncated).to(dtype=torch.bool)
    # Count a fall only if this env was still in its first episode.
    newly_fallen = alive & done & (~truncated.to(dtype=torch.bool))
    fell |= newly_fallen

    if step < warmup_steps:
      alive = alive & ~done
      continue

    mask = alive
    if not bool(mask.any()):
      alive = alive & ~done
      break

    root_lin_b = robot.data.root_link_lin_vel_b
    root_ang_b = robot.data.root_link_ang_vel_b
    grav = robot.data.projected_gravity_b
    height = robot.data.root_link_pos_w[:, 2]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    joint_acc = robot.data.joint_acc if hasattr(robot.data, "joint_acc") else None
    foot_pos = robot.data.site_pos_w[:, [left_id, right_id], :]
    foot_vel = robot.data.site_lin_vel_w[:, [left_id, right_id], :2]
    in_contact = (contact.data.found > 0).float()
    if in_contact.ndim == 3:
      in_contact = in_contact.squeeze(-1)
    # Contact sensor order follows primary names; assume left then right if 2.
    if in_contact.shape[1] != 2:
      in_contact = in_contact[:, :2]

    cmd = torch.tensor([vx, vy, wz], device=device)
    track_xy = torch.norm(root_lin_b[:, :2] - cmd[:2], dim=-1)
    track_yaw = torch.abs(root_ang_b[:, 2] - cmd[2])
    tilt = torch.norm(grav[:, :2], dim=-1)
    lin_xy = torch.norm(root_lin_b[:, :2], dim=-1)
    ang = torch.norm(root_ang_b, dim=-1)
    slip = torch.norm(foot_vel, dim=-1) * in_contact
    stance_width = torch.abs(foot_pos[:, 0, 1] - foot_pos[:, 1, 1])
    stance_fore = torch.abs(foot_pos[:, 0, 0] - foot_pos[:, 1, 0])
    swing = 1.0 - in_contact
    swing_h = foot_pos[:, :, 2] * swing
    knee = joint_pos[:, [l4, r4]]
    hip_roll = joint_pos[:, [l2, r2]]
    swing_knee = knee * swing
    stance_knee = knee * in_contact

    leg_pos = joint_pos[:, leg_idx]
    joint_still = torch.norm(joint_vel[:, leg_idx], dim=-1)
    joint_home_err = torch.norm(leg_pos - home, dim=-1)
    left = torch.stack(
      [joint_pos[:, l1], joint_pos[:, l2], joint_pos[:, l4], joint_pos[:, l5]], dim=-1
    )
    right = torch.stack(
      [joint_pos[:, r1], -joint_pos[:, r2], joint_pos[:, r4], joint_pos[:, r5]], dim=-1
    )
    joint_lr_asym = torch.norm(left - right, dim=-1)
    hip_roll_abs = hip_roll.abs().mean(dim=-1)

    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    action_rate = torch.norm(action - prev_action, dim=-1)

    if joint_acc is None:
      jacc = torch.zeros(n, device=device)
    else:
      jacc = torch.norm(joint_acc[:, leg_idx], dim=-1)

    m = mask
    sums["lin_vel_xy"] += _masked_mean(lin_xy, m)
    sums["lin_vel_z"] += _masked_mean(root_lin_b[:, 2].abs(), m)
    sums["ang_vel"] += _masked_mean(ang, m)
    sums["tilt"] += _masked_mean(tilt, m)
    sums["height"] += _masked_mean(height, m)
    sums["joint_still"] += _masked_mean(joint_still, m)
    sums["joint_home_err"] += _masked_mean(joint_home_err, m)
    sums["joint_lr_asym"] += _masked_mean(joint_lr_asym, m)
    sums["foot_slip"] += _masked_mean(slip, m)
    sums["stance_width"] += _masked_mean(stance_width, m)
    sums["stance_fore"] += _masked_mean(stance_fore, m)
    sums["swing_height"] += _masked_mean(swing_h, m)
    sums["swing_knee"] += _masked_mean(swing_knee, m)
    sums["stance_knee"] += _masked_mean(stance_knee, m)
    sums["hip_roll_abs"] += _masked_mean(hip_roll_abs, m)
    sums["track_xy"] += _masked_mean(track_xy, m)
    sums["track_yaw"] += _masked_mean(track_yaw, m)
    sums["action_rate"] += _masked_mean(action_rate, m)
    sums["joint_acc"] += _masked_mean(jacc, m)
    count += 1
    max_tilt = max(max_tilt, _masked_mean(tilt, m))
    if float(swing.sum().item()) > 0:
      swing_heights.append(float(swing_h[swing > 0].max().item()))

    alive = alive & ~done

  denom = max(count, 1)
  peak_clearance = float(np.mean(swing_heights)) if swing_heights else 0.0
  metrics = {k: v / denom for k, v in sums.items()}
  metrics["peak_foot_clearance"] = peak_clearance
  metrics["fall_rate"] = float(fell.float().mean().item())
  metrics["alive_frac"] = float(alive.float().mean().item())
  metrics["max_tilt"] = max_tilt
  metrics["height_err"] = abs(metrics["height"] - DEFAULT_HEIGHT)
  metrics["stance_width_err"] = abs(metrics["stance_width"] - TARGET_STANCE_WIDTH)
  metrics["is_stand"] = 1.0 if is_stand else 0.0
  metrics["cmd_vx"] = vx
  return metrics


def _percentile_ranks(values: np.ndarray, higher_better: bool) -> np.ndarray:
  x = np.asarray(values, dtype=np.float64)
  finite = np.isfinite(x)
  ranks = np.full_like(x, 0.5)
  if finite.sum() < 2:
    return ranks
  order = x[finite] if higher_better else -x[finite]
  # average ranks, then scale to 0-1
  sorter = np.argsort(order, kind="mergesort")
  inv = np.empty_like(sorter)
  inv[sorter] = np.arange(len(sorter))
  # handle ties by averaging
  raw = np.empty(len(order))
  raw[sorter] = np.arange(len(order), dtype=np.float64)
  # convert to percentile
  ranks[finite] = raw / max(len(order) - 1, 1)
  return ranks


def _band_score(values: np.ndarray, target: float, width: float) -> np.ndarray:
  x = np.asarray(values, dtype=np.float64)
  return np.exp(-((x - target) / width) ** 2)


def rank_models(rows: list[dict]) -> list[dict]:
  names = [r["ckpt"] for r in rows]
  n = len(rows)

  def col(key: str, scenario: str) -> np.ndarray:
    return np.array([r["scenarios"][scenario][key] for r in rows], dtype=np.float64)

  stand = "stand"
  walks = ["walk_0.3", "walk_0.6", "walk_1.0"]

  # --- Standing ---
  stand_parts = {
    "sway": 0.5 * _percentile_ranks(col("lin_vel_xy", stand), False)
    + 0.3 * _percentile_ranks(col("ang_vel", stand), False)
    + 0.2 * _percentile_ranks(col("lin_vel_z", stand), False),
    "posture": 0.45 * _percentile_ranks(col("tilt", stand), False)
    + 0.35 * _percentile_ranks(col("height_err", stand), False)
    + 0.20 * _percentile_ranks(col("stance_width_err", stand), False),
    "joints": 0.40 * _percentile_ranks(col("joint_still", stand), False)
    + 0.35 * _percentile_ranks(col("joint_home_err", stand), False)
    + 0.25 * _percentile_ranks(col("joint_lr_asym", stand), False),
    "slip": _percentile_ranks(col("foot_slip", stand), False),
    "survive": _percentile_ranks(1.0 - col("fall_rate", stand), True),
  }
  stand_score = (
    0.28 * stand_parts["sway"]
    + 0.27 * stand_parts["posture"]
    + 0.25 * stand_parts["joints"]
    + 0.10 * stand_parts["slip"]
    + 0.10 * stand_parts["survive"]
  )
  stand_score = stand_score * (1.0 - np.clip(col("fall_rate", stand), 0.0, 1.0) * 0.7)

  # --- Gait quality (walk) ---
  def mean_walk(key: str) -> np.ndarray:
    return np.mean([col(key, s) for s in walks], axis=0)

  clearance = mean_walk("peak_foot_clearance")
  clearance_fit = _band_score(clearance, TARGET_FOOT_CLEARANCE, 0.035)
  gait_parts = {
    "clearance": _percentile_ranks(clearance_fit, True),
    "slip": _percentile_ranks(mean_walk("foot_slip"), False),
    "leg": 0.35 * _percentile_ranks(_band_score(mean_walk("swing_knee"), 0.55, 0.25), True)
    + 0.25 * _percentile_ranks(_band_score(mean_walk("stance_knee"), 0.32, 0.18), True)
    + 0.20 * _percentile_ranks(mean_walk("hip_roll_abs"), False)
    + 0.20 * _percentile_ranks(mean_walk("joint_lr_asym"), False),
  }
  gait_score = (
    0.38 * gait_parts["clearance"]
    + 0.32 * gait_parts["slip"]
    + 0.30 * gait_parts["leg"]
  )

  # --- Walking ability ---
  walk_parts = {
    "track": _percentile_ranks(mean_walk("track_xy"), False),
    "survive": _percentile_ranks(1.0 - mean_walk("fall_rate"), True),
    "smooth": 0.5 * _percentile_ranks(mean_walk("action_rate"), False)
    + 0.3 * _percentile_ranks(mean_walk("ang_vel"), False)
    + 0.2 * _percentile_ranks(mean_walk("joint_acc"), False),
  }
  walk_score = (
    0.45 * walk_parts["track"]
    + 0.30 * walk_parts["survive"]
    + 0.25 * walk_parts["smooth"]
  )
  walk_score = walk_score * (1.0 - np.clip(mean_walk("fall_rate"), 0.0, 1.0) * 0.6)

  # User priority: stand > gait > walk ability
  total = 0.45 * stand_score + 0.32 * gait_score + 0.23 * walk_score

  ranked = []
  for i, row in enumerate(rows):
    out = dict(row)
    out["scores"] = {
      "stand": float(stand_score[i]),
      "gait": float(gait_score[i]),
      "walk": float(walk_score[i]),
      "total": float(total[i]),
      "stand_sway": float(stand_parts["sway"][i]),
      "stand_posture": float(stand_parts["posture"][i]),
      "stand_joints": float(stand_parts["joints"][i]),
      "gait_clearance": float(gait_parts["clearance"][i]),
      "gait_slip": float(gait_parts["slip"][i]),
      "gait_leg": float(gait_parts["leg"][i]),
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


def main(cfg: EvalConfig) -> None:
  configure_torch_backends()
  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  task_id = _resolve_task(cfg.task_id)
  run_dir = Path(cfg.run_dir).resolve()
  ckpts = sorted(run_dir.glob("model_*.pt"))
  selected = []
  for p in ckpts:
    try:
      it = int(p.stem.split("_")[1])
    except (IndexError, ValueError):
      continue
    if cfg.ckpt_min <= it <= cfg.ckpt_max and (it - cfg.ckpt_min) % cfg.ckpt_step == 0:
      selected.append((it, p))
  if not selected:
    raise FileNotFoundError(f"No checkpoints in {run_dir}")

  print(f"[INFO] Task={task_id}  envs={cfg.num_envs}  ckpts={len(selected)}")
  env_cfg = _prepare_env_cfg(task_id, cfg.num_envs, cfg.seed)
  agent_cfg = load_rl_cfg(task_id)
  device = cfg.device if torch.cuda.is_available() else "cpu"

  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(task_id)
  runner = runner_cls(env, asdict(agent_cfg), device=device)

  robot = raw_env.scene["robot"]
  joint_ids = _joint_index_map(robot)
  foot_ids = _site_ids(robot)
  print(f"[INFO] joints={list(joint_ids)}  feet={foot_ids}")

  scenarios = [
    ("stand", (0.0, 0.0, 0.0), cfg.stand_seconds),
    ("walk_0.3", (0.3, 0.0, 0.0), cfg.walk_seconds),
    ("walk_0.6", (0.6, 0.0, 0.0), cfg.walk_seconds),
    ("walk_1.0", (1.0, 0.0, 0.0), cfg.walk_seconds),
  ]

  rows = []
  for it, path in selected:
    print(f"\n===== Evaluating {path.name} =====")
    runner.load(str(path), load_optimizer=False)
    policy = runner.get_inference_policy(device=device)
    scenario_metrics = {}
    for name, cmd, dur in scenarios:
      print(f"  {name} cmd={cmd} ...", flush=True)
      scenario_metrics[name] = run_scenario(
        raw_env,
        policy,
        cmd,
        dur,
        cfg.warmup_seconds,
        joint_ids,
        foot_ids,
      )
      m = scenario_metrics[name]
      print(
        f"    fall={m['fall_rate']:.2f} tilt={m['tilt']:.3f} h={m['height']:.3f} "
        f"slip={m['foot_slip']:.3f} clear={m['peak_foot_clearance']:.3f} "
        f"track={m['track_xy']:.3f} still={m['joint_still']:.3f}"
      )
    rows.append({"ckpt": path.name, "iter": it, "scenarios": scenario_metrics})

  ranked = rank_models(rows)
  out_json = run_dir / "eval_results.json"
  out_md = run_dir / "eval_ranking.md"
  out_json.write_text(json.dumps(ranked, indent=2), encoding="utf-8")
  _write_report(out_md, ranked)
  print("\n========== RANKING ==========")
  for r in ranked:
    s = r["scores"]
    print(
      f"{r['rank']:2d}. {r['ckpt']:16s}  total={s['total']:.3f}  "
      f"stand={s['stand']:.3f} gait={s['gait']:.3f} walk={s['walk']:.3f}"
    )
  print(f"\nBest: {ranked[0]['ckpt']}")
  print(f"Wrote {out_json}")
  print(f"Wrote {out_md}")
  raw_env.close()


def _write_report(path: Path, ranked: list[dict]) -> None:
  best = ranked[0]
  lines = [
    "# CASBOT02-Leg-AMP checkpoint evaluation",
    "",
    f"**Best overall: `{best['ckpt']}`**",
    "",
    "Priority weights: standing 0.45, gait quality 0.32, walking ability 0.23.",
    "",
    "| Rank | Checkpoint | Total | Stand | Gait | Walk | Stand fall | Walk0.6 fall | Stand sway vel | Tilt | Height | Joint still | Home err | Peak clear | Walk slip | Track xy |",
    "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]
  for r in ranked:
    st = r["scenarios"]["stand"]
    w6 = r["scenarios"]["walk_0.6"]
    s = r["scores"]
    lines.append(
      f"| {r['rank']} | `{r['ckpt']}` | {s['total']:.3f} | {s['stand']:.3f} | "
      f"{s['gait']:.3f} | {s['walk']:.3f} | {st['fall_rate']:.2f} | {w6['fall_rate']:.2f} | "
      f"{st['lin_vel_xy']:.3f} | {st['tilt']:.3f} | {st['height']:.3f} | "
      f"{st['joint_still']:.3f} | {st['joint_home_err']:.3f} | "
      f"{w6['peak_foot_clearance']:.3f} | {w6['foot_slip']:.3f} | {w6['track_xy']:.3f} |"
    )
  path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
  main(tyro.cli(EvalConfig))
