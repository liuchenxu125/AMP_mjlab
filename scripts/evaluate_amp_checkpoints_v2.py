"""Checkpoint evaluation v2: heading-relative stance, pooled tails and incremental results."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.utils.torch import configure_torch_backends


def _checkpoint_step(path: Path) -> int:
  return int(path.stem.rsplit("_", 1)[1])


def _mean(values: list[float]) -> float:
  return float(np.mean(values)) if values else float("nan")


def _percentile(values: list[float], q: float) -> float:
  return float(np.percentile(values, q)) if values else float("nan")


def _append_mean(target: list[float], value: torch.Tensor) -> None:
  if value.numel():
    target.append(float(value.float().mean().item()))


def _score_rows(rows: list[dict[str, float | str | int]]) -> None:
  # Every top-level dimension has equal weight. Metrics within a dimension are
  # also equal-weighted percentile ranks so their physical units cannot dominate.
  dimensions: dict[str, list[tuple[str, bool]]] = {
    "speed_tracking_score": [
      ("moving_vel_xy_error", False),
      ("moving_vel_yaw_error", False),
      ("fall_rate_per_1000_env_steps", False),
    ],
    "landing_impact_score": [
      ("landing_force_mean_n", False),
      ("landing_force_p95_n", False),
    ],
    "foot_slip_score": [
      ("moving_slip_mean_mps", False),
      ("moving_slip_p95_mps", False),
    ],
    "leg_form_score": [
      ("fixed_judge_amp_style", True),
      ("action_acceleration_mean", False),
      ("joint_limit_violation_mean_rad", False),
    ],
    "standing_posture_score": [
      ("standing_tilt_mean_deg", False),
      ("standing_height_error_mean_m", False),
      ("standing_foot_fore_error_mean_m", False),
      ("standing_foot_lateral_error_mean_m", False),
    ],
    "standing_stability_score": [
      ("standing_root_speed_mean_mps", False),
      ("standing_ang_vel_xy_mean_rps", False),
      ("standing_slip_mean_mps", False),
      ("standing_fall_rate_per_1000_env_steps", False),
    ],
  }

  n = len(rows)
  for dim, specs in dimensions.items():
    total = np.zeros(n, dtype=np.float64)
    used = 0
    for metric, higher_is_better in specs:
      values = np.asarray([float(row[metric]) for row in rows])
      finite = np.isfinite(values)
      if not finite.any():
        continue
      fill = np.nanmedian(values[finite])
      values = np.where(finite, values, fill)
      ranks = rankdata(values, method="average")
      if not higher_is_better:
        ranks = n + 1 - ranks
      percentile = 100.0 * (ranks - 1) / max(n - 1, 1)
      total += percentile
      used += 1
    scores = total / max(used, 1)
    for row, score in zip(rows, scores, strict=True):
      row[dim] = float(score)

  score_names = list(dimensions)
  for row in rows:
    row["overall_score"] = float(np.mean([float(row[name]) for name in score_names]))


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("run_dirs", nargs="+", type=Path)
  parser.add_argument("--judge-checkpoint", type=Path)
  parser.add_argument("--task", default="Casbot02-Leg-AMP-Flat")
  parser.add_argument("--num-envs", type=int, default=128)
  parser.add_argument("--steps", type=int, default=600)
  parser.add_argument("--warmup-steps", type=int, default=50)
  parser.add_argument("--seed", type=int, default=20260903)
  parser.add_argument("--standing-only", action="store_true")
  parser.add_argument("--no-pushes", action="store_true")
  parser.add_argument("--lin-vel-x-range", nargs=2, type=float)
  parser.add_argument("--lin-vel-y-range", nargs=2, type=float)
  parser.add_argument("--ang-vel-z-range", nargs=2, type=float)
  parser.add_argument("--waist-com-x-range", nargs=2, type=float)
  parser.add_argument("--joint-acc-weight", type=float)
  parser.add_argument("--action-rate-weight", type=float)
  parser.add_argument("--foot-slip-weight", type=float)
  parser.add_argument("--soft-landing-weight", type=float)
  parser.add_argument("--output-dir", type=Path, default=Path("logs/evaluation"))
  args = parser.parse_args()

  import mjlab.tasks  # noqa: F401
  import src.tasks  # noqa: F401

  configure_torch_backends()
  random.seed(args.seed)
  np.random.seed(args.seed)
  torch.manual_seed(args.seed)

  checkpoints: list[Path] = []
  for run_path in args.run_dirs:
    if run_path.is_file():
      checkpoints.append(run_path)
    else:
      checkpoints.extend(sorted(run_path.glob("model_*.pt"), key=_checkpoint_step))
  if not checkpoints:
    raise FileNotFoundError("No model_*.pt checkpoints found")

  # Snapshot the checkpoint list before evaluation; an active training run may
  # create newer files while this script is running.
  checkpoints = list(checkpoints)
  # Copied run directories may not preserve chronological mtimes, so select the
  # common AMP judge by training iteration rather than filesystem timestamp.
  judge_checkpoint = (
    args.judge_checkpoint.resolve()
    if args.judge_checkpoint is not None
    else max(checkpoints, key=_checkpoint_step)
  )
  if not judge_checkpoint.is_file():
    raise FileNotFoundError(f"Judge checkpoint not found: {judge_checkpoint}")
  print(f"[EVAL] snapshot checkpoints={len(checkpoints)} judge={judge_checkpoint}", flush=True)

  env_cfg = load_env_cfg(args.task, play=False)
  agent_cfg = load_rl_cfg(args.task)
  env_cfg.scene.num_envs = args.num_envs
  env_cfg.seed = args.seed
  # Increase standing coverage while retaining the original command ranges,
  # observation corruption, pushes, reset distribution, and domain randomization.
  twist_cfg = env_cfg.commands["twist"]
  twist_cfg.rel_standing_envs = 0.25
  if args.lin_vel_x_range is not None:
    twist_cfg.ranges.lin_vel_x = tuple(args.lin_vel_x_range)
  if args.lin_vel_y_range is not None:
    twist_cfg.ranges.lin_vel_y = tuple(args.lin_vel_y_range)
  if args.ang_vel_z_range is not None:
    twist_cfg.ranges.ang_vel_z = tuple(args.ang_vel_z_range)
  if args.waist_com_x_range is not None:
    env_cfg.events["waist_com_backward"].params["ranges"][0] = tuple(
      args.waist_com_x_range
    )
  if args.joint_acc_weight is not None:
    env_cfg.rewards["joint_acc_l2"].weight = args.joint_acc_weight
  if args.action_rate_weight is not None:
    env_cfg.rewards["action_rate_l2"].weight = args.action_rate_weight
  if args.foot_slip_weight is not None:
    env_cfg.rewards["foot_slip"].weight = args.foot_slip_weight
  if args.soft_landing_weight is not None:
    env_cfg.rewards["soft_landing"].weight = args.soft_landing_weight

  if args.standing_only:
    env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.steps * env_cfg.decimation * env_cfg.sim.mujoco.timestep + 1.0)
    twist_cfg.rel_standing_envs = 1.0
    twist_cfg.heading_command = False
    twist_cfg.ranges.lin_vel_x = (0.0, 0.0)
    twist_cfg.ranges.lin_vel_y = (0.0, 0.0)
    twist_cfg.ranges.ang_vel_z = (0.0, 0.0)
    twist_cfg.ranges.heading = None
  if args.no_pushes:
    env_cfg.events.pop("push_robot", None)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  from mjlab.utils.os import dump_yaml
  dump_yaml(args.output_dir / "effective_env.yaml", env_cfg)
  dump_yaml(args.output_dir / "effective_agent.yaml", agent_cfg)
  base_env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda:0", render_mode=None)
  env = RslRlVecEnvWrapper(base_env, clip_actions=agent_cfg.clip_actions)
  runner_cls = load_runner_cls(args.task) or MjlabOnPolicyRunner
  runner = runner_cls(env, asdict(agent_cfg), log_dir=None, device="cuda:0")

  # A common discriminator and normalizer make the AMP-style score comparable
  # across checkpoints. Using each checkpoint's own discriminator would move the
  # measuring scale together with the policy.
  runner.load(str(judge_checkpoint), load_optimizer=False)
  fixed_judge = copy.deepcopy(runner.alg.discriminator).eval()
  fixed_amp_normalizer = copy.deepcopy(runner.alg.amp_normalizer)

  robot = base_env.scene["robot"]
  contact = base_env.scene["feet_ground_contact"]
  foot_ids, _ = robot.find_sites(("left_force", "right_force"), preserve_order=True)
  leg_ids, _ = robot.find_joints(
    (
      "leg_l1_joint", "leg_l2_joint", "leg_l3_joint", "leg_l4_joint",
      "leg_l5_joint", "leg_l6_joint", "leg_r1_joint", "leg_r2_joint",
      "leg_r3_joint", "leg_r4_joint", "leg_r5_joint", "leg_r6_joint",
    ),
    preserve_order=True,
  )

  rows: list[dict[str, float | str | int]] = []
  try:
    for index, checkpoint in enumerate(checkpoints, start=1):
      runner.load(str(checkpoint), load_optimizer=False)
      policy = runner.get_inference_policy(device="cuda:0")

      # Rewind all stochastic reset/command streams for paired comparisons.
      random.seed(args.seed)
      np.random.seed(args.seed)
      torch.manual_seed(args.seed)
      env.seed(args.seed)
      obs, _ = env.reset()
      prev_amp = obs["amp"].clone()
      prev_action: torch.Tensor | None = None
      prev_action_delta: torch.Tensor | None = None

      samples: dict[str, list[float]] = {
        name: []
        for name in (
          "vel_xy", "vel_yaw", "landing_mean", "landing_p95", "moving_slip_mean",
          "moving_slip_p95", "style", "action_acc", "joint_limit", "stand_tilt",
          "stand_height", "stand_fore", "stand_lateral", "stand_speed",
          "stand_angvel", "stand_slip",
        )
      }
      valid_env_steps = 0
      standing_env_steps = 0
      falls = 0
      standing_falls = 0

      # The environment may allocate/reset delay buffers during a rollout.
      # no_grad keeps them as ordinary mutable tensors; inference_mode would
      # make newly created tensors immutable outside its context.
      with torch.no_grad():
        for step in range(args.steps):
          command = base_env.command_manager.get_command("twist").clone()
          standing = torch.linalg.vector_norm(command, dim=1) < 1e-6
          moving = ~standing

          actual_lin = robot.data.root_link_lin_vel_b[:, :2].clone()
          actual_yaw = robot.data.root_link_ang_vel_b[:, 2].clone()
          foot_vel = torch.linalg.vector_norm(
            robot.data.site_lin_vel_w[:, foot_ids, :2], dim=-1
          )
          assert contact.data.found is not None
          in_contact = contact.data.found > 0
          assert contact.data.force is not None
          force_mag = torch.linalg.vector_norm(contact.data.force, dim=-1)
          first_contact = contact.compute_first_contact(dt=base_env.step_dt)

          # Snapshot all pose metrics at the same pre-action state as contacts.
          pre_q = robot.data.joint_pos[:, leg_ids].clone()
          pre_gravity = robot.data.projected_gravity_b.clone()
          pre_height = robot.data.root_link_pos_w[:, 2].clone()
          pre_foot_pos = robot.data.site_pos_w[:, foot_ids, :2].clone()
          pre_angvel = robot.data.root_link_ang_vel_b[:, :2].clone()
          pre_quat = robot.data.root_link_quat_w.clone()
          actions = policy(obs)
          next_obs, _, dones, _ = env.step(actions)
          terminated = base_env.reset_terminated.clone()

          if step >= args.warmup_steps:
            valid_env_steps += args.num_envs
            standing_env_steps += int(standing.sum().item())
            falls += int(terminated.sum().item())
            standing_falls += int((terminated & standing).sum().item())

            _append_mean(samples["vel_xy"], torch.linalg.vector_norm(
              command[moving, :2] - actual_lin[moving], dim=-1
            ))
            _append_mean(samples["vel_yaw"], torch.abs(command[moving, 2] - actual_yaw[moving]))

            landing_values = force_mag[first_contact & moving[:, None]]
            samples["landing_mean"].extend(landing_values.detach().cpu().tolist())
            if landing_values.numel():
              samples["landing_p95"].append(float(torch.quantile(landing_values.float(), 0.95).item()))

            moving_contacts = in_contact & moving[:, None]
            moving_slip = foot_vel[moving_contacts]
            samples["moving_slip_mean"].extend(moving_slip.detach().cpu().tolist())
            if moving_slip.numel():
              samples["moving_slip_p95"].append(float(torch.quantile(moving_slip.float(), 0.95).item()))

            valid_transition = ~dones.bool()
            if valid_transition.any():
              state = fixed_amp_normalizer.normalize_torch(prev_amp[valid_transition], "cuda:0")
              next_state = fixed_amp_normalizer.normalize_torch(next_obs["amp"][valid_transition], "cuda:0")
              disc = fixed_judge(torch.cat((state, next_state), dim=-1)).squeeze(-1)
              style = torch.clamp(1.0 - 0.25 * torch.square(disc - 1.0), min=0.0)
              _append_mean(samples["style"], style)

            if prev_action is not None and prev_action_delta is not None:
              action_delta = actions - prev_action
              _append_mean(samples["action_acc"], torch.abs(action_delta - prev_action_delta))

            q = pre_q
            limits = robot.data.soft_joint_pos_limits[:, leg_ids]
            violation = torch.relu(limits[..., 0] - q) + torch.relu(q - limits[..., 1])
            _append_mean(samples["joint_limit"], violation)

            if standing.any():
              projected_gravity = pre_gravity[standing]
              tilt = torch.rad2deg(torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)))
              _append_mean(samples["stand_tilt"], tilt)
              desired_height = robot.data.default_root_state[standing, 2]
              height = pre_height[standing]
              _append_mean(samples["stand_height"], torch.abs(height - desired_height))

              foot_pos = pre_foot_pos[standing]
              delta_w = foot_pos[:, 0] - foot_pos[:, 1]
              w, x, y, z = pre_quat[standing].unbind(-1)
              yaw = torch.atan2(2 * (w*z+x*y), 1-2*(y*y+z*z))
              c, s = yaw.cos(), yaw.sin()
              delta = torch.stack((c*delta_w[:,0]+s*delta_w[:,1],
                                   -s*delta_w[:,0]+c*delta_w[:,1]), dim=-1)
              _append_mean(samples["stand_fore"], torch.abs(delta[:, 0]))
              _append_mean(samples["stand_lateral"], torch.abs(torch.abs(delta[:, 1]) - 0.285))
              _append_mean(samples["stand_speed"], torch.linalg.vector_norm(actual_lin[standing], dim=-1))
              _append_mean(samples["stand_angvel"], torch.linalg.vector_norm(
                pre_angvel[standing], dim=-1
              ))
              standing_contacts = in_contact & standing[:, None]
              _append_mean(samples["stand_slip"], foot_vel[standing_contacts])

          if prev_action is None:
            prev_action_delta = None
          else:
            prev_action_delta = actions - prev_action
          prev_action = actions.clone()
          prev_amp = next_obs["amp"].clone()
          obs = next_obs

      row: dict[str, float | str | int] = {
        "run": checkpoint.parent.name,
        "checkpoint": checkpoint.name,
        "iteration": _checkpoint_step(checkpoint),
        "path": str(checkpoint.resolve()),
        "moving_vel_xy_error": _mean(samples["vel_xy"]),
        "moving_vel_yaw_error": _mean(samples["vel_yaw"]),
        "fall_rate_per_1000_env_steps": 1000.0 * falls / max(valid_env_steps, 1),
        "landing_force_mean_n": _mean(samples["landing_mean"]),
        "landing_force_p95_n": _percentile(samples["landing_mean"], 95),
        "moving_slip_mean_mps": _mean(samples["moving_slip_mean"]),
        "moving_slip_p95_mps": _percentile(samples["moving_slip_mean"], 95),
        "fixed_judge_amp_style": _mean(samples["style"]),
        "action_acceleration_mean": _mean(samples["action_acc"]),
        "joint_limit_violation_mean_rad": _mean(samples["joint_limit"]),
        "standing_tilt_mean_deg": _mean(samples["stand_tilt"]),
        "standing_height_error_mean_m": _mean(samples["stand_height"]),
        "standing_foot_fore_error_mean_m": _mean(samples["stand_fore"]),
        "standing_foot_lateral_error_mean_m": _mean(samples["stand_lateral"]),
        "standing_root_speed_mean_mps": _mean(samples["stand_speed"]),
        "standing_ang_vel_xy_mean_rps": _mean(samples["stand_angvel"]),
        "standing_slip_mean_mps": _mean(samples["stand_slip"]),
        "standing_fall_rate_per_1000_env_steps": 1000.0 * standing_falls / max(standing_env_steps, 1),
      }
      rows.append(row)
      (args.output_dir / "progress.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
      print(
        f"[EVAL] {index:02d}/{len(checkpoints)} {checkpoint.parent.name}/{checkpoint.name} "
        f"vel={row['moving_vel_xy_error']:.4f} impact={row['landing_force_mean_n']:.1f} "
        f"slip={row['moving_slip_mean_mps']:.4f} style={row['fixed_judge_amp_style']:.4f}",
        flush=True,
      )
  finally:
    env.close()

  _score_rows(rows)
  rows.sort(key=lambda row: float(row["overall_score"]), reverse=True)
  args.output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = args.output_dir / "casbot02_amp_checkpoint_scores.csv"
  json_path = args.output_dir / "casbot02_amp_checkpoint_scores.json"
  with csv_path.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  json_path.write_text(json.dumps({
    "task": args.task,
    "num_envs": args.num_envs,
    "steps": args.steps,
    "warmup_steps": args.warmup_steps,
    "seed": args.seed,
    "command_ranges": {
      "lin_vel_x": twist_cfg.ranges.lin_vel_x,
      "lin_vel_y": twist_cfg.ranges.lin_vel_y,
      "ang_vel_z": twist_cfg.ranges.ang_vel_z,
    },
    "waist_com_x_range": (env_cfg.events["waist_com_backward"].params["ranges"][0] if "waist_com_backward" in env_cfg.events else None),
    "metric_version": 2,
    "standing_only": args.standing_only,
    "no_pushes": args.no_pushes,
    "standing_definition": "command norm < 1e-6",
    "stance_frame": "root yaw frame",
    "tail_definition": "pooled contact samples",
    "joint_acc_weight": env_cfg.rewards["joint_acc_l2"].weight,
    "action_rate_weight": env_cfg.rewards["action_rate_l2"].weight,
    "foot_slip_weight": env_cfg.rewards["foot_slip"].weight,
    "soft_landing_weight": env_cfg.rewards["soft_landing"].weight,
    "judge_checkpoint": str(judge_checkpoint.resolve()),
    "checkpoints": len(checkpoints),
    "ranking": rows,
  }, ensure_ascii=False, indent=2))

  print("\n[EVAL] final ranking", flush=True)
  for rank, row in enumerate(rows, start=1):
    print(
      f"{rank:02d}. {row['run']}/{row['checkpoint']} overall={row['overall_score']:.2f} "
      f"speed={row['speed_tracking_score']:.1f} impact={row['landing_impact_score']:.1f} "
      f"slip={row['foot_slip_score']:.1f} leg={row['leg_form_score']:.1f} "
      f"posture={row['standing_posture_score']:.1f} stability={row['standing_stability_score']:.1f}",
      flush=True,
    )
  print(f"[EVAL] wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
  main()
