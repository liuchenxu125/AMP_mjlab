"""Batch evaluate CASBOT02 leg AMP checkpoints via headless sim2sim.

Uses the same MuJoCo model, sensor names, and observation format as
sim2sim_casbot02_leg_amp_onnx.py — the reference sim2sim script.

Usage:
  python scripts/eval_checkpoints_batch.py logs/rsl_rl/casbot02_leg_amp_locomotion/2026-08-10_09-42-44
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import deque
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
sys.path.insert(0, str(REPO_ROOT))

from src.assets.robots import (
    CASBOT02_22DOF_NO_WAIST_ACTION_SCALE,
    CASBOT02_22DOF_NO_WAIST_JOINT_NAMES,
    CASBOT02_23DOF_JOINT_NAMES,
    CASBOT02_LEG_ONLY_JOINT_NAMES,
    get_casbot02_23dof_robot_cfg,
)
from src.assets.robots.casbot02.casbot02_constants import HOME_KEYFRAME
from mjlab.entity import Entity

# --- constants (mirror sim2sim_casbot02_leg_amp_onnx.py) ---
SIM_TIMESTEP = 0.005
DECIMATION = 4
HISTORY_LENGTH = 4
NUM_OBS_JOINTS = len(CASBOT02_LEG_ONLY_JOINT_NAMES)      # 12
NUM_POLICY_ACTIONS = len(CASBOT02_22DOF_NO_WAIST_JOINT_NAMES)  # 22
NUM_FULL_JOINTS = len(CASBOT02_23DOF_JOINT_NAMES)          # 23
OBS_JOINT_INDICES = np.array(
    [CASBOT02_23DOF_JOINT_NAMES.index(n) for n in CASBOT02_LEG_ONLY_JOINT_NAMES],
    dtype=np.int64,
)
ACTION_JOINT_INDICES = np.array(
    [CASBOT02_23DOF_JOINT_NAMES.index(n) for n in CASBOT02_22DOF_NO_WAIST_JOINT_NAMES],
    dtype=np.int64,
)
CURRENT_SINGLE_FRAME_OBS_SIZE = 45  # 3+3+3+12+12+12

# --- command grid ---
COMMAND_GRID = [
    (0.4, 0.0, 0.0, "fwd_slow"),
    (0.8, 0.0, 0.0, "fwd_med"),
    (1.0, 0.0, 0.0, "fwd_fast"),
    (-0.3, 0.0, 0.0, "bwd"),
    (0.0, 0.0, 0.5, "turn"),
    (0.5, 0.0, 0.3, "fwd_turn"),
]
EVAL_DURATION = 20.0   # seconds per command
WARMUP_TIME = 3.0      # seconds warmup before collecting metrics

# --- foot body names for slip velocity ---
LEFT_FOOT_BODY = "leg_l6_link"   # or similar — check actual body names
RIGHT_FOOT_BODY = "leg_r6_link"


# ---------------------------------------------------------------------------
# helpers (copied / adapted from sim2sim)
# ---------------------------------------------------------------------------

def quat_apply_inverse_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate a world-frame vector into the body frame."""
    w, x, y, z = quat
    q_vec = -np.array([x, y, z], dtype=np.float64)
    t = 2.0 * np.cross(q_vec, vec)
    rotated = vec + w * t + np.cross(q_vec, t)
    return rotated


def build_model() -> mujoco.MjModel:
    entity = Entity(get_casbot02_23dof_robot_cfg())
    spec = entity.spec
    ground = spec.worldbody.add_geom(
        name="ground",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
    )
    ground.size = [0.0, 0.0, 1.0]
    ground.condim = 4
    ground.friction = [0.9, 0.2, 0.2]
    ground.material = "matplane"
    model = spec.compile()
    model.opt.timestep = SIM_TIMESTEP
    model.opt.iterations = 10
    model.opt.ls_iterations = 20
    return model


def make_default_joint_pos() -> np.ndarray:
    joint_pos = HOME_KEYFRAME.joint_pos
    if joint_pos is None:
        return np.zeros(NUM_FULL_JOINTS, dtype=np.float64)
    fallback = float(joint_pos.get(".*", 0.0))
    return np.array(
        [float(joint_pos.get(name, fallback)) for name in CASBOT02_23DOF_JOINT_NAMES],
        dtype=np.float64,
    )


def make_obs_joint_pos() -> np.ndarray:
    return make_default_joint_pos()[OBS_JOINT_INDICES]


def make_action_scale() -> np.ndarray:
    return np.array(
        [CASBOT02_22DOF_NO_WAIST_ACTION_SCALE[name] for name in CASBOT02_22DOF_NO_WAIST_JOINT_NAMES],
        dtype=np.float64,
    )


def ctrl_range(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    lo = model.actuator_ctrlrange[:, 0].astype(np.float64).copy()
    hi = model.actuator_ctrlrange[:, 1].astype(np.float64).copy()
    return lo, hi


def reset_state(model: mujoco.MjModel, data: mujoco.MjData, default_joint_pos: np.ndarray):
    mujoco.mj_resetData(model, data)
    data.qpos[0:3] = [0.0, 0.0, 0.92]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:] = default_joint_pos
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


# ---------------------------------------------------------------------------
# observation (EXACT copy of sim2sim's get_obs_frame)
# ---------------------------------------------------------------------------

def get_obs_frame(
    data: mujoco.MjData,
    command: np.ndarray,
    last_action: np.ndarray,
    default_joint_pos: np.ndarray,
) -> np.ndarray:
    base_ang_vel = np.asarray(
        data.sensor("angular-velocity").data, dtype=np.float64
    ).copy()
    quat_wxyz = np.asarray(data.qpos[3:7], dtype=np.float64)
    projected_gravity = quat_apply_inverse_wxyz(
        quat_wxyz,
        np.array([0.0, 0.0, -1.0], dtype=np.float64),
    )
    joint_pos_rel = (
        np.asarray(data.qpos[7:], dtype=np.float64)[OBS_JOINT_INDICES]
        - default_joint_pos
    )
    joint_vel_rel = np.asarray(data.qvel[6:], dtype=np.float64)[OBS_JOINT_INDICES]
    obs_parts = [
        base_ang_vel,
        projected_gravity,
        command,
        joint_pos_rel,
        joint_vel_rel,
        last_action,
    ]
    obs = np.concatenate(obs_parts, axis=0)
    return obs.astype(np.float32)


# ---------------------------------------------------------------------------
# ONNX policy
# ---------------------------------------------------------------------------

class OnnxPolicy:
    def __init__(self, path: Path):
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        self.session = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        obs = np.ascontiguousarray(obs.astype(np.float32))
        action = self.session.run([self.output_name], {self.input_name: obs})[0]
        return np.asarray(action, dtype=np.float32).reshape(-1)


# ---------------------------------------------------------------------------
# single-model evaluation
# ---------------------------------------------------------------------------

def evaluate_model(onnx_path: Path, verbose: bool = False) -> dict:
    policy = OnnxPolicy(onnx_path)
    model = build_model()
    data = mujoco.MjData(model)
    default_joint_pos = make_default_joint_pos()
    default_obs_joint_pos = make_obs_joint_pos()
    action_scale = make_action_scale()
    ctrl_lo, ctrl_hi = ctrl_range(model)

    # Find foot body indices for slip velocity
    try:
        left_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, LEFT_FOOT_BODY)
    except Exception:
        left_foot_id = -1
    try:
        right_foot_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_FOOT_BODY)
    except Exception:
        right_foot_id = -1

    results_by_cmd = []
    total_falls = 0
    total_cmds = len(COMMAND_GRID)

    for vx, vy, vyaw, label in COMMAND_GRID:
        command = np.array([vx, vy, vyaw], dtype=np.float64)
        reset_state(model, data, default_joint_pos)

        history: deque[np.ndarray] = deque(maxlen=HISTORY_LENGTH)
        last_action = np.zeros(NUM_OBS_JOINTS, dtype=np.float32)
        target_pos = default_joint_pos.copy()

        total_steps = int(EVAL_DURATION / SIM_TIMESTEP)
        warmup_steps = int(WARMUP_TIME / SIM_TIMESTEP)

        metrics = {
            "errXY": [], "errYaw": [],
            "landing": [], "slip": [],
            "foot_fore_err": [], "foot_lat_err": [],
        }
        fell = False
        step_count = 0

        for step in range(total_steps):
            if step % DECIMATION == 0:
                obs_frame = get_obs_frame(data, command, last_action, default_obs_joint_pos)
                if not history:
                    for _ in range(HISTORY_LENGTH):
                        history.append(obs_frame)
                else:
                    history.append(obs_frame)

                obs = np.concatenate(list(history), axis=0).reshape(1, -1)
                action = policy(obs)
                last_action = action[:NUM_OBS_JOINTS].astype(np.float32)
                target_pos = default_joint_pos.copy()
                target_pos[ACTION_JOINT_INDICES] = (
                    default_joint_pos[ACTION_JOINT_INDICES]
                    + action.astype(np.float64) * action_scale
                )
                target_pos = np.clip(target_pos, ctrl_lo, ctrl_hi)

            data.ctrl[:] = target_pos
            mujoco.mj_step(model, data)
            step_count += 1

            # Check for fall
            base_z = data.qpos[2]
            if base_z < 0.5:
                fell = True
                break

            if step >= warmup_steps:
                # --- velocity tracking ---
                lin_vel = data.sensor("linear-velocity").data[:2].copy()
                ang_vel_z = data.sensor("angular-velocity").data[2].copy()
                metrics["errXY"].append(float(np.linalg.norm(lin_vel - command[:2])))
                metrics["errYaw"].append(float(abs(ang_vel_z - command[2])))

                # --- landing force (left + right foot force magnitude) ---
                lf = data.sensor("left_force").data.copy()
                rf = data.sensor("right_force").data.copy()
                lf_norm = float(np.linalg.norm(lf))
                rf_norm = float(np.linalg.norm(rf))
                landing = lf_norm + rf_norm
                metrics["landing"].append(landing)

                # --- slip velocity (foot body velocity when in contact) ---
                slip_sum = 0.0
                slip_count = 0
                if left_foot_id >= 0 and lf_norm > 5.0:  # foot in contact
                    foot_vel = data.cvel[left_foot_id][3:6].copy()  # linear vel
                    slip_sum += float(np.linalg.norm(foot_vel))
                    slip_count += 1
                if right_foot_id >= 0 and rf_norm > 5.0:  # foot in contact
                    foot_vel = data.cvel[right_foot_id][3:6].copy()
                    slip_sum += float(np.linalg.norm(foot_vel))
                    slip_count += 1
                if slip_count > 0:
                    metrics["slip"].append(slip_sum / slip_count)

        if fell:
            total_falls += 1

        # Aggregate per-command
        cmd_result = {
            "label": label,
            "fell": fell,
            "eval_time": step_count * SIM_TIMESTEP,
        }
        for key in ["errXY", "errYaw", "landing", "slip"]:
            vals = metrics[key]
            cmd_result[key] = float(np.mean(vals)) if vals else (999.0 if "err" in key else 0.0)

        results_by_cmd.append(cmd_result)

    # Aggregate across commands (non-fallen only)
    good = [r for r in results_by_cmd if not r["fell"]]
    if good:
        agg = {
            "errXY": float(np.mean([r["errXY"] for r in good])),
            "errYaw": float(np.mean([r["errYaw"] for r in good])),
            "landing": float(np.mean([r["landing"] for r in good])),
            "slip": float(np.mean([r["slip"] for r in good])),
            "falls": total_falls,
            "n_cmds": total_cmds,
            "per_cmd": results_by_cmd,
        }
    else:
        agg = {
            "errXY": 999.0, "errYaw": 999.0,
            "landing": 9999.0, "slip": 999.0,
            "falls": total_falls, "n_cmds": total_cmds,
            "per_cmd": results_by_cmd,
        }

    return agg


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

def score_model(m: dict) -> float:
    WEIGHTS = {
        "errXY": 0.25,
        "errYaw": 0.15,
        "landing": 0.15,
        "slip": 0.10,
        "stability": 0.35,
    }
    TARGETS = {
        "errXY": 0.5,      # m/s tracking error
        "errYaw": 0.5,     # rad/s yaw error
        "landing": 300.0,  # N total foot force
        "slip": 0.05,      # m/s foot slip
    }
    total = 0.0
    for k in ["errXY", "errYaw", "landing", "slip"]:
        total += WEIGHTS[k] * (m[k] / TARGETS[k])
    # Stability: penalty for falls
    total += WEIGHTS["stability"] * (1.0 + 0.5 * m["falls"])
    return round(total * 100, 1)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Batch evaluate CASBOT02 leg AMP checkpoints")
    parser.add_argument("log_dir", type=str)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--max-models", type=int, default=0,
                        help="Evaluate at most N models (0 = all)")
    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    if not log_dir.is_absolute():
        log_dir = REPO_ROOT / log_dir

    export_dir = log_dir / "export"
    if not export_dir.is_dir():
        print(f"[ERROR] export directory not found: {export_dir}")
        return

    onnx_models = sorted(
        export_dir.glob("Casbot02-Leg-AMP-Flat_model_*.onnx"),
        key=lambda p: int(re.search(r"model_(\d+)", p.name).group(1)),
    )
    if not onnx_models:
        print(f"[ERROR] No ONNX models found in {export_dir}")
        return

    if args.max_models > 0:
        onnx_models = onnx_models[: args.max_models]

    print(f"[eval] {len(onnx_models)} models × {len(COMMAND_GRID)} cmds × {EVAL_DURATION}s")
    print(f"[eval] log_dir: {log_dir.name}\n")

    results = {}
    for onnx_path in onnx_models:
        name = onnx_path.stem.replace("Casbot02-Leg-AMP-Flat_", "")
        print(f"  [{name}] ... ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            m = evaluate_model(onnx_path, verbose=args.verbose)
            s = score_model(m)
            elapsed = time.perf_counter() - t0
            results[name] = {"score": s, **m}
            print(f"score={s:.1f}  errXY={m['errXY']:.3f}  errYaw={m['errYaw']:.3f}  "
                  f"land={m['landing']:.0f}N  slip={m['slip']:.3f}  "
                  f"falls={m['falls']}/{m['n_cmds']}  ({elapsed:.1f}s)")
        except Exception as e:
            elapsed = time.perf_counter() - t0
            print(f"FAILED ({elapsed:.1f}s): {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("[ERROR] All evaluations failed.")
        return

    # --- ranking ---
    ranked = sorted(results.items(), key=lambda x: x[1]["score"])
    print()
    print("=" * 90)
    print("RANKING  (lower score = better)")
    print("-" * 90)
    print(f"{'Rank':<5} {'Model':<15} {'Score':<8} {'errXY':<8} {'errYaw':<8} "
          f"{'Landing':<10} {'Slip':<8} {'Falls':<8}")
    print("-" * 90)
    for rank, (name, r) in enumerate(ranked, 1):
        marker = "  ← BEST" if rank == 1 else ""
        print(f"{rank:<5} {name:<15} {r['score']:<8.1f} {r['errXY']:<8.3f} "
              f"{r['errYaw']:<8.3f} {r['landing']:<10.0f} {r['slip']:<8.3f} "
              f"{r['falls']}/{r['n_cmds']}{marker}")
    print("=" * 90)

    # Per-command details for top 3
    print("\nPER-COMMAND BREAKDOWN (top 3):")
    for rank, (name, r) in enumerate(ranked[:3], 1):
        print(f"\n  #{rank} {name} (score={r['score']:.1f}):")
        for cmd in r.get("per_cmd", []):
            status = "FELL" if cmd["fell"] else "OK"
            print(f"    {cmd['label']:<12}  "
                  f"errXY={cmd['errXY']:.3f}  errYaw={cmd['errYaw']:.3f}  "
                  f"land={cmd['landing']:.0f}N  slip={cmd['slip']:.3f}  [{status}]")

    # Save
    out_path = log_dir / "eval_results.json"
    with open(out_path, "w") as f:
        json.dump({"ranked": [(n, r) for n, r in ranked]}, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
