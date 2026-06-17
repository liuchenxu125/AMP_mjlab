#!/usr/bin/env python3
"""Replay AMP motion NPZ file in MuJoCo viewer.

Usage:
  python scripts/replay_motion.py <npz_file> --robot casbot_skeleton
  python scripts/replay_motion.py <npz_file> --robot marathon_001 --speed 0.5 --loop
  python scripts/replay_motion.py <npz_file> --robot g1 --base-height 0.05

Supports any robot registered in src/assets/robots/ (g1, casbot_skeleton, marathon_001).
"""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

# ── Robot registry ──────────────────────────────────────────────────────────
# Each entry: (XML path relative to SRC_PATH, joint index function)
# For robots using the framework's Entity system, we provide XML paths directly.

from src import SRC_PATH

ROBOT_XML = {
    "g1": SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "scene_g1.xml",
    "g1_23dof": SRC_PATH / "assets" / "robots" / "unitree_g1" / "xmls" / "g1_23dof.xml",
    "casbot_skeleton": SRC_PATH / "assets" / "robots" / "casbot_skeleton" / "xmls" / "casbot_skeleton_25dof.xml",
    "marathon_001": SRC_PATH / "assets" / "robots" / "marathon_001" / "xmls" / "marathon_001.xml",
}

# Joint names for each robot, in the order they appear in NPZ joint_pos columns.
# Must match the CSV→NPZ conversion order in csv_to_npz.py.
ROBOT_JOINT_NAMES = {
    "g1": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ],
    "g1_23dof": [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint",
    ],
    "casbot_skeleton": [
        "left_leg_pelvic_pitch_joint", "left_leg_pelvic_roll_joint", "left_leg_pelvic_yaw_joint",
        "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint", "left_leg_ankle_roll_joint",
        "right_leg_pelvic_pitch_joint", "right_leg_pelvic_roll_joint", "right_leg_pelvic_yaw_joint",
        "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint", "right_leg_ankle_roll_joint",
        "waist_yaw_joint", "head_yaw_joint", "head_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_pitch_joint", "right_wrist_yaw_joint",
    ],
    "marathon_001": [
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_elbow_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_elbow_pitch_joint",
        "right_leg_pelvic_pitch_joint", "right_leg_pelvic_roll_joint", "right_leg_pelvic_yaw_joint",
        "right_leg_knee_pitch_joint", "right_leg_ankle_pitch_joint", "right_leg_ankle_roll_joint",
        "left_leg_pelvic_pitch_joint", "left_leg_pelvic_roll_joint", "left_leg_pelvic_yaw_joint",
        "left_leg_knee_pitch_joint", "left_leg_ankle_pitch_joint", "left_leg_ankle_roll_joint",
    ],
}


def _build_joint_index_map(model: mujoco.MjModel, joint_names: list[str]) -> np.ndarray:
    """Build an index array mapping NPZ column i → model.qpos index.

    Returns int32 array of shape (num_npz_dof,) where each entry is the
    qpos address of the corresponding joint in the MuJoCo model.
    """
    name_to_qpos = {}
    for j in range(model.njnt):
        jnt_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jnt_name is None:
            continue
        addr = model.jnt_qposadr[j]
        if addr >= 0 and jnt_name not in name_to_qpos:
            name_to_qpos[jnt_name] = addr

    mapping = np.empty(len(joint_names), dtype=np.int32)
    for i, name in enumerate(joint_names):
        if name not in name_to_qpos:
            raise KeyError(
                f"Joint '{name}' (NPZ column {i}) not found in model. "
                f"Available joints: {sorted(name_to_qpos.keys())}"
            )
        mapping[i] = name_to_qpos[name]

    return mapping


def main():
    parser = argparse.ArgumentParser(description="Replay AMP motion NPZ in MuJoCo viewer")
    parser.add_argument("npz_file", help="Path to NPZ motion file")
    parser.add_argument(
        "--robot", default="g1",
        choices=list(ROBOT_XML.keys()),
        help="Robot type (default: g1)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier")
    parser.add_argument("--loop", action="store_true", help="Loop playback")
    parser.add_argument("--xml", help="Custom robot XML path (overrides --robot)")
    parser.add_argument("--base-height", type=float, default=0.0,
                        help="Vertical offset added to root position (meters)")
    args = parser.parse_args()

    # ── Load motion ──────────────────────────────────────────────────────
    data_npz = np.load(args.npz_file)
    joint_pos = data_npz["joint_pos"]          # (N, dof)
    body_pos_w = data_npz["body_pos_w"]        # (N, nbodies, 3)
    body_quat_w = data_npz["body_quat_w"]      # (N, nbodies, 4)
    fps = float(data_npz["fps"].item())
    num_frames = joint_pos.shape[0]
    num_dof = joint_pos.shape[1]
    print(f"Motion: {num_frames} frames, {num_dof} DOF, {fps:.1f} fps")

    # ── Load model ───────────────────────────────────────────────────────
    xml_path = args.xml or str(ROBOT_XML[args.robot])
    if not Path(xml_path).exists():
        raise FileNotFoundError(f"XML not found: {xml_path}")
    print(f"Model:  {xml_path}")

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    # Build NPZ→model joint index mapping
    joint_names = ROBOT_JOINT_NAMES[args.robot]
    if num_dof != len(joint_names):
        print(f"WARNING: NPZ has {num_dof} DOF but config lists {len(joint_names)} "
              f"joints for '{args.robot}'. Motion may look wrong.")
    joint_qpos_map = _build_joint_index_map(model, joint_names)

    nq_joints = model.nq - 7
    print(f"Model:  {model.nq} qpos ({nq_joints} joints), {model.nv} qvel")

    # ── Viewer loop ──────────────────────────────────────────────────────
    frame_dt = 1.0 / fps / args.speed
    print(f"Playback speed: {args.speed:.1f}x  →  {1.0/frame_dt:.1f} fps")
    print("Viewer opened — close window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        frame_idx = 0
        last_update = time.time()

        while viewer.is_running():
            now = time.time()
            if now - last_update >= frame_dt:
                # Root pose (base_link / pelvis)
                data.qpos[0] = body_pos_w[frame_idx, 0, 0]
                data.qpos[1] = body_pos_w[frame_idx, 0, 1]
                data.qpos[2] = body_pos_w[frame_idx, 0, 2] + args.base_height
                data.qpos[3:7] = body_quat_w[frame_idx, 0, :]  # wxyz

                # Joint positions via name→qpos index map
                data.qpos[joint_qpos_map] = joint_pos[frame_idx, :]

                mujoco.mj_forward(model, data)
                viewer.sync()

                frame_idx += 1
                if frame_idx >= num_frames:
                    if args.loop:
                        frame_idx = 0
                        print("↻ Looping...")
                    else:
                        print("Done.")
                        break
                last_update = now
            else:
                time.sleep(0.002)


if __name__ == "__main__":
    main()
