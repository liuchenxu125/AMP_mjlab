"""
CasbotAMP — AMP locomotion policy for the Casbot Skeleton (25-DOF) robot.

Standalone policy class (no FSM dependency). Ported from the MJAMP (G1)
deployment, adapted for the 25-joint casbot_skeleton robot trained with
the same MJLAB AMP framework and parameters.

Observation layout (84 dims per frame, 4-frame history = 336 total):
    [ang_vel(3), proj_gravity(3), cmd_vel(3), dof_pos(25), dof_vel(25), last_action(25)]

Joint order (matches XML and MJLAB training order):
    Left leg (0-5):   pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
    Right leg (6-11): pelvic_pitch, pelvic_roll, pelvic_yaw, knee_pitch, ankle_pitch, ankle_roll
    Waist (12):       waist_yaw
    Head (13-14):     head_yaw, head_pitch
    Left arm (15-19): shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw
    Right arm (20-24):shoulder_pitch, shoulder_roll, shoulder_yaw, elbow_pitch, wrist_yaw
"""

import os
import numpy as np
import yaml
import onnxruntime as ort


class CasbotAMP:
    """AMP locomotion policy for the Casbot Skeleton robot."""

    def __init__(self, config_path=None):
        """
        Load configuration and ONNX model.

        Args:
            config_path: Path to YAML config. If None, uses default relative path.
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))

        if config_path is None:
            config_path = os.path.join(current_dir, "config", "CasbotAMP.yaml")

        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

        # ── Model path ──
        onnx_rel = config["onnx_path"]
        if os.path.isabs(onnx_rel):
            self.onnx_path = onnx_rel
        else:
            self.onnx_path = os.path.normpath(os.path.join(current_dir, onnx_rel))

        # ── Observation config ──
        self.num_actions = config["num_actions"]
        self.num_obs = config["num_obs"]
        self.history_length = config["history_length"]
        self.robot_state_dim = config["robot_state_dim"]
        self.clip_observations = config["clip_observations"]
        self.clip_actions = config["clip_actions"]

        # ── Scaling ──
        self.action_scale = config["action_scale"]
        self.ang_vel_scale = np.array(config["ang_vel_scale"], dtype=np.float32)
        self.dof_pos_scale = config["dof_pos_scale"]
        self.dof_vel_scale = config["dof_vel_scale"]

        # ── Velocity limits ──
        self.vx_lim = np.array(
            [config["vx_limit_min"], config["vx_limit_max"]], dtype=np.float32
        )
        self.vx_lim_slow = np.array(
            [config["vx_limit_min_slow"], config["vx_limit_max_slow"]],
            dtype=np.float32,
        )
        self.vy_lim = np.array(
            [config["vy_limit_min"], config["vy_limit_max"]], dtype=np.float32
        )
        self.wyaw_lim = np.array(
            [config["wyaw_limit_min"], config["wyaw_limit_max"]], dtype=np.float32
        )
        self.dead_zone = config["dead_zone"]
        self.cmd_smoothes = config["cmd_smoothes"]

        # ── Safety ──
        self.safe_projgravity_threshold = config["safe_projgravity_threshold"]

        # ── Motor parameters (25 elements) ──
        self.kps = np.array(config["kps"], dtype=np.float32)
        self.kds = np.array(config["kds"], dtype=np.float32)
        self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
        self.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

        # ── dof_mapping (identity) ──
        self.dof_mapping = np.array(config["dof_mapping"], dtype=np.int32)

        # ── Waist & gravity ──
        self.waist_yrp_idx = np.array(config["waist_yrp_idx"], dtype=np.int32)
        self.gravity_vec = np.array(config["gravity_vec"], dtype=np.float32)

        # ── Compute dof_action_scale per motor ──
        # dof_action_scale[i] = action_scale * tau_limit[i] / kps[i]
        self.dof_action_scale = self.action_scale * self.tau_limit / self.kps

        # ── Runtime state ──
        self._high_speed_mode = False
        self._vCmdBodyPast = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self._target_pos = np.zeros(self.num_actions, dtype=np.float32)

        # ── Load ONNX model ──
        self._load_policy()

        print(f"[CasbotAMP] Policy initialized.")
        print(f"  Model: {self.onnx_path}")
        print(f"  Observation: {self.num_obs} dims, Actions: {self.num_actions}")
        print(
            f"  Velocity limits (high): vx=[{self.vx_lim[0]}, {self.vx_lim[1]}], "
            f"vy=[{self.vy_lim[0]}, {self.vy_lim[1]}], "
            f"wyaw=[{self.wyaw_lim[0]}, {self.wyaw_lim[1]}]"
        )
        print(
            f"  Velocity limits (slow): vx=[{self.vx_lim_slow[0]}, {self.vx_lim_slow[1]}]"
        )

    def _load_policy(self):
        """Load ONNX policy model."""
        self.ort_session = ort.InferenceSession(self.onnx_path)
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        print(f"  ONNX input: {self.input_name}, output: {self.output_name}")

    # ────────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────────

    def reset(self):
        """Reset runtime buffers (call once before the first step)."""
        self._high_speed_mode = False
        self._vCmdBodyPast = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self._target_pos = np.zeros(self.num_actions, dtype=np.float32)

    def init_buffers(self, base_quat, ang_vel, q, dq):
        """
        Fill the observation history buffer by running the observation
        pipeline `history_length` times with the current state.

        Args:
            base_quat: [w, x, y, z] base orientation quaternion.
            ang_vel:   [3] angular velocity.
            q:         [25] joint positions.
            dq:        [25] joint velocities.
        """
        self._vCmdBodyPast = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)

        # Command zero velocity during init
        cmd_vel = np.zeros(3, dtype=np.float32)

        for _ in range(self.history_length):
            self._observations_compute(base_quat, ang_vel, cmd_vel, q, dq)

        print(f"[CasbotAMP] Buffers initialized ({self.history_length} frames)")

    def step(self, base_quat, ang_vel, cmd_vel, q, dq):
        """
        Run one policy step: compute observation → run inference → return actions.

        Args:
            base_quat: [w, x, y, z] base orientation quaternion.
            ang_vel:   [3] angular velocity.
            cmd_vel:   [3] commanded velocity [vx, vy, wyaw].
            q:         [25] joint positions.
            dq:        [25] joint velocities.

        Returns:
            dict with keys:
                actions:    [25] target joint positions.
                kps:        [25] stiffness gains.
                kds:        [25] damping gains.
                terminated: bool — True if anchor gravity safety triggered.
        """
        observation = self._observations_compute(base_quat, ang_vel, cmd_vel, q, dq)
        result = self._action_compute(observation)

        # Check anchor termination
        projected_gravity = self._compute_projected_gravity(base_quat)
        anchor_error = abs(projected_gravity[2] - (-1.0))
        terminated = anchor_error > self.safe_projgravity_threshold

        if terminated:
            print(
                f"[CasbotAMP WARNING] Large anchor proj_gravity error: "
                f"{anchor_error:.4f} (threshold: {self.safe_projgravity_threshold})"
            )

        return {
            "actions": result["actions"],
            "kps": result["kps"],
            "kds": result["kds"],
            "terminated": terminated,
        }

    def get_user_cmd(self, ly, lx, rx):
        """
        Process joystick-style velocity commands.

        Args:
            ly:  left stick Y  (forward/back,  -1..1).
            lx:  left stick X  (lateral,       -1..1).
            rx:  right stick X (yaw,           -1..1).

        Returns:
            numpy array [vx, vy, wyaw].
        """
        vx_lim = self.vx_lim_slow if not self._high_speed_mode else self.vx_lim

        # Forward velocity (vx) from ly
        if ly < -self.dead_zone:
            vx = ly * (-vx_lim[0])
        elif ly > self.dead_zone:
            vx = ly * vx_lim[1]
        else:
            vx = 0.0

        # Lateral velocity (vy) from lx
        if lx < -self.dead_zone:
            vy = lx * (-self.vy_lim[0])
        elif lx > self.dead_zone:
            vy = lx * self.vy_lim[1]
        else:
            vy = 0.0

        # Yaw rate (wyaw) from rx
        if rx < -self.dead_zone:
            wyaw = rx * (-self.wyaw_lim[0])
        elif rx > self.dead_zone:
            wyaw = rx * self.wyaw_lim[1]
        else:
            wyaw = 0.0

        new_cmd = np.array([vx, vy, wyaw], dtype=np.float32)
        vCmdBody = (
            self._vCmdBodyPast * self.cmd_smoothes
            + new_cmd * (1.0 - self.cmd_smoothes)
        )
        self._vCmdBodyPast = vCmdBody.copy()
        return vCmdBody

    @property
    def high_speed_mode(self):
        return self._high_speed_mode

    @high_speed_mode.setter
    def high_speed_mode(self, value: bool):
        self._high_speed_mode = value

    # ────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_projected_gravity(base_quat):
        """Compute gravity direction in the robot's local frame."""
        qw, qx, qy, qz = base_quat
        g = np.zeros(3, dtype=np.float32)
        g[0] = 2.0 * (-qz * qx + qw * qy)
        g[1] = -2.0 * (qz * qy + qw * qx)
        g[2] = 1.0 - 2.0 * (qw * qw + qz * qz)
        return g

    def _observations_compute(self, base_quat, ang_vel, cmd_vel, q, dq):
        """
        Build one frame of robot state and slide it into the obs buffer.

        Returns the full (clipped) observation buffer.
        """
        # 1. Projected gravity
        projected_gravity = self._compute_projected_gravity(base_quat)

        # 2. Angular velocity (ensure 3-element)
        ang_vel = np.asarray(ang_vel, dtype=np.float32).flatten()[:3]

        # 3. Command velocity
        vCmdBody = np.asarray(cmd_vel, dtype=np.float32).flatten()[:3]

        # 4. Joint positions — offset from default
        dof_pos_motor = np.asarray(q, dtype=np.float32).flatten()
        dof_pos_policy = dof_pos_motor[self.dof_mapping]
        default_policy = self.default_dof_pos[self.dof_mapping]
        dof_pos_scaled = (dof_pos_policy - default_policy) * self.dof_pos_scale

        # 5. Joint velocities
        dof_vel_motor = np.asarray(dq, dtype=np.float32).flatten()
        dof_vel_policy = dof_vel_motor[self.dof_mapping]
        dof_vel_scaled = dof_vel_policy * self.dof_vel_scale

        # 6. Scale angular velocity
        ang_vel_scaled = ang_vel * self.ang_vel_scale

        # 7. Build current robot state (84 dims)
        current_robot_state = np.concatenate(
            [
                ang_vel_scaled,       # 3
                projected_gravity,    # 3
                vCmdBody,             # 3
                dof_pos_scaled,       # 25
                dof_vel_scaled,       # 25
                self._last_action,    # 25
            ],
            axis=0,
            dtype=np.float32,
        )  # Total: 84

        # 8. Slide window: shift left, append new frame at end
        step = self.robot_state_dim
        self.obs_buffer[0 : self.num_obs - step] = self.obs_buffer[step : self.num_obs]
        self.obs_buffer[self.num_obs - step : self.num_obs] = current_robot_state

        # 9. Clip and return
        return np.clip(self.obs_buffer, -self.clip_observations, self.clip_observations)

    def _action_compute(self, observation):
        """Run ONNX inference and convert to motor targets."""
        try:
            obs_tensor = observation.reshape(1, -1).astype(np.float32)
            outputs = self.ort_session.run(
                [self.output_name], {self.input_name: obs_tensor}
            )
            action_policy = outputs[0].squeeze()

            # Clip
            action_policy = np.clip(
                action_policy, -self.clip_actions, self.clip_actions
            )

            # Scale to motor target positions
            target_pos_motor = np.zeros(self.num_actions, dtype=np.float32)
            for policy_idx in range(self.num_actions):
                motor_idx = self.dof_mapping[policy_idx]
                target_pos_motor[motor_idx] = (
                    action_policy[policy_idx] * self.dof_action_scale[motor_idx]
                    + self.default_dof_pos[motor_idx]
                )

            self._last_action = action_policy.copy()
            self._target_pos = target_pos_motor.copy()

            return {
                "actions": target_pos_motor.copy(),
                "kps": self.kps.copy(),
                "kds": self.kds.copy(),
            }

        except Exception as e:
            print(f"[CasbotAMP ERROR] ONNX inference failed: {e}")
            # Hold current position on error
            return {
                "actions": self._target_pos.copy(),
                "kps": self.kps.copy(),
                "kds": self.kds.copy(),
            }
