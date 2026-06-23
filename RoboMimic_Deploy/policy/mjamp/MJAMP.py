from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
from common.rotation_helper import get_gravity_orientation_real
from common.remote_controller import KeyMap
import numpy as np
import yaml
import onnx
import onnxruntime
import os


class MJAMP(FSMState):
    """
    MJAMP (Motion Joint AMP) locomotion policy for G1 robot.

    Ported from C++ State_MJAMP (State_MJAmp.cpp / State_MJamp.h).
    Uses an ONNX policy network that takes a 384-dim observation
    (4-frame history × 96-dim robot state) and outputs 29 joint target
    positions. Supports joystick velocity commands with dead-zone
    filtering, exponential smoothing, high/low speed modes, and anchor
    gravity safety termination.

    dof_mapping = identity (dof_mapping_mj in C++): policy action index
    equals motor index — no reordering needed. dof_action_scale is
    computed per motor index as action_scale * tau_limit / Kp, matching
    the C++ formula with ArmatureConstants.

    Observation layout (96 dims per frame):
        [ang_vel(3), proj_gravity(3), cmd_vel(3), dof_pos(29), dof_vel(29), last_action(29)]
    """

    def __init__(self, state_cmd: StateAndCmd, policy_output: PolicyOutput, remote_controller=None):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.remote_controller = remote_controller
        self.name = FSMStateName.MJAMP
        self.name_str = "mjamp"

        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "MJAMP.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            # Model path
            self.onnx_path = os.path.join(current_dir, "model", config["onnx_path"])

            # Observation config
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.history_length = config["history_length"]
            self.robot_state_dim = config["robot_state_dim"]
            self.clip_observations = config["clip_observations"]
            self.clip_actions = config["clip_actions"]

            # Scaling
            self.action_scale = config["action_scale"]
            self.ang_vel_scale = np.array(config["ang_vel_scale"], dtype=np.float32)
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]

            # Velocity limits
            self.vx_lim = np.array([config["vx_limit_min"], config["vx_limit_max"]], dtype=np.float32)
            self.vx_lim_slow = np.array([config["vx_limit_min_slow"], config["vx_limit_max_slow"]], dtype=np.float32)
            self.vy_lim = np.array([config["vy_limit_min"], config["vy_limit_max"]], dtype=np.float32)
            self.wyaw_lim = np.array([config["wyaw_limit_min"], config["wyaw_limit_max"]], dtype=np.float32)
            self.dead_zone = config["dead_zone"]
            self.cmd_smoothes = config["cmd_smoothes"]

            # Safety
            self.safe_projgravity_threshold = config["safe_projgravity_threshold"]

            # Motor parameters (motor order, 29 elements)
            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.tau_limit = np.array(config["tau_limit"], dtype=np.float32)
            self.default_dof_pos = np.array(config["default_dof_pos"], dtype=np.float32)

            # dof_mapping: policy action index → motor index
            self.dof_mapping = np.array(config["dof_mapping"], dtype=np.int32)

            # Waist joint indices
            self.waist_yrp_idx = np.array(config["waist_yrp_idx"], dtype=np.int32)
            self.gravity_vec = np.array(config["gravity_vec"], dtype=np.float32)

        # Compute dof_action_scale per motor index (matching C++ formula)
        # dof_action_scale[i] = action_scale * tau_limit[i] / kps[i]
        self.dof_action_scale = self.action_scale * self.tau_limit / self.kps

        # Runtime state
        self._high_speed_mode = False
        self._terminate_flag = False
        self._vCmdBodyPast = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self._target_pos = np.zeros(self.num_actions, dtype=np.float32)

        # Load ONNX model
        self._loadPolicy()

        print(f"[MJAMP] Policy initialized. Model: {self.onnx_path}")
        print(f"[MJAMP] Observation: {self.num_obs} dims, Actions: {self.num_actions}")
        print(f"[MJAMP] Velocity limits (high): vx=[{self.vx_lim[0]}, {self.vx_lim[1]}], "
              f"vy=[{self.vy_lim[0]}, {self.vy_lim[1]}], wyaw=[{self.wyaw_lim[0]}, {self.wyaw_lim[1]}]")
        print(f"[MJAMP] Velocity limits (slow): vx=[{self.vx_lim_slow[0]}, {self.vx_lim_slow[1]}]")
        print(f"[MJAMP] Safe proj_gravity threshold: {self.safe_projgravity_threshold}")

    def _loadPolicy(self):
        """Load ONNX policy model."""
        self.ort_session = onnxruntime.InferenceSession(self.onnx_path)
        input_info = self.ort_session.get_inputs()
        self.input_name = input_info[0].name
        self.output_name = self.ort_session.get_outputs()[0].name
        print(f"[MJAMP] ONNX model loaded. Input: {self.input_name}, Output: {self.output_name}")

    def _getUserCmd(self):
        """
        Process joystick commands with dead zone and velocity limits.
        Matches C++ State_MJAMP::_getUserCmd().

        Joystick mapping:
            ly -> forward velocity (vx)
            lx -> lateral velocity (vy)
            rx -> yaw rate (wyaw)

        Velocity limits are directional:
            vx_lim = [min (backward), max (forward)]
            When joystick is negative, multiply by -vx_lim[0] (negates min to get positive scale)
            When joystick is positive, multiply by vx_lim[1]
        """
        # Extract raw joystick values from state_cmd.vel_cmd
        # deploy_real stores: vel_cmd[0]=ly, vel_cmd[1]=-lx, vel_cmd[2]=-rx
        ly = self.state_cmd.vel_cmd[0]
        lx = -self.state_cmd.vel_cmd[1]
        rx = -self.state_cmd.vel_cmd[2]

        # Select velocity limits based on speed mode
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

        # Exponential smoothing
        new_cmd = np.array([vx, vy, wyaw], dtype=np.float32)
        vCmdBody = self._vCmdBodyPast * self.cmd_smoothes + new_cmd * (1.0 - self.cmd_smoothes)
        self._vCmdBodyPast = vCmdBody.copy()

        return vCmdBody

    def _observations_compute(self):
        """
        Compute the observation for the policy network.
        Matches C++ State_MJAMP::_observations_compute().

        Returns:
            numpy array of shape (384,) — clipped observation buffer
        """
        # 1. Projected gravity from IMU quaternion (QuatRotateInverse in C++)
        base_quat = np.array([
            self.state_cmd.base_quat[0],  # w
            self.state_cmd.base_quat[1],  # x
            self.state_cmd.base_quat[2],  # y
            self.state_cmd.base_quat[3],  # z
        ], dtype=np.float32)
        projected_gravity = get_gravity_orientation_real(base_quat)

        # 2. Body angular velocity (gyroscope)
        ang_vel = self.state_cmd.ang_vel.copy()
        # Ensure ang_vel is 1D with 3 elements
        ang_vel = np.asarray(ang_vel, dtype=np.float32).flatten()[:3]

        # 3. Velocity commands from joystick
        vCmdBody = self._getUserCmd()

        # 4. Joint positions (offset from default_dof_pos)
        # C++: dof_pos_vec[i] = motorState[dof_mapping_mj[i]].q - default_dof_pos[dof_mapping_mj[i]]
        # With identity dof_mapping: dof_pos[i] = motorState[i].q - default[i]
        dof_pos_motor = self.state_cmd.q.copy()
        dof_pos_policy = dof_pos_motor[self.dof_mapping]  # identity: same as dof_pos_motor
        default_policy = self.default_dof_pos[self.dof_mapping]
        dof_pos_scaled = (dof_pos_policy - default_policy) * self.dof_pos_scale

        # 5. Joint velocities
        # C++: dof_vel_vec[i] = motorState[dof_mapping_mj[i]].dq
        dof_vel_motor = self.state_cmd.dq.copy()
        dof_vel_policy = dof_vel_motor[self.dof_mapping]
        dof_vel_scaled = dof_vel_policy * self.dof_vel_scale

        # 6. Scale angular velocity
        ang_vel_scaled = ang_vel * self.ang_vel_scale

        # 7. Build current robot state (96 dims)
        current_robot_state = np.concatenate([
            ang_vel_scaled,          # 3
            projected_gravity,       # 3
            vCmdBody,                # 3
            dof_pos_scaled,          # 29
            dof_vel_scaled,          # 29
            self._last_action,       # 29
        ], axis=0, dtype=np.float32)  # Total: 96

        # 8. Update sliding window buffer (shift left, append new at end)
        self.obs_buffer[0:self.num_obs - self.robot_state_dim] = \
            self.obs_buffer[self.robot_state_dim:self.num_obs]
        self.obs_buffer[self.num_obs - self.robot_state_dim:self.num_obs] = current_robot_state

        # 9. Anchor termination check
        # projected_gravity[2] should be close to -1.0 (gravity pointing down in robot frame)
        anchor_proj_gravity_error = abs(projected_gravity[2] - (-1.0))
        if anchor_proj_gravity_error > self.safe_projgravity_threshold:
            self._terminate_flag = True
            print(f"[MJAMP Warning] Large anchor projected gravity error: {anchor_proj_gravity_error:.4f} "
                  f"(threshold: {self.safe_projgravity_threshold})")

        # 10. Return clipped observation
        observation = np.clip(self.obs_buffer, -self.clip_observations, self.clip_observations)
        return observation

    def _action_compute(self, observation):
        """
        Run ONNX inference and convert policy output to motor target positions.
        Matches C++ State_MJAMP::_action_compute().
        """
        try:
            # Prepare input tensor [1, 384]
            obs_tensor = observation.reshape(1, -1).astype(np.float32)

            # Run ONNX inference
            outputs = self.ort_session.run(
                [self.output_name],
                {self.input_name: obs_tensor}
            )
            action_policy = outputs[0].squeeze()

            # Clip actions
            action_policy = np.clip(action_policy, -self.clip_actions, self.clip_actions)

            # Scale actions to motor target positions
            # C++: _joint_q[dof_mapping_mj[i]] = action[i] * dof_action_scale[dof_mapping_mj[i]] + default[dof_mapping_mj[i]]
            # With identity dof_mapping: target[i] = action[i] * dof_action_scale[i] + default[i]
            target_pos_motor = np.zeros(self.num_actions, dtype=np.float32)
            for policy_idx in range(self.num_actions):
                motor_idx = self.dof_mapping[policy_idx]
                target_pos_motor[motor_idx] = (
                    action_policy[policy_idx] * self.dof_action_scale[motor_idx]
                    + self.default_dof_pos[motor_idx]
                )

            # Store last action for next observation
            self._last_action = action_policy.copy()
            self._target_pos = target_pos_motor.copy()

            # Write to policy output
            self.policy_output.actions = target_pos_motor.copy()
            self.policy_output.kps = self.kps.copy()
            self.policy_output.kds = self.kds.copy()

        except Exception as e:
            print(f"[MJAMP Error] ONNX inference failed: {e}")
            # Hold current position on error
            self.policy_output.actions = self.state_cmd.q.copy()
            self.policy_output.kps = self.kps.copy()
            self.policy_output.kds = self.kds.copy()

    def _init_buffers(self):
        """
        Initialize observation buffer by filling with initial state.
        Matches C++ State_MJAMP::_init_buffers().
        Calls _observations_compute() history_length times.
        """
        self._vCmdBodyPast = np.zeros(3, dtype=np.float32)
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        for _ in range(self.history_length):
            self._observations_compute()
        print(f"[MJAMP] Buffers initialized with {self.history_length} history frames")

    def enter(self):
        """
        Enter MJAMP state. Reset flags and initialize observation buffers.
        Matches C++ State_MJAMP::enter().
        """
        print("[MJAMP] Entering MJAMP state...")
        self._high_speed_mode = False
        self._terminate_flag = False

        # Set motor commands to current position (hold in place during transition)
        self.policy_output.actions = self.state_cmd.q.copy()
        self.policy_output.kps = self.kps.copy()
        self.policy_output.kds = self.kds.copy()

        # Initialize observation history buffer
        self._init_buffers()
        print("[MJAMP] State entered successfully")

    def run(self):
        """
        Main control loop iteration.
        Compute observation, run policy inference, and set motor commands.
        Matches C++ State_MJAMP::run().
        """
        observation = self._observations_compute()
        self._action_compute(observation)

    def exit(self):
        """
        Exit MJAMP state. Reset all buffers and flags.
        Matches C++ State_MJAMP::exit().
        """
        print("[MJAMP] Exiting MJAMP state...")
        self._last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.obs_buffer = np.zeros(self.num_obs, dtype=np.float32)
        self._vCmdBodyPast = np.zeros(3, dtype=np.float32)
        self._terminate_flag = False
        self._high_speed_mode = False
        self._target_pos = np.zeros(self.num_actions, dtype=np.float32)
        print("[MJAMP] State exited")

    def checkChange(self):
        """
        Check for state transitions based on remote controller input and safety flags.
        Matches C++ State_MJAMP::checkChange().

        Priority order:
            1. L2+B → PASSIVE (emergency stop)
            2. Anchor termination → PASSIVE (safety)
            3. skill_cmd transitions (PASSIVE, LOCO)
            4. R2+B → LOCOMODE
            5. R1 released → LOCOMODE
            6. R2/R2 release → speed toggle (stay in MJAMP)
            7. SELECT → PASSIVE
        """
        # 1. Emergency stop: L2 + B -> PASSIVE
        if (self.remote_controller is not None and
            self.remote_controller.is_button_pressed(KeyMap.L2) and
            self.remote_controller.is_button_pressed(KeyMap.B)):
            print("[MJAMP] L2+B pressed → switching to PASSIVE")
            return FSMStateName.PASSIVE

        # 2. Anchor termination safety trigger
        if self._terminate_flag:
            print("[MJAMP] Anchor termination triggered → switching to PASSIVE")
            return FSMStateName.PASSIVE

        # 3. Standard skill_cmd transitions
        if self.state_cmd.skill_cmd == FSMCommand.PASSIVE:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            print("[MJAMP] PASSIVE command → switching to PASSIVE")
            return FSMStateName.PASSIVE

        if self.state_cmd.skill_cmd == FSMCommand.POS_RESET:
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.FIXEDPOSE

        # 4. R2 + B -> LOCOMODE (exit to base locomotion)
        if (self.remote_controller is not None and
            self.remote_controller.is_button_pressed(KeyMap.R2) and
            self.remote_controller.is_button_pressed(KeyMap.B)):
            print("[MJAMP] R2+B pressed → switching to LOCOMODE")
            return FSMStateName.LOCOMODE

        # 5. R1 released -> LOCOMODE
        if (self.remote_controller is not None and
            self.remote_controller.is_button_released(KeyMap.R1)):
            print("[MJAMP] R1 released → switching to LOCOMODE")
            return FSMStateName.LOCOMODE

        # 6. SELECT → PASSIVE
        if (self.remote_controller is not None and
            self.remote_controller.is_button_pressed(KeyMap.select)):
            print("[MJAMP] SELECT pressed → switching to PASSIVE")
            return FSMStateName.PASSIVE

        # 7. R2 released (trigger up) → toggle high speed mode ON
        if (self.remote_controller is not None and
            self.remote_controller.is_button_released(KeyMap.R2)):
            if not self._high_speed_mode:
                print("[MJAMP] Switching to HIGH SPEED mode")
                self._high_speed_mode = True
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.MJAMP  # Same state → no transition in FSM

        # 8. R2 pressed (trigger down) → toggle high speed mode OFF
        if (self.remote_controller is not None and
            self.remote_controller.is_button_pressed(KeyMap.R2)):
            if self._high_speed_mode:
                print("[MJAMP] Switching to LOW SPEED mode")
                self._high_speed_mode = False
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.MJAMP  # Same state → no transition in FSM

        # 9. Default: stay in MJAMP
        self.state_cmd.skill_cmd = FSMCommand.INVALID
        return FSMStateName.MJAMP


