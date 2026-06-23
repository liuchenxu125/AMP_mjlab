from common.path_config import PROJECT_ROOT

from FSM.FSMState import FSMStateName, FSMState
from common.ctrlcomp import StateAndCmd, PolicyOutput, FSMCommand
from common.utils import scale_values
import numpy as np
import yaml
import torch
import os
# pip install roboticstoolbox-python spatialmath-python
from roboticstoolbox import ERobot
MUJOCO_DOF_NAMES = [
    'left_hip_pitch_joint',
    'left_hip_roll_joint',
    'left_hip_yaw_joint',
    'left_knee_joint',
    'left_ankle_pitch_joint',
    'left_ankle_roll_joint',
    'right_hip_pitch_joint',
    'right_hip_roll_joint',
    'right_hip_yaw_joint',
    'right_knee_joint',
    'right_ankle_pitch_joint',
    'right_ankle_roll_joint',
    'waist_yaw_joint',
    'waist_roll_joint',
    'waist_pitch_joint',
    'left_shoulder_pitch_joint',
    'left_shoulder_roll_joint',
    'left_shoulder_yaw_joint',
    'left_elbow_joint',
    'left_wrist_roll_joint',
    'left_wrist_pitch_joint',
    'left_wrist_yaw_joint',
    'right_shoulder_pitch_joint',
    'right_shoulder_roll_joint',
    'right_shoulder_yaw_joint',
    'right_elbow_joint',
    'right_wrist_roll_joint',
    'right_wrist_pitch_joint',
    'right_wrist_yaw_joint'
]

LAB_DOF_NAMES = [
'left_hip_pitch_joint',
'right_hip_pitch_joint',
'waist_yaw_joint',
'left_hip_roll_joint',
'right_hip_roll_joint',
'waist_roll_joint',
'left_hip_yaw_joint',
'right_hip_yaw_joint',
'waist_pitch_joint',
'left_knee_joint',
'right_knee_joint',
'left_shoulder_pitch_joint',
'right_shoulder_pitch_joint',
'left_ankle_pitch_joint',
'right_ankle_pitch_joint',
'left_shoulder_roll_joint',
'right_shoulder_roll_joint',
'left_ankle_roll_joint',
'right_ankle_roll_joint',
'left_shoulder_yaw_joint',
'right_shoulder_yaw_joint',
'left_elbow_joint',
'right_elbow_joint',
'left_wrist_roll_joint',
'right_wrist_roll_joint',
'left_wrist_pitch_joint',
'right_wrist_pitch_joint',
'left_wrist_yaw_joint',
'right_wrist_yaw_joint'
]

MUJOCO_BODY_NAMES=[
'world',
'pelvis',
'left_hip_pitch_link',
'left_hip_roll_link',
'left_hip_yaw_link',
'left_knee_link',
'left_ankle_pitch_link',
'left_ankle_roll_link',
'right_hip_pitch_link',
'right_hip_roll_link',
'right_hip_yaw_link',
'right_knee_link',
'right_ankle_pitch_link',
'right_ankle_roll_link',
'waist_yaw_link',
'waist_roll_link',
'torso_link',
'left_shoulder_pitch_link',
'left_shoulder_roll_link',
'left_shoulder_yaw_link',
'left_elbow_link',
'left_wrist_roll_link',
'left_wrist_pitch_link',
'left_wrist_yaw_link',
'right_shoulder_pitch_link',
'right_shoulder_roll_link',
'right_shoulder_yaw_link',
'right_elbow_link',
'right_wrist_roll_link',
'right_wrist_pitch_link',
'right_wrist_yaw_link'
]

KEY_BODY_NAMES = [
    "left_ankle_roll_link", 
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
]

class LocoMode(FSMState):
    def __init__(self, state_cmd:StateAndCmd, policy_output:PolicyOutput):
        super().__init__()
        self.state_cmd = state_cmd
        self.policy_output = policy_output
        self.name = FSMStateName.LOCOMODE
        self.name_str = "Loco_mode"
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, "config", "LocoMode.yaml")
        with open(config_path, "r") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)

            self.urdf_path = os.path.join(current_dir, "g1_urdf", "g1_29dof_rev_1_0.urdf")
            self.robot = ERobot.URDF(self.urdf_path)

            model_dir = os.path.join(current_dir, "model")
            model_files = [f for f in os.listdir(model_dir) if os.path.isfile(os.path.join(model_dir, f))]
            model_files.sort()
            if len(model_files) == 0:
                raise FileNotFoundError(f"No model file found in {model_dir}")
            self.policy_path = os.path.join(model_dir, model_files[0])

            self.kps = np.array(config["kps"], dtype=np.float32)
            self.kds = np.array(config["kds"], dtype=np.float32)
            self.tau_limit =  np.array(config["tau_limit"], dtype=np.float32)
            self.default_angles =  np.array(config["default_angles"], dtype=np.float32)
            self.velocity_limit =  np.array(config["velocity_limit"], dtype=np.float32)
            self.dof29_index =  np.array(config["dof29_index"], dtype=np.int32)
            
            self.num_actions = config["num_actions"]
            self.num_obs = config["num_obs"]
            self.ang_vel_scale = config["ang_vel_scale"]
            self.dof_pos_scale = config["dof_pos_scale"]
            self.dof_vel_scale = config["dof_vel_scale"]
            self.action_scale = config["action_scale"]
            self.cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)
            self.cmd_range = config["cmd_range"]
            self.range_velx = np.array([self.cmd_range["lin_vel_x"][0], self.cmd_range["lin_vel_x"][1]], dtype=np.float32)
            self.range_vely = np.array([self.cmd_range["lin_vel_y"][0], self.cmd_range["lin_vel_y"][1]], dtype=np.float32)
            self.range_velz = np.array([self.cmd_range["ang_vel_z"][0], self.cmd_range["ang_vel_z"][1]], dtype=np.float32)
            
            self.qj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.dqj_obs = np.zeros(self.num_actions, dtype=np.float32)
            self.cmd = np.array(config["cmd_init"], dtype=np.float32)
            self.obs = np.zeros(self.num_obs)
            self.actions = np.zeros(self.num_actions)
            self.last_actions = np.zeros(self.num_actions)
            self.history_length = 5
            # load policy
            self.policy = torch.jit.load(self.policy_path)

            print(f"Locomotion policy initializing ... Loaded model: {self.policy_path}")

    def enter(self):
        self.kps_reorder = np.zeros_like(self.kps)
        self.kds_reorder = np.zeros_like(self.kds)
        self.default_angles_reorder = np.zeros_like(self.default_angles)
        self.kps_reorder = self.kps.copy()
        self.kds_reorder = self.kds.copy()
        self.default_angles_reorder = self.default_angles.copy()
        self.buffer_obs_Init()

    def MJ29_to_LAB29(self, array1):
        """
        Reorder array1 (MUJOCO_DOF_NAMES order) to match the order of array2
        (LAB_DOF_NAMES). Extra elements in MUJOCO_DOF_NAMES ('waist_roll_joint',
        'waist_pitch_joint') are ignored.

        Args:
            array1 (list or np.ndarray): Input array in MUJOCO_DOF_NAMES order.
            array2 (list or np.ndarray): Output array in LAB_DOF_NAMES order.

        Returns:
            np.ndarray: Reordered array matching LAB_DOF_NAMES order.
        """
        # Create a mapping from MUJOCO_DOF_NAMES to their indices
        mujoco_indices = {
            name: idx for idx, name in enumerate(MUJOCO_DOF_NAMES)
        }

        # Reorder array1 based on LAB_DOF_NAMES
        reordered_array = [
            array1[mujoco_indices[name]] for name in LAB_DOF_NAMES
        ]

        return np.array(reordered_array)

    def LAB29_to_MJ29(self, array1):
        """
        Reorder array1 (LAB_DOF_NAMES order) to match the order of MUJOCO_DOF_NAMES.
        Extra elements in MUJOCO_DOF_NAMES ('waist_roll_joint', 'waist_pitch_joint')
        are set to 0.

        Args:
            array1 (list or np.ndarray): Input array in LAB_DOF_NAMES order.

        Returns:
            np.ndarray: Reordered array matching MUJOCO_DOF_NAMES order, with
            extra elements set to 0.
        """
        # Create a mapping from LAB_DOF_NAMES to their indices
        lab_indices = {
            name: idx for idx, name in enumerate(LAB_DOF_NAMES)
        }

        # Reorder array1 based on MUJOCO_DOF_NAMES, setting extra elements to 0
        reordered_array = [
            array1[lab_indices[name]] if name in lab_indices else 0
            for name in MUJOCO_DOF_NAMES
        ]

        return np.array(reordered_array)

    def root_local_rot_tan_norm(self, quat):
        """
        输入四元数（w, x, y, z），去除yaw分量后，返回局部旋转矩阵的切向量和法向量（6维，前3为切向量，后3为法向量）。
        quat: np.ndarray, shape=(4,) or (w, x, y, z)
        return: np.ndarray, shape=(6,)
        """
        quat = np.asarray(quat, dtype=np.float32)
        quat = quat / np.linalg.norm(quat)

        # 提取yaw分量
        qw, qx, qy, qz = quat
        # 计算yaw角
        yaw = np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy**2 + qz**2))
        # 构造仅含yaw的四元数
        cy = np.cos(yaw / 2)
        sy = np.sin(yaw / 2)
        yaw_quat = np.array([cy, 0.0, 0.0, sy], dtype=np.float32)

        # 四元数共轭
        yaw_quat_conj = np.array([yaw_quat[0], -yaw_quat[1], -yaw_quat[2], -yaw_quat[3]], dtype=np.float32)

        # 四元数乘法: q1 * q2
        def quat_mul(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            return np.array([
                w1*w2 - x1*x2 - y1*y2 - z1*z2,
                w1*x2 + x1*w2 + y1*z2 - z1*y2,
                w1*y2 - x1*z2 + y1*w2 + z1*x2,
                w1*z2 + x1*y2 - y1*x2 + z1*w2
            ], dtype=np.float32)

        # 去除yaw后的局部四元数
        quat_local = quat_mul(yaw_quat_conj, quat)

        # 转为旋转矩阵
        qw, qx, qy, qz = quat_local
        rotm = np.array([
            [1 - 2 * (qy**2 + qz**2), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx**2 + qy**2)]
        ], dtype=np.float32)

        tan_vec = rotm[:, 0]
        norm_vec = rotm[:, 2]
        obs = np.concatenate([tan_vec, norm_vec], axis=0)
        return obs

    def buffer_obs_Init(self):
        self.base_ang_vel_buffer = np.zeros((self.history_length, 3), dtype=np.float32)
        self.root_local_rot_tan_norm_buffer = np.zeros((self.history_length, 6), dtype=np.float32)
        self.velocity_commands_buffer = np.zeros((self.history_length, 3), dtype=np.float32)
        self.joint_pos_buffer = np.zeros((self.history_length, 29), dtype=np.float32)
        self.joint_vel_buffer = np.zeros((self.history_length, 29), dtype=np.float32)
        self.actions_buffer = np.zeros((self.history_length, 29), dtype=np.float32)
        self.key_body_pos_b_buffer = np.zeros((self.history_length, 18), dtype=np.float32)

    def clear_all_buffers(self):
        """
        一键清空所有观测缓冲区。
        """
        self.base_ang_vel_buffer[...] = 0
        self.root_local_rot_tan_norm_buffer[...] = 0
        self.velocity_commands_buffer[...] = 0
        self.joint_pos_buffer[...] = 0
        self.joint_vel_buffer[...] = 0
        self.actions_buffer[...] = 0
        self.key_body_pos_b_buffer[...] = 0

    def update_buffer(self, buffer: np.ndarray, new_value: np.ndarray):
        """
        通用缓冲区更新方法，将 new_value 插入到 buffer 的最前面（buffer[0]），其余后移。
        buffer: np.ndarray, shape=(history_length, ...)
        new_value: np.ndarray, shape=(...)
        return: 更新后的 buffer（原地修改并返回）
        """
        buffer[:-1] = buffer[1:]
        buffer[-1] = new_value
        return buffer

    def velocity_cmd(self):
        vel_cmd = self.state_cmd.vel_cmd.copy()
        vel_cmd = scale_values(vel_cmd, [self.range_velx, self.range_vely, self.range_velz])
        if vel_cmd[0] < -2.0:
            vel_cmd[0] = -2.0
        elif -0.1 < vel_cmd[0] < 0.1:
            vel_cmd[0] = 0
        return vel_cmd

    def forward_kinematics(self, q, link=None):
        """
        计算机器人正向运动学。
        参数：
            q: 关节角度数组，长度为self.robot.n
            link: 指定链末端（如'left_foot'），默认None为主末端
        返回：
            T: SE3对象（齐次变换）
            pos: 位置向量 (3,)
            rot: 旋转矩阵 (3,3)
        """
        if link is not None:
            T = self.robot.fkine(q, end=link)
        else:
            T = self.robot.fkine(q)
        pos = T.t
        rot = T.R
        return T, pos, rot

    def key_body_pos(self, KEY_BODY_NAMES):
        joint_pos = self.state_cmd.q.copy()
        key_body_pos_b = np.zeros((len(KEY_BODY_NAMES), 3), dtype=np.float32)
        for i, link_name in enumerate(KEY_BODY_NAMES):
            T, pos, rot = self.forward_kinematics(joint_pos, link=link_name)
            key_body_pos_b[i, :] = pos
        key_body_pos_b = key_body_pos_b.reshape(-1)
        return key_body_pos_b

    def run(self):
        base_ang_vel = self.state_cmd.ang_vel.copy()
        root_local_rot_tan_norm = self.root_local_rot_tan_norm(self.state_cmd.base_quat)
        velocity_commands = self.velocity_cmd()
        joint_pos = self.MJ29_to_LAB29(self.state_cmd.q.copy())
        joint_vel = self.MJ29_to_LAB29(self.state_cmd.dq.copy())
        actions = self.last_actions.copy()
        key_body_pos_b = self.key_body_pos(KEY_BODY_NAMES)

        base_ang_vel_arry = self.update_buffer(self.base_ang_vel_buffer, base_ang_vel)
        root_local_rot_tan_norm_arry = self.update_buffer(self.root_local_rot_tan_norm_buffer, root_local_rot_tan_norm)
        velocity_commands_arry = self.update_buffer(self.velocity_commands_buffer, velocity_commands)
        joint_pos_arry = self.update_buffer(self.joint_pos_buffer, joint_pos)
        joint_vel_arry = self.update_buffer(self.joint_vel_buffer, joint_vel)
        actions_arry = self.update_buffer(self.actions_buffer, actions)
        key_body_pos_b_arry = self.update_buffer(self.key_body_pos_b_buffer, key_body_pos_b)

        obs = np.concatenate([
            base_ang_vel_arry.flatten(),
            root_local_rot_tan_norm_arry.flatten(),
            velocity_commands_arry.flatten(),
            joint_pos_arry.flatten(),
            joint_vel_arry.flatten(),
            actions_arry.flatten(),
            key_body_pos_b_arry.flatten()
        ], axis=0)

        obs = obs.reshape(1, -1)
        obs_tensor = torch.from_numpy(obs)
        self.actions = self.policy(obs_tensor).detach().numpy().squeeze()
        self.last_actions = self.actions
        self.MJ_actions = self.LAB29_to_MJ29(self.actions)
        loco_action = self.MJ_actions * self.action_scale + self.default_angles

        self.policy_output.actions = loco_action.copy()
        self.policy_output.kps = self.kps_reorder.copy()
        self.policy_output.kds = self.kds_reorder.copy()

    def exit(self):
        self.actions = np.zeros(self.num_actions)
        self.last_actions = np.zeros(self.num_actions)
        self.observations = np.zeros(self.num_obs)
        self.clear_all_buffers()
        pass
    
    def checkChange(self):
        if(self.state_cmd.skill_cmd == FSMCommand.SKILL_MJAMP):
            self.state_cmd.skill_cmd = FSMCommand.INVALID
            return FSMStateName.MJAMP
        if(self.state_cmd.skill_cmd == FSMCommand.SKILL_1):
            return FSMStateName.SKILL_Dance
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_2):
            return FSMStateName.SKILL_KungFu
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_3):
            return FSMStateName.SKILL_KICK
        elif(self.state_cmd.skill_cmd == FSMCommand.SKILL_4):
            return FSMStateName.SKILL_BEYOND_MIMIC
        elif(self.state_cmd.skill_cmd == FSMCommand.PASSIVE):
            return FSMStateName.PASSIVE
        else:
            return FSMStateName.LOCOMODE

