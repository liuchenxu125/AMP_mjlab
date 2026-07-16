import os
import cv2
import numpy as np
from isaacgym import gymapi
from humanoid import LEGGED_GYM_ROOT_DIR

# import isaacgym
from humanoid.envs import *
from humanoid.utils import  get_args, export_policy_as_jit, task_registry, Logger
from isaacgym.torch_utils import *

import torch
from tqdm import tqdm
from datetime import datetime

import pygame
from threading import Thread
import pandas as pd
import time 
# from plot import SimpleLogger, record_data

x_vel_cmd, y_vel_cmd, yaw_vel_cmd = 0.0, 0.0, 0.0
# x_scale, y_scale, yaw_scale = 2.5, 2.0, 0.0
joystick_use = True
joystick_opened = False

if joystick_use:

    pygame.init()

    try:
        # 获取手柄
        joystick = pygame.joystick.Joystick(0)
        joystick.init()
        joystick_opened = True
    except Exception as e:
        print(f"无法打开手柄：{e}")

    # 用于控制线程退出的标志
    exit_flag = False


# 处理手柄输入的线程
    def handle_joystick_input():
        global exit_flag, x_vel_cmd, y_vel_cmd, yaw_vel_cmd, head_vel_cmd
        
        
        while not exit_flag:
            # 获取手柄输入
            pygame.event.get()

            # 更新机器人命令
            x_vel_cmd = -joystick.get_axis(1) * 1.5
            y_vel_cmd = -joystick.get_axis(0) * 1
            yaw_vel_cmd = -joystick.get_axis(3) * 3

            print(x_vel_cmd, y_vel_cmd, yaw_vel_cmd)

            # 等待一小段时间，可以根据实际情况调整
            pygame.time.delay(100)

        # 启动线程

    if joystick_opened and joystick_use:
        joystick_thread = Thread(target=handle_joystick_input)
        joystick_thread.start()

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # env_cfg.init_state.pos = [0.0, 0.0, 1.2]
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 1)
    env_cfg.sim.max_gpu_contact_pairs = 2**10
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.terrain.mesh_type = 'plane'
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False     
    env_cfg.terrain.max_init_terrain_level = 5
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False 
    env_cfg.domain_rand.push_robots = False 
    env_cfg.domain_rand.continuous_push = False 
    env_cfg.domain_rand.randomize_base_mass = False 
    env_cfg.domain_rand.randomize_com = False 
    env_cfg.domain_rand.randomize_gains = False 
    env_cfg.domain_rand.randomize_torque = False 
    env_cfg.domain_rand.randomize_link_mass = False 
    env_cfg.domain_rand.randomize_motor_offset = False 
    env_cfg.domain_rand.randomize_joint_friction = False
    env_cfg.domain_rand.randomize_joint_damping = False
    env_cfg.domain_rand.randomize_joint_armature = False
    env_cfg.domain_rand.randomize_lag_timesteps = False
    # env_cfg.domain_rand.joint_angle_noise = 0.

    env_cfg.domain_rand.lag_timesteps_range = [5, 15]
    env_cfg.domain_rand.add_obs_lag = True
    env_cfg.domain_rand.randomize_obs_motor_lag_timesteps = True
    env_cfg.domain_rand.randomize_obs_motor_lag_timesteps_perstep = False
    env_cfg.domain_rand.obs_motor_lag_timesteps_range = [5, 15]
    env_cfg.domain_rand.randomize_obs_actions_lag_timesteps = False
    env_cfg.domain_rand.randomize_obs_actions_lag_timesteps_perstep = False
    env_cfg.domain_rand.obs_actions_lag_timesteps_range = [2, 5]
    env_cfg.domain_rand.randomize_obs_imu_lag_timesteps = True
    env_cfg.domain_rand.randomize_obs_imu_lag_timesteps_perstep = False
    env_cfg.domain_rand.obs_imu_lag_timesteps_range = [1, 10]
    env_cfg.noise.curriculum = False
    # env_cfg.noise.noise_level = 1.0
    # env_cfg.sim.dt = 0.0005
    # env_cfg.sim.sim_duration = 60
    # env_cfg.control.decimation = 40
    env_cfg.commands.heading_command = False



    train_cfg.seed = 123145
    print("train_cfg.runner_class_name:", train_cfg.runner_class_name)

    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    env.set_camera(env_cfg.viewer.pos, env_cfg.viewer.lookat)



    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, _ = task_registry.make_alg_runner(env=env, name=args.task, args=args, train_cfg=train_cfg)
    policy = ppo_runner.get_inference_policy(device=env.device)
    
    # export policy as a jit module (used to run it from C++)
    current_date_str = datetime.now().strftime('%Y-%m-%d')
    current_time_str = datetime.now().strftime('%H-%M-%S')
    if EXPORT_POLICY:
        path = os.path.join(LEGGED_GYM_ROOT_DIR, 'logs', train_cfg.runner.experiment_name, '0_exported', 'policies')
        export_policy_as_jit(ppo_runner.alg.actor_critic, path)
        print('Exported policy as jit script to: ', path)

    logger = Logger(env_cfg.sim.dt)
    robot_index = 0 # which robot is used for logging
    joint_index = 5 # which joint is used for logging
    stop_state_log = 1000 # number of steps before plotting states
    # sloger = SimpleLogger(f'{LEGGED_GYM_ROOT_DIR}/logs/play_log', record_data())

    # current_directory = os.getcwd()
    # print("Current directory: ", current_directory)
    # data = pd.read_csv('test.csv')
    # left = "left_joint_"
    # right = "right_joint_"
    # res = []
    # res_l = []
    # res_r = []
    # for i in range (6):
    #     column_data = data[f"{left}{i}"]
    #     column_data = np.array(column_data)
    #     res_l = np.concatenate((res_l, column_data), axis = 0)
    # for i in range (6):
    #     column_data = data[f"{right}{i}"]
    #     column_data = np.array(column_data)
    #     res_r = np.concatenate((res_r, column_data), axis = 0)

    # res_l = res_l.reshape(6,221)
    # res_l = res_l.transpose()
    # res_r = res_r.reshape(6,221)
    # res_r = res_r.transpose()

    # res = np.concatenate((res_l, res_r), axis = 1)
    # traj1 = torch.tensor(res)
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # traj1 = traj1.to(device)

    if RENDER:
        camera_properties = gymapi.CameraProperties()
        camera_properties.width = 1920
        camera_properties.height = 1080
        h1 = env.gym.create_camera_sensor(env.envs[0], camera_properties)
        camera_offset = gymapi.Vec3(1, -1, 0.5)
        camera_rotation = gymapi.Quat.from_axis_angle(gymapi.Vec3(-0.3, 0.2, 1),
                                                    np.deg2rad(135))
        actor_handle = env.gym.get_actor_handle(env.envs[0], 0)
        body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], actor_handle, 0)
        env.gym.attach_camera_to_body(
            h1, env.envs[0], body_handle,
            gymapi.Transform(camera_offset, camera_rotation),
            gymapi.FOLLOW_POSITION)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos')
        experiment_dir = os.path.join(LEGGED_GYM_ROOT_DIR, 'videos', train_cfg.runner.experiment_name)
        dir = os.path.join(experiment_dir, datetime.now().strftime('%b%d_%H-%M-%S')+ args.run_name + '.mp4')
        if not os.path.exists(video_dir):
            os.makedirs(video_dir,exist_ok=True)
        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir,exist_ok=True)
        video = cv2.VideoWriter(dir, fourcc, 50.0, (1920, 1080))
    
    obs = env.get_observations()
    np.set_printoptions(formatter={'float': '{:0.4f}'.format})

    # camera_position = np.array(env_cfg.viewer.pos, dtype=np.float64)
    # camera_vel = np.array([1., 0., 0.])
    # camera_direction = np.array(env_cfg.viewer.lookat) - np.array(env_cfg.viewer.pos)
    for i in range(10*stop_state_log):
        
        # print("1111111111111111111111",obs[0].cpu().detach().numpy())
        # arr = obs[0].cpu().detach().numpy()
        # arr_reshaped = arr.reshape((15, 47))

        # # 遍历这个二维数组，打印每一行
        # for row in arr_reshaped:
        #     print(row)
        # if i<300:
        #     camera_position += camera_vel * 0.05
        #     env.set_camera(camera_position, camera_position + camera_direction)
        # env.set_camera(camera_position, camera_position + camera_direction)
        
        actions = policy(obs.detach()) # * 0.
        # print(actions)
        # print("\n",actions[0].cpu().detach().numpy())
        # sloger.save(obs[3:14], i, t1 - t0)
        # t0 = t1
        # t1 = time.time()
        
        if FIX_COMMAND:
            env.commands[:, 0] = 0.5    # 1.0
            env.commands[:, 1] = 0.
            env.commands[:, 2] = 0.
            env.commands[:, 3] = 0.
            
        else:
            env.commands[:, 0] = x_vel_cmd
            env.commands[:, 1] = y_vel_cmd
            env.commands[:, 2] = yaw_vel_cmd
            env.commands[:, 3] = 0.
        
        obs, critic_obs, rews, dones, infos = env.step(actions.detach())

        if RENDER:
            env.gym.fetch_results(env.sim, True)
            env.gym.step_graphics(env.sim)
            env.gym.render_all_camera_sensors(env.sim)
            img = env.gym.get_camera_image(env.sim, env.envs[0], h1, gymapi.IMAGE_COLOR)
            img = np.reshape(img, (1080, 1920, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            video.write(img[..., :3])

        if i < stop_state_log:
            logger.log_states(
                {
                    'dof_pos_target': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                    'dof_pos_target[0]': actions[robot_index, 0].item() * env.cfg.control.action_scale,
                    'dof_pos_target[1]': actions[robot_index, 1].item() * env.cfg.control.action_scale,
                    'dof_pos_target[2]': actions[robot_index, 2].item() * env.cfg.control.action_scale,
                    'dof_pos_target[3]': actions[robot_index, 3].item() * env.cfg.control.action_scale,
                    'dof_pos_target[4]': actions[robot_index, 4].item() * env.cfg.control.action_scale,
                    'dof_pos_target[5]': actions[robot_index, 5].item() * env.cfg.control.action_scale,
                    'dof_pos_target[6]': actions[robot_index, 6].item() * env.cfg.control.action_scale,
                    'dof_pos_target[7]': actions[robot_index, 7].item() * env.cfg.control.action_scale,
                    'dof_pos_target[8]': actions[robot_index, 8].item() * env.cfg.control.action_scale,
                    'dof_pos_target[9]': actions[robot_index, 9].item() * env.cfg.control.action_scale,
                    'dof_pos_target[10]': actions[robot_index,10].item() * env.cfg.control.action_scale,
                    'dof_pos_target[11]': actions[robot_index, 11].item() * env.cfg.control.action_scale,
                    'dof_pos': env.dof_pos[robot_index, 0].item(),
                    'dof_pos[0]': env.dof_pos[robot_index, 0].item(),
                    'dof_pos[1]': env.dof_pos[robot_index, 1].item(),
                    'dof_pos[2]': env.dof_pos[robot_index, 2].item(),
                    'dof_pos[3]': env.dof_pos[robot_index, 3].item(),
                    'dof_pos[4]': env.dof_pos[robot_index, 4].item(),
                    'dof_pos[5]': env.dof_pos[robot_index, 5].item(),
                    'dof_pos[6]': env.dof_pos[robot_index, 6].item(),
                    'dof_pos[7]': env.dof_pos[robot_index, 7].item(),
                    'dof_pos[8]': env.dof_pos[robot_index, 8].item(),
                    'dof_pos[9]': env.dof_pos[robot_index, 9].item(),
                    'dof_pos[10]': env.dof_pos[robot_index, 10].item(),
                    'dof_pos[11]': env.dof_pos[robot_index, 11].item(),
                    'dof_torque': env.torques[robot_index, 0].item(),
                    'dof_torque[0]': env.torques[robot_index, 0].item(),
                    'dof_torque[1]': env.torques[robot_index, 1].item(),
                    'dof_torque[2]': env.torques[robot_index, 2].item(),
                    'dof_torque[3]': env.torques[robot_index, 3].item(),
                    'dof_torque[4]': env.torques[robot_index, 4].item(),
                    'dof_torque[5]': env.torques[robot_index, 5].item(),
                    'dof_torque[6]': env.torques[robot_index, 6].item(),
                    'dof_torque[7]': env.torques[robot_index, 7].item(),
                    'dof_torque[8]': env.torques[robot_index, 8].item(),
                    'dof_torque[9]': env.torques[robot_index, 9].item(),
                    'dof_torque[10]': env.torques[robot_index, 10].item(),
                    'dof_torque[11]': env.torques[robot_index, 11].item(),
                    'dof_vel': env.dof_vel[robot_index, 0].item(),
                    'dof_vel[0]': env.dof_vel[robot_index, 0].item(),
                    'dof_vel[1]': env.dof_vel[robot_index, 1].item(),
                    'dof_vel[2]': env.dof_vel[robot_index, 2].item(),
                    'dof_vel[3]': env.dof_vel[robot_index, 3].item(),
                    'dof_vel[4]': env.dof_vel[robot_index, 4].item(),
                    'dof_vel[5]': env.dof_vel[robot_index, 5].item(),
                    'dof_vel[6]': env.dof_vel[robot_index, 6].item(),
                    'dof_vel[7]': env.dof_vel[robot_index, 7].item(),
                    'dof_vel[8]': env.dof_vel[robot_index, 8].item(),
                    'dof_vel[9]': env.dof_vel[robot_index, 9].item(),
                    'dof_vel[10]': env.dof_vel[robot_index, 10].item(),
                    'dof_vel[11]': env.dof_vel[robot_index, 11].item(),
                    'command_x': env.commands[robot_index, 0].item(),
                    'command_y': env.commands[robot_index, 1].item(),
                    'command_yaw': env.commands[robot_index, 2].item(),
                    'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                    'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                    'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                    'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                    'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy()
                }
                )
            
        elif i == stop_state_log:
            logger.plot_states()

        # ====================== Log states ======================
        if infos["episode"]:
            num_episodes = torch.sum(env.reset_buf).item()
            if num_episodes>0:
                logger.log_rewards(infos["episode"], num_episodes)

    # logger.print_rewards()
    
    # while True:
    #     True

    if RENDER:
        video.release()

if __name__ == '__main__':
    EXPORT_POLICY = True
    RENDER = False
    FIX_COMMAND = False
    # MOVE_CAMERA = True
    args = get_args()
    play(args)
