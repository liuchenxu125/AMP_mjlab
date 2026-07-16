from humanoid.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class L0Cfg(LeggedRobotCfg):
    """
    Configuration class for the XBotL humanoid robot.
    """
    class env(LeggedRobotCfg.env):
        # change the observation dim
        frame_stack = 15
        c_frame_stack = 3
        num_single_obs = 65

        num_observations = int(frame_stack * num_single_obs)
        single_num_privileged_obs = 97

        num_privileged_obs = int(c_frame_stack * single_num_privileged_obs)
        num_actions = 18

        num_envs = 4096
        episode_length_s = 24 # episode length in seconds
        use_ref_actions = False
        joint_num = 18

        
    class safety:
        # safety factors
        pos_limit = 1.0
        vel_limit = 1.0
        torque_limit = 0.85


    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/HL/urdf/L0.urdf'

        name = "L0"
        foot_name = "6_link"
        knee_name = "4_link"

        terminate_after_contacts_on = ['base_link', "4_link", "arm_link1", "arm_link2", "arm_link3"]
        penalize_contacts_on = ["base_link"]
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        flip_visual_attachments = False
        replace_cylinder_with_capsule = False
        fix_base_link = False

    class terrain(LeggedRobotCfg.terrain):
        # mesh_type = 'plane'
        mesh_type = 'trimesh'
        curriculum = False
        # rough terrain only:
        measure_heights = False
        static_friction = 0.6
        dynamic_friction = 0.6
        terrain_length = 8.
        terrain_width = 8.
        num_rows = 20  # number of terrain rows (levels)
        num_cols = 20  # number of terrain cols (types)
        max_init_terrain_level = 10  # starting curriculum state
        # plane; obstacles; uniform; slope_up; slope_down, stair_up, stair_down
        terrain_proportions = [0.2, 0.2, 0.4, 0.1, 0.1, 0, 0]
        restitution = 0.

    class noise(LeggedRobotCfg.noise):
        add_noise = True
        noise_level = 1.5    # scales other values

        class noise_scales(LeggedRobotCfg.noise.noise_scales):
            dof_pos = 0.02
            dof_vel = 2.5 
            ang_vel = 0.2   
            lin_vel = 0.1   
            quat = 0.1
            gravity = 0.05
            height_measurements = 0.1


    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.1]
        
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'left_joint_1': 0.0,
            'left_joint_2': 0.1,
            'left_joint_3': -0.53,
            'left_joint_4': 0.0,
            'left_joint_5': 0.0,
            'left_joint_6': -0.35,
            'left_joint_7': 0.70,
            'left_joint_8': -0.35,
            'left_joint_9': -0.0,

            'right_joint_1': 0.0,
            'right_joint_2': -0.1,
            'right_joint_3': -0.53,
            'right_joint_4': -0.0,
            'right_joint_5': 0.0,
            'right_joint_6': -0.35,
            'right_joint_7': 0.70,
            'right_joint_8': -0.35, 
            'right_joint_9': 0.0,
 
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'

        stiffness = {'_joint_1': 50, '_joint_2': 50, '_joint_3': 50,
                     '_joint_4': 50, '_joint_5': 50, '_joint_6': 70,
                     '_joint_7': 70, '_joint_8': 40, '_joint_9': 40}
        damping = {'_joint_1': 5.0, '_joint_2': 5.0, '_joint_3': 5.0,
                   '_joint_4': 5.0, '_joint_5': 5.0, '_joint_6': 7.0, 
                   '_joint_7': 7.0, '_joint_8': 0.1, '_joint_9': 0.1}

        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 10  # 50hz 100hz

    class sim(LeggedRobotCfg.sim):
        dt = 0.001  # 200 Hz 1000 Hz
        substeps = 1  # 2
        up_axis = 1  # 0 is y, 1 is z
     
        class physx(LeggedRobotCfg.sim.physx):
            num_threads = 10
            solver_type = 1  # 0: pgs, 1: tgs
            num_position_iterations = 4
            num_velocity_iterations = 0
            contact_offset = 0.01  # [m]
            rest_offset = 0.0   # [m]
            bounce_threshold_velocity = 0.5  # 0.5 #0.5 [m/s]
            max_depenetration_velocity = 1.0
            max_gpu_contact_pairs = 2**23  # 2**24 -> needed for 8000 envs and more
            default_buffer_size_multiplier = 5
            # 0: never, 1: last sub-step, 2: all sub-steps (default=2)
            contact_collection = 2

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.2, 1.3]
        restitution_range = [0.0, 0.4]

        push_robots = True
        push_interval_s = 8
        max_push_vel_xy = 0.4
        max_push_ang_vel = 0.6

        randomize_base_mass = True
        added_mass_range = [-4.0, 4.0]

        randomize_com = True
        com_displacement_range = [-0.06, 0.06]

        randomize_gains = True
        stiffness_multiplier_range = [0.8, 1.2]  # Factor
        damping_multiplier_range = [0.8, 1.2]    # Factor

        randomize_torque = True
        torque_multiplier_range = [0.8, 1.2]

        randomize_link_mass = True
        added_link_mass_range = [0.8, 1.2]

        randomize_motor_offset = True
        motor_offset_range = [-0.035, 0.035] # Offset to add to the motor angles

        randomize_joint_friction = True
        randomize_joint_friction_each_joint = False
        joint_friction_range = [0.01, 1.15]
        joint_1_friction_range = [0.01, 1.15]
        joint_2_friction_range = [0.01, 1.15]
        joint_3_friction_range = [0.01, 1.15]
        joint_4_friction_range = [0.5, 1.3]
        joint_5_friction_range = [0.5, 1.3]
        joint_6_friction_range = [0.01, 1.15]
        joint_7_friction_range = [0.01, 1.15]
        joint_8_friction_range = [0.01, 1.15]
        joint_9_friction_range = [0.5, 1.3]
        joint_10_friction_range = [0.5, 1.3]

        randomize_joint_damping = True
        randomize_joint_damping_each_joint = False
        joint_damping_range = [0.3, 1.5]
        joint_1_damping_range = [0.3, 1.5]
        joint_2_damping_range = [0.3, 1.5]
        joint_3_damping_range = [0.3, 1.5]
        joint_4_damping_range = [0.9, 1.5]
        joint_5_damping_range = [0.9, 1.5]
        joint_6_damping_range = [0.3, 1.5]
        joint_7_damping_range = [0.3, 1.5]
        joint_8_damping_range = [0.3, 1.5]
        joint_9_damping_range = [0.9, 1.5]
        joint_10_damping_range = [0.9, 1.5]

        randomize_joint_armature = True
        randomize_joint_armature_each_joint = False#True
        joint_armature_range = [0.008, 0.06]
        joint_1_armature_range = [0.008, 0.06]
        joint_2_armature_range = [0.008, 0.06]
        joint_3_armature_range = [0.008, 0.06]
        joint_4_armature_range = [0.008, 0.06]
        joint_5_armature_range = [0.0007, 0.005]
        joint_6_armature_range = [0.0007, 0.005]
        joint_7_armature_range = [0.008, 0.06]
        joint_8_armature_range = [0.008, 0.06]
        joint_9_armature_range = [0.008, 0.06]
        joint_10_armature_range = [0.008, 0.06]
        joint_11_armature_range = [0.0007, 0.005]
        joint_12_armature_range = [0.0007, 0.005]


        add_lag = True
        randomize_lag_timesteps = True
        randomize_lag_timesteps_perstep = False
        lag_timesteps_range = [30, 50]
        
        add_obs_lag = True
        randomize_obs_motor_lag_timesteps = True
        randomize_obs_motor_lag_timesteps_perstep = False
        obs_motor_lag_timesteps_range = [50, 70]
        randomize_obs_actions_lag_timesteps = False
        randomize_obs_actions_lag_timesteps_perstep = False
        obs_actions_lag_timesteps_range = [2, 5]
        randomize_obs_imu_lag_timesteps = True
        randomize_obs_imu_lag_timesteps_perstep = False
        obs_imu_lag_timesteps_range = [1, 10]
        
        randomize_coulomb_friction = False
        joint_coulomb_range = [0.1, 0.9]
        joint_viscous_range = [0.05, 0.1]
    class commands(LeggedRobotCfg.commands):
        curriculum = True
        max_curriculum = 1.7
        # Vers: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        num_commands = 4
        resampling_time = 8.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [-0.5, 1.5] # min max [m/s] 
            lin_vel_y = [-0.5, 0.5]   # min max [m/s]
            ang_vel_yaw = [-1.5, 1.5]    # min max [rad/s]
            heading = [-3.14, 3.14]

    class rewards:
        base_height_target = 0.805#0.827
        min_dist = 0.15
        max_dist = 0.8
        # put some settings here for LLM parameter tuning
        arm_scale = 0.25
        target_joint_pos_scale = 0.26    # rad ?
        target_feet_height = 0.08       # m  ?
        cycle_time = 0.8                # sec ??
        # if true negative total rewards are clipped at zero (avoids early termination problems)
        only_positive_rewards = True
        # tracking reward = exp(error*sigma)
        tracking_sigma = 5 
        max_contact_force = 500  # forces above this value are penalized
        
        class scales:
            joint_pos = 2.8
            feet_clearance = 1.6
            feet_contact_number = 1.4
            # gait
            feet_air_time = 1.5
            foot_slip = -0.1
            feet_distance = 0.2
            knee_distance = 0.2
            # contact 
            feet_contact_forces = -0.1
            # vel tracking
            tracking_lin_vel = 1.4
            tracking_ang_vel = 1.1
            vel_mismatch_exp = 0.5  # lin_z; ang x,y
            low_speed = 0.2
            track_vel_hard = 0.5
            # stand_still = 5
            # base pos
            default_joint_pos = 0.8
            orientation = 1.
            base_height = 0.2
            base_acc = 0.2
            # energy
            action_smoothness = -0.003
            torques = -1e-10
            dof_vel = -1e-5
            dof_acc = -5e-9
            collision = -1.

    class normalization:
        class obs_scales:
            lin_vel = 2.
            ang_vel = 1.
            dof_pos = 1.
            dof_vel = 0.05
            quat = 1.
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.


class L0CfgPPO(LeggedRobotCfgPPO):
    seed = 5
    runner_class_name = 'OnPolicyRunner'   # DWLOnPolicyRunner

    class policy:
        init_noise_std = 1.0
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [768, 256, 128]

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.001
        learning_rate = 1e-5
        num_learning_epochs = 2
        gamma = 0.994
        lam = 0.9
        num_mini_batches = 4

    class runner:
        policy_class_name = 'ActorCritic'
        algorithm_class_name = 'PPO'
        num_steps_per_env = 60  # per iteration
        max_iterations = 10000  # number of policy updates

        # logging
        save_interval = 500  # check for potential saves every this many iterations
        experiment_name = 'L0_ppo'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1  # -1 = last run
        checkpoint = -1  # -1 = last saved model
        resume_path = None  # updated from load_run and chkpt

    