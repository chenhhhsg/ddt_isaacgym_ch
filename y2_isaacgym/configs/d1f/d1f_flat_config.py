from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi
from configs.d1f.d1f_command import *

class D1FFlatCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 16 + 16 + 16
        n_proprio = 60 #
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 16  # fixed ABADOF joints
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,   # [rad]
            'FR_hip_joint': 0.0 ,  # [rad]
            'RL_hip_joint': 0.0,   # [rad]
            'RR_hip_joint': 0.0,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.0,   # [rad]
            'RR_thigh_joint': 1.0,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'RR_calf_joint': -1.5,    # [rad]

            'FL_foot_joint':0.0,
            'FR_foot_joint':0.0,
            'RL_foot_joint':0.0,
            'RR_foot_joint':0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 400.,
                     'thigh': 40.,
                     'calf': 40.,
                     'foot': 10.}  # [N*m/rad]
        damping = {'hip': 1.0,
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = True 
        max_curriculum = 3.0
        num_commands = 4  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]   # m/s
            lin_vel_y = [0.0, 0.0]   # 12 DOF 无法横向移动，设为 0
            ang_vel_yaw = [-1.0, 1.0] 
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/d1f/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1f"
        penalize_contacts_on = ["thigh", "calf", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-1., 3.]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]

        randomize_kpkd = False
        kp_range = [0.8,1.2]
        kd_range = [0.8,1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = True
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

    class rewards( LeggedRobotCfg.rewards ):
        only_positive_rewards = False  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        max_contact_force = 500.  # forces above this value are penalized

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = 0.0#-2e-5
            termination = 0.0
            tracking_ang_vel = 8.0
            lin_vel_z = -1.0
            tracking_lin_vel_x = 15.0
            tracking_lin_vel = 0.0

            orientation = -2.0         # 基础惩罚 (Combined Roll + Pitch)
            orientation_pitch = -5.0   # 强力惩罚 Pitch
            orientation_roll = -20.0    # 适度惩罚 Roll

            ang_vel_xy = -0.1
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -5.0
            feet_air_time = 2.0
            collision = -1.0
            # feet_stumble = -5.0
            action_rate = -0.01
            stand_still = 20.0
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            upward = 5.0
            feet_all_contact = -5.0
            # feet_contact_forces = -0.1
            # joint_power=-2e-5
            # powers_dist =-1.0e-5
            turn_yaw = 10.0

            
    class costs(LeggedRobotCfg.costs):
        num_costs = 5
        class scales:
            pos_limit = 1.0
            torque_limit = 1.0
            dof_vel_limits = 1.0
            hip_pos = 2.0
            default_joint= 0.2

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            hip_pos = 0.0
            default_joint = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type ='plane'
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        terrain_proportions = [0.5, 0.5, 0.0, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.3, 0.1, 0.1, 0.3]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        slope_treshold = 1.0  # slopes above this threshold will be corrected to vertical surfaces
        slope = [0, 0.4]

class D1FFlatCfg_Play( D1FFlatCfg ):
    class env(D1FFlatCfg.env):
        num_envs = 10
    class terrain(D1FFlatCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type = 'plane'
        num_rows = 5
        num_cols = 5
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        terrain_proportions = [1, 0, 0, 0, 0, 0, 0]
        curriculum = False
        # selected = True  # select a unique terrain type and pass all arguments
        # terrain_kwargs = {
        #     "type": "pit_terrain",  
        #     "depth": 0.5,                     
        #     "platform_size": 4.0               
        # } # Dict of arguments for selected terrain
    class noise( D1FFlatCfg.noise ):
        add_noise = False
    class control ( D1FFlatCfg.control ):
        use_filter = False
    class domain_rand( D1FFlatCfg.domain_rand ):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_friction = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False
    class commands( D1FFlatCfg.commands ):
        heading_command = False  # if true: compute ang vel command from heading error
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0.5, 0.5]  # min max [rad/s]
            heading = [-3.14, 3.14]

class D1FFlatCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4 # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        #priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 5

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1f_flat'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 5000
        max_iterations = 10000
        num_steps_per_env = 24
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = ''
