from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi
from configs.d1f_12DOF.d1f_12DOF import *

class D1F12DOFFlatCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 12 + 12 + 12
        n_proprio = 48 #  3+ 3+ 3+ 3+ 12+ 12+ 12 
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 12  # fixed ABADOF joints
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
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
        stiffness = {'thigh': 40.,
                     'calf': 40.,
                     'foot': 10.}  # [N*m/rad]
        damping = {
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        # hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = True 
        max_curriculum = 3.0
        num_commands = 4  # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]   # m/s
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-1.0, 1.0] 
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/d1f_12DOF/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1f_12DOF"
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
        feet_x_distance_target = 0.25
        feet_x_distance_sigma = 0.5
        # feet_y_distance_target = 0.474

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = 0.0#-2e-5
            termination = 0.0
            tracking_ang_vel = 20.0
            lin_vel_z = -2.0
            tracking_lin_vel_x = 20.0
            tracking_lin_vel = 0.0
            orientation = -1.0         # 基础惩罚 (Combined Roll + Pitch)
            orientation_pitch = -2.0   # 强力惩罚 Pitch
            orientation_roll = -1.0    # 适度惩罚 Roll
            ang_vel_xy = -0.05
            feet_air_time = 0.0
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -1.0
            collision = -1.0
            # feet_stumble = -5.0
            action_rate = -0.01
            stand_still = 20.0
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            upward = 0.5
            feet_all_contact = 5.0  
            feet_x_distance = -10.0
            roll_bias = 0
            # body_feet_distance_x = -5.0

            
    class costs(LeggedRobotCfg.costs):
        num_costs = 4
        class scales:
            pos_limit = 1.0
            torque_limit = 1.0
            dof_vel_limits = 1.0
            default_joint= 0.2

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            default_joint = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
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

class D1F12DOFFlatCfgPPO( LeggedRobotCfgPPO ):
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
        num_costs = 4

        teacher_act = True
        imi_flag = True
      
    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1f_12DOF_flat'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 1000
        max_iterations = 10000
        num_steps_per_env = 24
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = ''

class D1F12DOFFlat(D1F12DOF):
    def _reward_stand_still(self):
        cmd_small = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        deviation = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        reward = torch.exp(-deviation)
        return cmd_small * reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
 
    def _reward_orientation_roll(self):
        # Penalize Roll only (g_y^2)
        # 对应原来的 orientation_y，但可能给予较小的惩罚
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 1])
    
    def _reward_orientation_pitch(self):
        # Penalize Pitch only (g_x^2)
        # 防止屁股朝上或后仰，这对 Fixed ABAD 很重要
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 0])
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_feet_x_distance(self):
        # yaw 指令下，机体系前/后足的 x 间距低于 target 才惩罚，否则不惩罚
        gate = (torch.abs(self.commands[:, 2]) > 0.1).float()

        # 世界坐标 -> 机体系
        vec_l = self.feet_pos[:, 0, :] - self.feet_pos[:, 2, :]  # FL - RL
        vec_r = self.feet_pos[:, 1, :] - self.feet_pos[:, 3, :]  # FR - RR
        vec_l_body = quat_rotate_inverse(self.base_quat, vec_l)
        vec_r_body = quat_rotate_inverse(self.base_quat, vec_r)

        dist_l = torch.abs(vec_l_body[:, 0])
        dist_r = torch.abs(vec_r_body[:, 0])

        deficit_l = torch.clamp(self.cfg.rewards.feet_x_distance_target - dist_l, min=0.0)
        deficit_r = torch.clamp(self.cfg.rewards.feet_x_distance_target - dist_r, min=0.0)

        return 0.5 * (deficit_l + deficit_r) * gate

    def _reward_roll_bias(self):
        gate = torch.abs(self.commands[:, 2]>0.1).float() * torch.abs(self.commands[:, 0]>0.1).float()
        cmd_yaw = self.commands[:, 2]
        roll, _, _ = get_euler_xyz(self.base_quat)
        
        roll_max = 0.30      # 目标倾斜角，约 17 deg
        sigma_roll = 0.12    # 容忍范围
    
        # 固定角度：只要有转弯指令就偏到对应一侧
        roll_tgt = -roll_max * torch.sign(cmd_yaw)

        reward = torch.exp(-(roll - roll_tgt))
        return gate * reward


    
