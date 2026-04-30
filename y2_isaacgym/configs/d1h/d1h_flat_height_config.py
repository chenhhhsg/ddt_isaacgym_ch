from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi
from configs.d1h.d1h_flat_height_command import *

class D1HFlatHeightCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  2 + 1 + 4 + 1 + 1 + 8 + 8 + 8
        n_proprio = 37 # 3+3+3+4+8+8+8
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 8
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.5] # x,y,z [m]
        rot = [0, 0.0, 0.0, 1]  # x, y, z, w [quat]
        default_joint_angles = {
            'FL_hip_joint': 0,
            'FR_hip_joint': 0,

            'FL_thigh_joint': 0.8,
            'FR_thigh_joint': 0.8,

            'FL_calf_joint': -1.5,
            'FR_calf_joint': -1.5,

            'FL_foot_joint': 0,
            'FR_foot_joint': 0,
        }
        desired_feet_distance = 0.4

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 40.,
                     'thigh': 40.,
                     'calf': 40.,
                     'foot': 10.}  # [N*m/rad]
        damping = {'hip': 1.0,
                   'thigh': 1.0,
                   'calf': 1.0,
                   'foot': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.5
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.commands ):
        curriculum = True 
        max_curriculum = 1.0
        max_curriculum_x = 2.0
        max_curriculum_x_back = 1.0
        max_curriculum_y = 1.0
        max_curriculum_yaw = 1.0
        num_commands = 5  # default: lin_vel_x, lin_vel_y, ang_vel_yaw, heading , base_height(in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 5.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]
            base_height = [0.35, 0.45] # min max [m]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/d1h/urdf/robot.urdf'
        foot_name = "foot"
        name = "d1h"
        penalize_contacts_on = ["thigh", "calf"]
        penalize_contact_head_on = ["base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
  
    class rewards( LeggedRobotCfg.rewards ):

        only_positive_rewards = False
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_height_sigma = 0.05
        distance_sigma = 0.1  # distance reward = exp(-distance^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        max_contact_force = 500.  # forces above this value are penalized

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = -2e-5
            termination = -100.0
            tracking_lin_vel_x = 15.0
            tracking_lin_vel_y = 10.0
            tracking_ang_vel = 8.0
            lin_vel_z = -0.5
            orientation = -10.0
            ang_vel_xy = -0.05
            dof_thigh_vel = 0.0  #-0.05
            dof_acc = -2.5e-7
            # base_height = -10.0
            feet_air_time = 3.0
            collision = -2.0
            feet_stumble = 0.0
            action_rate = -0.01
            upward = 2.0
            # keep_still = -0.5
            tracking_base_height = 4.0
             
            # finetune
            collision_head = -5.0
            body_pos_to_feet_x = 0.5
            body_feet_distance_x = -0.2
            body_feet_distance_y = -0.8
            body_symmetry_y = 0.1
            body_symmetry_z = 0.3
        

    class costs(LeggedRobotCfg.costs):
        num_costs = 3
        class scales:
            pos_limit = 0.3
            torque_limit = 0.3
            dof_vel_limits = 0.3
            # hip_pos = 0.0
            # default_joint= 0.0

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            # hip_pos = 0.0
            # default_joint = 0.0

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        terrain_proportions = [0.7, 0.3, 0.0, 0.0, 0.0]
        slope_treshold = 1.0  # slopes above this threshold will be corrected to vertical surfaces
        slope = [0, 0.6]

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025


class D1HFlatHeightCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
        learning_rate = 1.e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy( LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        continue_from_last_std = True
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = []
        activation = 'elu'
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        tanh_encoder_output = False
        num_costs = 3

        teacher_act = True
        imi_flag = True

    class runner( LeggedRobotCfgPPO.runner ):
        run_name = ''
        experiment_name = 'd1h_flat_height'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 3000
        max_iterations = 6000
        num_steps_per_env = 24
        resume = False
        resume_path = ''


class D1HFlatHeight(D1HHeightCommand):

    ## 使用stand_still_vel + base_height 设置默认高度 or stand_still 设置
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.square(self.base_lin_vel[:, 2])
    

    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.commands[:,4]), 0, 1)
        base_height_sigma = 0.5 + 0.5 * torch.exp(-base_height_error / self.cfg.rewards.tracking_height_sigma)
        lin_vel_x_error = torch.clamp(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 0]))/(0.25+torch.abs(self.commands[:, 0]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_x_error/tracking_sigma) * base_height_sigma
        return reward
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.commands[:,4]), 0, 1)
        base_height_sigma = 0.5 + 0.5 * torch.exp(-base_height_error / self.cfg.rewards.tracking_height_sigma)
        lin_vel_y_error = torch.clamp(torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 1]))/(0.25+torch.abs(self.commands[:, 1]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_y_error/tracking_sigma) * base_height_sigma
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.commands[:,4]), 0, 1)
        base_height_sigma = 0.5 + 0.5 * torch.exp(-base_height_error / self.cfg.rewards.tracking_height_sigma)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 2]))/(0.25+torch.abs(self.commands[:, 2]))
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/tracking_sigma) * base_height_sigma
    
    def _reward_tracking_base_height(self):
        # Tracking of base_height_command (height_command)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.commands[:,4]), 0, 1)
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-base_height_error/self.cfg.rewards.tracking_height_sigma)
        return reward

    def _reward_feet_air_time(self):
        # Reward leg lifting mainly for side-walking commands.
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts)
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1)
        side_walk_cmd = (torch.abs(self.commands[:, 1]) > 0.1) & (torch.abs(self.commands[:, 1]) > torch.abs(self.commands[:, 0]))
        rew_airTime *= side_walk_cmd
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_stand_still(self):
        # 惩罚无命令下滑动
        cmd_still = ((torch.norm(self.commands[:, :2], dim=1) < 0.1)).float()
        base_motion = (torch.sum(torch.square(self.base_lin_vel), dim=1))
        return cmd_still * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * base_motion
