from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi
from configs.d1h.d1h_height_command import *

class D1HSlopeHeightCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  1 + 2 + 1 + 4 + 1 + 1 + 8 + 8 + 8 # 34
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
        max_curriculum = 2.0
        max_curriculum_x = 2.0
        max_curriculum_x_back = 1.0
        max_curriculum_y = 0.0
        max_curriculum_yaw = 1.0
        num_commands = 5  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, lin_vel_z
        resampling_time = 5.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False
        height_goal_prob = 0.3
        height_goal_kp = 1.0
        height_goal_tolerance = 0.02
        # Command category proportions: [x_only, x+yaw, yaw_only, stand_still].
        commands_proportion = [0.4, 0.2, 0.2, 0.2]
        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-1.0, 1.0]  # min max [rad/s]
            heading = [-3.14, 3.14]
            lin_vel_z = [-0.1, 0.1]  # min max [m/s]

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
        wheel_radius = 0.085
  
    class rewards( LeggedRobotCfg.rewards ):

        only_positive_rewards = False
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        tracking_height_sigma = 0.05
        distance_sigma = 0.1  # distance reward = exp(-distance^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        height_target_min = 0.25
        height_target_max = 0.5
        stand_height_tolerance = 0.03
        height_change_vz_threshold = 0.01
        height_change_xdist_sigma = 0.02
        head_feet_x_dist_target = 0.0
        max_contact_force = 500.  # forces above this value are penalized

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = -2e-5
            termination = -100.0
            tracking_lin_vel = 0.0
            feet_air_time = 0.0
            tracking_lin_vel_x = 15.0
            tracking_lin_vel_y = 10.0
            tracking_ang_vel = 8.0
            tracking_height_velocity = 1.0
            lin_vel_z = -0.0
            orientation = -5.0
            ang_vel_xy = -0.05
            dof_thigh_vel = -0.05
            dof_acc = -2.5e-7
            # base_height = -10.0
            collision = -5.0
            feet_stumble = 0.0
            action_rate = -0.01
            upward = 2.0
            # keep_still = -0.5
            tracking_base_height = 8.0

            stand_still_wheel = -5.0
            stand_still_base = -5.0
            wheel_vel_diff = -4.0

            # finetune
            collision_head = -5.0
            body_pos_to_feet_x = 0.5
            body_feet_distance_x = -1.0
            body_feet_distance_y = -3.0
            body_symmetry_y = 0.1
            body_symmetry_z = 0.3
            no_jump = -1.0
            collision_hard = -10.0
        

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
        # mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        terrain_proportions = [1.0, 0.0, 0.0, 0.0, 0.0]
        slope_treshold = 1.0  # slopes above this threshold will be corrected to vertical surfaces
        slope = [0, 0.4]

    class sim(LeggedRobotCfg.sim):
        dt = 0.0025


class D1HSlopeHeightCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = 'd1h_slope_height'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 5000
        max_iterations = 10000
        num_steps_per_env = 24
        resume = False
        resume_path = ''


class D1HSlopeHeight(D1HHeightCommand):

    ## 使用stand_still_vel + base_height 设置默认高度 or stand_still 设置
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity; gate off when actively tracking height
        if self.cfg.commands.num_commands >= 5:
            actual_vz = torch.abs(self.base_height - self.last_base_height) / self.dt
            # Hard gate: no vertical-velocity penalty while base height is changing (height tracking).
            mask = (actual_vz < 1e-2).float()
        else:
            mask = 1.0
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * mask * torch.square(self.base_lin_vel[:, 2])
    
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.target_height), 0, 1)
        base_height_sigma = 0.8 + 0.2 * torch.exp(-base_height_error / self.cfg.rewards.tracking_height_sigma)
        lin_vel_x_error = torch.clamp(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 0]))/(0.25+torch.abs(self.commands[:, 0]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_x_error/0.2) * base_height_sigma
        return reward
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.target_height), 0, 1)
        base_height_sigma = 0.8 + 0.2 * torch.exp(-base_height_error / self.cfg.rewards.tracking_height_sigma)
        lin_vel_y_error = torch.clamp(torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 1]))/(0.25+torch.abs(self.commands[:, 1]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_y_error/0.2) * base_height_sigma
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.target_height), 0, 1)
        base_height_sigma = 0.8 + 0.2 * torch.exp(-base_height_error / self.cfg.rewards.tracking_height_sigma)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 2]))/(0.25+torch.abs(self.commands[:, 2]))
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/0.2) * base_height_sigma
    
    def _reward_tracking_base_height(self):
        base_height_error = torch.clamp(torch.square(self._get_base_heights() - self.target_height), 0, 1)
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-base_height_error/self.cfg.rewards.tracking_height_sigma)
        return reward

    def _reward_tracking_height_velocity(self):
        # 辅助跟踪速度奖励函数
        lin_vel_z_cmd = (self.target_height - self.last_target_height) / self.dt
        height_vel_error = torch.square(lin_vel_z_cmd - self.base_lin_vel[:, 2])
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.exp( -height_vel_error / 1e-4)

    def _reward_stand_still_wheel(self):
        cmd_still_xy = getattr(self, "is_stand_cmd", torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        cmd_still_yaw = (torch.abs(self.commands[:, 2]) < 0.1).float()
        height_tol = getattr(self.cfg.commands, "height_goal_tolerance", 0.02)
        height_hold = (
            (torch.abs(self.height_goal - self.target_height) < height_tol)
            & (torch.abs(self.target_height - self.last_target_height) < height_tol)
        ).float()
        cmd_still_height = (torch.abs(self.commands[:, 4]) < 0.02).float()
        cmd_still_gate = cmd_still_xy * height_hold * cmd_still_height * cmd_still_yaw
        wheel_vel = torch.sum(torch.square(self.dof_vel[:, self.foot_joint_indices]), dim=1)
        return cmd_still_gate * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.clamp(wheel_vel / 0.5, 0.0, 1.0)

    def _reward_stand_still_base(self):
        # 惩罚无命令下滑动
        cmd_still_xy = getattr(self, "is_stand_cmd", torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        cmd_still_yaw = (torch.abs(self.commands[:, 2]) < 0.1).float()
        height_tol = getattr(self.cfg.commands, "height_goal_tolerance", 0.02)
        height_hold = (
            (torch.abs(self.height_goal - self.target_height) < height_tol)
            & (torch.abs(self.target_height - self.last_target_height) < height_tol)
        ).float()
        cmd_still_height = (torch.abs(self.commands[:, 4]) < 0.02).float()
        cmd_still_gate = cmd_still_xy * height_hold * cmd_still_height * cmd_still_yaw
        base_lin_motion = torch.sum(torch.square(self.base_lin_vel[:, :2]), dim=1)
        base_yaw_motion = torch.square(self.base_ang_vel[:, 2])
        base_motion = base_lin_motion +  base_yaw_motion
        return cmd_still_gate * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.clamp(base_motion / 0.1, 0.0, 1.0)

    def _reward_wheel_vel_diff(self):
        cmd_y_zero = (torch.abs(self.commands[:, 1]) < 0.1).float()
        cmd_yaw_zero = (torch.abs(self.commands[:, 2]) < 0.1).float()
        cmd_height_zero = (torch.abs(self.commands[:, 4]) < 0.02).float()
        wheel_diff_gate = cmd_y_zero * cmd_yaw_zero * cmd_height_zero
        wheel_vel = self.dof_vel[:, self.foot_joint_indices]
        wheel_vel_diff = torch.square(wheel_vel[:, 0] - wheel_vel[:, 1])
        return wheel_diff_gate * torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.clamp(wheel_vel_diff / 0.1, 0.0, 1.0)

    def _reward_no_jump(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 10.
        airborne = torch.sum(contacts.float(), dim=1) == 0
        moving_cmd = torch.norm(self.commands[:, :2], dim=1) > 0.1
        return airborne.float() * moving_cmd.float()

    def _reward_collision_hard(self):
        contact_force = torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1)
        return torch.sum((contact_force > 100.).float(), dim=1)
