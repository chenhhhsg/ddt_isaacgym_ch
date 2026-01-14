from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi
from configs.y2.y2_command import *

class Y2ClimbCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 12 + 12 + 12
        n_proprio = 48 #
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 12  # fixed ABADOF joints
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.6] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'front_left_thigh_joint': -0.2,     # [rad]
            'front_right_thigh_joint': -0.2,     # [rad]
            'back_left_thigh_joint': 0.2,   # [rad]
            'back_right_thigh_joint': 0.2,   # [rad]

            'front_left_knee_joint': 1.5,   # [rad]
            'front_right_knee_joint': 1.5,  # [rad]
            'back_left_knee_joint': -1.5,    # [rad]
            'back_right_knee_joint': -1.5,    # [rad]

            'front_left_wheel_joint': 0.0,
            'front_right_wheel_joint': 0.0,
            'back_left_wheel_joint': 0.0,
            'back_right_wheel_joint': 0.0,
        }

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {
                    'thigh': 40.,
                     'knee': 40.,
                     'wheel': 10.}  # [N*m/rad]
        damping = {
                   'thigh': 1.0,
                   'knee': 1.0,
                   'wheel': 0.5}     #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
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
        file = '{ROOT_DIR}/resources/y2/urdf/robot.urdf'
        foot_name = "wheel"
        name = "y2"
        penalize_contacts_on = ["thigh", "knee", "base"]
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
            tracking_ang_vel = 20.0
            lin_vel_z = 0.0
            tracking_lin_vel_x = 30.0
            tracking_lin_vel = 0.0
            orientation = -1.0
            orientation_roll = -2.0   
            ang_vel_xy = -0.05

            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -2.0
            collision = -1.0
            action_rate = -0.01
            stand_still = 20.0
            upward = 0.5
            # turn_yaw = 10.0
            tilt = -0.0 
            stumble = -1.0
            feet_contact_forces = -0.0
            feet_air_time = 0.0

            feet_all_contact = 1.0

            pitch_velocity = -0.2 # 惩罚俯仰角速度过大
            feet_traction = 0.0 
            climb_assist = 1.0
            feet_swing_height = 1.0
            
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
        mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        # mesh_type ='plane'
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        terrain_proportions = [0.2, 0.0, 0.8, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.3, 0.1, 0.1, 0.3]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        slope_treshold = 1.0  # slopes above this threshold will be corrected to vertical surfaces
        slope = [0, 0.4]

class Y2ClimbCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = 'y2_climb'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 5000
        max_iterations = 40000
        num_steps_per_env = 24
        # load_run = -1
        # checkpoint = -1
        # resume = True
        # resume_path = 'logs/d1f_12DOF_climb/Jan09_21-57-06_/model_40000.pt'

class Y2Climb(Y2Command):
    def _reward_tilt(self):
        gravity_z = self.projected_gravity[:,2]
        gravity_z_too_big = gravity_z > 0.0
        return 1.*gravity_z_too_big
    
    def _reward_stumble(self):
        # Penalize feet hitting vertical surfaces
        return torch.any(torch.norm(self.contact_forces[:, self.feet_indices, :2], dim=2) >\
             5 *torch.abs(self.contact_forces[:, self.feet_indices, 2]), dim=1)
    
    def _reward_feet_contact_forces(self):
        # penalize high contact forces
        # print(torch.norm(self.contact_forces[0, self.feet_indices, :], dim=-1))
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 400), dim=1)
    
    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) # reward only on first contact with the ground
        #rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)
        #rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_pitch_velocity(self):
        pitch_angular_vel = self.base_ang_vel[:, 1]
        # 使用平方惩罚，当俯仰角速度超过阈值时给予更大的惩罚
        pitch_vel_penalty = torch.square(pitch_angular_vel)
        return pitch_vel_penalty
    
    def _reward_feet_traction(self):
        gate = torch.abs( self.commands[: , :2] > 0.1 ).float()
        contact = (self.contact_forces[:, self.feet_indices, 2] > 1.).float()
        # Get xy velocity of the feet (we want this to be 0 when in contact)
        feet_vel_xy = self.rigid_body_states[:, self.feet_indices, 7:9]
        # Calculate penalty: sum of norm of velocities for contacting feet
        return torch.sum(torch.norm(feet_vel_xy, dim=-1) * contact, dim=1) * gate

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_climb_assist(self):
        # 惩罚v_z < 0, 忽略 v_z > 0
        penalize_falling = torch.square(torch.clamp(self.base_lin_vel[:, 2], max=0.0))
        return torch.clamp(-self.projected_gravity[:,2],0,1) * penalize_falling

    def _reward_orientation_roll(self):
        # Penalize Roll only (g_y^2)
        # 对应原来的 orientation_y，但可能给予较小的惩罚
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 1])

    def _reward_feet_swing_height(self):
        swing = (self.contact_forces[:, self.feet_indices, 2] <= 1.).float()

        # 离地高度（相对地形），动态适应台阶高度
        clearance = self.feet_heights  # 已在 post_physics_step 里更新

        min_clear = 0.12  # 最小离地高度，先给 12cm
        max_clear = 0.30  # 饱和上限，避免无限抬腿

        reward = torch.sum(
            torch.clamp(clearance - min_clear, min=0.0, max=max_clear - min_clear) * swing,
            dim=1
        )
        return reward

    
    # def _reward_feet_all_contact(self):
    #     # 奖励保持足够支撑：移动时至少 2 足，静止时 3 足以上
    #     contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
    #     support = torch.sum(contact, dim=1)

    #     cmd_mag = torch.norm(self.commands[:, :2], dim=1) + torch.abs(self.commands[:, 2])
    #     target_support = torch.where(cmd_mag > 0.2, torch.tensor(2.0, device=self.device), torch.tensor(3.0, device=self.device))

    #     deficit = torch.clamp(target_support - support, min=0.0)
    #     sigma = 0.5
    #     return torch.exp(-torch.square(deficit) / (sigma * sigma + 1e-6))
