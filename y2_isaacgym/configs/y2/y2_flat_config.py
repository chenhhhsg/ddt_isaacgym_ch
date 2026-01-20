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

class Y2FlatCfg( LeggedRobotCfg ):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 12 + 12 + 12
        n_proprio = 48 #  3+ 3+ 3+ 3+ 12+ 12+ 12 
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 12  # fixed ABADOF joints
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'front_left_thigh_joint': -0.6,     # [rad]
            'front_right_thigh_joint': -0.6,     # [rad]
            'back_left_thigh_joint': 0.4,   # [rad]
            'back_right_thigh_joint': 0.4,   # [rad]

            'front_left_knee_joint': 1.6,   # [rad]
            'front_right_knee_joint': 1.6,  # [rad]
            'back_left_knee_joint': -1.6,    # [rad]
            'back_right_knee_joint': -1.6,    # [rad]

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
        file = '{ROOT_DIR}/resources/y2/urdf/robot.urdf'
        foot_name = "wheel"
        name = "Y2"
        penalize_contacts_on = ["thigh", "knee", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
        wheel_radius = 0.085

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
        base_height_target = 0.3  # y2的站立默认高度
        max_contact_force = 500.  # forces above this value are penalized
        feet_x_distance_target = 0.30
        feet_x_distance_sigma = 0.5
        roll_max = 15  # 转弯允许最大倾斜角
        # feet_y_distance_target = 0.474

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = 0.0#-2e-5
            termination = 0.0
            tracking_ang_vel = 10.0
            lin_vel_z = -3.0
            tracking_lin_vel_x = 10.0
            lin_vel_y = -5.0          # 惩罚y方向出现速度，不允许横移
            tracking_lin_vel = 0.0
            orientation = -0.0         # 基础惩罚 (·Combined Roll + Pitch)
            orientation_pitch = -3.0   # 强力惩罚 Pitch
            orientation_roll = -0.5    # 适度惩罚 Roll
            ang_vel_xy = -0.2
            feet_air_time = -4.0
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = -0.0
            dof_acc = -2.5e-7
            base_height = -2.0
            collision = -1.0
            # feet_stumble = -5.0
            action_rate = -0.01
            stand_still = -0.0
            stand_still_vel = -10.0
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            upward = 0.5
            feet_all_contact = -10.0  
            feet_x_distance = -0.0
            roll_turn_assist = 10.0

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
        # mesh_type ='trimesh'
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

class Y2FlatCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = 'y2_flat'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 2000
        max_iterations = 6000
        num_steps_per_env = 24
        load_run = -1
        checkpoint = -1
        resume = False
        resume_path = ''

class Y2Flat(Y2Command):

    def _reward_stand_still_vel(self):
        # 现在使用的是奖励
        cmd_small = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        base_lin_speed = torch.norm(self.base_lin_vel, dim=1)
        base_ang_speed = torch.norm(self.base_ang_vel, dim=1)
        leg_speed = torch.mean(torch.abs(self.dof_vel), dim=1)
        deviation = base_lin_speed + base_ang_speed + leg_speed
        reward = torch.clamp(torch.square(-deviation), -2, 2)
        # reward = torch.exp(-deviation)
        return cmd_small * reward

    def _reward_stand_still(self):
        cmd_small = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        deviation = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        reward = torch.clamp(torch.square(-deviation), -2, 2)
        # reward = torch.exp(-deviation)
        return cmd_small * reward

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
 
    # def _reward_orientation_roll(self):
    #     # Penalize Roll only (g_y^2)
    #     gate = torch.abs(self.commands[:, 2] > 0.1).float
    #     return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 1])

    def _reward_orientation_roll(self):
        limit = np.sin(np.deg2rad(self.cfg.rewards.roll_max + 5))  # 0.5
        excess = torch.clamp(torch.abs(self.projected_gravity[:, 1]) - limit, min=0.0)
        # print("limit:",limit)
        # print("excess:",excess.mean())
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.square(excess)

    def _reward_orientation_pitch(self):
        # Penalize Pitch only (g_x^2)
        # 防止屁股朝上或后仰，这对 Fixed ABAD 很重要
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 0])
    
    # def _reward_base_height(self):
    #     # Penalize base height away from target
    #     base_height = self._get_base_heights()
    #     # print('base_height:',base_height.mean())
    #     reward_height = torch.clamp(self.cfg.rewards.base_height_target - base_height, min=0.0)
    #     return torch.clamp(-self.projected_gravity[:,2],0,1)* reward_height

    def _reward_base_height(self):
        # Penalize base height deviation from target (both too low and too high)
        base_height = self._get_base_heights()
        target = self.cfg.rewards.base_height_target

        # deadzone: within ±3cm no penalty (tuneable)
        err = base_height - self.cfg.rewards.base_height_target
        deadzone = 0.03
        err = torch.where(torch.abs(err) < deadzone, torch.zeros_like(err), err)

        # squared error, clipped
        penalty = torch.clamp(err * err, max=0.20)  # cap to keep stable
        # keep your "upright gate"
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * penalty

    def _reward_roll_turn_assist(self):
        cmd_yaw = self.commands[:, 2]
        # Use sin(roll) directly to avoid 2π jumps
        roll_sin = torch.clamp(self.projected_gravity[:, 1], -1.0, 1.0)

        k_roll = 0.25 # sin_max / cmd_yaw 也可
        sin_max = float(np.sin(np.deg2rad(self.cfg.rewards.roll_max)))
        roll_sin_tgt = torch.clamp(k_roll * cmd_yaw, -sin_max, sin_max)

        sigma = 0.12
        r_roll = torch.exp(-((roll_sin - roll_sin_tgt) / sigma) ** 2)
        gate = (torch.abs(cmd_yaw ) > 0.1 ).float()
        return gate * r_roll

    # def _reward_turn_assist(self):
    #     gate = (torch.abs(self.commands[:, 2]) > 0.1 ).float()


    # def _reward_feet_all_contact(self):
    #     no_contact = self.contact_forces[:, self.feet_indices, 2] < 1.
    #     any_off_ground = (torch.sum(no_contact, dim=1) > 0).float()
    #     return torch.clamp(-self.projected_gravity[:,2],0,1) * any_off_ground

    def _reward_feet_all_contact(self):
        # count how many feet are off ground
        off_ground = (self.contact_forces[:, self.feet_indices, 2] < 1.0).float()
        off_count = torch.sum(off_ground, dim=1)  # 0~4
        # penalty proportional to how many are off
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * off_count

    def _reward_feet_air_time(self):
        """
        Penalize long airtime (prevents "lift one leg forever" or jumping).
        We update self.feet_air_time here because current code only resets it.
        """
        # contact_filt: [N,4] bool (already computed in post_physics_step) :contentReference[oaicite:4]{index=4}
        contact = self.contact_filt

        # integrate airtime
        self.feet_air_time += (~contact).float() * self.dt
        self.feet_air_time *= (~contact).float()  # reset to 0 when contact is True

        # allow short airtime without penalty (tuneable)
        free_time = 0.12  # 120ms free swing
        excess = torch.clamp(self.feet_air_time - free_time, min=0.0)
        # sum over 4 feet
        penalty = torch.sum(excess, dim=1)

        # gate with upright as you already do
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * penalty

    def _reward_lin_vel_y(self):
        vy = self.base_lin_vel[:, 1]
        return vy * vy