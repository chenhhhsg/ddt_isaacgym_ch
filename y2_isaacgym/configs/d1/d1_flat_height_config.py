from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi
from configs.d1.d1_flat_height_command import *

class D1FlatHeightCfg ( LeggedRobotCfg ): 
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent =  4 + 1 + 4 + 1 + 1 + 16 + 16 + 16
        n_proprio = 60 #  3+ 3+ 3+ 3+ 16+ 16+ 16 
        history_len = 10
        num_observations = n_proprio + n_scan + history_len*n_proprio + n_priv_latent
        num_actions = 16  # fixed ABADOF joints
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.35] # x,y,z [m]
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
        # yaw_curriculum = False
        # heading_curriculum = False
        max_curriculum = 3.0
        num_commands = 5  # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-0.5, 0.5]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            lin_vel_z = [0.25, 0.5]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]
 
    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/d1/urdf/robot.urdf'
        foot_name = "wheel"
        name = "D1"
        penalize_contacts_on = ["thigh", "knee", "base"]
        terminate_after_contacts_on = []
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
        wheel_radius = 0.086

    class domain_rand:
        randomize_friction = True
        friction_range = [0.2, 2.75]
        randomize_restitution = True
        restitution_range = [0.0,1.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]  # y2有负载10kg的需求，随机化加入8~15kg质量
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]  # 重心往上偏移0~5cm
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
        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
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
            tracking_lin_vel = 2.0
            tracking_ang_vel = 1.0
            lin_vel_z = -2.0
            orientation = -1.0
            orientation_y = -10.0
            ang_vel_xy = -0.05
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            base_height = -1.0
            feet_air_time = 0.0
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.01
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            # hip_pos = 0.5
            upward = 0.5
            # feet_all_contact = -0.5
            # feet_contact_forces = -0.1
            # joint_power=-2e-5
            # powers_dist =-1.0e-5
        

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

class D1FlatHeightCfgPPO( LeggedRobotCfgPPO ):
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
        experiment_name = 'y2_flat'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        save_interval = 1000
        max_iterations = 6000
        num_steps_per_env = 24
        load_run = -1
        checkpoint = "logs/y2_flat/Feb04_10-28-44_/model_3000.pt"
        resume = False
        resume_path = ''

class D1FlatHeight(D1Command):

    ## 使用stand_still_vel + base_height 设置默认高度 or stand_still 设置
    def _reward_stand_still_vel(self):
        # 现在使用的是奖励
        gate = (torch.norm(self.commands[:, :3], dim=1) < 0.1).float()
        base_lin_speed = torch.norm(self.base_lin_vel, dim=1)
        base_ang_speed = torch.norm(self.base_ang_vel, dim=1)
        leg_speed = torch.mean(torch.abs(self.dof_vel), dim=1)
        deviation = base_lin_speed + base_ang_speed + leg_speed
        reward = torch.clamp(torch.square(-deviation), -2, 2)
        # reward = torch.exp(-deviation)
        return gate * reward

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :3], dim=1) < 0.1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_roll(self):
        limit = np.sin(np.deg2rad(self.cfg.rewards.roll_max + 2))  # 0.5
        excess = torch.clamp(torch.abs(self.projected_gravity[:, 1]) - limit, min=0.0)
        # print("limit:",limit)
        # print("excess:",excess.mean())
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.square(excess)

    def _reward_orientation_pitch(self):
        # Penalize Pitch only (g_x^2)
        # 防止屁股朝上或后仰，这对 Fixed ABAD 很重要
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 0])

    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        # print("base_height", base_height)
        gate = (torch.abs(self.commands[:, 2]) < 0.1).float()
        return gate * torch.abs(base_height - self.cfg.rewards.base_height_target)

    def _reward_roll_turn_assist(self):
        base_yaw = self.base_ang_vel[:, 2]
        cmd_yaw = self.commands[:, 2]
        # Use sin(roll) directly to avoid 2π jumps
        roll_sin = torch.clamp(self.projected_gravity[:, 1], -1.0, 1.0)
        # print("roll_sin:", roll_sin)
        k_roll = 0.9 # sin_max / cmd_yaw 也可
        sin_max = float(np.sin(np.deg2rad(self.cfg.rewards.roll_max)))
        cmd_x = self.commands[:, 0]
        dir_sign = torch.sign(cmd_x)
        dir_sign = torch.where(dir_sign == 0.0, torch.ones_like(dir_sign), dir_sign)
        roll_sin_tgt = torch.clamp(k_roll * cmd_yaw * dir_sign, -sin_max, sin_max)
        sigma = 0.20
        r_roll = torch.exp(-((roll_sin - roll_sin_tgt) / sigma) ** 2)
        gate = (torch.abs(cmd_yaw) > 0.1 ).float()
        return gate * r_roll

    def _reward_turn_assist(self):
        cmd_yaw = self.commands[:, 2]
        gate = (torch.abs(cmd_yaw) > 0.1).float()

        w = self.dof_vel[:, self.foot_joint_indices]
        v = w * self.cfg.asset.wheel_radius
        # print("dof_vel:", w)
        v_left = 0.5 * (v[:, 0] + v[:, 2])
        v_right = 0.5 * (v[:, 1] + v[:, 3])
        diff = v_right - v_left
        # print("机身速度",self.base_lin_vel[:,0])
        # print("左边轮速：",v_left,"右边轮速：",v_right)
        # print("右减左轮速差为:", diff)
        target = self.cfg.rewards.turn_assist_k * cmd_yaw
        reward = torch.exp(-((diff - target) / self.cfg.rewards.turn_assist_sigma) ** 2)
        return gate * reward

    def _reward_feet_all_contact(self):
        # penalize if any foot is off the ground
        off_ground = self.contact_forces[:, self.feet_indices, 2] < 1.0
        any_off = torch.sum(off_ground, dim=1).float()
        return any_off

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

    def _reward_inner_wheels_close(self):
        # turning: encourage inner-side wheels to get closer but stay above min distance
        cmd_yaw = self.commands[:, 2]
        cmd_x = self.commands[:, 0]
        gate = (torch.abs(cmd_yaw) > 0.1).float()

        vec_l = self.feet_pos[:, 0, :] - self.feet_pos[:, 2, :]  # FL - RL
        vec_r = self.feet_pos[:, 1, :] - self.feet_pos[:, 3, :]  # FR - RR
        vec_l_body = quat_rotate_inverse(self.base_quat, vec_l)
        vec_r_body = quat_rotate_inverse(self.base_quat, vec_r)

        dist_l = torch.abs(vec_l_body[:, 0])
        dist_r = torch.abs(vec_r_body[:, 0])
        # print("dist_r:", dist_r)
        dir_flag = torch.where(torch.abs(cmd_x) > 0.1, cmd_yaw * cmd_x, cmd_yaw) # 后退时候dir_flag与cmd_yaw反号
        inner_dist = torch.where(dir_flag >= 0.0, dist_l, dist_r)

        target = self.cfg.rewards.inner_wheel_dist_target  # 0.2
        sigma = self.cfg.rewards.inner_wheel_dist_sigma
        min_dist = self.cfg.rewards.inner_wheel_dist_min

        reward = torch.exp(-((inner_dist - target) / sigma) ** 2)
        too_close = (inner_dist < min_dist).float()
        penalty = torch.clamp(min_dist - inner_dist, min=0.0) / sigma
        return gate * (reward - too_close * penalty)

    # def _reward_pivot_turn(self):
    #     # in-place turn: lift outer side legs, keep inner side in contact
    #     cmd_x = self.commands[:, 0]
    #     cmd_yaw = self.commands[:, 2]
    #     gate = (torch.abs(cmd_yaw) > 0.1).float() * (torch.abs(cmd_x) < 0.1).float()

    #     contact = self.contact_forces[:, self.feet_indices, 2] > 1.0
    #     # feet order: FL, FR, BL, BR
    #     # fixed diagonal support: keep FL + BR, lift FR + BL
    #     fl = contact[:, 0].float()
    #     fr = contact[:, 1].float()
    #     bl = contact[:, 2].float()
    #     br = contact[:, 3].float()
    #     keep_on = fl * br
    #     lift_off = (1.0 - fr) * (1.0 - bl)
    #     reward = keep_on * lift_off
    #     return gate * reward

    def _reward_feet_pos_sym(self):
        # feet positions in body frame
        base_pos = self.root_states[:, :3]
        feet_world = self.feet_pos - base_pos.unsqueeze(1)
        feet_world_flat = feet_world.reshape(-1, 3)
        base_quat_rep = self.base_quat.repeat_interleave(feet_world.shape[1], dim=0)
        feet_body = quat_rotate_inverse(base_quat_rep, feet_world_flat).reshape(feet_world.shape)

        # left/right symmetry for front and rear
        x_err_front = feet_body[:, 0, 0] - feet_body[:, 1, 0]  # FL.x - FR.x

        x_err_back = feet_body[:, 2, 0] - feet_body[:, 3, 0]   # BL.x - BR.x

        diff = x_err_front**2 + x_err_back**2 
        # print("x_err_front:", x_err_front.mean())
        # print("x_err_back:", x_err_back.mean())
        # weaken symmetry during turning
        turning = (torch.abs(self.commands[:, 2]) > 0.1).float()
        gate = 1.0 - 1.0 * turning  # keep 30% weight when turning
        reward = torch.exp(-diff / 0.25)
        return torch.clamp(-self.projected_gravity[:,2],0,1) * reward * gate
