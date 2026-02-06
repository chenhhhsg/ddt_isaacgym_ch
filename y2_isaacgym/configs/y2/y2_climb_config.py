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
        pos = [0.0, 0.0, 0.4] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'front_left_thigh_joint': -0.6,     # [rad]
            'front_right_thigh_joint': -0.6,     # [rad]
            'back_left_thigh_joint': 0.6,   # [rad]
            'back_right_thigh_joint': 0.6,   # [rad]

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
        hip_scale_reduction = 0.5
        use_filter = True

    class commands( LeggedRobotCfg.control ):
        curriculum = True 
        yaw_curriculum = False
        max_curriculum = 2.0
        num_commands = 4  # default: lin_vel_x, ang_vel_yaw, heading (in heading mode ang_vel_yaw is recomputed from heading error)
        resampling_time = 10.  # time before command are changed[s]
        heading_command = False  # if true: compute ang vel command from heading error
        global_reference = False

        class ranges:
            lin_vel_x = [-1.0, 1.0]   # m/s
            lin_vel_y = [0, 0]
            ang_vel_yaw = [-0.8, 0.8]  # rad/s
            heading = [-3.14, 3.14]

    class asset( LeggedRobotCfg.asset ):
        file = '{ROOT_DIR}/resources/y2/urdf/robot.urdf'
        foot_name = "wheel"
        name = "y2"
        penalize_contacts_on = ["thigh", "knee", "base"]
        terminate_after_contacts_on = []     # 
        self_collisions = 0 # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False
        wheel_radius = 0.086  # needed for velocity estimation from wheel joint movement

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
        base_height_target = 0.31  # y2的站立默认高度
        max_contact_force = 500.  # forces above this value are penalized
        roll_max = 30  # 转弯允许最大倾斜角
        turn_assist_k = 1.0
        turn_assist_sigma = 0.5

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = 0.0
            powers = -5e-5 #-2e-5
            termination = 0.0
            tracking_ang_vel = 10.0
            lin_vel_z = -0.0
            tracking_lin_vel_x = 30.0
            tracking_lin_vel = 0.0
            orientation = -0.5
            orientation_roll = -5.0   # 没有髋关节，太容易侧翻了，roll惩罚给大一点
            ang_vel_xy = -0.05
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_acc = -2.5e-7
            action_smoothness = -0.01

            lin_vel_y = -5.0
            base_height = -100.0
            collision = -3.0
            action_rate = -0.05
            stand_still = 0.0
            stand_still_vel = -20.0
            upward = 0.0
            tilt = -0.0  # 翻身直接结束回合
            pitch_velocity = -0.2 # 惩罚俯仰角速度过大
            roll_turn_assist = 10.0
            turn_assist = 10.0

            stumble = -1.0
            feet_air_time = 0.0
            feet_all_contact = 2.0
            climb_assist = -1.0
            feet_swing_height = 2.0
            forward_progress = 0.0
            rear_knee_tuck = 5.0

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
        # terrain_proportions = [0.2, 0.0, 0.8, 0.0, 0.0]
        terrain_proportions = [0.0, 0.0, 1.0, 0.0, 0.0]
        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]
        step_height = [0.03,0.18] # max height of stairs, stepping stones etc
        step_width = 0.31    # depth of stairs
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
        save_interval = 1000
        max_iterations = 6000
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

    def _reward_orientation_roll(self):
        limit = np.sin(np.deg2rad(self.cfg.rewards.roll_max + 5))  # 0.5
        excess = torch.clamp(torch.abs(self.projected_gravity[:, 1]) - limit, min=0.0)
        # print("limit:",limit)
        # print("excess:",excess.mean())
        return torch.clamp(-self.projected_gravity[:, 2], 0, 1) * torch.square(excess)

    def _reward_pitch_velocity(self):
        pitch_angular_vel = self.base_ang_vel[:, 1]
        # 使用平方惩罚，当俯仰角速度超过阈值时给予更大的惩罚
        pitch_vel_penalty = torch.square(pitch_angular_vel)
        return pitch_vel_penalty
    
    def _reward_base_height(self):
        # Penalize base height too low
        base_height = self._get_base_heights()
        # reward = torch.clamp(self.cfg.rewards.base_height_target - base_height, min=0.0)
        # contact = (self.contact_forces[:, self.feet_indices, 2] > 1.).float()
        # stable = (torch.sum(contact, dim=1) >= 3).float()
        # print("base_height", base_height)
        reward = torch.square(base_height - self.cfg.rewards.base_height_target)
        return reward

    def _reward_tracking_lin_vel_x(self):
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        lin_vel_x_error = torch.clip(lin_vel_x_error,0,1)
        reward = torch.exp(-lin_vel_x_error/self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_stand_still_vel(self):
        # 令机器人在无命令时，尽量静止（否则会通过起跳去不断获取高度奖励）
        cmd_small = (torch.norm(self.commands[:, :3], dim=1) < 0.2).float()
        base_lin_speed = torch.norm(self.base_lin_vel, dim=1)
        leg_speed = torch.mean(torch.abs(self.dof_vel), dim=1)
        deviation = base_lin_speed + leg_speed
        reward = torch.clamp(torch.square(-deviation), -2, 2)
        # reward = torch.exp(-deviation)
        return cmd_small * reward

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_feet_all_contact(self):
        # require wheels to stay on ground when xy collision is small
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        xy_force = torch.norm(self.contact_forces[:, self.feet_indices, 0:2], dim=-1)
        xy_safe = xy_force < 20.0
        contact_safe = contact & xy_safe
        off_safe = (~contact) & xy_safe
        reward = torch.sum(contact_safe.float(), dim=1) - torch.sum(off_safe.float(), dim=1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)* 0.25 * reward

    def _reward_roll_turn_assist(self):
        cmd_yaw = self.commands[:, 2]
        # Use sin(roll) directly to avoid 2π jumps
        roll_sin = torch.clamp(self.projected_gravity[:, 1], -1.0, 1.0)
        # print("roll_sin:", roll_sin)
        k_roll = 0.50 # sin_max / cmd_yaw 也可
        sin_max = float(np.sin(np.deg2rad(self.cfg.rewards.roll_max)))
        cmd_x = self.commands[:, 0]
        dir_sign = torch.sign(cmd_x)
        dir_sign = torch.where(dir_sign == 0.0, torch.ones_like(dir_sign), dir_sign)
        roll_sin_tgt = torch.clamp(k_roll * cmd_yaw * dir_sign, -sin_max, sin_max)
        sigma = 0.12
        r_roll = torch.exp(-((roll_sin - roll_sin_tgt) / sigma) ** 2)
        gate = (torch.abs(cmd_yaw ) > 0.1 ).float()
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
        # print("base_lin_vel:",self.base_lin_vel[:,0])
        # print("wheel_left_lin_vel：",v_left,"wheel_right_lin_vel：",v_right)
        # print("wheel_diff_(right-left)=", diff)
        cmd_x = self.commands[:, 0]
        target = self.cfg.rewards.turn_assist_k * cmd_yaw
        reward = torch.exp(-((diff - target) / self.cfg.rewards.turn_assist_sigma) ** 2)
        return gate * reward

    def _reward_lin_vel_y(self):
        # 惩罚无yaw命令时的侧移(走直线)
        gate = (torch.abs( self.commands[:, 2] < 0.1 )).float()
        base_vel_y = self.base_lin_vel[:, 1]
        return gate * torch.square(base_vel_y)

    def _reward_feet_swing_height(self):
        # 鼓励摆相腿抬高
        swing = (self.contact_forces[:, self.feet_indices, 2] <= 1.).float()
        gate = (torch.abs(self.commands[:, 0]) > 0.1).float()
        # 离地高度（相对地形），动态适应台阶高度
        clearance = self.feet_heights  # 已在 post_physics_step 里更新
        # print('clearance', clearance)
        min_clear = 0.02  # 最小离地高度，先给 5cm
        max_clear = 0.07  # 饱和上限，避免无限抬腿

        reward = torch.sum(torch.clamp(clearance - min_clear, min=0.0, max=max_clear - min_clear) * swing, dim=1)
        return reward * gate

    def _reward_climb_assist(self):
        # 惩罚v_z < 0
        penalize_falling = torch.square(torch.clamp(self.base_lin_vel[:, 2], max=0.0))
        return torch.clamp(-self.projected_gravity[:,2],0,1) * penalize_falling

    def _reward_forward_progress(self):
        # Reward forward progress to discourage backing up when cmd_x > 0 
        # 主要想和 collision reward 配合使用，避免机器人为了减少碰撞而后退
        gate = (self.commands[:, 0] > 0.1).float()
        forward_vel = torch.clamp(self.base_lin_vel[:, 0], min=0.0)
        return forward_vel * gate

    def _reward_rear_knee_tuck(self):
        # 当前后轮出现高度差后，通过收紧后腿，令膝盖远离台阶，轮子先触碰到台阶
        # contact reorder -> [FL, FR, BL, BR]
        contact_z = self.contact_forces[:, self.feet_indices, 2][:, [2, 3, 0, 1]]
        swing = (contact_z <= 1.).float()
        swing_rear = swing[:, 2:4]  # BL, BR

        # gate: only active when front/rear terrain height differs (stairs-like)
        # feet_pos/feet_heights are in feet_indices order -> reorder to [FL, FR, BL, BR]
        terrain_z = (self.feet_pos[:, :, 2] - self.feet_heights)[:, [2, 3, 0, 1]]
        front_mean = 0.5 * (terrain_z[:, 0] + terrain_z[:, 1])
        rear_mean = 0.5 * (terrain_z[:, 2] + terrain_z[:, 3])
        stair_gate = (torch.abs(front_mean - rear_mean) > 0.03).float()  # 前后腿出现高度差时才激活

        # rear knee tuck toward lower limit
        knee = self.dof_pos[:, self.knee_joint_indices]   # [FL,FR,BL,BR]
        knee_rear = knee[:, 2:4]

        knee_lower = -1.92
        margin = 0.05
        knee_target = knee_lower + margin  # 留一点冗余，不让膝关节完全贴近极限位置
        sigma_knee = 0.18
        r_knee = torch.exp(-((knee_rear - knee_target) / sigma_knee) ** 2)

        # rear thigh in (target a negative "tucked" region, avoid slamming to -2.36)
        thigh = self.dof_pos[:, self.thigh_joint_indices]  # [FL,FR,BL,BR]
        thigh_rear = thigh[:, 2:4]

        # choose a practical target; start here, tune after watching behavior
        thigh_target = -0.9        # rad  经过URDF测定，当thigh接近极限的位置，小于0就可以先让轮子触碰到台阶
        sigma_thigh = 0.45         # tolerance
        r_thigh = torch.exp(-((thigh_rear - thigh_target) / sigma_thigh) ** 2)

        # combine
        w_knee, w_thigh = 0.8, 1.0
        r = w_knee * r_knee + w_thigh * r_thigh

        return stair_gate * torch.sum(r * swing_rear, dim=1)

    def _reward_feet_air_time(self):
        # Reward long steps
        # Need to filter the contacts because the contact reporting of PhysX is unreliable on meshes
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1) # reward only on first contact with the ground
        #rew_airTime = torch.sum((self.feet_air_time - 0.3) * first_contact, dim=1)
        #rew_airTime = torch.sum((self.feet_air_time - 0.2) * first_contact, dim=1)
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 #no reward for zero command
        self.feet_air_time *= ~contact_filt
        return rew_airTime
