from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
# config
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil
import numpy as np

class D1FClimb(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.hip_joint_indices = [0, 4, 8, 12]
        self.foot_joint_indices = [3, 7, 11, 15]
        # for rewards using delta-position
        self._last_base_z = self.root_states[:, 2].clone()
        self._last_base_pos = self.root_states[:, 0:3].clone()

    
    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        init_z = float(self.cfg.init_state.pos[2])
        # base position
        if self.custom_origins:
            self.cfg.init_state.pos
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            # self.root_states[env_ids, 2] = init_z + torch_rand_float(0., 0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            # self.root_states[env_ids, 2] = init_z + torch_rand_float(0., 0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        # print("init_z:", self.cfg.init_state.pos[2], "root_z:", self.root_states[env_ids[:4], 2])
        # base rotation
        random_roll = torch_rand_float(0, 0, (len(env_ids),1), device=self.device).squeeze(1)
        random_pitch = torch_rand_float(0, 0, (len(env_ids),1), device=self.device).squeeze(1)
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 8:10] = 0.0  # zero y linear velocity

        # keep delta-position buffers consistent after reset
        self._last_base_pos[env_ids] = self.root_states[env_ids, 0:3]
        self._last_base_z[env_ids]   = self.root_states[env_ids, 2]
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,
                                   dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self._get_base_heights() < 0

    def step(self, actions):
        """ Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # actions = self.reindex(actions)
        actions = actions.to(self.device)

        self.global_counter += 1   
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        
        # 增大ABAD的刚度和阻尼，达到fixed的效果，将动作设为0
        # fixed ABAD DOF 
        self.actions[:, self.hip_joint_indices] = 0.0

        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            # self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels 
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras
    

    def _compute_torques(self, actions):

        """ Compute torques from actions.
            Actions can be interpreted as position or velocity targets given to a PD controller, or directly as scaled torques.
            [NOTE]: torques must have the same dimension as the number of DOFs, even if some DOFs are not actuated.

        Args:
            actions (torch.Tensor): Actions

        Returns:
            [torch.Tensor]: Torques sent to the simulation
        """
        # 如果使用滤波器，则对动作进行滤波
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)

        #pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]
                - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)


    def _terrain_height_at(self, points_xy, env_ids=None):
        """Query terrain height at arbitrary XY points.
        points_xy: [N,2] world xy
        env_ids:   [N]   which env each point belongs to (needed for env origins)
        returns:   [N]   height (world z)
        """
        device = points_xy.device
        N = points_xy.shape[0]

        # Need env_ids to subtract env origins correctly on tiled terrains
        if env_ids is None:
            # fallback: assume all points belong to env 0
            env_ids = torch.zeros(N, dtype=torch.long, device=device)

        # ---- Heightfield sampling path (recommended) ----
        # Common in legged_gym-style envs:
        # self.height_samples: [H,W] int16/float (unscaled or scaled)
        # self.terrain.cfg.horizontal_scale, vertical_scale, border_size
        if hasattr(self, "height_samples") and hasattr(self, "terrain") and hasattr(self.terrain, "cfg"):
            cfg = self.terrain.cfg
            hs = float(getattr(cfg, "horizontal_scale", 1.0))
            vs = float(getattr(cfg, "vertical_scale", 1.0))
            border = int(getattr(cfg, "border_size", 0))

            # env origins: [E,3]
            if hasattr(self, "env_origins"):
                origins_xy = self.env_origins[env_ids, 0:2]
            elif hasattr(self.terrain, "env_origins"):
                origins_xy = self.terrain.env_origins[env_ids, 0:2]
            else:
                origins_xy = torch.zeros((N, 2), device=device)

            # world -> terrain local
            p_local = points_xy - origins_xy  # [N,2]

            # local meters -> heightfield indices
            # In many implementations: (x/hs) + border maps into [0..W-1]
            ix = torch.floor(p_local[:, 0] / hs).long() + border
            iy = torch.floor(p_local[:, 1] / hs).long() + border

            H, W = self.height_samples.shape[0], self.height_samples.shape[1]
            ix = torch.clamp(ix, 0, W - 1)
            iy = torch.clamp(iy, 0, H - 1)

            h = self.height_samples[iy, ix].to(device).float() * vs
            return h

        # ---- Fallbacks ----
        # If you don't have a heightfield, assume flat ground z=0.
        return torch.zeros(N, device=device)

    # ------------ cost functions----------------
    def _cost_torque_limit(self):
        # constaint torque over limit
        #return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
        # return 1.*(torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)>0.0)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)
    
    def _cost_pos_limit(self):
        # upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        # lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        # out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        # return out_limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        # return 1.*(torch.sum(out_of_limits, dim=1)>0.0)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(out_of_limits, dim=1)
   
    def _cost_dof_vel_limits(self):
        # return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
        # return 1.*(torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)>0.0)

        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum((torch.abs(self.dof_vel[:, [0,1,2,4,5,6,8,9,10,12,13,14]]) - self.dof_vel_limits[ [0,1,2,4,5,6,8,9,10,12,13,14]]*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
   
    def _cost_hip_pos(self):
        # max_rad = 0.05
        # hip_err = torch.where(torch.abs(self.dof_pos[:, self.hip_joint_indices] ) < max_rad, torch.zeros_like(self.dof_pos[:, self.hip_joint_indices] ), torch.abs(self.dof_pos[:, self.hip_joint_indices]) - max_rad)
        # # print('hip_err:', hip_err)
        # return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(hip_err), dim=1)

        #return torch.sum(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - self.default_dof_pos[:, [0, 3, 6, 9]]), dim=1)
        # return flag * torch.mean(torch.square(self.dof_pos[:, [0, 3, 6, 9]] - torch.zeros_like(self.dof_pos[:, [0, 3, 6, 9]])), dim=1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - 0.0),dim=-1)
    
    def _cost_default_joint(self):
        # Penalize motion at zero commands
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.abs(self.dof_pos[:, [1,2,5,6,9,10,13,14]] - self.default_dof_pos[:,[1,2,5,6,9,10,13,14]]), dim=1)


    #------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity
        return torch.clamp(-self.projected_gravity[:,2],0,1) * torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_tracking_lin_vel_x(self):
        lin_vel_x_error = torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0])
        lin_vel_x_error = torch.clip(lin_vel_x_error,0,1)
        reward = torch.exp(-lin_vel_x_error/self.cfg.rewards.tracking_sigma)
        return reward

    def _reward_base_ang_acc(self):
        # Penalize dof accelerations
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square((self.last_root_vel[:, 3:] - self.root_states[:, 10:13]) / self.dt), dim=1)
    
    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)
 
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        return torch.square(base_height - self.cfg.rewards.base_height_target)
    
    def _reward_torques(self):
        # Penalize torques
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.torques), dim=1)
    
    def _reward_powers(self):
        # Penalize torques
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.abs(self.torques)*torch.abs(self.dof_vel), dim=1)
        #return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_powers_dist(self):
        # Penalize power dist
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.var(self.torques*self.dof_vel, dim=1)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)
    
    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.last_actions - self.actions), dim=1)
    
    def _reward_action_smoothness(self):
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.square(self.action_history_buf[:,-1,:] - 2*self.action_history_buf[:,-2,:]+self.action_history_buf[:,-3,:]), dim=1)

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)
    
    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.) # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(out_of_limits, dim=1)
    
    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_upward(self):
        # print(self.projected_gravity[:,2])
        return 1 - torch.clamp(self.projected_gravity[:,2], -1, 1)
        # return 1 - self.projected_gravity[:,2]

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)
    
    def _reward_hip_pos(self):
        # penalty hip joint position not equal to zero
        reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - torch.zeros_like(self.default_dof_pos[:, self.hip_joint_indices])), dim=1)/0.05) 
        return torch.clamp(-self.projected_gravity[:,2],0,1) * reward  # torch.sum(torch.square(self.dof_pos[:, [0, 4, 8, 12]] - torch.zeros_like(self.dof_pos[:, [0, 4, 8, 12]])), dim=1)
    
    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        # reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)/0.05) +\
        #     torch.exp(-torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05)
        reward = torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1) +\
                 torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)
        return torch.clamp(-self.projected_gravity[:,2],0,1)*reward        
    
    def _reward_feet_all_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] < 1.
        return torch.clamp(-self.projected_gravity[:,2],0,1)*0.25 * torch.sum(contact, dim=1)

    def _reward_turn_yaw(self):
        # target yaw rate (cmd)
        cmd_yaw = self.commands[:, 2]
        yaw_rate = self.base_ang_vel[:, 2]

        sigma_yaw = 0.5  # rad/s，可调：0.3~1.0
        yaw_err = yaw_rate - cmd_yaw
        r_track = torch.exp(-(yaw_err / sigma_yaw) ** 2)

        wheel_vel = self.dof_vel[:, self.foot_joint_indices]  # [N,4]
        left = (wheel_vel[:, 0] + wheel_vel[:, 2]) / 2.0
        right = (wheel_vel[:, 1] + wheel_vel[:, 3]) / 2.0
        diff = right - left

        diff_scale = 5.0  # 越大越像 sign，可调 2~10
        sign_score = 0.5 * (1.0 + torch.tanh(diff_scale * cmd_yaw * diff))  # [0,1]

        k = 5.0  # 建议先给个常数，比如 1.0~5.0，再慢慢调
        sigma_diff = 2.0         # rad/s，可调：1~5
        diff_err = diff - k * cmd_yaw
        r_diff = torch.exp(-(diff_err / sigma_diff) ** 2)

        r_energy = torch.exp(-0.01 * (left ** 2 + right ** 2))

        gate = torch.clamp((torch.abs(cmd_yaw) - 0.05) / 0.10, 0.0, 1.0)
        reward = (
            0.75 * r_track +      # 主目标：实际 yaw-rate 跟踪
            0.15 * r_diff  +       # 辅助：幅度一致
            0.10 * sign_score      # 辅助：方向一致
        )
        reward = reward * r_energy
        return reward * gate

    def _reward_orientation_roll(self):
        # Penalize Roll only (g_y^2)
        # 对应原来的 orientation_y，但可能给予较小的惩罚
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(self.projected_gravity[:, 1])

    def _reward_feet_traction(self):
        # 接触判断：Z轴上有力
        is_contact = (self.contact_forces[:, self.feet_indices, 2] > 1.0).float()
        # 运动指令判断
        move_cmd = (torch.abs(self.commands[:, 0]) > 0.1).float()
        # 速度一致性奖励：基座速度与指令方向一致
        vx = self.base_lin_vel[:, 0]
        vel_tracking = torch.clamp(vx * torch.sign(self.commands[:, 0]), min=0.0)
        rew = torch.sum(is_contact, dim=1) * vel_tracking * move_cmd
        return rew

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.sum(1.*(torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_climb_assist(self):
        move_cmd = (torch.abs(self.commands[:, 0]) > 0.1).float()
        vel_z = torch.clamp(self.base_lin_vel[:, 2], min=0.0)
        is_contact = (torch.sum(self.contact_forces[:, self.feet_indices, 2] > 1.0, dim=1) > 0).float()
        return vel_z * move_cmd * is_contact

    def _reward_orientation_pitch(self):
        pitch = self.projected_gravity[:, 0] # g_x projection ~ sin(pitch)
        
        backward_lean = torch.clamp(pitch - 0.05, min=0.0) 
        forward_lean = torch.clamp(-0.5 - pitch, min=0.0)
        return torch.clamp(-self.projected_gravity[:,2],0,1) * (torch.square(backward_lean) + torch.square(forward_lean))

    def _reward_pitch_velocity(self):
        pitch_angular_vel = self.base_ang_vel[:, 1]
        # 使用平方惩罚，当俯仰角速度超过阈值时给予更大的惩罚
        pitch_vel_penalty = torch.square(pitch_angular_vel)
        return pitch_vel_penalty
    
    def _reward_stand_still(self):
        cmd_small = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        deviation = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        reward = torch.exp(-deviation)
        return cmd_small * reward

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
    