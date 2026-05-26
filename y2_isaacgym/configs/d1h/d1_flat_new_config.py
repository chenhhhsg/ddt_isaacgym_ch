from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch

# config
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi


class D1FlatNew(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        # 5th command = lin_vel_z; extend commands_scale (base uses 3) without modifying legged_robot.py
        if self.cfg.commands.num_commands >= 5:
            vz_cmd_scale = getattr(self.obs_scales, 'lin_vel_z_cmd', self.obs_scales.lin_vel)
            self.commands_scale = torch.tensor(
                [self.obs_scales.lin_vel, self.obs_scales.lin_vel, self.obs_scales.ang_vel, vz_cmd_scale],
                device=self.device,
                requires_grad=False,
            )
        self.hip_joint_indices = [0, 4, 8, 12]
        self.foot_joint_indices = [3, 7, 11, 15]
        # target height for height velocity tracking: initialised to cfg base_height_target (tensor for .clone / indexing)
        h0 = float(self.cfg.rewards.base_height_target)
        self.target_height = torch.full((self.num_envs,), h0, device=self.device, dtype=torch.float)
        self.last_target_height = self.target_height.clone()
        self.base_height = torch.full((self.num_envs,), h0, device=self.device, dtype=torch.float)
        self.last_base_height = self.base_height.clone()

    def _resample_commands(self, env_ids):
        # Proportional command sampling with explicit stand_still category.
        # Categories: 1.x  2.y  3.xy  4.turn  5.x+turn  6.xy+turn  7.stand_still
        # stand_still sets all planar commands to 0, preventing under-training on
        # zero-input — the root cause of forward-rolling in real deployment.
        command_select = torch.rand(len(env_ids), device=self.device)
        cp = torch.tensor(self.cfg.commands.commands_proportion, device=self.device)

        sel_x        = command_select < cp[0]
        sel_y        = (command_select >= cp[0])                    & (command_select < cp[0]+cp[1])
        sel_xy       = (command_select >= cp[0]+cp[1])              & (command_select < cp[0]+cp[1]+cp[2])
        sel_turn     = (command_select >= cp[:3].sum())             & (command_select < cp[:4].sum())
        sel_x_turn   = (command_select >= cp[:4].sum())             & (command_select < cp[:5].sum())
        sel_xy_turn  = (command_select >= cp[:5].sum())             & (command_select < cp[:6].sum())
        # sel_stand_still: everything remaining (cp[6])

        self.commands[env_ids, :] = 0.0

        x_ids   = torch.cat([env_ids[sel_x],  env_ids[sel_xy],  env_ids[sel_x_turn],  env_ids[sel_xy_turn]])
        y_ids   = torch.cat([env_ids[sel_y],  env_ids[sel_xy],                         env_ids[sel_xy_turn]])
        ang_ids = torch.cat([env_ids[sel_turn], env_ids[sel_x_turn], env_ids[sel_xy_turn]])

        self.commands[x_ids, 0] = torch_rand_float(
            self.command_ranges['lin_vel_x'][0], self.command_ranges['lin_vel_x'][1],
            (len(x_ids), 1), device=self.device).squeeze(1)
        self.commands[y_ids, 1] = torch_rand_float(
            self.command_ranges['lin_vel_y'][0], self.command_ranges['lin_vel_y'][1],
            (len(y_ids), 1), device=self.device).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[ang_ids, 3] = torch_rand_float(
                self.command_ranges['heading'][0], self.command_ranges['heading'][1],
                (len(ang_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[ang_ids, 2] = torch_rand_float(
                self.command_ranges['ang_vel_yaw'][0], self.command_ranges['ang_vel_yaw'][1],
                (len(ang_ids), 1), device=self.device).squeeze(1)

        if self.cfg.commands.num_commands >= 5:
            sampled = torch_rand_float(
                self.command_ranges['lin_vel_z'][0], self.command_ranges['lin_vel_z'][1],
                (len(env_ids), 1), device=self.device).squeeze(1)
            zero_mask = torch.rand(len(env_ids), device=self.device) < self.cfg.commands.zero_height_cmd_prob
            self.commands[env_ids, 4] = torch.where(zero_mask, torch.zeros(len(env_ids), device=self.device), sampled)

            # vz_cmd=0 envs: target_height stays constant for the whole episode, so randomise
            # it now to train the policy to hold still at all heights.
            # vz_cmd≠0 envs: start from nominal and let the command integrate over time.
            zero_height_ids = env_ids[zero_mask]
            nonzero_height_ids = env_ids[~zero_mask]
            if len(zero_height_ids) > 0:
                self.target_height[zero_height_ids] = torch_rand_float(
                    self.cfg.rewards.height_target_min,
                    self.cfg.rewards.height_target_max,
                    (len(zero_height_ids), 1), device=self.device).squeeze(1)
            if len(nonzero_height_ids) > 0:
                self.target_height[nonzero_height_ids] = self.cfg.rewards.base_height_target

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        self.last_base_height = self.base_height.clone()
        self.base_height = self._get_base_heights()

        if self.cfg.commands.num_commands >= 5:
            h_min = self.cfg.rewards.height_target_min
            h_max = self.cfg.rewards.height_target_max
            self.last_target_height = self.target_height.clone()
            self.target_height = torch.clamp(
                self.target_height + self.commands[:, 4] * self.dt, h_min, h_max
            )

        # Mid-episode lateral command flip: randomly negate lin_vel_y to expose policy
        # to sudden direction reversals (trains against joystick transient instability).
        flip_prob = getattr(self.cfg.commands, 'mid_episode_flip_prob', 0.002)
        if flip_prob > 0:
            flip_mask = torch.rand(self.num_envs, device=self.device) < flip_prob
            self.commands[flip_mask, 1] = -self.commands[flip_mask, 1]

    def compute_observations(self):
        if self.cfg.commands.num_commands < 5:
            return super().compute_observations()
        
        cmd_z = self.target_height - self.cfg.rewards.base_height_target
        cmd_obs = torch.cat(
            (self.commands[:, :3] * self.commands_scale[:3], (cmd_z * self.commands_scale[3]).unsqueeze(-1)),
            dim=-1,
        )
        # print(f"cmd_obs: {cmd_obs}")
        proprioceptive_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                cmd_obs,
                (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                self.dof_vel * self.obs_scales.dof_vel,
                self.action_history_buf[:, -1],
            ),
            dim=-1,
        )

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat(
            (
                torch.ones(3) * noise_scales.ang_vel * noise_level,
                torch.ones(3) * noise_scales.gravity * noise_level,
                torch.zeros(4),
                torch.ones(self.cfg.env.num_actions) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                torch.ones(self.cfg.env.num_actions) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                torch.zeros(self.num_actions),
            ),
            dim=-1,
        )

        if self.cfg.noise.add_noise:
            proprioceptive_obs += (2 * torch.rand_like(proprioceptive_obs) - 1) * noise_vec.to(self.device)

        self.obs_buf = torch.cat([proprioceptive_obs, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([proprioceptive_obs] * self.cfg.env.history_len, dim=1),
            torch.cat([self.obs_history_buf[:, 1:], proprioceptive_obs.unsqueeze(1)], dim=1),
        )

        # Privileged only: integrated height reference (not in on-policy proprio / history)
        th_scale = getattr(self.obs_scales, 'target_height', self.obs_scales.height_measurements)
        target_h_obs = (self.target_height.unsqueeze(-1) * th_scale).to(proprioceptive_obs.dtype)
        # print(f"target_h_obs: {target_h_obs}")
        privileged_proprioceptive_obs = torch.cat(
            (self.base_lin_vel * self.obs_scales.lin_vel, proprioceptive_obs, target_h_obs),
            dim=-1,
        )
        priv_latent = torch.cat(
            (
                self.contact_filt.float() - 0.5,
                self.randomized_lag_tensor,
                self.mass_params_tensor,
                self.friction_coeffs_tensor,
                self.restitution_coeffs_tensor,
                self.motor_strength,
                self.kp_factor,
                self.kd_factor,
            ),
            dim=-1,
        )

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.0) * self.obs_scales.height_measurements
            self.privileged_obs_buf = torch.cat([privileged_proprioceptive_obs, heights, priv_latent, self.obs_privileged_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.privileged_obs_buf = torch.cat([privileged_proprioceptive_obs, priv_latent, self.obs_privileged_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_privileged_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([privileged_proprioceptive_obs] * self.cfg.env.history_len, dim=1),
            torch.cat([self.obs_privileged_history_buf[:, 1:], privileged_proprioceptive_obs.unsqueeze(1)], dim=1),
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([self.contact_buf[:, 1:], self.contact_filt.float().unsqueeze(1)], dim=1),
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            pure_obs_hist = self.obs_history_buf[:, :, : -self.num_actions].reshape(self.num_envs, -1)
            act_hist = self.action_history_buf.view(self.num_envs, -1)
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, pure_obs_hist, act_hist], dim=-1)

    def _reset_root_states(self, env_ids):
        """Resets ROOT states position and velocities of selected environmments
            Sets base position based on the curriculum
            Selects randomized base velocities within -0.5:0.5 [m/s, rad/s]
        Args:
            env_ids (List[int]): Environemnt ids
        """
        # base position
        if self.custom_origins:
            self.cfg.init_state.pos
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)  # xy position within 1m of the center
            # self.root_states[env_ids, 2] += torch_rand_float(0., 0.2, (len(env_ids), 1), device=self.device).squeeze(1) # z position within 0.2m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 2] += torch_rand_float(0.0, 0.2, (len(env_ids), 1), device=self.device).squeeze(1)
        # base rotation
        random_roll = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        random_pitch = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)  # [7:10]: lin vel, [10:13]: ang vel

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def check_termination(self):
        """Check if environments need to be reset"""
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length  # no terminal reward for time-outs
        self.reset_buf |= self.time_out_buf
        self.reset_buf |= self._get_base_heights() < 0

    def step(self, actions):
        """Apply actions, simulate, call self.post_physics_step()

        Args:
            actions (torch.Tensor): Tensor of shape (num_envs, num_actions_per_env)
        """
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        # actions = self.reindex(actions)
        actions = actions.to(self.device)

        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices] = 0  # zero position of wheels
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        """Compute torques from actions.
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

        # pd controller
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:, 1:, :].clone(), actions_scaled.unsqueeze(1).clone()], dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes, self.randomized_lag, :] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == 'P':
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains * (joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
                torques[:, self.foot_joint_indices] = (
                    self.p_gains[self.foot_joint_indices] * actions_scaled[:, self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:, self.foot_joint_indices]
                )
            else:
                torques = self.kp_factor * self.p_gains * (joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains * self.dof_vel
                torques[:, self.foot_joint_indices] = self.kp_factor[:, self.foot_joint_indices] * self.p_gains[self.foot_joint_indices] * actions_scaled[:, self.foot_joint_indices]
                -self.kd_factor[:, self.foot_joint_indices] * self.d_gains[self.foot_joint_indices] * self.dof_vel[:, self.foot_joint_indices]
        else:
            raise NameError(f'Unknown controller type: {control_type}')
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _target_height_extreme_proximity(self):
        """0 at mid of [height_target_min, height_target_max], 1 at either bound (uses target_height)."""
        h_min = float(self.cfg.rewards.height_target_min)
        h_max = float(self.cfg.rewards.height_target_max)
        half_span = max(h_max - h_min, 1e-6) * 0.5
        mid = 0.5 * (h_min + h_max)
        return torch.clamp(torch.abs(self.target_height - mid) / half_span, 0.0, 1.0)

    # ------------ reward functions----------------
    def _reward_lin_vel_z(self):
        # Penalize z axis base linear velocity; gate off when actively tracking height
        if self.cfg.commands.num_commands >= 5:
            actual_vz = torch.abs(self.base_height - self.last_base_height) / self.dt
            # Hard gate: no vertical-velocity penalty while base height is changing (height tracking).
            mask = (actual_vz < 1e-2).float()
        else:
            mask = 1.0
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * mask * torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        # Penalize xy axes base angular velocity
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_ang_acc(self):
        # Penalize dof accelerations
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square((self.last_root_vel[:, 3:] - self.root_states[:, 10:13]) / self.dt), dim=1)

    def _reward_orientation(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square(self.projected_gravity[:, :2]), dim=1)

    def _reward_orientation_y(self):
        # Penalize non flat base orientation
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.square(self.projected_gravity[:, 1])

    def _reward_torques(self):
        # Penalize torques
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square(self.torques), dim=1)

    def _reward_powers(self):
        # Penalize torques
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)
        # return torch.sum(torch.multiply(self.torques, self.dof_vel), dim=1)

    def _reward_powers_dist(self):
        # Penalize power dist
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.var(self.torques * self.dof_vel, dim=1)

    def _reward_dof_vel(self):
        # Penalize dof velocities
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square(self.dof_vel[:, ~self.foot_joint_indices]), dim=1)

    def _reward_dof_wheels_vel(self):
        # Penalize wheel dof velocities only when planar standstill and not height-tracking
        # (commands[:, 4] = lin_vel_z); rotating wheels helps height regulation.
        zero_planar = torch.norm(self.commands[:, :2], dim=1) < 0.1
        if self.cfg.commands.num_commands >= 5:
            actual_vz = torch.abs(self.base_height - self.last_base_height) / self.dt
            mask = (actual_vz > 1e-2).float()
        else:
            mask = 1.0
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 *\
             torch.sum(torch.square(self.dof_vel[:, self.foot_joint_indices]), dim=1) * zero_planar * mask
    
    def _reward_feet_distance(self):
        current_time = self.episode_length_buf * self.dt
        cur_footsteps_translated = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        footsteps_in_body_frame = torch.zeros(self.num_envs, 4, 3, device=self.device)
        for i in range(4):
            footsteps_in_body_frame[:, i, :] = quat_rotate_inverse(self.base_quat, cur_footsteps_translated[:, i, :])

        stance_length = 0.45 * torch.zeros([self.num_envs, 1], device=self.device)
        stance_width = 0.4 * torch.ones([self.num_envs, 1], device=self.device)
        desired_xs = torch.cat([stance_length / 2, stance_length / 2, -stance_length / 2, -stance_length / 2], dim=1)
        desired_ys = torch.cat([stance_width / 2, -stance_width / 2, stance_width / 2, -stance_width / 2], dim=1)
        stance_diff_x = torch.zeros(self.num_envs, device=self.device)
        for i in range(2):
            x_error = torch.abs(footsteps_in_body_frame[:, 2*i, 0] - footsteps_in_body_frame[:, 2*i + 1, 0])
            # print(f"x_error i: {i}, x_error: {x_error}")
            stance_diff_x += torch.square(x_error)
        stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1]).sum(dim=1)
        # print(f"footsteps_in_body_frame: {footsteps_in_body_frame[:, :, 1]}")
        return stance_diff_x + stance_diff_y

    def _reward_dof_wheel_action(self):
        # Penalise non-zero wheel *actions* at zero planar commands.
        # Unlike dof_wheels_vel, this directly shapes the policy output regardless of
        # whether wheels physically spin in simulation (IsaacGym high-friction masks
        # rotation, other sims/real hardware do not — causing forward rolling at low height).
        zero_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        return (torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7
                * torch.sum(torch.square(self.actions[:, self.foot_joint_indices]), dim=1)
                * zero_cmd)

    def _reward_dof_acc(self):
        # Penalize dof accelerations
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        # Penalize changes in actions
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        return (
            torch.clamp(-self.projected_gravity[:, 2], 0, 0.7)
            / 0.7
            * torch.sum(torch.square(self.action_history_buf[:, -1, :] - 2 * self.action_history_buf[:, -2, :] + self.action_history_buf[:, -3, :]), dim=1)
        )

    def _reward_collision(self):
        # Penalize collisions on selected bodies
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1)

    def _reward_dof_pos_limits(self):
        # Penalize dof positions too close to the limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel(self):
        # Tracking of linear velocity commands (xy axes)
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.exp(-lin_vel_error / self.cfg.rewards.tracking_sigma)

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.exp(-ang_vel_error / self.cfg.rewards.tracking_sigma)
    
    def _reward_base_height(self):
        # Penalize base height away from target
        base_height = self._get_base_heights()
        g = torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7
        prox = self._target_height_extreme_proximity()
        boost = float(getattr(self.cfg.rewards, 'height_extreme_penalty_boost', 1.0))
        stress = 1.0 + boost * prox
        return g * stress * torch.square((base_height - self.target_height) / 0.05)


    # def _reward_tracking_height_velocity(self):
    #     h_min = float(self.cfg.rewards.height_target_min)
    #     h_max = float(self.cfg.rewards.height_target_max)
    #     span = max(h_max - h_min, 1e-6)
    #     vz_cmd = self.commands[:, 4]
    #     current_h = self._get_base_heights()

    #     vel_error = torch.square(vz_cmd - self.base_lin_vel[:, 2])
    #     g = torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7

    #     # boundary_scale: 0 at h_min/h_max, 1 at mid-range (clamped to [0,1])
    #     d1 = torch.clamp((current_h - h_min) / span, 0.0, 1.0)
    #     d2 = torch.clamp((h_max - current_h) / span, 0.0, 1.0)
    #     boundary_scale = torch.clamp(torch.minimum(d1, d2) * 4, 0.0, 1.0)

    #     # zero cmd: always full scale so robot learns to hold height precisely anywhere
    #     # non-zero cmd: scale down near boundaries where the command is physically unreachable
    #     zero_cmd = (torch.abs(vz_cmd) < 1e-3).float()
    #     effective_scale = zero_cmd + (1.0 - zero_cmd) * boundary_scale

    #     return g * effective_scale * torch.exp(-vel_error / 1e-4)
    
    def _reward_tracking_height_velocity(self):
        vz_cmd = (self.target_height - self.last_target_height) / self.dt
        vel_error = torch.square(vz_cmd - self.base_lin_vel[:, 2])
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.exp(-vel_error / 1e-4)

    def _reward_height_change_xdist(self):
        # During height change, penalize x-offset drift of base relative to mean foot position
        # in robot body frame. Keeps the robot from arcing forward/backward while ascending/descending.
        actual_vz = torch.abs(self.base_height - self.last_base_height) / self.dt
        height_changing = actual_vz > 1e-2

        mean_foot_pos = self.feet_pos.mean(dim=1)          # (N, 3) world frame
        diff_world = self.root_states[:, :3] - mean_foot_pos  # base minus feet, world frame
        diff_body = quat_rotate_inverse(self.root_states[:, 3:7], diff_world)  # rotate to body frame
        x_dist = diff_body[:, 0]
        # print(f"x_dist: {x_dist}")

        desired = float(getattr(self.cfg.rewards, 'head_feet_x_dist_target', 0.0))
        return height_changing * torch.square((x_dist - desired)/2e-2)

    def _reward_upward(self):
        # print(self.projected_gravity[:,2])
        return torch.square(1 - self.projected_gravity[:, 2])
        # return 1 - self.projected_gravity[:,2]

    def _reward_stand_still(self):
        # Penalize motion at zero commands
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 *\
             torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1) * (torch.norm(self.commands[:, :2], dim=1) < 0.1)

    def _reward_hip_pos(self):
        # penalty hip joint position not equal to zero
        flag = 1.0 * (torch.abs(self.commands[:, 1]) < 0.1)
        return (
            torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 *\
                 flag * torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - self.default_dof_pos[:, self.hip_joint_indices]), dim=1)
        )

    def _reward_foot_mirror(self):
        # penalty when feet contact not mirror, RL foot mirror RR foot, FL foot mirror FR foot
        mirror = torch.tensor([-1, 1, 1], device=self.device)
        # reward = torch.exp(-torch.sum(torch.square(self.dof_pos[:,[0,1,2]] - self.dof_pos[:,[12,13,14]] * mirror),dim=-1)/0.05) +\
        #     torch.exp(-torch.sum(torch.square(self.dof_pos[:,[8,9,10]] - self.dof_pos[:,[4,5,6]] * mirror),dim=-1)/0.05)
        reward = torch.sum(torch.square(self.dof_pos[:, [0, 1, 2]] - self.dof_pos[:, [12, 13, 14]] * mirror), dim=-1) + torch.sum(
            torch.square(self.dof_pos[:, [8, 9, 10]] - self.dof_pos[:, [4, 5, 6]] * mirror), dim=-1
        )
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * reward

    def _reward_default_joint(self):
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
    
    def _reward_leg_symmetry(self):
        # All terms: sum_i (q_a,i - q_b,i * m_i)^2 with m = [-1,1,1] (hip opposite) or [1,1,1] (same sign).
        m_lr = torch.tensor([-1.0, 1.0, 1.0], device=self.device)
        m_eq = torch.ones(3, device=self.device)
        pairs = (
            ([0, 1, 2], [12, 13, 14], m_lr),   # FL–RR diagonal
            ([8, 9, 10], [4, 5, 6], m_lr),    # RL–FR diagonal
            ([0, 1, 2], [4, 5, 6], m_lr),     # FL–FR same row
            ([8, 9, 10], [12, 13, 14], m_lr), # RL–RR same row
            ([0, 1, 2], [8, 9, 10], m_eq),    # FL–RL same side
            ([4, 5, 6], [12, 13, 14], m_eq), # FR–RR same side
        )
        dof_pos_diff = torch.zeros(self.num_envs, device=self.device, dtype=self.dof_pos.dtype)
        for ia, ib, mir in pairs:
            dof_pos_diff += torch.sum(
                torch.square(self.dof_pos[:, ia] - self.dof_pos[:, ib] * mir),
                dim=-1,
            )
        return dof_pos_diff * (torch.abs(self.commands[:, 1]) < 0.1)

    def _reward_leg_symmetry_same_row(self):
        # All terms: sum_i (q_a,i - q_b,i * m_i)^2 with m = [-1,1,1].
        m_lr = torch.tensor([-1.0, 1.0, 1.0], device=self.device)
        pairs = (
            ([0, 1, 2], [4, 5, 6], m_lr),     # FL–FR same row
            ([8, 9, 10], [12, 13, 14], m_lr), # RL–RR same row
        )
        dof_pos_diff = torch.zeros(self.num_envs, device=self.device, dtype=self.dof_pos.dtype)
        for ia, ib, mir in pairs:
            dof_pos_diff += torch.sum(
                torch.square(self.dof_pos[:, ia] - self.dof_pos[:, ib] * mir),
                dim=-1,
            )
        return dof_pos_diff * (torch.abs(self.commands[:, 1]) < 0.1)

    # ------------ cost functions----------------
    def _cost_torque_limit(self):
        # constaint torque over limit
        # return 1.*(torch.sum(1.*(torch.abs(self.torques) > self.torque_limits*self.cfg.rewards.soft_torque_limit),dim=1)>0.0)
        # return 1.*(torch.sum((torch.abs(self.torques) - self.torque_limits*self.cfg.rewards.soft_torque_limit).clip(min=0.), dim=1)>0.0)
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum((torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1)

    def _cost_pos_limit(self):
        # upper_limit = 1.*(self.dof_pos > self.dof_pos_limits[:, 1])
        # lower_limit = 1.*(self.dof_pos < self.dof_pos_limits[:, 0])
        # out_limit = 1.*(torch.sum(upper_limit + lower_limit,dim=1) > 0.0)
        # return out_limit
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)  # lower limit
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        # return 1.*(torch.sum(out_of_limits, dim=1)>0.0)
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(out_of_limits, dim=1)

    def _cost_dof_vel_limits(self):
        # return 1.*(torch.sum(1.*(torch.abs(self.dof_vel) > self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit),dim=1) > 0.0)
        # return 1.*(torch.sum((torch.abs(self.dof_vel) - self.dof_vel_limits*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)>0.0)

        return (
            torch.clamp(-self.projected_gravity[:, 2], 0, 0.7)
            / 0.7
            * torch.sum(
                (torch.abs(self.dof_vel[:, [0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]]) - self.dof_vel_limits[[0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14]] * self.cfg.rewards.soft_dof_vel_limit).clip(
                    min=0.0, max=1.0
                ),
                dim=1,
            )
        )

    def _cost_hip_pos(self):
        return torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 *\
             torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - 0.0), dim=-1)

    def _cost_default_joint(self):
        # Penalize motion at zero commands
        return (
            torch.clamp(-self.projected_gravity[:, 2], 0, 0.7) / 0.7 * torch.sum(torch.abs(self.dof_pos[:, [1, 2, 5, 6, 9, 10, 13, 14]] - self.default_dof_pos[:, [1, 2, 5, 6, 9, 10, 13, 14]]), dim=1)
        )

    


class D1FlatNewCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent = 4 + 1 + 4 + 1 + 1 + 16 + 16 + 16
        n_proprio = 58  # +1 vs 4 cmd: height command in proprio (vx, vy, yaw, height)
        n_privileged_proprio = 62  # +1 vs 61: privileged target_height (scaled)
        history_len = 10
        num_observations = n_proprio + history_len * n_proprio
        num_privileged_obs = n_privileged_proprio + n_scan + history_len * n_privileged_proprio + n_priv_latent
        num_actions = 16

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 0.60]  # x,y,z [m]
        default_joint_angles = {  # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.0,  # [rad]
            'FR_hip_joint': -0.0,  # [rad]
            'RL_hip_joint': 0.0,  # [rad]
            'RR_hip_joint': -0.0,  # [rad]
            'FL_thigh_joint': 0.8,  # [rad]
            'FR_thigh_joint': 0.8,  # [rad]
            'RL_thigh_joint': 0.8,  # [rad]
            'RR_thigh_joint': 0.8,  # [rad]
            'FL_calf_joint': -1.5,  # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RL_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,  # [rad]
            'FL_foot_joint': 0.0,
            'FR_foot_joint': 0.0,
            'RL_foot_joint': 0.0,
            'RR_foot_joint': 0.0,
        }

    class control(LeggedRobotCfg.control):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'hip': 50.0, 'thigh': 50.0, 'calf': 50.0, 'foot': 10.0}  # [N*m/rad]
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0, 'foot': 0.5}  #  [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel_z_cmd = 5.0  # scales commands[:, 4] (vertical velocity cmd) in proprio

    class commands(LeggedRobotCfg.control):
        curriculum = True
        max_curriculum = 3.0
        num_commands = 5  # lin_vel_x, lin_vel_y, ang_vel_yaw, heading, lin_vel_z (obs: vx,vy,yaw + vz cmd)
        resampling_time = 5.0  # time before command are changed[s]
        heading_command = True  # if true: compute ang vel command from heading error
        global_reference = False
        zero_height_cmd_prob = 0.3  # probability of sampling zero lin_vel_z command
        mid_episode_flip_prob = 0.002  # per-step prob of negating lin_vel_y (trains against joystick transients)
        # Proportions for command categories (must sum to 1.0):
        # [x, y, xy, turn, x+turn, xy+turn, stand_still]
        # stand_still (last bucket) trains the robot on zero planar commands,
        # which reduces forward-rolling at zero input in real deployment.
        commands_proportion = [0.25, 0.2, 0.15, 0.15, 0.1, 0.05, 0.1]

        class ranges:
            lin_vel_x = [-1.0, 1.0]  # min max [m/s]
            lin_vel_y = [-1.0, 1.0]  # min max [m/s]
            ang_vel_yaw = [-1, 1]  # min max [rad/s]
            heading = [-3.14, 3.14]
            lin_vel_z = [-0.1, 0.1]  # [m/s] commanded base vertical velocity

    class asset(LeggedRobotCfg.asset):
        file = '{ROOT_DIR}/resources/d1/urdf/robot1.urdf'
        foot_name = 'foot'
        name = 'd1'
        penalize_contacts_on = ['thigh', 'calf', 'base']
        terminate_after_contacts_on = []
        self_collisions = 0  # 1 to disable, 0 to enable...bitwise filter
        replace_cylinder_with_capsule = False  # replace collision cylinders with capsules, leads to faster/more stable simulation
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            torques = 0.0
            powers = 0.0  # -2e-5
            termination = 0.0
            tracking_lin_vel = 2.5
            tracking_ang_vel = 1.5
            tracking_height_velocity = 1.0  # tracking reward for height velocity
            lin_vel_z = -0.1
            orientation = -1.0
            # orientation_y = -10.0
            ang_vel_xy = -0.05
            # ang_vel_y = -1.0 # avoid flipping
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_wheels_vel = -0.02
            dof_acc = -2.5e-7
            base_height = -0.5
            feet_air_time = 0.0
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.01
            # action_smoothness= -0.01
            # foot_mirror = -0.05
            hip_pos = -0.2
            default_joint = -0.0
            upward = 0.5
            # feet_all_contact = -0.5
            # feet_contact_forces = -0.1
            # joint_power=-2e-5
            # powers_dist =-1.0e-5
            leg_symmetry = -0.8
            # leg_symmetry_same_row = -0.8
            dof_wheel_action = -0.0  # penalise wheel action output at zero commands (sim-to-real fix)
            # stand_still = -1.0
            height_change_xdist = -1.0  # penalize base x-drift during height change
            feet_distance = -5.0

        only_positive_rewards = True  # if true negative total rewards are clipped at zero (avoids early termination problems)
        tracking_sigma = 0.25  # tracking reward = exp(-error^2/sigma)
        soft_dof_pos_limit = 0.9  # percentage of urdf limits, values above this limit are penalized
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45  # legacy / reference only

        max_contact_force = 500.0  # forces above this value are penalized
        height_target_min = 0.3   # [m] lower height bound above terrain
        height_target_max = 0.55   # [m] upper height bound above terrain
        height_correction_speed = 0.02  # [m/s] fixed correction speed toward target when vz_cmd is small
        # base_height penalty multiplier at bounds: stress = 1 + height_extreme_penalty_boost * proximity(target_height)
        height_extreme_penalty_boost = 0.5
        head_feet_x_dist_target = -0.0257  # [m] desired base x-offset from mean foot in body frame



    class costs(LeggedRobotCfg.costs):
        class scales:
            pos_limit = 1.0
            torque_limit = 1.0
            dof_vel_limits = 1.0
            hip_pos = 2.0
            default_joint = 0.2

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            hip_pos = 0.0
            default_joint = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        friction_range = [0.1, 1.0]  # lowered upper bound; trains policy on low-friction (smooth floor) conditions

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'  # "heightfield" # none, plane, heightfield or trimesh
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        static_friction = 0.4   # lowered from 1.0: allows wheels to slip in training,
        dynamic_friction = 0.4  # making dof_wheel_action / dof_wheels_vel penalties effective
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete, stepping stones, gap]
        # terrain_proportions = [0.1, 0.1, 0.35, 0.25, 0.2]
        terrain_proportions = [0.6, 0.4, 0.0, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.2, 0.2, 0.2, 0.2, 0.0, 0.0]

        # terrain_proportions = [0.2, 0.3, 0.1, 0.1, 0.3]
        # terrain_proportions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        # slope_treshold = 1.0  # slopes above this threshold will be corrected to vertical surfaces
        # slope = [0, 0.6]


class D1FlatNewCfg_Play(D1FlatNewCfg):
    class env(D1FlatNewCfg.env):
        num_envs = 10

    class terrain(D1FlatNewCfg.terrain):
        # mesh_type = 'trimesh'  # "heightfield" # none, plane, heightfield or trimesh
        num_rows = 5
        num_cols = 5
        # terrain types: [smooth slope, rough slope, stairs up, stairs down, discrete]
        # terrain_proportions = [0, 0, 0, 0, 0, 0, 0]
        curriculum = False
        # selected = True  # select a unique terrain type and pass all arguments
        # terrain_kwargs = {
        #     "type": "pit_terrain",
        #     "depth": 0.5,
        #     "platform_size": 4.0
        # } # Dict of arguments for selected terrain

    class noise(D1FlatNewCfg.noise):
        add_noise = False

    class control(D1FlatNewCfg.control):
        use_filter = True

    class domain_rand(D1FlatNewCfg.domain_rand):
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

    class commands(D1FlatNewCfg.commands):
        heading_command = False  # if true: compute ang vel command from heading error

        class ranges:
            lin_vel_x = [3.0, 3.0]  # min max [m/s]
            lin_vel_y = [-0.0, 0.0]  # min max [m/s]
            ang_vel_yaw = [-0, 0]  # min max [rad/s]
            heading = [-0.0, 0.0]
            lin_vel_z = [0.0, 0.0]  # no vertical velocity command in play


class D1FlatNewCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4  # mini batch size = num_envs*nsteps / nminibatches
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        # priv_encoder_dims = [64, 20]
        priv_encoder_dims = []
        activation = 'elu'  # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        # only for 'ActorCriticRecurrent':
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1

        teacher_act = True
        imi_flag = True

    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'd1_flat_new'
        policy_class_name = 'ActorCriticBarlowTwins'
        # policy_class_name = 'ActorCriticTransBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 10000
        save_interval = 500
        num_steps_per_env = 24
        resume = False
        resume_path = ''
        # resume = True
        # resume_path = '/home/martin/Projects/LocomotionWithNP3O/runs/d1_flat_new/20260423140049'
