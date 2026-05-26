from isaacgym.torch_utils import *
from isaacgym import gymtorch

import torch

from ..legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.d1h_new.d1h_new import D1hNew
from utils.math import wrap_to_pi, get_euler_xyz


class D1hLqr(D1hNew):
    """高度/平面(vx,vy) + 俯仰/横滚积分目标；线速度/角速度与姿态跟踪在指令姿态系下计算。"""

    def _init_buffers(self):
        super()._init_buffers()
        nc = self.cfg.commands.num_commands
        if nc >= 5:
            os_ = self.obs_scales
            vz_cmd_scale = getattr(os_, 'lin_vel_z_cmd', os_.lin_vel)
            pitch_rate_scale = getattr(os_, 'ang_vel_pitch_cmd', os_.ang_vel)
            roll_rate_scale = getattr(os_, 'ang_vel_roll_cmd', os_.ang_vel)
            scales = [os_.lin_vel, os_.lin_vel, os_.ang_vel, vz_cmd_scale]
            if nc >= 7:
                scales.extend([pitch_rate_scale, roll_rate_scale])
            self.commands_scale = torch.tensor(scales, device=self.device, requires_grad=False)
        self.hip_joint_indices = [0, 4]
        self.foot_joint_indices = [3, 7]
        rw = self.cfg.rewards
        h0 = float(rw.base_height_target)
        self.target_height = torch.full((self.num_envs,), h0, device=self.device, dtype=torch.float)
        self.last_target_height = self.target_height.clone()
        self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.last_base_height = self.base_height.clone()
        p0 = float(getattr(rw, 'base_pitch_target', 0.0))
        r0 = float(getattr(rw, 'base_roll_target', 0.0))
        self.target_pitch = torch.full((self.num_envs,), p0, device=self.device, dtype=torch.float)
        self.target_roll = torch.full((self.num_envs,), r0, device=self.device, dtype=torch.float)
        self.last_target_pitch = self.target_pitch.clone()
        self.last_target_roll = self.target_roll.clone()
        if self.cfg.env.num_privileged_obs is not None and self.cfg.env.history_encoding:
            self.obs_privileged_history_buf = torch.zeros(
                self.num_envs,
                self.cfg.env.history_len,
                self.cfg.env.n_privileged_proprio,
                device=self.device,
                dtype=torch.float,
            )
        self._refresh_command_frame_quat()

    def reset_idx(self, env_ids):
        super().reset_idx(env_ids)
        if hasattr(self, 'obs_privileged_history_buf'):
            self.obs_privileged_history_buf[env_ids, :, :] = 0.0
        if self.cfg.commands.num_commands >= 7:
            rw = self.cfg.rewards
            self.target_pitch[env_ids] = float(getattr(rw, 'base_pitch_target', 0.0))
            self.target_roll[env_ids] = float(getattr(rw, 'base_roll_target', 0.0))
            self.last_target_pitch[env_ids] = self.target_pitch[env_ids]
            self.last_target_roll[env_ids] = self.target_roll[env_ids]

    def _command_frame_yaw(self):
        if self.cfg.commands.heading_command:
            return self.commands[:, 3]
        _, _, yaw = get_euler_xyz(self.base_quat)
        return yaw

    def _command_frame_quat(self):
        return quat_from_euler_xyz(self.target_roll, self.target_pitch, self._command_frame_yaw())

    def _refresh_command_frame_quat(self):
        self._cmd_frame_quat_cache = self._command_frame_quat()
        # yaw-only frame: horizontal velocity/yaw-rate tracking is independent of commanded pitch/roll
        zeros = torch.zeros(self.num_envs, device=self.device)
        self._yaw_quat_cache = quat_from_euler_xyz(zeros, zeros, self._command_frame_yaw())

    def _lin_vel_in_command_frame(self):
        """Velocity in yaw-only frame so pitch/roll commands don't corrupt vx/vy direction."""
        return quat_rotate_inverse(self._yaw_quat_cache, self.base_lin_vel)

    def _ang_vel_in_command_frame(self):
        """Angular velocity in yaw-only frame for yaw-rate and roll/pitch rate tracking."""
        return quat_rotate_inverse(self._yaw_quat_cache, self.base_ang_vel)

    def _command_gravity_gate(self):
        return 1.0

    def _foot_distance_in_yaw_frame(self):
        foot_v = self.foot_positions[:, 0, :] - self.foot_positions[:, 1, :]
        return quat_rotate_inverse(self._yaw_quat_cache, foot_v)

    def _tracking_cmd_factor(self, axis_idx, range_key):
        """|cmd|=0 -> min factor (default 0.5); |cmd| at range limit -> 1.0."""
        r = self.command_ranges[range_key]
        max_cmd = max(abs(float(r[0])), abs(float(r[1])), 1e-6)
        norm = torch.clamp(torch.abs(self.commands[:, axis_idx]) / max_cmd, 0.0, 1.0)
        f_min = float(getattr(self.cfg.rewards, 'tracking_cmd_factor_min', 0.5))
        return f_min + (1.0 - f_min) * norm

    def _sample_integrated_angle_cmd(self, env_ids, cmd_idx, target_buf, rate_key, zero_prob_key, min_key, max_key, nominal_key):
        r = self.command_ranges
        sampled = torch_rand_float(r[rate_key][0], r[rate_key][1], (len(env_ids), 1), device=self.device).squeeze(1)
        zero_prob = float(getattr(self.cfg.commands, zero_prob_key, 0.3))
        zero_mask = torch.rand(len(env_ids), device=self.device) < zero_prob
        self.commands[env_ids, cmd_idx] = torch.where(zero_mask, torch.zeros(len(env_ids), device=self.device), sampled)
        rw = self.cfg.rewards
        zero_ids = env_ids[zero_mask]
        nonzero_ids = env_ids[~zero_mask]
        if len(zero_ids) > 0:
            target_buf[zero_ids] = torch_rand_float(
                getattr(rw, min_key), getattr(rw, max_key), (len(zero_ids), 1), device=self.device
            ).squeeze(1)
        if len(nonzero_ids) > 0:
            target_buf[nonzero_ids] = float(getattr(rw, nominal_key))

    def _resample_commands(self, env_ids):
        # Categories: 1.x  2.y  3.xy  4.turn  5.x+turn  6.xy+turn  7.stand_still（y/xy 份额在 cfg 中为 0）
        command_select = torch.rand(len(env_ids), device=self.device)
        cp = torch.tensor(self.cfg.commands.commands_proportion, device=self.device)

        sel_x = command_select < cp[0]
        sel_y = (command_select >= cp[0]) & (command_select < cp[0] + cp[1])
        sel_xy = (command_select >= cp[0] + cp[1]) & (command_select < cp[0] + cp[1] + cp[2])
        sel_turn = (command_select >= cp[:3].sum()) & (command_select < cp[:4].sum())
        sel_x_turn = (command_select >= cp[:4].sum()) & (command_select < cp[:5].sum())
        sel_xy_turn = (command_select >= cp[:5].sum()) & (command_select < cp[:6].sum())

        self.commands[env_ids, :] = 0.0

        x_ids = torch.cat([env_ids[sel_x], env_ids[sel_xy], env_ids[sel_x_turn], env_ids[sel_xy_turn]])
        y_ids = torch.cat([env_ids[sel_y], env_ids[sel_xy], env_ids[sel_xy_turn]])
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
            # Task group assignment per env (see cfg.commands.task_proportions):
            #   grp_pure   → only vx/vy/yaw
            #   grp_height → vx/vy/yaw + height
            #   grp_att    → vx/vy/yaw + pitch/roll
            #   grp_all    → all dimensions
            tp = torch.tensor(
                getattr(self.cfg.commands, 'task_proportions', [0.4, 0.3, 0.2, 0.1]),
                device=self.device,
            )
            rand_task = torch.rand(len(env_ids), device=self.device)
            c = torch.cumsum(tp, dim=0)
            use_height = (rand_task >= c[0]) & ((rand_task < c[1]) | (rand_task >= c[2]))
            use_att    = rand_task >= c[1]

            # Default: lock height at nominal, zero rate command
            self.commands[env_ids, 4] = 0.0
            self.target_height[env_ids] = self.cfg.rewards.base_height_target

            h_ids = env_ids[use_height]
            if len(h_ids) > 0:
                sampled = torch_rand_float(
                    self.command_ranges['lin_vel_z'][0], self.command_ranges['lin_vel_z'][1],
                    (len(h_ids), 1), device=self.device).squeeze(1)
                zero_mask = torch.rand(len(h_ids), device=self.device) < self.cfg.commands.zero_height_cmd_prob
                self.commands[h_ids, 4] = torch.where(zero_mask, torch.zeros(len(h_ids), device=self.device), sampled)
                zero_h_ids = h_ids[zero_mask]
                nonzero_h_ids = h_ids[~zero_mask]
                if len(zero_h_ids) > 0:
                    self.target_height[zero_h_ids] = torch_rand_float(
                        self.cfg.rewards.height_target_min,
                        self.cfg.rewards.height_target_max,
                        (len(zero_h_ids), 1), device=self.device).squeeze(1)
                if len(nonzero_h_ids) > 0:
                    self.target_height[nonzero_h_ids] = self.cfg.rewards.base_height_target

            if self.cfg.commands.num_commands >= 7:
                rw = self.cfg.rewards
                # Default: lock attitude at nominal, zero rate commands
                self.commands[env_ids, 5] = 0.0
                self.commands[env_ids, 6] = 0.0
                self.target_pitch[env_ids] = float(getattr(rw, 'base_pitch_target', 0.0))
                self.target_roll[env_ids] = float(getattr(rw, 'base_roll_target', 0.0))

                att_ids = env_ids[use_att]
                if len(att_ids) > 0:
                    self._sample_integrated_angle_cmd(
                        att_ids, 5, self.target_pitch,
                        'ang_vel_pitch', 'zero_pitch_cmd_prob',
                        'pitch_target_min', 'pitch_target_max', 'base_pitch_target',
                    )
                    self._sample_integrated_angle_cmd(
                        att_ids, 6, self.target_roll,
                        'ang_vel_roll', 'zero_roll_cmd_prob',
                        'roll_target_min', 'roll_target_max', 'base_roll_target',
                    )

    def _post_physics_step_callback(self):
        super()._post_physics_step_callback()
        cc = self.cfg.commands
        if getattr(cc, 'cmd_jitter_enabled', False):
            st = max(1, int(getattr(cc, 'cmd_jitter_interval_s', 0.5) / self.dt))
            p = float(getattr(cc, 'cmd_jitter_prob', 0.1))
            lucky = (self.episode_length_buf % st == 0) & (torch.rand(self.num_envs, device=self.device) < p)
            if lucky.any():
                e = lucky.nonzero(as_tuple=False).flatten()
                n = len(e)
                r = self.command_ranges
                dev = self.device

                def t3(lo, hi):
                    return torch.tensor([lo, hi, 0.0], device=dev, dtype=torch.float)[torch.randint(0, 3, (n,), device=dev)]

                self.commands[e, 0] = t3(r['lin_vel_x'][0], r['lin_vel_x'][1])
                self.commands[e, 1] = t3(r['lin_vel_y'][0], r['lin_vel_y'][1])
                if cc.heading_command:
                    self.commands[e, 3] = t3(r['heading'][0], r['heading'][1])
                    forward = quat_apply(self.base_quat, self.forward_vec)
                    heading = torch.atan2(forward[:, 1], forward[:, 0])
                    self.commands[:, 2] = torch.clip(wrap_to_pi(self.commands[:, 3] - heading), -1.0, 1.0)
                else:
                    self.commands[e, 2] = t3(r['ang_vel_yaw'][0], r['ang_vel_yaw'][1])
        if self.cfg.commands.num_commands >= 5:
            h_min = self.cfg.rewards.height_target_min
            h_max = self.cfg.rewards.height_target_max
            self.last_target_height = self.target_height.clone()
            self.target_height = torch.clamp(
                self.target_height + self.commands[:, 4] * self.dt, h_min, h_max
            )
            self.last_base_height = self.base_height.clone()
            self.base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)

        if self.cfg.commands.num_commands >= 7:
            rw = self.cfg.rewards
            self.last_target_pitch = self.target_pitch.clone()
            self.last_target_roll = self.target_roll.clone()
            self.target_pitch = torch.clamp(
                self.target_pitch + self.commands[:, 5] * self.dt,
                float(rw.pitch_target_min), float(rw.pitch_target_max),
            )
            self.target_roll = torch.clamp(
                self.target_roll + self.commands[:, 6] * self.dt,
                float(rw.roll_target_min), float(rw.roll_target_max),
            )

        self._refresh_command_frame_quat()

    def compute_observations(self):
        if self.cfg.commands.num_commands < 5:
            return super().compute_observations()

        rw = self.cfg.rewards
        os_ = self.obs_scales
        cmd_z = self.target_height - rw.base_height_target
        cmd_parts = [self.commands[:, :3] * self.commands_scale[:3], (cmd_z * self.commands_scale[3]).unsqueeze(-1)]
        if self.cfg.commands.num_commands >= 7:
            pitch_scale = getattr(os_, 'pitch_angle_cmd', 1.0)
            roll_scale = getattr(os_, 'roll_angle_cmd', 1.0)
            cmd_pitch = (self.target_pitch - float(getattr(rw, 'base_pitch_target', 0.0))) * pitch_scale
            cmd_roll = (self.target_roll - float(getattr(rw, 'base_roll_target', 0.0))) * roll_scale
            cmd_parts.extend([cmd_pitch.unsqueeze(-1), cmd_roll.unsqueeze(-1)])
        cmd_obs = torch.cat(cmd_parts, dim=-1)

        dof_list = [0, 1, 2, 4, 5, 6]
        dof_pos = (self.dof_pos - self.default_dof_pos)[:, dof_list]

        proprioceptive_obs = torch.cat(
            (
                self.base_ang_vel * self.obs_scales.ang_vel,
                self.projected_gravity,
                cmd_obs,
                dof_pos * self.obs_scales.dof_pos,
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
                torch.zeros(cmd_obs.shape[1]),
                torch.ones(6) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                torch.ones(8) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
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

        th_scale = getattr(self.obs_scales, 'target_height', self.obs_scales.height_measurements)
        target_h_obs = (self.target_height.unsqueeze(-1) * th_scale).to(proprioceptive_obs.dtype)

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
            pure_obs_hist = self.obs_history_buf[:, :, :-self.num_actions].reshape(self.num_envs, -1)
            act_hist = self.action_history_buf.view(self.num_envs, -1)
            self.privileged_obs_buf = torch.cat([self.privileged_obs_buf, pure_obs_hist, act_hist], dim=-1)

    def _reset_root_states(self, env_ids):
        if self.custom_origins:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
            self.root_states[env_ids, :2] += torch_rand_float(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        self.root_states[env_ids, 2] += torch_rand_float(0.0, 0.1, (len(env_ids), 1), device=self.device).squeeze(1)
        zero_roll = torch.zeros(len(env_ids), device=self.device)
        zero_pitch = torch.zeros(len(env_ids), device=self.device)
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids), 1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(zero_roll, zero_pitch, random_yaw)
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device)
        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self.root_states), gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

    def check_termination(self):
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.0, dim=1)
        self.time_out_buf = self.episode_length_buf > self.max_episode_length
        self.reset_buf |= self.time_out_buf
        base_h = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        self.reset_buf |= base_h < 0

    def step(self, actions):
        self.action_history_buf = torch.cat([self.action_history_buf[:, 1:].clone(), actions[:, None, :].clone()], dim=1)
        actions = actions.to(self.device)
        clip_actions = self.cfg.normalization.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.render()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices] = 0
        self.post_physics_step()
        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.cost_buf, self.reset_buf, self.extras

    def _compute_torques(self, actions):
        if self.cfg.control.use_filter:
            actions = self._low_pass_action_filter(actions)
        actions_scaled = actions * self.cfg.control.action_scale
        actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:, 1:, :].clone(), actions_scaled.unsqueeze(1).clone()], dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes, self.randomized_lag, :] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == 'P':
            if not self.cfg.domain_rand.randomize_kpkd:
                torques = self.p_gains * (joint_pos_target - self.dof_pos) - self.d_gains * self.dof_vel
                torques[:, self.foot_joint_indices] = (
                    self.p_gains[self.foot_joint_indices] * actions_scaled[:, self.foot_joint_indices]
                    - self.d_gains[self.foot_joint_indices] * self.dof_vel[:, self.foot_joint_indices]
                )
            else:
                torques = self.kp_factor * self.p_gains * (joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains * self.dof_vel
                torques[:, self.foot_joint_indices] = (
                    self.kp_factor[:, self.foot_joint_indices] * self.p_gains[self.foot_joint_indices] * actions_scaled[:, self.foot_joint_indices]
                    - self.kd_factor[:, self.foot_joint_indices] * self.d_gains[self.foot_joint_indices] * self.dof_vel[:, self.foot_joint_indices]
                )
        else:
            raise NameError(f'Unknown controller type: {control_type}')
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _target_height_extreme_proximity(self):
        h_min = float(self.cfg.rewards.height_target_min)
        h_max = float(self.cfg.rewards.height_target_max)
        half_span = max(h_max - h_min, 1e-6) * 0.5
        mid = 0.5 * (h_min + h_max)
        return torch.clamp(torch.abs(self.target_height - mid) / half_span, 0.0, 1.0)

    def _reward_lin_vel_z(self):
        z_v = ((self.base_height - self.last_base_height) / self.dt)
        z_error = self.target_height - self.base_height
        z_mask = torch.logical_and(torch.abs(z_v) > 0.01, torch.abs(z_error) > 0.01).float()
        lin_cmd = self._lin_vel_in_command_frame()
        return self._command_gravity_gate() * torch.square(lin_cmd[:, 2]) * z_mask

    def _attitude_tracking_sigma(self):
        return float(getattr(self.cfg.rewards, 'attitude_tracking_sigma', self.cfg.rewards.tracking_sigma))

    def _reward_ang_vel_xy(self):
        # 仅在未指令俯仰/横滚变化时抑制多余 roll/pitch 角速度
        ang_cmd = self._ang_vel_in_command_frame()
        w_pitch_cmd = (self.target_pitch - self.last_target_pitch) / self.dt
        w_roll_cmd = (self.target_roll - self.last_target_roll) / self.dt
        excess = torch.stack([ang_cmd[:, 0] - w_roll_cmd, ang_cmd[:, 1] - w_pitch_cmd], dim=1)
        # gate only on rate commands; non-zero targets still need angular damping
        hold_att = (torch.abs(w_pitch_cmd) < 0.05) & (torch.abs(w_roll_cmd) < 0.05)
        return self._command_gravity_gate() * torch.sum(torch.square(excess), dim=1) * hold_att.float()

    def _reward_orientation(self):
        # 保留接口；主姿态跟踪见 tracking_pitch / tracking_roll
        return torch.zeros(self.num_envs, device=self.device)

    def _reward_tracking_pitch(self):
        _, pitch, _ = get_euler_xyz(self.base_quat)
        err = wrap_to_pi(pitch - self.target_pitch)
        return self._command_gravity_gate() * torch.exp(-torch.square(err) / (self._attitude_tracking_sigma()*4.0))

    def _reward_tracking_roll(self):
        roll, _, _ = get_euler_xyz(self.base_quat)
        err = wrap_to_pi(roll - self.target_roll)
        return self._command_gravity_gate() * torch.exp(-torch.square(err) / self._attitude_tracking_sigma())

    def _reward_torques(self):
        return self._command_gravity_gate() * torch.sum(torch.square(self.torques), dim=1)

    def _reward_powers(self):
        return self._command_gravity_gate() * torch.sum(torch.abs(self.torques) * torch.abs(self.dof_vel), dim=1)

    def _reward_dof_vel(self):
        leg_joints = [0, 1, 2, 4, 5, 6]
        return self._command_gravity_gate() * torch.sum(torch.square(self.dof_vel[:, leg_joints]), dim=1)

    def _reward_dof_wheels_vel(self):
        zero_planar = torch.norm(self.commands[:, :2], dim=1) < 0.1
        return (
            self._command_gravity_gate()
            * torch.sum(torch.square(self.dof_vel[:, self.foot_joint_indices]), dim=1)
            * zero_planar
        )

    def _reward_dof_wheel_action(self):
        zero_cmd = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        return (
            self._command_gravity_gate()
            * torch.sum(torch.square(self.actions[:, self.foot_joint_indices]), dim=1)
            * zero_cmd
        )

    def _reward_dof_acc(self):
        return self._command_gravity_gate() * torch.sum(torch.square((self.last_dof_vel - self.dof_vel) / self.dt), dim=1)

    def _reward_action_rate(self):
        return self._command_gravity_gate() * torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_action_smoothness(self):
        return (
            self._command_gravity_gate()
            * torch.sum(
                torch.square(
                    self.action_history_buf[:, -1, :]
                    - 2 * self.action_history_buf[:, -2, :]
                    + self.action_history_buf[:, -3, :]
                ),
                dim=1,
            )
        )

    def _reward_collision(self):
        return self._command_gravity_gate() * torch.sum(
            1.0 * (torch.norm(self.contact_forces[:, self.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
        )

    def _reward_dof_pos_limits(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return self._command_gravity_gate() * torch.sum(out_of_limits, dim=1)

    def _reward_tracking_lin_vel_x(self):
        lin_cmd = self._lin_vel_in_command_frame()
        # print(f"lin_cmd x: {lin_cmd[:, 0]}")
        lin_vel_x_error = torch.square(self.commands[:, 0] - lin_cmd[:, 0])
        base_rew =  torch.exp(-lin_vel_x_error / self.cfg.rewards.tracking_sigma_x_scale)
        return base_rew * self._tracking_cmd_factor(0, 'lin_vel_x')

    def _reward_tracking_lin_vel_y(self):
        lin_cmd = self._lin_vel_in_command_frame()
        # print(f"lin_cmd y: {lin_cmd[:, 1]}")
        lin_vel_y_error = torch.square(self.commands[:, 1] - lin_cmd[:, 1])
        base_rew = (
            torch.exp(-lin_vel_y_error / self.cfg.rewards.tracking_sigma_y_scale)
            * self._tracking_cmd_factor(1, 'lin_vel_y')
        )
        # When vy is commanded, only give full reward if at least one foot is in the air
        # (prevents policy from achieving vy purely by body tilting without stepping)
        vy_active = (torch.abs(self.commands[:, 1]) > 0.15).float()
        one_foot_up = (self.contact_filt.sum(dim=1) < 2).float()
        stepping_gate = vy_active * one_foot_up + (1.0 - vy_active)
        # print(f"stepping_gate: {stepping_gate}")
        # print(f"base_rew: {base_rew}")
        return base_rew * stepping_gate

    def _reward_tracking_ang_vel(self):
        ang_cmd = self._ang_vel_in_command_frame()
        ang_vel_error = torch.square(self.commands[:, 2] - ang_cmd[:, 2])
        return self._command_gravity_gate() * torch.exp(-ang_vel_error / float(self.cfg.rewards.tracking_sigma))

    def _reward_base_height(self):
        base_height = torch.mean(self.root_states[:, 2].unsqueeze(1) - self.measured_heights, dim=1)
        prox = self._target_height_extreme_proximity()
        boost = float(getattr(self.cfg.rewards, 'height_extreme_penalty_boost', 1.0))
        stress = 1.0 + boost * prox
        return self._command_gravity_gate() * stress * torch.square((base_height - self.target_height) / 0.05)

    def _reward_tracking_height_velocity(self):
        vz_cmd = (self.target_height - self.last_target_height) / self.dt
        lin_cmd = self._lin_vel_in_command_frame()
        vel_error = torch.square(vz_cmd - lin_cmd[:, 2])
        return self._command_gravity_gate() * torch.exp(-vel_error / 1e-2)

    def _reward_tracking_pitch_velocity(self):
        w_pitch_cmd = (self.target_pitch - self.last_target_pitch) / self.dt
        ang_cmd = self._ang_vel_in_command_frame()
        vel_error = torch.square(w_pitch_cmd - ang_cmd[:, 1])
        return self._command_gravity_gate() * torch.exp(-vel_error / 1e-2)

    def _reward_tracking_roll_velocity(self):
        w_roll_cmd = (self.target_roll - self.last_target_roll) / self.dt
        ang_cmd = self._ang_vel_in_command_frame()
        vel_error = torch.square(w_roll_cmd - ang_cmd[:, 0])
        return self._command_gravity_gate() * torch.exp(-vel_error / 1e-2)

    def _reward_upward(self):
        # gate off when non-zero attitude is commanded, otherwise fights tracking_pitch/roll
        near_upright = (torch.abs(self.target_pitch) < 0.05) & (torch.abs(self.target_roll) < 0.05)
        return torch.square(1 - self.projected_gravity[:, 2]) * near_upright.float()

    def _reward_stand_still(self):
        leg_joints = [0, 1, 2, 4, 5, 6]
        return (
            self._command_gravity_gate()
            * torch.sum(torch.abs(self.dof_pos[:, leg_joints] - self.default_dof_pos[:, leg_joints]), dim=1)
            * (torch.norm(self.commands[:, :2], dim=1) < 0.1)
        )

    def _reward_hip_pos(self):
        flag = 1.0 * (torch.abs(self.commands[:, 1]) < 0.1)
        return (
            self._command_gravity_gate()
            * flag
            * torch.sum(
                torch.square(self.dof_pos[:, self.hip_joint_indices] - self.default_dof_pos[:, self.hip_joint_indices]),
                dim=1,
            )
        )

    def _reward_body_feet_distance_x(self):
        foot_yaw = self._foot_distance_in_yaw_frame()
        foot_x_err = torch.abs(foot_yaw[:, 0])
        return foot_x_err ** 2

    def _reward_body_feet_distance_y(self):
        ry = self.command_ranges['lin_vel_y']
        max_vy_cmd = max(abs(float(ry[0])), abs(float(ry[1])), 1e-6)
        vy_n = torch.abs(self.commands[:, 1]) / max_vy_cmd
        mask_floor = float(getattr(self.cfg.rewards, 'body_feet_distance_y_cmd_mask_floor', 0.08))
        mask = torch.clamp(1.0 - vy_n, 0.0, 1.0) * (1.0 - mask_floor) + mask_floor
        foot_yaw = self._foot_distance_in_yaw_frame()
        foot_y_err = torch.abs(torch.abs(foot_yaw[:, 1]) - self.cfg.init_state.desired_feet_distance)
        return foot_y_err ** 2 * mask

    def _reward_lateral_step(self):
        """Reward yaw-frame lateral velocity matching vy sign while one foot is in swing.

        Do NOT use (L_foot - R_foot).y: standing stance already has positive separation,
        so sign(vy) * foot_sep_y is always >= 0 for vy > 0 and always 0 for vy < 0.
        """
        vy_cmd = self.commands[:, 1]
        vy_active = torch.abs(vy_cmd) > 0.15
        lin_cmd = self._lin_vel_in_command_frame()
        lateral_vel_align = torch.sign(vy_cmd) * lin_cmd[:, 1]
        one_foot_up = (self.contact_filt.sum(dim=1) < 2).float()
        return (
            vy_active.float()
            * one_foot_up
            * torch.clamp(lateral_vel_align, 0.0, None)
        )

    def _cost_torque_limit(self):
        return self._command_gravity_gate() * torch.sum(
            (torch.abs(self.torques) - self.torque_limits * self.cfg.rewards.soft_torque_limit).clip(min=0.0), dim=1
        )

    def _cost_pos_limit(self):
        out_of_limits = -(self.dof_pos - self.dof_pos_limits[:, 0]).clip(max=0.0)
        out_of_limits += (self.dof_pos - self.dof_pos_limits[:, 1]).clip(min=0.0)
        return self._command_gravity_gate() * torch.sum(out_of_limits, dim=1)

    def _cost_dof_vel_limits(self):
        leg_joints = [0, 1, 2, 4, 5, 6]
        return self._command_gravity_gate() * torch.sum(
            (
                torch.abs(self.dof_vel[:, leg_joints])
                - self.dof_vel_limits[leg_joints] * self.cfg.rewards.soft_dof_vel_limit
            ).clip(min=0.0, max=1.0),
            dim=1,
        )

    def _cost_hip_pos(self):
        return self._command_gravity_gate() * torch.sum(
            torch.square(self.dof_pos[:, self.hip_joint_indices] - 0.0), dim=-1
        )


class D1hLqrCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 4096
        n_scan = 187
        n_priv_latent = 2 + 1 + 4 + 1 + 1 + 8 + 8 + 8
        n_proprio = 34  # +2: target pitch / roll (integrated, in proprio cmd block)
        n_privileged_proprio = 38
        history_len = 10
        num_observations = n_proprio + history_len * n_proprio
        num_privileged_obs = n_privileged_proprio + n_scan + history_len * n_privileged_proprio + n_priv_latent
        num_actions = 8

    class cost(LeggedRobotCfg.cost):
        num_costs = 4

    class init_state(LeggedRobotCfg.init_state):
        random_dof_pos_probability = 0.0
        pos = [0.0, 0.0, 0.55]
        default_joint_angles = {
            'L_hip_joint': 0.0,
            'R_hip_joint': 0.0,
            'L_thigh_joint': 0.8,
            'R_thigh_joint': 0.8,
            'L_calf_joint': -1.5,
            'R_calf_joint': -1.5,
            'L_foot_joint': 0.0,
            'R_foot_joint': 0.0,
        }
        desired_feet_distance = 0.4

    class control(LeggedRobotCfg.control):
        control_type = 'P'
        stiffness = {'hip': 60.0, 'thigh': 60.0, 'calf': 60.0, 'foot': 10.0}
        damping = {'hip': 2.0, 'thigh': 2.0, 'calf': 2.0, 'foot': 0.5}
        action_scale = 0.5
        decimation = 4
        hip_scale_reduction = 0.5
        use_filter = True

    class normalization(LeggedRobotCfg.normalization):
        class obs_scales(LeggedRobotCfg.normalization.obs_scales):
            lin_vel_z_cmd = 5.0
            ang_vel_pitch_cmd = 2.0
            ang_vel_roll_cmd = 2.0
            pitch_angle_cmd = 2.0
            roll_angle_cmd = 5.0

    class commands(LeggedRobotCfg.control):
        curriculum = True
        max_curriculum = 3.0
        # vx, vy, yaw_rate, heading, height_rate, pitch_rate, roll_rate
        num_commands = 7
        resampling_time = 5.0
        heading_command = False
        global_reference = False
        zero_height_cmd_prob = 0.3
        zero_pitch_cmd_prob = 0.3
        zero_roll_cmd_prob = 0.3
        cmd_jitter_enabled = True
        cmd_jitter_interval_s = 0.5
        cmd_jitter_prob = 0.1
        # [x, y, xy, turn, x+turn, xy+turn, stand_still]
        commands_proportion = [0.25, 0.20, 0.1, 0.15, 0.15, 0.05, 0.1]
        # [pure_loco, loco+height, loco+attitude, all_dims]
        task_proportions = [0.40, 0.30, 0.20, 0.10]

        class ranges:
            lin_vel_x = [-1.0, 1.5]
            lin_vel_y = [-1.0, 1.0]
            ang_vel_yaw = [-1.0, 1.0]
            heading = [-3.14, 3.14]
            lin_vel_z = [-0.1, 0.1]
            ang_vel_pitch = [-0.5, 0.5]
            ang_vel_roll = [-0.25, 0.25]

    class asset(LeggedRobotCfg.asset):
        file = '{ROOT_DIR}/resources/d1h/urdf/d1h_new.urdf'
        foot_name = 'foot'
        name = 'd1h_new'
        penalize_contacts_on = ['thigh', 'calf', 'base']
        terminate_after_contacts_on = ['base']
        self_collisions = 0
        replace_cylinder_with_capsule = False
        flip_visual_attachments = False

    class rewards(LeggedRobotCfg.rewards):
        class scales(LeggedRobotCfg.rewards.scales):
            torques = 0.0
            powers = 0.0
            termination = 0.0
            tracking_lin_vel = 0.0
            tracking_lin_vel_x = 12.0
            tracking_lin_vel_y = 10.0
            tracking_ang_vel = 4.0
            tracking_height_velocity = 1.0
            tracking_pitch = 2.0
            tracking_roll = 2.0
            tracking_pitch_velocity = 0.1
            tracking_roll_velocity = 0.1
            lin_vel_z = -0.1
            orientation = 0.0
            ang_vel_xy = -0.02
            dof_pos_limits = -10.0
            dof_vel = 0.0
            dof_wheels_vel = -0.02
            dof_acc = -2.5e-7
            base_height = -0.5
            feet_air_time = 0.0
            collision = -1.0
            feet_stumble = 0.0
            action_rate = -0.01
            hip_pos = -0.2
            upward = 0.5
            dof_wheel_action = 0.0
            stand_still = 0.0
            body_feet_distance_x = -50.0
            body_feet_distance_y = -50.0
            lateral_step = 0.0

        only_positive_rewards = True
        tracking_sigma = 0.5
        tracking_sigma_x_scale = 0.25
        tracking_sigma_y_scale = 0.5
        # vx/vy tracking: factor = factor_min at |cmd|=0, 1.0 at |cmd|=range max
        tracking_cmd_factor_min = 0.5
        attitude_tracking_sigma = 0.15
        body_feet_distance_y_cmd_mask_floor = 0.08
        soft_dof_pos_limit = 0.9
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        base_height_target = 0.45
        base_pitch_target = 0.0
        base_roll_target = 0.0

        max_contact_force = 500.0
        height_target_min = 0.30
        height_target_max = 0.55
        pitch_target_min = -1.57
        pitch_target_max = 1.57
        roll_target_min = -0.8
        roll_target_max = 0.8
        height_extreme_penalty_boost = 0.5

    class costs:
        class scales:
            pos_limit = 1.0
            torque_limit = 1.0
            dof_vel_limits = 1.0
            hip_pos = 2.0

        class d_values:
            pos_limit = 0.0
            torque_limit = 0.0
            dof_vel_limits = 0.0
            hip_pos = 0.0

    class domain_rand(LeggedRobotCfg.domain_rand):
        randomize_friction = True
        friction_range = [0.1, 1.0]
        randomize_restitution = False
        restitution_range = [0.0, 1.0]
        randomize_base_mass = True
        added_mass_range = [-1.0, 3.0]
        randomize_base_com = True
        added_com_range = [-0.1, 0.1]
        push_robots = True
        push_interval_s = 15
        max_push_vel_xy = 1.0

        randomize_motor = True
        motor_strength_range = [0.8, 1.2]
        randomize_kpkd = True
        kp_range = [0.8, 1.2]
        kd_range = [0.8, 1.2]

        randomize_lag_timesteps = True
        lag_timesteps = 3

        disturbance = False
        disturbance_range = [-30.0, 30.0]
        disturbance_interval = 8

    class terrain(LeggedRobotCfg.terrain):
        mesh_type = 'plane'
        curriculum = True
        measure_heights = True
        include_act_obs_pair_buf = False
        static_friction = 0.4
        dynamic_friction = 0.4
        terrain_proportions = [0.6, 0.4, 0.0, 0.0, 0.0]


class D1hLqrCfg_Play(D1hLqrCfg):
    class env(D1hLqrCfg.env):
        num_envs = 10

    class terrain(D1hLqrCfg.terrain):
        num_rows = 5
        num_cols = 5
        curriculum = False

    class noise(D1hLqrCfg.noise):
        add_noise = False

    class control(D1hLqrCfg.control):
        use_filter = True

    class domain_rand(D1hLqrCfg.domain_rand):
        push_robots = False
        randomize_friction = False
        randomize_base_com = False
        randomize_base_mass = False
        randomize_motor = False
        randomize_lag_timesteps = False
        randomize_restitution = False
        disturbance = False
        randomize_kpkd = False

    class commands(D1hLqrCfg.commands):
        heading_command = False
        cmd_jitter_enabled = False

        class ranges:
            lin_vel_x = [1.0, 1.0]
            lin_vel_y = [0.0, 0.0]
            ang_vel_yaw = [0.0, 0.0]
            heading = [0.0, 0.0]
            lin_vel_z = [0.0, 0.0]
            ang_vel_pitch = [0.0, 0.0]
            ang_vel_roll = [0.0, 0.0]


class D1hLqrCfgPPO(LeggedRobotCfgPPO):
    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
        learning_rate = 1.0e-3
        max_grad_norm = 0.01
        num_learning_epochs = 5
        num_mini_batches = 4
        cost_value_loss_coef = 0.1
        cost_viol_loss_coef = 0.1

    class policy(LeggedRobotCfgPPO.policy):
        init_noise_std = 1.0
        scan_encoder_dims = [128, 64, 32]
        actor_hidden_dims = [512, 256, 128]
        critic_hidden_dims = [512, 256, 128]
        priv_encoder_dims = []
        activation = 'elu'
        num_costs = 4
        rnn_type = 'lstm'
        rnn_hidden_size = 512
        rnn_num_layers = 1
        teacher_act = True
        imi_flag = True

    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'd1h_lqr'
        experiment_name = 'd1h_lqr'
        policy_class_name = 'ActorCriticBarlowTwins'
        runner_class_name = 'OnConstraintPolicyRunner'
        algorithm_class_name = 'NP3O'
        max_iterations = 10000
        save_interval = 500
        num_steps_per_env = 24
        resume = False
        resume_path = ''
        # resume = True
        # resume_path = 'logs/d1h_lqr/May19_16-37-58_d1h_lqr/model_1000.pt'
