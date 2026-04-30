from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
import os
import random
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi

class D1HHeightCommand(LeggedRobot):
    def _init_buffers(self):
        super()._init_buffers()
        self.commands_scale = torch.tensor(
            [
                self.obs_scales.lin_vel,
                self.obs_scales.lin_vel,
                self.obs_scales.ang_vel,
                1.0,
            ],
            device=self.device,
            requires_grad=False,
        )
        self.hip_joint_indices = [0, 4]
        self.thigh_joint_indices = [1, 5]
        self.foot_joint_indices = [3, 7]
    
    def _create_envs(self):
        """ Creates environments:
             1. loads the robot URDF/MJCF asset,
             2. For each environment
                2.1 creates the environment, 
                2.2 calls DOF and Rigid shape properties callbacks,
                2.3 create actor with these properties and add them to the env
             3. Store indices of different bodies of the robot
        """
        asset_path = self.cfg.asset.file.format(ROOT_DIR=ROOT_DIR)
        asset_root = os.path.dirname(asset_path)
        asset_file = os.path.basename(asset_path)

        asset_options = gymapi.AssetOptions()
        asset_options.default_dof_drive_mode = self.cfg.asset.default_dof_drive_mode
        asset_options.collapse_fixed_joints = self.cfg.asset.collapse_fixed_joints
        asset_options.replace_cylinder_with_capsule = self.cfg.asset.replace_cylinder_with_capsule
        asset_options.flip_visual_attachments = self.cfg.asset.flip_visual_attachments
        asset_options.fix_base_link = self.cfg.asset.fix_base_link
        asset_options.density = self.cfg.asset.density
        asset_options.angular_damping = self.cfg.asset.angular_damping
        asset_options.linear_damping = self.cfg.asset.linear_damping
        asset_options.max_angular_velocity = self.cfg.asset.max_angular_velocity
        asset_options.max_linear_velocity = self.cfg.asset.max_linear_velocity
        asset_options.armature = self.cfg.asset.armature
        asset_options.thickness = self.cfg.asset.thickness
        asset_options.disable_gravity = self.cfg.asset.disable_gravity

        robot_asset = self.gym.load_asset(self.sim, asset_root, asset_file, asset_options)
        self.num_dof = self.gym.get_asset_dof_count(robot_asset)
        self.num_bodies = self.gym.get_asset_rigid_body_count(robot_asset)
        dof_props_asset = self.gym.get_asset_dof_properties(robot_asset)
        rigid_shape_props_asset = self.gym.get_asset_rigid_shape_properties(robot_asset)

        # save body names from the asset
        body_names = self.gym.get_asset_rigid_body_names(robot_asset)
        self.dof_names = self.gym.get_asset_dof_names(robot_asset)
        self.num_bodies = len(body_names)
        self.num_dofs = len(self.dof_names)
        feet_names = [s for s in body_names if self.cfg.asset.foot_name in s]

        for s in feet_names:
            feet_idx = self.gym.find_asset_rigid_body_index(robot_asset, s)
            sensor_pose = gymapi.Transform(gymapi.Vec3(0.0, 0.0, 0.0))
            self.gym.create_asset_force_sensor(robot_asset, feet_idx, sensor_pose)
        
        penalized_contact_names = []
        for name in self.cfg.asset.penalize_contacts_on:
            penalized_contact_names.extend([s for s in body_names if name in s])
        penalized_contact_head_names = []
        for name in self.cfg.asset.penalize_contact_head_on:
            penalized_contact_head_names.extend([s for s in body_names if name in s])
        termination_contact_names = []
        for name in self.cfg.asset.terminate_after_contacts_on:
            termination_contact_names.extend([s for s in body_names if name in s])

        base_init_state_list = self.cfg.init_state.pos + self.cfg.init_state.rot + self.cfg.init_state.lin_vel + self.cfg.init_state.ang_vel
        self.base_init_state = to_torch(base_init_state_list, device=self.device, requires_grad=False)
        start_pose = gymapi.Transform()
        start_pose.p = gymapi.Vec3(*self.base_init_state[:3])

        self._get_env_origins()
        env_lower = gymapi.Vec3(0., 0., 0.)
        env_upper = gymapi.Vec3(0., 0., 0.)
        self.actor_handles = []
        self.envs = []
        self.cam_handles = []
        self.cam_tensors = []
        self.mass_params_tensor = torch.zeros(self.num_envs, 4, dtype=torch.float, device=self.device, requires_grad=False)

        print("Creating env...")
        for i in range(self.num_envs):
            # create env instance
            env_handle = self.gym.create_env(self.sim, env_lower, env_upper, int(np.sqrt(self.num_envs)))
            pos = self.env_origins[i].clone()
            pos[:2] += torch_rand_float(-1., 1., (2,1), device=self.device).squeeze(1)
            pos[2] += self.base_init_state[2]
            start_pose.p = gymapi.Vec3(*pos)
            rigid_shape_props = self._process_rigid_shape_props(rigid_shape_props_asset, i)
            self.gym.set_asset_rigid_shape_properties(robot_asset, rigid_shape_props)
            actor_handle = self.gym.create_actor(env_handle, robot_asset, start_pose, self.cfg.asset.name, i, self.cfg.asset.self_collisions, 0)
            dof_props = self._process_dof_props(dof_props_asset, i)
            self.gym.set_actor_dof_properties(env_handle, actor_handle, dof_props)
            body_props = self.gym.get_actor_rigid_body_properties(env_handle, actor_handle)
            body_props, mass_params = self._process_rigid_body_props(body_props, i)
            self.gym.set_actor_rigid_body_properties(env_handle, actor_handle, body_props, recomputeInertia=True)
            self.envs.append(env_handle)
            self.actor_handles.append(actor_handle)
            self.mass_params_tensor[i, :] = torch.from_numpy(mass_params).to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_friction:
            self.friction_coeffs_tensor = self.friction_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            friction_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].friction
            self.friction_coeffs_tensor = friction_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_restitution:
            self.restitution_coeffs_tensor = self.restitution_coeffs.to(self.device).to(torch.float).squeeze(-1)
        else:
            restitution_coeffs_tensor = torch.ones(self.num_envs,1)*rigid_shape_props_asset[0].restitution
            self.restitution_coeffs_tensor = restitution_coeffs_tensor.to(self.device).to(torch.float)

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [random.randint(0,self.cfg.domain_rand.lag_timesteps-1) for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False
        else:
            self.num_envs_indexes = list(range(0,self.num_envs))
            self.randomized_lag = [self.cfg.domain_rand.lag_timesteps-1 for i in range(self.num_envs)]
            self.randomized_lag_tensor = torch.FloatTensor(self.randomized_lag).view(-1,1)/(self.cfg.domain_rand.lag_timesteps-1)
            self.randomized_lag_tensor = self.randomized_lag_tensor.to(self.device)
            self.randomized_lag_tensor.requires_grad_ = False

        self.feet_indices = torch.zeros(len(feet_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(feet_names)):
            self.feet_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], feet_names[i])

        self.penalised_contact_indices = torch.zeros(len(penalized_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_names)):
            self.penalised_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_names[i])

        self.termination_contact_indices = torch.zeros(len(termination_contact_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(termination_contact_names)):
            self.termination_contact_indices[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], termination_contact_names[i])

        self.penalised_contact_head_index = torch.zeros(len(penalized_contact_head_names), dtype=torch.long, device=self.device, requires_grad=False)
        for i in range(len(penalized_contact_head_names)):
            self.penalised_contact_head_index[i] = self.gym.find_actor_rigid_body_handle(self.envs[0], self.actor_handles[0], penalized_contact_head_names[i])

    def _reset_root_states(self, env_ids):
        """ Resets ROOT states position and velocities of selected environmments
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
            self.root_states[env_ids, :2] += torch_rand_float(-1., 1., (len(env_ids), 2), device=self.device) # xy position within 1m of the center
            self.root_states[env_ids, 2] += torch_rand_float(0., 0.2, (len(env_ids), 1), device=self.device).squeeze(1) # z position within 0.2m of the center
        else:
            self.root_states[env_ids] = self.base_init_state
            self.root_states[env_ids, :3] += self.env_origins[env_ids]
        # base rotation
        random_roll = torch_rand_float(-0.15, 0.15, (len(env_ids),1), device=self.device).squeeze(1)
        random_pitch = torch_rand_float(-0.15, 0.15, (len(env_ids),1), device=self.device).squeeze(1)
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)
        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel

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
        # step physics and render each frame
        self.render()

        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions).view(self.torques.shape)
            self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self.torques))
            self.gym.simulate(self.sim)
            self.gym.fetch_results(self.sim, True)
            self.gym.refresh_dof_state_tensor(self.sim)
            self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels 
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
                torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        torques *= self.motor_strength
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self):
        # 3 + 3 + 3 + 4 + 8 + 8 + 8 = 37
        obs_buf = torch.cat((
            self.base_lin_vel * self.obs_scales.lin_vel,
            self.base_ang_vel * self.obs_scales.ang_vel,
            self.projected_gravity,
            self.commands[:, [0, 1, 2, 4]] * self.commands_scale,
            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
            self.dof_vel * self.obs_scales.dof_vel,
            self.action_history_buf[:, -1],
        ), dim=-1)

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((
            torch.zeros(3),
            torch.ones(3) * noise_scales.ang_vel * noise_level,
            torch.ones(3) * noise_scales.gravity * noise_level,
            torch.zeros(4),
            torch.ones(self.cfg.env.num_actions) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
            torch.ones(self.cfg.env.num_actions) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
            torch.zeros(self.num_actions),
        ), dim=0)

        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

        priv_latent = torch.cat((
            self.contact_filt.float() - 0.5,
            self.randomized_lag_tensor,
            self.mass_params_tensor,
            self.friction_coeffs_tensor,
            self.restitution_coeffs_tensor,
            self.motor_strength,
            self.kp_factor,
            self.kd_factor,
        ), dim=-1)

        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.) * self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1),
            ], dim=1),
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None],
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1),
            ], dim=1),
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            pure_obs_hist = self.obs_history_buf[:, :, :-self.num_actions].reshape(self.num_envs, -1)
            act_hist = self.action_history_buf.view(self.num_envs, -1)
            self.obs_buf = torch.cat([self.obs_buf, pure_obs_hist, act_hist], dim=-1)

    def _post_physics_step_callback(self):
        env_ids = (self.episode_length_buf % int(self.cfg.commands.resampling_time / self.dt) == 0).nonzero(as_tuple=False).flatten()
        self._resample_commands(env_ids)

        if self.cfg.commands.heading_command:
            forward = quat_apply(self.base_quat, self.forward_vec)
            heading = torch.atan2(forward[:, 1], forward[:, 0])
            self.commands[:, 2] = torch.clip(0.5 * wrap_to_pi(self.commands[:, 3] - heading), -1., 1.)

        if self.cfg.terrain.measure_heights:
            self.measured_heights = self._get_heights()
            self.feet_heights = self._get_feet_heights()
            self.feet_body_frame_height = self._get_feet_local_heights()

        if self.cfg.domain_rand.push_robots and (self.common_step_counter % self.cfg.domain_rand.push_interval == 0):
            self._push_robots()

        if self.cfg.domain_rand.disturbance and (self.common_step_counter % self.cfg.domain_rand.disturbance_interval == 0):
            self._disturbance_robots()

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return

        self.commands[env_ids, 0] = torch_rand_float(self.command_ranges["lin_vel_x"][0], self.command_ranges["lin_vel_x"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        self.commands[env_ids, 1] = torch_rand_float(self.command_ranges["lin_vel_y"][0], self.command_ranges["lin_vel_y"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        num_envs = len(env_ids)
        self.commands[env_ids, 4] = torch_rand_float(self.command_ranges["base_height"][0], self.command_ranges["base_height"][1], (num_envs, 1), device=self.device).squeeze(1)

        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0], self.command_ranges["heading"][1], (len(env_ids), 1), device=self.device).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0], self.command_ranges["ang_vel_yaw"][1], (len(env_ids), 1), device=self.device).squeeze(1)

        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if "tracking_lin_vel" not in self.reward_scales:
            if "tracking_lin_vel_x" in self.reward_scales:
                if torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_x"]:
                    self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum_x_back, 0.)
                    self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum_x)
            
            if "tracking_lin_vel_y" in self.reward_scales:
                if torch.mean(self.episode_sums["tracking_lin_vel_y"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_y"]:
                    self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum_y, 0.)
                    self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum_y)

        elif "tracking_lin_vel" in self.reward_scales:
            if torch.mean(self.episode_sums["tracking_lin_vel"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel"]:
                self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
                self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

    #------------ reward functions----------------
    def _reward_tracking_lin_vel_x(self):
        # Tracking of linear velocity commands (x axis)
        base_height_sigma = torch.clamp(self._get_base_heights()/self.cfg.rewards.base_height_target, 0, 1) + 0.2
        lin_vel_x_error = torch.clamp(torch.square(self.commands[:, 0] - self.base_lin_vel[:, 0]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 0]))/(0.25+torch.abs(self.commands[:, 0]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_x_error/tracking_sigma)*base_height_sigma
        return reward
    
    def _reward_tracking_lin_vel_y(self):
        # Tracking of linear velocity commands (y axis)
        base_height_sigma = torch.clamp(self._get_base_heights()/self.cfg.rewards.base_height_target, 0, 1) + 0.2
        lin_vel_y_error = torch.clamp(torch.square(self.commands[:, 1] - self.base_lin_vel[:, 1]), 0, 1)
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 1]))/(0.25+torch.abs(self.commands[:, 1]))
        reward = torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-lin_vel_y_error/tracking_sigma)*base_height_sigma
        return reward

    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        base_height_sigma = torch.clamp(self._get_base_heights()/self.cfg.rewards.base_height_target, 0, 1) + 0.2
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        tracking_sigma = self.cfg.rewards.tracking_sigma * (0.1+torch.abs(self.commands[:, 2]))/(0.25+torch.abs(self.commands[:, 2]))
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/tracking_sigma)*base_height_sigma
    
    def _reward_feet_air_time(self):
        # Reward long steps
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact
        first_contact = (self.feet_air_time > 0.) * contact_filt
        self.feet_air_time += self.dt
        rew_airTime = torch.sum((self.feet_air_time - 0.5) * first_contact, dim=1) 
        rew_airTime *= torch.norm(self.commands[:, :2], dim=1) > 0.1 
        self.feet_air_time *= ~contact_filt
        return rew_airTime

    def _reward_upward(self):
        return 1 - torch.clamp(self.projected_gravity[:,2], -1, 1)

    def _reward_keep_still(self):
        reward = (torch.norm(self.commands[:, :2], dim=1) < 0.1) * (1+10*torch.norm(self.base_lin_vel[:, :2], dim=1))
        return reward

    def _reward_body_pos_to_feet_x(self):
        # keep body relative position to Los small
        base_derivation = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        base_derivation_xyz = torch.zeros_like(base_derivation[:,:,:])
        
        for i in range(base_derivation.shape[1]):
            base_derivation_xyz[:, i, :] = quat_rotate_inverse(self.base_quat, base_derivation[:, i, :])
        
        distance_x = torch.abs(torch.mean(base_derivation_xyz[:,:,0], dim=1))
        reward = torch.exp(-distance_x / self.cfg.rewards.distance_sigma)
        return reward

    def _reward_body_feet_distance_x(self):
        foot_distance_world = self.feet_pos[:,0,:]-self.feet_pos[:,1,:]
        foot_distance_base = quat_rotate_inverse(self.base_quat, foot_distance_world)
        foot_x_err = torch.abs(foot_distance_base[:,0])/self.cfg.rewards.distance_sigma
        reward = foot_x_err**2
        return reward

    def _reward_body_feet_distance_y(self):
        foot_distance_world = self.feet_pos[:,0,:]-self.feet_pos[:,1,:] 
        foot_distance_base = quat_rotate_inverse(self.base_quat, foot_distance_world)
        foot_y_err = torch.abs(torch.abs(foot_distance_base[:,1])-self.cfg.init_state.desired_feet_distance)/self.cfg.rewards.distance_sigma
        reward = foot_y_err**2
        return reward

    def _reward_body_symmetry_y(self):
        foot_position_base_world = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_y_err = torch.abs(torch.abs(foot1_base[:, 1]) - torch.abs(foot2_base[:, 1]))
        reward = torch.exp(-symmetry_y_err / self.cfg.rewards.distance_sigma)
        return reward

    def _reward_body_symmetry_z(self):
        foot_position_base_world = self.feet_pos - self.root_states[:, 0:3].unsqueeze(1)
        foot1_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 0, :])
        foot2_base = quat_rotate_inverse(self.base_quat, foot_position_base_world[:, 1, :])
        symmetry_z_err = torch.abs(torch.abs(foot1_base[:, 2]) - torch.abs(foot2_base[:, 2]))
        reward = torch.exp(-symmetry_z_err / self.cfg.rewards.distance_sigma)
        return reward

    def _reward_collision_head(self):
        head_contact_force = torch.norm(self.contact_forces[:, self.penalised_contact_head_index, :], dim=-1)   
        return torch.sum(1.*(head_contact_force > 10), dim=1)

    def _reward_dof_thigh_vel(self):
        return torch.sum(torch.square(self.dof_vel[:, self.thigh_joint_indices]), dim=1)
    
    # # ------------ cost functions----------------
    def _cost_hip_pos(self):
        return torch.sum(torch.square(self.dof_pos[:, self.hip_joint_indices] - 0.0),dim=-1)
    
    def _cost_default_joint(self):
        # Penalize motion at zero commands
        non_hip_foot_indices = [i for i in range(self.num_dof) if i not in self.hip_joint_indices and i not in self.foot_joint_indices]
        return torch.sum(torch.abs(self.dof_pos[:, non_hip_foot_indices] - self.default_dof_pos[:,non_hip_foot_indices]), dim=1)
