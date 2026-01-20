from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
import numpy as np
# config
from global_config import ROOT_DIR
from configs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO
from configs.base.legged_robot import LeggedRobot
from utils.math import wrap_to_pi

class Y2Command(LeggedRobot):
    def _init_buffers(self):

        super()._init_buffers()
        self.thigh_joint_indices = [0, 3, 6, 9]
        self.foot_joint_indices = [2, 5, 8, 11]
        print("dof_names",self.dof_names)
        print("feet_indices:",self.feet_indices)
        
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
        random_roll = torch_rand_float(0, 0, (len(env_ids),1), device=self.device).squeeze(1)
        random_pitch = torch_rand_float(0, 0, (len(env_ids),1), device=self.device).squeeze(1)
        random_yaw = torch_rand_float(-np.pi, np.pi, (len(env_ids),1), device=self.device).squeeze(1)
        self.root_states[env_ids, 3:7] = quat_from_euler_xyz(random_roll, random_pitch, random_yaw)

        # base velocities
        self.root_states[env_ids, 7:13] = torch_rand_float(-0.5, 0.5, (len(env_ids), 6), device=self.device) # [7:10]: lin vel, [10:13]: ang vel
        self.root_states[env_ids, 8:10] = 0.0  # zero y linear velocity

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self.root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
    def check_termination(self):
        """ Check if environments need to be reset
        """
        self.reset_buf = torch.any(torch.norm(self.contact_forces[:, self.termination_contact_indices, :], dim=-1) > 1.,dim=1)
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
            
        self.post_physics_step()

        clip_obs = self.cfg.normalization.clip_observations
        self.obs_buf = torch.clip(self.obs_buf, -clip_obs, clip_obs)
        if self.privileged_obs_buf is not None:
            self.privileged_obs_buf = torch.clip(self.privileged_obs_buf, -clip_obs, clip_obs)

        return self.obs_buf,self.privileged_obs_buf,self.rew_buf,self.cost_buf,self.reset_buf, self.extras

    #------------- Callbacks --------------
    def post_physics_step(self):
        """ check terminations, compute observations and rewards
            calls self._post_physics_step_callback() for common computations
            calls self._draw_debug_vis() if needed
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_force_sensor_tensor(self.sim)

        self.episode_length_buf += 1
        self.common_step_counter += 1

        # prepare quantities
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 7:10])
        self.base_ang_vel[:] = quat_rotate_inverse(self.base_quat, self.root_states[:, 10:13])
        self.projected_gravity[:] = quat_rotate_inverse(self.base_quat, self.gravity_vec)

        self.feet_pos = self.rigid_body_states[:, self.feet_indices, 0:3]
        self.feet_vel = self.rigid_body_states[:, self.feet_indices, 7:10]

        #self.roll, self.pitch, self.yaw = euler_from_quaternion(self.base_quat)
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        self.contact_filt = torch.logical_or(contact, self.last_contacts) 
        self.last_contacts = contact

        self._post_physics_step_callback()

        # g_y = self.projected_gravity[:, 1]
        # if self.common_step_counter % 200 == 0:
        #     print(f"[tilt] g_y={g_y[0]:+.3f}")

        # compute observations, rewards, resets, ...
        self.check_termination()
        self.compute_reward()
        self.compute_cost()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)

        self.compute_observations()

        self.last_actions[:] = self.actions[:]
        self.last_dof_vel[:] = self.dof_vel[:]
        self.last_torques[:] = self.torques[:]
        self.last_root_vel[:] = self.root_states[:, 7:13]

        if self.viewer and self.enable_viewer_sync and self.debug_viz:
            self._draw_debug_vis()

    def compute_observations(self):
        # 3 + 3 + 3 + 3 + 12 + 12+ 12 = 48
        obs_buf =torch.cat((self.base_lin_vel * self.obs_scales.lin_vel,
                            self.base_ang_vel  * self.obs_scales.ang_vel,
                            self.projected_gravity,
                            self.commands[:, :3] * self.commands_scale,
                            (self.dof_pos - self.default_dof_pos) * self.obs_scales.dof_pos,
                            self.dof_vel * self.obs_scales.dof_vel,
                            self.action_history_buf[:,-1]),dim=-1)

        noise_scales = self.cfg.noise.noise_scales
        noise_level = self.cfg.noise.noise_level
        noise_vec = torch.cat((torch.zeros(3),
                               torch.ones(3) * noise_scales.ang_vel * noise_level,
                               torch.ones(3) * noise_scales.gravity * noise_level,
                               torch.zeros(3),
                               torch.ones(
                                   self.cfg.env.num_actions) * noise_scales.dof_pos * noise_level * self.obs_scales.dof_pos,
                               torch.ones(
                                   self.cfg.env.num_actions) * noise_scales.dof_vel * noise_level * self.obs_scales.dof_vel,
                               torch.zeros(self.num_actions),
                               ), dim=0)
        
        if self.cfg.noise.add_noise:
            obs_buf += (2 * torch.rand_like(obs_buf) - 1) * noise_vec.to(self.device)

        priv_latent = torch.cat(( # 私有潜在状态
            # self.base_lin_vel * self.obs_scales.lin_vel,
            # self.reindex_feet(self.contact_filt.float()-0.5),   # 足端接触状态（4足）           *4
            self.contact_filt.float()-0.5,                      # 足端接触状态（4足）           *4
            self.randomized_lag_tensor,                         # 动作延迟参数（模拟响应延迟）    *1
            #self.base_ang_vel  * self.obs_scales.ang_vel,
            # self.base_lin_vel * self.obs_scales.lin_vel,
            self.mass_params_tensor,                            # 随机化的质量参数（躯干质量分布） *4
            self.friction_coeffs_tensor,                        # 随机化的地面摩擦系数           *1    
            self.restitution_coeffs_tensor,                     # 随机化的碰撞恢复系数           *1
            self.motor_strength,                                # 电机强度比例因子               *12   
            self.kp_factor,                                     # 位置环比例系数因子             *12
            self.kd_factor), dim=-1)                            # 微分环系数因子                *12
        
        # add perceptive inputs if not blind
        if self.cfg.terrain.measure_heights:
            heights = torch.clip(self.root_states[:, 2].unsqueeze(1) - 0.5 - self.measured_heights, -1, 1.)*self.obs_scales.height_measurements
            self.obs_buf = torch.cat([obs_buf, heights, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)
        else:
            self.obs_buf = torch.cat([obs_buf, priv_latent, self.obs_history_buf.view(self.num_envs, -1)], dim=-1)

        # update buffer
        self.obs_history_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([obs_buf] * self.cfg.env.history_len, dim=1),
            torch.cat([
                self.obs_history_buf[:, 1:],
                obs_buf.unsqueeze(1)
            ], dim=1)
        )

        self.contact_buf = torch.where(
            (self.episode_length_buf <= 1)[:, None, None], 
            torch.stack([self.contact_filt.float()] * self.cfg.env.contact_buf_len, dim=1),
            torch.cat([
                self.contact_buf[:, 1:],
                self.contact_filt.float().unsqueeze(1)
            ], dim=1)
        )

        if self.cfg.terrain.include_act_obs_pair_buf:
            # add to full observation history and action history to obs
            pure_obs_hist = self.obs_history_buf[:,:,:-self.num_actions].reshape(self.num_envs,-1)
            act_hist = self.action_history_buf.view(self.num_envs,-1)
            self.obs_buf = torch.cat([self.obs_buf,pure_obs_hist,act_hist], dim=-1)

    def _compute_torques(self, actions):
        self.dof_pos[:, self.foot_joint_indices]  = 0  # zero position of wheels
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
        # actions_scaled[:, self.hip_joint_indices] *= self.cfg.control.hip_scale_reduction

        if self.cfg.domain_rand.randomize_lag_timesteps:
            self.lag_buffer = torch.cat([self.lag_buffer[:,1:,:].clone(),actions_scaled.unsqueeze(1).clone()],dim=1)
            joint_pos_target = self.lag_buffer[self.num_envs_indexes,self.randomized_lag,:] + self.default_dof_pos
        else:
            joint_pos_target = actions_scaled + self.default_dof_pos

        control_type = self.cfg.control.control_type
        if control_type == "P":
            if not self.cfg.domain_rand.randomize_kpkd:  # TODO add strength to gain directly
                torques = self.p_gains*(joint_pos_target - self.dof_pos) - self.d_gains*self.dof_vel
                # torques[:,self.foot_joint_indices] = self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices] - self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]                
            else:
                torques = self.kp_factor * self.p_gains*(joint_pos_target - self.dof_pos) - self.kd_factor * self.d_gains*self.dof_vel
                # torques[:,self.foot_joint_indices] = self.kp_factor[:,self.foot_joint_indices]  * self.p_gains[self.foot_joint_indices] * actions_scaled[:,self.foot_joint_indices]- self.kd_factor[:,self.foot_joint_indices] *self.d_gains[self.foot_joint_indices] * self.dof_vel[:,self.foot_joint_indices]
        else: 
            raise NameError(f"Unknown controller type: {control_type}")
        
        torques *= self.motor_strength

        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def _init_yaw_curriculum(self):
        if hasattr(self, "_yaw_curriculum_scale"):
            return
        yaw_min, yaw_max = self.command_ranges["ang_vel_yaw"]
        self._yaw_curriculum_target = (float(yaw_min), float(yaw_max))
        self._yaw_curriculum_scale = 0.2
        self._yaw_curriculum_step = 0.1
        self.command_ranges["ang_vel_yaw"][0] = yaw_min * self._yaw_curriculum_scale
        self.command_ranges["ang_vel_yaw"][1] = yaw_max * self._yaw_curriculum_scale

    def _resample_commands(self, env_ids):
        if self.cfg.commands.curriculum and not self.cfg.commands.heading_command:
            self._init_yaw_curriculum()

        self.commands[env_ids, 0] = torch_rand_float(
            self.command_ranges["lin_vel_x"][0],self.command_ranges["lin_vel_x"][1],(len(env_ids), 1),device=self.device,).squeeze(1)
        if self.cfg.commands.heading_command:
            self.commands[env_ids, 3] = torch_rand_float(self.command_ranges["heading"][0],self.command_ranges["heading"][1],(len(env_ids), 1),device=self.device,).squeeze(1)
        else:
            self.commands[env_ids, 2] = torch_rand_float(self.command_ranges["ang_vel_yaw"][0],self.command_ranges["ang_vel_yaw"][1],(len(env_ids), 1),device=self.device,).squeeze(1)

        # set small commands to zero
        self.commands[env_ids, :2] *= (torch.norm(self.commands[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command_curriculum(self, env_ids):
        """ Implements a curriculum of increasing commands

        Args:
            env_ids (List[int]): ids of environments being reset
        """
        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(self.episode_sums["tracking_lin_vel_x"][env_ids]) / self.max_episode_length > 0.8 * self.reward_scales["tracking_lin_vel_x"]:
            self.command_ranges["lin_vel_x"][0] = np.clip(self.command_ranges["lin_vel_x"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            self.command_ranges["lin_vel_x"][1] = np.clip(self.command_ranges["lin_vel_x"][1] + 0.5, 0., self.cfg.commands.max_curriculum)
            # self.command_ranges["lin_vel_y"][0] = np.clip(self.command_ranges["lin_vel_y"][0] - 0.5, -self.cfg.commands.max_curriculum, 0.)
            # self.command_ranges["lin_vel_y"][1] = np.clip(self.command_ranges["lin_vel_y"][1] + 0.5, 0., self.cfg.commands.max_curriculum)

        if not self.cfg.commands.heading_command:
            if "tracking_ang_vel" in self.episode_sums and "tracking_ang_vel" in self.reward_scales:
                mean_tracking = torch.mean(self.episode_sums["tracking_ang_vel"][env_ids]) / self.max_episode_length
                if mean_tracking > 0.8 * self.reward_scales["tracking_ang_vel"]:
                    self._init_yaw_curriculum()
                    if self._yaw_curriculum_scale < 1.0:
                        self._yaw_curriculum_scale = min(
                            1.0, self._yaw_curriculum_scale + self._yaw_curriculum_step
                        )
                        self.command_ranges["ang_vel_yaw"][0] = (
                            self._yaw_curriculum_target[0] * self._yaw_curriculum_scale
                        )
                        self.command_ranges["ang_vel_yaw"][1] = (
                            self._yaw_curriculum_target[1] * self._yaw_curriculum_scale
                        )

    def reset_idx(self, env_ids):
        """ Reset some environments.
            Calls self._reset_dofs(env_ids), self._reset_root_states(env_ids), and self._resample_commands(env_ids)
            [Optional] calls self._update_terrain_curriculum(env_ids), self.update_command_curriculum(env_ids) and
            Logs episode info
            Resets some buffers

        Args:
            env_ids (list[int]): List of environment ids which must be reset
        """
        if len(env_ids) == 0:
            return
        # update curriculum
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
        # avoid updating command curriculum at each step since the maximum command is common to all envs
        if self.cfg.commands.curriculum and (self.common_step_counter % self.max_episode_length==0):
            self._update_command_curriculum(env_ids)

        # reset robot states
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)
        self._resample_commands(env_ids)
        self.gym.refresh_dof_state_tensor(self.sim)
        
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_net_contact_force_tensor(self.sim)

        # reset buffers
        self.last_actions[env_ids] = 0.
        self.last_dof_vel[env_ids] = 0.
        self.last_torques[env_ids] = 0.
        self.last_root_vel[env_ids] = 0.
        self.feet_air_time[env_ids] = 0.
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history_buf[env_ids, :, :] = 0.
        self.contact_buf[env_ids, :, :] = 0.
        self.action_history_buf[env_ids, :, :] = 0.

        # fill extras
        self.extras["episode"] = {}
        for key in self.episode_sums.keys():
            self.extras["episode"]['rew_' + key] = torch.mean(self.episode_sums[key][env_ids]) / self.max_episode_length_s
            self.episode_sums[key][env_ids] = 0.
        for key in self.cost_episode_sums.keys():
            self.extras["episode"]['cost_'+ key] = torch.mean(self.cost_episode_sums[key][env_ids]) / self.max_episode_length_s
            self.cost_episode_sums[key][env_ids] = 0.
        # log additional curriculum info
        if self.cfg.terrain.curriculum:
            self.extras["episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())
        if self.cfg.commands.curriculum:
            self.extras["episode"]["max_command_x"] = self.command_ranges["lin_vel_x"][1]
            self.extras["episode"]["max_command_yaw"] = self.command_ranges["ang_vel_yaw"][1]
            self.extras["episode"]["yaw_curriculum_scale"] = getattr(self, "_yaw_curriculum_scale", 1.0)
        # send timeout info to the algorithm
        if self.cfg.env.send_timeouts:
            self.extras["time_outs"] = self.time_out_buf

        # for i in range(len(self.lag_buffer)):
        #     self.lag_buffer[i][env_ids, :] = 0
        self.lag_buffer[env_ids,:,:] = 0

    def compute_reward(self):
        """ Compute rewards
            Calls each reward function which had a non-zero scale (processed in self._prepare_reward_function())
            adds each terms to the episode sums and to the total reward
        """
        self.rew_buf[:] = 0.
        for i in range(len(self.reward_functions)):
            name = self.reward_names[i]
            rew = self.reward_functions[i]() * self.reward_scales[name]
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.cfg.rewards.only_positive_rewards:
            self.rew_buf[:] = torch.clip(self.rew_buf[:], min=0.)
        # add termination reward after clipping
        if "termination" in self.reward_scales:
            rew = self._reward_termination() * self.reward_scales["termination"]
            self.rew_buf += rew
            self.episode_sums["termination"] += rew


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

        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum((torch.abs(self.dof_vel[:, [0,1,3,4,6,7,9,10]]) - self.dof_vel_limits[ [0,1,3,4,6,7,9,10]]*self.cfg.rewards.soft_dof_vel_limit).clip(min=0., max=1.), dim=1)
    
    def _cost_default_joint(self):
        # Penalize motion at zero commands
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.sum(torch.abs(self.dof_pos[:, [0,1,3,4,6,7,9,10]] - self.default_dof_pos[:,[0,1,3,4,6,7,9,10]]), dim=1)

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
        base_height = self._get_base_heights()
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.square(base_height - self.cfg.rewards.base_height_target)

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
    
    def _reward_tracking_ang_vel(self):
        # Tracking of angular velocity commands (yaw)
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.clamp(-self.projected_gravity[:,2],0,1)*torch.exp(-ang_vel_error/self.cfg.rewards.tracking_sigma)

    def _reward_upward(self):
        # print(self.projected_gravity[:,2])
        return 1 - torch.clamp(self.projected_gravity[:,2], -1, 1)
    
    def _reward_feet_all_contact(self):
        contact = self.contact_forces[:, self.feet_indices, 2] > 1.
        return torch.clamp(-self.projected_gravity[:,2],0,1)*0.25 * torch.sum(contact, dim=1)

    def _reward_stand_still(self):
        cmd_small = (torch.norm(self.commands[:, :2], dim=1) < 0.1).float()
        deviation = torch.mean(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)
        reward = torch.exp(-deviation)
        return cmd_small * reward
    
