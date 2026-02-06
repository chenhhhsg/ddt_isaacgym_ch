import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from multiprocessing import Process, Value


class Logger:
    def __init__(self, dt):
        self.state_log = defaultdict(list)
        self.rew_log = defaultdict(list)
        self.dt = dt
        self.num_episodes = 0
        self.plot_process = None
        self.joint_names = None

    def log_state(self, key, value):
        self.state_log[key].append(value)

    def log_states(self, dict):
        for key, value in dict.items():
            if key == "joint_names":
                self.joint_names = list(value)
                continue
            self.log_state(key, value)

    def set_joint_names(self, joint_names):
        self.joint_names = list(joint_names)

    def log_rewards(self, dict, num_episodes):
        for key, value in dict.items():
            if 'rew' in key:
                self.rew_log[key].append(value.item() * num_episodes)
        self.num_episodes += num_episodes

    def reset(self):
        self.state_log.clear()
        self.rew_log.clear()

    def plot_states(self, save_path=None):
        if save_path:
            self._plot(save_path)
            return
        self.plot_process = Process(target=self._plot, args=(save_path,))
        self.plot_process.start()

    def _plot(self, save_path=None):
        nb_rows = 3
        nb_cols = 3
        fig, axs = plt.subplots(nb_rows, nb_cols)
        log = self.state_log
        time = None
        for key in (
            "dof_pos",
            "dof_vel",
            "base_vel_x",
            "command_x",
            "base_vel_y",
            "command_y",
            "base_vel_yaw",
            "command_yaw",
            "base_vel_z",
        ):
            if log[key]:
                time = np.linspace(0, len(log[key]) * self.dt, len(log[key]))
                break
        if time is None:
            time = np.array([])
        # plot joint targets and measured positions
        a = axs[1, 0]
        if log["dof_pos"]: a.plot(time, log["dof_pos"], label='measured')
        if log["dof_pos_target"]: a.plot(time, log["dof_pos_target"], label='target')
        a.set(xlabel='time [s]', ylabel='Position [rad]', title='DOF Position')
        a.legend()
        # plot joint velocity
        a = axs[1, 1]
        if log["dof_vel"]:
            dof_vel_rpm = np.array(log["dof_vel"]) * 60.0 / (2.0 * np.pi)
            a.plot(time, dof_vel_rpm, label='measured')
        if log["dof_vel_target"]:
            dof_vel_tgt_rpm = np.array(log["dof_vel_target"]) * 60.0 / (2.0 * np.pi)
            a.plot(time, dof_vel_tgt_rpm, label='target')
        a.set(xlabel='time [s]', ylabel='Velocity [rpm]', title='Joint Velocity')
        a.legend()
        # plot base vel x
        a = axs[0, 0]
        if log["base_vel_x"]: a.plot(time, log["base_vel_x"], label='measured')
        if log["command_x"]: a.plot(time, log["command_x"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity x')
        a.legend()
        # plot base vel y
        a = axs[0, 1]
        if log["base_vel_y"]: a.plot(time, log["base_vel_y"], label='measured')
        if log["command_y"]: a.plot(time, log["command_y"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity y')
        a.legend()
        # plot base vel yaw
        a = axs[0, 2]
        if log["base_vel_yaw"]: a.plot(time, log["base_vel_yaw"], label='measured')
        if log["command_yaw"]: a.plot(time, log["command_yaw"], label='commanded')
        a.set(xlabel='time [s]', ylabel='base ang vel [rad/s]', title='Base velocity yaw')
        a.legend()
        # plot base vel z
        a = axs[1, 2]
        if log["base_vel_z"]: a.plot(time, log["base_vel_z"], label='measured')
        a.set(xlabel='time [s]', ylabel='base lin vel [m/s]', title='Base velocity z')
        a.legend()
        # plot robot mass (fallback to contact forces if mass not logged)
        a = axs[2, 0]
        if log["robot_mass"]:
            a.plot(time, log["robot_mass"], label='mass')
            a.set(xlabel='time [s]', ylabel='Mass [kg]', title='Robot Mass')
        elif log["contact_forces_z"]:
            forces = np.array(log["contact_forces_z"])
            for i in range(forces.shape[1]):
                a.plot(time, forces[:, i], label=f'force {i}')
            a.set(xlabel='time [s]', ylabel='Forces z [N]', title='Vertical Contact forces')
        a.legend()
        # plot torque/vel curves
        a = axs[2, 1]
        if log["dof_vel"] != [] and log["dof_torque"] != []:
            dof_vel_rpm = np.array(log["dof_vel"]) * 60.0 / (2.0 * np.pi)
            a.plot(dof_vel_rpm, log["dof_torque"], 'x', label='measured')
        a.set(xlabel='Joint vel [rpm]', ylabel='Joint Torque [Nm]', title='Torque/velocity curves')
        a.legend()
        # plot torque
        # a = axs[2, 2]
        # if log["dof_torque"] != []: a.plot(time, log["dof_torque"], label='measured')
        # a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title='Torque')
        # a.legend()
        # plot torques for each joint

        # base height
        a = axs[2, 2]
        if log["base_height"] : a.plot(time, log["base_height"], label='measured')
        if log["command_height"]: a.plot(time, log["command_height"], label='target')
        a.set(xlabel='time [s]', ylabel='base height [m]', title='Base height')
        a.legend()

        # per-joint torque/speed over time
        fig2 = None
        fig3 = None
        fig4 = None
        if log["torques"]:
            num_joints = len(log["torques"][0])
            rows, cols = 3, 4
            fig2, axs2 = plt.subplots(rows, cols, figsize=(12, 8))
            max_plots = rows * cols
            for joint_idx in range(min(num_joints, max_plots)):
                a = axs2[joint_idx // cols, joint_idx % cols]
                joint_torques = [torque[joint_idx] for torque in log["torques"]]
                if self.joint_names and joint_idx < len(self.joint_names):
                    title_name = self.joint_names[joint_idx]
                else:
                    title_name = f'Joint {joint_idx}'
                a.plot(time, joint_torques, label='torque')
                a.set(xlabel='time [s]', ylabel='Joint Torque [Nm]', title=title_name)
                a.legend()
                if log["velocities"]:
                    a2 = a.twinx()
                    joint_velocities = [vel[joint_idx] for vel in log["velocities"]]
                    rpm = np.array(joint_velocities) * 60.0 / (2.0 * np.pi)
                    a2.plot(time, rpm, 'r--', label='Speed')
                    a2.set_ylabel('Speed [rpm]', color='r')
                    a2.legend()

            # per-joint power over time with max power limits
            if log["velocities"]:
                fig3, axs3 = plt.subplots(rows, cols, figsize=(12, 8))
                for joint_idx in range(min(num_joints, max_plots)):
                    a = axs3[joint_idx // cols, joint_idx % cols]
                    joint_torques = np.array([torque[joint_idx] for torque in log["torques"]])
                    joint_velocities = np.array([vel[joint_idx] for vel in log["velocities"]])
                    joint_power = np.abs(joint_torques * joint_velocities)
                    if self.joint_names and joint_idx < len(self.joint_names):
                        title_name = self.joint_names[joint_idx]
                        is_wheel = "wheel" in title_name
                    else:
                        title_name = f'Joint {joint_idx}'
                        is_wheel = False
                    max_power = 20 * 30 if is_wheel else 24 * 6
                    a.plot(time, joint_power, label='power')
                    a.axhline(max_power, color='r', linestyle='--', linewidth=1, label='max power')
                    a.set(xlabel='time [s]', ylabel='Power [W]', title=title_name)
                    a.legend()

        fig.tight_layout()
        if fig2 is not None:
            fig2.tight_layout()
        if fig3 is not None:
            fig3.tight_layout()
        if log["wheel_left_vel"] or log["wheel_right_vel"]:
            fig4, axs4 = plt.subplots(5, 1, figsize=(10, 12), sharex=True)
            left = np.array(log["wheel_left_vel"])
            right = np.array(log["wheel_right_vel"])
            a = axs4[0]
            if left.size:
                a.plot(time, left, label='left wheel')
            if right.size:
                a.plot(time, right, label='right wheel')
            limit = None
            if log["wheel_vel_limit"]:
                limit = float(np.nanmean(log["wheel_vel_limit"]))
                if not np.isfinite(limit):
                    limit = None
            if limit is not None:
                a.axhline(limit, color='r', linestyle='--', linewidth=1, label='limit')
                a.axhline(-limit, color='r', linestyle='--', linewidth=1)
                if left.size:
                    hit_left = np.abs(left) >= 0.98 * limit
                    if np.any(hit_left):
                        a.scatter(time[hit_left], left[hit_left], s=10, color='r', label='left hit')
                if right.size:
                    hit_right = np.abs(right) >= 0.98 * limit
                    if np.any(hit_right):
                        a.scatter(time[hit_right], right[hit_right], s=10, color='m', label='right hit')
            a.set(ylabel='wheel speed [rad/s]', title='Wheel speeds')
            a.legend()

            a = axs4[1]
            if log["base_vel_x"]:
                a.plot(time, log["base_vel_x"], label='base vel x')
            if log["command_x"]:
                a.plot(time, log["command_x"], label='cmd x')
            a.set(ylabel='m/s', title='Base linear velocity vs command')
            a.legend()

            a = axs4[2]
            if log["base_vel_yaw"]:
                a.plot(time, log["base_vel_yaw"], label='base yaw vel')
            if log["command_yaw"]:
                a.plot(time, log["command_yaw"], label='cmd yaw')
            a.set(ylabel='rad/s', title='Base yaw velocity vs command')
            a.legend()

            a = axs4[3]
            if log["base_roll_deg"]:
                a.plot(time, log["base_roll_deg"], label='base roll')
            a.set(ylabel='deg', title='Base roll')
            a.legend()

            a = axs4[4]
            if log["wheel_diff_vel"]:
                a.plot(time, log["wheel_diff_vel"], label='right - left')
            a.set(xlabel='time [s]', ylabel='diff [rad/s]', title='Wheel speed difference')
            a.legend()
            fig4.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=150)
            if fig2 is not None:
                base, ext = os.path.splitext(save_path)
                fig2.savefig(f"{base}_torques{ext}", dpi=150)
            if fig3 is not None:
                base, ext = os.path.splitext(save_path)
                fig3.savefig(f"{base}_power{ext}", dpi=150)
            if fig4 is not None:
                base, ext = os.path.splitext(save_path)
                fig4.savefig(f"{base}_wheel_speeds{ext}", dpi=150)
            print(f"Saved plot to {save_path}")
            return
        plt.show()

    def print_rewards(self):
        print("Average rewards per second:")
        for key, values in self.rew_log.items():
            mean = np.sum(np.array(values)) / self.num_episodes
            print(f" - {key}: {mean}")
        print(f"Total number of episodes: {self.num_episodes}")

    def __del__(self):
        if self.plot_process is not None:
            self.plot_process.kill()
