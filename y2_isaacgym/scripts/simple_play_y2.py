import cv2
import os
import sys
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs import *
from isaacgym import gymapi
from modules import *
from utils import  get_args, export_policy_as_jit, task_registry, Logger, get_load_path
from utils.helpers import class_to_dict
from utils.task_registry import task_registry
import numpy as np
import torch
from global_config import ROOT_DIR
import xml.etree.ElementTree as ET


def _get_wheel_vel_limit_rad_s(urdf_path, wheel_joint_names):
    try:
        root = ET.parse(urdf_path).getroot()
    except Exception:
        return None
    limits = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in wheel_joint_names:
            continue
        limit = joint.find("limit")
        if limit is None:
            continue
        vel = limit.get("velocity")
        if vel is None:
            continue
        try:
            limits.append(float(vel))
        except ValueError:
            continue
    if not limits:
        return None
    return min(limits)

def play(args):
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    # override some parameters for testing
    env_cfg.env.num_envs = min(env_cfg.env.num_envs, 3)
    env_cfg.terrain.num_rows = 6
    env_cfg.terrain.num_cols = 6
    env_cfg.terrain.curriculum = True
    # env_cfg.terrain.terrain_proportions = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0]
    env_cfg.noise.add_noise = False
    # env_cfg.terrain.mesh_type = 'plane'
    # env_cfg.terrain.mesh_type = 'trimesh'
    # env_cfg.terrain.static_friction = 0.5
    # env_cfg.terrain.dynamic_friction = 0.5
    # env_cfg.terrain.mesh_type = 'trimesh'
    env_cfg.domain_rand.push_robots = True
    #env_cfg.domain_rand.randomize_friction = False
    
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_motor = False 
    env_cfg.domain_rand.randomize_lag_timesteps = False
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = False
    env_cfg.domain_rand.friction_range=[0.1,0.2]
    env_cfg.domain_rand.randomize_restitution = False
    env_cfg.control.use_filter = True
    env_cfg.control.jump_resampling_time = 10.0

    env_cfg.commands.heading_command = False
    
    env_cfg.init_state.random_dof_pos_probability = 0.0
    env_cfg.init_state.random_ori_probability = 0.0
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
    thigh_idx = [i for i, n in enumerate(env.dof_names) if "thigh_joint" in n]
    knee_idx = [i for i, n in enumerate(env.dof_names) if "knee_joint" in n]
    wheel_joint_names = [
        "front_left_wheel_joint",
        "front_right_wheel_joint",
        "back_left_wheel_joint",
        "back_right_wheel_joint",
    ]
    wheel_name_to_idx = {n: i for i, n in enumerate(env.dof_names)}
    wheel_indices = [wheel_name_to_idx[n] for n in wheel_joint_names if n in wheel_name_to_idx]
    left_wheel_indices = [wheel_name_to_idx[n] for n in wheel_joint_names if "left" in n and n in wheel_name_to_idx]
    right_wheel_indices = [wheel_name_to_idx[n] for n in wheel_joint_names if "right" in n and n in wheel_name_to_idx]
    urdf_path = os.path.join(ROOT_DIR, "resources/y2/urdf/robot.urdf")
    wheel_vel_limit_rad_s = _get_wheel_vel_limit_rad_s(urdf_path, wheel_joint_names)
    # load policy partial_checkpoint_load
    policy_cfg_dict = class_to_dict(train_cfg.policy)
    runner_cfg_dict = class_to_dict(train_cfg.runner)
    actor_critic_class = eval(runner_cfg_dict["policy_class_name"])
    policy: ActorCriticRMA = actor_critic_class(env.cfg.env.n_proprio,
                                                      env.cfg.env.n_scan,
                                                      env.num_obs,
                                                      env.cfg.env.n_priv_latent,
                                                      env.cfg.env.history_len,
                                                      env.num_actions,
                                                      **policy_cfg_dict)
    
    # policy: ActorCriticBarlowTwinsWithStochasticEncoder = actor_critic_class(env.cfg.env.n_proprio,
    #                                                                               env.cfg.env.n_scan,
    #                                                                               env.num_obs,
    #                                                                               env.cfg.env.n_priv_latent,
    #                                                                               env.cfg.env.history_len,
    #                                                                               env.num_actions,
    #                                                                               env.cfg.env.num_dynamics_params,
    #                                                                               env.cfg.env.z_dims,
    #                                                                               **policy_cfg_dict)
    # print(policy)
    # model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/d1_flat/Nov12_18-27-36_/model_6000.pt'))
    model_path = os.path.join(ROOT_DIR, 'logs/d1_flat_height/Apr23_12-47-43_/model_6000.pt')
    model_dict = torch.load(model_path)

    policy.load_state_dict(model_dict['model_state_dict'])
    # policy.half()
    policy.eval()
    policy = policy.to(env.device)
    run_dir = os.path.dirname(model_path)
    run_parent = os.path.basename(os.path.dirname(run_dir))
    run_name = os.path.basename(run_dir).rstrip("_")
    policy.export_tag = f"{run_parent}_{run_name}"
    policy.save_torch_jit_policy('model.pt',env.device)
    model_base = os.path.basename(model_path)
    plot_name = f"{run_parent}_{run_name}_{model_base}.png"
    plot_path = os.path.join(run_dir, plot_name)

    # logger for plot
    logger = Logger(env.dt)
    logger.set_joint_names(env.dof_names)
    body_props = env.gym.get_actor_rigid_body_properties(env.envs[0], env.actor_handles[0])
    robot_mass = sum(p.mass for p in body_props)
    robot_index = 0 # which robot is used for logging
    joint_index = 1 # which joint is used for logging
    start_state_log = 100 # number of steps before plotting states
    stop_state_log = 600 # number of steps before plotting states
    stop_rew_log = env.max_episode_length + 1 # number of steps before print average episode rewards

    # set rgba camera sensor for debug and doudle check
    camera_local_transform = gymapi.Transform()
    camera_local_transform.p = gymapi.Vec3(-0.5, -1, 0.1)
    camera_local_transform.r = gymapi.Quat.from_axis_angle(gymapi.Vec3(0,0,1), np.deg2rad(90))
    camera_props = gymapi.CameraProperties()
    camera_props.width = 512
    camera_props.height = 512

    cam_handle = env.gym.create_camera_sensor(env.envs[0], camera_props)
    body_handle = env.gym.get_actor_rigid_body_handle(env.envs[0], env.actor_handles[0], 0)
    env.gym.attach_camera_to_body(cam_handle, env.envs[0], body_handle, camera_local_transform, gymapi.FOLLOW_TRANSFORM)

    img_idx = 0

    video_duration = 1000
    num_frames = int(video_duration / env.dt)
    video = None

    #torch.sum(self.last_actions - self.actions, dim=1)
    # self.base_lin_vel[:, 2]
    #torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    action_rate = 0
    z_vel = 0
    xy_vel = 0
    feet_air_time = 0

    # 初始化命令值（用于平滑控制）
    command_x = 0.0
    command_y = 0.0
    command_yaw = 0.0
    has_z_vel_command = env.commands.shape[1] >= 5
    if has_z_vel_command and hasattr(env_cfg.commands.ranges, "lin_vel_z"):
        z_vel_min, z_vel_max = env_cfg.commands.ranges.lin_vel_z
        command_z = 0.5 * (z_vel_min + z_vel_max)
        print(f"z velocity command enabled: range=({z_vel_min:.3f}, {z_vel_max:.3f}), init={command_z:.3f}")
    else:
        z_vel_min = None
        z_vel_max = None
        command_z = None
        print("z velocity command disabled for this task")
    
    # 命令速度和加速度限制
    max_x_vel = 1.5
    max_y_vel = 1.0
    # heading_command = 0.5
    max_yaw_vel = 1.0
    max_z_vel = 0.5

    # 按键状态跟踪字典（用于持续检测按键）
    key_states = {
        "w": False, "s": False, "a": False, "d": False,
        "left": False, "right": False, "up": False, "down": False,
    }

    # 订阅键盘事件（如果环境有viewer）
    if hasattr(env, 'viewer') and env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "w_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "s_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "a_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "d_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT, "left_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT, "right_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_UP, "up_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_DOWN, "down_pressed")
    else:
        pass

    for i in range(num_frames):
        action_rate += torch.sum(torch.abs(env.last_actions - env.actions),dim=1)
        z_vel += torch.square(env.base_lin_vel[:, 2])
        xy_vel += torch.sum(torch.square(env.base_ang_vel[:, :2]), dim=1)

        # 处理键盘输入
        if hasattr(env, 'viewer') and env.viewer is not None:
            # 查询键盘事件并更新按键状态
            # evt.value > 0 表示按下，evt.value == 0 表示释放
            for evt in env.gym.query_viewer_action_events(env.viewer):
                if evt.action == "w_pressed":
                    key_states["w"] = (evt.value > 0)
                elif evt.action == "s_pressed":
                    key_states["s"] = (evt.value > 0)
                elif evt.action == "a_pressed":
                    key_states["a"] = (evt.value > 0)
                elif evt.action == "d_pressed":
                    key_states["d"] = (evt.value > 0)
                elif evt.action == "left_pressed":
                    key_states["left"] = (evt.value > 0)
                elif evt.action == "right_pressed":
                    key_states["right"] = (evt.value > 0)
                elif evt.action == "up_pressed":
                    key_states["up"] = (evt.value > 0)
                elif evt.action == "down_pressed":
                    key_states["down"] = (evt.value > 0)
            
            # 根据按键状态设置命令值
            # W/S 控制 x 方向（前进/后退）
            if key_states["w"]:
                command_x = max_x_vel
            elif key_states["s"]:
                command_x = -max_x_vel
            else:
                command_x = 0.0
            
            # A/D 控制 yaw 方向（左转/右转）
            if key_states["a"]:
                command_yaw = max_yaw_vel
            elif key_states["d"]:
                command_yaw = -max_yaw_vel
            else:
                command_yaw = 0.0
            
            # ←/→ 控制 y 方向（左移/右移）
            if key_states["left"]:
                command_y = max_y_vel
            elif key_states["right"]:
                command_y = -max_y_vel
            else:
                command_y = 0.0

            # ↑/↓ 控制 z 方向（上升/下降）
            if has_z_vel_command:
                if key_states["up"]:
                    command_z = min(max_z_vel, z_vel_max)
                elif key_states["down"]:
                    command_z = max(-max_z_vel, z_vel_min)
                else:
                    command_z = 0.0

        # 设置命令值
        env.commands[:,0] = command_x  # x方向速度
        env.commands[:,1] = command_y  # y方向速度
        env.commands[:,2] = command_yaw #command_yaw # yaw角速度
        env.commands[:,3] = 0
        if has_z_vel_command:
            env.commands[:,4] = command_z
        # print("env.commands:",env.commands)

        # if i % 100 == 0:
        actions = policy.act_teacher(obs)

        # actions = policy.act_teacher(obs, env.get_dynamics_params().to(env.device))
        # actions = torch.clamp(actions,-1.2,1.2)

        # if i % 10 == 0:
        #     print("before step:", env.commands[0,:3])
        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)

        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)
        if RECORD_FRAMES:
            if i < stop_state_log and i > start_state_log:
                img = env.gym.get_camera_image(
                    env.sim,
                    env.envs[0],
                    cam_handle,
                    gymapi.IMAGE_COLOR,
                ).reshape((512, 512, 4))[:, :, :3]
                if video is None:
                    video = cv2.VideoWriter(
                        'record.mp4',
                        cv2.VideoWriter_fourcc(*'MP4V'),
                        int(1 / env.dt),
                        (img.shape[1], img.shape[0]),
                    )
                video.write(img)
                img_idx += 1
        if PLOT_STATES:
            if i < stop_state_log and i > start_state_log:
                wheel_radius = 1
                if left_wheel_indices:
                    left_wheel_vel = torch.mean(env.dof_vel[robot_index, left_wheel_indices]).item()
                else:
                    left_wheel_vel = float("nan")
                if right_wheel_indices:
                    right_wheel_vel = torch.mean(env.dof_vel[robot_index, right_wheel_indices]).item()
                else:
                    right_wheel_vel = float("nan")
                left_wheel_lin = left_wheel_vel * wheel_radius
                right_wheel_lin = right_wheel_vel * wheel_radius
                wheel_diff_lin = right_wheel_lin - left_wheel_lin
                if wheel_vel_limit_rad_s is not None:
                    wheel_vel_limit_lin = wheel_vel_limit_rad_s * wheel_radius
                else:
                    wheel_vel_limit_lin = float("nan")
                roll_sin = torch.clamp(env.projected_gravity[robot_index, 1], -1.0, 1.0)
                base_roll_deg = (torch.asin(roll_sin) * 180.0 / np.pi).item()
                logger.log_states(
                    {
                        'dof_pos_target': actions[robot_index, joint_index].item() * env.cfg.control.action_scale,
                        'dof_pos': env.dof_pos[robot_index, joint_index].item(),
                        'dof_vel': env.dof_vel[robot_index, joint_index].item(),
                        'dof_torque': env.torques[robot_index, joint_index].item(),
                        'command_x': env.commands[robot_index, 0].item(),
                        'command_y': env.commands[robot_index, 1].item(),
                        'command_yaw': env.commands[robot_index, 2].item(),
                        'base_vel_x': env.base_lin_vel[robot_index, 0].item(),
                        'base_vel_y': env.base_lin_vel[robot_index, 1].item(),
                        'base_vel_z': env.base_lin_vel[robot_index, 2].item(),
                        'base_vel_yaw': env.base_ang_vel[robot_index, 2].item(),
                        'contact_forces_z': env.contact_forces[robot_index, env.feet_indices, 2].cpu().numpy(),
                        'robot_mass': robot_mass,
                        'base_height': env._get_base_heights()[robot_index].item(),
                        'command_z': env.commands[robot_index, 4].item() if has_z_vel_command else 0.0,
                        'torques': env.torques[robot_index, :].tolist(),
                        'velocities': env.dof_vel[robot_index, :].tolist(),
                        'wheel_left_vel': left_wheel_lin,
                        'wheel_right_vel': right_wheel_lin,
                        'wheel_diff_vel': wheel_diff_lin,
                        'wheel_vel_limit': wheel_vel_limit_lin,
                        'base_roll_deg': base_roll_deg,
                    }
                )
            elif i==stop_state_log:
                logger.plot_states(save_path=plot_path)
        # if i % 10 == 0:
        #     print("after step:", env.commands[0,:3])
        # print(f"env.commands[:,:3]: {env.commands[:,:3]}")

    if video is not None:
        video.release()

if __name__ == '__main__':
    RECORD_FRAMES = False
    PLOT_STATES = True
    args = get_args()
    play(args)
