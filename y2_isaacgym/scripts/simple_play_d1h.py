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
    
    env_cfg.init_state.random_dof_pos_probability = 0.0
    env_cfg.init_state.random_ori_probability = 0.0
    # prepare environment
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()
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
    model_dict = torch.load(os.path.join(ROOT_DIR, 'logs/y2_flat/Jan20_14-28-25_/model_3000.pt'))

    policy.load_state_dict(model_dict['model_state_dict'])
    # policy.half()
    policy.eval()
    policy = policy.to(env.device)
    policy.save_torch_jit_policy('model.pt',env.device)

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
    # print(f'gathering {num_frames} frames')
    # video = None

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
    
    # 命令速度和加速度限制
    max_x_vel = 1.0
    max_y_vel = 1.0
    max_yaw_vel = 1.0

    # 按键状态跟踪字典（用于持续检测按键）
    key_states = {
        "w": False, "s": False, "a": False, "d": False,
        "left": False, "right": False
    }

    # 订阅键盘事件（如果环境有viewer）
    if hasattr(env, 'viewer') and env.viewer is not None:
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_W, "w_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_S, "s_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_A, "a_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_D, "d_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_LEFT, "left_pressed")
        env.gym.subscribe_viewer_keyboard_event(env.viewer, gymapi.KEY_RIGHT, "right_pressed")
        print("Keyboard control enabled: W/S for x, A/D for yaw, ←/→ for y")
    else:
        print("Warning: No viewer found. Keyboard control will not work. Make sure headless=False in config.")

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
        
        # 设置命令值
        env.commands[:,0] = command_x  # x方向速度
        env.commands[:,1] = command_y  # y方向速度
        env.commands[:,2] = command_yaw # yaw角速度
        env.commands[:,3] = 0.0
        # print("env.commands:",env.commands)

        # if i % 100 == 0:
        actions = policy.act_teacher(obs)

        # actions = policy.act_teacher(obs, env.get_dynamics_params().to(env.device))
        # actions = torch.clamp(actions,-1.2,1.2)

        # if i % 10 == 0:
        #     print("before step:", env.commands[0,:3])
        obs, privileged_obs, rewards,costs,dones, infos = env.step(actions)
        # if i % 10 == 0:
        #     print("after step:", env.commands[0,:3])
        # print(f"env.commands[:,:3]: {env.commands[:,:3]}")
        env.gym.step_graphics(env.sim) # required to render in headless mode
        env.gym.render_all_camera_sensors(env.sim)

    # video.release()
if __name__ == '__main__':
    RECORD_FRAMES = False
    args = get_args()
    play(args)
