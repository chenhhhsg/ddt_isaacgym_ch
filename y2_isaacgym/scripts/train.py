import numpy as np
import os
import sys
import json
# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
from configs import *

import isaacgym
from utils.helpers import get_args
from utils.task_registry import task_registry

def save_configs_to_log_dir(env_cfg, train_cfg, log_dir):
    """
    保存环境配置和训练配置到训练日志目录中，并记录用户输入的修改说明
    
    Args:
        env_cfg: 环境配置对象
        train_cfg: 训练配置对象  
        log_dir: 训练日志目录路径
    """
    if log_dir is None:
        print("Warning: log_dir is None, skipping config save")
        return
    
    # 获取用户输入的修改说明
    print("\n" + "="*60)
    print("请描述本次训练相对于resume策略的修改内容:")
    print("例如:")
    print("- 修改了奖励函数权重")
    print("- 调整了时间参数") 
    print("- 添加了新的奖励项")
    print("- 修改了网络结构")
    print("="*60)
    
    changes = input("请输入修改说明 (按Enter跳过): ").strip()
    changes = changes if changes else "无修改说明"
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 将配置对象转换为字典
    env_cfg_dict = class_to_dict(env_cfg)
    train_cfg_dict = class_to_dict(train_cfg)
    
    # 创建统一的配置记录
    unified_config = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "training_changes": changes,
        "resume_path": getattr(train_cfg.runner, 'resume_path', None),
        "environment_config": env_cfg_dict,
        "training_config": train_cfg_dict
    }
    
    # 保存统一的配置文件
    config_path = os.path.join(log_dir, 'unified_config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(unified_config, f, indent=2, ensure_ascii=False)
    print(f"统一配置文件已保存到: {config_path}")
    
    # 显示resume路径信息
    if hasattr(train_cfg.runner, 'resume_path') and train_cfg.runner.resume_path:
        print(f"Resume from: {train_cfg.runner.resume_path}")

def train(args):
    env, env_cfg = task_registry.make_env(name=args.task, args=args)
    ppo_runner, train_cfg = task_registry.make_alg_runner(env=env, name=args.task, args=args)
    save_configs_to_log_dir(env_cfg, train_cfg, ppo_runner.log_dir)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)

if __name__ == '__main__':
  
    args = get_args()
    train(args)
