
#!/usr/bin/env python3

import gymnasium as gym
import h5py
import torch
import numpy as np
from PIL import Image
import mani_skill.envs
import os
import argparse
import yaml

# 导入自定义环境，确保其被注册
import odpc.envs

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="从 rollout 分析结果中重建并渲染 ManiSkill 环境状态。"
    )
    parser.add_argument(
        "--analysis-dir",
        type=str,
        required=True,
        help="包含 config.yaml 和 key_env_states.h5 的分析结果目录的路径。"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./reconstructed_images_generic",
        help="保存重建图像的输出目录路径。 (默认: ./reconstructed_images_generic)"
    )
    args = parser.parse_args()

    # --- 1. 路径构建 ---
    print(f"[*] 使用分析目录: {args.analysis_dir}")
    analysis_config_path = os.path.join(args.analysis_dir, "config.yaml")
    key_states_path = os.path.join(args.analysis_dir, "key_env_states.h5")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. 解析配置 ---
    print(f"[*] 正在解析分析配置文件: {analysis_config_path}")
    with open(analysis_config_path, 'r') as f:
        analysis_config = yaml.safe_load(f)
    
    rollout_h5_rel_path = analysis_config["rollout_path"]
    # 假设所有相对路径都基于 /home/xucuijie/Projects/ManiSkill/odpc
    # 这是根据我们之前的对话确定的
    odpc_root = "/home/xucuijie/Projects/ManiSkill/odpc"
    rollout_h5_abs_path = os.path.join(odpc_root, rollout_h5_rel_path)
    rollout_config_path = os.path.join(os.path.dirname(rollout_h5_abs_path), "config.yaml")

    print(f"[*] 正在解析 Rollout 配置文件: {rollout_config_path}")
    with open(rollout_config_path, 'r') as f:
        rollout_config = yaml.safe_load(f)

    # 提取环境创建所需的所有参数
    env_creation_params = rollout_config["evaluator"]["env_configs"]["ind"]["env"]
    # 将 env_kwargs 中的内容提升到主字典中
    env_kwargs = env_creation_params.pop("env_kwargs", {})
    env_creation_params.update(env_kwargs)
    # 移除一些不需要的或由脚本控制的参数
    env_creation_params.pop("_target_", None)
    env_creation_params.pop("other_kwargs", None)
    env_creation_params.pop("output_dir", None)
    env_creation_params.pop("traj_name", None)
    env_creation_params.pop("num_envs", None)

    print(f"[*] 提取到的环境参数: {env_creation_params}")

    # --- 3. 加载状态和种子 ---
    print(f"[*] 正在加载状态和种子: {key_states_path}")
    with h5py.File(key_states_path, "r") as f:
        if 'env_states' not in f or 'episode_seed' not in f:
            raise ValueError("HDF5 文件中必须包含 'env_states' 和 'episode_seed' 数据集。")
        
        seeds = f['episode_seed'][:]
        num_envs = len(seeds)
        print(f"[*] 找到 {num_envs} 个种子和状态。")

        env_states_group = f['env_states']
        state_dict = {
            'actors': {key: torch.from_numpy(value[:]) for key, value in env_states_group['actors'].items()},
            'articulations': {key: torch.from_numpy(value[:]) for key, value in env_states_group['articulations'].items()}
        }

    # --- 4. 创建并恢复环境 ---
    print(f"[*] 正在创建 {num_envs} 个并行环境...")
    env_id = env_creation_params.pop("env_id")
    env = gym.make(
        env_id,
        num_envs=num_envs,
        **env_creation_params
    )

    print("[*] 正在使用种子重置环境以配置场景...")
    seed_list = [int(s) for s in seeds]
    env.reset(seed=seed_list, options=dict(reconfigure=True))
    
    print("[*] 正在使用 set_state_dict 设置动态状态...")
    env.unwrapped.set_state_dict(state_dict)
    print("[*] 环境状态已完美恢复。")

    # --- 5. 渲染并保存图像 ---
    print("[*] 正在获取观测数据...")
    obs = env.unwrapped.get_obs()

    base_cam_arrays = obs['sensor_data']['base_camera']['rgb'].cpu().numpy().astype(np.uint8)
    hand_cam_arrays = obs['sensor_data']['hand_camera']['rgb'].cpu().numpy().astype(np.uint8)

    print(f"[*] 正在拼接图像并保存到: {args.output_dir}")
    for i in range(num_envs):
        concatenated_image = np.concatenate((base_cam_arrays[i], hand_cam_arrays[i]), axis=1)
        img = Image.fromarray(concatenated_image)
        output_path = os.path.join(args.output_dir, f"reconstructed_{i}_seed_{seeds[i]}.png")
        img.save(output_path)

    # --- 6. 清理 ---
    env.close()
    print("[*] 完成！")

if __name__ == "__main__":
    main()
