
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
import cv2

# 导入自定义环境，确保其被注册
import odpc.envs

def detect_current_stage(env, env_idx: int) -> str:
    """根据环境状态检测当前最可能的专家策略阶段（V5 - 基于官方 is_grasping）。"""
    # 定义一些合理的阈值
    ALIGN_THRESH_XY = 0.03      # 对齐阶段的 XY 平面距离阈值
    LIFT_CLEARANCE_Z = 0.015    # peg 必须比其初始高度高出这个距离才算离地

    # --- 1. 首先使用官方方法判断 peg 是否被抓取 ---
    is_grasped = env.agent.is_grasping(env.peg)[env_idx]

    if not is_grasped:
        return "INITIAL (pre-grasp or failed grasp)"

    # --- 如果已成功抓取，则进入后续判断 ---
    peg_p = env.peg.pose.p[env_idx].squeeze(0)
    goal_p = env.goal_pose.p[env_idx].squeeze(0)
    
    initial_peg_z = env.peg_half_sizes[env_idx, 2]
    is_lifted = peg_p[2] > (initial_peg_z + LIFT_CLEARANCE_Z)

    if not is_lifted:
        return "LIFT (grasped on table)"

    is_aligned_xy = torch.linalg.norm(peg_p[:2] - goal_p[:2]) < ALIGN_THRESH_XY

    if is_aligned_xy:
        return "INSERT"

    return "ALIGN (in air, moving to hole)"

def main():
    """主执行函数"""
    parser = argparse.ArgumentParser(
        description="从 rollout 分析结果中加载状态和种子，验证其能否精确复现 ManiSkill 环境，并检测当前所处的专家策略阶段。"
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
        default="./outputs/verified_images",
        help="保存已标注图像的输出目录路径。 (默认: ./outputs/verified_images)"
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
    odpc_root = "/home/xucuijie/Projects/ManiSkill/odpc"
    rollout_h5_abs_path = os.path.join(odpc_root, rollout_h5_rel_path)
    rollout_config_path = os.path.join(os.path.dirname(rollout_h5_abs_path), "config.yaml")

    print(f"[*] 正在解析 Rollout 配置文件: {rollout_config_path}")
    with open(rollout_config_path, 'r') as f:
        rollout_config = yaml.safe_load(f)

    env_creation_params = rollout_config["evaluator"]["env_configs"]["ind"]["env"]
    env_kwargs = env_creation_params.pop("env_kwargs", {})
    env_creation_params.update(env_kwargs)
    env_creation_params.pop("_target_", None)
    env_creation_params.pop("other_kwargs", None)
    env_creation_params.pop("output_dir", None)
    env_creation_params.pop("traj_name", None)
    env_creation_params.pop("num_envs", None)
    env_id = env_creation_params.pop("env_id")

    print(f"[*] 提取到的环境参数: {env_creation_params}")

    # --- 3. 加载状态和种子 ---
    print(f"[*] 正在加载状态和种子: {key_states_path}")
    with h5py.File(key_states_path, "r") as f:
        seeds = f['episode_seed'][:]
        num_envs = len(seeds)
        print(f"[*] 找到 {num_envs} 个种子和状态。")

        env_states_group = f['env_states']
        state_dict = {
            'actors': {key: torch.from_numpy(value[:]) for key, value in env_states_group['actors'].items()},
            'articulations': {key: torch.from_numpy(value[:]) for key, value in env_states_group['articulations'].items()}
        }

    # --- 4. 创建并恢复环境 ---
    # 强制将控制模式设置为 pd_joint_pos，以匹配专家策略和我们的“保持状态”动作
    env_creation_params["control_mode"] = "pd_joint_pos"
    print(f"[*] [!] 强制使用控制模式: {env_creation_params['control_mode']}")

    print(f"[*] 正在创建 {num_envs} 个并行环境...")
    env = gym.make(
        env_id,
        num_envs=num_envs,
        **env_creation_params
    )

    print("[*] 正在使用种子重置环境以配置场景...")
    seed_list = [int(s) for s in seeds]
    env.reset(seed=seed_list, options=dict(reconfigure=True))
    
    print("[*] ---------------- 状态恢复与物理同步 ----------------")
    print("[*] 正在使用 set_state_dict 设置物理状态...")
    env.unwrapped.set_state_dict(state_dict)

    print("[*] 正在同步控制器目标...")
    # 9-DoF qpos to 8-DoF action for pd_joint_pos controller
    # The gripper joints are mimic, so we only need one of them.
    current_qpos = env.unwrapped.agent.robot.get_qpos()
    action = torch.cat([current_qpos[:, :7], current_qpos[:, 7:8]], dim=1)
    env.unwrapped.agent.set_action(action)

    print("[*] 正在推进一帧物理仿真以更新接触力...")
    env.unwrapped.scene.step()
    if env.unwrapped.gpu_sim_enabled:
        env.unwrapped.scene._gpu_fetch_all()
    print("[*] 环境状态已完美恢复并同步。")

    # --- 5. 检测阶段、渲染、标注并保存图像 ---
    print("[*] 正在获取观测数据...")
    obs = env.unwrapped.get_obs()

    base_cam_arrays = obs['sensor_data']['base_camera']['rgb'].cpu().numpy().astype(np.uint8)
    hand_cam_arrays = obs['sensor_data']['hand_camera']['rgb'].cpu().numpy().astype(np.uint8)

    print(f"[*] 正在检测阶段、拼接并标注图像，并保存到: {args.output_dir}")
    for i in range(num_envs):
        stage_name = detect_current_stage(env.unwrapped, i)
        concatenated_image = np.concatenate((base_cam_arrays[i], hand_cam_arrays[i]), axis=1)
        annotated_image = concatenated_image.copy()
        cv2.putText(
            annotated_image, 
            f"Stage: {stage_name}", 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (0, 255, 0), 
            2, 
            cv2.LINE_AA
        )
        img = Image.fromarray(annotated_image)
        output_path = os.path.join(args.output_dir, f"verified_{i}_seed_{seeds[i]}.png")
        img.save(output_path)

    # --- 6. 清理 ---
    env.close()
    print("[*] 完成！")

if __name__ == "__main__":
    main()
