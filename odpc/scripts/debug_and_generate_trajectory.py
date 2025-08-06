
#!/usr/bin/env python3

import gymnasium as gym
import h5py
import torch
import numpy as np
import mani_skill.envs
import os
import argparse
import yaml
import sapien

import odpc.envs
from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.examples.motionplanning.panda.motionplanner import PandaArmMotionPlanningSolver, OPEN, CLOSED
from mani_skill.examples.motionplanning.panda.utils import (
    compute_grasp_info_by_obb, get_actor_obb)

# =====================================================================================
# Core Expert Logic (V6 - Replicating run.py)
# =====================================================================================

def detect_current_stage(env, env_idx: int) -> str:
    """Detects stage based on the official is_grasping method."""
    ALIGN_THRESH_XY = 0.03
    LIFT_CLEARANCE_Z = 0.015
    is_grasped = env.agent.is_grasping(env.peg)[env_idx]
    if not is_grasped:
        return "INITIAL"
    peg_p = env.peg.pose.p[env_idx].squeeze(0)
    goal_p = env.goal_pose.p[env_idx].squeeze(0)
    initial_peg_z = env.peg_half_sizes[env_idx, 2]
    is_lifted = peg_p[2] > (initial_peg_z + LIFT_CLEARANCE_Z)
    if not is_lifted:
        return "LIFT"
    is_aligned_xy = torch.linalg.norm(peg_p[:2] - goal_p[:2]) < ALIGN_THRESH_XY
    if is_aligned_xy:
        return "INSERT"
    else:
        return "ALIGN"

def run_expert_from_stage(env, start_stage: str, planner: PandaArmMotionPlanningSolver):
    """Exactly mimics the logic from the original solve() function."""
    print(f"  [Expert] 从 {start_stage} 阶段开始执行...")
    current_stage = start_stage

    if current_stage != "INITIAL":
        print("  [Expert] 检测到已抓取，正在同步规划器的夹爪状态到 'CLOSED'...")
        planner.gripper_state = CLOSED

    if current_stage == "INITIAL":
        print("  [Expert] 阶段: INITIAL -> ... -> LIFT")
        reach_pose = planner.grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return -1
        planner.close_gripper()
        current_stage = "LIFT"

    if current_stage == "LIFT":
        print("  [Expert] 阶段: LIFT")
        reach_pose = planner.grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose_with_screw(reach_pose)
        if res == -1: return -1
        current_stage = "ALIGN"

    ee_cur_pose = env.unwrapped.agent.tcp.pose

    if current_stage == "ALIGN":
        print("  [Expert] 阶段: ALIGN")
        offset = 0.01 + env.unwrapped.peg_half_sizes[0, 0].item()
        fine_insert_pose = env.unwrapped.goal_pose * sapien.Pose([-offset, 0, 0])
        for _ in range(3):
            delta_pose = fine_insert_pose * env.unwrapped.peg.pose.inv()
            ee_cur_pose = delta_pose * ee_cur_pose
            res = planner.move_to_pose_with_screw(ee_cur_pose)
            if res == -1: return -1
        current_stage = "INSERT"

    if current_stage == "INSERT":
        print("  [Expert] 阶段: INSERT")
        delta_pose = env.unwrapped.goal_pose * sapien.Pose([0.03, 0, 0]) * env.unwrapped.peg.pose.inv()
        ee_cur_pose = delta_pose * ee_cur_pose
        res = planner.move_to_pose_with_screw(ee_cur_pose)
        if res == -1: return -1

    print("  [Expert] 专家策略执行完毕。")
    return 0

# =====================================================================================
# Main Execution (Final Version)
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="从恢复的状态开始，执行专家策略，生成轨迹并进行可视化调试。")
    parser.add_argument("--analysis-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/generated_trajectories")
    parser.add_argument("--process-index", type=int, default=0)
    parser.add_argument("--vis", action="store_true", help="开启交互式可视化调试。")
    parser.add_argument("--save-video", action="store_true", help="保存视频轨迹。")
    args = parser.parse_args()

    # --- 1. 配置解析 ---
    analysis_config_path = os.path.join(args.analysis_dir, "config.yaml")
    key_states_path = os.path.join(args.analysis_dir, "key_env_states.h5")
    with open(analysis_config_path, 'r') as f: analysis_config = yaml.safe_load(f)
    rollout_h5_rel_path = analysis_config["rollout_path"]
    odpc_root = "/home/xucuijie/Projects/ManiSkill/odpc"
    rollout_config_path = os.path.join(os.path.dirname(os.path.join(odpc_root, rollout_h5_rel_path)), "config.yaml")
    with open(rollout_config_path, 'r') as f: rollout_config = yaml.safe_load(f)
    env_creation_params = rollout_config["evaluator"]["env_configs"]["ind"]["env"]
    env_kwargs = env_creation_params.pop("env_kwargs", {})
    env_creation_params.update(env_kwargs)
    for key in ["_target_", "other_kwargs", "output_dir", "traj_name", "num_envs"]:
        env_creation_params.pop(key, None)
    env_id = env_creation_params.pop("env_id")

    # --- 2. 加载状态和种子 ---
    with h5py.File(key_states_path, "r") as f:
        seed = int(f['episode_seed'][args.process_index])
        state_dict = {k: {sub_k: torch.from_numpy(sub_v[args.process_index:args.process_index+1]) for sub_k, sub_v in v.items()} for k, v in f['env_states'].items()}

    # --- 3. 创建环境 (遵循 run.py 的设计模式) ---
    env_creation_params["control_mode"] = "pd_joint_pos"
    env_creation_params["sim_backend"] = "physx_cpu"
    env_creation_params["render_mode"] = "rgb_array" # 始终使用 rgb_array

    env = gym.make(env_id, num_envs=1, **env_creation_params)
    
    # 使用 trajectory_name 参数来控制输出文件名
    base_name = f"from_state_idx_{args.process_index}_seed_{seed}"
    env = RecordEpisode(env, args.output_dir, save_trajectory=True, save_video=args.save_video, trajectory_name=base_name)
    
    env.reset(seed=seed, options=dict(reconfigure=True))
    
    env.unwrapped.set_state_dict(state_dict)
    current_qpos = env.unwrapped.agent.robot.get_qpos()
    action = torch.cat([current_qpos[:, :7], current_qpos[:, 7:8]], dim=1)
    env.unwrapped.agent.set_action(action)
    env.unwrapped.scene.step()

    # --- 4. 执行与调试 ---
    stage = detect_current_stage(env.unwrapped, 0)
    print(f"[*] 检测到阶段: {stage}")

    planner = PandaArmMotionPlanningSolver(env, debug=False, vis=args.vis, base_pose=env.unwrapped.agent.robot.pose)

    FINGER_LENGTH = 0.025
    obb = get_actor_obb(env.unwrapped.peg)
    approaching = np.array([0, 0, -1])
    target_closing = env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
    grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
    closing, center = grasp_info["closing"], grasp_info["center"]
    planner.grasp_pose = env.unwrapped.agent.build_grasp_pose(approaching, closing, center)
    offset = max(0.05, env.unwrapped.peg_half_sizes[0, 0].item() / 2 + 0.01)
    planner.grasp_pose.p = (env.unwrapped.peg.pose * sapien.Pose([-offset, 0, 0])).p[0].cpu().numpy()

    success = run_expert_from_stage(env, stage, planner)

    # --- 5. 保存与清理 ---
    if success == -1:
        print("[*] 专家策略执行失败。")
    else:
        print("[*] 专家策略成功执行。")

    env.close()
    final_h5_path = os.path.join(args.output_dir, f"{base_name}.h5")
    print(f"[*] 轨迹已保存到: {final_h5_path}")
    if args.save_video:
        final_mp4_path = os.path.join(args.output_dir, f"{base_name}.mp4")
        print(f"[*] 视频已保存到: {final_mp4_path}")
    print("[*] 脚本执行完毕。")

if __name__ == "__main__":
    main()
