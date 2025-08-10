
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
import multiprocessing as mp
from copy import deepcopy
from tqdm import tqdm

from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.examples.motionplanning.panda.utils import compute_grasp_info_by_obb, get_actor_obb
from mani_skill.utils import common

import odpc.envs
from odpc.data.demo.motionplanner import PandaArmMotionPlanningClipSolver

# =====================================================================================
# Core Expert Logic (Copied from your debugged script)
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

def run_expert_from_stage(env, start_stage: str, planner: PandaArmMotionPlanningClipSolver):
    """Exactly mimics the logic from the original solve() function."""
    current_stage = start_stage

    if current_stage != "INITIAL":
        planner.close_gripper()
    else:
        planner.open_gripper()
    if current_stage == "INITIAL":
        reach_pose = planner.grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose(reach_pose)
        if res == -1: return -1
        res = planner.move_to_pose(planner.grasp_pose)
        if res == -1: return -1
        planner.close_gripper()
        current_stage = "LIFT"

    if current_stage == "LIFT":
        reach_pose = planner.grasp_pose * sapien.Pose([0, 0, -0.05])
        res = planner.move_to_pose(reach_pose)
        if res == -1: return -1
        current_stage = "ALIGN"

    ee_cur_pose = env.agent.tcp.pose

    if current_stage == "ALIGN":
        offset = 0.01 + env.peg_half_sizes[0, 0].item()
        fine_insert_pose = env.goal_pose * sapien.Pose([-offset, 0, 0])
        for _ in range(3):
            delta_pose = fine_insert_pose * env.peg.pose.inv()
            ee_cur_pose = delta_pose * ee_cur_pose
            res = planner.move_to_pose(ee_cur_pose)
            if res == -1: return -1
        current_stage = "INSERT"

    if current_stage == "INSERT":
        delta_pose = env.goal_pose * sapien.Pose([0.00, 0, 0]) * env.peg.pose.inv()
        ee_cur_pose = delta_pose * ee_cur_pose
        res = planner.move_to_pose(ee_cur_pose)
        if res == -1: return -1

    return res

# =====================================================================================
# Worker Function for Multiprocessing
# =====================================================================================

def _main(args, proc_id: int, indices_to_process: list):
    # --- 1. 配置解析 (与之前相同) ---
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

    # --- 2. 每个进程创建一个独立的环境 ---
    env_creation_params["control_mode"] = "pd_joint_pos"
    env_creation_params["sim_backend"] = "physx_cpu"
    env_creation_params["render_mode"] = "rgb_array"
    env = gym.make(env_id, num_envs=1, **env_creation_params)

    # --- 3. 使用 RecordEpisode 包装器，每个进程只生成一个文件 ---
    # 文件名将是 traj_name.proc_id.h5
    env = RecordEpisode(
        env, 
        output_dir=args.output_dir, 
        save_trajectory=True, 
        save_video=args.save_video, 
        trajectory_name=f"{args.traj_name}.{proc_id}"
    )
    output_h5_path = env._h5_file.filename

    # --- 4. 循环处理分配到的所有索引 ---
    successes = []
    solution_episode_lengths = []
    
    pbar = tqdm(indices_to_process, desc=f"Process {proc_id}")
    for process_index in indices_to_process:
        with h5py.File(key_states_path, "r") as f:
            seed = int(f['episode_seed'][process_index])
            state_dict = {k: {sub_k: torch.from_numpy(sub_v[process_index:process_index+1]) for sub_k, sub_v in v.items()} for k, v in f['env_states'].items()}

        env.reset(seed=seed, options=dict(reconfigure=True))
        env.unwrapped.set_state_dict(state_dict)

        # 更新 RecordEpisode 中记录的初始状态和观测
        # 因为 RecordEpisode 在 reset 时已经记录了初始状态，但我们在 reset 后又修改了状态
        # 所以需要手动更新 RecordEpisode 的记录
        if hasattr(env, '_trajectory_buffer') and env._trajectory_buffer is not None:
            # 更新环境状态记录
            def recursive_replace(x, y):
                if isinstance(x, np.ndarray):
                    x[-1, :] = y[-1, :]
                else:
                    for k in x.keys():
                        recursive_replace(x[k], y[k])

            # 更新状态记录
            if env.record_env_state:
                recursive_replace(
                    env._trajectory_buffer.state, 
                    common.to_numpy(common.batch(state_dict))
                )
            
            # 更新观测记录
            updated_obs = env.base_env.get_obs()
            recursive_replace(
                env._trajectory_buffer.observation,
                common.to_numpy(common.batch(updated_obs))
            )

        # 通过一次step更新物理仿真器的状态，为后续计算 is_grasp 做准备
        # Warning: 这个 step 的 action 可能是有问题的
        current_qpos = env.unwrapped.agent.robot.get_qpos()
        action = torch.cat([current_qpos[:, :7], current_qpos[:, 7:8]], dim=1)
        env.unwrapped.agent.set_action(action)
        env.unwrapped.scene.step()

        stage = detect_current_stage(env.unwrapped, 0)
        
        # 注意：在多进程中，可视化通常是关闭的
        planner = PandaArmMotionPlanningClipSolver(
            env, 
            debug=False, 
            vis=args.vis, 
            base_pose=env.unwrapped.agent.robot.pose, 
            print_env_info=False,
        )

        FINGER_LENGTH = 0.025
        obb = get_actor_obb(env.unwrapped.peg)
        approaching = np.array([0, 0, -1])
        target_closing = env.unwrapped.agent.tcp.pose.to_transformation_matrix()[0, :3, 1].cpu().numpy()
        grasp_info = compute_grasp_info_by_obb(obb, approaching=approaching, target_closing=target_closing, depth=FINGER_LENGTH)
        closing, center = grasp_info["closing"], grasp_info["center"]
        planner.grasp_pose = env.unwrapped.agent.build_grasp_pose(approaching, closing, center)
        offset = max(0.05, env.unwrapped.peg_half_sizes[0, 0].item() / 2 + 0.01)
        planner.grasp_pose.p = (env.unwrapped.peg.pose * sapien.Pose([-offset, 0, 0])).p[0].cpu().numpy()

        res = run_expert_from_stage(env, stage, planner)
        if res == -1:
            success = False
        else:
            success = res[-1]["success"].item()
            elapsed_steps = res[-1]["elapsed_steps"].item()
            solution_episode_lengths.append(elapsed_steps)
        successes.append(success)
        
        if args.only_count_success and not success:
            env.flush_trajectory(save=False)
            if args.save_video:
                env.flush_video(save=False)
        else:
            env.flush_trajectory()
            if args.save_video:
                env.flush_video()

        pbar.update(1)
        pbar.set_postfix(
            dict(
                success_rate=np.mean(successes),
                avg_episode_length=np.mean(solution_episode_lengths),
                max_episode_length=np.max(solution_episode_lengths),
            )
        )

    env.close()
    return output_h5_path

# =====================================================================================
# Main Function for Dispatching
# =====================================================================================

def main():
    parser = argparse.ArgumentParser(description="从所有恢复的状态开始，并行执行专家策略，生成轨迹。")
    parser.add_argument("--analysis-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./outputs/generated_trajectories_from_states")
    parser.add_argument("--traj-name", type=str, default="from_states_expert")
    parser.add_argument("--num-procs", type=int, default=1, help="用于并行生成的进程数。")
    parser.add_argument("--vis", action="store_true", help="开启交互式可视化调试（仅在 num-procs=1 时有效）。")
    parser.add_argument("--save-video", action="store_true", help="保存视频轨迹。")
    parser.add_argument("--only-count-success", action="store_true", help="仅统计成功轨迹。")
    args = parser.parse_args()

    if args.vis and args.num_procs > 1:
        print("[!] [警告] 可视化模式不支持多进程，将强制使用单进程。")
        args.num_procs = 1

    key_states_path = os.path.join(args.analysis_dir, "key_env_states.h5")
    with h5py.File(key_states_path, "r") as f:
        num_total_states = f['episode_seed'].shape[0]
    
    print(f"[*] 发现 {num_total_states} 个关键状态。将使用 {args.num_procs} 个进程进行处理。")

    indices = np.arange(num_total_states)
    indices_split = np.array_split(indices, args.num_procs)

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    mp.set_start_method("spawn", force=True)
    pool = mp.Pool(args.num_procs)
    
    proc_args = [(deepcopy(args), i, split.tolist()) for i, split in enumerate(indices_split)]
    
    # res 将会是每个进程产生的 h5 文件的路径列表
    res = pool.starmap(_main, proc_args)
    pool.close()
    pool.join()

    print("[*] 所有子进程执行完毕。开始合并轨迹...")

    # --- 合并轨迹文件 ---
    output_path = os.path.join(args.output_dir, f"{args.traj_name}.h5")
    merge_trajectories(output_path, res)

    # --- 清理临时文件 ---
    for h5_path in res:
        print(f"  [Cleanup] 正在删除 {h5_path}")
        os.remove(h5_path)
        json_path = h5_path.replace(".h5", ".json")
        if os.path.exists(json_path):
            print(f"  [Cleanup] 正在删除 {json_path}")
            os.remove(json_path)

    print(f"[*] 合并完毕。最终轨迹已保存到: {output_path}")

if __name__ == "__main__":
    main()
