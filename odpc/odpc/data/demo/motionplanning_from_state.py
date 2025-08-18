
#!/usr/bin/env python3

import json
import gymnasium as gym
import h5py
import torch
import numpy as np
import os
import argparse
import yaml
import multiprocessing as mp
from copy import deepcopy
from tqdm import tqdm

from mani_skill.utils.wrappers import RecordEpisode
from mani_skill.trajectory.merge_trajectory import merge_trajectories
from mani_skill.utils import common

import odpc.envs
from odpc.data.demo.panda_solutions import solvePegInsertionSidev2FromStage

MP_SOLUTIONS = {
    "PegInsertionSide-v2": solvePegInsertionSidev2FromStage,
}

def _main(args, proc_id: int, indices_to_process: list):
 
    env_id, env_kwargs = parse_rollout_config(args.analysis_dir)
    env = gym.make(env_id, **env_kwargs)

    key_states_path = os.path.join(args.analysis_dir, "key_env_states.h5")
    
    env = RecordEpisode(
        env, 
        output_dir=args.output_dir, 
        save_trajectory=True, 
        save_video=args.save_video, 
        trajectory_name=f"{args.traj_name}.{proc_id}"
    )
    output_h5_path = env._h5_file.filename

    solve = MP_SOLUTIONS[env_id]
    if hasattr(args, "solve_kwargs"):
        solve_kwargs = args.solve_kwargs
    else:
        solve_kwargs = {}

    pbar = tqdm(indices_to_process, desc=f"Process {proc_id}")
    successes = []
    solution_episode_lengths = []
    
    for process_index in indices_to_process:

        with h5py.File(key_states_path, "r") as f:
            def get_data(data, index):
                if isinstance(data, (h5py.File, h5py.Group)):
                    return {k: get_data(v, index) for k, v in data.items()}
                elif isinstance(data, h5py.Dataset):
                    return data[index]
                else:
                    raise ValueError(f"Unsupported data type: {type(data)}")

            seed = int(f['episode_seed'][process_index])
            state_dict = get_data(f['env_states'], slice(process_index, process_index+1))

        env.reset(seed=seed, options=dict(reconfigure=True))
        env.unwrapped.set_state_dict(state_dict)

        # 更新 RecordEpisode 中记录的初始状态和观测
        # 因为 RecordEpisode 在 reset 时已经记录了初始状态，但我们在 reset 后又修改了状态
        # 所以需要手动更新 RecordEpisode 的记录
        if hasattr(env, '_trajectory_buffer') and env._trajectory_buffer is not None:
            def recursive_replace(x, y):
                if isinstance(x, np.ndarray):
                    x[-1, :] = y[-1, :]
                else:
                    for k in x.keys():
                        recursive_replace(x[k], y[k])
            if env.record_env_state:
                recursive_replace(
                    env._trajectory_buffer.state, 
                    common.to_numpy(common.batch(state_dict))
                )
        
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

        try:
            res = solve(env=env, seed=seed, debug=False, vis=args.vis, **solve_kwargs)
        except Exception as e:
            print(f"Episode {process_index} failed: {e}")
            res = -1
        
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
        if len(solution_episode_lengths) > 0:
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

def get_args():
    parser = argparse.ArgumentParser(description="从所有恢复的状态开始，并行执行专家策略，生成轨迹。")
    parser.add_argument("--analysis-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./demos")
    parser.add_argument("--traj-name", type=str, default="motionplanning_from_state")
    parser.add_argument("--num-procs", type=int, default=1, help="用于并行生成的进程数。")
    parser.add_argument("--vis", action="store_true", help="开启交互式可视化调试（仅在 num-procs=1 时有效）。")
    parser.add_argument("--save-video", action="store_true", help="保存视频轨迹。")
    parser.add_argument("--only-count-success", action="store_true", help="仅统计成功轨迹。")
    parser.add_argument("--solve-kwargs", type=str, default="{}", help="传递给专家策略的额外参数。")
    args = parser.parse_args()

    if args.vis and args.num_procs > 1:
        print("[!] [警告] 可视化模式不支持多进程，将强制使用单进程。")
        args.num_procs = 1
    
    args.solve_kwargs = json.loads(args.solve_kwargs)

    return args

def parse_rollout_config(analysis_dir: str):
    analysis_config_path = os.path.join(analysis_dir, "config.yaml")
    with open(analysis_config_path, 'r') as f: 
        analysis_config = yaml.safe_load(f)
    rollout_path = analysis_config["rollout_path"]
    rollout_config_path = os.path.join(os.path.dirname(rollout_path), "config.yaml")
    with open(rollout_config_path, 'r') as f: 
        rollout_config = yaml.safe_load(f)

    env_id = rollout_config["evaluator"]["env_configs"]["ind"]["env"]["env_id"]
    env_kwargs = rollout_config["evaluator"]["env_configs"]["ind"]["env"]["env_kwargs"]
    env_kwargs["control_mode"] = "pd_joint_pos"
    env_kwargs["sim_backend"] = "physx_cpu"
    env_kwargs["num_envs"] = 1
    return env_id, env_kwargs

def main(args):
    key_states_path = os.path.join(args.analysis_dir, "key_env_states.h5")
    with h5py.File(key_states_path, "r") as f:
        num_total_states = f['episode_seed'].shape[0]
    
    indices = np.arange(num_total_states)
    if args.num_trajs is not None:
        rng = np.random.default_rng(args.seed)
        indices = rng.choice(indices, size=min(args.num_trajs, num_total_states), replace=False)
    indices_split = np.array_split(indices, args.num_procs)

    print(f"[*] 发现 {num_total_states} 个关键状态，处理 {len(indices)} 个状态，将使用 {args.num_procs} 个进程进行处理。")

    os.makedirs(args.output_dir, exist_ok=True)
    
    proc_args = [(deepcopy(args), i, split.tolist()) for i, split in enumerate(indices_split)]

    with mp.Pool(args.num_procs) as pool:
        res = pool.starmap(_main, proc_args)

    print("[*] 所有子进程执行完毕。开始合并轨迹...")

    # --- 合并轨迹文件 ---
    output_path = os.path.join(args.output_dir, f"{args.traj_name}.h5")
    merge_trajectories(output_path, res)

    # --- 清理临时文件 ---
    for h5_path in res:
        print(f"  [Cleanup] 正在删除 {h5_path}")
        os.remove(h5_path)
        json_path = h5_path.replace(".h5", ".json")
        print(f"  [Cleanup] 正在删除 {json_path}")
        os.remove(json_path)

    print(f"[*] 合并完毕。最终轨迹已保存到: {output_path}")

    return output_path

if __name__ == "__main__":
    args = get_args()
    main(args)
