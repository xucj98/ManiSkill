# odpc/odpc/data/demo/generation.py
import os
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf, DictConfig

from odpc.data.demo.motionplanning import main as motion_planning
from odpc.data.demo.clip_demo import main as clip_demo
from odpc.data.demo.replay_trajectory import main as replay_trajectory
from odpc.data.demo.replay_trajectory import Args as ReplayTrajectoryArgs
from odpc.data.demo.compress import main as compress_demo


def run_generation_workflow(cfg: DictConfig, record_dir: str, jpg_quality: int = 90) -> str:
    """
    运行完整的专家演示生成工作流。
    这个函数封装了从运动规划、控制模式重放、裁剪到压缩的整个流程。

    Args:
        cfg: 包含所有步骤所需参数的 OmegaConf 配置对象。
        record_dir: 用于存放所有产物的根目录。
        jpg_quality: JPEG 压缩质量。

    Returns:
        最终生成的、经过压缩的演示文件的路径。
    """
    mp.set_start_method("spawn", force=True) # 强制设置，避免上下文问题
    
    traj_paths = []

    # 1. 运动规划生成基础轨迹
    cfg.motion_planning_args.record_dir = record_dir
    traj_path = motion_planning(cfg.motion_planning_args)
    traj_paths.append(traj_path)

    # 2. (可选) 控制模式重放
    if cfg.control_mode != cfg.motion_planning_args.env_kwargs.control_mode:
        replay_trajectory_args = ReplayTrajectoryArgs(
            traj_path=traj_path,
            obs_mode=None,
            target_control_mode=cfg.control_mode,
            save_traj=True,
            allow_failure=not cfg.only_count_success,
            num_envs=cfg.num_procs,
        )
        traj_path = replay_trajectory(replay_trajectory_args)
        traj_paths.append(traj_path)

    # 3. (可选) 裁剪轨迹
    if cfg.get("clip", False):
        traj_path = clip_demo(traj_path)
        traj_paths.append(traj_path)

    # 4. 观测模式重放，生成最终观测
    replay_trajectory_args = ReplayTrajectoryArgs(
        traj_path=traj_path,
        obs_mode=cfg.obs_mode,
        target_control_mode=cfg.control_mode,
        save_traj=True,
        allow_failure=not cfg.only_count_success,
        num_envs=cfg.num_procs,
        use_env_states=True,
    )
    final_traj_path = replay_trajectory(replay_trajectory_args)
    OmegaConf.save(cfg, final_traj_path.replace(".h5", ".yaml"), resolve=True)

    # 5. 压缩最终轨迹
    compress_args = argparse.Namespace(
        traj_path=final_traj_path,
        jpg_quality=jpg_quality,
        num_procs=cfg.num_procs,
    )
    compressed_traj_path = compress_demo(compress_args)
    
    # 6. 清理所有中间文件, 
    #   final_traj_path 对应的 json 文件需要保留, compress 不会生成对应的 json 文件, 需要使用原来的
    for path in traj_paths:
        if os.path.exists(path):
            os.remove(path)
        json_path = path.replace(".h5", ".json")
        if os.path.exists(json_path):
            os.remove(json_path)
    if os.path.exists(final_traj_path):
        os.remove(final_traj_path)

    print(f"Workflow complete. Final compressed trajectory at: {compressed_traj_path}")
    return compressed_traj_path 