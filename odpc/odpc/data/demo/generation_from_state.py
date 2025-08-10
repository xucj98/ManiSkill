# odpc/odpc/data/demo/generation.py
import os
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf, DictConfig
import warnings

from odpc.data.demo.motionplanning_from_state import main as motion_planning
from odpc.data.demo.replay_trajectory import main as replay_trajectory
from odpc.data.demo.replay_trajectory import Args as ReplayTrajectoryArgs
from odpc.data.demo.compress import main as compress_demo


def run_generation_workflow(cfg: DictConfig) -> str:
    """
    运行完整的专家演示生成工作流。
    这个函数封装了从运动规划、控制模式重放到压缩的整个流程。

    Args:
        cfg: 包含所有步骤所需参数的 OmegaConf 配置对象。
       
    Returns:
        最终生成的、经过压缩的演示文件的路径。
    """
    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        # 在某些情况下（例如，如果已在另一个地方设置），再次设置会引发 RuntimeError。
        # 如果已经设置，我们可以安全地忽略这个错误。
        current_method = mp.get_start_method(allow_none=True)
        if current_method != "spawn":
            warnings.warn(f"multiprocessing start method is already set to '{current_method}', not 'spawn'. This might cause issues with CUDA.")
    
    traj_paths = []

    # 1. 运动规划生成基础轨迹
    traj_path = motion_planning(cfg.motion_planning_args)
    traj_paths.append(traj_path)

    # 2. 控制模式和观测模式重放
    replay_trajectory_args = ReplayTrajectoryArgs(
        traj_path=traj_path,
        obs_mode=cfg.obs_mode,
        target_control_mode=cfg.control_mode,
        save_traj=True,
        allow_failure=False,
        num_envs=cfg.num_procs,
        use_first_env_state=True,
    )
    final_traj_path = replay_trajectory(replay_trajectory_args)
    OmegaConf.save(cfg, final_traj_path.replace(".h5", ".yaml"), resolve=True)

    # 3. 压缩最终轨迹
    compress_args = argparse.Namespace(
        traj_path=final_traj_path,
        jpg_quality=cfg.jpg_quality,
        num_procs=cfg.num_procs,
    )
    compressed_traj_path = compress_demo(compress_args)
    
    # 4. 清理所有中间文件, 
    #   final_traj_path 对应的 json 文件需要保留, compress 不会生成对应的 json 文件, 需要使用原来的
    for path in traj_paths:
        os.remove(path)
        json_path = path.replace(".h5", ".json")
        os.remove(json_path)
    os.remove(final_traj_path)

    print(f"Workflow complete. Final compressed trajectory at: {compressed_traj_path}")
    return compressed_traj_path 