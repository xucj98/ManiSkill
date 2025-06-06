import os
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf
from dataclasses import dataclass


from odpc.data.demo.motionplanning import main as motion_planning
from odpc.data.demo.clip_demo import main as clip_demo
from odpc.data.demo.replay_trajectory import main as replay_trajectory
from odpc.data.demo.replay_trajectory import Args as ReplayTrajectoryArgs

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/demo/peg-insertion.yaml")
    parser.add_argument("--record_dir", type=str, default="demos")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)
 
    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()
    mp.set_start_method("spawn")
    
    # todo: 支持 xarm6, 
    cfg.motion_planning_args.record_dir = args.record_dir
    motion_planning(cfg.motion_planning_args)
    traj_path = os.path.join(args.record_dir, f"{cfg.traj_name}.h5")
    
    if cfg.control_mode != cfg.motion_planning_args.env_kwargs.control_mode:
        cfg.control_mode = cfg.motion_planning_args.env_kwargs.control_mode
        #todo: 转换控制模式

    clip_demo(traj_path)

    traj_path = traj_path.replace('.h5', f'_clip.h5')
    replay_trajectory_args = ReplayTrajectoryArgs(
        traj_path=traj_path,
        obs_mode=cfg.obs_mode,
        target_control_mode=cfg.control_mode,
        save_traj=True,
        allow_failure=not cfg.only_count_success,
        num_envs=cfg.num_procs,
        use_env_states=True,
    )
    replay_trajectory(replay_trajectory_args)