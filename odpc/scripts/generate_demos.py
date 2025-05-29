import os
import shutil
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf
from dataclasses import dataclass

from mani_skill.trajectory.replay_trajectory import main as replay_trajectory
from mani_skill.trajectory.replay_trajectory import Args as ReplayTrajectoryArgs

import odpc.envs
import odpc.data.demo.panda_solutions as panda_solutions
from odpc.data.demo.motionplanning import main as motion_planning, MP_SOLUTIONS

MP_SOLUTIONS["PegInsertionSide-v2"] = panda_solutions.solvePegInsertionSide


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/demo/peg_insertion.yaml")
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
    
    src_dir = os.path.join(args.record_dir, cfg.env_id, "motionplanning")
    
    replay_trajectory_args = ReplayTrajectoryArgs(
        traj_path=os.path.join(src_dir, f"{cfg.traj_name}.h5"),
        obs_mode=cfg.obs_mode,
        target_control_mode=cfg.control_mode,
        save_traj=True,
        allow_failure=not cfg.only_count_success,
        num_envs=cfg.num_procs,
    )
    replay_trajectory(replay_trajectory_args)