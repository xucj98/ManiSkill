import os
import shutil
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf
from odpc.configs.paths import *
from dataclasses import dataclass

from mani_skill.examples.motionplanning.panda.run import main as panda_motion_planning
from mani_skill.examples.motionplanning.xarm6.run import main as xarm6_motion_planning
from mani_skill.trajectory.replay_trajectory import main as replay_trajectory
from mani_skill.trajectory.replay_trajectory import Args as ReplayTrajectoryArgs

@dataclass
class MotionPlanningArgs:
    env_id: str
    obs_mode: str = None
    num_traj: int = 10
    only_count_success: bool = True
    reward_mode: str = None
    sim_backend: str = "auto"
    render_mode: str = "rgb_array"
    vis: bool = False
    save_video: bool = False
    traj_name: str = "demo"
    shader: str = "default"
    record_dir: str = DATASET_ROOT
    num_procs: int = 16


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data/peg_insertion_demo.yaml")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    cfg = OmegaConf.load(args.config)
    mp.set_start_method("spawn")
    
    # todo: 支持 xarm6, 支持设置 env_kwargs
    motion_planning_args = MotionPlanningArgs(
        env_id=cfg.env_id,
        num_traj=cfg.num_traj,
        only_count_success=cfg.only_count_success,
        traj_name=cfg.traj_name,
        num_procs=cfg.num_procs,
    )
    panda_motion_planning(motion_planning_args)
    
    src_dir = os.path.join(DATASET_ROOT, cfg.env_id, "motionplanning")
    
    replay_trajectory_args = ReplayTrajectoryArgs(
        traj_path=os.path.join(src_dir, f"{cfg.traj_name}.h5"),
        obs_mode=cfg.obs_mode,
        target_control_mode=cfg.control_mode,
        save_traj=True,
        allow_failure=not cfg.only_count_success,
        num_envs=cfg.num_procs,
    )
    replay_trajectory(replay_trajectory_args)