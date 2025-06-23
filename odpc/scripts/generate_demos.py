import os
import argparse
import multiprocessing as mp
from omegaconf import OmegaConf


from odpc.data.demo.motionplanning import main as motion_planning
from odpc.data.demo.clip_demo import main as clip_demo
from odpc.data.demo.replay_trajectory import main as replay_trajectory
from odpc.data.demo.replay_trajectory import Args as ReplayTrajectoryArgs
from odpc.data.demo.compress import main as compress_demo

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/demo/peg-insertion.yaml")
    parser.add_argument("--record_dir", type=str, default="demos")
    parser.add_argument("--compress", action="store_true", help="Enable JPEG compression for RGB data")
    parser.add_argument("--jpg-quality", type=int, default=85, help="JPEG compression quality (0-100)")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    cli = OmegaConf.from_dotlist(unknown)
    cfg = OmegaConf.merge(cfg, cli)
 
    print(OmegaConf.to_yaml(cfg, resolve=True))

    return args, cfg


if __name__ == "__main__":
    args, cfg = get_args()
    mp.set_start_method("spawn")
    
    traj_paths = []

    # todo: 支持 xarm6, 
    cfg.motion_planning_args.record_dir = args.record_dir
    traj_path = motion_planning(cfg.motion_planning_args)
    traj_paths.append(traj_path)

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

    if cfg.clip:
        traj_path = clip_demo(traj_path)
        traj_paths.append(traj_path)

    replay_trajectory_args = ReplayTrajectoryArgs(
        traj_path=traj_path,
        obs_mode=cfg.obs_mode,
        target_control_mode=cfg.control_mode,
        save_traj=True,
        allow_failure=not cfg.only_count_success,
        num_envs=cfg.num_procs,
        use_env_states=True,
    )
    traj_path = replay_trajectory(replay_trajectory_args)
    OmegaConf.save(cfg, traj_path.replace(".h5", ".yaml"), resolve=True)

    # 添加JPEG压缩步骤
    if args.compress:
        print(f"Starting JPEG compression with quality {args.jpg_quality}...")
        compress_args = argparse.Namespace(
            traj_path=traj_path,
            jpg_quality=args.jpg_quality
        )
        compressed_traj_path = compress_demo(compress_args)
        print(f"Compression completed. Compressed file: {compressed_traj_path}")
        # 更新最终路径为压缩后的文件
        final_traj_path = compressed_traj_path
    else:
        final_traj_path = traj_path

    # 清理中间文件
    for path in traj_paths:
        if os.path.exists(path):
            os.remove(path)
        json_path = path.replace(".h5", ".json")
        if os.path.exists(json_path):
            os.remove(json_path)

    print(f"Generate {final_traj_path} successfully")
