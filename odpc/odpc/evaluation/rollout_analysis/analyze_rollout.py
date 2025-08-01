import os
import json
import h5py
import pathlib
from typing import List, Dict

import numpy as np

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from mani_skill.utils import common

from odpc.data.rollout_dataset import RolloutDataset
from odpc.utils.utils import instantiate_from_config, parse_config_expr
from odpc.models.policy import BasePolicy
from odpc.evaluation.rollout_analysis.key_frames_finder.base import BaseKeyFrameFinder


def filter_failed_traj(
        success_type: str,
        rollout_path: str,
) -> List[str]:
    
    success_traj_keys = []
    failed_traj_keys = []
    
    with h5py.File(rollout_path, "r") as f:
        for traj_key in f.keys():
            traj_data = f[traj_key]
            success_array = traj_data["success"][:]
            if success_type == "success_once":
                success = success_array.any()
            elif success_type == "success_at_end":
                success = success_array[-1]
            else:
                raise ValueError(f"Invalid success type: {success_type}")

            if success:
                success_traj_keys.append(traj_key)
            else:
                failed_traj_keys.append(traj_key)

        print(f"Success trajectories: {len(success_traj_keys)}/{len(f.keys())}")

    return failed_traj_keys


def concat_traj_data(
        traj_data: List[Dict],
) -> Dict:
    
    res = {}
    for key, data in traj_data[0].items():
        if isinstance(data, np.ndarray):
            res[key] = np.concatenate([d[key] for d in traj_data], axis=0)
        elif isinstance(data, torch.Tensor):
            res[key] = torch.cat([d[key] for d in traj_data], dim=0)
        elif isinstance(data, dict):
            res[key] = concat_traj_data([d[key] for d in traj_data])
        else:
            raise ValueError(f"Invalid data type: {type(data)}")
    return res


def process_traj(
        traj_data: List[Dict],
        key_frame_finder: BaseKeyFrameFinder,
        device: torch.device,
):
    traj_data = concat_traj_data(traj_data)
    traj_data = common.to_tensor(traj_data, device)
    key_frames = key_frame_finder.find_key_frames_from_trajectory(traj_data)
    key_frames["frame_idx"] = traj_data["frame_idx"].tolist()
    key_frames["is_key_frame"] = key_frames["is_key_frame"].tolist()
    key_frames["metric_values"] = key_frames["metric_values"].tolist()
    return key_frames


def analyze_rollout(
        config: DictConfig,
) -> pathlib.Path:

    config.save_dir = parse_config_expr(config.save_dir)
    os.makedirs(config.save_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.save_dir, "config.yaml"))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model: BasePolicy = instantiate_from_config(config.model)
    ckpt = torch.load(config.ckpt_path)
    if config.use_ema:
        model.load_state_dict(ckpt["ema_model"])
    else:
        model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()

    train_dataset = instantiate_from_config(config.train_dataset)

    key_frame_finder: BaseKeyFrameFinder = instantiate_from_config(
        config.key_frame_finder,
        model=model,
        train_dataset=train_dataset,
    )

    failed_traj_keys = filter_failed_traj(config.success_type, config.rollout_path)
    rollout_dataset: RolloutDataset = instantiate_from_config(
        config.rollout_dataset,
        traj_keys=failed_traj_keys,
    )

    dataloader = DataLoader(
        rollout_dataset,
        batch_size=1,
        num_workers=config.num_workers,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )

    cur_traj_idx = None
    traj_data = []
    results = {}

    for data in dataloader:
        if cur_traj_idx != data["traj_idx"]:
            if cur_traj_idx is not None:
                key_frames = process_traj(traj_data, key_frame_finder, device)
                traj_key = rollout_dataset.traj_info[cur_traj_idx][1]
                results[traj_key] = key_frames
            cur_traj_idx = data["traj_idx"]
            traj_data = []
        traj_data.append(data)

    if cur_traj_idx is not None:
        key_frames = process_traj(traj_data, key_frame_finder, device)
        traj_key = rollout_dataset.traj_info[cur_traj_idx][1]
        results[traj_key] = key_frames

    result_path = pathlib.Path(config.save_dir) / "rollout_analysis_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f)
    print(f"Rollout analysis results saved to {result_path}")

    return result_path
