import json
import h5py
import pathlib
from typing import List, Dict

import numpy as np

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from mani_skill.utils import common

from odpc.utils.utils import instantiate_from_config, get_data_shape
from odpc.models.policy import BasePolicy
from odpc.evaluation.ood_finder.base_ood_finder import BaseOODFinder
from odpc.utils.visualize import visualize_video_with_metric

def filter_failed_traj(
        config: DictConfig,
) -> pathlib.Path:
    
    success_traj_keys = []
    failed_traj_keys = []
    
    with h5py.File(config.rollout_path, "r") as f:
        print(get_data_shape(f[list(f.keys())[0]]))
        for traj_key in f.keys():
            traj_data = f[traj_key]
            success_array = traj_data["success"][:]
            if config.success_type == "success_once":
                success = success_array.any()
            elif config.success_type == "success_at_end":
                success = success_array[-1]
            else:
                raise ValueError(f"Invalid success type: {config.success_type}")

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
        ood_finder: BaseOODFinder,
        device: torch.device,
):
    traj_data = concat_traj_data(traj_data)
    traj_data = common.to_tensor(traj_data, device)
    loss = ood_finder.compute_diffusion_loss(
        traj_data["observations"],
        traj_data["actions"],
    )
    video = traj_data["observations"]["sensor_data"]["base_camera"]["rgb"].cpu().numpy()[:, 0, ...]
    video = np.transpose(video, (0, 2, 3, 1))
    loss = loss.cpu().numpy()

    visualize_video_with_metric(video, loss)

def rollout_analysis(
        config: DictConfig,
) -> pathlib.Path:

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

    ood_finder: BaseOODFinder = instantiate_from_config(
        config.ood_finder,
        model=model,
        dataset=train_dataset,
    )

    failed_traj_keys = filter_failed_traj(config)
    rollout_dataset = instantiate_from_config(
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

    for data in dataloader:
        if cur_traj_idx != data["traj_idx"]:
            if cur_traj_idx is not None:
                process_traj(traj_data, ood_finder, device)
            cur_traj_idx = data["traj_idx"]
            traj_data = []
        traj_data.append(data)

    if cur_traj_idx is not None:
        process_traj(traj_data, ood_finder, device)

