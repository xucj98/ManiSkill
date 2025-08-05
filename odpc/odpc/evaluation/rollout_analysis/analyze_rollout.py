import os
import json
import h5py
import pathlib
from tqdm import tqdm
from typing import List, Dict, Optional, Tuple

import cv2
import numpy as np

import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf

from mani_skill.utils import common

from odpc.data.rollout_dataset import RolloutDataset
from odpc.utils.utils import instantiate_from_config, parse_config_expr, save_h5_data
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


def get_sample_data(
        data: dict,
        indexes: any,
) -> dict:
    res = {}
    for key, value in data.items():
        if isinstance(value, dict):
            res[key] = get_sample_data(value, indexes)
        else:
            res[key] = value[indexes]
    return res


def sample_key_frames(
        key_frames: dict,
        sample_step: int = 10,
) -> tuple:
    """
    从所有可能的关键帧中进行采样，根据 sample_step 采样间隔，并返回对应帧的索引
    """
    
    is_key_frame = key_frames["is_key_frame"]
    sample_idx = []
    for idx in range(len(is_key_frame)):
        if is_key_frame[idx]:
            if len(sample_idx) == 0 or idx - sample_idx[-1] >= sample_step:
                sample_idx.append(idx)

    return sample_idx

def process_traj(
        traj_key: str,
        traj_data: List[Dict],
        key_frame_finder: BaseKeyFrameFinder,
        sample_step: int,
        device: torch.device,
        save_dir: Optional[pathlib.Path] = None,
)->Tuple[Dict[str, list], Dict[str, np.ndarray]]:
    """
    处理单条（失败）轨迹，
        1. 拼接轨迹数据
        2. 找到关键帧
        3. 采样关键帧数据，得到对应的 env_states 和 sensor_data
    """
    traj_data = concat_traj_data(traj_data)
    traj_data = common.to_tensor(traj_data, device)
   
    key_frames = key_frame_finder.find_key_frames_from_trajectory(traj_data)
    key_frames["frame_idx"] = traj_data["frame_idx"].tolist()
    key_frames["is_key_frame"] = key_frames["is_key_frame"].tolist()
    key_frames["metric_values"] = key_frames["metric_values"].tolist()

    sample_idx = sample_key_frames(key_frames, sample_step)
    key_frames["sample_idx"] = sample_idx
    env_states = get_sample_data(traj_data["env_states"], sample_idx)
    env_states = common.to_numpy(env_states)
    
    if save_dir is not None:
        sensor_data = get_sample_data(traj_data["observations"]["sensor_data"], sample_idx)
        sensor_names = [k for k in sensor_data if "rgb" in sensor_data[k]]
        for i in range(len(sensor_data[sensor_names[0]]["rgb"])):
            images = []
            for sensor_name in sensor_names:
                rgb = sensor_data[sensor_name]["rgb"][i, 0]  # [3, h, w]
                rgb = rgb.cpu().numpy()
                rgb = np.transpose(rgb, (1, 2, 0))  # [h, w, 3]
                rgb = (rgb * 255).astype(np.uint8)
                images.append(rgb)
            images = np.concatenate(images, axis=1)  # Warning: 这里假设所有 sensor_data 的 rgb 维度相同
            img_path = os.path.join(save_dir, f"{traj_key}_{sample_idx[i]}.jpg")
            cv2.imwrite(img_path, images[:, :, ::-1])

    return key_frames, env_states


def analyze_rollout(
        config: DictConfig,
) -> pathlib.Path:

    config.save_dir = parse_config_expr(config.save_dir)
    os.makedirs(config.save_dir, exist_ok=True)
    OmegaConf.save(config, os.path.join(config.save_dir, "config.yaml"))
    if config.save_rgb:
        os.makedirs(os.path.join(config.save_dir, "rgb"), exist_ok=True)

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

    # results: {
    #     traj_key: {                   # traj_key 是轨迹的唯一标识
    #         is_key_frame: [bool],     # 当前帧是否是关键帧
    #         metric_values: [float],   # 当前帧的 metric 值
    #         frame_idx: [int],         # 当前帧在原始的 rollout.h5 轨迹中的帧索引
    #         sample_idx: [int]         # 最终采样出来的关键状态的帧索引，注意这个索引是当前帧的索引，不是原始的 rollout.h5 轨迹中的帧索引
    #     }
    # }
    results = {}  
    
    key_env_states = []  # key_env_states: [{env_states}]

    for data in tqdm(dataloader, desc="Analyzing rollout"):
        if cur_traj_idx != data["traj_idx"]:
            if cur_traj_idx is not None:
                traj_key = rollout_dataset.traj_info[cur_traj_idx][1]        
                key_frames, env_states = process_traj(
                    traj_key,
                    traj_data,
                    key_frame_finder,
                    config.key_frame_sample_step,
                    device,
                    os.path.join(config.save_dir, "rgb") if config.save_rgb else None,
                )
                results[traj_key] = key_frames
                key_env_states.append(env_states)

            cur_traj_idx = data["traj_idx"]
            traj_data = []
        traj_data.append(data)

    if cur_traj_idx is not None:
        traj_key = rollout_dataset.traj_info[cur_traj_idx][1]
        key_frames, env_states = process_traj(
            traj_key,
            traj_data,
            key_frame_finder,
            config.key_frame_sample_step,
            device,
            os.path.join(config.save_dir, "rgb") if config.save_rgb else None,
        )
        results[traj_key] = key_frames
        key_env_states.append(env_states)

    key_env_states = concat_traj_data(key_env_states)
    key_env_states_path = pathlib.Path(config.save_dir) / "key_env_states.h5"
    with h5py.File(key_env_states_path, "w") as f:
        save_h5_data(key_env_states, f)
    print(f"Key env states saved to {key_env_states_path}")

    result_path = pathlib.Path(config.save_dir) / "rollout_analysis_results.json"
    with open(result_path, "w") as f:
        json.dump(results, f)
    print(f"Rollout analysis results saved to {result_path}")

    return result_path
