import h5py
import pathlib
import numpy as np
from tqdm import tqdm
import random
from torch.utils.data.dataset import Dataset
from typing import List, Union, Optional, Tuple
from omegaconf import DictConfig

from odpc.utils import utils
from odpc.data.obs_processors.base_processor import BaseObsProcessor
from odpc.data.act_processors.base_processor import BaseActProcessor
from odpc.data.utils import decode_jpeg_sequence, is_compressed_rgb_dataset

class ImitationDataset(Dataset):
    def __init__(
            self,
            data_paths: List[Union[str, pathlib.Path]],
            obs_horizon: int,
            pred_horizon: int,
            slices_step: int = 1,
            num_traj: Optional[int] = None,
            valid: bool = False,
            obs_processor_configs: List[DictConfig] = [],
            act_processor_configs: List[DictConfig] = [],
            seed: int = 42,
            used_obs: Optional[List[str]] = None,
    ):
        self.data_paths = data_paths
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon
        self.used_obs = used_obs

        self.obs_processors: List[BaseObsProcessor] = []
        for obs_processor_config in obs_processor_configs:
            self.obs_processors.append(utils.instantiate_from_config(obs_processor_config))

        self.act_processors: List[BaseActProcessor] = []
        for act_processor_config in act_processor_configs:
            self.act_processors.append(utils.instantiate_from_config(act_processor_config))

        self.traj_info: List[Tuple[int, str, int]] = []  # dataset_id, traj_key, traj_len
        self.slices: List[Tuple[int, int, int]] = []  # traj_idx, start, end
        total_transitions = 0

        for dataset_id, data_path in enumerate(self.data_paths):
            with h5py.File(data_path, "r") as file:
                keys = list(file.keys())
                for k in keys:
                    self.traj_info.append((dataset_id, k, file[k]["actions"].shape[0]))
             
        if num_traj is not None:
            rng = random.Random(seed)
            rng.shuffle(self.traj_info)
            self.traj_info = self.traj_info[:num_traj] if not valid else self.traj_info[-num_traj:]

        pbar = tqdm(total=len(self.traj_info), desc="Prepare dataset.")

        for traj_idx in range(len(self.traj_info)):
            dataset_id, traj_key, traj_len = self.traj_info[traj_idx]
            total_transitions += traj_len

            # |o|o|                             observations: 2
            # | |a|a|a|a|a|a|a|a|               actions executed: 8
            # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
            pad_after = pred_horizon - obs_horizon
            # Pad after the trajectory, so all the observations are utilized in training
            # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
            self.slices += [
                (traj_idx, start, start + pred_horizon)
                for start in range(0, traj_len - pred_horizon + pad_after, slices_step)
            ]  # slice indices follow convention [start, end)

            pbar.update(1)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self._h5_files: Optional[List[h5py.File]] = None

    def _ensure_h5_open(self):
        if self._h5_files is None:
            self._h5_files = [h5py.File(data_path, 'r') for data_path in self.data_paths]

    def _get_slice_data(self, file, slice_indices, used_key=None, cur_key=""):
        """获取切片数据，支持压缩数据的自动解码"""
        if isinstance(file, (h5py.File, h5py.Group)):
            res = {}
            for key in file.keys():
                new_cur_key = key if cur_key == "" else cur_key + "." + key
                data = self._get_slice_data(file[key], slice_indices, used_key=used_key, cur_key=new_cur_key)
                if data is not None:
                    res[key] = data
            return res if len(res) > 0 else None
        
        elif isinstance(file, h5py.Dataset):
            if used_key is None or cur_key in used_key:
                data = file[slice_indices]
                if is_compressed_rgb_dataset(file):
                    data = decode_jpeg_sequence(data)
                return data
            else:
                return None
            
        else:
            raise NotImplementedError(f"H5 file type {type(file)} not supported")

    def __getitem__(self, index):
        self._ensure_h5_open()

        traj_idx, start, end = self.slices[index]
        dataset_id, traj_key, traj_len = self.traj_info[traj_idx]
        h5_file = self._h5_files[dataset_id]

        obs = self._get_slice_data(
            h5_file[f"{traj_key}/obs"], 
            slice(start, start + self.obs_horizon),
            used_key=self.used_obs,
        )
        for obs_processor in self.obs_processors:
            obs = obs_processor.process(obs)

        act = self._get_slice_data(h5_file[f"{traj_key}/obs/extra"], slice(start, end + 1))
        for key in act.keys():
            act[key] = utils.expand_dim_to(act[key], 0, self.pred_horizon + 1)
        act["actions"] = self._get_slice_data(h5_file[f"{traj_key}/actions"], slice(start, end))
        act["actions"] = utils.expand_dim_to(act["actions"], 0, self.pred_horizon)
        for act_processor in self.act_processors:
            act = act_processor.process(act)

        if "sensor_data" in obs:
            for sensor in obs["sensor_data"].values():
                for modality, data in sensor.items():
                    sensor[modality] = np.transpose(data, (0, 3, 1, 2))

        return {
            "observations": obs,
            "actions": act["actions"],
            "traj_idx": traj_idx,
        }

    def __len__(self):
        return len(self.slices)

    def __del__(self):
        if self._h5_files is not None:
            for h5_file in self._h5_files:
                h5_file.close()
            self._h5_files = None
