import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
from typing import List
from omegaconf import DictConfig

from odpc.utils import utils
from odpc.data.obs_processors.base_processor import BaseObsProcessor
from odpc.data.act_processors.base_processor import BaseActProcessor
class ODPCDataset(Dataset):
    def __init__(
            self,
            data_path,
            obs_horizon,
            pred_horizon,
            slices_step=1,
            num_traj=None,
            clip_traj=False,
            obs_processor_configs: List[DictConfig] = [],
            act_processor_configs: List[DictConfig] = [],
    ):
        self.data_path = data_path
        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon

        self.obs_processors: List[BaseObsProcessor] = []
        for obs_processor_config in obs_processor_configs:
            self.obs_processors.append(utils.instantiate_from_config(obs_processor_config))

        self.act_processors: List[BaseActProcessor] = []
        for act_processor_config in act_processor_configs:
            self.act_processors.append(utils.instantiate_from_config(act_processor_config))

        with h5py.File(self.data_path, "r") as file:
            keys = list(file.keys())
            if num_traj is not None:
                # assert num_traj <= len(keys), f"num_traj: {num_traj} > len(keys): {len(keys)}"
                keys = sorted(keys, key=lambda x: int(x.split("_")[-1]))
                keys = keys[:num_traj]

            self.traj_keys = keys
            self.traj_lens = []
            self.slices = []
            total_transitions = 0

            pbar = tqdm(total=len(keys), desc="Prepare dataset.")

            for traj_idx, traj_key in enumerate(self.traj_keys):
                traj_len = file[f'{traj_key}/actions'].shape[0]
                self.traj_lens.append(traj_len)
                total_transitions += traj_len

                # poses_cam0_peg = file[f'{traj_key}/obs/extra/cam0_peg_pose']
                poses_world_peg = file[f'{traj_key}/obs/extra/peg_pose']
                # cam0_extrinsic = file[f'{traj_key}/obs/sensor_param/base_camera/extrinsic_cv']

                if clip_traj:
                    peg_z = poses_world_peg[:-1, 2]
                    peg_z = peg_z - np.min(peg_z)
                    traj_start = np.where(peg_z > 1e-3)[0][0]
                else:
                    traj_start = 0

                # |o|o|                             observations: 2
                # | |a|a|a|a|a|a|a|a|               actions executed: 8
                # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
                pad_after = pred_horizon - obs_horizon
                # Pad after the trajectory, so all the observations are utilized in training
                # Note that in the original code, pad_after = act_horizon - 1, but I think this is not the best choice
                self.slices += [
                    (traj_idx, start, start + pred_horizon)
                    for start in range(traj_start, traj_len - pred_horizon + pad_after, slices_step)
                ]  # slice indices follow convention [start, end)

                pbar.update(1)

        print(
            f"Total transitions: {total_transitions}, Total obs sequences: {len(self.slices)}"
        )

        self._h5_file = None

    def _ensure_h5_open(self):
        if self._h5_file is None:
            self._h5_file = h5py.File(self.data_path, 'r')

    def __getitem__(self, index):
        self._ensure_h5_open()

        traj_idx, start, end = self.slices[index]
        traj_key = self.traj_keys[traj_idx]
        L = self.traj_lens[traj_idx]

        def get_slice_data(file, slice):
            if isinstance(file, (h5py.File, h5py.Group)):
                return {key: get_slice_data(file[key], slice) for key in file.keys()}
            elif isinstance(file, h5py.Dataset):
                return file[slice]
            else:
                raise NotImplementedError(f"H5 file type {type(file)} not supported")

        obs = get_slice_data(self._h5_file[f"{traj_key}/obs"], slice(start, start + self.obs_horizon))
        for obs_processor in self.obs_processors:
            obs = obs_processor.process(obs)

        act = get_slice_data(self._h5_file[f"{traj_key}/obs/extra"], slice(start, end + 1))
        for key in act.keys():
            act[key] = utils.expand_dim_to(act[key], 0, self.pred_horizon + 1)
        for act_processor in self.act_processors:
            act = act_processor.process(act)

        for sensor in obs["sensor_data"].values():
            for modality, data in sensor.items():
                sensor[modality] = np.transpose(data, (0, 3, 1, 2))

        return {
            "observations": obs,
            "actions": act["actions"],
        }

    def __len__(self):
        return len(self.slices)

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()  # 进程退出时，安全关闭自己的句柄
            self._h5_file = None
