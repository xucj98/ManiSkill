import numpy as np
from typing import List, Optional
from omegaconf import DictConfig

from odpc.utils import utils
from odpc.data.imitation_dataset import ImitationDataset

class RolloutDataset(ImitationDataset):
    def __init__(
            self,
            data_path,
            obs_horizon,
            pred_horizon,
            slices_step=1,
            traj_keys: Optional[List[str]] = None,
            obs_processor_configs: List[DictConfig] = [],
            act_processor_configs: List[DictConfig] = [],
    ):
        super().__init__(
            data_paths=[data_path],
            obs_horizon=obs_horizon,
            pred_horizon=pred_horizon,
            slices_step=slices_step,
            obs_processor_configs=obs_processor_configs,
            act_processor_configs=act_processor_configs,
        )
        
        if traj_keys is not None:
            traj_keys = set(traj_keys)
            new_slices = []
            for traj_idx, start, end in self.slices:
                traj_key = self.traj_info[traj_idx][1]
                if traj_key in traj_keys:
                    new_slices.append((traj_idx, start, end))
            self.slices = new_slices

        print(f"After filtering, total obs sequences: {len(self.slices)}")

    def __getitem__(self, index):
        self._ensure_h5_open()

        traj_idx, start, end = self.slices[index]
        dataset_id, traj_key, _ = self.traj_info[traj_idx]
        h5_file = self._h5_files[dataset_id]
    
        # 在 rollout dataset中，由于使用了 FrameStackWrapper, 因此保存下来的 obs 是 [T, obs_horizon, ...]
        # 这里就不用截取 slice(start, start + self.obs_horizon) 了，直接取 start 的 obs 就好了
        obs = self._get_slice_data(h5_file[f"{traj_key}/obs"], start)
        for obs_processor in self.obs_processors:
            obs = obs_processor.process(obs)

        act = self._get_slice_data(h5_file[f"{traj_key}/obs/extra"], slice(start, end + 1))
        for key in act.keys():
            act[key] = utils.expand_dim_to(act[key], 0, self.pred_horizon + 1)
        act["actions"] = self._get_slice_data(h5_file[f"{traj_key}/actions"], slice(start, end))
        act["actions"] = utils.expand_dim_to(act["actions"], 0, self.pred_horizon)
        for act_processor in self.act_processors:
            act = act_processor.process(act)

        for sensor in obs["sensor_data"].values():
            for modality, data in sensor.items():
                sensor[modality] = np.transpose(data, (0, 3, 1, 2))

        return {
            "observations": obs,
            "actions": act["actions"],
            "traj_idx": traj_idx,
            "frame_idx": start,
        }