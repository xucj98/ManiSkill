import h5py
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Dataset


class ODPCDataset(Dataset):
    def __init__(
            self,
            data_path,
            obs_horizon,
            pred_horizon,
            slices_step=1,
            num_traj=None,
    ):
        self.data_path = data_path

        self.obs_horizon, self.pred_horizon = obs_horizon, pred_horizon

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

                peg_z = poses_world_peg[:-1, 2]
                peg_z = peg_z - np.min(peg_z)
                traj_start = np.where(peg_z > 1e-3)[0][0]

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

        def get_slice_data(file):
            if isinstance(file, (h5py.File, h5py.Group)):
                return {key: get_slice_data(file[key]) for key in file.keys()}
            elif isinstance(file, h5py.Dataset):
                return file[start: start + self.obs_horizon]
            else:
                raise NotImplementedError(f"H5 file type {type(file)} not supported")

        sensor_data = get_slice_data(self._h5_file[f"{traj_key}/obs/sensor_data"])
        obs_seq = {
            key: np.transpose(
                np.concatenate([v[key] for v in sensor_data.values()], axis=-1), axes=(0, 3, 1, 2)
            ) for key in ["rgb", "depth"]
        }

        poses_peg = self._h5_file[f'{traj_key}/obs/extra/peg_pose'][start: end + 1]
        poses_ee = self._h5_file[f'{traj_key}/obs/extra/tcp_pose'][start: end + 1]
        poses_base = self._h5_file[f'{traj_key}/obs/extra/base_pose'][start: end + 1]
        poses_cam0_world = self._h5_file[f'{traj_key}/obs/extra/cam0_world_pose'][start: end + 1]
        intrinsic = self._h5_file[f'{traj_key}/obs/sensor_param/base_camera/intrinsic_cv'][0]
        act_seq = self._h5_file[f"{traj_key}/actions"][start: end]
        if end > L:
            poses_peg, poses_base, poses_ee, poses_cam0_world = [
                np.concatenate([raw, raw[-1:].repeat(end - L, 0)], axis=0)
                for raw in [poses_peg, poses_base, poses_ee, poses_cam0_world]
            ]

            act_seq = np.concatenate([act_seq, np.zeros_like(act_seq[-1:]).repeat(end - L, 0)], axis=0)

        return {
            "observations": obs_seq,
            "actions": act_seq,
            "poses_peg": poses_peg,
            "poses_ee": poses_ee,
            "poses_base": poses_base,
            "poses_cam0_world": poses_cam0_world,
            "intrinsic": intrinsic,
        }

    def __len__(self):
        return len(self.slices)

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()  # 进程退出时，安全关闭自己的句柄
            self._h5_file = None
