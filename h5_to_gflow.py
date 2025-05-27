import argparse

import cv2
import h5py
import os
import gymnasium as gym
from tqdm import tqdm
import numpy as np

import mani_skill
from mani_skill.utils import io_utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-path', required=True)
    parser.add_argument('--output-path', required=True)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()

    json_path = args.traj_path.replace('.h5', '.json')
    json_data = io_utils.load_json(json_path)

    env_info = json_data["env_info"]
    env_id = env_info["env_id"]
    ori_env_kwargs = env_info["env_kwargs"]
    env_kwargs = ori_env_kwargs.copy()

    env = gym.make(env_id, **env_kwargs)
    segmentation_id_map = env.unwrapped.segmentation_id_map
    # for k, v in segmentation_id_map.items():
    #     print(k, v.name)

    with h5py.File(args.traj_path, 'r') as file:
        traj_keys = list(file.keys())
        traj_keys.sort(key = lambda x: int(x.split('_')[-1]))
        # traj_keys = traj_keys[:10]
        for traj_key in tqdm(traj_keys):
            outputs = os.path.join(args.output_path, traj_key)
            os.makedirs(outputs, exist_ok=True)

            peg_poses = file[f'{traj_key}/obs/extra/peg_pose'][:]
            z = peg_poses[:, 2]
            dz = z - np.min(z)
            start = np.min(np.where(dz > 2e-3)[0])

            rgb = file[f'{traj_key}/obs/sensor_data/base_camera/rgb'][start:]
            bgr = rgb[..., ::-1]

            depth = file[f'{traj_key}/obs/sensor_data/base_camera/depth'][start:]
            depth[depth < 0] = 0
            depth[depth > 3000] = 0
            depth = depth.astype(np.uint16)

            segmentation_raw = file[f'{traj_key}/obs/sensor_data/base_camera/segmentation'][start:]
            segmentation_raw = segmentation_raw[..., 0]
            segmentation = np.zeros_like(segmentation_raw)

            for k, v in segmentation_id_map.items():
                if 'panda' in v.name:
                    segmentation[segmentation_raw == k] = 2
                if 'peg' in v.name:
                    segmentation[segmentation_raw == k] = 1

            cam_peg_poses = file[f'{traj_key}/obs/extra/cam0_peg_pose'][start:]
            intrinsic = file[f'{traj_key}/obs/sensor_param/base_camera/intrinsic_cv'][start:]

            np.save(os.path.join(outputs, 'camera_in.npy'), intrinsic[0])
            np.save(os.path.join(outputs, 'object_poses.npy'), cam_peg_poses)

            for i in range(len(rgb)):
                cv2.imwrite(os.path.join(outputs, f'rgb_{i}.png'), bgr[i])
                cv2.imwrite(os.path.join(outputs, f'depth_{i}.png'), depth[i])
                np.save(os.path.join(outputs, f'segmentation_{i}.npy'), segmentation[i])


            # print(rgb.shape, depth.shape, segmentation.shape, cam_peg_poses.shape)


            # print(file[f'{traj_key}/obs/extra'].keys())