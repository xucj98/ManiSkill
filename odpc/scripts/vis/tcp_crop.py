import os
import argparse

import numpy as np
import h5py
import cv2

from odpc.data.obs_processors.tcp_crop import TcpCropProcessor
from odpc.utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-path', type=str, default='demos/peg_insertion_demo.state_dict+rgb+depth+segmentation.pd_ee_pose.physx_cpu.h5')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    processor = TcpCropProcessor(
        crop_size=(128, 128),
        padding_value=0,
        camera_name="base_camera",
        output_postfix="_crop",
    )

    with h5py.File(args.traj_path, 'r') as file:
        
        for traj_key in file.keys():
            obs = {
                "sensor_data": {
                    "base_camera": {
                        "rgb": file[f'{traj_key}/obs/sensor_data/base_camera/rgb'][:],
                        "depth": file[f'{traj_key}/obs/sensor_data/base_camera/depth'][:],
                        "segmentation": file[f'{traj_key}/obs/sensor_data/base_camera/segmentation'][:],
                    }
                },
                "sensor_param": {
                    "base_camera": {
                        "intrinsic_cv": file[f'{traj_key}/obs/sensor_param/base_camera/intrinsic_cv'][:],
                    }
                },
                "extra": {
                    "tcp_pose": file[f'{traj_key}/obs/extra/tcp_pose'][:],
                    "cam0_world_pose": file[f'{traj_key}/obs/extra/cam0_world_pose'][:],
                }
            }
            
            obs = processor.process(obs)
            images = obs["sensor_data"]["base_camera"]
            
            key = None

            for i in range(images["rgb"].shape[0]):
                rgb = images["rgb"][i]
                rgb_crop = images["rgb_crop"][i]
                cv2.imshow('rgb', rgb[:, :, ::-1])
                cv2.imshow('rgb_crop', rgb_crop[:, :, ::-1])
                depth = images["depth"][i]
                depth_crop = images["depth_crop"][i]
                cv2.imshow('depth', depth / 3000)
                cv2.imshow('depth_crop', depth_crop / 3000)
                
                key = cv2.waitKey(10)
                if key == 27 or key == ord('q'):
                    break
            
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()
            