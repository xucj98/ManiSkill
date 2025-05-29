import os
import argparse

import numpy as np
import h5py
import cv2
from diffusion_policy.math_utils import (
    pose_to_matrix, extrinsic_to_t_cam_world,
)

from odpc.utils.visualize import visualize_pose


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-path', type=str, default='demos/peg_insertion_demo.state_dict+rgb+depth+segmentation.pd_ee_pose.physx_cpu.h5')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    with h5py.File(args.traj_path, 'r') as file:
        
        for traj_key in file.keys():
            cam0_peg_poses = file[f'{traj_key}/obs/extra/cam0_peg_pose'][:]
            images_raw = file[f'{traj_key}/obs/sensor_data/base_camera/rgb'][:]
            intrinsic = file[f'{traj_key}/obs/sensor_param/base_camera/intrinsic_cv'][0]
            # peg_half_size = file[f'{traj_key}/obs/extra/peg_half_size'][0]
            
            key = None

            for pose, rgb in zip(cam0_peg_poses, images_raw):
                vis = visualize_pose(rgb, pose, intrinsic)
                cv2.imshow('vis', vis[:, :, ::-1])
                key = cv2.waitKey(10)
                if key == 27 or key == ord('q'):
                    break
            
            if key == 27 or key == ord('q'):
                break

        cv2.destroyAllWindows()
            