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
    parser.add_argument('--headless', action='store_true')
    return parser.parse_args()



if __name__ == "__main__":
    args = get_args()

    with h5py.File(args.traj_path, 'r') as file:
        
        if args.headless:
            h, w = file[list(file.keys())[0]]['obs/sensor_data/base_camera/rgb'].shape[1:3]
            writer = cv2.VideoWriter(f'peg_pose.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 50, (w, h))
       
        index = 0

        for traj_key in file.keys():
            cam0_peg_poses = file[f'{traj_key}/obs/extra/cam0_peg_pose'][:]
            images_raw = file[f'{traj_key}/obs/sensor_data/base_camera/rgb'][:]
            intrinsic = file[f'{traj_key}/obs/sensor_param/base_camera/intrinsic_cv'][0]
            # peg_half_size = file[f'{traj_key}/obs/extra/peg_half_size'][0]
            
            key = None

            for pose, rgb in zip(cam0_peg_poses, images_raw):
                vis = visualize_pose(rgb, pose, intrinsic)
                index += 1
                print(index, end='\r')
                if args.headless:
                    writer.write(vis[:, :, ::-1])
                    if index > 5000:
                        break
                else:
                    cv2.imshow('vis', vis[:, :, ::-1])
                    key = cv2.waitKey(10)
                    if key == 27 or key == ord('q'):
                        break
            if args.headless:
                if index > 5000:
                    break
            elif key == 27 or key == ord('q'):
                break

        if args.headless:
            writer.release()
        else:
            cv2.destroyAllWindows()
            