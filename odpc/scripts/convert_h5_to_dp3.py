import cv2
import h5py
import argparse
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import time

from odpc.data.utils import create_point_cloud, batch_farthest_point_sampling

def decode_jpeg(jpeg_bytes):
    bgr_image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

def main(args):
    print(f"Starting conversion of {args.input_file}...")
    print("Output will be a high-fidelity copy with point clouds added and RGB/Depth removed.")
    t_start = time.time()

    with h5py.File(args.input_file, 'r') as f_in, h5py.File(args.output_file, 'w') as f_out:
        # Copy top-level attributes
        for key, value in f_in.attrs.items():
            f_out.attrs[key] = value

        traj_keys = sorted([key for key in f_in.keys() if key.startswith('traj_')])
        if not traj_keys:
            print("No trajectories found in the input file.")
            return

        camera_names = list(f_in[traj_keys[0]]['obs']['sensor_data'].keys())
        print(f"Detected cameras: {camera_names}")

        if args.num_trajectories > 0:
            traj_keys = traj_keys[:args.num_trajectories]
        print(f"Processing {len(traj_keys)} trajectories for {len(camera_names)} camera(s)...")

        for traj_key in tqdm(traj_keys, desc="Processing Trajectories"):
            # 1. Copy the entire trajectory group from source to destination
            f_in.copy(traj_key, f_out)

            # 2. For each camera, process point clouds and replace the old data
            for camera_name in camera_names:
                traj_group_in = f_in[traj_key]
                num_steps = traj_group_in['actions'].shape[0]
                
                cropped_pcs_list = []
                for i in range(num_steps):
                    rgb_image = decode_jpeg(traj_group_in['obs']['sensor_data'][camera_name]['rgb'][i])
                    depth_image = traj_group_in['obs']['sensor_data'][camera_name]['depth'][i].squeeze()
                    intrinsic = traj_group_in['obs']['sensor_param'][camera_name]['intrinsic_cv'][i]
                    extrinsic_3x4 = traj_group_in['obs']['sensor_param'][camera_name]['extrinsic_cv'][i]
                    pc = create_point_cloud(rgb_image, depth_image, intrinsic, extrinsic_3x4=extrinsic_3x4, crop_range=args.crop_range)
                    cropped_pcs_list.append(pc)
                
                final_pcs = batch_farthest_point_sampling(cropped_pcs_list, num_points=args.num_points)
                
                # 3. In the destination file, delete old data and add new data
                cam_group_out = f_out[traj_key]['obs']['sensor_data'][camera_name]
                del cam_group_out['rgb']
                del cam_group_out['depth']
                cam_group_out.create_dataset('point_cloud', data=final_pcs, compression='gzip')

    total_time = time.time() - t_start
    print(f"\nConversion complete. Total time: {total_time:.2f}s")

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Convert H5 dataset to DP3 point cloud format with trajectory structure.")
    parser.add_argument('--input_file', type=str, default='PegIns_EeDeltaPose_HandCam_HoleSize5_DemoVal200.rgb+depth.pd_ee_delta_pose.physx_cpu.compressed.h5')
    parser.add_argument('--output_file', type=str, default='peg_insertion_pointcloud_full.h5')
    parser.add_argument('--num_points', type=int, default=1024)
    parser.add_argument('--num_trajectories', type=int, default=-1, help='Number of trajectories to process. Set to -1 for all.')
    parser.add_argument('--crop_range', nargs=6, type=float, default=[-1.0, 1.0, -1.0, 1.0, -0.1, 1.0], help='Final workspace crop range.')
    args = parser.parse_args()

    demo_config_path = args.input_file.replace('.compressed', '').replace('.h5', '.yaml')
    with open(demo_config_path, 'r') as f:
        demo_config = OmegaConf.load(f)
    demo_config.sparse_pointcloud = dict(num_points=args.num_points, crop_range=args.crop_range)
    output_config_path = args.output_file.replace('.h5', '.yaml')
    OmegaConf.save(demo_config, output_config_path)

    main(args)