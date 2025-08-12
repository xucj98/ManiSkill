import h5py
import numpy as np
import open3d as o3d
import torch
import pytorch3d.ops as torch3d_ops
import cv2
from tqdm import tqdm
import argparse
import time

def decode_jpeg(jpeg_bytes):
    bgr_image = cv2.imdecode(np.frombuffer(jpeg_bytes, np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)

def create_point_cloud(rgb_img, depth_img, intrinsic, extrinsic_4x4, depth_scale=1000.0):
    depth_img[depth_img <= 0] = 0
    height, width, _ = rgb_img.shape
    o3d_rgb = o3d.geometry.Image(rgb_img)
    o3d_depth = o3d.geometry.Image(depth_img)
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=depth_scale, depth_trunc=3.0, convert_rgb_to_intensity=False
    )
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, o3d_intrinsic)
    transform_matrix = np.linalg.inv(extrinsic_4x4)
    pcd.transform(transform_matrix)
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    return np.hstack((points, colors))

def batch_farthest_point_sampling(points_list, num_points=1024):
    max_points = max(pc.shape[0] for pc in points_list if pc.shape[0] > 0)
    if max_points == 0:
        return np.zeros((len(points_list), num_points, 6), dtype=np.float32)

    padded_points = np.zeros((len(points_list), max_points, 6), dtype=np.float32)
    lengths = np.zeros(len(points_list), dtype=np.int64)
    for i, pc in enumerate(points_list):
        if pc.shape[0] > 0:
            padded_points[i, :pc.shape[0], :] = pc
            lengths[i] = pc.shape[0]

    points_torch = torch.from_numpy(padded_points).cuda()
    lengths_torch = torch.from_numpy(lengths).cuda()
    xyz_torch = points_torch[:, :, :3].float()
    rgb_torch = points_torch[:, :, 3:].float()
    sampled_indices = torch3d_ops.sample_farthest_points(xyz_torch, lengths=lengths_torch, K=num_points)[1]
    batch_indices = torch.arange(len(points_list), device='cuda').unsqueeze(1)
    sampled_xyz = xyz_torch[batch_indices, sampled_indices]
    sampled_rgb = rgb_torch[batch_indices, sampled_indices]
    sampled_points = torch.cat([sampled_xyz, sampled_rgb], dim=-1).cpu().numpy()
    return sampled_points

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
                    extrinsic_4x4 = np.eye(4); extrinsic_4x4[:3, :] = extrinsic_3x4
                    raw_pc = create_point_cloud(rgb_image, depth_image, intrinsic, extrinsic_4x4)
                    x_min, x_max = args.crop_range[0]; y_min, y_max = args.crop_range[1]; z_min, z_max = args.crop_range[2]
                    cropped_pc = raw_pc[(raw_pc[:, 0] >= x_min) & (raw_pc[:, 0] <= x_max) & (raw_pc[:, 1] >= y_min) & (raw_pc[:, 1] <= y_max) & (raw_pc[:, 2] >= z_min) & (raw_pc[:, 2] <= z_max)]
                    cropped_pcs_list.append(cropped_pc)
                
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
    args.crop_range = [[args.crop_range[0], args.crop_range[1]], [args.crop_range[2], args.crop_range[3]], [args.crop_range[4], args.crop_range[5]]]
    main(args)