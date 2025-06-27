import argparse
import numpy as np
import h5py
import cv2
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from odpc.data.odpc_dataset import ODPCDataset

def get_args():
    parser = argparse.ArgumentParser(description="Visualize ODPC dataset and save as MP4 video")
    parser.add_argument('--data-path', type=str, required=True, help='Path to the HDF5 dataset file')
    parser.add_argument('--output-dir', type=str, default='vis_output', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--obs-horizon', type=int, default=1, help='Observation horizon')
    parser.add_argument('--pred-horizon', type=int, default=16, help='Prediction horizon')
    parser.add_argument('--num-traj', type=int, default=5, help='Number of trajectories to visualize')
    parser.add_argument('--add-camera-labels', action='store_true', help='Add camera name labels to concatenated images')
    return parser.parse_args()

def create_video_writer(output_path, fps, frame_size):
    """创建视频写入器"""
    # 尝试不同的编码器
    codecs = ['mp4v', 'avc1', 'XVID', 'MJPG']
    video_writer = None
    
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            video_writer = cv2.VideoWriter(output_path, fourcc, fps, frame_size)
            if video_writer.isOpened():
                print(f"Using codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to use codec {codec}: {e}")
            continue
    
    if video_writer is None or not video_writer.isOpened():
        raise RuntimeError(f"Could not create video writer for {output_path}")
    
    return video_writer

def visualize_all_samples(dataset, output_path, fps=10, add_camera_labels=False):
    """依次展示数据集中的所有样本，同时显示所有相机的图像"""
    camera_names = None
    all_frames = []
    
    print(f"Processing all {len(dataset)} samples...")
    
    # 收集所有样本的帧
    for sample_idx in tqdm(range(len(dataset)), desc="Collecting frames"):
        traj_data = dataset[sample_idx]
        current_sensor_data = traj_data['observations']['sensor_data']
        
        if camera_names is None:
            camera_names = list(current_sensor_data.keys())
            if not camera_names:
                print("No camera data found in trajectory")
                return None
            print(f"Found cameras: {camera_names}")
        
        # 收集当前样本所有相机的帧
        sample_frames = []
        for camera_name in camera_names:
            if camera_name not in current_sensor_data:
                print(f"Warning: Camera {camera_name} not found in sample {sample_idx}")
                continue
                
            rgb_data = current_sensor_data[camera_name]['rgb']
            
            # 支持 (N, C, H, W) 或 (N, H, W, C)
            if len(rgb_data.shape) == 4:
                if rgb_data.shape[1] == 3:  # (N, C, H, W)
                    num_frames, channels, height, width = rgb_data.shape
                    rgb_data = np.transpose(rgb_data, (0, 2, 3, 1))  # (N, H, W, C)
                else:  # (N, H, W, C)
                    num_frames, height, width, channels = rgb_data.shape
            else:
                print(f"Unexpected RGB data shape for camera {camera_name}: {rgb_data.shape}")
                continue
            
            # 收集当前相机所有帧
            camera_frames = []
            for frame_idx in range(num_frames):
                frame = rgb_data[frame_idx]
                if frame.dtype != np.uint8:
                    frame = (frame * 255).astype(np.uint8)
                camera_frames.append(frame)
            
            sample_frames.append(camera_frames)
        
        # 将当前样本的所有相机帧添加到总帧列表
        if sample_frames:
            all_frames.append(sample_frames)
    
    if not all_frames:
        print("No frames collected")
        return None
    
    # 获取第一个样本的帧数作为基准
    num_frames_per_sample = len(all_frames[0][0]) if all_frames and all_frames[0] else 0
    print(f"Each sample has {num_frames_per_sample} frames")
    
    try:
        # 计算拼接后的图像尺寸
        first_frame = all_frames[0][0][0]  # 第一个样本第一个相机的第一帧
        height, width = first_frame.shape[:2]
        total_width = width * len(camera_names)
        
        video_writer = create_video_writer(output_path, fps, (total_width, height))
        
        print(f"Creating video with {len(all_frames)} samples, {num_frames_per_sample} frames per sample")
        print(f"Image size: {total_width}x{height} (concatenated from {len(camera_names)} cameras)")
        print(f"Duration: {len(all_frames) * num_frames_per_sample / fps:.2f} seconds")
        
        # 为每个样本的每一帧创建拼接图像
        for sample_idx, sample_frames in enumerate(tqdm(all_frames, desc="Processing samples")):
            for frame_idx in range(num_frames_per_sample):
                # 拼接当前帧的所有相机图像
                concatenated_frames = []
                for camera_idx, camera_frames in enumerate(sample_frames):
                    if frame_idx < len(camera_frames):
                        frame = camera_frames[frame_idx].copy()
                        
                        # 添加相机标签
                        if add_camera_labels:
                            camera_name = camera_names[camera_idx]
                            # 在图像左上角添加标签
                            cv2.putText(frame, camera_name, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(frame, camera_name, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        
                        concatenated_frames.append(frame)
                    else:
                        # 如果某个相机帧数不足，用黑色填充
                        black_frame = np.zeros((height, width, 3), dtype=np.uint8)
                        if add_camera_labels:
                            camera_name = camera_names[camera_idx]
                            cv2.putText(black_frame, camera_name, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                            cv2.putText(black_frame, camera_name, (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1)
                        concatenated_frames.append(black_frame)
                
                # 在x轴上拼接所有相机图像
                if concatenated_frames:
                    concatenated_image = np.concatenate(concatenated_frames, axis=1)
                    
                    # 转换为BGR格式
                    if concatenated_image.shape[-1] == 3:
                        frame_bgr = cv2.cvtColor(concatenated_image, cv2.COLOR_RGB2BGR)
                    else:
                        frame_bgr = concatenated_image
                    
                    video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Video saved to: {output_path}")
        return (total_width, height)
        
    except Exception as e:
        print(f"Error creating video {output_path}: {e}")
        return None

def main():
    args = get_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
       
    print(f"Loading dataset from: {args.data_path}")
    dataset = ODPCDataset(
        data_path=args.data_path,
        obs_horizon=args.obs_horizon,
        pred_horizon=args.pred_horizon,
        num_traj=args.num_traj,
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # 生成包含所有样本的视频
    base_name = os.path.splitext(os.path.basename(args.data_path))[0]
    output_filename = f"{base_name}.mp4"
    output_path = os.path.join(args.output_dir, output_filename)
    
    visualize_all_samples(dataset, output_path, args.fps, args.add_camera_labels)
    
    print(f"\nVisualization completed. Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 