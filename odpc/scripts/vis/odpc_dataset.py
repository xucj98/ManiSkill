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
    parser.add_argument('--obs-horizon', type=int, default=2, help='Observation horizon')
    parser.add_argument('--pred-horizon', type=int, default=16, help='Prediction horizon')
    parser.add_argument('--num-traj', type=int, default=5, help='Number of trajectories to visualize')
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

def visualize_all_samples(dataset, output_path, fps=10):
    """依次展示数据集中的所有样本"""
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
            first_camera = camera_names[0]
        
        rgb_data = current_sensor_data[first_camera]['rgb']
        
        # 支持 (N, C, H, W) 或 (N, H, W, C)
        if len(rgb_data.shape) == 4:
            if rgb_data.shape[1] == 3:  # (N, C, H, W)
                num_frames, channels, height, width = rgb_data.shape
                rgb_data = np.transpose(rgb_data, (0, 2, 3, 1))  # (N, H, W, C)
            else:  # (N, H, W, C)
                num_frames, height, width, channels = rgb_data.shape
        else:
            print(f"Unexpected RGB data shape: {rgb_data.shape}")
            continue
        
        # 收集所有帧
        for frame_idx in range(num_frames):
            frame = rgb_data[frame_idx]
            if frame.dtype != np.uint8:
                frame = (frame * 255).astype(np.uint8)
            all_frames.append(frame)
    
    if not all_frames:
        print("No frames collected")
        return None
    
    try:
        video_writer = create_video_writer(output_path, fps, (width, height))
        
        print(f"Creating video with {len(all_frames)} frames, size: {width}x{height}")
        print(f"Duration: {len(all_frames)/fps:.2f} seconds")
        
        for frame in tqdm(all_frames, desc="Writing frames"):
            if frame.shape[-1] == 3:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame
            video_writer.write(frame_bgr)
        
        video_writer.release()
        print(f"Video saved to: {output_path}")
        return (width, height)
        
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
    
    visualize_all_samples(dataset, output_path, args.fps)
    
    print(f"\nVisualization completed. Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 