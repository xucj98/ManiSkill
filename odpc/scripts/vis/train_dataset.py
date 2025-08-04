import argparse
import numpy as np
import h5py
import cv2
import os
from tqdm import tqdm
from omegaconf import OmegaConf

from odpc.utils.utils import load_config_with_defaults, instantiate_from_config
from odpc.utils.visualize import visualize_dataset_samples

def get_args():
    parser = argparse.ArgumentParser(description="Visualize ODPC dataset and save as MP4 video")
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--output-dir', type=str, default='vis_output', help='Output directory for videos')
    parser.add_argument('--fps', type=int, default=30, help='FPS for output video')
    parser.add_argument('--add-camera-labels', action='store_true', help='Add camera name labels to concatenated images')
    return parser.parse_args()

def main():
    args = get_args()

    config = load_config_with_defaults(args.config)
    dataset = instantiate_from_config(config.train_dataset)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 生成包含所有样本的视频
    base_name = config.exp_name
    output_filename = f"{base_name}.mp4"
    output_path = os.path.join(args.output_dir, output_filename)
    
    visualize_dataset_samples(dataset, output_path, args.fps, args.add_camera_labels)
    
    print(f"\nVisualization completed. Output directory: {args.output_dir}")

if __name__ == "__main__":
    main() 