import os
import argparse
import json
import yaml
import h5py
import numpy as np
from odpc.utils.visualize import visualize_video_with_metric


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-path", type=str, required=True, help="Path to the rollout analysis results .json file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    config_path = os.path.join(os.path.dirname(args.result_path), "config.yaml")
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)
    rollout_path = config["rollout_path"]

    analysis_results = json.load(open(args.result_path))
    with h5py.File(rollout_path, "r") as f:
        print("Total trajectories: ", len(analysis_results))
        for traj_key, result in analysis_results.items():
            print(f"Visualizing trajectory: {traj_key}")
            traj_data = f[traj_key]

            frame_idx = result["frame_idx"]
            is_key_frame = np.array(result["is_key_frame"])
            metric_values = np.array(result["metric_values"])

            # Load the full video
            video = traj_data["obs/sensor_data/base_camera/rgb"][:]
            
            # Select the frames that were actually analyzed
            video = video[frame_idx]
            if video.ndim == 5:
                video = video.squeeze(1)

            # Ensure video and metrics have the same length
            assert len(video) == len(metric_values), "Mismatch between video and metric length after indexing!"
            assert len(video) == len(is_key_frame), "Mismatch between video and key_frame length after indexing!"

            visualize_video_with_metric(
                video,
                metric_values,
                is_key_frame=is_key_frame,
            )