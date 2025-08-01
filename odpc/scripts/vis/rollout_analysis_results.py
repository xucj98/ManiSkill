import argparse
import json
import h5py
import numpy as np
from odpc.utils.visualize import visualize_video_with_metric


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-path", type=str, required=True, help="Path to the rollout.h5 file.")
    parser.add_argument("--result-path", type=str, required=True, help="Path to the rollout analysis results .json file.")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    analysis_results = json.load(open(args.result_path))
    
    with h5py.File(args.rollout_path, "r") as f:
        for traj_key, result in analysis_results.items():
            traj_data = f[traj_key]

            frame_idx = result["frame_idx"]
            is_key_frame = result["is_key_frame"]
            metric_values = result["metric_values"]

            video = traj_data["obs/sensor_data/base_camera/rgb"][:]
            video = video[frame_idx, 0]
            metric = np.array(metric_values)

            visualize_video_with_metric(
                video,
                metric,
            )
