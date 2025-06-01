import os
import h5py
import json
import argparse
import numpy as np

from odpc.utils.utils import get_data_shape


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--traj-path', type=str, required=True, help='Path to the trajectory file.')
    return parser.parse_args()

def clip_traj(traj, start):
    if isinstance(traj, h5py.Dataset):
        return traj[start:]
    elif isinstance(traj, h5py.Group):
        return {k: clip_traj(v, start) for k, v in traj.items()}
    else:
        raise ValueError(f"Unsupported type: {type(traj)}")

def save_h5(data, dst, name):
    if isinstance(data, np.ndarray):
        chunk_shape = (1,) + data.shape[1:]
        dst.create_dataset(
            name,
            data=data,
            compression="gzip",
            compression_opts=5,
            # chunks=chunk_shape,
        )
    elif isinstance(data, dict):
        group = dst.create_group(name)
        for k, v in data.items():
            save_h5(v, group, k)
    else:
        raise ValueError(f"Unsupported type: {type(data)}")
    
def main(
        traj_path: str,
):
    with open(traj_path.replace('.h5', '.json'), 'r') as f:
        traj_info = json.load(f)
        
        episode_infos = {}
        for episode_info in traj_info['episodes']:
            episode_infos[episode_info['episode_id']] = episode_info

        if os.path.exists(traj_path.replace('.h5', f'_clip.h5')):
            os.remove(traj_path.replace('.h5', f'_clip.h5'))

        src_file = h5py.File(traj_path, 'r')
        dst_file = h5py.File(traj_path.replace('.h5', f'_clip.h5'), 'w')

        traj_keys = list(src_file.keys())
        for traj_key in traj_keys:
            traj = src_file[traj_key]
            episode_id = int(traj_key.split('_')[-1])
            start = episode_infos[episode_id]['start_step']
            traj_clip = clip_traj(traj, start)
            save_h5(traj_clip, dst_file, traj_key)
            episode_infos[episode_id]['elapsed_steps'] -= start
            episode_infos[episode_id]['start_step'] = 0
        
        with open(traj_path.replace('.h5', f'_clip.json'), 'w') as f:
            json.dump(traj_info, f, indent=2)

        src_file.close()
        dst_file.close()
    

if __name__ == "__main__":
    args = get_args()
    main(args.traj_path)