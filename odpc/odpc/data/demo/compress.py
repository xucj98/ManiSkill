import os
import h5py
from tqdm import tqdm
import argparse
import numpy as np
import multiprocessing as mp

from typing import Union, List

from odpc.data.utils import compress_rgb_sequence_to_jpg_bytes

# JPG压缩质量
DEFAULT_JPG_QUALITY = 85

def compress_recursive(
        src_obj: Union[h5py.Group, h5py.Dataset], 
        dst_obj: Union[h5py.File, h5py.Group, h5py.Dataset],
        name: str,
        jpg_quality: int = DEFAULT_JPG_QUALITY,
        src_path: str='',
) -> None:
    if isinstance(src_obj, h5py.Group):
        dst_group = dst_obj.create_group(name)
        for attr_name, attr_val in src_obj.attrs.items():
            dst_group.attrs[attr_name] = attr_val

        for _name, _src_obj in src_obj.items():
            compress_recursive(_src_obj, dst_group, _name, jpg_quality, src_path + "." + _name)
    elif isinstance(src_obj, h5py.Dataset):
        p = src_path.split(".")
        # traj_{id}.obs.sensor_data.{camera_name}.rgb
        if len(p) == 5 and p[2] == "sensor_data" and p[4] == "rgb":
            rgb_data_np = src_obj[:] # 加载原始RGB数据
            jpg_byte_list = compress_rgb_sequence_to_jpg_bytes(rgb_data_np, jpg_quality)
            num_frames = len(jpg_byte_list)
            dst_dataset = dst_obj.create_dataset(name, shape=(num_frames,), dtype=h5py.special_dtype(vlen=np.uint8), chunks=(1,))
            for i in range(num_frames):
                dst_dataset[i] = np.array(list(jpg_byte_list[i]), dtype=np.uint8)
            dst_dataset.attrs['original_shape'] = rgb_data_np.shape
            dst_dataset.attrs['original_dtype'] = str(rgb_data_np.dtype)
            dst_dataset.attrs['compression_type'] = 'jpeg'
            dst_dataset.attrs['jpeg_quality'] = jpg_quality 
        else:
            dst_obj.create_dataset(
                name, 
                data=src_obj[()], 
                shape=src_obj.shape, 
                dtype=src_obj.dtype,
                chunks=src_obj.chunks, 
                compression=src_obj.compression,
                compression_opts=src_obj.compression_opts
            )

def process_trajectories(
    proc_id: int,
    traj_keys: List[str],
    src_path: str,
    output_path: str,
    jpg_quality: int
) -> str:
    """
    Worker function for each process to compress a subset of trajectories.
    """
    partial_output_path = output_path.replace('.h5', f'.part.{proc_id}.h5')
    
    with h5py.File(src_path, 'r') as src_file, h5py.File(partial_output_path, 'w') as dst_file:
        pbar = tqdm(traj_keys, desc=f"Proc {proc_id}", position=proc_id, leave=False)
        for traj_key in pbar:
            compress_recursive(src_file[traj_key], dst_file, traj_key, jpg_quality, src_path=traj_key)
    
    return partial_output_path

def merge_h5_files(output_path: str, partial_h5_paths: List[str]):
    """
    Merges multiple HDF5 files into a single file.
    """
    with h5py.File(output_path, 'w') as merged_f:
        for partial_path in tqdm(partial_h5_paths, desc="Merging files"):
            with h5py.File(partial_path, 'r') as partial_f:
                for key in partial_f.keys():
                    partial_f.copy(key, merged_f)

def get_args():
    parser = argparse.ArgumentParser(description="Compress RGB data in HDF5 trajectory files using JPG.")
    parser.add_argument('--traj-path', type=str, required=True, 
                        help='Path to the source trajectory HDF5 file.')
    parser.add_argument('--jpg-quality', type=int, default=DEFAULT_JPG_QUALITY, 
                        help=f'JPEG compression quality (0-100). Default: {DEFAULT_JPG_QUALITY}')
    parser.add_argument('--num-procs', type=int, default=1, 
                        help='Number of parallel processes to use.')
    return parser.parse_args()

def main(args):
    traj_path = args.traj_path
    output_path = args.traj_path.replace('.h5', '.compressed.h5')
    jpg_quality = args.jpg_quality
    num_procs = args.num_procs

    print(f"Source HDF5: {traj_path}")
    print(f"Output HDF5: {output_path}")
    print(f"JPEG Quality: {jpg_quality}")
    print(f"Num Procs: {num_procs}")

    with h5py.File(traj_path, 'r') as src_file:
        traj_keys = list(src_file.keys())
    
    if num_procs > 1 and len(traj_keys) > 0:
        if len(traj_keys) < num_procs:
            print(f"Warning: Number of trajectories ({len(traj_keys)}) is less than number of processes ({num_procs}). Reducing num_procs to {len(traj_keys)}.")
            num_procs = len(traj_keys)

        key_chunks = np.array_split(traj_keys, num_procs)
        
        proc_args = []
        for i in range(num_procs):
            if len(key_chunks[i]) > 0:
                proc_args.append((i, key_chunks[i], traj_path, output_path, jpg_quality))
        
        with mp.Pool(processes=num_procs) as pool:
            partial_h5_paths = pool.starmap(process_trajectories, proc_args)
        
        print(f"Merging {len(partial_h5_paths)} partial files into {output_path}...")
        merge_h5_files(output_path, partial_h5_paths)
        
        for h5_path in partial_h5_paths:
            # print(f"Removing partial file: {h5_path}")
            os.remove(h5_path)

    else:
        with h5py.File(traj_path, 'r') as src_file, h5py.File(output_path, 'w') as dst_file:
            for traj_key in tqdm(traj_keys, desc="Processing trajectories"):
                compress_recursive(src_file[traj_key], dst_file, traj_key, jpg_quality, src_path=traj_key)

    print(f"Processing finished. \n"
          f"Before Compress: size={os.path.getsize(traj_path) / (1024*1024):.2f} MB \n"
          f"After Compress: size={os.path.getsize(output_path) / (1024*1024):.2f} MB \n")
    
    return output_path

if __name__ == '__main__':
    args = get_args()
    main(args)