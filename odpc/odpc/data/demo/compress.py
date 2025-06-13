import os
import h5py
from tqdm import tqdm
import argparse
import numpy as np
from PIL import Image
import io

from typing import Union, List

# JPG压缩质量
DEFAULT_JPG_QUALITY = 85

def compress_rgb_sequence_to_jpg_bytes(rgb_sequence_np: np.ndarray, jpg_quality: int) -> list:
    """
    将 (t, h, w, 3) uint8 NumPy 数组的RGB序列压缩为JPG字节列表。
    """
    jpg_byte_list = []
    for i in range(rgb_sequence_np.shape[0]):
        frame_np = rgb_sequence_np[i]
        pil_image = Image.fromarray(frame_np)
        with io.BytesIO() as output_buffer:
            pil_image.save(output_buffer, format="JPEG", quality=jpg_quality)
            jpg_bytes = output_buffer.getvalue()
        jpg_byte_list.append(jpg_bytes)
    return jpg_byte_list

def _main(
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
            _main(_src_obj, dst_group, _name, jpg_quality, src_path + "." + _name)
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

def get_args():
    parser = argparse.ArgumentParser(description="Compress RGB data in HDF5 trajectory files using JPG.")
    parser.add_argument('--traj-path', type=str, required=True, 
                        help='Path to the source trajectory HDF5 file.')
    parser.add_argument('--jpg-quality', type=int, default=DEFAULT_JPG_QUALITY, 
                        help=f'JPEG compression quality (0-100). Default: {DEFAULT_JPG_QUALITY}')
    return parser.parse_args()

def main(args):
    traj_path = args.traj_path
    output_path = args.traj_path.replace('.h5', '.compressed.h5')
    jpg_quality = args.jpg_quality

    print(f"Source HDF5: {traj_path}")
    print(f"Output HDF5: {output_path}")
    print(f"JPEG Quality: {jpg_quality}")

    src_file = h5py.File(traj_path, 'r')
    dst_file = h5py.File(output_path, 'w')
    traj_keys = list(src_file.keys())
       
    for traj_key in tqdm(traj_keys, desc="Processing trajectories"):
        _main(src_file[traj_key], dst_file, traj_key, jpg_quality, src_path=traj_key)
        
    src_file.close()
    dst_file.close()

    print(f"Processing finished. \n"
          f"Before Compress: size={os.path.getsize(traj_path) / (1024*1024):.2f} MB \n"
          f"After Compress: size={os.path.getsize(output_path) / (1024*1024):.2f} MB \n")


if __name__ == '__main__':
    args = get_args()
    main(args)