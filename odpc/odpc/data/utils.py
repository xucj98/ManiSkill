"""
数据工具函数模块
包含压缩、解压缩等数据处理相关的工具函数
"""

import numpy as np
import cv2
from PIL import Image
import io
from typing import List, Union


def compress_rgb_sequence_to_jpg_bytes(rgb_sequence_np: np.ndarray, jpg_quality: int) -> List[bytes]:
    """
    将 (t, h, w, 3) uint8 NumPy 数组的RGB序列压缩为JPG字节列表。
    
    Args:
        rgb_sequence_np: RGB图像序列，形状为 (num_frames, height, width, 3)
        jpg_quality: JPEG压缩质量 (0-100)
    
    Returns:
        List[bytes]: 压缩后的JPEG字节列表
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


def decode_jpeg_sequence(compressed_data: Union[List[bytes], np.ndarray]) -> np.ndarray:
    """
    解码JPEG压缩的RGB序列数据
    
    Args:
        compressed_data: 压缩的JPEG字节数据，可以是字节列表或numpy数组
    
    Returns:
        numpy.ndarray: 解码后的RGB图像序列，形状为 (num_frames, height, width, 3)
    """
    decoded_frames = []
    
    # 确保compressed_data是可迭代的
    if isinstance(compressed_data, np.ndarray):
        data_iter = compressed_data
    else:
        data_iter = compressed_data
    
    for frame_bytes in data_iter:
        # 检查数据类型并正确处理
        if isinstance(frame_bytes, bytes):
            frame_array = np.frombuffer(frame_bytes, dtype=np.uint8)
        elif isinstance(frame_bytes, np.ndarray):
            frame_array = frame_bytes
        else:
            # 如果是其他类型，尝试转换为numpy数组
            frame_array = np.array(frame_bytes, dtype=np.uint8)
        
        # 使用OpenCV解码JPEG
        decoded_frame = cv2.imdecode(frame_array, cv2.IMREAD_COLOR)
        if decoded_frame is None:
            raise ValueError(f"Failed to decode JPEG frame: {frame_bytes[:10]}...")
        
        # 转换为RGB格式
        decoded_frame = cv2.cvtColor(decoded_frame, cv2.COLOR_BGR2RGB)
        decoded_frames.append(decoded_frame)
    
    return np.array(decoded_frames)


def is_compressed_rgb_dataset(dataset) -> bool:
    """
    检查数据集是否为JPEG压缩的RGB数据
    
    Args:
        dataset: h5py数据集对象
    
    Returns:
        bool: 是否为压缩的RGB数据
    """
    return (hasattr(dataset, 'attrs') and 
            'compression_type' in dataset.attrs and 
            dataset.attrs['compression_type'] == 'jpeg')
