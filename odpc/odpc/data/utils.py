"""
数据工具函数模块
包含压缩、解压缩等数据处理相关的工具函数
"""

import io
import cv2
import numpy as np
from PIL import Image
import open3d as o3d

import torch
import pytorch3d.ops as torch3d_ops

from typing import List, Union, Optional, Tuple


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


def create_point_cloud(
        rgb_img: np.ndarray, 
        depth_img: np.ndarray, 
        intrinsic: np.ndarray, 
        extrinsic_3x4: Optional[np.ndarray] = None, 
        extrinsic_4x4: Optional[np.ndarray] = None, 
        depth_scale: float = 1000.0,
        depth_trunc: float = 3.0,
        crop_range: Optional[Tuple[float, float, float, float, float, float]] = None,
)-> np.ndarray:
    """
    Create a point cloud from a RGB image and a depth image.
    """
    depth_img[depth_img <= 0] = 0
    height, width, _ = rgb_img.shape
    o3d_rgb = o3d.geometry.Image(rgb_img)
    o3d_depth = o3d.geometry.Image(depth_img)
    o3d_rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d_rgb, o3d_depth, depth_scale=depth_scale, depth_trunc=depth_trunc, convert_rgb_to_intensity=False
    )
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width, height, intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    )
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(o3d_rgbd, o3d_intrinsic)
    
    if extrinsic_3x4 is not None:
        extrinsic_4x4 = np.eye(4)
        extrinsic_4x4[:3, :] = extrinsic_3x4
    if extrinsic_4x4 is not None:   
        transform_matrix = np.linalg.inv(extrinsic_4x4)
        pcd.transform(transform_matrix)
    
    points = np.asarray(pcd.points)
    colors = (np.asarray(pcd.colors) * 255).astype(np.uint8)
    pc = np.hstack((points, colors))

    if crop_range is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = crop_range
        pc = pc[(pc[:, 0] >= x_min) & (pc[:, 0] <= x_max) & 
                (pc[:, 1] >= y_min) & (pc[:, 1] <= y_max) & 
                (pc[:, 2] >= z_min) & (pc[:, 2] <= z_max)]

    return pc


def batch_farthest_point_sampling(
        points_list: List[np.ndarray], 
        num_points: int = 1024,
        return_numpy: bool = True,
)-> Union[np.ndarray, torch.Tensor]:
    """
    Batch farthest point sampling.
    """
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
    sampled_points = torch.cat([sampled_xyz, sampled_rgb], dim=-1)
    if return_numpy:
        sampled_points = sampled_points.cpu().numpy()
    return sampled_points