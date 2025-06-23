"""
数据工具函数的核心功能测试
"""

import pytest
import numpy as np
from unittest.mock import Mock

from odpc.data.utils import (
    compress_rgb_sequence_to_jpg_bytes,
    decode_jpeg_sequence,
    is_compressed_rgb_dataset,
    get_compression_info
)


class TestCompressionCore:
    """测试压缩核心功能"""
    
    @pytest.mark.unit
    def test_compress_rgb_sequence_to_jpg_bytes(self):
        """测试RGB序列压缩功能"""
        # 创建测试数据
        rgb_data = np.random.randint(0, 256, (3, 64, 64, 3), dtype=np.uint8)
        
        # 测试压缩
        compressed_bytes = compress_rgb_sequence_to_jpg_bytes(rgb_data, 85)
        
        # 验证结果
        assert len(compressed_bytes) == 3
        assert all(isinstance(frame_bytes, bytes) for frame_bytes in compressed_bytes)
        assert all(len(frame_bytes) > 0 for frame_bytes in compressed_bytes)
        
        # 验证压缩后的数据确实比原始数据小
        original_size = rgb_data.nbytes
        compressed_size = sum(len(frame) for frame in compressed_bytes)
        assert compressed_size < original_size
    
    @pytest.mark.unit
    def test_decode_jpeg_sequence(self):
        """测试JPEG序列解码功能"""
        # 创建测试数据并压缩
        rgb_data = np.random.randint(0, 256, (2, 32, 32, 3), dtype=np.uint8)
        compressed_bytes = compress_rgb_sequence_to_jpg_bytes(rgb_data, 85)
        
        # 测试解码
        decoded_data = decode_jpeg_sequence(compressed_bytes)
        
        # 验证结果
        assert decoded_data.shape == rgb_data.shape
        assert decoded_data.dtype == np.uint8
        # 验证数据在合理范围内
        assert decoded_data.min() >= 0
        assert decoded_data.max() <= 255


class TestDatasetDetection:
    """测试数据集检测功能"""
    
    @pytest.mark.unit
    def test_is_compressed_rgb_dataset(self):
        """测试压缩数据集检测"""
        # 创建模拟的压缩数据集
        mock_dataset = Mock()
        mock_dataset.attrs = {
            'compression_type': 'jpeg',
            'original_shape': (10, 64, 64, 3),
            'jpeg_quality': 85
        }
        
        assert is_compressed_rgb_dataset(mock_dataset) is True
        
        # 测试非压缩数据
        mock_dataset.attrs = {}
        assert is_compressed_rgb_dataset(mock_dataset) is False
    
    @pytest.mark.unit
    def test_get_compression_info(self):
        """测试获取压缩信息"""
        mock_dataset = Mock()
        mock_dataset.attrs = {
            'compression_type': 'jpeg',
            'original_shape': (10, 64, 64, 3),
            'original_dtype': 'uint8',
            'jpeg_quality': 85
        }
        
        info = get_compression_info(mock_dataset)
        
        assert info['compression_type'] == 'jpeg'
        assert info['original_shape'] == (10, 64, 64, 3)
        assert info['jpeg_quality'] == 85


class TestIntegration:
    """集成测试 - 核心工作流程"""
    
    @pytest.mark.integration
    def test_compress_decompress_roundtrip(self):
        """测试压缩-解压缩的完整流程（使用平滑渐变图片）"""
        # 创建平滑渐变测试数据
        original_data = np.zeros((5, 64, 64, 3), dtype=np.uint8)
        for i in range(5):
            for c in range(3):
                # 生成从0到255的线性渐变
                original_data[i, :, :, c] = np.linspace(0, 255, 64, dtype=np.uint8)[None, :]
        
        # 压缩
        compressed_bytes = compress_rgb_sequence_to_jpg_bytes(original_data, 85)
        
        # 解压缩
        decompressed_data = decode_jpeg_sequence(compressed_bytes)
        
        # 验证形状一致
        assert decompressed_data.shape == original_data.shape
        assert decompressed_data.dtype == np.uint8
        # 验证MAE - 这是集成测试的核心验证
        mae = np.mean(np.abs(original_data.astype(float) - decompressed_data.astype(float)))
        print(f"MAE for smooth image: {mae}")
        assert mae < 10  # 平滑图片下，JPEG误差应很小 