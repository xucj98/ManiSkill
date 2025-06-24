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
)


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
        
        # 验证压缩率 - 同时测试压缩效果
        original_size = original_data.nbytes
        compressed_size = sum(len(frame) for frame in compressed_bytes)
        compression_ratio = original_size / compressed_size
        
        print(f"原始大小: {original_size / 1024:.2f} KB")
        print(f"压缩后大小: {compressed_size / 1024:.2f} KB")
        print(f"压缩比: {compression_ratio:.2f}x")
        
        # 验证压缩确实有效
        assert compressed_size < original_size
        assert compression_ratio > 1.0
        # 对于平滑图片，在质量85下应该达到至少5倍压缩比
        assert compression_ratio >= 5.0, f"平滑图片在质量85下应该达到至少5倍压缩比，实际为 {compression_ratio:.2f}x" 