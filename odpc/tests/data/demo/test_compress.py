"""
H5文件压缩功能的核心测试
"""

import pytest
import numpy as np
import h5py
import tempfile
import os
import argparse
from unittest.mock import patch

from odpc.data.odpc_dataset import ODPCDataset


class TestH5CompressionCore:
    """测试H5压缩核心功能"""
    
    @pytest.fixture
    def temp_h5_file(self):
        """创建临时H5文件"""
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        # 创建测试H5文件
        test_rgb = np.random.randint(0, 256, (3, 64, 64, 3), dtype=np.uint8)
        
        with h5py.File(temp_path, 'w') as f:
            # 创建轨迹结构
            traj_group = f.create_group('traj_0')
            obs_group = traj_group.create_group('obs')
            sensor_group = obs_group.create_group('sensor_data')
            camera_group = sensor_group.create_group('camera_0')
            
            # 存储原始RGB数据
            camera_group.create_dataset('rgb', data=test_rgb)
            
            # 存储动作数据
            actions = np.random.randn(3, 7)
            traj_group.create_dataset('actions', data=actions)
            
            # 添加extra数据
            extra_group = obs_group.create_group('extra')
            extra_group.create_dataset('example_data', data=np.random.randn(3, 10))
        
        yield temp_path
        
        # 清理
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    @pytest.mark.integration
    def test_h5_compression_workflow(self, temp_h5_file):
        """测试H5压缩工作流程"""
        from odpc.data.demo.compress import main as compress_main
        
        # 获取原始文件大小
        original_size = os.path.getsize(temp_h5_file)
        
        # 执行压缩
        compress_args = argparse.Namespace(
            traj_path=temp_h5_file,
            jpg_quality=85
        )
        
        with patch('builtins.print'):  # 抑制打印输出
            compressed_path = compress_main(compress_args)
        
        # 验证函数返回了正确的路径
        expected_path = temp_h5_file.replace('.h5', '.compressed.h5')
        assert compressed_path == expected_path, f"Expected {expected_path}, got {compressed_path}"
        
        # 检查压缩后的文件
        assert os.path.exists(compressed_path)
        
        compressed_size = os.path.getsize(compressed_path)
        
        # 验证压缩效果
        assert compressed_size < original_size
        
        # 清理压缩文件
        os.unlink(compressed_path)
    
    @pytest.mark.integration
    def test_odpc_dataset_with_compressed_data(self, temp_h5_file):
        """测试ODPCDataset读取压缩数据"""
        from odpc.data.demo.compress import main as compress_main
        
        # 执行压缩
        compress_args = argparse.Namespace(
            traj_path=temp_h5_file,
            jpg_quality=85
        )
        
        with patch('builtins.print'):
            compressed_path = compress_main(compress_args)
        
        # 验证函数返回了正确的路径
        expected_path = temp_h5_file.replace('.h5', '.compressed.h5')
        assert compressed_path == expected_path, f"Expected {expected_path}, got {compressed_path}"
        
        # 使用ODPCDataset读取压缩数据
        dataset = ODPCDataset(
            data_path=compressed_path,
            obs_horizon=2,
            pred_horizon=3,
            num_traj=1
        )
        
        # 验证数据集加载成功
        assert len(dataset) > 0
        
        # 获取一个样本
        sample = dataset[0]
        
        # 验证数据结构
        assert 'observations' in sample
        assert 'actions' in sample
        assert 'sensor_data' in sample['observations']
        assert 'camera_0' in sample['observations']['sensor_data']
        assert 'rgb' in sample['observations']['sensor_data']['camera_0']
        
        # 验证RGB数据形状
        rgb_data = sample['observations']['sensor_data']['camera_0']['rgb']
        assert rgb_data.shape[0] == 2  # obs_horizon
        assert rgb_data.shape[1] == 3  # channels
        assert rgb_data.shape[2] == 64  # height
        assert rgb_data.shape[3] == 64  # width
        
        # 清理
        os.unlink(compressed_path) 