import h5py
import numpy as np
from typing import Optional, Union, Dict, Any, List


def print_h5_structure(file_path: str, indent: int = 0, max_depth: Optional[int] = None) -> None:
    """
    打印h5文件的结构，包括所有组和数据集的信息。
    
    参数:
        file_path: h5文件的路径
        indent: 缩进级别，用于递归打印
        max_depth: 最大递归深度，None表示无限制
    """
    def _print_structure(name, obj, current_depth=0):
        # 检查是否达到最大深度
        if max_depth is not None and current_depth > max_depth:
            return
        
        # 计算缩进
        prefix = "  " * indent + "  " * current_depth
        
        if isinstance(obj, h5py.Dataset):
            # 打印数据集信息
            shape = obj.shape
            dtype = obj.dtype
            print(f"{prefix}数据集: {name}")
            print(f"{prefix}  形状: {shape}")
            print(f"{prefix}  数据类型: {dtype}")
            
            # 如果数据集很小，打印其内容
            if np.prod(shape) < 10 and obj.dtype.kind in 'iuf':
                try:
                    data = obj[()]
                    print(f"{prefix}  内容: {data}")
                except Exception as e:
                    print(f"{prefix}  无法读取内容: {e}")
        elif isinstance(obj, h5py.Group):
            # 打印组信息
            print(f"{prefix}组: {name}")
            # 递归遍历组内的所有项目
            for key in obj.keys():
                _print_structure(f"{name}/{key}", obj[key], current_depth + 1)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件: {file_path}")
            # 遍历文件中的所有顶级项目
            for key in f.keys():
                _print_structure(key, f[key])
    except Exception as e:
        print(f"无法打开文件 {file_path}: {e}")


def analyze_h5_dataset(file_path: str, dataset_path: Optional[str] = None) -> Dict[str, Any]:
    """
    分析h5文件中的特定数据集，返回详细信息。
    
    参数:
        file_path: h5文件的路径
        dataset_path: 数据集的路径，如果为None则分析所有数据集
        
    返回:
        包含数据集信息的字典
    """
    result = {}
    
    try:
        with h5py.File(file_path, 'r') as f:
            if dataset_path is not None:
                # 分析特定数据集
                if dataset_path in f:
                    ds = f[dataset_path]
                    result[dataset_path] = {
                        'shape': ds.shape,
                        'dtype': ds.dtype,
                        'size': ds.size,
                        'chunks': ds.chunks if ds.chunks else None,
                        'compression': ds.compression,
                        'compression_opts': ds.compression_opts,
                        'shuffle': ds.shuffle,
                        'fletcher32': ds.fletcher32,
                        'maxshape': ds.maxshape,
                        'scaleoffset': ds.scaleoffset,
                    }
                else:
                    print(f"数据集 {dataset_path} 不存在")
            else:
                # 分析所有数据集
                def _analyze_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        result[name] = {
                            'shape': obj.shape,
                            'dtype': obj.dtype,
                            'size': obj.size,
                        }
                
                f.visititems(_analyze_datasets)
    
    except Exception as e:
        print(f"分析文件时出错: {e}")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="分析H5文件结构")
    parser.add_argument("file_path", type=str, help="H5文件路径")
    parser.add_argument("--dataset", type=str, help="要分析的特定数据集路径", default=None)
    parser.add_argument("--max-depth", type=int, help="最大递归深度", default=None)
    
    args = parser.parse_args()
    
    print_h5_structure(args.file_path, max_depth=args.max_depth)
    
    if args.dataset:
        print("\n数据集详细信息:")
        dataset_info = analyze_h5_dataset(args.file_path, args.dataset)
        for path, info in dataset_info.items():
            print(f"\n数据集: {path}")
            for key, value in info.items():
                print(f"  {key}: {value}")
