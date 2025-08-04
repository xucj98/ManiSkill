import os
import h5py
import argparse
from tqdm import tqdm
from collections import defaultdict
from h5py import h5o
import numpy as np # 需要 numpy 来使用 np.sum

def format_bytes(byte_count):
    """将字节数格式化为可读的字符串 (KB, MB, GB)。"""
    if byte_count is None:
        return "N/A"
    power = 1024
    n = 0
    power_labels = {0: '', 1: 'K', 2: 'M', 3: 'G', 4: 'T'}
    while byte_count >= power and n < len(power_labels) - 1:
        byte_count /= power
        n += 1
    return f"{byte_count:.2f} {power_labels[n]}B"

def get_dataset_size(dset: h5py.Dataset) -> int:
    """
    根据数据集类型，准确计算其在磁盘上的存储大小。
    """
    # --- 关键修改在这里 ---
    # 检查数据集是否是可变长度类型 (VLEN)
    if h5py.check_vlen_dtype(dset.dtype):
        # 对于VLEN数据集，必须遍历所有元素并累加其大小。
        # get_storage_size() 对VLEN的行为不符合预期。
        # 使用 np.sum 和生成器表达式以提高效率和内存使用。
        if dset.shape[0] == 0:
            return 0
        return np.sum([item.nbytes for item in dset])
    else:
        # 对于所有其他普通数据集（固定大小元素），
        # 使用 get_storage_size() 可以正确处理GZIP等压缩。
        return dset.id.get_storage_size()

def find_datasets_recursively(
    group: h5py.Group, 
    size_summary: defaultdict, 
    processed_addresses: set, 
    base_path: str = ""
):
    """递归地查找组内的所有数据集并累加它们的大小，避免重复计算共享数据。"""
    for name, item in group.items():
        current_path = os.path.join(base_path, name)
        
        if isinstance(item, h5py.Dataset):
            info = h5o.get_info(item.id)
            addr = info.addr
            
            if addr in processed_addresses:
                continue
            
            processed_addresses.add(addr)
            
            size = get_dataset_size(item)
            size_summary[current_path] += size
        elif isinstance(item, h5py.Group):
            find_datasets_recursively(item, size_summary, processed_addresses, base_path=current_path)

def analyze_space_usage(h5_path: str, dry_run: bool = False):
    """分析HDF5文件内各部分的空间占用。"""
    if not os.path.exists(h5_path):
        print(f"错误: 文件不存在 '{h5_path}'")
        return

    print("=" * 80)
    print(f"分析文件: {h5_path}")
    total_file_size = os.path.getsize(h5_path)
    print(f"总文件大小: {format_bytes(total_file_size)}")
    print("-" * 80)

    size_summary = defaultdict(int)
    processed_addresses = set()
    
    with h5py.File(h5_path, 'r') as hf:
        traj_keys = [key for key in hf.keys() if key.startswith('traj_')]
        if not traj_keys:
            print("文件中未找到以 'traj_' 开头的轨迹。")
            find_datasets_recursively(hf, size_summary, processed_addresses)
        else:
            print(f"找到 {len(traj_keys)} 条轨迹，开始统计（将自动处理共享数据和压缩）...")
            if dry_run:
                traj_keys = traj_keys[:10]
            for traj_key in tqdm(traj_keys):
                traj_group = hf[traj_key]
                find_datasets_recursively(traj_group, size_summary, processed_addresses)

    if not size_summary:
        print("未在轨迹中找到任何数据集。")
        return

    sorted_summary = sorted(size_summary.items(), key=lambda item: item[1], reverse=True)
    total_summarized_size = sum(size_summary.values())

    print(f"\n各数据集在所有轨迹中的总空间占用 (从大到小):")
    print("-" * 80)
    print(f"{'数据集路径':<60} {'总大小':>15} {'百分比':>10}")
    print("-" * 80)

    for path, size in sorted_summary:
        percentage = (size / total_summarized_size) * 100 if total_summarized_size > 0 else 0
        print(f"{path:<60} {format_bytes(size):>15} {percentage:>9.2f}%")

    print("-" * 80)
    print(f"{'所有数据集总计':<60} {format_bytes(total_summarized_size):>15} {'100.00%':>10}")
    
    overhead = total_file_size - total_summarized_size
    if overhead < 0:
        print(f"{'HDF5文件元数据等开销 (计算异常)':<60} {format_bytes(overhead):>15}")
    else:
        print(f"{'HDF5文件元数据等开销':<60} {format_bytes(overhead):>15}")
    print("=" * 80 + "\n")

def get_args():
    parser = argparse.ArgumentParser(description="分析 HDF5 文件中每个数据集key在所有轨迹中的总空间占用。")
    parser.add_argument('h5_paths', nargs='+', help='一个或多个HDF5文件的路径，用于分析。')
    parser.add_argument('--dry_run', action='store_true')
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    for path in args.h5_paths:
        analyze_space_usage(path, args.dry_run)