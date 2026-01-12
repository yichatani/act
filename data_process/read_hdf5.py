"""
# 基础查看
python read_hdf5.py episode_0.hdf5

# 详细模式（显示更多数据样本）
python read_hdf5.py episode_0.hdf5 --verbose

# 提取特定数据集
python read_hdf5.py episode_0.hdf5 --extract /observations/qpos

# 提取并保存为numpy文件
python read_hdf5.py episode_0.hdf5 --extract /observations/qpos --save qpos_data.npy

# 比较多个episode文件
python read_hdf5.py episode_0.hdf5 --compare episode_0.hdf5 episode_1.hdf5 episode_2.hdf5
"""

# python ./data_process/read_hdf5.py ./data/sim_transfer_cube_scripted/episode_0.hdf5 --verbose

import h5py
import numpy as np
import argparse
from pathlib import Path


def print_structure(name, obj, indent=0):
    """递归打印HDF5文件结构"""
    prefix = "  " * indent
    
    if isinstance(obj, h5py.Group):
        print(f"{prefix}📁 Group: {name}")
        print(f"{prefix}   Keys: {list(obj.keys())}")
    elif isinstance(obj, h5py.Dataset):
        print(f"{prefix}📄 Dataset: {name}")
        print(f"{prefix}   Shape: {obj.shape}")
        print(f"{prefix}   Dtype: {obj.dtype}")
        print(f"{prefix}   Size: {obj.size} elements")
        
        # 打印属性
        if len(obj.attrs) > 0:
            print(f"{prefix}   Attributes:")
            for attr_name, attr_value in obj.attrs.items():
                print(f"{prefix}     - {attr_name}: {attr_value}")


def explore_hdf5(file_path, verbose=False):
    """完整探索HDF5文件"""
    print("=" * 80)
    print(f"📖 Reading HDF5 file: {file_path}")
    print("=" * 80)
    
    with h5py.File(file_path, 'r') as f:
        print("\n🌲 File Structure:")
        print("-" * 80)
        f.visititems(print_structure)
        
        print("\n" + "=" * 80)
        print("📊 Detailed Analysis:")
        print("=" * 80)
        
        # 递归读取所有数据
        def analyze_group(group, path=""):
            for key in group.keys():
                current_path = f"{path}/{key}" if path else key
                item = group[key]
                
                if isinstance(item, h5py.Group):
                    print(f"\n📂 Group: {current_path}")
                    analyze_group(item, current_path)
                elif isinstance(item, h5py.Dataset):
                    print(f"\n📋 Dataset: {current_path}")
                    print(f"   Shape: {item.shape}, Dtype: {item.dtype}")
                    
                    # 读取数据
                    data = item[()]
                    
                    # 根据数据类型显示信息
                    if isinstance(data, np.ndarray):
                        print(f"   Array shape: {data.shape}")
                        print(f"   Array dtype: {data.dtype}")
                        
                        if data.size > 0:
                            if np.issubdtype(data.dtype, np.number):
                                print(f"   Min: {np.min(data):.6f}, Max: {np.max(data):.6f}")
                                print(f"   Mean: {np.mean(data):.6f}, Std: {np.std(data):.6f}")
                            
                            # 显示前几个元素
                            if verbose:
                                if data.size <= 20:
                                    print(f"   Data: {data}")
                                else:
                                    print(f"   First few elements:")
                                    if data.ndim == 1:
                                        print(f"   {data[:10]}")
                                    elif data.ndim == 2:
                                        print(f"   {data[:3, :]}")
                                    elif data.ndim == 3:
                                        print(f"   {data[:2, :2, :]}")
                                    else:
                                        print(f"   Shape: {data.shape} (too large to display)")
                    else:
                        print(f"   Value: {data}")
                    
                    # 显示属性
                    if len(item.attrs) > 0:
                        print(f"   Attributes:")
                        for attr_name, attr_value in item.attrs.items():
                            print(f"     {attr_name}: {attr_value}")
        
        analyze_group(f)
        
        print("\n" + "=" * 80)
        print("✅ Analysis complete!")
        print("=" * 80)


def extract_data(file_path, dataset_path=None):
    """提取特定数据集或所有数据"""
    with h5py.File(file_path, 'r') as f:
        if dataset_path:
            # 提取特定数据集
            if dataset_path in f:
                data = f[dataset_path][()]
                print(f"Extracted dataset: {dataset_path}")
                print(f"Shape: {data.shape if isinstance(data, np.ndarray) else type(data)}")
                return data
            else:
                print(f"Dataset {dataset_path} not found!")
                print(f"Available datasets:")
                def print_datasets(name, obj):
                    if isinstance(obj, h5py.Dataset):
                        print(f"  - {name}")
                f.visititems(print_datasets)
                return None
        else:
            # 提取所有数据到字典
            data_dict = {}
            
            def collect_data(name, obj):
                if isinstance(obj, h5py.Dataset):
                    data_dict[name] = obj[()]
            
            f.visititems(collect_data)
            print(f"Extracted {len(data_dict)} datasets")
            return data_dict


def compare_episodes(file_paths):
    """比较多个episode文件"""
    print("=" * 80)
    print("🔄 Comparing multiple HDF5 files:")
    print("=" * 80)
    
    all_data = {}
    for fp in file_paths:
        print(f"\n📁 Loading: {fp}")
        with h5py.File(fp, 'r') as f:
            file_data = {}
            def collect(name, obj):
                if isinstance(obj, h5py.Dataset):
                    file_data[name] = {
                        'shape': obj.shape,
                        'dtype': obj.dtype,
                        'data': obj[()]
                    }
            f.visititems(collect)
            all_data[fp] = file_data
    
    # 比较结构
    print("\n📊 Structure comparison:")
    all_keys = set()
    for data in all_data.values():
        all_keys.update(data.keys())
    
    for key in sorted(all_keys):
        print(f"\n  Dataset: {key}")
        for fp, data in all_data.items():
            if key in data:
                print(f"    {Path(fp).name}: shape={data[key]['shape']}, dtype={data[key]['dtype']}")
            else:
                print(f"    {Path(fp).name}: ❌ Missing")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Read and analyze HDF5 files')
    parser.add_argument('file', type=str, help='Path to HDF5 file')
    parser.add_argument('--verbose', '-v', action='store_true', 
                        help='Show more detailed information including data samples')
    parser.add_argument('--extract', type=str, default=None,
                        help='Extract specific dataset (e.g., /observations/qpos)')
    parser.add_argument('--compare', type=str, nargs='+', default=None,
                        help='Compare multiple episode files')
    parser.add_argument('--save', type=str, default=None,
                        help='Save extracted data to numpy file')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_episodes(args.compare)
    elif args.extract:
        data = extract_data(args.file, args.extract)
        if data is not None and args.save:
            np.save(args.save, data)
            print(f"Saved to {args.save}")
    else:
        explore_hdf5(args.file, args.verbose)
        
        # 提示可用的提取命令
        print("\n💡 Tip: To extract specific data, use:")
        print(f"   python read_hdf5.py {args.file} --extract /path/to/dataset")
        print(f"   python read_hdf5.py {args.file} --extract /path/to/dataset --save output.npy")