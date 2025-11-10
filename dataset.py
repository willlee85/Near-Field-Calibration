"""
数据加载器，处理192x192x1的电场数据
"""
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, List
import h5py
import pickle
import re
from scipy.io import loadmat


class ElectricFieldDataset(Dataset):
    """
    电场数据集
    每对数据包括偏移电场图（输入）和标准电场图（groundtruth）
    """
    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        transform: Optional[callable] = None,
        normalize: bool = True,
        data_format: str = 'mat',  # 'h5' 或 'npy' 或 'pkl' 或 'mat'
        mat_var_name: Optional[str] = None  # MAT文件中的变量名，如果为None则自动检测
    ):
        """
        Args:
            data_dir: 数据目录路径
            split: 数据集划分 ('train', 'val', 'test')
            transform: 数据变换函数
            normalize: 是否归一化
            data_format: 数据格式
            mat_var_name: MAT文件中的变量名，如果为None则自动检测第一个非元数据的变量
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.normalize = normalize
        self.data_format = data_format
        self.mat_var_name = mat_var_name
        
        # 加载数据文件列表
        self.data_files = self._load_data_files()
        
        # 计算归一化参数（如果启用）
        if self.normalize:
            self.mean, self.std = self._compute_stats()
        else:
            self.mean, self.std = 0.0, 1.0
    
    def _extract_base_name(self, filename: str) -> str:
        """
        从文件名中提取基础名称（去掉off参数部分）
        
        例如：
        array0.25lam0.18rfN_32x27_off_0.3_0.2_0.1.mat 
        -> array0.25lam0.18rfN_32x27
        """
        # 使用正则表达式匹配off后面的部分
        pattern = r'_off_([-\d.]+)_([-\d.]+)_([-\d.]+)\.mat$'
        match = re.search(pattern, filename)
        
        if match:
            # 去掉_off_xxx_xxx_xxx.mat部分
            base_name = re.sub(pattern, '', filename)
            return base_name
        else:
            raise ValueError(f"无法从文件名中提取基础名称: {filename}")
    
    def _get_groundtruth_filename(self, offset_filename: str) -> str:
        """
        从偏移文件名生成groundtruth文件名
        
        例如：
        array0.25lam0.18rfN_32x27_off_0.3_0.2_0.1.mat 
        -> array0.25lam0.18rfN_32x27_off_0.0_0.0_0.0.mat
        """
        # 使用正则表达式匹配off后面的三个数字
        pattern = r'_off_([-\d.]+)_([-\d.]+)_([-\d.]+)\.mat$'
        match = re.search(pattern, offset_filename)
        
        if match:
            # 替换为0.0_0.0_0.0
            gt_filename = re.sub(pattern, '_off_0.0_0.0_0.0.mat', offset_filename)
            return gt_filename
        else:
            raise ValueError(f"无法从文件名中提取偏移信息: {offset_filename}")
    
    def _load_data_files(self):
        """加载数据文件列表"""
        if self.data_format == 'h5':
            # HDF5格式：假设所有数据在一个文件中
            h5_file = self.data_dir / f'{self.split}.h5'
            if h5_file.exists():
                with h5py.File(h5_file, 'r') as f:
                    num_samples = len(f['offset'])  # 假设键名为'offset'和'groundtruth'
                return list(range(num_samples))
            else:
                raise FileNotFoundError(f"数据文件不存在: {h5_file}")
        
        elif self.data_format == 'npy':
            # NPY格式：每对数据两个文件
            offset_dir = self.data_dir / self.split / 'offset'
            gt_dir = self.data_dir / self.split / 'groundtruth'
            
            if not offset_dir.exists() or not gt_dir.exists():
                raise FileNotFoundError(f"数据目录不存在: {offset_dir} 或 {gt_dir}")
            
            offset_files = sorted(offset_dir.glob('*.npy'))
            gt_files = sorted(gt_dir.glob('*.npy'))
            
            assert len(offset_files) == len(gt_files), "偏移数据和groundtruth数据数量不匹配"
            
            return list(zip(offset_files, gt_files))
        
        elif self.data_format == 'pkl':
            # Pickle格式：单个文件包含所有数据
            pkl_file = self.data_dir / f'{self.split}.pkl'
            if pkl_file.exists():
                with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
                return data
            else:
                raise FileNotFoundError(f"数据文件不存在: {pkl_file}")
        
        elif self.data_format == 'mat':
            # MAT格式：根据命名规则自动匹配文件
            # 首先尝试查找split子目录，如果不存在则使用data_dir本身
            split_path = self.data_dir / self.split
            if split_path.exists() and split_path.is_dir():
                data_path = split_path
            else:
                # 所有文件都在同一个目录下，需要按其他方式划分
                data_path = self.data_dir
            
            # 查找所有MAT文件
            all_mat_files = sorted(data_path.glob('*.mat'))
            
            if len(all_mat_files) == 0:
                raise FileNotFoundError(f"在 {data_path} 中没有找到MAT文件")
            
            # 过滤出偏移文件（off后面不是0.0_0.0_0.0的文件）
            offset_files = []
            gt_files_dict = {}  # 存储groundtruth文件，以基础名称为键
            
            for mat_file in all_mat_files:
                filename = mat_file.name
                # 检查是否是groundtruth文件（off_0.0_0.0_0.0）
                if re.search(r'_off_0\.0_0\.0_0\.0\.mat$', filename):
                    # 提取基础名称（去掉_off_0.0_0.0_0.0.mat）
                    base_name = filename.replace('_off_0.0_0.0_0.0.mat', '')
                    gt_files_dict[base_name] = mat_file
                else:
                    offset_files.append(mat_file)
            
            print(f"找到 {len(offset_files)} 个偏移文件, {len(gt_files_dict)} 个groundtruth文件")
            
            # 验证每个偏移文件都有对应的groundtruth文件
            valid_pairs = []
            missing_gt = []
            
            for offset_file in offset_files:
                # 从偏移文件名提取基础名称
                base_name = self._extract_base_name(offset_file.name)
                
                if base_name in gt_files_dict:
                    gt_file = gt_files_dict[base_name]
                    valid_pairs.append((offset_file, gt_file))
                else:
                    # 尝试使用原来的方法（向后兼容）
                    gt_filename = self._get_groundtruth_filename(offset_file.name)
                    gt_file = offset_file.parent / gt_filename
                    if gt_file.exists():
                        valid_pairs.append((offset_file, gt_file))
                    else:
                        missing_gt.append(gt_filename)
            
            if missing_gt:
                print(f"警告: 有 {len(missing_gt)} 个groundtruth文件缺失:")
                for f in missing_gt[:5]:  # 只显示前5个
                    print(f"  - {f}")
                if len(missing_gt) > 5:
                    print(f"  ... 还有 {len(missing_gt) - 5} 个")
            
            # 如果没有split子目录，需要手动划分数据集
            if not (self.data_dir / self.split).exists() and len(valid_pairs) > 0:
                # 使用固定的随机种子确保划分一致性
                import random
                random.seed(42)
                random.shuffle(valid_pairs)
                
                # 按比例划分：train 80%, val 10%, test 10%
                total = len(valid_pairs)
                if self.split == 'train':
                    valid_pairs = valid_pairs[:int(total * 0.9)]
                elif self.split == 'val':
                    valid_pairs = valid_pairs[int(total * 0.9):int(total * 0.92)]
                elif self.split == 'test':
                    valid_pairs = valid_pairs[int(total * 0.92):]
                
                print(f"自动划分 {self.split} 集: {len(valid_pairs)} 个样本")
            
            print(f"有效数据对: {len(valid_pairs)}")
            return valid_pairs
        
        else:
            raise ValueError(f"不支持的数据格式: {self.data_format}")
    
    def _get_mat_variable(self, mat_file: Path):
        """从MAT文件中获取数据变量"""
        mat_data = loadmat(str(mat_file))
        
        # 过滤掉MAT文件的元数据（以__开头的键）
        data_keys = [k for k in mat_data.keys() if not k.startswith('__')]
        
        if self.mat_var_name:
            # 使用指定的变量名
            if self.mat_var_name in mat_data:
                return mat_data[self.mat_var_name]
            else:
                raise KeyError(f"变量 '{self.mat_var_name}' 不存在于文件 {mat_file}。可用变量: {data_keys}")
        else:
            # 自动检测：使用第一个非元数据的变量
            if len(data_keys) == 0:
                raise ValueError(f"MAT文件 {mat_file} 中没有找到数据变量")
            elif len(data_keys) == 1:
                var_name = data_keys[0]
            else:
                # 如果有多个变量，优先选择形状为(192, 192)的
                for key in data_keys:
                    data = mat_data[key]
                    if isinstance(data, np.ndarray) and data.shape[:2] == (192, 192):
                        var_name = key
                        break
                else:
                    # 如果没找到，使用第一个
                    var_name = data_keys[0]
                    print(f"警告: 文件 {mat_file} 有多个变量 {data_keys}，使用 {var_name}")
            
            # 首次使用时保存变量名
            if not hasattr(self, '_detected_mat_var_name'):
                self._detected_mat_var_name = var_name
                print(f"检测到MAT文件变量名: {var_name}")
            
            return mat_data[var_name]
    
    def _compute_stats(self, sample_size: int = 1000):
        """计算数据集的均值和标准差（用于归一化）"""
        print(f"计算数据集统计信息（采样 {sample_size} 个样本）...")
        
        if self.data_format == 'h5':
            h5_file = self.data_dir / f'{self.split}.h5'
            with h5py.File(h5_file, 'r') as f:
                indices = np.random.choice(len(f['offset']), min(sample_size, len(f['offset'])), replace=False)
                samples = []
                for idx in indices:
                    offset = f['offset'][idx]
                    gt = f['groundtruth'][idx]
                    samples.extend([offset, gt])
                all_data = np.concatenate(samples)
        
        elif self.data_format == 'npy':
            sample_indices = np.random.choice(len(self.data_files), min(sample_size, len(self.data_files)), replace=False)
            samples = []
            for idx in sample_indices:
                offset_path, gt_path = self.data_files[idx]
                offset = np.load(offset_path)
                gt = np.load(gt_path)
                samples.extend([offset, gt])
            all_data = np.concatenate(samples)
        
        elif self.data_format == 'mat':
            sample_indices = np.random.choice(len(self.data_files), min(sample_size, len(self.data_files)), replace=False)
            samples = []
            for idx in sample_indices:
                offset_path, gt_path = self.data_files[idx]
                offset = self._get_mat_variable(offset_path)
                gt = self._get_mat_variable(gt_path)
                samples.extend([offset, gt])
            all_data = np.concatenate([s.flatten() for s in samples])
        
        else:
            # 对于其他格式，使用默认值
            return 0.0, 1.0
        
        mean = float(np.mean(all_data))
        std = float(np.std(all_data))
        print(f"均值: {mean:.4f}, 标准差: {std:.4f}")
        return mean, std
    
    def _load_sample(self, idx):
        """加载单个样本"""
        if self.data_format == 'h5':
            h5_file = self.data_dir / f'{self.split}.h5'
            with h5py.File(h5_file, 'r') as f:
                offset = f['offset'][idx]
                groundtruth = f['groundtruth'][idx]
                # 转换为numpy数组
                offset = np.array(offset)
                groundtruth = np.array(groundtruth)
        
        elif self.data_format == 'npy':
            offset_path, gt_path = self.data_files[idx]
            offset = np.load(offset_path)
            groundtruth = np.load(gt_path)
        
        elif self.data_format == 'pkl':
            data = self.data_files[idx]
            offset = data['offset']
            groundtruth = data['groundtruth']
        
        elif self.data_format == 'mat':
            offset_path, gt_path = self.data_files[idx]
            offset = self._get_mat_variable(offset_path)
            groundtruth = self._get_mat_variable(gt_path)
            
            # 确保是numpy数组
            offset = np.array(offset)
            groundtruth = np.array(groundtruth)
        
        # 确保形状为 (H, W) 或 (H, W, 1)
        if offset.ndim == 2:
            offset = offset[:, :, np.newaxis]
        if groundtruth.ndim == 2:
            groundtruth = groundtruth[:, :, np.newaxis]
        
        # 确保形状为 192x192x1
        assert offset.shape[:2] == (192, 192), f"偏移数据形状错误: {offset.shape}"
        assert groundtruth.shape[:2] == (192, 192), f"Groundtruth形状错误: {groundtruth.shape}"
        
        # 归一化
        if self.normalize:
            offset = (offset - self.mean) / (self.std + 1e-8)
            groundtruth = (groundtruth - self.mean) / (self.std + 1e-8)
        
        return offset, groundtruth
    
    def __len__(self):
        if self.data_format == 'h5':
            return len(self.data_files)
        else:
            return len(self.data_files)
    
    def __getitem__(self, idx):
        offset, groundtruth = self._load_sample(idx)
        
        # 转换为torch tensor
        offset = torch.from_numpy(offset).float()
        groundtruth = torch.from_numpy(groundtruth).float()
        
        # 转换为CHW格式
        if offset.dim() == 3:
            offset = offset.permute(2, 0, 1)  # HWC -> CHW
        if groundtruth.dim() == 3:
            groundtruth = groundtruth.permute(2, 0, 1)  # HWC -> CHW
        
        # 应用变换
        if self.transform:
            offset = self.transform(offset)
            groundtruth = self.transform(groundtruth)
        
        return {
            'offset': offset,  # [1, 192, 192] 条件图像
            'groundtruth': groundtruth  # [1, 192, 192] 目标图像
        }


def create_dataloader(
    data_dir: str,
    split: str = 'train',
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    **kwargs
) -> DataLoader:
    """
    创建数据加载器
    
    Args:
        data_dir: 数据目录
        split: 数据集划分
        batch_size: 批次大小
        num_workers: 工作进程数
        shuffle: 是否打乱
        **kwargs: 传递给Dataset的其他参数
    
    Returns:
        DataLoader实例
    """
    dataset = ElectricFieldDataset(data_dir, split=split, **kwargs)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True if split == 'train' else False
    )
    return dataloader


if __name__ == "__main__":
    # 测试数据加载器
    # 注意：需要实际的数据文件才能运行
    print("数据加载器测试")
    print("请确保数据文件存在并格式正确")

