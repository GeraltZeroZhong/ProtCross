# File: src/evopoint_da/data/dataset.py

import os
import glob
import torch
import numpy as np
import random
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

class EvoPointDataset(InMemoryDataset):
    def __init__(self, root, split="train", augment=False):
        self.split = split
        self.augment = augment
        super().__init__(root)
        
        # 自动加载处理好的缓存文件
        # 注意：如果修改了 process 逻辑，必须先手动删除 processed/ 目录下的 .pt 文件
        if not os.path.exists(self.processed_paths[0]):
            self.process()
            
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self): 
        return [f'data_cache_{self.split}.pt']

    def _augment(self, pos):
        """
        训练时对点云进行随机旋转，增强模型对旋转的不变性/鲁棒性
        """
        theta = np.random.uniform(0, 2 * np.pi)
        # 构建旋转矩阵 (z轴旋转示例，也可改为随机 SO3 旋转)
        rot = torch.tensor([
            [np.cos(theta), -np.sin(theta), 0], 
            [np.sin(theta), np.cos(theta), 0], 
            [0, 0, 1]
        ], dtype=pos.dtype)
        
        # 应用旋转并添加微小抖动噪声
        return pos @ rot + torch.randn_like(pos) * 0.01

    def get(self, idx):
        data = super().get(idx)
        # 仅在训练集的训练阶段应用增强
        if self.augment and self.split == "train":
            data.pos = self._augment(data.pos)
        return data

    def process(self):
        data_list = []
        
        # 1. 获取所有 .pt 文件
        # 使用 sorted 保证初始顺序一致
        raw_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
        
        # === 核心：固定随机种子并打乱 ===
        # 这确保了 Train/Val/Test 的划分在任何机器上都是一致的
        # 同时也消除了文件名排序带来的潜在分布偏差
        random.seed(42) 
        random.shuffle(raw_files)
        # =================================

        # 2. 按比例划分数据集
        # Train: 0% - 80%
        # Val:   80% - 90%
        # Test:  90% - 100%
        num = len(raw_files)
        if self.split == "train": 
            files = raw_files[:int(num*0.8)]
        elif self.split == "val": 
            files = raw_files[int(num*0.8):int(num*0.9)]
        else: 
            files = raw_files[int(num*0.9):] # Test Set

        print(f"[Dataset] Processing split: '{self.split}'")
        print(f"   - Source Directory: {self.root}")
        print(f"   - Candidate Files: {len(files)}")

        valid_count = 0
        for f in tqdm(files, desc=f"Loading {self.split}"):
            try:
                # 加载数据
                d = torch.load(f, weights_only=False)
                
                # === 数据清洗与过滤 ===
                
                # 1. 检查是否存在标签键
                if d.get('y') is None:
                    continue
                
                # 2. 统一转换为 Tensor 以便检查
                y_temp = d['y'] if isinstance(d['y'], torch.Tensor) else torch.tensor(d['y'])
                
                # 3. 严格过滤：丢弃标签全为 0 的样本
                # 这通常意味着对齐失败、Pocket Collapse 导致标签丢失，或该链确实无结合位点
                # 包含这些数据会引入 False Negatives，干扰训练
                if y_temp.sum() == 0:
                    continue
                
                # === 数据类型标准化 (防止 Double/Float 不匹配报错) ===
                def to_tensor(x):
                    if isinstance(x, np.ndarray): return torch.from_numpy(x).float()
                    if isinstance(x, torch.Tensor): return x.float()
                    return torch.tensor(x).float()

                # 构建 PyG Data 对象
                data = Data(
                    x=to_tensor(d['x']), 
                    pos=to_tensor(d['pos']), 
                    plddt=to_tensor(d['plddt']), 
                    y=to_tensor(d['y'])
                )
                data_list.append(data)
                valid_count += 1
                
            except Exception as e:
                print(f"⚠️ Error loading {f}: {e}")
        
        print(f"   - Final Valid Samples: {valid_count} / {len(files)}")
        
        # === 空数据保护 ===
        if not data_list:
            print(f"⚠️ WARNING: No valid data found for split '{self.split}'!")
            print(f"   This might crash the DataLoader. Creating a dummy sample.")
            # 创建一个空的 dummy 数据，防止程序直接崩溃
            dummy_x = torch.zeros((1, 1024), dtype=torch.float)
            dummy_pos = torch.zeros((1, 3), dtype=torch.float)
            dummy_y = torch.zeros((1,), dtype=torch.float)
            dummy_plddt = torch.zeros((1,), dtype=torch.float)
            data_list = [Data(x=dummy_x, pos=dummy_pos, y=dummy_y, plddt=dummy_plddt)]

        # 保存处理后的数据到缓存文件
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"   - Cache saved to: {self.processed_paths[0]}")
