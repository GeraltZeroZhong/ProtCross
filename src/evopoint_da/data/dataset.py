# File: dataset.py | Path: evopoint_da/src/evopoint_da/data/dataset.py
import os
import glob
import torch
import numpy as np
from torch_geometric.data import InMemoryDataset, Data
from tqdm import tqdm

class EvoPointDataset(InMemoryDataset):
    def __init__(self, root, split="train", augment=False):
        self.split = split
        self.augment = augment
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)

    @property
    def processed_file_names(self): return [f'data_cache_{self.split}.pt']

    def _augment(self, pos):
        # 随机旋转
        theta = np.random.uniform(0, 2 * np.pi)
        rot = torch.tensor([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0,0,1]])
        return pos @ rot + torch.randn_like(pos) * 0.01

    def get(self, idx):
        data = super().get(idx)
        if self.augment and self.split == "train":
            data.pos = self._augment(data.pos)
        return data

    def process(self):
        data_list = []
        raw_files = sorted(glob.glob(os.path.join(self.root, "*.pt")))
        # 划分逻辑：Train (前80%), Val (80-90%), Test (全部)
        num = len(raw_files)
        if self.split == "train": files = raw_files[:int(num*0.8)]
        elif self.split == "val": files = raw_files[int(num*0.8):int(num*0.9)]
        else: files = raw_files # Test 模式加载全部

        print(f"[Dataset] Loading .pt files from {self.root}...")
        for f in tqdm(files):
            try:
                # 兼容加载方式
                d = torch.load(f, weights_only=False)
                
                # 过滤无标签数据 (仅针对 val/test)
                if self.split in ['val', 'test'] and (d.get('y') is None or d['y'].sum() == 0): 
                    continue
                
                # FIX: 检查数据类型，避免 double conversion error
                # 如果已经是 Tensor，直接用；如果是 numpy，转 Tensor
                def to_tensor(x):
                    if isinstance(x, np.ndarray): return torch.from_numpy(x)
                    if isinstance(x, torch.Tensor): return x
                    return torch.tensor(x)

                data = Data(
                    x=to_tensor(d['x']), 
                    pos=to_tensor(d['pos']), 
                    plddt=to_tensor(d['plddt']), 
                    y=to_tensor(d['y'])
                )
                data_list.append(data)
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        if not data_list:
            print(f"Warning: No valid data found for split {self.split}")
            # 创建一个空的 dataset 防止 crash，或者 raise error
            # raise RuntimeError("No data found")
        
        torch.save(self.collate(data_list), self.processed_paths[0])