import os
import glob
import random
import warnings
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

# 忽略警告
warnings.filterwarnings("ignore")


class EvalDataset(Dataset):
    """
    专门用于评估的 Dataset。
    修改：逻辑与 EvoPointDataset (src/evopoint_da/data/dataset.py) 保持严格一致
    1. Sort -> Random Shuffle (Seed 42) -> Split
    2. 过滤掉 y.sum() == 0 的无效样本
    3. 保留 pdb_id 用于 external baselines
    """
    def __init__(self, root, split="test"):
        super().__init__()
        self.root = root

        # 1. 获取所有 .pt 文件并排序 (保证初始顺序一致)
        all_files = sorted(glob.glob(os.path.join(root, "*.pt")))

        # 2. 核心：固定随机种子并打乱 (复制 EvoPointDataset 逻辑)
        random.seed(42)
        random.shuffle(all_files)

        num = len(all_files)

        # 3. 按比例划分 (8:1:1)
        if split == "train":
            candidate_files = all_files[:int(num * 0.8)]
        elif split == "val":
            candidate_files = all_files[int(num * 0.8):int(num * 0.9)]
        elif split == "test":
            candidate_files = all_files[int(num * 0.9):]
        elif split == "all":
            candidate_files = all_files
        else:
            candidate_files = []

        print(f"[{split}] Scanning {len(candidate_files)} candidate files from {root}...")

        # 4. 过滤无效数据 (y.sum() == 0) 并保留有效文件列表
        self.files = []
        for fpath in tqdm(candidate_files, desc=f"Filtering {split}"):
            try:
                # 只读取 label 进行检查，减少内存消耗
                data = torch.load(fpath, weights_only=False)

                # 兼容字典或 Data 对象
                y = data["y"] if isinstance(data, dict) else data.y

                # 转换为 tensor 检查
                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y)

                # 严格过滤：丢弃标签全为 0 的样本
                if y.sum() > 0:
                    self.files.append(fpath)
            except Exception as e:
                print(f"Error checking {fpath}: {e}")
                continue

        print(f"[{split}] Final valid files: {len(self.files)} (Filtered {len(candidate_files) - len(self.files)})")

    def len(self):
        return len(self.files)

    def get(self, idx):
        file_path = self.files[idx]
        try:
            payload = torch.load(file_path, weights_only=False)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return Data()  # Return empty data to avoid crash

        data = Data(**payload) if isinstance(payload, dict) else payload

        # [关键] 强制添加 pdb_id
        filename = os.path.basename(file_path)
        pdb_id = os.path.splitext(filename)[0]
        data.pdb_id = pdb_id

        return data
