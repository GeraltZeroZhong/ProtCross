import os
import glob
import random
import warnings
from tqdm import tqdm

import torch
from torch_geometric.data import Dataset, Data

warnings.filterwarnings("ignore")


class EvalDataset(Dataset):
    """
    Dataset used by legacy evaluation scripts.

    The split and filtering behavior intentionally mirrors EvoPointDataset:
    1. Sort -> Random Shuffle (Seed 42) -> Split
    2. Drop samples whose labels are all zero
    3. Keep pdb_id for external baselines
    """
    def __init__(self, root, split="test"):
        super().__init__()
        self.root = root

        all_files = sorted(glob.glob(os.path.join(root, "*.pt")))

        random.seed(42)
        random.shuffle(all_files)

        num = len(all_files)

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

        self.files = []
        for fpath in tqdm(candidate_files, desc=f"Filtering {split}"):
            try:
                data = torch.load(fpath, weights_only=False)

                y = data["y"] if isinstance(data, dict) else data.y

                if not isinstance(y, torch.Tensor):
                    y = torch.tensor(y)

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

        filename = os.path.basename(file_path)
        pdb_id = os.path.splitext(filename)[0]
        data.pdb_id = pdb_id

        return data
