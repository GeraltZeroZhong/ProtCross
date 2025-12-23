import pytorch_lightning as pl
from torch_geometric.loader import DataLoader
from typing import Optional
from .dataset import EvoPointDataset

class EvoPointDataModule(pl.LightningDataModule):
    def __init__(self, 
                 data_dir_pdb: str = "./data/processed_pdb",
                 data_dir_af2: str = "./data/processed_af2",
                 batch_size: int = 16,
                 num_workers: int = 0): # WSL上有时多线程可能有问题，先设为0测试，稳定后改4
        super().__init__()
        self.save_hyperparameters()
        
    def setup(self, stage: Optional[str] = None):
        # 实例化 Dataset (会自动触发 process 生成缓存)
        if stage == 'fit' or stage is None:
            self.train_set_pdb = EvoPointDataset(root=self.hparams.data_dir_pdb, split='train')
            # 暂时假设 AF2 数据也存在，如果不存在可能会报错，测试时可注释掉
            # self.train_set_af2 = EvoPointDataset(root=self.hparams.data_dir_af2, split='train')
            self.val_set = EvoPointDataset(root=self.hparams.data_dir_pdb, split='val')

    def train_dataloader(self):
        # 返回字典以支持多流加载 (Source + Target)
        loaders = {
            'source': DataLoader(self.train_set_pdb, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers),
            # 'target': DataLoader(self.train_set_af2, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)
        }
        return loaders

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.hparams.batch_size, num_workers=self.hparams.num_workers)
