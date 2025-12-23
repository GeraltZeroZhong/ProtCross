import hydra
from omegaconf import DictConfig, OmegaConf
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import CSVLogger
import os
import sys

# 添加 src 到 Python 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evopoint_da.data.datamodule import EvoPointDataModule
from evopoint_da.models.module import EvoPointDALitModule

@hydra.main(version_base="1.3", config_path="configs", config_name="train")
def main(cfg: DictConfig):
    seed_everything(cfg.seed, workers=True)
    
    # 获取原始项目根目录 (不受 Hydra 影响)
    try:
        orig_cwd = hydra.utils.get_original_cwd()
    except:
        orig_cwd = os.getcwd()
        
    print(f"[Train] Project Root: {orig_cwd}")
    
    # 2. 实例化 DataModule
    print(f"[Train] Instantiating DataModule <{cfg.data._target_}>")
    # 确保数据路径也是绝对路径
    if not os.path.isabs(cfg.data.data_dir_pdb):
        cfg.data.data_dir_pdb = os.path.join(orig_cwd, cfg.data.data_dir_pdb)
    if not os.path.isabs(cfg.data.data_dir_af2):
        cfg.data.data_dir_af2 = os.path.join(orig_cwd, cfg.data.data_dir_af2)
        
    datamodule = hydra.utils.instantiate(cfg.data)
    
    # 3. 实例化 Model
    print(f"[Train] Instantiating Model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)
    
    # 4. 回调函数 (Callbacks)
    ckpt_dir = os.path.join(orig_cwd, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    print(f"[Train] Checkpoints will be saved to: {ckpt_dir}")
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="best-{epoch:02d}", # 简化文件名，避免包含 metrics 导致解析错误
        monitor="train/loss_total",
        mode="min",
        save_top_k=1,
        save_last=True, 
        every_n_epochs=1, # 明确每个 Epoch 检查一次
    )
    
    # 显式使用 CSVLogger，避免 TensorBoard 依赖问题
    logger = CSVLogger(save_dir=os.path.join(orig_cwd, "logs"), name="evopoint_da")

    # 5. 实例化 Trainer
    print(f"[Train] Instantiating Trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=[checkpoint_callback, RichProgressBar()],
        logger=logger,
        enable_checkpointing=True # 确保开启
    )
    
    # 6. 开始训练
    print("[Train] Starting training...")
    trainer.fit(model=model, datamodule=datamodule)
    
    print(f"[Train] Done! Checkpoints saved in {ckpt_dir}")

if __name__ == "__main__":
    main()