# src/evopoint_da/data/components.py

import torch
import os
import numpy as np
import pickle
import json

# === 导入 ESMC 相关组件 ===
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein
# 尝试导入 Tokenizer，兼容不同 esm 版本路径
try:
    from esm.tokenization import EsmSequenceTokenizer
except ImportError:
    try:
        from esm.tokenization.sequence_tokenizer import EsmSequenceTokenizer
    except ImportError:
        pass 
# ========================

from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch, Selection
from Bio.SeqUtils import seq1
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple

STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", 
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
}

class ESMFeatureExtractor:
    def __init__(self, model_path: str, device: str = None):
        """
        Args:
            model_path: 本地 .pth 文件的完整路径
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESMC] Initializing model structure (600M) and loading local weights...")
        
        # 1. 实例化 Tokenizer
        try:
            # 尝试无参初始化 (通常使用库内置词表)
            self.tokenizer = EsmSequenceTokenizer()
        except Exception as e:
            print(f"Warning: Default tokenizer init failed ({e}), attempting fallback with model name...")
            self.tokenizer = EsmSequenceTokenizer(model_name="esmc_600m")

        # 2. 定义 ESMC-600M 模型参数 (硬编码以避免依赖 config.json)
        model_args = {
            "d_model": 1152,
            "n_layers": 36,
            "n_heads": 18,
        }
        
        # 3. 手动初始化模型结构
        try:
            self.model = ESMC(tokenizer=self.tokenizer, **model_args)
        except Exception as e:
            raise RuntimeError(f"ESMC init failed: {e}. Please check 'esm' library version.")

        # 4. 加载本地权重
        self.model = self.model.to(self.device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ 找不到权重文件: {model_path}")
            
        print(f"[ESMC] Loading state dict from {model_path}...")
        state_dict = torch.load(model_path, map_location=self.device)
        
        # 兼容性处理: 提取 state_dict
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]
            
        # 移除前缀 (module. 或 model.) 以匹配手动初始化的模型 keys
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if new_key.startswith("module."): new_key = new_key[7:]
            if new_key.startswith("model."): new_key = new_key[6:]
            new_state_dict[new_key] = v
            
        # 加载
        msg = self.model.load_state_dict(new_state_dict, strict=False)
        print(f"[ESMC] Weights loaded. {msg}")
        
        self.model.eval()

    @torch.no_grad()
    def extract_residue_embeddings(self, sequence: str) -> torch.Tensor:
        # 1. 截断序列
        if len(sequence) > 1022: sequence = sequence[:1022]
        
        # 2. Tokenize
        protein = ESMProtein(sequence=sequence)
        tokenized_output = self.model.encode(protein)
        
        # === 修复: 提取 Tensor 并增加 Batch 维度 ===
        # tokenized_output.sequence 是 Tensor (SeqLen,)
        input_ids = tokenized_output.sequence
        
        # 关键修正：必须是 (1, SeqLen) 才能正确广播 Attention Heads
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            
        input_ids = input_ids.to(self.device)
        
        # 可选：显式构建 mask (虽非必须，但更稳健)
        attention_mask = torch.ones_like(input_ids, dtype=torch.bool, device=self.device)
        
        # 3. Forward
        try:
            output = self.model(input_ids, attention_mask=attention_mask)
        except TypeError:
            # 如果旧版 API 不接受 attention_mask
            output = self.model(input_ids)
        
        # 4. 提取输出
        # output.embeddings Shape: (1, SeqLen, Dim)
        # 切片: [0] 取 batch, [1:-1] 去除 BOS/EOS
        return output.embeddings[0, 1:-1, :].cpu()

class StructureParser:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def parse_file_with_labels(self, file_path: str, chain_id: Optional[str] = None) -> Optional[Dict]:
        is_pdb = file_path.endswith('.pdb')
        parser = self.pdb_parser if is_pdb else self.cif_parser
        try:
            structure = parser.get_structure('protein', file_path)
            model = list(structure)[0]
        except: return None

        all_atoms = [a for a in Selection.unfold_entities(model, 'A') if a.get_parent().get_resname().strip().upper() not in {"HOH", "WAT", "H2O"}]
        if not all_atoms: return None
        ns = NeighborSearch(all_atoms)

        coords, seq_chars, plddts, residue_ids, labels = [], [], [], [], []
        
        for chain in model:
            if chain_id and chain.id != chain_id: continue
            
            for res in chain:
                res_name = res.get_resname().strip().upper()
                is_standard = (res.id[0] == ' ') and (res_name in STANDARD_AA)
                
                if not is_standard: continue 
                if 'CA' not in res: continue
                
                ca = res['CA']
                coords.append(ca.get_coord())
                plddts.append(ca.get_bfactor()) 
                residue_ids.append(f"{chain.id}_{res.id[1]}")
                
                try: aa = seq1(res_name)
                except: aa = 'X'
                seq_chars.append(aa if len(aa)==1 else 'X')
                
                is_binding = 0.0
                neighbors = ns.search(ca.get_coord(), 6.0)
                
                for n_atom in neighbors:
                    n_res = n_atom.get_parent()
                    if n_res == res: continue 

                    n_name = n_res.get_resname().strip().upper()
                    is_ligand_res = (n_name not in STANDARD_AA) or (n_res.id[0] != ' ')
                    
                    if is_ligand_res:
                        is_binding = 1.0
                        break
                
                labels.append(is_binding)

        if not coords: return None

        coords_np = np.array(coords, dtype=np.float32)
        coords_np -= coords_np.mean(axis=0)

        return {
            "coords": coords_np, 
            "sequence": "".join(seq_chars),
            "plddts": np.array(plddts, dtype=np.float32),
            "residue_ids": residue_ids,
            "labels": np.array(labels, dtype=np.float32)
        }

class PCAReducer:
    def __init__(self, n_components: int = 128):
        self.n_components = n_components
        self.pca = PCA(n_components=n_components)
        self.is_fitted = False
    def fit(self, data_list: List[torch.Tensor]):
        X = torch.cat(data_list, dim=0).numpy()
        self.pca.fit(X)
        self.is_fitted = True
    def transform(self, x: torch.Tensor) -> torch.Tensor:
        if not self.is_fitted: raise RuntimeError("PCA Not fitted")
        return torch.from_numpy(self.pca.transform(x.numpy())).float()
    def save(self, path: str):
        with open(path, 'wb') as f: pickle.dump(self.pca, f)
    def load(self, path: str):
        with open(path, 'rb') as f: self.pca = pickle.load(f)
        self.is_fitted = True
