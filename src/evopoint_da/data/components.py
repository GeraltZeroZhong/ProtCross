import torch
import esm
import numpy as np
import pickle
import json
import os
from Bio.PDB import PDBParser, MMCIFParser, NeighborSearch, Selection
from Bio.SeqUtils import seq1
from sklearn.decomposition import PCA
from typing import List, Dict, Optional, Tuple

class ESMFeatureExtractor:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ESM] Loading {model_name} to {self.device}...")
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def extract_residue_embeddings(self, sequence: str) -> torch.Tensor:
        # ESM2 限制 1022 AA
        if len(sequence) > 1022: sequence = sequence[:1022]
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
        # 去掉 cls 和 eos token
        return results["representations"][self.model.num_layers][0, 1:-1].cpu()

class StructureParser:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def parse_file_with_labels(self, file_path: str, chain_id: Optional[str] = None) -> Optional[Dict]:
        """解析结构并自动计算结合位点标签 (基于 HETATM 距离)"""
        is_pdb = file_path.endswith('.pdb')
        parser = self.pdb_parser if is_pdb else self.cif_parser
        try:
            structure = parser.get_structure('protein', file_path)
            model = list(structure)[0]
        except: return None

        # 提取配体原子用于标签计算
        ligand_atoms = [a for a in Selection.unfold_entities(model, 'A') 
                       if a.get_parent().id[0] != ' ' and a.get_parent().id[0] != 'W']
        has_ligand = len(ligand_atoms) > 0
        ns = NeighborSearch(ligand_atoms) if has_ligand else None

        coords, seq_chars, plddts, residue_ids, labels = [], [], [], [], []
        
        for chain in model:
            if chain_id and chain.id != chain_id: continue
            for res in chain:
                if res.id[0] != ' ' or 'CA' not in res: continue
                
                # 基础信息
                ca = res['CA']
                coords.append(ca.get_coord())
                plddts.append(ca.get_bfactor()) # AF2 pLDDT 存放在 B-factor
                residue_ids.append(f"{chain.id}_{res.id[1]}")
                
                # 序列
                try: aa = seq1(res.get_resname())
                except: aa = 'X'
                seq_chars.append(aa if len(aa)==1 else 'X')
                
                # 标签计算 (6.0A 阈值)
                is_binding = 0.0
                if has_ligand:
                    if len(ns.search(ca.get_coord(), 6.0)) > 0:
                        is_binding = 1.0
                labels.append(is_binding)

        if not coords: return None

        # --- 核心修复：坐标中心化 (Centering) ---
        coords_np = np.array(coords, dtype=np.float32)
        centroid = coords_np.mean(axis=0)
        coords_np -= centroid 

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

class LabelManager:
    """处理外部 JSON 标签映射"""
    def __init__(self, labels_json_path: str = None):
        self.labels = {}
        if labels_json_path and os.path.exists(labels_json_path):
            with open(labels_json_path, 'r') as f:
                self.labels = json.load(f)

    def get_labels(self, pdb_id: str, residue_ids: List[str]) -> torch.Tensor:
        y = torch.zeros(len(residue_ids))
        pdb_key = pdb_id.lower()
        if pdb_key not in self.labels: return y
            
        annotated = self.labels[pdb_key] 
        for i, res_tag in enumerate(residue_ids):
            chain, res_num = res_tag.split('_')
            if chain in annotated and int(res_num) in annotated[chain]:
                y[i] = 1.0
        return y