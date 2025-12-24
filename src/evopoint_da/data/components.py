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

# 定义标准氨基酸集合（用于区分蛋白质和配体）
STANDARD_AA = {
    "ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", 
    "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"
}

class ESMFeatureExtractor:
    def __init__(self, model_name: str = "esm2_t33_650M_UR50D", device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        # print(f"[ESM] Loading {model_name}...") 
        self.model, self.alphabet = esm.pretrained.load_model_and_alphabet(model_name)
        self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()

    @torch.no_grad()
    def extract_residue_embeddings(self, sequence: str) -> torch.Tensor:
        if len(sequence) > 1022: sequence = sequence[:1022]
        data = [("protein", sequence)]
        _, _, batch_tokens = self.batch_converter(data)
        batch_tokens = batch_tokens.to(self.device)
        results = self.model(batch_tokens, repr_layers=[self.model.num_layers])
        return results["representations"][self.model.num_layers][0, 1:-1].cpu()

class StructureParser:
    def __init__(self):
        self.pdb_parser = PDBParser(QUIET=True)
        self.cif_parser = MMCIFParser(QUIET=True)

    def parse_file_with_labels(self, file_path: str, chain_id: Optional[str] = None) -> Optional[Dict]:
        """
        解析结构文件并生成结合位点标签。
        改进版逻辑：通过残基名称判断配体，支持 Open Babel 生成的 ATOM 格式配体。
        """
        is_pdb = file_path.endswith('.pdb')
        parser = self.pdb_parser if is_pdb else self.cif_parser
        try:
            structure = parser.get_structure('protein', file_path)
            model = list(structure)[0]
        except: return None

        # 1. 收集所有非水原子用于搜索环境
        # 只要不是水 (HOH, WAT)，都可能是潜在的结合对象
        all_atoms = [a for a in Selection.unfold_entities(model, 'A') if a.get_parent().get_resname().strip().upper() not in {"HOH", "WAT", "H2O"}]
        if not all_atoms: return None
        ns = NeighborSearch(all_atoms)

        coords, seq_chars, plddts, residue_ids, labels = [], [], [], [], []
        
        for chain in model:
            if chain_id and chain.id != chain_id: continue
            
            for res in chain:
                # === 核心修正 1: 仅提取标准氨基酸作为输入序列 ===
                # 排除 HETATM 和 名称不在标准列表中的残基 (如 HEM, CPZ)
                res_name = res.get_resname().strip().upper()
                is_standard = (res.id[0] == ' ') and (res_name in STANDARD_AA)
                
                if not is_standard: continue 
                if 'CA' not in res: continue
                
                # 提取特征
                ca = res['CA']
                coords.append(ca.get_coord())
                plddts.append(ca.get_bfactor()) 
                residue_ids.append(f"{chain.id}_{res.id[1]}")
                
                try: aa = seq1(res_name)
                except: aa = 'X'
                seq_chars.append(aa if len(aa)==1 else 'X')
                
                # === 核心修正 2: 标签判定逻辑 ===
                is_binding = 0.0
                neighbors = ns.search(ca.get_coord(), 6.0) # 6.0A 阈值
                
                for n_atom in neighbors:
                    n_res = n_atom.get_parent()
                    if n_res == res: continue # 排除自身

                    n_name = n_res.get_resname().strip().upper()
                    
                    # 判定为配体的条件：
                    # 1. 残基名称不是标准氨基酸 (捕捉 HEM, CPZ, 即使它是 ATOM)
                    # 2. 或者它是 HETATM (id[0] != ' ')
                    # 3. 且不是水
                    is_ligand_res = (n_name not in STANDARD_AA) or (n_res.id[0] != ' ')
                    
                    if is_ligand_res:
                        is_binding = 1.0
                        break
                
                labels.append(is_binding)

        if not coords: return None

        coords_np = np.array(coords, dtype=np.float32)
        coords_np -= coords_np.mean(axis=0) # 中心化

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
