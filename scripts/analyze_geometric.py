import torch
import os
import sys
import glob
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio.PDB import Superimposer, Atom
from Bio.Align import PairwiseAligner
import warnings
from collections import defaultdict

warnings.filterwarnings("ignore")

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import StructureParser

# ==========================================
# 复用核心组件 (保持与 map_labels.py 一致)
# ==========================================
def create_atoms(coords):
    atoms = []
    for i, coord in enumerate(coords):
        a = Atom.Atom(
            name='CA',
            coord=coord.numpy() if isinstance(coord, torch.Tensor) else coord,
            bfactor=0, occupancy=1, altloc=' ', fullname=' CA ', serial_number=i, element='C'
        )
        atoms.append(a)
    return atoms

def split_data_by_chain(full_data):
    if not full_data or 'residue_ids' not in full_data: return {}
    chains = defaultdict(lambda: {'coords': [], 'sequence': [], 'residue_ids': [], 'labels': [], 'indices': []})
    full_seq = full_data['sequence']
    for i, res_id in enumerate(full_data['residue_ids']):
        parts = res_id.split('_'); cid = parts[0]
        chains[cid]['coords'].append(full_data['coords'][i])
        chains[cid]['residue_ids'].append(res_id)
        chains[cid]['labels'].append(full_data['labels'][i])
        chains[cid]['indices'].append(i)
        if i < len(full_seq): chains[cid]['sequence'].append(full_seq[i])
    final_chains = {}
    for cid, data in chains.items():
        if len(data['sequence']) < 5: continue
        final_chains[cid] = {
            'coords': np.array(data['coords'], dtype=np.float32),
            'sequence': "".join(data['sequence']),
            'labels': np.array(data['labels'], dtype=np.float32),
            'residue_ids': data['residue_ids']
        }
    return final_chains

def find_best_matching_chain_memory(full_pdb_data, af2_seq):
    chain_map = split_data_by_chain(full_pdb_data)
    if not chain_map: return None, None, 0.0
    
    best_chain = None; best_seq = None; best_score = -1.0
    aligner = PairwiseAligner(); aligner.mode = 'global'
    aligner.match_score = 2; aligner.mismatch_score = -1
    aligner.open_gap_score = -2.0; aligner.extend_gap_score = -0.5
    aligner.target_end_open_gap_score = 0.0; aligner.target_end_extend_gap_score = 0.0
    
    af2_len = len(af2_seq)
    for cid, data in chain_map.items():
        pdb_seq = data['sequence']; pdb_len = len(pdb_seq)
        if pdb_len > af2_len * 1.5: continue
        try:
            score = aligner.score(af2_seq, pdb_seq)
            norm_score = score / (2 * pdb_len) # 覆盖率优先策略
            if norm_score > best_score:
                best_score = norm_score; best_chain = data; best_seq = pdb_seq
        except: continue
    return best_chain, best_seq, best_score

# ==========================================
# 🎯 核心逻辑：纯净环境下的几何测试
# ==========================================
def evaluate_geometric_fidelity(af2_data, pdb_data, af2_seq, pdb_seq, thresholds=[4.0, 8.0, 12.0]):
    """
    在已知链匹配正确的前提下，计算不同距离阈值下的召回率。
    """
    # 1. 对齐与叠加
    aligner = PairwiseAligner(); aligner.mode = 'global'
    aligner.match_score = 2; aligner.mismatch_score = -1
    aligner.open_gap_score = -10.0; aligner.extend_gap_score = -0.5
    aligner.target_end_open_gap_score = 0.0; aligner.target_end_extend_gap_score = 0.0 # Semiglobal
    
    try: alignment = aligner.align(af2_seq, pdb_seq)[0]
    except: return None
    
    af2_idxs, pdb_idxs = [], []
    aligned_af2, aligned_pdb = alignment.aligned[0], alignment.aligned[1]
    for (a_s, a_e), (p_s, p_e) in zip(aligned_af2, aligned_pdb):
        length = a_e - a_s
        for i in range(length):
            if af2_seq[a_s+i] == pdb_seq[p_s+i]:
                af2_idxs.append(a_s+i); pdb_idxs.append(p_s+i)
                
    if len(af2_idxs) < 10: return None
    
    # 2. 计算刚体变换 (Superposition)
    fixed_coords = af2_data['pos'][af2_idxs]
    pdb_coords_vec = pdb_data['coords'][pdb_idxs] # PDB data from memory dict
    
    sup = Superimposer()
    sup.set_atoms(create_atoms(fixed_coords), create_atoms(pdb_coords_vec))
    rot, tran = sup.rotran
    rot = torch.from_numpy(rot).float(); tran = torch.from_numpy(tran).float()
    
    # 3. 准备 PDB 真实位点
    pdb_y = pdb_data['labels']
    pdb_pos_indices = np.where(pdb_y > 0.5)[0]
    n_pdb_pos = len(pdb_pos_indices)
    
    if n_pdb_pos == 0: return None # 无阳性位点
    
    # 4. 将 PDB 阳性位点变换到 AF2 空间
    p_coords = torch.from_numpy(pdb_data['coords'][pdb_pos_indices])
    p_trans = torch.matmul(p_coords, rot.T) + tran
    
    # 5. 计算距离矩阵 (AF2 所有原子 vs PDB 阳性原子)
    # Shape: [N_AF2, N_PDB_Pos]
    dists = torch.cdist(af2_data['pos'], p_trans)
    
    # 6. 对每个阈值计算 Recall
    # 这里的 Recall 定义为：有多少个 PDB 真实位点，在 AF2 中找到了 < Threshold 的邻居？
    # 这就是 "Geometric Coverage"
    min_dists_per_pdb, _ = dists.min(dim=0) # [N_PDB_Pos]
    
    results = {}
    for th in thresholds:
        covered = (min_dists_per_pdb < th).sum().item()
        results[f"recall_{th}"] = covered
        
    results["total_sites"] = n_pdb_pos
    results["rmsd"] = sup.rms
    
    return results

def main():
    processed_af2_dir = "./data/processed_af2"
    raw_pdb_dir = "./data/raw_pdb" 
    raw_af2_dir = "./data/raw_af2" 
    mapping_file = "./pdb_uniprot_mapping.json"
    
    # 定义测试阈值
    THRESHOLDS = [4.0, 6.0, 8.0, 12.0, 16.0, 20.0]
    
    parser = StructureParser()
    if not os.path.exists(mapping_file): return
    with open(mapping_file, 'r') as f:
        uniprot_to_pdb = {v.lower(): k.lower() for k, v in json.load(f).items()}

    af2_files = glob.glob(os.path.join(processed_af2_dir, "*.pt"))
    print(f"🚀 Starting Geometric Fidelity Analysis (Clean Chain Mode)...")
    print(f"   Testing Thresholds: {THRESHOLDS} Å")
    
    stats = {
        "valid_proteins": 0,
        "total_pdb_sites": 0,
        "covered_sites": {th: 0 for th in THRESHOLDS},
        "rmsds": []
    }
    
    def find_file_robust(d, l, u):
        c = glob.glob(os.path.join(d, l)); 
        if c: return c
        return glob.glob(os.path.join(d, u))

    for af2_pt_path in tqdm(af2_files):
        basename = os.path.basename(af2_pt_path)
        
        target_pdb_id = None; current_uid = None
        for uid, pid in uniprot_to_pdb.items():
            if uid in basename.lower(): target_pdb_id = pid; current_uid = uid; break
        if not target_pdb_id: continue
            
        raw_af2_c = find_file_robust(raw_af2_dir, f"*{current_uid}*.pdb", f"*{current_uid.upper()}*.pdb")
        raw_pdb_c = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.pdb", f"{target_pdb_id.upper()}.pdb")
        if not raw_pdb_c: raw_pdb_c = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.cif", f"{target_pdb_id.upper()}.cif")

        if not (raw_af2_c and raw_pdb_c): continue
        
        try:
            # 1. 基础数据准备
            af2_data = torch.load(af2_pt_path, weights_only=False)
            try:
                parsed_af2 = parser.parse_file_with_labels(raw_af2_c[0])
                af2_seq = parsed_af2['sequence']
            except: continue

            full_pdb_data = parser.parse_file_with_labels(raw_pdb_c[0], chain_id=None)
            if not full_pdb_data: continue

            # 2. 核心步骤：找到正确的链 (排除干扰)
            best_chain_data, best_pdb_seq, score = find_best_matching_chain_memory(full_pdb_data, af2_seq)
            
            if not best_chain_data or score < 0.15: continue # 没找到对应链，跳过

            # 3. 核心测试：计算几何一致性
            res = evaluate_geometric_fidelity(af2_data, best_chain_data, af2_seq, best_pdb_seq, thresholds=THRESHOLDS)
            
            if res:
                stats["valid_proteins"] += 1
                stats["total_pdb_sites"] += res["total_sites"]
                stats["rmsds"].append(res["rmsd"])
                
                for th in THRESHOLDS:
                    stats["covered_sites"][th] += res[f"recall_{th}"]
                
        except Exception: continue

    # === 生成报告 ===
    print("\n" + "="*60)
    print("🔬 GEOMETRIC FIDELITY REPORT (Chain-Corrected)")
    print("="*60)
    print(f"Proteins Analyzed : {stats['valid_proteins']}")
    print(f"Total Binding Sites: {stats['total_pdb_sites']}")
    print(f"Avg Structural RMSD: {np.mean(stats['rmsds']):.2f} Å (Global Align)")
    print("-" * 60)
    
    df_rows = []
    if stats["total_pdb_sites"] > 0:
        for th in THRESHOLDS:
            recall = stats["covered_sites"][th] / stats["total_pdb_sites"]
            loss = 1.0 - recall
            print(f"Threshold {th:4.1f} Å : Geometric Recall = {recall:.2%} | Loss = {loss:.2%}")
            df_rows.append({"Threshold_Angstrom": th, "Geometric_Recall": recall, "Geometric_Loss": loss})
    else:
        print("No valid sites found.")

    print("="*60)
    print("Interpretation:")
    print(" -> If Loss @ 4.0A is high (>20%), it confirms 'Pocket Collapse' is real")
    print("    and not caused by wrong chain selection.")
    
    pd.DataFrame(df_rows).to_csv("geometric_fidelity_test.csv", index=False)
    print("📁 Saved to geometric_fidelity_test.csv")

if __name__ == "__main__":
    main()
