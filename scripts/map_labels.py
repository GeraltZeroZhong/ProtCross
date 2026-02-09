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

# 引入项目路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import StructureParser

# ==========================================
# 辅助函数
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
    """
    将 StructureParser 解析的整块数据按链拆分。
    依靠 full_data['residue_ids'] 格式为 "CHAIN_RESID"
    """
    if not full_data or 'residue_ids' not in full_data:
        return {}

    chains = defaultdict(lambda: {
        'coords': [], 'sequence': [], 'residue_ids': [], 'labels': [], 'indices': []
    })
    
    full_seq = full_data['sequence']
    
    for i, res_id in enumerate(full_data['residue_ids']):
        # 解析 Chain ID (假设格式 "A_10" 或 " _10")
        parts = res_id.split('_')
        cid = parts[0]
        
        chains[cid]['coords'].append(full_data['coords'][i])
        chains[cid]['residue_ids'].append(res_id)
        chains[cid]['labels'].append(full_data['labels'][i])
        chains[cid]['indices'].append(i) # 记录原始索引
        
        # 处理序列
        if i < len(full_seq):
            chains[cid]['sequence'].append(full_seq[i])
            
    # 转换回 numpy/string 格式
    final_chains = {}
    for cid, data in chains.items():
        # 放宽最小长度限制，有些结合肽可能很短
        if len(data['sequence']) < 5: continue 
        
        final_chains[cid] = {
            'coords': np.array(data['coords'], dtype=np.float32),
            'sequence': "".join(data['sequence']),
            'labels': np.array(data['labels'], dtype=np.float32),
            'residue_ids': data['residue_ids']
        }
    
    return final_chains

def find_best_matching_chain_memory(full_pdb_data, af2_seq, debug=False):
    """
    在内存中寻找最佳匹配链 (修复版：优化短片段匹配)
    """
    chain_map = split_data_by_chain(full_pdb_data)
    
    if not chain_map:
        if debug: print(f"   [Debug] Failed to split data or empty data.")
        return None, None, 0.0

    best_chain_data = None
    best_pdb_seq = None
    best_score = -1.0
    best_cid = None
    
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1
    # 关键修改：Gap 惩罚调优，允许更灵活的对齐
    aligner.open_gap_score = -2.0 
    aligner.extend_gap_score = -0.5
    # 确保末端 Gap 免费 (Semi-global)，这对短 PDB 匹配长 AF2 至关重要
    aligner.target_end_open_gap_score = 0.0
    aligner.target_end_extend_gap_score = 0.0
    aligner.query_end_open_gap_score = 0.0
    aligner.query_end_extend_gap_score = 0.0

    af2_len = len(af2_seq)
    
    for cid, data in chain_map.items():
        pdb_seq = data['sequence']
        pdb_len = len(pdb_seq)
        
        # 如果 PDB 比 AF2 还长很多 (异常情况)，可以考虑跳过
        if pdb_len > af2_len * 1.5: continue

        try:
            score = aligner.score(af2_seq, pdb_seq)
            
            # === 核心修复 ===
            # 计算 "PDB Coverage Score"：PDB 序列中有多少比例匹配上了 AF2
            # 理论最大分 = 2 * pdb_len
            # 我们用 pdb_len 作为分母，而不是 max(af2, pdb)
            # 这允许短 PDB 片段 (Domain) 也能获得高分
            norm_score = score / (2 * pdb_len)
            
            if debug: print(f"   [Debug] Chain {cid}: Len {pdb_len} | Raw {score:.1f} | Norm {norm_score:.3f}")

            if norm_score > best_score:
                best_score = norm_score
                best_chain_data = data
                best_pdb_seq = pdb_seq
                best_cid = cid
        except:
            continue

    return best_chain_data, best_pdb_seq, best_score

def sequence_based_mapping(af2_data, pdb_data, af2_seq, pdb_seq):
    # === 序列比对 ===
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2
    aligner.mismatch_score = -1     
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aligner.target_end_open_gap_score = 0.0
    aligner.target_end_extend_gap_score = 0.0
    aligner.query_end_open_gap_score = 0.0
    aligner.query_end_extend_gap_score = 0.0
    
    try:
        alignment = aligner.align(af2_seq, pdb_seq)[0]
    except:
        return None, 999.0, "Alignment Error", 0, 0, 0.0
    
    map_af2_to_pdb = {}
    af2_indices_rmsd = []
    pdb_indices_rmsd = []
    
    aligned_af2, aligned_pdb = alignment.aligned[0], alignment.aligned[1]
    
    for (a_start, a_end), (p_start, p_end) in zip(aligned_af2, aligned_pdb):
        length = a_end - a_start
        for i in range(length):
            if af2_seq[a_start+i] == pdb_seq[p_start+i]:
                map_af2_to_pdb[a_start+i] = p_start+i
                af2_indices_rmsd.append(a_start+i)
                pdb_indices_rmsd.append(p_start+i)

    if len(map_af2_to_pdb) < 10:
        return None, 999.0, "Too few aligned residues", 0, 0, 0.0

    # === RMSD Check ===
    fixed_coords = af2_data['pos'][af2_indices_rmsd]
    pdb_coords_vec = pdb_data['coords'][pdb_indices_rmsd]
    
    sup = Superimposer()
    sup.set_atoms(create_atoms(fixed_coords), create_atoms(pdb_coords_vec))
    
    # 放宽 RMSD 到 30.0，因为我们已经确认序列是对的，大 RMSD 只是说明结构变化大
    # 这正是我们要保留的数据
    if sup.rms > 30.0: 
        return None, sup.rms, f"Extreme RMSD ({sup.rms:.1f})", 0, 0, 0.0

    # === 标签迁移 ===
    num_af2_res = af2_data['pos'].shape[0]
    new_labels = torch.zeros(num_af2_res, dtype=torch.float32)
    
    pdb_y = pdb_data['labels']
    n_pdb_pos = (pdb_y > 0.5).sum()
    
    mapped_count = 0
    site_displacements = []
    rot, tran = sup.rotran
    rot = torch.from_numpy(rot).float()
    tran = torch.from_numpy(tran).float()
    
    for af2_idx, pdb_idx in map_af2_to_pdb.items():
        if pdb_idx < len(pdb_y) and pdb_y[pdb_idx] > 0.5:
            new_labels[af2_idx] = 1.0
            mapped_count += 1
            
            p_coord = torch.from_numpy(pdb_data['coords'][pdb_idx])
            p_trans = torch.matmul(p_coord, rot.T) + tran
            a_coord = af2_data['pos'][af2_idx]
            dist = torch.norm(a_coord - p_trans).item()
            site_displacements.append(dist)
            
    mean_shift = np.mean(site_displacements) if site_displacements else 0.0
    
    return new_labels, sup.rms, "Success", mapped_count, n_pdb_pos, mean_shift

def find_file_robust(directory, pattern_lower, pattern_upper):
    c = glob.glob(os.path.join(directory, pattern_lower))
    if c: return c
    c = glob.glob(os.path.join(directory, pattern_upper))
    if c: return c
    return []

def get_sequence_simple(parser, raw_path):
    try:
        parsed = parser.parse_file_with_labels(raw_path) 
        if not parsed: return None
        return parsed['sequence']
    except: return None

# ==========================================
# 主程序
# ==========================================
def main():
    processed_pdb_dir = "./data/processed_pdb"
    processed_af2_dir = "./data/processed_af2"
    raw_pdb_dir = "./data/raw_pdb" 
    raw_af2_dir = "./data/raw_af2" 
    mapping_file = "./pdb_uniprot_mapping.json"
    OUTPUT_CSV = "mapping_report_final.csv"

    parser = StructureParser()
    if not os.path.exists(mapping_file): 
        print("Mapping file missing."); return
    with open(mapping_file, 'r') as f:
        uniprot_to_pdb = {v.lower(): k.lower() for k, v in json.load(f).items()}

    af2_files = glob.glob(os.path.join(processed_af2_dir, "*.pt"))
    print(f"Scanning {len(af2_files)} processed AF2 files (Final Optimized Mode)...")
    
    stats = {
        "matched": 0, "failed": 0, "skip": 0, "no_chain_found": 0,
        "total_pdb_sites": 0, "mapped_af2_sites": 0, "site_shifts": []
    }
    
    # 调试前 5 个
    debug_limit = 5
    debug_counter = 0

    for af2_pt_path in tqdm(af2_files):
        basename = os.path.basename(af2_pt_path)
        
        target_pdb_id = None; current_uid = None
        for uid, pid in uniprot_to_pdb.items():
            if uid in basename.lower(): target_pdb_id = pid; current_uid = uid; break
        
        if not target_pdb_id: stats["skip"] += 1; continue
            
        raw_af2_c = find_file_robust(raw_af2_dir, f"*{current_uid}*.pdb", f"*{current_uid.upper()}*.pdb")
        raw_pdb_c = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.pdb", f"{target_pdb_id.upper()}.pdb")
        if not raw_pdb_c: raw_pdb_c = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.cif", f"{target_pdb_id.upper()}.cif")

        if not (raw_af2_c and raw_pdb_c): stats["skip"] += 1; continue
        
        is_debug = (debug_counter < debug_limit)
        if is_debug: 
            print(f"\n[Processing] {basename} -> PDB: {target_pdb_id}")
            debug_counter += 1

        try:
            af2_data = torch.load(af2_pt_path, weights_only=False)
            af2_seq = get_sequence_simple(parser, raw_af2_c[0])
            if not af2_seq: 
                stats["failed"] += 1; continue

            full_pdb_data = parser.parse_file_with_labels(raw_pdb_c[0], chain_id=None)
            if not full_pdb_data:
                stats["failed"] += 1; continue

            best_chain_data, best_pdb_seq, score = find_best_matching_chain_memory(
                full_pdb_data, af2_seq, debug=is_debug
            )
            
            # 阈值从 0.2 放宽到 0.15，以召回短片段
            if not best_chain_data or score < 0.15:
                stats["no_chain_found"] += 1
                stats["failed"] += 1
                if is_debug: print(f"   [Fail] No matching chain found (Best Score: {score:.3f})")
                continue

            new_labels, rmsd, msg, n_mapped, n_total, mean_shift = sequence_based_mapping(
                af2_data, best_chain_data, af2_seq, best_pdb_seq
            )
            
            if is_debug:
                print(f"   [Result] {msg} | PDB Sites: {n_total} -> Mapped: {n_mapped}")

            if new_labels is not None:
                stats["total_pdb_sites"] += int(n_total)
                stats["mapped_af2_sites"] += int(n_mapped)
                if mean_shift > 0: stats["site_shifts"].append(mean_shift)
                
                if n_mapped > 0:
                    af2_data['y'] = new_labels
                    torch.save(af2_data, af2_pt_path)
                    stats["matched"] += 1
                else:
                    stats["skip"] += 1 
            else:
                stats["failed"] += 1
                
        except Exception as e:
            stats["failed"] += 1

    total = stats['matched'] + stats['failed'] + stats['skip'] + stats['no_chain_found']
    success_rate = stats['matched'] / total if total > 0 else 0.0
    loss_rate = 1.0 - (stats["mapped_af2_sites"] / stats["total_pdb_sites"]) if stats["total_pdb_sites"] > 0 else 0.0
    avg_shift = np.mean(stats["site_shifts"]) if stats["site_shifts"] else 0.0
    
    print(f"\n✅ Final Processing Complete.")
    print("="*50)
    print(f"Alignment Success Rate : {success_rate:.2%}")
    print(f"Sequence Label Loss    : {loss_rate:.2%}")
    print(f"Avg Binding Site Shift : {avg_shift:.2f} Å")
    print(f"Mapped/Total Sites     : {stats['mapped_af2_sites']}/{stats['total_pdb_sites']}")
    print(f"Failed (No Chain)      : {stats['no_chain_found']}")
    print("="*50)
    
    pd.DataFrame({
        "Total_Samples": [total],
        "Success_Rate": [success_rate],
        "Sequence_Label_Loss": [loss_rate],
        "Avg_Site_Shift": [avg_shift]
    }).to_csv(OUTPUT_CSV, index=False)
    print(f"📁 Report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()