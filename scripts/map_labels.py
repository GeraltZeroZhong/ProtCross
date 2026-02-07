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

# 忽略 Biopython 的其他潜在警告，保持输出整洁
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 添加 src 路径以导入 StructureParser
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import StructureParser

def create_atoms(coords):
    """辅助函数：将坐标转换为 Atom 对象"""
    atoms = []
    for i, coord in enumerate(coords):
        a = Atom.Atom(
            name='CA',
            coord=coord.numpy() if isinstance(coord, torch.Tensor) else coord,
            bfactor=0, occupancy=1, altloc=' ', fullname=' CA ', serial_number=i, element='C'
        )
        atoms.append(a)
    return atoms

def get_sequence_and_indices(parser, raw_path):
    """从原始文件解析序列"""
    try:
        parsed = parser.parse_file_with_labels(raw_path)
        if not parsed: return None, None
        
        MAX_LEN = 1022
        seq = parsed['sequence']
        if len(seq) > MAX_LEN:
            seq = seq[:MAX_LEN]
        return seq, parsed
    except Exception as e:
        return None, None

def robust_align_and_map(af2_data, pdb_data, af2_seq, pdb_seq, threshold=4.0):
    # === 1. 参数优化后的序列比对 ===
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    
    # 核心参数：防止溢出并确保准确性
    aligner.match_score = 2
    aligner.mismatch_score = -1     
    aligner.open_gap_score = -10.0  
    aligner.extend_gap_score = -0.5
    
    aligner.target_end_open_gap_score = 0.0
    aligner.target_end_extend_gap_score = 0.0
    aligner.query_end_open_gap_score = 0.0
    aligner.query_end_extend_gap_score = 0.0
    
    try:
        alignments = aligner.align(af2_seq, pdb_seq)
        alignment = alignments[0]
    except OverflowError:
        return None, 999.0, "Alignment Overflow"
    except Exception as e:
        return None, 999.0, f"Alignment Error: {e}"
    
    # === 2. 提取匹配索引 ===
    af2_indices = []
    pdb_indices = []
    
    aligned_af2 = alignment.aligned[0]
    aligned_pdb = alignment.aligned[1]
    
    for (a_start, a_end), (p_start, p_end) in zip(aligned_af2, aligned_pdb):
        segment_len = a_end - a_start
        for i in range(segment_len):
            idx_a = a_start + i
            idx_p = p_start + i
            if af2_seq[idx_a] == pdb_seq[idx_p]:
                af2_indices.append(idx_a)
                pdb_indices.append(idx_p)

    if len(af2_indices) < 10:
        return None, 999.0, f"Too few aligned residues ({len(af2_indices)})"

    # === 3. 结构叠加 ===
    fixed_coords = af2_data['pos'][af2_indices]
    moving_coords = pdb_data['pos'][pdb_indices]
    
    fixed_atoms = create_atoms(fixed_coords)
    moving_atoms = create_atoms(moving_coords)
    
    sup = Superimposer()
    sup.set_atoms(fixed_atoms, moving_atoms)
    
    if sup.rms > 10.0:
        return None, sup.rms, f"High RMSD ({sup.rms:.2f})"

    # === 4. 迁移标签 ===
    pdb_pos_indices = torch.where(pdb_data['y'] > 0.5)[0]
    if len(pdb_pos_indices) == 0:
        return None, sup.rms, "No positive labels in PDB"
        
    pdb_positive_coords = pdb_data['pos'][pdb_pos_indices]
    pdb_pos_atoms = create_atoms(pdb_positive_coords)
    sup.apply(pdb_pos_atoms)
    
    transformed_pos_sites = torch.tensor(
        np.array([a.get_coord() for a in pdb_pos_atoms]), dtype=torch.float32
    )
    
    dists = torch.cdist(af2_data['pos'], transformed_pos_sites)
    min_dists, _ = dists.min(dim=1)
    new_labels = (min_dists < threshold).float()
    
    return new_labels, sup.rms, "Success"

def find_file_robust(directory, pattern_lower, pattern_upper):
    """辅助函数：同时尝试小写和大写文件名的匹配"""
    candidates = glob.glob(os.path.join(directory, pattern_lower))
    if candidates: return candidates
    candidates = glob.glob(os.path.join(directory, pattern_upper))
    if candidates: return candidates
    return []

def main():

    processed_pdb_dir = "./data/processed_pdb"
    processed_af2_dir = "./data/processed_af2"
    raw_pdb_dir = "./data/raw_pdb" 
    raw_af2_dir = "./data/raw_af2" 
    mapping_file = "./pdb_uniprot_mapping.json"
    DIST_THRESHOLD = 4.0 
    OUTPUT_CSV = "mapping_report.csv"  

    print(f"Checking directories...")
    if not os.path.exists(raw_pdb_dir):
        print(f" Error: Raw PDB directory not found: {raw_pdb_dir}")
        return
    if not os.path.exists(raw_af2_dir):
        print(f" Error: Raw AF2 directory not found: {raw_af2_dir}")
        return

    parser = StructureParser()

    if not os.path.exists(mapping_file):
        print("Error: Mapping file not found.")
        return
    with open(mapping_file, 'r') as f:
        raw_mapping = json.load(f)
    uniprot_to_pdb = {v.lower(): k.lower() for k, v in raw_mapping.items()}

    af2_files = glob.glob(os.path.join(processed_af2_dir, "*.pt"))
    print(f"Scanning {len(af2_files)} processed AF2 files...")
    

    stats = {
        "matched": 0, "failed": 0, "skip": 0,
        "total_pdb_sites": 0,
        "mapped_af2_sites": 0,
        "high_rmsd_drops": 0
    }
    
    debug_limit = 5
    debug_count = 0

    for af2_pt_path in tqdm(af2_files):
        basename = os.path.basename(af2_pt_path)
        
        target_pdb_id = None
        current_uid = None
        for uid, pid in uniprot_to_pdb.items():
            if uid in basename.lower():
                target_pdb_id = pid
                current_uid = uid
                break
        
        if not target_pdb_id:
            stats["skip"] += 1
            continue
            
        pdb_pt_path = os.path.join(processed_pdb_dir, f"{target_pdb_id}.pt")
        
        raw_af2_candidates = find_file_robust(raw_af2_dir, f"*{current_uid}*.pdb", f"*{current_uid.upper()}*.pdb")
        raw_pdb_candidates = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.pdb", f"{target_pdb_id.upper()}.pdb")
        if not raw_pdb_candidates:
             raw_pdb_candidates = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.cif", f"{target_pdb_id.upper()}.cif")

        missing = []
        if not os.path.exists(pdb_pt_path): missing.append("PT File")
        if not raw_af2_candidates: missing.append("Raw AF2")
        if not raw_pdb_candidates: missing.append("Raw PDB")
        
        if missing:
            stats["skip"] += 1
            if debug_count < debug_limit:
                print(f"️Skip {basename}: Missing {missing}")
                debug_count += 1
            continue
            
        raw_af2_path = raw_af2_candidates[0]
        raw_pdb_path = raw_pdb_candidates[0]

        try:
            af2_data = torch.load(af2_pt_path, weights_only=False)
            pdb_data = torch.load(pdb_pt_path, weights_only=False)
            
            if pdb_data['y'].sum() == 0: 
                stats["skip"] += 1
                continue

            af2_seq, _ = get_sequence_and_indices(parser, raw_af2_path)
            pdb_seq, _ = get_sequence_and_indices(parser, raw_pdb_path)
            
            if not af2_seq or not pdb_seq:
                stats["failed"] += 1
                continue

            new_labels, rmsd, msg = robust_align_and_map(
                af2_data, pdb_data, af2_seq, pdb_seq, threshold=DIST_THRESHOLD
            )
            
            if msg.startswith("High RMSD"):
                stats["high_rmsd_drops"] += 1

            if new_labels is not None:
                n_pdb_pos = (pdb_data['y'] > 0.5).sum().item()
                n_af2_pos = new_labels.sum().item()
                
                stats["total_pdb_sites"] += int(n_pdb_pos)
                stats["mapped_af2_sites"] += int(n_af2_pos)
                
                if n_af2_pos > 0:
                    af2_data['y'] = new_labels
                    torch.save(af2_data, af2_pt_path)
                    stats["matched"] += 1
                else:
                    stats["failed"] += 1
            else:
                stats["failed"] += 1
                
        except Exception as e:
            print(f"Error {basename}: {e}")
            stats["failed"] += 1

    # === 计算派生指标 ===
    total_samples = stats['matched'] + stats['failed'] + stats['skip']
    success_rate = stats['matched'] / total_samples if total_samples > 0 else 0.0
    
    if stats["total_pdb_sites"] > 0:
        loss_rate = 1 - (stats["mapped_af2_sites"] / stats["total_pdb_sites"])
    else:
        loss_rate = 0.0

    # === 打印报告 ===
    print(f"\n Processing Complete.")
    print("\n" + "="*40)
    print(" Mapping Statistics & Bias Report")
    print("="*40)
    print(f"Alignment Success Rate : {success_rate:.2%}")
    print(f"High RMSD Exclusions : {stats['high_rmsd_drops']} proteins")
    print(f"Label Loss Rate      : {loss_rate:.2%}")
    print(f"Mapped/Total Sites   : {stats['mapped_af2_sites']}/{stats['total_pdb_sites']}")

    # === 保存 CSV ===
    report_data = {
        "Total_Samples": [total_samples],
        "Matched_Proteins": [stats['matched']],
        "Failed_Proteins": [stats['failed']],
        "Skipped_Proteins": [stats['skip']],
        "Success_Rate": [success_rate],
        "High_RMSD_Drops": [stats['high_rmsd_drops']],
        "Total_PDB_Sites": [stats['total_pdb_sites']],
        "Mapped_AF2_Sites": [stats['mapped_af2_sites']],
        "Label_Loss_Rate": [loss_rate]
    }
    
    df = pd.DataFrame(report_data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n Report saved to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
