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


# 忽略警告
warnings.filterwarnings("ignore")

# 添加 src 路径
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import StructureParser

# ==========================================
# 核心映射逻辑 (复用自 map_labels.py)
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

def get_sequence_and_indices(parser, raw_path):
    try:
        parsed = parser.parse_file_with_labels(raw_path)
        if not parsed: return None, None
        MAX_LEN = 1022
        seq = parsed['sequence']
        if len(seq) > MAX_LEN: seq = seq[:MAX_LEN]
        return seq, parsed
    except: return None, None

def calculate_mapping_stats(af2_data, pdb_data, af2_seq, pdb_seq, threshold):
    """
    执行对齐和映射，但只返回统计数据，不返回新标签。
    """
    # 1. 序列比对 (Biopython PairwiseAligner)
    aligner = PairwiseAligner()
    aligner.mode = 'global'
    aligner.match_score = 2; aligner.mismatch_score = -1     
    aligner.open_gap_score = -10.0; aligner.extend_gap_score = -0.5
    aligner.target_end_open_gap_score = 0.0; aligner.target_end_extend_gap_score = 0.0; 
    aligner.query_end_open_gap_score = 0.0; aligner.query_end_extend_gap_score = 0.0
    
    try:
        alignment = aligner.align(af2_seq, pdb_seq)[0]
    except:
        return "Alignment Failed", 0, 0, 0

    # 2. 提取索引
    af2_indices, pdb_indices = [], []
    aligned_af2, aligned_pdb = alignment.aligned[0], alignment.aligned[1]
    for (a_start, a_end), (p_start, p_end) in zip(aligned_af2, aligned_pdb):
        length = a_end - a_start
        for i in range(length):
            if af2_seq[a_start+i] == pdb_seq[p_start+i]:
                af2_indices.append(a_start+i); pdb_indices.append(p_start+i)

    if len(af2_indices) < 10: return "Too Few Aligned", 0, 0, 0

    # 3. 结构叠加
    fixed_coords = af2_data['pos'][af2_indices]
    moving_coords = pdb_data['pos'][pdb_indices]
    sup = Superimposer()
    sup.set_atoms(create_atoms(fixed_coords), create_atoms(moving_coords))
    
    if sup.rms > 10.0: return f"High RMSD ({sup.rms:.1f})", 0, 0, 0

    # 4. 计算距离统计
    pdb_pos_indices = torch.where(pdb_data['y'] > 0.5)[0]
    if len(pdb_pos_indices) == 0: return "No PDB Labels", 0, 0, 0
    
    # 变换 PDB 坐标到 AF2 空间
    pdb_positive_coords = pdb_data['pos'][pdb_pos_indices]
    rot, tran = sup.rotran
    rot = torch.from_numpy(rot).float()
    tran = torch.from_numpy(tran).float()
    transformed_pos = torch.matmul(pdb_positive_coords, rot.T) + tran
    
    # 计算距离矩阵 [Num_AF2, Num_PDB_Pos]
    dists = torch.cdist(af2_data['pos'], transformed_pos)
    
    # 指标 A: AF2 上有多少个点被标记 (Mapped Sites)
    min_dists_per_af2, _ = dists.min(dim=1)
    n_mapped_af2 = (min_dists_per_af2 < threshold).float().sum().item()
    
    # 指标 B: PDB 上有多少个点被找回 (Covered PDB Sites)
    min_dists_per_pdb, _ = dists.min(dim=0)
    n_covered_pdb = (min_dists_per_pdb < threshold).float().sum().item()
    
    return "Success", len(pdb_pos_indices), n_covered_pdb, n_mapped_af2

def find_file_robust(directory, pattern_lower, pattern_upper):
    c = glob.glob(os.path.join(directory, pattern_lower))
    if c: return c
    c = glob.glob(os.path.join(directory, pattern_upper))
    if c: return c
    return []

# ==========================================
# 主程序
# ==========================================
def main():
    # 配置
    THRESHOLDS = [3.0, 4.0, 5.0, 6.0, 8.0]
    processed_pdb_dir = "./data/processed_pdb"
    processed_af2_dir = "./data/processed_af2"
    raw_pdb_dir = "./data/raw_pdb" 
    raw_af2_dir = "./data/raw_af2" 
    mapping_file = "./pdb_uniprot_mapping.json"
    
    parser = StructureParser()
    if not os.path.exists(mapping_file): 
        print("Mapping file missing."); return
    with open(mapping_file, 'r') as f:
        uniprot_to_pdb = {v.lower(): k.lower() for k, v in json.load(f).items()}

    # 获取所有待处理文件列表
    af2_files = sorted(glob.glob(os.path.join(processed_af2_dir, "*.pt")))
    # 为了快速演示，如果是超大数据集，可以随机采样 50-100 个文件进行分析
    # af2_files = np.random.choice(af2_files, 100, replace=False) 
    
    print(f"🚀 Starting Sensitivity Analysis on {len(af2_files)} proteins...")
    print(f"🎯 Thresholds to test: {THRESHOLDS} Å")

    # 预加载数据以避免重复 IO (如果内存允许)
    # 这里为了稳妥，还是在循环内加载，或者更高效的做法是：加载一次数据，跑所有阈值
    
    dataset_cache = []
    print("📥 Pre-loading dataset for speed...")
    for af2_pt_path in tqdm(af2_files):
        basename = os.path.basename(af2_pt_path)
        target_pdb_id = None; current_uid = None
        for uid, pid in uniprot_to_pdb.items():
            if uid in basename.lower(): target_pdb_id = pid; current_uid = uid; break
        if not target_pdb_id: continue
        
        pdb_pt_path = os.path.join(processed_pdb_dir, f"{target_pdb_id}.pt")
        raw_af2_cands = find_file_robust(raw_af2_dir, f"*{current_uid}*.pdb", f"*{current_uid.upper()}*.pdb")
        raw_pdb_cands = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.pdb", f"{target_pdb_id.upper()}.pdb")
        if not raw_pdb_cands: raw_pdb_cands = find_file_robust(raw_pdb_dir, f"{target_pdb_id}.cif", f"{target_pdb_id.upper()}.cif")
        
        if os.path.exists(pdb_pt_path) and raw_af2_cands and raw_pdb_cands:
            try:
                # 仅解析序列一次
                af2_seq, _ = get_sequence_and_indices(parser, raw_af2_cands[0])
                pdb_seq, _ = get_sequence_and_indices(parser, raw_pdb_cands[0])
                if af2_seq and pdb_seq:
                    dataset_cache.append({
                        "af2_path": af2_pt_path, "pdb_path": pdb_pt_path,
                        "af2_seq": af2_seq, "pdb_seq": pdb_seq
                    })
            except: pass
    
    print(f"✅ Loaded {len(dataset_cache)} valid pairs. Running analysis loops...")

    results = []

    # === 大循环：遍历阈值 ===
    for dist_thresh in THRESHOLDS:
        stats = {
            "total_pdb_sites": 0,
            "covered_pdb_sites": 0,
            "mapped_af2_sites": 0,
            "valid_proteins": 0
        }
        
        print(f"\n👉 Testing Distance Threshold: {dist_thresh} Å")
        
        for item in tqdm(dataset_cache, leave=False):
            try:
                # 加载点云数据 (必须在这里加载，因为 PyTorch Tensor 占用显存/内存)
                af2_data = torch.load(item['af2_path'], weights_only=False)
                pdb_data = torch.load(item['pdb_path'], weights_only=False)
                
                status, n_pdb, n_covered, n_af2 = calculate_mapping_stats(
                    af2_data, pdb_data, item['af2_seq'], item['pdb_seq'], dist_thresh
                )
                
                if status == "Success" and n_pdb > 0:
                    stats["total_pdb_sites"] += n_pdb
                    stats["covered_pdb_sites"] += n_covered
                    stats["mapped_af2_sites"] += n_af2
                    stats["valid_proteins"] += 1
            except: pass
        
        # 计算该阈值的汇总指标
        if stats["total_pdb_sites"] > 0 and stats["covered_pdb_sites"] > 0:
            recall = stats["covered_pdb_sites"] / stats["total_pdb_sites"]
            loss_rate = 1.0 - recall
            inflation = stats["mapped_af2_sites"] / stats["covered_pdb_sites"]
        else:
            loss_rate = 1.0; inflation = 0.0
            
        results.append({
            "Threshold (Å)": dist_thresh,
            "Label Loss Rate": loss_rate,
            "Inflation Ratio": inflation,
            "Total PDB Sites": stats["total_pdb_sites"],
            "Covered PDB Sites": stats["covered_pdb_sites"],
            "Mapped AF2 Sites": stats["mapped_af2_sites"]
        })
        
        print(f"   [Result] Loss: {loss_rate:.2%} | Inflation: {inflation:.2f}x")

    # === 结果展示 ===
    df = pd.DataFrame(results)
    print("\n" + "="*60)
    print("🏆 SENSITIVITY ANALYSIS REPORT")
    print("="*60)
    print(df.to_string(index=False, float_format="%.4f"))
    
    # 保存
    df.to_csv("mapping_sensitivity_results.csv", index=False)
    print("\n📁 Saved to mapping_sensitivity_results.csv")
    
    # 简单的推荐逻辑
    # 我们希望 Loss 尽可能低 (<5%)，同时 Inflation 尽可能接近 1.0 (例如 <1.5)
    # 这是一个 Trade-off
    print("\n💡 Analysis:")
    for i, row in df.iterrows():
        t = row['Threshold (Å)']
        l = row['Label Loss Rate']
        inf = row['Inflation Ratio']
        
        eval_str = ""
        if l < 0.05 and inf < 1.5: eval_str = "🌟 Optimal Candidate"
        elif l < 0.10 and inf < 2.0: eval_str = "✅ Acceptable"
        elif l > 0.20: eval_str = "❌ High Loss (Missed Sites)"
        elif inf > 3.0: eval_str = "⚠️ High Inflation (False Positives)"
        
        print(f"   {t}Å : Loss={l*100:.1f}%, Inflation={inf:.2f}x  {eval_str}")

if __name__ == "__main__":
    main()
