import torch
import os
import glob
import json
from tqdm import tqdm

def main():
    # === 配置路径 ===
    src_dir = "./data/processed_pdb"
    tgt_dir = "./data/processed_af2"
    mapping_file = "./pdb_uniprot_mapping.json"

    # === 1. 加载映射 ===
    print(f"Loading mapping from {mapping_file}...")
    with open(mapping_file, 'r') as f:
        raw_mapping = json.load(f)
    uniprot_to_pdb = {v.lower(): k.lower() for k, v in raw_mapping.items()}
    print(f"Loaded {len(uniprot_to_pdb)} mappings.")

    # === 2. 建立 PDB 索引 ===
    print("Indexing PDB residues...")
    pdb_pool = {}
    pdb_files = glob.glob(os.path.join(src_dir, "*.pt"))
    print(f"Found {len(pdb_files)} processed PDB files.")
    
    for p in pdb_files:
        try:
            pdb_id = os.path.basename(p).split(".")[0].lower()
            data = torch.load(p, weights_only=False)
            
            # 统计正样本
            positive_res_nums = set()
            for rid, label in zip(data['residue_ids'], data['y']):
                if label > 0.5:
                    # 尝试提取数字部分 'A_123' -> '123'
                    parts = rid.split('_')
                    if len(parts) > 1: positive_res_nums.add(parts[1])
                    else: positive_res_nums.add(rid) # fallback
            
            pdb_pool[pdb_id] = {
                "positive_nums": positive_res_nums,
                "has_sites": len(positive_res_nums) > 0,
                "all_sample_res": [r.split('_')[1] for r in data['residue_ids'][:5] if '_' in r] # 采样几个用于debug
            }
        except Exception as e:
            print(f"Warning: Failed to load {p}: {e}")

    # === 3. 诊断式映射 ===
    af2_files = glob.glob(os.path.join(tgt_dir, "*.pt"))
    print(f"Scanning {len(af2_files)} AF2 files...")

    debug_limit = 5 # 只详细打印前5个错误
    debug_count = 0
    match_count = 0

    for af2_path in tqdm(af2_files):
        basename = os.path.basename(af2_path)
        
        # 1. 查找目标 PDB
        target_pdb_id = None
        matched_uid = None
        for uid, pid in uniprot_to_pdb.items():
            if uid in basename.lower():
                target_pdb_id = pid
                matched_uid = uid
                break
        
        # === 失败原因诊断 1: 找不到映射 ===
        if not target_pdb_id:
            if debug_count < debug_limit:
                print(f"\n❌ [DEBUG] File: {basename}")
                print(f"   -> Reason: No matching UniProt ID found in filename.")
                debug_count += 1
            continue

        # === 失败原因诊断 2: PDB 文件不存在 ===
        if target_pdb_id not in pdb_pool:
            if debug_count < debug_limit:
                print(f"\n❌ [DEBUG] File: {basename} (Match: {matched_uid} -> {target_pdb_id})")
                print(f"   -> Reason: {target_pdb_id}.pt NOT found in {src_dir}.")
                print(f"      (Did preprocess_esm.py run for this PDB?)")
                debug_count += 1
            continue

        pdb_info = pdb_pool[target_pdb_id]

        # === 失败原因诊断 3: PDB 本身没有结合位点标签 ===
        if not pdb_info["has_sites"]:
            if debug_count < debug_limit:
                print(f"\n❌ [DEBUG] File: {basename} (Target PDB: {target_pdb_id})")
                print(f"   -> Reason: Corresponding PDB has NO positive labels (all y=0).")
                print(f"      (Ligand parsing failed or PDB has no ligand?)")
                debug_count += 1
            continue

        # === 失败原因诊断 4: 残基编号不匹配 ===
        af2_data = torch.load(af2_path, weights_only=False)
        mapped_positives = 0
        new_y = torch.zeros(len(af2_data['residue_ids']))
        
        af2_sample_res = [] # 用于debug
        
        for i, rid in enumerate(af2_data['residue_ids']):
            parts = rid.split('_')
            r_num = parts[1] if len(parts) > 1 else rid
            if i < 5: af2_sample_res.append(r_num) # 采样

            if r_num in pdb_info["positive_nums"]:
                new_y[i] = 1.0
                mapped_positives += 1
        
        if mapped_positives > 0:
            af2_data['y'] = new_y
            torch.save(af2_data, af2_path)
            match_count += 1
        else:
            if debug_count < debug_limit:
                print(f"\n❌ [DEBUG] File: {basename} (Target PDB: {target_pdb_id})")
                print(f"   -> Reason: ID Mismatch. No residues overlapped.")
                print(f"      AF2 Residues (Example): {af2_sample_res}")
                print(f"      PDB Positive Residues : {list(pdb_info['positive_nums'])[:5]}")
                print(f"      (Check if numbering systems are different)")
                debug_count += 1

    print(f"\n✅ Diagnostic Complete.")
    print(f"   Matches: {match_count}")

if __name__ == "__main__":
    main()
