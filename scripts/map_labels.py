# File: scripts/map_labels.py
import torch
import os
import glob
from tqdm import tqdm

def main():
    src_dir = "./data/processed_pdb"
    tgt_dir = "./data/processed_af2"
    
    # 1. 建立 PDB 索引
    print("Indexing PDB residues...")
    pdb_pool = {}
    pdb_files = glob.glob(os.path.join(src_dir, "*.pt"))
    
    for p in tqdm(pdb_files):
        try:
            data = torch.load(p, weights_only=False)
            # 假设文件名格式为 "1a2b.pt"，提取 "1a2b"
            pdb_id = os.path.basename(p).split(".")[0].lower()
            
            res_map = {rid: label.item() for rid, label in zip(data['residue_ids'], data['y'])}
            pdb_pool[pdb_id] = {
                "res_map": res_map,
                "res_set": set(data['residue_ids'])
            }
        except Exception as e: 
            print(f"Skipping {p}: {e}")
            continue

    # 2. 映射标签
    af2_files = glob.glob(os.path.join(tgt_dir, "*.pt"))
    print(f"Mapping labels for {len(af2_files)} AF2 files (No pLDDT filtering)...")
    
    match_count = 0
    skipped_count = 0
    
    for af2_path in tqdm(af2_files):
        try:
            af2_data = torch.load(af2_path, weights_only=False)
            
            # 尝试从文件名提取 PDB ID (假设格式如 "1a2b_af2.pt" 或 "1a2b.pt")
            # 这里简单尝试提取前4个字符作为ID，或者根据你的文件名规则修改
            current_filename = os.path.basename(af2_path).lower()
            best_pdb = None
            
            # --- 策略 A: 精确文件名匹配 (推荐) ---
            for pid in pdb_pool.keys():
                if pid in current_filename:
                    best_pdb = pid
                    break
            
            # --- 策略 B: 如果文件名不匹配，回退到重合度匹配 (不推荐但保留) ---
            if best_pdb is None:
                af2_res_set = set(af2_data['residue_ids'])
                max_overlap = 0
                for pdb_id, pdb_info in pdb_pool.items():
                    overlap = len(af2_res_set & pdb_info["res_set"])
                    if overlap > max_overlap:
                        max_overlap = overlap
                        best_pdb = pdb_id
                
                # 如果重合度太低，可能是误匹配
                if max_overlap < 10: 
                    best_pdb = None

            # 开始映射
            if best_pdb:
                res_map = pdb_pool[best_pdb]["res_map"]
                # 初始化为全 0
                new_y = torch.zeros(len(af2_data['residue_ids']))
                mapped_positives = 0
                
                for i, rid in enumerate(af2_data['residue_ids']):
                    if rid in res_map:
                        # [FIX] 直接映射真实标签，不再根据 pLDDT 过滤
                        # 结合位点是客观事实，不应受预测置信度影响
                        if res_map[rid] > 0.5: # 假设 >0.5 为正样本
                            new_y[i] = 1.0
                            mapped_positives += 1
                
                # 只有当成功映射了至少一个正样本时才保存
                # (或者你可以选择保留全负样本作为 Negative Control)
                if mapped_positives > 0:
                    af2_data['y'] = new_y
                    torch.save(af2_data, af2_path)
                    match_count += 1
                else:
                    # print(f"Matched {best_pdb} but no binding sites found.")
                    skipped_count += 1
            else:
                skipped_count += 1
                
        except Exception as e:
            print(f"Error processing {af2_path}: {e}")
            continue

    print(f"\n✅ Mapping complete.")
    print(f"   - Successfully labeled: {match_count} files")
    print(f"   - Skipped (No match/No binding sites): {skipped_count} files")

if __name__ == "__main__":
    main()