import os
import sys
import argparse
import glob
import torch
import numpy as np
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import ESMFeatureExtractor, StructureParser, PCAReducer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--pca_model_path", type=str, default="pca_128.pkl")
    parser.add_argument("--fit_pca", action="store_true")
    parser.add_argument("--pca_dim", type=int, default=128)
    parser.add_argument("--is_af2", action="store_true") # 关键标志
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    raw_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pdb"))) + sorted(glob.glob(os.path.join(args.data_dir, "*.cif")))
    
    if not raw_files:
        print(f"No files found in {args.data_dir}")
        return

    esm_extractor = ESMFeatureExtractor(model_name="esm2_t33_650M_UR50D", device=args.device)
    struct_parser = StructureParser()
    pca_reducer = PCAReducer(n_components=args.pca_dim)

    # Phase 1: PCA Fitting (通常只用 Source 数据 Fit)
    if args.fit_pca:
        print("Fitting PCA...")
        sample_files = np.random.choice(raw_files, min(len(raw_files), 500), replace=False)
        buffer = []
        for fpath in tqdm(sample_files):
            # Fit 阶段不需要标签
            parsed = struct_parser.parse_file_with_labels(fpath) # 这里复用 parse，即使没标签也没事
            if not parsed: 
                # 尝试用无标签 parser (如果 components 里有的话，或者 parse_file_with_labels 鲁棒性够好)
                # 假设 parse_file_with_labels 如果没配体返回全0 label，这是可以接受的
                continue
                
            if len(parsed['sequence']) > 1022: parsed['sequence'] = parsed['sequence'][:1022]
            try:
                buffer.append(esm_extractor.extract_residue_embeddings(parsed['sequence']))
            except: pass
        if buffer:
            pca_reducer.fit(buffer)
            pca_reducer.save(args.pca_model_path)
        else:
            return
    else:
        if os.path.exists(args.pca_model_path):
            pca_reducer.load(args.pca_model_path)
        else:
            print("PCA model not found.")
            return

    # Phase 2: Processing
    print(f"Processing files (Is AF2: {args.is_af2})...")
    
    for fpath in tqdm(raw_files):
        file_id = os.path.splitext(os.path.basename(fpath))[0]
        output_path = os.path.join(args.output_dir, f"{file_id}.pt")
        
        if os.path.exists(output_path): os.remove(output_path)

        try:
            # 关键分支：AF2 数据 vs PDB 数据
            if args.is_af2:
                # AF2 数据：不计算结合位点标签 (因为没有 HETATM)
                # 我们只需要提取坐标和 pLDDT
                # 这里我们调用 StructureParser 的基础解析功能 (需要确保 components.py 有基础 parse_file，或者 parse_file_with_labels 能处理无配体情况)
                # 假设 parse_file_with_labels 在无配体时返回全 0 labels，这对于 Target Domain 训练是完全没问题的
                # 因为在 training_step 中，对于 Target 数据，我们根本不计算 loss_seg，只计算 loss_domain
                parsed = struct_parser.parse_file_with_labels(fpath)
                # 对于 AF2，labels 全为 0 是符合预期的 (Dummy Labels)
            else:
                # PDB 数据：需要计算标签
                parsed = struct_parser.parse_file_with_labels(fpath)

            if not parsed: continue

            # Cropping
            MAX_LEN = 1022
            if len(parsed['sequence']) > MAX_LEN:
                parsed['sequence'] = parsed['sequence'][:MAX_LEN]
                parsed['coords'] = parsed['coords'][:MAX_LEN]
                parsed['plddts'] = parsed['plddts'][:MAX_LEN]
                parsed['residue_ids'] = parsed['residue_ids'][:MAX_LEN]
                parsed['labels'] = parsed['labels'][:MAX_LEN]

            raw_emb = esm_extractor.extract_residue_embeddings(parsed['sequence'])
            reduced_emb = pca_reducer.transform(raw_emb)
            
            # AF2 pLDDT 归一化
            if args.is_af2:
                plddt = torch.from_numpy(parsed['plddts']) / 100.0 
            else:
                plddt = torch.ones(len(parsed['coords'])) # PDB 默认为 1

            y = torch.from_numpy(parsed['labels']).float()

            data_dict = {
                "pos": torch.from_numpy(parsed['coords']), 
                "x": reduced_emb,                          
                "plddt": plddt.unsqueeze(1),               
                "y": y, # 对于 AF2，这是 Dummy (全0)；对于 PDB，这是 Real
                "residue_ids": parsed['residue_ids']       
            }
            torch.save(data_dict, output_path)

        except Exception as e:
            print(f"Failed {fpath}: {e}")

if __name__ == "__main__":
    main()