import os
import sys
import argparse
import glob
import torch
import numpy as np
from tqdm import tqdm

# 添加 src 到路径以导入组件
sys.path.append(os.path.join(os.path.dirname(__file__), "../src"))
from evopoint_da.data.components import ESMFeatureExtractor, StructureParser, PCAReducer

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="输入 PDB/CIF 文件夹路径")
    parser.add_argument("--output_dir", type=str, required=True, help="输出 .pt 文件夹路径")
    
    # === 修改: 适配 ESMC 模型名称参数 ===
    parser.add_argument("--model_name", type=str, default="esmc_600m", 
                        help="ESMC 模型名称 (如 esmc_600m) 或本地权重路径")
    
    # 修改默认 PCA 文件名，避免与旧版混淆
    parser.add_argument("--pca_model_path", type=str, default="pca_esmc_128.pkl")
    parser.add_argument("--fit_pca", action="store_true", help="是否重新拟合 PCA (处理 Source 数据时开启)")
    parser.add_argument("--pca_dim", type=int, default=128, help="PCA 降维后的目标维度")
    parser.add_argument("--is_af2", action="store_true", help="是否为 AF2 数据 (归一化 pLDDT，忽略标签)")
    parser.add_argument("--sample_ratio", type=float, default=0.1)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()

def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    
    raw_files = sorted(glob.glob(os.path.join(args.data_dir, "*.pdb"))) + \
                sorted(glob.glob(os.path.join(args.data_dir, "*.cif")))
    
    if not raw_files:
        print(f"No files found in {args.data_dir}")
        return

    # 初始化 ESMC 特征提取器
    # 注意: 这里传入 model_name

    esm_extractor = ESMFeatureExtractor(model_path=args.model_name, device=args.device)
    struct_parser = StructureParser()
    pca_reducer = PCAReducer(n_components=args.pca_dim)

    # Phase 1: PCA Fitting (通常只在处理 Source PDB 数据时运行一次)
    if args.fit_pca:
        print(f"Fitting PCA (Target Dim: {args.pca_dim})...")
        # 随机采样 500 个文件用于拟合 PCA
        sample_files = np.random.choice(raw_files, min(len(raw_files), 500), replace=False)
        buffer = []
        for fpath in tqdm(sample_files, desc="PCA Sampling"):
            parsed = struct_parser.parse_file_with_labels(fpath)
            if not parsed: continue
            
            # 截断过长序列
            if len(parsed['sequence']) > 1022: parsed['sequence'] = parsed['sequence'][:1022]
            
            try:
                # 使用 ESMC 提取特征
                emb = esm_extractor.extract_residue_embeddings(parsed['sequence'])
                buffer.append(emb)
            except Exception as e:
                print(f"PCA Fit Skip {fpath}: {e}")
                pass
                
        if buffer:
            pca_reducer.fit(buffer)
            pca_reducer.save(args.pca_model_path)
            print(f"PCA model saved to {args.pca_model_path}")
        else:
            print("Error: No valid data extracted for PCA fitting.")
            return
    else:
        if os.path.exists(args.pca_model_path):
            pca_reducer.load(args.pca_model_path)
            print(f"Loaded PCA model from {args.pca_model_path}")
        else:
            print(f"PCA model not found at {args.pca_model_path}. Please run with --fit_pca first.")
            return

    # Phase 2: Processing All Files
    print(f"Processing {len(raw_files)} files (Is AF2: {args.is_af2})...")
    
    success_count = 0
    for fpath in tqdm(raw_files, desc="Processing"):
        file_id = os.path.splitext(os.path.basename(fpath))[0]
        output_path = os.path.join(args.output_dir, f"{file_id}.pt")
        
        # 如果需重新处理，先删除旧文件
        if os.path.exists(output_path): 
            try: os.remove(output_path)
            except: pass

        try:
            parsed = struct_parser.parse_file_with_labels(fpath)
            if not parsed: continue

            # Cropping (Match ESMC context window)
            MAX_LEN = 1022
            if len(parsed['sequence']) > MAX_LEN:
                parsed['sequence'] = parsed['sequence'][:MAX_LEN]
                parsed['coords'] = parsed['coords'][:MAX_LEN]
                parsed['plddts'] = parsed['plddts'][:MAX_LEN]
                parsed['residue_ids'] = parsed['residue_ids'][:MAX_LEN]
                parsed['labels'] = parsed['labels'][:MAX_LEN]

            # 1. Extract (ESMC -> 1152 dim)
            raw_emb = esm_extractor.extract_residue_embeddings(parsed['sequence'])
            
            # 2. Reduce (PCA -> 128 dim)
            reduced_emb = pca_reducer.transform(raw_emb)
            
            # 3. Handle pLDDT
            if args.is_af2:
                plddt = torch.from_numpy(parsed['plddts']) / 100.0 
            else:
                plddt = torch.ones(len(parsed['coords'])) # PDB default 1.0

            # 4. Handle Labels
            y = torch.from_numpy(parsed['labels']).float()

            data_dict = {
                "pos": torch.from_numpy(parsed['coords']), 
                "x": reduced_emb,                          
                "plddt": plddt.unsqueeze(1),               
                "y": y, 
                "residue_ids": parsed['residue_ids']       
            }
            torch.save(data_dict, output_path)
            success_count += 1

        except Exception as e:
            print(f"Failed {fpath}: {e}")
            
    print(f"Done. Processed {success_count}/{len(raw_files)} files.")

if __name__ == "__main__":
    main()
