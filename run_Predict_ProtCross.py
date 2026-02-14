import argparse
import os
import sys
import torch
import numpy as np
from Bio.PDB import PDBParser, PDBIO

# Add src to path to import project components
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from evopoint_da.data.components import ESMFeatureExtractor, StructureParser, PCAReducer
from evopoint_da.models.module import EvoPointDALitModule
from torch_geometric.data import Data

def get_args():
    parser = argparse.ArgumentParser(description="ProtCross single PDB binding site prediction script")
    
    # Required arguments
    parser.add_argument("--pdb_file", type=str, required=True, help="Input PDB file path")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Trained model checkpoint (.ckpt) path")
    parser.add_argument("--esm_weights", type=str, required=True, help="ESM-C model weights (.pth) path")
    parser.add_argument("--pca_path", type=str, required=True, help="Pretrained PCA model (.pkl) path")
    
    # Optional arguments
    parser.add_argument("--output_pdb", type=str, default=None, help="Output PDB path (predicted probabilities will be written to B-factor column)")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for binding site classification (default: 0.5)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Inference device")
    parser.add_argument("--pca_dim", type=int, default=128, help="PCA reduced dimension (must match training)")
    
    return parser.parse_args()

def main():
    args = get_args()
    
    # 1. Check if files exist
    for f in [args.pdb_file, args.ckpt_path, args.esm_weights, args.pca_path]:
        if not os.path.exists(f):
            print(f"Error: File not found -> {f}")
            return

    print(f"Start prediction: {args.pdb_file}")
    print(f"Device: {args.device}")

    # 2. Initialize components
    print("Loading feature extractor and PCA...")
    try:
        # Load ESM extractor
        esm_extractor = ESMFeatureExtractor(model_path=args.esm_weights, device=args.device)
        
        # Load PCA reducer
        pca_reducer = PCAReducer(n_components=args.pca_dim)
        pca_reducer.load(args.pca_path)
        
        # Structure parser
        struct_parser = StructureParser()
    except Exception as e:
        print(f"Component initialization failed: {e}")
        return

    # 3. Parse PDB and extract features
    print("Parsing structure and extracting features...")
    parsed = struct_parser.parse_file_with_labels(args.pdb_file)
    if not parsed:
        print("PDB parsing failed or no standard amino acids found.")
        return

    # Truncate sequence to match ESM context window limit (consistent with training)
    MAX_LEN = 1022
    if len(parsed['sequence']) > MAX_LEN:
        print(f"Warning: Sequence length ({len(parsed['sequence'])}) exceeds limit, truncating to {MAX_LEN}.")
        parsed['sequence'] = parsed['sequence'][:MAX_LEN]
        parsed['coords'] = parsed['coords'][:MAX_LEN]
        parsed['residue_ids'] = parsed['residue_ids'][:MAX_LEN]

    try:
        # Extract ESM embeddings
        raw_emb = esm_extractor.extract_residue_embeddings(parsed['sequence'])
        # Apply PCA reduction
        reduced_emb = pca_reducer.transform(raw_emb)
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return

    # 4. Build PyG Data object
    pos = torch.from_numpy(parsed['coords'])
    x = reduced_emb
    # Create batch index (all zeros for single graph)
    batch_idx = torch.zeros(pos.shape[0], dtype=torch.long)

    data = Data(x=x, pos=pos, batch=batch_idx)
    data = data.to(args.device)

    # 5. Load model and run inference
    print(f"Loading model: {os.path.basename(args.ckpt_path)}...")
    try:
        model = EvoPointDALitModule.load_from_checkpoint(args.ckpt_path, map_location=args.device)
        model.eval()
        model.to(args.device)
    except Exception as e:
        print(f"Model loading failed: {e}")
        return

    print("Running inference...")
    with torch.no_grad():
        # Forward pass
        # No plddt or targets required during inference
        feats, _ = model.backbone(data.x, data.pos, data.batch)
        logits = model.seg_head(feats)
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

    # 6. Output results
    print(f"DEBUG: Probability statistics -> Max: {probs.max():.4f}, Mean: {probs.mean():.4f}, Min: {probs.min():.4f}")
    res_ids = parsed['residue_ids']
    predicted_indices = np.where(probs > args.threshold)[0]
    
    print("\n" + "="*50)
    print(f"Prediction completed (Threshold={args.threshold})")
    print("="*50)
    
    print(f"Detected {len(predicted_indices)} potential binding residues:")
    results_list = []
    for idx in predicted_indices:
        rid = res_ids[idx]
        p = probs[idx]
        results_list.append(f"{rid}({p:.2f})")
    
    # Print first 50 results to avoid excessive output
    print(", ".join(results_list[:50]))
    if len(results_list) > 50:
        print(f"... (Total {len(results_list)} residues)")

    # 7. (Optional) Save PDB with B-factor values
    if args.output_pdb:
        print(f"\nSaving results to: {args.output_pdb}")
        try:
            # Reload PDB structure to modify B-factor
            bio_parser = PDBParser(QUIET=True)
            structure = bio_parser.get_structure("pred", args.pdb_file)
            
            # Build {residue_id: prob} mapping
            # StructureParser residue_ids format: "Chain_ResSeq" (e.g., "A_10")
            score_map = {rid: prob for rid, prob in zip(res_ids, probs)}
            
            for model in structure:
                for chain in model:
                    for res in chain:
                        # Construct ID for matching
                        rid = f"{chain.id}_{res.id[1]}"
                        
                        # Get predicted score (default 0.0 if not predicted)
                        score = score_map.get(rid, 0.0)
                        
                        # Set B-factor for all atoms in the residue
                        for atom in res:
                            atom.set_bfactor(score)
            
            io = PDBIO()
            io.set_structure(structure)
            io.save(args.output_pdb)
            print("Save successful. You can visualize probabilities in PyMOL using 'spectrum b'.")
            
        except Exception as e:
            print(f"Warning: Failed to save PDB: {e}")

if __name__ == "__main__":
    main()
