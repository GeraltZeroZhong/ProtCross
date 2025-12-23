import os
import requests
import glob
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def get_uniprot_ids_from_pdb(pdb_path):
    """从 PDB 文件头解析所有唯一的 UniProt ID"""
    uniprot_ids = set()
    try:
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("DBREF"):
                    parts = line.split()
                    # 寻找 UNP 或 UNIPROT 标记
                    for keyword in ["UNP", "UNIPROT"]:
                        if keyword in parts:
                            try:
                                idx = parts.index(keyword)
                                if idx + 1 < len(parts):
                                    uid = parts[idx+1]
                                    # 简单的格式检查
                                    if len(uid) >= 6:
                                        uniprot_ids.add(uid)
                            except ValueError:
                                continue
    except Exception as e:
        print(f"Error reading {pdb_path}: {e}")
        
    return list(uniprot_ids)

def download_af2(uniprot_id, save_dir):
    """根据 UniProt ID 下载 AlphaFold 预测结构 (v6)"""
    uniprot_id = uniprot_id.strip()
    
    # 使用用户指定的 v6 版本
    filename = f"AF-{uniprot_id}-F1-model_v6.pdb"
    save_path = os.path.join(save_dir, filename)
    
    if os.path.exists(save_path):
        return True 

    url = f"https://alphafold.ebi.ac.uk/files/{filename}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            return False
    except Exception as e:
        print(f"Exception for {uniprot_id}: {e}")
        return False

def main():
    save_dir = "./data/raw_af2"
    os.makedirs(save_dir, exist_ok=True)
    
    raw_pdb_dir = "./data/raw_pdb"
    if not os.path.exists(raw_pdb_dir):
        print(f"No PDB directory found at {raw_pdb_dir}")
        return
        
    pdb_files = glob.glob(os.path.join(raw_pdb_dir, "*.pdb"))
    
    print(f"Scanning {len(pdb_files)} PDBs for UniProt IDs...")
    
    all_uniprots = set()
    for p in pdb_files:
        uids = get_uniprot_ids_from_pdb(p)
        if uids:
            all_uniprots.update(uids)
            
    print(f"Found {len(all_uniprots)} unique UniProt IDs in PDB headers.")
    
    tasks = [(uid, save_dir) for uid in all_uniprots]
    
    success_count = 0
    if tasks:
        print(f"Attempting to download {len(tasks)} AF2 structures (v6)...")
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(tqdm(executor.map(lambda args: download_af2(*args), tasks), total=len(tasks)))
        success_count = sum(results)
        print(f"Downloaded {success_count} structures.")

    if success_count == 0:
        print("❌ Critical: Failed to download ANY AlphaFold structures. Please check if v6 files exist for these IDs.")
    else:
        print(f"✅ Total AF2 structures downloaded: {success_count}")

if __name__ == "__main__":
    main()