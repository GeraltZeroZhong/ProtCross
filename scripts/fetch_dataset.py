import os
import requests
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def download_pdb(pdb_id, save_dir):
    pdb_id = pdb_id.lower()
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    save_path = os.path.join(save_dir, f"{pdb_id}.pdb")
    
    if os.path.exists(save_path):
        return True # Already exists
        
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            with open(save_path, "wb") as f:
                f.write(response.content)
            return True
        else:
            return False
    except:
        return False

def main():
    # ScanNet Test Set (部分) - 这些都是经过验证的有结合位点的蛋白
    # 我们先下载 50 个用于快速实验
    pdb_list = [
        "1a2k", "1a2y", "1a30", "1a4y", "1a8t", "1a9x", "1b34", "1b6c", "1buh", "1bun",
        "1c1y", "1c4z", "1cgi", "1ciq", "1cse", "1dfj", "1e6e", "1e96", "1eaw", "1eer",
        "1efn", "1f34", "1f51", "1f6m", "1fc2", "1fle", "1g2y", "1g3n", "1g4u", "1g4y",
        "1gc1", "1gla", "1gp2", "1gqw", "1grw", "1h9d", "1hcf", "1hcl", "1hcr", "1he1",
        "1he8", "1hlz", "1hqr", "1hv2", "1i2m", "1i4d", "1i4e", "1i85", "1ib1", "1ibr"
    ]
    
    save_dir = "./data/raw_pdb"
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"Downloading {len(pdb_list)} PDB files to {save_dir}...")
    
    # 多线程下载
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(lambda p: download_pdb(p, save_dir), pdb_list), total=len(pdb_list)))
        
    success = sum(results)
    print(f"Downloaded {success}/{len(pdb_list)} files successfully.")

if __name__ == "__main__":
    main()