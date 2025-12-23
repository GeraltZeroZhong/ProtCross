import os
import glob
import requests
import concurrent.futures
import threading
import json

# 配置路径
SAVE_DIR = "./data/raw_af2"
RAW_PDB_DIR = "./data/raw_pdb"
MAPPING_FILE = "pdb_uniprot_mapping.json"

# 线程锁
print_lock = threading.Lock()
mapping_lock = threading.Lock()

# 全局映射字典
mapping_dict = {}

def safe_print(msg):
    with print_lock:
        print(msg)

def get_uniprot_ids(pdb_id):
    """通过 UniProt API 获取候选 UniProt ID 列表 (Top 3)"""
    url = f"https://rest.uniprot.org/uniprotkb/search?query=xref:pdb-{pdb_id}&fields=accession&size=3"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['results']:
            return [res['primaryAccession'] for res in data['results']]
    except Exception as e:
        safe_print(f"[-] 映射 PDB ID {pdb_id} 失败: {e}")
    return []

def download_af2(uniprot_id, output_dir):
    """下载 AlphaFold 结构，返回 True/False"""
    url = f"https://alphafold.ebi.ac.uk/files/AF-{uniprot_id}-F1-model_v6.pdb"
    filename = os.path.join(output_dir, f"AF-{uniprot_id}.pdb")

    # 如果文件已存在，也视为成功，直接返回 True
    if os.path.exists(filename):
        safe_print(f"[*] {uniprot_id} 已存在，跳过。")
        return True

    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            with open(filename, 'wb') as f:
                f.write(response.content)
            safe_print(f"[+] 下载成功: {filename}")
            return True
        elif response.status_code == 404:
            return False
        else:
            safe_print(f"[-] 下载错误 {uniprot_id}: HTTP {response.status_code}")
            return False
    except Exception as e:
        safe_print(f"[-] 下载异常 {uniprot_id}: {e}")
        return False

def process_single_pdb(pdb_file):
    """处理单个 PDB 文件并记录映射"""
    filename = os.path.basename(pdb_file)
    pdb_id = os.path.splitext(filename)[0].upper()
    
    # 1. 获取 ID 列表
    uniprot_ids = get_uniprot_ids(pdb_id)
    
    if not uniprot_ids:
        safe_print(f"[-] PDB {pdb_id} 未找到对应的 UniProt ID")
        return

    # 2. 遍历尝试下载
    success = False
    for uid in uniprot_ids:
        if download_af2(uid, SAVE_DIR):
            # 成功后记录映射关系
            with mapping_lock:
                mapping_dict[pdb_id] = uid
            success = True
            break
    
    if not success:
        safe_print(f"[-] PDB {pdb_id} (候选: {uniprot_ids}) 全部下载失败")

def main():
    if not os.path.exists(RAW_PDB_DIR):
        print(f"错误: 输入目录不存在 -> {RAW_PDB_DIR}")
        return

    os.makedirs(SAVE_DIR, exist_ok=True)
    
    search_pattern = os.path.join(RAW_PDB_DIR, "*.pdb")
    pdb_files = glob.glob(search_pattern)
    
    if not pdb_files:
        print(f"在 {RAW_PDB_DIR} 中未发现 .pdb 文件。")
        return

    print(f"发现 {len(pdb_files)} 个 PDB 文件，开始多线程处理...")

    # 多线程执行
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        executor.map(process_single_pdb, pdb_files)

    # 3. 保存映射字典
    print(f"正在保存映射字典到 {MAPPING_FILE} ...")
    try:
        with open(MAPPING_FILE, "w") as f:
            json.dump(mapping_dict, f, indent=4)
        print(f"成功保存 {len(mapping_dict)} 条映射记录。")
    except Exception as e:
        print(f"保存映射文件失败: {e}")

if __name__ == "__main__":
    main()