import os
import glob
import logging

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

# ==========================================
# Helpers: Data Parsers
# ==========================================
def get_residue_ids_from_pdb(pdb_path):
    res_ids = []
    seen = set()
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    chain_id = line[21].strip()
                    if chain_id == "":
                        chain_id = "A"
                    res_seq = line[22:27].strip()
                    unique_id = f"{chain_id}_{res_seq}"
                    if unique_id not in seen:
                        res_ids.append(unique_id)
                        seen.add(unique_id)
        return res_ids
    except Exception:
        return []


def load_pesto_scores(pdb_path):
    scores = []
    seen = set()
    try:
        with open(pdb_path, "r") as f:
            for line in f:
                if line.startswith(("ATOM", "HETATM")):
                    res_seq = line[22:27].strip()
                    chain = line[21]
                    uid = f"{chain}_{res_seq}"
                    if uid not in seen:
                        try:
                            scores.append(float(line[60:66].strip()))
                            seen.add(uid)
                        except Exception:
                            continue
        return np.array(scores)
    except Exception:
        return None


def load_p2rank_scores(csv_path, target_len, residue_ids_map):
    scores = np.zeros(target_len)
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path, skipinitialspace=True)
        for _, row in df.iterrows():
            prob = row["probability"]
            res_str_list = str(row["residue_ids"]).strip().split()
            for res_str in res_str_list:
                target_id = res_str.strip()
                if target_id in residue_ids_map:
                    idx = residue_ids_map.index(target_id)
                    scores[idx] = max(scores[idx], prob)
        return scores
    except Exception:
        return np.zeros(target_len)


# ==========================================
# Checkpoint Scanner
# ==========================================
def get_grouped_checkpoints(root_dir):
    """
    扫描 saved_weights，将实验分为 'A' (Baseline) 和 'D' (ProtCross) 两组
    """
    if not os.path.exists(root_dir):
        print(f"❌ Error: {root_dir} not found.")
        return {"A": [], "D": []}

    groups = {"A": [], "D": []}
    subdirs = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )

    print(f"📂 Scanning {len(subdirs)} folders in '{root_dir}'...")

    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)

        # 找 ckpt
        target_ckpt = os.path.join(folder_path, "last.ckpt")
        if not os.path.exists(target_ckpt):
            cands = glob.glob(os.path.join(folder_path, "*.ckpt"))
            if cands:
                target_ckpt = cands[0]
            else:
                continue

        # 分组
        if folder_name.startswith("A_"):
            groups["A"].append(target_ckpt)
        elif folder_name.startswith("D_"):
            groups["D"].append(target_ckpt)

    print(f"   ✅ Found {len(groups['A'])} checkpoints for Baseline (A)")
    print(f"   ✅ Found {len(groups['D'])} checkpoints for ProtCross (D)")
    return groups


# ==========================================
# Inference Engine
# ==========================================
def run_inference(ckpt_path, loader, device="cuda"):
    """运行单个模型的推理"""
    try:
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)

        # 延迟导入，确保 sys.path 已在主脚本中设置好
        from src.evopoint_da.models.module import EvoPointDALitModule

        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"❌ Error loading {os.path.basename(ckpt_path)}: {e}")
        return None, None

    all_y = []
    all_p = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)

            # 兼容处理：只有 use_esm 才用 batch.x
            src_x = batch.x if (hasattr(model.hparams, "use_esm") and model.hparams.use_esm) else None

            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]

            all_y.append(batch.y.detach().cpu().numpy())
            all_p.append(probs.detach().cpu().numpy())

    return np.concatenate(all_y), np.concatenate(all_p)


# ==========================================
# Statistical Aggregator (ROC / PR)
# ==========================================
def aggregate_roc_results(ckpt_list, loader, device):
    """收集多个模型的 ROC 数据并计算 Mean ± Std"""
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    pbar = tqdm(ckpt_list, desc="Aggregating ROC", leave=False)
    for ckpt in pbar:
        y_true, y_score = run_inference(ckpt, loader, device)
        if y_true is None:
            continue

        fpr, tpr, _ = roc_curve(y_true, y_score)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # 去重 FPR 以避免插值错误
        _, unique_indices = np.unique(fpr, return_index=True)
        fpr_unique = fpr[unique_indices]
        tpr_unique = tpr[unique_indices]

        interp_tpr = np.interp(mean_fpr, fpr_unique, tpr_unique)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    return mean_fpr, mean_tpr, std_tpr, mean_auc, std_auc


def aggregate_pr_results(ckpt_list, loader, device):
    """收集多个模型的 PR 数据并计算 Mean ± Std（修复起点）"""
    precs = []
    aps = []
    mean_recall = np.linspace(0, 1, 100)

    pbar = tqdm(ckpt_list, desc="Aggregating PR", leave=False)
    for ckpt in pbar:
        y_true, y_score = run_inference(ckpt, loader, device)
        if y_true is None:
            continue

        precision, recall, _ = precision_recall_curve(y_true, y_score)
        ap = average_precision_score(y_true, y_score)
        aps.append(ap)

        # 1) 翻转为升序 recall
        recall_rev = recall[::-1]
        precision_rev = precision[::-1]

        # 2) 去重
        _, unique_indices = np.unique(recall_rev, return_index=True)
        recall_unique = recall_rev[unique_indices]
        precision_unique = precision_rev[unique_indices]

        # 3) 强制起点 (Recall=0 -> Precision=1)
        if len(recall_unique) > 0 and recall_unique[0] == 0:
            precision_unique[0] = 1.0

        interp_prec = np.interp(mean_recall, recall_unique, precision_unique)
        precs.append(interp_prec)

    mean_prec = np.mean(precs, axis=0)
    std_prec = np.std(precs, axis=0)
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)

    return mean_recall, mean_prec, std_prec, mean_ap, std_ap


# ==========================================
# best-F1 threshold + confusion-matrix metrics
# ==========================================
def compute_bestf1_metrics(y_true, y_score, eps=1e-12):
    """
    返回 dict：
      best_f1, best_thr,
      prec (Precision@bestF1),
      rec  (Recall/TPR@bestF1),
      tnr  (Specificity),
      bal_acc (Balanced Accuracy = (TPR+TNR)/2)
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    # thresholds 长度 = len(precision)-1
    if thresholds is None or len(thresholds) == 0:
        thr = 0.5
        y_pred = (y_score >= thr).astype(int)

        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))

        tpr = tp / (tp + fn + eps)
        tnr = tn / (tn + fp + eps)
        prec = tp / (tp + fp + eps)
        rec = tpr
        f1 = 2 * prec * rec / (prec + rec + eps)
        bal_acc = 0.5 * (tpr + tnr)

        return {
            "best_f1": float(f1),
            "best_thr": float(thr),
            "prec": float(prec),
            "rec": float(rec),
            "tnr": float(tnr),
            "bal_acc": float(bal_acc),
        }

    prec_t = precision[:-1]
    rec_t = recall[:-1]
    f1 = 2 * prec_t * rec_t / (prec_t + rec_t + eps)

    idx = int(np.nanargmax(f1))
    thr = float(thresholds[idx])
    best_f1 = float(f1[idx])
    best_prec = float(prec_t[idx])
    best_rec = float(rec_t[idx])

    y_pred = (y_score >= thr).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    tpr = tp / (tp + fn + eps)
    tnr = tn / (tn + fp + eps)
    bal_acc = 0.5 * (tpr + tnr)

    return {
        "best_f1": float(best_f1),
        "best_thr": float(thr),
        "prec": float(best_prec),
        "rec": float(best_rec),
        "tnr": float(tnr),
        "bal_acc": float(bal_acc),
    }


def aggregate_bestf1_metrics(ckpt_list, loader, device):
    """多 ckpt：对 best-F1 指标做 mean±std"""
    rows = []
    pbar = tqdm(ckpt_list, desc="Aggregating best-F1", leave=False)
    for ckpt in pbar:
        y_true, y_score = run_inference(ckpt, loader, device)
        if y_true is None:
            continue
        rows.append(compute_bestf1_metrics(y_true, y_score))

    if len(rows) == 0:
        return None

    def _ms(key):
        arr = np.array([r[key] for r in rows], dtype=float)
        return float(np.mean(arr)), float(np.std(arr))

    out = {}
    for k in ["best_f1", "best_thr", "prec", "rec", "tnr", "bal_acc"]:
        out[k] = _ms(k)
    return out


# ==========================================
# External Baselines
# ==========================================
def get_external_baselines(loader, pesto_dir, p2rank_dir, af2_dir):
    """收集 PeSTo 和 P2Rank 的结果"""
    if len(loader.dataset) == 0:
        return None, None, None

    sample = loader.dataset[0]
    if not hasattr(sample, "pdb_id"):
        print("⚠️ Warning: No 'pdb_id' in dataset. Skipping external baselines.")
        return None, None, None

    y_true_all = []
    pesto_scores = []
    p2rank_scores = []

    has_pesto = pesto_dir and os.path.exists(pesto_dir)
    has_p2rank = p2rank_dir and os.path.exists(p2rank_dir)

    print("   Running External Baselines...")
    for batch in tqdm(loader, desc="Ext Baselines", leave=False):
        # batch_size=1
        pid = batch.pdb_id[0] if isinstance(batch.pdb_id, list) else batch.pdb_id

        y_true = batch.y.detach().cpu().numpy()
        gt_len = len(y_true)
        y_true_all.append(y_true)

        # PeSTo
        if has_pesto:
            p_path = os.path.join(pesto_dir, f"{pid}_i3.pdb")
            if not os.path.exists(p_path):
                p_path = os.path.join(pesto_dir, f"{pid}.pdb")

            sc = load_pesto_scores(p_path)
            if sc is not None and len(sc) >= gt_len:
                pesto_scores.append(sc[:gt_len])
            else:
                pesto_scores.append(np.zeros(gt_len))

        # P2Rank
        if has_p2rank:
            raw_pdb = os.path.join(
                af2_dir.replace("processed_af2", "raw_af2"), f"{pid}.pdb"
            )
            if not os.path.exists(raw_pdb):
                raw_pdb = os.path.join(pesto_dir, f"{pid}.pdb")

            res_map = get_residue_ids_from_pdb(raw_pdb)
            csv_path = os.path.join(p2rank_dir, f"{pid}.pdb_predictions.csv")

            if res_map and len(res_map) >= gt_len:
                sc = load_p2rank_scores(csv_path, gt_len, res_map[:gt_len])
                p2rank_scores.append(sc)
            else:
                p2rank_scores.append(np.zeros(gt_len))

    y_cat = np.concatenate(y_true_all)
    p_cat = np.concatenate(pesto_scores) if pesto_scores else None
    p2_cat = np.concatenate(p2rank_scores) if p2rank_scores else None

    return y_cat, p_cat, p2_cat

def get_grouped_checkpoints_abcd(root_dir, exps=("A", "B", "C", "D")):
    """
    新增：扫描 saved_weights，按前缀 A_/B_/C_/D_ 分组。
    不影响旧的 get_grouped_checkpoints()。
    """
    if not os.path.exists(root_dir):
        print(f"❌ Error: {root_dir} not found.")
        return {k: [] for k in exps}

    groups = {k: [] for k in exps}
    subdirs = sorted(
        [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    )

    print(f"📂 Scanning {len(subdirs)} folders in '{root_dir}' (ABCD mode).")

    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)

        target_ckpt = os.path.join(folder_path, "last.ckpt")
        if not os.path.exists(target_ckpt):
            cands = glob.glob(os.path.join(folder_path, "*.ckpt"))
            if not cands:
                continue
            target_ckpt = cands[0]

        for k in exps:
            if folder_name.startswith(f"{k}_"):
                groups[k].append(target_ckpt)
                break

    for k in exps:
        print(f" Found {len(groups[k])} checkpoints for Experiment {k}")
    return groups

