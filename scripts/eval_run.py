import os
import sys
import warnings
import json
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch_geometric.loader import DataLoader
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from eval_dataset import EvalDataset
from eval_utils import (
    get_grouped_checkpoints,
    aggregate_roc_results,
    aggregate_pr_results,
    get_external_baselines,
    aggregate_bestf1_metrics,
    compute_bestf1_metrics,
)

warnings.filterwarnings("ignore")


def _add_src_to_path():
    # 添加项目根目录到 sys.path，保证 `import src...` 可用
    this_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(this_dir, ".."))

def _find_ckpt_in_folder(folder_path):
    target_ckpt = os.path.join(folder_path, "last.ckpt")
    if os.path.exists(target_ckpt):
        return target_ckpt
    cands = sorted([p for p in glob.glob(os.path.join(folder_path, "*.ckpt"))])
    return cands[0] if cands else None


def _scan_checkpoints_by_prefix(root_dir, prefixes):
    """
    仅供新增 ABCD-only 输出使用：按文件夹前缀扫描 ckpt。
    约定：saved_weights 子目录形如 'B_xxx' / 'C_xxx' 等。
    """
    groups = {p: [] for p in prefixes}
    if not os.path.exists(root_dir):
        return groups

    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    for folder_name in subdirs:
        for p in prefixes:
            if folder_name.startswith(f"{p}_"):
                ckpt = _find_ckpt_in_folder(os.path.join(root_dir, folder_name))
                if ckpt is not None:
                    groups[p].append(ckpt)
                break
    return groups


def _serialize_meanstd_dict(d):
    # d: {'best_f1': (mean,std), ...} 或 None
    if d is None:
        return None
    out = {}
    for k, (m, s) in d.items():
        out[k] = {"mean": float(m), "std": float(s)}
    return out



def main():
    _add_src_to_path()

    # --- Config ---
    WEIGHTS_DIR = "saved_weights"
    AF2_DATA = "data/processed_af2"
    PESTO_DIR = "data/PeSTo_results"
    P2RANK_DIR = "data/p2rank_results"

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) Load Dataset
    print(f"📦 Loading Dataset from {AF2_DATA}...")
    try:
        ds = EvalDataset(root=AF2_DATA, split="test")
        if len(ds) == 0:
            raise ValueError("Empty split")
    except Exception:
        print("   ⚠️ Test split empty or failed, trying ALL files.")
        ds = EvalDataset(root=AF2_DATA, split="all")

    loader = DataLoader(ds, batch_size=1, shuffle=False)

    # 2) Group Checkpoints
    groups = get_grouped_checkpoints(WEIGHTS_DIR)
    if not groups["A"] and not groups["D"]:
        print("❌ No checkpoints found.")
        return

    # 3) Calculate Stats
    stats = {}
    print_table_rows = []

    for name, label in [("A", "Baseline"), ("D", "ProtCross")]:
        if not groups[name]:
            continue

        print(f"\n📈 Processing {label} ({len(groups[name])} runs)...")

        # ROC Stats
        fpr, tpr, tpr_std, auc_m, auc_s = aggregate_roc_results(groups[name], loader, device)

        # PR Stats
        rec, prec, prec_std, ap_m, ap_s = aggregate_pr_results(groups[name], loader, device)

        # best-F1 Stats (threshold + confusion-matrix derived)
        bf1 = aggregate_bestf1_metrics(groups[name], loader, device)

        stats[name] = {
            "roc": (fpr, tpr, tpr_std, auc_m, auc_s),
            "pr": (rec, prec, prec_std, ap_m, ap_s),
            "bf1": bf1,
        }

        print_table_rows.append(
            {
                "Model": label,
                "AUC": f"{auc_m:.4f} ± {auc_s:.4f}",
                "AP": f"{ap_m:.4f} ± {ap_s:.4f}",
                "MaxF1": f"{bf1['best_f1'][0]:.4f} ± {bf1['best_f1'][1]:.4f}",
                "Thr@F1": f"{bf1['best_thr'][0]:.4f} ± {bf1['best_thr'][1]:.4f}",
                "Prec@F1": f"{bf1['prec'][0]:.4f} ± {bf1['prec'][1]:.4f}",
                "Rec@F1": f"{bf1['rec'][0]:.4f} ± {bf1['rec'][1]:.4f}",
                "TNR": f"{bf1['tnr'][0]:.4f} ± {bf1['tnr'][1]:.4f}",
                "BalAcc": f"{bf1['bal_acc'][0]:.4f} ± {bf1['bal_acc'][1]:.4f}",
            }
        )

    # 4) External Baselines
    y_ext, p_ext, p2_ext = get_external_baselines(loader, PESTO_DIR, P2RANK_DIR, AF2_DATA)

    if p_ext is not None:
        fpr_p, tpr_p, _ = roc_curve(y_ext, p_ext)
        p_auc = auc(fpr_p, tpr_p)
        p_ap = average_precision_score(y_ext, p_ext)
        m = compute_bestf1_metrics(y_ext, p_ext)

        print_table_rows.append(
            {
                "Model": "PeSTo",
                "AUC": f"{p_auc:.4f}",
                "AP": f"{p_ap:.4f}",
                "MaxF1": f"{m['best_f1']:.4f}",
                "Thr@F1": f"{m['best_thr']:.4f}",
                "Prec@F1": f"{m['prec']:.4f}",
                "Rec@F1": f"{m['rec']:.4f}",
                "TNR": f"{m['tnr']:.4f}",
                "BalAcc": f"{m['bal_acc']:.4f}",
            }
        )

    if p2_ext is not None:
        fpr_2, tpr_2, _ = roc_curve(y_ext, p2_ext)
        p2_auc = auc(fpr_2, tpr_2)
        p2_ap = average_precision_score(y_ext, p2_ext)
        m2 = compute_bestf1_metrics(y_ext, p2_ext)

        print_table_rows.append(
            {
                "Model": "P2Rank",
                "AUC": f"{p2_auc:.4f}",
                "AP": f"{p2_ap:.4f}",
                "MaxF1": f"{m2['best_f1']:.4f}",
                "Thr@F1": f"{m2['best_thr']:.4f}",
                "Prec@F1": f"{m2['prec']:.4f}",
                "Rec@F1": f"{m2['rec']:.4f}",
                "TNR": f"{m2['tnr']:.4f}",
                "BalAcc": f"{m2['bal_acc']:.4f}",
            }
        )

    # 5) Plotting
    plt.rcParams.update({"font.size": 14, "axes.linewidth": 1.5, "font.family": "sans-serif"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)

    # Colors (保持原始配色)
    c_A = "#7f8c8d"  # Grey (Baseline)
    c_D = "#d35400"  # Orange (ProtCross)
    c_P = "#27ae60"  # Green (PeSTo)
    c_P2 = "#2980b9"  # Blue (P2Rank)

    # --- Plot ROC (ax1) ---
    ax1.plot([0, 1], [0, 1], "k:", alpha=0.5)

    if "A" in stats:
        fpr, tpr, std, auc_m, auc_s = stats["A"]["roc"]
        ax1.plot(fpr, tpr, color=c_A, lw=2, label=f"Baseline (AUC={auc_m:.2f}±{auc_s:.2f})")
        ax1.fill_between(fpr, np.maximum(tpr - std, 0), np.minimum(tpr + std, 1), color=c_A, alpha=0.2)

    if "D" in stats:
        fpr, tpr, std, auc_m, auc_s = stats["D"]["roc"]
        ax1.plot(fpr, tpr, color=c_D, lw=3, label=f"ProtCross (AUC={auc_m:.2f}±{auc_s:.2f})")
        ax1.fill_between(fpr, np.maximum(tpr - std, 0), np.minimum(tpr + std, 1), color=c_D, alpha=0.2)

    if p_ext is not None:
        fpr_p, tpr_p, _ = roc_curve(y_ext, p_ext)
        ax1.plot(fpr_p, tpr_p, color=c_P, lw=2, ls="--", label=f"PeSTo (AUC={p_auc:.2f})")

    if p2_ext is not None:
        fpr_2, tpr_2, _ = roc_curve(y_ext, p2_ext)
        ax1.plot(fpr_2, tpr_2, color=c_P2, lw=2, ls="--", label=f"P2Rank (AUC={p2_auc:.2f})")

    ax1.set_xlabel("False Positive Rate")
    ax1.set_ylabel("True Positive Rate")
    ax1.set_title("ROC Curve")
    ax1.legend(loc="lower right", frameon=True, fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # --- Plot PR (ax2) ---
    if "A" in stats:
        rec, prec, std, ap_m, ap_s = stats["A"]["pr"]
        ax2.plot(rec, prec, color=c_A, lw=2, label=f"Baseline (AP={ap_m:.2f}±{ap_s:.2f})")
        ax2.fill_between(rec, np.maximum(prec - std, 0), np.minimum(prec + std, 1), color=c_A, alpha=0.2)

    if "D" in stats:
        rec, prec, std, ap_m, ap_s = stats["D"]["pr"]
        ax2.plot(rec, prec, color=c_D, lw=3, label=f"ProtCross (AP={ap_m:.2f}±{ap_s:.2f})")
        ax2.fill_between(rec, np.maximum(prec - std, 0), np.minimum(prec + std, 1), color=c_D, alpha=0.2)

    if p_ext is not None:
        p_p, r_p, _ = precision_recall_curve(y_ext, p_ext)
        ax2.plot(r_p, p_p, color=c_P, lw=2, ls="--", label=f"PeSTo (AP={p_ap:.2f})")

    if p2_ext is not None:
        p_2, r_2, _ = precision_recall_curve(y_ext, p2_ext)
        ax2.plot(r_2, p_2, color=c_P2, lw=2, ls="--", label=f"P2Rank (AP={p2_ap:.2f})")

    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("Precision-Recall Curve")
    ax2.legend(loc="lower left", frameon=True, fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_file = "aggregated_metrics_plot.png"
    plt.savefig(out_file)
    print(f"\n✅ Plot saved to {out_file}")

    # 6) Print Summary Tables
    print("\n" + "=" * 70)
    print(f"{'Model':<15} | {'AUC':<18} | {'AP':<18}")
    print("-" * 70)
    for row in print_table_rows:
        print(f"{row['Model']:<15} | {row['AUC']:<18} | {row['AP']:<18}")
    print("=" * 70)

    print("\n" + "=" * 120)
    print(
        f"{'Model':<15} | {'MaxF1':<18} | {'Thr@F1':<18} | {'Prec@F1':<18} | {'Rec@F1':<18} | {'TNR':<18} | {'BalAcc':<18}"
    )
    print("-" * 120)
    for row in print_table_rows:
        print(
            f"{row['Model']:<15} | "
            f"{row.get('MaxF1','-'):<18} | {row.get('Thr@F1','-'):<18} | "
            f"{row.get('Prec@F1','-'):<18} | {row.get('Rec@F1','-'):<18} | "
            f"{row.get('TNR','-'):<18} | {row.get('BalAcc','-'):<18}"
        )
    print("=" * 120 + "\n")

        # =========================================================
    # 7) EXTRA OUTPUT: ABCD-only (no PeSTo/P2Rank), new files
    # =========================================================
    # 扫描 B/C（A/D 复用原来的 groups）
    bc = _scan_checkpoints_by_prefix(WEIGHTS_DIR, prefixes=("B", "C"))
    groups_abcd = {"A": groups.get("A", []), "B": bc.get("B", []), "C": bc.get("C", []), "D": groups.get("D", [])}

    # 复用已算好的 A/D；只额外算 B/C
    stats_abcd = {}
    labels_abcd = {"A": "Baseline (Pure Geom)", "B": "+ ESM Embeddings", "C": "+ Standard DA (w/o pLDDT)", "D": "ProtCross"}

    for k in ["A", "B", "C", "D"]:
        if not groups_abcd[k]:
            continue

        if k in stats:  # A/D 已在旧流程算过
            stats_abcd[k] = stats[k]
            continue

        print(f"\n📈 [ABCD-only] Processing {labels_abcd[k]} ({len(groups_abcd[k])} runs).")
        fpr, tpr, tpr_std, auc_m, auc_s = aggregate_roc_results(groups_abcd[k], loader, device)
        rec, prec, prec_std, ap_m, ap_s = aggregate_pr_results(groups_abcd[k], loader, device)
        bf1 = aggregate_bestf1_metrics(groups_abcd[k], loader, device)

        stats_abcd[k] = {
            "roc": (fpr, tpr, tpr_std, auc_m, auc_s),
            "pr": (rec, prec, prec_std, ap_m, ap_s),
            "bf1": bf1,
        }

        # --- 构造 ABCD-only 的打印表（同旧格式）---
    print_table_rows_abcd = []
    for k in ["A", "B", "C", "D"]:
        if k not in stats_abcd:
            continue

        fpr, tpr, tpr_std, auc_m, auc_s = stats_abcd[k]["roc"]
        rec, prec, prec_std, ap_m, ap_s = stats_abcd[k]["pr"]
        bf1 = stats_abcd[k]["bf1"]

        row = {
            "Model": labels_abcd[k],
            "AUC": f"{auc_m:.4f} ± {auc_s:.4f}",
            "AP": f"{ap_m:.4f} ± {ap_s:.4f}",
            "MaxF1": f"{bf1['best_f1'][0]:.4f} ± {bf1['best_f1'][1]:.4f}" if bf1 else "-",
            "Thr@F1": f"{bf1['best_thr'][0]:.4f} ± {bf1['best_thr'][1]:.4f}" if bf1 else "-",
            "Prec@F1": f"{bf1['prec'][0]:.4f} ± {bf1['prec'][1]:.4f}" if bf1 else "-",
            "Rec@F1": f"{bf1['rec'][0]:.4f} ± {bf1['rec'][1]:.4f}" if bf1 else "-",
            "TNR": f"{bf1['tnr'][0]:.4f} ± {bf1['tnr'][1]:.4f}" if bf1 else "-",
            "BalAcc": f"{bf1['bal_acc'][0]:.4f} ± {bf1['bal_acc'][1]:.4f}" if bf1 else "-",
        }
        print_table_rows_abcd.append(row)

    # --- 打印：ABCD-only AUC/AP 表 ---
    print("\n" + "=" * 70)
    print("[ABCD-only] Summary (AUC / AP)")
    print(f"{'Model':<15} | {'AUC':<18} | {'AP':<18}")
    print("-" * 70)
    for row in print_table_rows_abcd:
        print(f"{row['Model']:<15} | {row['AUC']:<18} | {row['AP']:<18}")
    print("=" * 70)

    # --- 打印：ABCD-only MaxF1/Threshold/Precision/Recall/TNR/BalAcc 表 ---
    print("\n" + "=" * 120)
    print("[ABCD-only] Best-F1 Metrics")
    print(
        f"{'Model':<15} | {'MaxF1':<18} | {'Thr@F1':<18} | {'Prec@F1':<18} | "
        f"{'Rec@F1':<18} | {'TNR':<18} | {'BalAcc':<18}"
    )
    print("-" * 120)
    for row in print_table_rows_abcd:
        print(
            f"{row['Model']:<15} | {row['MaxF1']:<18} | {row['Thr@F1']:<18} | "
            f"{row['Prec@F1']:<18} | {row['Rec@F1']:<18} | {row['TNR']:<18} | {row['BalAcc']:<18}"
        )
    print("=" * 120 + "\n")

    
    # --- 保存 ABCD-only stats（JSON）---
    payload_abcd = {"models": {}}
    for k, v in stats_abcd.items():
        fpr, tpr, tpr_std, auc_m, auc_s = v["roc"]
        rec, prec, prec_std, ap_m, ap_s = v["pr"]
        payload_abcd["models"][labels_abcd[k]] = {
            "roc": {"fpr": fpr.tolist(), "tpr": tpr.tolist(), "tpr_std": tpr_std.tolist(),
                    "auc_mean": float(auc_m), "auc_std": float(auc_s)},
            "pr": {"recall": rec.tolist(), "precision": prec.tolist(), "prec_std": prec_std.tolist(),
                   "ap_mean": float(ap_m), "ap_std": float(ap_s)},
            "bf1": _serialize_meanstd_dict(v["bf1"]),
        }

    out_stat_abcd = "aggregated_metrics_stats_abcd.json"
    with open(out_stat_abcd, "w", encoding="utf-8") as f:
        json.dump(payload_abcd, f, indent=2)

    # --- 画 ABCD-only 新图 ---
    fig2, (bx1, bx2) = plt.subplots(1, 2, figsize=(16, 7), dpi=300)
    colors = {"A": "#7f8c8d", "B": "#8e44ad", "C": "#16a085", "D": "#d35400"}

    bx1.plot([0, 1], [0, 1], "k:", alpha=0.5)
    for k in ["A", "B", "C", "D"]:
        if k not in stats_abcd:
            continue
        fpr, tpr, std, auc_m, auc_s = stats_abcd[k]["roc"]
        bx1.plot(fpr, tpr, color=colors[k], lw=3 if k == "D" else 2,
                 label=f"{labels_abcd[k]} (AUC={auc_m:.2f}±{auc_s:.2f})")
        bx1.fill_between(fpr, np.maximum(tpr - std, 0), np.minimum(tpr + std, 1), color=colors[k], alpha=0.2)
    bx1.set_xlabel("False Positive Rate")
    bx1.set_ylabel("True Positive Rate")
    bx1.set_title("ROC Curve")
    bx1.legend(loc="lower right", frameon=True, fontsize=10)
    bx1.grid(True, linestyle="--", alpha=0.3)

    for k in ["A", "B", "C", "D"]:
        if k not in stats_abcd:
            continue
        rec, prec, std, ap_m, ap_s = stats_abcd[k]["pr"]
        bx2.plot(rec, prec, color=colors[k], lw=3 if k == "D" else 2,
                 label=f"{labels_abcd[k]} (AP={ap_m:.2f}±{ap_s:.2f})")
        bx2.fill_between(rec, np.maximum(prec - std, 0), np.minimum(prec + std, 1), color=colors[k], alpha=0.2)
    bx2.set_xlabel("Recall")
    bx2.set_ylabel("Precision")
    bx2.set_title("Precision-Recall Curve")
    bx2.legend(loc="lower left", frameon=True, fontsize=10)
    bx2.grid(True, linestyle="--", alpha=0.3)

    plt.tight_layout()
    out_fig_abcd = "aggregated_metrics_plot_abcd.png"
    plt.savefig(out_fig_abcd)
    plt.close(fig2)

    print(f"\n✅ [ABCD-only] Plot saved to {out_fig_abcd}")
    print(f"✅ [ABCD-only] Stats saved to {out_stat_abcd}")



if __name__ == "__main__":
    main()
