import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch_geometric.loader import DataLoader
import pytorch_lightning as pl

WEIGHTS_DIR = "saved_weights"
AF2_DATA_FOLDER = "data/processed_af2"
OUTPUT_CSV = "standard_iou_results.csv"
SEED = 42

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from evopoint_da.models.module import EvoPointDALitModule
from evopoint_da.data.dataset import EvoPointDataset


def get_checkpoints_from_dir(root_dir):
    if not os.path.exists(root_dir):
        print(f"❌ Error: Directory '{root_dir}' not found.")
        return []

    ckpt_list = []
    subdirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    print(f"📂 Found {len(subdirs)} experiment folders in '{root_dir}':")

    for folder_name in subdirs:
        folder_path = os.path.join(root_dir, folder_name)

        target_ckpt = os.path.join(folder_path, "last.ckpt")
        if not os.path.exists(target_ckpt):
            candidates = glob.glob(os.path.join(folder_path, "*.ckpt"))
            if candidates:
                target_ckpt = candidates[0]
            else:
                continue

        parts = folder_name.split('_')
        if len(parts) >= 2 and parts[-1].isdigit():
            seed = int(parts[-1])
            exp_id = "_".join(parts[:-1])
        else:
            seed = "Unknown"
            exp_id = folder_name

        ckpt_list.append({"path": target_ckpt, "exp_id": exp_id, "seed": seed})

    return ckpt_list


def _binary_confusion_counts(y_true: np.ndarray, y_pred: np.ndarray):
    # y_true/y_pred: 0/1 int arrays
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tn, fp, fn, tp


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den != 0 else 0.0


def evaluate_standard_iou_and_cm(ckpt_info, data_folder, thresh_start=0.10, thresh_end=0.95, thresh_step=0.01):
    ckpt_path = ckpt_info['path']
    seed = ckpt_info['seed']
    exp_id = ckpt_info['exp_id']

    print(f"\n📊 Processing {exp_id} | Seed {seed}...")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 加载模型
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path)
        model.eval()
        model.to(device)
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return None

    # 2) 加载数据
    try:
        dataset = EvoPointDataset(root=data_folder, split="test")
        print(f"   ✅ Loaded EvoPointDataset (split='test', n={len(dataset)})")
    except Exception as e:
        print(f"   ⚠️ 'test' split failed, trying 'train' split ({e})")
        dataset = EvoPointDataset(root=data_folder, split="train")
        print(f"   ✅ Loaded EvoPointDataset (split='train', n={len(dataset)})")

    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    all_labels = []
    all_scores = []

    # 3) 推理
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Inference", leave=False):
            batch = batch.to(device)

            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)
            logits = model.seg_head(feats)
            probs = torch.softmax(logits, dim=1)[:, 1]  # class-1 score

            all_labels.append(batch.y.detach().cpu().numpy())
            all_scores.append(probs.detach().cpu().numpy())

    y_true = np.concatenate(all_labels).astype(np.int32)
    y_scores = np.concatenate(all_scores).astype(np.float32)

    # 4) 扫阈值找最佳 IoU（class-1）
    best_iou = -1.0
    best_thresh = thresh_start

    thresholds = np.arange(thresh_start, thresh_end, thresh_step)
    for thresh in thresholds:
        y_pred = (y_scores > thresh).astype(np.int32)
        tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred)
        denom = tp + fp + fn
        iou = _safe_div(tp, denom)  # IoU for positive class

        if iou > best_iou:
            best_iou = iou
            best_thresh = float(thresh)

    # 5) 在最佳阈值处算混淆矩阵 + 衍生指标
    y_pred_best = (y_scores > best_thresh).astype(np.int32)
    tn, fp, fn, tp = _binary_confusion_counts(y_true, y_pred_best)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall)
    acc = _safe_div(tp + tn, tp + tn + fp + fn)

    print(f"   ✅ Best Standard IoU: {best_iou*100:.2f}% (Thresh={best_thresh:.2f})")
    print(f"   🧮 Confusion Matrix @best thresh: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

    return {
        "Experiment": exp_id,
        "Seed": seed,
        "Best_Standard_IoU": best_iou * 100,
        "Best_Threshold": best_thresh,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Accuracy": acc * 100,
        "Precision": precision * 100,
        "Recall": recall * 100,
        "F1": f1 * 100,
        "Checkpoint": ckpt_path
    }


def main():
    pl.seed_everything(SEED, workers=True)
    print(f"🔒 Global seed set to {SEED}")

    tasks = get_checkpoints_from_dir(WEIGHTS_DIR)
    if not tasks:
        print("No checkpoints found. Please ensure 'saved_weights' exists.")
        return

    results = []
    for task in tasks:
        metrics = evaluate_standard_iou_and_cm(task, AF2_DATA_FOLDER)
        if metrics:
            results.append(metrics)
            pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)

    if results:
        df = pd.DataFrame(results)

        print("\n" + "="*80)
        print("🏆 SUMMARY (Mean ± Std) + Confusion (Summed)")
        print("="*80)

        if "Experiment" in df.columns:
            groups = df.groupby("Experiment")
            summary_rows = []

            header = f"{'Experiment':<25} | {'IoU% (best)':<16} | {'F1%':<12} | {'TN FP FN TP (sum)':<22}"
            print(header)
            print("-" * len(header))

            for name, g in groups:
                m_iou, s_iou = g["Best_Standard_IoU"].mean(), g["Best_Standard_IoU"].std()
                s_iou = 0.0 if np.isnan(s_iou) else s_iou

                m_f1, s_f1 = g["F1"].mean(), g["F1"].std()
                s_f1 = 0.0 if np.isnan(s_f1) else s_f1

                tn_sum = int(g["TN"].sum())
                fp_sum = int(g["FP"].sum())
                fn_sum = int(g["FN"].sum())
                tp_sum = int(g["TP"].sum())

                print(f"{name:<25} | {m_iou:6.2f} ± {s_iou:5.2f} | {m_f1:6.2f} ± {s_f1:5.2f} | "
                      f"{tn_sum} {fp_sum} {fn_sum} {tp_sum}")

                summary_rows.append({
                    "Experiment": f"{name} (Mean)",
                    "Seed": "Aggregated",
                    "Best_Standard_IoU": f"{m_iou:.2f} ± {s_iou:.2f}",
                    "Best_Threshold": "-",
                    "TN": tn_sum, "FP": fp_sum, "FN": fn_sum, "TP": tp_sum,
                    "Accuracy": f"{g['Accuracy'].mean():.2f} ± {(0.0 if np.isnan(g['Accuracy'].std()) else g['Accuracy'].std()):.2f}",
                    "Precision": f"{g['Precision'].mean():.2f} ± {(0.0 if np.isnan(g['Precision'].std()) else g['Precision'].std()):.2f}",
                    "Recall": f"{g['Recall'].mean():.2f} ± {(0.0 if np.isnan(g['Recall'].std()) else g['Recall'].std()):.2f}",
                    "F1": f"{m_f1:.2f} ± {s_f1:.2f}",
                    "Checkpoint": "-"
                })

            df_final = pd.concat([df, pd.DataFrame(summary_rows)], ignore_index=True)
            df_final.to_csv(OUTPUT_CSV, index=False)

        print(f"\n📁 Results saved to: {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
