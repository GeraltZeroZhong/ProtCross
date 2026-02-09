import os
import sys
import glob
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix

# === 引入项目模块 ===
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from evopoint_da.models.module import EvoPointDALitModule
from evopoint_da.data.dataset import EvoPointDataset

# === 配置区域 ===
CKPT_C_DIR = "saved_weights/C_1"  # Standard DANN
CKPT_D_DIR = "saved_weights/D_1"  # ProtCross
DATA_DIR = "data/processed_af2"    # AF2 测试集
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------
# Adaptive threshold logic (same as test.py)
# -------------------------
def compute_best_iou(y_true, y_probs):
    """遍历 0.1-0.95 寻找最佳 IoU 和对应的阈值"""
    best_iou = 0.0
    best_thresh = 0.5

    for thresh in np.arange(0.1, 1.0, 0.05):
        y_pred = (y_probs > thresh).astype(int)

        intersection = np.sum((y_true == 1) & (y_pred == 1))
        union = np.sum((y_true == 1) | (y_pred == 1))

        if union == 0:
            iou = 0.0 if np.sum(y_true) > 0 else 1.0
        else:
            iou = intersection / union

        if iou > best_iou:
            best_iou = iou
            best_thresh = thresh

    return best_iou, best_thresh


def normalize_plddt_like_test(plddt_np):
    """test.py: if max>1.5 treat as 0-100 else already 0-1"""
    if plddt_np.max() > 1.5:
        return plddt_np / 100.0
    return plddt_np


def calculate_metrics_at_threshold(y_true, y_probs, plddt, threshold):
    """test.py-style stratified metrics (High/Med/Low) at a given threshold"""
    y_pred = (y_probs > threshold).astype(int)

    # normalize plddt to 0-1 (same logic as test.py)
    if plddt.max() > 1.5:
        p_norm = plddt / 100.0
    else:
        p_norm = plddt

    mask_high = p_norm >= 0.9
    mask_med = (p_norm >= 0.7) & (p_norm < 0.9)
    mask_low = p_norm < 0.7

    def calc_iou(mask):
        if mask.sum() == 0:
            return 0.0, 0
        yt = y_true[mask]
        yp = y_pred[mask]
        inter = np.sum((yt == 1) & (yp == 1))
        union = np.sum((yt == 1) | (yp == 1))
        return (inter / union * 100.0) if union > 0 else 0.0, int(mask.sum())

    def calc_fpr(mask):
        if mask.sum() == 0:
            return 0.0, 0
        yt = y_true[mask]
        yp = y_pred[mask]
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        negatives = tn + fp
        return (fp / negatives * 100.0) if negatives > 0 else 0.0, int(mask.sum())

    high_iou, high_tot = calc_iou(mask_high)
    med_iou, med_tot = calc_iou(mask_med)
    low_fpr, low_tot = calc_fpr(mask_low)

    return {
        "High_IoU": high_iou, "High_Total": high_tot,
        "Med_IoU": med_iou,   "Med_Total": med_tot,
        "Low_FPR": low_fpr,   "Low_Total": low_tot,
    }


# -------------------------
# Checkpoint / model / data
# -------------------------
def find_best_ckpt(folder):
    """自动查找最佳权重: 优先 best*.ckpt，否则任意 *.ckpt；按修改时间取最新"""
    if not os.path.exists(folder):
        print(f"❌ Error: Folder not found: {folder}")
        return None

    candidates = glob.glob(os.path.join(folder, "best*.ckpt"))
    if not candidates:
        candidates = glob.glob(os.path.join(folder, "*.ckpt"))

    if not candidates:
        print(f"❌ No checkpoints in {folder}")
        return None

    return max(candidates, key=os.path.getmtime)


def load_model(ckpt_path):
    print(f"🚀 Loading model from: {ckpt_path}")
    try:
        model = EvoPointDALitModule.load_from_checkpoint(ckpt_path, map_location=DEVICE)
        model.eval()
        model.to(DEVICE)
        return model
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return None


def collect_data(model, loader, model_name):
    """
    收集 residue-level 数据，并使用 test.py 的 adaptive 阈值搜索生成 Pred。
    逻辑：
      1) 推理收集 probs/labels/plddt + 判别器 P(Target)
      2) 若 use_plddt_weight=True，则 probs *= (normalize(plddt)>0.65)
      3) 在全体残基上扫阈值找 best_t
      4) 用 best_t 生成 Pred
    """
    plddt_list, ptarget_list, label_list, prob_list = [], [], [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Scanning {model_name}", leave=False):
            batch = batch.to(DEVICE)

            # 1) backbone feats
            src_x = batch.x if model.hparams.use_esm else None
            feats, _ = model.backbone(src_x, batch.pos, batch.batch)

            # 2) discriminator P(Target)
            if hasattr(model, "domain_disc") and model.domain_disc is not None:
                d_logits = model.domain_disc(feats)
                p_source = torch.sigmoid(d_logits)
                p_target = 1.0 - p_source
            else:
                p_target = torch.zeros_like(feats[:, 0]).unsqueeze(1) + 0.5

            # 3) seg probs
            seg_logits = model.seg_head(feats)
            probs = torch.softmax(seg_logits, dim=1)[:, 1]

            

            # 5) pLDDT output stored as 0-100 for plots/stats
            plddt = batch.plddt
            if plddt.max() <= 1.5:
                plddt = plddt * 100.0

            plddt_list.append(plddt.detach().cpu().numpy().flatten())
            ptarget_list.append(p_target.detach().cpu().numpy().flatten())
            label_list.append(batch.y.detach().cpu().numpy().flatten())
            prob_list.append(probs.detach().cpu().numpy().flatten())

    plddt_all = np.concatenate(plddt_list)
    ptarget_all = np.concatenate(ptarget_list)
    y_true = np.concatenate(label_list).astype(int)
    y_probs = np.concatenate(prob_list)

    best_iou, best_t = compute_best_iou(y_true, y_probs)
    y_pred = (y_probs > best_t).astype(float)

    print(f"[{model_name}] Adaptive best_t={best_t:.2f}, best_IoU={best_iou*100:.2f}%")

    return pd.DataFrame({
        "pLDDT": plddt_all,
        "Disc_Prob_Target": ptarget_all,
        "Label": y_true,
        "Prob": y_probs,
        "Pred": y_pred,
        "Best_Thresh": best_t,
        "Best_IoU": best_iou,
        "Model": model_name,
    })


# -------------------------
# Analysis / plots / stats
# -------------------------
def plot_analysis(df_c, df_d):
    """绘制机制诊断图表"""
    df_all = pd.concat([df_c, df_d], ignore_index=True)

    # pLDDT binning
    df_all["pLDDT_Bin"] = (df_all["pLDDT"] // 10 * 10).astype(int)

    plt.figure(figsize=(12, 5))

    # (1) Mechanism: discriminator confusion vs pLDDT
    plt.subplot(1, 2, 1)
    sns.lineplot(
        data=df_all,
        x="pLDDT_Bin",
        y="Disc_Prob_Target",
        hue="Model",
        style="Model",
        markers=True,
        dashes=False,
        palette=["gray", "red"],
    )
    plt.axhline(0.5, color="black", linestyle="--", alpha=0.3, label="Confusion Line (0.5)")
    plt.title("Mechanism: Discriminator Confusion vs pLDDT")
    plt.xlabel("pLDDT Score (Bin)")
    plt.ylabel("P(Target) Output by Discriminator")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # (2) Effect: High-confidence IoU (pLDDT>90) using adaptive Pred
    plt.subplot(1, 2, 2)

    high_conf_c = df_c[df_c["pLDDT"] > 90]
    high_conf_d = df_d[df_d["pLDDT"] > 90]

    def calc_iou_df(df):
        inter = ((df["Label"] == 1) & (df["Pred"] == 1)).sum()
        union = ((df["Label"] == 1) | (df["Pred"] == 1)).sum()
        return inter / union if union > 0 else 0.0

    iou_c = calc_iou_df(high_conf_c)
    iou_d = calc_iou_df(high_conf_d)

    bars = plt.bar(["Standard DANN", "ProtCross"], [iou_c, iou_d], color=["gray", "red"], alpha=0.8)
    plt.title("Effect: Segmentation IoU (pLDDT > 90)")
    plt.ylabel("IoU Score")
    plt.ylim(0, max(iou_c, iou_d) * 1.2 if max(iou_c, iou_d) > 0 else 1.0)

    for bar in bars:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 0.01,
            f"{yval:.3f}",
            ha="center",
            va="bottom",
            fontsize=12,
        )

    plt.tight_layout()
    out_file = "mechanism_diagnosis_plot.png"
    plt.savefig(out_file, dpi=300)
    print(f"\n✅ Plot saved to {out_file}")


def print_stats(df, model_name):
    """打印数值统计（含 test.py 风格的分层指标）"""
    print(f"\n📊 --- Stats for {model_name} ---")

    # Mechanism: low pLDDT confusion
    low_mask = df["pLDDT"] < 70
    mean_disc_low = df.loc[low_mask, "Disc_Prob_Target"].mean()
    print(f"  [Mechanism] Mean Disc Prob (Target) @ Low pLDDT (<70): {mean_disc_low:.4f}")
    if mean_disc_low > 0.7:
        print("    -> Discriminator easily identifies Target (Rejection).")
    elif 0.4 < mean_disc_low < 0.6:
        print("    -> Discriminator is Confused (Forced Alignment).")

    # Adaptive summary (overall)
    best_t = float(df["Best_Thresh"].iloc[0]) if "Best_Thresh" in df.columns else 0.5
    best_iou = float(df["Best_IoU"].iloc[0]) if "Best_IoU" in df.columns else float("nan")
    print(f"  [Adaptive]  Best_Threshold (overall): {best_t:.2f} | Overall_IoU: {best_iou*100:.2f}%")

    # Effect: keep your original high pLDDT IoU using Pred (adaptive)
    high_mask = df["pLDDT"] > 90
    y_true_h = df.loc[high_mask, "Label"]
    y_pred_h = df.loc[high_mask, "Pred"]
    inter = ((y_true_h == 1) & (y_pred_h == 1)).sum()
    union = ((y_true_h == 1) | (y_pred_h == 1)).sum()
    iou_high = inter / union if union > 0 else 0.0
    print(f"  [Effect]    Segmentation IoU @ High pLDDT (>90):       {iou_high:.4f}")

    # Effect: test.py stratified metrics at best_t using Prob
    y_true = df["Label"].to_numpy().astype(int)
    y_probs = df["Prob"].to_numpy()
    plddt = df["pLDDT"].to_numpy()
    m = calculate_metrics_at_threshold(y_true, y_probs, plddt, best_t)
    print(f"  [Effect]    High_IoU (pLDDT>=0.9): {m['High_IoU']:.2f}% (n={m['High_Total']})")
    print(f"  [Effect]    Med_IoU  (0.7-0.9):    {m['Med_IoU']:.2f}% (n={m['Med_Total']})")
    print(f"  [Effect]    Low_FPR  (pLDDT<0.7):  {m['Low_FPR']:.2f}% (n={m['Low_Total']})")


# -------------------------
# Main
# -------------------------
def main():
    print("🔬 Starting Stratified Mechanism Diagnosis...")

    ckpt_c = find_best_ckpt(CKPT_C_DIR)
    ckpt_d = find_best_ckpt(CKPT_D_DIR)

    if not ckpt_c or not ckpt_d:
        print("❌ Cannot find checkpoints. Please check 'saved_weights/' paths.")
        return

    # dataset
    try:
        dataset = EvoPointDataset(root=DATA_DIR, split="test")
    except Exception:
        print("⚠️ Test split not found, using train split for demo.")
        dataset = EvoPointDataset(root=DATA_DIR, split="train")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # models
    model_c = load_model(ckpt_c)
    if model_c is None:
        return
    df_c = collect_data(model_c, loader, "Standard DANN (Exp C)")

    model_d = load_model(ckpt_d)
    if model_d is None:
        return
    df_d = collect_data(model_d, loader, "ProtCross (Exp D)")

    # stats & plots
    print_stats(df_c, "Standard DANN")
    print_stats(df_d, "ProtCross")
    plot_analysis(df_c, df_d)

    # save
    df_c.to_csv("diagnosis_data_C.csv", index=False)
    df_d.to_csv("diagnosis_data_D.csv", index=False)
    print("✅ Raw data saved to .csv files.")


if __name__ == "__main__":
    main()
