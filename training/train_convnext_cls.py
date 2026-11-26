#!/usr/bin/env python3
"""
Train ConvNeXt-Tiny (`convnext_tiny.fb_in22k_ft_in1k_384`) on storm-centered HDF5
datasets under `data/processed/cls_h5_448/` with a coupled dual-head objective:
1. Primary head: Ordinal Class-Distance Weighted Cross-Entropy + head consistency (7 bins)
2. Auxiliary head: Smooth L1 regression on continuous sustained wind speed (knots)

Includes Model EMA, warmup+cosine scheduling, dihedral + radiometric augmentation,
4-fold rotational Test-Time Augmentation (TTA), early stopping, and diagnostic plots.
"""

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Headless backend for server/cluster environments
import matplotlib.pyplot as plt
import numpy as np
import timm
from timm.utils import ModelEmaV2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

# Ensure project root is importable so `src.cls_dataset` resolves cleanly
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.cls_dataset import CycloneH5Dataset, GPUCycloneTensorBuilder

DEFAULT_CLASS_NAMES = ["TD", "TS", "Cat 1", "Cat 2", "Cat 3", "Cat 4", "Cat 5"]
# Saffir-Simpson bin midpoints (knots) corresponding to TD, TS, Cat 1..5
CLASS_WIND_CENTERS_KTS = [25.0, 48.0, 73.0, 89.0, 104.0, 124.0, 148.0]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def apply_cyclone_augmentations(
    x: torch.Tensor, radiometric_jitter: float = 0.05
) -> torch.Tensor:
    """
    Apply exact dihedral symmetries (90-deg rotations + flips) and mild
    channel-wise gain/offset jitter to reduce overfitting on brightness temperatures.
    """
    k = random.randint(0, 3)
    if k > 0:
        x = torch.rot90(x, k=k, dims=(-2, -1))
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1])
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-2])

    if radiometric_jitter > 0.0:
        b, c, _, _ = x.shape
        scale = 1.0 + torch.empty((b, c, 1, 1), device=x.device, dtype=x.dtype).uniform_(
            -radiometric_jitter, radiometric_jitter
        )
        shift = torch.empty((b, c, 1, 1), device=x.device, dtype=x.dtype).uniform_(
            -radiometric_jitter, radiometric_jitter
        )
        x = x * scale + shift
    return x


class DualHeadConvNeXt(nn.Module):
    """Wraps a `timm` backbone with a 7-class intensity head and wind-speed head."""

    def __init__(
        self,
        model_name: str = "convnext_tiny.fb_in22k_ft_in1k_384",
        num_classes: int = 7,
        pretrained: bool = True,
        drop_path_rate: float = 0.3,
        head_dropout: float = 0.25,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3,
            num_classes=0,
            drop_path_rate=drop_path_rate,
        )
        feat_dim = self.backbone.num_features
        self.head_drop = nn.Dropout(head_dropout)
        self.cls_head = nn.Linear(feat_dim, num_classes)
        self.wind_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 4),
            nn.GELU(),
            nn.Linear(feat_dim // 4, 1),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.head_drop(self.backbone(x))
        logits = self.cls_head(feats)
        wind_pred_kts = F.softplus(self.wind_head(feats).squeeze(-1)) * 100.0
        return logits, wind_pred_kts


class OrdinalCycloneLoss(nn.Module):
    """
    Coupled classification loss combining:
    1. Label-smoothed Cross-Entropy
    2. Class-Distance Weighted penalty (heavily penalizes far jumps like Cat 4 -> TD)
    3. Head-consistency loss aligning expected category wind speed with wind_head
    """

    def __init__(
        self,
        num_classes: int = 7,
        weight: Optional[torch.Tensor] = None,
        label_smoothing: float = 0.05,
        ordinal_alpha: float = 1.5,
        ordinal_weight: float = 0.35,
        consistency_weight: float = 0.15,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.ordinal_weight = ordinal_weight
        self.consistency_weight = consistency_weight

        idx = torch.arange(num_classes, dtype=torch.float32)
        dist_matrix = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs().pow(ordinal_alpha)
        self.register_buffer("dist_matrix", dist_matrix)

        centers = CLASS_WIND_CENTERS_KTS[:num_classes]
        if len(centers) < num_classes:
            centers = [25.0 + 20.0 * i for i in range(num_classes)]
        self.register_buffer("wind_centers", torch.tensor(centers, dtype=torch.float32))

    def forward(
        self, logits: torch.Tensor, targets: torch.Tensor, wind_pred_kts: torch.Tensor
    ) -> torch.Tensor:
        loss_ce = self.ce(logits, targets)

        probs = F.softmax(logits.float(), dim=-1)
        sample_dists = self.dist_matrix[targets]  # (B, C)
        loss_ord = (probs * sample_dists).sum(dim=-1).mean()

        expected_wind = (probs * self.wind_centers.unsqueeze(0)).sum(dim=-1)
        loss_cons = F.smooth_l1_loss(
            expected_wind / 100.0, wind_pred_kts.float().detach() / 100.0, beta=0.1
        )

        return loss_ce + self.ordinal_weight * loss_ord + self.consistency_weight * loss_cons


def predict_with_4x_tta(
    model: nn.Module, x: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average logits and wind speed predictions across 0, 90, 180, and 270 degree rotations."""
    logits_sum = torch.zeros(0, device=x.device)
    wind_sum = torch.zeros(0, device=x.device)
    for k in range(4):
        x_rot = torch.rot90(x, k=k, dims=(-2, -1)) if k > 0 else x
        logits, wind_pred = model(x_rot)
        logits_sum = logits if k == 0 else (logits_sum + logits)
        wind_sum = wind_pred if k == 0 else (wind_sum + wind_pred)
    return logits_sum / 4.0, wind_sum / 4.0


def compute_classification_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> Dict[str, Any]:
    """Compute confusion matrix, per-class precision/recall/F1, and ordinal error."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    per_class = []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = int(cm[c, :].sum())
        prec = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        per_class.append(
            {"class_id": c, "precision": prec, "recall": rec, "f1": f1, "support": support}
        )

    macro_f1 = float(np.mean([m["f1"] for m in per_class]))
    mean_cat_err = float(np.mean(np.abs(y_pred - y_true))) if len(y_true) > 0 else 0.0
    return {
        "confusion_matrix": cm,
        "per_class": per_class,
        "macro_f1": macro_f1,
        "mean_cat_err": mean_cat_err,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    tensor_builder: GPUCycloneTensorBuilder,
    loader: DataLoader,
    cls_criterion: OrdinalCycloneLoss,
    wind_criterion: nn.Module,
    wind_weight: float,
    device: torch.device,
    use_amp: bool,
    num_classes: int = 7,
    use_tta: bool = True,
    collect_samples: bool = False,
    max_vis_samples: int = 16,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    model.eval()
    total_loss = 0.0
    total_cls_loss = 0.0
    total_wind_loss = 0.0
    total_samples = 0

    all_true_cls: List[np.ndarray] = []
    all_pred_cls: List[np.ndarray] = []
    all_probs: List[np.ndarray] = []
    all_true_wind: List[np.ndarray] = []
    all_pred_wind: List[np.ndarray] = []
    vis_crops: List[np.ndarray] = []

    for tb_x100, class_ids, wind_kts in loader:
        tb_x100 = tb_x100.to(device, non_blocking=True)
        class_ids = class_ids.to(device, non_blocking=True)
        wind_kts = wind_kts.to(device, non_blocking=True)

        x = tensor_builder(tb_x100)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            if use_tta:
                logits, wind_pred = predict_with_4x_tta(model, x)
            else:
                logits, wind_pred = model(x)
            loss_cls = cls_criterion(logits, class_ids, wind_pred)
            loss_wind = wind_criterion(wind_pred / 100.0, wind_kts / 100.0)
            loss = loss_cls + wind_weight * loss_wind

        bs = class_ids.size(0)
        total_loss += float(loss.item()) * bs
        total_cls_loss += float(loss_cls.item()) * bs
        total_wind_loss += float(loss_wind.item()) * bs
        total_samples += bs

        probs = torch.softmax(logits.float(), dim=-1)
        preds = logits.argmax(dim=-1)

        all_true_cls.append(class_ids.cpu().numpy())
        all_pred_cls.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_true_wind.append(wind_kts.float().cpu().numpy())
        all_pred_wind.append(wind_pred.float().cpu().numpy())

        if collect_samples and sum(len(c) for c in vis_crops) < max_vis_samples:
            needed = max_vis_samples - sum(len(c) for c in vis_crops)
            vis_crops.append(x[:needed, 0].detach().cpu().numpy())

    y_true_cls = np.concatenate(all_true_cls)
    y_pred_cls = np.concatenate(all_pred_cls)
    y_probs = np.concatenate(all_probs)
    y_true_wind = np.concatenate(all_true_wind)
    y_pred_wind = np.concatenate(all_pred_wind)

    denom = max(total_samples, 1)
    acc = float(np.mean(y_pred_cls == y_true_cls))
    within_one_cat = float(np.mean(np.abs(y_pred_cls - y_true_cls) <= 1))

    err = y_pred_wind - y_true_wind
    wind_rmse = float(np.sqrt(np.mean(err ** 2)))
    wind_mae = float(np.mean(np.abs(err)))
    wind_bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true_wind - np.mean(y_true_wind)) ** 2)) + 1e-8
    wind_r2 = 1.0 - (ss_res / ss_tot)

    cls_details = compute_classification_metrics(y_true_cls, y_pred_cls, num_classes)

    for c_info in cls_details["per_class"]:
        cid = c_info["class_id"]
        mask = y_true_cls == cid
        c_info["wind_mae_kts"] = float(np.mean(np.abs(err[mask]))) if np.any(mask) else 0.0

    metrics = {
        "loss": total_loss / denom,
        "cls_loss": total_cls_loss / denom,
        "wind_loss": total_wind_loss / denom,
        "acc": acc,
        "macro_f1": cls_details["macro_f1"],
        "within_1_cat_acc": within_one_cat,
        "mean_cat_err": cls_details["mean_cat_err"],
        "wind_rmse_kts": wind_rmse,
        "wind_mae_kts": wind_mae,
        "wind_bias_kts": wind_bias,
        "wind_r2": wind_r2,
    }

    artifacts = {
        "y_true_cls": y_true_cls,
        "y_pred_cls": y_pred_cls,
        "y_probs": y_probs,
        "y_true_wind": y_true_wind,
        "y_pred_wind": y_pred_wind,
        "confusion_matrix": cls_details["confusion_matrix"],
        "per_class": cls_details["per_class"],
        "vis_crops": np.concatenate(vis_crops, axis=0) if vis_crops else None,
    }
    return metrics, artifacts


# ---------------------------------------------------------------------------
# Detailed Visualization Suite
# ---------------------------------------------------------------------------


def save_training_visualizations(
    history: List[Dict[str, Any]],
    artifacts: Dict[str, Any],
    class_names: List[str],
    save_dir: Path,
    prefix: str = "val",
) -> None:
    """Generate diagnostic plots for training curves, classification, and wind regression."""
    plots_dir = save_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})

    # 1. Learning Curves (2x2)
    if len(history) > 0:
        epochs = [h["epoch"] for h in history]
        best_idx = int(np.argmax([h["score"] for h in history]))
        best_ep = epochs[best_idx]

        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

        ax = axes[0, 0]
        ax.plot(epochs, [h["train_loss"] for h in history], label="Train Total Loss", lw=2)
        ax.plot(epochs, [h["loss"] for h in history], label="Val Total Loss", lw=2)
        ax.plot(epochs, [h["cls_loss"] for h in history], "--", label="Val Cls Loss", alpha=0.8)
        ax.axvline(best_ep, color="gray", linestyle=":", label=f"Best Epoch ({best_ep})")
        ax.set_title("Loss Trajectory")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()

        ax = axes[0, 1]
        ax.plot(epochs, [h["acc"] * 100 for h in history], label="Exact Acc (%)", lw=2)
        ax.plot(epochs, [h["within_1_cat_acc"] * 100 for h in history], label="±1 Cat Acc (%)", lw=2)
        ax.plot(epochs, [h["macro_f1"] * 100 for h in history], "--", label="Macro F1 (%)", lw=1.8)
        ax.axvline(best_ep, color="gray", linestyle=":")
        ax.set_title("Intensity Classification Performance")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Percentage (%)")
        ax.legend()

        ax = axes[1, 0]
        ax.plot(epochs, [h["wind_mae_kts"] for h in history], label="Wind MAE (kt)", lw=2, color="#d95f02")
        ax.plot(epochs, [h["wind_rmse_kts"] for h in history], label="Wind RMSE (kt)", lw=2, color="#7570b3")
        ax.axvline(best_ep, color="gray", linestyle=":")
        ax.set_title("Sustained Wind Speed Error")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Error (knots)")
        ax.legend()

        ax = axes[1, 1]
        l1 = ax.plot(epochs, [h["score"] for h in history], label="Composite Score", lw=2, color="#1b9e77")
        ax.axvline(best_ep, color="gray", linestyle=":")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Composite Score", color="#1b9e77")
        ax2 = ax.twinx()
        ax2.grid(False)
        l2 = ax2.plot(epochs, [h["lr"] for h in history], "--", label="Learning Rate", color="#e7298a", alpha=0.75)
        ax2.set_ylabel("Learning Rate", color="#e7298a")
        ax.set_title("Selection Score & LR Schedule")
        lines = l1 + l2
        ax.legend(lines, [l.get_label() for l in lines], loc="lower right")

        plt.tight_layout()
        fig.savefig(plots_dir / "training_curves.png")
        plt.close(fig)

    # 2. Confusion Matrix (Counts + Normalized Recall)
    cm = artifacts["confusion_matrix"]
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(
        cm.astype(float), row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums > 0
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=150)
    for ax, mat, title, fmt in [
        (axes[0], cm, f"{prefix.upper()} Confusion Matrix (Counts)", "d"),
        (axes[1], cm_norm * 100.0, f"{prefix.upper()} Row-Normalized Recall (%)", ".1f"),
    ]:
        im = ax.imshow(
            mat, interpolation="nearest", cmap="Blues", vmin=0, vmax=(100.0 if fmt == ".1f" else None)
        )
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(len(class_names)))
        ax.set_yticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=35, ha="right")
        ax.set_yticklabels(class_names)
        ax.set_xlabel("Predicted Intensity Category")
        ax.set_ylabel("True Intensity Category")
        ax.set_title(title)

        thresh = mat.max() * 0.55
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax.text(
                    j,
                    i,
                    f"{mat[i, j]:{fmt}}",
                    ha="center",
                    va="center",
                    color="white" if mat[i, j] > thresh else "black",
                    fontsize=9,
                )

    plt.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_confusion_matrix.png")
    plt.close(fig)

    # 3. Wind Speed Regression Diagnostics (3-panel)
    y_true_w = artifacts["y_true_wind"]
    y_pred_w = artifacts["y_pred_wind"]
    residuals = y_pred_w - y_true_w

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)

    ax = axes[0]
    hb = ax.hexbin(y_true_w, y_pred_w, gridsize=35, cmap="viridis", mincnt=1)
    plt.colorbar(hb, ax=ax, label="Sample Count")
    max_w = max(float(y_true_w.max()), float(y_pred_w.max()), 160.0)
    lims = [0, max_w * 1.05]
    ax.plot(lims, lims, "r--", lw=1.8, label="1:1 Ideal")
    ax.fill_between(
        lims, [x - 10 for x in lims], [x + 10 for x in lims], color="red", alpha=0.1, label="±10 kt Band"
    )
    mae = float(np.mean(np.abs(residuals)))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    bias = float(np.mean(residuals))
    ax.text(
        0.05,
        0.95,
        f"MAE: {mae:.2f} kt\nRMSE: {rmse:.2f} kt\nBias: {bias:+.2f} kt",
        transform=ax.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
    )
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Best-Track Wind Speed (kt)")
    ax.set_ylabel("Predicted Wind Speed (kt)")
    ax.set_title(f"{prefix.upper()} Wind Speed Parity")
    ax.legend(loc="lower right")

    ax = axes[1]
    ax.scatter(y_true_w, residuals, s=8, alpha=0.25, color="#2b8cbe")
    ax.axhline(0, color="red", linestyle="--", lw=1.5)
    ax.axhline(10, color="gray", linestyle=":", lw=1)
    ax.axhline(-10, color="gray", linestyle=":", lw=1)
    ax.set_xlabel("Best-Track Wind Speed (kt)")
    ax.set_ylabel("Residual: Pred − True (kt)")
    ax.set_title("Residuals vs. Intensity (Bias Check)")

    ax = axes[2]
    ax.hist(residuals, bins=40, color="#31a354", edgecolor="white", alpha=0.85, density=True)
    ax.axvline(0, color="red", linestyle="--", lw=1.5, label="Zero Error")
    ax.axvline(bias, color="black", linestyle="-", lw=1.5, label=f"Mean ({bias:+.2f} kt)")
    ax.set_xlabel("Prediction Error (kt)")
    ax.set_ylabel("Density")
    ax.set_title("Wind Error Distribution")
    ax.legend()

    plt.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_wind_diagnostics.png")
    plt.close(fig)

    # 4. Per-Class F1, Recall & Wind MAE Breakdown
    per_class = artifacts["per_class"]
    x_idx = np.arange(len(class_names))
    width = 0.28

    fig, ax1 = plt.subplots(figsize=(12, 5.2), dpi=150)
    f1s = [c["f1"] * 100 for c in per_class]
    recalls = [c["recall"] * 100 for c in per_class]
    supports = [c["support"] for c in per_class]
    c_maes = [c["wind_mae_kts"] for c in per_class]

    b1 = ax1.bar(x_idx - width / 2, f1s, width, label="F1-Score (%)", color="#1f78b4")
    b2 = ax1.bar(x_idx + width / 2, recalls, width, label="Recall (%)", color="#a6cee3")
    ax1.set_ylabel("Score (%)")
    ax1.set_ylim(0, 112)
    ax1.set_xticks(x_idx)
    ax1.set_xticklabels([f"{name}\n(n={sup})" for name, sup in zip(class_names, supports)])
    ax1.set_title(f"{prefix.upper()} Per-Category Classification & Wind Speed Error")

    ax2 = ax1.twinx()
    ax2.grid(False)
    l3 = ax2.plot(x_idx, c_maes, "o-", color="#e31a1c", lw=2, markersize=6, label="Wind MAE (kt)")
    ax2.set_ylabel("Wind MAE (knots)", color="#e31a1c")
    ax2.set_ylim(0, max(max(c_maes) * 1.25, 15.0))

    handles = [b1, b2] + l3
    ax1.legend(handles, [h.get_label() for h in handles], loc="upper right")

    plt.tight_layout()
    fig.savefig(plots_dir / f"{prefix}_per_class_metrics.png")
    plt.close(fig)

    # 5. Sample Satellite Crop Predictions Grid
    vis_crops = artifacts.get("vis_crops")
    if vis_crops is not None and len(vis_crops) > 0:
        n_show = min(len(vis_crops), 16)
        cols = 4
        rows = math.ceil(n_show / cols)
        fig, axes = plt.subplots(rows, cols, figsize=(13, 3.3 * rows), dpi=140)
        axes = np.array(axes).reshape(-1)

        for idx in range(rows * cols):
            ax = axes[idx]
            ax.axis("off")
            if idx >= n_show:
                continue
            crop = vis_crops[idx]
            t_cls = int(artifacts["y_true_cls"][idx])
            p_cls = int(artifacts["y_pred_cls"][idx])
            conf = float(artifacts["y_probs"][idx, p_cls]) * 100.0
            t_w = float(artifacts["y_true_wind"][idx])
            p_w = float(artifacts["y_pred_wind"][idx])

            ax.imshow(crop, cmap="inferno_r")
            is_exact = t_cls == p_cls
            is_close = abs(t_cls - p_cls) <= 1
            color = "green" if is_exact else ("darkorange" if is_close else "crimson")
            ax.set_title(
                f"True: {class_names[t_cls]} ({t_w:.0f} kt)\n"
                f"Pred: {class_names[p_cls]} [{conf:.0f}%] ({p_w:.0f} kt)",
                fontsize=9,
                color=color,
                fontweight="bold",
            )

        plt.tight_layout()
        fig.savefig(plots_dir / f"{prefix}_sample_predictions.png")
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main Training Entrypoint
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny tropical cyclone classifier.")
    parser.add_argument("--data-dir", type=Path, default=Path("./../data/processed/cls_h5_448"))
    parser.add_argument("--save-dir", type=Path, default=Path("run/convnext_tiny_cls384"))
    parser.add_argument("--model-name", type=str, default="convnext_tiny.fb_in22k_ft_in1k_384")
    parser.add_argument("--img-size", type=int, default=384)
    parser.add_argument("--crop-km", type=float, default=360.0, help="Tighter inner-core FOV (km)")
    parser.add_argument("--max-jitter-km", type=float, default=28.0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--grad-accum", type=int, default=2, help="Effective batch = batch_size * grad_accum")
    parser.add_argument("--lr", type=float, default=1.0e-4, help="Head LR; backbone uses backbone-lr-mult")
    parser.add_argument("--backbone-lr-mult", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=3e-2)
    parser.add_argument("--drop-path", type=float, default=0.3)
    parser.add_argument("--head-dropout", type=float, default=0.25)
    parser.add_argument("--radiometric-jitter", type=float, default=0.05)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--ordinal-alpha", type=float, default=1.5)
    parser.add_argument("--ordinal-weight", type=float, default=0.35)
    parser.add_argument("--consistency-weight", type=float, default=0.15)
    parser.add_argument("--wind-loss-weight", type=float, default=0.35)
    parser.add_argument("--sampler-power", type=float, default=0.35, help="0=none, 0.35=mild, 1=full inverse")
    parser.add_argument("--use-loss-weights", action="store_true", help="Also apply class weights in CE loss")
    parser.add_argument("--ema-decay", type=float, default=0.999, help="0 to disable ModelEmaV2")
    parser.add_argument("--no-tta", action="store_true", help="Disable 4-fold rotational TTA during eval")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience (0 to disable)")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    use_tta = not args.no_tta
    args.save_dir.mkdir(parents=True, exist_ok=True)

    train_h5 = args.data_dir / "train.h5"
    val_h5 = args.data_dir / "val.h5"
    test_h5 = args.data_dir / "test.h5"
    if not train_h5.exists() or not val_h5.exists():
        raise FileNotFoundError(f"Missing train.h5 or val.h5 under {args.data_dir}")

    train_ds = CycloneH5Dataset(
        train_h5, out_size=args.img_size, is_train=True, max_jitter_km=args.max_jitter_km
    )
    val_ds = CycloneH5Dataset(
        val_h5, out_size=args.img_size, is_train=False, max_jitter_km=0.0
    )

    sampler = None
    shuffle = True
    if args.sampler_power > 0.0:
        sample_weights = train_ds.get_sample_weights(power=args.sampler_power)
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(train_ds), replacement=True
        )
        shuffle = False

    loader_kwargs = dict(
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=sampler, shuffle=shuffle, drop_last=True, **loader_kwargs
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs
    )

    tensor_builder = GPUCycloneTensorBuilder(
        img_size=args.img_size, crop_km=args.crop_km, imagenet_norm=True
    ).to(device)

    class_weights_tensor = None
    class_names = DEFAULT_CLASS_NAMES
    summary_path = args.data_dir / "dataset_summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        if "class_names" in summary and len(summary["class_names"]) == 7:
            class_names = summary["class_names"]
        if args.use_loss_weights and "train_class_weights" in summary:
            if args.sampler_power > 0.0:
                print("[Warning] Both WeightedRandomSampler and CE class weights are active.")
            class_weights_tensor = torch.tensor(
                summary["train_class_weights"], dtype=torch.float32, device=device
            )

    cls_criterion = OrdinalCycloneLoss(
        num_classes=len(class_names),
        weight=class_weights_tensor,
        label_smoothing=args.label_smoothing,
        ordinal_alpha=args.ordinal_alpha,
        ordinal_weight=args.ordinal_weight,
        consistency_weight=args.consistency_weight,
    ).to(device)
    wind_criterion = nn.SmoothL1Loss(beta=0.1)

    model = DualHeadConvNeXt(
        model_name=args.model_name,
        num_classes=len(class_names),
        pretrained=not args.no_pretrained,
        drop_path_rate=args.drop_path,
        head_dropout=args.head_dropout,
    ).to(device)

    model_ema: Optional[ModelEmaV2] = None
    if args.ema_decay > 0.0:
        model_ema = ModelEmaV2(model, decay=args.ema_decay, device=device)

    param_groups = [
        {"params": model.backbone.parameters(), "lr": args.lr * args.backbone_lr_mult},
        {"params": model.cls_head.parameters(), "lr": args.lr},
        {"params": model.wind_head.parameters(), "lr": args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)

    warmup_epochs = min(args.warmup_epochs, max(args.epochs - 1, 0))
    if warmup_epochs > 0:
        warmup_sched = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 0.01
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs]
        )
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
        )

    scaler = torch.amp.GradScaler(device=device.type, enabled=use_amp)

    start_epoch = 1
    best_score = -float("inf")
    epochs_no_improve = 0
    history: List[Dict[str, Any]] = []

    if args.resume is not None and args.resume.exists():
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if model_ema is not None and "ema_state_dict" in ckpt:
            model_ema.module.load_state_dict(ckpt["ema_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_score = float(ckpt.get("best_score", -float("inf")))
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    serializable_args = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        seen = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [train]")
        for step, (tb_x100, class_ids, wind_kts) in enumerate(pbar, start=1):
            tb_x100 = tb_x100.to(device, non_blocking=True)
            class_ids = class_ids.to(device, non_blocking=True)
            wind_kts = wind_kts.to(device, non_blocking=True)

            x = apply_cyclone_augmentations(
                tensor_builder(tb_x100), radiometric_jitter=args.radiometric_jitter
            )

            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                logits, wind_pred = model(x)
                loss_cls = cls_criterion(logits, class_ids, wind_pred)
                loss_wind = wind_criterion(wind_pred / 100.0, wind_kts / 100.0)
                loss = (loss_cls + args.wind_loss_weight * loss_wind) / args.grad_accum

            scaler.scale(loss).backward()

            if step % args.grad_accum == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if model_ema is not None:
                    model_ema.update(model)

            bs = class_ids.size(0)
            running_loss += float(loss.item()) * args.grad_accum * bs
            seen += bs
            head_lr = optimizer.param_groups[-1]["lr"]
            pbar.set_postfix(loss=f"{running_loss / max(seen, 1):.4f}", lr=f"{head_lr:.2e}")

        current_lr = optimizer.param_groups[-1]["lr"]
        scheduler.step()

        eval_target = model_ema.module if model_ema is not None else model
        val_metrics, val_artifacts = evaluate(
            model=eval_target,
            tensor_builder=tensor_builder,
            loader=val_loader,
            cls_criterion=cls_criterion,
            wind_criterion=wind_criterion,
            wind_weight=args.wind_loss_weight,
            device=device,
            use_amp=use_amp,
            num_classes=len(class_names),
            use_tta=use_tta,
            collect_samples=True,
        )

        # Composite score rewarding exact accuracy, macro-F1, ±1 cat accuracy, and low wind MAE
        composite_score = (
            0.40 * val_metrics["acc"]
            + 0.25 * val_metrics["macro_f1"]
            + 0.35 * val_metrics["within_1_cat_acc"]
            - 0.01 * val_metrics["wind_mae_kts"]
        )

        row = {
            "epoch": epoch,
            "lr": current_lr,
            "train_loss": running_loss / max(seen, 1),
            **val_metrics,
            "score": composite_score,
        }
        history.append(row)

        print(
            f"Epoch {epoch:02d} | val_loss={val_metrics['loss']:.4f} | "
            f"acc={val_metrics['acc']:.4f} | F1={val_metrics['macro_f1']:.4f} | "
            f"±1cat={val_metrics['within_1_cat_acc']:.4f} | "
            f"MAE={val_metrics['wind_mae_kts']:.2f} kt | RMSE={val_metrics['wind_rmse_kts']:.2f} kt | "
            f"R2={val_metrics['wind_r2']:.3f}"
        )

        ckpt_payload = {
            "epoch": epoch,
            "model_state_dict": eval_target.state_dict(),
            "raw_model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_metrics": val_metrics,
            "best_score": max(best_score, composite_score),
            "class_names": class_names,
            "args": serializable_args,
        }
        if model_ema is not None:
            ckpt_payload["ema_state_dict"] = model_ema.module.state_dict()

        torch.save(ckpt_payload, args.save_dir / "last_model.pt")
        (args.save_dir / "metrics.json").write_text(json.dumps(history, indent=2))

        if composite_score > best_score:
            best_score = composite_score
            epochs_no_improve = 0
            torch.save(ckpt_payload, args.save_dir / "best_model.pt")
            save_training_visualizations(
                history, val_artifacts, class_names, args.save_dir, prefix="val_best"
            )
            print(f"  ↳ New best score ({best_score:.4f})! Saved checkpoint & updated plots.")
        else:
            epochs_no_improve += 1
            save_training_visualizations(
                history, val_artifacts, class_names, args.save_dir, prefix="val_latest"
            )

        if args.patience > 0 and epochs_no_improve >= args.patience:
            print(f"\nEarly stopping triggered after {args.patience} epochs without improvement.")
            break

    print(f"\nTraining complete. Best checkpoint saved to: {args.save_dir / 'best_model.pt'}")

    # Final evaluation on held-out test set using best checkpoint
    if test_h5.exists():
        print(f"\nEvaluating best checkpoint on test.h5 (4x TTA={'ON' if use_tta else 'OFF'})...")
        best_ckpt = torch.load(args.save_dir / "best_model.pt", map_location=device, weights_only=True)
        model.load_state_dict(best_ckpt["model_state_dict"])
        test_ds = CycloneH5Dataset(test_h5, out_size=args.img_size, is_train=False, max_jitter_km=0.0)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size * 2, shuffle=False, **loader_kwargs)
        test_metrics, test_artifacts = evaluate(
            model=model,
            tensor_builder=tensor_builder,
            loader=test_loader,
            cls_criterion=cls_criterion,
            wind_criterion=wind_criterion,
            wind_weight=args.wind_loss_weight,
            device=device,
            use_amp=use_amp,
            num_classes=len(class_names),
            use_tta=use_tta,
            collect_samples=True,
        )
        (args.save_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2))
        save_training_visualizations(history, test_artifacts, class_names, args.save_dir, prefix="test")
        print(
            f"Test Results | acc={test_metrics['acc']:.4f} | F1={test_metrics['macro_f1']:.4f} | "
            f"±1cat={test_metrics['within_1_cat_acc']:.4f} | MAE={test_metrics['wind_mae_kts']:.2f} kt | "
            f"RMSE={test_metrics['wind_rmse_kts']:.2f} kt"
        )


if __name__ == "__main__":
    main()
