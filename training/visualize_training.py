from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib.gridspec as gridspec
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent

# Light, elegant publication-grade palette
PALETTE = {
    "fig_bg": "#F8FAFC",      # Alabaster slate-50
    "card_bg": "#FFFFFF",     # Pure white panel
    "border": "#CBD5E1",      # Slate-300
    "grid": "#E2E8F0",        # Slate-200
    "text_main": "#0F172A",   # Slate-900
    "text_sub": "#475569",    # Slate-600
    "blue": "#1D4ED8",        # Royal Blue (Box / Primary)
    "sky": "#0284C7",         # Ocean Blue (Secondary / Outer Kpts)
    "emerald": "#059669",     # Emerald Green (Pose mAP / GT)
    "teal": "#0D9488",        # Teal (Pose mAP@50-95)
    "crimson": "#E11D48",     # Crimson Rose (Cyclone Center K0 / RLE)
    "amber": "#D97706",       # Warm Amber (Pred Box / Highlights)
    "gold_bright": "#F59E0B", # Vivid Gold (Pred Box Stroke)
    "violet": "#6D28D9",      # Deep Violet (Cls / Other)
}

plt.rcParams.update({
    "figure.facecolor": PALETTE["fig_bg"],
    "axes.facecolor": PALETTE["card_bg"],
    "axes.edgecolor": PALETTE["border"],
    "axes.linewidth": 1.1,
    "axes.labelcolor": PALETTE["text_main"],
    "axes.titlecolor": PALETTE["text_main"],
    "xtick.color": PALETTE["text_sub"],
    "ytick.color": PALETTE["text_sub"],
    "text.color": PALETTE["text_main"],
    "grid.color": PALETTE["grid"],
    "grid.linestyle": "--",
    "grid.linewidth": 0.8,
    "grid.alpha": 0.9,
    "font.family": "sans-serif",
    "font.size": 10,
})


def resolve_path_smart(raw_path: Path) -> Path:
    """Resolves paths relative to CWD first, falling back to script directory."""
    if raw_path.exists():
        return raw_path.resolve()
    alt = (SCRIPT_DIR / raw_path).resolve()
    if alt.exists():
        return alt
    return raw_path.resolve()


def ema_smooth(values: np.ndarray, alpha: float = 0.25) -> np.ndarray:
    """Exponential moving average for smooth trend lines."""
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1.0 - alpha) * out[i - 1]
    return out


def infer_satellite(filename: str) -> str:
    lower = filename.lower()
    for sat in ("goes_east", "goes_west", "himawari"):
        if sat in lower:
            return sat
    return "other"


def parse_yolo_pose_label(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parses YOLO-Pose ground-truth boxes and pixel keypoints from a .txt file."""
    if not label_path.exists():
        return []
    instances = []
    for line in label_path.read_text(encoding="utf-8").strip().splitlines():
        parts = [float(x) for x in line.strip().split()]
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, w, h = parts[1] * img_w, parts[2] * img_h, parts[3] * img_w, parts[4] * img_h
        box_xyxy = np.array([xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2])

        kp_vals = parts[5:]
        stride = 3 if len(kp_vals) % 3 == 0 else 2
        kpts, vis = [], []
        for i in range(0, len(kp_vals), stride):
            kx = kp_vals[i] * img_w
            ky = kp_vals[i + 1] * img_h
            kv = kp_vals[i + 2] if stride == 3 else 2.0
            kpts.append([kx, ky])
            vis.append(kv)

        instances.append({
            "cls": cls_id,
            "box": box_xyxy,
            "kpts": np.array(kpts, dtype=float),
            "vis": np.array(vis, dtype=float),
        })
    return instances


def compute_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter + 1e-8)


def match_instances_one_to_one(
    gt_instances: list[dict],
    pred_boxes: np.ndarray,
    iou_thresh: float = 0.30,
) -> list[tuple[int, int, float]]:
    """Globally matches GT and predicted boxes one-to-one in descending IoU order."""
    if not gt_instances or len(pred_boxes) == 0:
        return []

    candidates: list[tuple[float, int, int]] = []
    for gt_i, gt in enumerate(gt_instances):
        for pred_j, pb in enumerate(pred_boxes):
            iou = compute_iou(gt["box"], pb)
            if iou >= iou_thresh:
                candidates.append((iou, gt_i, pred_j))

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for iou, gt_i, pred_j in candidates:
        if gt_i in used_gt or pred_j in used_pred:
            continue
        used_gt.add(gt_i)
        used_pred.add(pred_j)
        matches.append((gt_i, pred_j, iou))

    return matches


def _style_clean_axes(ax: plt.Axes, hide_right: bool = True) -> None:
    ax.spines["top"].set_visible(False)
    if hide_right:
        ax.spines["right"].set_visible(False)
    ax.grid(True, zorder=0)


def plot_training_telemetry(df: pd.DataFrame, out_dir: Path) -> dict:
    """Generates a 4-panel light-themed telemetry report with dual-axis loss scaling."""
    epochs = df["epoch"].to_numpy()
    fig = plt.figure(figsize=(19, 12.5), dpi=300)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.36, wspace=0.24)

    fig.suptitle(
        "YOLO26-Pose Tropical Cyclone Detection & Center Localization — Training Report",
        fontsize=16,
        fontweight="bold",
        color=PALETTE["text_main"],
        y=0.97,
    )
    fig.text(
        0.5,
        0.938,
        "Geostationary Satellite Multi-Sensor Run (GOES-East, GOES-West, Himawari) • Smoothed EMA Trends Overlaid on Raw Epoch Values",
        ha="center",
        fontsize=10.5,
        color=PALETTE["text_sub"],
    )

    # -------------------------------------------------------------------------
    # Panel 1: Bounding Box & Cyclone Classification Losses
    # -------------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    _style_clean_axes(ax1)
    box_cols = [
        ("train/box_loss", "Train Box Loss (Localization)", PALETTE["blue"], "-"),
        ("val/box_loss", "Validation Box Loss", PALETTE["sky"], "--"),
        ("train/cls_loss", "Train Classification Loss", PALETTE["violet"], "-"),
        ("val/cls_loss", "Validation Classification Loss", "#9333EA", "--"),
    ]
    for col, label, color, ls in box_cols:
        if col in df.columns:
            raw = df[col].to_numpy(dtype=float)
            ax1.plot(epochs, raw, color=color, alpha=0.18, lw=1.0)
            ax1.plot(epochs, ema_smooth(raw), color=color, linestyle=ls, lw=2.3, label=label)

    if "val/box_loss" in df.columns and len(df) > 1:
        v_box = df["val/box_loss"].to_numpy(dtype=float)
        change_pct = 100.0 * (v_box[-1] - v_box[0]) / (abs(v_box[0]) + 1e-8)
        ax1.text(
            0.98, 0.94,
            f"Val Box Loss: {v_box[0]:.3f} → {v_box[-1]:.3f} ({change_pct:+.1f}%)",
            transform=ax1.transAxes, ha="right", va="top", fontsize=8.5, fontweight="bold",
            color=PALETTE["blue"],
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#EFF6FF", edgecolor="#BFDBFE"),
        )

    ax1.set_title("1. Cyclone Bounding Box & Classification Losses\n(Lower is better — tracks cyclone extent & false-alarm suppression)", fontsize=11, fontweight="bold", pad=10)
    ax1.set_xlabel("Training Epoch", fontweight="bold")
    ax1.set_ylabel("Loss Value (Dimensionless)", fontweight="bold")
    ax1.legend(facecolor=PALETTE["card_bg"], edgecolor=PALETTE["border"], fontsize=8.8, loc="upper right", bbox_to_anchor=(1.0, 0.84))

    # -------------------------------------------------------------------------
    # Panel 2: Dual-Axis Keypoint Pose Loss (Left) & RLE Uncertainty Loss (Right)
    # -------------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    _style_clean_axes(ax2, hide_right=False)
    lines_for_legend = []

    pose_cols = [
        ("train/pose_loss", "Train Keypoint Pose Loss (Left Axis)", PALETTE["amber"], "-"),
        ("val/pose_loss", "Val Keypoint Pose Loss (Left Axis)", "#B45309", "--"),
        ("val/kobj_loss", "Val Keypoint Visibility Loss (Left Axis)", PALETTE["teal"], "-."),
    ]
    for col, label, color, ls in pose_cols:
        if col in df.columns:
            raw = df[col].to_numpy(dtype=float)
            if np.nanmax(np.abs(raw)) > 1e-8:
                ax2.plot(epochs, raw, color=color, alpha=0.18, lw=1.0)
                (ln,) = ax2.plot(epochs, ema_smooth(raw), color=color, linestyle=ls, lw=2.3, label=label)
                lines_for_legend.append(ln)

    ax2.set_ylabel("Pose / Visibility Loss", fontweight="bold", color="#B45309")
    ax2.tick_params(axis="y", labelcolor="#B45309")

    ax2_rle = ax2.twinx()
    ax2_rle.spines["top"].set_visible(False)
    ax2_rle.spines["right"].set_color(PALETTE["crimson"])
    ax2_rle.grid(False)

    rle_cols = [
        ("train/rle_loss", "Train RLE Uncertainty Loss (Right Axis)", PALETTE["crimson"], "-"),
        ("val/rle_loss", "Val RLE Uncertainty Loss (Right Axis)", "#9F1239", "--"),
    ]
    has_rle = False
    for col, label, color, ls in rle_cols:
        if col in df.columns:
            raw = df[col].to_numpy(dtype=float)
            if np.nanmax(np.abs(raw)) > 1e-8:
                has_rle = True
                ax2_rle.plot(epochs, raw, color=color, alpha=0.18, lw=1.0)
                (ln,) = ax2_rle.plot(epochs, ema_smooth(raw), color=color, linestyle=ls, lw=2.3, label=label)
                lines_for_legend.append(ln)

    if has_rle:
        ax2_rle.set_ylabel("RLE Uncertainty Loss", fontweight="bold", color=PALETTE["crimson"])
        ax2_rle.tick_params(axis="y", labelcolor=PALETTE["crimson"])
    else:
        ax2_rle.set_yticks([])

    if lines_for_legend:
        ax2.legend(
            lines_for_legend,
            [ln.get_label() for ln in lines_for_legend],
            facecolor=PALETTE["card_bg"],
            edgecolor=PALETTE["border"],
            fontsize=8.3,
            loc="upper right",
        )

    ax2.set_title("2. Cyclone Center Pose & RLE Uncertainty Losses (Dual-Axis)\n(Lower is better — separates sub-pixel pose error from RLE scale)", fontsize=11, fontweight="bold", pad=10)
    ax2.set_xlabel("Training Epoch", fontweight="bold")

    # -------------------------------------------------------------------------
    # Panel 3: Validation mAP Progression & Tightness Gap
    # -------------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    _style_clean_axes(ax3)
    if "metrics/mAP50(P)" in df.columns and "metrics/mAP50-95(P)" in df.columns:
        p50 = df["metrics/mAP50(P)"].to_numpy(dtype=float)
        p5095 = df["metrics/mAP50-95(P)"].to_numpy(dtype=float)
        ax3.plot(epochs, p50, color=PALETTE["emerald"], lw=2.4, label="Pose mAP@50 (Coarse Keypoint Match)")
        ax3.plot(epochs, p5095, color=PALETTE["teal"], linestyle="--", lw=2.4, label="Pose mAP@50-95 (Strict Sub-Pixel Match)")
        ax3.fill_between(
            epochs, p5095, p50, color=PALETTE["emerald"], alpha=0.12,
            label="Localization Gap (Narrower = Tighter Center Fit)",
        )

    if "metrics/mAP50-95(B)" in df.columns:
        ax3.plot(epochs, df["metrics/mAP50-95(B)"], color=PALETTE["blue"], linestyle="-.", lw=2.0, label="Cyclone Box mAP@50-95")

    safe_pose_scores = df["metrics/mAP50-95(P)"].fillna(-1.0)
    best_idx = int(safe_pose_scores.idxmax())
    best_ep = int(df.loc[best_idx, "epoch"])
    best_pose = float(max(0.0, safe_pose_scores.loc[best_idx]))
    ax3.scatter([best_ep], [best_pose], color=PALETTE["crimson"], s=110, zorder=6, edgecolor="#FFFFFF", lw=1.8)
    ax3.annotate(
        f"Peak Pose mAP@50-95: {best_pose:.4f}\nSaved as best.pt (Epoch {best_ep})",
        xy=(best_ep, best_pose),
        xytext=(max(epochs[0], best_ep - len(epochs) * 0.38), max(0.08, best_pose - 0.22)),
        arrowprops=dict(arrowstyle="->", color=PALETTE["crimson"], lw=1.6),
        bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF1F2", edgecolor=PALETTE["crimson"], lw=1.2),
        fontsize=9,
        fontweight="bold",
        color=PALETTE["text_main"],
    )

    ax3.set_title("3. Validation Accuracy — Mean Average Precision (mAP)\n(Higher is better, 0.0 to 1.0 — primary checkpoint selection metric)", fontsize=11, fontweight="bold", pad=10)
    ax3.set_xlabel("Training Epoch", fontweight="bold")
    ax3.set_ylabel("Mean Average Precision (mAP)", fontweight="bold")
    ax3.set_ylim(0, 1.06)
    ax3.legend(facecolor=PALETTE["card_bg"], edgecolor=PALETTE["border"], fontsize=8.5, loc="lower right")

    # -------------------------------------------------------------------------
    # Panel 4: Precision–Recall Phase Trajectory with Iso-F1 Contours
    # -------------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    _style_clean_axes(ax4)
    grid_x = np.linspace(0.01, 1.0, 200)
    grid_y = np.linspace(0.01, 1.0, 200)
    R_grid, P_grid = np.meshgrid(grid_x, grid_y)
    F1_grid = 2 * P_grid * R_grid / (P_grid + R_grid)
    cs = ax4.contour(R_grid, P_grid, F1_grid, levels=[0.3, 0.5, 0.7, 0.8, 0.9, 0.95], colors="#CBD5E1", linestyles=":")
    ax4.clabel(cs, inline=True, fontsize=8, fmt="F1=%.2f")

    if "metrics/precision(P)" in df.columns and "metrics/recall(P)" in df.columns:
        rec = df["metrics/recall(P)"].fillna(0.0).to_numpy(dtype=float)
        prec = df["metrics/precision(P)"].fillna(0.0).to_numpy(dtype=float)
        ax4.plot(rec, prec, color="#94A3B8", alpha=0.5, lw=1.3, zorder=2)
        sc = ax4.scatter(rec, prec, c=epochs, cmap="viridis", s=46, zorder=4, edgecolor="#FFFFFF", lw=0.7)
        ax4.scatter(
            [rec[best_idx]], [prec[best_idx]],
            color=PALETTE["crimson"], marker="*", s=220, edgecolor="#FFFFFF", lw=1.0, zorder=6,
            label=f"Best Checkpoint (Ep {best_ep}: P={prec[best_idx]:.2f}, R={rec[best_idx]:.2f})",
        )
        cbar = plt.colorbar(sc, ax=ax4, pad=0.02)
        cbar.set_label("Training Epoch", color=PALETTE["text_main"], fontweight="bold")

    ax4.set_title("4. Keypoint Precision–Recall Trade-off & F1 Contours\n(Top-right corner = high detection rate with minimal false alarms)", fontsize=11, fontweight="bold", pad=10)
    ax4.set_xlabel("Recall (Fraction of Ground-Truth Cyclones Detected)", fontweight="bold")
    ax4.set_ylabel("Precision (Fraction of Predicted Cyclones Valid)", fontweight="bold")
    ax4.set_xlim(0, 1.04)
    ax4.set_ylim(0, 1.04)
    ax4.legend(facecolor=PALETTE["card_bg"], edgecolor=PALETTE["border"], fontsize=8.5, loc="lower left")

    out_path = out_dir / "1_training_telemetry.png"
    plt.savefig(out_path, bbox_inches="tight", facecolor=PALETTE["fig_bg"])
    plt.close(fig)
    print(f"[Saved] {out_path}")
    return {"best_epoch": best_ep, "best_pose_map50_95": best_pose}


def _draw_prominent_cyclone_box(
    ax: plt.Axes,
    box: np.ndarray,
    label_text: str,
    color: str,
    dashed: bool = False,
    label_bottom: bool = False,
) -> None:
    """Draws a high-contrast haloed bounding box that stands out clearly over bright clouds."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    ls = "--" if dashed else "-"

    ax.add_patch(
        plt.Rectangle((x1, y1), w, h, fill=False, edgecolor="#0F172A", linestyle=ls, lw=3.6, zorder=4)
    )
    ax.add_patch(
        plt.Rectangle((x1, y1), w, h, fill=False, edgecolor=color, linestyle=ls, lw=2.3, zorder=5)
    )

    if not dashed:
        ax.add_patch(
            plt.Rectangle((x1, y1), w, h, fill=True, facecolor=color, alpha=0.08, edgecolor="none", zorder=3)
        )

    text_y = min(y2 + 13, ax.get_ylim()[0] - 4) if label_bottom else max(14, y1 - 6)
    ax.text(
        x1,
        text_y,
        label_text,
        color="#0F172A" if not dashed else "#FFFFFF",
        fontsize=8.0,
        fontweight="bold",
        zorder=9,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor=color,
            edgecolor="#0F172A",
            lw=0.9,
        ),
    )


def _draw_prominent_center_and_keypoints(
    ax: plt.Axes,
    gk: np.ndarray,
    pk: np.ndarray,
    gv: np.ndarray,
) -> float | None:
    """Draws prominent K0 Cyclone Center bullseye vs GT Center star, plus outer keypoints."""
    center_err_px: float | None = None
    halo_stroke = [pe.withStroke(linewidth=2.4, foreground="#0F172A")]

    for k_idx in range(1, len(gk)):
        if gv[k_idx] <= 0 or (gk[k_idx][0] <= 0 and gk[k_idx][1] <= 0):
            continue
        gx, gy = gk[k_idx]
        px, py = pk[k_idx]

        if pk[0][0] > 0 and pk[0][1] > 0:
            ax.plot(
                [pk[0][0], px], [pk[0][1], py],
                color="#E0F2FE", linestyle=":", lw=1.3, alpha=0.85, zorder=5,
            )

        ax.plot([gx, px], [gy, py], color="#FB7185", lw=1.4, alpha=0.85, zorder=6)
        ax.scatter(gx, gy, marker="D", s=42, color="#10B981", edgecolor="#0F172A", lw=0.8, zorder=7)
        ax.scatter(px, py, marker="o", s=46, color="#38BDF8", edgecolor="#0F172A", lw=0.9, zorder=8)
        ax.text(
            px + 5, py - 4, f"K{k_idx}",
            color="#FFFFFF", fontsize=7.5, fontweight="bold", zorder=9, path_effects=halo_stroke,
        )

    if len(gk) > 0 and gv[0] > 0 and (gk[0][0] > 0 or gk[0][1] > 0):
        gx0, gy0 = gk[0]
        px0, py0 = pk[0]
        center_err_px = float(np.linalg.norm(pk[0] - gk[0]))

        ax.plot(
            [gx0, px0], [gy0, py0],
            color="#FF1E56", lw=2.4, zorder=8,
            path_effects=[pe.withStroke(linewidth=3.6, foreground="#FFFFFF")],
        )
        ax.scatter(
            gx0, gy0, marker="*", s=210, color="#10B981",
            edgecolor="#0F172A", lw=1.1, zorder=10,
        )
        ax.scatter(px0, py0, s=360, color="#FF1E56", alpha=0.28, edgecolor="none", zorder=9)
        ax.scatter(px0, py0, s=145, color="#FFFFFF", edgecolor="#0F172A", lw=1.2, zorder=10)
        ax.scatter(px0, py0, s=85, color="#E11D48", edgecolor="none", zorder=11)
        ax.scatter(px0, py0, marker="+", s=130, color="#FFFFFF", lw=1.6, zorder=12)

        ax.text(
            px0 + 8,
            py0 + 10,
            f"Center (K0) • Δ{center_err_px:.1f}px",
            color="#FFFFFF",
            fontsize=8.0,
            fontweight="bold",
            zorder=13,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#E11D48", edgecolor="#FFFFFF", lw=0.9),
        )

    return center_err_px


def evaluate_and_plot_spatial_diagnostics(
    weights_path: Path,
    images_dir: Path,
    labels_dir: Path,
    out_dir: Path,
    split_name: str = "test",
    conf_thresh: float = 0.55,
    batch_size: int = 16,
    edge_margin: int = 48,
) -> dict[str, Any]:
    """Computes keypoint pixel errors plus operational false-positive / precision / recall audit."""
    model = YOLO(str(weights_path))
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        raise FileNotFoundError(f"No PNG tiles found in {images_dir}")

    split_tag = split_name.lower()
    split_upper = split_name.upper()
    records = []
    tile_diagnostics = []

    total_gt_cyclones = 0
    matched_tp = 0
    unmatched_fp = 0
    bg_tiles_total = 0
    bg_tiles_with_fp = 0
    bg_fp_detections = 0

    for start in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[start : start + batch_size]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            conf=conf_thresh,
            stream=True,
            verbose=False,
        )

        for img_path, res in zip(batch_paths, results):
            img = cv2.imread(str(img_path))
            h, w = img.shape[:2]
            sat = infer_satellite(img_path.name)
            gt_instances = parse_yolo_pose_label(labels_dir / f"{img_path.stem}.txt", w, h)

            pred_boxes = res.boxes.xyxy.cpu().numpy() if res.boxes is not None and len(res.boxes) > 0 else np.empty((0, 4))
            pred_kpts = res.keypoints.xy.cpu().numpy() if res.keypoints is not None and res.keypoints.xy.numel() > 0 else np.empty((0, 0, 2))
            pred_confs = res.boxes.conf.cpu().numpy() if res.boxes is not None and len(res.boxes) > 0 else np.empty((0,))

            total_gt_cyclones += len(gt_instances)
            if len(gt_instances) == 0:
                bg_tiles_total += 1
                if len(pred_boxes) > 0:
                    bg_tiles_with_fp += 1
                    bg_fp_detections += len(pred_boxes)

            matches = match_instances_one_to_one(gt_instances, pred_boxes, iou_thresh=0.30)
            matched_tp += len(matches)
            unmatched_fp += max(0, len(pred_boxes) - len(matches))

            matched_pairs = []
            tile_errors = []
            center_errors = []
            tile_edge_clipped = False

            for gt_i, best_j, matched_iou in matches:
                if best_j >= len(pred_kpts):
                    continue
                gt = gt_instances[gt_i]
                pk = pred_kpts[best_j]
                gk = gt["kpts"]
                gv = gt["vis"]
                n_k = min(len(gk), len(pk))

                gx0, gy0 = float(gk[0][0]), float(gk[0][1])
                edge_dist_px = min(gx0, gy0, float(w) - gx0, float(h) - gy0)
                is_edge_clipped = edge_dist_px < edge_margin
                if is_edge_clipped:
                    tile_edge_clipped = True

                matched_pairs.append((gt["box"], pred_boxes[best_j], gk[:n_k], pk[:n_k], gv[:n_k], float(pred_confs[best_j])))

                for k_idx in range(n_k):
                    if gv[k_idx] <= 0 or (gk[k_idx][0] <= 0 and gk[k_idx][1] <= 0):
                        continue
                    err_px = float(np.linalg.norm(pk[k_idx] - gk[k_idx]))
                    tile_errors.append(err_px)
                    if k_idx == 0:
                        center_errors.append(err_px)
                    records.append({
                        "split": split_tag,
                        "image": img_path.name,
                        "satellite": sat,
                        "keypoint_idx": "K0 (Center)" if k_idx == 0 else f"K{k_idx}",
                        "kpt_order": k_idx,
                        "error_px": err_px,
                        "iou": float(matched_iou),
                        "conf": float(pred_confs[best_j]),
                        "edge_dist_px": round(edge_dist_px, 2),
                        "edge_clipped": is_edge_clipped,
                    })

            if matched_pairs:
                tile_diagnostics.append({
                    "path": img_path,
                    "satellite": sat,
                    "mean_err": float(np.mean(tile_errors)) if tile_errors else 999.0,
                    "center_err": float(np.mean(center_errors)) if center_errors else 999.0,
                    "edge_clipped": tile_edge_clipped,
                    "pairs": matched_pairs,
                })

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    op_prec = matched_tp / max(1, matched_tp + unmatched_fp)
    op_rec = matched_tp / max(1, total_gt_cyclones)
    op_f1 = 2 * op_prec * op_rec / max(1e-8, op_prec + op_rec)
    bg_fp_rate = 100.0 * bg_tiles_with_fp / max(1, bg_tiles_total)

    detection_stats = {
        "split": split_tag,
        "conf_thresh": conf_thresh,
        "total_gt": total_gt_cyclones,
        "tp": matched_tp,
        "fp": unmatched_fp,
        "fn": max(0, total_gt_cyclones - matched_tp),
        "precision": round(op_prec, 4),
        "recall": round(op_rec, 4),
        "f1": round(op_f1, 4),
        "bg_tiles_total": bg_tiles_total,
        "bg_tiles_with_fp": bg_tiles_with_fp,
        "bg_fp_detections": bg_fp_detections,
        "bg_tile_fp_rate_pct": round(bg_fp_rate, 2),
    }

    print(
        f"[Audit {split_upper} @ conf≥{conf_thresh:.2f}] "
        f"TP={matched_tp}/{total_gt_cyclones} | FP={unmatched_fp} (on bg tiles: {bg_fp_detections} across {bg_tiles_with_fp}/{bg_tiles_total} tiles, {bg_fp_rate:.1f}%) | "
        f"P={op_prec:.3f}, R={op_rec:.3f}, F1={op_f1:.3f}"
    )

    err_df = pd.DataFrame(records)
    if err_df.empty:
        print(f"[Warning] [{split_upper}] No matched GT-Prediction pairs above conf={conf_thresh:.2f}, IoU=0.30; skipping spatial error plots.")
        return {"err_df": err_df, "stats": detection_stats}

    csv_out = out_dir / f"keypoint_errors_{split_tag}.csv"
    err_df.to_csv(csv_out, index=False)
    print(f"[Saved] {csv_out}")

    # -------------------------------------------------------------------------
    # Figure 2: Per-Satellite & Per-Keypoint Pixel Error Breakdown
    # -------------------------------------------------------------------------
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(18, 7.4), dpi=300)
    _style_clean_axes(ax_left)
    _style_clean_axes(ax_right)

    fig.suptitle(
        f"[{split_upper} SET] Geostationary Sensor & Cyclone Keypoint Localization Error Analysis (conf ≥ {conf_thresh:.2f})",
        fontsize=14.5,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5, 0.935,
        f"Operational P={op_prec:.2f}, R={op_rec:.2f}, F1={op_f1:.2f} • Bg Tile False Alarm Rate: {bg_fp_rate:.1f}% ({bg_tiles_with_fp}/{bg_tiles_total}) • Interior Margin ≥ {edge_margin} px",
        ha="center", fontsize=9.8, color=PALETTE["text_sub"],
    )

    sats = sorted(err_df["satellite"].unique())
    sat_palette = {
        "goes_east": PALETTE["sky"],
        "goes_west": PALETTE["emerald"],
        "himawari": PALETTE["amber"],
        "other": PALETTE["violet"],
    }

    data_by_sat = [err_df.loc[err_df["satellite"] == s, "error_px"].to_numpy() for s in sats]
    parts = ax_left.violinplot(data_by_sat, positions=range(len(sats)), showmeans=True, showmedians=True)
    for idx, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(sat_palette.get(sats[idx], PALETTE["sky"]))
        pc.set_edgecolor(PALETTE["text_main"])
        pc.set_alpha(0.38)
    for stat_key in ("cmeans", "cmedians", "cbars", "cmins", "cmaxes"):
        if stat_key in parts:
            parts[stat_key].set_color(PALETTE["text_main"])
            parts[stat_key].set_linewidth(1.4)

    global_max_y = max(float(np.max(v)) for v in data_by_sat if len(v) > 0)
    ax_left.set_ylim(-global_max_y * 0.03, global_max_y * 1.25)

    for idx, vals in enumerate(data_by_sat):
        sat_name = sats[idx]
        interior_vals = err_df.loc[(err_df["satellite"] == sat_name) & (~err_df["edge_clipped"]), "error_px"].to_numpy()
        int_med_str = f"{np.median(interior_vals):.1f} px" if len(interior_vals) > 0 else "N/A"

        jitter = np.random.default_rng(42).normal(0, 0.045, size=len(vals))
        ax_left.scatter(
            np.full(len(vals), idx) + jitter, vals,
            color=sat_palette.get(sat_name, PALETTE["sky"]),
            edgecolor="#FFFFFF", lw=0.6, alpha=0.85, s=26, zorder=4,
        )
        box_y = float(np.max(vals)) + global_max_y * 0.04
        ax_left.text(
            idx,
            box_y,
            f"All Median: {np.median(vals):.1f} px (90th: {np.percentile(vals, 90):.1f})\n"
            f"Interior Median: {int_med_str} (n={len(vals)} kpts)",
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="bold",
            color=PALETTE["text_main"],
            bbox=dict(boxstyle="round,pad=0.32", facecolor=PALETTE["card_bg"], edgecolor=PALETTE["border"]),
        )

    ax_left.set_xticks(range(len(sats)))
    ax_left.set_xticklabels([s.replace("_", "-").upper() for s in sats], fontweight="bold", fontsize=10)
    ax_left.set_xlabel("Geostationary Satellite Source", fontweight="bold")
    ax_left.set_ylabel("Localization Error (Pixels)", fontweight="bold")
    ax_left.set_title("A. Overall Keypoint Error Distribution by Satellite Sensor", fontweight="bold", pad=10)

    pivot_mean = err_df.pivot_table(index=["kpt_order", "keypoint_idx"], columns="satellite", values="error_px", aggfunc="mean")
    pivot_mean = pivot_mean.sort_index(level=0)
    kpt_labels = [idx[1] for idx in pivot_mean.index]
    x = np.arange(len(kpt_labels))
    bar_w = 0.78 / max(1, len(sats))

    ax_right.axvspan(-0.45, 0.45, color="#FFE4E6", alpha=0.55, zorder=0, label="Cyclone Eye Center (K0)")

    max_bar = 0.0
    for i, sat in enumerate(sats):
        if sat in pivot_mean.columns:
            vals = pivot_mean[sat].to_numpy()
            max_bar = max(max_bar, float(np.nanmax(vals)))
            offset = (i - (len(sats) - 1) / 2) * bar_w
            bars = ax_right.bar(
                x + offset, vals, width=bar_w * 0.90,
                label=sat.replace("_", "-").upper(),
                color=sat_palette.get(sat, PALETTE["sky"]),
                edgecolor="#FFFFFF", lw=0.8, zorder=3,
            )
            for bar, v in zip(bars, vals):
                if not np.isnan(v):
                    ax_right.text(
                        bar.get_x() + bar.get_width() / 2, v + 0.25, f"{v:.1f}px",
                        ha="center", fontsize=8, fontweight="bold", color=PALETTE["text_main"],
                    )

    if max_bar > 0:
        ax_right.set_ylim(0, max_bar * 1.18)

    ax_right.set_xticks(x)
    ax_right.set_xticklabels(kpt_labels, fontweight="bold", fontsize=9.5)
    ax_right.set_xlabel("Keypoint Index (K0 = Cyclone Eye Center; K1+ = Structural Band Keypoints)", fontweight="bold")
    ax_right.set_ylabel("Mean Euclidean Error (Pixels)", fontweight="bold")
    ax_right.set_title("B. Mean Pixel Error per Keypoint Across Sensors", fontweight="bold", pad=10)
    ax_right.legend(facecolor=PALETTE["card_bg"], edgecolor=PALETTE["border"], fontsize=8.8)

    err_fig_path = out_dir / f"2_{split_tag}_satellite_keypoint_errors.png"
    plt.savefig(err_fig_path, bbox_inches="tight", facecolor=PALETTE["fig_bg"])
    plt.close(fig)
    print(f"[Saved] {err_fig_path}")

    # -------------------------------------------------------------------------
    # Figure 3: Prominent Cyclone Box & Eye Center Overlay Grid
    # -------------------------------------------------------------------------
    sorted_tiles = sorted(tile_diagnostics, key=lambda d: d["center_err"])
    selected = sorted_tiles[:3] + sorted_tiles[-3:] if len(sorted_tiles) >= 6 else sorted_tiles[:6]

    fig, axes = plt.subplots(2, 3, figsize=(18, 11.8), dpi=300)
    fig.suptitle(
        f"[{split_upper} SET] Qualitative Audit — Cyclone Bounding Box & Eye Center (K0) Localization",
        fontsize=15,
        fontweight="bold",
        y=0.975,
    )
    fig.text(
        0.5, 0.942,
        "Top Row: Highest-Precision Center Predictions  •  Bottom Row: Largest Center Offset Cases (Failure Audit)",
        ha="center", fontsize=10.5, color=PALETTE["text_sub"],
    )

    for idx, (ax, item) in enumerate(zip(axes.flat, selected)):
        img_rgb = cv2.cvtColor(cv2.imread(str(item["path"])), cv2.COLOR_BGR2RGB)
        ax.imshow(img_rgb)
        ax.set_xlim(0, img_rgb.shape[1])
        ax.set_ylim(img_rgb.shape[0], 0)

        for gt_box, pred_box, gk, pk, gv, conf in item["pairs"]:
            _draw_prominent_cyclone_box(
                ax, gt_box, label_text="GT Cyclone", color="#10B981", dashed=True, label_bottom=True
            )
            _draw_prominent_cyclone_box(
                ax, pred_box, label_text=f"Pred Cyclone ({conf:.2f})", color="#F59E0B", dashed=False, label_bottom=False
            )
            _draw_prominent_center_and_keypoints(ax, gk, pk, gv)

        if len(selected) >= 6 and idx < 3:
            tier = "BEST FIT"
        else:
            tier = "EDGE-CLIPPED AUDIT" if item["edge_clipped"] else "INTERIOR AUDIT"

        ax.set_title(
            f"[{tier}] {item['path'].name}\n"
            f"Sensor: {item['satellite'].upper()} | Center (K0) Error: {item['center_err']:.2f} px | All-Kpt Mean: {item['mean_err']:.2f} px",
            fontsize=8.3,
            fontweight="bold",
            pad=6,
        )
        ax.axis("off")

    for ax in axes.flat[len(selected):]:
        ax.axis("off")

    legend_handles = [
        plt.Line2D([0], [0], color="#F59E0B", lw=3, label="Predicted Cyclone Box (Amber)"),
        plt.Line2D([0], [0], color="#10B981", lw=2.2, linestyle="--", label="Ground-Truth Box (Emerald)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#E11D48", markeredgecolor="#0F172A", markersize=11, label="Predicted Cyclone Center K0 (Crimson Bullseye)"),
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#10B981", markeredgecolor="#0F172A", markersize=13, label="Ground-Truth Center K0 (Emerald Star)"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#38BDF8", markeredgecolor="#0F172A", markersize=8, label="Outer Spiral Keypoints (K1+)"),
        plt.Line2D([0], [0], color="#FF1E56", lw=2.5, label="Center Offset Vector (GT → Pred)"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=True,
        facecolor=PALETTE["card_bg"],
        edgecolor=PALETTE["border"],
        fontsize=9.2,
        bbox_to_anchor=(0.5, 0.01),
    )

    plt.tight_layout(rect=[0, 0.06, 1, 0.93])
    vec_fig_path = out_dir / f"3_{split_tag}_gt_vs_pred_vectors.png"
    plt.savefig(vec_fig_path, bbox_inches="tight", facecolor=PALETTE["fig_bg"])
    plt.close(fig)
    print(f"[Saved] {vec_fig_path}")
    return {"err_df": err_df, "stats": detection_stats}


def _build_split_summary_table(err_df: pd.DataFrame) -> str:
    rows = []
    for (sat, kpt), grp in err_df.groupby(["satellite", "keypoint_idx"]):
        interior = grp[~grp["edge_clipped"]]["error_px"]
        rows.append({
            "Satellite Sensor": sat,
            "Keypoint": kpt,
            "Matched Samples": len(grp),
            "All Mean (px)": round(float(grp["error_px"].mean()), 2),
            "All Median (px)": round(float(grp["error_px"].median()), 2),
            "Interior Median (px)": round(float(interior.median()), 2) if len(interior) > 0 else np.nan,
            "90th Pct (px)": round(float(np.percentile(grp["error_px"], 90)), 2),
        })
    summary = pd.DataFrame(rows)
    return summary.to_html(index=False, classes="metrics-table")


def export_interactive_html(
    df: pd.DataFrame,
    eval_reports: dict[str, dict[str, Any]],
    out_dir: Path,
    conf_thresh: float = 0.55,
    edge_margin: int = 48,
) -> None:
    """Exports a light, elegant interactive Plotly HTML dashboard with KPI summary cards."""
    val_rle_raw = df["val/rle_loss"].tolist() if "val/rle_loss" in df else []
    val_rle_valid = val_rle_raw if any(abs(x) > 1e-8 for x in val_rle_raw) else []

    payload = {
        "epochs": df["epoch"].tolist(),
        "train_box": df["train/box_loss"].tolist() if "train/box_loss" in df else [],
        "val_box": df["val/box_loss"].tolist() if "val/box_loss" in df else [],
        "train_rle": df["train/rle_loss"].tolist() if "train/rle_loss" in df else [],
        "val_rle": val_rle_valid,
        "train_pose": df["train/pose_loss"].tolist() if "train/pose_loss" in df else [],
        "val_pose": df["val/pose_loss"].tolist() if "val/pose_loss" in df else [],
        "pose_map50": df["metrics/mAP50(P)"].tolist() if "metrics/mAP50(P)" in df else [],
        "pose_map50_95": df["metrics/mAP50-95(P)"].tolist() if "metrics/mAP50-95(P)" in df else [],
        "box_map50_95": df["metrics/mAP50-95(B)"].tolist() if "metrics/mAP50-95(B)" in df else [],
    }

    best_pose = f"{df['metrics/mAP50-95(P)'].max():.4f}" if "metrics/mAP50-95(P)" in df else "N/A"
    best_box = f"{df['metrics/mAP50-95(B)'].max():.4f}" if "metrics/mAP50-95(B)" in df else "N/A"
    total_ep = int(df["epoch"].max()) if "epoch" in df and not df.empty else 0

    primary_split = "test" if "test" in eval_reports and not eval_reports["test"]["err_df"].empty else next(
        (k for k, v in eval_reports.items() if not v["err_df"].empty), None
    )
    center_med = "N/A"
    center_label = "Median Cyclone Center (K0) Error"
    if primary_split is not None:
        sp_df = eval_reports[primary_split]["err_df"]
        k0_rows = sp_df[sp_df["kpt_order"] == 0]
        if not k0_rows.empty:
            int_k0 = k0_rows[~k0_rows["edge_clipped"]]
            all_med = k0_rows["error_px"].median()
            int_med = int_k0["error_px"].median() if not int_k0.empty else all_med
            center_med = f"{int_med:.2f} px <span style='font-size:13px;color:#64748B;'>(all: {all_med:.2f})</span>"
            center_label = f"Interior Center K0 Median ({primary_split.upper()})"

    tables_sections = []
    for sp_name, report in eval_reports.items():
        sp_df = report["err_df"]
        st = report["stats"]
        if sp_df.empty:
            continue
        badge_html = (
            f"<div style='font-size:13px;color:#475569;margin-bottom:8px;'>"
            f"<b>Operational Detection (conf ≥ {conf_thresh:.2f}):</b> "
            f"Precision = {st['precision']:.3f} | Recall = {st['recall']:.3f} | F1 = {st['f1']:.3f} • "
            f"<b>Background Tile False Alarms:</b> {st['bg_fp_detections']} detections across "
            f"{st['bg_tiles_with_fp']}/{st['bg_tiles_total']} background tiles ({st['bg_tile_fp_rate_pct']:.1f}%)"
            f"</div>"
        )
        tables_sections.append(
            f"""<div class="card" style="margin-top:18px;">
              <h3 style="margin-top:0; margin-bottom:6px; color:#0F172A;">Per-Satellite & Keypoint Localization Accuracy ({sp_name.upper()} Set • Interior Margin ≥ {edge_margin}px)</h3>
              {badge_html}
              {_build_split_summary_table(sp_df)}
            </div>"""
        )

    if not tables_sections:
        tables_sections.append(
            """<div class="card" style="margin-top:18px;">
              <p style='color:#64748B; margin:0;'>Run spatial evaluation on validation/test tiles to populate sensor error breakdown.</p>
            </div>"""
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>YOLO26-Pose Cyclone Telemetry Dashboard</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    body {{ background: #F8FAFC; color: #0F172A; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 28px; }}
    h1 {{ font-size: 24px; margin-bottom: 4px; color: #0F172A; }}
    .sub {{ color: #475569; margin-bottom: 22px; font-size: 14px; }}
    .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; margin-bottom: 20px; }}
    .kpi {{ background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
    .kpi-label {{ font-size: 12px; text-transform: uppercase; letter-spacing: 0.04em; color: #64748B; font-weight: 600; }}
    .kpi-val {{ font-size: 24px; font-weight: 700; margin-top: 4px; color: #0F172A; }}
    .grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; }}
    .card {{ background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 10px; padding: 16px; box-shadow: 0 1px 2px rgba(15,23,42,0.04); }}
    .metrics-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; font-size: 13.5px; }}
    .metrics-table th, .metrics-table td {{ border: 1px solid #E2E8F0; padding: 9px 12px; text-align: left; }}
    .metrics-table th {{ background: #F1F5F9; color: #0F172A; font-weight: 600; }}
    .metrics-table tr:nth-child(even) {{ background: #F8FAFC; }}
  </style>
</head>
<body>
  <h1>YOLO26-Pose Tropical Cyclone Telemetry</h1>
  <div class="sub">Interactive loss convergence, validation mAP accuracy, and geostationary satellite center localization audit</div>
  <div class="kpis">
    <div class="kpi"><div class="kpi-label">Peak Val Pose mAP@50-95</div><div class="kpi-val" style="color:#059669;">{best_pose}</div></div>
    <div class="kpi"><div class="kpi-label">Peak Val Box mAP@50-95</div><div class="kpi-val" style="color:#1D4ED8;">{best_box}</div></div>
    <div class="kpi"><div class="kpi-label">{center_label}</div><div class="kpi-val" style="color:#E11D48;">{center_med}</div></div>
    <div class="kpi"><div class="kpi-label">Completed Epochs</div><div class="kpi-val">{total_ep}</div></div>
  </div>
  <div class="grid">
    <div class="card"><div id="lossPlot" style="height:370px;"></div></div>
    <div class="card"><div id="mapPlot" style="height:370px;"></div></div>
  </div>
  {''.join(tables_sections)}
  <script>
    const d = {json.dumps(payload)};
    const layoutBase = {{
      paper_bgcolor: '#FFFFFF', plot_bgcolor: '#FFFFFF',
      font: {{ color: '#0F172A' }}, margin: {{ t: 48, r: 58, b: 48, l: 58 }},
      xaxis: {{ title: 'Training Epoch', gridcolor: '#E2E8F0', linecolor: '#CBD5E1' }},
      hovermode: 'x unified'
    }};
    const lossTraces = [
      {{ x: d.epochs, y: d.train_pose, name: 'Train Pose Loss', line: {{ color: '#D97706', width: 2.3 }} }},
      {{ x: d.epochs, y: d.val_pose, name: 'Val Pose Loss', line: {{ color: '#B45309', dash: 'dash', width: 2.3 }} }},
      {{ x: d.epochs, y: d.train_rle, name: 'Train RLE Loss (Right)', yaxis: 'y2', line: {{ color: '#E11D48', width: 2.2 }} }},
      {{ x: d.epochs, y: d.val_rle, name: 'Val RLE Loss (Right)', yaxis: 'y2', line: {{ color: '#9F1239', dash: 'dash', width: 2.2 }} }}
    ].filter(t => t.y && t.y.length > 0);
    Plotly.newPlot('lossPlot', lossTraces, {{
      ...layoutBase,
      title: 'Pose Localization & RLE Uncertainty Losses (Dual-Axis)',
      yaxis: {{ title: 'Pose Loss', gridcolor: '#E2E8F0', linecolor: '#CBD5E1' }},
      yaxis2: {{ title: 'RLE Loss', overlaying: 'y', side: 'right', showgrid: false, tickfont: {{ color: '#E11D48' }}, titlefont: {{ color: '#E11D48' }} }}
    }});

    const mapTraces = [
      {{ x: d.epochs, y: d.pose_map50, name: 'Pose mAP@50', line: {{ color: '#059669', width: 2.5 }} }},
      {{ x: d.epochs, y: d.pose_map50_95, name: 'Pose mAP@50-95', line: {{ color: '#0D9488', dash: 'dash', width: 2.5 }} }},
      {{ x: d.epochs, y: d.box_map50_95, name: 'Box mAP@50-95', line: {{ color: '#1D4ED8', dash: 'dot', width: 2.2 }} }}
    ].filter(t => t.y && t.y.length > 0);
    Plotly.newPlot('mapPlot', mapTraces, {{ ...layoutBase, title: 'Validation mAP Progression (Higher = Better)', yaxis: {{ title: 'mAP (0–1)', range: [0, 1.05], gridcolor: '#E2E8F0' }} }});
  </script>
</body>
</html>"""

    html_path = out_dir / "dashboard.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[Saved] {html_path}")


def _resolve_split_dirs(data_dir: Path, split: str) -> tuple[Path, Path]:
    """Resolves (images_dir, labels_dir) from either a dataset root or cyclone_pose.yaml."""
    if data_dir.is_file() and data_dir.suffix in {".yaml", ".yml"}:
        cfg = yaml.safe_load(data_dir.read_text(encoding="utf-8"))
        raw_root = Path(cfg.get("path", data_dir.parent))
        root = raw_root if raw_root.is_absolute() else (data_dir.parent / raw_root).resolve()
        rel_img = cfg.get(split, f"images/{split}")
        img_dir = Path(rel_img) if Path(rel_img).is_absolute() else (root / rel_img).resolve()
    else:
        root = data_dir
        img_dir = root / "images" / split

    parts = list(img_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return img_dir, Path(*parts)
    return img_dir, root / "labels" / split


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize YOLO26-Pose cyclone training & spatial errors.")
    parser.add_argument("--run-dir", type=Path, default=Path("./runs/cyclone_yolo26"))
    parser.add_argument("--data-dir", type=Path, default=Path("../data/processed/yolo26data"), help="Dataset root or path to cyclone_pose.yaml")
    parser.add_argument("--split", default="both", choices=["test", "val", "both"], help="Evaluate spatial errors on val, test, or both")
    parser.add_argument("--conf", type=float, default=0.55, help="Operational confidence threshold for spatial diagnostics")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size for spatial diagnostics")
    parser.add_argument("--edge-margin", type=int, default=48, help="Tile border margin (px) to flag truncated storms")
    args = parser.parse_args()

    run_dir = resolve_path_smart(args.run_dir)
    data_dir = resolve_path_smart(args.data_dir)

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing {csv_path}. Run training first.")

    out_dir = run_dir / "viz"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    plot_training_telemetry(df, out_dir)

    best_weights = run_dir / "weights" / "best.pt"
    eval_reports: dict[str, dict[str, Any]] = {}

    if best_weights.exists():
        target_splits = ["val", "test"] if args.split == "both" else [args.split]
        for sp in target_splits:
            img_dir, lbl_dir = _resolve_split_dirs(data_dir, sp)
            if img_dir.exists():
                eval_reports[sp] = evaluate_and_plot_spatial_diagnostics(
                    weights_path=best_weights,
                    images_dir=img_dir,
                    labels_dir=lbl_dir,
                    out_dir=out_dir,
                    split_name=sp,
                    conf_thresh=args.conf,
                    batch_size=args.batch,
                    edge_margin=args.edge_margin,
                )
            else:
                print(f"[Info] Skipping '{sp}' spatial diagnostics: {img_dir} not found.")

    export_interactive_html(df, eval_reports, out_dir, conf_thresh=args.conf, edge_margin=args.edge_margin)


if __name__ == "__main__":
    main()
