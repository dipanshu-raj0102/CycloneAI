from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from ultralytics import YOLO

# Ensure sibling modules (e.g., visualize_training.py) resolve regardless of CWD
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    RICH_AVAILABLE = True
    console = Console()
except ImportError:
    RICH_AVAILABLE = False
    console = None


@dataclass
class TrainConfig:
    data_yaml: str = "../data/processed/yolo26data/cyclone_pose.yaml"
    model_weights: str = "yolo26n-pose.pt"  # yolo26{n,s,m,l,x}-pose.pt
    epochs: int = 100
    imgsz: int = 640
    batch: int = 128
    device: str = "0"
    optimizer: str = "AdamW"  # AdamW + cosine decay prevents early memorization on overlapping satellite tiles
    lr0: float | None = 0.001
    lrf: float = 0.01
    cos_lr: bool = True
    warmup_epochs: float = 5.0
    patience: int = 25
    weight_decay: float = 0.001
    dropout: float = 0.10
    erasing: float = 0.15    # Simulates partial IR scan dropouts and prevents single-patch memorization
    box_gain: float = 9.0    # Tightens cyclone extent box around core
    pose_gain: float = 14.0  # Emphasizes sub-pixel K0 eye localization
    cls_gain: float = 0.8    # Penalizes background ITCZ/convective cloud false positives
    workers: int = min(8, os.cpu_count() or 4)
    seed: int = 42
    project: str = "./runs"
    run_name: str = "cyclone_yolo26"
    resume: bool = False
    viz: bool = True         # Generate full telemetry & spatial plots once after training finishes
    viz_split: str = "both"  # "test", "val", or "both"
    viz_conf: float = 0.55   # F1-optimal operational confidence threshold (0.486-0.70 plateau)
    viz_batch: int = 16
    edge_margin: int = 48    # Pixel margin from tile border to flag boundary-truncated cyclones
    batch_log_interval: int = 25


def _fmt_duration(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _infer_satellite(name: str) -> str:
    lower = name.lower()
    for sat in ("goes_east", "goes_west", "himawari"):
        if sat in lower:
            return sat
    return "other"


def resolve_yaml_path(raw_path: str) -> Path:
    """Resolves dataset YAML relative to CWD first, falling back to script directory."""
    candidate = Path(raw_path)
    if candidate.exists():
        return candidate.resolve()
    script_relative = (SCRIPT_DIR / candidate).resolve()
    if script_relative.exists():
        return script_relative
    raise FileNotFoundError(
        f"Dataset YAML not found at '{candidate.resolve()}' or '{script_relative}'."
    )


def resolve_project_path(raw_project: str) -> Path:
    """Resolves output project directory consistently relative to CWD."""
    p = Path(raw_project)
    return p if p.is_absolute() else p.resolve()


def _resolve_split_img_dir(root: Path, cfg_entry: str | None, default_rel: str) -> Path | None:
    rel = cfg_entry or default_rel
    candidate = Path(rel)
    if candidate.is_absolute():
        return candidate if candidate.exists() else None
    resolved = (root / candidate).resolve()
    return resolved if resolved.exists() else None


def _labels_dir_for_images(root: Path, img_dir: Path) -> Path:
    """Maps .../images/<split> -> .../labels/<split> cleanly."""
    parts = list(img_dir.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == "images":
            parts[idx] = "labels"
            return Path(*parts)
    return root / "labels" / img_dir.name


def _inspect_split(
    split_name: str,
    img_dir: Path,
    lbl_dir: Path,
    expected_cols: int,
    kpt_shape: list[int],
    imgsz: int,
    edge_margin: int,
) -> dict[str, Any]:
    imgs = sorted(img_dir.glob("*.png"))
    sat_counts: dict[str, int] = {}
    positive_tiles = 0
    total_cyclones = 0
    edge_clipped_cyclones = 0
    norm_margin = edge_margin / max(1, imgsz)

    for img_p in imgs:
        sat = _infer_satellite(img_p.name)
        sat_counts[sat] = sat_counts.get(sat, 0) + 1
        lbl_p = lbl_dir / f"{img_p.stem}.txt"
        if lbl_p.exists():
            lines = [ln.strip() for ln in lbl_p.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if lines:
                positive_tiles += 1
                for line_num, ln in enumerate(lines, start=1):
                    parts = ln.split()
                    cols = len(parts)
                    if cols != expected_cols:
                        raise ValueError(
                            f"[{split_name}] Label column mismatch in {lbl_p.name}:{line_num}: "
                            f"found {cols} values, but kpt_shape={kpt_shape} requires {expected_cols} columns."
                        )
                    total_cyclones += 1
                    k0_x, k0_y = float(parts[5]), float(parts[6])
                    if (
                        k0_x < norm_margin
                        or k0_x > (1.0 - norm_margin)
                        or k0_y < norm_margin
                        or k0_y > (1.0 - norm_margin)
                    ):
                        edge_clipped_cyclones += 1

    return {
        "images_dir": str(img_dir),
        "labels_dir": str(lbl_dir),
        "total": len(imgs),
        "positive": positive_tiles,
        "background": len(imgs) - positive_tiles,
        "total_cyclones": total_cyclones,
        "edge_clipped": edge_clipped_cyclones,
        "satellites": sat_counts,
    }


def validate_and_inspect_dataset(yaml_path: Path, imgsz: int, edge_margin: int) -> dict[str, Any]:
    """Verifies YAML paths, kpt_shape consistency across train/val/test, and summarizes tiles."""
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    raw_root = Path(cfg.get("path", yaml_path.parent))
    if raw_root.is_absolute():
        root = raw_root
    elif (yaml_path.parent / raw_root).exists():
        root = (yaml_path.parent / raw_root).resolve()
    else:
        root = raw_root.resolve()

    train_dir = _resolve_split_img_dir(root, cfg.get("train"), "images/train")
    val_dir = _resolve_split_img_dir(root, cfg.get("val"), "images/val")
    test_dir = _resolve_split_img_dir(root, cfg.get("test"), "images/test")

    if train_dir is None:
        raise FileNotFoundError(f"Training image directory missing under {root} (checked '{cfg.get('train', 'images/train')}').")
    if val_dir is None:
        raise FileNotFoundError(f"Validation image directory missing under {root} (checked '{cfg.get('val', 'images/val')}').")

    kpt_shape = cfg.get("kpt_shape")
    if not kpt_shape or len(kpt_shape) != 2:
        raise ValueError(f"Invalid or missing `kpt_shape` in {yaml_path}. Expected e.g. [5, 3], got {kpt_shape}")

    num_kpts, kpt_dims = int(kpt_shape[0]), int(kpt_shape[1])
    expected_cols = 5 + num_kpts * kpt_dims

    splits: dict[str, dict[str, Any]] = {
        "train": _inspect_split(
            "train", train_dir, _labels_dir_for_images(root, train_dir), expected_cols, kpt_shape, imgsz, edge_margin
        ),
        "val": _inspect_split(
            "val", val_dir, _labels_dir_for_images(root, val_dir), expected_cols, kpt_shape, imgsz, edge_margin
        ),
    }
    if test_dir is not None:
        splits["test"] = _inspect_split(
            "test", test_dir, _labels_dir_for_images(root, test_dir), expected_cols, kpt_shape, imgsz, edge_margin
        )

    return {
        "root": str(root),
        "yaml_cfg": cfg,
        "kpt_shape": kpt_shape,
        "splits": splits,
    }


def _fmt_split_summary(s: dict[str, Any]) -> str:
    bg_pct = 100.0 * s["background"] / max(1, s["total"])
    edge_pct = 100.0 * s["edge_clipped"] / max(1, s["total_cyclones"])
    return (
        f"{s['total']} tiles ({s['positive']} cyclone, {s['background']} bg [{bg_pct:.0f}%]) | "
        f"{s['total_cyclones']} storms ({s['edge_clipped']} edge-clipped, {edge_pct:.1f}%)"
    )


def print_preflight_summary(
    cfg: TrainConfig, ds_info: dict[str, Any], resolved_opt: str, resolved_lr0: float, run_dir: Path
) -> None:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and cfg.device != "cpu" else "CPU"
    vram_gb = (
        f"{torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f} GB"
        if torch.cuda.is_available() and cfg.device != "cpu"
        else "N/A"
    )
    splits = ds_info["splits"]
    train_sats = set(splits["train"]["satellites"].keys())
    sat_str = ", ".join(f"{k}: {v}" for k, v in sorted(splits["train"]["satellites"].items()))
    test_str = _fmt_split_summary(splits["test"]) if "test" in splits else "Not configured"

    alerts: list[str] = []
    for sp_name in ("val", "test"):
        if sp_name in splits:
            missing = sorted(train_sats - set(splits[sp_name]["satellites"].keys()))
            if missing:
                alerts.append(f"{sp_name.upper()} missing {', '.join(missing)}")

    train_bg_pct = 100.0 * splits["train"]["background"] / max(1, splits["train"]["total"])
    if train_bg_pct < 20.0:
        alerts.append(f"Train background share is low ({train_bg_pct:.1f}%; target 25-35% hard negatives)")

    if RICH_AVAILABLE:
        table = Table(title="YOLO26-Pose Cyclone Training — Pre-Flight Audit", border_style="cyan", show_header=True)
        table.add_column("Parameter", style="bold cyan", width=24)
        table.add_column("Value", style="white")
        table.add_row("Compute Device", f"{gpu_name} (VRAM: {vram_gb}) | PyTorch {torch.__version__}")
        table.add_row("Dataset Root", ds_info["root"])
        table.add_row("Output Directory", str(run_dir))
        table.add_row("Base Checkpoint", f"{cfg.model_weights} | Image Size: {cfg.imgsz}px | Batch: {cfg.batch}")
        table.add_row(
            "Optimizer & Schedule",
            f"{resolved_opt} (lr0={resolved_lr0}, lrf={cfg.lrf}, cos_lr={cfg.cos_lr}, wd={cfg.weight_decay})",
        )
        table.add_row(
            "Loss & Regularization",
            f"box={cfg.box_gain} | pose={cfg.pose_gain} | cls={cfg.cls_gain} | dropout={cfg.dropout} | erasing={cfg.erasing}",
        )
        table.add_row("Epochs & Patience", f"{cfg.epochs} epochs (early-stop patience={cfg.patience})")
        table.add_row("Train Split", _fmt_split_summary(splits["train"]))
        table.add_row("Validation Split", _fmt_split_summary(splits["val"]))
        table.add_row("Held-Out Test Split", test_str)
        table.add_row("Satellite Mix (Train)", sat_str or "N/A")
        if alerts:
            table.add_row("Dataset Audit Alerts", f"[yellow]{'; '.join(alerts)}[/yellow]")
        table.add_row("Keypoint Schema", f"kpt_shape = {ds_info['kpt_shape']} | Chirality-Safe (fliplr=0, flipud=0)")
        console.print(table)
    else:
        print("=" * 78)
        print(f"YOLO26-POSE CYCLONE PRE-FLIGHT | Device: {gpu_name} ({vram_gb})")
        print(f"Dataset Root: {ds_info['root']} | Output Dir: {run_dir}")
        print(f"Model: {cfg.model_weights} | Imgsz: {cfg.imgsz} | Batch: {cfg.batch} | Opt: {resolved_opt} (lr0={resolved_lr0})")
        print(f"Gains/Reg: box={cfg.box_gain}, pose={cfg.pose_gain}, cls={cfg.cls_gain}, wd={cfg.weight_decay}, dropout={cfg.dropout}")
        print(f"Train: {_fmt_split_summary(splits['train'])}")
        print(f"Val  : {_fmt_split_summary(splits['val'])}")
        print(f"Test : {test_str}")
        print(f"Satellites (Train): {sat_str}")
        if alerts:
            print(f"[Warn] Dataset Audit Alerts: {'; '.join(alerts)}")
        print("=" * 78)


class CycloneProgressTracker:
    """Tracks intra-epoch speed/VRAM, epoch-over-epoch deltas, and best checkpoints."""

    def __init__(self, cfg: TrainConfig) -> None:
        self.cfg = cfg
        self.train_start_time = time.time()
        self.epoch_start_time = time.time()
        self.best_pose_map50_95 = -1.0
        self.best_epoch = 0
        self.epochs_without_improvement = 0
        self.prev_metrics: dict[str, float] = {}

    def on_train_start(self, trainer: Any) -> None:
        self.train_start_time = time.time()
        save_dir = Path(trainer.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        csv_path = save_dir / "results.csv"
        if self.cfg.resume and csv_path.exists():
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]
            if "metrics/mAP50-95(P)" in df.columns and not df.empty:
                scores = df["metrics/mAP50-95(P)"].fillna(-1.0)
                best_idx = int(scores.idxmax())
                self.best_pose_map50_95 = float(scores.loc[best_idx])
                self.best_epoch = int(df.loc[best_idx, "epoch"])
                self.epochs_without_improvement = max(0, len(df) - 1 - best_idx)

        (save_dir / "run_config.json").write_text(json.dumps(asdict(self.cfg), indent=2), encoding="utf-8")

    def on_train_epoch_start(self, trainer: Any) -> None:
        self.epoch_start_time = time.time()

    def on_train_batch_end(self, trainer: Any) -> None:
        batch_idx = getattr(trainer, "batch_i", 0)
        total_batches = len(getattr(trainer, "train_loader", [])) or 1
        step = batch_idx + 1

        if step % self.cfg.batch_log_interval != 0 and step != total_batches:
            return

        tloss = getattr(trainer, "tloss", None)
        loss_names = getattr(trainer, "loss_names", ())
        loss_str = ""
        if tloss is not None and len(loss_names) > 0:
            vals = tloss.detach().cpu().tolist() if hasattr(tloss, "detach") else list(tloss)
            loss_str = " | ".join(f"{n}: {v:.4f}" for n, v in zip(loss_names, vals))

        vram = f"{torch.cuda.memory_reserved() / (1024**3):.2f}GB" if torch.cuda.is_available() else "CPU"
        pct = 100.0 * step / total_batches
        ep = getattr(trainer, "epoch", 0) + 1
        tot_ep = getattr(trainer, "epochs", self.cfg.epochs)

        print(f"  [Epoch {ep:03d}/{tot_ep:03d} | {pct:5.1f}% ({step}/{total_batches}) | VRAM {vram}] {loss_str}", flush=True)

    def _delta_tag(self, key: str, current: float, higher_is_better: bool = True) -> str:
        if key not in self.prev_metrics:
            return ""
        diff = current - self.prev_metrics[key]
        if abs(diff) < 1e-5:
            return " (0.000)"
        improved = (diff > 0) if higher_is_better else (diff < 0)
        arrow = "▲" if diff > 0 else "▼"
        if RICH_AVAILABLE:
            color = "green" if improved else "yellow"
            return f" [{color}]({arrow}{abs(diff):.4f})[/{color}]"
        return f" ({arrow}{abs(diff):.4f})"

    def _extract_epoch_row(self, trainer: Any, save_dir: Path) -> dict[str, Any]:
        """Reads metrics from trainer memory first, falling back to results.csv."""
        row: dict[str, Any] = {}
        metrics = getattr(trainer, "metrics", None)
        if isinstance(metrics, dict) and metrics:
            row.update(metrics)
        lr_dict = getattr(trainer, "lr", None)
        if isinstance(lr_dict, dict):
            row.update(lr_dict)

        if "metrics/mAP50-95(P)" not in row:
            csv_path = save_dir / "results.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df.columns = [c.strip() for c in df.columns]
                if not df.empty:
                    row.update(df.iloc[-1].to_dict())
        return row

    def on_fit_epoch_end(self, trainer: Any) -> None:
        save_dir = Path(trainer.save_dir)
        row = self._extract_epoch_row(trainer, save_dir)
        if not row:
            return

        epoch = int(row.get("epoch", getattr(trainer, "epoch", 0) + 1))
        total_epochs = int(getattr(trainer, "epochs", self.cfg.epochs))

        box_loss = float(row.get("val/box_loss", row.get("train/box_loss", 0.0)))
        pose_loss = float(row.get("val/pose_loss", row.get("train/pose_loss", 0.0)))
        cls_loss = float(row.get("val/cls_loss", row.get("train/cls_loss", 0.0)))
        box_map50 = float(row.get("metrics/mAP50(B)", 0.0))
        box_map = float(row.get("metrics/mAP50-95(B)", 0.0))
        pose_map50 = float(row.get("metrics/mAP50(P)", 0.0))
        pose_map = float(row.get("metrics/mAP50-95(P)", 0.0))
        prec_p = float(row.get("metrics/precision(P)", 0.0))
        rec_p = float(row.get("metrics/recall(P)", 0.0))
        f1_p = 2 * prec_p * rec_p / (prec_p + rec_p + 1e-8)
        lr_now = float(row.get("lr/pg0", 0.0))

        is_new_best = pose_map > self.best_pose_map50_95
        if is_new_best:
            self.best_pose_map50_95 = pose_map
            self.best_epoch = epoch
            self.epochs_without_improvement = 0
        else:
            self.epochs_without_improvement += 1

        ep_sec = time.time() - self.epoch_start_time
        elapsed_sec = time.time() - self.train_start_time
        avg_ep_sec = elapsed_sec / max(1, epoch)
        eta_sec = avg_ep_sec * max(0, total_epochs - epoch)

        self._render_epoch_summary(
            epoch=epoch,
            total_epochs=total_epochs,
            ep_sec=ep_sec,
            eta_sec=eta_sec,
            lr_now=lr_now,
            box_loss=box_loss,
            pose_loss=pose_loss,
            cls_loss=cls_loss,
            box_map50=box_map50,
            box_map=box_map,
            pose_map50=pose_map50,
            pose_map=pose_map,
            prec_p=prec_p,
            rec_p=rec_p,
            f1_p=f1_p,
            is_new_best=is_new_best,
        )

        event = {
            "epoch": epoch,
            "epoch_sec": round(ep_sec, 2),
            "eta_sec": round(eta_sec, 2),
            "is_new_best": is_new_best,
            "best_pose_map50_95": round(self.best_pose_map50_95, 5),
            "patience_used": self.epochs_without_improvement,
            **{k: float(v) for k, v in row.items() if isinstance(v, (int, float))},
        }
        with (save_dir / "training_events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")

        self.prev_metrics = {
            "box_loss": box_loss,
            "pose_loss": pose_loss,
            "cls_loss": cls_loss,
            "box_map": box_map,
            "pose_map50": pose_map50,
            "pose_map": pose_map,
            "f1_p": f1_p,
        }

    def _render_epoch_summary(self, **m: Any) -> None:
        badge = " ★ NEW BEST CHECKPOINT" if m["is_new_best"] else f" (Best: {self.best_pose_map50_95:.4f} @ Ep {self.best_epoch})"
        patience_str = f"{self.epochs_without_improvement}/{self.cfg.patience}"

        if RICH_AVAILABLE:
            tbl = Table(
                title=f"Epoch {m['epoch']:03d}/{m['total_epochs']:03d} Summary | Ep Time: {m['ep_sec']:.1f}s | ETA: {_fmt_duration(m['eta_sec'])}{badge}",
                border_style="bright_green" if m["is_new_best"] else "blue",
                show_header=True,
            )
            tbl.add_column("Metric Group", style="bold")
            tbl.add_column("Primary", justify="right")
            tbl.add_column("Secondary", justify="right")
            tbl.add_column("Context", justify="right")

            tbl.add_row(
                "Val Losses",
                f"Pose: {m['pose_loss']:.4f}{self._delta_tag('pose_loss', m['pose_loss'], False)}",
                f"Box: {m['box_loss']:.4f}{self._delta_tag('box_loss', m['box_loss'], False)}",
                f"Cls: {m['cls_loss']:.4f}{self._delta_tag('cls_loss', m['cls_loss'], False)}",
            )
            tbl.add_row(
                "Pose Keypoints",
                f"mAP@50-95: [bold]{m['pose_map']:.4f}[/bold]{self._delta_tag('pose_map', m['pose_map'], True)}",
                f"mAP@50: {m['pose_map50']:.4f}{self._delta_tag('pose_map50', m['pose_map50'], True)}",
                f"F1: {m['f1_p']:.4f} (P={m['prec_p']:.3f}, R={m['rec_p']:.3f})",
            )
            tbl.add_row(
                "Cyclone Box & LR",
                f"Box mAP@50-95: {m['box_map']:.4f}{self._delta_tag('box_map', m['box_map'], True)}",
                f"Box mAP@50: {m['box_map50']:.4f}",
                f"LR: {m['lr_now']:.6f} | Patience: {patience_str}",
            )
            console.print(tbl)
        else:
            print(
                f"── Epoch {m['epoch']:03d}/{m['total_epochs']:03d} ({m['ep_sec']:.1f}s | ETA {_fmt_duration(m['eta_sec'])}){badge} ──\n"
                f"   Losses : ValPose={m['pose_loss']:.4f} | ValBox={m['box_loss']:.4f} | ValCls={m['cls_loss']:.4f}\n"
                f"   Pose   : mAP@50-95={m['pose_map']:.4f} | mAP@50={m['pose_map50']:.4f} | F1={m['f1_p']:.4f}\n"
                f"   Box/LR : Box mAP@50-95={m['box_map']:.4f} | LR={m['lr_now']:.6f} | Patience={patience_str}"
            )


def _ensure_resolved_yaml_with_all_splits(yaml_path: Path, ds_info: dict[str, Any]) -> tuple[Path, Path | None]:
    """Ensures YAML has explicit absolute root and `test` entry when images/test exists."""
    cfg = dict(ds_info["yaml_cfg"])
    splits = ds_info["splits"]
    needs_temp = False

    if not Path(str(cfg.get("path", ""))).is_absolute():
        cfg["path"] = ds_info["root"]
        needs_temp = True
    if "test" in splits and "test" not in cfg:
        cfg["test"] = str(Path(splits["test"]["images_dir"]).relative_to(ds_info["root"]))
        needs_temp = True

    if not needs_temp:
        return yaml_path, None

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix="_cyclone_pose.yaml", delete=False, encoding="utf-8")
    yaml.safe_dump(cfg, tmp, sort_keys=False)
    tmp.close()
    tmp_path = Path(tmp.name)
    return tmp_path, tmp_path


def generate_final_visualizations(
    run_dir: Path,
    splits_info: dict[str, dict[str, Any]],
    best_weights: Path,
    viz_split: str = "both",
    conf_thresh: float = 0.55,
    batch_size: int = 16,
    edge_margin: int = 48,
) -> None:
    """Generates telemetry plots, per-split spatial diagnostics, and interactive HTML."""
    from visualize_training import (
        evaluate_and_plot_spatial_diagnostics,
        export_interactive_html,
        plot_training_telemetry,
    )

    csv_path = run_dir / "results.csv"
    if not csv_path.exists():
        print(f"[Warn] Skipping final plots: {csv_path} not found.")
        return

    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    plot_training_telemetry(df, viz_dir)

    if viz_split == "both":
        target_splits = [s for s in ("val", "test") if s in splits_info]
    elif viz_split in splits_info:
        target_splits = [viz_split]
    else:
        target_splits = ["test"] if "test" in splits_info else ["val"]

    eval_reports: dict[str, dict[str, Any]] = {}
    if best_weights.exists():
        for sp in target_splits:
            sp_data = splits_info[sp]
            eval_reports[sp] = evaluate_and_plot_spatial_diagnostics(
                weights_path=best_weights,
                images_dir=Path(sp_data["images_dir"]),
                labels_dir=Path(sp_data["labels_dir"]),
                out_dir=viz_dir,
                split_name=sp,
                conf_thresh=conf_thresh,
                batch_size=batch_size,
                edge_margin=edge_margin,
            )

    export_interactive_html(df, eval_reports, viz_dir, conf_thresh=conf_thresh, edge_margin=edge_margin)


def train_cyclone_pose(cfg: TrainConfig) -> Path:
    yaml_path = resolve_yaml_path(cfg.data_yaml)
    ds_info = validate_and_inspect_dataset(yaml_path, imgsz=cfg.imgsz, edge_margin=cfg.edge_margin)
    splits = ds_info["splits"]

    project_abs = resolve_project_path(cfg.project)
    run_dir = project_abs / cfg.run_name

    if cfg.optimizer == "auto":
        resolved_opt = "AdamW"
        resolved_lr0 = cfg.lr0 if cfg.lr0 is not None else 0.001
    else:
        resolved_opt = cfg.optimizer
        resolved_lr0 = cfg.lr0 if cfg.lr0 is not None else (0.01 if resolved_opt == "MuSGD" else 0.001)

    print_preflight_summary(cfg, ds_info, resolved_opt, resolved_lr0, run_dir)
    effective_yaml, temp_yaml = _ensure_resolved_yaml_with_all_splits(yaml_path, ds_info)

    try:
        if cfg.resume:
            last_ckpt = run_dir / "weights" / "last.pt"
            if not last_ckpt.exists():
                raise FileNotFoundError(f"Cannot resume: checkpoint not found at {last_ckpt}")
            model = YOLO(str(last_ckpt))
        else:
            model = YOLO(cfg.model_weights)

        tracker = CycloneProgressTracker(cfg)
        model.add_callback("on_train_start", tracker.on_train_start)
        model.add_callback("on_train_epoch_start", tracker.on_train_epoch_start)
        model.add_callback("on_train_batch_end", tracker.on_train_batch_end)
        model.add_callback("on_fit_epoch_end", tracker.on_fit_epoch_end)

        device_arg: int | str = int(cfg.device) if cfg.device.isdigit() else cfg.device

        model.train(
            data=str(effective_yaml),
            epochs=cfg.epochs,
            imgsz=cfg.imgsz,
            batch=cfg.batch,
            device=device_arg,
            project=str(project_abs),
            name=cfg.run_name,
            exist_ok=True,
            resume=cfg.resume,
            pretrained=True,
            optimizer=resolved_opt,
            lr0=resolved_lr0,
            lrf=cfg.lrf,
            cos_lr=cfg.cos_lr,
            warmup_epochs=cfg.warmup_epochs,
            patience=cfg.patience,
            weight_decay=cfg.weight_decay,
            dropout=cfg.dropout,
            box=cfg.box_gain,
            pose=cfg.pose_gain,
            cls=cfg.cls_gain,
            workers=cfg.workers,
            seed=cfg.seed,
            deterministic=True,
            amp=True,
            save=True,
            save_period=10,
            plots=True,
            verbose=False,
            # Geostationary cyclone domain augmentations (preserves Coriolis spiral chirality)
            fliplr=0.0,
            flipud=0.0,
            mosaic=0.0,
            close_mosaic=0,
            mixup=0.0,
            copy_paste=0.0,
            degrees=15.0,
            translate=0.05,
            scale=0.10,
            erasing=cfg.erasing,
            hsv_h=0.0,
            hsv_s=0.1,
            hsv_v=0.2,
        )

        actual_run_dir = Path(getattr(model.trainer, "save_dir", run_dir))
        best_weights = actual_run_dir / "weights" / "best.pt"

        if best_weights.exists():
            best_model = YOLO(str(best_weights))
            eval_lines = [f"Best Checkpoint: {best_weights} (Operational Audit Conf = {cfg.viz_conf:.2f})"]

            for eval_split in [s for s in ("val", "test") if s in splits]:
                val_std = best_model.val(
                    data=str(effective_yaml),
                    split=eval_split,
                    imgsz=cfg.imgsz,
                    device=device_arg,
                    nms=True,
                    project=str(actual_run_dir),
                    name=f"eval_{eval_split}_nms",
                    exist_ok=True,
                    verbose=False,
                )
                val_e2e = best_model.val(
                    data=str(effective_yaml),
                    split=eval_split,
                    imgsz=cfg.imgsz,
                    device=device_arg,
                    nms=False,
                    project=str(actual_run_dir),
                    name=f"eval_{eval_split}_e2e",
                    exist_ok=True,
                    verbose=False,
                )
                label = eval_split.upper()
                eval_lines.append(
                    f"[{label}] Standard Head (nms=True)   -> Box mAP50-95: {val_std.box.map:.4f} | Pose mAP50-95: {val_std.pose.map:.4f}"
                )
                eval_lines.append(
                    f"[{label}] End-to-End Head (nms=False) -> Box mAP50-95: {val_e2e.box.map:.4f} | Pose mAP50-95: {val_e2e.pose.map:.4f}"
                )

            summary_msg = "\n".join(eval_lines)
            if RICH_AVAILABLE:
                console.print(Panel(summary_msg, title="Final Validation & Test Evaluation Complete", border_style="green"))
            else:
                print("\n" + "=" * 78 + "\n" + summary_msg + "\n" + "=" * 78)

        if cfg.viz:
            try:
                generate_final_visualizations(
                    run_dir=actual_run_dir,
                    splits_info=splits,
                    best_weights=best_weights,
                    viz_split=cfg.viz_split,
                    conf_thresh=cfg.viz_conf,
                    batch_size=cfg.viz_batch,
                    edge_margin=cfg.edge_margin,
                )
            except Exception as exc:
                print(f"[Warn] Post-training visualization skipped: {exc}")

        return actual_run_dir
    finally:
        if temp_yaml is not None and temp_yaml.exists():
            temp_yaml.unlink(missing_ok=True)


def parse_args() -> TrainConfig:
    defaults = TrainConfig()
    parser = argparse.ArgumentParser(description="Train YOLO26-Pose on geostationary tropical cyclone tiles.")
    parser.add_argument("--data", dest="data_yaml", default=defaults.data_yaml)
    parser.add_argument("--weights", dest="model_weights", default=defaults.model_weights)
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--imgsz", type=int, default=defaults.imgsz)
    parser.add_argument("--batch", type=int, default=defaults.batch)
    parser.add_argument("--device", default=defaults.device)
    parser.add_argument("--optimizer", default=defaults.optimizer, choices=["auto", "AdamW", "MuSGD", "SGD"])
    parser.add_argument("--lr0", type=float, default=defaults.lr0)
    parser.add_argument("--no-cos-lr", dest="cos_lr", action="store_false", help="Disable cosine learning-rate schedule")
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--erasing", type=float, default=defaults.erasing)
    parser.add_argument("--box-gain", type=float, default=defaults.box_gain, help="Bounding-box loss gain")
    parser.add_argument("--pose-gain", type=float, default=defaults.pose_gain, help="Keypoint pose loss gain")
    parser.add_argument("--cls-gain", type=float, default=defaults.cls_gain, help="Classification loss gain")
    parser.add_argument("--project", default=defaults.project)
    parser.add_argument("--name", dest="run_name", default=defaults.run_name)
    parser.add_argument("--resume", action="store_true", help="Resume training from last.pt")
    parser.add_argument("--viz-split", default=defaults.viz_split, choices=["test", "val", "both"], help="Dataset split(s) for spatial diagnostics")
    parser.add_argument("--viz-conf", type=float, default=defaults.viz_conf, help="Operational confidence threshold for spatial diagnostics")
    parser.add_argument("--viz-batch", type=int, default=defaults.viz_batch, help="Inference batch size for spatial diagnostics")
    parser.add_argument("--edge-margin", type=int, default=defaults.edge_margin, help="Tile border margin (px) to flag truncated storms")
    parser.add_argument("--no-viz", dest="viz", action="store_false", help="Skip post-training plot generation")
    args = parser.parse_args()
    return TrainConfig(**vars(args))


if __name__ == "__main__":
    train_cyclone_pose(parse_args())
