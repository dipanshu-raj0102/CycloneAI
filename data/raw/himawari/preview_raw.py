import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import bz2
from collections import defaultdict
import gzip
from pathlib import Path
import random
import re
import tempfile
import warnings

import dask
import numpy as np
from PIL import Image
from satpy import Scene

warnings.filterwarnings("ignore", category=RuntimeWarning)
dask.config.set(scheduler="synchronous")

HSD_PATTERN = re.compile(
    r"^(HS_H0[89]_(\d{8})_(\d{4})_(B\d{2})_FLDK_R\d{2})_S(\d{2})(\d{2})\.DAT(?:\.bz2|\.gz)?$",
    re.IGNORECASE,
)
SKIP_DIRS = {".venv", ".git", "__pycache__", "data"}


def discover_scenes(root_dir: Path, min_segments: int = 8) -> list[dict]:
    """Recursively walk YYYY/MM/DD folders and group raw HSD segments by timestamp."""
    grouped: dict[str, dict] = defaultdict(
        lambda: {"files": [], "band": "B13", "total_segs": 10}
    )

    for dirpath, dirnames, filenames in os.walk(root_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fname in filenames:
            if "FLDK" not in fname:
                continue
            m = HSD_PATTERN.match(fname)
            if not m:
                continue
            scene_id, _, _, band, seg_idx, seg_total = m.groups()
            entry = grouped[scene_id]
            entry["files"].append((int(seg_idx), Path(dirpath) / fname))
            entry["band"] = band.upper()
            entry["total_segs"] = int(seg_total)

    valid_scenes = []
    for scene_id, info in grouped.items():
        files = [p for _, p in sorted(info["files"], key=lambda x: x[0])]
        if info["total_segs"] == 1 or len(files) >= min_segments:
            valid_scenes.append(
                {
                    "scene_id": scene_id,
                    "band": info["band"],
                    "seg_count": len(files),
                    "expected_segs": info["total_segs"],
                    "files": files,
                }
            )
    return valid_scenes


def prepare_uncompressed_hsd(filepaths: list[Path], tmpdir: str) -> list[str]:
    """Detect bzip2 (BZh) or gzip magic bytes regardless of filename and unpack to .DAT."""
    prepared = []
    for path in filepaths:
        data = path.read_bytes()
        unpacked = False

        for _ in range(2):  # Handles single or double compression
            if data[:3] == b"BZh":
                data = bz2.decompress(data)
                unpacked = True
            elif data[:2] == b"\x1f\x8b":
                data = gzip.decompress(data)
                unpacked = True
            else:
                break

        if not unpacked:
            prepared.append(str(path))
            continue

        clean_name = re.sub(r"\.(bz2|gz)$", "", path.name, flags=re.IGNORECASE)
        out_path = os.path.join(tmpdir, clean_name)
        with open(out_path, "wb") as f:
            f.write(data)
        prepared.append(out_path)

    return prepared


def render_ir_grayscale(tb: np.ndarray, vmin: float, vmax: float) -> Image.Image:
    norm = np.clip((tb - vmin) / (vmax - vmin), 0.0, 1.0)
    inv = np.nan_to_num(1.0 - norm, nan=0.0)
    return Image.fromarray((inv * 255).astype(np.uint8), mode="L")


def render_ir_enhanced(tb: np.ndarray) -> Image.Image:
    k_pts = np.array([180, 195, 205, 215, 225, 235, 240, 240.1, 310], dtype=np.float32)
    r_pts = np.array([255, 180, 220, 255, 255,  40,  30,   190,  15], dtype=np.float32)
    g_pts = np.array([255,  20,  20, 140, 240, 200,  90,   190,  15], dtype=np.float32)
    b_pts = np.array([255, 180,  20,   0,  30, 220, 200,   190,  15], dtype=np.float32)

    valid = np.isfinite(tb)
    tb_clamped = np.clip(np.where(valid, tb, 310.0), 180.0, 310.0)

    rgb = np.stack(
        [
            np.interp(tb_clamped, k_pts, r_pts),
            np.interp(tb_clamped, k_pts, g_pts),
            np.interp(tb_clamped, k_pts, b_pts),
        ],
        axis=-1,
    ).astype(np.uint8)
    rgb[~valid] = 0
    return Image.fromarray(rgb, mode="RGB")


def generate_preview(
    scene_info: dict,
    out_dir: Path,
    color: bool,
    max_dim: int | None,
    vmin: float,
    vmax: float,
):
    with tempfile.TemporaryDirectory(prefix="hsd_preview_") as tmpdir:
        unpacked = prepare_uncompressed_hsd(scene_info["files"], tmpdir)
        scn = Scene(filenames=unpacked, reader="ahi_hsd")
        scn.load([scene_info["band"]], calibration="brightness_temperature", pad_data=True)
        tb = scn[scene_info["band"]].values.astype(np.float32)

    valid = np.isfinite(tb)
    if not valid.any():
        raise RuntimeError(f"All pixels are NaN for {scene_info['scene_id']}")

    img = render_ir_enhanced(tb) if color else render_ir_grayscale(tb, vmin, vmax)
    if max_dim:
        img.thumbnail((max_dim, max_dim), Image.Resampling.BILINEAR)

    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_ir_color.png" if color else "_ir.png"
    out_path = out_dir / f"{scene_info['scene_id']}{suffix}"
    img.save(out_path)

    print(f"Scene:    {scene_info['scene_id']} ({scene_info['seg_count']}/{scene_info['expected_segs']} segments)")
    print(f"Shape:    {tb.shape} -> Saved {img.size[0]}x{img.size[1]} | Valid: {100.0 * valid.mean():.1f}%")
    print(f"Temp:     {float(np.nanmin(tb)):.2f} K – {float(np.nanmax(tb)):.2f} K")
    print(f"Saved to: {out_path}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly pick a raw Himawari timestamp and render an IR PNG preview."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Root folder containing YYYY/MM/DD (default: .)",
    )
    parser.add_argument(
        "-n", "--count", type=int, default=10, help="Number of random timestamps to preview (default: 1)"
    )
    parser.add_argument(
        "-o", "--out-dir", type=Path, default=Path("previews"), help="Output folder for PNGs (default: previews/)"
    )
    parser.add_argument(
        "--color", action="store_true", help="Use color-enhanced IR (< 240 K highlighted)"
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=1600,
        help="Resize longest edge to N px (0 for full 5500x5500; default: 1600)",
    )
    parser.add_argument("--vmin", type=float, default=180.0, help="Coldest Kelvin clamp (default: 180)")
    parser.add_argument("--vmax", type=float, default=310.0, help="Warmest Kelvin clamp (default: 310)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility")
    args = parser.parse_args()

    scenes = discover_scenes(args.root_dir)
    if not scenes:
        raise FileNotFoundError(f"No valid Himawari HSD timestamps found under {args.root_dir.resolve()}")

    if args.seed is not None:
        random.seed(args.seed)

    picks = random.sample(scenes, k=min(args.count, len(scenes)))
    print(f"Found {len(scenes)} valid timestamps. Randomly selected {len(picks)}:\n")

    max_dim = args.max_dim if args.max_dim and args.max_dim > 0 else None
    for scene_info in picks:
        generate_preview(scene_info, args.out_dir, args.color, max_dim, args.vmin, args.vmax)


if __name__ == "__main__":
    main()
