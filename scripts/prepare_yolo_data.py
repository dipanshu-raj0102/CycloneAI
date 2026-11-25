"""
GOES-18/19 + Himawari (Band 13) + IBTrACS -> YOLO26-Pose dataset
(cyclone identification + exact single center keypoint)

Pipeline highlights:
  * Analytic Plate-Carree (eqc) geolocation with sub-pixel precision & exact inverse.
  * Strict agency center fixes (USA_LAT / USA_LON) at main synoptic hours (00/06/12/18 UTC)
    with extratropical (ET) filtering; zero interpolated coordinates.
  * 8-deg domain overlap at both 182E (Himawari/GOES-West) and 254E (GOES-West/GOES-East).
  * 0.02 deg bilinear -> 2x2 NaN-aware block mean (area average) to 0.04 deg (~4.45 km).
  * NaN-aware normalized convolution for convective anomaly & eroded valid mask for gradients.
  * Snap-to-edge 640x640 tiling (zero artificial black border padding).
  * Negative tiles avoid ALL IBTrACS positions (any wind/agency) + disk-aware empty-scan sampling.
  * Exports cyclone_pose.yaml, grid_meta.json, tiles_manifest.csv, and visual QC previews.
"""
import os

# Pin BLAS/OpenMP threads before numpy/cv2 import (inherited by spawned workers).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import multiprocessing  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from datetime import datetime, timedelta, timezone  # noqa: E402
from pathlib import Path  # noqa: E402

import cv2  # noqa: E402
import dask  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from pyresample import create_area_def  # noqa: E402
from satpy import Scene  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("YOLO26_Cyclone_Pose_Prep")

RAW_DIR = Path(__file__).resolve().parent

# ------------------------------------------------------------------------------
# Grid & Tiling Constants
# ------------------------------------------------------------------------------
RES_DEG = 0.04                                   # final grid spacing (~4.45 km at equator)
KM_PER_DEG = 111.195
M_PER_DEG = float(np.deg2rad(1.0) * 6378137.0)   # WGS84 eqc (lat_ts=0): x = a*lon, y = a*lat
TILE = 640
STRIDE = 480
EXCL_BUFFER_PX = int(round(10.0 / RES_DEG))      # keep negatives >= 10 deg from ANY IBTrACS fix

# Satellite routing handover longitudes (deg E in [0, 360))
HIMA_GOESW_BOUNDARY = 182.0                      # midpoint of 140.7E and 222.8E
GOESW_GOESE_BOUNDARY = 254.0                     # midpoint of 222.8E and 284.8E (~106W)
GOESE_MAX_LON = 345.0

# Padded domains provide >= 4 deg context across both handovers (182E and 254E)
DOMAINS = {
    "himawari": {"lon_min": 80.0, "lon_max": 186.0, "lat_min": -40.0, "lat_max": 45.0},
    "goes_west": {"lon_min": 178.0, "lon_max": 260.0, "lat_min": -40.0, "lat_max": 45.0},
    "goes_east": {"lon_min": 250.0, "lon_max": GOESE_MAX_LON, "lat_min": -40.0, "lat_max": 45.0},
}


def grid_shape(dom: dict) -> tuple[int, int]:
    w = int(round((dom["lon_max"] - dom["lon_min"]) / RES_DEG))
    h = int(round((dom["lat_max"] - dom["lat_min"]) / RES_DEG))
    return h, w


def tile_starts(length: int, tile: int, stride: int) -> list[int]:
    """Full-size tile origins snapped inward at the far edge (eliminates black zero-padding)."""
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


def lonlat_to_pixel(lon: float, lat: float, dom: dict) -> tuple[float, float]:
    """Continuous pixel coords (col, row); (0, 0) is top-left corner of pixel (0, 0)."""
    dlon = ((lon - dom["lon_min"]) + 180.0) % 360.0 - 180.0
    return dlon / RES_DEG, (dom["lat_max"] - lat) / RES_DEG


def pixel_to_lonlat(sensor: str, gx: float, gy: float) -> tuple[float, float]:
    """Inverse of lonlat_to_pixel. gx, gy = tile (x0, y0) + tile pixel. Returns lon in [-180, 180)."""
    dom = DOMAINS[sensor]
    lon = (dom["lon_min"] + gx * RES_DEG) % 360.0
    lon = (lon + 180.0) % 360.0 - 180.0
    return lon, dom["lat_max"] - gy * RES_DEG


def hash_split(key: str) -> str:
    bucket = int(hashlib.md5(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "val"
    return "test"


def year_split(year: int) -> str:
    if year <= 2022:
        return "train"
    if year == 2023:
        return "val"
    return "test"


# ==============================================================================
# 1. Parse IBTrACS (Strict Agency Fixes + Storm-ID Split + Global Exclusion Map)
# ==============================================================================
def parse_ibtracs(
    csv_path: Path,
    min_year: int = 2021,
    min_wind_kt: float = 25.0,
    split_mode: str = "sid_hash",
    hours: tuple[int, ...] = (0, 6, 12, 18),
    exclude_et: bool = True,
) -> tuple[pd.DataFrame, dict]:
    usecols = [
        "SID", "SEASON", "NATURE", "ISO_TIME", "LAT", "LON",
        "USA_LAT", "USA_LON", "USA_WIND",
        "USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW",
    ]
    df = pd.read_csv(csv_path, skiprows=[1], usecols=usecols, low_memory=False)
    for col in usecols:
        if col not in ("SID", "NATURE", "ISO_TIME"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["NATURE"] = df["NATURE"].astype(str).str.strip().str.upper()
    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce", utc=True)
    df = df[df["SEASON"] >= min_year].dropna(subset=["SID", "ISO_TIME"]).copy()

    # Global exclusion list: EVERY IBTrACS position (any wind, nature, or agency) per timestamp
    ex = df.dropna(subset=["LAT", "LON"])
    excl_by_time: dict = {}
    for t, lon, lat in zip(ex["ISO_TIME"], (ex["LON"] % 360.0), ex["LAT"]):
        excl_by_time.setdefault(t, []).append((float(lon), float(lat)))

    # Labeled fixes: real agency fixes only (USA_LAT/USA_LON) at main synoptic hours
    mask = (
        (df["USA_WIND"] >= min_wind_kt)
        & (df["ISO_TIME"].dt.hour.isin(list(hours)))
        & (df["ISO_TIME"].dt.minute == 0)
    )
    if exclude_et:
        mask &= df["NATURE"] != "ET"

    lab = df[mask].dropna(subset=["USA_LAT", "USA_LON", "USA_WIND"]).copy()
    lab["lat"] = lab["USA_LAT"]
    lab["lon360"] = lab["USA_LON"] % 360.0

    # Box half-width (deg) scaled from mean 34-kt wind radii (1 deg ~ 60 nm); 3 deg default
    r34_cols = ["USA_R34_NE", "USA_R34_SE", "USA_R34_SW", "USA_R34_NW"]
    mean_r34_nm = lab[r34_cols].mean(axis=1)
    lab["half_deg"] = np.clip((mean_r34_nm / 60.0).fillna(3.0), 3.0, 8.0)

    # Route each storm to its best-viewing satellite domain
    lon = lab["lon360"]
    is_hima = (lon >= DOMAINS["himawari"]["lon_min"]) & (lon < HIMA_GOESW_BOUNDARY)
    is_goes_w = (lon >= HIMA_GOESW_BOUNDARY) & (lon < GOESW_GOESE_BOUNDARY) & (lab["SEASON"] >= 2022)
    is_goes_e = (lon >= GOESW_GOESE_BOUNDARY) & (lon <= GOESE_MAX_LON) & (lab["SEASON"] >= 2025)
    lab["sensor"] = np.select(
        [is_hima, is_goes_w, is_goes_e],
        ["himawari", "goes_west", "goes_east"],
        default="unsupported",
    )
    lab = lab[lab["sensor"] != "unsupported"].copy()

    # Leak-free split strictly grouped by storm SID
    if split_mode == "year":
        sid_year = lab.groupby("SID")["SEASON"].min()
        lab["split"] = lab["SID"].map(sid_year.apply(year_split))
    else:
        lab["split"] = lab["SID"].map({sid: hash_split(sid) for sid in lab["SID"].unique()})

    logger.info(
        f"Parsed {len(lab)} labeled fixes across {lab['SID'].nunique()} storms "
        f"(train={(lab['split']=='train').sum()}, val={(lab['split']=='val').sum()}, "
        f"test={(lab['split']=='test').sum()}); exclusion map covers {len(excl_by_time)} timestamps."
    )
    return lab, excl_by_time


# ==============================================================================
# 2. Local File Resolvers & Disk-Aware Storm-Free Scan Sampler
# ==============================================================================
def resolve_himawari_files(hima_root: Path, dt: datetime) -> list[str]:
    """Raw/himawari/{YYYY}/{MM}/{DD}/HS_H0[89]_YYYYMMDD_HHMM_B13_FLDK_R20_S*.DAT.bz2"""
    yyyy, mm, dd = dt.strftime("%Y"), dt.strftime("%m"), dt.strftime("%d")
    ymd, hm = dt.strftime("%Y%m%d"), dt.strftime("%H%M")
    candidate_dirs = [hima_root / yyyy / mm / dd, hima_root / yyyy / mm / dd / hm]

    preferred_sat = "H09" if dt >= datetime(2022, 12, 13, 5, 0, tzinfo=timezone.utc) else "H08"
    fallback_sat = "H08" if preferred_sat == "H09" else "H09"

    for sat in (preferred_sat, fallback_sat):
        pattern = f"HS_{sat}_{ymd}_{hm}_B13_FLDK_R20_S*.DAT.bz2"
        for scan_dir in candidate_dirs:
            if scan_dir.is_dir():
                files = sorted(str(p) for p in scan_dir.glob(pattern))
                if len(files) == 10:
                    return files
    return []


def resolve_goes_files(goes_root: Path, dt: datetime, sensor: str) -> list[str]:
    """Raw/goes/noaa-goes{18,19}/{YYYY}/{DDD}/*C13_G1x_sYYYYDDDHH*.nc (handles :59:50 boundary jitter)."""
    bucket, sat_code = ("noaa-goes18", "G18") if sensor == "goes_west" else ("noaa-goes19", "G19")
    ts_re = re.compile(r"_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})")

    # Check dt's hour first, then dt - 10 min in case a scan started at :59:50 of the prior hour/day
    probe_dts = [dt]
    prev_dt = dt - timedelta(minutes=10)
    if prev_dt.hour != dt.hour:
        probe_dts.append(prev_dt)

    matches = []
    for pdt in probe_dts:
        yyyy, ddd, hh = pdt.strftime("%Y"), pdt.strftime("%j"), pdt.strftime("%H")
        candidate_dirs = [
            goes_root / bucket / yyyy / ddd,
            goes_root / bucket / "ABI-L1b-RadF" / yyyy / ddd / hh,
        ]
        for scan_dir in candidate_dirs:
            if not scan_dir.is_dir():
                continue
            for f in scan_dir.glob(f"*C13_{sat_code}_s{yyyy}{ddd}{hh}*.nc"):
                m = ts_re.search(f.name)
                if m:
                    yr, doy, h, mn, sec = m.groups()
                    f_dt = datetime.strptime(f"{yr}-{doy} {h}:{mn}:{sec}", "%Y-%j %H:%M:%S").replace(
                        tzinfo=timezone.utc
                    )
                    diff_sec = abs((f_dt - dt).total_seconds())
                    if diff_sec <= 600:
                        matches.append((diff_sec, str(f)))
        if matches:
            break

    if matches:
        matches.sort(key=lambda x: x[0])
        return [matches[0][1]]
    return []


def resolve_files(sensor: str, dt: datetime, goes_dir: Path, himawari_dir: Path) -> list[str]:
    if sensor == "himawari":
        return resolve_himawari_files(himawari_dir, dt)
    return resolve_goes_files(goes_dir, dt, sensor)


def discover_local_dates(sensor: str, goes_dir: Path, himawari_dir: Path) -> list[datetime]:
    """Lists actual calendar dates present on disk for each sensor so empty-scan sampling never guesses dead years."""
    dates = []
    if sensor == "himawari":
        for y_dir in sorted(himawari_dir.glob("[0-9][0-9][0-9][0-9]")):
            if not y_dir.is_dir():
                continue
            for m_dir in sorted(y_dir.glob("[0-9][0-9]")):
                for d_dir in sorted(m_dir.glob("[0-9][0-9]")):
                    if d_dir.is_dir():
                        try:
                            dates.append(
                                datetime(int(y_dir.name), int(m_dir.name), int(d_dir.name), tzinfo=timezone.utc)
                            )
                        except ValueError:
                            pass
    else:
        bucket = "noaa-goes18" if sensor == "goes_west" else "noaa-goes19"
        base = goes_dir / bucket
        for y_dir in sorted(base.glob("[0-9][0-9][0-9][0-9]")):
            if not y_dir.is_dir():
                continue
            for ddd_dir in sorted(y_dir.glob("[0-9][0-9][0-9]")):
                if ddd_dir.is_dir():
                    try:
                        dates.append(
                            datetime.strptime(f"{y_dir.name}-{ddd_dir.name}", "%Y-%j").replace(tzinfo=timezone.utc)
                        )
                    except ValueError:
                        pass
    return dates


def build_scan_tasks(
    df: pd.DataFrame,
    excl_by_time: dict,
    goes_dir: Path,
    himawari_dir: Path,
    split_mode: str,
) -> list[dict]:
    tasks, missing = [], 0
    cols = ["SID", "lat", "lon360", "half_deg", "USA_WIND", "split"]
    for (sensor, iso_time), grp in df.groupby(["sensor", "ISO_TIME"], sort=True):
        dt = iso_time.to_pydatetime()
        files = resolve_files(sensor, dt, goes_dir, himawari_dir)
        if not files:
            missing += 1
            continue
        tasks.append({
            "sensor": sensor,
            "synoptic_dt": dt,
            "file_paths": files,
            "storms": grp[cols].to_dict(orient="records"),
            "excl": excl_by_time.get(iso_time, []),
            "empty": False,
            "split_mode": split_mode,
        })
    logger.info(f"Matched {len(tasks)} storm scans ({missing} storm timestamps had no local satellite file).")
    return tasks


def sample_empty_scans(
    df: pd.DataFrame,
    excl_by_time: dict,
    goes_dir: Path,
    himawari_dir: Path,
    n_per_sensor: int,
    hours: tuple[int, ...],
    split_mode: str,
    seed: int = 42,
) -> list[dict]:
    """Samples storm-free synoptic scans exclusively from dates actually present on disk."""
    rng = random.Random(seed)
    tasks = []
    for sensor in ("himawari", "goes_west", "goes_east"):
        used = set(df.loc[df["sensor"] == sensor, "ISO_TIME"])
        local_dates = discover_local_dates(sensor, goes_dir, himawari_dir)
        candidates = [
            pd.Timestamp(d.replace(hour=h))
            for d in local_dates
            for h in hours
            if pd.Timestamp(d.replace(hour=h)) not in used
        ]
        rng.shuffle(candidates)
        found = 0
        for ts in candidates:
            if found >= n_per_sensor:
                break
            dt = ts.to_pydatetime()
            files = resolve_files(sensor, dt, goes_dir, himawari_dir)
            if not files:
                continue
            tasks.append({
                "sensor": sensor,
                "synoptic_dt": dt,
                "file_paths": files,
                "storms": [],
                "excl": excl_by_time.get(ts, []),
                "empty": True,
                "split_mode": split_mode,
            })
            found += 1
        logger.info(f"Sampled {found} storm-free scans for {sensor} (from {len(local_dates)} local date folders).")
    return tasks


# ==============================================================================
# 3. Band 13 -> Kelvin Tb -> 0.04 deg Grid -> 3-Channel uint8
# ==============================================================================
def block_mean_nan(a: np.ndarray, f: int) -> np.ndarray:
    """f x f block mean ignoring NaNs; requires at least half the sub-pixels valid."""
    if f == 1:
        return a
    h, w = a.shape
    if h % f or w % f:
        raise ValueError(f"Array shape {a.shape} not divisible by {f}")
    blocks = a.reshape(h // f, f, w // f, f)
    ok = np.isfinite(blocks)
    n = ok.sum(axis=(1, 3))
    s = np.where(ok, blocks, 0.0).sum(axis=(1, 3), dtype=np.float64)
    out = np.where(n >= (f * f + 1) // 2, s / np.maximum(n, 1), np.nan)
    return out.astype(np.float32)


def encode_channels(tb: np.ndarray, valid: np.ndarray, lat_max: float) -> np.ndarray:
    """tb: Kelvin with NaN where invalid. Returns HxWx3 uint8; invalid pixels are 0 in all channels."""
    H, _ = tb.shape
    sigma = 25.0
    tb0 = np.where(valid, tb, 0.0).astype(np.float32)

    # Ch0: Inverted Brightness Temperature [310 K -> 0, 180 K -> 1]
    ch0 = np.where(valid, np.clip((310.0 - tb0) / 130.0, 0.0, 1.0), 0.0)

    # NaN-aware Gaussian smooth (normalized convolution prevents limb/missing-pixel bleed)
    num = cv2.GaussianBlur(tb0, (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    den = cv2.GaussianBlur(valid.astype(np.float32), (0, 0), sigmaX=sigma, sigmaY=sigma, borderType=cv2.BORDER_REFLECT)
    smooth = np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    del num, den

    # Ch2: Mesoscale convective anomaly (colder than local surroundings -> high)
    anom = smooth - tb0
    ch2 = np.where(valid & np.isfinite(anom), np.clip((anom + 20.0) / 80.0, 0.0, 1.0), 0.0)
    del anom

    # Ch1: Latitude-corrected ||grad Tb|| in K/km, zeroed wherever 3x3 window touches invalid pixels
    tb_f = np.where(valid, tb0, np.where(np.isfinite(smooth), smooth, 290.0)).astype(np.float32)
    del tb0, smooth
    lats = lat_max - (np.arange(H, dtype=np.float32) + 0.5) * RES_DEG
    dy_km = RES_DEG * KM_PER_DEG
    dx_km = np.maximum(RES_DEG * KM_PER_DEG * np.cos(np.deg2rad(lats)), 1.0)[:, None]
    gy = np.gradient(tb_f, dy_km, axis=0)
    gx = np.gradient(tb_f, axis=1) / dx_km
    del tb_f
    valid_er = cv2.erode(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    ch1 = np.where(valid_er, np.clip(np.hypot(gx, gy) / 3.5, 0.0, 1.0), 0.0)

    return np.round(np.stack([ch0, ch1, ch2], axis=-1) * 255.0).astype(np.uint8)


def build_3ch_grid(
    file_paths: list[str],
    sensor: str,
    cache_dir: Path,
    supersample: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    dom = DOMAINS[sensor]
    H, W = grid_shape(dom)

    # Plate-Carree (eqc, lat_ts=0) centered on domain -> exact uniform degree pixels
    half_w = (dom["lon_max"] - dom["lon_min"]) / 2.0
    lon0 = ((dom["lon_min"] + dom["lon_max"]) / 2.0 + 180.0) % 360.0 - 180.0
    extent = (-half_w * M_PER_DEG, dom["lat_min"] * M_PER_DEG, half_w * M_PER_DEG, dom["lat_max"] * M_PER_DEG)
    proj = {"proj": "eqc", "lat_ts": 0, "lat_0": 0, "lon_0": lon0, "datum": "WGS84", "units": "m"}
    fine_area = create_area_def(
        area_id=f"{sensor}_{RES_DEG / supersample:.3f}deg_{H}x{W}",
        projection=proj,
        shape=(H * supersample, W * supersample),
        area_extent=extent,
    )

    if sensor == "himawari":
        reader, candidates = "ahi_hsd", ["B13"]
    else:
        reader = "abi_l2_nc" if "CMIP" in Path(file_paths[0]).name else "abi_l1b"
        candidates = ["C13", "CMIP_C13", "CMI"]

    scn = Scene(reader=reader, filenames=file_paths)
    available = scn.available_dataset_names()
    band = next((c for c in candidates if c in available), None)
    if band is None:
        raise ValueError(f"Band 13 missing in {Path(file_paths[0]).name}; wanted {candidates}, got {available}")
    scn.load([band])

    resampled = scn.resample(fine_area, resampler="bilinear", cache_dir=str(cache_dir))
    tb_fine = np.asarray(resampled[band].values, dtype=np.float32)
    del scn, resampled  # release Satpy/Dask buffers before channel encoding

    tb = block_mean_nan(tb_fine, supersample)
    del tb_fine
    if tb.shape != (H, W):
        raise ValueError(f"Unexpected grid shape {tb.shape}, expected {(H, W)}")

    valid = np.isfinite(tb) & (tb > 150.0) & (tb < 350.0)
    if valid.mean() < 0.70:
        raise ValueError(f"Low valid pixel coverage ({valid.mean():.1%})")
    tb = np.where(valid, tb, np.nan).astype(np.float32)

    return encode_channels(tb, valid, dom["lat_max"]), valid


# ==============================================================================
# 4. Tile Worker: Full 640x640 BBox + Exact Center Keypoint Tiles
# ==============================================================================
def draw_preview(tile_rgb: np.ndarray, labels: list[str], out_path: Path) -> None:
    vis = tile_rgb[:, :, ::-1].copy()  # BGR for OpenCV drawing
    for line in labels:
        parts = line.split()
        xc, yc, bw, bh = (float(parts[i]) * TILE for i in range(1, 5))
        kx, ky = float(parts[5]) * TILE, float(parts[6]) * TILE
        x1, y1 = int(round(xc - bw / 2)), int(round(yc - bh / 2))
        x2, y2 = int(round(xc + bw / 2)), int(round(yc + bh / 2))
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.drawMarker(vis, (int(round(kx)), int(round(ky))), (0, 0, 255), cv2.MARKER_CROSS, 16, 2)
    cv2.imwrite(str(out_path), vis, [cv2.IMWRITE_PNG_COMPRESSION, 1])


def process_scan_task(
    task: dict,
    out_dir: Path,
    cache_dir: Path,
    neg_tile_ratio: float = 0.15,
    supersample: int = 2,
) -> tuple[int, int, int, list[dict]]:
    cv2.setNumThreads(0)
    dask.config.set(scheduler="single-threaded")

    sensor = task["sensor"]
    dom = DOMAINS[sensor]
    dt = task["synoptic_dt"]
    ts_key = dt.strftime("%Y%m%d_%H%M")
    save_preview = bool(task.get("preview", False))

    try:
        rgb_u8, valid = build_3ch_grid(task["file_paths"], sensor, cache_dir, supersample=supersample)
    except Exception as exc:
        logger.warning(f"Skipping {sensor} {ts_key}: {exc}")
        return 0, 0, 0, []

    H, W, _ = rgb_u8.shape
    storm_boxes = []
    for st in task["storms"]:
        cx, cy = lonlat_to_pixel(st["lon360"], st["lat"], dom)
        if not (0.0 <= cx < W and 0.0 <= cy < H):
            continue
        ix, iy = int(cx), int(cy)
        # Resilient 3x3 check so a single dead pixel does not discard an otherwise valid eye
        if not valid[max(0, iy - 1) : min(H, iy + 2), max(0, ix - 1) : min(W, ix + 2)].any():
            continue
        storm_boxes.append(
            (float(cx), float(cy), float(st["half_deg"]) / RES_DEG, st["split"], st["SID"], float(st["USA_WIND"]))
        )

    if not storm_boxes and not task.get("empty", False):
        return 0, 0, 0, []

    excl_px = [lonlat_to_pixel(lon, lat, dom) for lon, lat in task.get("excl", [])]
    if storm_boxes:
        scan_split = storm_boxes[0][3]
    elif task.get("split_mode") == "year":
        scan_split = year_split(dt.year)
    else:
        scan_split = hash_split(f"{sensor}_{ts_key}")

    seed = int(hashlib.md5(f"{sensor}_{ts_key}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    pos_tiles, neg_tiles, total_storms = 0, 0, 0
    manifest_rows = []
    preview_written = False

    for y0 in tile_starts(H, TILE, STRIDE):
        y1 = y0 + TILE
        for x0 in tile_starts(W, TILE, STRIDE):
            x1 = x0 + TILE

            # Drop mostly off-disk / corrupt tiles for both positives and negatives
            if valid[y0:y1, x0:x1].mean() < 0.50:
                continue

            labels, tile_splits, tile_sids, tile_winds = [], set(), [], []
            for cx, cy, half_px, st_split, sid, wind_kt in storm_boxes:
                # Require true storm center inside tile (16 px margin)
                if not ((x0 + 16 <= cx <= x1 - 16) and (y0 + 16 <= cy <= y1 - 16)):
                    continue

                raw_area = (2.0 * half_px) ** 2
                bx1 = np.clip(cx - half_px - x0, 0.0, TILE)
                by1 = np.clip(cy - half_px - y0, 0.0, TILE)
                bx2 = np.clip(cx + half_px - x0, 0.0, TILE)
                by2 = np.clip(cy + half_px - y0, 0.0, TILE)
                clip_w, clip_h = bx2 - bx1, by2 - by1

                # Keep only if >= 35% of the storm bounding box lies inside the tile
                if (clip_w * clip_h) / raw_area < 0.35:
                    continue

                bcx, bcy = ((bx1 + bx2) / 2.0) / TILE, ((by1 + by2) / 2.0) / TILE
                bw, bh = clip_w / TILE, clip_h / TILE
                # Exact unclipped IBTrACS agency center as keypoint
                kpt_x, kpt_y = (cx - x0) / TILE, (cy - y0) / TILE

                labels.append(f"0 {bcx:.6f} {bcy:.6f} {bw:.6f} {bh:.6f} {kpt_x:.6f} {kpt_y:.6f} 2")
                tile_splits.add(st_split)
                tile_sids.append(sid)
                tile_winds.append(wind_kt)

            if labels:
                if len(tile_splits) > 1:  # multi-storm split collision guard
                    continue
                target_split = next(iter(tile_splits))
            else:
                near = any(
                    (x0 - max(1.5 * hp, EXCL_BUFFER_PX) <= cx <= x1 + max(1.5 * hp, EXCL_BUFFER_PX))
                    and (y0 - max(1.5 * hp, EXCL_BUFFER_PX) <= cy <= y1 + max(1.5 * hp, EXCL_BUFFER_PX))
                    for cx, cy, hp, _, _, _ in storm_boxes
                ) or any(
                    (x0 - EXCL_BUFFER_PX <= px <= x1 + EXCL_BUFFER_PX)
                    and (y0 - EXCL_BUFFER_PX <= py <= y1 + EXCL_BUFFER_PX)
                    for px, py in excl_px
                )
                if near or rng.random() > neg_tile_ratio:
                    continue
                target_split = scan_split

            tile = rgb_u8[y0:y1, x0:x1]
            stem = f"{sensor}_{ts_key}_y{y0:04d}_x{x0:04d}"
            img_rel = f"images/{target_split}/{stem}.png"

            cv2.imwrite(str(out_dir / img_rel), tile[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 1])
            (out_dir / f"labels/{target_split}/{stem}.txt").write_text("\n".join(labels))

            if save_preview and labels and not preview_written:
                draw_preview(tile, labels, out_dir / f"previews/{stem}_qc.png")
                preview_written = True

            manifest_rows.append({
                "image": img_rel,
                "sensor": sensor,
                "timestamp": dt.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "split": target_split,
                "y0": y0,
                "x0": x0,
                "num_storms": len(labels),
                "sids": ";".join(tile_sids),
                "max_wind_kt": max(tile_winds) if tile_winds else 0.0,
            })

            if labels:
                pos_tiles += 1
                total_storms += len(labels)
            else:
                neg_tiles += 1

    return pos_tiles, neg_tiles, total_storms, manifest_rows


# ==============================================================================
# 5. Metadata & Main Entrypoint
# ==============================================================================
def write_grid_meta(out_dir: Path, supersample: int) -> None:
    meta = {
        "resolution_deg": RES_DEG,
        "supersample": supersample,
        "tile_size": TILE,
        "stride": STRIDE,
        "pixel_convention": "continuous pixel coords, (0,0) = top-left corner of pixel (0,0); x right, y down",
        "tile_name": "{sensor}_{YYYYmmdd}_{HHMM}_y{y0:04d}_x{x0:04d}",
        "global_pixel": "gx = x0 + tile_x_px ; gy = y0 + tile_y_px  (tile_*_px = normalized * tile_size)",
        "inverse": "lon = (lon_min + gx*res) mod 360 (then wrap to [-180,180)) ; lat = lat_max - gy*res",
        "channels": {
            "R": "inverted_tb_180_to_310K",
            "G": "lat_corrected_grad_tb_0_to_3.5K_per_km",
            "B": "mesoscale_convective_anomaly_-20_to_60K",
        },
        "sensors": {
            s: {**d, "height": grid_shape(d)[0], "width": grid_shape(d)[1]} for s, d in DOMAINS.items()
        },
    }
    (out_dir / "grid_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="GOES-18/19 & Himawari + IBTrACS preprocessing for YOLO26-Pose (identification + center localization)"
    )
    parser.add_argument("--ibtracs", type=Path, default=RAW_DIR / "ibtracs/ibtracs.ALL.list.v04r01.csv")
    parser.add_argument("--goes-dir", type=Path, default=RAW_DIR / "goes")
    parser.add_argument("--himawari-dir", type=Path, default=RAW_DIR / "himawari")
    parser.add_argument("--out-dir", type=Path, default=RAW_DIR.parent / "processed/yolo26data")
    parser.add_argument("--min-year", type=int, default=2021)
    parser.add_argument("--min-wind", type=float, default=25.0, help="Minimum USA_WIND in knots")
    parser.add_argument("--hours", type=str, default="0,6,12,18", help="UTC hours of agency fixes to use as labels")
    parser.add_argument("--keep-et", action="store_true", help="Include extratropical (ET) systems in labels")
    parser.add_argument("--split-mode", choices=["sid_hash", "year"], default="sid_hash")
    parser.add_argument("--neg-ratio", type=float, default=0.15, help="Fraction of eligible background tiles to keep")
    parser.add_argument("--empty-scans", type=int, default=100, help="Storm-free scans to sample per sensor (0 = off)")
    parser.add_argument("--supersample", type=int, choices=[1, 2], default=2, help="2 = 0.02 deg area average (higher RAM)")
    parser.add_argument("--previews", type=int, default=5, help="QC preview PNGs to save per sensor")
    parser.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 4) // 2)))
    args = parser.parse_args()

    hours = tuple(int(h) for h in args.hours.split(",") if h.strip() != "")

    for split in ["train", "val", "test"]:
        (args.out_dir / f"images/{split}").mkdir(parents=True, exist_ok=True)
        (args.out_dir / f"labels/{split}").mkdir(parents=True, exist_ok=True)
    if args.previews > 0:
        (args.out_dir / "previews").mkdir(parents=True, exist_ok=True)

    cache_dir = args.out_dir / ".resample_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    yaml_path = args.out_dir / "cyclone_pose.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(args.out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "nc": 1,
                "names": {0: "cyclone"},
                "kpt_shape": [1, 3],
                "flip_idx": [0],
            },
            sort_keys=False,
        )
    )
    write_grid_meta(args.out_dir, args.supersample)

    df_lab, excl_by_time = parse_ibtracs(
        args.ibtracs,
        min_year=args.min_year,
        min_wind_kt=args.min_wind,
        split_mode=args.split_mode,
        hours=hours,
        exclude_et=not args.keep_et,
    )
    tasks = build_scan_tasks(df_lab, excl_by_time, args.goes_dir, args.himawari_dir, args.split_mode)
    if args.empty_scans > 0:
        tasks += sample_empty_scans(
            df_lab, excl_by_time, args.goes_dir, args.himawari_dir, args.empty_scans, hours, args.split_mode
        )

    # Mark first N storm scans per sensor for QC preview rendering
    if args.previews > 0:
        preview_counts = {"himawari": 0, "goes_west": 0, "goes_east": 0}
        for t in tasks:
            s = t["sensor"]
            if not t["empty"] and preview_counts[s] < args.previews:
                t["preview"] = True
                preview_counts[s] += 1

    # Pre-warm pyresample bilinear cache (1 scan per sensor) to prevent concurrent write races
    cv2.setNumThreads(0)
    dask.config.set(scheduler="single-threaded")
    warmed = set()
    for t in tasks:
        if t["sensor"] in warmed:
            continue
        try:
            build_3ch_grid(t["file_paths"], t["sensor"], cache_dir, supersample=args.supersample)
            warmed.add(t["sensor"])
            logger.info(f"Cached bilinear resampling grid for {t['sensor']} (supersample={args.supersample})")
        except Exception as exc:
            logger.warning(f"Cache warmup failed for {t['sensor']}: {exc}")

    logger.info(f"Dispatching {len(tasks)} scans across {args.workers} spawned workers...")

    total_pos, total_neg, total_storms, failed = 0, 0, 0, 0
    all_manifest_rows = []
    pool = ProcessPoolExecutor(
        max_workers=args.workers,
        max_tasks_per_child=250,
        mp_context=multiprocessing.get_context("spawn"),
    )
    try:
        futures = {
            pool.submit(process_scan_task, t, args.out_dir, cache_dir, args.neg_ratio, args.supersample): t
            for t in tasks
        }
        for i, fut in enumerate(as_completed(futures), 1):
            t = futures[fut]
            try:
                p, n, b, rows = fut.result()
            except Exception as exc:
                failed += 1
                logger.error(f"Worker failed on {t['sensor']} {t['synoptic_dt']:%Y%m%d_%H%M}: {exc!r}")
                continue
            total_pos += p
            total_neg += n
            total_storms += b
            all_manifest_rows.extend(rows)
            if i % 50 == 0 or i == len(futures):
                logger.info(
                    f"Progress: {i}/{len(futures)} | Positive tiles: {total_pos} | Background tiles: {total_neg} "
                    f"| Labeled cyclones: {total_storms} | Failed scans: {failed}"
                )
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Shutting down workers...")
        pool.shutdown(wait=False, cancel_futures=True)
        return
    else:
        pool.shutdown(wait=True)

    if all_manifest_rows:
        manifest_path = args.out_dir / "tiles_manifest.csv"
        pd.DataFrame(all_manifest_rows).to_csv(manifest_path, index=False)
        logger.info(f"Saved tile manifest ({len(all_manifest_rows)} rows) to {manifest_path}")

    logger.info(f"Finished. Config: {yaml_path} | Metadata: {args.out_dir / 'grid_meta.json'}")


if __name__ == "__main__":
    main()
