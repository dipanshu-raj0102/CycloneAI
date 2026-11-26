#!/usr/bin/env python3
"""
Build storm-centred, calibrated Band-13 brightness-temperature datasets from
Himawari-8/9 AHI HSD and GOES-R ABI L2 CMIPF files, matched to IBTrACS.

Outputs:
    data/processed/intensity_kelvin_final/current/train.h5
    data/processed/intensity_kelvin_final/current/val.h5
    data/processed/intensity_kelvin_final/current/test.h5
    data/processed/manifest.csv
    data/processed/rejections.csv
    data/processed/processing_summary.json

Design:
- Uses Satpy readers for sensor-specific calibration and geolocation.
- Uses a common local azimuthal-equidistant grid centred on each IBTrACS fix.
- Keeps temperatures as uint16, encoded as round(K * 100).
- Stores a per-pixel validity mask; invalid pixels are not silently treated as data.
- Splits by storm SID to prevent storm leakage between train/val/test.
- Does not infer sensor/band from file extension alone.
- Uses USA_WIND by default. These thresholds are Saffir-Simpson-style
  one-minute wind thresholds and must not be used as IMD three-minute labels.

Important:
- Validate the chosen IBTrACS wind field and agency convention for your study.
- The script deliberately rejects ambiguous/unsupported files rather than guessing.
- AHI HSD input should be a directory of B13 segment files for each observation.
- GOES input must be ABI L2 CMIPF channel 13 files.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, ExitStack
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import multiprocessing as mp
import time
import importlib.metadata
import fcntl
import csv
import hashlib
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import traceback
from collections import defaultdict, Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable, Optional

import h5py
import numpy as np
import pandas as pd
from pyproj import CRS, Transformer
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from pyresample.geometry import AreaDefinition
    from satpy import Scene
from pyproj import CRS, Transformer
import xarray as xr
from tqdm import tqdm

LOG = logging.getLogger("cyclone_preprocess")

# ----------------------------- Defaults ------------------------------------

DEFAULT_CROP_KM = 600.0
DEFAULT_SIZE = 448
DEFAULT_TOLERANCE_MIN = 5.0
DEFAULT_SEED = 42
DEFAULT_SPLITS = (0.70, 0.15, 0.15)
TEMP_SCALE = 100.0
TEMP_OFFSET = 0.0  # stored_value = round((Tb_K - offset) * scale)

HSD_RE = re.compile(
    r"^HS_H(?P<sat>08|09)_(?P<date>\d{8})_(?P<hm>\d{4})_B13_FLDK_R20_S(?P<seg>\d{2})10\.DAT(?:\.bz2)?$",
    re.IGNORECASE,
)
GOES_RE = re.compile(
    r"^OR_ABI-L2-CMIPF-M(?P<mode>\d+)C13_G(?P<sat>16|17|18)_"
    r"s(?P<year>\d{4})(?P<doy>\d{3})(?P<hms>\d{6})\d*_e\d+_c\d+\.nc$",
    re.IGNORECASE,
)

STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


@dataclass
class Candidate:
    sensor: str
    sensor_id: str
    timestamp: datetime
    files: list[str]


@dataclass
class Sample:
    sample_id: str
    sid: str
    timestamp: str
    nominal_timestamp: str
    sensor: str
    sensor_id: str
    basin: str
    lat: float
    lon: float
    wind_kts: float
    class_id: int
    ibtracs_timestamp: str
    time_offset_min: float
    viewing_zenith_deg: float
    flipped: int
    image: np.ndarray
    valid_mask: np.ndarray
    source_files: str
    nature: str
    wind_source: str
    pressure_hpa: float
    valid_fraction: float
    inner_valid_fraction: float


def utc_parse(value) -> Optional[datetime]:
    if value is None or pd.isna(value):
        return None
    dt = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(dt):
        return None
    return dt.to_pydatetime().astimezone(timezone.utc)


def normalize_lon(lon: float) -> float:
    return ((float(lon) + 180.0) % 360.0) - 180.0


def stable_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def classify_usa_wind(wind_kts: float) -> int:
    """0=TD/weak, 1=TS, 2=Cat1, ... 6=Cat5; requires USA_WIND."""
    if not np.isfinite(wind_kts) or wind_kts < 0:
        raise ValueError("Wind must be a finite non-negative value")
    if wind_kts < 34:
        return 0
    if wind_kts < 64:
        return 1
    if wind_kts < 83:
        return 2
    if wind_kts < 96:
        return 3
    if wind_kts < 113:
        return 4
    if wind_kts < 137:
        return 5
    return 6


CLASS_NAMES = [
    "TD_or_below_TS_threshold",
    "TS",
    "Category_1",
    "Category_2",
    "Category_3",
    "Category_4",
    "Category_5",
]


def parse_hsd_candidate(path: Path) -> Optional[tuple[str, datetime, str]]:
    m = HSD_RE.search(path.name)
    if not m:
        return None
    stamp = datetime.strptime(m.group("date") + m.group("hm"), "%Y%m%d%H%M").replace(tzinfo=timezone.utc)
    return f"himawari{m.group('sat')}", stamp, m.group("seg")


def parse_goes_candidate(path: Path) -> Optional[tuple[str, datetime]]:
    m = GOES_RE.search(path.name)
    if not m:
        return None
    year, doy = int(m.group("year")), int(m.group("doy"))
    hms = m.group("hms")
    dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=doy - 1)
    dt = dt.replace(hour=int(hms[:2]), minute=int(hms[2:4]), second=int(hms[4:6]))
    return f"goes{m.group('sat')}", dt


def discover_candidates(raw_root: Path) -> list[Candidate]:
    groups: dict[tuple[str, datetime], list[str]] = defaultdict(list)
    for p in raw_root.rglob("*"):
        if not p.is_file():
            continue
        hsd = parse_hsd_candidate(p)
        if hsd:
            sensor_id, stamp, _ = hsd
            groups[(sensor_id, stamp)].append(str(p))
            continue
        goes = parse_goes_candidate(p)
        if goes:
            sensor_id, stamp = goes
            groups[(sensor_id, stamp)].append(str(p))

    candidates = []
    for (sensor_id, stamp), files in groups.items():
        sensor = "himawari" if sensor_id.startswith("himawari") else "goes"
        # HSD scenes must contain a complete, unique segment sequence.
        if sensor == "himawari":
            parsed = [parse_hsd_candidate(Path(f)) for f in files]
            segs = [int(x[2]) for x in parsed if x]
            if sorted(segs) != list(range(1, 11)):
                LOG.warning("Rejecting incomplete/duplicate HSD segments for %s %s", sensor_id, stamp)
                continue
        elif len(files) != 1:
            LOG.warning("Expected one GOES C13 file per scan, found %d: %s", len(files), files)
            continue
        candidates.append(Candidate(sensor, sensor_id, stamp, sorted(files)))
    return sorted(candidates, key=lambda x: (x.timestamp, x.sensor_id))


def load_ibtracs(path: Path, wind_col: str, basin_filter: Optional[list[str]]) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df["SID"] = df["SID"].astype("string").str.strip() if "SID" in df else pd.Series(dtype="string")
    required = {"SID", "ISO_TIME", "LAT", "LON", wind_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"IBTrACS is missing required columns: {sorted(missing)}")

    # IBTrACS CSV commonly has a units row immediately after the header.
    df["LAT"] = pd.to_numeric(df["LAT"], errors="coerce")
    df["LON"] = pd.to_numeric(df["LON"], errors="coerce")
    df[wind_col] = pd.to_numeric(df[wind_col], errors="coerce")
    df["ISO_TIME_PARSED"] = pd.to_datetime(df["ISO_TIME"].astype("string").str.strip(), format="%Y-%m-%d %H:%M:%S", utc=True, errors="coerce")
    df = df.dropna(subset=["SID", "ISO_TIME_PARSED", "LAT", "LON", wind_col]).copy()
    df = df[df["SID"].ne("")].copy()
    before = len(df)
    df = df[df["LAT"].between(-90, 90) & df["LON"].between(-180, 180)
            & np.isfinite(df[wind_col]) & df[wind_col].between(0, 250)].copy()
    LOG.info("Invalid coordinate/wind rows excluded: %d", before-len(df))
    df["LON"] = df["LON"].map(normalize_lon)
    df["SID"] = df["SID"].astype(str)
    if "BASIN" not in df.columns:
        df["BASIN"] = "UNKNOWN"
    df["BASIN"] = df["BASIN"].fillna("UNKNOWN").astype(str)
    if basin_filter:
        wanted = {x.upper() for x in basin_filter}
        df = df[df["BASIN"].str.upper().isin(wanted)]
    # Restrict to synoptic fixes; do not silently resolve conflicting labels.
    t = df["ISO_TIME_PARSED"]
    df = df[(t.dt.hour % 3 == 0) & (t.dt.minute == 0) & (t.dt.second == 0)]
    conflicts = df.groupby(["SID", "ISO_TIME_PARSED"])[[wind_col,"LAT","LON"]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        raise ValueError(f"Conflicting duplicate IBTrACS fixes: {int(conflicts.sum())}")
    df = (
        df.sort_values(wind_col)
          .drop_duplicates(["SID", "ISO_TIME_PARSED"], keep="last")
          .sort_values(["SID", "ISO_TIME_PARSED"])
    )
    if "NATURE" in df.columns:
        df = df[df["NATURE"].astype("string").str.strip().eq("TS")]
    if df.empty:
        raise ValueError("No usable tropical IBTrACS rows remain after filtering")
    return df


def matching_track_rows(
    tracks: pd.DataFrame,
    scan_time: datetime,
    tolerance_min: float,
) -> list[pd.Series]:
    """Return the nearest-in-time row for EVERY SID within tolerance.

    A satellite scene can contain multiple cyclones. Selecting one globally
    nearest IBTrACS row would silently assign the entire scene to one storm.
    """
    delta = (tracks["ISO_TIME_PARSED"] - pd.Timestamp(scan_time)).abs()
    eligible = tracks.loc[delta <= pd.Timedelta(minutes=tolerance_min)].copy()
    if eligible.empty:
        return []

    eligible["TIME_OFFSET_MIN"] = (
        eligible["ISO_TIME_PARSED"] - pd.Timestamp(scan_time)
    ).abs().dt.total_seconds() / 60.0

    # Keep one closest record per storm SID. In a tie, retain the first
    # deterministic row after sorting by time offset and timestamp.
    eligible = eligible.sort_values(
        ["SID", "TIME_OFFSET_MIN", "ISO_TIME_PARSED"]
    ).drop_duplicates("SID", keep="first")
    return [row.copy() for _, row in eligible.iterrows()]


def storm_area(lat: float, lon: float, crop_km: float, size: int) -> AreaDefinition:
    from pyresample.geometry import AreaDefinition
    half = crop_km * 1000.0
    projection = {
        "proj": "aeqd",
        "lat_0": float(lat),
        "lon_0": float(lon),
        "datum": "WGS84",
        "units": "m",
        "no_defs": None,
    }
    return AreaDefinition(
        area_id="storm_local_aeqd",
        description="Cyclone-centred azimuthal-equidistant grid",
        proj_id="storm_local_aeqd",
        projection=projection,
        width=size,
        height=size,
        area_extent=(-half, -half, half, half),
    )


def decompress_inputs(files: list[str], temp_dir: Path) -> list[str]:
    """Decompress .bz2 files for Satpy; uncompressed inputs pass through."""
    out = []
    for name in files:
        p = Path(name)
        if p.suffix.lower() != ".bz2":
            out.append(str(p))
            continue
        target = temp_dir / p.with_suffix("").name
        with open(p, "rb") as src, open(target, "wb") as dst:
            shutil.copyfileobj(__import__("bz2").BZ2File(src), dst)
        out.append(str(target))
    return out


def select_dataset_name(scene: Scene, sensor: str) -> str:
    available = {str(x) for x in scene.available_dataset_names()}
    expected = "B13" if sensor == "himawari" else "C13"
    if expected not in available:
        raise ValueError(f"Expected {expected}; available datasets: {sorted(available)}")
    return expected


def get_scene_time(scene: Scene, fallback: datetime) -> datetime:
    start = scene.attrs.get("start_time")
    if start is None:
        return fallback
    dt = utc_parse(start)
    return dt or fallback


def read_goes_patch(filename: str, area: AreaDefinition):
    """Decode L2 CMI once, map target pixel centres onto the ABI fixed grid."""
    with xr.open_dataset(filename, mask_and_scale=True) as ds:
        return read_goes_open(ds, area)

def read_goes_open(ds, area):
    required = {"CMI", "DQF", "x", "y", "goes_imager_projection", "band_id"}
    if not required.issubset(ds.variables):
        raise ValueError(f"Missing GOES variables: {required - set(ds.variables)}")
    if int(ds.band_id.values.item()) != 13 or ds.CMI.attrs.get("units") != "K":
        raise ValueError("Expected channel 13 brightness temperature in Kelvin")
    if ds.CMI.dims != ("y", "x") or ds.DQF.dims != ("y", "x"):
        raise ValueError("Unexpected CMI/DQF dimensions")
    if any(ds[v].attrs.get("units") not in ("rad", "radian", "radians") for v in ("x", "y")):
        raise ValueError("Expected ABI scan angles in radians")
    attrs = ds.goes_imager_projection.attrs
    height = float(attrs["perspective_point_height"])
    crs = CRS.from_cf(attrs)
    xx, yy = area.get_proj_coords()
    gx, gy = Transformer.from_crs(area.crs, crs, always_xy=True).transform(xx, yy)
    x = np.asarray(ds.x.values, dtype=float) * height
    y = np.asarray(ds.y.values, dtype=float) * height
    if min(len(x), len(y)) < 2 or not all(np.allclose(np.diff(v), v[1]-v[0], rtol=0.01) for v in (x,y)):
        raise ValueError("Expected regular ABI fixed grid")
    col = (gx - x[0]) / (x[1] - x[0])
    row = (gy - y[0]) / (y[1] - y[0])
    valid = np.isfinite(col) & np.isfinite(row) & (col >= -0.5) & (col < len(x)-0.5) & (row >= -0.5) & (row < len(y)-0.5)
    arr = np.full(xx.shape, np.nan, dtype=np.float32)
    if valid.any():
        ci = np.floor(col[valid]+0.5).astype(int)
        ri = np.floor(row[valid]+0.5).astype(int)
        ys = slice(int(ri.min()), int(ri.max())+1)
        xs = slice(int(ci.min()), int(ci.max())+1)
        bt = ds.CMI.isel(y=ys, x=xs).values
        dqf = ds.DQF.isel(y=ys, x=xs).values
        ii, jj = ri-ys.start, ci-xs.start
        arr[valid] = np.where(dqf[ii,jj] == 0, bt[ii,jj], np.nan)
    start = utc_parse(ds.attrs.get("time_coverage_start"))
    end = utc_parse(ds.attrs.get("time_coverage_end"))
    if start is None or end is None or end < start:
        raise ValueError("Missing or invalid GOES acquisition metadata")
    return arr, float(attrs["longitude_of_projection_origin"]), start, end


def read_ahi_window(data, area):
    """Map target centres to the native fixed grid; compute only intersecting chunks."""
    native = data.attrs["area"]
    if data.dims != ("y", "x") or data.shape != (native.height, native.width):
        raise ValueError("Unexpected AHI data geometry")
    xx, yy = area.get_proj_coords()
    gx, gy = Transformer.from_crs(area.crs, native.crs, always_xy=True).transform(xx, yy)
    xmin, ymin, xmax, ymax = native.area_extent
    dx = (xmax-xmin)/native.width
    dy = (ymax-ymin)/native.height
    col = (gx-xmin)/dx - 0.5
    row = (ymax-gy)/dy - 0.5
    valid = np.isfinite(col) & np.isfinite(row) & (col >= -0.5) & (col < native.width-0.5) & (row >= -0.5) & (row < native.height-0.5)
    out = np.full(xx.shape, np.nan, dtype=np.float32)
    if valid.any():
        ci = np.floor(col[valid]+0.5).astype(int)
        ri = np.floor(row[valid]+0.5).astype(int)
        y0, x0 = int(ri.min()), int(ci.min())
        block = data.isel(y=slice(y0,int(ri.max())+1), x=slice(x0,int(ci.max())+1)).values
        out[valid] = np.asarray(np.ma.filled(block, np.nan), dtype=np.float32)[ri-y0,ci-x0]
    return out


def process_candidate(
    candidate: Candidate,
    row: pd.Series,
    wind_col: str,
    crop_km: float,
    size: int,
    min_valid_fraction: float,
    max_zenith: float,
    flip_south: bool,
    tolerance_min: float = 5.0,
    opened=None,
    min_inner_valid_fraction=0.99,
    diagnostics=None,
) -> tuple[Optional[Sample], str]:
    if opened is None:
        with open_scene(candidate) as resource:
            return process_candidate(candidate, row, wind_col, crop_km, size, min_valid_fraction, max_zenith, flip_south, tolerance_min, resource, min_inner_valid_fraction, diagnostics)
    lat = float(row["LAT"])
    lon = normalize_lon(float(row["LON"]))
    wind = float(row[wind_col])
    try:
        class_id = classify_usa_wind(wind)
    except ValueError:
        return None, "invalid_wind"

    if diagnostics is None:
        diagnostics = {}
    fix = row["ISO_TIME_PARSED"].to_pydatetime()
    if candidate.sensor == "himawari":
        scene, dataset_name, actual_time, end_time = opened
        sensor_lon = 140.7
    else:
        actual_time = utc_parse(opened.attrs.get("time_coverage_start"))
        end_time = utc_parse(opened.attrs.get("time_coverage_end"))
        sensor_lon = float(opened.goes_imager_projection.attrs["longitude_of_projection_origin"])
    if actual_time is None or end_time is None or end_time < actual_time:
        raise ValueError("Missing or invalid acquisition interval")
    start_delta = abs((actual_time - candidate.timestamp).total_seconds())
    if start_delta > (60 if candidate.sensor == "goes" else 600):
        raise ValueError("Acquisition start disagrees with nominal scan timestamp")
    offset = max((actual_time-fix).total_seconds(), (fix-end_time).total_seconds(), 0)
    diagnostics.update(acquisition_start=actual_time.isoformat(), acquisition_end=end_time.isoformat(), interval_offset_sec=offset)
    if offset > tolerance_min*60:
        return None, "acquisition_interval_outside_tolerance"
    zenith = viewing_zenith(sensor_lon, lat, lon)
    diagnostics["viewing_zenith_deg"] = zenith
    if zenith > max_zenith:
        return None, "view_zenith_exceeds_limit"
    area = storm_area(lat, lon, crop_km, size)
    if candidate.sensor == "himawari":
        arr = read_ahi_window(scene[dataset_name], area)
    else:
        arr, _, _, _ = read_goes_open(opened, area)
    valid = np.isfinite(arr) & (arr >= 150.0) & (arr <= 350.0)
    valid_fraction = float(valid.mean())

    yy, xx = np.ogrid[:size, :size]
    inner = (yy-(size-1)/2)**2 + (xx-(size-1)/2)**2 <= (size/4)**2
    inner_valid = float(valid[inner].mean())
    diagnostics.update(valid_fraction=valid_fraction, inner_valid_fraction=inner_valid)
    if valid_fraction < min_valid_fraction:
        return None, "valid_fraction_below_limit"
    if inner_valid < min_inner_valid_fraction:
        return None, "inner_core_valid_fraction_below_limit"
    # Invalid values use zero sentinel; mask remains authoritative.
    if not valid.any():
        return None, "no_valid_pixels"
    fill_value = 0.0
    arr = np.where(valid, arr, fill_value)

    # Flip Southern Hemisphere patches only when explicitly enabled.
    flipped = int(flip_south and lat < 0)
    if flipped:
        arr = np.flipud(arr)
        valid = np.flipud(valid)

    # Encode Kelvin * 100. Clip only to representable range; physical
    # sanity filtering above ensures ordinary samples do not hit clipping.
    encoded = np.rint((arr - TEMP_OFFSET) * TEMP_SCALE)
    encoded = np.clip(encoded, 0, np.iinfo(np.uint16).max).astype(np.uint16)

    stamp = candidate.timestamp.strftime("%Y%m%dT%H%M%SZ")
    sid = str(row["SID"])
    sample_id = f"{candidate.sensor_id}_{stamp}_{sid}_{stable_hash('|'.join(candidate.files))}"
    sample = Sample(
        sample_id=sample_id,
        sid=sid,
        timestamp=actual_time.isoformat(),
        nominal_timestamp=candidate.timestamp.isoformat(),
        sensor=candidate.sensor,
        sensor_id=candidate.sensor_id,
        basin=str(row["BASIN"]),
        lat=lat,
        lon=lon,
        wind_kts=wind,
        class_id=class_id,
        ibtracs_timestamp=pd.Timestamp(row["ISO_TIME_PARSED"]).isoformat(),
        time_offset_min=(actual_time-fix).total_seconds()/60,
        viewing_zenith_deg=zenith,
        flipped=flipped,
        image=encoded,
        valid_mask=valid.astype(np.uint8),
        source_files=";".join(candidate.files),
        nature=str(row.get("NATURE", "UNKNOWN")),
        wind_source=str(row.get("USA_AGENCY", "UNKNOWN")),
        pressure_hpa=float(pd.to_numeric(row.get("USA_PRES", np.nan), errors="coerce")),
        valid_fraction=valid_fraction,
        inner_valid_fraction=inner_valid,
    )
    return sample, "accepted"


@contextmanager
def open_scene(candidate):
    """One decompression/open per scene; release all resources after its jobs."""
    with ExitStack() as stack:
        td = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="cyclone_sat_")))
        files = decompress_inputs(candidate.files, td)
        if candidate.sensor == "himawari":
            from satpy import Scene
            scene = Scene(filenames=files, reader="ahi_hsd")
            name = select_dataset_name(scene, candidate.sensor)
            scene.load([name], calibration="brightness_temperature")
            data = scene[name]
            if data.attrs.get("units") != "K":
                raise ValueError("Expected AHI brightness temperature in Kelvin")
            start = utc_parse(data.attrs.get("start_time")) or utc_parse(scene.attrs.get("start_time"))
            end = utc_parse(data.attrs.get("end_time")) or utc_parse(scene.attrs.get("end_time"))
            if start is None or end is None or end < start:
                raise ValueError("Missing or invalid AHI acquisition metadata")
            # Keep the calibrated array lazy; extract native windows on demand.
            yield scene, name, start, end
        else:
            yield stack.enter_context(xr.open_dataset(files[0], mask_and_scale=True))


def run_scene(task):
    import dask
    from dataclasses import fields
    candidate, rows, args, cache = task
    results = []
    source = [(str(Path(f).resolve()), Path(f).stat().st_size, Path(f).stat().st_mtime_ns) for f in candidate.files]
    with dask.config.set(scheduler="threads", num_workers=args.dask_threads), ExitStack() as stack:
        opened = None
        open_error = None
        for row in rows:
            key = stable_hash(json.dumps([source,str(row["SID"]),str(row["ISO_TIME_PARSED"])]))
            meta = cache / (key+".json")
            image_path = cache / (key+".image.npy")
            mask_path = cache / (key+".mask.npy")
            record = None
            if meta.exists():
                try:
                    record = json.loads(meta.read_text())
                    if record["status"] == "accepted":
                        a = np.load(image_path, mmap_mode="r", allow_pickle=False)
                        m = np.load(mask_path, mmap_mode="r", allow_pickle=False)
                        if a.shape != (args.size,args.size) or m.shape != a.shape or a.dtype != np.uint16 or m.dtype != np.uint8:
                            record = None
                    elif record["status"] == "error" or args.retry_rejections:
                        record = None
                except (OSError, ValueError, KeyError, TypeError):
                    record = None
            hit = record is not None
            if record is None:
                diagnostics = {}
                try:
                    if nominal_zenith(candidate.sensor_id,float(row["LAT"]),float(row["LON"])) > args.max_zenith + (0 if candidate.sensor == "himawari" else 2):
                        sample, reason = None, "nominal_view_zenith_prefilter"
                    else:
                        if open_error is not None:
                            raise RuntimeError(open_error)
                        if opened is None:
                            try:
                                opened = stack.enter_context(open_scene(candidate))
                            except Exception as exc:
                                open_error = f"{type(exc).__name__}: {exc}"
                                raise
                        sample, reason = process_candidate(candidate,row,args.wind_column,args.crop_km,args.size,args.min_valid_fraction,args.max_zenith,args.flip_south,args.tolerance_min,opened,args.min_inner_valid_fraction,diagnostics)
                    if sample is None:
                        record = {"status":"rejected","reason":reason,"diagnostics":diagnostics}
                    else:
                        for path, arr in ((image_path,sample.image),(mask_path,sample.valid_mask)):
                            tmp = path.with_suffix(".tmp")
                            with tmp.open("wb") as f:
                                np.save(f,arr,allow_pickle=False)
                            os.replace(tmp,path)
                        record = {"status":"accepted","sample":{f.name:getattr(sample,f.name) for f in fields(sample) if f.name not in {"image","valid_mask"}}}
                except Exception as exc:
                    LOG.exception("Failed %s %s",candidate.sensor_id,row["SID"])
                    record = {"status":"error","reason":f"{type(exc).__name__}:{exc}","diagnostics":diagnostics}
                tmp = meta.with_suffix(".tmp")
                tmp.write_text(json.dumps(record))
                os.replace(tmp,meta)
            if record["status"] == "accepted":
                results.append((Sample(**record["sample"],image=image_path,valid_mask=mask_path),None,hit))
            else:
                results.append((None,{"sensor_id":candidate.sensor_id,"timestamp":candidate.timestamp.isoformat(),"sid":str(row["SID"]),"reason":record["reason"],"status":record["status"],"files":";".join(candidate.files),"basin":str(row["BASIN"]),"wind_kts":float(row[args.wind_column]),"class_id":classify_usa_wind(float(row[args.wind_column])),"lat":float(row["LAT"]),"lon":float(row["LON"]),**record.get("diagnostics",{})},hit))
    return results


def scene_results(tasks, workers, pending_limit):
    if workers == 1:
        for task in tasks:
            yield run_scene(task)
        return
    # Spawn avoids inheriting HDF5/NetCDF locks and native thread pools.
    with ProcessPoolExecutor(max_workers=workers, mp_context=mp.get_context("spawn")) as pool:
        it = iter(tasks)
        pending = set()
        for _ in range(pending_limit):
            task = next(it,None)
            if task is not None:
                pending.add(pool.submit(run_scene,task))
        while pending:
            done,pending = wait(pending,return_when=FIRST_COMPLETED)
            for future in done:
                yield future.result()
                task = next(it,None)
                if task is not None:
                    pending.add(pool.submit(run_scene,task))


def split_sids(samples, ratios, seed, min_class_storms=0):
    """Find chronological cut points between non-overlapping storm intervals.

    Never divide a storm or silently discard crossing storms. Ratios approximate
    unique storm counts; fail if strict temporal splits are infeasible.
    """
    intervals = {}
    for sample in samples:
        t = utc_parse(sample.ibtracs_timestamp)
        lo, hi = intervals.get(sample.sid, (t, t))
        intervals[sample.sid] = (min(lo,t), max(hi,t))
    ordered = sorted(intervals, key=lambda sid: (intervals[sid][0],sid))
    n = len(ordered)
    cuts = []
    latest = None
    for i,sid in enumerate(ordered[:-1],1):
        latest = max(latest,intervals[sid][1]) if latest is not None else intervals[sid][1]
        if latest < intervals[ordered[i]][0]:
            cuts.append(i)
    if len(cuts) < 2:
        raise ValueError("No feasible strict chronological split. Collect more seasons; do not split overlapping storms randomly.")
    # Count unique storms representing each class in each interval.
    classes = defaultdict(set)
    for sample in samples:
        classes[sample.sid].add(sample.class_id)
    prefix = np.zeros((n+1,len(CLASS_NAMES)), dtype=np.int64)
    for i,sid in enumerate(ordered,1):
        prefix[i] = prefix[i-1]
        for cls in classes[sid]:
            prefix[i,cls] += 1
    best = None
    for a in cuts[:-1]:
        if np.any(prefix[a] < min_class_storms):
            continue
        for b in cuts:
            if b <= a or np.any(prefix[b]-prefix[a] < min_class_storms) or np.any(prefix[n]-prefix[b] < min_class_storms):
                continue
            candidate = ((a/n-ratios[0])**2+(b/n-sum(ratios[:2]))**2,a,b)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise ValueError("No chronological boundaries satisfy minimum class coverage")
    _,a,b = best
    mapping = {sid: name for name, group in zip(("train","val","test"),(ordered[:a],ordered[a:b],ordered[b:])) for sid in group}
    return {name:[s for s in samples if mapping[s.sid]==name] for name in ("train","val","test")}


def nominal_zenith(platform, lat, lon):
    sublon = {"himawari08":140.7,"himawari09":140.7,"goes16":-75.2,"goes17":-137.2,"goes18":-137.0}[platform]
    return viewing_zenith(sublon, lat, lon)

def viewing_zenith(sublon, lat, lon):
    c = math.cos(math.radians(lat))*math.cos(math.radians(normalize_lon(sublon-lon)))
    return math.degrees(math.acos(float(np.clip((42164*c-6378.137)/math.sqrt(42164**2+6378.137**2-2*42164*6378.137*c),-1,1))))


def write_h5(path: Path, samples: list[Sample], compression: str, crop_km: float, size: int,
             wind_col: str, args):
    if not samples:
        raise ValueError(f"Refusing to write empty split: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(samples)
    chunk_n = 1
    tmp = path.with_suffix(".h5.tmp")
    with h5py.File(tmp, "w") as h5:
        image_ds = h5.create_dataset(
            "images", shape=(n, size, size), dtype=np.uint16,
            chunks=(chunk_n, size, size), compression=compression,
            shuffle=True, fletcher32=True,
        )
        mask_ds = h5.create_dataset(
            "valid_mask", shape=(n, size, size), dtype=np.uint8,
            chunks=(chunk_n, size, size), compression=compression,
            shuffle=True, fletcher32=True,
        )
        string_fields = [
            "sample_id", "sid", "timestamp", "nominal_timestamp", "sensor", "sensor_id", "basin",
            "ibtracs_timestamp", "source_files", "nature", "wind_source",
        ]
        scalar_fields = [
            "lat", "lon", "wind_kts", "class_id", "time_offset_min",
            "viewing_zenith_deg", "flipped", "pressure_hpa", "valid_fraction", "inner_valid_fraction",
        ]
        for field in string_fields:
            h5.create_dataset(field, (n,), dtype=STRING_DTYPE)
        dtypes = {
            "lat": np.float32, "lon": np.float32, "wind_kts": np.float32,
            "class_id": np.int8, "time_offset_min": np.float32,
            "viewing_zenith_deg": np.float32, "flipped": np.uint8,
            "pressure_hpa": np.float32, "valid_fraction": np.float32, "inner_valid_fraction": np.float32,
        }
        for field in scalar_fields:
            h5.create_dataset(field, (n,), dtype=dtypes[field])

        for i, s in enumerate(tqdm(samples, desc=f"Writing {path.name}")):
            image_ds[i] = np.load(s.image, allow_pickle=False) if isinstance(s.image, Path) else s.image
            mask_ds[i] = np.load(s.valid_mask, allow_pickle=False) if isinstance(s.valid_mask, Path) else s.valid_mask
            for field in string_fields:
                h5[field][i] = getattr(s, field)
            for field in scalar_fields:
                h5[field][i] = getattr(s, field)

        h5.attrs["temperature_unit"] = "Kelvin"
        h5.attrs["temperature_encoding"] = "uint16 = round((Tb_K - temperature_offset_K) * temperature_scale)"
        h5.attrs["temperature_scale"] = TEMP_SCALE
        h5.attrs["temperature_offset_K"] = TEMP_OFFSET
        h5.attrs["crop_half_width_km"] = crop_km
        h5.attrs["image_size"] = size
        h5.attrs["band"] = 13
        h5.attrs["coordinate_system"] = "cyclone-centred local azimuthal-equidistant"
        h5.attrs["label_source"] = "IBTrACS"
        h5.attrs["wind_column"] = wind_col
        h5.attrs["class_names_json"] = json.dumps(CLASS_NAMES)
        h5.attrs["flip_southern_hemisphere"] = bool(args.flip_south)
        h5.attrs["processing_script_version"] = "6.0.0"
        h5.attrs["invalid_pixel_value"] = 0
        h5.attrs["wind_avg_period_minutes"] = 1
    os.replace(tmp, path)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-dir", type=Path, default=Path("data/raw"), help="Root containing GOES .nc and Himawari .DAT[.bz2] files")
    p.add_argument("--ibtracs", type=Path, default=Path("data/raw/ibtracs/ibtracs.ALL.list.v04r01.csv"), help="IBTrACS CSV file")
    p.add_argument("--output-dir", type=Path, default=Path("data/processed/intensity_kelvin_final"))
    p.add_argument("--sensor", choices=["all", "himawari", "goes"], default="all")
    p.add_argument("--inspect-only", action="store_true", help="Report discovered scans without processing")
    p.add_argument("--wind-column", choices=["USA_WIND"], default="USA_WIND", help="IBTrACS wind field; default USA_WIND")
    p.add_argument("--basins", nargs="*", default=None, help="Optional IBTrACS basin codes, e.g. NI WP EP")
    p.add_argument("--tolerance-min", type=float, default=DEFAULT_TOLERANCE_MIN)
    p.add_argument("--crop-km", type=float, default=DEFAULT_CROP_KM, help="Patch half-width in km; total width is twice this value")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--min-valid-fraction", type=float, default=0.95)
    p.add_argument("--min-inner-valid-fraction", type=float, default=0.99)
    p.add_argument("--plan-only", action="store_true", help="Match, filter and check split feasibility without satellite decoding")
    p.add_argument("--max-zenith", type=float, default=70.0)
    p.add_argument("--flip-south", action="store_true", help="Vertically flip Southern Hemisphere patches")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--train-ratio", type=float, default=DEFAULT_SPLITS[0])
    p.add_argument("--val-ratio", type=float, default=DEFAULT_SPLITS[1])
    p.add_argument("--test-ratio", type=float, default=DEFAULT_SPLITS[2])
    p.add_argument("--compression", choices=["lzf", "gzip"], default="lzf")
    p.add_argument("--overwrite", action="store_true", help="Publish a new validated generation; retain previous generations")
    p.add_argument("--min-class-storms", type=int, default=1, help="Minimum unique storms per class per split; 0 explicitly permits missing classes")
    p.add_argument("--retry-rejections", action="store_true", help="Retry cached QC/processing failures")
    p.add_argument("--limit", type=int, default=0, help="Debug limit; 0 means all candidates")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--dask-threads", type=int, default=2)
    p.add_argument("--max-pending-scenes", type=int, default=4)
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    if not args.raw_dir.exists():
        raise FileNotFoundError(args.raw_dir)
    if not args.ibtracs.exists():
        raise FileNotFoundError(args.ibtracs)
    if not all(math.isfinite(x) for x in (args.crop_km,args.tolerance_min,args.max_zenith,args.min_valid_fraction,args.min_inner_valid_fraction,args.train_ratio,args.val_ratio,args.test_ratio)):
        raise ValueError("Nonfinite configuration value")
    if not 0 <= args.tolerance_min <= 10 or not 0 < args.min_valid_fraction <= 1 or not 0 < args.max_zenith <= 90 or not 0 < args.min_inner_valid_fraction <= 1:
        raise ValueError("Invalid QC or temporal tolerance (maximum 10 minutes)")
    if args.size < 32 or args.crop_km <= 0:
        raise ValueError("size must be >= 32 and crop-km must be positive")
    ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    if any(x <= 0 for x in ratios) or not np.isclose(sum(ratios), 1.0, atol=1e-6):
        raise ValueError("Train/val/test ratios must all be positive and sum to 1")

    if min(args.workers,args.dask_threads,args.max_pending_scenes) < 1:
        raise ValueError("Worker, thread and pending scene counts must be positive")
    if args.limit < 0 or args.min_class_storms < 0:
        raise ValueError("Negative limit or class requirement")
    if args.limit:
        args.output_dir = args.output_dir / "debug"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = (args.output_dir / ".writer.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    expected_outputs = [args.output_dir / "current"]
    if not (args.overwrite or args.inspect_only or args.plan_only) and any(p.exists() for p in expected_outputs):
        raise FileExistsError(
            "One or more output HDF5 files already exist. Use --overwrite to replace them."
        )

    tracks = load_ibtracs(args.ibtracs, args.wind_column, args.basins)
    candidates = discover_candidates(args.raw_dir)
    if args.sensor != "all":
        candidates = [c for c in candidates if c.sensor == args.sensor]
    LOG.info("Discovery counts: %s", dict(__import__("collections").Counter(c.sensor_id for c in candidates)))
    if args.inspect_only:
        return
    # Match one nearest scan per platform and synoptic storm fix, before decoding.
    from bisect import bisect_left
    from dataclasses import fields
    by_platform = defaultdict(list)
    for c in candidates:
        by_platform[c.sensor_id].append(c)
    jobs = []
    excluded = Counter()
    for platform, scans in sorted(by_platform.items()):
        times = [c.timestamp for c in scans]
        platform_tracks = tracks[tracks["ISO_TIME_PARSED"].between(times[0]-timedelta(minutes=args.tolerance_min), times[-1]+timedelta(minutes=args.tolerance_min))]
        for _, row in platform_tracks.iterrows():
            fix = row["ISO_TIME_PARSED"].to_pydatetime()
            i = bisect_left(times, fix)
            options = scans[max(0,i-1):min(len(scans),i+1)]
            if not options:
                continue
            c = min(options, key=lambda x: (abs(x.timestamp-fix), x.timestamp))
            delta = (c.timestamp-fix).total_seconds()/60
            if abs(delta) > args.tolerance_min:
                continue
            if nominal_zenith(platform,float(row["LAT"]),float(row["LON"])) > args.max_zenith + (0 if c.sensor == "himawari" else 2):
                excluded[platform] += 1
                continue
            row = row.copy()
            row["TIME_OFFSET_MIN"] = delta
            jobs.append((c,row))
    jobs.sort(key=lambda x: (x[0].timestamp, x[0].sensor_id, str(x[1]["SID"])))
    if args.limit:
        jobs = jobs[:args.limit]
    LOG.info("Matched %d unique platform/storm/fix jobs", len(jobs))
    LOG.info("Geographic exclusions before decoding: %s", dict(excluded))
    if not jobs:
        raise ValueError("No eligible matched storm fixes")
    from types import SimpleNamespace
    if not args.limit:
        provisional = [SimpleNamespace(sid=str(r["SID"]), ibtracs_timestamp=r["ISO_TIME_PARSED"].isoformat(), class_id=classify_usa_wind(float(r[args.wind_column]))) for _,r in jobs]
        try:
            planned = split_sids(provisional, ratios, args.seed, args.min_class_storms)
            LOG.info("Preliminary split counts (before image QC): %s", {k:len(v) for k,v in planned.items()})
        except ValueError as exc:
            LOG.warning("Preliminary split not feasible: %s; QC may change feasibility", exc)
            if args.plan_only:
                raise
    if args.plan_only:
        return
    # Persistent, configuration-keyed cache. Atomic writes survive interruption.
    config = {k: str(v) for k,v in vars(args).items() if k not in {"overwrite","retry_rejections","log_level","workers","dask_threads","max_pending_scenes","plan_only"}}
    config["version"] = "6.0.0"
    config["code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config["dependencies"] = {}
    for package in ("satpy","pyresample","pyproj","numpy","xarray","netCDF4","h5py","pandas","dask"):
        try:
            config["dependencies"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            config["dependencies"][package] = "not-installed"
    config["ibtracs_sha256"] = hashlib.sha256(args.ibtracs.read_bytes()).hexdigest()
    cache = args.output_dir / ".cache" / stable_hash(json.dumps(config, sort_keys=True))
    cache.mkdir(parents=True, exist_ok=True)
    accepted = []
    rejected = []
    grouped = {}
    for candidate, row in jobs:
        key = (candidate.sensor_id,candidate.timestamp,tuple(candidate.files))
        if key not in grouped:
            grouped[key] = (candidate,[])
        grouped[key][1].append(row)
    tasks = [(c,rows,args,cache) for c,rows in grouped.values()]
    LOG.info("Processing %d unique scenes using %d workers, %d Dask threads per worker",len(tasks),args.workers,args.dask_threads)
    hits = errors = 0
    reasons = Counter()
    next_report = 100
    started = time.monotonic()
    with tqdm(total=len(jobs),desc="Storm fixes") as progress:
        for batch in scene_results(tasks,args.workers,args.max_pending_scenes):
            for sample,rejection,hit in batch:
                hits += int(hit)
                if sample is not None:
                    accepted.append(sample)
                else:
                    rejected.append(rejection)
                    errors += int(rejection["status"] == "error")
                    reasons[rejection["reason"]] += 1
            progress.update(len(batch))
            if progress.n >= next_report:
                LOG.info("Rejection reasons so far: %s", dict(reasons.most_common(8)))
                report = args.output_dir / "rejections_live.csv"
                tmp = report.with_suffix(".tmp")
                pd.DataFrame(rejected).to_csv(tmp,index=False)
                os.replace(tmp,report)
                next_report = progress.n + 100
            progress.set_postfix(accepted=len(accepted), rejected=len(rejected), errors=errors, cache=hits)
    pd.DataFrame(rejected, columns=None if rejected else ["status","reason","sid"]).to_csv(args.output_dir / "rejections_live.csv",index=False)
    # Completion order is nondeterministic with multiple workers; normalize it.
    accepted.sort(key=lambda s:(s.nominal_timestamp,s.sensor_id,s.sid,s.sample_id))
    rejected.sort(key=lambda r:(r["timestamp"],r["sensor_id"],r["sid"]))
    LOG.info("Scene processing finished in %.1f seconds; cache hits=%d",time.monotonic()-started,hits)
    if any(r["status"] == "error" for r in rejected):
        pd.DataFrame(rejected).to_csv(args.output_dir / "rejections.csv",index=False)
        raise RuntimeError("Source processing errors; fix and rerun with --retry-rejections before publishing dataset")

    pd.DataFrame(rejected).to_csv(args.output_dir / "rejections.csv", index=False)
    if not accepted:
        raise RuntimeError(
            "No samples were accepted. Inspect rejections/logs, filename patterns, "
            "Satpy reader support, and IBTrACS matching."
        )

    # De-duplicate exact same sensor/storm/time samples.
    dedup = {}
    for s in accepted:
        key = (s.sid, s.sensor_id, s.ibtracs_timestamp)
        dedup.setdefault(key, s)
    accepted = list(dedup.values())

    if args.limit:
        # Debug output is not a training split.
        splits = {"debug": accepted}
    else:
        splits = split_sids(accepted, ratios, args.seed, args.min_class_storms)
    seen = set()
    for name, rows in splits.items():
        ids = {s.sid for s in rows}
        if seen & ids:
            raise RuntimeError("Storm leakage")
        seen |= ids
    if not args.limit:
        for name, rows in splits.items():
            for cls in range(len(CLASS_NAMES)):
                count = len({s.sid for s in rows if s.class_id == cls})
                if count < args.min_class_storms:
                    raise ValueError(f"{name}: class {cls} has {count} storms; required {args.min_class_storms}. Collect data or explicitly lower --min-class-storms.")
    # Stage an immutable generation; only the current symlink is atomically switched.
    root = args.output_dir
    generations = root / "generations"
    generations.mkdir(exist_ok=True)
    args.output_dir = Path(tempfile.mkdtemp(prefix="generation-",dir=generations))
    (args.output_dir / "configuration.json").write_text(json.dumps(config,indent=2))
    for name, rows in splits.items():
        write_h5(
            args.output_dir / f"{name}.h5", rows, args.compression,
            args.crop_km, args.size, args.wind_column, args,
        )

    manifest_rows = []
    for split_name, rows in splits.items():
        for s in rows:
            manifest_rows.append({
                "split": split_name,
                "sample_id": s.sample_id,
                "sid": s.sid,
                "timestamp": s.timestamp,
                "nominal_timestamp": s.nominal_timestamp,
                "sensor": s.sensor,
                "sensor_id": s.sensor_id,
                "basin": s.basin,
                "lat": s.lat,
                "lon": s.lon,
                "wind_kts": s.wind_kts,
                "class_id": s.class_id,
                "ibtracs_timestamp": s.ibtracs_timestamp,
                "time_offset_min": s.time_offset_min,
                "viewing_zenith_deg": s.viewing_zenith_deg,
                "valid_fraction": s.valid_fraction,
                "inner_valid_fraction": s.inner_valid_fraction,
                "pressure_hpa": s.pressure_hpa,
                "nature": s.nature,
                "wind_source": s.wind_source,
                "flipped": s.flipped,
                "source_files": s.source_files,
            })
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    # Weights prevent multi-platform fixes receiving disproportionate influence.
    multiplicity = manifest.groupby(["sid","ibtracs_timestamp"])["sample_id"].transform("size")
    manifest["observation_weight"] = 1.0 / multiplicity
    manifest.to_csv(args.output_dir / "manifest.csv", index=False)
    for name, rows in splits.items():
        weights = manifest.loc[manifest.split.eq(name), "observation_weight"].to_numpy(dtype=np.float32)
        with h5py.File(args.output_dir / f"{name}.h5", "r+") as h5:
            h5.create_dataset("observation_weight", data=weights)
    for col in ("class_id", "sensor_id", "basin"):
        manifest.groupby(["split",col]).agg(samples=("sid","size"),storms=("sid","nunique")).to_csv(args.output_dir / f"distribution_{col}.csv")
    for name, rows in splits.items():
        with h5py.File(args.output_dir / f"{name}.h5", "r") as h5:
            if len(h5["images"]) != len(rows):
                raise RuntimeError("HDF5 sample count mismatch")
            for i in range(len(rows)):
                x = h5["images"][i]; m = h5["valid_mask"][i].astype(bool)
                if not np.all(np.isin(h5["valid_mask"][i], [0,1])) or not m.any() or np.any(x[~m] != 0) or np.any((x[m] < 15000)|(x[m] > 35000)):
                    raise RuntimeError(f"Invalid temperature encoding: {name}/{i}")
    # Training-only global statistics, excluding invalid pixels.
    if "train" in splits:
        count = 0; total = 0.; squares = 0.
        with h5py.File(args.output_dir / "train.h5", "r") as h5:
            for i in range(len(h5["images"])):
                x = h5["images"][i][h5["valid_mask"][i].astype(bool)].astype(np.float64)/TEMP_SCALE
                count += x.size; total += x.sum(); squares += np.square(x).sum()
        mean = total/count
        (args.output_dir / "normalization.json").write_text(json.dumps({"valid_pixels":count,"mean_kelvin":mean,"std_kelvin":math.sqrt(max(0.,squares/count-mean**2)),"fixed_range_baseline_kelvin":[180,330]},indent=2))
    pd.DataFrame(rejected).to_csv(args.output_dir / "rejections.csv", index=False)

    summary = {
        "script_version": "6.0.0",
        "matched_job_count": len(jobs),
        "geographic_exclusions": dict(excluded),
        "rejection_reasons": dict(reasons),
        "matched_scene_count": len(tasks),
        "cache_hits": hits,
        "workers": args.workers,
        "split_strategy": "strict_nonoverlapping_accepted_fix_intervals",
        "cache_directory": str(cache),
        "raw_dir": str(args.raw_dir.resolve()),
        "ibtracs": str(args.ibtracs.resolve()),
        "wind_column": args.wind_column,
        "basins": args.basins,
        "candidate_count": len(candidates),
        "accepted_count": len(accepted),
        "rejected_count": len(rejected),
        "storm_count": len({s.sid for s in accepted}),
        "class_counts": {
            str(i): int(sum(s.class_id == i for s in accepted))
            for i in range(len(CLASS_NAMES))
        },
        "split_summary": {
            name: {
                "samples": len(rows),
                "storms": len({s.sid for s in rows}),
                "class_counts": {
                    str(i): int(sum(s.class_id == i for s in rows))
                    for i in range(len(CLASS_NAMES))
                },
            }
            for name, rows in splits.items()
        },
        "parameters": {
            "crop_km": args.crop_km,
            "size": args.size,
            "tolerance_min": args.tolerance_min,
            "min_valid_fraction": args.min_valid_fraction,
            "max_zenith": args.max_zenith,
            "flip_south": args.flip_south,
            "seed": args.seed,
            "split_ratios": ratios,
        },
        "warnings": [
            "USA_WIND labels use one-minute wind thresholds; do not interpret as IMD three-minute categories.",
            "Validate Satpy reader calibration and quality-mask behavior against sample source files before full production runs.",
            "A SID is assigned wholly to one split; sensor observations of the same storm remain together.",
        ],
    }
    with open(args.output_dir / "processing_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Atomic publication on a local POSIX filesystem; old generations remain intact.
    (args.output_dir / "READY").write_text("validated\n")
    link = root / ".current-next"
    link.unlink(missing_ok=True)
    link.symlink_to(args.output_dir.relative_to(root), target_is_directory=True)
    os.replace(link, root / "current")
    LOG.info("Published dataset: %s", root / "current")
    LOG.info("Processing complete: %d accepted, %d rejected", len(accepted), len(rejected))
    LOG.info("Outputs written to %s", args.output_dir.resolve())
    for name, rows in splits.items():
        LOG.info("%s: %d samples, %d storms", name, len(rows), len({s.sid for s in rows}))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.error("Interrupted by user")
        sys.exit(130)
    except Exception:
        LOG.exception("Fatal preprocessing error")
        sys.exit(1)
