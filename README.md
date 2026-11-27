# CycloneAI

CycloneAI uses GOES-16 satellite imagery and NOAA IBTrACS storm tracks for two tasks. An Ultralytics pose model detects tropical cyclones and estimates their centers as single keypoints. A Swin-T model classifies the intensity of storm-centered crops from tropical depression (TD) through Category 5 and estimates wind speed in knots.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run this from the repository root.

```sh
uv sync --locked
```

Save the [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) CSV as `data/ibtracs.csv`. Place GOES-16 ABI L2 full-disk NetCDF files in `data/goes16/<YYYYMMDD_HHMM>/IR/` for channel C13 and `data/goes16/<YYYYMMDD_HHMM>/WV/` for channel C08. The timestamp is UTC.

To fetch imagery from NOAA's public archive, run `uv run python download_goes.py` from `scripts/`. The downloader uses dates configured in the script and writes to `data/goes16_dataset/` by default. Set its `OUTPUT_DIR` to `../data/goes16` before downloading, or move the downloaded timestamp folders into `data/goes16/`.

## Detection and center localization

Run from the repository root. The first command builds the dataset. The second validates it and prints split statistics before training.

```sh
uv run python scripts/preprocess_detection.py --goes data/goes16 --ibtracs data/ibtracs.csv --output data/processed/detection_pose_v2
uv run python scripts/train_detection.py --data data/processed/detection_pose_v2 --check-only
uv run python scripts/train_detection.py --data data/processed/detection_pose_v2 --project runs/cyclone_pose
```

Detection training defaults to GPU index `0`. Use `--device cpu` for CPU training. Checkpoint and confidence selection use the validation split. Use `--final-test` once the checkpoint and operating point are frozen. See the [detection evaluation summary](reports/detection_final_evaluation/presentation_summary.md) for recorded validation and test results.

## Intensity classification

Run from the repository root. The planning command audits candidate crops before the full preprocessing run.

```sh
uv run python scripts/preprocess_classification_raw.py --data-root data --plan-only
uv run python scripts/preprocess_classification_raw.py --data-root data
uv run python scripts/train_classification.py --data data/processed/classification_raw_v1 --output runs/classification/swin_t
```

Add `--validate-only` to the training command to check the dataset without training, or `--evaluate-test` to score the selected checkpoint on the held-out test split after training.

Datasets, checkpoints, experiment runs, and generated report files are stored locally. The detection evaluation write-up is kept under `reports/`. Installed dependencies are managed through `pyproject.toml` and `uv.lock`.
