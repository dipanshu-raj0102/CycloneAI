# CycloneAI

CycloneAI has two workflows. An Ultralytics YOLO pose model uses GOES-16 imagery and NOAA IBTrACS storm tracks to detect tropical cyclones and locate their centers. A six-class ClassCNN built with Keras and TensorFlow classifies cyclone intensity from a separate image dataset.

## Setup

Install Python 3.12 and [uv](https://docs.astral.sh/uv/), then run this from the repository root.

```sh
uv sync --locked
```

For detection, save the [NOAA IBTrACS](https://www.ncei.noaa.gov/products/international-best-track-archive) CSV as `data/ibtracs.csv`. Place GOES-16 ABI L2 full-disk NetCDF files in `data/goes16/<YYYYMMDD_HHMM>/IR/` for channel C13 and `data/goes16/<YYYYMMDD_HHMM>/WV/` for channel C08. The timestamp is UTC.

To fetch imagery from NOAA's public archive, run `uv run python download_goes.py` from `scripts/`. The downloader uses dates configured in the script and writes to `data/goes16_dataset/` by default. Set its `OUTPUT_DIR` to `../data/goes16` before downloading, or move the downloaded timestamp folders into `data/goes16/`.

## Detection and center localization

Run from the repository root. The first command builds the dataset. The second validates it and prints split statistics before training.

```sh
uv run python scripts/preprocess_detection.py --goes data/goes16 --ibtracs data/ibtracs.csv --output data/processed/detection_pose_v2
uv run python scripts/train_detection.py --data data/processed/detection_pose_v2 --check-only
uv run python scripts/train_detection.py --data data/processed/detection_pose_v2 --project runs/cyclone_pose
```

Detection training defaults to GPU index `0`. Use `--device cpu` for CPU training. The reported detector uses the selected `best_geo.pt` checkpoint, available locally as `models/detection/cyclone_detector.pt`. Both splits use a confidence threshold of **0.675** and a **100 km** geographic match radius. The checkpoint and threshold were frozen before test evaluation.

| Metric | Validation | Held-out test |
| --- | ---: | ---: |
| Geographic precision | 87.6% | 93.5% |
| Geographic recall | 83.9% | 74.8% |
| Geographic F1 | 85.7% | 83.1% |
| Matched / false / missed detections | 78 / 11 / 15 | 86 / 6 / 29 |
| Mean center error among matches | 32.95 km | 35.35 km |
| Median center error among matches | 22.70 km | 31.01 km |
| Recall within 50 km of covered targets | 63.4% | 57.4% |
| Box mAP50 / mAP50–95 | 92.5% / 60.1% | 88.9% / 59.3% |
| Pose mAP50 / mAP50–95 | 80.0% / 67.5% | 86.3% / 67.1% |

Geographic metrics deduplicate detections across overlapping tiles. Test coverage was 115 of 121 planned targets, with geographic recall of 71.1% when all planned targets are counted. Box AP evaluates proxy boxes and pose AP uses Object Keypoint Similarity (OKS). [Detection evaluation summary](reports/detection_final_evaluation/presentation_summary.md)

## Intensity classification

The evaluated ClassCNN takes 128 × 128 images with four channels and predicts one of six classes derived from the source wind labels. The classes are Low Pressure, Depression, Deep Depression, Cyclonic Storm, Severe Cyclonic Storm, and Very Severe Cyclonic Storm.

Place `Cyclone_Images.h5` and `Cyclone_Labels h5.npy` under `data/processed/classification/`. Use a trusted label file because the loader enables NumPy pickle deserialization. From the repository root, run the configuration used for the reported results.

```sh
uv run python scripts/ClassCNN_updated.py --data data/processed/classification --output runs/classification/classcnn_amp --epochs 20 --batch-size 32 --seed 42 --mixed-precision
```

The run used 21,076 images and a stratified random 80/20 image split. Both checkpoints were scored on the same 4,216 validation images. Epoch 14 had the highest validation accuracy and was selected as `models/classification/best.keras`. Storms can appear in both partitions under this split, so performance on independent storms remains unmeasured.

| Metric | Best checkpoint (epoch 14) | Last checkpoint (epoch 20) |
| --- | ---: | ---: |
| Correct predictions | 3,785 / 4,216 | 3,738 / 4,216 |
| Accuracy | 89.78% | 88.66% |
| Balanced accuracy (macro recall) | 88.05% | 87.03% |
| Macro precision | 89.97% | 90.78% |
| Macro F1 | 88.90% | 88.74% |
| Weighted F1 | 89.78% | 88.63% |
| Within-one-category accuracy | 97.39% | 95.73% |
| Quadratic weighted kappa | 0.951 | 0.923 |
| Macro ROC AUC | 0.983 | 0.979 |

[Best checkpoint metrics](reports/classification_final_evaluation/validation_best/metrics.json) · [Last checkpoint metrics](reports/classification_final_evaluation/validation_last/metrics.json)

Datasets, model weights, and experiment runs are local artifacts. The detection summary and classification metrics JSON files are eligible for version control.
