# CycloneAI

CycloneAI builds a YOLO pose dataset from geostationary satellite imagery and IBTrACS tropical cyclone best-track data, then trains a model to detect cyclones and locate their centers.

The preprocessing pipeline supports GOES-East, GOES-West, and Himawari imagery. It calibrates and reprojects satellite scans to a common grid, creates image tiles and YOLO pose labels, and writes metadata and quality-control previews. Training uses Ultralytics YOLO pose models; the visualization script reports training telemetry and spatial localization errors.

## Repository layout

```text
data/
  raw/                  Input satellite data, IBTrACS CSVs, and preprocessing code
  processed/            Generated YOLO dataset and metadata
scripts/
  prepare_yolo_data.py  Dataset preparation entry point
src/cycloneai/          Python package entry point
training/
  train_yolo.py         Dataset inspection and model training
  visualize_training.py Training and spatial evaluation reports
```

Raw satellite imagery, generated datasets, model weights, and training runs are local artifacts and are excluded by `.gitignore`. Keep the raw input files in the directory structure expected by the preprocessing script.

## Requirements

- Python 3.12 or newer
- `uv` (recommended) or `pip`
- GOES ABI Channel 13 and/or Himawari Band 13 source files
- IBTrACS best-track CSV data
- A compatible YOLO pose checkpoint for training; a CUDA-capable GPU is recommended

Install dependencies from the project metadata:

```bash
uv sync
```

Or install from `requirements.txt` with pip:

```bash
python -m pip install -r requirements.txt
```

## Prepare the dataset

Place inputs under `data/raw/` using the directory layout expected by the preprocessing code. By default it reads IBTrACS from `data/raw/ibtracs/ibtracs.ALL.list.v04r01.csv`, GOES data from `data/raw/goes/`, and Himawari data from `data/raw/himawari/`.

Run preprocessing from the repository root:

```bash
python data/raw/processing.py
```

The output defaults to `data/processed/yolo26data/`, including train/validation/test image and label splits, `cyclone_pose.yaml`, grid metadata, a tile manifest, and optional previews. Override input and output locations with `--ibtracs`, `--goes-dir`, `--himawari-dir`, and `--out-dir`. See all options with:

```bash
python data/raw/processing.py --help
```

## Train and evaluate

Inspect the prepared dataset and train a YOLO pose model:

```bash
python training/train_yolo.py \
  --data data/processed/yolo26data/cyclone_pose.yaml \
  --weights training/yolo26n-pose.pt \
  --project training/runs
```

Place a compatible YOLO pose checkpoint at the path passed to `--weights` first (the example expects `training/yolo26n-pose.pt`). Training options include `--epochs`, `--batch`, `--device`, `--project`, and `--name`; use `--help` for the complete list. The command above writes runs and checkpoints under `training/runs/`.

Generate or rerun spatial diagnostics for a training run:

```bash
python training/visualize_training.py \
  --run-dir training/runs/cyclone_yolo26 \
  --data-dir data/processed/yolo26data \
  --split both
```

## Data and model notes

The repository does not distribute the large satellite inputs, generated tiles, or model checkpoints. Obtain source data and model weights separately, and check their respective terms of use before redistribution. The preprocessing and training scripts expose additional parameters through their `--help` output.

## License

See [LICENSE](LICENSE).
