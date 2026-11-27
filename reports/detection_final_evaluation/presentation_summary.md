# GOES-16 cyclone detection — presentation results

Checkpoint: `best_geo.pt` (10,575,572 parameters).

Images: 11,870; covered storm IDs: 87.

## VAL — frozen checkpoint

Operating confidence: **0.675**; geographic match radius: **100 km**.

| Metric | Value |
|---|---:|
| Geographic precision | 87.6% |
| Geographic recall | 83.9% |
| Geographic F1 | 85.7% |
| Matched / false / missed | 78 / 11 / 15 |
| Mean / median matched centre error | 32.95 / 22.70 km |
| RMSE / P90 matched error | 41.35 / 67.28 km |
| Recall within 50 km (all covered targets) | 63.4% |
| Macro storm recall | 82.0% |
| Recall including uncovered planned targets | 83.9% |
| Planned target coverage | 93/93 |
| Tile false-positive rate | 0.5% |
| Scene false-positive rate | 1.1% |
| False detections per retained scene | 0.04 |
| Box mAP50 | 92.5% |
| Box mAP50–95 | 60.1% |
| Pose mAP50 | 80.0% |
| Pose mAP50–95 | 67.5% |

**Interpretation:** geographic matching counts each storm observation once per scene, after overlap deduplication. Native box AP evaluates proxy boxes, not actual storm extent. Pose AP uses OKS, not kilometres. Matched errors exclude misses and errors beyond the primary radius. Scene results cover retained tiles only. The test set must not be used to adjust threshold or choose a checkpoint.

## TEST — frozen checkpoint

Operating confidence: **0.675**; geographic match radius: **100 km**.

| Metric | Value |
|---|---:|
| Geographic precision | 93.5% |
| Geographic recall | 74.8% |
| Geographic F1 | 83.1% |
| Matched / false / missed | 86 / 6 / 29 |
| Mean / median matched centre error | 35.35 / 31.01 km |
| RMSE / P90 matched error | 41.94 / 69.28 km |
| Recall within 50 km (all covered targets) | 57.4% |
| Macro storm recall | 71.9% |
| Recall including uncovered planned targets | 71.1% |
| Planned target coverage | 115/121 |
| Tile false-positive rate | 0.4% |
| Scene false-positive rate | 0.0% |
| False detections per retained scene | 0.02 |
| Box mAP50 | 88.9% |
| Box mAP50–95 | 59.3% |
| Pose mAP50 | 86.3% |
| Pose mAP50–95 | 67.1% |

**Interpretation:** geographic matching counts each storm observation once per scene, after overlap deduplication. Native box AP evaluates proxy boxes, not actual storm extent. Pose AP uses OKS, not kilometres. Matched errors exclude misses and errors beyond the primary radius. Scene results cover retained tiles only. The test set must not be used to adjust threshold or choose a checkpoint.
