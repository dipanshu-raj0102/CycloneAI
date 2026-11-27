#!/usr/bin/env python3
"""Evaluate cyclone presence on every validation image; one vote per image.

This is a binary image-level evaluation, not object IoU or centre accuracy.
Run from CycloneAI root or its scripts directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.webp'}
DEFAULT_MODEL = Path('../runs/cyclone_pose/yolosT2/train/weights/best_geo.pt')
DEFAULT_DATA = Path('../data/processed/detection_pose_v2')
DEFAULT_CONFIDENCE = .675


def project_root():
    for start in (Path.cwd(), Path(__file__).resolve().parent):
        for path in (start, *start.parents):
            if (path / DEFAULT_DATA).is_dir():
                return path
    return Path.cwd()


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def label_count(path):
    rows = [line.split() for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    for row in rows:
        try:
            values = np.asarray([float(x) for x in row])
        except ValueError as exc:
            raise ValueError(f'Invalid numeric label: {path}') from exc
        if (values.shape != (8,) or not np.isfinite(values).all()
                or values[0] != 0 or values[7] != 2
                or np.any((values[1:7] < 0) | (values[1:7] > 1))
                or np.any(values[3:5] <= 0)):
            raise ValueError(f'Expected class 0, box and visible centre: {path}')
    return len(rows)


def load_validation(data):
    images = data / 'images/val'
    labels = data / 'labels/val'
    if not images.is_dir() or not labels.is_dir():
        raise FileNotFoundError(f'Missing validation image/label directories: {data}')
    paths = sorted(p for p in images.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS)
    if not paths:
        raise ValueError('No validation images found')
    records = {}
    expected_labels = set()
    for image in paths:
        label = labels / image.relative_to(images).with_suffix('.txt')
        if label in expected_labels:
            raise ValueError(f'Multiple images share label: {label}')
        expected_labels.add(label)
        if not label.is_file():
            raise FileNotFoundError(f'Missing label (not treated as negative): {label}')
        records[image.relative_to(data).as_posix()] = label_count(label)
    if expected_labels != set(labels.rglob('*.txt')):
        raise ValueError('Unexpected/orphan validation labels')
    manifest = data / 'samples.jsonl'
    if manifest.is_file():
        expected = {}
        with manifest.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                if row['split'] != 'val':
                    continue
                key = row['image']
                if key in expected:
                    raise ValueError(f'Duplicate manifest image: {key}')
                expected[key] = len(row['objects'])
        if expected != records:
            raise ValueError('Validation images/label counts disagree with samples.jsonl')
    summary = data / 'summary.json'
    if summary.is_file():
        expected = read_json(summary)['splits']['val']
        actual = dict(images=len(records), positive=sum(n>0 for n in records.values()),
                      negative=sum(n==0 for n in records.values()), objects=sum(records.values()))
        if any(actual[k] != expected[k] for k in actual):
            raise ValueError(f'Validation counts disagree with summary.json: {actual}')
    if not any(records.values()) or all(records.values()):
        raise ValueError('Validation must contain both positive and negative images')
    return records


def predict_all(model, data, records, confidence, imgsz, batch, device):
    rows = []
    keys = list(records)
    for start in range(0, len(keys), batch):
        group = keys[start:start+batch]
        results = model.predict(source=[str(data / key) for key in group],
                                imgsz=imgsz, batch=batch, device=device,
                                conf=.001, max_det=300, verbose=False, save=False,
                                augment=False)
        seen = {}
        for result in results:
            key = Path(result.path).resolve().relative_to(data).as_posix()
            if key not in group or key in seen:
                raise RuntimeError(f'Unexpected or duplicate prediction: {key}')
            boxes = result.boxes
            scores = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=float) if boxes is not None else np.array([])
            classes = np.asarray(boxes.cls.detach().cpu().numpy()) if boxes is not None else np.array([])
            if scores.ndim != 1 or scores.shape != classes.shape or np.any(classes != 0) or not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
                raise ValueError(f'Unexpected cyclone prediction values: {key}')
            actual = int(records[key] > 0)
            detections = int(np.count_nonzero(scores >= confidence))
            predicted = int(detections > 0)
            outcome = {(1,1):'TP', (1,0):'FN', (0,1):'FP', (0,0):'TN'}[actual,predicted]
            seen[key] = dict(image=key, actual=actual, predicted=predicted, outcome=outcome,
                             ground_truth_objects=records[key], predicted_boxes=detections,
                             max_box_confidence=float(scores.max()) if scores.size else None)
        if set(seen) != set(group):
            raise RuntimeError(f'Missing predictions: {set(group)-set(seen)}')
        rows.extend(seen[key] for key in group)
        print(f'Predicted {len(rows)}/{len(keys)} images', flush=True)
    return rows


def calculate(rows):
    tp, fn, fp, tn = (sum(row['outcome']==x for row in rows) for x in ('TP','FN','FP','TN'))
    total = tp+tn+fp+fn
    if total != len(rows) or total == 0:
        raise ValueError('Invalid outcome counts')
    def div(a,b): return a/b if b else None
    recall, specificity = div(tp,tp+fn), div(tn,tn+fp)
    return dict(total_images=total, TP=tp, FN=fn, FP=fp, TN=tn,
                actual_positive=tp+fn, actual_negative=tn+fp,
                predicted_positive=tp+fp, predicted_negative=tn+fn,
                accuracy=div(tp+tn,total), precision=div(tp,tp+fp), recall=recall,
                specificity=specificity, f1=div(2*tp,2*tp+fp+fn),
                false_positive_rate=div(fp,fp+tn), false_negative_rate=div(fn,fn+tp),
                negative_predictive_value=div(tn,tn+fn),
                balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                mcc=div(tp*tn-fp*fn, math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))))


def save_outputs(out, rows, metrics, metadata):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=False)
    for name, content in [('metrics.json', metrics), ('evaluation_config.json', metadata)]:
        (out / name).write_text(json.dumps(content, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (out/'image_predictions.csv').open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for outcome in ('TP','TN','FP','FN'):
        with (out/f'{outcome.lower()}_images.csv').open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(r for r in rows if r['outcome']==outcome)
    with (out/'metrics.csv').open('w', newline='', encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['metric','value']); w.writerows(metrics.items())
    counts=np.array([[metrics['TP'],metrics['FN']],[metrics['FP'],metrics['TN']]])
    normalized=counts/counts.sum(axis=1,keepdims=True)
    classes=['Cyclone','No cyclone']
    for name, matrix in [('confusion_matrix',counts),('confusion_matrix_normalized',normalized)]:
        is_norm=name.endswith('normalized')
        with (out/f'{name}.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f); w.writerow(['Actual / Predicted',*classes]); w.writerows([[classes[i],*matrix[i]] for i in range(2)])
        fig,ax=plt.subplots(figsize=(8.4,6.5),layout='constrained')
        vmax=1 if is_norm else max(1,int(counts.max()))
        im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=vmax)
        fig.colorbar(im,ax=ax,shrink=.8,label='Fraction of actual class' if is_norm else 'Number of images')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=classes,yticklabels=classes,
               xlabel='Predicted class',ylabel='Actual class')
        ax.set_title(f'Cyclone presence — validation\n{metadata["model_run"]} | confidence ≥ {metadata["confidence"]:.3f} | n = {len(rows):,}',fontsize=14,pad=16)
        for i,j in np.ndindex(2,2):
            code=[['TP','FN'],['FP','TN']][i][j]
            value=f'{matrix[i,j]:.1%}' if is_norm else f'{counts[i,j]:,}'
            ax.text(j,i,f'{code}\n{value}',ha='center',va='center',fontsize=19,fontweight='bold',color='white' if matrix[i,j]>.5*vmax else '#172B45')
        for ext in ('png','svg'):
            fig.savefig(out/f'{name}.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    names=['accuracy','precision','recall','specificity','f1','balanced_accuracy']
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    vals=[metrics[k]*100 if metrics[k] is not None else 0 for k in names]
    bars=ax.barh([k.replace('_',' ').title() for k in names],vals,color='#23669D')
    ax.invert_yaxis(); ax.set(xlim=(0,108),xlabel='Score (%)',title=f'Image-level validation performance | n = {len(rows):,}')
    ax.bar_label(bars,labels=[f'{v:.2f}%' if metrics[k] is not None else 'N/A' for k,v in zip(names,vals)],padding=4)
    ax.grid(axis='x',alpha=.2); ax.set_axisbelow(True)
    for ext in ('png','svg'):
        fig.savefig(out/f'performance_metrics.{ext}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=project_root())
    parser.add_argument('--data',type=Path,help='Dataset directory; default detection_pose_v2')
    parser.add_argument('--model',type=Path,help='Selected checkpoint; default yolosT2/best_geo.pt')
    parser.add_argument('--confidence',type=float,help='Explicit threshold; default saved selection or 0.675')
    parser.add_argument('--batch',type=int,default=8)
    parser.add_argument('--device',default='0')
    parser.add_argument('--imgsz',type=int,help='Default build.json image_size or 640')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    root=args.root.resolve(); data=(args.data or root/DEFAULT_DATA).resolve()
    checkpoint=(args.model or root/DEFAULT_MODEL).resolve()
    out=(args.output or root/'reports/custom_confusion_matrix').resolve()
    if out.exists(): parser.error(f'Output exists: {out}; select a new --output')
    if not checkpoint.is_file(): parser.error(f'Checkpoint missing: {checkpoint}')
    if args.batch<1: parser.error('Batch must be positive')
    sha=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    confidence=args.confidence; source='explicit CLI override'
    if confidence is None:
        if checkpoint != (root/DEFAULT_MODEL).resolve(): parser.error('Custom checkpoint requires --confidence')
        run=checkpoint.parent.parent.parent
        op=run/'evaluation/operating_point.json'; sel=run/'checkpoint_selection.json'
        if op.is_file():
            obj=read_json(op)
            if obj.get('checkpoint_sha256') and obj['checkpoint_sha256']!=sha:
                parser.error('Operating point checkpoint hash mismatch')
            confidence=float(obj['confidence']); source=str(op)
        elif sel.is_file():
            obj=read_json(sel)
            if obj.get('split')!='val': parser.error('Selection report is not validation-based')
            confidence=float(obj['chosen']['threshold']); source=str(sel)
        else:
            confidence=DEFAULT_CONFIDENCE; source='previously selected yolosT2 threshold (fallback)'
    if not math.isfinite(confidence) or not .001<=confidence<=1: parser.error('Confidence must be between 0.001 and 1')
    build=data/'build.json'; imgsz=args.imgsz or (int(read_json(build)['config']['image_size']) if build.is_file() else 640)
    if imgsz<1: parser.error('Image size must be positive')
    records=load_validation(data)
    print(f'Validation images: {len(records)} | positive: {sum(n>0 for n in records.values())} | negative: {sum(n==0 for n in records.values())}')
    print(f'Model: {checkpoint}\nConfidence: {confidence:g} ({source})')
    import ultralytics
    from ultralytics import YOLO
    model=YOLO(str(checkpoint))
    if model.task!='pose' or tuple(model.model.yaml.get('kpt_shape',()))!=(1,3):
        raise ValueError('Expected single-centre pose checkpoint')
    names=model.names
    if len(names)!=1 or str(names[0]).lower()!='cyclone': raise ValueError('Expected single cyclone class')
    rows=predict_all(model,data,records,confidence,imgsz,args.batch,args.device)
    metrics=calculate(rows)
    metadata=dict(split='val',evaluation_unit='image',positive_rule='any cyclone box at or above confidence; no IoU or centre constraint',
                  model=str(checkpoint),model_run=checkpoint.parent.parent.parent.name,checkpoint_sha256=sha,
                  data=str(data),confidence=confidence,confidence_source=source,imgsz=imgsz,device=args.device,batch=args.batch,
                  inference_confidence_floor=.001,max_det=300,ultralytics_version=ultralytics.__version__,
                  manifest_verified=(data/'samples.jsonl').is_file(),summary_verified=(data/'summary.json').is_file())
    save_outputs(out,rows,metrics,metadata)
    print('\nActual rows / predicted columns: Cyclone, No cyclone')
    print(f'[[{metrics["TP"]}, {metrics["FN"]}],\n [{metrics["FP"]}, {metrics["TN"]}]]')
    for k,v in metrics.items():
        print(f'{k:26s}: {v:.4f}' if isinstance(v,float) else f'{k:26s}: {v}')
    print(f'\nResults: {out}')


if __name__=='__main__':
    main()
