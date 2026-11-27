#!/usr/bin/env python3
"""Train and evaluate YOLO26-Pose on the GOES-16 detection_pose_v2 dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import platform
import tempfile
import shutil
import re
import importlib.util
from pathlib import Path

# Load the bundled helper by absolute path, not a generic module name.
_diag_path = Path(__file__).resolve().with_name("cyclone_detection_diagnostics.py")
if not _diag_path.is_file():
    raise FileNotFoundError(f"Missing bundled diagnostics helper: {_diag_path}")
_diag_spec = importlib.util.spec_from_file_location("cyclone_detection_diagnostics", _diag_path)
diag = importlib.util.module_from_spec(_diag_spec)
_diag_spec.loader.exec_module(diag)
for _name in ("diversity_plots", "training_plots", "checkpoint_plots", "report", "operating_rank"):
    if not callable(getattr(diag, _name, None)):
        raise ImportError(f"Incompatible diagnostics helper {_diag_path}: missing {_name}")
from collections import Counter, defaultdict
import cyclone_hard_negatives as hn

import numpy as np
import yaml
from pyproj import CRS, Geod, Transformer
from scipy.optimize import linear_sum_assignment

LOG = logging.getLogger('cyclone-training')
SPLITS = ('train', 'val', 'test')
GEOD = Geod(ellps='WGS84')
DISTANCES = (25, 50, 100, 150, 200, 300)
FLOOR = 0.001


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def write_csv(path, rows, fields):
    with Path(path).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def validate_split_counts(counts, summary, allow_approximate=False):
    total = sum(counts[s]['images'] for s in SPLITS)
    if total <= 0:
        raise ValueError('Empty dataset')
    fractions = {s: counts[s]['images'] / total for s in SPLITS}
    exact = all(counts[s]['images'] * 10 == total * n
                for s, n in zip(SPLITS, (8, 1, 1)))
    if summary.get('exact_image_split') is not exact:
        raise ValueError('Missing or inconsistent V2 exact_image_split metadata')
    reported = summary.get('image_split_fractions', {})
    if any(s not in reported or not math.isclose(reported[s], fractions[s], abs_tol=1e-9) for s in SPLITS):
        raise ValueError('Image split fractions disagree with actual counts')
    if not exact and not allow_approximate:
        raise ValueError(f'Expected exact 80:10:10 image split, found {fractions}')
    coverage = summary.get('planned_storm_observation_coverage')
    if coverage is None or not 0 <= coverage <= 1:
        raise ValueError('Missing or invalid V2 storm coverage metadata')
    LOG.info('Image split fractions: %s; planned storm coverage: %.2f%%', fractions, 100 * coverage)
    return fractions


def load_dataset(root, allow_approximate=False):
    root = root.resolve()
    names = ('build.json', 'samples.jsonl', 'scene_plan.json', 'summary.json', 'storm_splits.json')
    digest = hashlib.sha256()
    for name in names:
        with (root / name).open('rb') as f:
            digest.update(name.encode())
            for chunk in iter(lambda: f.read(1024 * 1024), b''):
                digest.update(chunk)
    build = json.loads((root / 'build.json').read_text())
    summary = json.loads((root / 'summary.json').read_text())
    if summary['storm_leakage'] or summary['scene_leakage']:
        raise ValueError('Preprocessing reported leakage')
    if json.loads((root / 'processing_failures.json').read_text()):
        raise ValueError('Dataset has preprocessing failures')
    plans = {s: {} for s in SPLITS}
    scene_ids, storm_splits = {}, {}
    storm_map = json.loads((root / 'storm_splits.json').read_text())
    for p in json.loads((root / 'scene_plan.json').read_text()):
        s, scene = p['split'], p['scene']['id']
        if scene in scene_ids or s not in SPLITS:
            raise ValueError(f'Duplicate or invalid scene: {scene}')
        scene_ids[scene] = s
        plans[s][scene] = p
        for sid in p['ids']:
            if storm_splits.setdefault(sid, s) != s or storm_map.get(sid) != s:
                raise ValueError(f'Storm split mismatch: {sid}')
    samples = {s: {} for s in SPLITS}
    labels = set()
    counts = {s: Counter() for s in SPLITS}
    with (root / 'samples.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            s, scene = r['split'], r['scene']
            im, lab = r['image'], r['label']
            if scene_ids.get(scene) != s or im in samples[s] or lab in labels:
                raise ValueError(f'Duplicate or inconsistent sample: {im}')
            if Path(im).parent != Path('images') / s or Path(lab) != Path('labels') / s / (Path(im).stem + '.txt'):
                raise ValueError(f'Unexpected paths: {im}, {lab}')
            if not (root / im).is_file() or not (root / lab).is_file():
                raise FileNotFoundError(im if not (root / im).is_file() else lab)
            objects = r['objects']
            expected = set(plans[s][scene]['positive_ids'])
            if any(o['sid'] not in expected for o in objects):
                raise ValueError(f'Unknown target in {im}')
            rows = (root / lab).read_text().splitlines()
            if len(rows) != len(objects):
                raise ValueError(f'Label count mismatch: {lab}')
            for row, obj in zip(rows, objects):
                a = np.array([float(v) for v in row.split()])
                if (a.shape != (8,) or not np.isfinite(a).all() or a[0] != 0 or a[7] != 2
                    or not ((a[1:7] >= 0) & (a[1:7] <= 1)).all() or min(a[3:5]) <= 0
                    or not np.allclose(a[1:7], obj['values'], rtol=0, atol=1e-7)):
                    raise ValueError(f'Invalid label: {lab}')
            # Check label contents and image stat in run fingerprint; not a pixel-content hash.
            st = (root / im).stat()
            digest.update(f'{im}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
            digest.update(('\n'.join(rows)).encode())
            samples[s][im] = r
            labels.add(lab)
            counts[s].update(images=1, positive=int(bool(objects)), negative=int(not objects), objects=len(objects))
    for s in SPLITS:
        actual = {str(p.relative_to(root)) for p in (root / 'images' / s).rglob('*') if p.is_file() and p.suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tif','.tiff','.webp','.dng','.mpo','.pfm','.heic'}}
        actual_labels = {str(p.relative_to(root)) for p in (root / 'labels' / s).rglob('*.txt')}
        if actual != set(samples[s]) or actual_labels != {r['label'] for r in samples[s].values()}:
            raise ValueError(f'Orphan image or label in {s}')
        if not counts[s]['positive'] or not counts[s]['negative'] or dict(counts[s]) != summary['splits'][s]:
            raise ValueError(f'Split counts disagree with preprocessing summary: {s}')
    validate_split_counts(counts, summary, allow_approximate)
    return build, plans, samples, digest.hexdigest(), counts


class Projection:
    def __init__(self, grid):
        self.grid = grid
        self.inverse = Transformer.from_crs(CRS.from_wkt(grid['crs_wkt']), 'EPSG:4326', always_xy=True)

    def geographic(self, r, u, v):
        g = self.grid
        col = r['x0'] + u * r['tile_size']
        row = r['y0'] + v * r['tile_size']
        x = g['x0_m'] + (col - .5) * g['dx_m']
        y = g['y0_m'] + (row - .5) * g['dy_m']
        lon, lat = self.inverse.transform(x, y)
        return float(lon), float(lat), float(col), float(row)


def distances(a, b):
    if not a or not b:
        return np.empty((len(a), len(b)))
    x, y = np.meshgrid(np.arange(len(a)), np.arange(len(b)), indexing='ij')
    lon1 = np.array([p['lon'] for p in a])[x]
    lat1 = np.array([p['lat'] for p in a])[x]
    lon2 = np.array([p['lon'] for p in b])[y]
    lat2 = np.array([p['lat'] for p in b])[y]
    return np.asarray(GEOD.inv(lon1.ravel().tolist(), lat1.ravel().tolist(), lon2.ravel().tolist(), lat2.ravel().tolist())[2]).reshape(x.shape) / 1000


def match(pred, truth, radius):
    """Maximize valid one-to-one matches first, then minimize distance."""
    d = distances(pred, truth)
    if not d.size:
        return []
    penalty = (min(d.shape) + 1) * (radius + 1)
    i, j = linear_sum_assignment(np.where(d <= radius, d, penalty))
    return [(int(a), int(b), float(d[a,b])) for a,b in zip(i,j) if d[a,b] <= radius]


def deduplicate(pred, radius, same_tile_radius=0.):
    """Score-ordered geographic suppression; invalid centres stay as false-positive candidates."""
    kept = []
    for p in sorted(pred, key=lambda p: (-p['score'], p['image'], p['index'])):
        if p['valid']:
            # Separate same-tile duplicate radius from overlapping-tile suppression.
            others = [q for q in kept if q['valid'] and q['image'] != p['image']
                      and abs(q['x0']-p['x0']) < min(q['tile_size'],p['tile_size'])
                      and abs(q['y0']-p['y0']) < min(q['tile_size'],p['tile_size'])]
            same = [q for q in kept if q['valid'] and q['image']==p['image']]
            if others and np.any(distances([p], others) <= radius):
                continue
            if same_tile_radius>0 and same and np.any(distances([p],same)<=same_tile_radius):
                continue
        kept.append(p)
    return kept


def ratio(a, b):
    return float(a / b) if b else None


def detection_counts(tp, fp, fn):
    return dict(tp=tp, fp=fp, fn=fn, precision=ratio(tp,tp+fp), recall=ratio(tp,tp+fn), f1=ratio(2*tp,2*tp+fp+fn))


def binary_counts(pairs):
    c = Counter((bool(t),bool(p)) for t,p in pairs)
    tp,fp,fn,tn = c[True,True],c[False,True],c[True,False],c[False,False]
    return {**detection_counts(tp,fp,fn), 'tn':tn, 'accuracy':ratio(tp+tn,tp+fp+fn+tn), 'specificity':ratio(tn,tn+fp), 'false_positive_rate':ratio(fp,tn+fp)}


def error_stats(errors):
    x = np.asarray(errors, dtype=float)
    out = {'count':len(x)}
    for name, func in [('mean',np.mean),('median',np.median),('rmse',lambda a: np.sqrt(np.mean(a*a))),('std',np.std),('p90',lambda a:np.percentile(a,90)),('p95',lambda a:np.percentile(a,95)),('max',np.max)]:
        out[name] = float(func(x)) if len(x) else None
    return out


def ground_truth(samples, plans):
    covered = defaultdict(dict)
    for r in samples.values():
        for o in r['objects']:
            old = covered[r['scene']].setdefault(o['sid'], {k:o[k] for k in ('sid','lat','lon','nature','match','offset_s')})
            if (old['lat'],old['lon']) != (o['lat'],o['lon']):
                raise ValueError('Inconsistent overlapping tile ground truth')
    expected = {scene:set(p['positive_ids']) for scene,p in plans.items()}
    return {scene:list(covered[scene].values()) for scene in plans}, expected


def infer(model, root, samples, projection, args, out):
    """Explicit batches; every tile must return exactly one Results object."""
    scenes = defaultdict(list)
    tile_scores = {}
    paths = list(samples)
    out.mkdir(parents=True, exist_ok=True)
    with (out / 'tile_predictions.jsonl').open('w') as f:
        for start in range(0,len(paths),args.eval_batch):
            group = paths[start:start+args.eval_batch]
            results = model.predict(source=[str(root / p) for p in group], imgsz=args.imgsz,
                                    batch=args.eval_batch, device=args.device, conf=FLOOR,
                                    max_det=300, verbose=False, save=False)
            seen = set()
            for result in results:
                rel = str(Path(result.path).resolve().relative_to(root))
                if rel not in group or rel in seen:
                    raise RuntimeError(f'Unexpected prediction: {rel}')
                seen.add(rel)
                r = samples[rel]
                h,w = result.orig_shape
                if (h,w) != (args.imgsz,args.imgsz):
                    raise ValueError(f'Unexpected image size: {rel}: {(h,w)}')
                boxes = result.boxes
                scores = boxes.conf.detach().cpu().numpy() if boxes is not None else np.array([])
                cls = boxes.cls.detach().cpu().numpy() if boxes is not None else np.array([])
                coords = boxes.xyxy.detach().cpu().numpy() if boxes is not None else np.empty((0,4))
                kpts = result.keypoints.data.detach().cpu().numpy() if result.keypoints is not None else np.empty((0,1,3))
                if kpts.shape != (len(scores),1,3) or np.any(cls != 0):
                    raise ValueError('Expected one cyclone class and one 3-value keypoint per box')
                tile_scores[rel] = float(max(scores, default=0.))
                for i,(score,box,kp) in enumerate(zip(scores,coords,kpts[:,0])):
                    u,v,kc = float(kp[0]/w),float(kp[1]/h),float(kp[2])
                    valid = bool(np.isfinite([u,v,kc]).all() and 0<=u<1 and 0<=v<1 and kc>=args.keypoint_conf and not (u==0 and v==0))
                    lon=lat=col=row=None
                    if valid:
                        lon,lat,col,row = projection.geographic(r,u,v)
                        valid = bool(np.isfinite([lon,lat]).all() and -90<=lat<=90)
                    p = dict(scene=r['scene'],image=rel,index=i,score=float(score),keypoint_conf=kc if math.isfinite(kc) else None,
                             u=u if math.isfinite(u) else None,v=v if math.isfinite(v) else None,
                             lon=lon if valid else None,lat=lat if valid else None,col=col if valid else None,row=row if valid else None,
                             valid=valid,box_xyxy=box.tolist(),x0=r['x0'],y0=r['y0'],tile_size=r['tile_size'])
                    scenes[r['scene']].append(p)
                    f.write(json.dumps(p,allow_nan=False)+'\n')
            if seen != set(group):
                raise RuntimeError('Missing prediction results')
            if start==0 or (start//args.eval_batch)%100==0:
                LOG.info('Predicted %d/%d tiles',min(start+args.eval_batch,len(paths)),len(paths))
    return scenes,tile_scores


def evaluate_at(scenes, truth, threshold, radius, details=False):
    tp=fp=fn=0
    errors=[]; records=[]; scene_rows=[]; storm_counts=defaultdict(Counter)
    for scene, targets in truth.items():
        all_pred = [p for p in scenes.get(scene,[]) if p['score']>=threshold]
        valid = [p for p in all_pred if p['valid']]
        pairs = match(valid,targets,radius)
        tp += len(pairs); fp += len(all_pred)-len(pairs); fn += len(targets)-len(pairs)
        errors.extend(d for _,_,d in pairs)
        if not details:
            continue
        by_target = {j:(i,d) for i,j,d in pairs}
        matched_pred = {id(valid[i]) for i,_,_ in pairs}
        scene_rows.append(dict(scene=scene,targets=len(targets),predictions=len(all_pred),tp=len(pairs),fp=len(all_pred)-len(pairs),fn=len(targets)-len(pairs)))
        for j,t in enumerate(targets):
            hit = j in by_target
            storm_counts[t['sid']].update(observations=1, matched=int(hit))
            row = dict(scene=scene,sid=t['sid'],status='TP' if hit else 'FN',true_lat=t['lat'],true_lon=t['lon'],nature=t['nature'],match=t['match'],offset_s=t['offset_s'])
            if hit:
                i,d=by_target[j]; p=valid[i]
                row.update(pred_lat=p['lat'],pred_lon=p['lon'],score=p['score'],error_km=d,image=p['image'])
            records.append(row)
        for p in all_pred:
            if id(p) not in matched_pred:
                records.append(dict(scene=scene,status='FP',pred_lat=p['lat'],pred_lon=p['lon'],score=p['score'],image=p['image']))
    return {**detection_counts(tp,fp,fn),'centre_error_km':error_stats(errors)}, records, scene_rows, storm_counts


def centre_ap(scenes, truth, radius):
    """Score-ranked greedy one-to-one geographic AP, all-point precision envelope."""
    ranked = sorted((p for group in scenes.values() for p in group), key=lambda p:(-p['score'],p['scene'],p['image'],p['index']))
    assigned = defaultdict(set); hits=[]; n=sum(map(len,truth.values()))
    if not n:
        return None
    for p in ranked:
        targets=truth[p['scene']]; hit=False
        if p['valid'] and targets:
            d=distances([p],targets)[0]
            for j in np.argsort(d):
                if d[j]>radius: break
                if int(j) not in assigned[p['scene']]:
                    assigned[p['scene']].add(int(j)); hit=True; break
        hits.append(hit)
    if not hits: return 0.
    tp=np.cumsum(hits); recall=tp/n; precision=tp/np.arange(1,len(hits)+1)
    # Group tied scores before integrating to avoid artificial rank gains.
    scores=np.array([p['score'] for p in ranked]); ends=np.r_[np.flatnonzero(scores[:-1]!=scores[1:]),len(scores)-1]
    r=np.r_[0,recall[ends],1]; q=np.r_[0,precision[ends],0]
    q=np.maximum.accumulate(q[::-1])[::-1]
    return float(np.sum(np.diff(r)*q[1:]))


def plots(out, curve, records, threshold):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=[r['threshold'] for r in curve]
    fig,ax=plt.subplots()
    for name in ('precision','recall','f1'):
        ax.plot(x,[r[name] if r[name] is not None else np.nan for r in curve],label=name)
    ax.axvline(threshold,ls='--',label='locked threshold'); ax.set(xlabel='Box confidence',ylabel='Score',ylim=(0,1.02)); ax.legend(); fig.tight_layout(); fig.savefig(out/'confidence_curves.png',dpi=160); plt.close(fig)
    fig,ax=plt.subplots(); ax.plot([r['recall'] for r in curve if r['precision'] is not None],[r['precision'] for r in curve if r['precision'] is not None],marker='.'); ax.set(xlabel='Recall',ylabel='Precision',xlim=(0,1.02),ylim=(0,1.02)); fig.tight_layout(); fig.savefig(out/'centre_pr.png',dpi=160); plt.close(fig)
    errors=sorted(r['error_km'] for r in records if r['status']=='TP')
    if errors:
        fig,ax=plt.subplots(); ax.step(errors,np.arange(1,len(errors)+1)/len(errors),where='post'); ax.set(xlabel='Matched centre error (km)',ylabel='Cumulative fraction'); fig.tight_layout(); fig.savefig(out/'centre_error_cdf.png',dpi=160); plt.close(fig)


def filtered_box_counts(raw, samples, threshold, imgsz, iou=.5):
    """Independent tile-box matching at fixed IoU, without cross-tile suppression."""
    by_image=defaultdict(list)
    for group in raw.values():
        for p in group:
            if p['score']>=threshold:
                by_image[p['image']].append(p['box_xyxy'])
    tp=fp=fn=0
    rows=[]
    for image,r in samples.items():
        gt=[]
        for obj in r['objects']:
            x,y,w,h=obj['values'][:4]
            gt.append([(x-w/2)*imgsz,(y-h/2)*imgsz,(x+w/2)*imgsz,(y+h/2)*imgsz])
        pred=by_image[image]; n=0
        if pred and gt:
            a=np.asarray(pred); b=np.asarray(gt)
            lo=np.maximum(a[:,None,:2],b[None,:,:2]); hi=np.minimum(a[:,None,2:],b[None,:,2:])
            inter=np.maximum(hi-lo,0).prod(axis=2)
            union=(a[:,2:]-a[:,:2]).prod(axis=1)[:,None]+(b[:,2:]-b[:,:2]).prod(axis=1)[None,:]-inter
            overlap=np.divide(inter,union,out=np.zeros_like(inter),where=union>0)
            ii,jj=linear_sum_assignment(np.where(overlap>=iou,1-overlap,min(overlap.shape)+1))
            n=int(np.sum(overlap[ii,jj]>=iou))
        tp+=n; fp+=len(pred)-n; fn+=len(gt)-n
        rows.append(dict(image=image,scene=r['scene'],targets=len(gt),predictions=len(pred),tp=n,fp=len(pred)-n,fn=len(gt)-n))
    return dict(threshold=threshold,iou=iou,**detection_counts(tp,fp,fn)),rows


def geographic_report(raw, tile_scores, samples, plans, args, out, threshold=None):
    truth, expected=ground_truth(samples,plans)
    evaluated={r['scene'] for r in samples.values()}
    # Scene universe is retained tiles only, not full-disk or rejected tiles.
    truth={s:t for s,t in truth.items() if s in evaluated}
    scenes={s:deduplicate(raw.get(s,[]),args.dedup_km,getattr(args,'same_tile_dedup_km',0.)) for s in truth}
    curve=[]
    # Fixed grid chosen before looking at test. No threshold selection on test.
    for t in sorted(set(confidence_grid()+([threshold] if threshold is not None else []))):
        m,_,_,_=evaluate_at(scenes,truth,t,args.match_km)
        curve.append({'threshold':float(t),**{k:v for k,v in m.items() if k!='centre_error_km'}})
    if threshold is None:
        threshold=choose_operating_point(curve,args)['threshold']
    metrics,records,scene_rows,storm_counts=evaluate_at(scenes,truth,threshold,args.match_km,True)
    total_expected=sum(map(len,expected.values())); total_covered=sum(map(len,truth.values()))
    errors=[r['error_km'] for r in records if r['status']=='TP']
    # Independent rematching at each radius so misses count in denominator.
    radius_metrics={}
    for d in DISTANCES:
        m,_,_,_=evaluate_at(scenes,truth,threshold,d)
        radius_metrics[str(d)]={**{k:v for k,v in m.items() if k!='centre_error_km'},'AP':centre_ap(scenes,truth,d)}
    metrics.update(threshold=threshold,match_km=args.match_km,dedup_km=args.dedup_km,keypoint_conf=args.keypoint_conf,
                   raw_predictions_at_threshold=sum(p['score']>=threshold for ps in raw.values() for p in ps),
                   deduplicated_predictions_at_threshold=sum(p['score']>=threshold for ps in scenes.values() for p in ps),
                   invalid_centres_at_threshold=sum(p['score']>=threshold and not p['valid'] for ps in scenes.values() for p in ps),
                   coverage=dict(planned_scenes=len(plans),evaluated_scenes=len(evaluated),planned_target_observations=total_expected,covered_target_observations=total_covered,uncovered_target_observations=total_expected-total_covered,coverage_fraction=ratio(total_covered,total_expected),recall_including_uncovered_as_misses=ratio(metrics['tp'],total_expected)),
                   centre_radius_metrics=radius_metrics,macro_storm_recall=float(np.mean([c['matched']/c['observations'] for c in storm_counts.values()])) if storm_counts else None,
                   tile_presence=binary_counts((bool(r['objects']),tile_scores[p]>=threshold) for p,r in samples.items()),
                   scene_presence=binary_counts((bool(truth[s]),any(p['score']>=threshold for p in scenes[s])) for s in truth),
                   mean_false_positives_per_evaluated_scene=ratio(metrics['fp'],len(truth)),
                   matched_error_within_km={str(d):ratio(sum(e<=d for e in errors),len(errors)) for d in DISTANCES})
    metrics['same_tile_dedup_km']=getattr(args,'same_tile_dedup_km',0.)
    metrics['precision_constraint_met'] = precision_ok(metrics,args)
    metrics['minimum_precision'] = getattr(args,'min_precision',0.)
    metrics['diagnostics'] = diag.report(out, metrics, records, scene_rows, storm_counts, expected, truth, scenes, tile_scores, samples, args, distances)
    dump(out/'geographic_metrics.json',metrics)
    write_csv(out/'confidence_curve.csv',curve,['threshold','tp','fp','fn','precision','recall','f1'])
    write_csv(out/'centre_matches.csv',records,['scene','sid','status','true_lat','true_lon','pred_lat','pred_lon','score','error_km','image','nature','match','offset_s'])
    write_csv(out/'scene_metrics.csv',scene_rows,['scene','targets','predictions','tp','fp','fn'])
    storm_errors=defaultdict(list)
    for r in records:
        if r['status']=='TP': storm_errors[r['sid']].append(r['error_km'])
    write_csv(out/'storm_metrics.csv',[dict(sid=s,**c,recall=ratio(c['matched'],c['observations']),**{f'error_km_{k}':v for k,v in error_stats(storm_errors[s]).items()}) for s,c in sorted(storm_counts.items())],['sid','observations','matched','recall','error_km_count','error_km_mean','error_km_median','error_km_rmse','error_km_std','error_km_p90','error_km_p95','error_km_max'])
    write_csv(out/'coverage.csv',[dict(scene=s,sid=sid,covered=int(sid in {o['sid'] for o in truth.get(s,[])})) for s,ids in expected.items() for sid in sorted(ids)],['scene','sid','covered'])
    plots(out,curve,records,threshold)
    return metrics


def native_metrics(model, data, split, args, out):
    # The package's default sigma=1/N is too permissive for a single keypoint.
    # The callback timing differs by version; use validator subclass for explicit verification.
    from ultralytics.models.yolo.pose.val import PoseValidator
    class CheckedValidator(PoseValidator):
        def init_metrics(self, model):
            super().init_metrics(model)
            if not np.allclose(self.sigma,[args.oks_sigma]):
                raise RuntimeError('Installed Ultralytics ignores custom OKS sigma; upgrade it')
    m=model.val(validator=CheckedValidator,data=str(data),split=split,imgsz=args.imgsz,batch=args.eval_batch,
                device=args.device,workers=args.workers,conf=FLOOR,max_det=300,plots=True,save_json=True,
                project=str(out),name='ultralytics',exist_ok=False)
    def finite(v):
        v=float(v); return v if math.isfinite(v) else None
    result={'oks_sigma':args.oks_sigma,'metrics':{str(k):finite(v) for k,v in m.results_dict.items()},'speed_ms_per_image':{str(k):finite(v) for k,v in m.speed.items()}}
    for name,metric in [('box',m.box),('pose',m.pose)]:
        result[name]={k:finite(getattr(metric,k)) for k in ('mp','mr','map50','map75','map')}
        result[name]['f1_at_framework_operating_point']=finite(np.mean(metric.f1)) if len(metric.f1) else None
    dump(out/'ultralytics_metrics.json',result)
    return result


def diversity_report(samples, plans):
    result = {}
    for split in SPLITS:
        truth, expected = ground_truth(samples[split], plans[split])
        observations = Counter(o['sid'] for targets in truth.values() for o in targets)
        tiles = Counter(o['sid'] for r in samples[split].values() for o in r['objects'])
        result[split] = dict(unique_storms=len(observations),
            unique_storm_observations=sum(observations.values()),
            planned_storm_observations=sum(map(len, expected.values())),
            positive_tiles=sum(bool(r['objects']) for r in samples[split].values()),
            unique_retained_scenes=len({r['scene'] for r in samples[split].values()}),
            tile_annotations=sum(tiles.values()),
            storms=[dict(sid=sid, observations=n, tile_annotations=tiles[sid])
                    for sid,n in sorted(observations.items())])
    return result


def training_kwargs(args, data, run):
    return dict(data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        device=args.device, workers=args.workers, seed=args.seed, deterministic=True,
        pose=args.pose, patience=args.patience, amp=not args.no_amp, optimizer='AdamW', lr0=args.lr0,
        lrf=.01, weight_decay=args.weight_decay, cos_lr=True, warmup_epochs=args.warmup_epochs,
        warmup_bias_lr=args.warmup_bias_lr, freeze=args.freeze or None,
        hsv_h=0., hsv_s=0., hsv_v=0., bgr=0., fliplr=0., flipud=0.,
        degrees=args.degrees, translate=args.translate, scale=args.scale,
        shear=0., perspective=0., mosaic=0., mixup=0., copy_paste=0., cutmix=0., erasing=0.,
        plots=True, save=True, save_period=args.save_period, project=str(run), name='train',
        exist_ok=False, cache=False, val=True)


def selection_metrics(raw, samples, plans, args):
    truth, expected = ground_truth(samples, plans)
    evaluated = {r['scene'] for r in samples.values()}
    truth = {s:t for s,t in truth.items() if s in evaluated}
    scenes = {s:deduplicate(raw.get(s, []), args.dedup_km,getattr(args,'same_tile_dedup_km',0.)) for s in truth}
    thresholds = [args.confidence] if args.confidence is not None else confidence_grid()
    rows = []
    for threshold in thresholds:
        m, _, _, _ = evaluate_at(scenes, truth, threshold, args.match_km)
        rows.append(dict(threshold=float(threshold), **m))
    winner = choose_operating_point(rows,args)
    m, _, _, storms = evaluate_at(scenes, truth, winner['threshold'], args.match_km, True)
    winner['selection_score'] = diag.operating_rank(winner,args)[0]
    winner['precision_constraint_met'] = precision_ok(winner,args)
    winner['macro_storm_recall'] = float(np.mean([c['matched']/c['observations'] for c in storms.values()])) if storms else None
    winner['recall_at_50_km'] = evaluate_at(scenes,truth,winner['threshold'],50)[0]['recall']
    winner['recall_including_uncovered'] = ratio(m['tp'], sum(map(len, expected.values())))
    return winner


def precision_ok(row,args):
    return bool(row['tp']>0 and (row.get('precision') or 0)>=getattr(args,'min_precision',0.))


def choose_operating_point(rows,args):
    eligible=[r for r in rows if precision_ok(r,args)]
    if eligible: return max(eligible,key=lambda r:diag.operating_rank(r,args))
    # Explicit fallback, reported; never treat an empty prediction set as feasible.
    return max(rows,key=lambda r:diag.operating_rank(r,args))


def confidence_grid():
    return sorted(set([FLOOR,.005,.01,.02,.03,.04,.05,.085,.124,.303,.305,*np.linspace(.075,.95,36).round(3),.99]))


def checkpoint_rank(row):
    return (row.get('precision_constraint_met',True), row.get('selection_score',row['f1']) or 0., row['recall'] or 0., row['macro_storm_recall'] or 0., row.get('recall_at_50_km') or 0.,
            -row['centre_error_km']['mean'] if row['centre_error_km']['mean'] is not None else -math.inf)


def select_checkpoint(YOLO, best, root, samples, plans, projection, args, run):
    # Post-training selection: framework early stopping is deliberately unchanged.
    baseline = getattr(args,'baseline',None)
    # Baseline is evaluated first; exact ties retain the original model.
    candidates = ([baseline] if baseline is not None else []) + [best]
    if args.selection == 'geographic':
        candidates += sorted(best.parent.glob('epoch*.pt'), key=lambda p:int(re.search(r'epoch(\d+)',p.stem).group(1)) if re.search(r'epoch(\d+)',p.stem) else -1)
        last = best.parent / 'last.pt'
        if last.is_file(): candidates.append(last)
    candidates = list(dict.fromkeys(p.resolve() for p in candidates))
    rows = []
    for path in candidates:
        LOG.info('Validation-only checkpoint comparison: %s', path.name)
        candidate = YOLO(str(path))
        with tempfile.TemporaryDirectory(prefix='checkpoint_eval_',dir=run) as temp:
            raw,_ = infer(candidate,root,samples,projection,args,Path(temp))
            row = selection_metrics(raw,samples,plans,args)
        rows.append(dict(checkpoint=str(path), **row))
        del candidate, raw
    chosen = max(rows,key=checkpoint_rank)
    baseline_row = next((r for r in rows if baseline is not None and Path(r['checkpoint'])==baseline.resolve()),None)
    destination = best.parent / 'best_geo.pt'
    if args.selection == 'geographic':
        shutil.copy2(chosen['checkpoint'], destination)
    else:
        destination = best
    dump(run/'checkpoint_selection.json',dict(method=args.selection, split='val', match_km=args.match_km,
        beta=getattr(args,'beta',1.), chosen=chosen, candidates=rows, selected_checkpoint=str(destination),
        baseline=baseline_row, retained_baseline=bool(baseline_row is not None and chosen is baseline_row),
        selection_score_delta=(chosen['selection_score']-baseline_row['selection_score']) if baseline_row is not None else None,
        f1_delta=(chosen['f1'] or 0)-(baseline_row['f1'] or 0) if baseline_row is not None else None,
        same_tile_dedup_km=getattr(args,'same_tile_dedup_km',0.)))
    diag.checkpoint_plots(run, rows)
    return destination, chosen['threshold']


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('../data/processed/detection_pose_v2'))
    p.add_argument('--project',type=Path,default=Path('../runs/cyclone_pose'))
    p.add_argument('--name',default='yolo26s_v6')
    p.add_argument('--model',default='yolo26s-pose.pt')
    p.add_argument('--baseline',type=Path,help='Existing trained checkpoint to include in validation-only selection')
    mode=p.add_mutually_exclusive_group(); mode.add_argument('--resume',type=Path); mode.add_argument('--checkpoint',type=Path); mode.add_argument('--check-only',action='store_true')
    p.add_argument('--epochs',type=int,default=20); p.add_argument('--batch',type=int,default=8); p.add_argument('--eval-batch',type=int,default=8)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--device',default='0'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=5); p.add_argument('--imgsz',type=int,default=None)
    p.add_argument('--oks-sigma',type=float,default=.1); p.add_argument('--match-km',type=float,default=150); p.add_argument('--dedup-km',type=float,default=100)
    p.add_argument('--same-tile-dedup-km',type=float,default=0.,help='Optional geographic suppression inside one tile; 0 disables')
    p.add_argument('--keypoint-conf',type=float,default=.25); p.add_argument('--confidence',type=float,default=None,help='Fixed box threshold; otherwise select on validation')
    p.add_argument('--allow-approximate-split', action='store_true', help='Accept a published V2 dataset with non-exact image ratios')
    test = p.add_mutually_exclusive_group()
    test.add_argument('--skip-test',action='store_true',help='Validation only (default)')
    test.add_argument('--final-test',action='store_true',help='Evaluate held-out test after all choices are frozen')
    p.add_argument('--no-amp',action='store_true')
    p.add_argument('--lr0',type=float,default=.00005)
    p.add_argument('--pose', type=float, default=12.0)
    p.add_argument('--weight-decay',type=float,default=.001)
    p.add_argument('--warmup-epochs',type=float,default=1.)
    p.add_argument('--warmup-bias-lr',type=float,default=.00005)
    p.add_argument('--degrees',type=float,default=10.)
    p.add_argument('--translate',type=float,default=.05)
    p.add_argument('--scale',type=float,default=.10)
    p.add_argument('--freeze',type=int,default=0)
    p.add_argument('--save-period',type=int,default=1)
    p.add_argument('--selection',choices=['geographic','native'],default='geographic')
    p.add_argument('--audit-only',action='store_true',help='Print storm diversity without training')
    p.add_argument('--beta',type=float,default=1.,help='Validation F-beta objective; 2 emphasizes recall, 1 reproduces F1')
    p.add_argument('--bootstrap',type=int,default=1000,help='Storm-cluster bootstrap replicates; 0 disables')
    p.add_argument('--min-precision',type=float,default=.85)
    p.add_argument('--mine-from',type=Path,help='Trained one-keypoint checkpoint for training-only mining')
    p.add_argument('--mine-only',action='store_true')
    p.add_argument('--mine-confidence',type=float,default=.30)
    p.add_argument('--approved-negatives',type=Path,help='Reviewed candidate CSV with approved=yes')
    p.add_argument('--hard-fraction',type=float,default=.15)
    p.add_argument('--max-hard-repeat',type=int,default=3)
    p.add_argument('--mining-cache',type=Path,help='Existing mining directory containing predictions and candidates')
    p.add_argument('--allow-legacy-mining-cache',action='store_true')
    p.add_argument('--screen-only',action='store_true',help='Screen cached candidates against IBTrACS-derived scene plan')
    p.add_argument('--accept-screened',action='store_true',help='Explicitly accept catalog-clear candidates as provisional negatives')
    p.add_argument('--ordinary-training',action='store_true',help='Skip mining and use original training split')
    p.add_argument('--screen-km',type=float,default=600.)
    p.add_argument('--storm-balance',action='store_true')
    args=p.parse_args(argv)
    if not math.isfinite(args.same_tile_dedup_km) or not 0<=args.same_tile_dedup_km<=args.dedup_km: p.error('Invalid same-tile dedup radius')
    if args.baseline is not None:
        args.baseline=args.baseline.resolve()
        if not args.baseline.is_file(): p.error('Baseline checkpoint not found')
        if args.selection!='geographic' or args.checkpoint or args.resume or args.mine_only or args.screen_only or args.final_test: p.error('--baseline requires new geographic training; test separately')
    if not 0<=args.min_precision<=1 or not FLOOR<=args.mine_confidence<=1 or not 0<args.hard_fraction<=.5: p.error('Invalid mining or precision parameters')
    if args.max_hard_repeat<1 or not math.isfinite(args.screen_km) or args.screen_km<=0: p.error('Invalid screening or repetition limit')
    if args.ordinary_training and any((args.mine_from,args.mine_only,args.mining_cache,args.approved_negatives,args.accept_screened,args.screen_only,args.storm_balance)): p.error('--ordinary-training cannot be combined with mining options')
    if (args.mine_only or args.mining_cache or args.approved_negatives or args.storm_balance or args.screen_only or args.accept_screened) and not args.mine_from: p.error('Mining options require --mine-from')
    if args.mine_only and any((args.approved_negatives,args.mining_cache,args.screen_only,args.accept_screened)): p.error('--mine-only creates a fresh cache')
    if args.mine_from and (args.resume or args.checkpoint or args.final_test): p.error('Mining is a separate training experiment, not resume/evaluation')
    if args.mine_from and not args.mine_only and not any((args.approved_negatives,args.screen_only,args.accept_screened)): p.error('Use --approved-negatives, --screen-only or explicit --accept-screened')
    if args.approved_negatives and args.accept_screened: p.error('Choose manual approvals or provisional automatic acceptance, not both')
    if args.screen_only and args.accept_screened: p.error('Screen-only does not train')

    if not math.isfinite(args.beta) or not 0<args.beta<=10 or args.bootstrap<0: p.error('Invalid beta or bootstrap count')
    if args.device not in ('cpu','mps') and not args.device.isdigit(): p.error('Use a single GPU index, cpu, or mps')
    if min(args.epochs,args.batch,args.eval_batch,args.patience)<=0 or args.workers<0 or not 0<args.oks_sigma<=1 or not 0<args.dedup_km<=args.match_km or not 0<=args.keypoint_conf<=1 or (args.confidence is not None and not FLOOR<=args.confidence<=1):
        p.error('Invalid training/evaluation arguments')
    if not all(math.isfinite(v) for v in (args.lr0,args.weight_decay,args.warmup_epochs,args.warmup_bias_lr,args.degrees,args.translate,args.scale)) or not (0<args.lr0<=1 and 0<=args.weight_decay<1 and args.warmup_epochs>=0 and 0<=args.warmup_bias_lr<=1 and 0<=args.degrees<=180 and 0<=args.translate<1 and 0<=args.scale<1 and args.freeze>=0 and args.save_period>0):
        p.error('Invalid optimization or augmentation arguments')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    root=args.data.resolve(); build,plans,samples,fingerprint,counts=load_dataset(root, args.allow_approximate_split)
    native_size=int(build['config']['image_size']); args.imgsz=args.imgsz or native_size
    if args.imgsz!=native_size or native_size%32:
        raise ValueError('Use processed image size from build.json')
    LOG.info('Dataset checks passed: %s', {s:dict(c) for s,c in counts.items()})
    diversity = diversity_report(samples,plans)
    LOG.info('Storm diversity: %s', {s:{k:v for k,v in d.items() if k!='storms'} for s,d in diversity.items()})
    if args.audit_only:
        print(json.dumps(diversity,indent=2)); return 0
    if args.check_only: return 0
    # Validate cached mining and approvals before creating a run or loading GPU weights.
    hard=None; mined=[]; screened=[]
    if args.mine_from and not args.mine_only:
        cache=args.mining_cache or (args.approved_negatives.parent if args.approved_negatives else None)
        if cache is None: p.error('Specify --mining-cache for screening')
        mined,raw=hn.load_cache(cache,samples['train'],fingerprint,args.mine_from,args.mine_confidence,args.imgsz,args.keypoint_conf,args.allow_legacy_mining_cache)
        if args.approved_negatives and not args.screen_only: hard=hn.approved(args.approved_negatives,mined)
        if args.allow_legacy_mining_cache and not (cache/'cache_identity.json').exists(): LOG.warning('Legacy cache: original weight bytes cannot be verified; relying on user confirmation')
        screened=hn.screen(mined,raw,samples['train'],plans['train'],args.mine_confidence,args.screen_km)
        clear={r['image'] for r in screened if r['screen_status']=='clear_of_catalogued_centres'}
        LOG.info('Screening: %s',dict(Counter(r['screen_status'] for r in screened)))
        if args.screen_only:
            out=(args.project/args.name).resolve(); out.mkdir(parents=True,exist_ok=False)
            write_csv(out/'screened_candidates.csv',screened,['image','score','scene','nearest_catalogued_km','screen_status','approved'])
            LOG.info('Review report: %s',out/'screened_candidates.csv'); return 0
        if args.accept_screened: hard=clear
        if not hard: raise ValueError('No usable hard negatives; review candidates or use ordinary training')
        if hard-clear: LOG.warning('Manual approvals include %d candidates flagged by catalogue screening',len(hard-clear))
        if args.accept_screened: LOG.warning('Using provisional catalogue-clear negatives; not independently verified meteorological negatives')
        del raw
    import torch
    import ultralytics
    from ultralytics import YOLO
    if args.device!='cpu' and (args.device.isdigit() or args.device.startswith('cuda')) and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Install CUDA-enabled PyTorch or use --device cpu.')
    # Do not modify the published dataset YAML (may reside on read-only Kaggle storage).
    run=(args.project/args.name).resolve()
    if args.resume:
        checkpoint=args.resume.resolve(); run=checkpoint.parent.parent.parent
        meta=json.loads((run/'run_config.json').read_text())
        if meta['dataset_fingerprint']!=fingerprint or meta['dataset_root']!=str(root) or meta['oks_sigma']!=args.oks_sigma:
            raise ValueError('Resume requires original dataset and OKS configuration')
        if (run/'evaluation').exists(): raise ValueError('Run already has evaluation; use --checkpoint with a new --name')
        if not (run/'dataset.yaml').is_file(): raise FileNotFoundError(run/'dataset.yaml')
        if meta['versions']['ultralytics']!=ultralytics.__version__: raise ValueError('Resume requires original Ultralytics version')
        saved=meta['arguments']
        if saved.get('mine_from') or saved.get('baseline'): raise ValueError('Resume of mined or baseline-guarded runs is not supported; start a new experiment')
        for key in ('imgsz','seed','batch','epochs','patience','no_amp','lr0','weight_decay','warmup_epochs','warmup_bias_lr','degrees','translate','scale','freeze','save_period','selection','beta','match_km','dedup_km','keypoint_conf','confidence','min_precision','same_tile_dedup_km'):
            if key not in saved or getattr(args,key)!=saved[key]: raise ValueError(f'Resume requires original --{key.replace("_","-")}')
    else:
        run.mkdir(parents=True,exist_ok=False)
    data=run/'dataset.yaml'
    if not args.resume:
        data.write_text(yaml.safe_dump(dict(path=str(root),train='images/train',val='images/val',test='images/test',names={0:'cyclone'},kpt_shape=[1,3],flip_idx=[0],kpt_oks_sigmas=[args.oks_sigma]),sort_keys=False))
        dump(run/'run_config.json',dict(dataset_fingerprint=fingerprint,dataset_root=str(root),oks_sigma=args.oks_sigma,arguments=vars(args)|{k:str(v) for k,v in vars(args).items() if isinstance(v,Path)},versions=dict(python=platform.python_version(),torch=str(torch.__version__),ultralytics=ultralytics.__version__,cuda=torch.version.cuda),counts={s:dict(c) for s,c in counts.items()}))
    dump(run/'dataset_diversity.json',diversity)
    diag.diversity_plots(run, diversity)
    if args.mine_only:
        miner=YOLO(str(args.mine_from.resolve()))
        if miner.task!='pose' or tuple(miner.model.yaml.get('kpt_shape',()))!=(1,3): raise ValueError('Mining requires trained cyclone pose checkpoint')
        negatives={k:r for k,r in samples['train'].items() if not r['objects']}
        mining=run/'mining'; raw,scores=infer(miner,root,negatives,Projection(build['grid']),args,mining)
        mined=hn.candidates(negatives,scores,args.mine_confidence)
        write_csv(mining/'candidates.csv',[dict(image=k,score=v,approved='') for k,v in mined],['image','score','approved'])
        hn.gallery(root,[p for ps in raw.values() for p in ps if p['score']>=args.mine_confidence],mining/'gallery','candidate')
        dump(mining/'summary.json',dict(candidates=len(mined),negative_tiles=len(negatives),checkpoint=str(args.mine_from.resolve())))
        dump(mining/'cache_identity.json',dict(checkpoint_sha256=hn.sha256(args.mine_from),predictions_sha256=hn.sha256(mining/'tile_predictions.jsonl')))
        del miner,raw
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        return 0
    if hard is not None:
        mining=run/'mining'; mining.mkdir()
        write_csv(mining/'screened_candidates.csv',screened,['image','score','scene','nearest_catalogued_km','screen_status','approved'])
        excluded=set(dict(mined))-hard
        selected=hn.sample_list(samples['train'],hard,args.seed,args.storm_balance,args.hard_fraction,args.max_hard_repeat,excluded)
        hard_draws=sum(k in hard for k in selected)
        LOG.info('Hard negative draws: %d/%d; approved=%d; max repeat=%d; unapproved candidates excluded=%d',hard_draws,len(selected),len(hard),args.max_hard_repeat,len(excluded))
        # Image lists with repeated paths can be collapsed by framework label caches.
        # Create unique hard-linked image/label aliases in the run, not in the dataset.
        view=run/'sampling_view'; (view/'images').mkdir(parents=True); (view/'labels').mkdir()
        import os
        for i,k in enumerate(selected):
            src=root/k; lab=root/samples['train'][k]['label']; stem=f'{i:06d}_{src.stem}'
            for original,dest in [(src,view/'images'/(stem+src.suffix)),(lab,view/'labels'/(stem+'.txt'))]:
                try: os.link(original,dest)
                except OSError: shutil.copy2(original,dest)
        config=yaml.safe_load(data.read_text()); config['train']=str(view/'images'); data.write_text(yaml.safe_dump(config,sort_keys=False))
        positive_draws=sum(bool(samples['train'][k]['objects']) for k in selected)
        selected_storms=Counter(o['sid'] for k in selected for o in samples['train'][k]['objects'])
        dump(mining/'sampling.json',dict(positive_draws=positive_draws,original_positive_tiles=sum(bool(r['objects']) for r in samples['train'].values()),unique_positive_tiles=len({k for k in selected if samples['train'][k]['objects']}),storm_annotation_draws=dict(selected_storms),selected=selected,hard=sorted(hard),unique_images=len(set(selected)),hard_draws=hard_draws,effective_hard_fraction=hard_draws/len(selected),max_hard_repeat=args.max_hard_repeat,excluded=sorted(excluded),cache=str(cache.resolve()),provisional=args.accept_screened))
    if args.baseline is not None:
        original_hash=hn.sha256(args.baseline)
        base_model=YOLO(str(args.baseline))
        if base_model.task!='pose' or tuple(base_model.model.yaml.get('kpt_shape',()))!=(1,3): raise ValueError('Baseline must be a trained one-centre pose model')
        del base_model
        dump(run/'baseline_identity.json',dict(checkpoint=str(args.baseline),sha256=original_hash))
    if args.checkpoint:
        best=args.checkpoint.resolve()
    else:
        if args.resume:
            model=YOLO(str(checkpoint)); model.train(resume=True,device=args.device,workers=args.workers)
        else:
            model=YOLO(args.model)
            if model.task!='pose': raise ValueError('Use a pose checkpoint')
            model.train(**training_kwargs(args,data,run))
        best=Path(model.trainer.best).resolve()
    if not best.is_file(): raise FileNotFoundError(best)
    diag.training_plots(run, run/'train'/'results.csv')
    projection=Projection(build['grid'])
    selected_threshold=args.confidence
    if not args.checkpoint:
        del model
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if args.baseline is not None and hn.sha256(args.baseline)!=original_hash: raise RuntimeError('Baseline changed during training')
        best,selected_threshold=select_checkpoint(YOLO,best,root,samples['val'],plans['val'],projection,args,run)
    model=YOLO(str(best))
    if model.task!='pose' or tuple(model.model.yaml.get('kpt_shape',()))!=(1,3):
        raise ValueError('Checkpoint must be trained for one centre keypoint')
    # Resume's actual run is .../train; config is one level above it.
    reports=run/'evaluation'; reports.mkdir(exist_ok=False)
    combined={}
    dump(reports/'model.json',dict(checkpoint=str(best),checkpoint_bytes=best.stat().st_size,parameters=sum(p.numel() for p in model.model.parameters())))
    threshold=selected_threshold
    for split in ('val','test') if args.final_test else ('val',):
        out=reports/split; out.mkdir()
        native=native_metrics(model,data,split,args,out)
        raw,scores=infer(model,root,samples[split],projection,args,out)
        geo=geographic_report(raw,scores,samples[split],plans[split],args,out,threshold)
        low,_=filtered_box_counts(raw,samples[split],FLOOR,args.imgsz)
        operating,box_rows=filtered_box_counts(raw,samples[split],geo['threshold'],args.imgsz)
        dump(out/'filtered_box_metrics.json',dict(note='Independent one-to-one tile-box IoU matching, not native AP or geographic detection.',ap_collection_floor=low,operating_point=operating))
        write_csv(out/'filtered_box_counts.csv',box_rows,['image','scene','targets','predictions','tp','fp','fn'])
        plt=diag.pyplot(); fig,ax=plt.subplots(); x=np.arange(3)
        ax.bar(x-.18,[low[k] for k in ('tp','fp','fn')],.36,label='AP collection floor')
        ax.bar(x+.18,[operating[k] for k in ('tp','fp','fn')],.36,label=f'Operating point {geo["threshold"]:.3f}')
        ax.set(xticks=x,xticklabels=['Matched boxes','Unmatched predictions','Missed boxes'],ylabel='Tile-object count'); ax.legend(); diag.save(plt,fig,out/'filtered_box_counts.png')
        fp_rows=[r for r in csv.DictReader((out/'centre_matches.csv').open()) if r['status']=='FP']
        for r in fp_rows: r['score']=float(r['score'])
        hn.gallery(root,fp_rows,out/'false_positive_gallery','fp')
        plt=diag.pyplot(); fig,ax=plt.subplots(); ax.hist([r['score'] for r in fp_rows],bins=np.linspace(0,1,21)); ax.set(xlabel='False positive confidence',ylabel='Count'); diag.save(plt,fig,out/'false_positive_confidence.png')
        if split=='val' and not geo['precision_constraint_met']: LOG.warning('No selected operating point meets minimum geographic precision; do not deploy without review')
        if split=='val':
            threshold=geo['threshold']; dump(reports/'operating_point.json',dict(confidence=threshold,selected_on='validation' if args.confidence is None else 'fixed by user',beta=args.beta,min_precision=args.min_precision,precision_constraint_met=geo['precision_constraint_met'],match_km=args.match_km,dedup_km=args.dedup_km,keypoint_conf=args.keypoint_conf,same_tile_dedup_km=args.same_tile_dedup_km,checkpoint=str(best),checkpoint_sha256=hn.sha256(best)))
        combined[split]={'ultralytics':native,'geographic':geo,'filtered_box_metrics':operating}
    dump(reports/'metrics.json',combined)
    LOG.info('Finished. Metrics: %s',reports/'metrics.json')
    return 0


if __name__=='__main__':
    raise SystemExit(main())
