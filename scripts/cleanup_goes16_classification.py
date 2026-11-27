#!/usr/bin/env python3
"""Audit and optionally quarantine GOES-16 scenes unusable for the current classifier.
Place beside preprocess_classification_raw.py. Never permanently deletes files.
Run without --apply to make a plan; --apply executes that exact saved plan.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import math
import os
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
import preprocess_classification_raw as pipeline

SLOT = re.compile(r'\d{8}_\d{4}')


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def snapshot(folder):
    """Only ordinary IR/WV directories and NetCDF files may be moved."""
    if folder.is_symlink() or not folder.is_dir():
        raise ValueError('not a real scene directory')
    rows = []
    for sub in sorted(folder.iterdir()):
        if sub.name not in ('IR', 'WV') or sub.is_symlink() or not sub.is_dir():
            raise ValueError(f'unexpected entry: {sub.name}')
        for p in sorted(sub.iterdir()):
            if p.is_symlink() or not p.is_file() or p.suffix != '.nc':
                raise ValueError(f'unexpected entry: {p.relative_to(folder)}')
            st = p.stat()
            rows.append([str(p.relative_to(folder)), st.st_size, st.st_mtime_ns, st.st_ino, st.st_dev])
    return rows


def quality(job):
    ir, wv, samples, size, width, minimum = job
    counts = Counter(); good = Counter(); max_valid = 0.
    try:
        with xr.open_dataset(ir, decode_times=False) as a, xr.open_dataset(wv, decode_times=False) as b:
            crs, x, y = pipeline.grid(a, 13)
            crs2, x2, y2 = pipeline.grid(b, 8)
            if crs != crs2 or not np.array_equal(x, x2) or not np.array_equal(y, y2):
                raise ValueError('channel grid mismatch')
            for row in samples:
                rr, cc = pipeline.coordinates(crs, x, y, row['lat'], row['lon'], size, width)
                _, m13 = pipeline.sample_band(a, rr, cc)
                _, m08 = pipeline.sample_band(b, rr, cc)
                valid = m13 & m08
                frac = float(valid.mean()); max_valid = max(max_valid, frac)
                if not valid[size//2-1:size//2+1, size//2-1:size//2+1].all():
                    counts['invalid_center_pixels'] += 1
                elif frac < minimum:
                    counts['insufficient_valid_pixels'] += 1
                else:
                    good[row['class_name']] += 1
        return dict(good), dict(counts), max_valid, ''
    except Exception as exc:
        return dict(good), dict(counts), max_valid, f'{type(exc).__name__}: {exc}'


def safe_separate(a, b):
    if a == b or a in b.parents or b in a.parents:
        raise ValueError(f'Paths must be separate: {a} and {b}')


def make_plan(args, raw, tracks, audit, quarantine):
    scenes, inventory = pipeline.discover(raw)
    # Ensure the actual IBTrACS snapshot covers each scene date. An incomplete
    # label archive must not turn missing ground truth into a deletion decision.
    margin = pd.Timedelta(minutes=args.max_offset_min)
    track_data, _ = pipeline.read_tracks(tracks, min(s[1] for s in scenes).floor('D')-margin,
                                         max(s[1] for s in scenes).ceil('D')+margin)
    covered_days = {r[0].date() for rs in track_data.values() for r in rs}
    if not covered_days:
        raise RuntimeError('No main-track IBTrACS fixes in raw archive date range; aborting.')
    # Use the same raw reference projection as classification preprocessing.
    for _, _, ir, wv in scenes:
        try:
            with xr.open_dataset(ir, decode_times=False) as a, xr.open_dataset(wv, decode_times=False) as b:
                crs, x, y = pipeline.grid(a, 13); c2, x2, y2 = pipeline.grid(b, 8)
                if crs != c2 or not np.array_equal(x,x2) or not np.array_equal(y,y2):
                    continue
            visibility = Transformer.from_crs('EPSG:4326', crs, always_xy=True)
            limit = float(max(np.abs(x).max(), np.abs(y).max()))
            break
        except Exception:
            continue
    else:
        raise RuntimeError('No readable compatible raw reference grid; aborting.')
    jobs, rejects, _, plan_audit = pipeline.plan(SimpleNamespace(goes=raw, ibtracs=tracks,
        max_offset_min=args.max_offset_min, visibility=visibility, grid_limit=limit))
    by_scene = {samples[0]['scene_id']: (ir,wv,samples) for ir,wv,samples in jobs}
    if not by_scene:
        raise RuntimeError('Zero eligible observations; check IBTrACS and matching settings. No cleanup plan produced.')
    rejection = defaultdict(Counter)
    for r in rejects:
        rejection[r['scene_id']][r['reason']] += 1
    issues = defaultdict(list)
    for r in inventory.get('discovery_issues', []):
        if r['reason'] != 'shifted_scan': issues[r['scene_id']].append(r['reason'])
    scene_time = {s[0]:s[1] for s in scenes}
    rows = []
    for folder in sorted(raw.iterdir()):
        if not SLOT.fullmatch(folder.name): continue
        sid = folder.name
        row = {'scene_id':sid, 'decision':'review', 'reason':'', 'bytes':0,
               'candidate_samples':0, 'usable_samples':None, 'classes':{}, 'reasons':{}, 'snapshot':[]}
        try:
            datetime.strptime(sid,'%Y%m%d_%H%M')
            row['snapshot'] = snapshot(folder)
            row['bytes'] = sum(x[1] for x in row['snapshot'])
        except Exception as exc:
            row['reason'] = str(exc); rows.append(row); continue
        if sid not in scene_time:
            row['reason'] = ';'.join(issues[sid]) or 'unpaired_or_unsupported_scene'
        elif sid in by_scene:
            samples = by_scene[sid][2]
            row.update(decision='keep',reason='eligible_before_quality',candidate_samples=len(samples),
                       classes=dict(Counter(r['class_name'] for r in samples)))
        elif scene_time[sid].date() not in covered_days:
            row['reason'] = 'no_IBTrACS_main_track_records_on_date'
        else:
            row.update(decision='quarantine',reason='no_eligible_classification_observation',
                       reasons=dict(rejection[sid]) or {'no_fix_within_tolerance':1},usable_samples=0)
        rows.append(row)
    eligible = [r for r in rows if r['decision']=='keep']
    if not args.metadata_only:
        def consume(results):
            for i,(row,result) in enumerate(zip(eligible,results),1):
                good,bad,frac,error=result
                row.update(usable_samples=sum(good.values()),classes=good,reasons=bad,max_valid_fraction=frac)
                if error: row.update(decision='review',reason='quality_read_error: '+error)
                elif good: row.update(decision='keep',reason='at_least_one_usable_crop')
                else: row.update(decision='quarantine',reason='all_candidate_crops_fail_quality')
                if i%25==0 or i==len(eligible): print(f'QC {i}/{len(eligible)}',flush=True)
        work = ((*by_scene[r['scene_id']],args.image_size,args.width_km,args.min_valid) for r in eligible)
        if args.workers==1: consume(map(quality,work))
        else:
            import multiprocessing as mp
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
                consume(pool.map(quality,work,chunksize=1))
        if not any(r['decision']=='keep' for r in rows):
            raise RuntimeError('No scene passed quality checks; refusing to make cleanup plan.')
    counts = Counter(r['decision'] for r in rows)
    sizes = {k:sum(r['bytes'] for r in rows if r['decision']==k) for k in ('keep','quarantine','review')}
    class_counts = Counter()
    for r in rows:
        if r['decision']=='keep': class_counts.update(r['classes'])
    settings = {'image_size':args.image_size,'width_km':args.width_km,'min_valid':args.min_valid,
                'max_offset_min':args.max_offset_min,'metadata_only':args.metadata_only}
    summary = {'counts':dict(counts),'bytes':sizes,'quarantine_gb':round(sizes['quarantine']/1e9,3),
               'retained_class_observations':dict(class_counts),'counts_are_pre_QC':args.metadata_only,
               'planning_audit':plan_audit,'settings':settings}
    plan = {'raw':str(raw),'ibtracs':str(tracks),'ibtracs_sha256':digest(tracks),
            'pipeline_sha256':digest(Path(pipeline.__file__)),'quarantine':str(quarantine),
            'created_at':datetime.now(timezone.utc).isoformat(),'summary':summary,'scenes':rows}
    audit.mkdir(parents=True,exist_ok=True)
    path=audit/'cleanup_plan.json'
    if path.exists(): raise FileExistsError(f'{path} exists; choose a fresh --audit-dir for a new plan.')
    path.write_text(json.dumps(plan,indent=2,allow_nan=False)+'\n')
    (audit/'cleanup_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    fields=['scene_id','decision','reason','bytes','candidate_samples','usable_samples','classes','reasons','max_valid_fraction']
    with (audit/'cleanup_plan.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader()
        for r in rows: w.writerow({**r,'classes':json.dumps(r['classes']),'reasons':json.dumps(r['reasons'])})
    print(json.dumps(summary,indent=2)); print(f'Plan: {path}\nNo raw files moved. Review CSV before applying.')


def apply_plan(raw, tracks, audit, quarantine):
    plan=json.loads((audit/'cleanup_plan.json').read_text())
    if (plan['raw'],plan['ibtracs'],plan['quarantine'])!=(str(raw),str(tracks),str(quarantine)):
        raise ValueError('Saved plan paths differ from current paths')
    if plan['ibtracs_sha256']!=digest(tracks) or plan['pipeline_sha256']!=digest(Path(pipeline.__file__)):
        raise ValueError('IBTrACS or preprocessing script changed; make a new plan.')
    selected=[r for r in plan['scenes'] if r['decision']=='quarantine']
    if not selected: print('No scenes to quarantine.'); return
    # Preflight all moves. Permit resuming a partially applied plan only when
    # the destination is identical to the recorded source snapshot.
    pending=[]; seen=set()
    for r in selected:
        sid=r['scene_id']
        if not SLOT.fullmatch(sid) or sid in seen: raise ValueError('Unsafe or duplicate scene ID')
        seen.add(sid); src=raw/sid; dst=quarantine/sid
        if dst.exists() or dst.is_symlink():
            if src.exists() or src.is_symlink() or snapshot(dst)!=r['snapshot']:
                raise RuntimeError(f'Conflicting destination: {dst}')
            continue
        if snapshot(src)!=r['snapshot']: raise RuntimeError(f'Scene changed since planning: {src}')
        pending.append((src,dst))
    quarantine.mkdir(parents=True,exist_ok=True)
    if raw.stat().st_dev!=quarantine.stat().st_dev: raise RuntimeError('Quarantine must be on the same filesystem for atomic moves.')
    with (audit/'moves.jsonl').open('a') as f:
        for i,(src,dst) in enumerate(pending,1):
            if dst.exists() or dst.is_symlink(): raise RuntimeError(f'Destination appeared: {dst}')
            src.rename(dst)
            f.write(json.dumps({'source':str(src),'destination':str(dst)})+'\n'); f.flush(); os.fsync(f.fileno())
            if i%100==0 or i==len(pending): print(f'Moved {i}/{len(pending)}',flush=True)
    print(f'Quarantined {len(pending)} new scenes to {quarantine}. No permanent deletion performed.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--goes',type=Path); p.add_argument('--ibtracs',type=Path)
    p.add_argument('--audit-dir',type=Path); p.add_argument('--quarantine',type=Path)
    p.add_argument('--image-size',type=int,default=384); p.add_argument('--width-km',type=float,default=800)
    p.add_argument('--min-valid',type=float,default=.90); p.add_argument('--max-offset-min',type=float,default=10)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--metadata-only',action='store_true')
    modes=p.add_mutually_exclusive_group(); modes.add_argument('--apply',action='store_true'); modes.add_argument('--dry-run',action='store_true')
    a=p.parse_args()
    if a.image_size<32 or a.image_size%2 or not 0<a.min_valid<=1 or not math.isfinite(a.width_km) or a.width_km<=0 or not 0<a.max_offset_min<=30 or a.workers<1: p.error('Invalid QC settings')
    data=a.data_root.resolve(); raw=(a.goes or data/'goes16').resolve(); tracks=(a.ibtracs or data/'ibtracs.csv').resolve()
    audit=(a.audit_dir or data/'cleanup_audit').resolve(); quarantine=(a.quarantine or data/'goes16_quarantine').resolve()
    for i,x in enumerate((raw,tracks,audit,quarantine)):
        for y in (raw,tracks,audit,quarantine)[i+1:]: safe_separate(x,y)
    if not raw.is_dir() or not tracks.is_file(): p.error('Missing raw archive or IBTrACS CSV')
    if a.apply: apply_plan(raw,tracks,audit,quarantine)
    else: make_plan(a,raw,tracks,audit,quarantine)

if __name__=='__main__': main()
