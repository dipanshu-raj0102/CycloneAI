#!/usr/bin/env python3
"""Download all missing GOES-16 C13/C08 pairs for IBTrACS intensity fixes, 2018–2025.
Default is a dry-run plan; --download writes only missing files. Linux/Python 3.10+.
Default combined class ceiling is 900 candidate observations (existing + new).
Uses public unsigned NOAA S3, no AWS account or detection data required.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import random
import re
import shutil
import time
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import importlib.util
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import fcntl
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from boto3.s3.transfer import TransferConfig
from pyproj import Transformer

UTC = timezone.utc
BUCKET = 'noaa-goes16'
CUTOFF = datetime(2025,4,7,15,10,tzinfo=UTC)
QC_LOCK = Lock()
PRODUCT = 'ABI-L2-CMIPF'
CLASSES = ('TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5')
BOUNDS = (34,64,83,96,113,137)
ABI = re.compile(r'^OR_ABI-L2-CMIPF-M([36])C(08|13)_G16_s(\d{14})_e(\d{14})_c(\d{14})\.nc$')
SLOT = re.compile(r'^\d{8}_\d{4}$')
MAGICS = (b'\x89HDF\r\n\x1a\n', b'CDF\x01', b'CDF\x02', b'CDF\x05')

@dataclass(frozen=True)
class Fix:
    sid: str
    time: datetime
    lat: float
    lon: float
    wind: float
    category: str

@dataclass(frozen=True)
class File:
    key: str
    size: int
    mode: str
    band: str
    start: datetime
    end: datetime
    creation: datetime
    @property
    def name(self): return self.key.rsplit('/',1)[-1]

@dataclass(frozen=True)
class Pair:
    ir: File
    wv: File
    @property
    def time(self): return self.ir.start + (self.ir.end-self.ir.start)/2
    @property
    def slot(self): return self.ir.start.strftime('%Y%m%d_%H%M')


def abi_time(s):
    return datetime.strptime(s,'%Y%j%H%M%S%f').replace(tzinfo=UTC)


def parse_file(key, size):
    m=ABI.fullmatch(key.rsplit('/',1)[-1])
    if not m or size<=0: return None
    mode,band,s,e,c=m.groups()
    try: start,end,creation=map(abi_time,(s,e,c))
    except ValueError: return None
    if not 0<(end-start).total_seconds()<=900: return None
    return File(key,int(size),mode,band,start,end,creation)


def pair_files(files):
    # Use actual acquisition timestamps, not nominal directory names.
    # Group by acquisition minute/mode then keep the newest compatible pair.
    groups=defaultdict(list)
    for f in files:
        if f: groups[(f.mode,f.start.replace(second=0,microsecond=0))].append(f)
    out=[]
    for group in groups.values():
        pairs=[Pair(a,b) for a in group if a.band=='13' for b in group if b.band=='08'
               and abs((a.start-b.start).total_seconds())<=3 and abs((a.end-b.end).total_seconds())<=30]
        if pairs: out.append(max(pairs,key=lambda p:(p.ir.creation,p.wv.creation,p.ir.key,p.wv.key)))
    return sorted(out,key=lambda p:p.time)


def looks_netcdf(path):
    try:
        if not path.is_file() or path.stat().st_size<8: return False
        with path.open('rb') as f: head=f.read(8)
        return any(head.startswith(m) for m in MAGICS)
    except OSError: return False


def local_inventory(root):
    pairs=[]; issues=Counter(); occupied=set(); slot_paths=defaultdict(set)
    if not root.exists(): return pairs,issues,occupied,slot_paths
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not SLOT.fullmatch(folder.name): continue
        files=[]; occupied.add(folder.name)
        for sub,band in (('IR','13'),('WV','08')):
            for p in (folder/sub).glob('*.nc'):
                f=parse_file(str(p),p.stat().st_size)
                if f is None or f.band!=band or not looks_netcdf(p):
                    issues['invalid_local_files']+=1; continue
                files.append(f)
                slot_paths[f.start.strftime('%Y%m%d_%H%M')].add(folder.name)
        pp=pair_files(files)
        if len(pp)==1: pairs.append(pp[0])
        else: issues['incomplete_or_ambiguous_folders']+=1
    issues['complete_local_pairs']=len(pairs)
    return pairs,issues,occupied,slot_paths


def zenith(lat,lon):
    # WGS84 geocentric vectors, approximate nominal GOES-East orbital position.
    # Surface normal is geodetic latitude/longitude, not radial vector.
    x,y,z=ECEF.transform(lon,lat,0.)
    dx,dy,dz=SAT[0]-x,SAT[1]-y,SAT[2]-z
    phi,lam=math.radians(lat),math.radians(lon)
    normal=(math.cos(phi)*math.cos(lam),math.cos(phi)*math.sin(lam),math.sin(phi))
    cosine=(dx*normal[0]+dy*normal[1]+dz*normal[2])/math.sqrt(dx*dx+dy*dy+dz*dz)
    return math.degrees(math.acos(max(-1.,min(1.,cosine))))

ECEF=Transformer.from_crs('EPSG:4979','EPSG:4978',always_xy=True)
SAT=ECEF.transform(-75.,0.,35786023.)


def load_fixes(path,start,end,max_zenith):
    required={'SID','ISO_TIME','NATURE','USA_WIND','LAT','LON','TRACK_TYPE'}
    raw=defaultdict(set); stats=Counter()
    with path.open(encoding='utf-8-sig',newline='') as stream:
        reader=csv.DictReader(stream)
        if reader.fieldnames is None: raise ValueError('Empty IBTrACS CSV')
        reader.fieldnames=[x.strip() for x in reader.fieldnames]
        if not required.issubset(reader.fieldnames): raise ValueError(f'Missing IBTrACS columns: {required-set(reader.fieldnames)}')
        for row in reader:
            sid=(row['SID'] or '').strip()
            if not sid or (row['TRACK_TYPE'] or '').strip().lower()!='main': continue
            try:
                t=datetime.fromisoformat((row['ISO_TIME'] or '').strip().replace('Z','+00:00'))
                t=t.replace(tzinfo=UTC) if t.tzinfo is None else t.astimezone(UTC)
            except ValueError: continue  # includes the units row
            if not start<=t<end: continue
            try: values=((row['NATURE'] or '').strip(),float(row['LAT']),float(row['LON']),float(row['USA_WIND']))
            except (ValueError,TypeError): values=('INVALID',0.,0.,0.)
            raw[(sid,t)].add(values)
    fixes=[]
    for (sid,t),values in sorted(raw.items()):
        if len(values)!=1: stats['ambiguous_fixes']+=1; continue
        nature,lat,lon,wind=next(iter(values))
        if nature!='TS': stats['not_tropical_or_invalid']+=1; continue
        if not all(map(math.isfinite,(lat,lon,wind))) or not -90<lat<90 or not -180<=lon<=360 or not 0<wind<=250:
            stats['invalid_coordinates_or_wind']+=1; continue
        lon=(lon+180)%360-180
        if zenith(lat,lon)>max_zenith: stats['view_zenith_excluded']+=1; continue
        fixes.append(Fix(sid,t,lat,lon,wind,CLASSES[bisect_right(BOUNDS,wind)]))
    return fixes,stats


def covered(t,times,tolerance):
    i=bisect_left(times,t)
    return any(abs((x-t).total_seconds())<=tolerance for x in times[max(0,i-1):i+1])


def ordered_fixes(fixes,seed):
    # Interleave distinct storms; shuffle deterministically within each storm.
    rng=random.Random(seed); groups=defaultdict(list)
    for f in fixes: groups[f.sid].append(f)
    sids=sorted(groups); rng.shuffle(sids)
    for sid in sids: rng.shuffle(groups[sid])
    queues=deque(deque(groups[sid]) for sid in sids)
    while queues:
        q=queues.popleft(); yield q.popleft()
        if q: queues.append(q)


class Archive:
    def __init__(self, pool=96):
        self.s3=boto3.client('s3',region_name='us-east-1',config=Config(signature_version=UNSIGNED,
            retries={'mode':'standard','max_attempts':6},connect_timeout=15,read_timeout=90,max_pool_connections=pool))
    @lru_cache(maxsize=4096)
    def hour(self,hour):
        prefix=f'{PRODUCT}/{hour:%Y}/{hour:%j}/{hour:%H}/'
        files=[]
        for page in self.s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET,Prefix=prefix):
            for o in page.get('Contents',[]):
                f=parse_file(o['Key'],o['Size'])
                if f: files.append(f)
        return pair_files(files)
    def candidates(self,t,tolerance):
        # Search the fix's hour and adjoining hours that can contain a matching scan.
        # A full-disk scan can start up to 7.5 min before its midpoint.
        lo=(t-timedelta(seconds=tolerance+450)).replace(minute=0,second=0,microsecond=0)
        hi=(t+timedelta(seconds=tolerance)).replace(minute=0,second=0,microsecond=0)
        pairs=[]
        while lo<=hi:
            pairs.extend(self.hour(lo)); lo+=timedelta(hours=1)
        # Prefer scans starting at/after the fix for ties; no interpolation of wind.
        return sorted((p for p in pairs if abs((p.time-t).total_seconds())<=tolerance),
                      key=lambda p:(abs((p.time-t).total_seconds()),p.ir.start<t,p.ir.key))


def destination(root,pair,slot_paths):
    # Respect pre-existing aliases (e.g. a nominal 18:00 folder with actual 18:30 scan).
    aliases=slot_paths.get(pair.slot,set())
    if len(aliases)>1: return None
    folder=root/(next(iter(aliases)) if aliases else pair.slot)
    files=[]
    for sub,f in (('IR',pair.ir),('WV',pair.wv)):
        target=folder/sub/f.name
        other=list((folder/sub).glob('*.nc'))
        if any(p!=target for p in other): return None  # don't mix acquisitions/revisions
        if target.exists() and (target.stat().st_size!=f.size or not looks_netcdf(target)): return None
        files.append((f,target))
    return files


def write_csv(path,rows,fields):
    tmp=path.with_suffix(path.suffix+'.tmp')
    with tmp.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    tmp.replace(path)


def write_json(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)


def needed_hours(t,tolerance):
    lo=(t-timedelta(seconds=tolerance+450)).replace(minute=0,second=0,microsecond=0)
    hi=(t+timedelta(seconds=tolerance)).replace(minute=0,second=0,microsecond=0)
    while lo<=hi:
        if lo<CUTOFF: yield lo
        lo+=timedelta(hours=1)


def plan(args,archive,fixes,pairs,slot_paths):
    times=sorted({p.time for p in pairs}); tolerance=args.max_offset_min*60
    existing=Counter(); missing=defaultdict(list)
    for f in fixes:
        if covered(f.time,times,tolerance): existing[f.category]+=1
        else: missing[f.category].append(f)
    fixes_by_time=sorted(fixes,key=lambda f:f.time); fix_times=[f.time for f in fixes_by_time]
    def nearest_labels(t):
        lo=bisect_left(fix_times,t-timedelta(seconds=tolerance))
        hi=bisect_right(fix_times,t+timedelta(seconds=tolerance))
        near={}
        for x in fixes_by_time[lo:hi]:
            if x.sid not in near or abs(x.time-t)<abs(near[x.sid].time-t): near[x.sid]=x
        return list(near.values())
    # A scene can contain more than one cyclone, and a fix can match multiple
    # existing acquisitions. Count scene-level candidate labels as well as unique
    # matched fixes; the larger baseline is conservative for the download cap.
    local_observations=Counter(x.category for p in pairs for x in nearest_labels(p.time))
    baseline=Counter({c:max(existing[c],local_observations[c]) for c in CLASSES})
    remaining={c:max(0,args.max_per_class-baseline[c]) for c in CLASSES}
    active_missing={c:missing[c] if remaining[c]>0 else [] for c in CLASSES}
    hours=sorted({h for group in active_missing.values() for f in group for h in needed_hours(f.time,tolerance)})
    print(f'Listing {len(hours):,} distinct archive hours using {args.search_workers} workers',flush=True)
    lookup={}; errors={}
    with ThreadPoolExecutor(max_workers=args.search_workers) as pool:
        pending={pool.submit(archive.hour,h):h for h in hours}
        for i,future in enumerate(as_completed(pending),1):
            h=pending[future]
            try: lookup[h]=future.result()
            except Exception as exc: errors[h.isoformat()]=str(exc)
            if i%250==0 or i==len(hours): print(f'Archive search {i:,}/{len(hours):,} hours; errors={len(errors)}',flush=True)
    if errors:
        write_json(args.audit/'archive_errors.json',errors)
        raise RuntimeError('Archive listing errors; retry rather than silently producing an incomplete plan. See archive_errors.json')
    selected=[]; counts=Counter(); reasons=Counter(); used_slots=set(); used_folders=set(); total=0
    limit=int(args.max_gb*1e9) if args.max_gb is not None else math.inf
    # Prefer rare classes, interleave storms, and reserve quota for every label
    # in a scene, not merely the primary class. Existing files are never removed.
    queues={c:deque(ordered_fixes(active_missing[c],args.seed+CLASSES.index(c))) for c in reversed(CLASSES)}
    planned_by_fix=set(); added=Counter()
    while any(queues.values()) and (args.max_scenes is None or len(selected)<args.max_scenes):
        for c,q in queues.items():
            if not q: continue
            if added[c]>=remaining[c]:
                reasons['class_cap_reached']+=len(q); q.clear(); continue
            if args.max_scenes is not None and len(selected)>=args.max_scenes: break
            f=q.popleft()
            if (f.sid,f.time) in planned_by_fix: continue
            candidates=sorted((p for h in needed_hours(f.time,tolerance) for p in lookup[h]
                              if p.time<CUTOFF and abs((p.time-f.time).total_seconds())<=tolerance),
                              key=lambda p:(abs((p.time-f.time).total_seconds()),p.ir.start<f.time,p.ir.key))
            if not candidates: reasons['no_archive_pair']+=1; continue
            for pair in candidates:
                if pair.slot in used_slots or covered(pair.time,times,1): continue
                dest=destination(args.goes,pair,slot_paths)
                if dest is None: continue
                folder=dest[0][1].parent.parent.name
                if folder in used_folders: continue
                n=sum(o.size for o,path in dest if not path.exists())
                if n<=0 or total+n>limit: continue
                labels=nearest_labels(pair.time)
                if f not in labels: continue
                increments=Counter(x.category for x in labels)
                if any(added[k]+v>remaining[k] for k,v in increments.items()):
                    reasons['scene_class_cap_conflict']+=1; continue
                selected.append((f,pair,dest,n,labels)); total+=n
                added.update(increments)
                counts[c]+=1; used_slots.add(pair.slot); used_folders.add(folder)
                planned_by_fix.update((x.sid,x.time) for x in labels)
                break
            else: reasons['duplicate_conflict_or_budget']+=1
    stats={'existing_matched_fixes':dict(existing),'missing_candidate_fixes':{c:len(missing[c]) for c in CLASSES},
           'max_total_per_class':args.max_per_class,
           'existing_candidate_observations_by_class':{c:local_observations[c] for c in CLASSES},
           'class_cap_baseline':dict(baseline),
           'remaining_class_capacity_before_download':remaining,
           'existing_above_cap':{c:baseline[c]-args.max_per_class for c in CLASSES if baseline[c]>args.max_per_class},
           'planned_new_observations_by_class':{c:added[c] for c in CLASSES},
           'projected_total_by_class':{c:baseline[c]+added[c] for c in CLASSES},
           'planned_primary_class_counts':dict(counts),'planned_scenes':len(selected),'planned_download_bytes':total,
           'planned_download_gb':total/1e9,'archive_hours':len(hours),'planning_skips':dict(reasons)}
    return selected,stats


class Progress:
    def __init__(self,total):
        self.total=total; self.bytes=0; self.start=time.monotonic(); self.last=self.start; self.lock=Lock()
    def add(self,n):
        with self.lock:
            self.bytes+=n; now=time.monotonic()
            if now-self.last>=5:
                rate=self.bytes/max(now-self.start,1e-6)
                print(f'Transfer {self.bytes/1e9:.2f}/{self.total/1e9:.2f} GB | average {rate*8/1e6:.1f} Mbps ({rate/1e6:.1f} MB/s)',flush=True)
                self.last=now


def load_qc_module(path):
    spec=importlib.util.spec_from_file_location('classification_raw_qc',path)
    if spec is None or spec.loader is None: raise RuntimeError(f'Cannot import {path}')
    module=importlib.util.module_from_spec(spec); sys.modules[spec.name]=module; spec.loader.exec_module(module)
    return module


def quality_check(qc,paths,labels,args):
    # NetCDF/HDF5 backends are not universally thread-safe. Serialize only QC;
    # transfers remain concurrent. Same functions/thresholds as raw preprocessing.
    accepted=[]
    with QC_LOCK:
        with qc.xr.open_dataset(paths[0],decode_times=False) as a,qc.xr.open_dataset(paths[1],decode_times=False) as b:
            crs,x,y=qc.grid(a,13); crs2,x2,y2=qc.grid(b,8)
            if crs!=crs2 or not qc.np.array_equal(x,x2) or not qc.np.array_equal(y,y2): raise ValueError('Channel grid mismatch')
            for f in labels:
                rr,cc=qc.coordinates(crs,x,y,f.lat,f.lon,args.image_size,args.width_km)
                _,m13=qc.sample_band(a,rr,cc); _,m08=qc.sample_band(b,rr,cc)
                valid=m13&m08; half=args.image_size//2
                if valid.mean()>=args.min_valid and valid[half-1:half+1,half-1:half+1].all(): accepted.append(f)
    return accepted


def download_pair(archive,job,args,progress,qc):
    fix,pair,dest,_,labels=job; folder=dest[0][1].parent.parent
    folder.mkdir(parents=True,exist_ok=True)
    staged=[]; paths=[]
    # Stable staging names retain completed individual files across interrupted runs.
    stage_dir=folder/'.download_staging'; stage_dir.mkdir(exist_ok=True)
    for f,target in dest:
        if target.exists():
            if target.stat().st_size!=f.size or not looks_netcdf(target): raise RuntimeError(f'Existing file conflicts: {target}')
            paths.append(target); continue
        stage=stage_dir/f.name
        if not (stage.is_file() and stage.stat().st_size==f.size and looks_netcdf(stage)):
            stage.unlink(missing_ok=True)
            for attempt in range(3):
                try:
                    archive.s3.download_file(BUCKET,f.key,str(stage),Callback=progress.add,
                        Config=TransferConfig(multipart_threshold=8*1024**2,multipart_chunksize=8*1024**2,
                                              max_concurrency=args.parts,num_download_attempts=5,use_threads=True))
                    if stage.stat().st_size!=f.size or not looks_netcdf(stage): raise IOError(f'Invalid size or NetCDF signature: {f.key}')
                    break
                except Exception:
                    stage.unlink(missing_ok=True)
                    if attempt==2: raise
                    time.sleep(2**attempt)
        staged.append((stage,target)); paths.append(stage)
    accepted=quality_check(qc,paths,labels,args) if qc else labels
    if not accepted:
        for stage,_ in staged: stage.unlink(missing_ok=True)
        if not any(stage_dir.iterdir()): stage_dir.rmdir()
        return {'scene_id':folder.name,'status':'quality_rejected','error':'No candidate passes crop QC','accepted_classes':'','accepted_storms':''}
    for stage,target in staged:
        target.parent.mkdir(parents=True,exist_ok=True)
        os.link(stage,target); stage.unlink()  # never overwrite an existing file
    if not any(stage_dir.iterdir()): stage_dir.rmdir()
    return {'scene_id':folder.name,'status':'downloaded','error':'','accepted_classes':';'.join(f.category for f in accepted),'accepted_storms':';'.join(f.sid for f in accepted)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data'))
    p.add_argument('--goes',type=Path); p.add_argument('--ibtracs',type=Path); p.add_argument('--audit-dir',type=Path)
    p.add_argument('--start-year',type=int,default=2018); p.add_argument('--end-year',type=int,default=2025)
    p.add_argument('--max-per-class',type=int,default=900,help='Maximum total candidate observations per class, including existing archive; never delete existing data')
    p.add_argument('--max-scenes',type=int,default=None); p.add_argument('--max-gb',type=float,default=None,help='Optional maximum new transfer in decimal GB')
    p.add_argument('--max-offset-min',type=float,default=10.); p.add_argument('--max-zenith',type=float,default=70.)
    p.add_argument('--workers',type=int,default=16); p.add_argument('--search-workers',type=int,default=24); p.add_argument('--parts',type=int,default=2)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--image-size',type=int,default=384); p.add_argument('--width-km',type=float,default=800.); p.add_argument('--min-valid',type=float,default=.90)
    p.add_argument('--qc-script',type=Path,default=Path(__file__).with_name('preprocess_classification_raw.py'))
    p.add_argument('--skip-quality-check',action='store_true',help='Only verify file sizes and signatures; defer crop QC to preprocessing')
    modes=p.add_mutually_exclusive_group(); modes.add_argument('--plan-only',action='store_true'); modes.add_argument('--download',action='store_true')
    args=p.parse_args()
    if args.max_per_class<1: p.error('--max-per-class must be positive')
    if not 2018<=args.start_year<=args.end_year<=2025: p.error('Use years 2018–2025')
    if not 0<args.max_offset_min<=10 or not 0<args.max_zenith<90 or not 0<args.min_valid<=1 or not math.isfinite(args.width_km) or args.width_km<=0 or args.image_size<32 or args.image_size%2: p.error('Invalid quality settings')
    if not 1<=args.workers<=64 or not 1<=args.search_workers<=64 or not 1<=args.parts<=8: p.error('Invalid worker/part counts')
    if args.max_scenes is not None and args.max_scenes<1 or args.max_gb is not None and (not math.isfinite(args.max_gb) or args.max_gb<=0): p.error('Invalid download limits')
    root=args.data_root.resolve(); args.goes=(args.goes or root/'goes16').resolve(); args.ibtracs=(args.ibtracs or root/'ibtracs.csv').resolve(); args.audit=(args.audit_dir or root/'download_audit').resolve()
    if not args.ibtracs.is_file(): p.error(f'IBTrACS CSV missing: {args.ibtracs}')
    if args.audit==args.goes or args.goes in args.audit.parents: p.error('Audit directory must be outside raw archive')
    qc=None
    if args.download and not args.skip_quality_check:
        if not args.qc_script.is_file(): p.error(f'QC script missing: {args.qc_script}; place preprocess_classification_raw.py beside this script')
        qc=load_qc_module(args.qc_script)
    args.audit.mkdir(parents=True,exist_ok=True); args.goes.mkdir(parents=True,exist_ok=True)
    with (args.goes/'.class_download.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        local,inventory,_,slot_paths=local_inventory(args.goes)
        end=min(datetime(args.end_year+1,1,1,tzinfo=UTC),CUTOFF)
        fixes,fix_stats=load_fixes(args.ibtracs,datetime(args.start_year,1,1,tzinfo=UTC),end,args.max_zenith)
        if not fixes: raise RuntimeError('No eligible IBTrACS fixes; check CSV coverage')
        print(f'{len(local):,} existing pairs; {len(fixes):,} eligible fixes; latest eligible fix {max(f.time for f in fixes).isoformat()}',flush=True)
        archive=Archive(max(96,args.search_workers,args.workers*args.parts+8))
        jobs,summary=plan(args,archive,fixes,local,slot_paths)
        summary.update({'source_bucket':BUCKET,'local_inventory':dict(inventory),'ibtracs_filters':dict(fix_stats),'eligible_fixes':len(fixes),'eligible_fixes_by_year':dict(Counter(str(f.time.year) for f in fixes)),'eligible_fixes_by_class':dict(Counter(f.category for f in fixes)), 'start_year':args.start_year,'end_year':args.end_year,'effective_end_utc':end.isoformat(),'max_view_zenith':args.max_zenith,'max_offset_min':args.max_offset_min,'quality_check_enabled':not args.skip_quality_check,'workers':args.workers,'search_workers':args.search_workers})
        fields=['scene_id','sid','label_time','scan_midpoint','offset_seconds','class_name','wind_kt','lat','lon','ir_key','wv_key','ir_path','wv_path','download_bytes','candidate_labels']
        rows=[]
        for f,pair,dest,n,labels in jobs:
            rows.append(dict(scene_id=dest[0][1].parent.parent.name,sid=f.sid,label_time=f.time.isoformat(),scan_midpoint=pair.time.isoformat(),offset_seconds=abs((pair.time-f.time).total_seconds()),class_name=f.category,wind_kt=f.wind,lat=f.lat,lon=f.lon,ir_key=pair.ir.key,wv_key=pair.wv.key,ir_path=str(dest[0][1]),wv_path=str(dest[1][1]),download_bytes=n,candidate_labels=';'.join(f'{x.sid}:{x.category}' for x in labels)))
        write_csv(args.audit/'download_plan.csv',rows,fields); write_json(args.audit/'download_summary.json',summary)
        print(json.dumps(summary,indent=2),flush=True)
        if not args.download or not jobs: return 0
        free=shutil.disk_usage(args.goes).free
        if free<summary['planned_download_bytes']+1024**3: raise RuntimeError(f'Insufficient space: {free/1e9:.2f} GB free; plan needs {summary["planned_download_gb"]:.2f} GB plus 1 GiB. Set --max-gb or free space.')
        progress=Progress(summary['planned_download_bytes']); results=[]
        # Append each result immediately, in completion order; no O(n²) CSV rewriting.
        with (args.audit/'download_results.csv').open('w',newline='') as stream, ThreadPoolExecutor(max_workers=args.workers) as pool:
            writer=csv.DictWriter(stream,fieldnames=['scene_id','status','error','accepted_classes','accepted_storms']); writer.writeheader(); stream.flush()
            pending={pool.submit(download_pair,archive,job,args,progress,qc):job for job in jobs}
            for i,future in enumerate(as_completed(pending),1):
                job=pending[future]
                try: result=future.result()
                except Exception as exc: result={'scene_id':job[2][0][1].parent.parent.name,'status':'failed','error':str(exc)}
                results.append(result); writer.writerow(result); stream.flush()
                if i%10==0 or i==len(jobs) or result['status']!='downloaded': print(f'{i}/{len(jobs)} {result["scene_id"]}: {result["status"]} {result["error"]}',flush=True)
        summary['download_status']=dict(Counter(r['status'] for r in results)); summary['accepted_observations_by_class']=dict(Counter(c for r in results for c in r.get('accepted_classes','').split(';') if c)); summary['average_transfer_mbps']=progress.bytes*8/1e6/max(time.monotonic()-progress.start,1e-6)
        write_json(args.audit/'download_summary.json',summary); print(json.dumps(summary,indent=2),flush=True)
        return int(any(r['status']=='failed' for r in results))

if __name__=='__main__': raise SystemExit(main())
