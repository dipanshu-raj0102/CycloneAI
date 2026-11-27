#!/usr/bin/env python3
"""Build GOES-16 intensity crops from raw ABI files and IBTrACS only.
Run --plan-only first. Inputs are read-only; output is published only after QC.
"""
from __future__ import annotations
import argparse
import csv
import json
import math
import os
import re
import hashlib
from collections import Counter, defaultdict
from datetime import datetime, timezone
from bisect import bisect_left, bisect_right
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import CRS, Transformer
from scipy.ndimage import map_coordinates
from PIL import Image

SPLITS = ('train', 'val', 'test')
CLASSES = ('TD', 'TS', 'CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5')
BOUNDS = (34, 64, 83, 96, 113, 137)
NORMALIZATION = ((180., 330.), (180., 280.), (-30., 70.))
FIELDS = ['sample_id','image','kelvin','mask','split','sid','scene_id','scene_time','label_time','offset_seconds','lat','lon','wind_kt','class_id','class_name','valid_fraction','ir','wv']

def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')

def write_csv(path, rows, fields):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def class_id(wind):
    return int(np.searchsorted(BOUNDS, wind, side='right'))

# GOES ABI filenames contain scan start, end and creation timestamps.
ABI = re.compile(r'^OR_ABI-L2-CMIPF-M([36])C(08|13)_G16_s(\d{14})_e(\d{14})_c(\d{14})\.nc$')

def abi_time(value):
    return datetime.strptime(value, '%Y%j%H%M%S%f').replace(tzinfo=timezone.utc)

def discover(root):
    """Read the archive's nominal timestamp directories; pair only actual co-scans.

    Folder ID is the nominal scene key, not necessarily the actual scan time.
    Never pair channels across folders or assume a folder name is an observation time.
    """
    counts=Counter(); issues=[]; selected=[]
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not re.fullmatch(r"\d{8}_\d{4}", folder.name):
            continue
        counts['timestamp_folders']+=1
        try:
            nominal=datetime.strptime(folder.name,'%Y%m%d_%H%M').replace(tzinfo=timezone.utc)
        except ValueError:
            counts['invalid_folder_timestamp']+=1; continue
        bands=defaultdict(list)
        for dirname,expected in (('IR','13'),('WV','08')):
            for path in sorted((folder/dirname).glob('*.nc')):
                counts['netcdf_files']+=1
                m=ABI.fullmatch(path.name)
                if not m or m[2]!=expected:
                    counts['unsupported_or_wrong_band_files']+=1; continue
                mode,band,start,end,creation=m.groups()
                try: a,b=abi_time(start),abi_time(end)
                except ValueError:
                    counts['invalid_filename_timestamp']+=1; continue
                if not 0<(b-a).total_seconds()<=900:
                    counts['invalid_scan_duration']+=1; continue
                bands[band].append((mode,a,b,creation,path))
        pairs=[(a,b) for a in bands['13'] for b in bands['08']
               if a[0]==b[0] and abs((a[1]-b[1]).total_seconds())<=3
               and abs((a[2]-b[2]).total_seconds())<=30]
        if not pairs:
            reason='missing_channel' if not bands['13'] or not bands['08'] else 'channel_time_mismatch'
            counts[reason]+=1; issues.append({'scene_id':folder.name,'reason':reason}); continue
        # Verify a single acquisition per nominal folder.
        # Prefer latest product creation when the acquisition itself is identical.
        acquisitions=defaultdict(list)
        for a,b in pairs:
            acquisitions[(a[0],a[1].strftime('%Y%m%d_%H%M'))].append((a,b))
        if len(acquisitions)>1:
            counts['multiple_acquisition_folders']+=1
            # A nominal ID no longer uniquely identifies a scan; avoid ambiguous scene identity.
            issues.append({'scene_id':folder.name,'reason':'multiple_acquisitions_in_folder'}); continue
        a,b=max(next(iter(acquisitions.values())),key=lambda q:(q[0][3],q[1][3],str(q[0][4]),str(q[1][4])))
        counts['duplicate_pairs_removed']+=len(pairs)-1
        midpoint=a[1]+(a[2]-a[1])/2
        actual_slot=a[1].strftime('%Y%m%d_%H%M')
        if actual_slot!=folder.name:
            counts['shifted_scan_folders']+=1
            issues.append({'scene_id':folder.name,'reason':'shifted_scan','actual_scan_start':a[1].isoformat(),'offset_minutes':(a[1]-nominal).total_seconds()/60})
        selected.append((actual_slot,folder.name,pd.Timestamp(midpoint),str(a[4]),str(b[4])))
    # Do not allow one actual scan to enter different splits via different folder aliases.
    by_acquisition=defaultdict(list)
    for row in selected: by_acquisition[row[0]].append(row)
    scenes=[]
    for actual,rows in sorted(by_acquisition.items()):
        if len(rows)>1:
            counts['duplicate_acquisition_folders_quarantined']+=len(rows)
            issues.extend({'scene_id':r[1],'reason':'duplicate_actual_acquisition','actual_slot':actual} for r in rows)
            continue
        _,sid,t,ir,wv=rows[0]; scenes.append((sid,t,ir,wv))
    counts['paired_scenes']=len(scenes)
    if not scenes: raise ValueError(f'No paired GOES-16 C08/C13 scenes under {root}; expected YYYYMMDD_HHMM/IR and WV')
    return scenes,{**dict(counts),'discovery_issues':issues}

def read_tracks(path, start=None, end=None):
    cols = ['SID','ISO_TIME','NATURE','USA_WIND','LAT','LON','TRACK_TYPE']
    f = pd.read_csv(path,usecols=lambda c:c.strip() in cols,dtype=str,keep_default_na=False,low_memory=False)
    f.columns=f.columns.str.strip()
    if not set(cols)<=set(f): raise ValueError('IBTrACS missing columns: '+str(set(cols)-set(f)))
    for c in f: f[c]=f[c].str.strip()
    f['time']=pd.to_datetime(f.ISO_TIME,utc=True,errors='coerce')
    f=f.dropna(subset=['time']); f=f[f.TRACK_TYPE.str.lower().eq('main') & f.SID.ne('')]
    if start is not None: f=f[f.time>=start]
    if end is not None: f=f[f.time<=end]
    tracks=defaultdict(list); ambiguous=0
    for (sid,t),g in f.groupby(['SID','time'],sort=False):
        if len(g[['NATURE','USA_WIND','LAT','LON']].drop_duplicates())!=1:
            # Keep ambiguous fix as a rejection, not a license to use a more distant label.
            tracks[sid].append((t,'AMBIGUOUS',float('nan'),float('nan'),float('nan'))); ambiguous+=1; continue
        r=g.iloc[0]
        wind,lat,lon=pd.to_numeric(r[['USA_WIND','LAT','LON']],errors='coerce')
        tracks[sid].append((t,r.NATURE,float(wind),float(lat),float(lon)))
    for rows in tracks.values(): rows.sort(key=lambda x:x[0])
    return tracks,ambiguous

def split_samples(rows, seed, ratios):
    """Stratify whole storms, keeping each source scene in only one split.

    Optimize sample and per-class balance, coverage and retention. If storms in
    a shared scene land in different splits, keep the largest compatible subset.
    This is a seeded heuristic, not an exact or guaranteed stratification.
    """
    if not rows: raise ValueError('No samples to split')
    storm_names, si = np.unique([r['sid'] for r in rows], return_inverse=True)
    scene_names, gi = np.unique([r['scene_id'] for r in rows], return_inverse=True)
    ci = np.array([r['class_id'] for r in rows], dtype=int)
    ns, ng = len(storm_names), len(scene_names)
    if ns < 3: raise ValueError('At least three independent storms required')
    ratios = np.asarray(ratios, dtype=float)
    totals = np.bincount(ci, minlength=7)
    storm_class = np.unique(si*7+ci)
    available = np.bincount(storm_class%7, minlength=7)
    rng = np.random.default_rng(seed)
    def evaluate(assign, details=False):
        sp = assign[si]
        votes = np.bincount(gi*3+sp, minlength=ng*3).reshape(ng,3)
        # Deterministic tie break: train, then validation, then test.
        scene_assignment = votes.argmax(axis=1)
        keep = sp == scene_assignment[gi]
        tab = np.bincount(sp[keep]*7+ci[keep],minlength=21).reshape(3,7)
        n = tab.sum(axis=1)
        unique = np.unique((si[keep]*7+ci[keep])*3+sp[keep])
        st = np.bincount((unique%3)*7+(unique//3)%7,minlength=21).reshape(3,7)
        if details: return keep,tab,scene_assignment
        present = totals>0
        target = ratios[:,None]*totals[None,:]
        # Relative class balance and overall sample balance; penalize omissions.
        balance = np.mean(((tab[:,present]-target[:,present])/np.maximum(totals[present],1))**2)
        size = np.sum((n/len(rows)-ratios)**2)
        miss_train = np.count_nonzero((tab[0]==0)&present)
        # Three-way coverage can only be possible with >=3 storms in that class.
        feasible = available>=3
        miss_eval = np.count_nonzero((tab[1:]==0)&feasible)
        scarce = np.count_nonzero((st<2)&(available>=6))
        lost = 1-keep.mean()
        return 15*miss_train+3*np.count_nonzero(n==0)+1.5*miss_eval+.12*scarce+8*size+5*balance+4*lost
    best=None; best_score=float('inf')
    # Several seeded restarts allow storm-level swaps out of poor initial splits.
    for restart in range(6):
        assign = rng.choice(3,size=ns,p=ratios)
        assign[rng.choice(ns,3,replace=False)] = np.arange(3)
        score=evaluate(assign)
        for step in range(1000):
            if score<best_score: best_score=score; best=assign.copy()
            trial=assign.copy(); i=int(rng.integers(ns)); trial[i]=int(rng.integers(3))
            if step%5==0:
                j=int(rng.integers(ns)); trial[i],trial[j]=assign[j],assign[i]
            value=evaluate(trial)
            temp=.18*(1-step/1000)+.005
            if value<=score or rng.random()<math.exp(min(0,(score-value)/temp)):
                assign,score=trial,value
        if score<best_score: best_score=score; best=assign.copy()
    keep,tab,scene_assignment=evaluate(best,True)
    mapping={str(k):SPLITS[int(v)] for k,v in zip(storm_names,best)}
    accepted=[]; rejected=[]
    for r,k in zip(rows,keep):
        r=dict(r,split=mapping[r['sid']])
        if k: accepted.append(r)
        else: rejected.append(dict(r,reason='multi_storm_scene_split_conflict'))
    return accepted,rejected,mapping

def plan(args):
    scenes,inventory=discover(args.goes)
    margin=pd.Timedelta(minutes=args.max_offset_min)
    tracks,ambiguous=read_tracks(args.ibtracs,min(s[1] for s in scenes)-margin,max(s[1] for s in scenes)+margin)
    # Sorted temporal index avoids scanning all storms for every scene.
    fixes=sorted((r[0].value,sid,r) for sid,rows in tracks.items() for r in rows)
    times=[r[0] for r in fixes]; tolerance=int(args.max_offset_min*60*1e9)
    candidates=[]; rejects=[]; source_pairs={}; audit=Counter()
    for scene_id,t,ir,wv in scenes:
        near=defaultdict(list)
        for _,sid,r in fixes[bisect_left(times,t.value-tolerance):bisect_right(times,t.value+tolerance)]:
            near[sid].append(r)
        if not near: audit['scenes_without_nearby_fix']+=1; continue
        source_pairs[scene_id]=(ir,wv)
        for sid,rows in near.items():
            base=dict(scene_id=scene_id,sid=sid,split='')
            def reject(reason): rejects.append(dict(base,reason=reason))
            audit['nearby_storm_observations']+=1
            rows.sort(key=lambda r:abs((r[0]-t).total_seconds()))
            fix,nature,wind,lat,lon=rows[0]; offset=abs((fix-t).total_seconds())
            if len(rows)>1 and abs((rows[1][0]-t).total_seconds())==offset: reject('ambiguous_time'); continue
            if nature=='AMBIGUOUS': reject('ambiguous_best_track'); continue
            if nature!='TS': reject('not_tropical_fix'); continue
            if not math.isfinite(wind) or not 0<wind<=250: reject('missing_or_invalid_USA_WIND'); continue
            if not (math.isfinite(lat) and math.isfinite(lon) and -90<lat<90 and -180<=lon<=360): reject('invalid_center'); continue
            lon=(lon+180)%360-180
            # Fast visibility check using the raw satellite grid; exact QC follows.
            gx,gy=args.visibility.transform(lon,lat)
            if not (math.isfinite(gx) and math.isfinite(gy)) or abs(gx)>args.grid_limit or abs(gy)>args.grid_limit: reject('outside_GOES_disk'); continue
            if not re.fullmatch(r'[A-Za-z0-9_-]+',sid): raise ValueError('Unsafe storm ID')
            c=class_id(wind)
            candidates.append(dict(base,sample_id=f'{scene_id}_{sid}',scene_time=t.isoformat(),label_time=fix.isoformat(),offset_seconds=offset,lat=lat,lon=lon,wind_kt=wind,class_id=c,class_name=CLASSES[c],ir=ir,wv=wv))
    audit['eligible_before_split']=len(candidates)
    by_scene=defaultdict(list)
    for r in candidates: by_scene[r['scene_id']].append(r)
    jobs=[(*source_pairs[k],v) for k,v in sorted(by_scene.items())]
    audit['ambiguous_best_track_fixes']=ambiguous
    return jobs,rejects,dict(inventory),dict(audit)

def grid(ds, expected_band):
    if int(np.asarray(ds['band_id']).item()) != expected_band: raise ValueError('Unexpected ABI band ID')
    if ds['CMI'].dims != ('y','x') or ds['DQF'].dims != ('y','x'): raise ValueError('Unexpected CMI/DQF dimensions')
    if ds['CMI'].attrs.get('units','').strip().lower() not in ('k','kelvin'): raise ValueError('CMI must be Kelvin')
    p = dict(ds['goes_imager_projection'].attrs)
    h = float(p['perspective_point_height'])
    crs = CRS.from_cf(p)
    x,y = np.asarray(ds.x,dtype=float),np.asarray(ds.y,dtype=float)
    if ds.x.attrs.get('units','') not in ('rad','radian','radians') or ds.y.attrs.get('units','') not in ('rad','radian','radians'): raise ValueError('Expected scan angles in radians')
    x,y=x*h,y*h
    for a in (x,y):
        if len(a)<2 or not np.isfinite(a).all() or not (np.all(np.diff(a)>0) or np.all(np.diff(a)<0)): raise ValueError('Nonmonotonic satellite grid')
    return crs,x,y

def fractional(axis, coords):
    idx = np.arange(len(axis),dtype=float)
    if axis[0]>axis[-1]: axis,idx=axis[::-1],idx[::-1]
    return np.interp(coords,axis,idx,left=np.nan,right=np.nan)

def coordinates(crs,x,y,lat,lon,size,width_km):
    local = CRS.from_proj4(f'+proj=aeqd +lat_0={lat} +lon_0={lon} +datum=WGS84 +units=m +no_defs')
    # Pixel centers: extent exactly width_km; north at top.
    axis = (np.arange(size)+.5-size/2)*width_km*1000/size
    xx,yy = np.meshgrid(axis,axis[::-1])
    east,north = Transformer.from_crs(local,crs,always_xy=True).transform(xx,yy)
    return fractional(y,north),fractional(x,east)

def sample_band(ds, rr, cc):
    finite=np.isfinite(rr)&np.isfinite(cc)
    out=np.full(rr.shape,np.nan,dtype=np.float32)
    if not finite.any(): return out,np.zeros(rr.shape,dtype=bool)
    r0=max(0,int(np.floor(rr[finite].min()))); r1=min(ds.sizes['y'],int(np.ceil(rr[finite].max()))+1)
    c0=max(0,int(np.floor(cc[finite].min()))); c1=min(ds.sizes['x'],int(np.ceil(cc[finite].max()))+1)
    sub=ds.isel(y=slice(r0,r1),x=slice(c0,c1))
    a=np.asarray(sub.CMI.values,dtype=np.float32); q=np.asarray(sub.DQF.values)
    valid=np.isfinite(a)&(q==0)&(a>=100)&(a<=400)
    coord=np.stack([np.where(finite,rr-r0,0),np.where(finite,cc-c0,0)])
    weights=map_coordinates(valid.astype(np.float32),coord,order=1,mode='constant',cval=0,prefilter=False)
    values=map_coordinates(np.where(valid,a,0),coord,order=1,mode='constant',cval=0,prefilter=False)
    good=finite&(weights>=1-1e-6)
    out[good]=values[good]/weights[good]
    return out,good

def rgb_features(a,b,mask):
    out=[]
    for v,(lo,hi) in zip((a,b,a-b),NORMALIZATION):
        out.append(np.rint(np.clip((np.where(mask,v,lo)-lo)/(hi-lo),0,1)*255).astype(np.uint8))
    return np.stack(out,axis=-1)

def process(job):
    ir,wv,samples,stage,size,width,min_valid = job
    stage=Path(stage); rows=[]; rejected=[]
    with xr.open_dataset(ir,decode_times=False) as a,xr.open_dataset(wv,decode_times=False) as b:
        crs,x,y=grid(a,13); crs2,x2,y2=grid(b,8)
        if crs!=crs2 or not np.array_equal(x,x2) or not np.array_equal(y,y2): raise ValueError(f'Channel grid mismatch: {ir}')
        for r in samples:
            rr,cc=coordinates(crs,x,y,r['lat'],r['lon'],size,width)
            c13,m13=sample_band(a,rr,cc); c08,m08=sample_band(b,rr,cc)
            valid=m13&m08; ratio=float(valid.mean())
            center=valid[size//2-1:size//2+1,size//2-1:size//2+1]
            if ratio<min_valid or not center.all():
                rejected.append(dict(scene_id=r['scene_id'],sid=r['sid'],split=r['split'],reason='invalid_center_pixels' if not center.all() else 'insufficient_valid_pixels',valid_fraction=ratio,class_name=r['class_name'],wind_kt=r['wind_kt'])); continue
            name=r['sample_id']; folder=Path('_unsplit')
            im=Path('images')/folder/(name+'.png'); kelvin=Path('kelvin')/folder/(name+'.npz'); mask=Path('masks')/folder/(name+'.png')
            Image.fromarray(rgb_features(c13,c08,valid)).save(stage/im)
            Image.fromarray(valid.astype(np.uint8)*255).save(stage/mask)
            # Preserve original calibrated values, not just display PNGs.
            np.savez_compressed(stage/kelvin,c13=np.where(valid,c13,np.nan),c08=np.where(valid,c08,np.nan),valid=valid)
            rows.append(dict(r,image=str(im),kelvin=str(kelvin),mask=str(mask),valid_fraction=ratio,ir=ir,wv=wv))
    return rows,rejected

def summarize(rows):
    return {s:{'samples':sum(r['split']==s for r in rows),'storms':len({r['sid'] for r in rows if r['split']==s}),'classes':{c:sum(r['split']==s and r['class_name']==c for r in rows) for c in CLASSES},'storms_by_class':{c:len({r['sid'] for r in rows if r['split']==s and r['class_name']==c}) for c in CLASSES}} for s in SPLITS}

def sha(path):
    with path.open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,help='Directory containing goes16/, ibtracs.csv and processed/')
    p.add_argument('--ibtracs',type=Path,help='Override ibtracs.csv')
    p.add_argument('--goes',type=Path,help='Override goes16/ (timestamp/IR and timestamp/WV)')
    p.add_argument('--output',type=Path,help='Override processed/classification_raw_v1')
    p.add_argument('--image-size',type=int,default=384); p.add_argument('--width-km',type=float,default=800)
    p.add_argument('--min-valid',type=float,default=.90); p.add_argument('--max-offset-min',type=float,default=10)
    p.add_argument('--audit-dir',type=Path,help='Optional plan-only audit directory'); p.add_argument('--workers',type=int,default=2); p.add_argument('--plan-only',action='store_true')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--split-ratios',type=float,nargs=3,default=(.8,.1,.1),metavar=('TRAIN','VAL','TEST'))
    args=p.parse_args()
    if any(not math.isfinite(v) or v<=0 for v in args.split_ratios) or not math.isclose(sum(args.split_ratios),1,abs_tol=1e-8): p.error('Split ratios must be positive and sum to one')
    # Work from project root, from data/, or with --data-root /absolute/path.
    if args.data_root is None:
        choices=[Path.cwd(),Path.cwd()/'data',Path(__file__).resolve().parent.parent/'data']
        args.data_root=next((d for d in choices if (d/'goes16').is_dir() and (d/'ibtracs.csv').is_file()),Path.cwd()/'data')
    data=args.data_root.resolve()
    args.goes=(args.goes or data/'goes16').resolve()
    args.ibtracs=(args.ibtracs or data/'ibtracs.csv').resolve()
    args.output=(args.output or data/'processed'/'classification_raw_v1').resolve()
    if args.plan_only and args.audit_dir is None:
        args.audit_dir=data/'processed'/'classification_raw_v1_audit'
    required=[args.ibtracs]
    for path in required:
        if not path.is_file(): p.error(f'Missing required input: {path}')
    print(json.dumps({'goes':str(args.goes),'ibtracs':str(args.ibtracs),'output':str(args.output)},indent=2),flush=True)
    if args.image_size<32 or args.image_size%2 or not 0<args.min_valid<=1 or not 0<args.width_km or not 0<args.max_offset_min<=30 or args.workers<1: p.error('Invalid arguments')
    if not args.goes.is_dir(): raise FileNotFoundError(args.goes)
    # Obtain projection and scan extent directly from a valid raw C13/C08 pair.
    discovered,_=discover(args.goes)
    for _,_,ir,wv in discovered:
        try:
            with xr.open_dataset(ir,decode_times=False) as a, xr.open_dataset(wv,decode_times=False) as b:
                crs,x,y=grid(a,13); crs2,x2,y2=grid(b,8)
                if crs!=crs2 or not np.array_equal(x,x2) or not np.array_equal(y,y2): raise ValueError('Channel grid mismatch')
            args.visibility=Transformer.from_crs('EPSG:4326',crs,always_xy=True)
            args.grid_limit=max(np.abs(x).max(),np.abs(y).max())
            break
        except Exception as exc: print(f'Reference grid rejected: {ir}: {exc}',flush=True)
    else: raise RuntimeError('No valid raw reference grid')
    jobs,rejects,inventory,audit=plan(args)
    planned=[r for _,_,samples in jobs for r in samples]
    print(json.dumps({'inventory':inventory,'audit':audit,'planned_scenes':len(jobs),'planned_samples':len(planned),'rejections':dict(Counter(r['reason'] for r in rejects))},indent=2),flush=True)
    if not planned: raise RuntimeError('No eligible cyclone observations')
    if args.plan_only:
        preview,conflicts,_=split_samples(planned,args.seed,args.split_ratios)
        rejects.extend(conflicts)
        print(json.dumps({'provisional_splits':summarize(preview),'provisional_conflicts':len(conflicts)},indent=2),flush=True)
        if args.audit_dir:
            audit_path=args.audit_dir.resolve()
            if any(audit_path==root or root in audit_path.parents or audit_path in root.parents for root in (args.goes.resolve(),args.ibtracs.resolve())): raise ValueError('Audit directory must be separate from inputs')
            args.audit_dir.mkdir(parents=True,exist_ok=True)
            dump(args.audit_dir/'plan_summary.json',{'inventory':inventory,'audit':audit,'splits':summarize(preview),'pre_quality_filter':True,'provisional_scene_conflicts':len(conflicts),'rejections':dict(Counter(r['reason'] for r in rejects))})
            write_csv(args.audit_dir/'plan_rejections.csv',rejects,['scene_id','sid','split','reason','class_name','wind_kt','valid_fraction'])
            write_csv(args.audit_dir/'planned_samples.csv',preview,FIELDS)
        return
    out=args.output.resolve(); stage=out.with_name(out.name+'.work')
    if out.exists() or stage.exists(): raise FileExistsError(f'Output/staging exists: {out} / {stage}; review before choosing a fresh path')
    if any(out==root or root in out.parents or out in root.parents for root in (args.goes.resolve(),args.ibtracs.resolve())): raise ValueError('Output must be separate from raw inputs')
    stage.mkdir(parents=True)
    for kind in ('images','kelvin','masks'): (stage/kind/'_unsplit').mkdir(parents=True)
    for s in SPLITS:
        for c in CLASSES:
            for kind in ('images','kelvin','masks'): (stage/kind/s/c).mkdir(parents=True)
    work=[(ir,wv,samples,str(stage),args.image_size,args.width_km,args.min_valid) for ir,wv,samples in jobs]
    rows=[]
    def consume(results):
        for i,(accepted,denied) in enumerate(results,1):
            rows.extend(accepted); rejects.extend(denied)
            if i%25==0 or i==len(jobs): print(f'{i}/{len(jobs)} scenes; {len(rows)} crops',flush=True)
    if args.workers==1: consume(map(process,work))
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool: consume(pool.map(process,work,chunksize=1))
    # Allocate only after quality checks so rejected crops cannot distort stratification.
    quality_accepted=len(rows)
    rows,conflicts,mapping=split_samples(rows,args.seed,args.split_ratios)
    rejects.extend(conflicts)
    for r in rows:
        for field in ('image','mask','kelvin'):
            old=Path(r[field]); new=old.parent.parent/r['split']/r['class_name']/old.name
            (stage/old).rename(stage/new); r[field]=str(new)
    for r in conflicts:
        for field in ('image','mask','kelvin'): (stage/r[field]).unlink()
    for kind in ('images','kelvin','masks'): (stage/kind/'_unsplit').rmdir()
    audit['quality_accepted_before_split']=quality_accepted
    audit['eligible_after_split']=len(rows)
    audit['scene_conflicts']=len(conflicts)
    seen=set(); sids={}; scenes={}
    for r in rows:
        if r['sample_id'] in seen: raise ValueError('Duplicate sample')
        seen.add(r['sample_id'])
        if sids.setdefault(r['sid'],r['split'])!=r['split'] or scenes.setdefault(r['scene_id'],r['split'])!=r['split'] or mapping[r['sid']]!=r['split']: raise ValueError('Split leakage')
        for field in ('image','mask','kelvin'):
            if not (stage/r[field]).is_file(): raise FileNotFoundError(r[field])
    rows.sort(key=lambda r:(r['split'],r['scene_id'],r['sid']))
    write_csv(stage/'manifest.csv',rows,FIELDS)
    for s in SPLITS: write_csv(stage/f'{s}.csv',[r for r in rows if r['split']==s],FIELDS)
    write_csv(stage/'rejections.csv',rejects,['scene_id','sid','split','reason','valid_fraction','class_name','wind_kt'])
    dump(stage/'storm_splits.json',mapping); dump(stage/'classes.json',dict(enumerate(CLASSES)))
    counts=summarize(rows); warnings=[]
    for s,v in counts.items():
        if not v['samples']: warnings.append(f'{s}: no samples')
        for c,n in v['classes'].items():
            if n==0: warnings.append(f'{s}: missing {c}')
            elif v['storms_by_class'][c]<2: warnings.append(f'{s}: {c} represented by one storm')
    info={'version':'3.0-raw-only','inventory':inventory,'audit':audit,'split_strategy':'class-aware_storm-wise_with_scene_conflict_filter','seed':args.seed,'requested_split_ratios':args.split_ratios,'samples':len(rows),'planned_samples':len(planned),'splits':counts,'rejections':dict(Counter(r['reason'] for r in rejects)),'storm_leakage':False,'scene_leakage':False,'image_size':args.image_size,'width_km':args.width_km,'min_valid':args.min_valid,'max_offset_min':args.max_offset_min,'normalization':dict(zip(('C13','C08','C13-C08'),NORMALIZATION)),'ibtracs_sha256':sha(args.ibtracs),'warnings':warnings,'trainable':all(counts[s]['samples']>0 for s in SPLITS) and all(counts['train']['classes'].values())}
    dump(stage/'summary.json',info)
    dump(stage/'quality_audit.json',{'planned_classes':dict(Counter(r['class_name'] for r in planned)),'quality_accepted_before_split':quality_accepted,'accepted':counts,'rejections':dict(Counter(r['reason'] for r in rejects)),'quality_rejections_by_class':{c:dict(Counter(r['reason'] for r in rejects if r.get('class_name')==c)) for c in CLASSES}})
    if not rows or any(not counts[s]['samples'] for s in SPLITS): raise RuntimeError(f'Empty split; inspect {stage}/summary.json')
    stage.rename(out)
    print(json.dumps(info,indent=2),flush=True)
    print(f'Dataset saved: {out}',flush=True)

if __name__=='__main__': main()
