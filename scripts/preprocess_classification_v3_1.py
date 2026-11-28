#!/usr/bin/env python3
"""Build V3.1 full-precision GOES-16 intensity dataset from raw ABI and IBTrACS.
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
import shutil
import tempfile
import fcntl
from concurrent.futures import as_completed
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
FIELDS = ['sample_id','image','kelvin','mask','split','sid','scene_id','scene_time','label_time','offset_seconds','lat','lon','wind_kt','class_id','class_name','valid_fraction','inner_valid_fraction','master_valid_fraction','view_zenith_deg','center_source','basin','season','iflag','ir','wv','usa_agency','usa_iflag','label_provenance','suggested_label_weight','quality_tier','view_azimuth_deg','native_pixel_km','native_radial_footprint_approx_km','tensor_sha256']

def dump(path, obj):
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp,path)

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
            rows.sort(key=lambda r:(r[1]!=actual,r[1]))
            counts['duplicate_acquisition_folders_ignored']+=len(rows)-1
            issues.extend({'scene_id':r[1],'reason':'duplicate_actual_acquisition','actual_slot':actual} for r in rows[1:])
        _,sid,t,ir,wv=rows[0]; scenes.append((sid,t,ir,wv))
    counts['paired_scenes']=len(scenes)
    if not scenes: raise ValueError(f'No paired GOES-16 C08/C13 scenes under {root}; expected YYYYMMDD_HHMM/IR and WV')
    return scenes,{**dict(counts),'discovery_issues':issues}

def label_provenance(flag):
    # IBTrACS v4: first character describes the USA agency, not other agencies.
    first=flag[:1].upper()
    kind='original' if first in ('O','V') else 'interpolated' if first in ('P','I') else 'unknown'
    return dict(usa_iflag=first,label_provenance=kind,suggested_label_weight={'original':1.0,'interpolated':0.75,'unknown':1.0}[kind])

def read_tracks(path, start=None, end=None):
    required = {'SID','ISO_TIME','NATURE','USA_WIND','LAT','LON','TRACK_TYPE'}
    optional = {'USA_LAT','USA_LON','IFLAG','BASIN','SEASON','USA_AGENCY'}
    f = pd.read_csv(path,usecols=lambda c:c.strip() in required|optional,dtype=str,keep_default_na=False,low_memory=False)
    f.columns=f.columns.str.strip()
    if not required<=set(f): raise ValueError('IBTrACS missing columns: '+str(required-set(f)))
    for c in optional-set(f): f[c]=''
    for c in f: f[c]=f[c].str.strip()
    f['time']=pd.to_datetime(f.ISO_TIME,utc=True,errors='coerce')
    f=f.dropna(subset=['time']); f=f[f.TRACK_TYPE.str.lower().eq('main') & f.SID.ne('')]
    if start is not None: f=f[f.time>=start]
    if end is not None: f=f[f.time<=end]
    tracks=defaultdict(list); ambiguous=0
    for (sid,t),g in f.groupby(['SID','time'],sort=False):
        fixes=set()
        for r in g.itertuples(index=False):
            def num(v):
                try: return float(v)
                except (TypeError,ValueError): return float('nan')
            lat,lon=num(r.USA_LAT),num(r.USA_LON)
            source='USA'
            if not (math.isfinite(lat) and math.isfinite(lon) and -90<lat<90 and -180<=lon<=360):
                lat,lon=num(r.LAT),num(r.LON); source='IBTrACS'
            wind=num(r.USA_WIND)
            # None makes missing values comparable for duplicate checking.
            fixes.add((r.NATURE,*(v if math.isfinite(v) else None for v in (wind,lat,lon)),source,r.BASIN,r.SEASON,r.IFLAG,r.USA_AGENCY))
        if len(fixes)!=1:
            tracks[sid].append((t,'AMBIGUOUS',float('nan'),float('nan'),float('nan'),{})); ambiguous+=1; continue
        nature,wind,lat,lon,source,basin,season,flag,agency=next(iter(fixes))
        tracks[sid].append((t,nature,*(float('nan') if v is None else v for v in (wind,lat,lon)),dict(center_source=source,basin=basin,season=season,iflag=flag,usa_agency=agency,**label_provenance(flag))))
    for rows in tracks.values(): rows.sort(key=lambda x:x[0])
    return tracks,ambiguous

def split_samples(rows, seed, ratios, frozen=None, restarts=8, steps=1200):
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
    fixed=np.array([SPLITS.index(frozen.get(str(k), "train")) if frozen is not None else -1 for k in storm_names])
    def evaluate(assign, details=False):
        sp = assign[si]
        votes = np.bincount(gi*3+sp, minlength=ng*3).reshape(ng,3)
        # Deterministic tie break: train, then validation, then test.
        scene_assignment = votes.argmax(axis=1)
        if frozen is not None:
            # Preserve held-out observations in shared scenes before train observations.
            scene_assignment=np.where(votes[:,2]>0,2,np.where(votes[:,1]>0,1,0))
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
        return 15*miss_train+1000*np.count_nonzero(n==0)+1.5*miss_eval+.12*scarce+8*size+5*balance+4*lost
    best=None; best_score=float('inf')
    # Several seeded restarts allow storm-level swaps out of poor initial splits.
    for restart in range(1 if frozen is not None else restarts):
        assign = rng.choice(3,size=ns,p=ratios)
        assign[rng.choice(ns,3,replace=False)] = np.arange(3)
        if frozen is not None: assign=fixed.copy()
        score=evaluate(assign)
        for step in range(0 if frozen is not None else steps):
            if score<best_score: best_score=score; best=assign.copy()
            trial=assign.copy(); i=int(rng.integers(ns)); trial[i]=int(rng.integers(3))
            if step%5==0:
                j=int(rng.integers(ns)); trial[i],trial[j]=assign[j],assign[i]
            value=evaluate(trial)
            temp=.18*(1-step/steps)+.005
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
    if frozen is not None: mapping={**frozen,**mapping}
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
            fix,nature,wind,lat,lon,meta=rows[0]; offset=abs((fix-t).total_seconds())
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
            angle=view_zenith(lat,lon,args.sat_lon,args.sat_height)
            if angle>args.max_zenith: reject('view_zenith_exceeded'); continue
            c=class_id(wind)
            candidates.append(dict(base,**meta,view_zenith_deg=angle,sample_id=f'{scene_id}_{sid}',scene_time=t.isoformat(),label_time=fix.isoformat(),offset_seconds=offset,lat=lat,lon=lon,wind_kt=wind,class_id=c,class_name=CLASSES[c],ir=ir,wv=wv))
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


ECEF=Transformer.from_crs('EPSG:4979','EPSG:4978',always_xy=True)
def view_zenith(lat,lon,sat_lon,height):
    xyz=np.asarray(ECEF.transform(lon,lat,0.)); sat=np.asarray(ECEF.transform(sat_lon,0.,height))
    p,l=np.deg2rad([lat,lon]); normal=np.array([np.cos(p)*np.cos(l),np.cos(p)*np.sin(l),np.sin(p)])
    ray=sat-xyz
    return float(np.rad2deg(np.arccos(np.clip(np.dot(ray,normal)/np.linalg.norm(ray),-1.,1.))))

def center_slice(master, size):
    if size>master or (master-size)%2: raise ValueError('Incompatible centered crop dimensions')
    start=(master-size)//2
    return slice(start,start+size)

def physical_features(a,b,valid,spacing_km):
    """Mask-aware, image-only descriptors; NaN marks unavailable statistics."""
    n=a.shape[0]; axis=(np.arange(n)+.5-n/2)*spacing_km
    xx,yy=np.meshgrid(axis,axis[::-1]); radius=np.hypot(xx,yy)
    az=(np.arctan2(yy,xx)+2*np.pi)%(2*np.pi)
    result={}
    def mean(v,m): return float(v[m].mean(dtype=np.float64)) if m.any() else None
    for name,v in [('c13',a),('c08',b),('btd',a-b)]:
        previous=None
        for lo,hi in zip((0,25,50,100,150,200,300),(25,50,100,150,200,300,400)):
            region=(radius>=lo)&(radius<hi); m=region&valid
            prefix=f'{name}_{lo}_{hi}km'
            enough=m.sum()>=max(4,.8*region.sum()) and region.any()
            mu=mean(v,m) if enough else None
            result[prefix+'_mean']=mu
            result[prefix+'_std']=float(v[m].std(dtype=np.float64)) if enough else None
            result[prefix+'_radial_delta']=mu-previous if mu is not None and previous is not None else None
            previous=mu
            sectors=[]
            for k in range(8):
                sector=region&(az>=k*np.pi/4)&(az<(k+1)*np.pi/4); sm=sector&valid
                if sm.sum()>=max(2,.8*sector.sum()): sectors.append(mean(v,sm))
            result[prefix+'_azimuth_std']=float(np.std(sectors)) if len(sectors)==8 else None
    for threshold in (200,220,235):
        region=radius<200; m=region&valid
        result[f'c13_cold_fraction_lt{threshold}K_200km']=float((a[m]<threshold).mean()) if m.sum()>=.8*region.sum() and m.any() else None
    eye=result['c13_0_25km_mean']; ring=result['c13_25_50km_mean']
    result['c13_center_ring_contrast']=eye-ring if eye is not None and ring is not None else None
    return result


def viewing_geometry(lat,lon,sat_lon,height,x,y,angle):
    xyz=np.asarray(ECEF.transform(lon,lat,0.)); sat=np.asarray(ECEF.transform(sat_lon,0.,height)); ray=sat-xyz
    p,l=np.deg2rad([lat,lon]); east=np.array([-np.sin(l),np.cos(l),0]); north=np.array([-np.sin(p)*np.cos(l),-np.sin(p)*np.sin(l),np.cos(p)])
    e,n=np.dot(ray,east),np.dot(ray,north)
    az=float(np.degrees(np.arctan2(e,n))%360) if np.hypot(e,n)>1e-6 else None
    spacing=float(np.sqrt(abs(np.median(np.diff(x))*np.median(np.diff(y))))/1000)
    # Simple secant proxy, not an exact off-nadir footprint or parallax correction.
    return dict(view_azimuth_deg=az,native_pixel_km=spacing,native_radial_footprint_approx_km=spacing/max(np.cos(np.deg2rad(angle)),1e-6))

def process(job):
    ir,wv,samples,stage,cfg=job
    stage=Path(stage); rows=[]; rejected=[]
    n=cfg['master_size']; p=center_slice(n,cfg['image_size']); q=center_slice(n,cfg['inner_size'])
    try:
        with xr.open_dataset(ir,decode_times=False) as a,xr.open_dataset(wv,decode_times=False) as b:
            crs,x,y=grid(a,13); crs2,x2,y2=grid(b,8)
            if crs!=crs2: raise ValueError('Channel projection mismatch')
            projection=a['goes_imager_projection'].attrs
            for r in samples:
                try:
                    angle=view_zenith(r['lat'],r['lon'],float(projection['longitude_of_projection_origin']),float(projection['perspective_point_height']))
                    rr,cc=coordinates(crs,x,y,r['lat'],r['lon'],n,cfg['master_km'])
                    r2,c2=coordinates(crs2,x2,y2,r['lat'],r['lon'],n,cfg['master_km'])
                    c13,m13=sample_band(a,rr,cc); c08,m08=sample_band(b,r2,c2)
                    valid=m13&m08; main=float(valid[p,p].mean()); inner=float(valid[q,q].mean()); master=float(valid.mean())
                    r=dict(r,valid_fraction=main,inner_valid_fraction=inner,master_valid_fraction=master,view_zenith_deg=angle,**viewing_geometry(r['lat'],r['lon'],float(projection['longitude_of_projection_origin']),float(projection['perspective_point_height']),x,y,angle))
                    r['quality_tier']='high' if main>=cfg['high_valid'] and inner>=cfg['high_inner_valid'] and angle<=cfg['high_zenith'] else 'acceptable'
                    if angle>cfg['max_zenith']: reason='view_zenith_exceeded'
                    elif not valid[n//2-1:n//2+1,n//2-1:n//2+1].all(): reason='invalid_center_pixels'
                    elif inner<cfg['min_inner_valid']: reason='insufficient_inner_validity'
                    elif main<cfg['min_valid']: reason='insufficient_primary_validity'
                    else: reason=None
                    if reason: rejected.append(dict(r,reason=reason)); continue
                    name=r['sample_id']; year=pd.Timestamp(r['scene_time']).year
                    tensor=Path('tensors')/str(year)/(name+'.npz'); preview=Path('previews')/r['class_name']/(name+'.png')
                    (stage/tensor).parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(stage/tensor,c13=c13.astype('float32'),c08=c08.astype('float32'),mask_c13=m13,mask_c08=m08,valid=valid)
                    if cfg['previews']: Image.fromarray(rgb_features(c13[p,p],c08[p,p],valid[p,p])).save(stage/preview)
                    with (stage/tensor).open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
                    rows.append(dict(r,kelvin=str(tensor),image=str(preview) if cfg['previews'] else '',mask='',tensor_sha256=digest,features=physical_features(c13,c08,valid,cfg['master_km']/n)))
                except Exception as exc:
                    rejected.append(dict(r,reason='crop_processing_error',detail=f'{type(exc).__name__}: {exc}'))
    except Exception as exc:
        rejected.extend(dict(r,reason='scene_read_error',detail=f'{type(exc).__name__}: {exc}') for r in samples)
    return rows,rejected

CACHE_VERSION='3.1.0'
CROP_KEYS=('master_size','master_km','image_size','inner_size','min_valid','min_inner_valid','max_zenith','high_valid','high_inner_valid','high_zenith','previews')
def fingerprint(ir,wv,samples,cfg):
    sources=[]
    for path in (ir,wv):
        st=Path(path).stat(); sources.append([str(Path(path).resolve()),st.st_size,st.st_mtime_ns,st.st_ino])
    content=dict(version=CACHE_VERSION,sources=sources,samples=samples,config={k:cfg[k] for k in CROP_KEYS})
    return hashlib.sha256(json.dumps(content,sort_keys=True,allow_nan=False).encode()).hexdigest()

def sha256_file(path):
    with Path(path).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()

def cache_process(job):
    ir,wv,samples,cache,cfg,only=job; cache=Path(cache)
    key=fingerprint(ir,wv,samples,cfg); target=cache/key
    if (target/'result.json').is_file():
        data=json.loads((target/'result.json').read_text())
        if all((target/r['kelvin']).is_file() and sha256_file(target/r['kelvin'])==r['tensor_sha256'] and (not r['image'] or (target/r['image']).is_file()) for r in data['rows']):
            return data['rows'],data['rejected'],str(target),True
        raise RuntimeError(f'Damaged cache entry: {target}; remove this entry and rerun')
    if only: raise RuntimeError(f'Cache miss for {samples[0]["scene_id"]}; run normally or --cache-only first')
    temp=Path(tempfile.mkdtemp(prefix='.partial-',dir=cache))
    try:
        for c in CLASSES: (temp/'previews'/c).mkdir(parents=True)
        rows,rejected=process((ir,wv,samples,str(temp),cfg))
        if fingerprint(ir,wv,samples,cfg)!=key: raise RuntimeError('Source changed during crop processing')
        if any(r['reason'] in ('crop_processing_error','scene_read_error') for r in rejected):
            # Keep the results for this build, but never cache transient failures.
            return rows,rejected,str(temp),False
        dump(temp/'result.json',dict(rows=rows,rejected=rejected)); temp.rename(target)
        return rows,rejected,str(target),False
    except BaseException:
        shutil.rmtree(temp,ignore_errors=True); raise

def link_or_copy(src,dst):
    dst.parent.mkdir(parents=True,exist_ok=True)
    # Copy rather than hardlink so edits to an exported dataset cannot mutate cache.
    shutil.copy2(src,dst)

def feature_report(stage,rows):
    fields=sorted({k for r in rows for k in r.get('features',{})})
    data=[dict(sample_id=r['sample_id'],split=r['split'],**r.get('features',{})) for r in rows]
    write_csv(stage/'manifests/physical_features.csv',data,['sample_id','split']+fields)
    stats={}
    for k in fields:
        v=np.asarray([r['features'][k] for r in rows if r['split']=='train' and r['features'].get(k) is not None],dtype=float)
        stats[k]=dict(mean=float(v.mean()) if v.size else None,std=float(v.std()) if v.size else None,count=int(v.size))
    dump(stage/'metadata/physical_feature_normalization.json',dict(method='training_only',features=stats,missing_policy='impute training mean; preserve missing indicator; drop all-missing or constant features'))

def summarize(rows):
    return {s:{'samples':sum(r['split']==s for r in rows),'storms':len({r['sid'] for r in rows if r['split']==s}),'classes':{c:sum(r['split']==s and r['class_name']==c for r in rows) for c in CLASSES},'storms_by_class':{c:len({r['sid'] for r in rows if r['split']==s and r['class_name']==c}) for c in CLASSES}} for s in SPLITS}

def deduplicate_fixes(rows):
    # Select best temporal match among QC-passing acquisitions, then best validity.
    seen=set(); keep=[]; excluded=[]
    for r in sorted(rows,key=lambda r:(r['offset_seconds'],r.get('quality_tier')!='high',-r['valid_fraction'],-r['inner_valid_fraction'],r['sample_id'])):
        k=(r['sid'],r['label_time'])
        if k in seen: excluded.append(dict(r,reason='duplicate_label_fix',split='excluded'))
        else: seen.add(k); keep.append(r)
    return keep,excluded

def normalization(stage,rows,cfg):
    s=center_slice(cfg['master_size'],cfg['image_size']); count=np.zeros(3,dtype=np.int64); sums=np.zeros(3); sums2=np.zeros(3)
    clip=np.zeros(3,dtype=np.int64); mins=np.full(3,np.inf); maxs=np.full(3,-np.inf)
    for r in rows:
        if r['split']!='train': continue
        with np.load(stage/r['kelvin'],allow_pickle=False) as z:
            a,b=z['c13'][s,s],z['c08'][s,s]; valid=z['mask_c13'][s,s]&z['mask_c08'][s,s]
            for i,v in enumerate((a,b,a-b)):
                v=v[valid].astype(np.float64); lo,hi=NORMALIZATION[i]
                count[i]+=v.size; sums[i]+=v.sum(); sums2[i]+=np.square(v).sum(); clip[i]+=np.count_nonzero((v<lo)|(v>hi)); mins[i]=min(mins[i],v.min()); maxs[i]=max(maxs[i],v.max())
    if np.any(count==0): raise ValueError('No valid training pixels')
    mean=sums/count; std=np.sqrt(np.maximum(sums2/count-mean**2,0))
    if np.any(std<1e-8): raise ValueError('Degenerate training normalization')
    return {'channels':['C13','C08','C13-C08'],'method':'train_valid_joint_pixels_zscore','crop_size':cfg['image_size'],'mean':mean.tolist(),'std':std.tolist(),'valid_pixel_count':count.tolist(),'min':mins.tolist(),'max':maxs.tolist(),'audit_range_outside_fraction':(clip/count).tolist(),'audit_ranges':NORMALIZATION,'clip':False,'invalid_normalized_fill':0.0}

def plots(stage,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=stage/'reports'
    for key,title in [('classes','Observations by class'),('storms_by_class','Independent storms by class')]:
        fig,ax=plt.subplots(figsize=(9,4)); x=np.arange(7)
        for i,s in enumerate(SPLITS): ax.bar(x+(i-1)*.25,[stats[s][key][c] for c in CLASSES],.25,label=s)
        ax.set_xticks(x,CLASSES); ax.set_ylabel('Count'); ax.set_title(title); ax.legend(); fig.tight_layout(); fig.savefig(folder/('class_distribution.png' if key=='classes' else 'storm_distribution.png'),dpi=150); plt.close(fig)
    for field,title,xlabel in [('wind_kt','Wind speed distribution','Wind (knots)'),('valid_fraction','Primary crop validity','Valid fraction'),('view_zenith_deg','Viewing geometry','Zenith (degrees)')]:
        fig,ax=plt.subplots(figsize=(8,4))
        for s in SPLITS: ax.hist([r[field] for r in rows if r['split']==s],bins=25,alpha=.5,label=s)
        ax.set(title=title,xlabel=xlabel,ylabel='Count'); ax.legend(); fig.tight_layout(); fig.savefig(folder/({'wind_kt':'wind_distribution','valid_fraction':'quality_distribution','view_zenith_deg':'view_zenith_distribution'}[field]+'.png'),dpi=150); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5))
    for s in SPLITS:
        subset=[r for r in rows if r['split']==s]; ax.scatter([r['lon'] for r in subset],[r['lat'] for r in subset],s=7,alpha=.5,label=s)
    ax.set(xlabel='Longitude (degrees)',ylabel='Latitude (degrees)',title='Storm observation locations'); ax.legend(); fig.tight_layout(); fig.savefig(folder/'geographical_distribution.png',dpi=150); plt.close(fig)
    # Bounded deterministic subsample of valid training pixels for visual audit.
    rng=np.random.default_rng(42); vals=[[],[],[]]
    for r in rows:
        if r['split']!='train': continue
        with np.load(stage/r['kelvin'],allow_pickle=False) as z:
            a,b=z['c13'],z['c08']; v=np.flatnonzero(z['valid']); idx=rng.choice(v,min(128,len(v)),replace=False)
            aa,bb=a.ravel()[idx],b.ravel()[idx]
            for i,x in enumerate((aa,bb,aa-bb)): vals[i].extend(x.tolist())
    fig,axs=plt.subplots(1,3,figsize=(12,4))
    for ax,v,c in zip(axs,vals,['C13','C08','C13-C08']): ax.hist(v,bins=60); ax.set(title=c,xlabel='Kelvin',ylabel='Sampled valid pixels')
    fig.tight_layout(); fig.savefig(folder/'temperature_distribution.png',dpi=150); plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=Path('data')); p.add_argument('--goes',type=Path); p.add_argument('--ibtracs',type=Path); p.add_argument('--output',type=Path)
    p.add_argument('--master-size',type=int,default=480); p.add_argument('--master-km',type=float,default=1000.); p.add_argument('--image-size',type=int,default=384); p.add_argument('--inner-size',type=int,default=192)
    p.add_argument('--min-valid',type=float,default=.90); p.add_argument('--min-inner-valid',type=float,default=.95); p.add_argument('--max-offset-min',type=float,default=10); p.add_argument('--max-zenith',type=float,default=70)
    p.add_argument('--split-ratios',nargs=3,type=float,default=(.7,.15,.15)); p.add_argument('--seed',type=int,default=42); p.add_argument('--workers',type=int,default=4); p.add_argument('--plan-only',action='store_true'); p.add_argument('--audit-dir',type=Path); p.add_argument('--no-previews',action='store_true'); p.add_argument('--no-plots',action='store_true')
    p.add_argument('--high-valid',type=float,default=.95); p.add_argument('--high-inner-valid',type=float,default=.98); p.add_argument('--high-zenith',type=float,default=60)
    p.add_argument('--cache-dir',type=Path); p.add_argument('--cache-only',action='store_true'); p.add_argument('--cached-only',action='store_true'); p.add_argument('--freeze-splits',type=Path); p.add_argument('--freeze-test-manifest',type=Path); p.add_argument('--split-restarts',type=int,default=8); p.add_argument('--split-steps',type=int,default=1200)
    args=p.parse_args(); root=args.data_root.resolve(); args.goes=(args.goes or root/'goes16').resolve(); args.ibtracs=(args.ibtracs or root/'ibtracs.csv').resolve(); out=(args.output or root/'processed/classification_v3_1').resolve()
    if not (0<args.inner_size<=args.image_size<=args.master_size and all(x%2==0 for x in (args.inner_size,args.image_size,args.master_size)) and args.master_km>0 and 0<args.min_valid<=1 and 0<args.min_inner_valid<=1 and 0<args.max_offset_min<=30 and 0<args.max_zenith<90 and args.workers>=1): p.error('Invalid crop, quality or worker arguments')
    if any(x<=0 for x in args.split_ratios) or not math.isclose(sum(args.split_ratios),1): p.error('Split ratios must be positive and sum to 1')
    if not (args.min_valid<=args.high_valid<=1 and args.min_inner_valid<=args.high_inner_valid<=1 and 0<args.high_zenith<=args.max_zenith and args.split_restarts>0 and args.split_steps>0): p.error('Invalid high-quality thresholds or split search settings')
    if sum([args.plan_only,args.cache_only,args.cached_only])>1: p.error('Choose only one execution mode')
    if args.freeze_test_manifest and not args.freeze_splits: p.error('--freeze-test-manifest requires --freeze-splits')
    frozen=json.loads(args.freeze_splits.read_text()) if args.freeze_splits else None
    if frozen is not None and (not isinstance(frozen,dict) or any(v not in SPLITS for v in frozen.values())): p.error('Invalid frozen storm mapping')
    cfg=dict(high_valid=args.high_valid,high_inner_valid=args.high_inner_valid,high_zenith=args.high_zenith,split_restarts=args.split_restarts,split_steps=args.split_steps,master_size=args.master_size,master_km=args.master_km,image_size=args.image_size,inner_size=args.inner_size,primary_km=args.master_km*args.image_size/args.master_size,inner_km=args.master_km*args.inner_size/args.master_size,min_valid=args.min_valid,min_inner_valid=args.min_inner_valid,max_zenith=args.max_zenith,max_offset_min=args.max_offset_min,split_ratios=args.split_ratios,seed=args.seed,previews=not args.no_previews,workers=args.workers)
    def safe(path):
        if any(path==a or path in a.parents or a in path.parents for a in (args.goes,args.ibtracs)): raise ValueError('Output must be separate from raw inputs')
    safe(out)
    discovered,_=discover(args.goes)
    for _,_,ir,wv in discovered:
        try:
            with xr.open_dataset(ir,decode_times=False) as a,xr.open_dataset(wv,decode_times=False) as b:
                crs,x,y=grid(a,13); grid(b,8); projection=a['goes_imager_projection'].attrs
                args.sat_lon=float(projection['longitude_of_projection_origin']); args.sat_height=float(projection['perspective_point_height'])
            args.visibility=Transformer.from_crs('EPSG:4326',crs,always_xy=True); args.grid_limit=max(np.abs(x).max(),np.abs(y).max()); break
        except Exception as exc: print(f'Reference grid rejected: {ir}: {exc}',flush=True)
    else: raise RuntimeError('No readable raw reference pair')
    jobs,rejects,inventory,audit=plan(args); planned=[r for _,_,samples in jobs for r in samples]
    print(f'{len(discovered)} paired scenes; {len(jobs)} eligible scenes; {len(planned)} candidate crops',flush=True)
    if args.plan_only:
        d=(args.audit_dir or out.with_name(out.name+'_audit')).resolve(); safe(d); d.mkdir(parents=True,exist_ok=True)
        unique,dups=deduplicate_fixes([dict(r,valid_fraction=0.,inner_valid_fraction=0.) for r in planned]); preview,conflicts,_=split_samples(unique,args.seed,args.split_ratios,frozen,args.split_restarts,args.split_steps)
        info=dict(pre_quality_filter=True,inventory=inventory,audit=audit,planned_crops=len(planned),unique_fixes=len(unique),provisional_splits=summarize(preview),provisional_conflicts=len(conflicts),rejections=dict(Counter(r['reason'] for r in rejects)))
        dump(d/'plan_summary.json',info); write_csv(d/'planned_samples.csv',planned,FIELDS); write_csv(d/'plan_rejections.csv',rejects,['scene_id','sid','reason']); print(json.dumps(info,indent=2),flush=True); return
    cache=(args.cache_dir or root/'processed/.classification_v3_1_cache').resolve(); safe(cache)
    if cache==out or cache in out.parents or out in cache.parents: raise ValueError('Cache and output must be separate')
    if out.exists() and not args.cache_only: raise FileExistsError(f'Output exists: {out}; choose a new --output')
    cache.mkdir(parents=True,exist_ok=True); out.parent.mkdir(parents=True,exist_ok=True)
    # Linux advisory lock prevents concurrent writers sharing this cache.
    lock=(cache/'.lock').open('a'); fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stage=None if args.cache_only else Path(tempfile.mkdtemp(prefix=out.name+'.work-',dir=out.parent))
    if stage is not None:
        for d in ['tensors','previews','manifests','metadata','reports']: (stage/d).mkdir(parents=True,exist_ok=True)
    work=[(ir,wv,samples,str(cache),cfg,args.cached_only) for ir,wv,samples in jobs]; rows=[]; hits=0; errors=0
    def consume(results):
        nonlocal hits,errors
        for i,(accepted,denied,location,hit) in enumerate(results,1):
            loc=Path(location); hits+=int(hit)
            errors+=sum(r['reason'] in ('crop_processing_error','scene_read_error') for r in denied)
            if stage is not None:
                for r in accepted:
                    for k in ('kelvin','image'):
                        if r[k]: link_or_copy(loc/r[k],stage/r[k])
            rows.extend(accepted); rejects.extend(denied)
            if loc.name.startswith('.partial-'): shutil.rmtree(loc)
            if i%25==0 or i==len(work): print(f'QC {i}/{len(work)} scenes; accepted {len(rows)} crops; cache hits {hits}; processing errors {errors}',flush=True)
    if args.workers==1: consume(map(cache_process,work))
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn')) as pool:
            futures=[pool.submit(cache_process,j) for j in work]; consume(f.result() for f in as_completed(futures))
    if args.cache_only:
        print(f'Cache populated: {len(rows)} accepted crops, {hits} cache hits, {errors} processing errors',flush=True); return
    rows.sort(key=lambda r:r['sample_id'])
    if errors: print(f'WARNING: {errors} crop/scene processing errors; inspect rejection report. Failed scenes will be retried on future runs.',flush=True)
    quality_count=len(rows); unique,duplicates=deduplicate_fixes(rows)
    accepted,conflicts,mapping=split_samples(unique,args.seed,args.split_ratios,frozen,args.split_restarts,args.split_steps)
    if args.freeze_test_manifest:
        with args.freeze_test_manifest.open(newline='') as f: old={r['sample_id']:r for r in csv.DictReader(f)}
        if not old: raise ValueError('Empty frozen test manifest')
        actual={r['sample_id']:r for r in accepted if r['split']=='test'}
        for k,v in old.items():
            if k not in actual or any(str(actual[k][field])!=v.get(field) for field in ('tensor_sha256','sid','label_time','wind_kt','class_id')):
                raise ValueError(f'Frozen test sample missing or changed: {k}')
        added=[r for r in accepted if r['split']=='test' and r['sample_id'] not in old]
        accepted=[r for r in accepted if r['split']!='test' or r['sample_id'] in old]
        conflicts.extend(dict(r,reason='new_sample_in_frozen_test_storm') for r in added)
    # Preserve all QC-passing tensors for later experiments; excluded rows never enter split manifests.
    excluded=[dict(r,split='excluded') for r in duplicates+conflicts]; rejects.extend(excluded)
    seen=set(); sids={}; scenes={}; fixes=set()
    for r in accepted:
        k=(r['sid'],r['label_time'])
        if r['sample_id'] in seen or k in fixes or sids.setdefault(r['sid'],r['split'])!=r['split'] or scenes.setdefault(r['scene_id'],r['split'])!=r['split']: raise ValueError('Sample, fix, storm or scene leakage')
        seen.add(r['sample_id']); fixes.add(k)
        if not (stage/r['kelvin']).is_file(): raise FileNotFoundError(r['kelvin'])
    stats=summarize(accepted)
    if any(not stats[s]['samples'] for s in SPLITS): raise RuntimeError('Empty split; inspect staged output')
    norm=normalization(stage,accepted,cfg)
    feature_report(stage,accepted)
    accepted.sort(key=lambda r:(r['split'],r['sid'],r['label_time'])); master=accepted+excluded
    write_csv(stage/'manifests/master.csv',master,FIELDS+['reason']); write_csv(stage/'manifests/excluded.csv',excluded,FIELDS+['reason'])
    for s in SPLITS: write_csv(stage/'manifests'/f'{s}.csv',[r for r in accepted if r['split']==s],FIELDS)
    breakdown=[]
    for split in SPLITS:
        for c in CLASSES:
            for tier in ('high','acceptable'):
                for provenance in ('original','interpolated','unknown'):
                    selected=[r for r in accepted if r['split']==split and r['class_name']==c and r['quality_tier']==tier and r['label_provenance']==provenance]
                    breakdown.append(dict(split=split,class_name=c,quality_tier=tier,label_provenance=provenance,samples=len(selected),storms=len({r['sid'] for r in selected})))
    write_csv(stage/'reports/quality_label_breakdown.csv',breakdown,['split','class_name','quality_tier','label_provenance','samples','storms'])
    write_csv(stage/'reports/rejections.csv',rejects,FIELDS+['reason','detail']); dump(stage/'manifests/storm_splits.json',mapping)
    dump(stage/'metadata/classes.json',dict(enumerate(CLASSES))); dump(stage/'metadata/normalization.json',norm); dump(stage/'metadata/preprocessing_config.json',cfg)
    with args.ibtracs.open('rb') as f: digest=hashlib.file_digest(f,'sha256').hexdigest()
    warnings=[f'{s}: {c} has {stats[s]["storms_by_class"][c]} independent storms' for s in SPLITS for c in CLASSES if stats[s]['storms_by_class'][c]<2]
    info=dict(version='classification_v3_1',cache_hits=hits,processing_errors=errors,quality_tiers=dict(Counter(r['quality_tier'] for r in accepted)),label_provenance=dict(Counter(r['label_provenance'] for r in accepted)),config=cfg,inventory=inventory,audit=audit,planned_samples=len(planned),quality_accepted=quality_count,duplicate_fixes_excluded=len(duplicates),scene_conflicts_excluded=len(conflicts),samples=len(accepted),splits=stats,actual_ratios={s:stats[s]['samples']/len(accepted) for s in SPLITS},rejections=dict(Counter(r['reason'] for r in rejects)),ibtracs_sha256=digest,warnings=warnings,trainable=all(stats['train']['classes'].values()),storm_leakage=False,scene_leakage=False,duplicate_label_fixes=False)
    dump(stage/'metadata/dataset_summary.json',info); dump(stage/'reports/leakage_report.json',dict(storm_leakage=False,scene_leakage=False,duplicate_label_fixes=False,checked_samples=len(accepted),excluded_conflicts=len(conflicts)))
    dump(stage/'reports/quality_audit.json',dict(planned_classes=dict(Counter(r['class_name'] for r in planned)),quality_accepted=quality_count,quality_tiers_by_split={s:dict(Counter(r['quality_tier'] for r in accepted if r['split']==s)) for s in SPLITS},label_provenance_by_split={s:dict(Counter(r['label_provenance'] for r in accepted if r['split']==s)) for s in SPLITS},accepted_splits=stats,rejections_by_class={c:dict(Counter(r['reason'] for r in rejects if r.get('class_name')==c)) for c in CLASSES}))
    if not args.no_plots: plots(stage,accepted,stats)
    stage.rename(out); print(json.dumps(info,indent=2),flush=True); print(f'Dataset saved: {out}',flush=True)

if __name__=='__main__': main()
