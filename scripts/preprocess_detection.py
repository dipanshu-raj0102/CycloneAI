#!/usr/bin/env python3
"""GOES-16 ABI L2 C08/C13 + IBTrACS -> Ultralytics one-keypoint pose dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import multiprocessing as mp
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict
from functools import lru_cache
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw
from pyproj import CRS, Geod, Transformer

LOG = logging.getLogger('goes_pose')
SPLITS = ('train', 'val', 'test')
RATIOS = np.array([.8, .1, .1])
TARGET = {'TS', 'SS'}
GEOD = Geod(ellps='WGS84')
PATTERN = re.compile(r'OR_ABI-L2-CMIPF-M\dC(?P<band>08|13)_G16_s(?P<start>\d{14})_e(?P<end>\d{14})_c\d{14}\.nc$')


@dataclass(frozen=True)
class Scene:
    id: str
    ir: str
    wv: str
    midpoint: str
    nominal_offset_min: float


def dump(path: Path, obj) -> None:
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(temp, path)


def stamp(value: str) -> datetime:
    return datetime.strptime(value[:13], '%Y%j%H%M%S').replace(tzinfo=timezone.utc) + timedelta(seconds=int(value[13])/10)


def discover(root: Path, max_pair_seconds: float):
    scenes, issues = [], []
    seen = set()
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not re.fullmatch(r'\d{8}_\d{4}', folder.name):
            continue
        try:
            nominal = datetime.strptime(folder.name, '%Y%m%d_%H%M').replace(tzinfo=timezone.utc)
            pairs = []
            for sub, band in [('IR', '13'), ('WV', '08')]:
                files = sorted((folder/sub).glob('*.nc'))
                candidates = [(p, PATTERN.fullmatch(p.name)) for p in files]
                matches = [(p,m) for p,m in candidates if m and m['band']==band]
                if len(matches) != 1:
                    raise ValueError(f'{sub}: expected exactly one matching NetCDF, found {len(matches)}')
                p,m = matches[0]
                start,end = stamp(m['start']),stamp(m['end'])
                if not timedelta(seconds=0) < end-start <= timedelta(minutes=20):
                    raise ValueError('invalid scan interval')
                pairs.append((p,start,end))
            (ir, a,b),(wv,c,d) = pairs
            if max(abs((a-c).total_seconds()),abs((b-d).total_seconds())) > max_pair_seconds:
                raise ValueError('channel scan intervals do not match')
            mid = a+(b-a)/2
            if abs((mid-nominal).total_seconds()) > 3600:
                raise ValueError('scan midpoint more than one hour from folder timestamp')
            if mid in seen:
                raise ValueError('duplicate actual scan midpoint')
            seen.add(mid)
            scenes.append(Scene(folder.name,str(ir.resolve()),str(wv.resolve()),mid.isoformat(),(mid-nominal).total_seconds()/60))
        except (ValueError, OSError) as exc:
            issues.append({'scene':folder.name,'reason':str(exc)})
    return sorted(scenes,key=lambda s:s.midpoint), issues


@lru_cache(maxsize=8)
def transformer(crs, inverse=False):
    return Transformer.from_crs(crs if inverse else crs.geodetic_crs, crs.geodetic_crs if inverse else crs, always_xy=True)


@dataclass
class Grid:
    x: np.ndarray  # projected metres at pixel centres
    y: np.ndarray
    crs: CRS
    h: float

    @property
    def dx(self): return float(self.x[1]-self.x[0])
    @property
    def dy(self): return float(self.y[1]-self.y[0])
    @property
    def shape(self): return (len(self.y),len(self.x))

    def pixel(self, lon, lat):
        tr = transformer(self.crs)
        x,y = tr.transform(lon,lat)
        # Pixel-edge coordinates: first centre is (0.5, 0.5).
        return (np.asarray(x)-self.x[0])/self.dx+.5,(np.asarray(y)-self.y[0])/self.dy+.5

    def geographic(self, col, row):
        tr = transformer(self.crs,True)
        return tr.transform(self.x[0]+(np.asarray(col)-.5)*self.dx,self.y[0]+(np.asarray(row)-.5)*self.dy)

    def metadata(self):
        return {'crs_wkt':self.crs.to_wkt(),'height':self.h,'shape':list(self.shape),'x0_m':float(self.x[0]),'y0_m':float(self.y[0]),'dx_m':self.dx,'dy_m':self.dy,'coordinate_convention':'pixel edges; first pixel centre is 0.5,0.5'}


def get_grid(ds):
    attrs = ds['goes_imager_projection'].attrs
    h = float(attrs['perspective_point_height'])
    crs = CRS.from_cf(attrs)
    if attrs.get('sweep_angle_axis') != 'x' or attrs.get('grid_mapping_name') != 'geostationary':
        raise ValueError('not a GOES geostationary fixed grid')
    axes = []
    for name in ('x','y'):
        var = ds[name]
        if var.attrs.get('units') not in ('rad','radian','radians'):
            raise ValueError(f'{name} must use radians')
        axis = np.asarray(var.values,dtype=np.float64)*h
        dif = np.diff(axis)
        if len(axis)<2 or not np.isfinite(axis).all() or not np.allclose(dif,dif[0],rtol=.002,atol=.1) or dif[0]==0:
            raise ValueError(f'nonuniform {name} grid')
        axes.append(axis)
    if axes[0][1] <= axes[0][0] or axes[1][1] >= axes[1][0]:
        raise ValueError('unexpected GOES grid orientation')
    return Grid(*axes,crs,h)


def same_grid(a,b):
    return a.shape==b.shape and a.crs.equals(b.crs) and np.allclose(a.x,b.x,rtol=0,atol=.25) and np.allclose(a.y,b.y,rtol=0,atol=.25)


def read_band(path, band, expected=None):
    with xr.open_dataset(path,mask_and_scale=True,decode_times=False,cache=False) as ds:
        grid=get_grid(ds)
        if expected is not None and not same_grid(grid,expected):
            raise ValueError('channel/scene grid differs from reference grid')
        if 'band_id' in ds and int(ds['band_id'].values.item())!=band:
            raise ValueError('band_id disagrees with filename')
        cmi=ds['CMI']; dqf=ds['DQF']
        if cmi.attrs.get('units') not in ('K','kelvin','Kelvin'):
            raise ValueError('CMI is not brightness temperature in Kelvin')
        if set(cmi.dims)!={'y','x'} or set(dqf.dims)!={'y','x'}:
            raise ValueError('unexpected CMI/DQF dimensions')
        arr=np.array(cmi.transpose('y','x').values,dtype=np.float32,copy=True)
        quality=np.asarray(dqf.transpose('y','x').values)
        valid=np.isfinite(arr)&(arr>=100)&(arr<=400)&(quality==0)
        arr[~valid]=np.nan
        return arr,grid


def load_tracks(path, start, end):
    df=pd.read_csv(path,dtype=str,keep_default_na=False,low_memory=False)
    df.columns=df.columns.str.strip().str.upper()
    required={'SID','ISO_TIME','LAT','LON','NATURE'}
    if not required.issubset(df.columns):
        raise ValueError(f'IBTrACS missing columns: {sorted(required-set(df.columns))}')
    df=df[list(sorted(required))].copy()
    for c in df.columns: df[c]=df[c].str.strip()
    # NOAA CSV includes a units row; reject other invalid timestamp rows with a SID.
    df=df[df.SID!=''].copy()
    df['time']=pd.to_datetime(df.ISO_TIME,utc=True,errors='coerce',format='mixed') if int(pd.__version__.split('.')[0])>=2 else pd.to_datetime(df.ISO_TIME,utc=True,errors='coerce')
    if df.time.isna().any(): raise ValueError('invalid IBTrACS timestamp with nonempty SID')
    df['lat']=pd.to_numeric(df.LAT,errors='coerce'); df['lon']=pd.to_numeric(df.LON,errors='coerce')
    df.loc[~df.lat.between(-90,90),'lat']=np.nan
    df.loc[~df.lon.between(-180,360),'lon']=np.nan
    df['lon']=(df.lon+180)%360-180
    if not set(range(start.year,end.year+1)).issubset(set(df.time.dt.year)):
        raise ValueError('IBTrACS has no observations in one or more scene years')
    relevant=df[(df.time>=start-timedelta(days=2))&(df.time<=end+timedelta(days=2))]
    df=df[df.SID.isin(relevant.SID)]
    tracks=[]
    for sid,g in df.groupby('SID',sort=True):
        g=g.drop_duplicates(subset=['time','lat','lon','NATURE']).sort_values('time')
        if g.time.duplicated().any(): raise ValueError(f'conflicting IBTrACS records for {sid}')
        tracks.append((sid,g.time.to_numpy(dtype='datetime64[ns]').astype('int64')//10**9,g.lat.to_numpy(),g.lon.to_numpy(),g.NATURE.to_numpy()))
    if not tracks: raise ValueError('no overlapping IBTrACS tracks')
    return tracks


def match_track(track, t, max_gap=21600, near=900):
    sid,times,lat,lon,nature=track
    # Outside a track lifetime: ignore unless within the endpoint tolerance.
    if t<times[0]-near or t>times[-1]+near: return None
    j=int(np.searchsorted(times,t))
    ids=[i for i in (j-1,j) if 0<=i<len(times)]
    nearest=min(ids,key=lambda i:abs(times[i]-t))
    if times[nearest]==t and np.isfinite([lat[nearest],lon[nearest]]).all():
        return {'sid':sid,'lat':float(lat[nearest]),'lon':float(lon[nearest]),'nature':str(nature[nearest]),'match':'exact','offset_s':0.0}
    if 0<j<len(times) and times[j]-times[j-1]<=max_gap and nature[j]==nature[j-1] and np.isfinite([lat[j],lon[j],lat[j-1],lon[j-1]]).all():
        f=(t-times[j-1])/(times[j]-times[j-1])
        az,_,dist=GEOD.inv(lon[j-1],lat[j-1],lon[j],lat[j])
        lo,la,_=GEOD.fwd(lon[j-1],lat[j-1],az,dist*f)
        return {'sid':sid,'lat':float(la),'lon':float(lo),'nature':str(nature[j]),'match':'interpolated','offset_s':float(abs(times[nearest]-t))}
    if abs(times[nearest]-t)<=near and np.isfinite([lat[nearest],lon[nearest]]).all():
        # A transition between target and non-target types is ambiguous, not a positive.
        state=str(nature[nearest])
        if 0<j<len(times) and (nature[j] in TARGET)!=(nature[j-1] in TARGET): state='AMBIGUOUS'
        return {'sid':sid,'lat':float(lat[nearest]),'lon':float(lon[nearest]),'nature':state,'match':'nearest','offset_s':float(abs(times[nearest]-t))}
    # Cannot safely establish presence/absence within the active track interval.
    return {'sid':sid,'unknown':True}


def box_for(grid, obs, radius_km):
    lon,lat=obs['lon'],obs['lat']
    x,y=map(float,grid.pixel(lon,lat))
    h,w=grid.shape
    if not np.isfinite([x,y]).all() or not (0<=x<w and 0<=y<h): return None
    az=np.arange(0,360,15,dtype=float)
    lons,lats,_=GEOD.fwd(np.full_like(az,lon),np.full_like(az,lat),az,np.full_like(az,radius_km*1000))
    xx,yy=grid.pixel(lons,lats)
    if not np.isfinite(xx).all() or not np.isfinite(yy).all():
        # Limb-crossing proxy: exclude surrounding tiles conservatively.
        return dict(obs,x=x,y=y,box=[max(0,x-radius_km*1000/abs(grid.dx)),max(0,y-radius_km*1000/abs(grid.dy)),min(w,x+radius_km*1000/abs(grid.dx)),min(h,y+radius_km*1000/abs(grid.dy))],nature='LIMB')
    return dict(obs,x=x,y=y,box=[float(min(xx.min(),x)),float(min(yy.min(),y)),float(max(xx.max(),x)),float(max(yy.max(),y))])


def origins(n,size,stride):
    if n<size: raise ValueError('tile larger than source grid')
    return sorted(set(range(0,n-size+1,stride))|{n-size})


def tile_grid(grid,size,stride):
    h,w=grid.shape
    # Fixed grid, independent of labels. Limit tile centres to tropical/subtropical
    # usable viewing region; still label any known centre within selected tiles.
    coords=[(x,y) for y in origins(h,size,stride) for x in origins(w,size,stride)]
    x,y=np.array(coords).T
    lon,lat=grid.geographic(x+size/2,y+size/2)
    sublon=float(grid.crs.to_cf()['longitude_of_projection_origin'])
    cosang=np.cos(np.deg2rad(lat))*np.cos(np.deg2rad(lon-sublon))
    return [(int(a),int(b)) for a,b,ok in zip(x,y,np.isfinite(lat)&(np.abs(lat)<=45)&(cosang>=.5)) if ok]


def intersects(box,x,y,size):
    return box[0]<x+size and box[2]>x and box[1]<y+size and box[3]>y


def outside_possible_view(track, t, grid, radius_km):
    """Conservative exclusion of unresolved distant tracks using a motion envelope.

    Assumes maximum translation of 150 km/h; without a nearby valid fix,
    retain uncertainty. This is not a proof of physical absence.
    """
    _, times, lat, lon, _ = track
    valid = np.flatnonzero(np.isfinite(lat) & np.isfinite(lon))
    if not len(valid):
        return False
    i = valid[np.argmin(abs(times[valid] - t))]
    hours = abs(times[i] - t) / 3600
    if hours > 24:
        return False
    sublon = grid.crs.to_cf()['longitude_of_projection_origin']
    _, _, distance = GEOD.inv(sublon, 0, lon[i], lat[i])
    # 10,000 km is deliberately larger than the geostationary visible radius.
    return distance / 1000 > 10000 + 150 * hours + radius_km


def plan_scenes(scenes, tracks, grid, tiles, size, radius):
    plans=[]; issues=[]
    for scene in scenes:
        t=datetime.fromisoformat(scene.midpoint).timestamp()
        obs=[o for tr in tracks if (o:=match_track(tr,t)) is not None
             and not (o.get('unknown') and outside_possible_view(tr,t,grid,radius))]
        # Missing positions could conceal a storm anywhere in the domain.
        unknown=[o['sid'] for o in obs if o.get('unknown')]
        if unknown:
            issues.append({'scene':scene.id,'reason':'unresolved active track','sids':unknown}); continue
        objects=[]
        for o in obs:
            p=box_for(grid,o,radius)
            if p is not None and any(intersects(p['box'],x,y,size) for x,y in tiles): objects.append(p)
        # All matched visible track IDs, including ignored phases, participate in split isolation.
        eligible = [labels for x,y in tiles if (labels := label_for_tile(objects,x,y,size)) is not None]
        positive_tiles = sum(bool(labels) for labels in eligible)
        plans.append({'scene':asdict(scene),'objects':objects,'ids':sorted({o['sid'] for o in objects}),'positive_ids':sorted({o['sid'] for o in objects if o['nature'] in TARGET}), 'estimated_positive_tiles':positive_tiles, 'estimated_negative_tiles':len(eligible)-positive_tiles})
    return plans,issues


def allocate(weights, rng):
    """Weighted independent group assignment, approximate 80/10/10."""
    keys=list(weights); rng.shuffle(keys)
    # Randomized order within a broad weight scale keeps large storms manageable.
    keys.sort(key=lambda k: -(weights[k] * rng.uniform(.5,1.5)))
    total=sum(weights.values()); target=RATIOS*total; count=np.zeros(3); out={}
    for k in keys:
        # Greatest remaining fraction of target, with randomized ties.
        idx=max(range(3),key=lambda i:((target[i]-count[i])/max(target[i],1),rng.random()))
        out[k]=SPLITS[idx]; count[idx]+=weights[k]
    return out


def split_plans(plans, seed, purge_hours=24):
    rng=random.Random(seed)
    weights=Counter()
    for p in plans:
        for sid in p['ids']:
            weights[sid] += max(1, p.get('estimated_positive_tiles',1))
    positives=[p for p in plans if p['positive_ids']]
    if len({s for p in positives for s in p['positive_ids']})<3:
        raise ValueError('need at least three eligible storm IDs for disjoint train/val/test')
    best=None
    for _ in range(192):
        mapping=allocate(weights,rng)
        counts=Counter(); kept=0
        for p in positives:
            ss={mapping[s] for s in p['ids']}
            if len(ss)==1:
                counts[next(iter(ss))]+=p.get('estimated_positive_tiles',1); kept+=1
        if any(counts[s]==0 for s in SPLITS): continue
        error=sum(abs(counts[s]/sum(counts.values())-RATIOS[i]) for i,s in enumerate(SPLITS))
        score=error+1.5*(1-kept/len(positives))
        if best is None or score<best[0]: best=(score,mapping)
    if best is None: raise ValueError('cannot split positive scenes without storm leakage; inspect overlapping storms')
    mapping=best[1]; issues=[]; assigned=[]; no_track=[]; anchors=[]
    for p in plans:
        t=datetime.fromisoformat(p['scene']['midpoint']).timestamp()
        ss={mapping[s] for s in p['ids']}
        if len(ss)>1:
            issues.append({'scene':p['scene']['id'],'reason':'mixed storm splits','sids':p['ids']}); anchors.append((t,None)); continue
        if ss:
            p['split']=next(iter(ss)); assigned.append(p); anchors.append((t,p['split']))
        else: no_track.append(p)
    block_weights=Counter(int(datetime.fromisoformat(p['scene']['midpoint']).timestamp()//(7*86400)) for p in no_track)
    blocks=allocate(block_weights,rng) if block_weights else {}
    for p in no_track:
        t=datetime.fromisoformat(p['scene']['midpoint']).timestamp()
        near={s for at,s in anchors if abs(at-t)<=purge_hours*3600}
        if None in near or len(near)>1:
            issues.append({'scene':p['scene']['id'],'reason':'background near mixed split'}); continue
        p['split']=next(iter(near)) if near else blocks[int(t//(7*86400))]
        assigned.append(p)
    # Purge background scenes at split boundaries to reduce weather-pattern leakage.
    timeline=sorted((datetime.fromisoformat(p['scene']['midpoint']).timestamp(),p['scene']['id'],p['split']) for p in assigned)
    times=np.array([v[0] for v in timeline]); final=[]
    for p in assigned:
        t=datetime.fromisoformat(p['scene']['midpoint']).timestamp()
        lo=int(np.searchsorted(times,t-purge_hours*3600,side='left'))
        hi=int(np.searchsorted(times,t+purge_hours*3600,side='right'))
        if not p['ids'] and any(v[2]!=p['split'] for v in timeline[lo:hi]):
            issues.append({'scene':p['scene']['id'],'reason':'background temporal split buffer'}); continue
        final.append(p)
    return sorted(final,key=lambda p:p['scene']['id']),mapping,issues


def label_for_tile(objects,x,y,size):
    labels=[]
    for o in objects:
        box=o['box']
        if not intersects(box,x,y,size): continue
        if o['nature'] not in TARGET or not (x<=o['x']<x+size and y<=o['y']<y+size):
            return None  # uncertain or truncated centre: never label as background
        x1,y1,x2,y2=max(x,box[0]),max(y,box[1]),min(x+size,box[2]),min(y+size,box[3])
        if (x2-x1)*(y2-y1)<.75*(box[2]-box[0])*(box[3]-box[1]): return None
        vals=[((x1+x2)/2-x)/size,((y1+y2)/2-y)/size,(x2-x1)/size,(y2-y1)/size,(o['x']-x)/size,(o['y']-y)/size]
        labels.append({'sid':o['sid'],'values':vals,'lat':o['lat'],'lon':o['lon'],'match':o['match'],'offset_s':o['offset_s'],'nature':o['nature']})
    return labels


def encode(ir,wv):
    valid=np.isfinite(ir)&np.isfinite(wv)
    def scale(a,lo,hi):
        # Fixed physical bounds; 0 reserved for invalid pixels.
        return np.where(valid,1+254*np.clip((np.nan_to_num(a,nan=lo)-lo)/(hi-lo),0,1),0).round().astype(np.uint8)
    return np.stack((scale(ir,180,330),scale(wv,180,280),scale(ir-wv,-30,70)),axis=-1),valid


def atomic_image(path, image):
    temp=path.with_suffix('.tmp')
    image.save(temp,format='PNG',compress_level=3); os.replace(temp,path)


def atomic_text(path, content):
    temp=path.with_suffix('.tmp'); temp.write_text(content); os.replace(temp,path)


_WORK=None

def init_worker(grid,tiles,stage,config):
    global _WORK
    _WORK=(grid,tiles,Path(stage),config)


def process_scene(plan):
    grid,tiles,stage,cfg=_WORK; scene=plan['scene']; sid=scene['id']; split=plan['split']; size=cfg['tile_size']
    journal=stage/'records'/f'{sid}.json'
    if journal.exists():
        saved=json.loads(journal.read_text())
        if all((stage/r['image']).is_file() and (stage/r['label']).is_file() for r in saved['samples']): return sid,None
        journal.unlink()
    image_dir=stage/'images'/split; label_dir=stage/'labels'/split
    for directory in (image_dir,label_dir):
        for p in directory.glob(f'{sid}_*'): p.unlink()
    try:
        ir,_=read_band(scene['ir'],13,grid); wv,_=read_band(scene['wv'],8,grid)
        rgb,valid=encode(ir,wv); del ir,wv
        candidates=[]; rejects=Counter(); expected=set(plan['positive_ids'])
        for x,y in tiles:
            labels=label_for_tile(plan['objects'],x,y,size)
            if labels is None: rejects['ambiguous_or_truncated']+=1; continue
            vm=valid[y:y+size,x:x+size]
            fraction=float(vm.mean())
            if fraction<cfg['min_valid']: rejects['low_valid_fraction']+=1; continue
            if any(not vm[min(size-1,int(l['values'][5]*size)),min(size-1,int(l['values'][4]*size))] for l in labels):
                rejects['invalid_centre']+=1; continue
            candidates.append((x,y,labels,fraction))
        pos=[v for v in candidates if v[2]]; neg=[v for v in candidates if not v[2]]
        rr=random.Random(f"{cfg['seed']}:{sid}"); rr.shuffle(neg)
        neg=neg[:max(cfg['background_per_scene'],cfg['negative_ratio']*len(pos))]
        rows=[]
        for x,y,labels,fraction in pos+neg:
            name=f'{sid}_x{x:04d}_y{y:04d}'
            image=image_dir/f'{name}.png'; label=label_dir/f'{name}.txt'
            tile=Image.fromarray(rgb[y:y+size,x:x+size])
            if cfg['image_size']!=size: tile=tile.resize((cfg['image_size'],cfg['image_size']),Image.Resampling.BILINEAR)
            atomic_image(image,tile)
            atomic_text(label,''.join('0 '+' '.join(f'{a:.8f}' for a in l['values'])+' 2\n' for l in labels))
            rows.append({'scene':sid,'split':split,'image':str(image.relative_to(stage)),'label':str(label.relative_to(stage)),'x0':x,'y0':y,'tile_size':size,'valid_fraction':fraction,'objects':labels,'midpoint':scene['midpoint']})
        if pos:
            # One labelled preview per positive scene, available during processing.
            r=rows[0]
            with Image.open(stage/r['image']) as im:
                preview=im.copy()
            draw=ImageDraw.Draw(preview); w,h=preview.size
            for o in r['objects']:
                cx,cy,bw,bh,kx,ky=o['values']
                draw.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline='yellow',width=2)
                draw.line((kx*w-6,ky*h,kx*w+6,ky*h),fill='red',width=2)
                draw.line((kx*w,ky*h-6,kx*w,ky*h+6),fill='red',width=2)
            (stage/'qa_progress').mkdir(exist_ok=True)
            atomic_image(stage/'qa_progress'/f'{split}_{sid}.png',preview)
        covered={l['sid'] for r in rows for l in r['objects']}
        dump(journal,{'scene':sid,'split':split,'samples':rows,'rejected_tiles':dict(rejects),'uncovered_sids':sorted(expected-covered)})
        return sid,None
    except Exception as exc:
        return sid,f'{type(exc).__name__}: {exc}'


def balance_images(stage, plans, config):
    """Keep all positives; trim only negatives to attain exact 8:1:1 if feasible.

    Persist selection before removing anything, so retries are idempotent.
    """
    selection_path = stage / 'image_balance.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
    else:
        positives = Counter(); negatives = defaultdict(list)
        for p in plans:
            record = json.loads((stage/'records'/f"{p['scene']['id']}.json").read_text())
            for r in record['samples']:
                if r['objects']:
                    positives[r['split']] += 1
                else:
                    negatives[r['split']].append(r['image'])
        factors = dict(zip(SPLITS,(8,1,1)))
        # At least one negative per split and every positive retained.
        low = max((positives[s]+1+factors[s]-1)//factors[s] for s in SPLITS)
        high = min((positives[s]+len(negatives[s]))//factors[s] for s in SPLITS)
        if high < low:
            if not config['allow_approximate_split']:
                raise ValueError(f'Exact 80/10/10 infeasible without discarding positives: positives={dict(positives)}, negatives={ {s:len(negatives[s]) for s in SPLITS} }. Use --allow-approximate-split explicitly or revise sampling/split settings.')
            selection = {'exact':False,'drop_images':[], 'reason':'insufficient negative tiles for exact split'}
        else:
            drop=[]
            for split in SPLITS:
                choices=sorted(negatives[split])
                random.Random(f"{config['seed']}:balance:{split}").shuffle(choices)
                keep=high*factors[split]-positives[split]
                drop.extend(choices[keep:])
            selection={'exact':True,'drop_images':sorted(drop),'target_counts':{s:high*factors[s] for s in SPLITS}}
        dump(selection_path,selection)
    drop=set(selection['drop_images'])
    for p in plans:
        journal=stage/'records'/f"{p['scene']['id']}.json"
        record=json.loads(journal.read_text())
        removed=[r for r in record['samples'] if r['image'] in drop]
        # Delete first; process_scene can regenerate an interrupted record on restart.
        for r in removed:
            (stage/r['image']).unlink(missing_ok=True)
            (stage/r['label']).unlink(missing_ok=True)
        if removed:
            record['samples']=[r for r in record['samples'] if r['image'] not in drop]
            dump(journal,record)
    return selection


def write_previews(stage):
    """Select deterministic positive and negative QA images, outside training paths."""
    selected=defaultdict(list)
    with (stage/'samples.jsonl').open() as f:
        for line in f:
            r=json.loads(line); key=(r['split'],bool(r['objects']))
            # Spread examples across distinct scenes, not adjacent overlapping tiles.
            if len(selected[key])<4 and all(v['scene']!=r['scene'] for v in selected[key]):
                selected[key].append(r)
    folder=stage/'qa'; folder.mkdir(exist_ok=True)
    for (split,positive),rows in selected.items():
        for r in rows:
            with Image.open(stage/r['image']) as image:
                image=image.copy(); draw=ImageDraw.Draw(image); w,h=image.size
                for o in r['objects']:
                    cx,cy,bw,bh,kx,ky=o['values']; kx*=w; ky*=h
                    draw.rectangle(((cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h),outline='yellow',width=2)
                    draw.line((kx-6,ky,kx+6,ky),fill='red',width=2)
                    draw.line((kx,ky-6,kx,ky+6),fill='red',width=2)
                atomic_image(folder/f"{split}_{'positive' if positive else 'negative'}_{Path(r['image']).name}",image)


def validate(stage,plans,config):
    counts={s:Counter() for s in SPLITS}; sid_splits=defaultdict(set); scene_splits={}; paths=set(); labelpaths=set(); missed=[]; rejects=Counter(); uncovered=[]
    with (stage/'samples.jsonl').open('w') as out:
        for p in plans:
            scene=p['scene']['id']; split=p['split']; journal=stage/'records'/f'{scene}.json'
            if not journal.exists(): raise ValueError(f'missing scene record: {scene}')
            record=json.loads(journal.read_text()); rejects.update(record['rejected_tiles'])
            if record['uncovered_sids']: uncovered.append({'scene':scene,'sids':record['uncovered_sids']})
            if not record['samples']: missed.append(scene)
            for sid in p['ids']: sid_splits[sid].add(split)
            for r in record['samples']:
                if r['split']!=split or r['scene']!=scene or scene_splits.setdefault(scene,split)!=split: raise ValueError('scene split leakage')
                im,lab=r['image'],r['label']
                if im in paths or lab in labelpaths: raise ValueError('duplicate output sample')
                paths.add(im); labelpaths.add(lab)
                with Image.open(stage/im) as image:
                    if image.mode!='RGB' or image.size!=(config['image_size'],)*2: raise ValueError(f'invalid image: {im}')
                    image.verify()
                lines=(stage/lab).read_text().splitlines()
                if len(lines)!=len(r['objects']): raise ValueError(f'annotation count mismatch: {lab}')
                for line,obj in zip(lines,r['objects']):
                    a=np.array([float(x) for x in line.split()])
                    if len(a)!=8 or not np.isfinite(a).all() or a[0]!=0 or a[7]!=2 or not ((a[1:7]>=0)&(a[1:7]<=1)).all() or min(a[3:5])<=0: raise ValueError(f'invalid pose label: {lab}')
                    if not np.allclose(a[1:7],obj['values'],rtol=0,atol=1e-7): raise ValueError('label and manifest mismatch')
                    if abs(a[5]-a[1])>a[3]/2+1e-6 or abs(a[6]-a[2])>a[4]/2+1e-6: raise ValueError('keypoint outside box')
                counts[split].update({'images':1,'positive':int(bool(lines)),'negative':int(not lines),'objects':len(lines)})
                out.write(json.dumps(r,allow_nan=False)+'\n')
    if any(len(v)>1 for v in sid_splits.values()): raise ValueError('storm ID leakage')
    if paths!={str(p.relative_to(stage)) for p in (stage/'images').rglob('*.png')} or labelpaths!={str(p.relative_to(stage)) for p in (stage/'labels').rglob('*.txt')}: raise ValueError('orphan image or label')
    if any(not counts[s]['positive'] or not counts[s]['negative'] for s in SPLITS): raise ValueError(f'each split needs positive and negative images: {counts}')
    total=sum(counts[s]['images'] for s in SPLITS)
    fractions={s:counts[s]['images']/total for s in SPLITS}
    exact=all(counts[s]['images']*10==total*f for s,f in zip(SPLITS,(8,1,1)))
    expected=sum(len(p['positive_ids']) for p in plans)
    missing=sum(len(r['sids']) for r in uncovered)
    coverage=(expected-missing)/max(expected,1)
    if not exact and not config['allow_approximate_split']:
        raise ValueError(f'final image split is not exact: {fractions}')
    if coverage < config['min_storm_coverage']:
        raise ValueError(f'planned storm observation coverage {coverage:.2%} below threshold {config["min_storm_coverage"]:.2%}')
    return {'image_split_fractions':fractions,'exact_image_split':exact,'planned_storm_observation_coverage':coverage,'planned_storm_observations':expected,'covered_storm_observations':expected-missing,'splits':{s:dict(counts[s]) for s in SPLITS},'unique_storms_per_split':{s:sum(s in v for v in sid_splits.values()) for s in SPLITS},'scene_counts':dict(Counter(p['split'] for p in plans)),'storm_leakage':False,'scene_leakage':False,'rejected_tiles':dict(rejects),'scenes_without_samples':missed,'uncovered_storm_observations':uncovered}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goes',type=Path,default=Path('./../data/goes16'))
    parser.add_argument('--ibtracs',type=Path,default=Path('./../data/ibtracs.csv'))
    parser.add_argument('--output',type=Path,default=Path('./../data/processed/detection_pose_v2'))
    parser.add_argument('--tile-size',type=int,default=1024)
    parser.add_argument('--stride',type=int,default=512)
    parser.add_argument('--image-size',type=int,default=640)
    parser.add_argument('--radius-km',type=float,default=300)
    parser.add_argument('--min-valid',type=float,default=.7)
    parser.add_argument('--background-per-scene',type=int,default=1)
    parser.add_argument('--negative-ratio',type=int,default=3)
    parser.add_argument('--seed',type=int)
    parser.add_argument('--workers',type=int,default=1)
    parser.add_argument('--min-storm-coverage',type=float,default=.90)
    parser.add_argument('--allow-approximate-split',action='store_true')
    parser.add_argument('--plan-only',action='store_true',help='write inventory/split plan without generating images')
    args=parser.parse_args(argv)
    if not (0<args.min_storm_coverage<=1 and 0<args.stride<=args.tile_size and args.image_size>0 and args.image_size%32==0 and 0<args.min_valid<=1 and args.radius_km>0 and args.workers>0 and args.background_per_scene>=1 and args.negative_ratio>=0): parser.error('invalid preprocessing parameters')
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    root=args.goes.resolve(); csvpath=args.ibtracs.resolve(); output=args.output.resolve(); stage=output.with_name(output.name+'.work')
    if output.exists(): raise FileExistsError(f'output already exists: {output}; use a new output path')
    if output==root or root in output.parents or output in root.parents or output==csvpath or output in csvpath.parents: raise ValueError('output must not contain or be inside raw inputs')
    if not root.is_dir() or not csvpath.is_file(): raise FileNotFoundError('missing GOES directory or IBTrACS CSV')
    scenes,discovery_issues=discover(root,60)
    if not scenes: raise ValueError('no valid GOES scene pairs')
    grid=None
    for s in scenes:
        try:
            with xr.open_dataset(s.ir,decode_times=False) as a,xr.open_dataset(s.wv,decode_times=False) as b:
                candidate=get_grid(a)
                if not same_grid(candidate,get_grid(b)): raise ValueError('reference channels differ')
                grid=candidate; break
        except Exception as exc: LOG.warning('reference candidate %s rejected: %s',s.id,exc)
    if grid is None: raise ValueError('no valid reference projection')
    tiles=tile_grid(grid,args.tile_size,args.stride)
    if not tiles: raise ValueError('no usable fixed-grid tiles')
    tracks=load_tracks(csvpath,datetime.fromisoformat(scenes[0].midpoint),datetime.fromisoformat(scenes[-1].midpoint))
    plans,match_issues=plan_scenes(scenes,tracks,grid,tiles,args.tile_size,args.radius_km)
    before_split_observations=sum(len(p['positive_ids']) for p in plans)
    plans,mapping,split_issues=split_plans(plans,args.seed)
    cfg={k:v for k,v in vars(args).items() if k not in ('goes','ibtracs','output','workers','plan_only')}
    fingerprint=hashlib.sha256()
    fingerprint.update(Path(__file__).read_bytes()); fingerprint.update(csvpath.read_bytes())
    fingerprint.update(json.dumps(cfg,sort_keys=True).encode())
    for s in scenes:
        for name in (s.ir,s.wv):
            st=Path(name).stat(); fingerprint.update(f'{name}:{st.st_size}:{st.st_mtime_ns}\n'.encode())
    signature=fingerprint.hexdigest()
    stage.mkdir(parents=True,exist_ok=True)
    # Prevent concurrent writers. Stale lock after abrupt termination must be removed manually.
    lock=stage/'BUILD.lock'
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY); os.write(fd,str(os.getpid()).encode()); os.close(fd)
    try:
        build=stage/'build.json'
        if build.exists() and json.loads(build.read_text())['signature']!=signature: raise ValueError('staging inputs/config changed; use new output or remove old staging after review')
        dump(build,{'signature':signature,'config':cfg,'goes':str(root),'ibtracs':str(csvpath),'grid':grid.metadata(),'tile_origins':tiles,'target_natures':sorted(TARGET),'box_definition':'fixed geodesic radius proxy, not annotated cloud extent','rgb_channels':['C13 180–330 K','C08 180–280 K','C13 minus C08 -30–70 K']})
        dump(stage/'scene_plan.json',plans); dump(stage/'storm_splits.json',mapping)
        dump(stage/'inventory.json',{'discovered_pairs':len(scenes),'planned_scenes':len(plans),'fixed_tiles_per_scene':len(tiles),'scene_split_counts':dict(Counter(p['split'] for p in plans)),'positive_scene_split_counts':dict(Counter(p['split'] for p in plans if p['positive_ids'])),'estimated_positive_tiles_per_split':{s:sum(p['estimated_positive_tiles'] for p in plans if p['split']==s) for s in SPLITS},'positive_observations_before_split':before_split_observations,'positive_observations_after_split':sum(len(p['positive_ids']) for p in plans),'discovery_issues':discovery_issues,'matching_issues':match_issues,'split_issues':split_issues})
        LOG.info('planned positive scenes: %s',dict(Counter(p['split'] for p in plans if p['positive_ids'])))
        LOG.info('planned %d scenes of %d pairs; %d fixed tiles per scene',len(plans),len(scenes),len(tiles))
        if args.plan_only: return 0
        for split in SPLITS:
            for kind in ('images','labels'): (stage/kind/split).mkdir(parents=True,exist_ok=True)
        (stage/'records').mkdir(exist_ok=True)
        failures=[]
        (stage/'qa_progress').mkdir(exist_ok=True)
        plans.sort(key=lambda p:(not bool(p['positive_ids']),p['scene']['id']))
        def consume(results):
            for i,(sid,error) in enumerate(results,1):
                if error: failures.append({'scene':sid,'error':error}); LOG.error('%s: %s',sid,error)
                if i%25==0 or i==len(plans): LOG.info('processed %d/%d scenes',i,len(plans))
        if args.workers==1:
            init_worker(grid,tiles,stage,cfg); consume(map(process_scene,plans))
        else:
            with ProcessPoolExecutor(max_workers=args.workers,mp_context=mp.get_context('spawn'),initializer=init_worker,initargs=(grid,tiles,stage,cfg)) as pool:
                consume(pool.map(process_scene,plans,chunksize=1))
        dump(stage/'processing_failures.json',failures)
        if failures: raise RuntimeError(f'{len(failures)} scene failures; staging preserved, dataset not published')
        balance_images(stage,plans,cfg)
        summary=validate(stage,plans,cfg)
        summary['positive_observations_before_split']=before_split_observations
        summary['coverage_relative_to_pre_split_plan']=summary['covered_storm_observations']/max(before_split_observations,1)
        dump(stage/'summary.json',summary)
        write_previews(stage)
        # JSON-quoted absolute path is valid YAML; relocate by editing this path.
        atomic_text(stage/'data.yaml',f'path: {json.dumps(str(output))}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: cyclone\nkpt_shape: [1, 3]\nflip_idx: [0]\n')
        LOG.info('validation passed: %s',summary['splits'])
        lock.unlink(); stage.rename(output)
        return 0
    finally:
        if lock.exists(): lock.unlink()


if __name__=='__main__':
    raise SystemExit(main())
