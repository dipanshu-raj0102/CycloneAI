#!/usr/bin/env python3
"""Read-only CycloneAI v6 HDF5 readiness audit. Exit 0: checks pass; 1: errors; 2: warnings.
Requires numpy, pandas, h5py, matplotlib. No raw satellite decoding required.
"""
import argparse
import hashlib
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
import h5py

SPLITS = ('train', 'val', 'test')
CLASSES = ('TD', 'TS', 'C1', 'C2', 'C3', 'C4', 'C5')
STRINGS = 'sample_id sid timestamp nominal_timestamp sensor sensor_id basin ibtracs_timestamp source_files nature wind_source'.split()
NUMBERS = 'lat lon wind_kts class_id time_offset_min viewing_zenith_deg flipped pressure_hpa valid_fraction inner_valid_fraction observation_weight'.split()

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', type=Path, default=Path('data/processed/intensity_kelvin_final/current'))
    p.add_argument('--output-dir', type=Path, default=Path('reports/dataset_readiness'))
    p.add_argument('--ibtracs', type=Path, default=Path('data/raw/ibtracs/ibtracs.ALL.list.v04r01.csv'))
    p.add_argument('--previews-per-class', type=int, default=3)
    p.add_argument('--min-class-storms', type=int, default=5, help='Advisory threshold, not a statistical guarantee')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    if a.previews_per_class < 0 or a.min_class_storms < 1: p.error('Invalid preview or storm count')
    root = a.data_dir.resolve(); out = a.output_dir.resolve()
    if out == root or root in out.parents: p.error('Report directory must be outside dataset generation')
    out.mkdir(parents=True, exist_ok=True)
    issues=[]; frames=[]; previews=[]; pixel_rows=[]; hashes={}; duplicates=[]
    def issue(level, check, detail):
        issues.append(dict(level=level, check=check, detail=str(detail)))
        print(f'[{level}] {check}: {detail}', flush=True)
    def read_json(name):
        try: return json.loads((root/name).read_text())
        except Exception as e: issue('ERROR',name,e); return {}
    summary=read_json('processing_summary.json'); norm=read_json('normalization.json')
    if not (root/'READY').is_file(): issue('ERROR','publication','Missing READY marker')
    params=summary.get('parameters',{}); size=int(params.get('size',448)); minvalid=float(params.get('min_valid_fraction',.95)); zenith=float(params.get('max_zenith',70)); tolerance=float(params.get('tolerance_min',5))
    # v6 default; configuration file may contain a user override.
    config=read_json('configuration.json'); inner_min=float(config.get('min_inner_valid_fraction',.99))
    rng=np.random.default_rng(a.seed)
    train_n=0; train_sum=0.; train_sq=0.
    for split in SPLITS:
        print(f'\nScanning {split}.h5',flush=True)
        try:
            with h5py.File(root/f'{split}.h5','r') as h:
                missing=set(STRINGS+NUMBERS+['images','valid_mask'])-set(h.keys())
                if missing: raise ValueError(f'Missing datasets {sorted(missing)}')
                x=h['images']; m=h['valid_mask']; n=len(x)
                if not n or x.shape!=(n,size,size) or m.shape!=x.shape or x.dtype!=np.uint16 or m.dtype!=np.uint8: raise ValueError('Invalid image/mask shape or dtype')
                for k in STRINGS+NUMBERS:
                    if h[k].shape!=(n,): raise ValueError(f'Invalid {k} shape')
                for k,v in {'temperature_unit':'Kelvin','temperature_scale':100.,'temperature_offset_K':0.,'invalid_pixel_value':0,'wind_column':'USA_WIND','band':13,'image_size':size,'wind_avg_period_minutes':1}.items():
                    if h.attrs.get(k)!=v: issue('ERROR','HDF5 attributes',f'{split}: {k} expected {v}, got {h.attrs.get(k)}')
                d=pd.DataFrame({k:h[k].asstr()[:] for k in STRINGS}|{k:h[k][:] for k in NUMBERS}); d['split']=split; d['row']=np.arange(n); frames.append(d)
                counts=Counter(); examples={}
                def bad(key,i):
                    counts[key]+=1; examples.setdefault(key,[])
                    if len(examples[key])<5: examples[key].append(int(i))
                wind=d.wind_kts.to_numpy(); cls=d.class_id.to_numpy()
                expected=np.searchsorted([34,64,83,96,113,137],wind,side='right')
                if np.any(~np.isfinite(wind)|(wind<0)|(wind>250)|(cls!=expected)): issue('ERROR','wind labels',f'{split}: invalid wind or class mapping')
                for k,lo,hi in [('lat',-90,90),('lon',-180,180),('viewing_zenith_deg',0,zenith),('valid_fraction',minvalid,1),('inner_valid_fraction',inner_min,1),('flipped',0,1)]:
                    v=d[k].to_numpy(); invalid=~np.isfinite(v)|(v<lo-1e-6)|(v>hi+1e-6)
                    if invalid.any(): issue('ERROR','metadata bounds',f'{split}: {k}: {invalid.sum()} invalid')
                if not np.isfinite(d.observation_weight).all() or (d.observation_weight<=0).any(): issue('ERROR','weights',f'{split}: nonpositive/nonfinite weights')
                for k in ['sample_id','sid','sensor_id','source_files']:
                    if d[k].str.strip().eq('').any(): issue('ERROR','empty metadata',f'{split}: {k}')
                dates={k:pd.to_datetime(d[k],format='ISO8601',utc=True,errors='coerce') for k in ['timestamp','nominal_timestamp','ibtracs_timestamp']}
                for field, values in dates.items():
                    if values.isna().any():
                        bad = values.isna()
                        issue('ERROR','timestamps',f'{split}/{field}: {bad.sum()} unparseable; examples: {d.loc[bad, ["sample_id", field]].head(5).to_dict("records")}')
                delta=(dates['nominal_timestamp']-dates['ibtracs_timestamp']).dt.total_seconds()/60
                actual=(dates['timestamp']-dates['ibtracs_timestamp']).dt.total_seconds()/60
                if (delta.abs()>tolerance+1e-6).any(): issue('ERROR','nominal time matching',split)
                if not np.allclose(actual,d.time_offset_min,atol=1e-4): issue('ERROR','actual time offset metadata',split)
                t=dates['ibtracs_timestamp'];
                if ((t.dt.hour%3!=0)|(t.dt.minute!=0)|(t.dt.second!=0)).any(): issue('ERROR','synoptic fixes',split)
                yy,xx=np.ogrid[:size,:size]; inner=(yy-(size-1)/2)**2+(xx-(size-1)/2)**2<=(size/4)**2
                chosen=set()
                for _,g in d.groupby(['sensor_id','class_id']):
                    # Prefer different storms in previews.
                    pool=g.sample(frac=1,random_state=a.seed).drop_duplicates('sid').index.to_numpy()
                    chosen.update(rng.choice(pool,min(len(pool),a.previews_per_class),replace=False).tolist())
                for i in range(n):
                    image=x[i]; mask=m[i]; valid=mask==1
                    if np.any((mask!=0)&(mask!=1)): bad('nonbinary_mask',i)
                    if np.any(image[~valid]!=0): bad('invalid_sentinel',i)
                    v=image[valid].astype(np.float64)/100
                    vf=float(valid.mean()); iv=float(valid[inner].mean())
                    if vf<minvalid or iv<inner_min: bad('pixel_coverage',i)
                    if abs(vf-d.valid_fraction[i])>1e-6 or abs(iv-d.inner_valid_fraction[i])>1e-6: bad('mask_metadata_mismatch',i)
                    if not len(v) or np.any((v<150)|(v>350)): bad('temperature_range',i)
                    if len(v) and np.ptp(v)==0: bad('constant_image',i)
                    if len(v):
                        pixel_rows.append(dict(split=split,sample_id=d.sample_id[i],sensor_id=d.sensor_id[i],class_id=int(cls[i]),sid=d.sid[i],valid_fraction=vf,inner_valid_fraction=iv,mean_kelvin=float(v.mean()),std_kelvin=float(v.std()),min_kelvin=float(v.min()),max_kelvin=float(v.max())))
                        if split=='train': train_n+=len(v); train_sum+=v.sum(); train_sq+=np.square(v).sum()
                    digest=hashlib.sha256(image.tobytes()+mask.tobytes()).hexdigest()
                    prior=hashes.setdefault(digest,(split,d.sample_id[i],d.sid[i]))
                    if prior[1]!=d.sample_id[i]: duplicates.append(dict(split=split,sample_id=d.sample_id[i],sid=d.sid[i],other_split=prior[0],other_sample_id=prior[1],other_sid=prior[2]))
                    if i in chosen: previews.append((split,d.sensor_id[i],int(cls[i]),d.sid[i],d.sample_id[i],float(wind[i]),np.where(valid,image.astype(np.float32)/100,np.nan)))
                    if (i+1)%500==0: print(f'  {i+1}/{n}',flush=True)
                for key,count in counts.items(): issue('ERROR',key,f'{split}: {count} samples; example rows {examples[key]}')
                print(f'  Scanned {n} images',flush=True)
        except Exception as e: issue('ERROR','split read',f'{split}: {type(e).__name__}: {e}')
    d=pd.concat(frames,ignore_index=True) if frames else pd.DataFrame()
    stats={}
    if train_n:
        mean=train_sum/train_n; std=float(np.sqrt(max(0,train_sq/train_n-mean**2)))
        stats=dict(valid_pixels=train_n,mean_kelvin=float(mean),std_kelvin=std)
        for k,v in stats.items():
            if k not in norm or not np.isclose(v,norm[k],rtol=0,atol=0 if k=='valid_pixels' else 1e-6): issue('ERROR','training normalization',f'{k}: actual {v}, stored {norm.get(k)}')
        if std<=0: issue('ERROR','training normalization','zero standard deviation')
    if len(d):
        for k in ['sample_id']:
            if d[k].duplicated().any(): issue('ERROR','duplicate IDs',k)
        if d.duplicated(['sid','sensor_id','ibtracs_timestamp']).any(): issue('ERROR','duplicate platform/storm/fix','Duplicate observations')
        if (d.groupby('sid').split.nunique()>1).any(): issue('ERROR','storm leakage','SID occurs in multiple splits')
        # Same source scene in multiple splits is also undesirable.
        scene=d[['split','sensor_id','nominal_timestamp']].drop_duplicates()
        if (scene.groupby(['sensor_id','nominal_timestamp']).split.nunique()>1).any(): issue('ERROR','scene leakage','Satellite scene spans splits')
        dt=pd.to_datetime(d.ibtracs_timestamp,format='ISO8601',utc=True,errors='coerce'); intervals={s:(dt[d.split==s].min(),dt[d.split==s].max()) for s in SPLITS}
        if not (intervals['train'][1]<intervals['val'][0] and intervals['val'][1]<intervals['test'][0]): issue('ERROR','chronology',str(intervals))
        sums=d.groupby(['sid','ibtracs_timestamp']).observation_weight.sum()
        if not np.allclose(sums,1,atol=1e-6): issue('ERROR','observation weights','Weights do not sum to 1 per storm fix')
        for s in SPLITS:
            sub=d[d.split==s]; stored=summary.get('split_summary',{}).get(s,{})
            if len(sub)!=stored.get('samples') or sub.sid.nunique()!=stored.get('storms'): issue('ERROR','summary counts',s)
            for c in range(7):
                g=sub[sub.class_id==c]; nstorms=g.sid.nunique()
                if len(g)!=stored.get('class_counts',{}).get(str(c)): issue('ERROR','summary class counts',f'{s}/{c}')
                if nstorms<a.min_class_storms: issue('WARNING','rare class',f'{s}/{CLASSES[c]}: {len(g)} images from {nstorms} storms')
            for sensor in sorted(d.sensor_id.unique()):
                if sensor not in set(sub.sensor_id): issue('WARNING','sensor coverage',f'{s} lacks {sensor}')
            for basin in sorted(set(d.basin)-{'UNKNOWN',''}):
                if basin not in set(sub.basin): issue('WARNING','basin coverage',f'{s} lacks {basin}')
        if len(d)!=summary.get('accepted_count') or d.sid.nunique()!=summary.get('storm_count'): issue('ERROR','global summary counts','Mismatch')
        for col in ['class_id','sensor_id','basin']:
            dist=d.groupby(['split',col]).agg(samples=('sid','size'),storms=('sid','nunique')).reset_index().sort_values(['split',col]).reset_index(drop=True)
            dist.to_csv(out/f'distribution_{col}.csv',index=False)
            try:
                old=pd.read_csv(root/f'distribution_{col}.csv',keep_default_na=False).sort_values(['split',col]).reset_index(drop=True)
                pd.testing.assert_frame_equal(dist,old,check_dtype=False)
            except Exception as e: issue('ERROR','distribution report mismatch',f'{col}: {e}')
        try:
            man=pd.read_csv(root/'manifest.csv',keep_default_na=False)
            if man.sample_id.duplicated().any() or set(man.sample_id)!=set(d.sample_id): raise ValueError('Manifest sample IDs mismatch')
            left=d.set_index('sample_id'); right=man.set_index('sample_id').reindex(left.index)
            for k in ['split']+STRINGS+NUMBERS:
                if k=='sample_id': continue
                if k in NUMBERS:
                    match=np.allclose(left[k],pd.to_numeric(right[k],errors='coerce'),equal_nan=True,atol=1e-5)
                else: match=left[k].astype(str).equals(right[k].astype(str))
                if not match: issue('ERROR','manifest mismatch',k)
        except Exception as e: issue('ERROR','manifest',e)
        if a.ibtracs.is_file():
            try:
                track=pd.read_csv(a.ibtracs,usecols=['SID','ISO_TIME','LAT','LON','USA_WIND','BASIN'],keep_default_na=False,low_memory=False)
                track['fix']=pd.to_datetime(track.ISO_TIME,format='%Y-%m-%d %H:%M:%S',utc=True,errors='coerce'); track=track.dropna(subset=['fix'])
                for k in ['LAT','LON','USA_WIND']: track[k]=pd.to_numeric(track[k],errors='coerce')
                track=track.drop_duplicates(['SID','fix','LAT','LON','USA_WIND','BASIN']); dup=track.duplicated(['SID','fix'],keep=False); track=track[~dup]
                joined=d.assign(fix=dt).merge(track,left_on=['sid','fix'],right_on=['SID','fix'],how='left',validate='many_to_one')
                missing=joined.SID.isna();
                if missing.any(): issue('ERROR','IBTrACS join',f'{missing.sum()} unmatched or ambiguous fixes')
                q=joined[~missing]; lon_error=(q.lon-q.LON+180)%360-180
                if ((q.lat-q.LAT).abs()>1e-4).any() or (lon_error.abs()>1e-4).any() or not np.allclose(q.wind_kts,q.USA_WIND,atol=1e-5): issue('ERROR','IBTrACS labels','Coordinate or wind mismatch')
                mismatch=q[q.basin!=q.BASIN]; mismatch[['sample_id','sid','ibtracs_timestamp','basin','BASIN']].to_csv(out/'basin_mismatches.csv',index=False)
                if len(mismatch): issue('WARNING','IBTrACS basin mismatch',f'{len(mismatch)} rows; see basin_mismatches.csv')
            except Exception as e: issue('ERROR','IBTrACS verification',e)
        else: issue('WARNING','IBTrACS verification','Source CSV unavailable; independent label check skipped')
    if duplicates:
        pd.DataFrame(duplicates).to_csv(out/'exact_duplicate_images.csv',index=False)
        cross=sum(x['split']!=x['other_split'] for x in duplicates); issue('ERROR' if cross else 'WARNING','exact duplicate images',f'{len(duplicates)} pairs, {cross} cross-split pairs; investigate')
    pd.DataFrame(pixel_rows).to_csv(out/'pixel_statistics.csv',index=False)
    if pixel_rows:
        pd.DataFrame(pixel_rows).groupby(['split','sensor_id','class_id']).agg(samples=('sample_id','size'),mean_sample_kelvin=('mean_kelvin','mean'),mean_sample_std=('std_kelvin','mean')).to_csv(out/'sensor_temperature_statistics.csv')
    if previews:
        try:
            import matplotlib; matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            for split in SPLITS:
                for sensor in sorted({p[1] for p in previews if p[0]==split}):
                    rows=[p for p in previews if p[:2]==(split,sensor)]; cols=3; nr=(len(rows)+cols-1)//cols
                    fig,axes=plt.subplots(nr,cols,figsize=(10,3.5*nr),squeeze=False)
                    for ax in axes.flat: ax.axis('off')
                    for ax,(_,_,c,sid,ident,w,img) in zip(axes.flat,rows):
                        im=ax.imshow(img,cmap='gray',vmin=180,vmax=330); ax.plot((size-1)/2,(size-1)/2,'r+',ms=10); ax.set_title(f'{CLASSES[c]} | {w:g} kt\n{sid}',fontsize=9)
                    fig.colorbar(im,ax=list(axes.flat),label='Brightness temperature (K)',shrink=.65); fig.suptitle(f'{split} / {sensor} — red cross = expected centre'); fig.savefig(out/f'previews_{split}_{sensor}.png',dpi=120,bbox_inches='tight'); plt.close(fig)
        except Exception as e: issue('WARNING','preview generation',e)
    issue('WARNING','scientific review','Manual preview inspection and independent raw-scene calibration/geolocation comparison are still required. Sensor temperature differences may reflect basin/intensity differences.')
    errors=sum(i['level']=='ERROR' for i in issues); warnings=sum(i['level']=='WARNING' for i in issues)
    report=dict(dataset=str(root),status='FAIL' if errors else 'REVIEW_REQUIRED' if warnings else 'PASS',errors=errors,warnings=warnings,samples=len(d),storms=int(d.sid.nunique()) if len(d) else 0,training_normalization=stats,issues=issues)
    (out/'readiness_report.json').write_text(json.dumps(report,indent=2)); pd.DataFrame(issues).to_csv(out/'issues.csv',index=False)
    print(f"\n{report['status']}: {errors} errors, {warnings} warnings. Reports: {out}")
    return 1 if errors else 2 if warnings else 0

if __name__=='__main__':
    raise SystemExit(main())
