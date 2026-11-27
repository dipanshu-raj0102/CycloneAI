"""Training-only mining and reproducible sampling via a YOLO image-list file."""
import csv
import hashlib
import json
import math
from pyproj import Geod
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np


def candidates(samples, scores, threshold):
    if any(r['split'] != 'train' for r in samples.values()):
        raise ValueError('Mining accepts training samples only')
    return sorted([(k, scores[k]) for k,r in samples.items() if not r['objects'] and scores[k]>=threshold], key=lambda x:(-x[1],x[0]))


def approved(path, mined):
    allowed=set(k for k,_ in mined)
    with Path(path).open(newline='') as f:
        rows=list(csv.DictReader(f))
    selected={r['image'] for r in rows if r['approved'].strip().lower() in ('yes','true','1')}
    if not selected:
        raise ValueError('No approved hard negatives. Review candidates and set approved=yes, or explicitly use --accept-screened or --ordinary-training.')
    invalid=selected-allowed
    if invalid:
        raise ValueError('Approved images not in cached candidates: '+', '.join(sorted(invalid)))
    return selected


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''): h.update(chunk)
    return h.hexdigest()


def load_cache(path, samples, fingerprint, checkpoint, threshold, imgsz, keypoint_conf, allow_legacy=False):
    """Read original V5 cache without inference; legacy weights cannot be cryptographically verified."""
    folder=Path(path).resolve(); config=json.loads((folder.parent/'run_config.json').read_text()); args=config['arguments']
    summary=json.loads((folder/'summary.json').read_text())
    if config['dataset_fingerprint']!=fingerprint: raise ValueError('Mining dataset fingerprint differs; remine explicitly')
    if Path(summary['checkpoint']).resolve()!=Path(checkpoint).resolve(): raise ValueError('Mining checkpoint path differs')
    for key,value in [('mine_confidence',threshold),('imgsz',imgsz),('keypoint_conf',keypoint_conf)]:
        if args.get(key)!=value: raise ValueError(f'Mining cache requires original --{key.replace("_","-")}={args.get(key)}')
    manifest=folder/'cache_identity.json'
    if manifest.exists():
        meta=json.loads(manifest.read_text())
        if meta['checkpoint_sha256']!=sha256(checkpoint) or meta['predictions_sha256']!=sha256(folder/'tile_predictions.jsonl'):
            raise ValueError('Mining checkpoint or predictions changed; remine explicitly')
    elif not allow_legacy:
        raise ValueError('V5 cache has no weight hash. Use --allow-legacy-mining-cache only if original checkpoint is unchanged, or remine explicitly.')
    negatives={k:r for k,r in samples.items() if not r['objects'] and r['split']=='train'}
    if summary['negative_tiles']!=len(negatives): raise ValueError('Mining negative count differs')
    raw=defaultdict(list); scores={k:0. for k in negatives}
    with (folder/'tile_predictions.jsonl').open() as f:
        for line in f:
            p=json.loads(line); k=p['image']; score=float(p['score'])
            if k not in negatives or p['scene']!=negatives[k]['scene'] or not math.isfinite(score) or not 0<=score<=1:
                raise ValueError(f'Invalid cached prediction: {k}')
            raw[k].append(p); scores[k]=max(scores[k],score)
    mined=candidates(negatives,scores,threshold)
    if len(mined)!=summary['candidates']: raise ValueError('Mining candidate count differs')
    with (folder/'candidates.csv').open(newline='') as f:
        rows=list(csv.DictReader(f))
    recorded={r['image']:float(r['score']) for r in rows}
    if len(recorded)!=len(rows) or set(recorded)!=dict(mined).keys() or any(not math.isclose(recorded[k],v,abs_tol=1e-7) for k,v in mined):
        raise ValueError('Candidate CSV does not match saved predictions')
    return mined,raw


def screen(mined, raw, samples, plans, threshold, clearance_km=600.):
    """Conservative screening against IBTrACS-derived scene plan, not independent truth."""
    geod=Geod(ellps='WGS84'); rows=[]
    for k,score in mined:
        scene=samples[k]['scene']; plan=plans[scene]; obs=plan['objects']; nearest=float('inf'); reason='clear_of_catalogued_centres'
        for p in raw.get(k,[]):
            if p['score']<threshold: continue
            if not p['valid'] or p.get('lon') is None or p.get('lat') is None:
                reason='invalid_keypoint'; break
            for o in obs:
                if o.get('unknown') or o.get('lon') is None or o.get('lat') is None:
                    reason='unresolved_catalogue_position'; break
                d=geod.inv(p['lon'],p['lat'],o['lon'],o['lat'])[2]/1000
                if not math.isfinite(d): reason='unresolved_catalogue_position'; break
                nearest=min(nearest,d)
            if reason!='clear_of_catalogued_centres': break
        if not raw.get(k): reason='missing_predictions'
        if reason=='clear_of_catalogued_centres' and nearest<clearance_km: reason='near_catalogued_storm'
        rows.append(dict(image=k,score=score,scene=scene,nearest_catalogued_km=nearest if math.isfinite(nearest) else '',screen_status=reason,approved=''))
    return rows


def sample_list(samples, hard, seed, storm_balance=False, hard_fraction=.05, max_repeat=3, excluded=None):
    if any(r['split']!='train' for r in samples.values()): raise ValueError('Training samples only')
    if not hard <= {k for k,r in samples.items() if not r['objects']}: raise ValueError('Invalid hard negatives')
    rng=np.random.default_rng(seed); excluded=set(excluded or ())-hard
    pos=[k for k,r in samples.items() if r['objects']]; neg=[k for k,r in samples.items() if not r['objects'] and k not in hard and k not in excluded]
    if not pos or not neg or not hard: raise ValueError('Need all three sample groups')
    n=len(samples); nh=min(round(n*hard_fraction),len(hard)*max_repeat); np_=max(len(pos),round(n*.35)); nh=min(nh,n-np_-1); nn=n-nh-np_
    if nh<1: raise ValueError('Hard negative budget rounds to zero')
    weights=None
    if storm_balance:
        counts=Counter(o['sid'] for k in pos for o in samples[k]['objects'])
        weights=np.array([sum(1/math.sqrt(counts[o['sid']]) for o in samples[k]['objects']) for k in pos]); weights/=weights.sum()
    def draw(pool,n,w=None):
        if w is None and n>=len(pool): return list(pool)+rng.choice(pool,n-len(pool),replace=True).tolist()
        return rng.choice(pool,n,replace=w is not None or n>len(pool),p=w).tolist()
    hard_pool=sorted(hard); q,r=divmod(nh,len(hard_pool)); hard_draw=hard_pool*q+rng.choice(hard_pool,r,replace=False).tolist()
    positive_draw=list(pos)+rng.choice(pos,np_-len(pos),replace=True,p=weights).tolist()
    result=positive_draw+draw(neg,nn)+hard_draw; rng.shuffle(result)
    return result


def gallery(root, rows, out, title, limit=48):
    from PIL import Image, ImageDraw
    out.mkdir(parents=True,exist_ok=True)
    rows=sorted(rows,key=lambda r:-r['score'])[:limit]
    for start in range(0,len(rows),12):
        canvas=Image.new('RGB',(4*256,3*280),'white'); d=ImageDraw.Draw(canvas)
        for i,r in enumerate(rows[start:start+12]):
            x=i%4*256;y=i//4*280
            with Image.open(root/r['image']) as im:
                w,h=im.size; canvas.paste(im.convert('RGB').resize((256,256)),(x,y))
            if 'box_xyxy' in r:
                a,b,c,e=r['box_xyxy']; d.rectangle((x+a*256/w,y+b*256/h,x+c*256/w,y+e*256/h),outline='red',width=2)
            d.text((x+3,y+257),f"{r['score']:.3f} {Path(r['image']).stem[:20]}",fill='black')
        canvas.save(out/f'{title}_{start//12:02}.png')
