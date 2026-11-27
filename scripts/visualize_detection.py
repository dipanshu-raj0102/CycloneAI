#!/usr/bin/env python3
"""Presentation evaluation for a frozen GOES-16 cyclone detector. No training."""
from __future__ import annotations
import argparse
import csv
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
from PIL import Image
import yaml

HERE = Path(__file__).resolve().parent
if (HERE/'evaluation_core'/'train_detection.py').is_file():
    sys.path.insert(0, str(HERE/'evaluation_core'))
import train_detection as core

BLUE='#1565A8'; TEAL='#008A83'; ORANGE='#D97706'; RED='#C24141'; GRAY='#64748B'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','axes.facecolor':'white','savefig.facecolor':'white','axes.labelcolor':'#243447','text.color':'#243447','axes.edgecolor':'#CBD5E1','grid.color':'#E2E8F0'})


def save(fig, out, name, note=None):
    if note:
        fig.text(.02,.018,note,fontsize=9,color=GRAY,wrap=True)
    fig.tight_layout(rect=(0,.06 if note else 0,1,1))
    for ext in ('png','svg'):
        fig.savefig(out/f'{name}.{ext}',dpi=260,bbox_inches='tight')
    plt.close(fig)


def read_csv(path):
    with Path(path).open(newline='') as f:
        return [{k.strip():v for k,v in row.items() if k is not None} for row in csv.DictReader(f)]


def pct(v):
    return 'N/A' if v is None else f'{100*v:.1f}%'


def number(v):
    return 'N/A' if v is None else f'{v:.2f}'


def val(v):
    return np.nan if v is None else v


def digest(path):
    return core.hn.sha256(path)


def history_comparison(paths, out):
    """Report per-metric maxima, and epochs; never call maxima checkpoint metrics."""
    cols={'box50':'metrics/mAP50(B)','box5095':'metrics/mAP50-95(B)', 'pose50':'metrics/mAP50(P)','pose5095':'metrics/mAP50-95(P)'}
    rows=[]; histories=[]; seen={}
    for label,path in paths:
        path=Path(path)
        if not path.is_file():
            print(f'Skip missing training history: {path}'); continue
        h=digest(path)
        if h in seen:
            print(f'Skip duplicate history {label} (same bytes as {seen[h]})'); continue
        data=read_csv(path)
        if not data: continue
        if not all(k in data[0] for k in cols.values()):
            print(f'Skip incompatible training CSV: {path}'); continue
        a={k:np.array([float(r[c]) for r in data]) for k,c in cols.items()}
        if any(not np.isfinite(v).all() for v in a.values()): raise ValueError(f'Non-finite history: {path}')
        epochs=np.array([int(float(r['epoch'])) for r in data]); seen[h]=label
        row=dict(run=label,source=str(path.resolve()),epochs=len(data))
        for k,v in a.items():
            i=int(v.argmax()); row[k]=float(v[i]); row[k+'_epoch']=int(epochs[i])
        rows.append(row); histories.append((label,epochs,a,data))
    if not rows: return []
    core.write_csv(out/'model_comparison.csv',rows,list(rows[0]))
    fig,ax=plt.subplots(figsize=(11,max(4,1+.65*len(rows)))); y=np.arange(len(rows))
    ax.barh(y-.18,[r['box50']*100 for r in rows],.34,color=BLUE,label='Box mAP50')
    ax.barh(y+.18,[r['pose50']*100 for r in rows],.34,color=TEAL,label='Pose mAP50')
    for j,r in enumerate(rows):
        for dy,k in [(-.18,'box50'),(.18,'pose50')]: ax.text(r[k]*100+.7,j+dy,f'{r[k]*100:.1f}',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=[r['run'] for r in rows],xlim=(0,108),xlabel='Best validation score (%)',title='Detection model experiments'); ax.invert_yaxis(); ax.legend(loc='lower right'); ax.grid(axis='x',alpha=.5); ax.set_axisbelow(True)
    save(fig,out,'01_model_comparison','Maxima may come from different epochs. Compare only runs using the same validation data and OKS settings.')
    fig,ax=plt.subplots(figsize=(10,5));
    for label,epochs,a,_ in histories: ax.plot(epochs,a['pose5095']*100,marker='.',label=label)
    ax.set(xlabel='Epoch',ylabel='Pose mAP50–95 (%)',title='Centre-localization learning curves'); ax.grid(alpha=.5); ax.legend(fontsize=9)
    save(fig,out,'02_pose_learning_curves','Native pose AP uses OKS, not a distance in kilometres. Same metric configuration required for comparison.')
    return rows


def selection_comparison(path,out):
    data=json.loads(Path(path).read_text()); chosen=data['chosen']; rows=[]
    for i,r in enumerate(data['candidates']):
        p=Path(r['checkpoint']); name=f'{p.parent.parent.parent.name} / {p.name}'
        rows.append(dict(checkpoint=r['checkpoint'],name=name,selected=r['checkpoint']==chosen['checkpoint'],threshold=r['threshold'],precision=r['precision'],recall=r['recall'],f1=r['f1'],tp=r['tp'],fp=r['fp'],fn=r['fn'],mean_error_km=r['centre_error_km']['mean']))
    core.write_csv(out/'checkpoint_comparison.csv',rows,list(rows[0]))
    fig,ax=plt.subplots(figsize=(11,max(4,1+.42*len(rows)))); y=np.arange(len(rows));
    ax.barh(y,[100*val(r['f1']) for r in rows],color=[TEAL if r['selected'] else BLUE for r in rows]);
    for j,r in enumerate(rows):
        if r['f1'] is not None: ax.text(r['f1']*100+.5,j,f'{r["f1"]*100:.1f}%',va='center',fontsize=10)
    ax.set(yticks=y,yticklabels=[r['name'] for r in rows],xlim=(0,108),xlabel='Geographic F1 (%)',title='Why this checkpoint was selected'); ax.invert_yaxis(); ax.grid(axis='x',alpha=.5); ax.set_axisbelow(True)
    save(fig,out,'03_checkpoint_selection',f"Saved {data['split']} selection, {data['match_km']:g} km radius. Each candidate uses its own validation operating point. Teal = selected.")
    return data


def dataset_chart(counts, diversity,out):
    fig,ax=plt.subplots(figsize=(9,5)); names=list(core.SPLITS); x=np.arange(3)
    pos=[counts[s]['positive'] for s in names]; neg=[counts[s]['negative'] for s in names]
    ax.bar(x,pos,color=TEAL,label='Cyclone tiles'); ax.bar(x,neg,bottom=pos,color=BLUE,label='Background tiles')
    for i,s in enumerate(names): ax.text(i,pos[i]+neg[i]+max(pos+neg)*.04,f'{pos[i]+neg[i]:,} tiles\n{diversity[s]["unique_storms"]} storms',ha='center')
    ax.set(xticks=x,xticklabels=['Train','Validation','Test'],ylabel='Image tiles',title='Storm-independent data split',ylim=(0,max(np.array(pos)+neg)*1.24)); ax.legend(); ax.grid(axis='y',alpha=.5); ax.set_axisbelow(True)
    save(fig,out,'04_dataset_split','Storm counts are unique covered IBTrACS IDs. Tile count is not an independent cyclone count.')


def presentation_plots(m, out, split, native=None):
    rows=read_csv(out/'centre_matches.csv'); curve=read_csv(out/'confidence_curve.csv'); storms=read_csv(out/'storm_metrics.csv')
    errors=np.array([float(r['error_km']) for r in rows if r['status']=='TP']); radius=m['match_km']; threshold=m['threshold']; tag=f'{split.upper()} | confidence ≥ {threshold:.3f} | retained tiles'
    fig,ax=plt.subplots(figsize=(10,5)); names=['Precision','Recall','F1']; v=[m[k] for k in ('precision','recall','f1')]; ax.bar(names,[100*val(k) for k in v],color=[BLUE,TEAL,BLUE]);
    for i,k in enumerate(v):
        if k is not None: ax.text(i,k*100+2,pct(k),ha='center',fontweight='bold',fontsize=17)
    ax.set(ylim=(0,110),ylabel='Score (%)',title=f'Geographic detection within {radius:g} km'); ax.grid(axis='y',alpha=.5); ax.set_axisbelow(True)
    save(fig,out,'05_geographic_performance',tag+f" | {m['tp']} matched, {m['fp']} false detections, {m['fn']} missed covered observations")
    fig,ax=plt.subplots(figsize=(9,4.8)); counts=[m['tp'],m['fp'],m['fn']]; ax.bar(['Matched','False detections','Missed targets'],counts,color=[TEAL,RED,ORANGE]);
    for i,n in enumerate(counts): ax.text(i,n+max(1,max(counts)*.025),str(n),ha='center',fontweight='bold')
    ax.set(ylabel='Count',title='Geographic detection outcomes',ylim=(0,max(1,max(counts))*1.18)); save(fig,out,'06_detection_outcomes',tag+' | One-to-one matching; repeated tile views are deduplicated.')
    fig,ax=plt.subplots(figsize=(10,5));
    for key,c in [('precision',BLUE),('recall',TEAL),('f1',ORANGE)]: ax.plot([float(r['threshold']) for r in curve],[float(r[key])*100 if r[key] else np.nan for r in curve],label=key.capitalize(),color=c,lw=2.5)
    ax.axvline(threshold,color=GRAY,ls='--',label=f'Frozen threshold {threshold:.3f}'); ax.set(xlabel='Box confidence threshold',ylabel='Score (%)',ylim=(0,103),title='Confidence trade-off'); ax.grid(alpha=.5); ax.legend(); save(fig,out,'07_confidence_tradeoff',tag+' | Threshold not optimized on test.')
    ds=sorted(m['centre_radius_metrics'],key=int); fig,ax=plt.subplots(figsize=(10,5));
    for key,c in [('precision',BLUE),('recall',TEAL),('f1',ORANGE)]: ax.plot(list(map(int,ds)),[100*val(m['centre_radius_metrics'][d][key]) for d in ds],marker='o',lw=2.5,color=c,label=key.capitalize())
    ax.axvline(radius,color=GRAY,ls='--',label='Primary match radius'); ax.set(xlabel='Maximum centre error (km)',ylabel='Score (%)',ylim=(0,103),title='Location accuracy at multiple tolerances'); ax.grid(alpha=.5); ax.legend(); save(fig,out,'08_localization_tolerances',tag+' | Each radius rematches all covered targets; missed targets remain in recall denominator.')
    if len(errors):
        fig,ax=plt.subplots(figsize=(10,5)); ax.hist(errors,bins=np.linspace(0,radius,11),color=BLUE,edgecolor='white'); ax.axvline(np.mean(errors),color=ORANGE,ls='--',lw=2,label=f'Mean {np.mean(errors):.1f} km'); ax.axvline(np.median(errors),color=TEAL,ls=':',lw=2,label=f'Median {np.median(errors):.1f} km'); ax.set(xlabel='Centre error (km)',ylabel='Matched observations',title='Distribution of matched centre errors'); ax.legend(); save(fig,out,'09_centre_error_distribution',tag+f' | Matched only (n={len(errors)}); errors beyond {radius:g} km and misses excluded.')
        fig,ax=plt.subplots(figsize=(10,5)); e=np.sort(errors); ax.step(np.r_[0,e],np.r_[0,np.arange(1,len(e)+1)/len(e)*100],where='post',color=TEAL,lw=2.5); ax.set(xlabel='Centre error (km)',ylabel='Matched observations within distance (%)',ylim=(0,103),title='Cumulative centre accuracy'); ax.grid(alpha=.5); save(fig,out,'10_centre_error_cdf',tag+' | Conditional on matched observations, not overall cyclone recall.')
    for level in ('tile','scene'):
        b=m[level+'_presence']; matrix=np.array([[b['tn'],b['fp']],[b['fn'],b['tp']]]); fig,ax=plt.subplots(figsize=(7,5)); ax.imshow(matrix,cmap='Blues');
        for i in range(2):
            for j in range(2): ax.text(j,i,f'{matrix[i,j]:,}',ha='center',va='center',fontsize=23,color='white' if matrix[i,j]>matrix.max()/2 else '#243447',fontweight='bold')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Absent','Present'],yticklabels=['Absent','Present'],xlabel='Predicted presence',ylabel='Actual presence',title=f'{level.capitalize()} presence confusion matrix'); save(fig,out,f'11_{level}_confusion_matrix',tag+' | Presence only; centre correctness is measured separately.')
    if storms:
        fig,ax=plt.subplots(figsize=(10,max(4,1+.45*len(storms)))); y=np.arange(len(storms)); ax.barh(y,[float(r['recall'])*100 for r in storms],color=TEAL)
        for i,r in enumerate(storms): ax.text(float(r['recall'])*100+.7,i,f"{r['matched']}/{r['observations']}",va='center')
        ax.set(yticks=y,yticklabels=[r['sid'] for r in storms],xlim=(0,115),xlabel='Geographic recall (%)',title='Generalization across held-out storms'); ax.invert_yaxis(); save(fig,out,'12_storm_recall',tag+' | Labels show matched / covered observations per IBTrACS storm ID.')
    fig,ax=plt.subplots(figsize=(11,6));
    for status,c,marker in [('TP',TEAL,'o'),('FN',ORANGE,'x'),('FP',RED,'^')]:
        prefix='pred' if status=='FP' else 'true'; pts=[(float(r[prefix+'_lon']),float(r[prefix+'_lat'])) for r in rows if r['status']==status and r.get(prefix+'_lon') and r.get(prefix+'_lat')]
        if pts: ax.scatter(*zip(*pts),s=25,c=c,marker=marker,alpha=.7,label=status)
    ax.set(xlabel='Longitude (°)',ylabel='Latitude (°)',title='Spatial distribution of detection outcomes'); ax.grid(alpha=.5)
    if ax.has_data(): ax.legend()
    save(fig,out,'13_geographic_outcomes',tag+' | Coordinate scatter, not a basemap. Invalid FP centres cannot be plotted.')
    fp=[float(r['score']) for r in rows if r['status']=='FP' and r['score']]; fig,ax=plt.subplots(figsize=(9,5)); ax.hist(fp,bins=np.linspace(0,1,21),color=RED); ax.axvline(threshold,color=GRAY,ls='--'); ax.set(xlabel='Box confidence',ylabel='False detections',title='Confidence of false detections'); save(fig,out,'14_false_alarm_confidence',tag+' | Includes unmatched detections on both positive and negative tiles.')
    if native:
        fig,ax=plt.subplots(figsize=(9,5)); x=np.arange(2)
        for offset,key,c in [(-.18,'box',BLUE),(.18,'pose',TEAL)]:
            values=[100*val(native[key][k]) for k in ('map50','map')]; ax.bar(x+offset,values,.34,label=key.capitalize(),color=c)
            for j,v in enumerate(values):
                if np.isfinite(v): ax.text(j+offset,v+2,f'{v:.1f}%',ha='center')
        ax.set(xticks=x,xticklabels=['mAP50','mAP50–95'],ylim=(0,110),ylabel='Average precision (%)',title='Native box and keypoint accuracy'); ax.legend(); save(fig,out,'15_native_average_precision',f'{split.upper()} | Box IoU / pose OKS (sigma={native["oks_sigma"]}). Native AP uses score ranking, not frozen threshold.')


def gallery(root,samples,raw,out,m,split,n=6):
    rows=read_csv(out/'centre_matches.csv'); by_image=defaultdict(list); by_target=defaultdict(list)
    for ps in raw.values():
        for p in ps: by_image[p['image']].append(p)
    for im,r in samples.items():
        for o in r['objects']: by_target[(r['scene'],o['sid'])].append(im)
    def candidates(row):
        if row['status']=='FN':
            ims=by_target[(row['scene'],row['sid'])]
            return max(ims,key=lambda im: min(min(o['values'][4],1-o['values'][4],o['values'][5],1-o['values'][5]) for o in samples[im]['objects'] if o['sid']==row['sid'])) if ims else None
        return row['image'] or None
    def panel(ax,im,r):
        with Image.open(root/im) as image: ax.imshow(image.convert('RGB'))
        w=h=int(m['image_size']); status=r['status']
        for o in samples[im]['objects']:
            x,y,bw,bh,u,v=o['values']; ax.add_patch(Rectangle(((x-bw/2)*w,(y-bh/2)*h),bw*w,bh*h,fill=False,edgecolor='#5EF0AB',lw=2,ls='--')); ax.scatter(u*w,v*h,c='#5EF0AB',s=65,marker='+',linewidths=2.5)
        ps=[p for p in by_image[im] if p['score']>=m['threshold']]
        for p in ps:
            x1,y1,x2,y2=p['box_xyxy']; ax.add_patch(Rectangle((x1,y1),x2-x1,y2-y1,fill=False,edgecolor='#FFD166',lw=2));
            if p['valid']: ax.scatter(p['u']*w,p['v']*h,c='#FFD166',s=35,marker='x',linewidths=2)
            ax.text(x1,max(10,y1-4),f'{p["score"]:.2f}',color='#FFD166',fontsize=9,bbox=dict(facecolor='#152238',alpha=.8,pad=1,edgecolor='none'))
        title={'TP':'Matched','FP':'Unmatched prediction','FN':'Missed target','TN':'Correct negative tile'}[status]
        if status=='TP': title+=f' | {float(r["error_km"]):.1f} km'
        if status=='FP': title+=f' | score {float(r["score"]):.2f}'
        if status=='TP' and not any(o['sid']==r['sid'] for o in samples[im]['objects']): title+='*'
        ax.set_title(title,fontsize=11); ax.set_xlabel(Path(im).stem,fontsize=8); ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlim(-.5,w-.5); ax.set_ylim(h-.5,-.5)
    def emit(name,items,note):
        if not items:return
        cols=min(3,len(items)); nr=math.ceil(len(items)/cols); fig,axs=plt.subplots(nr,cols,figsize=(4.5*cols,4.6*nr),squeeze=False)
        for ax in axs.flat: ax.axis('off')
        for ax,(im,r) in zip(axs.flat,items): ax.axis('on'); panel(ax,im,r)
        fig.suptitle(f'{split.upper()} | {name.replace("_"," ").title()}',fontsize=16,fontweight='bold'); save(fig,out,name,note+' Green dashed/+ = reference; yellow solid/x = prediction. *Reference on another tile.')
        for i,(im,r) in enumerate(items,1):
            fig,ax=plt.subplots(figsize=(6,6)); panel(ax,im,r); save(fig,out,f'{name}_{i:02}',f'{split.upper()} | {r["status"]} | confidence ≥ {m["threshold"]:.3f}. Green = reference; yellow = prediction.')
    def unique(rs):
        result=[]; seen=set()
        for r in rs:
            im=candidates(r)
            if im and r['scene'] not in seen:
                seen.add(r['scene']); result.append((im,r))
        return result
    tp=unique(sorted((r for r in rows if r['status']=='TP'),key=lambda r:float(r['error_km'])))
    # Explicitly chosen best examples, not a random or representative sample.
    best=[]; storms=set()
    for item in tp:
        if item[1]['sid'] not in storms: best.append(item); storms.add(item[1]['sid'])
    best=(best+[item for item in tp if item not in best])[:n]
    emit('16_best_localization_examples',best,'Selected low-error examples, prioritizing distinct storms; not representative of all predictions.')
    if tp:
        ii=np.unique(np.linspace(0,len(tp)-1,min(n,len(tp))).round().astype(int)); emit('17_error_range_examples',[tp[i] for i in ii],'Deterministic examples spanning matched errors, including large errors.')
    fp=unique(sorted((r for r in rows if r['status']=='FP'),key=lambda r:-float(r['score'])))[:n]
    fn=unique(sorted((r for r in rows if r['status']=='FN'),key=lambda r:(r['sid'],r['scene'])))[:n]
    emit('18_false_detection_examples',fp,'Highest-confidence unmatched detections across distinct scenes. Other predictions on each tile are also shown.')
    emit('19_missed_target_examples',fn,'Selected missed observations, shown on a reference-containing tile; miss is defined after geographic matching.')
    tn=[]; seen=set()
    for im,r in sorted(samples.items()):
        if not r['objects'] and not any(p['score']>=m['threshold'] for p in by_image[im]) and r['scene'] not in seen:
            tn.append((im,dict(status='TN'))); seen.add(r['scene'])
            if len(tn)>=n:break
    emit('20_correct_negative_examples',tn,'Selected negative tiles without a prediction at the frozen threshold; not proof of full-disk absence.')
    items=[]
    for group,ls in [('best',best),('error_range',[tp[i] for i in ii] if tp else []),('false_detection',fp),('missed_target',fn),('correct_negative',tn)]:
        for im,r in ls: items.append(dict(group=group,image=im,scene=samples[im]['scene'],status=r['status'],sid=r.get('sid',''),score=r.get('score',''),error_km=r.get('error_km','')))
    core.write_csv(out/'gallery_manifest.csv',items,['group','image','scene','status','sid','score','error_km'])


def metric_table(m,native,split,out):
    rows=[]
    def add(metric,value,unit,definition):rows.append(dict(split=split,metric=metric,value=value,unit=unit,definition=definition))
    for key,defn in [('precision','Matched centres / retained predictions'),('recall','Matched centres / covered reference observations'),('f1','Harmonic mean of geographic precision and recall')]: add('Geographic '+key,m[key],'fraction',defn)
    for key,defn in [('tp','One-to-one matches within radius'),('fp','Unmatched predictions, including invalid centres'),('fn','Unmatched covered reference observations')]:add(key.upper(),m[key],'count',defn)
    for k,v in m['centre_error_km'].items():add('Matched centre '+k,v,'count' if k=='count' else 'km','Matched observations only; WGS84 geodesic distance')
    for k,v in m['coverage'].items():add(k,v,'fraction' if k in ('coverage_fraction','recall_including_uncovered_as_misses') else 'count','Coverage of the retained preprocessing plan, not the full disk')
    for d,r in m['centre_radius_metrics'].items():
        for key in ('precision','recall','f1','AP'):add(f'Geographic {key} @ {d} km',r[key],'fraction','Independent rematching at each radius; AP uses ranked predictions')
    for level in ('tile','scene'):
        for k,v in m[level+'_presence'].items():add(f'{level} presence {k}',v,'count' if k in ('tp','fp','fn','tn') else 'fraction','Binary presence, regardless of centre correctness; retained tiles only')
        for k,v in m['diagnostics'][level+'_presence'].items():add(f'{level} presence {k}',v,'fraction','Binary presence derived statistic')
    for k in ('macro_storm_recall','mean_false_positives_per_evaluated_scene','raw_predictions_at_threshold','deduplicated_predictions_at_threshold','invalid_centres_at_threshold'):add(k,m[k],'fraction' if k=='macro_storm_recall' else 'count/scene' if k.startswith('mean_') else 'count','Geographic post-processing statistic')
    if native:
        for group in ('box','pose'):
            for k,v in native[group].items():add(f'Native {group} {k}',v,'fraction','Native framework metric; AP uses ranked scores, precision/recall use framework operating point')
        for k,v in native['speed_ms_per_image'].items():add(f'Native {k} latency',v,'ms/image','Framework validation timing; hardware and batch dependent')
    core.write_csv(out/'presentation_metrics.csv',rows,['split','metric','value','unit','definition'])
    return rows


def summary_text(m,native,split):
    e=m['centre_error_km']; b=m['tile_presence']; s=m['scene_presence']; c=m['coverage']
    lines=[f'## {split.upper()} — frozen checkpoint', '',f'Operating confidence: **{m["threshold"]:.3f}**; geographic match radius: **{m["match_km"]:g} km**.', '', '| Metric | Value |','|---|---:|',f'| Geographic precision | {pct(m["precision"])} |',f'| Geographic recall | {pct(m["recall"])} |',f'| Geographic F1 | {pct(m["f1"])} |',f'| Matched / false / missed | {m["tp"]} / {m["fp"]} / {m["fn"]} |',f'| Mean / median matched centre error | {number(e["mean"])} / {number(e["median"])} km |',f'| RMSE / P90 matched error | {number(e["rmse"])} / {number(e["p90"])} km |',f'| Recall within 50 km (all covered targets) | {pct(m["centre_radius_metrics"]["50"]["recall"])} |',f'| Macro storm recall | {pct(m["macro_storm_recall"])} |',f'| Recall including uncovered planned targets | {pct(c["recall_including_uncovered_as_misses"])} |',f'| Planned target coverage | {c["covered_target_observations"]}/{c["planned_target_observations"]} |',f'| Tile false-positive rate | {pct(b["fp"]/(b["fp"]+b["tn"]) if b["fp"]+b["tn"] else None)} |',f'| Scene false-positive rate | {pct(s["fp"]/(s["fp"]+s["tn"]) if s["fp"]+s["tn"] else None)} |',f'| False detections per retained scene | {number(m["mean_false_positives_per_evaluated_scene"])} |']
    if native:
        for name in ('box','pose'): lines.extend([f'| {name.title()} mAP50 | {pct(native[name]["map50"])} |',f'| {name.title()} mAP50–95 | {pct(native[name]["map"])} |'])
    return '\n'.join(lines)+'\n\n**Interpretation:** geographic matching counts each storm observation once per scene, after overlap deduplication. Native box AP evaluates proxy boxes, not actual storm extent. Pose AP uses OKS, not kilometres. Matched errors exclude misses and errors beyond the primary radius. Scene results cover retained tiles only. The test set must not be used to adjust threshold or choose a checkpoint.\n'


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.cwd(),help='CycloneAI project root')
    p.add_argument('--data',type=Path); p.add_argument('--checkpoint',type=Path); p.add_argument('--selection-report',type=Path)
    p.add_argument('--operating-point',type=Path,help='Original evaluation/operating_point.json (if available)')
    p.add_argument('--output',type=Path); p.add_argument('--runs',type=Path); p.add_argument('--history',action='append',default=[],metavar='LABEL=CSV')
    p.add_argument('--comparison-only',action='store_true',help='Only generate charts from existing CSV/selection report; no model/data needed')
    p.add_argument('--split',choices=['val','test','both'],default='val'); p.add_argument('--confidence',type=float)
    p.add_argument('--match-km',type=float); p.add_argument('--dedup-km',type=float); p.add_argument('--same-tile-dedup-km',type=float); p.add_argument('--keypoint-conf',type=float); p.add_argument('--oks-sigma',type=float,default=.1)
    p.add_argument('--device',default='0'); p.add_argument('--batch',type=int,default=8); p.add_argument('--workers',type=int,default=2); p.add_argument('--examples',type=int,default=6); p.add_argument('--skip-native',action='store_true'); p.add_argument('--no-gallery',action='store_true')
    args=p.parse_args(argv); root=args.root.resolve(); runs=(args.runs or root/'runs/cyclone_pose').resolve(); out=(args.output or root/'reports/detection_presentation').resolve(); selected=runs/'yolosT2'; selection=args.selection_report or selected/'checkpoint_selection.json'; op_path=args.operating_point or selected/'evaluation/operating_point.json'; checkpoint=(args.checkpoint or selected/'train/weights/best_geo.pt').resolve()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    if args.batch<1 or args.workers<0 or not 1<=args.examples<=12: p.error('Invalid batch, workers or example count')
    if out.exists(): p.error(f'Output already exists: {out}. Choose a new --output (no overwrite).')
    paths=[]
    if args.history:
        for item in args.history:
            if '=' not in item: p.error('--history requires LABEL=CSV')
            label,path=item.split('=',1); paths.append((label,Path(path)))
    else: paths=[(x.parent.parent.name,x) for x in sorted(runs.glob('*/train/results.csv'))]
    if args.comparison_only:
        out.mkdir(parents=True); rows=history_comparison(paths,out); s=selection_comparison(selection,out) if selection.is_file() else None
        if not rows and s is None: raise ValueError('No comparison input found')
        print(f'Comparison charts: {out}'); return 0
    if not checkpoint.is_file(): p.error(f'Missing checkpoint: {checkpoint}')
    if not selection.is_file() and args.selection_report: p.error(f'Missing selection report: {selection}')
    if not op_path.is_file() and args.operating_point: p.error(f'Missing operating point: {op_path}')
    custom = args.checkpoint is not None and checkpoint != (selected/'train/weights/best_geo.pt').resolve()
    sel=json.loads(selection.read_text()) if selection.is_file() and (not custom or args.selection_report) else None
    op=json.loads(op_path.read_text()) if op_path.is_file() and (not custom or args.operating_point) else {}
    sha=digest(checkpoint)
    if op.get('checkpoint_sha256') and op['checkpoint_sha256']!=sha: p.error('Operating point belongs to different checkpoint bytes')
    if sel and sel.get('split')!='val': p.error('Checkpoint selection must be from validation')
    # Defaults below reproduce the selected T2 operating point. Other checkpoints require their own explicit confidence/report.
    if args.checkpoint and not op and not args.selection_report and args.confidence is None: p.error('Custom checkpoint requires --confidence or its own --operating-point/--selection-report')
    def choose(key,override,default):return override if override is not None else op.get(key,default)
    confidence=choose('confidence',args.confidence,sel['chosen']['threshold'] if sel else .675)
    match=choose('match_km',args.match_km,sel['match_km'] if sel else 100.)
    dedup=choose('dedup_km',args.dedup_km,100.); same=choose('same_tile_dedup_km',args.same_tile_dedup_km,0.); kp=choose('keypoint_conf',args.keypoint_conf,.25)
    if not all(math.isfinite(x) for x in (confidence,match,dedup,same,kp,args.oks_sigma)) or not(core.FLOOR<=confidence<=1 and 0<dedup<=match and 0<=same<=dedup and 0<=kp<=1 and 0<args.oks_sigma<=1):p.error('Invalid inference settings')
    if args.split in ('test','both'):
        locked=op or ({'confidence':sel['chosen']['threshold'],'match_km':sel['match_km']} if sel else {})
        for key,v in [('confidence',confidence),('match_km',match),('dedup_km',dedup),('same_tile_dedup_km',same),('keypoint_conf',kp)]:
            if key in locked and not math.isclose(v,locked[key]):p.error(f'Test settings differ from saved operating point: {key}')
        if not locked and args.confidence is None: p.error('Test evaluation requires a saved or explicitly frozen confidence')
    data=(args.data or root/'data/processed/detection_pose_v2').resolve(); build,plans,samples,fingerprint,counts=core.load_dataset(data); diversity=core.diversity_report(samples,plans)
    import ultralytics,torch
    from ultralytics import YOLO
    model=YOLO(str(checkpoint));
    if model.task!='pose' or tuple(model.model.yaml.get('kpt_shape',()))!=(1,3):p.error('Expected a trained single-centre pose model')
    size=int(build['config']['image_size']); a=SimpleNamespace(imgsz=size,device=args.device,eval_batch=args.batch,workers=args.workers,confidence=confidence,match_km=match,dedup_km=dedup,same_tile_dedup_km=same,keypoint_conf=kp,oks_sigma=args.oks_sigma,beta=1.,min_precision=.85,bootstrap=1000,seed=42)
    out.mkdir(parents=True); comparison=out/'comparison'; comparison.mkdir(); history_comparison(paths,comparison)
    if sel: selection_comparison(selection,comparison)
    dataset_chart(counts,diversity,comparison); core.dump(out/'dataset_summary.json',dict(counts={s:dict(c) for s,c in counts.items()},diversity=diversity))
    metadata=dict(checkpoint=str(checkpoint),checkpoint_sha256=sha,dataset=str(data),dataset_fingerprint=fingerprint,threshold=confidence,match_km=match,dedup_km=dedup,same_tile_dedup_km=same,keypoint_conf=kp,oks_sigma=args.oks_sigma,image_size=size,parameters=sum(p.numel() for p in model.model.parameters()),checkpoint_bytes=checkpoint.stat().st_size,device=args.device,batch=args.batch,ultralytics=ultralytics.__version__,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(int(args.device)) if args.device.isdigit() and torch.cuda.is_available() else None,selection_report=str(selection) if sel else None,operating_point=str(op_path) if op else None,settings_source='saved operating point + explicit overrides' if op else 'selection report + assumed defaults / explicit overrides' if sel else 'explicit confidence or T2 defaults (unverified)',split=args.split)
    core.dump(out/'evaluation_config.json',metadata); y=out/'dataset.yaml'; y.write_text(yaml.safe_dump(dict(path=str(data),train='images/train',val='images/val',test='images/test',names={0:'cyclone'},kpt_shape=[1,3],flip_idx=[0],kpt_oks_sigmas=[args.oks_sigma]),sort_keys=False))
    projection=core.Projection(build['grid']); combined={}; all_rows=[]; summaries=['# GOES-16 cyclone detection — presentation results','',f'Checkpoint: `{checkpoint.name}` ({metadata["parameters"]:,} parameters).','',f'Images: {sum(c["images"] for c in counts.values()):,}; covered storm IDs: {sum(d["unique_storms"] for d in diversity.values())}.','']
    for split in ('val','test') if args.split=='both' else (args.split,):
        folder=out/split; folder.mkdir(); print(f'Evaluating {split} at frozen confidence {confidence:.3f}',flush=True)
        native=None if args.skip_native else core.native_metrics(model,y,split,a,folder)
        raw,scores=core.infer(model,data,samples[split],projection,a,folder); m=core.geographic_report(raw,scores,samples[split],plans[split],a,folder,confidence); m['image_size']=size
        box,boxrows=core.filtered_box_counts(raw,samples[split],confidence,size); core.dump(folder/'filtered_box_metrics.json',box); core.write_csv(folder/'filtered_box_counts.csv',boxrows,['image','scene','targets','predictions','tp','fp','fn'])
        presentation_plots(m,folder,split,native)
        if not args.no_gallery: gallery(data,samples[split],raw,folder,m,split,args.examples)
        all_rows.extend(metric_table(m,native,split,folder)); summaries.append(summary_text(m,native,split)); combined[split]=dict(geographic=m,native=native,filtered_box=box)
    core.dump(out/'metrics.json',combined); core.write_csv(out/'presentation_metrics.csv',all_rows,['split','metric','value','unit','definition']); (out/'presentation_summary.md').write_text('\n'.join(summaries)); print(f'Presentation results: {out}'); return 0


if __name__=='__main__': raise SystemExit(main())
