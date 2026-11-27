"""Additional evaluation diagnostics; no training or test-set tuning here."""
import csv
import math
from collections import Counter, defaultdict
import numpy as np


def fbeta(tp,fp,fn,beta=2.):
    d=(1+beta**2)*tp+fp+beta**2*fn
    return (1+beta**2)*tp/d if d else None


def operating_rank(row,args):
    return (fbeta(row['tp'],row['fp'],row['fn'],getattr(args,'beta',1.)) or 0.,row['recall'] or 0.,row['threshold'])


def div(a,b): return float(a/b) if b else None


def binary_extra(m):
    tp,fp,fn,tn=(m[k] for k in ('tp','fp','fn','tn'))
    den=math.sqrt((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))
    return dict(balanced_accuracy=(m['recall']+m['specificity'])/2 if m['recall'] is not None and m['specificity'] is not None else None,
                mcc=div(tp*tn-fp*fn,den),npv=div(tn,tn+fn),f2=fbeta(tp,fp,fn))


def cluster_ci(storm_counts,n=1000,seed=42):
    a=np.array([[c['matched'],c['observations']] for c in storm_counts.values()],dtype=float)
    if len(a)<2 or n==0: return None
    rng=np.random.default_rng(seed); micro=[]; macro=[]
    for _ in range(n):
        b=a[rng.integers(0,len(a),len(a))]
        micro.append(b[:,0].sum()/b[:,1].sum()); macro.append(np.mean(b[:,0]/b[:,1]))
    return dict(storms=len(a),replicates=n,micro_recall_95=np.percentile(micro,[2.5,97.5]).tolist(),macro_recall_95=np.percentile(macro,[2.5,97.5]).tolist(),warning='Exploratory cluster bootstrap; few independent storms yield unreliable intervals.')


def pyplot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    return plt


def save(plt,fig,path):
    fig.tight_layout(); fig.savefig(path,dpi=160); plt.close(fig)


def report(out,m,records,scene_rows,storms,expected,truth,scenes,tile_scores,samples,args,distance_fn):
    plt=pyplot(); errors=[r['error_km'] for r in records if r['status']=='TP']; ids=sorted(storms)
    extra=dict(f2=fbeta(m['tp'],m['fp'],m['fn']),selection_beta=getattr(args,'beta',1.),selection_fbeta=fbeta(m['tp'],m['fp'],m['fn'],getattr(args,'beta',1.)),scene_presence=binary_extra(m['scene_presence']),tile_presence=binary_extra(m['tile_presence']),storm_bootstrap=cluster_ci(storms,getattr(args,'bootstrap',1000),getattr(args,'seed',42)),worst_storm_recall=min((c['matched']/c['observations'] for c in storms.values()),default=None),independent_storms=len(storms))
    extra['recall_including_uncovered_by_radius']={k:div(v['tp'],m['coverage']['planned_target_observations']) for k,v in m['centre_radius_metrics'].items()}
    # Diagnose misses using closest valid retained prediction, not just matched detections.
    rows=[]; groups=defaultdict(Counter)
    for r in records:
        if r['status'] not in ('TP','FN'): continue
        target=dict(lat=r['true_lat'],lon=r['true_lon']); valid=[p for p in scenes.get(r['scene'],[]) if p['valid']]
        above=[p for p in valid if p['score']>=m['threshold']]
        near=lambda ps: float(distance_fn(ps,[target]).min()) if ps else None
        d=near(above); low=near(valid)
        reason='matched' if r['status']=='TP' else ('matching_conflict' if d is not None and d<=m['match_km'] else 'below_threshold_candidate' if low is not None and low<=m['match_km'] else 'no_valid_nearby_candidate')
        rows.append(dict(scene=r['scene'],sid=r['sid'],status=r['status'],reason=reason,nearest_above_threshold_km=d,nearest_any_confidence_km=low,true_lat=r['true_lat'],true_lon=r['true_lon']))
        for name,key in [('latitude_band',str(int(abs(r['true_lat'])//10)*10)+'–'+str(int(abs(r['true_lat'])//10)*10+10)),('year',r['scene'][:4]),('nature',r['nature'])]:
            groups[(name,key)].update(observations=1,matched=int(r['status']=='TP'))
    extra['miss_reasons']=dict(Counter(r['reason'] for r in rows if r['status']=='FN'))
    def write(name,rows,fields):
        with (out/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(rows)
    write('failure_analysis.csv',rows,['scene','sid','status','reason','nearest_above_threshold_km','nearest_any_confidence_km','true_lat','true_lon'])
    strata=[dict(group=g,category=k,**c,recall=div(c['matched'],c['observations'])) for (g,k),c in sorted(groups.items())]
    write('stratified_metrics.csv',strata,['group','category','observations','matched','recall'])
    for level in ('scene','tile'):
        b=m[level+'_presence']; matrix=np.array([[b['tn'],b['fp']],[b['fn'],b['tp']]])
        fig,ax=plt.subplots(); im=ax.imshow(matrix,cmap='Blues'); fig.colorbar(im,ax=ax)
        for i in range(2):
            for j in range(2): ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Absent','Present'],yticklabels=['Absent','Present'],xlabel='Predicted',ylabel='Actual',title=level+' presence (retained tiles)'); save(plt,fig,out/(level+'_confusion_matrix.png'))
    if ids:
        fig,ax=plt.subplots(figsize=(max(7,len(ids)*.55),4)); ax.bar(ids,[storms[i]['matched']/storms[i]['observations'] for i in ids]); ax.tick_params(axis='x',rotation=90); ax.set(ylabel='Recall',ylim=(0,1),title='Recall per covered storm'); save(plt,fig,out/'storm_recall.png')
    if errors:
        fig,ax=plt.subplots(); ax.hist(errors,bins='auto'); ax.axvline(np.mean(errors),ls='--',label='Mean'); ax.set(xlabel='Matched centre error (km)',ylabel='Count'); ax.legend(); save(plt,fig,out/'centre_error_histogram.png')
    fig,ax=plt.subplots(); ds=sorted(m['centre_radius_metrics'],key=int)
    for key in ('precision','recall','f1','AP'): ax.plot(list(map(int,ds)),[m['centre_radius_metrics'][d][key] for d in ds],marker='o',label=key)
    ax.set(xlabel='Matching radius (km)',ylabel='Score',ylim=(0,1.02)); ax.legend(); save(plt,fig,out/'distance_metrics.png')
    fig,ax=plt.subplots(figsize=(9,5))
    for status in ('TP','FN','FP'):
        pts=[(r.get('true_lon') if status!='FP' else r.get('pred_lon'),r.get('true_lat') if status!='FP' else r.get('pred_lat')) for r in records if r['status']==status]; pts=[p for p in pts if None not in p]
        if pts: ax.scatter(*zip(*pts),s=12,alpha=.6,label=status)
    ax.set(xlabel='Longitude',ylabel='Latitude',title='Geographic outcomes (coordinate scatter, no basemap)');
    if ax.has_data(): ax.legend()
    save(plt,fig,out/'geographic_outcomes.png')
    fig,ax=plt.subplots(); ax.hist([tile_scores[k] for k,r in samples.items() if r['objects']],bins=np.linspace(0,1,21),alpha=.6,label='Positive'); ax.hist([tile_scores[k] for k,r in samples.items() if not r['objects']],bins=np.linspace(0,1,21),alpha=.6,label='Negative'); ax.set(xlabel='Maximum tile confidence',ylabel='Tiles'); ax.legend(); save(plt,fig,out/'tile_confidence_histogram.png')
    return extra


def diversity_plots(run,diversity):
    plt=pyplot(); fig,ax=plt.subplots(); x=list(diversity); ax.bar(x,[diversity[s]['unique_storms'] for s in x]); ax.set(ylabel='Unique covered storms'); save(plt,fig,run/'storm_diversity.png')
    for split,d in diversity.items():
        rows=d['storms']; fig,ax=plt.subplots(figsize=(max(7,len(rows)*.35),4)); ax.bar([r['sid'] for r in rows],[r['tile_annotations'] for r in rows],label='Tile annotations'); ax.bar([r['sid'] for r in rows],[r['observations'] for r in rows],label='Unique observations'); ax.tick_params(axis='x',rotation=90); ax.legend(); save(plt,fig,run/f'{split}_storm_representation.png')


def training_plots(run,path):
    if not path.exists(): return
    plt=pyplot(); out=run/'training_diagnostics'; out.mkdir(exist_ok=True)
    with path.open() as f: rows=[{k.strip():v for k,v in r.items()} for r in csv.DictReader(f)]
    if not rows:return
    cols=rows[0]; x=[float(r['epoch']) for r in rows]
    for name in [k[6:] for k in cols if k.startswith('train/') and 'val/'+k[6:] in cols]:
        fig,ax=plt.subplots()
        for split in ('train','val'): ax.plot(x,[float(r[split+'/'+name]) for r in rows],label=split)
        ax.set(xlabel='Epoch',ylabel=name); ax.legend(); save(plt,fig,out/(name+'.png'))
    for name,keys in [('validation_metrics',[k for k in cols if k.startswith('metrics/')]),('learning_rate',[k for k in cols if k.startswith('lr/')])]:
        if not keys:continue
        fig,ax=plt.subplots(figsize=(9,5));
        for k in keys: ax.plot(x,[float(r[k]) for r in rows],label=k)
        ax.set(xlabel='Epoch'); ax.legend(fontsize=8); save(plt,fig,out/(name+'.png'))


def checkpoint_plots(run,rows):
    plt=pyplot(); fig,ax=plt.subplots(figsize=(max(7,len(rows)*.45),4)); x=np.arange(len(rows)); ax.plot(x,[r['f1'] or 0 for r in rows],marker='o',label='F1'); ax.plot(x,[r['recall'] or 0 for r in rows],marker='o',label='Recall'); ax.plot(x,[r.get('selection_score',0) for r in rows],marker='o',label='Selection F-beta'); ax.set(xticks=x,xticklabels=[r['checkpoint'].split('/')[-1] for r in rows],ylim=(0,1.02)); ax.tick_params(axis='x',rotation=90); ax.legend(); save(plt,fig,run/'checkpoint_comparison.png')
