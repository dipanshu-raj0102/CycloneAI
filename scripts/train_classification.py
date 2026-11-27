#!/usr/bin/env python3
"""Train Swin-T for GOES-16 cyclone intensity using classification_v1 manifests.
PyTorch >=2.3 and compatible torchvision; see --help. Only load trusted checkpoints.
"""
from __future__ import annotations
import argparse, hashlib, json, math, random, time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import swin_t, Swin_T_Weights
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
    confusion_matrix, cohen_kappa_score, classification_report)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5']
SPLITS = ['train','val','test']
BOUNDS = np.array([34,64,83,96,113,137])
RANGES = np.array([[180,330],[180,280],[-30,70]], np.float32)
MEAN = torch.tensor([.485,.456,.406])[:,None,None]
STD = torch.tensor([.229,.224,.225])[:,None,None]
WIND_SCALE = 100.


def dump(path, value):
    path = Path(path); tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); tmp.replace(path)


def safe_path(root, rel):
    p = (root / rel).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise ValueError(f'Invalid/missing data path: {rel}')
    return p


def load_data(root, source):
    root = root.resolve(); key = 'image' if source == 'png' else 'kelvin'
    summary = json.loads((root/'summary.json').read_text())
    mapping = json.loads((root/'storm_splits.json').read_text())
    classes = json.loads((root/'classes.json').read_text())
    if classes != {str(i): c for i,c in enumerate(CLASSES)}: raise ValueError('Class ordering mismatch')
    if summary.get('storm_leakage') is not False or summary.get('scene_leakage') is not False: raise ValueError('Dataset not certified leakage-free')
    if any(not np.array_equal(np.asarray(summary['normalization'][k]), r) for k,r in zip(['C13','C08','C13-C08'], RANGES)): raise ValueError('Normalization mismatch')
    digest = hashlib.sha256(); frames = {}; seen = set(); paths = set(); storms = {}; scenes = {}
    required = {'sample_id','split','sid','scene_id','wind_kt','class_id','class_name',key}
    for name in ['summary.json','storm_splits.json','classes.json','manifest.csv',*[s+'.csv' for s in SPLITS]]:
        digest.update(name.encode()); digest.update((root/name).read_bytes())
    for s in SPLITS:
        df = pd.read_csv(root/f'{s}.csv', dtype={'sample_id':str,'sid':str,'scene_id':str}, keep_default_na=False)
        if not required <= set(df.columns) or df.empty: raise ValueError(f'Missing columns or empty split: {s}')
        for r in df.to_dict('records'):
            w = float(r['wind_kt']); c = int(r['class_id'])
            if not np.isfinite(w) or not 0<w<=250 or c != float(r['class_id']) or c != np.searchsorted(BOUNDS,w,side='right') or r['class_name'] != CLASSES[c]: raise ValueError(f'Invalid label: {r}')
            if not all(r[k] for k in ('sample_id','sid','scene_id')) or r['sample_id'] in seen or r['split']!=s or mapping.get(r['sid'])!=s or storms.setdefault(r['sid'],s)!=s or scenes.setdefault(r['scene_id'],s)!=s: raise ValueError(f'Duplicate or split leakage: {r}')
            p = safe_path(root,r[key]); st=p.stat()
            if p in paths: raise ValueError(f'Duplicate source crop: {p}')
            seen.add(r['sample_id']); paths.add(p)
            digest.update(f'{r[key]}:{st.st_size}\n'.encode())
        frames[s] = df
        print(f'{s}: {len(df)} crops, {df.sid.nunique()} storms, class counts {np.bincount(df.class_id.astype(int),minlength=7).tolist()}',flush=True)
    manifest=pd.read_csv(root/'manifest.csv', dtype={'sample_id':str,'sid':str,'scene_id':str},keep_default_na=False)
    combined=pd.concat(list(frames.values()),ignore_index=True)
    pd.testing.assert_frame_equal(manifest.sort_values('sample_id').reset_index(drop=True), combined[manifest.columns].sort_values('sample_id').reset_index(drop=True), check_dtype=False)
    counts=np.bincount(frames['train'].class_id.astype(int),minlength=7)
    if not counts.all(): raise ValueError(f'Missing training classes: {[CLASSES[i] for i in np.flatnonzero(counts==0)]}. Review preprocessing before training.')
    for s in ['val','test']:
        absent=set(range(7))-set(frames[s].class_id.astype(int))
        if absent: print(f'WARNING: {s} lacks {[CLASSES[i] for i in sorted(absent)]}; report includes both supported-class and fixed-seven macro F1.',flush=True)
    return frames, counts, digest.hexdigest(), int(summary['image_size'])


class Crops(Dataset):
    def __init__(self,root,frame,source,size,native_size,augment=False):
        self.root=root; self.rows=frame.to_dict('records'); self.source=source; self.size=size; self.native=native_size; self.augment=augment
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]
        if self.source=='png':
            with Image.open(self.root/r['image']) as im:
                if im.mode!='RGB': raise ValueError(f'Expected RGB: {r["image"]}')
                a=np.asarray(im,dtype=np.float32)/255.
        else:
            with np.load(self.root/r['kelvin'],allow_pickle=False) as z:
                c13,c08,valid=z['c13'],z['c08'],z['valid'].astype(bool)
            if c13.shape!=c08.shape or valid.shape!=c13.shape or not np.isfinite(c13[valid]).all() or not np.isfinite(c08[valid]).all(): raise ValueError('Corrupt Kelvin crop')
            a=np.stack([c13,c08,c13-c08],axis=-1).astype(np.float32)
            a=np.where(valid[...,None],a,RANGES[:,0]); a=np.clip((a-RANGES[:,0])/(RANGES[:,1]-RANGES[:,0]),0,1)
        if a.shape!=(self.native,self.native,3) or not np.isfinite(a).all(): raise ValueError(f'Bad crop: {r["sample_id"]}')
        x=torch.from_numpy(np.ascontiguousarray(a.transpose(2,0,1)))
        if self.size!=self.native: x=F.interpolate(x[None],size=(self.size,self.size),mode='bilinear',align_corners=False,antialias=True)[0]
        # Rotation preserves brightness-temperature values and the crop center; no RGB jitter or mixup.
        if self.augment: x=torch.rot90(x,int(torch.randint(4,()).item()),dims=(1,2))
        return (x-MEAN)/STD, int(r['class_id']), float(r['wind_kt'])/WIND_SCALE, i


class CycloneSwin(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__(); self.backbone=swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        d=self.backbone.head.in_features; self.backbone.head=nn.Identity()
        self.classifier=nn.Sequential(nn.Dropout(.2),nn.Linear(d,7))
        self.regressor=nn.Sequential(nn.Dropout(.2),nn.Linear(d,1))
    def forward(self,x):
        x=self.backbone(x); return self.classifier(x), F.softplus(self.regressor(x).squeeze(-1))


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def seed_worker(worker):
    np.random.seed(torch.initial_seed()%2**32); random.seed(torch.initial_seed()%2**32)


def make_loader(dataset,args,training,device,epoch=0):
    g=torch.Generator().manual_seed(args.seed+epoch+(0 if training else 100000))
    return DataLoader(dataset,batch_size=args.batch_size,shuffle=training,num_workers=args.workers,
                      pin_memory=device.type=='cuda',worker_init_fn=seed_worker,generator=g,drop_last=False)


def losses(logits,pwind,y,wind,weights,args):
    ce=F.cross_entropy(logits.float(),y,weight=weights,reduction='none',label_smoothing=args.label_smoothing)
    reg=F.smooth_l1_loss(pwind.float(),wind.float(),beta=.1,reduction='none')
    return ce+args.regression_weight*reg,ce,reg


def run_epoch(model,loader,device,weights,args,optimizer=None,scaler=None):
    training=optimizer is not None; model.train(training); amp=device.type=='cuda' and not args.no_amp
    totals=np.zeros(3); results=[]; n=0
    if training: optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(training):
        for step,(x,y,w,idx) in enumerate(loader):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True); w=w.to(device,non_blocking=True,dtype=torch.float32)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
                logits,pw=model(x); loss,ce,reg=losses(logits,pw,y,w,weights,args)
            if not torch.isfinite(loss).all(): raise FloatingPointError('Nonfinite training/evaluation loss')
            if training:
                first=(step//args.accumulation)*args.accumulation*args.batch_size
                group_n=min(args.accumulation*args.batch_size,len(loader.dataset)-first)
                scaler.scale(loss.sum()/group_n).backward()
                if (step+1)%args.accumulation==0 or step+1==len(loader):
                    scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.)
                    scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
            totals+=np.array([loss.detach().sum().item(),ce.detach().sum().item(),reg.detach().sum().item()]); n+=len(y)
            prob=logits.detach().float().softmax(1).cpu().numpy(); pred_w=pw.detach().float().cpu().numpy()*WIND_SCALE
            for j,i in enumerate(idx.tolist()): results.append((i,prob[j],float(pred_w[j])))
            if training and ((step+1)%25==0 or step+1==len(loader)):
                print(f'  batch {step+1}/{len(loader)} loss={totals[0]/n:.4f}',flush=True)
    results.sort(key=lambda r:r[0]); rows=loader.dataset.rows
    if [r[0] for r in results]!=list(range(len(rows))): raise RuntimeError('Missing or duplicate evaluation predictions')
    df=pd.DataFrame(rows)[['sample_id','sid','scene_id','class_id','wind_kt']].copy()
    probs=np.stack([r[1] for r in results]); df['pred_class']=probs.argmax(1); df['pred_wind_kt']=[r[2] for r in results]
    for i,c in enumerate(CLASSES): df['prob_'+c]=probs[:,i]
    m=metrics(df); m.update(dict(zip(['loss','class_loss','wind_loss'],(totals/n).tolist())))
    return m,df


def metrics(df):
    y=df.class_id.to_numpy(int); p=df.pred_class.to_numpy(int); err=df.pred_wind_kt.to_numpy(float)-df.wind_kt.to_numpy(float)
    _,rec,f1,support=precision_recall_fscore_support(y,p,labels=list(range(7)),zero_division=0)
    kappa=cohen_kappa_score(y,p,labels=list(range(7)),weights='quadratic') if len(np.unique(np.r_[y,p]))>1 else np.nan
    correct=(y==p).astype(float); abs_err=np.abs(err)
    storm=pd.DataFrame({'sid':df.sid,'correct':correct,'abs_err':abs_err}).groupby('sid').mean()
    return {'accuracy':float(accuracy_score(y,p)), 'macro_f1':float(f1.mean()),'macro_f1_supported':float(f1[support>0].mean()),
            'balanced_accuracy':float(rec[support>0].mean()),'quadratic_kappa':float(kappa) if np.isfinite(kappa) else None,
            'within_one_category':float((np.abs(y-p)<=1).mean()),'category_mae':float(np.abs(y-p).mean()),
            'wind_mae_kt':float(abs_err.mean()),'wind_rmse_kt':float(np.sqrt(np.mean(err**2))), 'wind_bias_kt':float(err.mean()),
            'storm_mean_accuracy':float(storm.correct.mean()),'storm_mean_wind_mae_kt':float(storm.abs_err.mean()),
            'head_agreement':float((np.searchsorted(BOUNDS,df.pred_wind_kt,side='right')==p).mean())}


def save_eval(out,name,m,df):
    folder=out/name; folder.mkdir(exist_ok=True); dump(folder/'metrics.json',m); df.to_csv(folder/'predictions.csv',index=False)
    y,p=df.class_id,df.pred_class
    report=classification_report(y,p,labels=list(range(7)),target_names=CLASSES,output_dict=True,zero_division=0)
    pd.DataFrame(report).T.to_csv(folder/'classification_report.csv')
    cm=confusion_matrix(y,p,labels=list(range(7))); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(folder/'confusion_matrix.csv')
    for normalized in (False,True):
        z=cm if not normalized else np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0)
        fig,ax=plt.subplots(figsize=(8,7)); im=ax.imshow(z,cmap='Blues',vmin=0,vmax=1 if normalized else max(1,cm.max()))
        for i in range(7):
            for j in range(7): ax.text(j,i,f'{z[i,j]:.2f}' if normalized else str(z[i,j]),ha='center',va='center',color='white' if z[i,j]>im.norm.vmax*.55 else 'black')
        ax.set(xticks=range(7),yticks=range(7),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title=f'{name}: confusion matrix'+(' (row normalized)' if normalized else ''))
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(folder/('confusion_matrix_normalized.png' if normalized else 'confusion_matrix.png'),dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6)); ax.scatter(df.wind_kt,df.pred_wind_kt,s=10,alpha=.45); hi=max(df.wind_kt.max(),df.pred_wind_kt.max())+5; ax.plot([0,hi],[0,hi],'--'); ax.set(xlim=(0,hi),ylim=(0,hi),xlabel='IBTrACS wind (kt)',ylabel='Predicted wind (kt)',title=f'Wind MAE: {m["wind_mae_kt"]:.2f} kt'); fig.tight_layout(); fig.savefig(folder/'wind_scatter.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar(CLASSES,[report[c]['f1-score'] for c in CLASSES]); ax.set(ylim=(0,1),ylabel='F1',title=f'{name}: class-wise F1'); fig.tight_layout(); fig.savefig(folder/'per_class_f1.png',dpi=180); plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4)); ax.hist(df.pred_wind_kt-df.wind_kt,bins=30); ax.axvline(0,linestyle='--'); ax.set(xlabel='Predicted − actual wind (kt)',ylabel='Crops',title='Wind-speed residuals'); fig.tight_layout(); fig.savefig(folder/'wind_residuals.png',dpi=180); plt.close(fig)
    print(f'{name}: {json.dumps(m)}',flush=True)


def save_history(out,history):
    df=pd.DataFrame(history); df.to_csv(out/'history.csv',index=False)
    for name,cols in {'loss':['train_loss','val_loss'],'macro_f1':['train_macro_f1','val_macro_f1'],'accuracy':['train_accuracy','val_accuracy'],'wind_mae':['train_wind_mae_kt','val_wind_mae_kt'],'learning_rate':['backbone_lr','head_lr']}.items():
        fig,ax=plt.subplots(figsize=(8,4))
        for c in cols: ax.plot(df.epoch,df[c],label=c)
        ax.set(xlabel='Epoch',title=name.replace('_',' ').title()); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=160); plt.close(fig)


def save_checkpoint(path,state):
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('../data/processed/classification_v1')); p.add_argument('--output',type=Path,default=Path('../runs/classification/swin_t'))
    p.add_argument('--source',choices=['png','kelvin'],default='png'); p.add_argument('--image-size',type=int,default=384); p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=16); p.add_argument('--accumulation',type=int,default=2)
    p.add_argument('--workers',type=int,default=4); p.add_argument('--backbone-lr',type=float,default=1e-5); p.add_argument('--head-lr',type=float,default=5e-4); p.add_argument('--weight-decay',type=float,default=.01); p.add_argument('--warmup',type=int,default=5); p.add_argument('--patience',type=int,default=15); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--regression-weight',type=float,default=.5); p.add_argument('--label-smoothing',type=float,default=0.); p.add_argument('--no-pretrained',action='store_true'); p.add_argument('--no-amp',action='store_true'); p.add_argument('--no-augmentation',action='store_true')
    p.add_argument('--device',default='auto'); p.add_argument('--resume',action='store_true'); p.add_argument('--evaluate-test',action='store_true',help='Evaluate selected best checkpoint on held-out test after training'); p.add_argument('--eval-only',action='store_true',help='Load existing best checkpoint, no training'); p.add_argument('--validate-only',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.batch_size,a.accumulation,a.patience)<1 or a.workers<0 or a.image_size<32 or a.image_size%32 or not 0<=a.label_smoothing<1 or a.warmup<0 or min(a.backbone_lr,a.head_lr)<=0 or min(a.regression_weight,a.weight_decay)<0: p.error('Invalid hyperparameters')
    a.data=a.data.resolve(); a.output=a.output.resolve()
    if a.output==a.data or a.output.is_relative_to(a.data) or a.data.is_relative_to(a.output): p.error('Output must be separate from dataset')
    frames,counts,fingerprint,native=load_data(a.data,a.source)
    if a.validate_only: print('Manifest validation passed. Pixel contents not exhaustively checked.'); return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and not torch.cuda.is_available()): p.error('Requested device unavailable; use CPU or CUDA')
    print(f'Device: {device}; visible GPUs: {torch.cuda.device_count()}; single-device training',flush=True)
    seed_all(a.seed); torch.set_float32_matmul_precision('high')
    out=a.output; best_path=out/'best.pt'; last_path=out/'last.pt'
    if a.resume and a.eval_only: p.error('Use only one of --resume or --eval-only')
    existing=a.resume or a.eval_only
    if existing and not (best_path if a.eval_only else last_path).is_file(): raise FileNotFoundError('Requested checkpoint does not exist')
    if not existing and out.exists() and any(out.iterdir()): raise FileExistsError('Output not empty; choose a fresh run or --resume')
    out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ('data','output','resume','eval_only','evaluate_test','validate_only','workers','device')}; config['dataset_fingerprint']=fingerprint
    ckpt=torch.load(best_path if a.eval_only else last_path,map_location='cpu',weights_only=False) if existing else None
    if ckpt and (ckpt['config']['dataset_fingerprint']!=fingerprint or ckpt['config']['source']!=a.source or ckpt['config']['image_size']!=a.image_size): raise ValueError('Checkpoint data/input mismatch')
    if a.resume and ckpt['config']!=config: raise ValueError('Resume requires same training configuration, including total epochs')
    weights=counts.astype(float)**(-.5); weights/=np.dot(weights,counts)/counts.sum(); weights=torch.tensor(weights,dtype=torch.float32,device=device)
    sets={s:Crops(a.data,frames[s],a.source,a.image_size,native,s=='train' and not a.no_augmentation) for s in SPLITS}
    model=CycloneSwin(pretrained=not a.no_pretrained and not existing).to(device)
    if ckpt: model.load_state_dict(ckpt['model'])
    if not a.eval_only:
        optim=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':a.backbone_lr},{'params':list(model.classifier.parameters())+list(model.regressor.parameters()),'lr':a.head_lr}],weight_decay=a.weight_decay)
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and not a.no_amp)
        history=[]; start=1; best=-1.; best_epoch=0; stale=0
        if a.resume:
            optim.load_state_dict(ckpt['optimizer']); scaler.load_state_dict(ckpt['scaler']); history=ckpt['history']; start=ckpt['epoch']+1; best=ckpt['best']; best_epoch=ckpt['best_epoch']; stale=ckpt['stale']
            if not best_path.is_file(): raise FileNotFoundError('Resume also requires best.pt')
        dump(out/'config.json',dict(config,classes=CLASSES,normalization=RANGES.tolist(),imagenet_mean=MEAN.flatten().tolist(),imagenet_std=STD.flatten().tolist(),wind_scale=WIND_SCALE,class_weights=weights.cpu().tolist(),torch_version=str(torch.__version__)))
        warm=min(a.warmup,max(0,a.epochs-1))
        for epoch in range(start,a.epochs+1):
            if stale>=a.patience: break
            seed_all(a.seed+epoch)
            factor=(.1+.9*epoch/warm) if epoch<=warm else (.05+.95*(1+math.cos(math.pi*(epoch-warm-1)/max(1,a.epochs-warm-1)))/2)
            for group,base in zip(optim.param_groups,[a.backbone_lr,a.head_lr]): group['lr']=base*factor
            tick=time.monotonic(); print(f'Epoch {epoch}/{a.epochs}',flush=True)
            train,_=run_epoch(model,make_loader(sets['train'],a,True,device,epoch),device,weights,a,optim,scaler)
            val,_=run_epoch(model,make_loader(sets['val'],a,False,device),device,weights,a)
            history.append({'epoch':epoch,'seconds':time.monotonic()-tick,'backbone_lr':optim.param_groups[0]['lr'],'head_lr':optim.param_groups[1]['lr'],**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()}})
            score=val['macro_f1']; improved=score>best+1e-6
            if improved: best=score; best_epoch=epoch; stale=0
            else: stale+=1
            state={'model':model.state_dict(),'optimizer':optim.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'config':config,'history':history}
            if improved: save_checkpoint(best_path,{'model':state['model'],'config':config,'epoch':epoch,'best':best})
            save_checkpoint(last_path,state); save_history(out,history)
            print(f'  train loss={train["loss"]:.4f} val F1={score:.4f} val acc={val["accuracy"]:.4f} wind MAE={val["wind_mae_kt"]:.2f} kt best epoch={best_epoch}',flush=True)
    best_ckpt=torch.load(best_path,map_location='cpu',weights_only=False); model.load_state_dict(best_ckpt['model'])
    # Restore loss settings for evaluation-only runs.
    for key in ('regression_weight','label_smoothing'): setattr(a,key,best_ckpt['config'][key])
    val,df=run_epoch(model,make_loader(sets['val'],a,False,device),device,weights,a); save_eval(out,'validation',val,df)
    if a.evaluate_test:
        test,df=run_epoch(model,make_loader(sets['test'],a,False,device),device,weights,a); save_eval(out,'test',test,df)
    dump(out/'selection.json',{'best_epoch':best_ckpt['epoch'],'validation_macro_f1':best_ckpt['best'],'test_evaluated_this_run':a.evaluate_test})

if __name__=='__main__': main()
