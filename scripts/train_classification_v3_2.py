#!/usr/bin/env python3
"""Train V3.2 models on V3.1 GOES-16 Kelvin crops; independent storm-disjoint evaluation.
PyTorch >=2.3, matching torchvision, numpy, pandas, scikit-learn, matplotlib.
Only load trusted checkpoints. Test evaluation is opt-in.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.models import swin_t, Swin_T_Weights, convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.ops import StochasticDepth
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, cohen_kappa_score, classification_report
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
CLASSES=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5']; SPLITS=['train','val','test']; BOUNDS=np.array([34,64,83,96,113,137]); WIND_SCALE=100.

def dump(path, value):
    path=Path(path); tmp=path.with_suffix(path.suffix+'.tmp'); tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); tmp.replace(path)

def read(path): return json.loads(path.read_text())
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
def seed_worker(worker):
    np.random.seed(torch.initial_seed()%2**32); random.seed(torch.initial_seed()%2**32)

def load_data(a):
    root=a.data; meta=root/'metadata'; man=root/'manifests'
    classes=read(meta/'classes.json'); norm=read(meta/'normalization.json'); cfg=read(meta/'preprocessing_config.json'); mapping=read(man/'storm_splits.json')
    if classes!={str(i):c for i,c in enumerate(CLASSES)}: raise ValueError('Class mapping mismatch')
    if norm['channels']!=['C13','C08','C13-C08'] or norm['method']!='train_valid_joint_pixels_zscore': raise ValueError('Unexpected normalization schema')
    native=int(cfg['master_size']); crop=int(norm['crop_size']); mean=np.asarray(norm['mean'],np.float32); std=np.asarray(norm['std'],np.float32)
    if mean.shape!=(3,) or std.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std<=0).any(): raise ValueError('Invalid normalization')
    if crop>native or (native-crop)%2 or a.center_shift_px>(native-crop)//2: raise ValueError('Crop/shift exceeds master grid')
    names=['metadata/classes.json','metadata/normalization.json','metadata/preprocessing_config.json','manifests/storm_splits.json']+[f'manifests/{s}.csv' for s in SPLITS]
    feats=[]; feature_map={}; fnorm={}
    if a.physical:
        names+=['metadata/physical_feature_normalization.json','manifests/physical_features.csv']
        fnorm=read(meta/'physical_feature_normalization.json')['features']
        feats=sorted(k for k,v in fnorm.items() if v['mean'] is not None and v['std'] is not None and np.isfinite(v['mean']) and np.isfinite(v['std']) and v['std']>1e-8)
        if not feats: raise ValueError('No nonconstant physical features')
        ff=pd.read_csv(man/'physical_features.csv',dtype={'sample_id':str,'split':str})
        if ff.sample_id.duplicated().any(): raise ValueError('Duplicate feature rows')
        for r in ff.to_dict('records'):
            vals=np.array([r[k] for k in feats],dtype=np.float32); valid=np.isfinite(vals); mu=np.array([fnorm[k]['mean'] for k in feats]); sd=np.array([fnorm[k]['std'] for k in feats]); v=np.where(valid,(vals-mu)/sd,0)
            feature_map[r['sample_id']]=(r['split'],np.r_[np.clip(v,-8,8),~valid].astype(np.float32))
    digest=hashlib.sha256(); frames={}; seen=set(); fixes=set(); paths=set(); storms={}; scenes={}; hashes={}
    for name in names: digest.update(name.encode()); digest.update((root/name).read_bytes())
    for s in SPLITS:
        df=pd.read_csv(man/f'{s}.csv',dtype={k:str for k in ['sample_id','sid','scene_id','label_time','tensor_sha256']},keep_default_na=False)
        required={'sample_id','sid','scene_id','label_time','split','kelvin','wind_kt','class_id','class_name','tensor_sha256','suggested_label_weight','quality_tier','label_provenance','view_zenith_deg'}
        if df.empty or not required<=set(df): raise ValueError(f'Empty or invalid {s} manifest')
        for r in df.to_dict('records'):
            i=r['sample_id']; sid=r['sid']; scene=r['scene_id']; fix=(sid,r['label_time']); w=float(r['wind_kt']); c=int(r['class_id']); h=r['tensor_sha256']
            if not all([i,sid,scene,r['label_time']]) or not 0<w<=250 or float(r['class_id'])!=c or c!=np.searchsorted(BOUNDS,w,side='right') or r['class_name']!=CLASSES[c]: raise ValueError(f'Invalid label: {i}')
            if r['split']!=s or mapping.get(sid)!=s or storms.setdefault(sid,s)!=s or scenes.setdefault(scene,s)!=s or i in seen or fix in fixes: raise ValueError(f'Split leakage or duplicate: {i}')
            p=(root/r['kelvin']).resolve()
            if not p.is_relative_to(root) or not p.is_file() or p in paths: raise ValueError(f'Invalid tensor path: {p}')
            if len(h)!=64 or any(ch not in '0123456789abcdef' for ch in h): raise ValueError(f'Invalid tensor digest: {i}')
            if hashes.setdefault(h,s)!=s: raise ValueError(f'Identical tensor across splits: {i}')
            if a.verify_hashes:
                with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=h: raise ValueError(f'Tensor checksum mismatch: {i}')
            if a.physical and (i not in feature_map or feature_map[i][0]!=s): raise ValueError(f'Physical feature join mismatch: {i}')
            if r['quality_tier'] not in ('high','acceptable') or not 0<float(r['suggested_label_weight'])<=1: raise ValueError(f'Invalid quality metadata: {i}')
            seen.add(i); fixes.add(fix); paths.add(p); digest.update(f'{r["kelvin"]}:{p.stat().st_size}:{h}'.encode())
        frames[s]=df
        print(f'{s}: {len(df)} samples, {df.sid.nunique()} storms; class counts {np.bincount(df.class_id.astype(int),minlength=7).tolist()}; class storm counts {df.groupby("class_id").sid.nunique().reindex(range(7),fill_value=0).tolist()}',flush=True)
        missing=set(range(7))-set(df.class_id.astype(int));
        if missing: print(f'WARNING: {s} missing {[CLASSES[i] for i in sorted(missing)]}',flush=True)
        if s=='train' and missing: raise ValueError('Training does not cover all seven classes')
    return frames,dict(native=native,crop=crop,mean=mean.tolist(),std=std.tolist(),feature_names=feats,feature_stats={k:fnorm[k] for k in feats}),feature_map,digest.hexdigest()

class Crops(Dataset):
    def __init__(self,a,df,spec,features,train=False): self.a=a; self.rows=df.to_dict('records'); self.spec=spec; self.features=features; self.train=train
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; a=self.a; spec=self.spec; n=spec['native']; c=spec['crop']; st=(n-c)//2; dy=dx=0
        if self.train and not a.no_augmentation and a.center_shift_px and not a.physical:
            dy,dx=np.random.randint(-a.center_shift_px,a.center_shift_px+1,size=2)
        with np.load(a.data/r['kelvin'],allow_pickle=False) as z:
            ir,wv,m1,m2=[z[k] for k in ('c13','c08','mask_c13','mask_c08')]
        if any(v.shape!=(n,n) for v in (ir,wv,m1,m2)): raise ValueError(f'Bad tensor shape: {r["sample_id"]}')
        # Normalize the master grid before mask-aware spatial transformations.
        valid=m1.astype(bool)&m2.astype(bool)
        if not valid.any() or not np.isfinite(ir[valid]).all() or not np.isfinite(wv[valid]).all(): raise ValueError(f'Bad pixels: {r["sample_id"]}')
        x=np.stack([ir,wv,ir-wv]).astype(np.float32)
        x=(x-np.asarray(spec['mean'],np.float32)[:,None,None])/np.asarray(spec['std'],np.float32)[:,None,None]
        x=np.where(valid[None],x,0)
        if a.channels=='c13': x=np.repeat(x[:1],3,axis=0)
        elif a.channels=='c13_c08': x[2]=0
        x=torch.from_numpy(np.ascontiguousarray(x)); mask=torch.from_numpy(valid.astype(np.float32))[None]
        center_x=x[:,st:st+c,st:st+c]; center_mask=mask[:,st:st+c,st:st+c]
        augment=self.train and not a.no_augmentation
        if augment and a.rotation=='arbitrary':
            # Sample the rotated primary crop directly from the master, not a padded primary crop.
            # align_corners=False: translation in source pixels is 2*shift/n.
            ang=float(np.random.uniform(0,2*math.pi)); co=math.cos(ang)*c/n; si=math.sin(ang)*c/n
            theta=x.new_tensor([[co,-si,2*dx/n],[si,co,2*dy/n]])[None]
            grid=F.affine_grid(theta,(1,3,c,c),align_corners=False)
            den=F.grid_sample(mask[None],grid,align_corners=False)[0]
            num=F.grid_sample(x[None],grid,align_corners=False)[0]
            x=torch.where(den>=.99,num/den.clamp_min(1e-6),0.); mask=(den>=.99).float()
            # 1000 km master cannot contain all corners of an arbitrarily rotated 800 km crop.
            # Reject extreme validity loss and fall back to the centered, unrotated crop.
            if mask.mean().item()<a.aug_min_valid:
                x,mask=center_x,center_mask
                k=int(torch.randint(4,()).item()); x=torch.rot90(x,k,(1,2)); mask=torch.rot90(mask,k,(1,2))
        else:
            sl=np.s_[st+dy:st+dy+c,st+dx:st+dx+c]; x=x[:,sl[0],sl[1]]; mask=mask[:,sl[0],sl[1]]
            if augment and mask.mean().item()<a.aug_min_valid:
                # Reconstruct the centered crop if a translation reaches invalid data.
                x,mask=center_x,center_mask
            if augment and a.rotation=='right-angle':
                k=int(torch.randint(4,()).item()); x=torch.rot90(x,k,(1,2)); mask=torch.rot90(mask,k,(1,2))
        if a.image_size!=c:
            kwargs=dict(size=(a.image_size,a.image_size),mode='bilinear',align_corners=False,antialias=True)
            den=F.interpolate(mask[None],**kwargs)[0]; x=F.interpolate(x[None],**kwargs)[0]/den.clamp_min(1e-6); x=torch.where(den>=.99,x,0.)
        feat=self.features[r['sample_id']][1] if a.physical else np.zeros(0,np.float32)
        quality=float(r['suggested_label_weight']) if a.label_weighting else 1.
        if a.quality_weighting and r['quality_tier']=='acceptable': quality*=.9
        return x.contiguous(),torch.from_numpy(feat.copy()),int(r['class_id']),float(r['wind_kt'])/WIND_SCALE,quality,i

class CycloneModel(nn.Module):
    def __init__(self,feature_dim,dropout=.4,pretrained=True,architecture='swin_t',stochastic_depth=.2):
        super().__init__(); self.architecture=architecture
        if architecture=='swin_t':
            self.backbone=swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
            d=self.backbone.head.in_features; self.backbone.head=nn.Identity()
        else:
            self.backbone=convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            d=self.backbone.classifier[-1].in_features; self.backbone.classifier[-1]=nn.Identity()
        depth=[m for m in self.backbone.modules() if isinstance(m,StochasticDepth)]
        for i,m in enumerate(depth): m.p=stochastic_depth*i/max(1,len(depth)-1)
        self.physical=nn.Sequential(nn.Linear(feature_dim,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(dropout)) if feature_dim else None
        self.shared=nn.Sequential(nn.LayerNorm(d+(128 if feature_dim else 0)),nn.Linear(d+(128 if feature_dim else 0),256),nn.GELU(),nn.Dropout(dropout))
        self.ordinal=nn.Linear(256,6); self.wind=nn.Linear(256,1)
    def forward(self,x,f):
        z=self.backbone(x)
        if self.physical is not None: z=torch.cat([z,self.physical(f)],dim=1)
        z=self.shared(z); raw=self.ordinal(z).float(); logits=torch.cat([raw[:,:1],raw[:,:1]-F.softplus(raw[:,1:]).cumsum(1)],dim=1)
        return logits,F.softplus(self.wind(z).float().squeeze(1))
    def stage_modules(self):
        # torchvision stages: stem+stage1, merge+stage2, merge+stage3, merge+stage4.
        f=self.backbone.features
        final=self.backbone.norm if self.architecture=='swin_t' else self.backbone.classifier
        return [list(f[0:2]),list(f[2:4]),list(f[4:6]),list(f[6:8])+[final]]
    def set_phase(self,a,epoch):
        if a.fine_tune=='heads-only': count=0
        elif a.fine_tune=='last-stage': count=1 if epoch>a.freeze_epochs else 0
        elif a.fine_tune=='full': count=4 if epoch>a.freeze_epochs else 0
        else: count=0 if epoch<=a.freeze_epochs else (1 if epoch<=a.stage3_after else 2)
        self.backbone.requires_grad_(False)
        for stage in self.stage_modules()[4-count:]:
            for module in stage: module.requires_grad_(True)
        return count
    def frozen_eval(self):
        # Disable stochastic depth in frozen stages even when model.train() was called.
        for stage in self.stage_modules():
            for module in stage:
                if not any(p.requires_grad for p in module.parameters()): module.eval()

def optimizer_groups(model,a):
    groups=[]; seen=set()
    def add(name,params,lr):
        params=list(params)
        for p in params:
            if id(p) in seen: raise RuntimeError('Duplicate optimizer parameter')
            seen.add(id(p))
        for decay in (True,False):
            chosen=[p for p in params if (p.ndim>1)==decay]
            if chosen: groups.append(dict(params=chosen,lr=lr,base_lr=lr,weight_decay=a.weight_decay if decay else 0.,group_name=name))
    for i,stage in enumerate(model.stage_modules()):
        add(f'stage{i+1}',(p for m in stage for p in m.parameters()),a.backbone_lr*a.layer_decay**(3-i))
    add('heads',(p for k,p in model.named_parameters() if not k.startswith('backbone.')),a.head_lr)
    if seen!={id(p) for p in model.parameters()}: raise RuntimeError('Optimizer groups do not cover model')
    return groups

def ordinal_probs(logits):
    q=logits.float().sigmoid(); return torch.cat([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],1).clamp_min(0)

def soft_targets(y,w,tau,mix):
    hard=(y[:,None]>torch.arange(6,device=y.device)).float()
    if tau<=0 or mix<=0: return hard
    # Half-knot boundary preserves the integer label convention at the thresholds.
    b=torch.as_tensor(BOUNDS,device=w.device,dtype=torch.float32)-.5
    return (1-mix)*hard+mix*torch.sigmoid((w.float()[:,None]*WIND_SCALE-b)/tau)

def loss_fn(logits,pw,y,w,a,train=True):
    target=soft_targets(y,w,a.soft_tau if train else 0,a.soft_mix)
    cl=F.binary_cross_entropy_with_logits(logits.float(),target,reduction='none').mean(1)
    cat=torch.cat([1-target[:,:1],target[:,:-1]-target[:,1:],target[:,-1:]],1).clamp_min(0)
    nll=-(cat*ordinal_probs(logits).clamp_min(1e-7).log()).sum(1)
    cl=cl+a.class_nll_weight*nll
    reg=F.smooth_l1_loss(pw.float(),w.float(),beta=.1,reduction='none')
    return cl+a.regression_weight*reg,cl,reg

def sample_weights(df,a):
    counts=df.groupby('class_id').sample_id.transform('count').to_numpy(float)
    within=df.groupby(['class_id','sid']).sample_id.transform('count').to_numpy(float)
    if a.sampling=='legacy':
        weight=counts**(-a.class_alpha)*within**(-a.storm_alpha)
    else:
        # Exact hierarchical mass: P(class) ∝ N_c^(1-alpha), P(storm|class) ∝ N_sc^(1-beta).
        # Dividing by N_sc distributes each storm's mass uniformly among its fixes.
        norm=pd.Series(within**(-a.storm_alpha),index=df.index).groupby(df.class_id).transform('sum').to_numpy()
        weight=counts**(1-a.class_alpha)*within**(-a.storm_alpha)/norm
    return weight/weight.sum()

def make_loader(ds,a,device,epoch=0,train=False):
    g=torch.Generator().manual_seed(a.seed+epoch+(0 if train else 100000)); sampler=None
    if train and (a.class_alpha or a.storm_alpha):
        weight=sample_weights(pd.DataFrame(ds.rows),a)
        sampler=WeightedRandomSampler(torch.as_tensor(weight,dtype=torch.double),len(ds),replacement=True,generator=g)
    return DataLoader(ds,batch_size=a.batch_size,shuffle=train and sampler is None,sampler=sampler,num_workers=a.workers,pin_memory=device.type=='cuda',worker_init_fn=seed_worker,generator=g)

@torch.no_grad()
def update_ema(ema,model,decay):
    for ep,p in zip(ema.parameters(),model.parameters()): ep.lerp_(p.detach(),1-decay)
    for eb,b in zip(ema.buffers(),model.buffers()): eb.copy_(b)

def run_epoch(model,loader,device,a,optimizer=None,scaler=None,ema=None):
    train=optimizer is not None; model.train(train)
    if train: model.frozen_eval()
    amp=device.type=='cuda' and not a.no_amp; dtype=torch.bfloat16 if amp and torch.cuda.is_bf16_supported() else torch.float16
    totals=np.zeros(3); hard_total=0.; results=[]; n=0
    if train: optimizer.zero_grad(set_to_none=True)
    with torch.set_grad_enabled(train):
        for step,(x,f,y,w,quality,idx) in enumerate(loader):
            x,f,y,w,quality=[v.to(device,non_blocking=True) for v in (x,f,y,w,quality)]; w=w.float(); quality=quality.float()
            with torch.autocast(device_type=device.type,dtype=dtype,enabled=amp):
                logits,pw=model(x,f); loss,cl,reg=loss_fn(logits,pw,y,w,a,train=train)
                prob=ordinal_probs(logits)
                if not train and a.tta==4:
                    for k in (1,2,3):
                        l,v=model(torch.rot90(x,k,(2,3)).contiguous(),f); prob=prob+ordinal_probs(l); pw=pw+v
                    prob=prob/4; pw=pw/4
                    # Evaluation loss corresponds to aggregated cumulative probabilities and wind.
                    q=1-prob.cumsum(1)[:,:6]; loss,cl,reg=loss_fn(torch.logit(q.clamp(1e-6,1-1e-6)),pw,y,w,a,train=False)
            if not torch.isfinite(loss).all(): raise FloatingPointError('Nonfinite loss')
            if train:
                group_n=min(a.accumulation*a.batch_size,len(loader.dataset)-(step//a.accumulation)*a.accumulation*a.batch_size)
                scaler.scale((loss*quality).sum()/group_n).backward()
                if (step+1)%a.accumulation==0 or step+1==len(loader):
                    scaler.unscale_(optimizer); nn.utils.clip_grad_norm_(model.parameters(),1.)
                    before=scaler.get_scale(); scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
                    if ema is not None and scaler.get_scale()>=before: update_ema(ema,model,a.ema)
            with torch.no_grad():
                hard_total+=float(loss_fn(logits,pw,y,w,a,train=False)[0].sum().item()) if train else float(loss.sum().item())
            totals+=np.array([loss.detach().sum().item(),cl.detach().sum().item(),reg.detach().sum().item()]); n+=len(y)
            probs=prob.detach().cpu().numpy(); winds=pw.detach().cpu().numpy()*WIND_SCALE
            results.extend((i,probs[j],float(winds[j])) for j,i in enumerate(idx.tolist()))
            if train and ((step+1)%25==0 or step+1==len(loader)): print(f'  batch {step+1}/{len(loader)} loss={totals[0]/n:.4f}',flush=True)
    if not train and sorted(r[0] for r in results)!=list(range(len(loader.dataset))): raise RuntimeError('Evaluation index coverage error')
    rows=loader.dataset.rows; df=pd.DataFrame([rows[r[0]] for r in results])[['sample_id','sid','scene_id','class_id','wind_kt','quality_tier','label_provenance','view_zenith_deg']].copy()
    probs=np.stack([r[1] for r in results]); df['pred_class']=probs.argmax(1); df['pred_wind_kt']=[r[2] for r in results]; df['pred_class_from_wind']=np.searchsorted(BOUNDS,df.pred_wind_kt,side='right')
    for i,c in enumerate(CLASSES): df['prob_'+c]=probs[:,i]
    m=metrics(df); m.update(dict(zip(['loss','class_loss','wind_loss'],(totals/n).tolist()))); m['wind_class_accuracy']=float((df.pred_class_from_wind==df.class_id).mean()); m['hard_target_loss']=hard_total/n; return m,df

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
    for name,cols in {'loss':['train_hard_target_loss','val_loss'],'macro_f1':['train_macro_f1','val_macro_f1'],'accuracy':['train_accuracy','val_accuracy'],'wind_mae':['train_wind_mae_kt','val_wind_mae_kt'],'learning_rate':['backbone_lr','head_lr']}.items():
        fig,ax=plt.subplots(figsize=(8,4))
        for c in cols: ax.plot(df.epoch,df[c],label=c)
        extra={'loss':'natural_train_hard_target_loss','macro_f1':'natural_train_macro_f1','accuracy':'natural_train_accuracy','wind_mae':'natural_train_wind_mae_kt'}.get(name)
        if extra in df: ax.plot(df.epoch[df[extra].notna()],df[extra].dropna(),marker='o',label=extra)
        ax.set(xlabel='Epoch',title=name.replace('_',' ').title()); ax.legend(); fig.tight_layout(); fig.savefig(out/f'{name}.png',dpi=160); plt.close(fig)


def save_checkpoint(path,state):
    tmp=path.with_suffix('.tmp'); torch.save(state,tmp); tmp.replace(path)



def extra_reports(out,name,df):
    folder=out/name; records=[]
    groups={'quality_tier':df.quality_tier,'label_provenance':df.label_provenance,'view_zenith':pd.cut(pd.to_numeric(df.view_zenith_deg),[-1,40,60,70,90],include_lowest=True).astype(str)}
    for key,values in groups.items():
        for v in values.unique():
            sub=df[values==v]; records.append(dict(group=key,value=str(v),samples=len(sub),storms=sub.sid.nunique(),**metrics(sub)))
    pd.DataFrame(records).to_csv(folder/'subgroup_metrics.csv',index=False)
    per=[]
    for sid,sub in df.groupby('sid'): per.append(dict(sid=sid,samples=len(sub),**metrics(sub)))
    pd.DataFrame(per).to_csv(folder/'storm_metrics.csv',index=False)
    # Cluster bootstrap: resample independent storms, not adjacent satellite fixes.
    groups=[g for _,g in df.groupby('sid')]; rng=np.random.default_rng(42); vals=[]
    for _ in range(300):
        b=pd.concat([groups[i] for i in rng.integers(0,len(groups),len(groups))],ignore_index=True); m=metrics(b); vals.append([m[k] for k in ('accuracy','macro_f1','wind_mae_kt')])
    ci=np.percentile(vals,[2.5,97.5],axis=0); dump(folder/'storm_bootstrap_ci.json',{'storms':len(groups),'resamples':300,'intervals':{k:ci[:,i].tolist() for i,k in enumerate(('accuracy','macro_f1','wind_mae_kt'))},'warning':'Descriptive intervals; unreliable with very few independent storms.'})

def fused_predictions(df,weight,scale):
    cols=['prob_'+c for c in CLASSES]; ordinal=df[cols].to_numpy(float)
    q=1/(1+np.exp(np.clip((BOUNDS[None,:]-.5-df.pred_wind_kt.to_numpy(float)[:,None])/scale,-60,60)))
    wind=np.concatenate([1-q[:,:1],q[:,:-1]-q[:,1:],q[:,-1:]],axis=1)
    prob=(1-weight)*ordinal+weight*wind
    out=df.copy(); out[cols]=prob; out['pred_class']=prob.argmax(1)
    return out

def select_fusion(df):
    # Small validation-only search. No test observations enter selection.
    candidates=[(0.,10.)]+[(w,t) for w in (.25,.5,.75) for t in (6.,10.,14.)]
    records=[]
    for w,t in candidates:
        m=metrics(fused_predictions(df,w,t)); records.append(dict(wind_weight=w,scale_kt=t,macro_f1=m['macro_f1'],accuracy=m['accuracy']))
    best=max(records,key=lambda r:(r['macro_f1'],-r['wind_weight']))
    return dict(selected=best,candidates=records,warning='Validation-tuned fusion; not an independent performance estimate. Wind distribution uses a heuristic logistic scale.')

def apply_fusion(out,name,df,selection):
    cfg=selection['selected']; df=fused_predictions(df,cfg['wind_weight'],cfg['scale_kt']); m=metrics(df)
    m['wind_class_accuracy']=float((df.pred_class_from_wind==df.class_id).mean())
    save_eval(out,name+'_fused',m,df); extra_reports(out,name+'_fused',df)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed/classification_v3_1')); p.add_argument('--output',type=Path,default=Path('runs/classification/swin_v3_2'))
    p.add_argument('--epochs',type=int,default=35); p.add_argument('--batch-size',type=int,default=8); p.add_argument('--accumulation',type=int,default=4); p.add_argument('--workers',type=int,default=4); p.add_argument('--image-size',type=int,default=320)
    p.add_argument('--backbone-lr',type=float,default=2e-5); p.add_argument('--head-lr',type=float,default=3e-4); p.add_argument('--weight-decay',type=float,default=.08); p.add_argument('--dropout',type=float,default=.4); p.add_argument('--warmup',type=int,default=4); p.add_argument('--freeze-epochs',type=int,default=3); p.add_argument('--patience',type=int,default=8)
    p.add_argument('--class-alpha',type=float,default=.35); p.add_argument('--storm-alpha',type=float,default=.5); p.add_argument('--regression-weight',type=float,default=1.); p.add_argument('--class-nll-weight',type=float,default=.5); p.add_argument('--ema',type=float,default=.99); p.add_argument('--center-shift-px',type=int,default=14); p.add_argument('--tta',type=int,choices=[1,4],default=4); p.add_argument('--channels',choices=['all','c13','c13_c08'],default='all'); p.add_argument('--physical',action='store_true'); p.add_argument('--label-weighting',action='store_true'); p.add_argument('--quality-weighting',action='store_true')
    p.add_argument('--seed',type=int,default=42); p.add_argument('--device',default='auto'); p.add_argument('--no-pretrained',action='store_true'); p.add_argument('--no-amp',action='store_true'); p.add_argument('--no-augmentation',action='store_true'); p.add_argument('--verify-hashes',action='store_true'); p.add_argument('--validate-only',action='store_true'); p.add_argument('--resume',action='store_true'); p.add_argument('--eval-only',action='store_true'); p.add_argument('--evaluate-test',action='store_true')
    p.add_argument('--architecture',choices=['swin_t','convnext_tiny'],default='swin_t')
    p.add_argument('--fine-tune',choices=['progressive','last-stage','heads-only','full'],default='progressive'); p.add_argument('--stage3-after',type=int,default=10); p.add_argument('--layer-decay',type=float,default=.25)
    p.add_argument('--stochastic-depth',type=float,default=.2); p.add_argument('--rotation',choices=['arbitrary','right-angle','none'],default='arbitrary'); p.add_argument('--aug-min-valid',type=float,default=.85)
    p.add_argument('--soft-tau',type=float,default=3.); p.add_argument('--soft-mix',type=float,default=.5); p.add_argument('--sampling',choices=['hierarchical','legacy'],default='hierarchical')
    p.add_argument('--natural-train-every',type=int,default=0,help='Evaluate unaugmented natural train distribution every N epochs (extra cost)')
    p.add_argument('--calibrate-fusion',action='store_true',help='Optional validation-only selection of ordinal/wind probability fusion after checkpoint selection')
    a=p.parse_args(); a.data=a.data.resolve(); a.output=a.output.resolve()
    if any(not math.isfinite(v) for v in vars(a).values() if isinstance(v,float)): p.error('Nonfinite hyperparameter')
    if min(a.epochs,a.batch_size,a.accumulation,a.patience)<1 or a.image_size<32 or a.image_size%32 or min(a.workers,a.warmup,a.freeze_epochs,a.center_shift_px)<0 or not 0<=a.ema<1 or not 0<=a.dropout<1 or not 0<=a.class_alpha<=1 or not 0<=a.storm_alpha<=1 or min(a.backbone_lr,a.head_lr)<=0 or min(a.weight_decay,a.regression_weight,a.class_nll_weight)<0: p.error('Invalid hyperparameters')
    if not 0<a.layer_decay<=1 or not 0<=a.stochastic_depth<1 or not 0<a.aug_min_valid<=1 or not 0<=a.soft_mix<=1 or not math.isfinite(a.soft_tau) or a.soft_tau<0 or a.natural_train_every<0 or a.stage3_after<0: p.error('Invalid V3.2 settings')
    if a.fine_tune=='progressive' and a.stage3_after<a.freeze_epochs: p.error('stage3-after must be >= freeze-epochs')
    if a.no_pretrained and not (a.resume or a.eval_only) and (a.fine_tune!='full' or a.freeze_epochs): p.error('Training from scratch requires --fine-tune full --freeze-epochs 0')
    if a.evaluate_test and not a.eval_only: p.error('Test evaluation requires --eval-only; select using validation first')
    if a.physical and a.channels!='all': p.error('Physical descriptors include both channels; use --channels all for fusion')
    if a.physical and a.rotation=='arbitrary':
        print('Physical fusion uses right-angle rotations to preserve the precomputed azimuth-sector statistics.',flush=True); a.rotation='right-angle'
    if a.physical and a.center_shift_px: print('Physical fusion disables center shifts to keep descriptors aligned with the image.',flush=True); a.center_shift_px=0
    if a.resume and a.eval_only: p.error('Choose resume or eval-only')
    if a.output==a.data or a.output.is_relative_to(a.data) or a.data.is_relative_to(a.output): p.error('Output must be separate from dataset')
    frames,spec,features,digest=load_data(a)
    if a.validate_only: print('Manifest and metadata validation passed; use --verify-hashes for full file checksum audit.'); return
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else torch.device(a.device)
    if device.type not in ('cpu','cuda') or (device.type=='cuda' and not torch.cuda.is_available()): p.error('CPU or CUDA required')
    seed_all(a.seed); torch.set_float32_matmul_precision('high'); out=a.output; existing=a.resume or a.eval_only
    if not existing and out.exists() and any(out.iterdir()): raise FileExistsError('Use a new run directory or --resume')
    out.mkdir(parents=True,exist_ok=True); best_path=out/'best.pt'; last_path=out/'last.pt'
    ckpt=torch.load(best_path if a.eval_only else last_path,map_location='cpu',weights_only=False) if existing else None
    config={k:v for k,v in vars(a).items() if k not in ('data','output','resume','eval_only','evaluate_test','validate_only','workers','device','verify_hashes','calibrate_fusion')}; config.update(dataset_fingerprint=digest,spec=spec,trainer_version='3.2')
    if ckpt:
        for k in ('trainer_version','dataset_fingerprint','spec','architecture','image_size','physical','channels','dropout','tta','stochastic_depth'):
            if ckpt['config'].get(k)!=config[k]: raise ValueError(f'Checkpoint mismatch: {k}')
        if a.resume and ckpt['config']!=config: raise ValueError('Resume requires identical training settings')
        if a.eval_only:
            a.regression_weight=ckpt['config']['regression_weight']; a.class_nll_weight=ckpt['config']['class_nll_weight']
    sets={s:Crops(a,frames[s],spec,features,s=='train') for s in SPLITS}; model=CycloneModel(2*len(spec['feature_names']),a.dropout,not a.no_pretrained and not existing,a.architecture,a.stochastic_depth).to(device)
    if ckpt: model.load_state_dict(ckpt['model'])
    else:
        # Initialize ordered thresholds near the effective training sampling prior.
        df=frames['train']; y=df.class_id.to_numpy(int)
        weights=sample_weights(df,a)
        q=np.clip(np.array([weights[y>k].sum() for k in range(6)]),1e-4,1-1e-4)
        l=np.log(q/(1-q)); delta=np.maximum(l[:-1]-l[1:],1e-3)
        with torch.no_grad():
            model.ordinal.bias.copy_(torch.tensor(np.r_[l[0],np.log(np.expm1(delta))],dtype=torch.float32,device=device))
            model.wind.bias.fill_(float(np.log(np.expm1(np.dot(weights,df.wind_kt.to_numpy(float))/WIND_SCALE))))
    if not a.eval_only:
        ema=copy.deepcopy(model).eval().requires_grad_(False) if a.ema else None
        opt=torch.optim.AdamW(optimizer_groups(model,a))
        scaler=torch.amp.GradScaler('cuda',enabled=device.type=='cuda' and not a.no_amp and not torch.cuda.is_bf16_supported()); history=[]; start=1; best=-1.; stale=0; best_epoch=0
        if a.resume:
            opt.load_state_dict(ckpt['optimizer']); scaler.load_state_dict(ckpt['scaler']); history=ckpt['history']; start=ckpt['epoch']+1; best=ckpt['best']; stale=ckpt['stale']; best_epoch=ckpt['best_epoch']
            if ema is not None: ema.load_state_dict(ckpt['ema'])
            if not best_path.is_file(): raise FileNotFoundError('Resume also requires best.pt')
        dump(out/'config.json',config)
        warm=min(a.warmup,max(0,a.epochs-1))
        previous_phase=None if start==1 else model.set_phase(a,start-1)
        natural=Crops(a,frames['train'],spec,features,False)
        for epoch in range(start,a.epochs+1):
            phase=model.set_phase(a,epoch)
            if previous_phase is not None and phase!=previous_phase: stale=0
            transitions=[] if a.fine_tune=='heads-only' else [a.freeze_epochs+1]
            if a.fine_tune=='progressive': transitions.append(a.stage3_after+1)
            last_transition=max([0]+[e for e in transitions if e<=a.epochs])
            if stale>=a.patience and epoch>last_transition: break
            previous_phase=phase; seed_all(a.seed+epoch)
            factor=(.1+.9*epoch/warm) if epoch<=warm else .05+.95*(1+math.cos(math.pi*(epoch-warm-1)/max(1,a.epochs-warm-1)))/2
            for group in opt.param_groups: group['lr']=group['base_lr']*factor
            active=sum(p.numel() for p in model.parameters() if p.requires_grad)
            tick=time.monotonic(); print(f'Epoch {epoch}/{a.epochs} | active backbone stages={phase} | trainable parameters={active:,}',flush=True)
            train,_=run_epoch(model,make_loader(sets['train'],a,device,epoch,True),device,a,opt,scaler,ema)
            selected=ema if ema is not None else model; val,_=run_epoch(selected,make_loader(sets['val'],a,device),device,a)
            score=val['macro_f1']; history.append({'epoch':epoch,'seconds':time.monotonic()-tick,'backbone_lr':a.backbone_lr*factor if phase else 0.,'head_lr':a.head_lr*factor,'active_stages':phase,'trainable_parameters':active,**{'train_'+k:v for k,v in train.items()},**{'val_'+k:v for k,v in val.items()}})
            if a.natural_train_every and epoch%a.natural_train_every==0:
                nat,_=run_epoch(selected,make_loader(natural,a,device),device,a)
                history[-1].update({'natural_train_'+k:v for k,v in nat.items()})
            if score>best+1e-6:
                best=score; best_epoch=epoch; stale=0; save_checkpoint(best_path,{'model':selected.state_dict(),'config':config,'epoch':epoch,'best':best})
            else: stale+=1
            save_checkpoint(last_path,{'model':model.state_dict(),'ema':ema.state_dict() if ema is not None else None,'config':config,'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'best_epoch':best_epoch,'stale':stale,'history':history})
            pd.DataFrame(history).to_csv(out/'history.csv',index=False)
            print(f'  sampled train F1={train["macro_f1"]:.4f} val F1={score:.4f} val acc={val["accuracy"]:.4f} wind MAE={val["wind_mae_kt"]:.2f} kt best={best_epoch}',flush=True)
        if history: save_history(out,history)
    best_ckpt=torch.load(best_path,map_location='cpu',weights_only=False); model.load_state_dict(best_ckpt['model'])
    fusion=None
    for split,name in [('val','validation')]+([('test','test')] if a.evaluate_test else []):
        m,df=run_epoch(model,make_loader(sets[split],a,device),device,a); save_eval(out,name,m,df); extra_reports(out,name,df)
        if a.calibrate_fusion:
            if split=='val':
                fusion=select_fusion(df); fusion.update(checkpoint_epoch=best_ckpt['epoch'],dataset_fingerprint=digest); dump(out/'fusion.json',fusion)
            apply_fusion(out,name,df,fusion)
    dump(out/'selection.json',{'best_epoch':best_ckpt['epoch'],'validation_macro_f1':best_ckpt['best'],'test_evaluated_this_run':a.evaluate_test,'checkpoint_metric':'fixed-seven validation macro F1','training_metrics':'sampled training distribution, not natural prevalence; natural_train_* are optional unsampled EMA evaluations; train_loss is soft objective, train_hard_target_loss and val_loss use hard labels','tta':a.tta,'fusion_evaluated':a.calibrate_fusion,'architecture':a.architecture})

if __name__=='__main__': main()
