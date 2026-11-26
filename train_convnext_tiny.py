#!/usr/bin/env python3
"""ConvNeXt-Tiny single-image cyclone intensity training. Existing HDF5 files are read-only."""
import argparse, csv, hashlib, json, math, os, random, time, subprocess
from tqdm import tqdm
from pathlib import Path
import h5py
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

NAMES = ['TD','TS','C1','C2','C3','C4','C5']
THRESHOLDS = np.array([34,64,83,96,113,137])

def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def worker_seed(_):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

def dump(path, obj):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False)); os.replace(tmp,path)

def save(path,obj):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); os.replace(tmp,path)

class Cyclones(Dataset):
    def __init__(self, path, mean, std, size, augment=False, limit=0):
        self.path=str(path); self.mean=mean; self.std=std; self.size=size; self.augment=augment; self.h=None
        with h5py.File(path,'r') as h:
            keys={'images','valid_mask','class_id','wind_kts','sid','sample_id','sensor_id','observation_weight'}
            if not keys <= set(h): raise ValueError(f'{path}: missing {keys-set(h)}')
            self.n=len(h['images']); self.scale=float(h.attrs['temperature_scale']); self.offset=float(h.attrs['temperature_offset_K'])
            if self.n==0 or h['images'].ndim!=3 or h['images'].shape!=h['valid_mask'].shape: raise ValueError('Invalid image shapes')
            self.labels=h['class_id'][:].astype(np.int64); self.wind=h['wind_kts'][:].astype(np.float32)
            self.weights=h['observation_weight'][:].astype(np.float32)
            self.sid=h['sid'].asstr()[:]; self.ids=h['sample_id'].asstr()[:]; self.sensor=h['sensor_id'].asstr()[:]
            if any(len(x)!=self.n for x in (self.labels,self.wind,self.weights,self.sid,self.ids,self.sensor)): raise ValueError('Metadata length mismatch')
            if not np.isfinite(self.wind).all() or (self.wind<0).any() or not np.array_equal(np.digitize(self.wind,THRESHOLDS),self.labels): raise ValueError('Invalid intensity labels')
            if not np.isfinite(self.weights).all() or (self.weights<=0).any(): raise ValueError('Invalid observation weights')
        if limit: self.n=min(self.n,limit)
    def __len__(self): return self.n
    def __getstate__(self):
        state=self.__dict__.copy(); state['h']=None; return state
    def __getitem__(self,i):
        if self.h is None: self.h=h5py.File(self.path,'r')
        m=torch.from_numpy(self.h['valid_mask'][i].astype(np.float32))[None,None]
        x=torch.from_numpy(self.h['images'][i].astype(np.float32))[None,None]
        x=((x/self.scale+self.offset-self.mean)/self.std)*m
        # Mask-aware area averaging prevents invalid sentinel pixels affecting valid values.
        if x.shape[-2:]!=(self.size,self.size):
            mode='area' if min(x.shape[-2:])>=self.size else 'bilinear'
            kwargs={} if mode=='area' else {'align_corners':False}
            x=F.interpolate(x,size=(self.size,self.size),mode=mode,**kwargs)
            m=F.interpolate(m,size=(self.size,self.size),mode=mode,**kwargs)
            x=torch.where(m>=0.5,x/m.clamp_min(1e-6),0.)
        if self.augment: x=torch.rot90(x,random.randrange(4),(-2,-1))
        return x.squeeze(0), int(self.labels[i]), float(self.wind[i]), float(self.weights[i]), i

class Model(nn.Module):
    def __init__(self, pretrained=True, dropout=.3):
        super().__init__()
        self.backbone=convnext_tiny(weights=ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None)
        old=self.backbone.features[0][0]
        stem=nn.Conv2d(1,old.out_channels,kernel_size=old.kernel_size,stride=old.stride,padding=old.padding,bias=old.bias is not None)
        with torch.no_grad():
            stem.weight.copy_(old.weight.sum(dim=1,keepdim=True))
            if old.bias is not None: stem.bias.copy_(old.bias)
        self.backbone.features[0][0]=stem
        self.backbone.classifier[2]=nn.Identity()
        self.classifier=nn.Sequential(nn.Dropout(dropout),nn.Linear(768,7))
        self.regressor=nn.Sequential(nn.Linear(768,128),nn.GELU(),nn.Dropout(dropout),nn.Linear(128,1))
    def forward(self,x):
        z=self.backbone(x); return self.classifier(z), F.softplus(self.regressor(z).squeeze(1))

def metrics(y,p,w,wp):
    cm=confusion_matrix(y,p,labels=range(7)); pr,rc,f1,support=precision_recall_fscore_support(y,p,labels=range(7),zero_division=0)
    present=support>0; err=wp-w
    return {'n':int(len(y)), 'accuracy':float(np.mean(y==p)), 'macro_f1':float(f1.mean()),'balanced_accuracy':float(rc[present].mean()),'mae_kt':float(np.abs(err).mean()),'rmse_kt':float(np.sqrt(np.mean(err**2))), 'ordinal_mae':float(np.abs(y-p).mean()),'within_one_category':float(np.mean(np.abs(y-p)<=1)),'wind_bias_kt':float(err.mean()),'regression_class_accuracy':float(np.mean(np.digitize(wp,THRESHOLDS)==y)), 'confusion_matrix':cm.tolist(),'per_class':{name:{'precision':float(pr[i]),'recall':float(rc[i]),'f1':float(f1[i]),'support':int(support[i])} for i,name in enumerate(NAMES)}}

class GPUStats:
    """Best-effort NVML via nvidia-smi; CUDA allocator statistics are independent."""
    def __init__(self, device, interval=5):
        self.device=device; self.interval=interval; self.last=-float('inf'); self.cached={}
    def read(self):
        if self.device.type!='cuda': return {}
        now=time.monotonic()
        if now-self.last>=self.interval:
            self.last=now; self.cached={}
            try:
                # Resolve the physical GPU by UUID if PyTorch exposes it, otherwise
                # use the CUDA_VISIBLE_DEVICES mapping (single-GPU default).
                i=self.device.index if self.device.index is not None else torch.cuda.current_device()
                prop=torch.cuda.get_device_properties(i)
                visible=os.environ.get('CUDA_VISIBLE_DEVICES','').split(',')
                selector=str(getattr(prop,'uuid','') or (visible[i].strip() if i<len(visible) and visible[i].strip() else i))
                line=subprocess.check_output(['nvidia-smi','-i',selector,'--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw','--format=csv,noheader,nounits'],text=True,timeout=2,stderr=subprocess.DEVNULL).strip().splitlines()[0]
                keys=('gpu_util_pct','gpu_used_mib','gpu_total_mib','gpu_temp_c','gpu_power_w')
                for k,v in zip(keys,line.split(',')):
                    try:
                        x=float(v.strip())
                        if math.isfinite(x): self.cached[k]=x
                    except ValueError: pass
            except (OSError,subprocess.SubprocessError,IndexError,RuntimeError): pass
        return {**self.cached,'cuda_allocated_mib':torch.cuda.memory_allocated(self.device)/2**20,'cuda_reserved_mib':torch.cuda.memory_reserved(self.device)/2**20,'cuda_peak_allocated_mib':torch.cuda.max_memory_allocated(self.device)/2**20}


def run_epoch(model,loader,device,amp,weights,reg_weight,optimizer=None,scaler=None, *, desc=None, monitor=None, log_every=10, batch_log=None, epoch=0, accum=1, clip=1.):
    training=optimizer is not None; model.train(training)
    n=correct=0; abs_error=0.; ce_num=ce_den=reg_num=reg_den=0.; ys=[]; ps=[]; ws=[]; wps=[]; ids=[]; probs=[]
    started=time.monotonic(); telemetry=[]; grad_norm=None; updates=0; skipped=0; wait_s=0.; last_end=started
    if training: optimizer.zero_grad(set_to_none=True)
    if device.type=='cuda': torch.cuda.reset_peak_memory_stats(device)
    with torch.set_grad_enabled(training), tqdm(loader,desc=desc or ('Train' if training else 'Evaluate'),unit='batch',dynamic_ncols=True, mininterval=.5,leave=True) as bar:
        for step,(x,y,w,ow,idx) in enumerate(bar,1):
            wait_s+=time.monotonic()-last_end
            x=x.to(device,non_blocking=True); y=y.to(device); w=w.to(device,dtype=torch.float32); ow=ow.to(device,dtype=torch.float32)
            with torch.autocast(device_type=device.type,dtype=amp,enabled=amp is not None):
                logits,pred=model(x)
                ce=F.cross_entropy(logits,y,reduction='none')
                reg=F.smooth_l1_loss(pred,w/100.,beta=.1,reduction='none')
                cw=weights[y] if training else torch.ones_like(ow)
                cn=(ce*cw*ow).sum(); cd=(cw*ow).sum(); rn=(reg*ow).sum(); rd=ow.sum()
                loss=cn/cd+reg_weight*rn/rd
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite loss')
            if training:
                # Accumulate equally weighted microbatch objectives, including the final partial group.
                group_size=min(accum,len(loader)-((step-1)//accum)*accum)
                scaler.scale(loss/group_size).backward()
                if step%accum==0 or step==len(loader):
                    scaler.unscale_(optimizer)
                    gn=nn.utils.clip_grad_norm_(model.parameters(),clip,error_if_nonfinite=not scaler.is_enabled())
                    grad_norm=float(gn.detach()); grad_norm=grad_norm if math.isfinite(grad_norm) else None
                    old_scale=scaler.get_scale()
                    scaler.step(optimizer); scaler.update()
                    if scaler.get_scale()<old_scale: skipped+=1
                    else: updates+=1
                    optimizer.zero_grad(set_to_none=True)
            ce_num+=float(cn.detach()); ce_den+=float(cd.detach()); reg_num+=float(rn.detach()); reg_den+=float(rd.detach())
            yp=logits.detach().argmax(1); wind_pred=pred.detach().float()*100
            n+=len(y); correct+=int((yp==y).sum()); abs_error+=float((wind_pred-w).abs().sum())
            ys.append(y.cpu().numpy()); ps.append(yp.cpu().numpy()); ws.append(w.cpu().numpy()); wps.append(wind_pred.cpu().numpy()); ids.append(idx.numpy()); probs.append(logits.detach().float().softmax(1).cpu().numpy())
            if step==1 or step%log_every==0 or step==len(loader):
                elapsed=max(time.monotonic()-started,1e-9); gpu=monitor.read() if monitor else {}; telemetry.append(gpu)
                stats={'epoch':epoch,'phase':'train' if training else (desc or 'eval'),'batch':step,'batches':len(loader),'samples':n,'elapsed_s':elapsed,'samples_per_sec':n/elapsed,'loss':ce_num/ce_den+reg_weight*reg_num/reg_den,'classification_loss':ce_num/ce_den,'regression_loss':reg_num/reg_den,'accuracy':correct/n,'mae_kt':abs_error/n,'lr':optimizer.param_groups[0]['lr'] if training else None,'grad_norm':grad_norm,'optimizer_updates':updates,'skipped_updates':skipped,'data_wait_s':wait_s,'head_lr':optimizer.param_groups[-1]['lr'] if training else None,**gpu}
                post={'loss':f"{stats['loss']:.3f}",'acc':f'{correct/n:.1%}','MAE':f'{abs_error/n:.1f}kt','img/s':f'{n/elapsed:.1f}'}
                if 'gpu_util_pct' in gpu: post['GPU']=f"{gpu['gpu_util_pct']:.0f}%"
                post['F1']=f"{metrics(np.concatenate(ys),np.concatenate(ps),np.concatenate(ws),np.concatenate(wps))['macro_f1']:.3f}"
                if 'gpu_temp_c' in gpu: post['temp']=f"{gpu['gpu_temp_c']:.0f}C"
                if 'gpu_power_w' in gpu: post['W']=f"{gpu['gpu_power_w']:.0f}"
                if 'cuda_reserved_mib' in gpu: post['VRAM']=f"{gpu['cuda_reserved_mib']/1024:.2f}GiB"
                bar.set_postfix(post,refresh=False)
                if batch_log:
                    with batch_log.open('a') as f: f.write(json.dumps(stats,allow_nan=False)+'\n')
            last_end=time.monotonic()
    arrays=[np.concatenate(a) for a in (ys,ps,ws,wps,ids,probs)]
    result=metrics(*arrays[:4]); result.update(loss=ce_num/ce_den+reg_weight*reg_num/reg_den,classification_loss=ce_num/ce_den,regression_loss=reg_num/reg_den,elapsed_s=time.monotonic()-started)
    result['samples_per_sec']=n/max(result['elapsed_s'],1e-9)
    result['gpu_telemetry']={k:{'mean':float(np.mean([v[k] for v in telemetry if k in v])),'max':float(max(v[k] for v in telemetry if k in v))} for k in sorted({k for v in telemetry for k in v})}
    result.update(optimizer_updates=updates,skipped_updates=skipped,data_wait_s=wait_s)
    result['by_sensor']={sensor:metrics(*[a[loader.dataset.sensor[arrays[4]]==sensor] for a in arrays[:4]]) for sensor in sorted(set(loader.dataset.sensor[arrays[4]]))}
    result['storm_macro_mae_kt']=float(np.mean([np.abs(arrays[3][loader.dataset.sid[arrays[4]]==sid]-arrays[2][loader.dataset.sid[arrays[4]]==sid]).mean() for sid in set(loader.dataset.sid[arrays[4]])]))
    return result,arrays


def epoch_report(epoch,tr,va,lr,best,stale,seconds):
    print(f"\nEpoch {epoch} | {seconds:.1f}s | next LR={lr:.3g} | best F1={best:.4f} | patience used={stale}",flush=True)
    print(f"{'Metric':<26}{'Train':>13}{'Validation':>13}")
    for key in ('loss','classification_loss','regression_loss','accuracy','macro_f1','balanced_accuracy','mae_kt','rmse_kt','ordinal_mae','regression_class_accuracy','within_one_category','wind_bias_kt','storm_macro_mae_kt','samples_per_sec'):
        print(f'{key:<26}{tr[key]:13.4f}{va[key]:13.4f}')
    print('Optimizer updates:',tr['optimizer_updates'],'skipped:',tr['skipped_updates'],'data wait seconds:',round(tr['data_wait_s'],2))
    for sensor,m in va['by_sensor'].items(): print(f"Validation {sensor}: n={m['n']} F1={m['macro_f1']:.4f} MAE={m['mae_kt']:.2f} kt")
    print('Validation per class:')
    print(f"{'Class':<8}{'Precision':>12}{'Recall':>12}{'F1':>12}{'Support':>10}")
    for name,m in va['per_class'].items(): print(f"{name:<8}{m['precision']:12.4f}{m['recall']:12.4f}{m['f1']:12.4f}{m['support']:10d}")
    print('Validation confusion matrix (rows=true, columns=predicted):')
    print(np.asarray(va['confusion_matrix']))
    if tr['gpu_telemetry']: print('Training GPU sampled telemetry:',json.dumps(tr['gpu_telemetry']))


def export_history(out,history):
    rows=[]
    for e in history:
        row={'epoch':e['epoch'],'lr':e['lr'],'head_lr':e['head_lr']}
        for phase in ('train','val'):
            for k,v in e[phase].items():
                if isinstance(v,(int,float)): row[f'{phase}_{k}']=v
            for k,m in e[phase]['gpu_telemetry'].items():
                for stat,v in m.items(): row[f'{phase}_{k}_{stat}']=v
            for c,m in e[phase]['per_class'].items():
                for k,v in m.items(): row[f'{phase}_{c}_{k}']=v
        rows.append(row)
    keys=list(dict.fromkeys(k for row in rows for k in row))
    with (out/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=keys); writer.writeheader(); writer.writerows(rows)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
        for ax,key,label in zip(axes.flat,('loss','macro_f1','accuracy','mae_kt','rmse_kt','balanced_accuracy'),('Loss','Macro F1','Accuracy','Wind MAE (kt)','Wind RMSE (kt)','Balanced accuracy')):
            for phase in ('train','val'): ax.plot([e['epoch'] for e in history],[e[phase][key] for e in history],label=phase)
            ax.set(xlabel='Epoch',title=label); ax.grid(alpha=.2); ax.legend()
        fig.savefig(out/'training_curves.png',dpi=140); plt.close(fig)
    except ImportError: pass

def evaluate(model,loader,device,amp,weights,args,out,name):
    result,a=run_epoch(model,loader,device,amp,weights,args.reg_weight,desc=name,monitor=GPUStats(device,args.gpu_interval),log_every=args.log_every,batch_log=out/'batch_metrics.jsonl')
    y,p,w,wp,idx,prob=a; ds=loader.dataset
    result['by_sensor']={}
    for sensor in sorted(set(ds.sensor[idx])):
        mask=ds.sensor[idx]==sensor; result['by_sensor'][sensor]=metrics(y[mask],p[mask],w[mask],wp[mask])
    result['storm_macro_mae_kt']=float(np.mean([np.abs(wp[ds.sid[idx]==s]-w[ds.sid[idx]==s]).mean() for s in set(ds.sid[idx])]))
    dump(out/f'{name}_metrics.json',result)
    with (out/f'{name}_predictions.csv').open('w',newline='') as f:
        writer=csv.writer(f); writer.writerow(['sample_id','sid','sensor_id','true_class','pred_class','true_wind_kt','pred_wind_kt']+['prob_'+s for s in NAMES])
        for i,j in enumerate(idx): writer.writerow([ds.ids[j],ds.sid[j],ds.sensor[j],int(y[i]),int(p[i]),float(w[i]),float(wp[i]),*map(float,prob[i])])
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir',type=Path,default=Path('data/processed/intensity_kelvin_final'))
    p.add_argument('--output-dir',type=Path,default=Path('training/runs/convnext_tiny'))
    p.add_argument('--epochs',type=int,default=60); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--size',type=int,default=384); p.add_argument('--workers',type=int,default=2); p.add_argument('--lr',type=float,default=3e-4); p.add_argument('--patience',type=int,default=10); p.add_argument('--seed',type=int,default=42); p.add_argument('--reg-weight',type=float,default=1.); p.add_argument('--amp',choices=['auto','none','bf16','fp16'],default='auto'); p.add_argument('--device',choices=['auto','cuda','cpu'],default='auto'); p.add_argument('--class-weight',choices=['sqrt','none'],default='sqrt'); p.add_argument('--resume',action='store_true'); p.add_argument('--test-only',action='store_true'); p.add_argument('--smoke',action='store_true'); p.add_argument('--log-every',type=int,default=10); p.add_argument('--gpu-interval',type=float,default=5); p.add_argument('--accum',type=int,default=4); p.add_argument('--backbone-lr',type=float,default=5e-5); p.add_argument('--dropout',type=float,default=.3); p.add_argument('--weight-decay',type=float,default=.01); p.add_argument('--no-pretrained',action='store_true'); args=p.parse_args()
    if args.accum<1 or not 0<=args.dropout<1 or not math.isfinite(args.backbone_lr) or args.backbone_lr<=0 or not math.isfinite(args.weight_decay) or args.weight_decay<0: p.error('Invalid model/optimizer arguments')
    if args.log_every<1 or not math.isfinite(args.gpu_interval) or args.gpu_interval<=0: p.error('Invalid monitoring interval')
    if not all(math.isfinite(v) for v in (args.lr,args.reg_weight)) or min(args.epochs,args.batch_size,args.patience)<1 or args.size<32 or args.workers<0 or args.lr<=0 or args.reg_weight<0: p.error('Invalid configuration')
    if args.smoke and (args.resume or args.test_only): p.error('Smoke mode cannot resume or test')
    seed_all(args.seed); device=torch.device(('cuda' if torch.cuda.is_available() else 'cpu') if args.device=='auto' else args.device)
    if device.type=='cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    amp=None
    if device.type=='cuda' and args.amp!='none':
        amp=torch.bfloat16 if args.amp=='bf16' or (args.amp=='auto' and torch.cuda.is_bf16_supported()) else torch.float16
        if amp==torch.bfloat16 and not torch.cuda.is_bf16_supported(): raise RuntimeError('BF16 unsupported')
    elif args.amp in ('bf16','fp16'): raise ValueError('Explicit AMP requires CUDA')
    root=args.data_dir.resolve(); out=args.output_dir / 'smoke' if args.smoke else args.output_dir
    if not (root/'READY').is_file(): raise RuntimeError('Dataset READY marker missing')
    norm=json.loads((root/'normalization.json').read_text()); mean=float(norm['mean_kelvin']); std=float(norm['std_kelvin'])
    if not math.isfinite(mean) or not math.isfinite(std) or std<=0: raise ValueError('Invalid normalization')
    # Test membership is checked, but its images are never read during training.
    datasets={s:Cyclones(root/f'{s}.h5',mean,std,args.size,augment=s=='train',limit=32 if args.smoke else 0) for s in ('train','val','test')}
    seen=set(); all_ids=set()
    for s,ds in datasets.items():
        if seen.intersection(ds.sid): raise ValueError('Storm leakage')
        if len(set(ds.ids))!=len(ds.ids) or all_ids.intersection(ds.ids): raise ValueError('Duplicate sample IDs')
        seen.update(ds.sid); all_ids.update(ds.ids)
        print(s,ds.n,'samples;',len(set(ds.sid)),'storms;',dict(zip(*np.unique(ds.sensor,return_counts=True))))
    if set(datasets['train'].sensor)-set(datasets['val'].sensor): print('WARNING: validation lacks training sensors; checkpoint selection has limited representativeness.')
    # Fingerprint dataset metadata and actual immutable file identity, not all pixels.
    identity=hashlib.sha256(json.dumps({'root':str(root),'norm':norm,'files':[(s,(root/f'{s}.h5').stat().st_size,(root/f'{s}.h5').stat().st_mtime_ns,datasets[s].ids.tolist()) for s in datasets]},sort_keys=True).encode()).hexdigest()
    cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('resume','test_only','epochs','output_dir','workers','data_dir','log_every','gpu_interval')}; cfg['dataset_id']=identity
    out.mkdir(parents=True,exist_ok=True)
    if args.resume or args.test_only:
        if json.loads((out/'config.json').read_text())!=cfg: raise ValueError('Checkpoint configuration/dataset mismatch')
    else:
        if (out/'config.json').exists(): raise FileExistsError('Run already exists; use --resume or choose a new output directory')
        dump(out/'config.json',cfg)
    gen=torch.Generator().manual_seed(args.seed)
    loaders={s:DataLoader(ds,batch_size=args.batch_size,shuffle=s=='train',num_workers=args.workers,pin_memory=device.type=='cuda',worker_init_fn=worker_seed,generator=gen if s=='train' else None,multiprocessing_context='spawn' if args.workers else None) for s,ds in datasets.items()}
    count=np.bincount(datasets['train'].labels,weights=datasets['train'].weights,minlength=7)
    if (count==0).any(): raise ValueError('Training class missing')
    weights=torch.tensor(1/np.sqrt(count) if args.class_weight=='sqrt' else np.ones(7),dtype=torch.float32,device=device); weights/=weights.mean()
    model=Model(pretrained=not (args.no_pretrained or args.resume or args.test_only),dropout=args.dropout).to(device)
    opt=torch.optim.AdamW([{'params':model.backbone.parameters(),'lr':args.backbone_lr},{'params':list(model.classifier.parameters())+list(model.regressor.parameters()),'lr':args.lr}],weight_decay=args.weight_decay)
    print(f'Parameters: {sum(p.numel() for p in model.parameters()):,}; effective batch={args.batch_size*args.accum}; backbone LR={args.backbone_lr:g}; head LR={args.lr:g}',flush=True)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='max',factor=.5,patience=3)
    scaler=torch.amp.GradScaler('cuda',enabled=amp==torch.float16)
    start=0; best=-1.; stale=0; history=[]
    if args.test_only:
        ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['model']); result=evaluate(model,loaders['test'],device,amp,weights,args,out,'test'); print(json.dumps(result,indent=2)); return
    if args.resume:
        ck=torch.load(out/'last.pt',map_location='cpu',weights_only=True); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scheduler.load_state_dict(ck['scheduler']); scaler.load_state_dict(ck['scaler']); start=ck['epoch']+1; best=ck['best']; stale=ck['stale']; history=ck['history']; gen.set_state(ck['generator']); torch.set_rng_state(ck['torch_rng']); random.setstate(ck['python_rng']);
        if device.type=='cuda' and ck['cuda_rng']: torch.cuda.set_rng_state_all(ck['cuda_rng'])
    print(f'Device={device}, AMP={amp}, class weights={weights.tolist()}')
    monitor=GPUStats(device,args.gpu_interval)
    total_epochs=1 if args.smoke else args.epochs
    if device.type=='cuda': print('GPU:',torch.cuda.get_device_name(device),flush=True)
    for epoch in range(start,total_epochs):
        t=time.monotonic(); tr,_=run_epoch(model,loaders['train'],device,amp,weights,args.reg_weight,opt,scaler,desc=f'Epoch {epoch+1}/{total_epochs} train',monitor=monitor,log_every=args.log_every,batch_log=out/'batch_metrics.jsonl',epoch=epoch+1,accum=args.accum); va,_=run_epoch(model,loaders['val'],device,amp,weights,args.reg_weight,desc=f'Epoch {epoch+1}/{total_epochs} val',monitor=monitor,log_every=args.log_every,batch_log=out/'batch_metrics.jsonl',epoch=epoch+1); score=va['macro_f1']; scheduler.step(score)
        improved=score>best+1e-6
        if improved: best=score; stale=0; save(out/'best.pt',{'model':model.state_dict(),'epoch':epoch,'macro_f1':best})
        else: stale+=1
        history.append({'epoch':epoch+1,'train':tr,'val':va,'lr':opt.param_groups[0]['lr'],'head_lr':opt.param_groups[1]['lr']}); dump(out/'history.json',history)
        save(out/'last.pt',{'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'epoch':epoch,'best':best,'stale':stale,'history':history,'generator':gen.get_state(),'torch_rng':torch.get_rng_state(),'python_rng':random.getstate(),'cuda_rng':torch.cuda.get_rng_state_all() if device.type=='cuda' else []})
        epoch_report(epoch+1,tr,va,opt.param_groups[0]['lr'],best,stale,time.monotonic()-t)
        export_history(out,history)
        if stale>=args.patience: print('Early stopping'); break
    ck=torch.load(out/'best.pt',map_location=device,weights_only=True); model.load_state_dict(ck['model']); evaluate(model,loaders['val'],device,amp,weights,args,out,'val'); print('Training finished. Test images remain untouched. Use --test-only once model selection is complete.')

if __name__=='__main__': main()
