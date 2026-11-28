#!/usr/bin/env python3
"""CycloneAI custom dual-scale CNN V3 with radial thermal profiles, four NOAA intensity groups, randomly initialized; TensorFlow/Keras 3.
Uses V3.1 Kelvin tensors and frozen manifests. KERAS_BACKEND=torch may be
used only as an alternative for testing. No pretrained weights or external models.
"""
import argparse, hashlib, json, os, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
os.environ.setdefault("KERAS_BACKEND", "tensorflow")
import keras
from keras import layers, ops
from scipy.ndimage import affine_transform
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, recall_score, cohen_kappa_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SOURCE_CLASSES=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5']
SOURCE_BOUNDS=[34,64,83,96,113,137]
CLASSES=['TD','TS','HU','MH']
BOUNDS=[34,64,96]
CLASS_NAMES=['Tropical Depression','Tropical Storm','Hurricane (Cat 1–2)','Major Hurricane (Cat 3–5)']
N_CLASSES=len(CLASSES)
CLASS_MAP=dict(zip(SOURCE_CLASSES,['TD','TS','HU','HU','MH','MH','MH']))
SCHEME='noaa_four_v1'
TRAINER_VERSION='custom_tf_v3_4class'
RING_EDGES=(0,25,50,100,150,200,300,400)
N_RINGS=len(RING_EDGES)-1
RADIAL_DIM=N_RINGS*15+6 
SPLITS=('train','val','test')

def read(p): return json.loads(p.read_text())
def dump(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def inventory(root, verify=False):
    meta=root/'metadata'; man=root/'manifests'
    norm=read(meta/'normalization.json'); cfg=read(meta/'preprocessing_config.json')
    if read(meta/'classes.json')!={str(i):c for i,c in enumerate(SOURCE_CLASSES)}: raise ValueError('Source class mapping mismatch')
    if norm['channels']!=['C13','C08','C13-C08'] or norm['method']!='train_valid_joint_pixels_zscore': raise ValueError('Normalization schema mismatch')
    mean=np.asarray(norm['mean'],np.float32); std=np.asarray(norm['std'],np.float32)
    native=int(cfg['master_size']); crop=int(norm['crop_size'])
    if mean.shape!=(3,) or std.shape!=(3,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std<=0) or not 0<crop<=native or (native-crop)%2: raise ValueError('Invalid normalization/crop')
    mapping=read(man/'storm_splits.json'); frames={}; seen=set(); fixes=set(); paths=set(); storms={}; scenes={}; hashes={}
    digest=hashlib.sha256()
    for p in [meta/'classes.json',meta/'normalization.json',meta/'preprocessing_config.json',man/'storm_splits.json',*[man/f'{s}.csv' for s in SPLITS]]: digest.update(p.read_bytes())
    for s in SPLITS:
        df=pd.read_csv(man/f'{s}.csv',keep_default_na=False,dtype={k:str for k in ['sample_id','sid','scene_id','label_time','tensor_sha256']})
        req={'sample_id','sid','scene_id','label_time','split','kelvin','wind_kt','class_id','class_name','tensor_sha256'}
        if df.empty or not req<=set(df): raise ValueError(f'Invalid {s} manifest')
        for r in df.to_dict('records'):
            i,sid,scene=r['sample_id'],r['sid'],r['scene_id']; fix=(sid,r['label_time']); w=float(r['wind_kt']); c=int(r['class_id']); h=r['tensor_sha256']; p=(root/r['kelvin']).resolve()
            if not all([i,sid,scene,r['label_time']]) or not 0<w<=250 or c!=float(r['class_id']) or c!=np.searchsorted(SOURCE_BOUNDS,w,side='right') or r['class_name']!=SOURCE_CLASSES[c]: raise ValueError(f'Invalid label {i}')
            if r['split']!=s or mapping.get(sid)!=s or storms.setdefault(sid,s)!=s or scenes.setdefault(scene,s)!=s or i in seen or fix in fixes: raise ValueError(f'Duplicate/leakage: {i}')
            if not p.is_relative_to(root) or not p.is_file() or p in paths: raise ValueError(f'Invalid tensor {p}')
            if len(h)!=64 or any(ch not in '0123456789abcdef' for ch in h) or hashes.setdefault(h,s)!=s: raise ValueError(f'Invalid checksum or cross-split duplicate: {i}')
            if verify:
                with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=h: raise ValueError(f'Checksum mismatch {i}')
            seen.add(i); fixes.add(fix); paths.add(p); digest.update(f'{r["kelvin"]}:{p.stat().st_size}:{h}'.encode())
        df['source_class_id']=df.class_id.astype(int)
        df['source_class_name']=df.class_name
        df['class_id']=np.searchsorted(BOUNDS,df.wind_kt.astype(float),side='right')
        df['class_name']=[CLASSES[i] for i in df.class_id]
        if not (df.source_class_name.map(CLASS_MAP)==df.class_name).all(): raise ValueError('Remapping mismatch')
        counts=np.bincount(df.class_id.astype(int),minlength=N_CLASSES)
        print(f'{s}: {len(df)} samples, {df.sid.nunique()} storms, counts={counts.tolist()}',flush=True)
        if s=='train' and np.any(counts==0): raise ValueError('Training missing classes')
        if np.any(counts==0): print('WARNING: missing classes:',[CLASSES[i] for i in np.flatnonzero(counts==0)])
        frames[s]=df
    return frames, (native,crop,mean,std), digest.hexdigest()


def resize_float(x,n):
    if x.shape==(n,n): return np.asarray(x,dtype=np.float32)
    return np.asarray(Image.fromarray(np.asarray(x,np.float32)).resize((n,n),Image.Resampling.BILINEAR),dtype=np.float32)

class RadialProfile:
    """Image-only radial statistics on the aligned native context crop.
    Fixed training z-score coordinates; explicit missing/coverage indicators.
    """
    def __init__(self, native, crop, master_km, mean, std):
        self.crop=crop; self.start=(native-crop)//2
        axis=(np.arange(crop,dtype=np.float64)+.5-crop/2)*float(master_km)/native
        xx,yy=np.meshgrid(axis,axis)
        radius=np.hypot(xx,yy).ravel()
        rings=np.searchsorted(RING_EDGES,radius,side='right')-1
        sectors=np.floor((np.arctan2(yy,xx).ravel()%(2*np.pi))/(np.pi/4)).astype(int)%8
        self.inside=rings<N_RINGS
        self.ring_ids=rings[self.inside]
        self.sector_ids=self.ring_ids*8+sectors[self.inside]
        self.total=np.bincount(self.ring_ids,minlength=N_RINGS)
        self.sector_total=np.bincount(self.sector_ids,minlength=N_RINGS*8)
        self.cold_region=radius<200
        self.cold_thresholds=(np.array([200,220,235],np.float32)-mean[0])/std[0]
    def __call__(self, x, mask):
        start=self.start; stop=start+self.crop
        m=mask[start:stop,start:stop].ravel()
        # x contains masked numerators after affine interpolation.
        good=m>=.95
        v=x[start:stop,start:stop].reshape(-1,3)/np.maximum(m[:,None],1e-6)
        good &= np.isfinite(v).all(axis=1)
        keep=good[self.inside]; rid=self.ring_ids[keep]; sid=self.sector_ids[keep]
        val=v[self.inside][keep]
        count=np.bincount(rid,minlength=N_RINGS)
        scount=np.bincount(sid,minlength=N_RINGS*8)
        coverage=count/np.maximum(self.total,1)
        usable=(count>=4)&(coverage>=.8)
        sector_ok=(scount>=2)&(scount>=.8*self.sector_total)
        all_sectors=sector_ok.reshape(N_RINGS,8).all(axis=1)&usable
        means=np.zeros((N_RINGS,3)); sigmas=means.copy(); az=means.copy()
        for k in range(3):
            t=val[:,k].astype(np.float64)
            mu=np.bincount(rid,weights=t,minlength=N_RINGS)/np.maximum(count,1)
            var=np.bincount(rid,weights=t*t,minlength=N_RINGS)/np.maximum(count,1)-mu*mu
            means[:,k]=np.where(usable,mu,0.)
            sigmas[:,k]=np.where(usable,np.sqrt(np.maximum(var,0.)),0.)
            sm=np.bincount(sid,weights=t,minlength=N_RINGS*8)/np.maximum(scount,1)
            az[:,k]=np.where(all_sectors,sm.reshape(N_RINGS,8).std(axis=1),0.)
        deltas=np.zeros_like(means)
        deltas[1:]=np.where((usable[1:]&usable[:-1])[:,None],means[1:]-means[:-1],0.)
        profile=np.column_stack([means,sigmas,deltas,az,coverage,usable,all_sectors]).ravel()
        cold_mask=good&self.cold_region
        cold_ok=cold_mask.sum()>=max(4,.8*self.cold_region.sum())
        cold=[float((v[cold_mask,0]<t).mean()) if cold_ok else 0. for t in self.cold_thresholds]
        contrast_ok=usable[0]&usable[1]
        out=np.r_[profile,cold,float(cold_ok),means[0,0]-means[1,0] if contrast_ok else 0.,float(contrast_ok)].astype(np.float32)
        if out.shape!=(RADIAL_DIM,) or not np.isfinite(out).all(): raise ValueError('Invalid radial profile')
        return out

class Crops(keras.utils.PyDataset):
    def __init__(self, root, df, spec, a, inner, train=False):
        super().__init__(workers=a.workers,use_multiprocessing=False,max_queue_size=4)
        self.root=root; self.df=df.reset_index(drop=True); self.rows=self.df.to_dict('records')
        self.spec=spec; self.a=a; self.inner=inner; self.train=train; self.epoch=0
        self.radial=RadialProfile(spec[0],spec[1],a.master_km,spec[2],spec[3]) if a.radial else None
        self.y=self.df.class_id.to_numpy(np.int32); self.wind=self.df.wind_kt.to_numpy(np.float32)/50.
        self.weights=np.ones(len(df),np.float32)
        self.wind_weights=np.ones(len(df),np.float32)
        if train:
            counts=self.df.class_id.value_counts(); group=self.df.groupby(['sid','class_id']).class_id.transform('size').to_numpy()
            self.weights=np.array([counts[c]**(-a.class_alpha) for c in self.y],np.float32)*group.astype(np.float32)**(-a.storm_alpha)
            self.weights/=self.weights.mean()
            # Independent regression weighting: do not double-count class weights.
            self.wind_weights=(1+a.intensity_weight*np.clip((self.wind*50-64)/73,0,1)).astype(np.float32)
            self.wind_weights/=self.wind_weights.mean()
        self.reset_indices()
    def reset_indices(self):
        self.indices=np.arange(len(self.rows))
        if self.train: np.random.default_rng(self.a.seed+self.epoch).shuffle(self.indices)
    def on_epoch_end(self):
        self.epoch+=1; self.reset_indices()
    def __len__(self): return (len(self.rows)+self.a.batch_size-1)//self.a.batch_size
    @staticmethod
    def crop(x,mask,crop,n):
        s=(x.shape[0]-crop)//2; x=x[s:s+crop,s:s+crop]; mask=mask[s:s+crop,s:s+crop]
        den=resize_float(mask,n); out=np.stack([resize_float(x[...,i],n) for i in range(3)],axis=-1)
        # x contains masked numerators; divide only where support is adequate.
        good=den>=.95
        out=np.where(good[...,None],out/np.maximum(den[...,None],1e-6),0.)
        return np.ascontiguousarray(np.concatenate([out,good[...,None].astype(np.float32)],axis=-1),dtype=np.float32)
    def image(self,i):
        r=self.rows[i]; native,primary,mean,std=self.spec
        with np.load(self.root/r['kelvin'],allow_pickle=False) as z:
            ir,wv,m1,m2=[z[k] for k in ('c13','c08','mask_c13','mask_c08')]
        if any(v.shape!=(native,native) for v in (ir,wv,m1,m2)): raise ValueError(f'Invalid tensor shape: {r["sample_id"]}')
        mask=m1.astype(bool)&m2.astype(bool)
        if not mask.any() or not np.isfinite(ir[mask]).all() or not np.isfinite(wv[mask]).all(): raise ValueError(f'Bad valid pixels: {r["sample_id"]}')
        ir=np.where(mask,ir,0.); wv=np.where(mask,wv,0.)
        x=np.where(mask[...,None],(np.stack([ir,wv,ir-wv],axis=-1)-mean)/std,0.).astype(np.float32)
        mask=mask.astype(np.float32)
        def pair(z,m):
            return (self.crop(z,m,primary,self.a.image_size),self.crop(z,m,self.inner,self.a.inner_image_size))
        def finish(pair,z,m):
            return (*pair,self.radial(z,m)) if self.radial is not None else pair
        if self.train and not self.a.no_augmentation:
            rng=np.random.default_rng(np.random.SeedSequence([self.a.seed,self.epoch,i]))
            k=int(rng.integers(4)); z=np.rot90(x,k); m=np.rot90(mask,k)
            theta=np.deg2rad(rng.uniform(-self.a.rotation,self.a.rotation)); co,si=np.cos(theta),np.sin(theta)
            mat=np.array([[co,-si],[si,co]]); center=np.full(2,(native-1)/2.)
            angle=rng.uniform(0,2*np.pi); radius=self.a.center_shift_px*np.sqrt(rng.random())
            shift=radius*np.array([np.cos(angle),np.sin(angle)])
            offset=center-mat@center+shift
            if self.a.rotation or self.a.center_shift_px:
                m=affine_transform(m,mat,offset,order=1,mode='constant',cval=0.,prefilter=False)
                z=np.stack([affine_transform(z[...,j],mat,offset,order=1,mode='constant',cval=0.,prefilter=False) for j in range(3)],axis=-1)
            a,b=pair(z,m)
            if a[...,3].mean()>=.88 and b[...,3].mean()>=.93: return finish((a,b),z,m)
        return finish(pair(x,mask),x,mask)
    def __getitem__(self,b):
        ix=self.indices[b*self.a.batch_size:(b+1)*self.a.batch_size]
        pairs=[self.image(int(i)) for i in ix]
        x={'context':np.stack([p[0] for p in pairs]),'core':np.stack([p[1] for p in pairs])}
        if self.radial is not None: x['radial']=np.stack([p[2] for p in pairs])
        if self.a.wind_weight>0:
            y={'category':self.y[ix], 'wind':self.wind[ix,None]}
            if self.train: return x,y,{'category':self.weights[ix],'wind':self.wind_weights[ix]}
            return x,y
        return (x,self.y[ix],self.weights[ix]) if self.train else (x,self.y[ix])

@keras.saving.register_keras_serializable(package='CycloneAI')
class OrderedCrossEntropy(keras.losses.Loss):
    """Hard categorical CE plus distance-aware cumulative-distribution penalty."""
    def __init__(self,ordinal_weight=.25,**kwargs):
        super().__init__(**kwargs); self.ordinal_weight=ordinal_weight
    def call(self,y_true,y_pred):
        y=ops.cast(ops.reshape(y_true,(-1,)), 'int32'); p=ops.cast(y_pred,'float32')
        ce=keras.losses.sparse_categorical_crossentropy(y,p)
        target=ops.one_hot(y,N_CLASSES); err=ops.cumsum(p-target,axis=-1)[...,:N_CLASSES-1]
        return ce+self.ordinal_weight*ops.mean(ops.square(err),axis=-1)
    def get_config(self): return {**super().get_config(),'ordinal_weight':self.ordinal_weight}


@keras.saving.register_keras_serializable(package='CycloneAI')
class MaskedTemperatureStats(layers.Layer):
    """Per-channel valid-pixel mean and std, retaining absolute thermal signal.
    Input temperatures have fixed training-set z-score normalization, not
    per-image normalization. Invalid pixels are excluded. No label input.
    """
    def call(self, inputs):
        x=ops.cast(inputs[...,:3],'float32')
        m=ops.cast(inputs[...,3:4]>.5,'float32')
        count=ops.maximum(ops.sum(m,axis=(1,2)),1.)
        mean=ops.sum(x*m,axis=(1,2))/count
        diff=x-mean[:,None,None,:]
        variance=ops.sum(ops.square(diff)*m,axis=(1,2))/count
        return ops.concatenate([mean,ops.sqrt(variance+1e-6)],axis=-1)
    def compute_output_shape(self,shape): return (shape[0],6)

@keras.saving.register_keras_serializable(package='CycloneAI')
class WindHuber(keras.losses.Loss):
    """Huber in units of delta knots, keeping the wind objective numerically relevant.
    Model outputs and labels remain in units of 50 knots for compatibility.
    """
    def __init__(self,delta_kt=10.,**kwargs):
        super().__init__(**kwargs); self.delta_kt=delta_kt
    def call(self,y_true,y_pred):
        e=ops.abs(ops.reshape(ops.cast(y_pred,'float32'),(-1,))-ops.reshape(ops.cast(y_true,'float32'),(-1,)))*50./self.delta_kt
        return ops.where(e<=1.,.5*ops.square(e),e-.5)
    def get_config(self): return {**super().get_config(),'delta_kt':self.delta_kt}


def epoch_lr(epoch,epochs,peak,warmup,min_ratio):
    """Epoch-indexed linear warmup then cosine; final epoch reaches the floor."""
    if warmup and epoch<warmup: return peak*(.1+.9*(epoch+1)/warmup)
    progress=(epoch-warmup)/max(1,epochs-warmup-1)
    return peak*(min_ratio+(1-min_ratio)*.5*(1+math.cos(math.pi*min(1.,max(0.,progress)))))


def build(a):
    def width(n): return max(8,int(round(n*a.width/8))*8)
    def conv(x,f,k=3,stride=1):
        return layers.Conv2D(f,k,strides=stride,padding='same',use_bias=False,kernel_initializer='he_normal')(x)
    def norm(x): return layers.GroupNormalization(groups=8,epsilon=1e-5)(x)
    def block(x,f,stride=1):
        skip=x; y=layers.Activation('swish')(norm(conv(x,f,3,stride)))
        y=norm(conv(y,f))
        if stride!=1 or int(x.shape[-1])!=f: skip=norm(conv(skip,f,1,stride))
        return layers.Activation('swish')(layers.Add()([skip,y]))
    def branch(inp,filters):
        x=layers.Activation('swish')(norm(conv(inp,width(filters[0]),5,2)))
        pooled=[]
        for j,f in enumerate(filters):
            f=width(f); x=block(x,f,2 if j else 1)
            if j>=1: x=block(x,f)
            # Retain earlier-stage texture and convective structure, not only deepest features.
            if a.multilevel and j in (1,2): pooled.append(layers.GlobalAveragePooling2D()(x))
            if j>=1 and a.spatial_dropout: x=layers.SpatialDropout2D(a.spatial_dropout)(x)
        pooled.extend([layers.GlobalAveragePooling2D()(x),layers.GlobalMaxPooling2D()(x)])
        return layers.Concatenate()(pooled)
    context=layers.Input((a.image_size,a.image_size,4),name='context',dtype='float32')
    core=layers.Input((a.inner_image_size,a.inner_image_size,4),name='core',dtype='float32')
    x=branch(context,[32,48,80,128]); y=branch(core,[24,40,64,96])
    z=layers.Concatenate()([x,y]); z=layers.LayerNormalization()(z)
    if a.thermal_stats:
        # Bypass per-sample group/layer normalization for the physical statistics.
        z=layers.Concatenate()([z,MaskedTemperatureStats(dtype='float32')(context),MaskedTemperatureStats(dtype='float32')(core)])
    inputs={'context':context,'core':core}
    if a.radial:
        radial=layers.Input((RADIAL_DIM,),name='radial',dtype='float32')
        inputs['radial']=radial
        r=layers.Dense(64,activation='swish',kernel_initializer='he_normal',name='radial_dense_1')(radial)
        r=layers.Dense(64,activation='swish',kernel_initializer='he_normal',name='radial_dense_2')(r)
        z=layers.Concatenate(name='radial_fusion')([z,r])
    z=layers.Dense(width(192),activation='swish',kernel_initializer='he_normal')(z); z=layers.Dropout(a.dropout)(z)
    cat=layers.Dense(N_CLASSES,activation='softmax',name='category',dtype='float32')(z)
    output=cat
    if a.wind_weight>0:
        # Independent final representation mitigates direct competition at the output layer.
        wind=layers.Dense(width(64),activation='swish',kernel_initializer='he_normal')(z)
        output={'category':cat,'wind':layers.Dense(1,activation='softplus',name='wind',dtype='float32')(wind)}
    return keras.Model(inputs,output,name='CycloneCoreCNN_V3_4Class')

def metrics(df,p):
    if p.shape!=(len(df),N_CLASSES): raise ValueError('Unexpected classification prediction shape')
    y=df.class_id.to_numpy(int); pred=p.argmax(1); k=cohen_kappa_score(y,pred,labels=list(range(N_CLASSES)),weights='quadratic'); sup=np.unique(y)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=list(range(N_CLASSES)),average='macro',zero_division=0)), 'macro_f1_supported':float(f1_score(y,pred,labels=sup,average='macro',zero_division=0)), 'balanced_accuracy':float(recall_score(y,pred,labels=sup,average='macro',zero_division=0)), 'quadratic_kappa':float(k) if np.isfinite(k) else None,'within_one_category':float(np.mean(np.abs(y-pred)<=1)),'category_mae':float(np.mean(np.abs(y-pred))), 'storm_mean_accuracy':float(pd.DataFrame({'sid':df.sid,'correct':y==pred}).groupby('sid').correct.mean().mean())}


def predict(model,ds,tta=1):
    if tta==1:
        out=model.predict(ds,verbose=0)
        return (out['category'],out['wind'].reshape(-1)*50.) if isinstance(out,dict) else (out,None)
    ps=[]; ws=[]
    for b in range(len(ds)):
        x=ds[b][0]; pp=[]; ww=[]
        for k in range(4):
            z={name:np.ascontiguousarray(np.rot90(v,k,axes=(1,2))) if name in ('context','core') else v for name,v in x.items()}
            out=model.predict_on_batch(z)
            if isinstance(out,dict): pp.append(out['category']); ww.append(out['wind'].reshape(-1)*50.)
            else: pp.append(out)
        ps.append(np.mean(pp,axis=0))
        if ww: ws.append(np.mean(ww,axis=0))
    return np.concatenate(ps),np.concatenate(ws) if ws else None

def all_metrics(df,p,w):
    if not np.isfinite(p).all() or (w is not None and not np.isfinite(w).all()): raise ValueError('Nonfinite predictions; check learning rate and precision')
    m=metrics(df,p); y=df.class_id.to_numpy(int)
    m['cross_entropy']=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-7,1)).mean())
    if w is not None:
        e=w-df.wind_kt.to_numpy(float); m.update(wind_mae_kt=float(np.abs(e).mean()),wind_rmse_kt=float(np.sqrt(np.mean(e**2))),wind_bias_kt=float(e.mean()),storm_mean_wind_mae_kt=float(pd.DataFrame({'sid':df.sid,'ae':np.abs(e)}).groupby('sid').ae.mean().mean()))
    return m

class EvalCallback(keras.callbacks.Callback):
    def __init__(self,val,natural,a): super().__init__(); self.val=val; self.natural=natural; self.a=a
    def on_epoch_end(self,epoch,logs=None):
        p,w=predict(self.model,self.val,self.a.tta); m=all_metrics(self.val.df,p,w)
        for k,v in m.items(): logs['val_'+k]=v if v is not None else float('nan')
        if self.a.natural_train_every:
            nm=all_metrics(self.natural.df,*predict(self.model,self.natural,self.a.tta)) if (epoch+1)%self.a.natural_train_every==0 else {k:float('nan') for k in m}
            for k,v in nm.items(): logs['natural_train_'+k]=v if v is not None else float('nan')
        print(f'val macro F1={m["macro_f1"]:.4f}; accuracy={m["accuracy"]:.4f}',flush=True)

def evaluate(model,ds,out,tta):
    out.mkdir(exist_ok=True,parents=True); p,w=predict(model,ds,tta); df=ds.df.copy(); y=df.class_id.to_numpy(int); pred=p.argmax(1); m=all_metrics(df,p,w); m.update(samples=len(df),storms=int(df.sid.nunique()),tta=tta,class_scheme=SCHEME,classes=CLASSES); dump(out/'metrics.json',m)
    df['pred_class']=pred; df['pred_class_name']=[CLASSES[i] for i in pred]
    for i,c in enumerate(CLASSES): df['prob_'+c]=p[:,i]
    if w is not None: df['pred_wind_kt']=w; df['wind_error_kt']=w-df.wind_kt; df['pred_class_from_wind']=np.searchsorted(BOUNDS,w,side='right')
    df.to_csv(out/'predictions.csv',index=False)
    report=classification_report(y,pred,labels=list(range(N_CLASSES)),target_names=CLASSES,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(out/'classification_report.csv')
    cm=confusion_matrix(y,pred,labels=list(range(N_CLASSES))); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_matrix.csv')
    for normalize in (False,True):
        z=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0) if normalize else cm
        fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(z,cmap='Blues',vmin=0,vmax=1 if normalize else max(1,cm.max())); ax.set(xticks=range(N_CLASSES),yticks=range(N_CLASSES),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title='Confusion matrix')
        for i in range(N_CLASSES):
            for j in range(N_CLASSES): ax.text(j,i,f'{z[i,j]:.2f}' if normalize else str(z[i,j]),ha='center',va='center',color='white' if z[i,j]>.55*im.norm.vmax else 'black')
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/('confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'),dpi=160); plt.close(fig)
    storms=[]; byclass=[]
    for sid,g in df.groupby('sid'):
        row=dict(sid=sid,samples=len(g),accuracy=float((g.class_id==g.pred_class).mean()))
        if w is not None: row['wind_mae_kt']=float(g.wind_error_kt.abs().mean())
        storms.append(row)
    for c,g in df.groupby('class_id'):
        row=dict(class_id=int(c),class_name=CLASSES[int(c)],samples=len(g),storms=g.sid.nunique(),recall=float((g.class_id==g.pred_class).mean()))
        if w is not None: row.update(wind_mae_kt=g.wind_error_kt.abs().mean(),wind_bias_kt=g.wind_error_kt.mean())
        byclass.append(row)
    pd.DataFrame(storms).to_csv(out/'storm_metrics.csv',index=False); pd.DataFrame(byclass).to_csv(out/'by_class.csv',index=False)
    fig,ax=plt.subplots(); ax.bar(CLASSES,[report[c]['f1-score'] for c in CLASSES]); ax.set(ylim=(0,1),ylabel='F1'); fig.tight_layout(); fig.savefig(out/'per_class_f1.png',dpi=160); plt.close(fig)
    if w is not None:
        fig,ax=plt.subplots(); ax.scatter(df.wind_kt,w,s=12,alpha=.5); lo=min(df.wind_kt.min(),w.min()); hi=max(df.wind_kt.max(),w.max()); ax.plot([lo,hi],[lo,hi],'k--'); ax.set(xlabel='IBTrACS wind (kt)',ylabel='Predicted wind (kt)'); fig.tight_layout(); fig.savefig(out/'wind_scatter.png',dpi=160); plt.close(fig)
    print(out.name,m,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/processed/classification_v3_1')); p.add_argument('--output',type=Path,default=Path('runs/classification/custom_tf_v3_4class'))
    p.add_argument('--image-size',type=int,default=256); p.add_argument('--inner-image-size',type=int,default=160); p.add_argument('--width',type=float,default=1.5,help='Width multiplier; 0.75 small, 1 baseline, 1.5 large')
    p.add_argument('--batch-size',type=int,default=8); p.add_argument('--epochs',type=int,default=80); p.add_argument('--lr',type=float,default=.0005); p.add_argument('--weight-decay',type=float,default=.0001); p.add_argument('--workers',type=int,default=4); p.add_argument('--seed',type=int,default=42); p.add_argument('--patience',type=int,default=20); p.add_argument('--dropout',type=float,default=.20); p.add_argument('--spatial-dropout',type=float,default=.03)
    p.add_argument('--class-alpha',type=float,default=.25); p.add_argument('--storm-alpha',type=float,default=.25); p.add_argument('--ordinal-weight',type=float,default=.25); p.add_argument('--wind-weight',type=float,default=.2,help='Set zero for classification only'); p.add_argument('--rotation',type=float,default=12.); p.add_argument('--center-shift-px',type=float,default=12.); p.add_argument('--no-augmentation',action='store_true'); p.add_argument('--natural-train-every',type=int,default=5); p.add_argument('--tta',type=int,choices=[1,4],default=1)
    p.add_argument('--warmup',type=int,default=5); p.add_argument('--min-lr-ratio',type=float,default=.02)
    p.add_argument('--schedule',choices=['cosine','plateau'],default='cosine')
    p.add_argument('--wind-delta-kt',type=float,default=10.); p.add_argument('--intensity-weight',type=float,default=0.,help='Optional extra relative wind loss for intense storms')
    p.add_argument('--no-multilevel',dest='multilevel',action='store_false'); p.add_argument('--no-thermal-stats',dest='thermal_stats',action='store_false')
    p.add_argument('--no-radial',dest='radial',action='store_false',help='Disable radial profile for controlled V2-style ablation')
    p.add_argument('--eval-tta',type=int,choices=[1,4],default=None,help='Override TTA in eval-only; output goes to separate folders')
    p.set_defaults(multilevel=True,thermal_stats=True,radial=True)
    p.add_argument('--mixed-precision',action='store_true'); p.add_argument('--verify-hashes',action='store_true'); p.add_argument('--validate-only',action='store_true'); p.add_argument('--eval-only',action='store_true'); p.add_argument('--evaluate-test',action='store_true'); a=p.parse_args(); a.data=a.data.resolve(); a.output=a.output.resolve()
    if min(a.image_size,a.inner_image_size)<32 or min(a.batch_size,a.epochs,a.workers,a.patience)<=0 or not .5<=a.width<=2 or a.lr<=0 or a.weight_decay<0 or not 0<=a.dropout<1 or not 0<=a.spatial_dropout<1 or not 0<=a.class_alpha<=1 or not 0<=a.storm_alpha<=1 or min(a.ordinal_weight,a.wind_weight,a.rotation,a.center_shift_px,a.natural_train_every)<0 or a.rotation>45: p.error('Invalid training parameters')
    if a.warmup<0 or not 0<a.min_lr_ratio<=1 or a.wind_delta_kt<=0 or not 0<=a.intensity_weight<=4: p.error('Invalid schedule or wind parameters')
    if a.evaluate_test and not a.eval_only: p.error('Test requires --eval-only')
    if a.eval_tta is not None and not a.eval_only: p.error('--eval-tta requires --eval-only')
    if keras.backend.backend()=='tensorflow':
        import tensorflow as tf
        gpus=tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            try: tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError: pass
        print('TensorFlow GPUs:',gpus,flush=True)
        if not gpus: print('WARNING: TensorFlow sees no GPU; training will use CPU.',flush=True)
        if a.mixed_precision and not gpus: p.error('Mixed precision requires a supported GPU in TensorFlow mode')
    keras.utils.set_random_seed(a.seed)
    if a.mixed_precision: keras.mixed_precision.set_global_policy('mixed_float16')
    frames,spec,digest=inventory(a.data,a.verify_hashes); cfg=read(a.data/'metadata/preprocessing_config.json'); inner=int(cfg['inner_size']); a.master_km=float(cfg['master_km'])
    if not np.isfinite(a.master_km) or a.master_km<=0: raise ValueError('Invalid master coverage')
    if not 0<inner<=spec[1] or (spec[0]-inner)%2: raise ValueError('Invalid inner crop size')
    print(f'Physical coverage: context {cfg["master_km"]*spec[1]/spec[0]:.1f} km, core {cfg["master_km"]*inner/spec[0]:.1f} km')
    print('Four-class mapping:',CLASS_MAP,flush=True)
    if a.validate_only:
        for s in SPLITS:
            ds=Crops(a.data,frames[s],spec,a,inner); ds[0]
        print('Manifest checks and first batch of each split passed.'); return
    if a.eval_only:
        old=read(a.output/'config.json')
        if old.get('class_scheme')!=SCHEME: raise ValueError('Four-class checkpoint required; use the original trainer for seven-class runs')
        if old['dataset_digest']!=digest: raise ValueError('Dataset has changed since training')
        for k in ('image_size','inner_image_size','width','wind_weight','tta'): setattr(a,k,old[k])
        a.radial=old.get('radial',False)  # V2 checkpoints are evaluation-compatible.
        if a.eval_tta is not None: a.tta=a.eval_tta
        model=keras.models.load_model(a.output/'best.keras',compile=False)
        if model.get_layer('category').units!=N_CLASSES: raise ValueError('Checkpoint output size mismatch')
    else:
        if a.output.exists() and any(a.output.iterdir()): raise ValueError('Output directory not empty; choose a new run')
        a.output.mkdir(parents=True,exist_ok=True)
        config={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}; config.update(trainer_version=TRAINER_VERSION,radial_dim=RADIAL_DIM if a.radial else 0,ring_edges_km=list(RING_EDGES) if a.radial else [],class_scheme=SCHEME,classes=CLASSES,class_names=CLASS_NAMES,wind_thresholds_kt=BOUNDS,source_class_mapping=CLASS_MAP,dataset_digest=digest,backend=keras.backend.backend(),normalization=read(a.data/'metadata/normalization.json'),preprocessing=cfg); dump(a.output/'config.json',config)
        model=build(a); opt=keras.optimizers.AdamW(learning_rate=a.lr,weight_decay=a.weight_decay,global_clipnorm=1.)
        opt.exclude_from_weight_decay(var_names=['bias','beta','gamma'])
        loss=OrderedCrossEntropy(a.ordinal_weight)
        if a.wind_weight>0:
            model.compile(optimizer=opt,loss={'category':loss,'wind':WindHuber(a.wind_delta_kt)},loss_weights={'category':1.,'wind':a.wind_weight},metrics={'category':[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],'wind':[keras.metrics.MeanAbsoluteError(name='mae_scaled')]},jit_compile=False)
        else: model.compile(optimizer=opt,loss=loss,metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')],jit_compile=False)
        model.summary(); (a.output/'model_summary.txt').write_text('\n'.join([]))
        with (a.output/'model_summary.txt').open('w') as f: model.summary(print_fn=lambda line:f.write(line+'\n'))
        train=Crops(a.data,frames['train'],spec,a,inner,True); natural=Crops(a.data,frames['train'],spec,a,inner); val=Crops(a.data,frames['val'],spec,a,inner)
        warmup=min(a.warmup,max(0,a.epochs-1))
        scheduler=(keras.callbacks.LearningRateScheduler(lambda e,lr:epoch_lr(e,a.epochs,a.lr,warmup,a.min_lr_ratio),verbose=1) if a.schedule=='cosine' else keras.callbacks.ReduceLROnPlateau(monitor='val_macro_f1',mode='max',factor=.5,patience=4,min_lr=a.lr*a.min_lr_ratio))
        callbacks=[EvalCallback(val,natural,a),keras.callbacks.ModelCheckpoint(str(a.output/'best.keras'),monitor='val_macro_f1',mode='max',save_best_only=True),scheduler,keras.callbacks.CSVLogger(str(a.output/'history.csv')),keras.callbacks.EarlyStopping(monitor='val_macro_f1',mode='max',patience=a.patience,start_from_epoch=warmup if a.schedule=='cosine' else 0)]
        model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks,shuffle=False); model.save(a.output/'last.keras')
        h=pd.read_csv(a.output/'history.csv'); row=h.loc[h.val_macro_f1.idxmax()]; dump(a.output/'selection.json',dict(epoch=int(row.epoch)+1,val_macro_f1=float(row.val_macro_f1),val_accuracy=float(row.val_accuracy),criterion='validation macro F1',parameters=model.count_params()))
        for name,cols in [('accuracy',['accuracy','category_accuracy','val_accuracy','natural_train_accuracy']),('macro_f1',['val_macro_f1','natural_train_macro_f1']),('loss',['loss','val_loss']),('cross_entropy',['val_cross_entropy','natural_train_cross_entropy']),('wind_mae',['val_wind_mae_kt','natural_train_wind_mae_kt']),('learning_rate',['learning_rate'])]:
            present=[c for c in cols if c in h and h[c].notna().any()]
            if not present: continue
            fig,ax=plt.subplots()
            for col in present:
                good=h[col].notna(); ax.plot(h.epoch[good]+1,h[col][good],label=col,marker='o' if col.startswith('natural') else None)
            ax.set(xlabel='Epoch',title=name); ax.legend(); fig.tight_layout(); fig.savefig(a.output/f'{name}.png',dpi=160); plt.close(fig)
        model=keras.models.load_model(a.output/'best.keras',compile=False)
    suffix=f'_tta{a.tta}' if a.eval_tta is not None else ''
    evaluate(model,Crops(a.data,frames['val'],spec,a,inner),a.output/('validation'+suffix),a.tta)
    if a.evaluate_test: evaluate(model,Crops(a.data,frames['test'],spec,a,inner),a.output/('test'+suffix),a.tta)

if __name__=='__main__': main()
