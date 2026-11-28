#!/usr/bin/env python3
"""Keras adaptation of supplied ResNet50 and 3-convolution CNN notebook.
Default backend TensorFlow; KERAS_BACKEND=torch is also supported by Keras 3.
Uses classification_v3_1 manifests, never random image splitting.
"""
import argparse, hashlib, json, os
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import keras
from keras import layers
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                             f1_score, recall_score, cohen_kappa_score)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES=['TD','TS','CAT1','CAT2','CAT3','CAT4','CAT5']
BOUNDS=[34,64,83,96,113,137]
SPLITS=('train','val','test')

def read(p): return json.loads(p.read_text())
def dump(p,v): p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def inventory(root, verify=False):
    meta=root/'metadata'; man=root/'manifests'
    norm=read(meta/'normalization.json'); cfg=read(meta/'preprocessing_config.json')
    if read(meta/'classes.json')!={str(i):c for i,c in enumerate(CLASSES)}: raise ValueError('Class mapping mismatch')
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
            if not all([i,sid,scene,r['label_time']]) or not 0<w<=250 or c!=float(r['class_id']) or c!=np.searchsorted(BOUNDS,w,side='right') or r['class_name']!=CLASSES[c]: raise ValueError(f'Invalid label {i}')
            if r['split']!=s or mapping.get(sid)!=s or storms.setdefault(sid,s)!=s or scenes.setdefault(scene,s)!=s or i in seen or fix in fixes: raise ValueError(f'Duplicate/leakage: {i}')
            if not p.is_relative_to(root) or not p.is_file() or p in paths: raise ValueError(f'Invalid tensor {p}')
            if len(h)!=64 or any(ch not in '0123456789abcdef' for ch in h) or hashes.setdefault(h,s)!=s: raise ValueError(f'Invalid checksum or cross-split duplicate: {i}')
            if verify:
                with p.open('rb') as f: actual=hashlib.file_digest(f,'sha256').hexdigest()
                if actual!=h: raise ValueError(f'Checksum mismatch {i}')
            seen.add(i); fixes.add(fix); paths.add(p); digest.update(f'{r["kelvin"]}:{p.stat().st_size}:{h}'.encode())
        counts=np.bincount(df.class_id.astype(int),minlength=7)
        print(f'{s}: {len(df)} samples, {df.sid.nunique()} storms, counts={counts.tolist()}',flush=True)
        if s=='train' and np.any(counts==0): raise ValueError('Training missing classes')
        if np.any(counts==0): print('WARNING: missing classes:',[CLASSES[i] for i in np.flatnonzero(counts==0)])
        frames[s]=df
    return frames, (native,crop,mean,std), digest.hexdigest()

def resize_float(x,n):
    return np.asarray(Image.fromarray(np.asarray(x,np.float32)).resize((n,n),Image.Resampling.BILINEAR),dtype=np.float32)

class Crops(keras.utils.PyDataset):
    def __init__(self,root,df,spec,a,train=False):
        super().__init__(workers=a.workers,use_multiprocessing=False,max_queue_size=4)
        self.root=root; self.df=df.reset_index(drop=True); self.rows=self.df.to_dict('records'); self.spec=spec; self.a=a; self.train=train; self.epoch=0
        counts=np.bincount(self.df.class_id.astype(int),minlength=7)
        self.weights=np.ones(7,np.float32)
        if train and a.class_alpha:
            self.weights=counts.astype(np.float32)**(-a.class_alpha); self.weights/=np.average(self.weights,weights=counts)
        self.indices=np.arange(len(df)); self.reset_indices()
    def reset_indices(self):
        self.indices=np.arange(len(self.rows))
        if self.train: np.random.default_rng(self.a.seed+self.epoch).shuffle(self.indices)
    def on_epoch_end(self): self.epoch+=1; self.reset_indices()
    def __len__(self): return (len(self.rows)+self.a.batch_size-1)//self.a.batch_size
    def image(self,i):
        r=self.rows[i]; native,crop,mean,std=self.spec; start=(native-crop)//2; sl=np.s_[start:start+crop]
        with np.load(self.root/r['kelvin'],allow_pickle=False) as z:
            ir,wv,m1,m2=[z[k] for k in ('c13','c08','mask_c13','mask_c08')]
        if any(v.shape!=(native,native) for v in (ir,wv,m1,m2)): raise ValueError(f'Invalid shape: {r["sample_id"]}')
        ir,wv=ir[sl,sl],wv[sl,sl]; mask=m1[sl,sl].astype(bool)&m2[sl,sl].astype(bool)
        if not mask.any() or not np.isfinite(ir[mask]).all() or not np.isfinite(wv[mask]).all(): raise ValueError(f'Bad pixels: {r["sample_id"]}')
        x=np.stack([ir,wv,ir-wv],axis=-1).astype(np.float32); x=np.where(mask[...,None],(x-mean)/std,0.)
        if self.train and self.a.rotate:
            k=np.random.default_rng(np.random.SeedSequence([self.a.seed,self.epoch,i])).integers(4); x=np.rot90(x,k); mask=np.rot90(mask,k)
        n=self.a.image_size
        if n!=crop:
            den=resize_float(mask,n); x=np.stack([resize_float(x[...,j],n) for j in range(3)],axis=-1); x=np.where(den[...,None]>=.99,x/np.maximum(den[...,None],1e-6),0.)
        if self.a.model=='resnet50':
            # Fixed pseudo-RGB mapping of z-scores, followed by Keras ResNet BGR centering.
            x=np.clip(127.5+32*x,0,255); x=x[...,::-1]-np.array([103.939,116.779,123.68],np.float32)
        return np.ascontiguousarray(x,dtype=np.float32)
    def __getitem__(self,b):
        ix=self.indices[b*self.a.batch_size:(b+1)*self.a.batch_size]; x=np.stack([self.image(int(i)) for i in ix]); y=self.df.class_id.to_numpy(dtype=np.int32)[ix]
        if self.train and self.a.class_alpha: return x,y,self.weights[y]
        return x,y

def build(a):
    inp=layers.Input((a.image_size,a.image_size,3),dtype='float32'); x=inp
    if a.model=='resnet50':
        base=keras.applications.ResNet50(include_top=False,weights=None if a.no_pretrained else 'imagenet',input_shape=(224,224,3)); base.trainable=False
        if a.no_pretrained: base.trainable=True
        x=layers.Resizing(224,224)(x); x=base(x,training=False if not a.no_pretrained else None); x=layers.GlobalAveragePooling2D()(x); x=layers.Dense(128,activation='relu',name='head_dense_128')(x)
    else:
        for filters in (32,64,128):
            x=layers.Conv2D(filters,3,padding='same')(x); x=layers.LeakyReLU(negative_slope=.2)(x); x=layers.MaxPooling2D()(x)
        x=layers.Flatten()(x) if a.model=='cnn' else layers.GlobalAveragePooling2D()(x)
        x=layers.Dense(256)(x); x=layers.LeakyReLU(negative_slope=.2)(x); x=layers.Dropout(.3)(x)
    return keras.Model(inp,layers.Dense(7,activation='softmax',dtype='float32',name='predictions')(x),name=a.model)

def metrics(df,p):
    y=df.class_id.to_numpy(int); pred=p.argmax(1); k=cohen_kappa_score(y,pred,labels=list(range(7)),weights='quadratic'); sup=np.unique(y)
    return {'accuracy':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=list(range(7)),average='macro',zero_division=0)), 'macro_f1_supported':float(f1_score(y,pred,labels=sup,average='macro',zero_division=0)), 'balanced_accuracy':float(recall_score(y,pred,labels=sup,average='macro',zero_division=0)), 'quadratic_kappa':float(k) if np.isfinite(k) else None,'within_one_category':float(np.mean(np.abs(y-pred)<=1)),'category_mae':float(np.mean(np.abs(y-pred))), 'storm_mean_accuracy':float(pd.DataFrame({'sid':df.sid,'correct':y==pred}).groupby('sid').correct.mean().mean())}

class ValF1(keras.callbacks.Callback):
    def __init__(self,ds): super().__init__(); self.ds=ds
    def on_epoch_end(self,epoch,logs=None):
        p=self.model.predict(self.ds,verbose=0); m=metrics(self.ds.df,p); logs['val_macro_f1']=m['macro_f1']; logs['val_balanced_accuracy']=m['balanced_accuracy']; print(f' — val_macro_f1={m["macro_f1"]:.4f}',flush=True)

def evaluate(model,ds,out):
    out.mkdir(exist_ok=True,parents=True); p=model.predict(ds,verbose=0); df=ds.df.copy(); y=df.class_id.to_numpy(int); pred=p.argmax(1); m=metrics(df,p); m['samples']=len(df); m['storms']=int(df.sid.nunique()); m['loss']=float(-np.log(np.clip(p[np.arange(len(y)),y],1e-7,1)).mean()); dump(out/'metrics.json',m)
    df['pred_class']=pred; df['pred_class_name']=[CLASSES[i] for i in pred]
    for i,c in enumerate(CLASSES): df['prob_'+c]=p[:,i]
    df.to_csv(out/'predictions.csv',index=False); report=classification_report(y,pred,labels=list(range(7)),target_names=CLASSES,output_dict=True,zero_division=0); pd.DataFrame(report).T.to_csv(out/'classification_report.csv')
    cm=confusion_matrix(y,pred,labels=list(range(7))); pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(out/'confusion_matrix.csv')
    for norm in (False,True):
        z=np.divide(cm,cm.sum(1,keepdims=True),out=np.zeros_like(cm,dtype=float),where=cm.sum(1,keepdims=True)>0) if norm else cm
        fig,ax=plt.subplots(figsize=(8,6)); im=ax.imshow(z,cmap='Blues',vmin=0,vmax=1 if norm else max(1,cm.max())); ax.set(xticks=range(7),yticks=range(7),xticklabels=CLASSES,yticklabels=CLASSES,xlabel='Predicted',ylabel='Actual',title='Confusion matrix')
        for i in range(7):
            for j in range(7): ax.text(j,i,f'{z[i,j]:.2f}' if norm else str(z[i,j]),ha='center',va='center',color='white' if z[i,j]>.55*im.norm.vmax else 'black')
        fig.colorbar(im,ax=ax); fig.tight_layout(); fig.savefig(out/('confusion_matrix_normalized.png' if norm else 'confusion_matrix.png'),dpi=160); plt.close(fig)
    pd.DataFrame([dict(sid=sid,samples=len(g),accuracy=float((g.class_id==g.pred_class).mean())) for sid,g in df.groupby('sid')]).to_csv(out/'storm_metrics.csv',index=False)
    print(out.name,m,flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--data',type=Path,default=Path('data/processed/classification_v3_1')); p.add_argument('--output',type=Path,default=Path('runs/classification/reference_cnn')); p.add_argument('--model',choices=['cnn','cnn-gap','resnet50'],default='cnn'); p.add_argument('--image-size',type=int,default=128); p.add_argument('--batch-size',type=int,default=32); p.add_argument('--epochs',type=int,default=20); p.add_argument('--lr',type=float,default=.001); p.add_argument('--workers',type=int,default=2); p.add_argument('--seed',type=int,default=42); p.add_argument('--patience',type=int,default=0,help='0: train all epochs'); p.add_argument('--class-alpha',type=float,default=0.); p.add_argument('--rotate',action='store_true'); p.add_argument('--no-pretrained',action='store_true'); p.add_argument('--mixed-precision',action='store_true'); p.add_argument('--verify-hashes',action='store_true'); p.add_argument('--validate-only',action='store_true'); p.add_argument('--eval-only',action='store_true'); p.add_argument('--evaluate-test',action='store_true'); a=p.parse_args(); a.data=a.data.resolve(); a.output=a.output.resolve()
    if min(a.image_size,a.batch_size,a.epochs,a.workers)<=0 or a.image_size<8 or a.lr<=0 or a.patience<0 or not 0<=a.class_alpha<=1: p.error('Invalid training parameters')
    if a.evaluate_test and not a.eval_only: p.error('Test requires --eval-only to keep selection independent')
    keras.utils.set_random_seed(a.seed)
    if keras.backend.backend()=='tensorflow':
        import tensorflow as tf
        for gpu in tf.config.list_physical_devices('GPU'):
            try: tf.config.experimental.set_memory_growth(gpu,True)
            except RuntimeError: pass
        print('TensorFlow GPUs:',tf.config.list_physical_devices('GPU'),flush=True)
    if a.mixed_precision: keras.mixed_precision.set_global_policy('mixed_float16')
    frames,spec,digest=inventory(a.data,a.verify_hashes)
    if a.validate_only: return
    val=Crops(a.data,frames['val'],spec,a)
    if a.eval_only:
        cfg=read(a.output/'config.json')
        for k,v in {'model':a.model,'image_size':a.image_size,'dataset_digest':digest}.items():
            if cfg[k]!=v: raise ValueError(f'Evaluation configuration mismatch: {k}')
        model=keras.models.load_model(a.output/'best.keras',compile=False)
    else:
        if a.output.exists() and any(a.output.iterdir()): raise ValueError('Output directory not empty; choose a new run')
        a.output.mkdir(parents=True,exist_ok=True); cfg={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}; cfg.update(dataset_digest=digest,backend=keras.backend.backend(),normalization=read(a.data/'metadata/normalization.json')); dump(a.output/'config.json',cfg)
        model=build(a); model.compile(optimizer=keras.optimizers.Adam(a.lr),loss=keras.losses.SparseCategoricalCrossentropy(),metrics=[keras.metrics.SparseCategoricalAccuracy(name='accuracy')]); model.summary()
        train=Crops(a.data,frames['train'],spec,a,True)
        callbacks=[ValF1(val),keras.callbacks.ModelCheckpoint(str(a.output/'best.keras'),monitor='val_macro_f1',mode='max',save_best_only=True),keras.callbacks.CSVLogger(str(a.output/'history.csv'))]
        if a.patience: callbacks.append(keras.callbacks.EarlyStopping(monitor='val_macro_f1',mode='max',patience=a.patience))
        model.fit(train,validation_data=val,epochs=a.epochs,callbacks=callbacks,shuffle=False); model.save(a.output/'last.keras'); h=pd.read_csv(a.output/'history.csv'); row=h.loc[h.val_macro_f1.idxmax()]; dump(a.output/'selection.json',{'epoch':int(row.epoch)+1,'val_macro_f1':float(row.val_macro_f1),'val_accuracy':float(row.val_accuracy),'criterion':'validation macro F1'})
        for name,cols in [('accuracy',['accuracy','val_accuracy']),('loss',['loss','val_loss']),('macro_f1',['val_macro_f1'])]:
            fig,ax=plt.subplots();
            for col in cols: ax.plot(h.epoch+1,h[col],label=col)
            ax.set(xlabel='Epoch',title=name); ax.legend(); fig.tight_layout(); fig.savefig(a.output/f'{name}.png',dpi=160); plt.close(fig)
        model=keras.models.load_model(a.output/'best.keras',compile=False)
    evaluate(model,val,a.output/'validation')
    if a.evaluate_test: evaluate(model,Crops(a.data,frames['test'],spec,a),a.output/'test')

if __name__=='__main__': main()
