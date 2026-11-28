#!/usr/bin/env python3
"""Train the original six-class ClassCNN on its HDF5/NPY dataset.

No V3.1 data, pretrained weights, regression or auxiliary objectives are used.
The default split reproduces the source's stratified random 80/20 image split.
This is NOT a storm-disjoint or independent test evaluation.
Only load a trusted NPY file: the original label format requires allow_pickle=True.
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    f1_score, precision_score, recall_score, cohen_kappa_score,
    matthews_corrcoef, log_loss, roc_curve, auc, precision_recall_curve,
    average_precision_score)
# TensorFlow by default; Keras backend override is only for portability.
os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
import keras
from keras import layers

CLASSES = ['Low Pressure', 'Depression', 'Deep Depression', 'Cyclonic Storm',
           'Severe Cyclonic Storm', 'Very Severe Cyclonic Storm']
SHORT = ['LP', 'D', 'DD', 'CS', 'SCS', 'VSCS']
IDS = np.arange(len(CLASSES))
IMG_SIZE = 128


def classify_wind(wind):
    """Preserve exact source boundaries. Invalid/out-of-range labels are -1."""
    wind = np.asarray(wind, dtype=np.float64)
    out = np.full(wind.shape, -1, dtype=np.int64)
    good = np.isfinite(wind) & (wind >= 0) & (wind <= 221)
    conditions = [wind < 31, (wind >= 31) & (wind <= 49),
                  (wind > 49) & (wind <= 61), (wind > 61) & (wind <= 88),
                  (wind > 88) & (wind <= 117), (wind > 117) & (wind <= 221)]
    for i, condition in enumerate(conditions):
        out[good & condition] = i
    return out


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def read_images(path, indices):
    """h5py fancy indexing requires sorted, unique indices; restore input order."""
    indices = np.asarray(indices, dtype=np.int64)
    unique, inverse = np.unique(indices, return_inverse=True)
    with h5py.File(path, 'r') as f:
        x = np.asarray(f['Images'][unique, :, :, :3], dtype=np.float32)[inverse]
    if not np.isfinite(x).all() or x.min() < 0 or x.max() > 255:
        raise ValueError('Expected finite 0..255 image values as in ClassCNN.py; check dataset scaling.')
    return x / np.float32(255.0)


class ImageBatches(keras.utils.PyDataset):
    def __init__(self, path, indices, y, batch_size, training=False, seed=42, workers=1):
        super().__init__(workers=workers, use_multiprocessing=False, max_queue_size=4)
        self.path = path
        self.indices = np.asarray(indices, dtype=np.int64)
        self.y = y
        self.batch_size = batch_size
        self.training = training
        self.rng = np.random.default_rng(seed)
        self.order = self.indices.copy()
        if training:
            self.rng.shuffle(self.order)

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def __getitem__(self, batch):
        idx = self.order[batch*self.batch_size:(batch+1)*self.batch_size]
        return read_images(self.path, idx), self.y[idx]

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.order)


def build_model(lr):
    # Same architecture, activations, optimizer family and loss as the source.
    model = keras.Sequential([
        layers.Input(shape=(128, 128, 3)),
        layers.Conv2D(32, (3, 3), activation='leaky_relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, (3, 3), activation='leaky_relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Conv2D(128, (3, 3), activation='leaky_relu', padding='same'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(256, activation='leaky_relu'),
        layers.Dropout(0.3),
        layers.Dense(len(CLASSES), activation='softmax', dtype='float32')
    ], name='ClassCNN')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


class ValidationMetrics(keras.callbacks.Callback):
    """Extra metrics per epoch; one additional inference pass, no optimization."""
    def __init__(self, ds, truth):
        super().__init__()
        self.ds, self.truth = ds, truth

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            return
        pred = self.model.predict(self.ds, verbose=0).argmax(axis=1)
        logs['val_macro_f1'] = float(f1_score(self.truth, pred, labels=IDS, average='macro', zero_division=0))
        logs['val_balanced_accuracy'] = float(balanced_accuracy_score(self.truth, pred))
        logs['val_macro_precision'] = float(precision_score(self.truth, pred, labels=IDS, average='macro', zero_division=0))
        logs['val_macro_recall'] = float(recall_score(self.truth, pred, labels=IDS, average='macro', zero_division=0))
        print(f" — val_macro_f1: {logs['val_macro_f1']:.4f} — val_balanced_accuracy: {logs['val_balanced_accuracy']:.4f}")


def history_plots(history, out, best_epoch):
    epochs = history['epoch'].to_numpy() + 1
    for metric in ['accuracy', 'loss', 'macro_f1', 'balanced_accuracy', 'macro_precision', 'macro_recall']:
        keys = [k for k in (metric, 'val_'+metric) if k in history]
        if not keys:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for k in keys:
            ax.plot(epochs, history[k], label=k.replace('_', ' ').title(), linewidth=2)
        ax.axvline(best_epoch, linestyle='--', color='gray', label='Best checkpoint')
        ax.set(xlabel='Epoch', ylabel=metric.replace('_', ' ').title(), title=metric.replace('_', ' ').title()+' by epoch')
        if metric != 'loss':
            ax.set_ylim(0, 1.04)
        ax.grid(alpha=.2); ax.legend()
        savefig(fig, out/f'{metric}.png')
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(epochs, 100*(history['accuracy']-history['val_accuracy']), label='Train − validation accuracy')
    ax.axhline(0, color='gray', linewidth=1)
    ax.set(xlabel='Epoch', ylabel='Percentage points', title='Accuracy generalization gap')
    ax.grid(alpha=.2); ax.legend()
    savefig(fig, out/'generalization_gap.png')


def distribution_plots(out, train, val, y, wind):
    counts = pd.DataFrame({'class': CLASSES, 'train': np.bincount(y[train], minlength=6),
                           'validation': np.bincount(y[val], minlength=6)})
    counts.to_csv(out/'class_distribution.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(IDS-.18, counts.train, .36, label='Train'); ax.bar(IDS+.18, counts.validation, .36, label='Validation')
    ax.set(xticks=IDS, xticklabels=SHORT, ylabel='Observations', title='Class distribution by split'); ax.legend()
    savefig(fig, out/'class_distribution.png')
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(wind[train], bins=np.linspace(0, 221, 45), alpha=.65, label='Train')
    ax.hist(wind[val], bins=np.linspace(0, 221, 45), alpha=.65, label='Validation')
    ax.set(xlabel='Wind speed (units as stored in source labels)', ylabel='Observations', title='Source wind-label distribution'); ax.legend()
    savefig(fig, out/'wind_label_distribution.png')


def example_grid(out, path, idx, truth, pred, conf, kind, rng):
    if len(idx) == 0:
        return
    positions = rng.choice(len(idx), size=min(16,len(idx)), replace=False)
    # idx and associated vectors represent the selected subset.
    images = read_images(path, idx[positions])
    n = len(positions)
    fig, axes = plt.subplots((n+3)//4, 4, figsize=(12, 3*((n+3)//4)), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for ax, x, p in zip(axes.flat, images, positions):
        ax.imshow(x); ax.set_title(f'True: {SHORT[truth[p]]} | Pred: {SHORT[pred[p]]}\nConfidence: {conf[p]:.1%}', fontsize=9)
    fig.suptitle(kind.replace('_',' ').title())
    savefig(fig, out/f'{kind}.png')


def evaluate(model, ds, indices, y, wind, image_path, out, seed):
    out.mkdir(parents=True, exist_ok=True)
    truth = y[indices]
    probs = np.asarray(model.predict(ds, verbose=1), dtype=np.float64)
    if probs.shape != (len(indices), len(CLASSES)) or not np.isfinite(probs).all():
        raise ValueError('Invalid predictions; check model outputs.')
    row_sums = probs.sum(axis=1, keepdims=True)
    if np.any(probs < 0) or not np.allclose(row_sums, 1.0, rtol=1e-5, atol=1e-6):
        raise ValueError('Expected nonnegative softmax probabilities summing to one.')
    # Remove float32 rounding error before float64 scoring.
    probs = probs / row_sums
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = truth == pred
    matrix = confusion_matrix(truth, pred, labels=IDS)
    report = classification_report(truth, pred, labels=IDS, target_names=CLASSES, zero_division=0, output_dict=True)
    pd.DataFrame(report).T.to_csv(out/'classification_report.csv', index_label='class_or_average')
    pd.DataFrame(matrix, index=CLASSES, columns=CLASSES).to_csv(out/'confusion_matrix.csv', index_label='actual')
    norm = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix,dtype=float), where=matrix.sum(axis=1,keepdims=True)!=0)
    pd.DataFrame(norm, index=CLASSES, columns=CLASSES).to_csv(out/'confusion_matrix_normalized.csv', index_label='actual')
    for name, values, fmt in [('confusion_matrix', matrix, 'd'), ('confusion_matrix_normalized', norm, '.1%')]:
        fig, ax = plt.subplots(figsize=(9,7))
        ConfusionMatrixDisplay(values, display_labels=SHORT).plot(ax=ax, cmap='Blues', values_format=fmt, colorbar=True)
        ax.set_title('Validation '+name.replace('_',' ')); savefig(fig,out/f'{name}.png')
    predictions = pd.DataFrame({'source_index': indices, 'wind_label': wind[indices], 'actual_id':truth, 'actual_class':np.asarray(CLASSES)[truth],
                               'predicted_id':pred, 'predicted_class':np.asarray(CLASSES)[pred], 'confidence':conf, 'correct':correct})
    for i, short in enumerate(SHORT):
        predictions[f'prob_{short}'] = probs[:,i]
    predictions.to_csv(out/'predictions.csv',index=False)
    metrics = {'samples':int(len(truth)), 'correct':int(correct.sum()), 'accuracy':float(accuracy_score(truth,pred)),
               'balanced_accuracy':float(balanced_accuracy_score(truth,pred)),
               'macro_precision':float(precision_score(truth,pred,labels=IDS,average='macro',zero_division=0)),
               'macro_recall':float(recall_score(truth,pred,labels=IDS,average='macro',zero_division=0)),
               'macro_f1':float(f1_score(truth,pred,labels=IDS,average='macro',zero_division=0)),
               'weighted_f1':float(f1_score(truth,pred,labels=IDS,average='weighted',zero_division=0)),
               'quadratic_weighted_kappa':float(cohen_kappa_score(truth,pred,labels=IDS,weights='quadratic')),
               'mcc':float(matthews_corrcoef(truth,pred)), 'log_loss':float(log_loss(truth,probs,labels=IDS)),
               'within_one_category_accuracy':float(np.mean(np.abs(truth-pred)<=1)),
               'mean_absolute_category_error':float(np.mean(np.abs(truth-pred))),
               'evaluation_protocol':'stratified random image validation, not an independent test'}
    # Per-class precision, recall and F1.
    fig, ax = plt.subplots(figsize=(10,5))
    for j, key in enumerate(['precision','recall','f1-score']):
        ax.bar(IDS+(j-1)*.25,[report[c][key] for c in CLASSES],.25,label=key)
    ax.set(xticks=IDS,xticklabels=SHORT,ylim=(0,1.05),ylabel='Score',title='Validation per-class metrics'); ax.legend()
    savefig(fig,out/'per_class_metrics.png')
    # One-vs-rest ROC and precision-recall curves and numerical curve data.
    roc_rows, pr_rows, scores = [], [], []
    fig_r, ax_r = plt.subplots(figsize=(8,6)); fig_p, ax_p = plt.subplots(figsize=(8,6))
    for i,c in enumerate(CLASSES):
        binary = truth == i
        if not binary.any() or binary.all():
            scores.append({'class':c,'roc_auc':None,'average_precision':None}); continue
        fpr,tpr,thresholds = roc_curve(binary,probs[:,i]); ra = float(auc(fpr,tpr))
        precision,recall,pr_thresholds = precision_recall_curve(binary,probs[:,i]); ap = float(average_precision_score(binary,probs[:,i]))
        ax_r.plot(fpr,tpr,label=f'{SHORT[i]} ({ra:.3f})'); ax_p.plot(recall,precision,label=f'{SHORT[i]} ({ap:.3f})')
        scores.append({'class':c,'roc_auc':ra,'average_precision':ap})
        roc_rows.extend({'class':c,'fpr':a,'tpr':b,'threshold':t} for a,b,t in zip(fpr,tpr,thresholds))
        pr_rows.extend({'class':c,'recall':a,'precision':b,'threshold':t} for a,b,t in zip(recall,precision,np.r_[pr_thresholds,np.nan]))
    ax_r.plot([0,1],[0,1],'--',color='gray'); ax_r.set(xlabel='False positive rate',ylabel='True positive rate',title='One-vs-rest ROC (AUC)',xlim=(0,1),ylim=(0,1.03)); ax_r.legend()
    ax_p.set(xlabel='Recall',ylabel='Precision',title='One-vs-rest precision-recall (AP)',xlim=(0,1),ylim=(0,1.03)); ax_p.legend()
    savefig(fig_r,out/'roc_curves.png'); savefig(fig_p,out/'precision_recall_curves.png')
    pd.DataFrame(roc_rows).to_csv(out/'roc_curve_data.csv',index=False); pd.DataFrame(pr_rows).to_csv(out/'pr_curve_data.csv',index=False)
    pd.DataFrame(scores).to_csv(out/'auc_ap_by_class.csv',index=False)
    metrics['macro_roc_auc'] = float(np.mean([s['roc_auc'] for s in scores if s['roc_auc'] is not None])) if roc_rows else None
    metrics['macro_average_precision'] = float(np.mean([s['average_precision'] for s in scores if s['average_precision'] is not None])) if pr_rows else None
    # Top-label calibration, confidence and severity of ordinal mistakes.
    bins = np.linspace(0,1,11); bucket = np.minimum(np.searchsorted(bins,conf,side='right')-1,9)
    calibration=[]; ece=0.
    for b in range(10):
        mask=bucket==b; n=int(mask.sum())
        avg_conf=float(conf[mask].mean()) if n else None
        avg_acc=float(correct[mask].mean()) if n else None
        if n: ece+=n/len(truth)*abs(avg_conf-avg_acc)
        calibration.append({'lower':float(bins[b]),'upper':float(bins[b+1]),'count':n,'mean_confidence':avg_conf,'accuracy':avg_acc})
    metrics['top_label_ece_10_bins']=float(ece)
    metrics['multiclass_brier_score']=float(np.mean(np.sum((probs-np.eye(len(CLASSES))[truth])**2,axis=1)))
    pd.DataFrame(calibration).to_csv(out/'calibration.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,6)); ax.plot([0,1],[0,1],'--',color='gray',label='Perfect calibration')
    nonempty=[r for r in calibration if r['count']]
    ax.plot([r['mean_confidence'] for r in nonempty],[r['accuracy'] for r in nonempty],'o-',label=f'ECE = {ece:.3f}')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean confidence',ylabel='Observed accuracy',title='Top-label reliability diagram'); ax.legend()
    savefig(fig,out/'reliability_diagram.png')
    fig,ax=plt.subplots(figsize=(8,5)); ax.hist(conf[correct],bins=bins,alpha=.7,label='Correct'); ax.hist(conf[~correct],bins=bins,alpha=.7,label='Incorrect')
    ax.set(xlabel='Maximum predicted probability',ylabel='Observations',title='Prediction confidence'); ax.legend(); savefig(fig,out/'confidence_distribution.png')
    error=pred-truth
    fig,ax=plt.subplots(figsize=(8,5)); offsets=np.arange(-5,6); ax.bar(offsets,[(error==d).sum() for d in offsets])
    ax.set(xticks=offsets,xlabel='Predicted class ID − true class ID',ylabel='Observations',title='Underestimation and overestimation'); savefig(fig,out/'category_errors.png')
    rng=np.random.default_rng(seed)
    for kind, mask in [('correct_examples',correct),('incorrect_examples',~correct)]:
        example_grid(out,image_path,indices[mask],truth[mask],pred[mask],conf[mask],kind,rng)
    # Most confident errors are useful to audit possible mislabeled/ambiguous samples.
    predictions.loc[~correct].sort_values('confidence',ascending=False).to_csv(out/'misclassified_samples.csv',index=False)
    # All metric values must be valid JSON, including undefined kappa in edge cases.
    metrics={k:(None if isinstance(v,float) and not np.isfinite(v) else v) for k,v in metrics.items()}
    write_json(out/'metrics.json',metrics)
    print(json.dumps(metrics,indent=2)); return metrics


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/processed/classification'))
    parser.add_argument('--output',type=Path,default=Path('runs/classification/classcnn'))
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--lr',type=float,default=.001)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--skip-invalid-labels',action='store_true',help='Explicitly exclude invalid labels; preserve original row IDs.')
    parser.add_argument('--mixed-precision',action='store_true')
    parser.add_argument('--no-epoch-metrics',action='store_true',help='Skip extra validation prediction pass per epoch; final analysis is still generated.')
    args=parser.parse_args()
    if args.epochs<1 or args.batch_size<1 or args.workers<1 or args.lr<=0: parser.error('Positive epochs, batch size, workers and learning rate required.')
    if args.output.exists() and any(args.output.iterdir()): parser.error('Output is not empty. Choose a new --output to avoid overwriting experiments.')
    keras.utils.set_random_seed(args.seed)
    if keras.backend.backend()=='tensorflow':
        import tensorflow as tf
        gpus=tf.config.list_physical_devices('GPU')
        print('TensorFlow GPUs:',gpus)
        for gpu in gpus: tf.config.experimental.set_memory_growth(gpu,True)
        if args.mixed_precision and not gpus: parser.error('Mixed precision requires a GPU in TensorFlow mode.')
    if args.mixed_precision: keras.mixed_precision.set_global_policy('mixed_float16')
    image_path=args.data/'Cyclone_Images.h5'; label_path=args.data/'Cyclone_Labels h5.npy'
    with h5py.File(image_path,'r') as f:
        shape=f['Images'].shape; dtype=str(f['Images'].dtype)
        if len(shape)!=4 or shape[1:3]!=(128,128) or shape[-1]<3:
            raise ValueError(f'Expected N x 128 x 128 x >=3 images, got {shape}; original CNN input must be preserved.')
    labels=np.load(label_path,allow_pickle=True)
    if len(labels)!=shape[0]: raise ValueError('Image and label counts do not match.')
    wind=np.full(len(labels),np.nan,dtype=np.float64)
    for i,row in enumerate(labels):
        try: wind[i]=float(row[5])
        except (TypeError,ValueError,IndexError,OverflowError,KeyError): pass
    y=classify_wind(wind); invalid=np.flatnonzero(y<0); valid=np.flatnonzero(y>=0)
    if len(invalid) and not args.skip_invalid_labels:
        raise ValueError(f'{len(invalid)} invalid labels (nonfinite, negative, or >221). First source rows: {invalid[:10].tolist()}. Check source units/labels or explicitly use --skip-invalid-labels.')
    counts=np.bincount(y[valid],minlength=6)
    if np.any(counts<2): raise ValueError(f'Each of the six classes needs at least two observations: {counts.tolist()}')
    train,val=train_test_split(valid,test_size=.2,random_state=args.seed,stratify=y[valid])
    if set(y[train])!=set(IDS) or set(y[val])!=set(IDS): raise ValueError('At least one class is missing from a split; collect more data.')
    out=args.output; out.mkdir(parents=True,exist_ok=True)
    write_json(out/'config.json',{'data':str(args.data.resolve()),'image_file':str(image_path.resolve()),'label_file':str(label_path.resolve()),'shape':list(shape),'dtype':dtype,'epochs':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'seed':args.seed,'workers':args.workers,'mixed_precision':args.mixed_precision,'backend':keras.backend.backend(),'keras_version':keras.__version__,'validation_fraction':.2,'split':'stratified random image split','selection_metric':'val_accuracy','invalid_excluded':int(len(invalid)),'normalization':'first 3 channels / 255','wind_units':'source comment says knots; not independently verified'})
    write_json(out/'classes.json',{'class_to_idx':dict(zip(CLASSES,range(6))),'thresholds':['[0,31)','[31,49]','(49,61]','(61,88]','(88,117]','(117,221]']})
    split=pd.DataFrame({'source_index':np.r_[train,val],'split':['train']*len(train)+['validation']*len(val),'class_id':y[np.r_[train,val]],'wind_label':wind[np.r_[train,val]]})
    split.to_csv(out/'split_indices.csv',index=False)
    pd.DataFrame({'source_index':invalid,'wind_label':wind[invalid]}).to_csv(out/'excluded_labels.csv',index=False)
    print('Class counts:',dict(zip(CLASSES,counts.tolist())))
    print(f'Train: {len(train)} | Validation: {len(val)} | Excluded: {len(invalid)}')
    print('WARNING: Random image split may include the same storm in both sets. Validation is not independent test performance.')
    distribution_plots(out,train,val,y,wind)
    train_ds=ImageBatches(image_path,train,y,args.batch_size,True,args.seed,args.workers)
    val_ds=ImageBatches(image_path,val,y,args.batch_size,False,args.seed,args.workers)
    model=build_model(args.lr)
    model.summary()
    with (out/'model_summary.txt').open('w') as f: model.summary(print_fn=lambda line:f.write(line+'\n'))
    callbacks=[]
    if not args.no_epoch_metrics: callbacks.append(ValidationMetrics(val_ds,y[val]))
    callbacks.extend([keras.callbacks.ModelCheckpoint(str(out/'best.keras'),monitor='val_accuracy',mode='max',save_best_only=True),
                      keras.callbacks.CSVLogger(str(out/'history.csv'))])
    model.fit(train_ds,validation_data=val_ds,epochs=args.epochs,callbacks=callbacks,shuffle=False)
    model.save(out/'last.keras')
    history=pd.read_csv(out/'history.csv'); best_epoch=int(history.loc[history.val_accuracy.idxmax(),'epoch'])+1
    write_json(out/'selection.json',{'best_epoch':best_epoch,'best_val_accuracy':float(history.val_accuracy.max()),'last_epoch':int(history.epoch.iloc[-1])+1,'criterion':'maximum validation accuracy; first epoch wins ties'})
    history_plots(history,out,best_epoch)
    print('\nFinal-epoch validation analysis')
    last=evaluate(model,val_ds,val,y,wind,image_path,out/'validation_last',args.seed)
    print('\nBest-checkpoint validation analysis')
    best_model=keras.models.load_model(out/'best.keras',compile=False)
    best=evaluate(best_model,val_ds,val,y,wind,image_path,out/'validation_best',args.seed)
    pd.DataFrame([{'checkpoint':'best',**best},{'checkpoint':'last',**last}]).to_csv(out/'checkpoint_comparison.csv',index=False)
    print('Saved all outputs to',out.resolve())


if __name__=='__main__':
    main()
