#!/usr/bin/env python3
"""Frozen ResNet50: four-class classification on CycloneAI V3.1 Kelvin tensors.
No regression, auxiliary loss, extra feature branch, or new data splitting.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('KERAS_BACKEND', 'tensorflow')
import keras
from keras import layers
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES = ['TD', 'TS', 'HU', 'MH']
SOURCE_CLASSES = ['TD', 'TS', 'CAT1', 'CAT2', 'CAT3', 'CAT4', 'CAT5']
SOURCE_BOUNDS = [34, 64, 83, 96, 113, 137]
BOUNDS = [34, 64, 96]


def load_dataset(root):
    meta = root / 'metadata'
    man = root / 'manifests'
    norm = json.loads((meta / 'normalization.json').read_text())
    cfg = json.loads((meta / 'preprocessing_config.json').read_text())
    source = json.loads((meta / 'classes.json').read_text())
    storm_split = json.loads((man / 'storm_splits.json').read_text())
    if source != {str(i): c for i, c in enumerate(SOURCE_CLASSES)}:
        raise ValueError('Expected the original seven-class V3.1 dataset')
    if (norm['channels'] != ['C13', 'C08', 'C13-C08'] or
            norm['method'] != 'train_valid_joint_pixels_zscore'):
        raise ValueError('Unexpected normalization metadata')
    mean = np.asarray(norm['mean'], dtype=np.float32)
    std = np.asarray(norm['std'], dtype=np.float32)
    native, crop = int(cfg['master_size']), int(norm['crop_size'])
    if (mean.shape != (3,) or std.shape != (3,) or not np.isfinite(mean).all()
            or not np.isfinite(std).all() or np.any(std <= 0)
            or not 0 < crop <= native or (native - crop) % 2):
        raise ValueError('Invalid normalization or crop metadata')
    frames = {}
    seen_ids, seen_paths = set(), set()
    storms, scenes, hashes = {}, {}, {}
    for split in ('train', 'val', 'test'):
        df = pd.read_csv(man / f'{split}.csv', keep_default_na=False,
                         dtype={'sid': str, 'scene_id': str, 'sample_id': str,
                                'tensor_sha256': str})
        wind = pd.to_numeric(df.wind_kt, errors='raise').to_numpy(float)
        old = np.searchsorted(SOURCE_BOUNDS, wind, side='right')
        if (df.empty or not np.isfinite(wind).all() or np.any((wind <= 0) | (wind > 250))
                or not np.array_equal(df.class_id.to_numpy(), old)
                or not np.array_equal(df.class_name.to_numpy(), np.asarray(SOURCE_CLASSES)[old])):
            raise ValueError(f'Invalid source labels in {split}')
        df['class_id'] = np.searchsorted(BOUNDS, wind, side='right').astype(np.int32)
        for row in df.itertuples(index=False):
            path = (root / row.kelvin).resolve()
            if (not row.sid or not row.scene_id or not row.sample_id or row.split != split
                    or storm_split.get(row.sid) != split
                    or storms.setdefault(row.sid, split) != split
                    or scenes.setdefault(row.scene_id, split) != split
                    or hashes.setdefault(row.tensor_sha256, split) != split
                    or row.sample_id in seen_ids or path in seen_paths
                    or not path.is_relative_to(root) or not path.is_file()):
                raise ValueError(f'Invalid or overlapping sample: {row.sample_id}')
            seen_ids.add(row.sample_id)
            seen_paths.add(path)
        counts = np.bincount(df.class_id, minlength=len(CLASSES))
        if split == 'train' and np.any(counts == 0):
            raise ValueError('Training split is missing a class')
        if split != 'test':
            print(f'{split}: {len(df)} samples, {df.sid.nunique()} storms, '
                  f'{dict(zip(CLASSES, counts.tolist()))}')
            frames[split] = df
    return frames, (native, crop, mean, std)


def resize(array, size):
    return np.asarray(Image.fromarray(np.asarray(array, np.float32)).resize(
        (size, size), Image.Resampling.BILINEAR), dtype=np.float32)


class SatelliteDataset(keras.utils.PyDataset):
    """Load only the current batch; use the fixed training normalization."""
    def __init__(self, root, frame, spec, batch_size=16, training=False, seed=42):
        super().__init__(workers=2, use_multiprocessing=False, max_queue_size=4)
        self.root, self.spec = root, spec
        self.paths = frame.kelvin.to_list()
        self.labels = frame.class_id.to_numpy(np.int32)
        self.batch_size, self.training = batch_size, training
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(len(frame))
        self.on_epoch_end()

    def __len__(self):
        return (len(self.indices) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.training:
            self.rng.shuffle(self.indices)

    def load_image(self, index):
        native, crop, mean, std = self.spec
        with np.load(self.root / self.paths[index], allow_pickle=False) as data:
            ir, wv, m1, m2 = (data[k] for k in ('c13', 'c08', 'mask_c13', 'mask_c08'))
        if any(a.shape != (native, native) for a in (ir, wv, m1, m2)):
            raise ValueError(f'Invalid tensor shape: {self.paths[index]}')
        start = (native - crop) // 2
        sl = slice(start, start + crop)
        ir, wv = ir[sl, sl], wv[sl, sl]
        mask = m1[sl, sl].astype(bool) & m2[sl, sl].astype(bool)
        if not mask.any() or not np.isfinite(ir[mask]).all() or not np.isfinite(wv[mask]).all():
            raise ValueError(f'Invalid temperatures: {self.paths[index]}')
        ir, wv = np.where(mask, ir, 0.), np.where(mask, wv, 0.)
        x = np.stack([ir, wv, ir - wv], axis=-1).astype(np.float32)
        x = np.where(mask[..., None], (x - mean) / std, 0.)
        # Resize the masked numerator and its support separately.
        support = resize(mask, 128)
        x = np.stack([resize(x[..., k], 128) for k in range(3)], axis=-1)
        x = np.where(support[..., None] >= .99,
                     x / np.maximum(support[..., None], 1e-6), 0.)
        # Fixed pseudo-RGB mapping, then correct ImageNet ResNet preprocessing.
        # This is not a natural RGB image or an image divided by 255.
        x = np.clip(127.5 + 32.0 * x, 0, 255).astype(np.float32)
        x = keras.applications.resnet50.preprocess_input(x)
        return np.ascontiguousarray(x, dtype=np.float32)

    def __getitem__(self, batch):
        idx = self.indices[batch*self.batch_size:(batch+1)*self.batch_size]
        x = np.stack([self.load_image(int(i)) for i in idx])
        return x, self.labels[idx]


def build_model(weights='imagenet'):
    base_model = keras.applications.ResNet50(
        input_shape=(224, 224, 3), include_top=False, weights=weights)
    base_model.trainable = False
    inputs = layers.Input(shape=(128, 128, 3), dtype='float32')
    x = layers.Resizing(224, 224)(inputs)
    x = base_model(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', name='head_dense_128')(x)
    outputs = layers.Dense(len(CLASSES), activation='softmax',
                           dtype='float32', name='predictions')(x)
    return keras.Model(inputs, outputs, name='cyclone_resnet50_4class')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/classification_v3_1'))
    parser.add_argument('--output', type=Path, default=Path('runs/classification/resnet50_4class'))
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    if args.batch_size < 1 or args.epochs < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('batch size, epochs and learning rate must be positive')
    keras.utils.set_random_seed(42)
    if keras.backend.backend() == 'tensorflow':
        import tensorflow as tf
        gpus = tf.config.list_physical_devices('GPU')
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print('TensorFlow GPUs:', gpus)
    root = args.data.resolve()
    frames, spec = load_dataset(root)
    train_ds = SatelliteDataset(root, frames['train'], spec, args.batch_size, training=True)
    val_ds = SatelliteDataset(root, frames['val'], spec, args.batch_size)
    args.output.mkdir(parents=True, exist_ok=True)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.lr),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    (args.output / 'classes.json').write_text(json.dumps(dict(enumerate(CLASSES)), indent=2))
    history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                        shuffle=False, callbacks=[keras.callbacks.ModelCheckpoint(
                            str(args.output / 'best.keras'), monitor='val_accuracy',
                            mode='max', save_best_only=True)])
    pd.DataFrame(history.history).to_csv(args.output / 'history.csv', index=False)
    model.save(args.output / 'last.keras')
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(history.history['accuracy'])+1), history.history['accuracy'], label='Train Accuracy')
    plt.plot(range(1, len(history.history['val_accuracy'])+1), history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Accuracy Curve'); plt.legend()
    plt.tight_layout(); plt.savefig(args.output / 'accuracy.png', dpi=160); plt.close()
    print(f'Best validation accuracy: {max(history.history["val_accuracy"]):.2%}')


if __name__ == '__main__':
    main()
