import os
import sys
import argparse
import glob
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import tensorflow as tf
from data.dataset_loader import load_manifest, split_dataset
from subsystem3.encoder3 import SubSystem3Encoder

def _load_npy(npy_path_bytes, label):
    npy_path = npy_path_bytes.numpy().decode('utf-8')
    data = np.load(npy_path).astype(np.float32)
    label_val = np.float32(label.numpy())
    return data, label_val

def build_tf_dataset3(entries, features_dir, batch_size=32, shuffle=True, feature_dim=64):
    # Mapping npz_path to subsystem3 npy path
    paths = []
    labels = []
    
    for npz_rel, label in entries:
        rel_path = os.path.relpath(npz_rel, features_dir)
        parts = rel_path.replace('\\', '/').split('/')
        category = parts[0]
        stem = os.path.splitext(parts[1])[0]
        
        npy_path = os.path.join(features_dir, 'subsystem3', category, f'{stem}_physics64.npy')
        paths.append(npy_path)
        labels.append(label)
        
    path_ds = tf.data.Dataset.from_tensor_slices((paths, np.array(labels, dtype=np.float32)))
    if shuffle:
        path_ds = path_ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)
        
    def _map_fn(path, label):
        f, lbl = tf.py_function(func=_load_npy, inp=[path, label], Tout=[tf.float32, tf.float32])
        f.set_shape((feature_dim,))
        lbl.set_shape(())
        return f, lbl

    dataset = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/default_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    ds_cfg = cfg['dataset']
    features_dir = ds_cfg['features_dir']
    feature_dim = cfg['extractors']['feature_dim']
    embed_dim = cfg['model']['embed_dim']
    epochs = cfg['training']['subsystem_epochs']
    lr = cfg['training']['subsystem_lr']
    batch_size = cfg['training']['batch_size']

    entries = load_manifest(features_dir)
    train_entries, val_entries, _ = split_dataset(entries,
        train_ratio=ds_cfg.get('train_ratio', 0.7),
        val_ratio=ds_cfg.get('val_ratio', 0.15),
        test_ratio=ds_cfg.get('test_ratio', 0.15),
        seed=ds_cfg.get('split_seed', 42),
    )

    train_ds = build_tf_dataset3(train_entries, features_dir, batch_size, True, feature_dim)
    val_ds = build_tf_dataset3(val_entries, features_dir, batch_size, False, feature_dim)

    input_f = tf.keras.layers.Input(shape=(feature_dim,), name='physics64')
    encoder = SubSystem3Encoder(feature_dim, embed_dim, name='subsystem3_physics')
    z3, sub3_logits = encoder(input_f)
    model = tf.keras.Model(inputs=input_f, outputs={'sub3_logits': sub3_logits}, name='SubSystem3_PAC')

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'sub3_logits': tf.keras.losses.BinaryCrossentropy(from_logits=True)},
        metrics={'sub3_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]}
    )

    os.makedirs('checkpoints', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/subsystem3_best.keras',
            monitor='val_sub3_logits_acc',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sub3_logits_acc',
            patience=5,
            mode='max',
            restore_best_weights=True,
            verbose=1
        )
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

if __name__ == '__main__':
    main()
