"""
dataset_loader.py
-----------------
tf.data.Dataset pipeline for the FaceForensics++ dataset.

Loads pre-extracted .npz feature files (produced by
pipelines/extract_features.py) and splits them into train / val / test
sets using a deterministic random seed.

Features per sample:
    f1  (64-D)  Lens distortion
    f2  (64-D)  Motion blur
    f3  (64-D)  Biomechanics / rPPG
    f4  (64-D)  Lighting spherical harmonics
    f5  (64-D)  FFT frequency spectrum
"""

import os
import csv
import json
import random

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Normalization
# ---------------------------------------------------------------------------

def normalize_and_clip_features(f1, f2, f3, f4, f5):
    """
    Standardize features channel-by-channel and clip extreme outliers to [-5, +5].
    Reads per-channel mean/std from config/feature_scaling_stats.json.
    """
    with open('config/feature_scaling_stats.json', 'r') as fh:
        stats = json.load(fh)

    def _norm(x, key):
        mean, std = stats[key][0], stats[key][1]
        return tf.clip_by_value((x - mean) / std, -5.0, 5.0)

    return (
        _norm(f1, 'f1'),
        _norm(f2, 'f2'),
        _norm(f3, 'f3'),
        _norm(f4, 'f4'),
        _norm(f5, 'f5'),
    )


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------

def load_manifest(features_dir):
    """
    Load the feature manifest CSV produced by extract_features.py.

    Returns:
        List of (absolute_npz_path, label) tuples.
    """
    manifest_path = os.path.join(features_dir, 'manifest.csv')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(
            f"Manifest not found at {manifest_path}. "
            f"Run  python pipelines/extract_features.py  first."
        )

    entries = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            path_key = 'npz_path' if 'npz_path' in row else 'path'
            abs_path = os.path.join(features_dir, row[path_key])
            label = int(row['label'])
            if os.path.exists(abs_path):
                entries.append((abs_path, label))

    return entries


# ---------------------------------------------------------------------------
# Train / val / test split
# ---------------------------------------------------------------------------

def split_dataset(entries, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """Stratified train / val / test split."""
    paths  = [e[0] for e in entries]
    labels = [e[1] for e in entries]

    val_test_ratio = val_ratio + test_ratio
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        paths, labels, test_size=val_test_ratio, stratify=labels, random_state=seed,
    )

    relative_test = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=relative_test, stratify=valtest_labels, random_state=seed,
    )

    return (
        list(zip(train_paths, train_labels)),
        list(zip(val_paths,   val_labels)),
        list(zip(test_paths,  test_labels)),
    )


# ---------------------------------------------------------------------------
# .npz loader (tf.py_function wrapper)
# ---------------------------------------------------------------------------

def _load_npz(npz_path_bytes, label):
    """Load one .npz file and return the five feature arrays + label."""
    npz_path = npz_path_bytes.numpy().decode('utf-8')
    data = np.load(npz_path)

    f1 = data['f1'].astype(np.float32)
    f2 = data['f2'].astype(np.float32)
    f3 = data['f3'].astype(np.float32)
    f4 = data['f4'].astype(np.float32)
    f5 = data['f5'].astype(np.float32) if 'f5' in data else np.zeros(64, dtype=np.float32)

    label_val = np.float32(label.numpy())
    return f1, f2, f3, f4, f5, label_val


# ---------------------------------------------------------------------------
# tf.data.Dataset builder
# ---------------------------------------------------------------------------

def build_tf_dataset(entries, batch_size=32, shuffle=True,
                     feature_dim=64, features_dir='DataSets/features',
                     normalize=True):
    """
    Build a tf.data.Dataset from a list of (npz_path, label) entries.

    Each element yields:
        inputs = {
            'f1_lens_distortion': (feature_dim,),
            'f2_motion_blur':     (feature_dim,),
            'f3_biomechanics':    (feature_dim,),
            'f4_lighting_sh':     (feature_dim,),
            'f5_frequency_fft':   (feature_dim,),
        }
        label = scalar float32 (0.0 or 1.0)
    """
    paths  = [e[0] for e in entries]
    labels = [e[1] for e in entries]

    path_ds = tf.data.Dataset.from_tensor_slices(
        (paths, np.array(labels, dtype=np.float32))
    )

    if shuffle:
        path_ds = path_ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        f1, f2, f3, f4, f5, lbl = tf.py_function(
            func=_load_npz,
            inp=[path, label],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        )
        f1.set_shape((feature_dim,))
        f2.set_shape((feature_dim,))
        f3.set_shape((feature_dim,))
        f4.set_shape((feature_dim,))
        f5.set_shape((feature_dim,))
        lbl.set_shape(())

        if normalize:
            f1, f2, f3, f4, f5 = normalize_and_clip_features(f1, f2, f3, f4, f5)

        return {
            'f1_lens_distortion': f1,
            'f2_motion_blur':     f2,
            'f3_biomechanics':    f3,
            'f4_lighting_sh':     f4,
            'f5_frequency_fft':   f5,
        }, lbl

    dataset = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if batch_size is not None:
        dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------

def get_strict_datasets(cfg):
    """
    Strict 1:1 balanced test/val sets, dynamically balanced 50/50 training.
    """
    ds_cfg       = cfg['dataset']
    features_dir = ds_cfg['features_dir']
    batch_size   = cfg['training']['batch_size']
    feature_dim  = cfg['extractors']['feature_dim']
    seed         = ds_cfg.get('split_seed', 42)

    entries = load_manifest(features_dir)
    print(f"Loaded manifest: {len(entries)} samples "
          f"({sum(1 for _, l in entries if l == 0)} real, "
          f"{sum(1 for _, l in entries if l == 1)} fake)")

    real_entries = []
    fake_by_method = {'Deepfakes': [], 'Face2Face': [], 'FaceSwap': [], 'NeuralTextures': []}
    other_fakes = []

    for path, label in entries:
        if label == 0:
            real_entries.append((path, label))
        else:
            matched = False
            for method in fake_by_method:
                if f'manipulated_sequences_{method}' in path:
                    fake_by_method[method].append((path, label))
                    matched = True
                    break
            if not matched:
                other_fakes.append((path, label))

    random.seed(seed)
    random.shuffle(real_entries)
    for method in fake_by_method:
        random.shuffle(fake_by_method[method])

    # 1. Select the requested subset for Train + Val
    trainval_real = real_entries[:1000]
    trainval_df   = fake_by_method['Deepfakes'][:750]
    trainval_f2f  = fake_by_method['Face2Face'][:750]

    trainval_entries = trainval_real + trainval_df + trainval_f2f
    random.shuffle(trainval_entries)

    # 2. Split 80/20 for Training and Validation
    split_idx = int(len(trainval_entries) * 0.8)
    train_entries = trainval_entries[:split_idx]
    val_entries   = trainval_entries[split_idx:]

    # Separate train into real/fake for 50-50 tf.data sampling
    train_real = [e for e in train_entries if e[1] == 0]
    train_fake = [e for e in train_entries if e[1] == 1]

    # 3. Test set: All remaining unused videos to prevent data leakage
    used_paths = {e[0] for e in trainval_entries}
    test_entries = [e for e in entries if e[0] not in used_paths]

    print(f"STRICT SPLIT:")
    print(f"  Train: {len(train_real)} Real, {len(train_fake)} Fake")
    print(f"  Val:   {sum(1 for e in val_entries if e[1]==0)} Real,  {sum(1 for e in val_entries if e[1]==1)} Fake")
    print(f"  Test:  {sum(1 for e in test_entries if e[1]==0)} Real,  {sum(1 for e in test_entries if e[1]==1)} Fake")

    train_real_ds = build_tf_dataset(train_real, batch_size=None, shuffle=True,
                                     feature_dim=feature_dim, features_dir=features_dir)
    train_fake_ds = build_tf_dataset(train_fake, batch_size=None, shuffle=True,
                                     feature_dim=feature_dim, features_dir=features_dir)

    train_ds = tf.data.Dataset.sample_from_datasets(
        [train_real_ds, train_fake_ds], weights=[0.5, 0.5], stop_on_empty_dataset=True
    ).batch(batch_size).prefetch(tf.data.AUTOTUNE)

    val_ds  = build_tf_dataset(val_entries,  batch_size, shuffle=False,
                                feature_dim=feature_dim, features_dir=features_dir)
    test_ds = build_tf_dataset(test_entries, batch_size, shuffle=False,
                                feature_dim=feature_dim, features_dir=features_dir)

    split_info = {
        'train': len(train_real) + len(train_fake),
        'val':   len(val_entries),
        'test':  len(test_entries),
    }
    return train_ds, val_ds, test_ds, split_info


def get_random_datasets(cfg):
    """Standard 70/15/15 stratified random split."""
    ds_cfg       = cfg['dataset']
    features_dir = ds_cfg['features_dir']
    batch_size   = cfg['training']['batch_size']
    feature_dim  = cfg['extractors']['feature_dim']

    entries = load_manifest(features_dir)
    print(f"Loaded manifest: {len(entries)} samples "
          f"({sum(1 for _, l in entries if l == 0)} real, "
          f"{sum(1 for _, l in entries if l == 1)} fake)")

    train_entries, val_entries, test_entries = split_dataset(
        entries,
        train_ratio = ds_cfg.get('train_ratio',  0.7),
        val_ratio   = ds_cfg.get('val_ratio',    0.15),
        test_ratio  = ds_cfg.get('test_ratio',   0.15),
        seed        = ds_cfg.get('split_seed',   42),
    )

    print(f"Split: train={len(train_entries)}, val={len(val_entries)}, "
          f"test={len(test_entries)}")

    train_ds = build_tf_dataset(train_entries, batch_size, shuffle=True,
                                feature_dim=feature_dim, features_dir=features_dir)
    val_ds   = build_tf_dataset(val_entries,   batch_size, shuffle=False,
                                feature_dim=feature_dim, features_dir=features_dir)
    test_ds  = build_tf_dataset(test_entries,  batch_size, shuffle=False,
                                feature_dim=feature_dim, features_dir=features_dir)

    split_info = {
        'train': len(train_entries),
        'val':   len(val_entries),
        'test':  len(test_entries),
    }
    return train_ds, val_ds, test_ds, split_info


def get_datasets(cfg):
    """Default: standard 70/15/15 random split."""
    return get_random_datasets(cfg)
