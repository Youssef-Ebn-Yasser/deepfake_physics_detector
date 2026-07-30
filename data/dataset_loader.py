"""
dataset_loader.py
-----------------
tf.data.Dataset pipeline for the FaceForensics++ dataset.

Loads pre-extracted .npz feature files (produced by
pipelines/extract_features.py) and splits them into train / val / test
sets using a deterministic random seed.
"""

import os
import csv
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


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
            abs_path = os.path.join(features_dir, row['npz_path'])
            label = int(row['label'])
            if os.path.exists(abs_path):
                entries.append((abs_path, label))

    return entries


def split_dataset(entries, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
                  seed=42):
    """
    Stratified train / val / test split.

    Returns:
        (train_entries, val_entries, test_entries)
    """
    paths = [e[0] for e in entries]
    labels = [e[1] for e in entries]

    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    train_paths, valtest_paths, train_labels, valtest_labels = train_test_split(
        paths, labels,
        test_size=val_test_ratio,
        stratify=labels,
        random_state=seed,
    )

    # Second split: val vs test
    relative_test = test_ratio / val_test_ratio
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        valtest_paths, valtest_labels,
        test_size=relative_test,
        stratify=valtest_labels,
        random_state=seed,
    )

    train_entries = list(zip(train_paths, train_labels))
    val_entries = list(zip(val_paths, val_labels))
    test_entries = list(zip(test_paths, test_labels))

    return train_entries, val_entries, test_entries


def _load_npz(npz_path_bytes, label):
    """
    tf.py_function wrapper to load a single .npz file.

    Returns:
        (f1, f2, f3, f4), label
    """
    npz_path = npz_path_bytes.numpy().decode('utf-8')
    data = np.load(npz_path)
    f1 = data['f1'].astype(np.float32)
    f2 = data['f2'].astype(np.float32)
    f3 = data['f3'].astype(np.float32)
    f4 = data['f4'].astype(np.float32)
    label_val = np.float32(label.numpy())
    return f1, f2, f3, f4, label_val


def build_tf_dataset(entries, batch_size=32, shuffle=True, feature_dim=64):
    """
    Build a tf.data.Dataset from a list of (npz_path, label) entries.

    Each element yields:
        inputs = {
            'f1_lens_distortion': (feature_dim,),
            'f2_motion_blur':     (feature_dim,),
            'f3_biomechanics':    (feature_dim,),
            'f4_lighting_sh':     (feature_dim,),
        }
        label = scalar float32 (0.0 or 1.0)
    """
    paths = [e[0] for e in entries]
    labels = [e[1] for e in entries]

    path_ds = tf.data.Dataset.from_tensor_slices(
        (paths, np.array(labels, dtype=np.float32))
    )

    if shuffle:
        path_ds = path_ds.shuffle(buffer_size=len(paths), reshuffle_each_iteration=True)

    def _map_fn(path, label):
        f1, f2, f3, f4, lbl = tf.py_function(
            func=_load_npz,
            inp=[path, label],
            Tout=[tf.float32, tf.float32, tf.float32, tf.float32, tf.float32],
        )
        # Set shapes so Keras knows the dimensions
        f1.set_shape((feature_dim,))
        f2.set_shape((feature_dim,))
        f3.set_shape((feature_dim,))
        f4.set_shape((feature_dim,))
        lbl.set_shape(())

        inputs = {
            'f1_lens_distortion': f1,
            'f2_motion_blur': f2,
            'f3_biomechanics': f3,
            'f4_lighting_sh': f4,
        }
        return inputs, lbl

    dataset = path_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def get_datasets(cfg):
    """
    High-level convenience function: load manifest, split, build datasets.

    Args:
        cfg: Parsed YAML config dict.

    Returns:
        (train_ds, val_ds, test_ds, split_info)
        where split_info is a dict with entry counts.
    """
    ds_cfg = cfg['dataset']
    features_dir = ds_cfg['features_dir']
    batch_size = cfg['training']['batch_size']
    feature_dim = cfg['extractors']['feature_dim']

    entries = load_manifest(features_dir)
    print(f"Loaded manifest: {len(entries)} samples "
          f"({sum(1 for _, l in entries if l == 0)} real, "
          f"{sum(1 for _, l in entries if l == 1)} fake)")

    train_entries, val_entries, test_entries = split_dataset(
        entries,
        train_ratio=ds_cfg.get('train_ratio', 0.7),
        val_ratio=ds_cfg.get('val_ratio', 0.15),
        test_ratio=ds_cfg.get('test_ratio', 0.15),
        seed=ds_cfg.get('split_seed', 42),
    )

    print(f"Split: train={len(train_entries)}, val={len(val_entries)}, "
          f"test={len(test_entries)}")

    train_ds = build_tf_dataset(train_entries, batch_size, shuffle=True,
                                feature_dim=feature_dim)
    val_ds = build_tf_dataset(val_entries, batch_size, shuffle=False,
                              feature_dim=feature_dim)
    test_ds = build_tf_dataset(test_entries, batch_size, shuffle=False,
                               feature_dim=feature_dim)

    split_info = {
        'train': len(train_entries),
        'val': len(val_entries),
        'test': len(test_entries),
    }
    return train_ds, val_ds, test_ds, split_info
