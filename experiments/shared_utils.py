"""
experiments/shared_utils.py
---------------------------
Common utilities shared across all experiment stages:
  - Data loading from manifest (video-level split safe)
  - Train-only normalization (no data leakage)
  - Full metrics computation
  - Results saving
"""

import os
import json
import pickle
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, average_precision_score,
    confusion_matrix
)

# ─────────────────────────────────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────────────────────────────────

FEATURE_KEYS = ['f1', 'f2', 'f3', 'f4', 'f5']
FEATURE_NAMES = {
    'f1': 'Lens Distortion',
    'f2': 'Motion Blur',
    'f3': 'Biomechanics',
    'f4': 'Lighting SH',
    'f5': 'Frequency FFT',
}


def load_split_csv(split_csv_path):
    """
    Load a split CSV (train/val/test) that contains columns: npz_path, label.
    Returns list of (npz_path, label) tuples.
    """
    entries = []
    with open(split_csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[0].strip().split(',')
    path_col = header.index('npz_path')
    label_col = header.index('label')
    for line in lines[1:]:
        line = line.strip()
        if not line:
            continue
        parts = line.split(',')
        entries.append((parts[path_col], int(parts[label_col])))
    return entries


def load_features(entries, base_dir, feature_keys=None):
    """
    Load requested feature arrays from .npz files.
    Returns:
        X_dict: dict[feature_key -> np.ndarray shape (N, D)]
        y:      np.ndarray shape (N,)
    """
    if feature_keys is None:
        feature_keys = FEATURE_KEYS

    X_dict = {k: [] for k in feature_keys}
    y_list = []
    missing = 0

    for rel_path, label in entries:
        full_path = os.path.join(base_dir, rel_path.replace('/', os.sep))
        if not os.path.exists(full_path):
            missing += 1
            continue
        data = np.load(full_path)
        ok = True
        for k in feature_keys:
            if k not in data:
                ok = False
                break
        if not ok:
            missing += 1
            continue
        for k in feature_keys:
            X_dict[k].append(data[k].astype(np.float32))
        y_list.append(label)

    if missing > 0:
        print(f"  [WARN] Skipped {missing} missing/incomplete .npz files.")

    for k in feature_keys:
        X_dict[k] = np.stack(X_dict[k], axis=0)   # (N, D)
    y = np.array(y_list, dtype=np.int32)
    return X_dict, y


# ─────────────────────────────────────────────────────────────────────────────
# Normalization (train-only, no leakage)
# ─────────────────────────────────────────────────────────────────────────────

class FeatureScaler:
    """Per-feature z-score scaler. Fit on train, apply to val/test."""

    def __init__(self):
        self.stats = {}   # key -> (mean, std)

    def fit(self, X_dict):
        for k, X in X_dict.items():
            mu = X.mean(axis=0)
            sigma = X.std(axis=0) + 1e-8
            self.stats[k] = (mu, sigma)

    def transform(self, X_dict):
        out = {}
        for k, X in X_dict.items():
            mu, sigma = self.stats[k]
            out[k] = np.clip((X - mu) / sigma, -5.0, 5.0).astype(np.float32)
        return out

    def fit_transform(self, X_dict):
        self.fit(X_dict)
        return self.transform(X_dict)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self.stats, f)

    @classmethod
    def load(cls, path):
        scaler = cls()
        with open(path, 'rb') as f:
            scaler.stats = pickle.load(f)
        return scaler


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics(y_true, y_pred_proba, threshold=0.5):
    """
    Compute the full set of classification metrics.
    Returns a dict with all metrics.
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, zero_division=0)
    recall      = recall_score(y_true, y_pred, zero_division=0)      # sensitivity
    f1          = f1_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    roc_auc     = roc_auc_score(y_true, y_pred_proba)
    pr_auc      = average_precision_score(y_true, y_pred_proba)

    return {
        'accuracy':    float(accuracy),
        'precision':   float(precision),
        'recall':      float(recall),       # = sensitivity
        'sensitivity': float(recall),
        'specificity': float(specificity),
        'f1':          float(f1),
        'roc_auc':     float(roc_auc),
        'pr_auc':      float(pr_auc),
        'tp':          int(tp),
        'tn':          int(tn),
        'fp':          int(fp),
        'fn':          int(fn),
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def print_metrics(name, metrics):
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    print(f"  Accuracy    : {metrics['accuracy']:.4f}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall/Sens : {metrics['sensitivity']:.4f}")
    print(f"  Specificity : {metrics['specificity']:.4f}")
    print(f"  F1 Score    : {metrics['f1']:.4f}")
    print(f"  ROC-AUC     : {metrics['roc_auc']:.4f}")
    print(f"  PR-AUC      : {metrics['pr_auc']:.4f}")
    print(f"  TP={metrics['tp']}  TN={metrics['tn']}  FP={metrics['fp']}  FN={metrics['fn']}")
    print(f"{'='*55}\n")


def save_metrics(metrics, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=4)


# ─────────────────────────────────────────────────────────────────────────────
# MLP Feature Encoder (shared building block)
# ─────────────────────────────────────────────────────────────────────────────

def build_mlp_encoder(input_dim, embed_dim=128, name='mlp_encoder'):
    """
    Single-feature MLP encoder:
        Linear(D → embed_dim) → LayerNorm → GELU → Dropout(0.2)
        → Linear(embed_dim → embed_dim) → LayerNorm → GELU
    Returns a Keras Model with input (input_dim,) and output (embed_dim,).
    """
    inp = tf.keras.Input(shape=(input_dim,), name=f'{name}_input')
    x = tf.keras.layers.Dense(embed_dim, name=f'{name}_fc1')(inp)
    x = tf.keras.layers.LayerNormalization(name=f'{name}_ln1')(x)
    x = tf.keras.layers.Activation('gelu', name=f'{name}_gelu1')(x)
    x = tf.keras.layers.Dropout(0.2, name=f'{name}_drop1')(x)
    x = tf.keras.layers.Dense(embed_dim, name=f'{name}_fc2')(x)
    x = tf.keras.layers.LayerNormalization(name=f'{name}_ln2')(x)
    x = tf.keras.layers.Activation('gelu', name=f'{name}_gelu2')(x)
    return tf.keras.Model(inputs=inp, outputs=x, name=name)


def build_single_feature_classifier(input_dim, embed_dim=128, name='feature_clf'):
    """
    Full single-feature binary classifier:
        MLP Encoder → Projection Head → Sigmoid
    """
    inp = tf.keras.Input(shape=(input_dim,), name='feature_input')
    encoder = build_mlp_encoder(input_dim, embed_dim, name=f'{name}_encoder')
    z = encoder(inp)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(z)
    model = tf.keras.Model(inputs=inp, outputs=out, name=name)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model
