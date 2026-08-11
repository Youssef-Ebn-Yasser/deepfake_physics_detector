"""
experiments/stage_b_simple_fusion.py
--------------------------------------
Stage B — Simple Fusion Baseline

Architecture:
    F1 → MLP Encoder(128) ─┐
    F2 → MLP Encoder(128) ─┤
    F3 → MLP Encoder(128) ─┤→ Concatenate(640) → Dense(256) → GELU → Dropout
    F4 → MLP Encoder(128) ─┤                  → Dense(64)  → GELU
    F5 → MLP Encoder(128) ─┘                  → Dense(1)   → Sigmoid

This is the simple-fusion baseline before the Transformer.
It loads all 5 features, normalises with a single shared FeatureScaler
(per-feature z-score), then trains a concatenation-based MLP.

Ablation sub-experiments run in order:
    F1+F2
    F1+F2+F3
    F1+F2+F3+F4
    F1+F2+F3+F4+F5  ← main

Usage:
    python experiments/stage_b_simple_fusion.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.shared_utils import (
    FEATURE_KEYS, FEATURE_NAMES,
    load_split_csv, load_features,
    FeatureScaler, compute_all_metrics, print_metrics, save_metrics,
    build_mlp_encoder,
)

# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIR     = os.path.join('DataSets', 'deepfakes_feature')
SPLIT_DIR       = FEATURE_DIR
CHECKPOINT_BASE = os.path.join('checkpoints', 'deepfakes', 'stage_b')

EMBED_DIM  = 128
EPOCHS     = 60
BATCH_SIZE = 64
PATIENCE   = 10
SEED       = 42

# Incremental fusion groups (ablation sub-experiments)
FUSION_GROUPS = [
    ['f1', 'f2'],
    ['f1', 'f2', 'f3'],
    ['f1', 'f2', 'f3', 'f4'],
    ['f1', 'f2', 'f3', 'f4', 'f5'],
]
# ─────────────────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_simple_fusion_model(feature_keys, input_dim=64, embed_dim=128):
    """Concatenation-fusion model."""
    inputs = {}
    embeddings = []
    for fk in feature_keys:
        inp = tf.keras.Input(shape=(input_dim,), name=f'input_{fk}')
        inputs[fk] = inp
        encoder = build_mlp_encoder(input_dim, embed_dim, name=f'enc_{fk}')
        embeddings.append(encoder(inp))

    concat = tf.keras.layers.Concatenate(name='concat')(embeddings)

    x = tf.keras.layers.Dense(256, name='fc1')(concat)
    x = tf.keras.layers.LayerNormalization(name='ln1')(x)
    x = tf.keras.layers.Activation('gelu', name='gelu1')(x)
    x = tf.keras.layers.Dropout(0.2, name='drop1')(x)
    x = tf.keras.layers.Dense(64, name='fc2')(x)
    x = tf.keras.layers.LayerNormalization(name='ln2')(x)
    x = tf.keras.layers.Activation('gelu', name='gelu2')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x)

    model = tf.keras.Model(inputs=list(inputs.values()), outputs=out,
                           name=f'SimpleFusion_{"_".join(feature_keys)}')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model, list(inputs.keys())


def train_fusion_group(feature_keys):
    group_name = '+'.join([k.upper() for k in feature_keys])
    print(f"\n{'#'*60}")
    print(f"  Stage B — Simple Fusion: {group_name}")
    print(f"{'#'*60}")

    # ── Load data ─────────────────────────────────────────────────────────────
    train_entries = load_split_csv(os.path.join(SPLIT_DIR, 'split_train.csv'))
    val_entries   = load_split_csv(os.path.join(SPLIT_DIR, 'split_val.csv'))
    test_entries  = load_split_csv(os.path.join(SPLIT_DIR, 'split_test.csv'))

    X_tr_raw, y_tr = load_features(train_entries, FEATURE_DIR, feature_keys)
    X_va_raw, y_va = load_features(val_entries,   FEATURE_DIR, feature_keys)
    X_te_raw, y_te = load_features(test_entries,  FEATURE_DIR, feature_keys)

    # ── Normalize (train-only) ─────────────────────────────────────────────────
    scaler = FeatureScaler()
    X_tr_n = scaler.fit_transform(X_tr_raw)
    X_va_n = scaler.transform(X_va_raw)
    X_te_n = scaler.transform(X_te_raw)

    input_dim = X_tr_n[feature_keys[0]].shape[1]
    print(f"  Features={group_name} | Train={len(y_tr)} | Val={len(y_va)} | Test={len(y_te)}")

    # ── Build model ───────────────────────────────────────────────────────────
    model, input_order = build_simple_fusion_model(feature_keys, input_dim, EMBED_DIM)

    # ── Checkpoint & callbacks ─────────────────────────────────────────────────
    safe_name = group_name.replace('+', '_')
    out_dir   = os.path.join(CHECKPOINT_BASE, safe_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best_model.keras')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss', save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
        ),
    ]

    # Keras expects inputs in the same order as model inputs
    train_X_list = [X_tr_n[k] for k in input_order]
    val_X_list   = [X_va_n[k] for k in input_order]
    test_X_list  = [X_te_n[k] for k in input_order]

    history = model.fit(
        train_X_list, y_tr,
        validation_data=(val_X_list, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred_proba = model.predict(test_X_list, verbose=0).flatten()
    metrics = compute_all_metrics(y_te, y_pred_proba)
    metrics['features']    = feature_keys
    metrics['group_name']  = group_name
    metrics['train_samples'] = int(len(y_tr))
    metrics['val_samples']   = int(len(y_va))
    metrics['test_samples']  = int(len(y_te))
    metrics['best_val_loss'] = float(min(history.history['val_loss']))

    print_metrics(f"Stage B — {group_name} TEST RESULTS", metrics)

    # ── Save ───────────────────────────────────────────────────────────────────
    scaler.save(os.path.join(out_dir, 'scaler.pkl'))
    save_metrics(metrics, os.path.join(out_dir, 'metrics.json'))
    print(f"  Saved → {out_dir}/")
    return metrics


def main():
    for split in ['split_train.csv', 'split_val.csv', 'split_test.csv']:
        if not os.path.exists(os.path.join(SPLIT_DIR, split)):
            print(f"ERROR: {split} not found. Run prepare_deepfakes_split.py first.")
            sys.exit(1)

    all_metrics = {}
    for group in FUSION_GROUPS:
        gname = '+'.join([k.upper() for k in group])
        m = train_fusion_group(group)
        all_metrics[gname] = m

    # ── Summary table ──────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  STAGE B — SIMPLE FUSION SUMMARY")
    print("="*65)
    print(f"  {'Features':<24} {'Acc':>7} {'F1':>7} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print(f"  {'-'*24} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for gname, m in all_metrics.items():
        print(f"  {gname:<24} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
              f"{m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f}")
    print("="*65)

    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    with open(os.path.join(CHECKPOINT_BASE, 'summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == '__main__':
    main()
