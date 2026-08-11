"""
experiments/stage_a_individual.py
----------------------------------
Stage A — Individual Feature Models (Ablation Study)

Trains one MLP classifier per feature (F1–F5) independently.
For each feature:
  1. Load train/val/test splits
  2. Fit scaler on TRAIN only, apply to val/test (no leakage)
  3. Train MLP with early stopping & model checkpoint
  4. Evaluate on test set with full metrics
  5. Save: best_model.keras, scaler.pkl, metrics.json

Architecture per feature:
    Input(D) → Dense(128) → LayerNorm → GELU → Dropout(0.2)
             → Dense(128) → LayerNorm → GELU → Dense(1) → Sigmoid

Usage:
    python experiments/stage_a_individual.py
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
    build_single_feature_classifier,
)

# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIR   = os.path.join('DataSets', 'deepfakes_feature')
SPLIT_DIR     = FEATURE_DIR
CHECKPOINT_BASE = os.path.join('checkpoints', 'deepfakes', 'stage_a')

EMBED_DIM   = 128
EPOCHS      = 50
BATCH_SIZE  = 64
PATIENCE    = 8       # early stopping patience
SEED        = 42
# ─────────────────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


def train_single_feature(feature_key: str):
    fname = FEATURE_NAMES[feature_key]
    print(f"\n{'#'*60}")
    print(f"  Stage A — {fname} ({feature_key})")
    print(f"{'#'*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    train_entries = load_split_csv(os.path.join(SPLIT_DIR, 'split_train.csv'))
    val_entries   = load_split_csv(os.path.join(SPLIT_DIR, 'split_val.csv'))
    test_entries  = load_split_csv(os.path.join(SPLIT_DIR, 'split_test.csv'))

    print("Loading features...")
    X_train_raw, y_train = load_features(train_entries, FEATURE_DIR, [feature_key])
    X_val_raw,   y_val   = load_features(val_entries,   FEATURE_DIR, [feature_key])
    X_test_raw,  y_test  = load_features(test_entries,  FEATURE_DIR, [feature_key])

    # ── Normalize (train-only fit) ────────────────────────────────────────────
    scaler = FeatureScaler()
    X_train_n = scaler.fit_transform(X_train_raw)
    X_val_n   = scaler.transform(X_val_raw)
    X_test_n  = scaler.transform(X_test_raw)

    X_train = X_train_n[feature_key]
    X_val   = X_val_n[feature_key]
    X_test  = X_test_n[feature_key]

    input_dim = X_train.shape[1]
    print(f"  Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)} | Feature dim={input_dim}")

    # ── Build model ──────────────────────────────────────────────────────────
    model = build_single_feature_classifier(input_dim, EMBED_DIM, name=f'clf_{feature_key}')

    # ── Checkpoint & callbacks ───────────────────────────────────────────────
    out_dir = os.path.join(CHECKPOINT_BASE, feature_key)
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
            monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=0
        ),
    ]

    # ── Train ────────────────────────────────────────────────────────────────
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate on test set ─────────────────────────────────────────────────
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    metrics = compute_all_metrics(y_test, y_pred_proba)
    metrics['feature']  = fname
    metrics['feature_key'] = feature_key
    metrics['train_samples'] = int(len(X_train))
    metrics['val_samples']   = int(len(X_val))
    metrics['test_samples']  = int(len(X_test))
    metrics['best_val_loss'] = float(min(history.history['val_loss']))

    print_metrics(f"Stage A — {fname} ({feature_key}) TEST RESULTS", metrics)

    # ── Save artifacts ────────────────────────────────────────────────────────
    scaler.save(os.path.join(out_dir, 'scaler.pkl'))
    save_metrics(metrics, os.path.join(out_dir, 'metrics.json'))
    print(f"  Saved → {out_dir}/")

    return metrics


def main():
    # Check splits exist
    for split in ['split_train.csv', 'split_val.csv', 'split_test.csv']:
        path = os.path.join(SPLIT_DIR, split)
        if not os.path.exists(path):
            print(f"ERROR: {path} not found. Run prepare_deepfakes_split.py first.")
            sys.exit(1)

    all_metrics = {}
    for fk in FEATURE_KEYS:
        m = train_single_feature(fk)
        all_metrics[fk] = m

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  STAGE A — ABLATION STUDY SUMMARY")
    print("="*65)
    print(f"  {'Feature':<22} {'Acc':>7} {'F1':>7} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for fk, m in all_metrics.items():
        print(f"  {FEATURE_NAMES[fk]:<22} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
              f"{m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f}")
    print("="*65)

    # Save combined summary
    summary_path = os.path.join(CHECKPOINT_BASE, 'summary.json')
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(all_metrics, f, indent=4)
    print(f"\nCombined summary saved to {summary_path}")


if __name__ == '__main__':
    main()
