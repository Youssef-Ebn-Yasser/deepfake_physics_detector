"""
experiments/eval_stage_e_cross_dataset.py
-------------------------------------------
Cross-Dataset Zero-Shot Evaluation for Stage E Models.

Evaluates EVERY Stage E model on:
  - DeeperForensics TEST (internal baseline)
  - Celeb-DF v2        (balanced: equal real / fake samples)
  - FaceForensics++    (balanced: equal real / fake samples)

Stage E models:
  checkpoints/deepfakes/stage_e/stage_e_f1f4/
  checkpoints/deepfakes/stage_e/stage_e_f1f5/
  checkpoints/deepfakes/stage_e/stage_e_f1f4f5/

Each model uses its OWN scaler.pkl (fitted on DeeperForensics TRAIN).
External data is balanced by undersampling the majority class.
No retraining. No refitting the scaler.

Usage:
    cd h:\\A-Windows\\deepfake_physics_detector
    python experiments/eval_stage_e_cross_dataset.py

Output:
    checkpoints/deepfakes/cross_dataset/stage_e_results.json
    (printed summary table in terminal)
"""

import os
import sys
import json
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from experiments.shared_utils import (
    FEATURE_KEYS, FEATURE_NAMES,
    FeatureScaler,
    load_split_csv, load_features,
    compute_all_metrics,
)
from experiments.stage_e_anchored_fusion import AnchoredFusionModel, EMBED_DIM

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

CKPT_E       = os.path.join('checkpoints', 'deepfakes', 'stage_e')
FEATURE_DIR  = os.path.join('DataSets', 'deepfakes_feature')
SPLIT_DIR    = FEATURE_DIR
CROSS_BASE   = os.path.join('DataSets', 'cross_dataset_features')
OUT_DIR      = os.path.join('checkpoints', 'deepfakes', 'cross_dataset')

SEED         = 42
np.random.seed(SEED)

SEP  = '=' * 72
SEP2 = '-' * 72

# ─────────────────────────────────────────────────────────────────────────────
# Stage E config
# ─────────────────────────────────────────────────────────────────────────────

STAGE_E_CONFIGS = [
    {
        'name':         'F1+F4',
        'short':        'StageE-F1F4',
        'dir':          os.path.join(CKPT_E, 'stage_e_f1f4'),
        'feature_keys': ['f1', 'f4'],
    },
    {
        'name':         'F1+F5',
        'short':        'StageE-F1F5',
        'dir':          os.path.join(CKPT_E, 'stage_e_f1f5'),
        'feature_keys': ['f1', 'f5'],
    },
    {
        'name':         'F1+F4+F5',
        'short':        'StageE-F1F4F5',
        'dir':          os.path.join(CKPT_E, 'stage_e_f1f4f5'),
        'feature_keys': ['f1', 'f4', 'f5'],
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_cross_dataset_npz(dataset_dir, feature_keys):
    manifest_path = os.path.join(dataset_dir, 'manifest.csv')
    entries = []

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        header    = lines[0].strip().split(',')
        path_col  = header.index('npz_path') if 'npz_path' in header else 0
        label_col = header.index('label')    if 'label'    in header else 1
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts    = line.split(',')
            npz_path = os.path.join(dataset_dir, parts[path_col].replace('/', os.sep))
            label    = int(parts[label_col])
            entries.append((npz_path, label))
    else:
        for lname, lval in [('real', 0), ('fake', 1)]:
            subdir = os.path.join(dataset_dir, lname)
            if os.path.isdir(subdir):
                for fname in sorted(os.listdir(subdir)):
                    if fname.endswith('.npz'):
                        entries.append((os.path.join(subdir, fname), lval))

    if not entries:
        return None, None

    X_dict  = {k: [] for k in feature_keys}
    y_list  = []
    missing = 0

    for npz_path, label in entries:
        if not os.path.exists(npz_path):
            missing += 1
            continue
        try:
            data = np.load(npz_path)
        except Exception:
            missing += 1
            continue
        if not all(k in data for k in feature_keys):
            missing += 1
            continue
        for k in feature_keys:
            X_dict[k].append(data[k].astype(np.float32))
        y_list.append(label)

    if missing > 0:
        print(f"    [WARN] Skipped {missing} missing/incomplete .npz files.")

    if not y_list:
        return None, None

    for k in feature_keys:
        X_dict[k] = np.stack(X_dict[k], axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X_dict, y


def balance_dataset(X_dict, y, seed=SEED):
    rng        = np.random.default_rng(seed)
    real_idx   = np.where(y == 0)[0]
    fake_idx   = np.where(y == 1)[0]
    n_min      = min(len(real_idx), len(fake_idx))

    real_sel   = rng.choice(real_idx, n_min, replace=False)
    fake_sel   = rng.choice(fake_idx, n_min, replace=False)
    sel        = np.sort(np.concatenate([real_sel, fake_sel]))

    X_balanced = {k: v[sel] for k, v in X_dict.items()}
    y_balanced = y[sel]
    return X_balanced, y_balanced


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print one result row
# ─────────────────────────────────────────────────────────────────────────────

def fmt_row(model_name, ds_name, n, metrics):
    return (
        f"  {model_name:<22}  {ds_name:<20}  {n:>6}  "
        f"{metrics['roc_auc']:>8.4f}  {metrics['accuracy']:>8.4f}  "
        f"{metrics['f1']:>7.4f}  {metrics['specificity']:>8.4f}  "
        f"{metrics['sensitivity']:>8.4f}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load internal DeeperForensics TEST ────────────────────────────────────
    print(f"\n{SEP}")
    print("  Loading DeeperForensics TEST split (internal baseline)")
    print(SEP)

    test_csv = os.path.join(SPLIT_DIR, 'split_test.csv')
    deeper_data = {}
    if os.path.exists(test_csv):
        test_entries = load_split_csv(test_csv)
        X_deep_all, y_deep = load_features(test_entries, FEATURE_DIR, FEATURE_KEYS)
        print(f"  DeeperForensics test: {len(y_deep)} samples "
              f"(real={(y_deep==0).sum()}, fake={(y_deep==1).sum()})")
        deeper_data = {'X': X_deep_all, 'y': y_deep}
    else:
        print(f"  [WARN] {test_csv} not found — skipping DeeperForensics baseline.")

    # ── Load external datasets ────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  Loading cross-dataset .npz files (all features)")
    print(SEP)

    EXT_DS = {}
    for ds_name, subdir in [
        ('Celeb-DF v2',    os.path.join(CROSS_BASE, 'celeb_df')),
        ('FaceForensics++', os.path.join(CROSS_BASE, 'ff_plus_plus')),
    ]:
        if not os.path.isdir(subdir):
            print(f"  [SKIP] {ds_name}: {subdir} not found.")
            continue
        X_raw, y = load_cross_dataset_npz(subdir, FEATURE_KEYS)
        if X_raw is None:
            continue
        X_bal, y_bal = balance_dataset(X_raw, y)
        EXT_DS[ds_name] = {'X_raw': X_raw, 'y_raw': y,
                           'X_bal': X_bal, 'y_bal': y_bal}
        print(f"  {ds_name}: balanced to {len(y_bal)} samples")

    all_results = {}
    summary_rows = []

    # ═════════════════════════════════════════════════════════════════════════
    # Evaluate Stage E Models
    # ═════════════════════════════════════════════════════════════════════════
    print(f"\n\n{SEP}")
    print("  STAGE E — ANCHORED FUSION MODELS")
    print(SEP)

    for cfg in STAGE_E_CONFIGS:
        model_dir = cfg['dir']
        fkeys     = cfg['feature_keys']
        name      = cfg['name']
        short     = cfg['short']

        model_path  = os.path.join(model_dir, 'best_model.weights.h5')
        scaler_path = os.path.join(model_dir, 'scaler.pkl')

        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            print(f"\n  [SKIP] {name}: model or scaler not found in {model_dir}")
            continue

        print(f"\n  Loading {name}")
        
        # We need to initialize the model with the same input_dim
        # Let's peek at DeeperForensics data to get input_dim
        input_dim = 64
        model = AnchoredFusionModel(fkeys, input_dim, EMBED_DIM)
        # Dummy call to build model
        dummy_input = {k: tf.zeros((1, input_dim)) for k in fkeys}
        model(dummy_input)
        model.load_weights(model_path)
        
        scaler = FeatureScaler.load(scaler_path)

        model_results = {'stage': 'E', 'name': name, 'features': fkeys, 'datasets': {}}

        # ── DeeperForensics baseline ─────────────────────────────────────────
        if deeper_data:
            X_sub = {k: deeper_data['X'][k] for k in fkeys}
            y_sub = deeper_data['y']
            X_n   = scaler.transform(X_sub)
            test_X_dict = {k: X_n[k] for k in fkeys}
            outputs = model(test_X_dict, training=False)
            yp = tf.sigmoid(outputs['main_logit']).numpy().flatten()
            m     = compute_all_metrics(y_sub, yp)
            model_results['datasets']['DeeperForensics_test'] = m
            summary_rows.append(fmt_row(short, 'DeeperForensics', len(y_sub), m))
            print(f"    DeeperForensics  ROC-AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"Spec={m['specificity']:.4f}  Sens={m['sensitivity']:.4f}")

        # ── External datasets (balanced) ──────────────────────────────────────
        for ds_name, ds in EXT_DS.items():
            X_sub  = {k: ds['X_bal'][k] for k in fkeys}
            y_sub  = ds['y_bal']
            X_n    = scaler.transform(X_sub)
            test_X_dict = {k: X_n[k] for k in fkeys}
            outputs = model(test_X_dict, training=False)
            yp = tf.sigmoid(outputs['main_logit']).numpy().flatten()
            m      = compute_all_metrics(y_sub, yp)
            safe   = ds_name.replace(' ', '_').replace('+', '_')
            model_results['datasets'][safe] = m
            summary_rows.append(fmt_row(short, ds_name, len(y_sub), m))
            print(f"    {ds_name:<22} ROC-AUC={m['roc_auc']:.4f}  Acc={m['accuracy']:.4f}  "
                  f"Spec={m['specificity']:.4f}  Sens={m['sensitivity']:.4f}")

        all_results[short] = model_results

    # ─────────────────────────────────────────────────────────────────────────
    # Final summary table
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("  FINAL CROSS-DATASET SUMMARY TABLE")
    print(SEP)
    header = (
        f"  {'Model':<22}  {'Dataset':<20}  {'n':>6}  "
        f"{'ROC-AUC':>8}  {'Accuracy':>8}  {'F1':>7}  {'Specif.':>8}  {'Sensit.':>8}"
    )
    print(header)
    print(f"  {'-'*22}  {'-'*20}  {'-'*6}  {'-'*8}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*8}")

    last_model = None
    for row in summary_rows:
        model_name = row.split('  ')[1].strip()
        if model_name != last_model and last_model is not None:
            print()
        last_model = model_name
        print(row)

    # ── Per-dataset cross-model AUC comparison ────────────────────────────────
    print(f"\n\n{SEP}")
    print("  ROC-AUC QUICK COMPARISON  (grouped by dataset)")
    print(SEP)

    all_ds_keys = ['DeeperForensics_test']
    for ds_name in EXT_DS:
        all_ds_keys.append(ds_name.replace(' ', '_').replace('+', '_'))

    col_w = 16
    ds_headers = [k[:col_w].ljust(col_w) for k in all_ds_keys]
    print(f"  {'Model':<22}  " + "  ".join(ds_headers))
    print(f"  {'-'*22}  " + "  ".join(['-'*col_w for _ in all_ds_keys]))

    for short, r in all_results.items():
        vals = []
        for dsk in all_ds_keys:
            m = r['datasets'].get(dsk)
            vals.append(f"{m['roc_auc']:.4f}".ljust(col_w) if m else 'N/A'.ljust(col_w))
        print(f"  {short:<22}  " + "  ".join(vals))

    out_path = os.path.join(OUT_DIR, 'stage_e_cross_dataset.json')
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"\n  Full results saved to: {out_path}")

if __name__ == '__main__':
    main()
