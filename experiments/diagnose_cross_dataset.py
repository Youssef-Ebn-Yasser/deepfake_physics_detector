"""
experiments/diagnose_cross_dataset.py
--------------------------------------
Comprehensive diagnostic for cross-dataset normalization / pipeline issues.

Runs the following checks for each dataset (DeeperForensics test, Celeb-DF, FF++):

  1. RAW feature statistics (before scaler)
  2. SCALED feature statistics (after DeeperForensics scaler)
  3. Scaler internals inspection (mean / std per feature)
  4. Feature shapes
  5. Prediction distribution (before & after scaler)
  6. ROC-AUC before scaler vs after scaler
  7. Input order verification (explicit, not inferred)

Usage:
    cd h:\\A-Windows\\deepfake_physics_detector
    python experiments/diagnose_cross_dataset.py
"""

import os
import sys
import json
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── path setup ────────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
tf.get_logger().setLevel('ERROR')

from experiments.stage_d_full_model import (
    TransformerEncoderBlock,
    CrossAttentionBlock,
    GatedFusion,
)
from experiments.shared_utils import (
    FEATURE_KEYS,
    FeatureScaler,
    load_split_csv,
    load_features,
    compute_all_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

STAGE_D_DIR    = os.path.join('checkpoints', 'deepfakes', 'stage_d')
SCALER_PATH    = os.path.join(STAGE_D_DIR, 'scaler.pkl')
MODEL_PATH     = os.path.join(STAGE_D_DIR, 'best_model.keras')

FEATURE_DIR    = os.path.join('DataSets', 'deepfakes_feature')
SPLIT_DIR      = FEATURE_DIR

CROSS_BASE     = os.path.join('DataSets', 'cross_dataset_features')

DATASETS_EXT   = {
    'Celeb-DF v2':     os.path.join(CROSS_BASE, 'celeb_df'),
    'FaceForensics++': os.path.join(CROSS_BASE, 'ff_plus_plus'),
}

SEP  = '=' * 68
SEP2 = '-' * 68

# ─────────────────────────────────────────────────────────────────────────────
# Helper: print feature stats table
# ─────────────────────────────────────────────────────────────────────────────

def print_feature_stats(X_dict, tag):
    print(f"\n  [{tag}] Per-feature statistics:")
    print(f"  {'key':<4}  {'shape':<12}  {'mean':>9}  {'std':>9}  {'min':>9}  {'max':>9}")
    print(f"  {'-'*4}  {'-'*12}  {'-'*9}  {'-'*9}  {'-'*9}  {'-'*9}")
    for k in FEATURE_KEYS:
        if k not in X_dict:
            print(f"  {k:<4}  MISSING")
            continue
        x = X_dict[k]
        print(
            f"  {k:<4}  {str(x.shape):<12}  "
            f"{x.mean():>9.4f}  {x.std():>9.4f}  "
            f"{x.min():>9.4f}  {x.max():>9.4f}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helper: print prediction distribution
# ─────────────────────────────────────────────────────────────────────────────

def print_pred_stats(y_pred_proba, y_true, tag):
    print(f"\n  [{tag}] Prediction probability distribution:")
    print(f"    Overall  -- min={y_pred_proba.min():.4f}  max={y_pred_proba.max():.4f}  "
          f"mean={y_pred_proba.mean():.4f}  std={y_pred_proba.std():.4f}")

    real_proba = y_pred_proba[y_true == 0]
    fake_proba = y_pred_proba[y_true == 1]

    if len(real_proba) > 0:
        print(f"    REAL     -- min={real_proba.min():.4f}  max={real_proba.max():.4f}  "
              f"mean={real_proba.mean():.4f}  std={real_proba.std():.4f}  "
              f"n={len(real_proba)}")
        print(f"    REAL first 10: {real_proba[:10].round(4)}")

    if len(fake_proba) > 0:
        print(f"    FAKE     -- min={fake_proba.min():.4f}  max={fake_proba.max():.4f}  "
              f"mean={fake_proba.mean():.4f}  std={fake_proba.std():.4f}  "
              f"n={len(fake_proba)}")
        print(f"    FAKE first 10: {fake_proba[:10].round(4)}")

    # Threshold sweep
    print(f"\n  [{tag}] Threshold sweep:")
    print(f"    {'Threshold':>10}  {'Acc':>7}  {'TPR':>7}  {'TNR':>7}  {'FPR':>7}")
    for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        y_pred = (y_pred_proba >= thr).astype(int)
        acc = (y_pred == y_true).mean()
        tp  = ((y_pred == 1) & (y_true == 1)).sum()
        fn  = ((y_pred == 0) & (y_true == 1)).sum()
        tn  = ((y_pred == 0) & (y_true == 0)).sum()
        fp  = ((y_pred == 1) & (y_true == 0)).sum()
        tpr = tp / (tp + fn + 1e-9)
        tnr = tn / (tn + fp + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        print(f"    {thr:>10.1f}  {acc:>7.4f}  {tpr:>7.4f}  {tnr:>7.4f}  {fpr:>7.4f}")


# ─────────────────────────────────────────────────────────────────────────────
# Helper: safe ROC-AUC (guard single-class edge case)
# ─────────────────────────────────────────────────────────────────────────────

def safe_roc_auc(y_true, y_proba):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y_true)) < 2:
        return float('nan')
    return roc_auc_score(y_true, y_proba)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: run model inference (explicit input order, no string heuristics)
# ─────────────────────────────────────────────────────────────────────────────

EXPLICIT_INPUT_ORDER = ['f1', 'f2', 'f3', 'f4', 'f5']   # must match training order

def run_model(model, X_dict):
    """Run model with explicit feature order. Returns flat probability array."""
    test_X = [X_dict[k] for k in EXPLICIT_INPUT_ORDER]
    raw = model.predict(test_X, verbose=0)
    if isinstance(raw, dict):
        return raw['logit'].flatten()
    return raw.flatten()


# ─────────────────────────────────────────────────────────────────────────────
# Helper: load cross-dataset npz files
# ─────────────────────────────────────────────────────────────────────────────

def load_cross_dataset(dataset_dir):
    manifest_path = os.path.join(dataset_dir, 'manifest.csv')
    entries = []

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        header    = lines[0].strip().split(',')
        path_col  = header.index('npz_path') if 'npz_path'  in header else 0
        label_col = header.index('label')    if 'label'     in header else 1
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts    = line.split(',')
            npz_path = os.path.join(dataset_dir, parts[path_col].replace('/', os.sep))
            label    = int(parts[label_col])
            entries.append((npz_path, label))
    else:
        for label_name, label_val in [('real', 0), ('fake', 1)]:
            subdir = os.path.join(dataset_dir, label_name)
            if os.path.isdir(subdir):
                for fname in sorted(os.listdir(subdir)):
                    if fname.endswith('.npz'):
                        entries.append((os.path.join(subdir, fname), label_val))

    if not entries:
        return None, None

    X_dict  = {k: [] for k in FEATURE_KEYS}
    y_list  = []
    missing = 0

    for npz_path, label in entries:
        if not os.path.exists(npz_path):
            missing += 1
            continue
        data = np.load(npz_path)
        if not all(k in data for k in FEATURE_KEYS):
            missing += 1
            continue
        for k in FEATURE_KEYS:
            X_dict[k].append(data[k].astype(np.float32))
        y_list.append(label)

    if missing > 0:
        print(f"  [WARN] Skipped {missing} missing / incomplete .npz files.")

    if not y_list:
        return None, None

    for k in FEATURE_KEYS:
        X_dict[k] = np.stack(X_dict[k], axis=0)
    y = np.array(y_list, dtype=np.int32)
    return X_dict, y


# ─────────────────────────────────────────────────────────────────────────────
# Section: Scaler inspection
# ─────────────────────────────────────────────────────────────────────────────

def inspect_scaler(scaler):
    print(f"\n{SEP}")
    print("  SCALER INSPECTION  (fitted on DeeperForensics TRAIN)")
    print(SEP)
    print(f"  {'key':<4}  {'dim':<6}  {'mean[0:3]':<30}  {'std[0:3]':<30}")
    print(f"  {'-'*4}  {'-'*6}  {'-'*30}  {'-'*30}")
    for k in FEATURE_KEYS:
        if k not in scaler.stats:
            print(f"  {k:<4}  NOT IN SCALER -- BUG!")
            continue
        mu, sigma = scaler.stats[k]
        print(
            f"  {k:<4}  {len(mu):<6}  "
            f"{str(mu[:3].round(4)):<30}  "
            f"{str(sigma[:3].round(4)):<30}"
        )
        # Check for degenerate std (close to 1e-8, meaning all values were identical)
        n_degenerate = (sigma < 1e-4).sum()
        if n_degenerate > 0:
            print(f"        WARNING: {n_degenerate}/{len(sigma)} dimensions have near-zero std "
                  "=> those dimensions will produce large scaled values for OOD inputs!")


# ─────────────────────────────────────────────────────────────────────────────
# Section: Model input order verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_input_order(model):
    print(f"\n{SEP}")
    print("  MODEL INPUT ORDER VERIFICATION")
    print(SEP)
    print(f"  Model has {len(model.inputs)} input(s):")
    for i, inp in enumerate(model.inputs):
        print(f"    [{i}]  name={inp.name}  shape={inp.shape}")
    print(f"\n  Using EXPLICIT order for inference: {EXPLICIT_INPUT_ORDER}")
    print("  (Bypasses fragile string-matching heuristic in eval script)")

    # Warn if model input names don't match explicit order
    for i, (inp, fk) in enumerate(zip(model.inputs, EXPLICIT_INPUT_ORDER)):
        if fk not in inp.name:
            print(f"  WARNING: model.inputs[{i}].name='{inp.name}' does not contain '{fk}'")
            print(f"     => Verify build_full_model() was called with FEATURE_KEYS = {FEATURE_KEYS}")


# ─────────────────────────────────────────────────────────────────────────────
# Core diagnostic: one dataset
# ─────────────────────────────────────────────────────────────────────────────

def diagnose_one(ds_name, X_raw, y, scaler, model):
    print(f"\n{SEP}")
    print(f"  DATASET: {ds_name}")
    print(f"  Samples: {len(y)}  (real={(y==0).sum()}, fake={(y==1).sum()})")
    print(SEP)

    # ── 1. Shape check ────────────────────────────────────────────────────────
    print(f"\n  [SHAPE CHECK]")
    shape_ok = True
    for k in FEATURE_KEYS:
        shape = X_raw[k].shape
        expected_dim = 64
        ok = (len(shape) == 2 and shape[1] == expected_dim)
        flag = "OK" if ok else "WRONG!"
        print(f"    {k}: {shape}  {flag}")
        if not ok:
            shape_ok = False
    if not shape_ok:
        print("  Shape mismatch -- cannot continue for this dataset.")
        return

    # ── 2. Raw statistics ─────────────────────────────────────────────────────
    print_feature_stats(X_raw, "RAW (before scaler)")

    # ── 3. Scaled statistics ──────────────────────────────────────────────────
    X_scaled = scaler.transform(X_raw)
    print_feature_stats(X_scaled, "SCALED (after DeeperForensics scaler)")

    # Distribution warning
    for k in FEATURE_KEYS:
        x = X_scaled[k]
        if abs(x.mean()) > 3.0 or x.std() > 5.0:
            print(f"  WARNING [{k}]: scaled mean={x.mean():.2f}, std={x.std():.2f} "
                  "-- likely far outside training feature space!")
        n_clipped = ((x <= -4.9) | (x >= 4.9)).sum()
        if n_clipped > 0:
            total = x.size
            pct   = 100.0 * n_clipped / total
            print(f"  WARNING [{k}]: {n_clipped}/{total} values ({pct:.1f}%) hit the +-5.0 clip "
                  "-- extreme domain shift detected!")

    # ── 4. Predict WITHOUT scaler (raw -> model) ──────────────────────────────
    print(f"\n  [EXPERIMENT A]  RAW features -> Model (bypassing scaler)")
    y_proba_raw = run_model(model, X_raw)
    auc_raw = safe_roc_auc(y, y_proba_raw)
    print(f"    ROC-AUC (raw, no scaler) = {auc_raw:.4f}")
    print_pred_stats(y_proba_raw, y, "RAW -> Model")

    # ── 5. Predict WITH DeeperForensics scaler (correct pipeline) ─────────────
    print(f"\n  [EXPERIMENT B]  SCALED features -> Model (correct pipeline)")
    y_proba_sc = run_model(model, X_scaled)
    auc_sc = safe_roc_auc(y, y_proba_sc)
    print(f"    ROC-AUC (scaled, correct) = {auc_sc:.4f}")
    print_pred_stats(y_proba_sc, y, "SCALED -> Model")

    # ── 6. Comparison ─────────────────────────────────────────────────────────
    print(f"\n  [COMPARISON SUMMARY for {ds_name}]")
    print(f"    ROC-AUC WITHOUT scaler : {auc_raw:.4f}")
    print(f"    ROC-AUC WITH    scaler : {auc_sc:.4f}")
    delta = auc_sc - auc_raw
    if abs(delta) < 0.02:
        verdict = "Scaler has minimal effect -- NOT a normalization bug (or both equally bad)"
    elif delta > 0:
        verdict = f"Scaler IMPROVES AUC by {delta:.4f} -- normalization was the issue!"
    else:
        verdict = f"Scaler HURTS AUC by {abs(delta):.4f} -- scaling might be mismatched"
    print(f"    Verdict: {verdict}")

    return {
        'roc_auc_raw':    auc_raw,
        'roc_auc_scaled': auc_sc,
        'n_samples':      int(len(y)),
        'n_real':         int((y == 0).sum()),
        'n_fake':         int((y == 1).sum()),
        'scaled_stats':   {k: {
            'mean': float(X_scaled[k].mean()),
            'std':  float(X_scaled[k].std()),
            'min':  float(X_scaled[k].min()),
            'max':  float(X_scaled[k].max()),
        } for k in FEATURE_KEYS},
        'raw_stats': {k: {
            'mean': float(X_raw[k].mean()),
            'std':  float(X_raw[k].std()),
            'min':  float(X_raw[k].min()),
            'max':  float(X_raw[k].max()),
        } for k in FEATURE_KEYS},
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print(SEP)
    print("  CROSS-DATASET NORMALIZATION DIAGNOSTIC")
    print(SEP)

    # ── Guard: model & scaler exist ───────────────────────────────────────────
    for path, label in [(MODEL_PATH, 'Model'), (SCALER_PATH, 'Scaler')]:
        if not os.path.exists(path):
            print(f"ERROR: {label} not found at: {path}")
            sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model  : {MODEL_PATH}")
    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False,
        custom_objects={
            'TransformerEncoderBlock': TransformerEncoderBlock,
            'CrossAttentionBlock':     CrossAttentionBlock,
            'GatedFusion':             GatedFusion,
        },
    )

    # ── Load scaler ───────────────────────────────────────────────────────────
    print(f"Loading scaler : {SCALER_PATH}")
    scaler = FeatureScaler.load(SCALER_PATH)

    # ── Inspect scaler ────────────────────────────────────────────────────────
    inspect_scaler(scaler)

    # ── Verify model input order ──────────────────────────────────────────────
    verify_input_order(model)

    all_results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # A. DeeperForensics TEST SET (the baseline: this MUST work)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  LOADING DeeperForensics TEST SET  (internal baseline)")
    print(SEP)

    test_csv = os.path.join(SPLIT_DIR, 'split_test.csv')
    if os.path.exists(test_csv):
        test_entries = load_split_csv(test_csv)
        X_deeper_raw, y_deeper = load_features(test_entries, FEATURE_DIR, FEATURE_KEYS)
        print(f"  Loaded {len(y_deeper)} samples  "
              f"(real={(y_deeper==0).sum()}, fake={(y_deeper==1).sum()})")
        res = diagnose_one('DeeperForensics (test)', X_deeper_raw, y_deeper, scaler, model)
        if res:
            all_results['DeeperForensics_test'] = res
    else:
        print(f"  [SKIP] {test_csv} not found -- skipping DeeperForensics internal test.")

    # ─────────────────────────────────────────────────────────────────────────
    # B. External datasets
    # ─────────────────────────────────────────────────────────────────────────
    for ds_name, ds_dir in DATASETS_EXT.items():
        if not os.path.isdir(ds_dir):
            print(f"\n  [SKIP] {ds_name}: directory not found at {ds_dir}")
            print("         Run: python scripts/extract_cross_dataset_features.py first.")
            continue

        print(f"\n{SEP}")
        print(f"  LOADING {ds_name}")
        print(SEP)
        X_raw, y = load_cross_dataset(ds_dir)
        if X_raw is None:
            print(f"  [SKIP] No valid samples found in {ds_dir}")
            continue
        print(f"  Loaded {len(y)} samples  (real={(y==0).sum()}, fake={(y==1).sum()})")

        res = diagnose_one(ds_name, X_raw, y, scaler, model)
        if res:
            safe_name = ds_name.replace(' ', '_').replace('+', '_').replace('/', '_')
            all_results[safe_name] = res

    # ─────────────────────────────────────────────────────────────────────────
    # Final comparative summary
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n\n{SEP}")
    print("  FINAL SUMMARY -- ROC-AUC COMPARISON")
    print(SEP)
    print(f"  {'Dataset':<30}  {'n':>6}  {'AUC (raw)':>10}  {'AUC (scaled)':>13}  {'Delta':>7}")
    print(f"  {'-'*30}  {'-'*6}  {'-'*10}  {'-'*13}  {'-'*7}")
    for ds_name, r in all_results.items():
        delta = r['roc_auc_scaled'] - r['roc_auc_raw']
        print(
            f"  {ds_name:<30}  {r['n_samples']:>6}  "
            f"{r['roc_auc_raw']:>10.4f}  {r['roc_auc_scaled']:>13.4f}  "
            f"{delta:>+7.4f}"
        )

    print(f"\n{SEP}")
    print("  INTERPRETATION GUIDE")
    print(SEP)
    print("""
  If DeeperForensics (scaled) ROC-AUC approx training result -> model & scaler are correct.

  For external datasets (Celeb-DF, FF++):

    AUC(scaled) ~ 0.5  AND  AUC(raw) ~ 0.5
      -> Domain shift is genuine. Scaler is NOT the problem.
         The model learned DeeperForensics-specific artifacts.

    AUC(scaled) >> AUC(raw)
      -> Normalization WAS the bug. Fixed by applying the correct scaler.

    AUC(raw) ~ AUC(scaled) ~ 0.5  BUT mean/std of scaled features are huge
      -> Feature extractor is producing incompatible values for external data.
         Check for dataset-specific compression / resolution artifacts.

    Model predicts almost all FAKE (low TNR, high TPR)
      -> Model is biased toward 'fake'. Check if training data was imbalanced
         or if the scaler clips most real-class features to extreme values.
    """)

    # Save summary JSON
    out_path = os.path.join('checkpoints', 'deepfakes', 'cross_dataset', 'diagnostic_results.json')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    print(f"  Full results saved to: {out_path}\n")


if __name__ == '__main__':
    main()
