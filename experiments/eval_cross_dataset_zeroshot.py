"""
experiments/eval_cross_dataset_zeroshot.py
-------------------------------------------
Zero-Shot Cross-Dataset Evaluation

Loads the BEST trained Stage D model (or Stage C/B) and evaluates
WITHOUT any retraining on two external datasets:

  1. Celeb-DF v2  (DataSets/Celeb-DF-v2/)
  2. FaceForensics++ (DataSets/deepfakes/) — using a held-out set

The model's scaler (fitted on DeeperForensics train) is applied to the
external features. No new training occurs.

Expected structure for cross-dataset .npz files:
  DataSets/cross_dataset_features/
      celeb_df/
          *.npz   (each with f1, f2, f3, f4, f5)
      ff_plus_plus/
          *.npz

These must be pre-extracted using scripts/deepfakes_extract_features.py
pointed at the respective dataset folders.

Usage:
    python experiments/eval_cross_dataset_zeroshot.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

from stage_d_full_model import (
    TransformerEncoderBlock, 
    CrossAttentionBlock, 
    GatedFusion
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.shared_utils import (
    FEATURE_KEYS,
    FeatureScaler, compute_all_metrics, print_metrics, save_metrics,
)

# ─────────────────────────────────────────────────────────────────────────────
STAGE_D_DIR = os.path.join('checkpoints', 'deepfakes', 'stage_d')
SCALER_PATH = os.path.join(STAGE_D_DIR, 'scaler.pkl')
MODEL_PATH  = os.path.join(STAGE_D_DIR, 'best_model.keras')

CROSS_DATASET_BASE = os.path.join('DataSets', 'cross_dataset_features')

DATASETS = {
    'Celeb-DF v2':      os.path.join(CROSS_DATASET_BASE, 'celeb_df'),
    'FaceForensics++':  os.path.join(CROSS_DATASET_BASE, 'ff_plus_plus'),
}

# Labels: the manifest.csv inside each cross-dataset folder must have npz_path,label
OUTPUT_DIR = os.path.join('checkpoints', 'deepfakes', 'cross_dataset')
# ─────────────────────────────────────────────────────────────────────────────


def load_cross_dataset(dataset_dir):
    """
    Load all .npz files from dataset_dir.
    If a manifest.csv exists, use it for labels.
    Otherwise infer label from folder name (real/fake subfolder).
    Returns X_dict, y.
    """
    manifest_path = os.path.join(dataset_dir, 'manifest.csv')
    entries = []

    if os.path.exists(manifest_path):
        with open(manifest_path, 'r') as f:
            lines = f.readlines()
        header = lines[0].strip().split(',')
        path_col  = header.index('npz_path') if 'npz_path' in header else 0
        label_col = header.index('label') if 'label' in header else 1
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            npz_path = os.path.join(dataset_dir, parts[path_col].replace('/', os.sep))
            label = int(parts[label_col])
            entries.append((npz_path, label))
    else:
        # Auto-discover: real/ and fake/ subfolders
        for label_name, label_val in [('real', 0), ('fake', 1), ('source', 0), ('manipulated', 1)]:
            subdir = os.path.join(dataset_dir, label_name)
            if os.path.isdir(subdir):
                for fname in os.listdir(subdir):
                    if fname.endswith('.npz'):
                        entries.append((os.path.join(subdir, fname), label_val))

    if not entries:
        return None, None

    X_dict = {k: [] for k in FEATURE_KEYS}
    y_list = []
    missing = 0

    for npz_path, label in entries:
        if not os.path.exists(npz_path):
            missing += 1
            continue
        data = np.load(npz_path)
        ok = all(k in data for k in FEATURE_KEYS)
        if not ok:
            missing += 1
            continue
        for k in FEATURE_KEYS:
            X_dict[k].append(data[k].astype(np.float32))
        y_list.append(label)

    if missing > 0:
        print(f"  [WARN] Skipped {missing} missing/incomplete .npz files.")

    if not y_list:
        return None, None

    for k in FEATURE_KEYS:
        X_dict[k] = np.stack(X_dict[k], axis=0)
    y = np.array(y_list, dtype=np.int32)
    print(f"  Loaded {len(y)} samples: real={( y==0).sum()}, fake={(y==1).sum()}")
    return X_dict, y


def prepare_cross_dataset_extraction_script():
    """
    Writes a helper script that tells the user how to extract features
    for cross-dataset evaluation. This is not auto-run.
    """
    script_content = '''"""
Cross-dataset feature extraction helper.

Run this ONCE for each external dataset BEFORE running eval_cross_dataset_zeroshot.py.
Adjust the target_folders to match your external dataset structure.
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from scripts.deepfakes_extract_features import main as extract_main

# To extract features from Celeb-DF v2:
# Set base_dir and output_dir to the correct paths, then call the extractor
# on the Celeb-real and Celeb-synthesis folders.

# Example manual extraction:
import glob
import numpy as np
import cv2
from tqdm import tqdm
from data.preprocessing import preprocess_video
from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur import extract_motion_blur_features
from extractors.biomechanics import extract_biomechanics_features
from extractors.lighting_sh import extract_lighting_sh_features
from extractors.frequency_fft import extract_fft_spectrum

DATASETS = {
    'celeb_df': {
        'real_dirs':  ['DataSets/Celeb-DF-v2/Celeb-real', 'DataSets/Celeb-DF-v2/YouTube-real'],
        'fake_dirs':  ['DataSets/Celeb-DF-v2/Celeb-synthesis'],
        'output_dir': 'DataSets/cross_dataset_features/celeb_df',
    },
    'ff_plus_plus': {
        'real_dirs':  ['DataSets/FaceForensics++/original_sequences/youtube/c40/videos'],
        'fake_dirs':  ['DataSets/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos',
                       'DataSets/FaceForensics++/manipulated_sequences/Face2Face/c40/videos'],
        'output_dir': 'DataSets/cross_dataset_features/ff_plus_plus',
    }
}

FEATURE_DIM = 64
N_FRAMES    = 16

def extract_video(video_path, output_path, n_frames=16, target_size=(256, 256), sigma=0.8, feature_dim=64):
    if os.path.exists(output_path):
        return False
    frames = preprocess_video(video_path, n_frames=n_frames, target_size=target_size, sigma=sigma)
    if len(frames) < 2:
        f1=f2=f3=f4=f5=np.zeros(feature_dim, dtype=np.float32)
    else:
        f1 = extract_lens_distortion_features(frames, feature_dim)
        f2 = extract_motion_blur_features(frames, feature_dim)
        f3 = extract_biomechanics_features(frames, feature_dim)
        f4 = extract_lighting_sh_features(frames, feature_dim)
        f5_list = []
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            f5_list.append(extract_fft_spectrum(gray, feature_dim))
        f5 = np.mean(f5_list, axis=0)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)
    return True

for ds_name, cfg in DATASETS.items():
    out_dir = cfg['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    manifest_rows = []

    for label_dirs, label in [(cfg['real_dirs'], 0), (cfg['fake_dirs'], 1)]:
        for d in label_dirs:
            if not os.path.isdir(d):
                print(f"WARNING: {d} not found. Skipping.")
                continue
            videos = glob.glob(os.path.join(d, '**', '*.mp4'), recursive=True)
            for vp in tqdm(videos, desc=f"{ds_name} label={label}"):
                stem = os.path.splitext(os.path.basename(vp))[0]
                out_path = os.path.join(out_dir, 'real' if label==0 else 'fake', f'{stem}.npz')
                extract_video(vp, out_path, N_FRAMES, (256,256), 0.8, FEATURE_DIM)
                rel = os.path.relpath(out_path, out_dir).replace(os.sep, '/')
                manifest_rows.append(f"{rel},{label}")

    with open(os.path.join(out_dir, 'manifest.csv'), 'w') as f:
        f.write("npz_path,label\\n")
        f.write("\\n".join(manifest_rows))

    print(f"Done: {ds_name}  ({len(manifest_rows)} videos)")
'''
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join('scripts', 'extract_cross_dataset_features.py')
    if not os.path.exists(path):
        with open(path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        print(f"  Helper script written to {path}")


def main():
    # ── Write helper extraction script (first-time setup) ─────────────────────
   # prepare_cross_dataset_extraction_script()
#
    # ── Validate model & scaler exist ─────────────────────────────────────────
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("  Run stage_d_full_model.py first, or update MODEL_PATH to stage_b/stage_c.")
        sys.exit(1)

    if not os.path.exists(SCALER_PATH):
        print(f"ERROR: Scaler not found at {SCALER_PATH}")
        sys.exit(1)

    print(f"Loading trained model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(
    MODEL_PATH, 
    compile=False,
    custom_objects={
        'TransformerEncoderBlock': TransformerEncoderBlock,
        'CrossAttentionBlock': CrossAttentionBlock,
        'GatedFusion': GatedFusion
    }
)
    # model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    print(f"Loading scaler from {SCALER_PATH} ...")
    scaler = FeatureScaler.load(SCALER_PATH)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_results = {}

    for ds_name, ds_dir in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"  Zero-Shot Evaluation: {ds_name}")
        print(f"  Directory: {ds_dir}")
        print(f"{'='*60}")

        if not os.path.isdir(ds_dir):
            print(f"  [SKIP] Directory does not exist. "
                  f"Run scripts/extract_cross_dataset_features.py first.")
            all_results[ds_name] = 'SKIPPED — features not found'
            continue

        X_dict, y_test = load_cross_dataset(ds_dir)
        if X_dict is None:
            print(f"  [SKIP] No valid .npz samples found in {ds_dir}.")
            all_results[ds_name] = 'SKIPPED — no samples'
            continue

        # Apply TRAIN scaler (no refit!)
        X_norm = scaler.transform(X_dict)

        # ── Predict ────────────────────────────────────────────────────────────
        # Determine model input order from model's input layers
        input_order = [inp.name.replace('_input:0', '').replace(':0', '').replace('input_', '') 
                       for inp in model.inputs]
        # Map back to feature keys
        fk_order = []
        for inp_name in model.inputs:
            for fk in FEATURE_KEYS:
                if fk in inp_name.name:
                    fk_order.append(fk)
                    break

        if len(fk_order) != len(model.inputs):
            # Fallback: use FEATURE_KEYS order
            fk_order = FEATURE_KEYS[:len(model.inputs)]

        test_X = [X_norm[k] for k in fk_order]
        raw_output = model.predict(test_X, verbose=0)

        if isinstance(raw_output, dict):
            y_pred_proba = raw_output['logit'].flatten()
        else:
            y_pred_proba = raw_output.flatten()

        metrics = compute_all_metrics(y_test, y_pred_proba)
        metrics['dataset']   = ds_name
        metrics['model_src'] = MODEL_PATH
        metrics['n_samples'] = int(len(y_test))
        metrics['zero_shot'] = True

        print_metrics(f"Zero-Shot: {ds_name}", metrics)

        ds_safe = ds_name.replace(' ', '_').replace('+', '_').lower()
        save_metrics(metrics, os.path.join(OUTPUT_DIR, f'metrics_{ds_safe}.json'))
        all_results[ds_name] = metrics

    # ── Combined summary ───────────────────────────────────────────────────────
    print("\n\n" + "="*65)
    print("  ZERO-SHOT CROSS-DATASET EVALUATION SUMMARY")
    print("="*65)
    print(f"  {'Dataset':<22} {'Acc':>7} {'F1':>7} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print(f"  {'-'*22} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for ds_name, m in all_results.items():
        if isinstance(m, str):
            print(f"  {ds_name:<22}  {m}")
        else:
            print(f"  {ds_name:<22} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
                  f"{m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f}")
    print("="*65)

    with open(os.path.join(OUTPUT_DIR, 'summary.json'), 'w') as f:
        json.dump({k: v if isinstance(v, str) else v for k, v in all_results.items()}, f, indent=4)
    print(f"\nSummary saved to {OUTPUT_DIR}/summary.json")


if __name__ == '__main__':
    main()
