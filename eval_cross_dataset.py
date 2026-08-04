"""
eval_cross_dataset.py
---------------------
Zero-shot cross-dataset evaluation for the Mid-Level Fusion model
(Sub-System 1 + Sub-System 2).

Works with ANY folder of videos — not just FaceForensics++.
Extracts features on-the-fly directly from raw .mp4 files, then
runs inference through the trained model.

Supported dataset layouts:
  Layout A — two separate folders:
      --real_dir  path/to/real_videos/
      --fake_dir  path/to/fake_videos/

  Layout B — single root with 'real' and 'fake' sub-folders:
      --data_dir  path/to/dataset/
      (expects: data_dir/real/*.mp4 and data_dir/fake/*.mp4)

  Layout C — single root with custom subfolder names:
      --data_dir  path/to/dataset/
      --real_subdir celeb-real
      --fake_subdir celeb-synthesis

Usage examples:
    # Celeb-DF, DFDC, DFF, or any custom dataset:
    python eval_cross_dataset.py --real_dir data/CelebDF/real --fake_dir data/CelebDF/fake

    # Using the v2 preprocessing (CLAHE + bilateral + alignment):
    python eval_cross_dataset.py --real_dir ... --fake_dir ... --preprocess_v2

    # Limit samples for quick testing:
    python eval_cross_dataset.py --real_dir ... --fake_dir ... --max_per_class 200
"""

import os
import sys
import glob
import argparse
import random

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import yaml
import tensorflow as tf
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score,
    precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur     import extract_motion_blur_features
from extractors.biomechanics    import extract_biomechanics_features
from extractors.lighting_sh     import extract_lighting_sh_features
from extractors.frequency_fft   import extract_fft_spectrum

from models.fusion_1_2_mid  import build_fusion_1_2_mid, MidLevelFusionBlock12
from models.master_fusion   import SubSystem1Encoder, SubSystem2Encoder


CUSTOM_OBJS = {
    'SubSystem1Encoder':     SubSystem1Encoder,
    'SubSystem2Encoder':     SubSystem2Encoder,
    'MidLevelFusionBlock12': MidLevelFusionBlock12,
}


# ---------------------------------------------------------------------------
# Video discovery
# ---------------------------------------------------------------------------

def discover_videos(real_dir=None, fake_dir=None,
                    data_dir=None, real_subdir='real', fake_subdir='fake',
                    max_per_class=None, seed=42):
    """Return list of (video_path, label) tuples (label 0=real, 1=fake)."""
    def _glob(folder):
        vids = []
        for ext in ('*.mp4', '*.avi', '*.mov', '*.mkv'):
            vids.extend(glob.glob(os.path.join(folder, '**', ext), recursive=True))
        return sorted(vids)

    # Resolve paths
    if real_dir and fake_dir:
        real_vids = _glob(real_dir)
        fake_vids = _glob(fake_dir)
    elif data_dir:
        real_vids = _glob(os.path.join(data_dir, real_subdir))
        fake_vids = _glob(os.path.join(data_dir, fake_subdir))
    else:
        raise ValueError("Provide either --real_dir + --fake_dir, or --data_dir")

    rng = random.Random(seed)
    rng.shuffle(real_vids)
    rng.shuffle(fake_vids)

    if max_per_class:
        real_vids = real_vids[:max_per_class]
        fake_vids = fake_vids[:max_per_class]

    entries = [(v, 0) for v in real_vids] + [(v, 1) for v in fake_vids]
    print(f"Found {len(real_vids)} real  +  {len(fake_vids)} fake  =  {len(entries)} total videos")
    return entries


# ---------------------------------------------------------------------------
# On-the-fly feature extraction
# ---------------------------------------------------------------------------

def extract_features_from_video(video_path, n_frames, target_size,
                                 feature_dim, use_v2=False):
    """Extract [f1, f2, f3, f4, f5] from a single video file."""
    if use_v2:
        from data.preprocessing_v2 import preprocess_video_v2
        frames = preprocess_video_v2(video_path, n_frames=n_frames,
                                     target_size=target_size)
    else:
        from data.preprocessing import preprocess_video
        frames = preprocess_video(video_path, n_frames=n_frames,
                                  target_size=target_size)

    if len(frames) < 2:
        zeros = np.zeros(feature_dim, dtype=np.float32)
        return zeros, zeros, zeros, zeros, zeros

    f1 = extract_lens_distortion_features(frames, feature_dim)
    f2 = extract_motion_blur_features(frames, feature_dim)
    f3 = extract_biomechanics_features(frames, feature_dim)
    f4 = extract_lighting_sh_features(frames, feature_dim)
    f5 = extract_fft_features(frames, feature_dim)
    return f1, f2, f3, f4, f5


# ---------------------------------------------------------------------------
# Metrics & plots
# ---------------------------------------------------------------------------

def compute_metrics(labels, probs, name):
    preds = (probs >= 0.5).astype(int)
    auc   = roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0
    return {
        'model':     name,
        'accuracy':  accuracy_score(labels, preds),
        'auc':       auc,
        'precision': precision_score(labels, preds, zero_division=0),
        'recall':    recall_score(labels, preds, zero_division=0),
        'f1':        f1_score(labels, preds, zero_division=0),
    }


def print_table(rows):
    hdr = f"{'Model':<52} {'Acc':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print('=' * len(hdr))
    print(hdr)
    print('-' * len(hdr))
    for r in rows:
        print(f"{r['model']:<52} "
              f"{r['accuracy']:>7.4f} "
              f"{r['auc']:>7.4f} "
              f"{r['precision']:>7.4f} "
              f"{r['recall']:>7.4f} "
              f"{r['f1']:>7.4f}")
    print('=' * len(hdr))


def save_cm(labels, probs, title, path):
    if not HAS_MPL:
        return
    cm = confusion_matrix(labels, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake'])
    ax.set_yticklabels(['Real', 'Fake'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(title)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=13, fontweight='bold')
    fig.colorbar(im); fig.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    fig.savefig(path, dpi=150); plt.close(fig)
    print(f"Confusion matrix saved → {path}")


def save_roc(roc_data, path):
    if not HAS_MPL or not roc_data:
        return
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for (name, labels, probs), color in zip(roc_data, colors):
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f'{name}  (AUC={auc_val:.4f})', color=color, lw=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.35, lw=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC — Cross-Dataset Evaluation (Fusion 1+2)', fontsize=13)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(path, dpi=150); plt.close()
    print(f"ROC curve saved → {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Cross-Dataset Evaluation for Mid-Level Fusion 1+2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Dataset paths
    parser.add_argument('--real_dir',     default=None,
                        help='Folder containing REAL video files')
    parser.add_argument('--fake_dir',     default=None,
                        help='Folder containing FAKE video files')
    parser.add_argument('--data_dir',     default=None,
                        help='Root dataset folder (expects real/ and fake/ subfolders)')
    parser.add_argument('--real_subdir',  default='real',
                        help='Real subfolder name under --data_dir (default: real)')
    parser.add_argument('--fake_subdir',  default='fake',
                        help='Fake subfolder name under --data_dir (default: fake)')
    parser.add_argument('--max_per_class', type=int, default=None,
                        help='Max videos per class (for quick tests)')

    # Model & config
    parser.add_argument('--model',  default='checkpoints/fusion12_mid_final_best.keras')
    parser.add_argument('--config', default='config/default_config.yaml')

    # Preprocessing
    parser.add_argument('--preprocess_v2', action='store_true',
                        help='Use v2 preprocessing (CLAHE + bilateral + eye alignment)')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Frames to sample per video (default: from config)')

    # Output
    parser.add_argument('--out_dir', default='results/cross_dataset',
                        help='Output directory for plots (default: results/cross_dataset)')

    args = parser.parse_args()

    if not args.real_dir and not args.fake_dir and not args.data_dir:
        parser.print_help()
        print("\n[ERROR] Provide --real_dir + --fake_dir, or --data_dir")
        sys.exit(1)

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim   = cfg['model']['embed_dim']
    n_frames    = args.n_frames or cfg['preprocessing']['n_frames']
    target_size = tuple(cfg['preprocessing']['target_size'])

    pp_label = "v2 (CLAHE+Bilateral)" if args.preprocess_v2 else "v1 (Gaussian)"
    print(f"\n{'='*60}")
    print(f"  Cross-Dataset Evaluation — Mid-Level Fusion 1+2")
    print(f"  Preprocessing : {pp_label}")
    print(f"  n_frames      : {n_frames}")
    print(f"  Model         : {args.model}")
    print(f"  Output dir    : {args.out_dir}")
    print(f"{'='*60}\n")

    # --- Discover videos ---
    entries = discover_videos(
        real_dir=args.real_dir, fake_dir=args.fake_dir,
        data_dir=args.data_dir,
        real_subdir=args.real_subdir, fake_subdir=args.fake_subdir,
        max_per_class=args.max_per_class,
    )

    if not entries:
        print("[ERROR] No videos found. Check your paths.")
        sys.exit(1)

    # --- Load model ---
    if os.path.exists(args.model):
        print(f"\nLoading model: {args.model}")
        model = tf.keras.models.load_model(
            args.model, compile=False, custom_objects=CUSTOM_OBJS
        )
    else:
        print(f"[WARN] Checkpoint not found — using random weights.")
        model = build_fusion_1_2_mid(feature_dim, embed_dim)

    # --- Extract features + run inference ---
    print(f"\nExtracting features and running inference on {len(entries)} videos ...\n")

    fusion_probs, sub1_probs, sub2_probs, all_labels = [], [], [], []
    failed = 0

    for video_path, label in tqdm(entries, desc='Evaluating'):
        try:
            f1, f2, f3, f4, f5 = extract_features_from_video(
                video_path, n_frames, target_size, feature_dim, args.preprocess_v2
            )

            inp = {
                'f1_lens_distortion': np.expand_dims(f1, 0),
                'f2_motion_blur':     np.expand_dims(f2, 0),
                'f3_biomechanics':    np.expand_dims(f3, 0),
                'f4_lighting_sh':     np.expand_dims(f4, 0),
                'f5_frequency_fft':   np.expand_dims(f5, 0),
            }

            out = model(inp, training=False)

            fusion_probs.append(float(tf.sigmoid(out['fusion_logits']).numpy()[0, 0]))
            sub1_probs.append(float(tf.sigmoid(out['sub1_logits']).numpy()[0, 0]))
            sub2_probs.append(float(tf.sigmoid(out['sub2_logits']).numpy()[0, 0]))
            all_labels.append(label)

        except Exception as e:
            print(f"  [WARN] Skipping {os.path.basename(video_path)}: {e}")
            failed += 1
            continue

    if not all_labels:
        print("[ERROR] No samples were successfully processed.")
        sys.exit(1)

    all_labels  = np.array(all_labels)
    fusion_probs = np.array(fusion_probs)
    sub1_probs   = np.array(sub1_probs)
    sub2_probs   = np.array(sub2_probs)

    print(f"\nProcessed: {len(all_labels)}  |  Failed/skipped: {failed}")
    print(f"Real: {int((all_labels == 0).sum())}  |  Fake: {int((all_labels == 1).sum())}\n")

    # --- Metrics ---
    all_metrics = [
        compute_metrics(all_labels, fusion_probs, 'Mid-Level Fusion (Sys 1 + Sys 2)'),
        compute_metrics(all_labels, sub1_probs,   'Sub-System 1 Auxiliary (Hardware Optics)'),
        compute_metrics(all_labels, sub2_probs,   'Sub-System 2 Auxiliary (Biological Domain)'),
    ]

    print('\n')
    print_table(all_metrics)

    # --- Plots ---
    os.makedirs(args.out_dir, exist_ok=True)
    roc_data = [
        ('Mid-Level Fusion', all_labels, fusion_probs),
        ('Sub-System 1 (aux)', all_labels, sub1_probs),
        ('Sub-System 2 (aux)', all_labels, sub2_probs),
    ]
    save_roc(roc_data, os.path.join(args.out_dir, 'roc_cross_dataset.png'))
    save_cm(all_labels, fusion_probs, 'Confusion Matrix — Mid-Level Fusion 1+2 (Cross-Dataset)',
            os.path.join(args.out_dir, 'cm_fusion12_cross.png'))
    save_cm(all_labels, sub1_probs,   'Confusion Matrix — Sub-System 1 Auxiliary (Cross-Dataset)',
            os.path.join(args.out_dir, 'cm_sub1_cross.png'))
    save_cm(all_labels, sub2_probs,   'Confusion Matrix — Sub-System 2 Auxiliary (Cross-Dataset)',
            os.path.join(args.out_dir, 'cm_sub2_cross.png'))

    print('\nCross-dataset evaluation complete.')


if __name__ == '__main__':
    main()
