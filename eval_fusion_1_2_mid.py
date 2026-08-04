"""
eval_fusion_1_2_mid.py
-----------------------
Standalone evaluation script for the Mid-Level Fusion of
Sub-System 1 (Hardware Optics) + Sub-System 2 (Biological Domain).

Produces:
  - Per-model metrics table (Acc, AUC, Precision, Recall, F1)
  - Confusion matrices for fusion head and both auxiliary heads
  - ROC curve comparing fusion vs each subsystem standalone

Usage:
    python eval_fusion_1_2_mid.py
    python eval_fusion_1_2_mid.py --model checkpoints/fusion12_mid_final_best.keras
    python eval_fusion_1_2_mid.py --model checkpoints/fusion12_mid_final_best.keras --config config/default_config.yaml
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import yaml
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
)

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from data.dataset_loader import get_datasets
from models.fusion_1_2_mid import build_fusion_1_2_mid, MidLevelFusionBlock12
from models.master_fusion import SubSystem1Encoder, SubSystem2Encoder


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def predict(model, dataset, logit_key='fusion_logits'):
    """Run inference; return (labels, probs) as numpy arrays."""
    all_labels, all_probs = [], []
    for inputs, labels in dataset:
        outputs = model(inputs, training=False)
        if isinstance(outputs, dict):
            logits = outputs.get(logit_key, list(outputs.values())[0])
        else:
            logits = outputs
        probs = tf.sigmoid(logits).numpy().ravel()
        lbl   = labels.numpy().ravel() if hasattr(labels, 'numpy') else labels
        all_probs.extend(probs)
        all_labels.extend(lbl)
    return np.array(all_labels), np.array(all_probs)


def metrics(labels, probs, name):
    """Compute and return a metrics dict."""
    if len(np.unique(labels)) < 2:
        auc = 0.0
    else:
        auc = roc_auc_score(labels, probs)
    preds = (probs >= 0.5).astype(int)
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
        print(
            f"{r['model']:<52} "
            f"{r['accuracy']:>7.4f} "
            f"{r['auc']:>7.4f} "
            f"{r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['f1']:>7.4f}"
        )
    print('=' * len(hdr))


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_cm(labels, probs, title, out_path):
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
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150); plt.close(fig)
    print(f"Confusion matrix saved → {out_path}")


def plot_roc(roc_data, out_path):
    if not HAS_MPL:
        return
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    colors = ['#4C72B0', '#DD8452', '#55A868']
    for (name, labels, probs), color in zip(roc_data, colors):
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f'{name}  (AUC = {auc_val:.4f})', color=color, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.35, linewidth=1)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves — Mid-Level Fusion 1+2 vs Standalone Sub-Systems', fontsize=13)
    plt.legend(loc='lower right', fontsize=11)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"ROC curve saved → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate Mid-Level Fusion of Sub-System 1 + Sub-System 2'
    )
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--model',  default='checkpoints/fusion12_mid_final_best.keras',
                        help='Path to the trained mid-level fusion checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim   = cfg['model']['embed_dim']

    custom_objs = {
        'SubSystem1Encoder':   SubSystem1Encoder,
        'SubSystem2Encoder':   SubSystem2Encoder,
        'MidLevelFusionBlock12': MidLevelFusionBlock12,
    }

    # --- Data ---
    _, _, test_ds, split_info = get_datasets(cfg)
    print(f"\nEvaluating on {split_info['test']} test samples\n")

    # --- Build or load fusion model ---
    if os.path.exists(args.model):
        print(f"Loading fusion model from: {args.model}")
        fusion_model = tf.keras.models.load_model(
            args.model, compile=False, custom_objects=custom_objs
        )
    else:
        print(f"[WARN] No checkpoint found at {args.model} — using random weights.")
        fusion_model = build_fusion_1_2_mid(feature_dim, embed_dim)

    # ---------------------------------------------------------------
    # 1. Mid-Level Fusion head (primary output)
    # ---------------------------------------------------------------
    fusion_ds = test_ds.map(lambda x, y: (
        {
            'f1_lens_distortion': x['f1_lens_distortion'],
            'f2_motion_blur':     x['f2_motion_blur'],
            'f3_biomechanics':    x['f3_biomechanics'],
            'f4_lighting_sh':     x['f4_lighting_sh'],
            'f5_frequency_fft':   x['f5_frequency_fft'],
        }, y
    ))

    f_labels, f_probs = predict(fusion_model, fusion_ds, 'fusion_logits')
    all_metrics = [metrics(f_labels, f_probs, 'Mid-Level Fusion (Sys 1 + Sys 2)')]
    roc_data    = [('Mid-Level Fusion', f_labels, f_probs)]

    # ---------------------------------------------------------------
    # 2. Sub-System 1 auxiliary head (from within the fusion model)
    # ---------------------------------------------------------------
    sub1_ds = test_ds.map(lambda x, y: (
        {
            'f1_lens_distortion': x['f1_lens_distortion'],
            'f2_motion_blur':     x['f2_motion_blur'],
            'f3_biomechanics':    x['f3_biomechanics'],
            'f4_lighting_sh':     x['f4_lighting_sh'],
            'f5_frequency_fft':   x['f5_frequency_fft'],
        }, y
    ))

    s1_labels, s1_probs = predict(fusion_model, sub1_ds, 'sub1_logits')
    all_metrics.append(metrics(s1_labels, s1_probs, 'Sub-System 1 Auxiliary (Hardware Optics)'))
    roc_data.append(('Sub-System 1 (aux)', s1_labels, s1_probs))

    # ---------------------------------------------------------------
    # 3. Sub-System 2 auxiliary head (from within the fusion model)
    # ---------------------------------------------------------------
    sub2_ds = test_ds.map(lambda x, y: (
        {
            'f1_lens_distortion': x['f1_lens_distortion'],
            'f2_motion_blur':     x['f2_motion_blur'],
            'f3_biomechanics':    x['f3_biomechanics'],
            'f4_lighting_sh':     x['f4_lighting_sh'],
            'f5_frequency_fft':   x['f5_frequency_fft'],
        }, y
    ))

    s2_labels, s2_probs = predict(fusion_model, sub2_ds, 'sub2_logits')
    all_metrics.append(metrics(s2_labels, s2_probs, 'Sub-System 2 Auxiliary (Biological Domain)'))
    roc_data.append(('Sub-System 2 (aux)', s2_labels, s2_probs))

    # ---------------------------------------------------------------
    # Print table
    # ---------------------------------------------------------------
    print('\n')
    print_table(all_metrics)

    # ---------------------------------------------------------------
    # Plots
    # ---------------------------------------------------------------
    os.makedirs('results', exist_ok=True)
    plot_roc(roc_data, 'results/roc_fusion12_mid.png')
    plot_cm(f_labels,  f_probs,  'Confusion Matrix — Mid-Level Fusion 1+2',         'results/cm_fusion12_mid.png')
    plot_cm(s1_labels, s1_probs, 'Confusion Matrix — Sub-System 1 Auxiliary',        'results/cm_fusion12_mid_sub1.png')
    plot_cm(s2_labels, s2_probs, 'Confusion Matrix — Sub-System 2 Auxiliary',        'results/cm_fusion12_mid_sub2.png')

    print('\nEvaluation complete.')


if __name__ == '__main__':
    main()
