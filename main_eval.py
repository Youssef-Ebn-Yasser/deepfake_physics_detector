"""
main_eval.py  --  Metrics & Ablation Evaluation Generator
---------------------------------------------------------
Evaluates trained models and produces:
    1. Per-model accuracy, AUC-ROC, precision, recall, F1-score.
    2. Ablation table comparing individual sub-systems vs. full fusion.
    3. Confusion-matrix and ROC-curve plots.

Usage:
    python main_eval.py
    python main_eval.py --config config/default_config.yaml
    python main_eval.py --master checkpoints/master_final_best.keras
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
from models.master_fusion import build_master_physics_detector
from models.subsystem_1 import build_subsystem1_model
from models.subsystem_2 import build_subsystem2_model


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def predict_on_dataset(model, dataset, logit_key=None):
    """
    Run inference and collect labels + probabilities.

    Args:
        model:      Keras model.
        dataset:    tf.data.Dataset yielding (inputs_dict, labels).
        logit_key:  Which output key to use (for multi-output models).

    Returns:
        (labels, probs) as numpy arrays.
    """
    all_labels, all_probs = [], []

    for inputs, labels in dataset:
        outputs = model(inputs, training=False)

        if isinstance(outputs, dict):
            logits = outputs.get(logit_key, list(outputs.values())[0])
        else:
            logits = outputs

        probs = tf.sigmoid(logits).numpy().ravel()
        all_probs.extend(probs)
        all_labels.extend(labels.numpy().ravel() if hasattr(labels, 'numpy') else labels)

    return np.array(all_labels), np.array(all_probs)


def compute_metrics(labels, probs, model_name='model'):
    """Compute classification metrics from labels and predicted probabilities."""
    preds = (probs >= 0.5).astype(int)
    return {
        'model': model_name,
        'accuracy': accuracy_score(labels, preds),
        'auc': roc_auc_score(labels, probs) if len(np.unique(labels)) > 1 else 0.0,
        'precision': precision_score(labels, preds, zero_division=0),
        'recall': recall_score(labels, preds, zero_division=0),
        'f1': f1_score(labels, preds, zero_division=0),
    }


def print_metrics_table(results_list):
    """Pretty-print a comparison table of evaluation results."""
    header = f"{'Model':<45} {'Acc':>7} {'AUC':>7} {'Prec':>7} {'Rec':>7} {'F1':>7}"
    print('=' * len(header))
    print(header)
    print('-' * len(header))
    for r in results_list:
        print(
            f"{r['model']:<45} "
            f"{r['accuracy']:>7.4f} "
            f"{r['auc']:>7.4f} "
            f"{r['precision']:>7.4f} "
            f"{r['recall']:>7.4f} "
            f"{r['f1']:>7.4f}"
        )
    print('=' * len(header))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_roc_curves(roc_data, output_path='results/roc_curves.png'):
    if not HAS_MPL:
        print('matplotlib not available -- skipping ROC plot.')
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.figure(figsize=(8, 6))
    for name, labels, probs in roc_data:
        if len(np.unique(labels)) < 2:
            continue
        fpr, tpr, _ = roc_curve(labels, probs)
        auc_val = roc_auc_score(labels, probs)
        plt.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves -- Ablation Study')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f'ROC curve saved to {output_path}')


def plot_confusion_matrix(labels, probs, model_name, output_path='results/cm.png'):
    if not HAS_MPL:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cm = confusion_matrix(labels, (probs >= 0.5).astype(int))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(['Real', 'Fake']); ax.set_yticklabels(['Real', 'Fake'])
    ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
    ax.set_title(f'Confusion Matrix -- {model_name}')
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black')
    fig.colorbar(im); fig.tight_layout()
    fig.savefig(output_path, dpi=150); plt.close(fig)
    print(f'Confusion matrix saved to {output_path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate & Ablation')
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--master', default='checkpoints/master_final_best.keras')
    parser.add_argument('--sub1', default='checkpoints/subsystem1_best.keras')
    parser.add_argument('--sub2', default='checkpoints/subsystem2_best.keras')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim = cfg['model']['embed_dim']

    from models.master_fusion import SubSystem1Encoder, SubSystem2Encoder, MasterFusionBlock
    custom_objs = {
        'SubSystem1Encoder': SubSystem1Encoder,
        'SubSystem2Encoder': SubSystem2Encoder,
        'MasterFusionBlock': MasterFusionBlock
    }

    # --- Data ---
    _, _, test_ds, split_info = get_datasets(cfg)
    print(f"\nEvaluating on {split_info['test']} test samples\n")

    roc_data = []
    all_metrics = []

    # --- Master Model ---
    if os.path.exists(args.master):
        print('Loading trained master model ...')
        master = tf.keras.models.load_model(args.master, compile=False, custom_objects=custom_objs)
    else:
        print('No trained master checkpoint found -- using random weights.')
        master = build_master_physics_detector(feature_dim, embed_dim)

    m_labels, m_probs = predict_on_dataset(master, test_ds, 'master_logits')
    m_metrics = compute_metrics(m_labels, m_probs, 'Master Fusion (Full)')
    all_metrics.append(m_metrics)
    roc_data.append(('Master Fusion', m_labels, m_probs))

    # --- Sub-System 1 ---
    sub1_ds = test_ds.map(lambda x, y: (
        {'f1_lens_distortion': x['f1_lens_distortion'],
         'f2_motion_blur': x['f2_motion_blur']}, y))

    if os.path.exists(args.sub1):
        sub1 = tf.keras.models.load_model(args.sub1, compile=False, custom_objects=custom_objs)
    else:
        sub1 = build_subsystem1_model(feature_dim, embed_dim)

    s1_labels, s1_probs = predict_on_dataset(sub1, sub1_ds, 'sub1_logits')
    s1_metrics = compute_metrics(s1_labels, s1_probs, 'Sub-System 1 (Hardware Optics)')
    all_metrics.append(s1_metrics)
    roc_data.append(('Sub-System 1', s1_labels, s1_probs))

    # --- Sub-System 2 ---
    sub2_ds = test_ds.map(lambda x, y: (
        {'f3_biomechanics': x['f3_biomechanics'],
         'f4_lighting_sh': x['f4_lighting_sh']}, y))

    if os.path.exists(args.sub2):
        sub2 = tf.keras.models.load_model(args.sub2, compile=False, custom_objects=custom_objs)
    else:
        sub2 = build_subsystem2_model(feature_dim, embed_dim)

    s2_labels, s2_probs = predict_on_dataset(sub2, sub2_ds, 'sub2_logits')
    s2_metrics = compute_metrics(s2_labels, s2_probs, 'Sub-System 2 (Biological Domain)')
    all_metrics.append(s2_metrics)
    roc_data.append(('Sub-System 2', s2_labels, s2_probs))

    # --- Print ablation table ---
    print('\n')
    print_metrics_table(all_metrics)

    # --- Plots ---
    os.makedirs('results', exist_ok=True)
    plot_roc_curves(roc_data, 'results/roc_curves.png')
    plot_confusion_matrix(m_labels, m_probs, 'Master Fusion', 'results/cm_master.png')
    plot_confusion_matrix(s1_labels, s1_probs, 'Sub-System 1', 'results/cm_sub1.png')
    plot_confusion_matrix(s2_labels, s2_probs, 'Sub-System 2', 'results/cm_sub2.png')

    print('\nEvaluation complete.')


if __name__ == '__main__':
    main()
