"""
train_fusion_1_2_mid.py
------------------------
Trains the Mid-Level Fusion of Sub-System 1 + Sub-System 2.

Two-stage training:
    Stage 1: Freeze both sub-encoders, train only the fusion block + head.
    Stage 2: Unfreeze everything for end-to-end fine-tuning.

Usage:
    python pipelines/train_fusion_1_2_mid.py
    python pipelines/train_fusion_1_2_mid.py --sub1 checkpoints/subsystem1_best.keras
    python pipelines/train_fusion_1_2_mid.py --sub1 ... --sub2 ...
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np
import tensorflow as tf

from data.dataset_loader import get_strict_datasets
from models.fusion_1_2_mid import build_fusion_1_2_mid


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_losses(aux_weight=0.2):
    bce   = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    focal = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True, gamma=2.0)
    losses = {
        'fusion_logits': focal,
        'sub1_logits':   bce,
        'sub2_logits':   bce,
    }
    loss_weights = {
        'fusion_logits': 1.0,
        'sub1_logits':   aux_weight,
        'sub2_logits':   aux_weight,
    }
    return losses, loss_weights


def make_labels_dict(dataset):
    """Project raw (inputs, label) dataset to the multi-output format."""
    return dataset.map(
        lambda x, y: (
            {
                'f1_lens_distortion': x['f1_lens_distortion'],
                'f2_motion_blur':     x['f2_motion_blur'],
                'f3_biomechanics':    x['f3_biomechanics'],
                'f4_lighting_sh':     x['f4_lighting_sh'],
                'f5_frequency_fft':   x['f5_frequency_fft'],
            },
            {
                'fusion_logits': y,
                'sub1_logits':   y,
                'sub2_logits':   y,
            },
        )
    )


def transfer_weights(model, sub1_path=None, sub2_path=None):
    """Load pre-trained encoder weights into the fusion model."""
    from models.master_fusion import SubSystem1Encoder, SubSystem2Encoder

    if sub1_path and os.path.exists(sub1_path):
        print(f"Loading Sub-System 1 weights from: {sub1_path}")
        sub1 = tf.keras.models.load_model(
            sub1_path, compile=False,
            custom_objects={'SubSystem1Encoder': SubSystem1Encoder}
        )
        model.get_layer('subsystem1_hardware').set_weights(
            sub1.get_layer('subsystem1_hardware').get_weights()
        )
        print("  -> Sub-System 1 encoder weights transferred.")

    if sub2_path and os.path.exists(sub2_path):
        print(f"Loading Sub-System 2 weights from: {sub2_path}")
        sub2 = tf.keras.models.load_model(
            sub2_path, compile=False,
            custom_objects={'SubSystem2Encoder': SubSystem2Encoder}
        )
        model.get_layer('subsystem2_biological').set_weights(
            sub2.get_layer('subsystem2_biological').get_weights()
        )
        print("  -> Sub-System 2 encoder weights transferred.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Mid-Level Fusion 1+2')
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--sub1', default='checkpoints/subsystem1_best.keras',
                        help='Pre-trained Sub-System 1 checkpoint')
    parser.add_argument('--sub2', default='checkpoints/subsystem2_best.keras',
                        help='Pre-trained Sub-System 2 checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim   = cfg['model']['embed_dim']
    aux_weight  = cfg['model']['master_aux_weight']
    lr_stage1   = cfg['training']['learning_rate_stage1']
    lr_stage2   = cfg['training']['learning_rate_stage2']
    epochs_s1   = cfg['training']['epochs_stage1']
    epochs_s2   = cfg['training']['epochs_stage2']

    # --- Data ---
    train_ds, val_ds, test_ds, split_info = get_strict_datasets(cfg)
    train_ds    = make_labels_dict(train_ds)
    val_ds      = make_labels_dict(val_ds)
    test_ds_eval= make_labels_dict(test_ds)

    print(f"Split: train={split_info['train']}, val={split_info['val']}, test={split_info['test']}")

    # --- Model ---
    model = build_fusion_1_2_mid(feature_dim, embed_dim)
    model.summary(line_length=110)
    transfer_weights(model, args.sub1, args.sub2)

    losses, loss_weights = make_losses(aux_weight)
    os.makedirs('checkpoints', exist_ok=True)

    # ==========================================================
    # STAGE 1 — Freeze both sub-encoders, train fusion block
    # ==========================================================
    print(f"\n{'='*60}")
    print("  STAGE 1: Mid-Level Fusion Block Training  (encoders frozen)")
    print(f"{'='*60}\n")

    model.get_layer('subsystem1_hardware').trainable  = False
    model.get_layer('subsystem2_biological').trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage1),
        loss=losses,
        loss_weights=loss_weights,
        metrics={'fusion_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]},
    )

    s1_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/fusion12_mid_stage1_best.keras',
            monitor='val_fusion_logits_acc', save_best_only=True, mode='max', verbose=1,
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_s1, callbacks=s1_callbacks)

    # ==========================================================
    # STAGE 2 — Unfreeze everything, end-to-end fine-tuning
    # ==========================================================
    print(f"\n{'='*60}")
    print("  STAGE 2: End-to-End Fine-Tuning  (all layers)")
    print(f"{'='*60}\n")

    model.get_layer('subsystem1_hardware').trainable  = True
    model.get_layer('subsystem2_biological').trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage2),
        loss=losses,
        loss_weights=loss_weights,
        metrics={'fusion_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]},
    )

    s2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/fusion12_mid_final_best.keras',
            monitor='val_fusion_logits_acc', save_best_only=True, mode='max', verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_fusion_logits_acc', mode='max',
            patience=6, restore_best_weights=True, verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1,
        ),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs_s2, callbacks=s2_callbacks)

    # ==========================================================
    # Evaluation
    # ==========================================================
    print(f"\n{'='*60}")
    print(f"  Final Test-Set Evaluation  ({split_info['test']} samples)")
    print(f"{'='*60}\n")

    results = model.evaluate(test_ds_eval, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    # Detailed metrics + confusion matrix
    from sklearn.metrics import (
        accuracy_score, roc_auc_score,
        precision_score, recall_score, f1_score, confusion_matrix,
    )
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    all_probs, all_labels = [], []
    for inputs, labels in test_ds_eval:
        logits = model(inputs, training=False)['fusion_logits']
        all_probs.extend(tf.sigmoid(logits).numpy().ravel())
        all_labels.extend(labels['fusion_logits'].numpy().ravel())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds      = (all_probs >= 0.5).astype(int)

    print("\n--- Mid-Level Fusion 1+2 Metrics ---")
    print(f"Accuracy : {accuracy_score(all_labels, preds):.4f}")
    if len(np.unique(all_labels)) > 1:
        print(f"AUC      : {roc_auc_score(all_labels, all_probs):.4f}")
    print(f"Precision: {precision_score(all_labels, preds, zero_division=0):.4f}")
    print(f"Recall   : {recall_score(all_labels, preds, zero_division=0):.4f}")
    print(f"F1 Score : {f1_score(all_labels, preds, zero_division=0):.4f}")

    # Confusion matrix plot
    if len(np.unique(all_labels)) > 1:
        cm = confusion_matrix(all_labels, preds)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap='Blues')
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(['Real', 'Fake'])
        ax.set_yticklabels(['Real', 'Fake'])
        ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        ax.set_title('Confusion Matrix — Mid-Level Fusion 1+2')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                        color='white' if cm[i, j] > cm.max() / 2 else 'black')
        fig.colorbar(im); fig.tight_layout()
        os.makedirs('results', exist_ok=True)
        fig.savefig('results/cm_fusion12_mid.png', dpi=150)
        plt.close(fig)
        print("Confusion matrix saved to results/cm_fusion12_mid.png")

    model.save('checkpoints/fusion12_mid_final.keras')
    print("Model saved to checkpoints/fusion12_mid_final.keras")


if __name__ == '__main__':
    main()
