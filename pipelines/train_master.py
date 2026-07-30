"""
train_master.py
---------------
End-to-end Master Model training with two-stage strategy:

    Stage 1: Freeze sub-encoders, train master cross-attention (5 epochs)
    Stage 2: Unfreeze all layers, fine-tune end-to-end (15 epochs)

Optionally loads pre-trained sub-system weights before Stage 1.

Usage:
    python pipelines/train_master.py
    python pipelines/train_master.py --config config/default_config.yaml
    python pipelines/train_master.py --sub1 checkpoints/subsystem1_best.keras \
                                      --sub2 checkpoints/subsystem2_best.keras
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import numpy as np
import tensorflow as tf
from data.dataset_loader import get_datasets
from models.master_fusion import build_master_physics_detector


# -----------------------------------------------------------------------
# Custom loss for multi-output training
# -----------------------------------------------------------------------

def make_losses(aux_weight=0.2):
    """Return a loss dict and loss-weight dict for model.compile()."""
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    losses = {
        'master_logits': bce,
        'sub1_logits': bce,
        'sub2_logits': bce,
        'sub3_logits': bce,
    }
    loss_weights = {
        'master_logits': 1.0,
        'sub1_logits': aux_weight,
        'sub2_logits': aux_weight,
        'sub3_logits': aux_weight,
    }
    return losses, loss_weights


def make_labels_dict(dataset):
    """
    Replicate the single label into a dict matching the model's 4 outputs.
    """
    return dataset.map(
        lambda x, y: (
            x,
            {
                'master_logits': y,
                'sub1_logits': y,
                'sub2_logits': y,
                'sub3_logits': y,
            },
        )
    )


# -----------------------------------------------------------------------
# Transfer pre-trained sub-system weights SAFELY
# -----------------------------------------------------------------------

# def transfer_subsystem_weights(master_model, sub1_path=None, sub2_path=None):
#     """
#     Load pre-trained sub-system checkpoints and copy their weights into 
#     the master model's corresponding sub-encoder layers without needing 
#     to deserialize custom Keras classes.
#     """
#     if sub1_path and os.path.exists(sub1_path):
#         print(f"Loading Sub-System 1 weights from: {sub1_path}")
#         try:
#             # First attempt layer weight transfer directly
#             dst = master_model.get_layer('subsystem1_hardware')
#             dst.load_weights(sub1_path, skip_mismatch=True)
#             print("  -> Sub-System 1 encoder weights transferred.")
#         except Exception as e:
#             print(f"  [!] Direct layer loading skipped ({e}). Falling back to full model load...")
#             sub1 = tf.keras.models.load_model(sub1_path, compile=False)
#             src = sub1.get_layer('subsystem1_hardware')
#             dst = master_model.get_layer('subsystem1_hardware')
#             dst.set_weights(src.get_weights())
#             print("  -> Sub-System 1 encoder weights transferred via full model fallback.")

#     if sub2_path and os.path.exists(sub2_path):
#         print(f"Loading Sub-System 2 weights from: {sub2_path}")
#         try:
#             dst = master_model.get_layer('subsystem2_biological')
#             dst.load_weights(sub2_path, skip_mismatch=True)
#             print("  -> Sub-System 2 encoder weights transferred.")
#         except Exception as e:
#             print(f"  [!] Direct layer loading skipped ({e}). Falling back to full model load...")
#             sub2 = tf.keras.models.load_model(sub2_path, compile=False)
#             src = sub2.get_layer('subsystem2_biological')
#             dst = master_model.get_layer('subsystem2_biological')
#             dst.set_weights(src.get_weights())
#             print("  -> Sub-System 2 encoder weights transferred via full model fallback.")
def transfer_subsystem_weights(master_model, sub1_path=None, sub2_path=None, sub3_path=None):
    """
    Loads pre-trained standalone sub-system models and transfers 
    their encoder layer weights directly into the master model.
    """
    if sub1_path and os.path.exists(sub1_path):
        print(f"Loading Sub-System 1 weights from: {sub1_path}")
        from models.master_fusion import SubSystem1Encoder
        sub1 = tf.keras.models.load_model(sub1_path, compile=False, custom_objects={'SubSystem1Encoder': SubSystem1Encoder})
        src = sub1.get_layer('subsystem1_hardware')
        dst = master_model.get_layer('subsystem1_hardware')
        dst.set_weights(src.get_weights())
        print("  -> Sub-System 1 encoder weights transferred successfully.")

    if sub2_path and os.path.exists(sub2_path):
        print(f"Loading Sub-System 2 weights from: {sub2_path}")
        from models.master_fusion import SubSystem2Encoder
        sub2 = tf.keras.models.load_model(sub2_path, compile=False, custom_objects={'SubSystem2Encoder': SubSystem2Encoder})
        src = sub2.get_layer('subsystem2_biological')
        dst = master_model.get_layer('subsystem2_biological')
        dst.set_weights(src.get_weights())
        print("  -> Sub-System 2 encoder weights transferred successfully.")

    if sub3_path and os.path.exists(sub3_path):
        print(f"Loading Sub-System 3 weights from: {sub3_path}")
        from subsystem3.encoder3 import SubSystem3Encoder
        sub3 = tf.keras.models.load_model(sub3_path, compile=False, custom_objects={'SubSystem3Encoder': SubSystem3Encoder})
        src = sub3.get_layer('subsystem3_physics')
        dst = master_model.get_layer('subsystem3_physics')
        dst.set_weights(src.get_weights())
        print("  -> Sub-System 3 encoder weights transferred successfully.")

# -----------------------------------------------------------------------
# Main training loop
# -----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train Master Fusion Model')
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--sub1', default='checkpoints/subsystem1_best.keras',
                        help='Path to pre-trained Sub-System 1 checkpoint')
    parser.add_argument('--sub2', default='checkpoints/subsystem2_best.keras',
                        help='Path to pre-trained Sub-System 2 checkpoint')
    parser.add_argument('--sub3', default='checkpoints/subsystem3_best.keras',
                        help='Path to pre-trained Sub-System 3 checkpoint')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim = cfg['model']['embed_dim']
    aux_weight = cfg['model']['master_aux_weight']
    batch_size = cfg['training']['batch_size']
    lr_stage1 = cfg['training']['learning_rate_stage1']
    lr_stage2 = cfg['training']['learning_rate_stage2']
    epochs_s1 = cfg['training']['epochs_stage1']
    epochs_s2 = cfg['training']['epochs_stage2']

    # --- Data ---
    train_ds, val_ds, test_ds, split_info = get_datasets(cfg)
    train_ds = make_labels_dict(train_ds)
    val_ds = make_labels_dict(val_ds)
    test_ds_eval = make_labels_dict(test_ds)  # for final evaluation

    # --- Model ---
    model = build_master_physics_detector(feature_dim, embed_dim)
    transfer_subsystem_weights(model, args.sub1, args.sub2, args.sub3)

    losses, loss_weights = make_losses(aux_weight)

    os.makedirs('checkpoints', exist_ok=True)

    # ===================================================================
    # STAGE 1: Freeze sub-encoders, train master cross-attention
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 1: Train Master Attention  (sub-encoders frozen)")
    print(f"  Epochs: {epochs_s1}  |  LR: {lr_stage1}")
    print(f"  Train: {split_info['train']}  |  Val: {split_info['val']}")
    print(f"{'='*60}\n")

    model.get_layer('subsystem1_hardware').trainable = False
    model.get_layer('subsystem2_biological').trainable = False
    model.get_layer('subsystem3_physics').trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage1),
        loss=losses,
        loss_weights=loss_weights,
        metrics={'master_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]},
    )
    model.summary()

    stage1_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/master_stage1_best.keras',
            monitor='val_master_logits_acc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_s1,
        callbacks=stage1_callbacks,
    )

    # ===================================================================
    # STAGE 2: Unfreeze everything, fine-tune end-to-end
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  STAGE 2: End-to-End Fine-Tuning  (all layers)")
    print(f"  Epochs: {epochs_s2}  |  LR: {lr_stage2}")
    print(f"{'='*60}\n")

    model.get_layer('subsystem1_hardware').trainable = True
    model.get_layer('subsystem2_biological').trainable = True
    model.get_layer('subsystem3_physics').trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr_stage2),
        loss=losses,
        loss_weights=loss_weights,
        metrics={'master_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]},
    )

    stage2_callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/master_final_best.keras',
            monitor='val_master_logits_acc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_master_logits_acc',
            mode='max',
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_s2,
        callbacks=stage2_callbacks,
    )

    # ===================================================================
    # Final evaluation on test set
    # ===================================================================
    print(f"\n{'='*60}")
    print(f"  Final Evaluation on Test Set  ({split_info['test']} samples)")
    print(f"{'='*60}\n")

    results = model.evaluate(test_ds_eval, return_dict=True)
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    model.save('checkpoints/master_final.keras')
    print("\nMaster model saved to checkpoints/master_final.keras")


if __name__ == '__main__':
    main()