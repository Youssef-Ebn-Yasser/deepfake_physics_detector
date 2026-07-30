"""
train_subsystem1.py
-------------------
Independent training loop for Sub-System 1 (Hardware Optics).
Trains on lens-distortion (F1) and motion-blur (F2) features only.

Usage:
    python pipelines/train_subsystem1.py
    python pipelines/train_subsystem1.py --config config/default_config.yaml
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
import tensorflow as tf
from data.dataset_loader import get_datasets
from models.subsystem_1 import build_subsystem1_model


def make_sub1_dataset(dataset):
    """Keep only f1 & f2 inputs from the full dataset."""
    return dataset.map(
        lambda x, y: (
            {
                'f1_lens_distortion': x['f1_lens_distortion'],
                'f2_motion_blur': x['f2_motion_blur'],
            },
            y,
        )
    )


def main():
    parser = argparse.ArgumentParser(description='Train Sub-System 1')
    parser.add_argument('--config', default='config/default_config.yaml')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim = cfg['model']['embed_dim']
    epochs = cfg['training']['subsystem_epochs']
    lr = cfg['training']['subsystem_lr']

    # --- Data ---
    train_ds, val_ds, _, split_info = get_datasets(cfg)
    train_ds = make_sub1_dataset(train_ds)
    val_ds = make_sub1_dataset(val_ds)

    # --- Model ---
    model = build_subsystem1_model(feature_dim, embed_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'sub1_logits': tf.keras.losses.BinaryCrossentropy(from_logits=True)},
        metrics={'sub1_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]},
    )

    model.summary()

    # --- Callbacks ---
    os.makedirs('checkpoints', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/subsystem1_best.keras',
            monitor='val_sub1_logits_acc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sub1_logits_acc',
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

    # --- Train ---
    print(f"\n{'='*60}")
    print(f"  Training Sub-System 1  (Hardware Optics)")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Train: {split_info['train']}  Val: {split_info['val']}")
    print(f"{'='*60}\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save('checkpoints/subsystem1_final.keras')
    print("\nSub-System 1 training complete. Saved to checkpoints/")


if __name__ == '__main__':
    main()
