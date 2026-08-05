"""
train_subsystem2.py
-------------------
Independent training loop for Sub-System 2 (Biological Domain).
Trains on biomechanics (F3) and lighting SH (F4) features only.

Usage:
    python pipelines/train_subsystem2.py
    python pipelines/train_subsystem2.py --config config/default_config.yaml
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import random
import yaml
import tensorflow as tf
from data.dataset_loader import get_strict_datasets, load_manifest, build_tf_dataset
from models.subsystem_2 import build_subsystem2_model

def get_balanced_datasets(cfg):
    features_dir = cfg['dataset']['features_dir']
    batch_size = cfg['training']['batch_size']
    feature_dim = cfg['extractors']['feature_dim']

    entries = load_manifest(features_dir)
    
    original = []
    deepfakes = []
    face2face = []
    other = []
    
    for path, label in entries:
        # Note: path is absolute, so we check for folder names
        if 'original_sequences' in path:
            original.append((path, label))
        elif 'manipulated_sequences_Deepfakes' in path:
            deepfakes.append((path, label))
        elif 'manipulated_sequences_Face2Face' in path:
            face2face.append((path, label))
        else:
            other.append((path, label))
            
    random.seed(cfg['dataset'].get('split_seed', 42))
    random.shuffle(original)
    random.shuffle(deepfakes)
    random.shuffle(face2face)
    
    train_entries = original[:1000] + deepfakes[:1000] + face2face[:1000]
    
    remaining = original[1000:] + deepfakes[1000:] + face2face[1000:] + other
    random.shuffle(remaining)
    
    val_size = int(len(remaining) * 0.5)
    val_entries = remaining[:val_size]
    test_entries = remaining[val_size:]
    
    print(f"Balanced Split: train={len(train_entries)} (1000 Orig, 1000 DF, 1000 F2F), val={len(val_entries)}, test={len(test_entries)}")
    
    train_ds = build_tf_dataset(train_entries, batch_size, shuffle=True, feature_dim=feature_dim, features_dir=features_dir)
    val_ds = build_tf_dataset(val_entries, batch_size, shuffle=False, feature_dim=feature_dim, features_dir=features_dir)
    test_ds = build_tf_dataset(test_entries, batch_size, shuffle=False, feature_dim=feature_dim, features_dir=features_dir)
    
    split_info = {'train': len(train_entries), 'val': len(val_entries), 'test': len(test_entries)}
    return train_ds, val_ds, test_ds, split_info


def make_sub2_dataset(dataset):
    """Keep only f3 & f4 inputs from the full dataset."""
    return dataset.map(
        lambda x, y: (
            {
                'f3_biomechanics': x['f3_biomechanics'],
                'f4_lighting_sh': x['f4_lighting_sh'],
            },
            y,
        )
    )


def main():
    parser = argparse.ArgumentParser(description='Train Sub-System 2')
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--balanced', action='store_true', help='Use balanced subset (1k Orig, 1k DF, 1k F2F) for training')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    feature_dim = cfg['extractors']['feature_dim']
    embed_dim = cfg['model']['embed_dim']
    epochs = cfg['training']['subsystem_epochs']
    lr = cfg['training']['subsystem_lr']

    # --- Data ---
    if args.balanced:
        train_ds, val_ds, _, split_info = get_balanced_datasets(cfg)
    else:
        train_ds, val_ds, _, split_info = get_strict_datasets(cfg)
        
    train_ds = make_sub2_dataset(train_ds)
    val_ds = make_sub2_dataset(val_ds)

    # --- Model ---
    model = build_subsystem2_model(feature_dim, embed_dim)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss={'sub2_logits': tf.keras.losses.BinaryCrossentropy(from_logits=True)},
        metrics={'sub2_logits': [tf.keras.metrics.BinaryAccuracy(name='acc')]},
    )

    model.summary()

    # --- Callbacks ---
    os.makedirs('checkpoints', exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            'checkpoints/subsystem2_best.keras',
            monitor='val_sub2_logits_acc',
            save_best_only=True,
            mode='max',
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_sub2_logits_acc',
            patience=5,
            mode='max',  # <--- ADD THIS LINE HERE
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
    print(f"  Training Sub-System 2  (Biological Domain)")
    print(f"  Epochs: {epochs}  |  LR: {lr}  |  Train: {split_info['train']}  Val: {split_info['val']}")
    print(f"{'='*60}\n")

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
    )

    model.save('checkpoints/subsystem2_final.keras')
    print("\nSub-System 2 training complete. Saved to checkpoints/")


if __name__ == '__main__':
    main()
