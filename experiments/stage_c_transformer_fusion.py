"""
experiments/stage_c_transformer_fusion.py
------------------------------------------
Stage C — Transformer Fusion

Architecture:
    F1 → MLP Encoder(128) ─→ Token 1 ─┐
    F2 → MLP Encoder(128) ─→ Token 2 ─┤
    F3 → MLP Encoder(128) ─→ Token 3 ─┤→ [B, 5, 128]
    F4 → MLP Encoder(128) ─→ Token 4 ─┤     ↓
    F5 → MLP Encoder(128) ─→ Token 5 ─┘  Transformer Encoder (2L, 4H)
                                              ↓
                                       Global Average Pool
                                              ↓
                                       Dense(64) → GELU → Dense(1) → Sigmoid

Comparison vs Stage B (Simple Fusion):
  Does the Transformer improve over simple concatenation?

Usage:
    python experiments/stage_c_transformer_fusion.py
"""

import os
import sys
import json
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from experiments.shared_utils import (
    FEATURE_KEYS, FEATURE_NAMES,
    load_split_csv, load_features,
    FeatureScaler, compute_all_metrics, print_metrics, save_metrics,
    build_mlp_encoder,
)

# ─────────────────────────────────────────────────────────────────────────────
FEATURE_DIR     = os.path.join('DataSets', 'deepfakes_feature')
SPLIT_DIR       = FEATURE_DIR
CHECKPOINT_BASE = os.path.join('checkpoints', 'deepfakes', 'stage_c')

EMBED_DIM       = 128
NUM_HEADS       = 4
TRANSFORMER_LAYERS = 2
FF_DIM          = 256    # Transformer feed-forward inner dim
TRANSFORMER_DROPOUT = 0.1
EPOCHS          = 60
BATCH_SIZE      = 64
PATIENCE        = 10
SEED            = 42
# ─────────────────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


class TransformerEncoderBlock(tf.keras.layers.Layer):
    """Single Transformer encoder layer (Multi-Head Self-Attention + FFN)."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads)
        self.ffn   = tf.keras.Sequential([
            tf.keras.layers.Dense(ff_dim, activation='gelu'),
            tf.keras.layers.Dense(embed_dim),
        ])
        self.ln1   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2   = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout_rate)
        self.drop2 = tf.keras.layers.Dropout(dropout_rate)

    def call(self, x, training=False):
        attn_out = self.attn(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        x = self.ln1(x + attn_out)

        ffn_out = self.ffn(x)
        ffn_out = self.drop2(ffn_out, training=training)
        return self.ln2(x + ffn_out)

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.attn.key_dim * self.attn.num_heads,
            'num_heads': self.attn.num_heads,
            'ff_dim':    self.ffn.layers[0].units,
            'dropout_rate': self.drop1.rate,
        })
        return config


def build_transformer_fusion_model(feature_keys, input_dim=64,
                                   embed_dim=128, num_heads=4,
                                   transformer_layers=2, ff_dim=256,
                                   dropout=0.1):
    """
    Treat each feature embedding as a sequence token → Transformer.
    Returns a Keras Model.
    """
    inputs = []
    token_list = []

    for fk in feature_keys:
        inp = tf.keras.Input(shape=(input_dim,), name=f'input_{fk}')
        inputs.append(inp)
        encoder = build_mlp_encoder(input_dim, embed_dim, name=f'enc_{fk}')
        tok = encoder(inp)                              # (B, embed_dim)
        tok = tf.keras.layers.Reshape((1, embed_dim))(tok)   # (B, 1, embed_dim)
        token_list.append(tok)

    # Stack: (B, n_features, embed_dim)
    tokens = tf.keras.layers.Concatenate(axis=1)(token_list)   # (B, 5, embed_dim)

    # Transformer blocks
    x = tokens
    for i in range(transformer_layers):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim,
                                    dropout, name=f'transformer_{i}')(x)

    # Global average pooling over token dimension
    x = tf.keras.layers.GlobalAveragePooling1D(name='gap')(x)  # (B, embed_dim)

    # Classification head
    x = tf.keras.layers.Dense(64, activation='gelu', name='head_fc')(x)
    x = tf.keras.layers.Dropout(dropout, name='head_drop')(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(x)

    model = tf.keras.Model(inputs=inputs, outputs=out,
                           name='TransformerFusion')
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    for split in ['split_train.csv', 'split_val.csv', 'split_test.csv']:
        if not os.path.exists(os.path.join(SPLIT_DIR, split)):
            print(f"ERROR: {split} not found. Run prepare_deepfakes_split.py first.")
            sys.exit(1)

    feature_keys = FEATURE_KEYS
    group_name   = '+'.join([k.upper() for k in feature_keys])
    print(f"\n{'#'*60}")
    print(f"  Stage C — Transformer Fusion: {group_name}")
    print(f"{'#'*60}")

    # ── Load data ──────────────────────────────────────────────────────────────
    train_entries = load_split_csv(os.path.join(SPLIT_DIR, 'split_train.csv'))
    val_entries   = load_split_csv(os.path.join(SPLIT_DIR, 'split_val.csv'))
    test_entries  = load_split_csv(os.path.join(SPLIT_DIR, 'split_test.csv'))

    X_tr_raw, y_tr = load_features(train_entries, FEATURE_DIR, feature_keys)
    X_va_raw, y_va = load_features(val_entries,   FEATURE_DIR, feature_keys)
    X_te_raw, y_te = load_features(test_entries,  FEATURE_DIR, feature_keys)

    # ── Normalize (train-only) ─────────────────────────────────────────────────
    scaler = FeatureScaler()
    X_tr_n = scaler.fit_transform(X_tr_raw)
    X_va_n = scaler.transform(X_va_raw)
    X_te_n = scaler.transform(X_te_raw)

    input_dim = X_tr_n[feature_keys[0]].shape[1]
    print(f"  Train={len(y_tr)} | Val={len(y_va)} | Test={len(y_te)} | Feature dim={input_dim}")

    # ── Build model ───────────────────────────────────────────────────────────
    model = build_transformer_fusion_model(
        feature_keys, input_dim, EMBED_DIM, NUM_HEADS,
        TRANSFORMER_LAYERS, FF_DIM, TRANSFORMER_DROPOUT
    )
    model.summary()

    # ── Checkpoint & callbacks ─────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_BASE, 'best_model.keras')

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss', save_best_only=True, verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=PATIENCE, restore_best_weights=True, verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0
        ),
    ]

    train_X = [X_tr_n[k] for k in feature_keys]
    val_X   = [X_va_n[k] for k in feature_keys]
    test_X  = [X_te_n[k] for k in feature_keys]

    history = model.fit(
        train_X, y_tr,
        validation_data=(val_X, y_va),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
    )

    # ── Evaluate ───────────────────────────────────────────────────────────────
    y_pred_proba = model.predict(test_X, verbose=0).flatten()
    metrics = compute_all_metrics(y_te, y_pred_proba)
    metrics['stage']         = 'C'
    metrics['architecture']  = 'TransformerFusion'
    metrics['features']      = feature_keys
    metrics['embed_dim']     = EMBED_DIM
    metrics['num_heads']     = NUM_HEADS
    metrics['transformer_layers'] = TRANSFORMER_LAYERS
    metrics['train_samples'] = int(len(y_tr))
    metrics['val_samples']   = int(len(y_va))
    metrics['test_samples']  = int(len(y_te))
    metrics['best_val_loss'] = float(min(history.history['val_loss']))

    print_metrics("Stage C — Transformer Fusion TEST RESULTS", metrics)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    scaler.save(os.path.join(CHECKPOINT_BASE, 'scaler.pkl'))
    save_metrics(metrics, os.path.join(CHECKPOINT_BASE, 'metrics.json'))
    print(f"  Saved → {CHECKPOINT_BASE}/")


if __name__ == '__main__':
    main()
