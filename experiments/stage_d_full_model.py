"""
experiments/stage_d_full_model.py
-----------------------------------
Stage D — Full Master Fusion Model

Architecture:
    F1 → MLP Encoder(128) ─→ E1 ─┐
    F2 → MLP Encoder(128) ─→ E2 ─┤
    F3 → MLP Encoder(128) ─→ E3 ─┤→ [B, 5, 128]
    F4 → MLP Encoder(128) ─→ E4 ─┤     ↓
    F5 → MLP Encoder(128) ─→ E5 ─┘  Cross-Attention
                                         ↓
                                    Gated Fusion
                                         ↓
                                 Transformer Encoder (2L, 4H)
                                         ↓
                                  Global Average Pool
                                         ↓
                                  Projection Head
                                         ↓
                              BCE + Supervised Contrastive Loss
                                         ↓
                                    Real / Fake

Loss:
    L_total = λ1 * BCE + λ2 * SupCon
    λ1 = 1.0, λ2 = 0.1 (configurable)

Usage:
    python experiments/stage_d_full_model.py
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
CHECKPOINT_BASE = os.path.join('checkpoints', 'deepfakes', 'stage_d')

EMBED_DIM          = 128
NUM_HEADS          = 4
TRANSFORMER_LAYERS = 2
FF_DIM             = 256
DROPOUT            = 0.1
LAMBDA_BCE         = 1.0
LAMBDA_CONTRASTIVE = 0.1
EPOCHS             = 80
BATCH_SIZE         = 64
PATIENCE           = 12
SEED               = 42
TEMPERATURE        = 0.07   # SupCon temperature
# ─────────────────────────────────────────────────────────────────────────────

tf.random.set_seed(SEED)
np.random.seed(SEED)


# ─── Transformer Block ────────────────────────────────────────────────────────

class TransformerEncoderBlock(tf.keras.layers.Layer):
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
        ffn_out  = self.ffn(x)
        ffn_out  = self.drop2(ffn_out, training=training)
        return self.ln2(x + ffn_out)


# ─── Cross-Attention Layer ────────────────────────────────────────────────────

class CrossAttentionBlock(tf.keras.layers.Layer):
    """
    Bidirectional cross-attention between Sub-System 1 (camera) and Sub-System 2 (bio).
    Sub-System 1 tokens: [E1, E2, E5]  (lens, motion, fft)
    Sub-System 2 tokens: [E3, E4]      (biomechanics, lighting)
    """
    def __init__(self, embed_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.ca1  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name='ca_1_attends_2')
        self.ca2  = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim // num_heads, name='ca_2_attends_1')
        self.ln1  = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.ln2  = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, ss1_tokens, ss2_tokens, training=False):
        # SS1 attends to SS2
        ss1_out = self.ca1(query=ss1_tokens, key=ss2_tokens, value=ss2_tokens, training=training)
        ss1_out = self.ln1(ss1_tokens + ss1_out)
        # SS2 attends to SS1
        ss2_out = self.ca2(query=ss2_tokens, key=ss1_tokens, value=ss1_tokens, training=training)
        ss2_out = self.ln2(ss2_tokens + ss2_out)
        return ss1_out, ss2_out


# ─── Gated Fusion Layer ───────────────────────────────────────────────────────

class GatedFusion(tf.keras.layers.Layer):
    """
    Learnable gating that determines how much SS1 vs SS2 information flows through.
    g = sigmoid(W * [SS1_pool; SS2_pool])
    fused = g * SS1_pool + (1 - g) * SS2_pool
    """
    def __init__(self, embed_dim, **kwargs):
        super().__init__(**kwargs)
        self.gate_fc = tf.keras.layers.Dense(embed_dim, activation='sigmoid', name='gate')

    def call(self, ss1_pool, ss2_pool):
        combined = tf.concat([ss1_pool, ss2_pool], axis=-1)
        gate = self.gate_fc(combined)
        return gate * ss1_pool + (1.0 - gate) * ss2_pool


# ─── Supervised Contrastive Loss ─────────────────────────────────────────────

def supervised_contrastive_loss(embeddings, labels, temperature=0.07):
    """
    Supervised Contrastive Loss (Khosla et al., 2020).
    Pulls same-class embeddings together, pushes different-class apart.
    """
    embeddings = tf.math.l2_normalize(embeddings, axis=1)
    similarity = tf.matmul(embeddings, embeddings, transpose_b=True) / temperature

    batch_size = tf.shape(labels)[0]
    labels = tf.cast(labels, tf.float32)
    labels_col = tf.reshape(labels, [-1, 1])
    labels_row = tf.reshape(labels, [1, -1])
    pos_mask = tf.equal(labels_col, labels_row)
    pos_mask = tf.cast(pos_mask, tf.float32)

    # Remove self from mask
    eye = tf.eye(batch_size)
    pos_mask = pos_mask - eye

    # Log-sum-exp denominator (all negatives)
    neg_mask = 1.0 - tf.eye(batch_size)
    neg_sim  = similarity * neg_mask
    log_denom = tf.reduce_logsumexp(neg_sim, axis=1, keepdims=True)

    log_prob = similarity - log_denom
    # Mean over positives
    n_pos = tf.reduce_sum(pos_mask, axis=1)
    loss = -tf.reduce_sum(log_prob * pos_mask, axis=1) / (n_pos + 1e-8)
    return tf.reduce_mean(loss)


# ─── Full Model ───────────────────────────────────────────────────────────────

def build_full_model(feature_keys, input_dim=64, embed_dim=128,
                     num_heads=4, transformer_layers=2, ff_dim=256, dropout=0.1):
    """
    Full Stage D model with cross-attention, gated fusion, and transformer.
    Returns (model_for_classification, embedding_model).
    """
    # SS1 = camera physics: f1, f2, f5
    # SS2 = biology:        f3, f4
    ss1_keys = [k for k in feature_keys if k in ('f1', 'f2', 'f5')]
    ss2_keys = [k for k in feature_keys if k in ('f3', 'f4')]

    inputs = {}
    encoders = {}
    for fk in feature_keys:
        inputs[fk] = tf.keras.Input(shape=(input_dim,), name=f'input_{fk}')
        encoders[fk] = build_mlp_encoder(input_dim, embed_dim, name=f'enc_{fk}')

    # ── Embed each feature ──────────────────────────────────────────────────
    embeddings = {fk: encoders[fk](inputs[fk]) for fk in feature_keys}   # each (B, embed_dim)

    # ── Sub-system token sequences ──────────────────────────────────────────
    def to_tokens(keys):
        toks = [tf.keras.layers.Reshape((1, embed_dim))(embeddings[k]) for k in keys]
        return tf.keras.layers.Concatenate(axis=1)(toks) if len(toks) > 1 else toks[0]

    ss1_tokens = to_tokens(ss1_keys) if ss1_keys else None   # (B, len(ss1), D)
    ss2_tokens = to_tokens(ss2_keys) if ss2_keys else None   # (B, len(ss2), D)

    # ── Cross-Attention (if both sub-systems present) ────────────────────────
    if ss1_tokens is not None and ss2_tokens is not None:
        ca_block = CrossAttentionBlock(embed_dim, num_heads, name='cross_attention')
        ss1_tokens, ss2_tokens = ca_block(ss1_tokens, ss2_tokens)

        # Pool each sub-system
        ss1_pool = tf.keras.layers.GlobalAveragePooling1D(name='ss1_pool')(ss1_tokens)  # (B, D)
        ss2_pool = tf.keras.layers.GlobalAveragePooling1D(name='ss2_pool')(ss2_tokens)  # (B, D)

        # ── Gated Fusion ─────────────────────────────────────────────────────
        fused = GatedFusion(embed_dim, name='gated_fusion')(ss1_pool, ss2_pool)  # (B, D)
        fused = tf.keras.layers.Reshape((1, embed_dim))(fused)  # (B, 1, D) as extra token

        # Concat all tokens for transformer
        all_tokens = tf.keras.layers.Concatenate(axis=1)([ss1_tokens, ss2_tokens, fused])
    else:
        # Fallback: stack all features as tokens
        all_keys_present = list(embeddings.keys())
        all_tokens = to_tokens(all_keys_present)

    # ── Transformer Encoder ──────────────────────────────────────────────────
    x = all_tokens
    for i in range(transformer_layers):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim,
                                    dropout, name=f'transformer_{i}')(x)

    # ── Global embedding ─────────────────────────────────────────────────────
    global_emb = tf.keras.layers.GlobalAveragePooling1D(name='global_pool')(x)   # (B, D)

    # ── Projection Head ──────────────────────────────────────────────────────
    proj = tf.keras.layers.Dense(embed_dim, activation='gelu', name='proj_fc')(global_emb)
    proj = tf.keras.layers.Dropout(dropout, name='proj_drop')(proj)
    proj = tf.keras.layers.LayerNormalization(name='proj_ln')(proj)

    # ── Classifier output ─────────────────────────────────────────────────────
    logit = tf.keras.layers.Dense(1, activation='sigmoid', name='classifier')(proj)

    # Two models: one for training (has both proj and logit), one embedding-only for SupCon
    full_model = tf.keras.Model(
        inputs=list(inputs.values()),
        outputs={'logit': logit, 'embedding': proj},
        name='StageDFullModel',
    )
    return full_model, list(inputs.keys())


class FullModelTrainer:
    """Custom training loop for Stage D (BCE + SupCon)."""

    def __init__(self, model, input_order, lambda_bce=1.0, lambda_con=0.1, temperature=0.07):
        self.model        = model
        self.input_order  = input_order
        self.lambda_bce   = lambda_bce
        self.lambda_con   = lambda_con
        self.temperature  = temperature
        self.optimizer    = tf.keras.optimizers.Adam(1e-4)
        self.bce_loss_fn  = tf.keras.losses.BinaryCrossentropy()

    @tf.function
    def train_step(self, x_batch, y_batch):
        with tf.GradientTape() as tape:
            outputs = self.model(x_batch, training=True)
            logits  = outputs['logit']
            embeds  = outputs['embedding']
            y_float = tf.cast(y_batch, tf.float32)
            bce     = self.bce_loss_fn(y_float, logits)
            con     = supervised_contrastive_loss(embeds, y_batch, self.temperature)
            loss    = self.lambda_bce * bce + self.lambda_con * con
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(logits > 0.5, tf.int32), tf.cast(y_batch, tf.int32)), tf.float32))
        return {'loss': loss, 'bce': bce, 'con': con, 'acc': acc}

    @tf.function
    def val_step(self, x_batch, y_batch):
        outputs = self.model(x_batch, training=False)
        logits  = outputs['logit']
        y_float = tf.cast(y_batch, tf.float32)
        bce     = self.bce_loss_fn(y_float, logits)
        acc = tf.reduce_mean(tf.cast(
            tf.equal(tf.cast(logits > 0.5, tf.int32), tf.cast(y_batch, tf.int32)), tf.float32))
        return {'val_loss': bce, 'val_acc': acc}

    def train(self, X_tr, y_tr, X_va, y_va, epochs, batch_size, patience, ckpt_path):
        N = len(y_tr)
        best_val_loss = np.inf
        patience_count = 0
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            # Shuffle
            idx = np.random.permutation(N)
            epoch_metrics = {'loss': [], 'acc': []}

            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                x_batch = [X_tr[k][batch_idx] for k in self.input_order]
                y_batch = y_tr[batch_idx]
                m = self.train_step(x_batch, y_batch)
                epoch_metrics['loss'].append(float(m['loss']))
                epoch_metrics['acc'].append(float(m['acc']))

            # Validation
            val_metrics = {'val_loss': [], 'val_acc': []}
            Nv = len(y_va)
            for start in range(0, Nv, batch_size):
                x_vb = [X_va[k][start:start + batch_size] for k in self.input_order]
                y_vb = y_va[start:start + batch_size]
                m = self.val_step(x_vb, y_vb)
                val_metrics['val_loss'].append(float(m['val_loss']))
                val_metrics['val_acc'].append(float(m['val_acc']))

            tr_loss = np.mean(epoch_metrics['loss'])
            tr_acc  = np.mean(epoch_metrics['acc'])
            va_loss = np.mean(val_metrics['val_loss'])
            va_acc  = np.mean(val_metrics['val_acc'])

            history['loss'].append(tr_loss)
            history['acc'].append(tr_acc)
            history['val_loss'].append(va_loss)
            history['val_acc'].append(va_acc)

            print(f"Epoch {epoch:3d}/{epochs} — loss={tr_loss:.4f} acc={tr_acc:.4f} "
                  f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_count = 0
                self.model.save(ckpt_path)
                print(f"   ✓ Saved best model (val_loss={va_loss:.4f})")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

        return history, best_val_loss


def main():
    for split in ['split_train.csv', 'split_val.csv', 'split_test.csv']:
        if not os.path.exists(os.path.join(SPLIT_DIR, split)):
            print(f"ERROR: {split} not found. Run prepare_deepfakes_split.py first.")
            sys.exit(1)

    feature_keys = FEATURE_KEYS
    print(f"\n{'#'*60}")
    print(f"  Stage D — Full Master Fusion Model")
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
    model, input_order = build_full_model(
        feature_keys, input_dim, EMBED_DIM, NUM_HEADS,
        TRANSFORMER_LAYERS, FF_DIM, DROPOUT
    )
    model.summary()

    # ── Train with custom loop ─────────────────────────────────────────────────
    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_BASE, 'best_model.keras')

    trainer = FullModelTrainer(
        model, input_order,
        lambda_bce=LAMBDA_BCE,
        lambda_con=LAMBDA_CONTRASTIVE,
        temperature=TEMPERATURE,
    )
    history, best_val_loss = trainer.train(
        X_tr_n, y_tr, X_va_n, y_va,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        patience=PATIENCE, ckpt_path=ckpt_path,
    )

    # ── Load best weights & evaluate ──────────────────────────────────────────
    best_model = tf.keras.models.load_model(
        ckpt_path,
        custom_objects={'TransformerEncoderBlock': TransformerEncoderBlock,
                        'CrossAttentionBlock': CrossAttentionBlock,
                        'GatedFusion': GatedFusion},
    )
    test_X = [X_te_n[k] for k in input_order]
    outputs = best_model.predict(test_X, verbose=0)
    y_pred_proba = outputs['logit'].flatten()

    metrics = compute_all_metrics(y_te, y_pred_proba)
    metrics['stage']              = 'D'
    metrics['architecture']       = 'CrossAttention_GatedFusion_Transformer_SupCon'
    metrics['features']           = feature_keys
    metrics['embed_dim']          = EMBED_DIM
    metrics['num_heads']          = NUM_HEADS
    metrics['transformer_layers'] = TRANSFORMER_LAYERS
    metrics['lambda_bce']         = LAMBDA_BCE
    metrics['lambda_contrastive'] = LAMBDA_CONTRASTIVE
    metrics['train_samples']      = int(len(y_tr))
    metrics['val_samples']        = int(len(y_va))
    metrics['test_samples']       = int(len(y_te))
    metrics['best_val_loss']      = float(best_val_loss)

    print_metrics("Stage D — Full Model TEST RESULTS", metrics)

    # ── Save artifacts ─────────────────────────────────────────────────────────
    scaler.save(os.path.join(CHECKPOINT_BASE, 'scaler.pkl'))
    save_metrics(metrics, os.path.join(CHECKPOINT_BASE, 'metrics.json'))
    with open(os.path.join(CHECKPOINT_BASE, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    print(f"  Saved → {CHECKPOINT_BASE}/")


if __name__ == '__main__':
    main()
