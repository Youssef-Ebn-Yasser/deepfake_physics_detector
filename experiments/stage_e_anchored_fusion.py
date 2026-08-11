"""
experiments/stage_e_anchored_fusion.py
--------------------------------------
Stage E — F1-Anchored Stability-Gated Fusion Model

This experiment attempts to fuse F1 (which showed better domain stability
on external datasets but weak internal performance) with powerful but brittle 
features (F4, F5) using a stability-anchored gating mechanism.

Architecture:
- F1 -> Enc_F1 (128) -> aux_logit
- F4/F5 -> Enc_F4/F5 (128)
- Gate computes disagreement: diff = |E1 - proj(E_strong)|
- fusion_input = concat([E1, gate * E_strong])
- fusion_input -> Dense(256) -> main_logit

Loss = BCE(main_logit, y) + 0.4 * BCE(aux_logit, y) + 0.1 * ConsistencyLoss
ConsistencyLoss = relu(|main_logit - aux_logit| - margin)

We train 3 variants:
1. f1 + f4
2. f1 + f5
3. f1 + f4 + f5

Usage:
    python experiments/stage_e_anchored_fusion.py
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
CHECKPOINT_BASE = os.path.join('checkpoints', 'deepfakes', 'stage_e')

EMBED_DIM   = 128
EPOCHS      = 60
BATCH_SIZE  = 64
PATIENCE    = 10
SEED        = 42

LAMBDA_AUX  = 0.4
LAMBDA_CONS = 0.1
MARGIN      = 2.0  # Logit margin for consistency loss

VARIANTS = [
    {
        'name': 'F1+F4',
        'dir_name': 'stage_e_f1f4',
        'feature_keys': ['f1', 'f4']
    },
    {
        'name': 'F1+F5',
        'dir_name': 'stage_e_f1f5',
        'feature_keys': ['f1', 'f5']
    },
    {
        'name': 'F1+F4+F5',
        'dir_name': 'stage_e_f1f4f5',
        'feature_keys': ['f1', 'f4', 'f5']
    }
]

tf.random.set_seed(SEED)
np.random.seed(SEED)
# ─────────────────────────────────────────────────────────────────────────────


class AnchoredFusionModel(tf.keras.Model):
    def __init__(self, feature_keys, input_dim=64, embed_dim=128, **kwargs):
        super().__init__(**kwargs)
        self.feature_keys = feature_keys
        self.strong_keys = [k for k in feature_keys if k != 'f1']
        
        self.encoders = {}
        for fk in feature_keys:
            self.encoders[fk] = build_mlp_encoder(input_dim, embed_dim, name=f'enc_{fk}')
            
        # Aux head for F1
        self.aux_head = tf.keras.layers.Dense(1, name='aux_classifier')
        
        # Projections to align strong features to F1 space for diff computation
        self.projections = {}
        for fk in self.strong_keys:
            self.projections[fk] = tf.keras.layers.Dense(embed_dim, name=f'proj_{fk}')
            
        # Gate network
        self.gate_fc1 = tf.keras.layers.Dense(64, activation='gelu', name='gate_fc1')
        self.gate_out = tf.keras.layers.Dense(1, activation='sigmoid', name='gate_out')
        
        # Main fusion head
        self.fusion_fc1 = tf.keras.layers.Dense(256, activation='gelu', name='fusion_fc1')
        self.fusion_drop = tf.keras.layers.Dropout(0.2, name='fusion_drop')
        self.main_head = tf.keras.layers.Dense(1, name='main_classifier')

    def call(self, inputs, training=False):
        # inputs is a dict of feature arrays
        embeddings = {fk: self.encoders[fk](inputs[fk], training=training) for fk in self.feature_keys}
        
        e1 = embeddings['f1']
        aux_logit = self.aux_head(e1)
        
        if not self.strong_keys:
            return {'main_logit': aux_logit, 'aux_logit': aux_logit, 'gate': tf.ones_like(aux_logit)}
            
        # Compute differences
        diffs = []
        strong_embs = []
        for fk in self.strong_keys:
            e_strong = embeddings[fk]
            strong_embs.append(e_strong)
            p_strong = self.projections[fk](e_strong)
            diff = tf.abs(e1 - p_strong)
            diffs.append(diff)
            
        gate_input = tf.concat([e1] + diffs, axis=-1)
        gate_hidden = self.gate_fc1(gate_input)
        gate_score = self.gate_out(gate_hidden)  # (B, 1)
        
        strong_concat = tf.concat(strong_embs, axis=-1)
        gated_strong = gate_score * strong_concat
        
        fusion_input = tf.concat([e1, gated_strong], axis=-1)
        f_hidden = self.fusion_fc1(fusion_input)
        f_hidden = self.fusion_drop(f_hidden, training=training)
        main_logit = self.main_head(f_hidden)
        
        # Return logits (apply sigmoid in loss or inference wrapper)
        return {
            'main_logit': main_logit,
            'aux_logit': aux_logit,
            'gate': gate_score
        }


class AnchoredTrainer:
    def __init__(self, model, input_order, lambda_aux=0.4, lambda_cons=0.1, margin=2.0):
        self.model = model
        self.input_order = input_order
        self.lambda_aux = lambda_aux
        self.lambda_cons = lambda_cons
        self.margin = margin
        self.optimizer = tf.keras.optimizers.Adam(1e-4)
        self.bce_loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    @tf.function
    def train_step(self, x_batch_dict, y_batch):
        y_float = tf.cast(y_batch, tf.float32)
        y_float_expanded = tf.expand_dims(y_float, -1)
        
        with tf.GradientTape() as tape:
            outputs = self.model(x_batch_dict, training=True)
            main_logit = outputs['main_logit']
            aux_logit = outputs['aux_logit']
            
            main_bce = self.bce_loss_fn(y_float_expanded, main_logit)
            aux_bce = self.bce_loss_fn(y_float_expanded, aux_logit)
            
            # Consistency: penalize if main logit diverges too far from aux logit
            cons_loss = tf.reduce_mean(tf.nn.relu(tf.abs(main_logit - aux_logit) - self.margin))
            
            loss = main_bce + self.lambda_aux * aux_bce + self.lambda_cons * cons_loss
            
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        main_prob = tf.sigmoid(main_logit)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(main_prob > 0.5, tf.int32), tf.expand_dims(tf.cast(y_batch, tf.int32), -1)), tf.float32))
        return {'loss': loss, 'main_bce': main_bce, 'aux_bce': aux_bce, 'cons': cons_loss, 'acc': acc}

    @tf.function
    def val_step(self, x_batch_dict, y_batch):
        y_float = tf.cast(y_batch, tf.float32)
        y_float_expanded = tf.expand_dims(y_float, -1)
        
        outputs = self.model(x_batch_dict, training=False)
        main_logit = outputs['main_logit']
        main_bce = self.bce_loss_fn(y_float_expanded, main_logit)
        
        main_prob = tf.sigmoid(main_logit)
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(main_prob > 0.5, tf.int32), tf.expand_dims(tf.cast(y_batch, tf.int32), -1)), tf.float32))
        return {'val_loss': main_bce, 'val_acc': acc}

    def train(self, X_tr, y_tr, X_va, y_va, epochs, batch_size, patience, ckpt_path):
        N = len(y_tr)
        best_val_loss = np.inf
        patience_count = 0
        history = {'loss': [], 'val_loss': [], 'acc': [], 'val_acc': []}

        for epoch in range(1, epochs + 1):
            idx = np.random.permutation(N)
            epoch_metrics = {'loss': [], 'acc': [], 'main_bce': [], 'aux_bce': [], 'cons': []}

            for start in range(0, N, batch_size):
                batch_idx = idx[start:start + batch_size]
                x_batch = {k: X_tr[k][batch_idx] for k in self.input_order}
                y_batch = y_tr[batch_idx]
                m = self.train_step(x_batch, y_batch)
                
                epoch_metrics['loss'].append(float(m['loss']))
                epoch_metrics['acc'].append(float(m['acc']))
                epoch_metrics['main_bce'].append(float(m['main_bce']))
                epoch_metrics['aux_bce'].append(float(m['aux_bce']))
                epoch_metrics['cons'].append(float(m['cons']))

            val_metrics = {'val_loss': [], 'val_acc': []}
            Nv = len(y_va)
            for start in range(0, Nv, batch_size):
                x_vb = {k: X_va[k][start:start + batch_size] for k in self.input_order}
                y_vb = y_va[start:start + batch_size]
                m = self.val_step(x_vb, y_vb)
                val_metrics['val_loss'].append(float(m['val_loss']))
                val_metrics['val_acc'].append(float(m['val_acc']))

            tr_loss = np.mean(epoch_metrics['loss'])
            tr_acc  = np.mean(epoch_metrics['acc'])
            va_loss = np.mean(val_metrics['val_loss'])
            va_acc  = np.mean(val_metrics['val_acc'])
            
            m_bce = np.mean(epoch_metrics['main_bce'])
            a_bce = np.mean(epoch_metrics['aux_bce'])
            c_loss = np.mean(epoch_metrics['cons'])

            history['loss'].append(tr_loss)
            history['acc'].append(tr_acc)
            history['val_loss'].append(va_loss)
            history['val_acc'].append(va_acc)

            print(f"Epoch {epoch:3d}/{epochs} — loss={tr_loss:.4f} (m={m_bce:.3f} a={a_bce:.3f} c={c_loss:.3f}) acc={tr_acc:.4f} val_loss={va_loss:.4f} val_acc={va_acc:.4f}")

            if va_loss < best_val_loss:
                best_val_loss = va_loss
                patience_count = 0
                self.model.save_weights(ckpt_path)
                print(f"   [OK] Saved best model weights (val_loss={va_loss:.4f})")
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"Early stopping at epoch {epoch} (patience={patience})")
                    break

        return history, best_val_loss


def train_variant(variant_cfg):
    feature_keys = variant_cfg['feature_keys']
    v_name = variant_cfg['name']
    dir_name = variant_cfg['dir_name']
    
    print(f"\n{'#'*60}")
    print(f"  Stage E — {v_name}")
    print(f"{'#'*60}")

    train_entries = load_split_csv(os.path.join(SPLIT_DIR, 'split_train.csv'))
    val_entries   = load_split_csv(os.path.join(SPLIT_DIR, 'split_val.csv'))
    test_entries  = load_split_csv(os.path.join(SPLIT_DIR, 'split_test.csv'))

    X_tr_raw, y_tr = load_features(train_entries, FEATURE_DIR, feature_keys)
    X_va_raw, y_va = load_features(val_entries,   FEATURE_DIR, feature_keys)
    X_te_raw, y_te = load_features(test_entries,  FEATURE_DIR, feature_keys)

    scaler = FeatureScaler()
    X_tr_n = scaler.fit_transform(X_tr_raw)
    X_va_n = scaler.transform(X_va_raw)
    X_te_n = scaler.transform(X_te_raw)

    input_dim = X_tr_n[feature_keys[0]].shape[1]
    print(f"  Train={len(y_tr)} | Val={len(y_va)} | Test={len(y_te)} | Feature dim={input_dim}")

    model = AnchoredFusionModel(feature_keys, input_dim, EMBED_DIM)
    
    out_dir = os.path.join(CHECKPOINT_BASE, dir_name)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best_model.weights.h5')

    trainer = AnchoredTrainer(model, feature_keys, LAMBDA_AUX, LAMBDA_CONS, MARGIN)
    history, best_val_loss = trainer.train(
        X_tr_n, y_tr, X_va_n, y_va,
        epochs=EPOCHS, batch_size=BATCH_SIZE,
        patience=PATIENCE, ckpt_path=ckpt_path
    )

    # Load best weights
    model.load_weights(ckpt_path)
    
    test_X_dict = {k: X_te_n[k] for k in feature_keys}
    outputs = model(test_X_dict, training=False)
    y_pred_proba = tf.sigmoid(outputs['main_logit']).numpy().flatten()
    
    metrics = compute_all_metrics(y_te, y_pred_proba)
    metrics['features'] = feature_keys
    metrics['variant_name'] = v_name
    metrics['best_val_loss'] = float(best_val_loss)
    
    print_metrics(f"Stage E — {v_name} TEST RESULTS", metrics)
    
    scaler.save(os.path.join(out_dir, 'scaler.pkl'))
    save_metrics(metrics, os.path.join(out_dir, 'metrics.json'))
    with open(os.path.join(out_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=4)
        
    return metrics


def main():
    for split in ['split_train.csv', 'split_val.csv', 'split_test.csv']:
        if not os.path.exists(os.path.join(SPLIT_DIR, split)):
            print(f"ERROR: {split} not found. Run prepare_deepfakes_split.py first.")
            sys.exit(1)

    all_metrics = {}
    for variant in VARIANTS:
        m = train_variant(variant)
        all_metrics[variant['name']] = m

    print("\n\n" + "="*65)
    print("  STAGE E — ANCHORED FUSION SUMMARY")
    print("="*65)
    print(f"  {'Variant':<24} {'Acc':>7} {'F1':>7} {'ROC-AUC':>9} {'PR-AUC':>9}")
    print(f"  {'-'*24} {'-'*7} {'-'*7} {'-'*9} {'-'*9}")
    for vname, m in all_metrics.items():
        print(f"  {vname:<24} {m['accuracy']:>7.4f} {m['f1']:>7.4f} "
              f"{m['roc_auc']:>9.4f} {m['pr_auc']:>9.4f}")
    print("="*65)

    os.makedirs(CHECKPOINT_BASE, exist_ok=True)
    with open(os.path.join(CHECKPOINT_BASE, 'summary.json'), 'w') as f:
        json.dump(all_metrics, f, indent=4)


if __name__ == '__main__':
    main()
