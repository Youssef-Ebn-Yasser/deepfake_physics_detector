"""
fusion_1_2_mid.py
-----------------
Mid-Level Fusion of Sub-System 1 (Hardware Optics) and
Sub-System 2 (Biological Domain).

Architecture:
    - Sub-System 1 Encoder:   [f1, f2, f5] -> z1  [B, 256]  (projection + LN)
    - Sub-System 2 Encoder:   [f3, f4]     -> z2  [B, 256]  (gated bilinear)
    - Mid-Level Fusion Block:
        * Bidirectional Cross-Attention  (z1 attends z2, z2 attends z1)
        * Residual + LayerNorm on each branch
        * Feature Pyramid Mixing  (element-wise product captures interaction)
        * Shared MLP bottleneck  [768-D -> 512-D -> 256-D]
    - Classification Head:    [B, 256] -> sigmoid logit
    - Auxiliary heads on z1 and z2 for regularization during training.

This is strictly mid-level because fusion happens at the intermediate
embedding space (not at raw features and not at final logits).
"""

import keras
import tensorflow as tf
from tensorflow.keras import layers, Model, ops


# ---------------------------------------------------------------------------
# Mid-Level Fusion Block
# ---------------------------------------------------------------------------

@keras.saving.register_keras_serializable()
class MidLevelFusionBlock12(layers.Layer):
    """
    Bidirectional cross-attention fusion between z1 (hardware) and z2 (bio).

    Steps:
        1. z1_enriched = CrossAttn(query=z1, key/value=z2)  + residual
        2. z2_enriched = CrossAttn(query=z2, key/value=z1)  + residual
        3. interaction  = z1_enriched ⊙ z2_enriched          (Hadamard product)
        4. fused = MLP([z1_enriched | z2_enriched | interaction])
    """

    def __init__(self, embed_dim=256, num_heads=8, mlp_dropout=0.3, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim

        # Bidirectional cross-attention
        self.cross_attn_1_to_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            name='cross_attn_z1_to_z2',
        )
        self.cross_attn_2_to_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads,
            name='cross_attn_z2_to_z1',
        )

        # Post-attention layer norms (residual)
        self.ln1 = layers.LayerNormalization(name='ln_z1_enriched')
        self.ln2 = layers.LayerNormalization(name='ln_z2_enriched')

        # Feature pyramid MLP: concat(z1', z2', z1'⊙z2') = 3*256 = 768 -> 256
        self.mlp = keras.Sequential([
            layers.Dense(512, activation='gelu', name='mid_mlp_dense1'),
            layers.Dropout(mlp_dropout),
            layers.Dense(256, activation='gelu', name='mid_mlp_dense2'),
            layers.Dropout(mlp_dropout),
            layers.Dense(embed_dim, activation=None, name='mid_mlp_dense3'),
        ], name='mid_level_mlp')

        self.ln_out = layers.LayerNormalization(name='ln_fusion_out')

    def call(self, inputs, training=False):
        z1, z2 = inputs  # Both [B, 256]

        # Add sequence dimension for MultiHeadAttention: [B, 1, 256]
        z1_seq = ops.expand_dims(z1, axis=1)
        z2_seq = ops.expand_dims(z2, axis=1)

        # z1 attends to z2 → enriches optics embedding with bio context
        z1_ctx = self.cross_attn_1_to_2(
            query=z1_seq, key=z2_seq, value=z2_seq, training=training
        )
        z1_ctx = ops.squeeze(z1_ctx, axis=1)  # [B, 256]
        z1_enriched = self.ln1(z1 + z1_ctx)   # residual + LN

        # z2 attends to z1 → enriches bio embedding with optics context
        z2_ctx = self.cross_attn_2_to_1(
            query=z2_seq, key=z1_seq, value=z1_seq, training=training
        )
        z2_ctx = ops.squeeze(z2_ctx, axis=1)
        z2_enriched = self.ln2(z2 + z2_ctx)

        # Hadamard product captures multiplicative interaction
        interaction = z1_enriched * z2_enriched  # [B, 256]

        # Feature pyramid concat: [z1' | z2' | z1'⊙z2'] = [B, 768]
        pyramid = ops.concatenate([z1_enriched, z2_enriched, interaction], axis=-1)

        # MLP bottleneck to 256
        fused = self.mlp(pyramid, training=training)
        fused = self.ln_out(fused)  # [B, 256]

        return fused, z1_enriched, z2_enriched


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

def build_fusion_1_2_mid(feature_dim=64, embed_dim=256):
    """
    Functional Keras model for mid-level fusion of Sub-System 1 and 2.

    Inputs:
        f1_lens_distortion  [B, feature_dim]
        f2_motion_blur      [B, feature_dim]
        f3_biomechanics     [B, feature_dim]
        f4_lighting_sh      [B, feature_dim]
        f5_frequency_fft    [B, feature_dim]

    Outputs (dict):
        fusion_logits   — primary classification logit
        sub1_logits     — Sub-System 1 auxiliary logit
        sub2_logits     — Sub-System 2 auxiliary logit
    """
    # Inputs
    input_f1 = layers.Input(shape=(feature_dim,), name='f1_lens_distortion')
    input_f2 = layers.Input(shape=(feature_dim,), name='f2_motion_blur')
    input_f3 = layers.Input(shape=(feature_dim,), name='f3_biomechanics')
    input_f4 = layers.Input(shape=(feature_dim,), name='f4_lighting_sh')
    input_f5 = layers.Input(shape=(feature_dim,), name='f5_frequency_fft')

    # Sub-System 1 Encoder (projection + LN)
    from models.master_fusion import SubSystem1Encoder, SubSystem2Encoder
    subsystem1 = SubSystem1Encoder(feature_dim, embed_dim, name='subsystem1_hardware')
    subsystem2 = SubSystem2Encoder(feature_dim, embed_dim, name='subsystem2_biological')

    z1, sub1_logits = subsystem1([input_f1, input_f2, input_f5])
    z2, sub2_logits = subsystem2([input_f3, input_f4])

    # Mid-Level Fusion
    fused, _, _ = MidLevelFusionBlock12(
        embed_dim=embed_dim, name='mid_level_fusion_12'
    )([z1, z2])

    # Classification Head
    x = layers.Dense(256, activation='relu', name='cls_dense1')(fused)
    x = layers.Dropout(0.4, name='cls_drop1')(x)
    x = layers.Dense(128, activation='relu', name='cls_dense2')(x)
    x = layers.Dropout(0.3, name='cls_drop2')(x)
    fusion_logits = layers.Dense(1, activation=None, name='fusion_logits')(x)

    model = Model(
        inputs=[input_f1, input_f2, input_f3, input_f4, input_f5],
        outputs={
            'fusion_logits': fusion_logits,
            'sub1_logits':   sub1_logits,
            'sub2_logits':   sub2_logits,
        },
        name='MidLevelFusion_1_2'
    )

    return model
