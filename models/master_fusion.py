import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, ops

@keras.saving.register_keras_serializable()
class ResidualBlock(layers.Layer):
    """Deep residual block with Skip Connection."""
    def __init__(self, units=256, dropout_rate=0.2, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.dropout_rate = dropout_rate
        self.dense1 = layers.Dense(units, activation='relu')
        self.dropout = layers.Dropout(dropout_rate)
        self.dense2 = layers.Dense(units, activation=None)
        self.layer_norm = layers.LayerNormalization()

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        x1 = self.dense1(inputs)
        x1 = self.dropout(x1, training=training)
        x2 = self.dense2(x1)
        # Skip connection
        out = inputs + x2
        out = self.layer_norm(out)
        return ops.relu(out)

    def get_config(self):
        config = super().get_config()
        config.update({
            "units": self.units,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable()
class TransformerEncoderBlock(layers.Layer):
    """Standard Transformer Encoder block."""
    def __init__(self, embed_dim=256, num_heads=4, ff_dim=512, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim//num_heads)
        self.ffn = keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dropout(dropout_rate),
            layers.Dense(embed_dim)
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "dropout_rate": self.dropout_rate,
        })
        return config


@keras.saving.register_keras_serializable()
class FeaturePipeline(layers.Layer):
    """Independent feature processing pipeline: Dense -> ResBlocks x2 -> Projection"""
    def __init__(self, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.proj_in = layers.Dense(embed_dim, activation='relu')
        self.res1 = ResidualBlock(embed_dim)
        self.res2 = ResidualBlock(embed_dim)
        self.proj_out = layers.Dense(embed_dim, activation='relu')

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        x = self.proj_in(inputs)
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        return self.proj_out(x)

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


@keras.saving.register_keras_serializable()
class SubSystem1Encoder(layers.Layer):
    """Sub-System 1: Hardware Optics (Lens Distortion + Motion Blur + FFT Frequency)"""
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        self.f1_pipe = FeaturePipeline(embed_dim)
        self.f2_pipe = FeaturePipeline(embed_dim)
        self.f5_pipe = FeaturePipeline(embed_dim)
        
        self.gate_dense = layers.Dense(3, activation='softmax')
        
        self.tf1 = TransformerEncoderBlock(embed_dim)
        self.tf2 = TransformerEncoderBlock(embed_dim)
        
        self.cross_att = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim//4)
        
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)
        ])

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        f1, f2, f5 = inputs
        
        # Independent Processing
        e1 = self.f1_pipe(f1, training=training)  # [B, embed_dim]
        e2 = self.f2_pipe(f2, training=training)
        e5 = self.f5_pipe(f5, training=training)
        
        # Feature Gates
        concat_feats = ops.concatenate([e1, e2, e5], axis=-1)  # [B, 3*embed_dim]
        g = self.gate_dense(concat_feats)  # [B, 3]
        
        e1_gated = e1 * ops.expand_dims(g[:, 0], axis=-1)
        e2_gated = e2 * ops.expand_dims(g[:, 1], axis=-1)
        e5_gated = e5 * ops.expand_dims(g[:, 2], axis=-1)
        
        # Sequence Stacking
        seq = ops.stack([e1_gated, e2_gated, e5_gated], axis=1)  # [B, 3, embed_dim]
        
        # Transformer Encoder x2
        seq = self.tf1(seq, training=training)
        seq = self.tf2(seq, training=training)
        
        # Attention Pooling (Cross-Attention)
        batch_size = ops.shape(seq)[0]
        cls_tokens = ops.repeat(self.cls_token, repeats=batch_size, axis=0)  # [B, 1, embed_dim]
        
        pooled = self.cross_att(
            query=cls_tokens, value=seq, key=seq, training=training
        )  # [B, 1, embed_dim]
        z1 = ops.squeeze(pooled, axis=1)  # [B, embed_dim]
        
        z1_for_classifier = ops.stop_gradient(z1) if training else z1
        sub_logits = self.classifier(z1_for_classifier, training=training)
        
        return z1, sub_logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "embed_dim": self.embed_dim,
        })
        return config


@keras.saving.register_keras_serializable()
class SubSystem2Encoder(layers.Layer):
    """Sub-System 2: Biological Domain (Biomechanics + Lighting SH)"""
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim

        self.f3_pipe = FeaturePipeline(embed_dim)
        self.f4_pipe = FeaturePipeline(embed_dim)
        
        self.gate_dense = layers.Dense(2, activation='softmax')
        
        self.tf1 = TransformerEncoderBlock(embed_dim)
        self.tf2 = TransformerEncoderBlock(embed_dim)
        
        self.cross_att = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim//4)

        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)
        ])

    def build(self, input_shape):
        self.cls_token = self.add_weight(
            shape=(1, 1, self.embed_dim),
            initializer='zeros',
            trainable=True,
            name='cls_token'
        )
        super().build(input_shape)

    def call(self, inputs, training=False):
        f3, f4 = inputs
        
        # Independent Processing
        e3 = self.f3_pipe(f3, training=training)
        e4 = self.f4_pipe(f4, training=training)
        
        # Feature Gates
        concat_feats = ops.concatenate([e3, e4], axis=-1)
        g = self.gate_dense(concat_feats)  # [B, 2]
        
        e3_gated = e3 * ops.expand_dims(g[:, 0], axis=-1)
        e4_gated = e4 * ops.expand_dims(g[:, 1], axis=-1)
        
        # Sequence Stacking
        seq = ops.stack([e3_gated, e4_gated], axis=1)  # [B, 2, embed_dim]
        
        # Transformer Encoder x2
        seq = self.tf1(seq, training=training)
        seq = self.tf2(seq, training=training)
        
        # Attention Pooling
        batch_size = ops.shape(seq)[0]
        cls_tokens = ops.repeat(self.cls_token, repeats=batch_size, axis=0)
        
        pooled = self.cross_att(
            query=cls_tokens, value=seq, key=seq, training=training
        )
        z2 = ops.squeeze(pooled, axis=1)
        
        z2_for_classifier = ops.stop_gradient(z2) if training else z2
        sub_logits = self.classifier(z2_for_classifier, training=training)
        
        return z2, sub_logits

    def get_config(self):
        config = super().get_config()
        config.update({
            "feature_dim": self.feature_dim,
            "embed_dim": self.embed_dim,
        })
        return config


@keras.saving.register_keras_serializable()
class MasterFusionBlock(layers.Layer):
    """Master-level cross-attention + gated fusion block for Sub-Systems 1 and 2."""
    def __init__(self, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.master_cross_attn = layers.MultiHeadAttention(
            num_heads=8, key_dim=embed_dim // 8, name='master_cross_attn'
        )
        self.gate_dense = layers.Dense(
            embed_dim, activation='sigmoid', name='master_fusion_gate'
        )
        self.layer_norm = layers.LayerNormalization(name='master_layer_norm')

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        z1, z2 = inputs  # Each [B, 256]

        # Stack vectors to form sequence dimension: [B, 2, 256]
        z_stack = ops.stack([z1, z2], axis=1)

        attn_out = self.master_cross_attn(
            query=z_stack, value=z_stack, key=z_stack, training=training
        )  # [B, 2, 256]

        attn_pooled = ops.mean(attn_out, axis=1)  # [B, 256]
        z_avg = ops.mean(z_stack, axis=1)

        g_master = self.gate_dense(
            ops.concatenate([z1, z2], axis=-1)
        )  # [B, 256]

        master_rep = self.layer_norm(
            g_master * z_avg + (1.0 - g_master) * attn_pooled
        )  # [B, 256]

        return master_rep

    def get_config(self):
        config = super().get_config()
        config.update({"embed_dim": self.embed_dim})
        return config


def build_master_physics_detector(feature_dim=64, embed_dim=256):
    """Functional Keras Master Model using Sub-Systems 1 and 2 only."""

    input_f1 = layers.Input(shape=(feature_dim,), name='f1_lens_distortion')
    input_f2 = layers.Input(shape=(feature_dim,), name='f2_motion_blur')
    input_f3 = layers.Input(shape=(feature_dim,), name='f3_biomechanics')
    input_f4 = layers.Input(shape=(feature_dim,), name='f4_lighting_sh')
    input_f5 = layers.Input(shape=(feature_dim,), name='f5_frequency_fft')

    subsystem1 = SubSystem1Encoder(feature_dim, embed_dim, name='subsystem1_hardware')
    subsystem2 = SubSystem2Encoder(feature_dim, embed_dim, name='subsystem2_biological')

    z1, sub1_logits = subsystem1([input_f1, input_f2, input_f5])
    z2, sub2_logits = subsystem2([input_f3, input_f4])

    master_rep = MasterFusionBlock(embed_dim, name='master_fusion')([z1, z2])

    x = layers.Dense(128, activation='relu')(master_rep)
    x = layers.Dropout(0.4)(x)
    master_logits = layers.Dense(1, activation=None, name='master_logits')(x)

    model = Model(
        inputs=[input_f1, input_f2, input_f3, input_f4, input_f5],
        outputs={
            'master_logits': master_logits,
            'sub1_logits':   sub1_logits,
            'sub2_logits':   sub2_logits,
        },
        name='MasterPhysicsDeepfakeDetector'
    )

    return model