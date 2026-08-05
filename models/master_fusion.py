import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, ops

@keras.saving.register_keras_serializable()
class SubSystem1Encoder(layers.Layer):
    """Sub-System 1: Hardware Optics (Lens Distortion + Motion Blur + FFT Frequency)"""
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.feature_dim = feature_dim
        self.embed_dim = embed_dim
        
        self.projection = layers.Dense(embed_dim, activation='relu')
        self.layer_norm = layers.LayerNormalization()
        
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)
        ])

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        f1, f2, f5 = inputs
        concat_features = ops.concatenate([f1, f2, f5], axis=-1)
        z1 = self.layer_norm(self.projection(concat_features))
        
        # فصل الـ Gradients أثناء تدريب النموذج الكامل لمنع تلوث z1 بالـ Auxiliary Loss
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

        self.fc_f3 = layers.Dense(embed_dim)
        self.fc_f4 = layers.Dense(embed_dim)
        self.gate_dense = layers.Dense(embed_dim, activation='sigmoid')
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)
        ])

    def build(self, input_shape):
        super().build(input_shape)

    def call(self, inputs, training=False):
        f3, f4 = inputs
        e3 = self.fc_f3(f3)
        e4 = self.fc_f4(f4)
        concat_feats = ops.concatenate([e3, e4], axis=-1)
        g = self.gate_dense(concat_feats)
        z2 = g * e3 + (1.0 - g) * e4
        
        # فصل الـ Gradients أثناء تدريب النموذج الكامل
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