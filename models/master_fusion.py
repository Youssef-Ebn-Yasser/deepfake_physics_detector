import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, ops

@keras.saving.register_keras_serializable()
class SubSystem1Encoder(layers.Layer):
    """Sub-System 1: Hardware Optics (Lens Distortion + Motion Blur)"""
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.fc_f1 = layers.Dense(embed_dim)
        self.fc_f2 = layers.Dense(embed_dim)
        
        # Keras Multi-Head Cross Attention
        self.cross_attn = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim // 4)
        self.layer_norm = layers.LayerNormalization()
        
        # Sub-Classifier Head
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)  # Output Logits
        ])

    def call(self, inputs, training=False):
        f1, f2 = inputs  # Shapes: [B, 64]
        
        e1 = ops.expand_dims(self.fc_f1(f1), axis=1)  # [B, 1, 256]
        e2 = ops.expand_dims(self.fc_f2(f2), axis=1)  # [B, 1, 256]
        
        # Cross Attention: Lens queries Blur
        attn_out = self.cross_attn(query=e1, value=e2, key=e2, training=training)
        z1 = ops.squeeze(self.layer_norm(e1 + attn_out), axis=1)  # [B, 256]
        
        sub_logits = self.classifier(z1, training=training)
        return z1, sub_logits

@keras.saving.register_keras_serializable()
class SubSystem2Encoder(layers.Layer):
    """Sub-System 2: Biological Domain (Biomechanics + Lighting SH)"""
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.fc_f3 = layers.Dense(embed_dim)
        self.fc_f4 = layers.Dense(embed_dim)
        
        # Gated Bilinear Fusion
        self.gate_dense = layers.Dense(embed_dim, activation='sigmoid')
        
        # Sub-Classifier Head
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)  # Output Logits
        ])

    def call(self, inputs, training=False):
        f3, f4 = inputs  # Shapes: [B, 64]
        
        e3 = self.fc_f3(f3)  # [B, 256]
        e4 = self.fc_f4(f4)  # [B, 256]
        
        concat_feats = ops.concatenate([e3, e4], axis=-1)  # [B, 512]
        g = self.gate_dense(concat_feats)                  # [B, 256]
        
        z2 = g * e3 + (1.0 - g) * e4                       # [B, 256]
        sub_logits = self.classifier(z2, training=training)
        return z2, sub_logits


@keras.saving.register_keras_serializable()
class MasterFusionBlock(layers.Layer):
    """
    Master-level cross-attention + gated fusion block.

    Wraps the master fusion logic in a Layer subclass so that all tensor
    operations (expand_dims, squeeze, concatenate) go through keras.ops
    and are fully compatible with Keras 3's Functional API tracing.
    """
    def __init__(self, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.master_cross_attn = layers.MultiHeadAttention(
            num_heads=8, key_dim=embed_dim // 8, name='master_cross_attn',
        )
        self.gate_dense = layers.Dense(
            embed_dim, activation='sigmoid', name='master_fusion_gate',
        )
        self.layer_norm = layers.LayerNormalization(name='master_layer_norm')

    def call(self, inputs, training=False):
        z1, z2 = inputs  # Each [B, 256]

        q1 = ops.expand_dims(z1, axis=1)  # [B, 1, 256]
        k2 = ops.expand_dims(z2, axis=1)  # [B, 1, 256]

        attn_out = self.master_cross_attn(
            query=q1, value=k2, key=k2, training=training,
        )
        attn_out = ops.squeeze(attn_out, axis=1)  # [B, 256]

        g_master = self.gate_dense(
            ops.concatenate([z1, z2], axis=-1)
        )  # [B, 256]

        master_rep = self.layer_norm(
            g_master * z1 + (1.0 - g_master) * attn_out
        )  # [B, 256]

        return master_rep


def build_master_physics_detector(feature_dim=64, embed_dim=256):
    """Functional Keras Master Model for Physics-Aware Deepfake Detection"""
    
    # Inputs for the 4 extracted feature vectors
    input_f1 = layers.Input(shape=(feature_dim,), name='f1_lens_distortion')
    input_f2 = layers.Input(shape=(feature_dim,), name='f2_motion_blur')
    input_f3 = layers.Input(shape=(feature_dim,), name='f3_biomechanics')
    input_f4 = layers.Input(shape=(feature_dim,), name='f4_lighting_sh')
    
    # Sub-Encoder Instantiations
    subsystem1 = SubSystem1Encoder(feature_dim, embed_dim, name='subsystem1_hardware')
    subsystem2 = SubSystem2Encoder(feature_dim, embed_dim, name='subsystem2_biological')
    
    z1, sub1_logits = subsystem1([input_f1, input_f2])
    z2, sub2_logits = subsystem2([input_f3, input_f4])
    
    # Master Cross Attention + Gated Fusion (wrapped in a Layer)
    master_rep = MasterFusionBlock(embed_dim, name='master_fusion')([z1, z2])
    
    # Master Classification Head
    x = layers.Dense(128, activation='relu')(master_rep)
    x = layers.Dropout(0.4)(x)
    master_logits = layers.Dense(1, activation=None, name='master_logits')(x)
    
    # Define multi-output Keras model
    model = Model(
        inputs=[input_f1, input_f2, input_f3, input_f4],
        outputs={
            'master_logits': master_logits,
            'sub1_logits': sub1_logits,
            'sub2_logits': sub2_logits
        },
        name='MasterPhysicsDeepfakeDetector'
    )
    
    return model
