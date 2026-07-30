import tensorflow as tf
import keras
from tensorflow.keras import layers, Model, ops

@keras.saving.register_keras_serializable()
class SubSystem3Encoder(layers.Layer):
    """Sub-System 3: Physics-Aware Consistency (PAC)"""
    def __init__(self, feature_dim=64, embed_dim=256, **kwargs):
        super().__init__(**kwargs)
        self.fc_in = layers.Dense(embed_dim)
        self.layer_norm = layers.LayerNormalization()
        
        # Residual + Cross Attention (using self-attention here since it's a single vector)
        self.attn = layers.MultiHeadAttention(num_heads=4, key_dim=embed_dim // 4)
        
        # Sub-Classifier Head
        self.classifier = keras.Sequential([
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(1, activation=None)  # Output Logits
        ])

    def call(self, inputs, training=False):
        # inputs: [B, 64] (physics64)
        e = self.fc_in(inputs)  # [B, 256]
        
        # self attention
        e_seq = ops.expand_dims(e, axis=1)  # [B, 1, 256]
        attn_out = self.attn(query=e_seq, value=e_seq, key=e_seq, training=training)
        
        z3 = ops.squeeze(self.layer_norm(e_seq + attn_out), axis=1)  # [B, 256]
        
        sub_logits = self.classifier(z3, training=training)
        return z3, sub_logits
