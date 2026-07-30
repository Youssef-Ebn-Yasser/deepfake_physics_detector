"""
subsystem_1.py  –  Keras Sub-model for Hardware Optics Domain
--------------------------------------------------------------
Standalone model wrapping SubSystem1Encoder so it can be trained
independently on lens-distortion (F1) and motion-blur (F2) features
before the master fusion stage.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from models.master_fusion import SubSystem1Encoder


def build_subsystem1_model(feature_dim=64, embed_dim=256):
    """
    Build a standalone Keras model for Sub-System 1 (Hardware Optics).

    Inputs:
        f1_lens_distortion : (batch, feature_dim)
        f2_motion_blur     : (batch, feature_dim)

    Outputs:
        sub1_logits : (batch, 1)  – binary classification logits
    """
    input_f1 = layers.Input(shape=(feature_dim,), name='f1_lens_distortion')
    input_f2 = layers.Input(shape=(feature_dim,), name='f2_motion_blur')

    encoder = SubSystem1Encoder(feature_dim, embed_dim, name='subsystem1_hardware')
    z1, sub1_logits = encoder([input_f1, input_f2])

    model = Model(
        inputs=[input_f1, input_f2],
        outputs={'sub1_logits': sub1_logits, 'z1_embedding': z1},
        name='SubSystem1_HardwareOptics',
    )
    return model
