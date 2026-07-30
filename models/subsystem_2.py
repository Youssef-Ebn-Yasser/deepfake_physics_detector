"""
subsystem_2.py  –  Keras Sub-model for Biological Domain
---------------------------------------------------------
Standalone model wrapping SubSystem2Encoder so it can be trained
independently on biomechanics (F3) and spherical-harmonics lighting
(F4) features before the master fusion stage.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from models.master_fusion import SubSystem2Encoder


def build_subsystem2_model(feature_dim=64, embed_dim=256):
    """
    Build a standalone Keras model for Sub-System 2 (Biological Domain).

    Inputs:
        f3_biomechanics : (batch, feature_dim)
        f4_lighting_sh  : (batch, feature_dim)

    Outputs:
        sub2_logits : (batch, 1)  – binary classification logits
    """
    input_f3 = layers.Input(shape=(feature_dim,), name='f3_biomechanics')
    input_f4 = layers.Input(shape=(feature_dim,), name='f4_lighting_sh')

    encoder = SubSystem2Encoder(feature_dim, embed_dim, name='subsystem2_biological')
    z2, sub2_logits = encoder([input_f3, input_f4])

    model = Model(
        inputs=[input_f3, input_f4],
        outputs={'sub2_logits': sub2_logits, 'z2_embedding': z2},
        name='SubSystem2_BiologicalDomain',
    )
    return model
