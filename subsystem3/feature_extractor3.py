import numpy as np
from .preprocessing3 import PhysicsPreprocessor
from .lighting import estimate_lighting
from .shading import compute_shading_consistency
from .shadow import extract_shadow_features
from .reflection import extract_reflection_features

class Subsystem3FeatureExtractor:
    def __init__(self):
        self.preprocessor = PhysicsPreprocessor()
        
    def extract_from_frames(self, frames, feature_dim=64):
        # We process multiple frames from the video.
        # Typically we average the features over all frames.
        frame_features = []
        
        for frame in frames:
            physics_frame = self.preprocessor.process_frame(frame)
            if physics_frame is None:
                continue
                
            L, I0, ambient = estimate_lighting(
                physics_frame.image_linear, 
                physics_frame.skin_mask, 
                physics_frame.normals
            )
            
            # Reconstruct expected shading for shadow/shading modules
            dot_product = np.sum(physics_frame.normals * L, axis=2)
            expected_shading = I0 * np.clip(dot_product, 0, None) + ambient
            expected_shading = np.clip(expected_shading, 0, 1)
            
            shading_feats = compute_shading_consistency(
                physics_frame.image_linear,
                physics_frame.skin_mask,
                physics_frame.normals,
                L, I0, ambient
            )
            
            shadow_feats = extract_shadow_features(
                physics_frame.image_linear,
                physics_frame.skin_mask,
                expected_shading
            )
            
            reflection_feats = extract_reflection_features(
                physics_frame.image_linear,
                physics_frame.landmarks
            )
            
            # Lighting features
            lighting_feats = np.zeros(16, dtype=np.float32)
            lighting_feats[0:3] = L
            lighting_feats[3] = I0
            lighting_feats[4] = ambient
            
            # Concatenate
            physics64 = np.concatenate([
                lighting_feats,   # 16
                shading_feats,    # 16
                reflection_feats, # 16
                shadow_feats      # 16
            ])
            
            frame_features.append(physics64)
            
        if len(frame_features) == 0:
            return np.zeros(feature_dim, dtype=np.float32)
            
        # Average across frames
        avg_features = np.mean(frame_features, axis=0)
        return avg_features.astype(np.float32)
