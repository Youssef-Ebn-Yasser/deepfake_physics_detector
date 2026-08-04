"""
diagnose_features.py
--------------------
Audits extracted features for NaN/Inf values, computes global Min/Max/Mean/Std,
and verifies class distribution from the manifest.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from data.dataset_loader import load_manifest, _load_npz
import tensorflow as tf

def main():
    features_dir = 'DataSets/features'
    out_file = 'results/feature_health.txt'
    
    os.makedirs('results', exist_ok=True)
    
    entries = load_manifest(features_dir)
    print(f"Loaded {len(entries)} entries from manifest.")
    
    class_counts = {0: 0, 1: 0}
    
    f1_all = []
    f2_all = []
    f3_all = []
    f4_all = []
    f5_all = []
    
    nan_count = 0
    inf_count = 0
    
    print("Scanning features...")
    for path, label in entries:
        class_counts[label] += 1
        
        try:
            # Fake tensor inputs to reuse existing _load_npz logic
            t_path = tf.constant(path)
            t_dir = tf.constant(features_dir)
            t_label = tf.constant(label)
            
            f1, f2, f3, f4, f5, _ = _load_npz(t_path, t_label, t_dir)
            
            # Check NaN/Inf
            for arr in [f1, f2, f3, f4, f5]:
                if np.isnan(arr).any():
                    nan_count += 1
                if np.isinf(arr).any():
                    inf_count += 1
                    
            f1_all.append(f1)
            f2_all.append(f2)
            f3_all.append(f3)
            f4_all.append(f4)
            f5_all.append(f5)
            
        except Exception as e:
            print(f"Error loading {path}: {e}")
            
    f1_all = np.array(f1_all)
    f2_all = np.array(f2_all)
    f3_all = np.array(f3_all)
    f4_all = np.array(f4_all)
    f5_all = np.array(f5_all)
    
    report = []
    report.append("==========================================")
    report.append("       FEATURE DIAGNOSTIC REPORT")
    report.append("==========================================\n")
    
    report.append(f"Total Samples Scanned: {len(f1_all)}")
    report.append(f"Class Distribution: Real={class_counts[0]}, Fake={class_counts[1]}")
    report.append(f"NaN Found: {'YES' if nan_count > 0 else 'NO'} ({nan_count} instances)")
    report.append(f"Inf Found: {'YES' if inf_count > 0 else 'NO'} ({inf_count} instances)\n")
    
    report.append("Global Statistics (per feature channel):")
    report.append("-" * 40)
    
    features = [
        ('f1_lens_distortion', f1_all),
        ('f2_motion_blur', f2_all),
        ('f3_biomechanics', f3_all),
        ('f4_lighting_sh', f4_all),
        ('physics64', f5_all)
    ]
    
    # We will output Mean and Std so Agent 2 can use them for Z-score norm
    for name, data in features:
        mean_val = np.mean(data)
        std_val = np.std(data)
        min_val = np.min(data)
        max_val = np.max(data)
        
        report.append(f"{name}:")
        report.append(f"  Shape: {data.shape}")
        report.append(f"  Mean:  {mean_val:.6f}")
        report.append(f"  Std:   {std_val:.6f}")
        report.append(f"  Min:   {min_val:.6f}")
        report.append(f"  Max:   {max_val:.6f}\n")
        
    report_text = "\n".join(report)
    print(report_text)
    
    with open(out_file, 'w') as f:
        f.write(report_text)
        
    print(f"Saved diagnostic report to {out_file}")

if __name__ == '__main__':
    main()
