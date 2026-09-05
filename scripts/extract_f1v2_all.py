"""
scripts/extract_f1v2_all.py
------------------------------------------
Extract the new F1-v2 feature (Fixed-Bin + Temporal) across:
1. Deepfakes (balanced train split)
2. Celeb-DF v2
3. FaceForensics++ (c40)

Output will be stored in:
- DataSets/f1v2_features/deepfakes/
- DataSets/f1v2_features/celeb_df/
- DataSets/f1v2_features/ff_plus_plus/
"""

import os
import sys
import glob
import time
import numpy as np
import cv2
from tqdm import tqdm
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import preprocess_video
from extractors.lens_distortion_v2 import extract_lens_distortion_v2_features

# Base Configurations
DS_CONFIG = {
    'deepfakes': {
        'sources': [
            (os.path.join('DataSets', 'deepfakes', 'source_videos_part_00'), 0),
            (os.path.join('DataSets', 'deepfakes', 'source_videos_part_01'), 0),
            (os.path.join('DataSets', 'deepfakes', 'source_videos_part_03'), 0),
            (os.path.join('DataSets', 'deepfakes', 'source_videos_part_04'), 0),
            (os.path.join('DataSets', 'deepfakes', 'manipulated_videos_part_00'), 1),
            (os.path.join('DataSets', 'deepfakes', 'manipulated_videos_part_01'), 1),
            (os.path.join('DataSets', 'deepfakes', 'manipulated_videos_part_02'), 1),
            (os.path.join('DataSets', 'deepfakes', 'manipulated_videos_part_11'), 1)
        ],
        'output_dir': os.path.join('DataSets', 'f1v2_features', 'deepfakes')
    },
    'celeb_df': {
        'sources': [
            (os.path.join('DataSets', 'Celeb-DF-v2', 'Celeb-real'), 0),
            (os.path.join('DataSets', 'Celeb-DF-v2', 'YouTube-real'), 0),
            (os.path.join('DataSets', 'Celeb-DF-v2', 'Celeb-synthesis'), 1)
        ],
        'output_dir': os.path.join('DataSets', 'f1v2_features', 'celeb_df')
    },
    'ff_plus_plus': {
        'sources': [
            (os.path.join('DataSets', 'FaceForensics++', 'original_sequences', 'youtube', 'c40', 'videos'), 0),
            (os.path.join('DataSets', 'FaceForensics++', 'original_sequences', 'actors', 'c40', 'videos'), 0),
            (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences', 'Deepfakes', 'c40', 'videos'), 1),
            (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences', 'Face2Face', 'c40', 'videos'), 1),
            (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences', 'FaceSwap', 'c40', 'videos'), 1),
            (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences', 'FaceShifter', 'c40', 'videos'), 1),
            (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences', 'NeuralTextures', 'c40', 'videos'), 1),
        ],
        'output_dir': os.path.join('DataSets', 'f1v2_features', 'ff_plus_plus')
    }
}

def discover_videos(source_dir):
    if not os.path.isdir(source_dir):
        return []
    return sorted(glob.glob(os.path.join(source_dir, '**', '*.mp4'), recursive=True))

def balance_videos(all_videos, dataset_name):
    reals = [v for v in all_videos if v[3] == 0]
    fakes = [v for v in all_videos if v[3] == 1]
    
    # Sort for deterministic shuffling
    reals.sort(key=lambda x: x[0])
    fakes.sort(key=lambda x: x[0])
    
    random.seed(42)
    random.shuffle(reals)
    random.shuffle(fakes)
    
    if dataset_name == 'deepfakes':
        target = 5000
    elif dataset_name == 'ff_plus_plus':
        target = 5000
    else: # celeb_df
        target = min(len(reals), len(fakes))
        
    min_count = min(target, len(reals), len(fakes))
    
    selected = reals[:min_count] + fakes[:min_count]
    random.shuffle(selected)
    
    print(f"  Balanced {dataset_name}: {min_count} real + {min_count} fake = {len(selected)} total.")
    return selected

def extract_and_save(video_path, output_path, label, n_frames=16, feature_dim=64):
    if os.path.exists(output_path):
        return False
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    try:
        frames = preprocess_video(video_path, n_frames=n_frames, target_size=(256, 256), sigma=0.8)
        if len(frames) < 2:
            f1_v2 = np.zeros(feature_dim, dtype=np.float32)
        else:
            f1_v2 = extract_lens_distortion_v2_features(frames, feature_dim)
            
        np.savez_compressed(output_path, f1_v2=f1_v2)
        return True
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        f1_v2 = np.zeros(feature_dim, dtype=np.float32)
        np.savez_compressed(output_path, f1_v2=f1_v2)
        return None

def process_dataset(name, config):
    print(f"\n--- Processing {name} ---")
    sources = config['sources']
    output_dir = config['output_dir']
    
    all_videos = []
    for source_dir, label in sources:
        videos = discover_videos(source_dir)
        for vp in videos:
            if name == 'deepfakes':
                base_dir = os.path.join('DataSets', 'deepfakes')
                rel = os.path.relpath(vp, base_dir)
                npz_rel = os.path.splitext(rel)[0] + ".npz"
            else:
                rel = os.path.relpath(vp, source_dir)
                stem = os.path.splitext(rel)[0].replace(os.sep, '_')
                subfolder = 'real' if label == 0 else 'fake'
                npz_rel = os.path.join(subfolder, f"{stem}.npz")
                
            npz_path = os.path.join(output_dir, npz_rel)
            all_videos.append((vp, npz_path, npz_rel, label))
            
    if not all_videos:
        print("No videos found.")
        return
        
    all_videos = balance_videos(all_videos, name)
        
    extracted, skipped = 0, 0
    manifest_entries = []
    
    for vp, npz_path, npz_rel, label in tqdm(all_videos, desc=name):
        manifest_entries.append((npz_rel.replace('\\', '/'), label))
        res = extract_and_save(vp, npz_path, label)
        if res is True: extracted += 1
        elif res is False: skipped += 1
        
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, 'manifest.csv'), 'w', encoding='utf-8') as f:
        f.write("npz_path,label\n")
        for p, l in manifest_entries:
            f.write(f"{p},{l}\n")
            
    print(f"Done {name}. Extracted: {extracted}, Skipped: {skipped}")


def main():
    for name, config in DS_CONFIG.items():
        process_dataset(name, config)
        
if __name__ == '__main__':
    main()
