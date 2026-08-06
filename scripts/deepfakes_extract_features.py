import os
import sys
import glob
import time
import numpy as np
from tqdm import tqdm
import cv2

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import preprocess_video
from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur import extract_motion_blur_features
from extractors.biomechanics import extract_biomechanics_features
from extractors.lighting_sh import extract_lighting_sh_features
from extractors.frequency_fft import extract_fft_spectrum

def extract_fft_features(frames, feature_dim=64):
    fft_features = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        f5 = extract_fft_spectrum(gray, num_bins=feature_dim)
        fft_features.append(f5)
    if not fft_features:
        return np.zeros(feature_dim, dtype=np.float32)
    return np.mean(fft_features, axis=0).astype(np.float32)

def main():
    base_dir = os.path.join("DataSets", "deepfakes")
    output_dir = os.path.join("DataSets", "deepfakes_feature")
    
    target_folders = [
        "source_videos_part_00",
        "source_videos_part_01",
        "source_videos_part_03",
        "source_videos_part_04",
        "manipulated_videos_part_00",
        "manipulated_videos_part_01",
        "manipulated_videos_part_02",
        "manipulated_videos_part_11"
    ]
    
    # Feature extraction settings
    n_frames = 16
    target_size = (256, 256)
    sigma = 0.8
    feature_dim = 64
    
    all_videos = []
    for folder in target_folders:
        folder_path = os.path.join(base_dir, folder)
        if not os.path.exists(folder_path):
            print(f"Warning: {folder_path} does not exist.")
            continue
        
        # Determine label (0 for real, 1 for manipulated)
        label = 0 if "source" in folder else 1
        
        # Recursively find all mp4 files
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mp4"):
                    video_path = os.path.join(root, file)
                    all_videos.append((video_path, label))
                    
    print(f"Found {len(all_videos)} videos to process.")
    
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = os.path.join(output_dir, "manifest.csv")
    manifest_entries = []
    
    extracted, skipped = 0, 0
    t0 = time.time()
    
    for video_path, label in tqdm(all_videos, desc='Extracting features'):
        # Create output path maintaining subfolder structure but saving in deepfakes_feature
        rel_path = os.path.relpath(video_path, base_dir)
        # Change extension to .npz
        npz_rel_path = os.path.splitext(rel_path)[0] + ".npz"
        out_path = os.path.join(output_dir, npz_rel_path)
        
        manifest_entries.append((npz_rel_path, label))
        
        if os.path.exists(out_path):
            skipped += 1
            continue
            
        # Ensure output dir exists
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        
        # Preprocess and extract
        frames = preprocess_video(video_path, n_frames=n_frames, target_size=target_size, sigma=sigma)
        
        if len(frames) < 2:
            # Video too short or unreadable — save zeros
            f1 = np.zeros(feature_dim, dtype=np.float32)
            f2 = np.zeros(feature_dim, dtype=np.float32)
            f3 = np.zeros(feature_dim, dtype=np.float32)
            f4 = np.zeros(feature_dim, dtype=np.float32)
            f5 = np.zeros(feature_dim, dtype=np.float32)
        else:
            f1 = extract_lens_distortion_features(frames, feature_dim)
            f2 = extract_motion_blur_features(frames, feature_dim)
            f3 = extract_biomechanics_features(frames, feature_dim)
            f4 = extract_lighting_sh_features(frames, feature_dim)
            f5 = extract_fft_features(frames, feature_dim)
            
        np.savez_compressed(out_path, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)
        extracted += 1
        
    # Write manifest
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("npz_path,label\n")
        for path, label in manifest_entries:
            # Use forward slashes for standard manifest formats
            path_fw = path.replace("\\", "/")
            f.write(f"{path_fw},{label}\n")
            
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s | Extracted: {extracted} | Skipped (cached): {skipped}")
    print(f"Manifest saved to {manifest_path}")

if __name__ == '__main__':
    main()
