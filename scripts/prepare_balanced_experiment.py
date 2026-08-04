import os
import sys
import pandas as pd
import numpy as np
import cv2
import json
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from extractors.frequency_fft import extract_fft_spectrum
from data.preprocessing import preprocess_video

def main():
    # 1. Load manifest.csv
    df = pd.read_csv('DataSets/features/manifest.csv')
    df.columns = ['path', 'label'] if len(df.columns) == 2 else df.columns
    
    # 2. Randomly select exactly 500 Real (label=0) and 500 Fake (label=1) samples.
    real_df = df[df['label'] == 0].sample(n=500, random_state=42)
    fake_df = df[df['label'] == 1].sample(n=500, random_state=42)
    
    # 3. Save this balanced subset to DataSets/features/manifest_1000.csv.
    subset_df = pd.concat([real_df, fake_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
    subset_df.to_csv('DataSets/features/manifest_1000.csv', index=False)
    
    print(f"Generated manifest_1000.csv with {len(subset_df)} samples.")
    
    # We need to compute f5 for these 1000 samples and update their .npz files.
    # Also collect all features to compute mean/std.
    all_f1, all_f2, all_f3, all_f4, all_f5, all_p64 = [], [], [], [], [], []
    
    features_dir = 'DataSets/features'
    
    print("Updating NPZ files with f5 and collecting stats...")
    for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
        npz_path = os.path.join(features_dir, row['path'] if 'path' in row else row['npz_path'])
        
        data = np.load(npz_path)
        f1 = data['f1']
        f2 = data['f2']
        f3 = data['f3']
        f4 = data['f4']
        p64 = data['physics64'] if 'physics64' in data else np.zeros(64, dtype=np.float32)
        
        if 'f5' not in data:
            # We need to find the original video to compute f5
            # The npz path is like DataSets/features/original_sequences_youtube/000.npz
            # We must map this back to the video path
            rel_npz = os.path.relpath(npz_path, features_dir)
            category, stem = os.path.split(rel_npz)
            stem = os.path.splitext(stem)[0]
            
            # Reconstruct video path
            # categories like original_sequences_youtube -> original_sequences/youtube
            # We know the first part is always 'original_sequences' or 'manipulated_sequences'
            if category.startswith('original_sequences_'):
                folder = 'original_sequences/' + category[len('original_sequences_'):]
            elif category.startswith('manipulated_sequences_'):
                folder = 'manipulated_sequences/' + category[len('manipulated_sequences_'):]
            else:
                folder = category
                
            # We assume c40 compression for now
            video_path = os.path.join('DataSets', folder, 'c40', 'videos', f"{stem}.mp4")
            if not os.path.exists(video_path):
                # Fallback
                video_path = video_path.replace('c40', 'c23')
            
            if os.path.exists(video_path):
                frames = preprocess_video(video_path, n_frames=16, target_size=(256, 256), sigma=0.8)
                f5_list = []
                for frame in frames:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                    f5_list.append(extract_fft_spectrum(gray, 64))
                f5 = np.mean(f5_list, axis=0) if len(f5_list) > 0 else np.zeros(64, dtype=np.float32)
            else:
                print(f"Warning: Video not found for {video_path}, using zeros for f5")
                f5 = np.zeros(64, dtype=np.float32)
                
            # Update NPZ
            np.savez_compressed(npz_path, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5, physics64=p64)
        else:
            f5 = data['f5']
            
        all_f1.append(f1)
        all_f2.append(f2)
        all_f3.append(f3)
        all_f4.append(f4)
        all_f5.append(f5)
        all_p64.append(p64)
        
    # 4. Calculate Mean and Standard Deviation
    stats = {
        'f1': (float(np.mean(all_f1)), float(np.std(all_f1) + 1e-8)),
        'f2': (float(np.mean(all_f2)), float(np.std(all_f2) + 1e-8)),
        'f3': (float(np.mean(all_f3)), float(np.std(all_f3) + 1e-8)),
        'f4': (float(np.mean(all_f4)), float(np.std(all_f4) + 1e-8)),
        'f5': (float(np.mean(all_f5)), float(np.std(all_f5) + 1e-8)),
        'physics64': (float(np.mean(all_p64)), float(np.std(all_p64) + 1e-8))
    }
    
    # 5. Save stats to config/feature_scaling_stats.json
    os.makedirs('config', exist_ok=True)
    with open('config/feature_scaling_stats.json', 'w') as f:
        json.dump(stats, f, indent=4)
        
    print("Saved stats to config/feature_scaling_stats.json.")

if __name__ == '__main__':
    main()
