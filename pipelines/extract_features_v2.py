"""
extract_features_v2.py
-----------------------
Alternative feature extraction pipeline using preprocessing_v2.py.

Key differences from extract_features.py (v1):
    - Uses preprocess_video_v2 instead of preprocess_video:
        * Bilateral filter + CLAHE instead of Gaussian blur
        * Eye-alignment before cropping
        * Scene-change-aware temporal sampling
    - Saves features to a SEPARATE output directory (DataSets/features_v2/)
      so that v1 features are never overwritten.
    - Writes manifest to DataSets/features_v2/manifest.csv

Usage:
    python pipelines/extract_features_v2.py
    python pipelines/extract_features_v2.py --config config/default_config.yaml
    python pipelines/extract_features_v2.py --n_frames 24  (more frames per video)
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing_v2 import preprocess_video_v2
from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur import extract_motion_blur_features
from extractors.biomechanics import extract_biomechanics_features
from extractors.lighting_sh import extract_lighting_sh_features


def load_config(config_path='config/default_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def discover_videos(dataset_root, sources, compression='c40'):
    videos = []
    for source in sources:
        video_dir = os.path.join(dataset_root, source, compression, 'videos')
        if not os.path.isdir(video_dir):
            print(f"  [WARN] Directory not found, skipping: {video_dir}")
            continue
        found = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
        print(f"  {source}/{compression}/videos/ -> {len(found)} videos")
        videos.extend(found)
    return videos


def get_feature_output_path(video_path, dataset_root, features_dir):
    """Map a video path to its .npz output path under features_dir."""
    rel = os.path.relpath(video_path, dataset_root)
    parts = rel.replace('\\', '/').split('/')

    category_parts = []
    for p in parts[:-1]:
        if p in ('c23', 'c40', 'videos'):
            continue
        category_parts.append(p)
    category = '_'.join(category_parts)

    stem = os.path.splitext(parts[-1])[0]
    return os.path.join(features_dir, category, f'{stem}.npz')


def extract_and_save_v2(video_path, output_path, n_frames=16,
                        target_size=(256, 256), feature_dim=64,
                        clip_limit=2.0, pad_ratio=0.10):
    """
    Extract all 4 feature vectors using preprocessing_v2 and save as .npz.
    Skips if the output already exists.
    """
    if os.path.exists(output_path):
        return False  # already extracted

    frames = preprocess_video_v2(
        video_path,
        n_frames=n_frames,
        target_size=target_size,
        clip_limit=clip_limit,
        pad_ratio=pad_ratio,
    )

    if len(frames) < 2:
        f1 = np.zeros(feature_dim, dtype=np.float32)
        f2 = np.zeros(feature_dim, dtype=np.float32)
        f3 = np.zeros(feature_dim, dtype=np.float32)
        f4 = np.zeros(feature_dim, dtype=np.float32)
    else:
        f1 = extract_lens_distortion_features(frames, feature_dim)
        f2 = extract_motion_blur_features(frames, feature_dim)
        f3 = extract_biomechanics_features(frames, feature_dim)
        f4 = extract_lighting_sh_features(frames, feature_dim)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, f1=f1, f2=f2, f3=f3, f4=f4)
    return True


def main():
    parser = argparse.ArgumentParser(
        description='Extract physics features (v2 preprocessing) from FF++ videos'
    )
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--features_dir', default=None,
                        help='Output directory override (default: DataSets/features_v2)')
    parser.add_argument('--n_frames', type=int, default=None,
                        help='Number of frames per video (default: from config)')
    parser.add_argument('--clip_limit', type=float, default=2.0,
                        help='CLAHE clip limit (higher = more contrast boost, default: 2.0)')
    parser.add_argument('--pad_ratio', type=float, default=0.10,
                        help='Face crop padding ratio (default: 0.10, v1 used 0.20)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg['dataset']
    pp = cfg['preprocessing']

    dataset_root = ds['root']
    compression = ds['compression']
    feature_dim = cfg['extractors']['feature_dim']

    n_frames = args.n_frames if args.n_frames else pp['n_frames']
    target_size = tuple(pp['target_size'])

    # Use a SEPARATE output dir so v1 features are preserved
    if args.features_dir:
        features_dir = args.features_dir
    else:
        features_dir = os.path.join(os.path.dirname(ds['features_dir']), 'features_v2')

    print(f"\n{'='*60}")
    print(f"  Extract Features V2")
    print(f"  n_frames   : {n_frames}")
    print(f"  target_size: {target_size}")
    print(f"  clip_limit : {args.clip_limit}  (CLAHE)")
    print(f"  pad_ratio  : {args.pad_ratio}  (face crop padding)")
    print(f"  output_dir : {features_dir}")
    print(f"{'='*60}\n")

    # --- Discover all videos ---
    print("=== Discovering REAL videos ===")
    real_videos = discover_videos(dataset_root, ds['real_sources'], compression)

    print("\n=== Discovering FAKE videos ===")
    fake_videos = discover_videos(dataset_root, ds['fake_sources'], compression)

    print(f"\nTotal: {len(real_videos)} real + {len(fake_videos)} fake = "
          f"{len(real_videos) + len(fake_videos)} videos\n")

    all_videos = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]

    t0 = time.time()
    extracted, skipped = 0, 0

    for video_path, label in tqdm(all_videos, desc='Extracting features (v2)'):
        out_path = get_feature_output_path(video_path, dataset_root, features_dir)
        was_new = extract_and_save_v2(
            video_path, out_path,
            n_frames=n_frames,
            target_size=target_size,
            feature_dim=feature_dim,
            clip_limit=args.clip_limit,
            pad_ratio=args.pad_ratio,
        )
        if was_new:
            extracted += 1
        else:
            skipped += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  Extracted: {extracted}  |  Skipped (cached): {skipped}")

    # --- Save manifest ---
    os.makedirs(features_dir, exist_ok=True)
    manifest_path = os.path.join(features_dir, 'manifest.csv')
    with open(manifest_path, 'w') as f:
        f.write('npz_path,label\n')
        for video_path, label in all_videos:
            out_path = get_feature_output_path(video_path, dataset_root, features_dir)
            rel_out = os.path.relpath(out_path, features_dir)
            f.write(f'{rel_out},{label}\n')

    print(f"Manifest saved to {manifest_path}  ({len(all_videos)} entries)")
    print("\n[TIP] To use v2 features for training, pass --features_dir DataSets/features_v2")
    print("      to your training scripts, or update the config yaml.")


if __name__ == '__main__':
    main()
