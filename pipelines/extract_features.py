"""
extract_features.py
-------------------
Batch feature extraction pipeline for FaceForensics++ videos.

Processes every video in the DataSets/ tree, runs all 4 physics-based
extractors (lens distortion, motion blur, biomechanics, lighting SH),
and saves the results as .npz archives under DataSets/features/.

Usage:
    python pipelines/extract_features.py
    python pipelines/extract_features.py --config config/default_config.yaml
"""

import os
import sys
import glob
import argparse
import time

import numpy as np
import yaml
from tqdm import tqdm

# Ensure project root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import preprocess_video
from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur import extract_motion_blur_features
from extractors.biomechanics import extract_biomechanics_features
from extractors.lighting_sh import extract_lighting_sh_features


def load_config(config_path='config/default_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def discover_videos(dataset_root, sources, compression='c40'):
    """
    Discover all .mp4 videos under the given sources.

    Args:
        dataset_root: Root DataSets directory.
        sources:      List of relative paths (e.g. 'original_sequences/youtube').
        compression:  Compression level subfolder ('c23' or 'c40').

    Returns:
        List of absolute paths to .mp4 files.
    """
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
    """
    Map a video path to its corresponding .npz feature file path.

    Example:
        video:  DataSets/original_sequences/youtube/c40/videos/000.mp4
        output: DataSets/features/original_sequences_youtube/000.npz
    """
    # Get the relative path from dataset root
    rel = os.path.relpath(video_path, dataset_root)
    # e.g. 'original_sequences/youtube/c40/videos/000.mp4'

    parts = rel.replace('\\', '/').split('/')
    # ['original_sequences', 'youtube', 'c40', 'videos', '000.mp4']

    # Build a flat category key: e.g. 'original_sequences_youtube'
    # Skip the compression and 'videos' parts
    category_parts = []
    for p in parts[:-1]:  # exclude filename
        if p in ('c23', 'c40', 'videos'):
            continue
        category_parts.append(p)
    category = '_'.join(category_parts)

    stem = os.path.splitext(parts[-1])[0]
    return os.path.join(features_dir, category, f'{stem}.npz')


def extract_and_save(video_path, output_path, n_frames=16,
                     target_size=(256, 256), sigma=0.8, feature_dim=64):
    """
    Extract all 4 feature vectors from a single video and save as .npz.
    """
    if os.path.exists(output_path):
        return False  # already extracted

    frames = preprocess_video(video_path, n_frames=n_frames,
                              target_size=target_size, sigma=sigma)
    if len(frames) < 2:
        # Video too short or unreadable — save zeros
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
    parser = argparse.ArgumentParser(description='Extract physics features from FF++ videos')
    parser.add_argument('--config', default='config/default_config.yaml')
    args = parser.parse_args()

    cfg = load_config(args.config)
    ds = cfg['dataset']
    pp = cfg['preprocessing']

    dataset_root = ds['root']
    compression = ds['compression']
    features_dir = ds['features_dir']
    feature_dim = cfg['extractors']['feature_dim']

    n_frames = pp['n_frames']
    target_size = tuple(pp['target_size'])
    sigma = pp['gaussian_blur_sigma']

    # --- Discover all videos ---
    print("\n=== Discovering REAL videos ===")
    real_videos = discover_videos(dataset_root, ds['real_sources'], compression)

    print("\n=== Discovering FAKE videos ===")
    fake_videos = discover_videos(dataset_root, ds['fake_sources'], compression)

    print(f"\nTotal: {len(real_videos)} real + {len(fake_videos)} fake = "
          f"{len(real_videos) + len(fake_videos)} videos")

    # --- Extract features ---
    all_videos = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]

    t0 = time.time()
    extracted, skipped = 0, 0

    for video_path, label in tqdm(all_videos, desc='Extracting features'):
        out_path = get_feature_output_path(video_path, dataset_root, features_dir)
        was_new = extract_and_save(
            video_path, out_path,
            n_frames=n_frames, target_size=target_size,
            sigma=sigma, feature_dim=feature_dim,
        )
        if was_new:
            extracted += 1
        else:
            skipped += 1

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s  |  Extracted: {extracted}  |  Skipped (cached): {skipped}")

    # --- Save a label manifest for the dataset loader ---
    manifest_path = os.path.join(features_dir, 'manifest.csv')
    with open(manifest_path, 'w') as f:
        f.write('npz_path,label\n')
        for video_path, label in all_videos:
            out_path = get_feature_output_path(video_path, dataset_root, features_dir)
            rel_out = os.path.relpath(out_path, features_dir)
            f.write(f'{rel_out},{label}\n')
    print(f"Manifest saved to {manifest_path}  ({len(all_videos)} entries)")


if __name__ == '__main__':
    main()
