"""
scripts/extract_cross_dataset_features.py
------------------------------------------
Extract all 5 physics-based features (F1–F5) from two external datasets
for zero-shot cross-dataset evaluation:

  1. FaceForensics++ (FF++)
     Real  : DataSets/FaceForensics++/original_sequences/youtube/c40/videos/
     Real  : DataSets/FaceForensics++/original_sequences/actors/c40/videos/
     Fake  : DataSets/FaceForensics++/manipulated_sequences/Deepfakes/c40/videos/
     Fake  : DataSets/FaceForensics++/manipulated_sequences/Face2Face/c40/videos/
     Fake  : DataSets/FaceForensics++/manipulated_sequences/FaceSwap/c40/videos/
     Fake  : DataSets/FaceForensics++/manipulated_sequences/FaceShifter/c40/videos/
     Fake  : DataSets/FaceForensics++/manipulated_sequences/NeuralTextures/c40/videos/

  2. Celeb-DF v2
     Real  : DataSets/Celeb-DF-v2/Celeb-real/
     Real  : DataSets/Celeb-DF-v2/YouTube-real/
     Fake  : DataSets/Celeb-DF-v2/Celeb-synthesis/

Output structure:
  DataSets/cross_dataset_features/
    ff_plus_plus/
      real/  *.npz
      fake/  *.npz
      manifest.csv
    celeb_df/
      real/  *.npz
      fake/  *.npz
      manifest.csv

Each .npz contains: f1, f2, f3, f4, f5  (each 64-dim float32)
manifest.csv columns: npz_path, label  (0=real, 1=fake)

Features are identical to those extracted for DeeperForensics training,
so the same trained scaler can be applied during zero-shot evaluation.

Usage:
    python scripts/extract_cross_dataset_features.py
    python scripts/extract_cross_dataset_features.py --dataset ff_plus_plus
    python scripts/extract_cross_dataset_features.py --dataset celeb_df
    python scripts/extract_cross_dataset_features.py --compression c23
"""

import os
import sys
import glob
import time
import argparse
import numpy as np
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import preprocess_video
from extractors.lens_distortion import extract_lens_distortion_features
from extractors.motion_blur import extract_motion_blur_features
from extractors.biomechanics import extract_biomechanics_features
from extractors.lighting_sh import extract_lighting_sh_features
from extractors.frequency_fft import extract_fft_spectrum

# ─────────────────────────────────────────────────────────────────────────────
# Dataset configurations
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset_config(compression='c40'):
    return {
        'ff_plus_plus': {
            'label': 'FaceForensics++',
            'output_dir': os.path.join('DataSets', 'cross_dataset_features', 'ff_plus_plus'),
            'sources': [
                # (directory, label, recursive)
                (os.path.join('DataSets', 'FaceForensics++', 'original_sequences', 'youtube',
                              compression, 'videos'), 0, False),
                (os.path.join('DataSets', 'FaceForensics++', 'original_sequences', 'actors',
                              compression, 'videos'), 0, False),
                # YouTube-real at root level (some FF++ releases)
                (os.path.join('DataSets', 'FaceForensics++', 'YouTube-real'), 0, True),
                (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences',
                              'Deepfakes', compression, 'videos'), 1, False),
                (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences',
                              'Face2Face', compression, 'videos'), 1, False),
                (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences',
                              'FaceSwap', compression, 'videos'), 1, False),
                (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences',
                              'FaceShifter', compression, 'videos'), 1, False),
                (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences',
                              'NeuralTextures', compression, 'videos'), 1, False),
                (os.path.join('DataSets', 'FaceForensics++', 'manipulated_sequences',
                              'DeepFakeDetection', compression, 'videos'), 1, False),
            ]
        },
        'celeb_df': {
            'label': 'Celeb-DF v2',
            'output_dir': os.path.join('DataSets', 'cross_dataset_features', 'celeb_df'),
            'sources': [
                (os.path.join('DataSets', 'Celeb-DF-v2', 'Celeb-real'), 0, False),
                (os.path.join('DataSets', 'Celeb-DF-v2', 'YouTube-real'), 0, False),
                (os.path.join('DataSets', 'Celeb-DF-v2', 'Celeb-synthesis'), 1, False),
            ]
        }
    }


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_fft_over_frames(frames, feature_dim=64):
    """Average FFT spectrum over all frames."""
    spectra = []
    for frame in frames:
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        spectra.append(extract_fft_spectrum(gray, num_bins=feature_dim))
    if not spectra:
        return np.zeros(feature_dim, dtype=np.float32)
    return np.mean(spectra, axis=0).astype(np.float32)


def extract_and_save(video_path, output_path,
                     n_frames=16, target_size=(256, 256),
                     sigma=0.8, feature_dim=64):
    """
    Extract all 5 features from one video and save as .npz.
    Returns:
        True  — newly extracted
        False — already existed (cache hit)
        None  — failed (error logged, zeros saved)
    """
    if os.path.exists(output_path):
        return False   # cache hit

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    try:
        frames = preprocess_video(
            video_path, n_frames=n_frames,
            target_size=target_size, sigma=sigma
        )

        if len(frames) < 2:
            # Video too short / unreadable → save zeros
            f1 = f2 = f3 = f4 = f5 = np.zeros(feature_dim, dtype=np.float32)
        else:
            f1 = extract_lens_distortion_features(frames, feature_dim)
            f2 = extract_motion_blur_features(frames, feature_dim)
            f3 = extract_biomechanics_features(frames, feature_dim)
            f4 = extract_lighting_sh_features(frames, feature_dim)
            f5 = extract_fft_over_frames(frames, feature_dim)

        np.savez_compressed(output_path, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)
        return True

    except Exception as e:
        print(f"\n  [ERROR] {os.path.basename(video_path)}: {e}")
        f1 = f2 = f3 = f4 = f5 = np.zeros(feature_dim, dtype=np.float32)
        np.savez_compressed(output_path, f1=f1, f2=f2, f3=f3, f4=f4, f5=f5)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Per-dataset pipeline
# ─────────────────────────────────────────────────────────────────────────────

def discover_videos(source_dir, recursive=False):
    """Return sorted list of .mp4 paths under source_dir."""
    if not os.path.isdir(source_dir):
        return []
    if recursive:
        return sorted(glob.glob(os.path.join(source_dir, '**', '*.mp4'), recursive=True))
    else:
        return sorted(glob.glob(os.path.join(source_dir, '*.mp4')))


def npz_path_for_video(video_path, source_dir, output_dir, label):
    """Map a video path to its output .npz path."""
    rel     = os.path.relpath(video_path, source_dir)
    stem    = os.path.splitext(rel)[0].replace(os.sep, '_')
    subfolder = 'real' if label == 0 else 'fake'
    return os.path.join(output_dir, subfolder, f'{stem}.npz')


def process_dataset(ds_key, ds_cfg, n_frames=16, feature_dim=64,
                    sigma=0.8, target_size=(256, 256)):
    """Process all sources in one dataset config."""
    output_dir = ds_cfg['output_dir']
    sources    = ds_cfg['sources']
    ds_label   = ds_cfg['label']

    print(f"\n{'='*60}")
    print(f"  Processing: {ds_label}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}")

    os.makedirs(os.path.join(output_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'fake'), exist_ok=True)

    # ── Collect all videos ─────────────────────────────────────────────────
    all_videos = []   # list of (video_path, output_npz_path, label)
    for source_dir, label, recursive in sources:
        videos = discover_videos(source_dir, recursive)
        if not videos:
            print(f"  [WARN] No videos found in: {source_dir}")
            continue
        tag = 'real' if label == 0 else 'fake'
        print(f"  Found {len(videos):>5} {tag:<4} videos in: {source_dir}")
        for vp in videos:
            npz_path = npz_path_for_video(vp, source_dir, output_dir, label)
            all_videos.append((vp, npz_path, label))

    if not all_videos:
        print(f"  [SKIP] No videos discovered for {ds_label}.")
        return []

    n_real  = sum(1 for _, _, l in all_videos if l == 0)
    n_fake  = sum(1 for _, _, l in all_videos if l == 1)
    print(f"\n  Total: {len(all_videos)} videos  (real={n_real}, fake={n_fake})")

    # ── Extract features ───────────────────────────────────────────────────
    extracted, skipped, errors = 0, 0, 0
    t0 = time.time()

    for video_path, npz_path, label in tqdm(all_videos, desc=f'  {ds_label}'):
        result = extract_and_save(
            video_path, npz_path,
            n_frames=n_frames, target_size=target_size,
            sigma=sigma, feature_dim=feature_dim,
        )
        if result is True:
            extracted += 1
        elif result is False:
            skipped += 1
        else:
            errors += 1

    elapsed = time.time() - t0
    print(f"\n  Done in {elapsed:.1f}s  |  "
          f"Extracted={extracted}  Skipped(cache)={skipped}  Errors={errors}")

    # ── Write manifest.csv ─────────────────────────────────────────────────
    manifest_rows = []
    for video_path, npz_path, label in all_videos:
        if os.path.exists(npz_path):
            rel_npz = os.path.relpath(npz_path, output_dir).replace(os.sep, '/')
            manifest_rows.append(f"{rel_npz},{label}")

    manifest_path = os.path.join(output_dir, 'manifest.csv')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        f.write("npz_path,label\n")
        f.write("\n".join(manifest_rows))

    print(f"  Manifest saved: {manifest_path}  ({len(manifest_rows)} entries)")
    return manifest_rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Extract F1–F5 features from FF++ and Celeb-DF v2 for cross-dataset evaluation'
    )
    parser.add_argument(
        '--dataset', default='all',
        choices=['all', 'ff_plus_plus', 'celeb_df'],
        help='Which dataset to extract (default: all)'
    )
    parser.add_argument(
        '--compression', default='c40',
        choices=['c23', 'c40'],
        help='FF++ compression level to use (default: c40)'
    )
    parser.add_argument('--n_frames',    type=int, default=16,  help='Frames per video (default: 16)')
    parser.add_argument('--feature_dim', type=int, default=64,  help='Feature vector size (default: 64)')
    parser.add_argument('--sigma',       type=float, default=0.8, help='Gaussian blur sigma (default: 0.8)')
    args = parser.parse_args()

    configs = get_dataset_config(args.compression)

    if args.dataset == 'all':
        datasets_to_run = list(configs.keys())
    else:
        datasets_to_run = [args.dataset]

    target_size = (256, 256)

    grand_total = 0
    for ds_key in datasets_to_run:
        rows = process_dataset(
            ds_key, configs[ds_key],
            n_frames=args.n_frames,
            feature_dim=args.feature_dim,
            sigma=args.sigma,
            target_size=target_size,
        )
        grand_total += len(rows)

    print(f"\n\n{'='*60}")
    print(f"  ALL DONE — {grand_total} total .npz files ready for zero-shot evaluation.")
    print(f"  Next step: python experiments/eval_cross_dataset_zeroshot.py")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
