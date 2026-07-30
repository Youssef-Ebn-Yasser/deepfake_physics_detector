import os
import sys
import glob
import argparse
import time
import numpy as np
import yaml
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from data.preprocessing import preprocess_video
from subsystem3.feature_extractor3 import Subsystem3FeatureExtractor

def load_config(config_path='config/default_config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def discover_videos(dataset_root, sources, compression='c40'):
    videos = []
    for source in sources:
        video_dir = os.path.join(dataset_root, source, compression, 'videos')
        if not os.path.isdir(video_dir):
            continue
        videos.extend(sorted(glob.glob(os.path.join(video_dir, '*.mp4'))))
    return videos

def get_feature_output_path(video_path, dataset_root, features_dir):
    rel = os.path.relpath(video_path, dataset_root)
    parts = rel.replace('\\', '/').split('/')
    category_parts = [p for p in parts[:-1] if p not in ('c23', 'c40', 'videos')]
    category = '_'.join(category_parts)
    stem = os.path.splitext(parts[-1])[0]
    return os.path.join(features_dir, 'subsystem3', category, f'{stem}_physics64.npy')

def main():
    parser = argparse.ArgumentParser()
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

    real_videos = discover_videos(dataset_root, ds['real_sources'], compression)
    fake_videos = discover_videos(dataset_root, ds['fake_sources'], compression)
    all_videos = [(v, 0) for v in real_videos] + [(v, 1) for v in fake_videos]

    extractor = Subsystem3FeatureExtractor()

    for video_path, label in tqdm(all_videos, desc='Extracting subsystem3 features'):
        out_path = get_feature_output_path(video_path, dataset_root, features_dir)
        if os.path.exists(out_path):
            continue
            
        frames = preprocess_video(video_path, n_frames=n_frames, target_size=target_size, sigma=sigma)
        physics64 = extractor.extract_from_frames(frames, feature_dim=feature_dim)
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        np.save(out_path, physics64)

if __name__ == '__main__':
    main()
