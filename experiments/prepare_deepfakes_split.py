"""
experiments/prepare_deepfakes_split.py
--------------------------------------
Step 2: Create the video-level 70/15/15 train/val/test split.

Reads:  DataSets/deepfakes_feature/manifest.csv
Writes: DataSets/deepfakes_feature/split_train.csv
        DataSets/deepfakes_feature/split_val.csv
        DataSets/deepfakes_feature/split_test.csv

Split logic (video-level, no leakage):
  - Extract a unique video ID from each .npz path (the stem filename, e.g. "000")
  - Group all samples by that video ID
  - Shuffle video IDs, split 70/15/15
  - Assign ALL samples from a video to the same partition

Target: 5000 real + 5000 fake = 10,000 samples
        Train ~7000 | Val ~1500 | Test ~1500
"""

import os
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# ─────────────────────────────────────────────────────────────────────────────
MANIFEST_PATH  = os.path.join('DataSets', 'deepfakes_feature', 'manifest.csv')
OUTPUT_DIR     = os.path.join('DataSets', 'deepfakes_feature')

N_REAL  = 5000
N_FAKE  = 5000
SEED    = 42

TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
# TEST_RATIO  = 0.15  (remainder)
# ─────────────────────────────────────────────────────────────────────────────


def video_id_from_path(npz_rel_path: str) -> str:
    """
    Extract a unique video identifier from the relative .npz path.
    e.g. 'source_videos_part_00/source_videos/000.npz' → 'source_videos_part_00__000'
    This ensures all feature samples from the same source video stay together.
    """
    parts = npz_rel_path.replace('\\', '/').split('/')
    folder = parts[0] if len(parts) > 1 else 'unknown'
    stem   = os.path.splitext(parts[-1])[0]
    return f"{folder}__{stem}"


def split_by_video(df: pd.DataFrame, n_target: int, seed: int) -> pd.DataFrame:
    """
    From a dataframe, sample n_target rows such that the selection is done
    at the video ID level (all rows sharing a video ID are selected together).
    """
    df = df.copy()
    df['_vid_id'] = df['npz_path'].apply(video_id_from_path)

    # Unique video IDs and their sizes
    vid_counts = df.groupby('_vid_id').size().reset_index(name='_cnt')
    rng = np.random.default_rng(seed)
    vid_counts = vid_counts.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    selected_vids = []
    total = 0
    for _, row in vid_counts.iterrows():
        if total >= n_target:
            break
        selected_vids.append(row['_vid_id'])
        total += row['_cnt']

    selected = df[df['_vid_id'].isin(selected_vids)].copy()
    selected.drop(columns=['_vid_id'], inplace=True)
    return selected


def stratified_video_split(df: pd.DataFrame, train_r: float, val_r: float, seed: int):
    """
    Perform a stratified (by label) video-level split into train/val/test.
    Returns three DataFrames.
    """
    df = df.copy()
    df['_vid_id'] = df['npz_path'].apply(video_id_from_path)

    # Get unique video IDs with their labels (a video should only have one label)
    vid_label = df.groupby('_vid_id')['label'].first().reset_index()

    rng = np.random.default_rng(seed)
    train_rows, val_rows, test_rows = [], [], []

    for lbl in [0, 1]:
        vids = vid_label[vid_label['label'] == lbl]['_vid_id'].values
        rng.shuffle(vids)

        n_total = len(vids)
        n_train = int(n_total * train_r)
        n_val   = int(n_total * val_r)

        train_vids = vids[:n_train]
        val_vids   = vids[n_train:n_train + n_val]
        test_vids  = vids[n_train + n_val:]

        train_rows.append(df[df['_vid_id'].isin(train_vids)])
        val_rows.append(df[df['_vid_id'].isin(val_vids)])
        test_rows.append(df[df['_vid_id'].isin(test_vids)])

    train_df = pd.concat(train_rows).drop(columns=['_vid_id']).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    val_df   = pd.concat(val_rows).drop(columns=['_vid_id']).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    test_df  = pd.concat(test_rows).drop(columns=['_vid_id']).sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return train_df, val_df, test_df


def main():
    print(f"Loading manifest: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)

    # Ensure correct column names
    if 'npz_path' not in df.columns:
        df.columns = ['npz_path', 'label']

    print(f"  Total samples in manifest: {len(df)}")
    print(f"  Real (0): {(df['label']==0).sum()}  |  Fake (1): {(df['label']==1).sum()}")

    # ── Balance to N_REAL + N_FAKE ────────────────────────────────────────────
    real_df = df[df['label'] == 0]
    fake_df = df[df['label'] == 1]

    if len(real_df) < N_REAL:
        print(f"  [WARN] Only {len(real_df)} real samples available (need {N_REAL}). Using all.")
        n_real = len(real_df)
    else:
        n_real = N_REAL

    if len(fake_df) < N_FAKE:
        print(f"  [WARN] Only {len(fake_df)} fake samples available (need {N_FAKE}). Using all.")
        n_fake = len(fake_df)
    else:
        n_fake = N_FAKE

    real_selected = split_by_video(real_df, n_real, SEED)
    fake_selected = split_by_video(fake_df, n_fake, SEED + 1)

    # Trim to exact target if over-selected
    real_selected = real_selected.iloc[:n_real]
    fake_selected = fake_selected.iloc[:n_fake]

    balanced_df = pd.concat([real_selected, fake_selected]).sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    print(f"\nBalanced subset: {len(balanced_df)} samples  "
          f"(real={( balanced_df['label']==0).sum()}, fake={(balanced_df['label']==1).sum()})")

    # ── Video-level stratified split ─────────────────────────────────────────
    train_df, val_df, test_df = stratified_video_split(balanced_df, TRAIN_RATIO, VAL_RATIO, SEED)

    print(f"\nSplit results (video-level, no leakage):")
    print(f"  Train : {len(train_df):>5}  (real={( train_df['label']==0).sum()}, fake={(train_df['label']==1).sum()})")
    print(f"  Val   : {len(val_df):>5}  (real={(   val_df['label']==0).sum()}, fake={(val_df['label']==1).sum()})")
    print(f"  Test  : {len(test_df):>5}  (real={(  test_df['label']==0).sum()}, fake={(test_df['label']==1).sum()})")

    # ── Save splits ──────────────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, 'split_train.csv'), index=False)
    val_df.to_csv(  os.path.join(OUTPUT_DIR, 'split_val.csv'),   index=False)
    test_df.to_csv( os.path.join(OUTPUT_DIR, 'split_test.csv'),  index=False)
    print(f"\nSaved split CSVs to {OUTPUT_DIR}/split_{{train,val,test}}.csv")


if __name__ == '__main__':
    main()
