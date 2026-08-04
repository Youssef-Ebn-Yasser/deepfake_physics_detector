import pandas as pd
import numpy as np

# Load full dataset manifest
df = pd.read_csv('DataSets/features/manifest.csv')

# Extract 500 Real and 500 Fake samples
real_df = df[df['label'] == 0].sample(n=500, random_state=42)
fake_df = df[df['label'] == 1].sample(n=500, random_state=42)

# Combine and shuffle
subset_df = pd.concat([real_df, fake_df]).sample(frac=1.0, random_state=42).reset_index(drop=True)
subset_df.to_csv('DataSets/features/manifest_1000.csv', index=False)

print(f"Successfully generated manifest_1000.csv with {len(subset_df)} samples.")
