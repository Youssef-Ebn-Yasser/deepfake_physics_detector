"""
plot_features.py
----------------
Visualizes the extracted feature vectors for Sub-System 1 and Sub-System 2 using t-SNE.
"""

import os
import sys
import argparse
import numpy as np
from sklearn.manifold import TSNE

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from data.dataset_loader import load_manifest


def load_feature_vector(npz_path):
    """Load f1, f2, f3, f4, f5 from a .npz file."""
    data = np.load(npz_path)
    f1 = data['f1'].astype(np.float32)
    f2 = data['f2'].astype(np.float32)
    f3 = data['f3'].astype(np.float32)
    f4 = data['f4'].astype(np.float32)
    f5 = data['f5'].astype(np.float32) if 'f5' in data else np.zeros(64, dtype=np.float32)
    return f1, f2, f3, f4, f5


def main():
    parser = argparse.ArgumentParser(description='Plot Feature Distributions')
    parser.add_argument('--features_dir', default='DataSets/features',
                        help='Path to features dir')
    parser.add_argument('--num_samples', type=int, default=500,
                        help='Number of samples per class to plot')
    parser.add_argument('--out_dir', default='results',
                        help='Output directory for plots')
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"Loading manifest from {args.features_dir}...")
    entries = load_manifest(args.features_dir)

    real_entries = [e for e in entries if e[1] == 0]
    fake_entries = [e for e in entries if e[1] == 1]

    np.random.seed(42)
    np.random.shuffle(real_entries)
    np.random.shuffle(fake_entries)

    n = min(args.num_samples, len(real_entries), len(fake_entries))
    selected_entries = real_entries[:n] + fake_entries[:n]

    print(f"Loading features for {n} real and {n} fake samples...")

    X_sys1, X_sys2 = [], []
    labels = []

    for path, label in selected_entries:
        try:
            f1, f2, f3, f4, f5 = load_feature_vector(path)

            sys1_feat = np.concatenate([f1, f2, f5])   # 192-D
            sys2_feat = np.concatenate([f3, f4])        # 128-D

            X_sys1.append(sys1_feat)
            X_sys2.append(sys2_feat)
            labels.append(label)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            continue

    X_sys1 = np.array(X_sys1)
    X_sys2 = np.array(X_sys2)
    labels = np.array(labels)

    print("Running t-SNE dimensionality reduction (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=42)

    print("  -> Sub-System 1 (Hardware/Optics + FFT)")
    X_sys1_2d = tsne.fit_transform(X_sys1)

    print("  -> Sub-System 2 (Biological/Lighting)")
    X_sys2_2d = tsne.fit_transform(X_sys2)

    print("Generating plots...")

    def plot_tsne(X_2d, y, title, filename):
        plt.figure(figsize=(8, 6))
        plt.scatter(X_2d[y == 0, 0], X_2d[y == 0, 1], c='blue', label='Real (0)', alpha=0.6, s=15)
        plt.scatter(X_2d[y == 1, 0], X_2d[y == 1, 1], c='red',  label='Fake (1)', alpha=0.6, s=15)
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, filename), dpi=150)
        plt.close()

    plot_tsne(X_sys1_2d, labels, 't-SNE: Sub-System 1 (Optics, Motion & FFT)', 'tsne_sys1.png')
    plot_tsne(X_sys2_2d, labels, 't-SNE: Sub-System 2 (Biology & Lighting)',   'tsne_sys2.png')

    # Combined side-by-side plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].scatter(X_sys1_2d[labels == 0, 0], X_sys1_2d[labels == 0, 1],
                    c='blue', alpha=0.5, s=10, label='Real')
    axes[0].scatter(X_sys1_2d[labels == 1, 0], X_sys1_2d[labels == 1, 1],
                    c='red',  alpha=0.5, s=10, label='Fake')
    axes[0].set_title('Sub-System 1 (Hardware + FFT)')
    axes[0].legend()

    axes[1].scatter(X_sys2_2d[labels == 0, 0], X_sys2_2d[labels == 0, 1],
                    c='blue', alpha=0.5, s=10, label='Real')
    axes[1].scatter(X_sys2_2d[labels == 1, 0], X_sys2_2d[labels == 1, 1],
                    c='red',  alpha=0.5, s=10, label='Fake')
    axes[1].set_title('Sub-System 2 (Biological)')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'tsne_combined.png'), dpi=200)
    plt.close()

    print(f"Done! Plots saved to {args.out_dir}")


if __name__ == '__main__':
    main()
