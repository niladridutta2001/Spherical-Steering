"""
Step 2 (Generic): Compute Contrastive Prototypes

Computes:  mu_T = normalize(mean(correct) - mean(incorrect)),  mu_H = -mu_T

Usage:
    python get_prototypes_generic.py --feature_file ./features_generic/llama3.1-8B_mmlu_global_train_l14.npz
"""

import argparse
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ellipsoid_geometry import fit_ellipsoid_geometry


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def main():
    parser = argparse.ArgumentParser(description="Step 2: Compute contrastive prototypes")
    parser.add_argument('--feature_file', type=str, required=True, help="Path to feature .npz file")
    parser.add_argument('--save_dir', type=str, default='./prototypes_generic')
    parser.add_argument('--geometry', choices=['sphere', 'ellipsoid-diag', 'ellipsoid-lowrank'], default='sphere')
    parser.add_argument('--covariance-source', choices=['pooled', 'global'], default='pooled')
    parser.add_argument('--shrinkage', type=float, default=0.1)
    parser.add_argument('--variance-floor', type=float, default=1e-5)
    parser.add_argument('--cov-rank', type=int, default=128)
    parser.add_argument('--radius-quantiles', default='0.5,0.9,0.95,0.99')
    args = parser.parse_args()

    data = np.load(args.feature_file)
    X, y = data['activations'], data['labels']
    print(f"Loaded {len(X)} samples ({sum(y)} correct, {len(y)-sum(y)} incorrect), dim={X.shape[1]}")

    if args.geometry == 'sphere':
        diff = np.mean(X[y == 1], axis=0) - np.mean(X[y == 0], axis=0)
        artifact = {'mu_T': normalize(diff), 'mu_H': -normalize(diff),
                    'center': np.mean(X, axis=0)}
    else:
        artifact = fit_ellipsoid_geometry(
            X, y, args.geometry, args.covariance_source, args.shrinkage,
            args.variance_floor, args.cov_rank,
            tuple(float(x) for x in args.radius_quantiles.split(',')))
        artifact = {k: v for k, v in artifact.items() if k not in ('m_T', 'm_H')}
    artifact.update(artifact_version=np.array(2), geometry=np.array(args.geometry),
                    covariance_source=np.array(args.covariance_source),
                    shrinkage=np.array(args.shrinkage, dtype=np.float32),
                    variance_floor=np.array(args.variance_floor, dtype=np.float32),
                    cov_rank=np.array(artifact.get('cov_rank', 0)))
    artifact = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and
                     np.issubdtype(v.dtype, np.floating) else v) for k, v in artifact.items()}

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.basename(args.feature_file).replace('.npz', '')
    save_path = os.path.join(args.save_dir, f"{base_name}_proto.npz")
    np.savez(save_path, **artifact)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
