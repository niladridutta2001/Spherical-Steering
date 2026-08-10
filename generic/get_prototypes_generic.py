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
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ellipsoid_geometry import fit_ellipsoid_geometry


def normalize(v):
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm


def split_development_questions(q_indices, validation_fraction=0.2, seed=42):
    """Return disjoint fit/validation question IDs for development data."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must lie strictly between 0 and 1")
    questions = np.unique(q_indices)
    fit_qs, validation_qs = train_test_split(
        questions, test_size=validation_fraction, random_state=seed, shuffle=True)
    return np.sort(fit_qs), np.sort(validation_qs)


def main():
    parser = argparse.ArgumentParser(description="Step 2: Compute contrastive prototypes")
    parser.add_argument('--feature_file', type=str, required=True, help="Path to feature .npz file")
    parser.add_argument('--save_dir', type=str, default='./prototypes_generic')
    parser.add_argument('--geometry', choices=['sphere', 'ellipsoid-diag', 'ellipsoid-lowrank'], default='sphere')
    parser.add_argument('--covariance-source', choices=['pooled', 'global'], default='pooled')
    parser.add_argument('--shrinkage', type=float, default=0.1)
    parser.add_argument('--variance-floor', type=float, default=1e-5)
    parser.add_argument('--cov-rank', type=int, default=128)
    parser.add_argument('--center-mode', choices=['zero', 'global', 'class-midpoint'],
                        default='global')
    parser.add_argument('--whitening-power', type=float, default=0.5)
    parser.add_argument('--radius-quantiles', default='0.5,0.9,0.95,0.99')
    parser.add_argument('--validation-fraction', type=float, default=0.0,
                        help='Question fraction held out for development validation')
    parser.add_argument('--split-seed', type=int, default=42)
    args = parser.parse_args()

    data = np.load(args.feature_file)
    X, y, q_indices = data['activations'], data['labels'], data['q_indices']
    sample_weights = (np.asarray(data['sample_weights'], dtype=np.float64)
                      if 'sample_weights' in data else np.ones(len(X), dtype=np.float64))
    print(f"Loaded {len(X)} samples ({sum(y)} correct, {len(y)-sum(y)} incorrect), dim={X.shape[1]}")

    if args.validation_fraction:
        fit_qs, validation_qs = split_development_questions(
            q_indices, args.validation_fraction, args.split_seed)
        fit_mask = np.isin(q_indices, fit_qs)
    else:
        fit_qs, validation_qs = np.unique(q_indices), np.array([], dtype=q_indices.dtype)
        fit_mask = np.ones(len(X), dtype=bool)
    X_fit, y_fit, weights_fit = X[fit_mask], y[fit_mask], sample_weights[fit_mask]
    print(f"Fit: {len(fit_qs)} questions | Development validation: {len(validation_qs)} questions")

    if args.geometry == 'sphere':
        mean_true = np.average(X_fit[y_fit == 1], axis=0, weights=weights_fit[y_fit == 1])
        mean_false = np.average(X_fit[y_fit == 0], axis=0, weights=weights_fit[y_fit == 0])
        diff = mean_true - mean_false
        artifact = {'mu_T': normalize(diff), 'mu_H': -normalize(diff),
                    'center': np.average(X_fit, axis=0, weights=weights_fit)}
    else:
        artifact = fit_ellipsoid_geometry(
            X_fit, y_fit, args.geometry, args.covariance_source, args.shrinkage,
            args.variance_floor, args.cov_rank,
            tuple(float(x) for x in args.radius_quantiles.split(',')),
            sample_weights=weights_fit,
            center_mode=args.center_mode,
            whitening_power=args.whitening_power)
        artifact = {k: v for k, v in artifact.items() if k not in ('m_T', 'm_H')}
    artifact.update(artifact_version=np.array(2), geometry=np.array(args.geometry),
                    covariance_source=np.array(args.covariance_source),
                    shrinkage=np.array(args.shrinkage, dtype=np.float32),
                    variance_floor=np.array(args.variance_floor, dtype=np.float32),
                    cov_rank=np.array(artifact.get('cov_rank', 0)),
                    center_mode=np.array(args.center_mode),
                    whitening_power=np.array(args.whitening_power, dtype=np.float32),
                    train_q_indices=fit_qs, validation_q_indices=validation_qs,
                    validation_fraction=np.array(args.validation_fraction, dtype=np.float32),
                    split_seed=np.array(args.split_seed),
                    dataset=np.array(str(data['dataset'].item()) if 'dataset' in data else ''),
                    data_seed=np.array(int(data['data_seed'].item()) if 'data_seed' in data else 42),
                    dev_num_samples=np.array(int(data['dev_num_samples'].item()) if 'dev_num_samples' in data else -1),
                    activation_positions=np.array(str(data['activation_positions'].item()) if 'activation_positions' in data else 'last'),
                    prompt_format=np.array(str(data['prompt_format'].item()) if 'prompt_format' in data else 'legacy'))
    artifact = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and
                     np.issubdtype(v.dtype, np.floating) else v) for k, v in artifact.items()}

    os.makedirs(args.save_dir, exist_ok=True)
    base_name = os.path.basename(args.feature_file).replace('.npz', '')
    save_path = os.path.join(args.save_dir, f"{base_name}_proto.npz")
    np.savez(save_path, **artifact)
    print(f"Saved to {save_path}")


if __name__ == '__main__':
    main()
