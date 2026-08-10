"""
Step 2: Compute Contrastive Prototypes

This script computes truthful (μ_T) and hallucination (μ_H) prototypes using
the difference vector method. This creates antipodal prototypes on the unit
sphere for maximum vMF discriminability.

The script uses K-Fold cross-validation at the question level to ensure
no data leakage between training and test sets.

Usage:
    python get_prototypes.py --feature_file ./features/llama3.1-8B_layer14.npz --save_dir ./prototypes

Output:
    Saves one .npz file per fold containing:
    - mu_T: Truthful prototype [hidden_dim]
    - mu_H: Hallucination prototype [hidden_dim]
    - test_q_indices: Question indices in the test fold
    - fold_idx: Fold index
    - train_accuracy: Classification accuracy on training set
    - test_accuracy: Classification accuracy on test set
"""

import argparse
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from ellipsoid_geometry import fit_ellipsoid_geometry


def normalize(v):
    """Normalize a vector to unit length."""
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def compute_contrastive_prototypes(X_train, y_train, sample_weights=None):
    """
    Compute contrastive prototypes using the difference vector method.
    
    This approach creates antipodal prototypes (μ_T = -μ_H) that maximize
    the separation between truthful and hallucination directions.
    
    Args:
        X_train: Training activations [N, D]
        y_train: Training labels [N] (1=truthful, 0=hallucination)
    
    Returns:
        mu_T: Truthful prototype (unit vector) [D]
        mu_H: Hallucination prototype (unit vector) [D]
        cos_sim: Cosine similarity between prototypes (should be -1.0)
    """
    X_true = X_train[y_train == 1]
    X_false = X_train[y_train == 0]
    
    weights = np.ones(len(X_train)) if sample_weights is None else np.asarray(sample_weights)
    mean_true = np.average(X_true, axis=0, weights=weights[y_train == 1])
    mean_false = np.average(X_false, axis=0, weights=weights[y_train == 0])
    
    # Compute difference vector (truthful direction)
    diff_vec = mean_true - mean_false
    
    # Normalize to create unit prototypes
    mu_T = normalize(diff_vec)
    mu_H = -mu_T  # Antipodal prototype
    
    # Verify they are antipodal
    cos_sim = np.dot(mu_T, mu_H)
    
    return mu_T, mu_H, cos_sim


def load_sample_weights(data, n_samples):
    """Load token weights, preserving uniform weighting for legacy features."""
    if 'sample_weights' not in data:
        return np.ones(n_samples, dtype=np.float64)
    weights = np.asarray(data['sample_weights'], dtype=np.float64)
    if weights.shape != (n_samples,) or not np.all(np.isfinite(weights)) or np.any(weights < 0):
        raise ValueError("sample_weights must be a finite non-negative vector matching activations")
    return weights


def split_question_folds(q_indices, num_folds=2, validation_fraction=0.0,
                         shuffle_folds=False, seed=42):
    """Yield fit/validation/test question IDs without overlap."""
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must satisfy 0 <= value < 1")
    unique_questions = np.unique(q_indices)
    kf = KFold(n_splits=num_folds, shuffle=shuffle_folds,
               random_state=seed if shuffle_folds else None)
    for fold_idx, (development_idx, test_idx) in enumerate(kf.split(unique_questions)):
        development_qs = unique_questions[development_idx]
        test_qs = unique_questions[test_idx]
        if validation_fraction:
            fit_qs, validation_qs = train_test_split(
                development_qs, test_size=validation_fraction,
                random_state=seed + fold_idx, shuffle=True)
        else:
            fit_qs, validation_qs = development_qs, np.array([], dtype=unique_questions.dtype)
        yield fold_idx, np.sort(fit_qs), np.sort(validation_qs), np.sort(test_qs)


def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Compute contrastive prototypes with K-Fold CV"
    )
    parser.add_argument(
        '--feature_file', 
        type=str, 
        required=True,
        help="Path to feature .npz file from step 1"
    )
    parser.add_argument(
        '--num_folds', 
        type=int, 
        default=2,
        help="Number of folds for cross-validation"
    )
    parser.add_argument(
        '--save_dir', 
        type=str, 
        default='./prototypes',
        help="Directory to save prototypes"
    )
    parser.add_argument('--geometry', choices=['sphere', 'ellipsoid-diag', 'ellipsoid-lowrank'],
                        default='sphere')
    parser.add_argument('--covariance-source', choices=['pooled', 'global'], default='pooled')
    parser.add_argument('--shrinkage', type=float, default=0.1)
    parser.add_argument('--variance-floor', type=float, default=1e-5)
    parser.add_argument('--cov-rank', type=int, default=128)
    parser.add_argument('--radius-quantiles', default='0.5,0.9,0.95,0.99')
    parser.add_argument('--validation-fraction', type=float, default=0.0,
                        help='Fraction of each development fold reserved for tuning')
    parser.add_argument('--shuffle-folds', action='store_true',
                        help='Randomize outer question folds reproducibly')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--center-mode', choices=['zero', 'global', 'class-midpoint'], default='global')
    parser.add_argument('--whitening-power', type=float, default=0.5)
    
    args = parser.parse_args()
    try:
        radius_quantiles = tuple(float(x) for x in args.radius_quantiles.split(','))
    except ValueError as exc:
        raise ValueError("--radius-quantiles must be comma-separated numbers") from exc
    
    # Load features
    print(f"Loading features from {args.feature_file}...")
    data = np.load(args.feature_file)
    X = data['activations']
    y = data['labels']
    q_indices = data['q_indices']
    sample_weights = load_sample_weights(data, len(X))
    if not 0.0 <= args.whitening_power <= 0.5:
        raise ValueError("--whitening-power must lie in [0, 0.5]")
    
    print(f"Loaded {len(X)} samples with {X.shape[1]} dimensions")
    
    # Setup K-Fold at question level
    unique_questions = np.unique(q_indices)
    print(f"Total unique questions: {len(unique_questions)}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Get base name for output files
    base_name = os.path.basename(args.feature_file).replace('.npz', '')
    
    folds = split_question_folds(
        q_indices, args.num_folds, args.validation_fraction,
        args.shuffle_folds, args.seed)
    for fold_idx, train_qs, validation_qs, test_qs in folds:
        print(f"\n{'='*60}")
        print(f"Processing Fold {fold_idx + 1}/{args.num_folds}")
        print(f"{'='*60}")
        
        # Create masks. Geometry is fitted from train_mask only.
        train_mask = np.isin(q_indices, train_qs)
        X_train = X[train_mask]
        y_train = y[train_mask]
        weights_train = sample_weights[train_mask]
        print(f"Train: {len(X_train)} samples from {len(train_qs)} questions")
        print(f"Validation: {len(validation_qs)} questions")
        print(f"Test: {len(test_qs)} questions")
        
        # Fit exclusively on this fold's training activations.
        if args.geometry == 'sphere':
            mu_T, mu_H, _ = compute_contrastive_prototypes(X_train, y_train, weights_train)
            artifact = dict(mu_T=mu_T, mu_H=mu_H,
                            center=np.average(X_train, axis=0, weights=weights_train))
        else:
            fitted = fit_ellipsoid_geometry(
                X_train, y_train, geometry=args.geometry,
                covariance_source=args.covariance_source,
                shrinkage=args.shrinkage, variance_floor=args.variance_floor,
                cov_rank=args.cov_rank, radius_quantiles=radius_quantiles,
                sample_weights=weights_train, center_mode=args.center_mode,
                whitening_power=args.whitening_power)
            artifact = {k: v for k, v in fitted.items() if k not in ('m_T', 'm_H')}
        
        # Save prototypes
        save_path = os.path.join(args.save_dir, f"{base_name}_fold{fold_idx}.npz")
        artifact.update(
            artifact_version=np.array(2, dtype=np.int64),
            geometry=np.array(args.geometry),
            covariance_source=np.array(args.covariance_source),
            shrinkage=np.array(args.shrinkage, dtype=np.float32),
            variance_floor=np.array(args.variance_floor, dtype=np.float32),
            cov_rank=np.array(artifact.get('cov_rank', 0), dtype=np.int64),
            train_q_indices=train_qs, validation_q_indices=validation_qs,
            test_q_indices=test_qs, fold_idx=np.array(fold_idx, dtype=np.int64),
            split_seed=np.array(args.seed, dtype=np.int64),
            validation_fraction=np.array(args.validation_fraction, dtype=np.float32),
            shuffled_folds=np.array(args.shuffle_folds))
        artifact.update(
            center_mode=np.array(args.center_mode),
            whitening_power=np.array(args.whitening_power, dtype=np.float32),
            activation_positions=np.array(str(data['activation_positions'].item()) if 'activation_positions' in data else 'last'),
            prompt_format=np.array(str(data['prompt_format'].item()) if 'prompt_format' in data else 'legacy'))
        artifact = {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and
                         np.issubdtype(v.dtype, np.floating) else v)
                    for k, v in artifact.items()}
        np.savez(save_path, **artifact)
        print(f"Saved to {save_path}")
    
    print(f"\n{'='*60}")
    print("Step 2 Complete!")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
