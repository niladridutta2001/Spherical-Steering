# LVLM Ellipsoid-Constrained Hallucination Steering

This package implements a training-free, model-agnostic steering framework.
Model weights remain frozen. Grounded covariance and hallucination modes are
offline statistics, not learned model parameters.

## Geometry

For each layer, `LowRankCovariance` stores actual principal-component
variances:

```text
Sigma = U diag(lambda) U^T + sigma_perp^2 (I - U U^T)
```

It does not store `lambda - sigma_perp^2`. Whitening, coloring, and displacement
whitening use matrix-free `[D,R]` operations. `whiten_delta` deliberately does
not subtract the activation mean.

The core update is:

```text
q       = Sigma^-1/2 (z - mean)
d       = sum_k alpha_k V_k g(lambda_k) V_k^T q
q_tmp   = q - beta d
q_new   = ||q|| q_tmp / ||q_tmp||
z_new   = mean + Sigma^1/2 q_new
```

## Initial offline workflow

Activation collection is deliberately separated from geometry fitting so a
small model-specific adapter can handle LLaVA, Qwen-VL, InternVL, or another
Hugging Face LVLM without entering the geometry modules.

```bash
python scripts/collect_grounded_stats.py \
  --activation-chunks 'cache/grounded_*.pt' \
  --output covariances.pt --model-name MODEL --dataset-name DATASET \
  --target-layers 12 16 20 24 --rank 128

python scripts/collect_hallucination_deltas.py \
  --paired-chunks 'cache/paired_*.pt' --covariances covariances.pt \
  --output-dir delta_chunks --mode aligned

python scripts/fit_hallucination_modes.py \
  --delta-chunks 'delta_chunks/*.pt' --covariances covariances.pt \
  --output geometry_with_modes.pt --num-modes 8 --hallucination-rank 16
```

Grounded activation chunks contain `{"activations": {layer: tensor}}` and
optional token masks. Paired chunks contain per-layer `grounded` and
`hallucinated` tensors. Local-window mode additionally requires explicit
`grounded_window` and `hallucinated_window`; whole-sequence averaging is never
implicit.

`run_steering.py` accepts a `module:function` factory returning a bundle with
the model, original inputs, perturbed-image inputs, token masks, and generation
arguments. Instance routing is the safe first autoregressive mode: routing is
computed on aligned prompt activations and reused during decoding. Teacher-
forced and position-aware modes fail loudly if masked and original trajectories
are not aligned.

## Synthetic milestone

```bash
python scripts/synthetic_demo.py --plot synthetic_ellipse.png
pytest -q tests/test_lvlm_*.py
```

The tests cover inverse consistency, Mahalanobis-radius preservation, energy
reduction, beta-zero and mask identities, routing, isotropic geometry, hooks,
serialization primitives, and a tiny Hugging Face transformer integration.
