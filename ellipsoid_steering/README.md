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

## Qwen2.5-VL + COCO/CHAIR pilot

Use a deterministic 200-image subset of COCO 2014 validation: 100 images fit
the geometry and a disjoint 100 images are evaluated once. The prompt is
`Please describe this image in detail.` and caption decoding is greedy.

Fresh Colab setup and data:

```bash
git clone https://github.com/niladridutta2001/Spherical-Steering.git
cd Spherical-Steering
pip install -U transformers accelerate qwen-vl-utils pycocotools pillow nltk scikit-learn tqdm
mkdir -p data/coco artifacts results cache
wget -q https://images.cocodataset.org/zips/val2014.zip -O data/coco/val2014.zip
wget -q https://images.cocodataset.org/annotations/annotations_trainval2014.zip -O data/coco/annotations.zip
unzip -q data/coco/val2014.zip -d data/coco
unzip -q data/coco/annotations.zip -d data/coco
python scripts/qwen_coco_chair.py make-split \
  --annotations data/coco/annotations/instances_val2014.json \
  --output data/coco/chair_100_100_seed42.json \
  --fit-count 100 --eval-count 100 --seed 42
```

Collect matched text-token activations and fit the geometry:

```bash
python scripts/qwen_coco_chair.py collect \
  --split data/coco/chair_100_100_seed42.json --image-dir data/coco/val2014 \
  --output-dir cache/qwen25vl3b_coco_fit --layers 19
python scripts/collect_grounded_stats.py \
  --activation-chunks 'cache/qwen25vl3b_coco_fit/grounded_*.pt' \
  --output artifacts/qwen25vl3b_coco_cov.pt \
  --model-name Qwen/Qwen2.5-VL-3B-Instruct --dataset-name coco2014-chair-fit100 \
  --target-layers 19 --rank 64 --token-policy text
python scripts/collect_hallucination_deltas.py \
  --paired-chunks 'cache/qwen25vl3b_coco_fit/paired_*.pt' \
  --covariances artifacts/qwen25vl3b_coco_cov.pt \
  --output-dir cache/qwen25vl3b_coco_deltas --mode aligned
python scripts/fit_hallucination_modes.py \
  --delta-chunks 'cache/qwen25vl3b_coco_deltas/*.pt' \
  --covariances artifacts/qwen25vl3b_coco_cov.pt \
  --output artifacts/qwen25vl3b_coco_geometry.pt \
  --num-modes 4 --hallucination-rank 8
```

Generate baseline and ellipsoid captions on the same held-out images:

```bash
python scripts/qwen_coco_chair.py generate \
  --split data/coco/chair_100_100_seed42.json --partition eval \
  --image-dir data/coco/val2014 --output results/qwen25vl3b_baseline_chair.json
python scripts/qwen_coco_chair.py generate \
  --split data/coco/chair_100_100_seed42.json --partition eval \
  --image-dir data/coco/val2014 --output results/qwen25vl3b_ellipsoid_chair.json \
  --geometry artifacts/qwen25vl3b_coco_geometry.pt --layers 19
```

Both files use official CHAIR fields (`image_id`, `caption`). Score them with
the unmodified official implementation:

```bash
git clone https://github.com/LisaAnne/Hallucination.git
python scripts/run_chair.py --captions results/qwen25vl3b_baseline_chair.json \
  --annotation-dir data/coco/annotations --chair-repo ./Hallucination
python scripts/run_chair.py --captions results/qwen25vl3b_ellipsoid_chair.json \
  --annotation-dir data/coco/annotations --chair-repo ./Hallucination
```

Report CHAIRs and CHAIRi (lower is better), along with the frozen split JSON,
prompt, decoding settings, layer, and geometry artifact.
