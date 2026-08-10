<div align="center">

### Evaluation Pipeline for Multiple-Choice Reasoning Benchmarks beyond TruthfulQA

</div>

---

## Data

Supported datasets:
- `copa`
- `storycloze`
- `mmlu`
- `winogrande`
- `boolq`

Artifacts are written to:
- `features_generic/`
- `prototypes_generic/`

## Reproduce

### Quick Reproduce (Provided Config Set)

Run the predefined sweep in `quick_set_generic.sh`:

```bash
bash quick_set_generic.sh
```

This script runs 5 datasets with fixed layer/alpha/beta settings:
- BoolQ
- COPA
- StoryCloze
- MMLU
- Winogrande

### Manual Reproduce

Step 1: extract features

```bash
python get_activations_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset storycloze \
  --split train \
  --layer 27
```

Step 2: compute prototypes

```bash
python get_prototypes_generic.py \
  --feature_file ./features_generic/llama3.1-8B-Instruct_storycloze_train_l27.npz
```

Add `--geometry ellipsoid-diag --covariance-source pooled` for diagonal
Mahalanobis geometry, or `--geometry ellipsoid-lowrank --cov-rank 128` for the
matrix-free low-rank representation. Evaluation discovers the artifact type by
default; `--steering-geometry {auto,sphere,ellipsoid}` can make the choice
explicit.

Step 3: evaluate steering

```bash
python evaluate_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset storycloze \
  --layer 27 \
  --prototype_path ./prototypes_generic/llama3.1-8B-Instruct_storycloze_train_l27_proto.npz \
  --kappa 20 --alpha 0.9 --beta -0.7
```

### Paper-compatible WinoGrande split

Sample exactly 1,000 official-training questions, reserve 200 for development
validation, and fit the artifact on the remaining 800:

```bash
python get_activations_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset winogrande --split train --num_samples 1000 --layer 19 \
  --activation-positions scored --feature-dtype float16 \
  --seed 42 --save_dir ./features_generic

python get_prototypes_generic.py \
  --feature_file ./features_generic/Qwen2.5-3B-Instruct_winogrande_train_l19_scored.npz \
  --save_dir ./prototypes_generic --geometry ellipsoid-lowrank \
  --covariance-source pooled --center-mode class-midpoint \
  --whitening-power 0.25 --cov-rank 32 --shrinkage 0.1 \
  --validation-fraction 0.2 --split-seed 42
```

Use only the 200 development-validation questions for hyperparameter selection:

```bash
python evaluate_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset winogrande --eval-split dev-validation --layer 19 \
  --prototype_path ./prototypes_generic/Qwen2.5-3B-Instruct_winogrande_train_l19_scored_proto.npz \
  --steering-geometry ellipsoid --kappa 20 --alpha 0.8 --beta -0.8
```

After freezing the configuration, run once on all 1,267 official validation
questions by replacing `--eval-split dev-validation` with
`--eval-split official`. Candidate scores use summed conditional token
log-likelihood, matching the paper rather than length-normalized likelihood.

Baseline (no steering):

```bash
python evaluate_generic.py \
  --model_name llama3.1-8B-Instruct \
  --dataset mmlu_global \
  --layer 27 \
  --prototype_path ./prototypes_generic/llama3.1-8B-Instruct_storycloze_train_l27_proto.npz \
  --disable_steering
```

## Key Modules

| Module | Description |
|---|---|
| `get_activations_generic.py` | Extract last-token hidden activations for training pairs |
| `get_prototypes_generic.py` | Compute contrastive prototypes from extracted features |
| `evaluate_generic.py` | Evaluate steering on benchmark tasks with MC1-style scoring |
| `utils_generic.py` | Dataset loaders, split rules, prompt formatting, feature extraction helpers |
| `quick_set_generic.sh` | Predefined multi-dataset run script with tuned hyperparameters |
