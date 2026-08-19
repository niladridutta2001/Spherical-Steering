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

Run the complete 24-configuration matched-token geometry sweep with:

```bash
python sweep_ellipsoid_generic.py \
  --model-name Qwen2.5-3B-Instruct \
  --feature-file ./features_generic/Qwen2.5-3B-Instruct_winogrande_train_l19_scored.npz \
  --layer 19 --stage all --output-dir ./sweeps_winogrande_geometry \
  --kappa 20 --alpha 0.8 --beta -0.8
```

It compares three centers, five whitening powers, and sixteen rank/shrinkage
combinations on only the 200-question development-validation subset. It saves
JSON/CSV summaries, supports `--resume`, and prints without executing the
frozen command for all 1,267 official validation questions.

### Category-balanced MMLU protocol

MMLU subjects are grouped into STEM, Humanities, Social Sciences, and Other.
For each category, the deterministically shuffled official test questions are
partitioned as follows: the first 500 form development data, split into 400 fit
and 100 validation questions; the next 200 are frozen evaluation questions.
Thus fitting uses 1,600 questions, validation uses 400, and final evaluation
uses 800. Candidate activations are extracted at exactly the token positions
whose conditional log-likelihoods are used during evaluation.

```bash
python get_activations_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset mmlu_global --split train --num_samples 500 --layer 19 \
  --activation-positions scored --feature-dtype float16 --seed 42 \
  --save_dir ./features_generic

python get_prototypes_generic.py \
  --feature_file ./features_generic/Qwen2.5-3B-Instruct_mmlu_global_train_l19_scored.npz \
  --save_dir ./prototypes_generic --geometry ellipsoid-lowrank \
  --covariance-source pooled --center-mode class-midpoint \
  --whitening-power 0.25 --cov-rank 64 --shrinkage 0.1 \
  --validation-fraction 0.2 --split-seed 42
```

Tune only on the balanced 400-question validation set:

```bash
python evaluate_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset mmlu_global --eval-split dev-validation --layer 19 \
  --prototype_path ./prototypes_generic/Qwen2.5-3B-Instruct_mmlu_global_train_l19_scored_proto.npz \
  --steering-geometry ellipsoid --kappa 20 --alpha 0.8 --beta -0.8
```

After freezing all settings, replace `dev-validation` with `evaluation` to run
once on the disjoint 800-question evaluation set. The evaluator prints the
four category accuracies and their balanced micro-average.

### Paper-compatible COPA split

The 400 examples in the official COPA training split are shuffled with seed 42
and divided question-wise into 320 fitting and 80 development-validation
examples. The official 100-example validation split remains untouched and is
used as the final test set. Both candidates are calibrated at the exact token
positions used by conditional-likelihood evaluation.

```bash
python get_activations_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset copa --split train --layer 19 --activation-positions scored \
  --feature-dtype float16 --seed 42 --save_dir ./features_generic

python get_prototypes_generic.py \
  --feature_file ./features_generic/Qwen2.5-3B-Instruct_copa_train_l19_scored.npz \
  --save_dir ./prototypes_generic --geometry ellipsoid-lowrank \
  --covariance-source pooled --center-mode class-midpoint \
  --whitening-power 0.25 --cov-rank 64 --shrinkage 0.1 \
  --validation-fraction 0.2 --split-seed 42

python evaluate_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset copa --eval-split dev-validation --layer 19 \
  --prototype_path ./prototypes_generic/Qwen2.5-3B-Instruct_copa_train_l19_scored_proto.npz \
  --steering-geometry ellipsoid --kappa 20 --alpha 0.8 --beta -0.8
```

After selection on the 80 examples, replace `dev-validation` with `official`
for the single final 100-example test run.

Run the complete staged COPA geometry sweep with:

```bash
python sweep_ellipsoid_generic.py --dataset copa \
  --model-name Qwen2.5-3B-Instruct \
  --feature-file ./features_generic/Qwen2.5-3B-Instruct_copa_train_l19_scored.npz \
  --layer 19 --stage all --output-dir ./sweeps_copa_geometry \
  --kappa 20 --alpha 0.8 --beta -0.8
```

It evaluates 24 staged configurations on only the 80 validation examples:
three centers, five whitening powers, then four ranks
`{16,32,64,128}` crossed with four shrinkages. The winning artifact and JSON
summary are saved, and the untouched 100-example command is printed but not
executed. Use `--resume` after a disconnected runtime.

### Paper-compatible BoolQ split

Exactly 1,000 examples are sampled from the shuffled official BoolQ training
split using seed 42. A question-wise 4:1 split provides 800 geometry-fitting
and 200 development-validation examples. The complete official validation set
of 3,270 examples remains untouched for final evaluation. Calibration extracts
the exact `no` and `yes` candidate-token activations scored by evaluation.

```bash
python get_activations_generic.py --model_name Qwen2.5-3B-Instruct \
  --dataset boolq --split train --num_samples 1000 --layer 19 \
  --activation-positions scored --feature-dtype float16 --seed 42 \
  --save_dir ./features_generic

python sweep_ellipsoid_generic.py --dataset boolq \
  --model-name Qwen2.5-3B-Instruct \
  --feature-file ./features_generic/Qwen2.5-3B-Instruct_boolq_train_l19_scored.npz \
  --layer 19 --stage all --output-dir ./sweeps_boolq_geometry \
  --kappa 20 --alpha 0.8 --beta -0.8 --resume
```

The sweep evaluates only the 200 development-validation examples and prints,
without executing, the frozen command for all 3,270 official validation
examples.

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
