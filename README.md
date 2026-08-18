<div align="center">

# Spherical Steering: Geometry-Aware Activation Rotation for Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2602.08169-b31b1b.svg)](https://arxiv.org/abs/2602.08169) [![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

</div>

---

*Spherical Steering* is an inference-time activation steering method for controlling language models via geometry-consistent interventions. Instead of the standard activation addition, Spherical Steering performs a rotation: it treats steering as a directional update in representation space and rotates hidden activations along a geodesic toward a target direction, while keeping activation magnitudes intact.

This code base contains the code to replicate the experiments presented in the paper "Spherical Steering: Geometry-Aware Activation Rotation for Language Models".

## Table of Contents

- [Preparation](#preparation)
- [Data](#data)
- [Usage](#usage)
- [Key Modules](#key-modules)
- [Citation](#citation)
- [Acknowledgements](#acknowledgements)

## Preparation

### Environment

- Python `3.10`
- Environment file: `environment.yml`

Create and activate:

```bash
conda env create -f environment.yml
conda activate spherical-steering
```

### TruthfulQA Repo

`evaluate_mc.py` imports evaluation metric from `./TruthfulQA`, so clone this repo first:

```bash
git clone https://github.com/sylinrl/TruthfulQA.git
```

## Data

- Main benchmark: `truthful_qa` (https://github.com/sylinrl/TruthfulQA).
- MC evaluation CSV default path: `./TruthfulQA/data/v1/TruthfulQA.csv`.
- Intermediate artifacts are written to:
  - `features/`
  - `prototypes/`
  - `results/`
  - `results_llm_judge/`
- Other benchmarks are under `./generic`.
- See `generic/README.md` for details.

## Usage

### Steering geometries

The original spherical method normalizes an activation `h` and rotates it on a
Euclidean sphere. It remains the default and its implementation is unchanged.
Mahalanobis ellipsoidal steering estimates a center `c` and covariance `Sigma`
from training questions only. The powered, scale-normalized map is
`z = W_p(h-c)`, where `W_p = tau^p Sigma^(-p)` and
`tau = tr(Sigma)/d`. Steering follows the same spherical geodesic in `z` space
and applies `W_p^{-1}` afterward. Thus `p=0` is the identity geometry and
`p=0.5` is full whitening; every value preserves `||z||` numerically.

The center is independently configurable as `zero`, `global`, or
`class-midpoint`. Pooled covariance always uses class-centered residuals, so a
center sweep changes the steering origin without adding the class mean
difference to covariance. Global covariance instead uses residuals about the
selected center.

Covariance controls ellipsoid axes: eigenvectors determine orientation and the
square roots of eigenvalues determine axis sizes. `ellipsoid-diag` stores only
per-coordinate variances and is the recommended low-memory baseline.
`ellipsoid-lowrank` stores leading axis directions plus an isotropic residual;
it captures correlations without allocating a dense hidden-dimension-squared
matrix, at additional fitting and inference cost. Pooled within-class
covariance is recommended because the class mean difference does not inflate an
axis. The ellipsoid is a calibration-data geometry estimator, not a learned
model.

For matched TruthfulQA calibration, extract every scored answer token. An
answer of length `L` gives each token weight `1/L`, so each answer has equal
total fitting weight. Prompt construction, scored positions, and hook bounds
are shared with evaluation.

Extract scored activations once, then create diagonal artifacts:

```bash
python get_activations.py Qwen2.5-3B-Instruct --layer 19 \
  --activation-positions scored --feature-dtype float16 --save_dir ./features
python get_prototypes.py \
  --feature_file ./features/Qwen2.5-3B-Instruct_layer19_scored.npz \
  --save_dir ./prototypes_ellipsoid_diag \
  --geometry ellipsoid-diag --covariance-source pooled \
  --center-mode global --whitening-power 0.5 \
  --shrinkage 0.1 --variance-floor 1e-5
python evaluate_mc.py Qwen2.5-3B-Instruct \
  --prototype_path ./prototypes_ellipsoid_diag/Qwen2.5-3B-Instruct_layer19_fold0.npz \
  --steering-geometry auto --layer 19 --kappa 20 --alpha 0.8 --beta -0.8
```

For low rank, replace the prototype command with:

```bash
python get_prototypes.py \
  --feature_file ./features/llama3.1-8B-Instruct_layer14.npz \
  --save_dir ./prototypes_ellipsoid_lowrank \
  --geometry ellipsoid-lowrank --covariance-source pooled \
  --cov-rank 128 --shrinkage 0.1 --variance-floor 1e-5
```

Evaluation modes are baseline (`--no_intervention`), legacy/default spherical
(omit new arguments), explicit spherical (`--steering-geometry sphere`), and
ellipsoidal (`--steering-geometry ellipsoid`). `auto` is the default and reads
artifact metadata. An explicit incompatible choice is rejected. Tune layer,
`kappa`, `alpha`, `beta`, covariance settings, and rank only on training or
validation data—not the held-out test fold.

For example, the corresponding evaluation forms are:

```bash
# Baseline
python evaluate_mc.py llama3.1-8B-Instruct --prototype_path prototypes/sphere_fold0.npz --no_intervention
# Spherical
python evaluate_mc.py llama3.1-8B-Instruct --prototype_path prototypes/sphere_fold0.npz --steering-geometry sphere
# Diagonal ellipsoid
python evaluate_mc.py llama3.1-8B-Instruct --prototype_path prototypes_ellipsoid_diag/diag_fold0.npz --steering-geometry ellipsoid
# Low-rank ellipsoid
python evaluate_mc.py llama3.1-8B-Instruct --prototype_path prototypes_ellipsoid_lowrank/lowrank_fold0.npz --steering-geometry ellipsoid
```

Version-2 artifacts contain `artifact_version`, `geometry`,
`covariance_source`, `shrinkage`, `variance_floor`, `center`, `mu_T`, `mu_H`,
`center_mode`, `whitening_power`, `tau`, feature metadata, fold metadata, axis
sizes, and training-radius quantiles. Diagonal artifacts
add `diag_var`; low-rank artifacts add `basis`, `eigvals`, `residual_var`, and
`cov_rank`. Ellipsoidal prototypes live in whitened coordinates. Legacy files
containing only prototypes and fold metadata are treated as spherical. Radius
quantiles are diagnostics only; every activation retains its own Mahalanobis
radius rather than being projected onto a fixed boundary.

Run the staged validation-only search (center, then power, then covariance):

```bash
python sweep_ellipsoid.py --model-name Qwen2.5-3B-Instruct --stage all \
  --feature-file ./features/Qwen2.5-3B-Instruct_layer19_scored.npz \
  --layer 19 --fold 0 --output-dir ./sweeps/qwen3b_fold0 \
  --kappa 20 --alpha 0.8 --beta -0.8
```

`--dry-run` prints configurations and `--resume` skips completed hashes. The
sweep evaluates validation only, caches raw spectra, writes JSON/CSV summaries,
freezes the best settings, and prints (without executing) the held-out test
command. With one validation split, the score is
`MC2 - 0.25 * abs(trigger_rate - 0.9)` and the unavailable cross-fold standard
deviation is marked explicitly. Older ellipsoid artifacts default to global
centering and `p=0.5`.

### TruthfulQA

Use the following scripts:

```bash
bash quickstart_llama.sh
bash quickstart_qwen.sh
```

For a paper-style methodology adapted to Qwen2.5-3B-Instruct, reserve validation
questions inside each randomized development fold:

```bash
python get_activations.py Qwen2.5-3B-Instruct --layer 19 --save_dir ./features
python get_prototypes.py \
  --feature_file ./features/Qwen2.5-3B-Instruct_layer19.npz \
  --save_dir ./prototypes_qwen3b_paper \
  --geometry sphere --num_folds 2 \
  --shuffle-folds --seed 42 --validation-fraction 0.2
```

Use `--eval-split validation` to select the layer and steering parameters, then
run the chosen configuration once with `--eval-split test`. Fit separate
artifacts for every candidate layer. Never select parameters from test metrics.
The 3B model is an adapted replication; the paper's reported numbers use
Qwen2.5-7B-Instruct and LLaMA-3.1-8B-Instruct.

Current quickstart defaults:

| Script | Model | Layer | `kappa` | `alpha` | `beta` |
|---|---|---|---|---|---|
| `quickstart_llama.sh` | `llama3.1-8B-Instruct` | 14 | 20.0 | 0.7 | -0.15 |
| `quickstart_qwen.sh` | `Qwen2.5-7B-Instruct` | 19 | 20.0 | 0.6 | 0.4 |

Spherical Steering is also applicable to smaller models (e.g., `Llama3.2-1B`, `Qwen2.5-3B-Instruct`) and larger models (e.g., `Qwen2.5-32B-Instruct`, `gpt-oss-20b`).

### Other Reasoning Benchmarks

For other reasoning multiple-choice benchmarks, use the pipeline in:

```bash
cd generic
```

See `generic/README.md` for details.

## Key Modules

| Module | Description |
|---|---|
| `get_activations.py` | Extract last-token hidden states from answer pairs |
| `get_prototypes.py` | Compute `mu_T`, `mu_H` with 2-fold question-level split |
| `evaluate_mc.py` | MC1/MC2/MC3 evaluation on held-out fold questions |
| `evaluate_llm_judge.py` | Open-ended generation + truth/info judge scoring |
| `spherical_steering.py` | Intervention hook and geometric steering logic |
| `ellipsoid_geometry.py` | Training-only covariance fitting and matrix-free whitening |
| `ellipsoidal_steering.py` | Mahalanobis ellipsoidal intervention hooks |
| `steering_artifacts.py` | Safe shared artifact loader and intervention factory |
| `utils.py` | Data loading and activation extraction helpers |

The isolated [`ellipsoid_steering`](ellipsoid_steering/README.md) package adds
the training-free LVLM hallucination-mode framework, including streaming
grounded covariance fitting, displacement whitening, routed spectral
suppression, two-pass hook control, cache validation, and synthetic invariant
tests. It does not modify the TruthfulQA or generic benchmark pipelines.

Its [Qwen2.5-VL COCO/CHAIR pilot](ellipsoid_steering/README.md#qwen25-vl--cocochair-pilot)
provides a reproducible 100-image fit / 100-image evaluation workflow, matched
original-versus-blurred activation collection, CHAIR-format caption generation,
and an official-scorer wrapper.

## Citation

If you find this work useful, please cite:

```bibtex
@misc{you2026sphericalsteeringgeometryawareactivation,
      title={Spherical Steering: Geometry-Aware Activation Rotation for Language Models}, 
      author={Zejia You and Chunyuan Deng and Hanjie Chen},
      year={2026},
      eprint={2602.08169},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.08169}, 
}
```

## Acknowledgements

- [baukit](https://github.com/davidbau/baukit)
- [TruthfulQA](https://github.com/sylinrl/TruthfulQA)
- [ITI](https://github.com/likenneth/honest_llama)
