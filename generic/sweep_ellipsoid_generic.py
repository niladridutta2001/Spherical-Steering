"""Staged validation-only powered-ellipsoid sweep for generic MC tasks."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ellipsoid_geometry import fit_raw_spectral, derive_lowrank_geometry, fitting_question_hash
from generic.get_prototypes_generic import split_development_questions


CENTERS = ("zero", "global", "class-midpoint")
POWERS = (0.0, 0.125, 0.25, 0.375, 0.5)
RANKS = (32, 64, 128, 256)
COPA_RANKS = (16, 32, 64, 128)
SHRINKAGES = (0.1, 0.3, 0.5, 0.7)
FIELDS = ("stage", "config_hash", "center_mode", "whitening_power", "rank",
          "shrinkage", "accuracy", "trigger_rate", "status", "artifact_path")


def config_hash(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:12]


def configurations(stage, center=None, power=None, ranks=RANKS):
    if stage == "center":
        return [dict(center_mode=c, whitening_power=0.5, rank=128, shrinkage=0.1)
                for c in CENTERS]
    if center is None:
        raise ValueError(f"{stage} stage requires a selected center")
    if stage == "power":
        return [dict(center_mode=center, whitening_power=p, rank=128, shrinkage=0.1)
                for p in POWERS]
    if power is None:
        raise ValueError("covariance stage requires a selected whitening power")
    return [dict(center_mode=center, whitening_power=power, rank=r, shrinkage=s)
            for r in ranks for s in SHRINKAGES]


def best_row(rows, stage):
    valid = [r for r in rows if r.get("stage") == stage and r.get("status") == "complete"]
    if not valid:
        return None
    return max(valid, key=lambda r: (float(r["accuracy"]),
                                    -abs(float(r.get("trigger_rate", 0.9)) - 0.9)))


def write_summary(rows, output):
    (output / "summary.json").write_text(json.dumps(rows, indent=2))
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in FIELDS})


def save_artifact(path, geometry, cfg, args, fit_qs, validation_qs, metadata, h):
    values = {k: v for k, v in geometry.items() if k not in ("m_T", "m_H")}
    values.update(
        artifact_version=np.array(3), geometry=np.array("ellipsoid-lowrank"),
        covariance_source=np.array(args.covariance_source),
        center_mode=np.array(cfg["center_mode"]),
        whitening_power=np.array(cfg["whitening_power"], dtype=np.float32),
        shrinkage=np.array(cfg["shrinkage"], dtype=np.float32),
        variance_floor=np.array(args.variance_floor, dtype=np.float32),
        cov_rank=np.array(cfg["rank"]), train_q_indices=fit_qs,
        validation_q_indices=validation_qs, dataset=np.array(args.dataset),
        data_seed=np.array(metadata["data_seed"]),
        dev_num_samples=np.array(metadata["dev_num_samples"]),
        split_seed=np.array(args.split_seed), validation_fraction=np.array(0.2),
        activation_positions=np.array(metadata["activation_positions"]),
        prompt_format=np.array(metadata["prompt_format"]), config_hash=np.array(h))
    values = {k: (v.astype(np.float32) if isinstance(v, np.ndarray)
                  and np.issubdtype(v.dtype, np.floating) else v)
              for k, v in values.items()}
    np.savez(path, **values)


def evaluate(artifact, args):
    command = [sys.executable, "evaluate_generic.py", "--model_name", args.model_name,
               "--dataset", args.dataset, "--eval-split", "dev-validation",
               "--layer", str(args.layer), "--prototype_path", str(artifact),
               "--steering-geometry", "ellipsoid", "--kappa", str(args.kappa),
               "--alpha", str(args.alpha), "--beta", str(args.beta)]
    if args.model_dir:
        command.extend(["--model_dir", args.model_dir])
    run = subprocess.run(command, cwd=Path(__file__).parent, text=True,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(run.stdout)
    accuracy = re.search(r"Accuracy:\s+([0-9.]+)", run.stdout)
    trigger = re.search(r"Steered=.*\(([0-9.]+)%\)", run.stdout)
    if run.returncode or accuracy is None:
        return None, None
    return float(accuracy.group(1)), (float(trigger.group(1))/100 if trigger else None)


def run_stage(stage, args, rows, output, selected_center=None, selected_power=None):
    with np.load(args.feature_file, allow_pickle=False) as data:
        X = np.asarray(data["activations"])
        y = np.asarray(data["labels"])
        q = np.asarray(data["q_indices"])
        weights = np.asarray(data["sample_weights"]) if "sample_weights" in data else np.ones(len(X))
        metadata = {
            "dataset": str(data["dataset"].item()) if "dataset" in data else "",
            "data_seed": int(data["data_seed"].item()) if "data_seed" in data else 42,
            "dev_num_samples": int(data["dev_num_samples"].item()) if "dev_num_samples" in data else -1,
            "activation_positions": str(data["activation_positions"].item()) if "activation_positions" in data else "last",
            "prompt_format": str(data["prompt_format"].item()) if "prompt_format" in data else "legacy"}
    valid_policy = (
        metadata["activation_positions"] == "scored" and
        metadata["prompt_format"] == "match-evaluation")
    if args.dataset == "boolq":
        valid_policy = (
            metadata["activation_positions"] == "last" and
            metadata["prompt_format"] == "answer-conditioned")
    if not valid_policy:
        expected = ("answer-conditioned last-token" if args.dataset == "boolq"
                    else "matched scored-token")
        raise ValueError(f"the {args.dataset} sweep requires {expected} features")
    expected_questions = {"winogrande": 1000, "copa": 400,
                          "boolq": 1000}.get(args.dataset,
                                              metadata["dev_num_samples"])
    if args.dataset == "storycloze" and expected_questions <= 0:
        raise ValueError("StoryCloze feature metadata lacks its development size")
    if metadata["dataset"] != args.dataset:
        raise ValueError(f"feature dataset {metadata['dataset']!r} does not match {args.dataset!r}")
    if metadata["dev_num_samples"] != expected_questions or len(np.unique(q)) != expected_questions:
        raise ValueError(
            f"paper-compatible {args.dataset} sweep requires exactly "
            f"{expected_questions} development questions")

    fit_qs, validation_qs = split_development_questions(q, 0.2, args.split_seed)
    mask = np.isin(q, fit_qs)
    ranks = COPA_RANKS if args.dataset == "copa" else RANKS
    configs = configurations(stage, selected_center, selected_power, ranks)
    max_rank = min(max(c["rank"] for c in configs), X.shape[1]-1, int(mask.sum())-1)
    raw = fit_raw_spectral(X[mask], y[mask], weights[mask], args.covariance_source,
                           configs[0]["center_mode"], max_rank, args.seed)
    question_hash = fitting_question_hash(fit_qs)
    np.savez(output / f"spectral_{stage}_{question_hash}.npz", basis=raw["basis"],
             raw_eigvals=raw["raw_eigvals"],
             total_covariance_trace=np.array(raw["total_covariance_trace"]))

    completed = {r.get("config_hash") for r in rows if r.get("status") == "complete"}
    for cfg in configs:
        base = dict(stage=stage, dataset=args.dataset,
                    feature_file=str(args.feature_file), layer=args.layer,
                    covariance_source=args.covariance_source, variance_floor=args.variance_floor,
                    kappa=args.kappa, alpha=args.alpha, beta=args.beta,
                    question_hash=question_hash, **cfg)
        h = config_hash(base)
        if args.resume and h in completed:
            print(f"SKIP complete {h}")
            continue
        if cfg["rank"] > raw["basis"].shape[1]:
            rows.append({**base, "config_hash": h, "status": "invalid-rank"})
            continue
        artifact = output / f"artifact_{h}.npz"
        if args.dry_run:
            print("DRY RUN", json.dumps(base, sort_keys=True))
            continue
        geometry = derive_lowrank_geometry(raw, cfg["rank"], cfg["shrinkage"],
                                           args.variance_floor, cfg["center_mode"],
                                           cfg["whitening_power"])
        save_artifact(artifact, geometry, cfg, args, fit_qs, validation_qs, metadata, h)
        accuracy, trigger = evaluate(artifact, args)
        row = {**base, "config_hash": h, "artifact_path": str(artifact),
               "accuracy": accuracy, "trigger_rate": trigger,
               "status": "complete" if accuracy is not None else "failed"}
        rows.append(row)
        write_summary(rows, output)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Validation-only generic ellipsoid sweep")
    parser.add_argument("--dataset", choices=("winogrande", "copa", "boolq", "storycloze"),
                        default="winogrande")
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--feature-file", required=True)
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--stage", choices=("center", "power", "covariance", "all"), default="all")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--kappa", type=float, default=20.0)
    parser.add_argument("--alpha", type=float, default=0.8)
    parser.add_argument("--beta", type=float, default=-0.8)
    parser.add_argument("--covariance-source", choices=("pooled",), default="pooled")
    parser.add_argument("--variance-floor", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary = output / "summary.json"
    rows = json.loads(summary.read_text()) if summary.exists() else []
    stages = ("center", "power", "covariance") if args.stage == "all" else (args.stage,)
    for stage in stages:
        center = best_row(rows, "center")
        power = best_row(rows, "power")
        rows = run_stage(stage, args, rows, output,
                         center["center_mode"] if center else None,
                         float(power["whitening_power"]) if power else None)
    write_summary(rows, output)
    winner = best_row(rows, stages[-1])
    if winner:
        best_path = output / f"best_{args.dataset}_ellipsoid.json"
        best_path.write_text(json.dumps(winner, indent=2))
        print("BEST", json.dumps(winner, indent=2))
        print("FROZEN OFFICIAL COMMAND (not executed):")
        print(f"python evaluate_generic.py --model_name {args.model_name} --dataset {args.dataset} "
              f"--eval-split official --layer {args.layer} --prototype_path {winner['artifact_path']} "
              f"--steering-geometry ellipsoid --kappa {args.kappa} --alpha {args.alpha} --beta {args.beta}")


if __name__ == "__main__":
    main()
