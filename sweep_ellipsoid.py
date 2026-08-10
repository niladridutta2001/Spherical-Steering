"""Staged, validation-only ellipsoid hyperparameter sweep."""

import argparse, csv, hashlib, json, os, subprocess, sys
from pathlib import Path
import numpy as np

from ellipsoid_geometry import (fit_raw_spectral, derive_lowrank_geometry,
                                fitting_question_hash)
from get_prototypes import split_question_folds


CENTER_MODES = ("zero", "global", "class-midpoint")
POWERS = (0.0, 0.125, 0.25, 0.375, 0.5)
RANKS = (32, 64, 128, 256)
SHRINKAGES = (0.1, 0.3, 0.5, 0.7)
FIELDS = ["stage","config_hash","fold","layer","feature_file","activation_positions",
          "prompt_format","center_mode","covariance_source","whitening_power","rank",
          "shrinkage","variance_floor","kappa","alpha","beta","trigger_rate","MC1",
          "MC2","MC3","MC_avg","selection_score","mean_metric_radius_before","mean_metric_radius_after",
          "max_relative_metric_radius_error","artifact_path","result_path","status"]


def stable_config_hash(config):
    payload = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def stage_configs(stage, best_center=None, best_power=None):
    if stage == "center":
        return [dict(center_mode=c, whitening_power=.5, rank=128, shrinkage=.1) for c in CENTER_MODES]
    if not best_center:
        raise ValueError(f"stage {stage} requires Stage A's selected center")
    if stage == "power":
        return [dict(center_mode=best_center, whitening_power=p, rank=128, shrinkage=.1) for p in POWERS]
    if best_power is None:
        raise ValueError("covariance stage requires Stage B's selected whitening power")
    return [dict(center_mode=best_center, whitening_power=best_power, rank=r, shrinkage=s)
            for r in RANKS for s in SHRINKAGES]


def choose_best(rows, stage):
    candidates = [r for r in rows if r.get("stage") == stage and r.get("status") == "complete"]
    if not candidates:
        return None
    return max(candidates, key=lambda r: float(r.get("selection_score", r["MC_avg"])))


def _load_rows(path):
    if not path.exists(): return []
    with path.open() as f: return json.load(f)


def _write_summaries(rows, output):
    rows = sorted(rows, key=lambda r: (r.get("stage", ""), -float(r.get("MC_avg") or -1)))
    (output/"summary.json").write_text(json.dumps(rows, indent=2))
    with (output/"summary.csv").open("w", newline="") as f:
        writer=csv.DictWriter(f,fieldnames=FIELDS); writer.writeheader()
        for row in rows: writer.writerow({k:row.get(k) for k in FIELDS})


def _save_raw(path, raw, question_hash):
    np.savez(path, basis=raw["basis"], raw_eigvals=raw["raw_eigvals"],
             total_covariance_trace=np.array(raw["total_covariance_trace"]),
             m_T=raw["m_T"], m_H=raw["m_H"], denominator=np.array(raw["denominator"]),
             center_zero=raw["centers"]["zero"], center_global=raw["centers"]["global"],
             center_midpoint=raw["centers"]["class-midpoint"],
             fitting_question_hash=np.array(question_hash))


def _load_raw(path):
    with np.load(path, allow_pickle=False) as d:
        return dict(basis=d["basis"],raw_eigvals=d["raw_eigvals"],
                    total_covariance_trace=float(d["total_covariance_trace"]),
                    m_T=d["m_T"],m_H=d["m_H"],denominator=float(d["denominator"]),
                    centers={"zero":d["center_zero"],"global":d["center_global"],
                             "class-midpoint":d["center_midpoint"]})


def _artifact(output, cfg, geom, meta, train_qs, val_qs, test_qs, args, h):
    path=output/f"artifact_{h}.npz"
    values={k:v for k,v in geom.items() if k not in ("m_T","m_H")}
    values.update(artifact_version=np.array(3),geometry=np.array("ellipsoid-lowrank"),
        covariance_source=np.array(args.covariance_source),center_mode=np.array(cfg["center_mode"]),
        whitening_power=np.array(cfg["whitening_power"],dtype=np.float32),
        shrinkage=np.array(cfg["shrinkage"],dtype=np.float32),
        variance_floor=np.array(args.variance_floor,dtype=np.float32),
        cov_rank=np.array(cfg["rank"]),train_q_indices=train_qs,
        validation_q_indices=val_qs,test_q_indices=test_qs,fold_idx=np.array(args.fold),
        activation_positions=np.array(meta["activation_positions"]),
        prompt_format=np.array(meta["prompt_format"]),config_hash=np.array(h))
    values={k:(v.astype(np.float32) if isinstance(v,np.ndarray) and
               np.issubdtype(v.dtype,np.floating) else v) for k,v in values.items()}
    np.savez(path,**values); return path


def _run_stage(stage, args, rows, output, best_center=None, best_power=None):
    with np.load(args.feature_file,allow_pickle=False) as data:
        X=np.asarray(data["activations"]); y=np.asarray(data["labels"]); q=np.asarray(data["q_indices"])
        w=np.asarray(data["sample_weights"]) if "sample_weights" in data else np.ones(len(X))
        meta={"activation_positions":str(data["activation_positions"].item()) if "activation_positions" in data else "last",
              "prompt_format":str(data["prompt_format"].item()) if "prompt_format" in data else "legacy"}
    if stage=="center" and (meta["activation_positions"]!="scored" or meta["prompt_format"]!="match-evaluation"):
        raise ValueError("center stage requires matched scored-position features")
    folds=list(split_question_folds(q,2,.2,True,args.seed)); _,train_qs,val_qs,test_qs=folds[args.fold]
    mask=np.isin(q,train_qs); question_hash=fitting_question_hash(train_qs)
    configs=stage_configs(stage,best_center,best_power)
    max_rank=min(max(c["rank"] for c in configs),X.shape[1]-1,int(mask.sum())-1)
    if max_rank<1: raise ValueError("insufficient fitting samples")
    raw_by_center={}
    cache_centers=(set(c["center_mode"] for c in configs) if args.covariance_source=="global" else {"pooled-independent"})
    for cache_center in cache_centers:
        fit_center=configs[0]["center_mode"] if cache_center=="pooled-independent" else cache_center
        cache_key=stable_config_hash(dict(feature_file=str(Path(args.feature_file).resolve()),fold=args.fold,
            layer=args.layer,covariance_source=args.covariance_source,question_hash=question_hash,
            center_mode=cache_center,max_rank=max_rank))
        cache=output/f"spectral_{cache_key}.npz"
        if cache.exists(): raw=_load_raw(cache)
        else:
            raw=fit_raw_spectral(X[mask],y[mask],w[mask],args.covariance_source,
                                 fit_center,max_rank,args.seed)
            _save_raw(cache,raw,question_hash)
        raw_by_center[cache_center]=raw
    done={r.get("config_hash") for r in rows if r.get("status")=="complete"}
    for cfg in configs:
        raw=raw_by_center[cfg["center_mode"] if args.covariance_source=="global" else "pooled-independent"]
        base=dict(stage=stage,fold=args.fold,layer=args.layer,feature_file=str(args.feature_file),
                  activation_positions=meta["activation_positions"],prompt_format=meta["prompt_format"],
                  covariance_source=args.covariance_source,variance_floor=args.variance_floor,
                  kappa=args.kappa,alpha=args.alpha,beta=args.beta,**cfg)
        h=stable_config_hash(base)
        if args.resume and h in done: print(f"SKIP complete {h}"); continue
        if cfg["rank"]>raw["basis"].shape[1] or cfg["rank"]>=X.shape[1]:
            row={**base,"config_hash":h,"status":"invalid-rank"}; rows.append(row); continue
        artifact=output/f"artifact_{h}.npz"; result=output/f"result_{h}.json"
        command=[sys.executable,"evaluate_mc.py",args.model_name,"--prototype_path",str(artifact),
                 "--eval-split","validation","--steering-geometry","ellipsoid","--layer",str(args.layer),
                 "--kappa",str(args.kappa),"--alpha",str(args.alpha),"--beta",str(args.beta),
                 "--output_path",str(result)]
        if args.model_dir: command.extend(["--model_dir",args.model_dir])
        if args.dry_run:
            print("DRY RUN",json.dumps(base,sort_keys=True)); print(" ".join(command)); continue
        geom=derive_lowrank_geometry(raw,cfg["rank"],cfg["shrinkage"],args.variance_floor,
                                     cfg["center_mode"],cfg["whitening_power"])
        artifact=_artifact(output,cfg,geom,meta,train_qs,val_qs,test_qs,args,h)
        completed=subprocess.run(command,cwd=Path(__file__).parent)
        row={**base,"config_hash":h,"artifact_path":str(artifact),"result_path":str(result)}
        if completed.returncode or not result.exists(): row["status"]="failed"
        else:
            payload=json.loads(result.read_text()); metrics=payload["metrics"]
            trigger=payload.get("trigger_rate")
            selection_score=metrics["MC2"] - (0.25*abs(trigger-0.9) if trigger is not None else 0.0)
            row.update(MC1=metrics["MC1"],MC2=metrics["MC2"],MC3=metrics["MC3"],
                MC_avg=(metrics["MC1"]+metrics["MC2"]+metrics["MC3"])/3,
                selection_score=selection_score,trigger_rate=trigger,
                mean_metric_radius_before=payload.get("mean_metric_radius_before"),
                mean_metric_radius_after=payload.get("mean_metric_radius_after"),
                max_relative_metric_radius_error=payload.get("max_relative_metric_radius_error"),status="complete")
        rows.append(row); _write_summaries(rows,output)
    return rows


def main():
    p=argparse.ArgumentParser(description="Validation-only staged ellipsoid sweep")
    p.add_argument("--stage",choices=["center","power","covariance","all"],required=True)
    p.add_argument("--feature-file",required=True); p.add_argument("--model-name",required=True)
    p.add_argument("--model-dir",default=None); p.add_argument("--layer",type=int,required=True)
    p.add_argument("--fold",type=int,choices=[0,1],required=True); p.add_argument("--output-dir",required=True)
    p.add_argument("--resume",action="store_true"); p.add_argument("--dry-run",action="store_true")
    p.add_argument("--kappa",type=float,default=20); p.add_argument("--alpha",type=float,default=.8)
    p.add_argument("--beta",type=float,default=-.8); p.add_argument("--covariance-source",choices=["pooled","global"],default="pooled")
    p.add_argument("--variance-floor",type=float,default=1e-5); p.add_argument("--seed",type=int,default=42)
    args=p.parse_args(); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    rows=_load_rows(output/"summary.json")
    stages=("center","power","covariance") if args.stage=="all" else (args.stage,)
    for stage in stages:
        center_best=choose_best(rows,"center"); power_best=choose_best(rows,"power")
        rows=_run_stage(stage,args,rows,output,
                        center_best and center_best["center_mode"],
                        power_best and float(power_best["whitening_power"]))
    _write_summaries(rows,output)
    final_stage=stages[-1]; best=choose_best(rows,final_stage)
    if best:
        frozen={k:best.get(k) for k in FIELDS}
        frozen["selection_note"]="single validation split: cross-fold standard-deviation penalty unavailable"
        frozen_path=output/f"best_ellipsoid_fold{args.fold}.json"; frozen_path.write_text(json.dumps(frozen,indent=2))
        print("BEST",json.dumps(frozen,indent=2))
        print("FROZEN TEST COMMAND (not executed):")
        print(f"python evaluate_mc.py {args.model_name} --prototype_path {best['artifact_path']} --eval-split test --steering-geometry ellipsoid --layer {args.layer} --kappa {args.kappa} --alpha {args.alpha} --beta {args.beta}")


if __name__=="__main__": main()
