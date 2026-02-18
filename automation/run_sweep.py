from __future__ import annotations

import argparse
import csv
import random
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from classify_failure import classify_metrics
from run_single_experiment import DEFAULT_RUNNER_CMD, run_trial
from sweep_utils import (
    DEFAULT_METRIC_FIELDS,
    compute_score,
    extract_param_specs,
    flatten_dict,
    load_yaml,
    parse_override_pairs,
    parse_bool,
    save_json,
    sample_params,
    to_float,
)


def read_existing_rows(csv_path: Path) -> List[Dict[str, Any]]:
    if not csv_path.exists():
        return []
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        return list(reader)


def append_row(csv_path: Path, row: Mapping[str, Any], fieldnames: List[str]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with csv_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in fieldnames})


def refresh_best_dir(sweep_dir: Path, rows: List[Dict[str, Any]], topk: int) -> None:
    best_dir = sweep_dir / "_best"
    best_dir.mkdir(parents=True, exist_ok=True)

    parsed: List[tuple[float, str]] = []
    for row in rows:
        exp_name = str(row.get("exp", "")).strip()
        if not exp_name:
            continue
        parsed.append((to_float(row.get("score", -1e9), -1e9), exp_name))

    parsed.sort(key=lambda x: x[0], reverse=True)
    keep = {exp_name for _, exp_name in parsed[:topk]}

    for child in best_dir.glob("exp_*"):
        if child.name not in keep and child.is_dir():
            shutil.rmtree(child)

    for exp_name in keep:
        src = sweep_dir / exp_name
        dst = best_dir / exp_name
        if not src.exists():
            continue
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auto experiment sweep with resume/timeout/topK support.")
    parser.add_argument("--config", required=True, help="Base config path")
    parser.add_argument("--space", required=True, help="Sweep search space YAML")
    parser.add_argument("--out", required=True, help="Sweep output directory")
    parser.add_argument("--trials", type=int, default=200, help="Number of trials")
    parser.add_argument("--topk", type=int, default=20, help="Top K trials to keep under _best")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="Timeout per trial")
    parser.add_argument("--sampler-seed", type=int, default=0xC0DE, help="Random seed for sampler")
    parser.add_argument("--resume", action="store_true", help="Skip existing exp_* folders")
    parser.add_argument("--start-index", type=int, default=1, help="Experiment index start")
    parser.add_argument("--policy-rules", default="automation/policy_rules.yaml", help="Policy/score YAML")
    parser.add_argument("--runner-cmd", default=DEFAULT_RUNNER_CMD, help="Runner command template")
    parser.add_argument("--workdir", default=".", help="Working directory for trial subprocess")
    parser.add_argument("--metrics-relpath", default="metrics.json", help="Runner-generated metrics path relative to outdir")
    parser.add_argument("--override", action="append", default=[], help="Global override key=value (repeatable)")
    parser.add_argument("--dry-run", action="store_true", help="Do not launch subprocess trials")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_dir = Path(args.out).resolve()
    sweep_dir.mkdir(parents=True, exist_ok=True)

    space_cfg = load_yaml(Path(args.space))
    specs = extract_param_specs(space_cfg)
    param_keys = sorted(specs.keys())

    policy_rules: Dict[str, Any] = {}
    policy_path = Path(args.policy_rules)
    if policy_path.exists():
        policy_rules = load_yaml(policy_path)

    global_overrides = parse_override_pairs(args.override)
    results_csv = sweep_dir / "results.csv"
    existing_rows = read_existing_rows(results_csv)

    fieldnames = [
        "exp",
        "score",
        "label",
        "success",
        "fail_reason",
        "returncode",
        "wall_time_sec",
    ]
    fieldnames.extend([f"param.{k}" for k in param_keys])
    for key in DEFAULT_METRIC_FIELDS:
        if key not in fieldnames:
            fieldnames.append(key)

    rng = random.Random(args.sampler_seed)
    created = 0
    attempted = 0
    start_time = time.time()

    for i in range(args.start_index, args.start_index + args.trials):
        attempted += 1
        exp_name = f"exp_{i:06d}"
        outdir = sweep_dir / exp_name
        if args.resume and outdir.exists():
            continue

        sampled = sample_params(space_cfg, rng)
        merged_overrides = dict(sampled)
        merged_overrides.update(global_overrides)
        seed = int(to_float(merged_overrides.get("seed", i), i))

        metrics = run_trial(
            config_path=Path(args.config),
            outdir=outdir,
            overrides=merged_overrides,
            seed=seed,
            timeout_sec=args.timeout_sec,
            runner_cmd=args.runner_cmd,
            metrics_relpath=args.metrics_relpath,
            workdir=Path(args.workdir),
            dry_run=args.dry_run,
        )

        stderr_text = ""
        stderr_path = outdir / "stderr.log"
        if stderr_path.exists():
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")

        label_info = classify_metrics(metrics, stderr_text=stderr_text, policy_rules=policy_rules)
        save_json(outdir / "failure_label.json", label_info)

        score = compute_score(metrics, policy_rules=policy_rules)
        row: Dict[str, Any] = {
            "exp": exp_name,
            "score": score,
            "label": label_info["label"],
            "success": parse_bool(metrics.get("success", False)),
            "fail_reason": metrics.get("fail_reason", ""),
            "returncode": metrics.get("returncode", ""),
            "wall_time_sec": metrics.get("wall_time_sec", ""),
        }
        for key in param_keys:
            row[f"param.{key}"] = merged_overrides.get(key, "")
        for key in DEFAULT_METRIC_FIELDS:
            row[key] = metrics.get(key, "")
        append_row(results_csv, row, fieldnames)
        existing_rows.append(row)
        refresh_best_dir(sweep_dir, existing_rows, args.topk)
        created += 1

        print(
            f"[sweep] {exp_name} "
            f"success={row['success']} label={row['label']} "
            f"score={score:.4f}"
        )

    elapsed = time.time() - start_time
    print(
        f"[sweep] done attempted={attempted} created={created} "
        f"elapsed_sec={elapsed:.1f} output={sweep_dir}"
    )


if __name__ == "__main__":
    main()
