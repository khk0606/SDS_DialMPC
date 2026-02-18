from __future__ import annotations

import argparse
import csv
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping

from sweep_utils import (
    build_nested_from_dotted,
    clamp_value,
    extract_param_specs,
    load_yaml,
    sample_params,
    save_yaml,
    to_float,
)


def read_rows(results_csv: Path) -> List[Dict[str, Any]]:
    if not results_csv.exists():
        return []
    with results_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def parse_param_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    for key, value in row.items():
        if not key.startswith("param."):
            continue
        pkey = key[len("param.") :]
        if value == "":
            continue
        try:
            if "." not in str(value):
                params[pkey] = int(value)
            else:
                params[pkey] = float(value)
        except (TypeError, ValueError):
            params[pkey] = value
    return params


def apply_label_adjustments(
    params: Dict[str, Any],
    label: str,
    policy_rules: Mapping[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
) -> Dict[str, Any]:
    adjusted = dict(params)
    label_rules = policy_rules.get("label_adjustments", {}).get(label, {})
    if not isinstance(label_rules, dict):
        return adjusted

    for key, rule in label_rules.items():
        if key not in adjusted:
            continue
        if not isinstance(rule, dict):
            continue
        op = str(rule.get("op", "mul")).lower()
        v = to_float(rule.get("value", 1.0), 1.0)
        base = adjusted[key]
        if isinstance(base, (int, float)):
            if op == "mul":
                new_v = float(base) * v
            elif op == "add":
                new_v = float(base) + v
            elif op == "set":
                new_v = v
            else:
                continue
            spec = specs.get(key, {})
            new_v = clamp_value(new_v, spec)
            if str(spec.get("dist", "")).lower() == "int_uniform":
                new_v = int(round(new_v))
            adjusted[key] = new_v
    return adjusted


def jitter_params(
    params: Dict[str, Any],
    specs: Mapping[str, Mapping[str, Any]],
    rng: random.Random,
    jitter_ratio: float,
) -> Dict[str, Any]:
    out = dict(params)
    for key, value in list(out.items()):
        spec = specs.get(key, {})
        dist = str(spec.get("dist", "")).lower()
        if dist not in {"uniform", "loguniform", "int_uniform"}:
            continue
        if not isinstance(value, (int, float)):
            continue
        span = to_float(spec.get("max", value), value) - to_float(spec.get("min", value), value)
        noise = rng.uniform(-1.0, 1.0) * span * jitter_ratio
        candidate = float(value) + noise
        candidate = clamp_value(candidate, spec)
        if dist == "int_uniform":
            candidate = int(round(candidate))
        out[key] = candidate
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Propose next trial params from results and rules.")
    parser.add_argument("--results", required=True, help="Path to sweep results.csv")
    parser.add_argument("--space", required=True, help="Path to sweep_space.yaml")
    parser.add_argument("--policy-rules", default="automation/policy_rules.yaml", help="Path to policy rules")
    parser.add_argument("--out", default="automation/proposed_params.yaml", help="Path to output params YAML")
    parser.add_argument("--sampler-seed", type=int, default=0xC0DE, help="Random seed")
    parser.add_argument("--explore-prob", type=float, default=0.2, help="Probability of full random resample")
    parser.add_argument("--jitter-ratio", type=float, default=0.05, help="Local perturbation ratio")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = random.Random(args.sampler_seed)

    space_cfg = load_yaml(Path(args.space))
    specs = extract_param_specs(space_cfg)
    policy_rules = {}
    policy_path = Path(args.policy_rules)
    if policy_path.exists():
        policy_rules = load_yaml(policy_path)

    rows = read_rows(Path(args.results))
    if not rows or rng.random() < args.explore_prob:
        proposed = sample_params(space_cfg, rng)
        reason = "explore_random"
    else:
        best = max(rows, key=lambda r: to_float(r.get("score", -1e9), -1e9))
        params = parse_param_row(best)
        last_label = str(rows[-1].get("label", "stable_good"))
        params = apply_label_adjustments(params, last_label, policy_rules, specs)
        proposed = jitter_params(params, specs, rng, jitter_ratio=args.jitter_ratio)
        reason = f"exploit_best_with_{last_label}"

    nested = build_nested_from_dotted(proposed)
    output = {
        "reason": reason,
        "generated_at_epoch": int(time.time()),
        "overrides": nested,
        "overrides_flat": proposed,
    }
    save_yaml(Path(args.out), output)
    print(f"[propose] reason={reason} wrote={args.out}")


if __name__ == "__main__":
    main()
