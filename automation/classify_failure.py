from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Mapping

from sweep_utils import load_yaml, parse_bool, save_json, to_float


def classify_metrics(
    metrics: Mapping[str, Any],
    stderr_text: str = "",
    policy_rules: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    policy_rules = policy_rules or {}
    cls = policy_rules.get("classification", {})

    early_fall_sec = to_float(cls.get("early_fall_sec", 2.0))
    roll_thr = to_float(cls.get("unstable_roll_abs", 0.30))
    pitch_thr = to_float(cls.get("unstable_pitch_abs", 0.30))
    contact_thr = to_float(cls.get("wrong_gait_contact_score", 0.65))
    standstill_vel_error = to_float(cls.get("standstill_vel_error", 0.30))
    high_energy_thr = cls.get("high_energy_threshold")

    rc = int(to_float(metrics.get("returncode", 0)))
    success = parse_bool(metrics.get("success", False))
    fail_reason = str(metrics.get("fail_reason", "") or "").lower()

    stderr_low = stderr_text.lower()
    if rc == 124 or "timeout" in fail_reason or "timeout" in stderr_low:
        return {
            "label": "compute_timeout",
            "reason": "Subprocess timeout or timeout marker found.",
            "confidence": 0.95,
        }

    if "nan" in stderr_low:
        return {
            "label": "unstable",
            "reason": "NaN detected in stderr.",
            "confidence": 0.85,
        }

    # Prefer pose-deviation metrics when available.
    mean_roll = abs(to_float(metrics.get("mean_roll_dev", metrics.get("mean_roll", 0.0))))
    mean_pitch = abs(to_float(metrics.get("mean_pitch_dev", metrics.get("mean_pitch", 0.0))))
    fall_time_sec = to_float(metrics.get("fall_time_sec", -1.0))
    mean_vel_error = to_float(metrics.get("mean_vel_error", 0.0))
    contact_match_score = to_float(metrics.get("contact_match_score", 1.0))
    energy = to_float(metrics.get("energy", 0.0))

    if not success:
        if 0.0 <= fall_time_sec < early_fall_sec:
            return {
                "label": "early_fall",
                "reason": f"Failure with fall_time_sec={fall_time_sec:.3f} < {early_fall_sec:.3f}.",
                "confidence": 0.9,
            }
        if mean_roll > roll_thr or mean_pitch > pitch_thr:
            return {
                "label": "unstable",
                "reason": "Failure with high body attitude oscillation.",
                "confidence": 0.8,
            }
        return {
            "label": "failure_other",
            "reason": "Failure without explicit subtype.",
            "confidence": 0.5,
        }

    if contact_match_score < contact_thr:
        return {
            "label": "wrong_gait",
            "reason": f"contact_match_score={contact_match_score:.3f} < {contact_thr:.3f}.",
            "confidence": 0.8,
        }

    if mean_vel_error > standstill_vel_error:
        return {
            "label": "standstill",
            "reason": f"mean_vel_error={mean_vel_error:.3f} > {standstill_vel_error:.3f}.",
            "confidence": 0.7,
        }

    if high_energy_thr is not None and energy > to_float(high_energy_thr):
        return {
            "label": "high_energy",
            "reason": f"energy={energy:.3f} > {to_float(high_energy_thr):.3f}.",
            "confidence": 0.65,
        }

    if mean_roll > roll_thr or mean_pitch > pitch_thr:
        return {
            "label": "unstable",
            "reason": "Success but body attitude oscillation is still high.",
            "confidence": 0.6,
        }

    return {
        "label": "stable_good",
        "reason": "No major failure mode detected.",
        "confidence": 0.7,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify trial result into a failure/success label.")
    parser.add_argument("--metrics", required=True, help="Path to metrics.json")
    parser.add_argument("--stderr", default=None, help="Path to stderr.log")
    parser.add_argument("--policy-rules", default=None, help="Path to policy_rules.yaml")
    parser.add_argument("--out", default=None, help="Output label JSON path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics_path = Path(args.metrics)
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics file not found: {metrics_path}")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

    stderr_text = ""
    if args.stderr:
        stderr_path = Path(args.stderr)
        if stderr_path.exists():
            stderr_text = stderr_path.read_text(encoding="utf-8", errors="ignore")

    policy = {}
    if args.policy_rules:
        policy = load_yaml(Path(args.policy_rules))

    label = classify_metrics(metrics=metrics, stderr_text=stderr_text, policy_rules=policy)

    out_path = Path(args.out) if args.out else metrics_path.parent / "failure_label.json"
    save_json(out_path, label)
    print(f"[classify] label={label['label']} confidence={label['confidence']}")


if __name__ == "__main__":
    main()
