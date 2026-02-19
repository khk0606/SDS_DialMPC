from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping

import yaml


DEFAULT_METRIC_FIELDS = [
    "success",
    "fail_reason",
    "returncode",
    "wall_time_sec",
    "fall_time_sec",
    "fall_count",
    "mean_roll",
    "mean_pitch",
    "mean_roll_dev",
    "mean_pitch_dev",
    "max_pitch_dev",
    "mean_height_error",
    "mean_vel_error",
    "yaw_rate_error",
    "forward_progress_m",
    "mean_forward_speed_mps",
    "forward_motion_ratio",
    "expected_gait",
    "foot_contact_source",
    "predicted_gait",
    "predicted_gait_confidence",
    "gait_match_score",
    "gait_diag_sync",
    "gait_lateral_sync",
    "gait_front_sync",
    "gait_hind_sync",
    "gait_switches_per_sec",
    "contact_match_score",
    "energy",
    "episode_return",
]


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"YAML file not found: {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def save_yaml(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(dict(data), sort_keys=False)
    path.write_text(text, encoding="utf-8")


def save_json(path: Path, data: Mapping[str, Any]) -> None:
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_scalar(raw: str) -> Any:
    text = raw.strip()
    low = text.lower()
    if low in {"true", "false"}:
        return low == "true"
    try:
        if "." not in text and "e" not in low:
            return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def parse_override_pairs(pairs: Iterable[str]) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Override must be key=value: {pair}")
        key, raw = pair.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Override key is empty: {pair}")
        overrides[key] = coerce_scalar(raw)
    return overrides


def set_dotted_key(data: MutableMapping[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    node: MutableMapping[str, Any] = data
    for key in keys[:-1]:
        child = node.get(key)
        if not isinstance(child, dict):
            child = {}
            node[key] = child
        node = child
    node[keys[-1]] = value


def build_nested_from_dotted(dotted_map: Mapping[str, Any]) -> Dict[str, Any]:
    nested: Dict[str, Any] = {}
    for key, value in dotted_map.items():
        set_dotted_key(nested, key, value)
    return nested


def flatten_dict(data: Mapping[str, Any], prefix: str = "") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for key, value in data.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            out.update(flatten_dict(value, dotted_key))
        else:
            out[dotted_key] = value
    return out


def extract_param_specs(space: Mapping[str, Any], prefix: str = "") -> Dict[str, Dict[str, Any]]:
    specs: Dict[str, Dict[str, Any]] = {}
    for key, value in space.items():
        dotted_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            if "dist" in value:
                specs[dotted_key] = dict(value)
            else:
                specs.update(extract_param_specs(value, dotted_key))
        else:
            specs[dotted_key] = {"dist": "fixed", "value": value}
    return specs


def sample_value(spec: Mapping[str, Any], rng: random.Random) -> Any:
    dist = str(spec.get("dist", "fixed")).lower()
    if dist == "fixed":
        return spec.get("value")
    if dist == "uniform":
        return rng.uniform(float(spec["min"]), float(spec["max"]))
    if dist == "loguniform":
        min_v = float(spec["min"])
        max_v = float(spec["max"])
        return math.exp(rng.uniform(math.log(min_v), math.log(max_v)))
    if dist == "int_uniform":
        min_v = int(spec["min"])
        max_v = int(spec["max"])
        return rng.randint(min_v, max_v)
    if dist == "choice":
        choices = spec.get("values")
        if not isinstance(choices, list) or not choices:
            raise ValueError("choice distribution requires non-empty list under `values`")
        return rng.choice(choices)
    raise ValueError(f"Unsupported dist: {dist}")


def sample_params(space: Mapping[str, Any], rng: random.Random) -> Dict[str, Any]:
    specs = extract_param_specs(space)
    sampled: Dict[str, Any] = {}
    for key, spec in specs.items():
        sampled[key] = sample_value(spec, rng)
    return sampled


def compute_score(metrics: Mapping[str, Any], policy_rules: Mapping[str, Any] | None = None) -> float:
    policy_rules = policy_rules or {}
    score_cfg = policy_rules.get("score", {})
    success = parse_bool(metrics.get("success", False))
    if not success:
        return to_float(score_cfg.get("hard_fail_score", -1e9))

    score = to_float(score_cfg.get("base", 0.0))
    weights = score_cfg.get("weights", {})
    if not isinstance(weights, dict):
        return score

    for key, weight in weights.items():
        # For attitude terms, prefer deviation-based metrics when present.
        if key == "mean_roll" and "mean_roll_dev" in metrics:
            val = metrics.get("mean_roll_dev", 0.0)
        elif key == "mean_pitch" and "mean_pitch_dev" in metrics:
            val = metrics.get("mean_pitch_dev", 0.0)
        else:
            val = metrics.get(key, 0.0)
        score += to_float(weight) * to_float(val, 0.0)
    return score


def clamp_value(value: float, spec: Mapping[str, Any]) -> float:
    dist = str(spec.get("dist", "")).lower()
    if dist in {"uniform", "loguniform", "int_uniform"}:
        min_v = to_float(spec.get("min", value))
        max_v = to_float(spec.get("max", value))
        return max(min_v, min(max_v, value))
    return value
