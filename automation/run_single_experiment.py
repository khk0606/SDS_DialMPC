from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Mapping

from sweep_utils import (
    flatten_dict,
    load_yaml,
    parse_override_pairs,
    parse_bool,
    save_json,
    save_yaml,
    set_dotted_key,
)

DEFAULT_RUNNER_CMD = (
    "python automation/run_pair_runner.py "
    "--config {config} "
    "--custom-env dial_mpc.envs.unitree_go2_env"
)


def _build_config(
    config_path: Path,
    outdir: Path,
    overrides: Mapping[str, Any],
    seed: int | None,
) -> Dict[str, Any]:
    config_data = load_yaml(config_path)
    set_dotted_key(config_data, "output_dir", str(outdir))
    if seed is not None:
        set_dotted_key(config_data, "seed", int(seed))

    # Safe defaults for batch trial execution via dial_sim + dial_plan pair runner.
    # These only apply when keys are not already specified in YAML/overrides.
    batch_defaults = {
        "headless": True,
        "plot": False,
        "draw_refs": False,
        "record": True,
        "max_time_sec": 12.0,
        "stop_on_fall": True,
        "metrics_filename": "metrics.json",
    }
    for key, value in batch_defaults.items():
        if key not in config_data:
            config_data[key] = value

    for key, value in overrides.items():
        set_dotted_key(config_data, key, value)
    return config_data


def _synthesize_metrics(
    returncode: int,
    wall_time_sec: float,
    existing: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    metrics: Dict[str, Any] = dict(existing or {})
    success = parse_bool(metrics.get("success", returncode == 0))
    fail_reason = metrics.get("fail_reason")
    if not success and not fail_reason:
        fail_reason = "timeout" if returncode == 124 else "process_error"

    metrics["success"] = bool(success)
    metrics["fail_reason"] = fail_reason
    metrics["returncode"] = int(returncode)
    metrics["wall_time_sec"] = float(wall_time_sec)
    return metrics


def run_trial(
    config_path: Path,
    outdir: Path,
    overrides: Mapping[str, Any],
    seed: int | None,
    timeout_sec: int,
    runner_cmd: str,
    metrics_relpath: str = "metrics.json",
    workdir: Path | None = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    outdir = outdir.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    base_config = config_path.resolve()
    merged_config = _build_config(base_config, outdir, overrides, seed)
    config_used_path = outdir / "config_used.yaml"
    save_yaml(config_used_path, merged_config)

    merged_flat = flatten_dict(merged_config)
    rendered_cmd = runner_cmd.format(
        config=shlex.quote(str(config_used_path)),
        original_config=shlex.quote(str(base_config)),
        outdir=shlex.quote(str(outdir)),
        seed=shlex.quote(str(merged_flat.get("seed", seed if seed is not None else 0))),
    )
    (outdir / "cmd.txt").write_text(rendered_cmd + "\n", encoding="utf-8")
    save_json(outdir / "overrides_used.json", dict(overrides))

    stdout_path = outdir / "stdout.log"
    stderr_path = outdir / "stderr.log"
    rc = 0
    t0 = time.time()

    if dry_run:
        rc = 0
    else:
        with stdout_path.open("w", encoding="utf-8") as stdout_f, stderr_path.open(
            "w", encoding="utf-8"
        ) as stderr_f:
            try:
                proc = subprocess.run(
                    rendered_cmd,
                    shell=True,
                    cwd=str(workdir.resolve() if workdir else Path.cwd()),
                    stdout=stdout_f,
                    stderr=stderr_f,
                    timeout=timeout_sec,
                    check=False,
                )
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                rc = 124
                stderr_f.write("TIMEOUT\n")

    wall_time_sec = time.time() - t0
    metrics_path = outdir / metrics_relpath
    existing_metrics: Dict[str, Any] | None = None
    if metrics_path.exists():
        try:
            existing_metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            existing_metrics = {"success": False, "fail_reason": "invalid_metrics_json"}

    metrics = _synthesize_metrics(rc, wall_time_sec, existing_metrics)
    save_json(outdir / "metrics.json", metrics)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one experiment trial and store standardized outputs.")
    parser.add_argument("--config", required=True, help="Path to base YAML config")
    parser.add_argument("--outdir", required=True, help="Output directory for this trial")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--timeout-sec", type=int, default=1800, help="Subprocess timeout in seconds")
    parser.add_argument(
        "--runner-cmd",
        default=DEFAULT_RUNNER_CMD,
        help="Command template. Placeholders: {config}, {original_config}, {outdir}, {seed}",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Dotted key override as key=value (repeatable)",
    )
    parser.add_argument("--metrics-relpath", default="metrics.json", help="Runner-generated metrics path relative to outdir")
    parser.add_argument("--workdir", default=".", help="Working directory for runner command")
    parser.add_argument("--dry-run", action="store_true", help="Skip subprocess and only write scaffolding files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    overrides = parse_override_pairs(args.override)
    metrics = run_trial(
        config_path=Path(args.config),
        outdir=Path(args.outdir),
        overrides=overrides,
        seed=args.seed,
        timeout_sec=args.timeout_sec,
        runner_cmd=args.runner_cmd,
        metrics_relpath=args.metrics_relpath,
        workdir=Path(args.workdir),
        dry_run=args.dry_run,
    )
    print(
        f"[trial] success={metrics.get('success')} "
        f"returncode={metrics.get('returncode')} "
        f"wall_time_sec={metrics.get('wall_time_sec')}"
    )


if __name__ == "__main__":
    main()
