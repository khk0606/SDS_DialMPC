from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from sweep_utils import parse_bool, to_float


def read_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"results.csv not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def pearson(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 3 or len(xs) != len(ys):
        return 0.0
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0.0 or vy <= 0.0:
        return 0.0
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    return cov / math.sqrt(vx * vy)


def numeric_columns(rows: List[Dict[str, Any]]) -> List[str]:
    if not rows:
        return []
    cols = list(rows[0].keys())
    keep: List[str] = []
    for col in cols:
        ok = 0
        for row in rows:
            val = row.get(col, "")
            if val in {"", None}:
                continue
            try:
                float(val)
                ok += 1
            except (TypeError, ValueError):
                pass
        if ok >= max(3, int(0.3 * len(rows))):
            keep.append(col)
    return keep


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze sweep results and generate top/correlation reports.")
    parser.add_argument("--results", required=True, help="Path to results.csv")
    parser.add_argument("--outdir", default=None, help="Output directory (default: <sweep>/analysis)")
    parser.add_argument("--topn", type=int, default=10, help="Top N rows by score")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_path = Path(args.results).resolve()
    rows = read_rows(results_path)
    if not rows:
        raise RuntimeError("results.csv has no rows")

    outdir = Path(args.outdir).resolve() if args.outdir else results_path.parent / "analysis"
    outdir.mkdir(parents=True, exist_ok=True)

    sorted_rows = sorted(rows, key=lambda r: to_float(r.get("score", -1e9), -1e9), reverse=True)
    top_rows = sorted_rows[: max(1, args.topn)]
    top_fields = list(top_rows[0].keys())
    write_csv(outdir / "top10.csv", top_rows, top_fields)

    cols = numeric_columns(rows)
    score_values = [to_float(r.get("score", 0.0), 0.0) for r in rows]
    corr_rows: List[Dict[str, Any]] = []
    for col in cols:
        if col == "score":
            continue
        xs = [to_float(r.get(col, 0.0), 0.0) for r in rows]
        corr = pearson(xs, score_values)
        corr_rows.append({"column": col, "pearson_with_score": corr, "abs_pearson": abs(corr)})
    corr_rows.sort(key=lambda r: to_float(r["abs_pearson"], 0.0), reverse=True)
    write_csv(outdir / "correlation.csv", corr_rows, ["column", "pearson_with_score", "abs_pearson"])

    labels = Counter(str(r.get("label", "unknown")) for r in rows)
    success_cnt = sum(1 for r in rows if parse_bool(r.get("success", False)))
    success_rate = success_cnt / max(1, len(rows))

    best = sorted_rows[0]
    summary = []
    summary.append("# Sweep Summary")
    summary.append("")
    summary.append(f"- total_trials: {len(rows)}")
    summary.append(f"- success_rate: {success_rate:.3f} ({success_cnt}/{len(rows)})")
    summary.append(f"- best_exp: {best.get('exp', '')}")
    summary.append(f"- best_score: {to_float(best.get('score', 0.0), 0.0):.6f}")
    summary.append("")
    summary.append("## Label Counts")
    for label, cnt in labels.most_common():
        summary.append(f"- {label}: {cnt}")
    summary.append("")
    summary.append("## Top Correlations (|r|)")
    for row in corr_rows[:10]:
        summary.append(f"- {row['column']}: {to_float(row['pearson_with_score'], 0.0):.4f}")
    (outdir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    print(f"[analyze] wrote {outdir / 'summary.md'}")
    print(f"[analyze] wrote {outdir / 'top10.csv'}")
    print(f"[analyze] wrote {outdir / 'correlation.csv'}")


if __name__ == "__main__":
    main()
