"""1回の学習で書かれたチェックポイントを、エンジン（cadex evaluate）で順に評価して表にする。

使い方:
  uv run python evaluation/evaluate_checkpoints.py artifacts/runs/cpu-cut --variant base
出力: <run>/checkpoints.json と checkpoints.csv（反復ごとの合格率・不合格の理由・主要指標の平均）
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

KEYS = ["completed", "mean_forward_speed_mm_s", "speed_ratio", "speed_error_rms_mm_s", "max_tilt_deg",
        "steps_min", "slip_share_max", "effort_rms_rated_ratio_max", "effort_peak_ratio_max", "cost_of_transport"]


def checkpoints(run_dir: Path) -> list[tuple[str, Path]]:
    found = []
    for p in run_dir.glob("*.cxpolicy"):
        m = re.fullmatch(r".*\.(\d+)\.cxpolicy", p.name)
        if m:
            found.append((int(m.group(1)), str(int(m.group(1))), p))
        elif p.name.count(".") == 1:  # walk.cxpolicy（最終）
            found.append((10 ** 9, "final", p))
    return [(label, p) for _, label, p in sorted(found)]


def row_for(label: str, ev: dict, keys=KEYS) -> dict:
    seeds = ev["seeds"]
    row = {"checkpoint": label, "pass_rate": sum(s["pass"] for s in seeds) / len(seeds),
           "failing": dict(Counter(f for s in seeds for f in s.get("failing", [])))}
    for k in keys:
        vals = [s["metrics"].get(k) for s in seeds if s["metrics"].get(k) is not None]
        row[k] = sum(vals) / len(vals) if vals else None
    return row


def main() -> int:
    from evaluate_variant import run_cadex_evaluate

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--variant", default="base")
    args = ap.parse_args()
    rows = []
    for label, path in checkpoints(args.run_dir):
        out = args.run_dir / f"eval-{label}"
        ev = (json.loads((out / "evaluation.json").read_text()) if (out / "evaluation.json").exists()
              else run_cadex_evaluate(args.variant, path, out, video=False))
        rows.append(row_for(label, ev))
        print(json.dumps(rows[-1], ensure_ascii=False))
    (args.run_dir / "checkpoints.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
    with (args.run_dir / "checkpoints.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, "failing": json.dumps(r["failing"], ensure_ascii=False)} for r in rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
