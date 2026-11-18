"""2つの案の、実機との差の評価（evaluation/evaluate_robustness.py）を、学習シード方向にまとめて比べる。

各回の方策を評価し（評価済み＝robustness.json がある回は飛ばす）、artifacts/<案>/robustness/<run>/ に置く。
まとめは artifacts/compare/robust_<a>_vs_<b>.{json,md}。評価に使う機体（モデルとタスク）は --variant。

例:
  uv run python scripts/compare_robustness.py --variant mini-r2-p7-eff \
      mini-r2-p7-eff=artifacts/training/full-mini-r2-p7-eff-s0 ... \
      mini-r2-p7-eff-dr=artifacts/training/full-mini-r2-p7-eff-dr-s0 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_seeds import POLICY_FILE, parse_run  # noqa: E402
from q8 import robustness as rb  # noqa: E402
from q8.paths import ARTIFACTS  # noqa: E402
from q8.portable import write_json  # noqa: E402


def run(runs: list[tuple[str, Path]], *, variant: str, artifacts: Path = ARTIFACTS, processes: int | None = None,
        evaluate=None) -> dict:
    labels = list(dict.fromkeys(label for label, _ in runs))
    if len(labels) != 2:
        raise ValueError(f"比べる案は2つ（渡されたのは {labels}）")
    if evaluate is None:
        from evaluate_robustness import evaluate  # noqa: PLC0415
    reports: dict[str, list[dict]] = {label: [] for label in labels}
    names: dict[str, list[str]] = {label: [] for label in labels}
    for label, folder in runs:
        out = artifacts / label / "robustness" / folder.name
        if not (out / "robustness.json").exists():
            evaluate(variant, folder / POLICY_FILE, out, processes=processes)
        reports[label].append(json.loads((out / "robustness.json").read_text(encoding="utf-8")))
        names[label].append(folder.name)
    a, b = labels
    result = rb.compare_reports(rb.across(reports[a]), rb.across(reports[b]), labels=(a, b))
    target = artifacts / "compare"
    target.mkdir(parents=True, exist_ok=True)
    write_json(target / f"robust_{a}_vs_{b}.json", {**result, "runs": names, "variant": variant}, indent=1)
    (target / f"robust_{a}_vs_{b}.md").write_text(rb.compare_markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="<案>=<学習の回のフォルダ>")
    parser.add_argument("--variant", required=True, help="評価に使う機体（モデルとタスク）")
    parser.add_argument("--processes", type=int, default=None)
    args = parser.parse_args()
    result = run([parse_run(r) for r in args.runs], variant=args.variant, processes=args.processes)
    print(rb.compare_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
