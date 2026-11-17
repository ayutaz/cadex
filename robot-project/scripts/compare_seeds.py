"""学習の乱数（学習シード）を変えた複数回の学習を評価し、二機体を比べる。

各回の方策（<run>/walk.cxpolicy、最終の方策）を evaluation/evaluate_variant.py で評価し、
artifacts/<variant>/evaluation-seeds/<run>/ に置く。評価済み（summary.json がある）の回は評価し直さない。
q8.seeds で学習シード方向に集め、artifacts/compare/seeds_<a>_vs_<b>.{json,md} に書く。

例:
  uv run python scripts/compare_seeds.py \
      base=artifacts/training/full-base-s0 base=artifacts/training/full-base-s1 ... \
      leg120=artifacts/training/full-leg120-s0 ...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

from q8 import seeds  # noqa: E402
from q8.paths import ARTIFACTS  # noqa: E402
from q8.portable import write_json  # noqa: E402

POLICY_FILE = "walk.cxpolicy"


def parse_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"<機体>=<学習の回のフォルダ> の形で渡す: {text!r}")
    variant, folder = text.split("=", 1)
    path = Path(folder)
    if not (path / POLICY_FILE).exists():
        raise FileNotFoundError(f"{path / POLICY_FILE} がありません")
    return variant, path


def run(runs: list[tuple[str, Path]], *, artifacts: Path = ARTIFACTS,
        evaluate: Callable | None = None, video: bool = False) -> dict:
    variants = list(dict.fromkeys(v for v, _ in runs))
    if len(variants) != 2:
        raise ValueError(f"比べる機体は2つ（渡されたのは {variants}）")
    if evaluate is None:
        from evaluate_variant import evaluate_variant as evaluate
    summaries: dict[str, list[dict]] = {v: [] for v in variants}
    names: dict[str, list[str]] = {v: [] for v in variants}
    for variant, folder in runs:
        out_dir = artifacts / variant / "evaluation-seeds" / folder.name
        saved = out_dir / "summary.json"
        if not saved.exists():
            # 戻り値は summary.json と形が違う（variant を含まない）ので、書かれたファイルを読む
            evaluate(variant, policy=folder / POLICY_FILE, out_dir=out_dir, video=video)
        summary = json.loads(saved.read_text(encoding="utf-8"))
        summaries[variant].append(summary)
        names[variant].append(folder.name)
    a, b = variants
    result = seeds.compare(seeds.across(summaries[a]), seeds.across(summaries[b]), labels=(a, b))
    out = artifacts / "compare"
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"seeds_{a}_vs_{b}.json", {**result, "runs": names}, indent=2)
    (out / f"seeds_{a}_vs_{b}.md").write_text(seeds.to_markdown(result), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="<機体>=<学習の回のフォルダ>")
    parser.add_argument("--video", action="store_true")
    args = parser.parse_args()
    result = run([parse_run(r) for r in args.runs], video=args.video)
    print(seeds.to_markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
