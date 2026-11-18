"""1つの方策を、実機との差を入れた評価にかける（段階A-2、q8/robustness.py）。

組（suite）は4つ。どれも評価シード（タスクの success.seeds、2101〜2110）と同じ成功条件で判定する。
  nominal  何も変えない（`cadex evaluate` と同じ数字になる）
  sweep    実機との差を1つずつ（摩擦、遅れ、周期のゆらぎ、ガタ、剛性、トルク、質量、重心）
  random   実機を想定した幅（REAL_RANGES）から全部を一度に引く（評価シードごとに --draws 組）
  push     3.0 秒目に横から 0.1 秒押す（強さ4段階 × 左右）。倒れずに戻った割合も出す

出力: <out>/robustness.json（設定・組ごとのまとめ・各エピソードの行）、<out>/summary.md

使い方:
  uv run python evaluation/evaluate_robustness.py --variant mini-r2-p7-eff \
      --policy artifacts/training/full-mini-r2-p7-eff-s3/walk.cxpolicy \
      --out artifacts/mini-r2-p7-eff/robustness/full-mini-r2-p7-eff-s3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q8 import robustness as rb  # noqa: E402
from q8.paths import MJCF_DIR  # noqa: E402
from q8.portable import portable_text, write_json  # noqa: E402

LABELS = {"completed": "完走", "max_tilt_deg": "最大傾斜", "slip_share_max": "滑り",
          "effort_rms_rated_ratio_max": "トルク比", "speed_ratio": "速度比", "steps_min": "最少歩数"}


def _f(v, digits=2):
    return "—" if v is None else f"{v:.{digits}f}"


def to_markdown(summary: dict, *, title: str) -> str:
    lines = [f"# {title}", "",
             "各行は評価シード10個（random は 10 × 引いた組の数、push は 10 × 左右）。値は平均。",
             "", "| 組 | 条件 | n | 合格率 | 完走 | 倒れず戻った | 滑り | トルク比 | 最大傾斜 | 速度比 | 不合格の理由 |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for entry in summary.values():
        m = entry["metrics"]
        fails = ", ".join(f"{k} {v}" for k, v in sorted(entry.get("failing", {}).items()))
        lines.append(
            f"| {entry['suite']} | {entry['name']} | {entry['n']} | {_f(entry['pass_rate'])} | "
            f"{_f(m.get('completed'))} | {_f(entry.get('recovered_rate'))} | {_f(m.get('slip_share_max'))} | "
            f"{_f(m.get('effort_rms_rated_ratio_max'))} | {_f(m.get('max_tilt_deg'), 1)} | "
            f"{_f(m.get('speed_ratio'))} | {fails} |")
    return "\n".join(lines) + "\n"


def evaluate(variant: str, policy: Path, out: Path, *, draws: int = 5, processes: int | None = None,
             suites: tuple[str, ...] = ("nominal", "sweep", "random", "push")) -> dict:
    xml = (MJCF_DIR / variant / "model-model.xml").read_bytes()
    task = json.loads((MJCF_DIR / variant / "walk-task.json").read_text(encoding="utf-8"))
    started = time.monotonic()
    rows = rb.run(xml, task, policy.read_bytes(), random_draws=draws, processes=processes, suites=suites)
    summary = rb.summarise(rows)
    kept = [{k: v for k, v in r.items() if k != "metrics"}
            | {"metrics": {k: r["metrics"].get(k) for k in rb.SUMMARY_METRICS}} for r in rows]
    report = {"schema": "q8-robustness-v1", "variant": variant, "policy": portable_text(str(policy.resolve())),
              "settings": rb.settings(), "random_draws": draws, "wall_time_s": time.monotonic() - started,
              "summary": summary, "rows": kept}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "robustness.json", report, indent=1)
    (out / "summary.md").write_text(to_markdown(summary, title=f"実機との差の評価: {policy.parent.name}"),
                                    encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True, help="評価に使う機体（モデルとタスク）。例 mini-r2-p7-eff")
    ap.add_argument("--policy", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--draws", type=int, default=5, help="random の組で、評価シードごとに引く組の数")
    ap.add_argument("--processes", type=int, default=None)
    ap.add_argument("--suites", nargs="+", default=["nominal", "sweep", "random", "push"])
    args = ap.parse_args()
    report = evaluate(args.variant, args.policy, args.out, draws=args.draws, processes=args.processes,
                      suites=tuple(args.suites))
    print(to_markdown(report["summary"], title=args.policy.parent.name))
    print(f"{len(report['rows'])} エピソード、{report['wall_time_s']:.0f} 秒")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
