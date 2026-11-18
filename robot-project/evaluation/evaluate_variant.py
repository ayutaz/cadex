"""Phase 7: 1機体を評価する。固定歩容（MuJoCo上で開ループ）と、学習済み方策（`cadex evaluate`）を、
同じ成功条件・同じシード・同じ指標計算（CadexEvaluation）で並べる。

出力: artifacts/<variant>/evaluation/
  fixed_gait.json   固定歩容のシードごとの判定と指標（四足でない機体では {"skipped": 理由}）
  policy/           cadex evaluate の出力（evaluation.json、シードごとのトレース、動画）
  summary.json      設計値・合格率・主要指標の平均/最小/最大
  metrics.csv       シードごとの主要指標（固定歩容と方策を縦に並べる）

使い方:
  uv run python evaluation/evaluate_variant.py --variant base [--policy artifacts/runs/xxx/walk.cxpolicy]
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from q8.compare import KEY_METRICS, design_summary, summarise_seeds  # noqa: E402
from q8.evaluation import evaluate_fixed_gait, metrics_table  # noqa: E402
from q8.gait import TrotParams, trot_targets  # noqa: E402
from q8.paths import ARTIFACTS, CADEX_CLI, MJCF_DIR, cadex_project  # noqa: E402
from q8.portable import portable_text, scrub_tree, write_json  # noqa: E402
from q8.robot import LEGS, leg_geometry, model_legs  # noqa: E402

FIXED = {"period_s": 0.5, "stride_m": 0.04, "lift_m": 0.008}   # simulation/fixed_gait.py の探索で決めた値
# 固定歩容は四足のトロット（q8/gait.py の trot_targets、lf/rf/lh/rh）の目標しか作らない。脚の組が違う機体
# （3本足の lf/rf/hc など）で回すと、無い脚の目標は捨てられ、ある脚（hc）は 0 のまま固定されて、四足の
# トロットとして黙って記録される。そこで評価せずに飛ばし、その理由を出力に残す。しっぽは脚に入らないので
# しっぽ付きの四足は評価する（しっぽの関節は目標が無いので 0 = 立位のまま）。
SKIP_NOT_QUADRUPED = "四足でないため（固定歩容は lf/rf/lh/rh のトロット専用）"


def fixed_gait_target(model, *, period_s: float, stride_m: float, lift_m: float):
    geo = leg_geometry(model)
    p = TrotParams(period_s=period_s, stride_m=stride_m, lift_m=lift_m)
    return (lambda t: trot_targets(t, geo, p)), p.nominal_speed_m_s * 1000.0


def read_cadex_evaluation(out_dir: Path) -> dict:
    ev = json.loads((out_dir / "evaluation.json").read_text())
    return {"verdict": ev.get("verdict"), "seeds": ev["seeds"], "wall_time_s": ev.get("wall_time_s"),
            "summary": ev.get("summary")}


def run_cadex_evaluate(variant: str, policy: Path, out_dir: Path, video: bool = True) -> dict:
    cmd = [str(CADEX_CLI), "--project", str(cadex_project(variant)), "--wait", "--json", "evaluate",
           "--task", "walk", "--weights", str(policy.resolve()), "--out", str(out_dir)]
    if not video:
        cmd.append("--no-video")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if not (out_dir / "evaluation.json").exists():
        raise RuntimeError(f"cadex evaluate が失敗: {proc.stderr[-2000:]}")
    scrub_tree(out_dir)  # Cadex は重みやエンジンの場所を絶対パスで記録する
    return read_cadex_evaluation(out_dir)


def evaluate_variant(variant: str, *, policy: Path | None, out_dir: Path | None = None, seconds: float | None = None,
                     video: bool = True) -> dict:
    out_dir = out_dir or ARTIFACTS / variant / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_xml = next((MJCF_DIR / variant).glob("*-model.xml"))
    task = json.loads(next((MJCF_DIR / variant).glob("*-task.json")).read_text())
    seeds = task["success"]["seeds"]
    seconds = seconds or task["success"]["episode"]["episode_seconds"]
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    legs = model_legs(model)
    if sorted(legs) != sorted(LEGS):
        fixed = {"skipped": SKIP_NOT_QUADRUPED}
        write_json(out_dir / "fixed_gait.json", {**fixed, "legs": legs}, indent=2)
    else:
        target, nominal = fixed_gait_target(model, **FIXED)
        fixed = evaluate_fixed_gait(model, task, target_fn=target, seeds=seeds, seconds=seconds,
                                    command_mm_s=nominal)
        fixed["gait"] = {**FIXED, "nominal_speed_mm_s": nominal, "seconds": seconds}
        write_json(out_dir / "fixed_gait.json", fixed, indent=2, default=str)
    pol = None
    if policy is not None:
        pol = run_cadex_evaluate(variant, policy, out_dir / "policy", video=video)
        pol["weights"] = portable_text(str(policy))
    design = json.loads((ARTIFACTS / variant / "design_report.json").read_text())
    summary = {"variant": variant, "design": design_summary(design),
               "fixed_gait": fixed if "skipped" in fixed else summarise_seeds(fixed, KEY_METRICS),
               "policy": summarise_seeds(pol, KEY_METRICS),
               "metric_labels": KEY_METRICS}
    write_json(out_dir / "summary.json", summary, indent=2)
    rows = [] if "skipped" in fixed else metrics_table(fixed, label="fixed_gait", keys=list(KEY_METRICS))
    if pol is not None:
        rows += metrics_table(pol, label="policy", keys=list(KEY_METRICS))
    with (out_dir / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["label", "seed", "pass", *KEY_METRICS])
        w.writeheader()
        w.writerows(rows)
    return {"design": summary["design"], "fixed_gait": fixed, "policy": pol, "summary": summary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--policy", type=Path, default=None)
    ap.add_argument("--no-video", action="store_true")
    args = ap.parse_args()
    res = evaluate_variant(args.variant, policy=args.policy, video=not args.no_video)
    s = res["summary"]
    print(json.dumps({"fixed_gait_pass_rate": s["fixed_gait"].get("pass_rate", s["fixed_gait"].get("skipped")),
                      "policy_pass_rate": None if s["policy"] is None else s["policy"]["pass_rate"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
