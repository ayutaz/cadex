"""Phase 9: 形状を変えて、再生成・再検証・（再学習）・評価・比較までを1コマンドで行う。

例: 「脚を20%長くして歩行性能を比較して」
  uv run python scripts/redesign.py --variant leg120 --scale thigh_length,shin_length=1.2

  1. build            基準機体のパラメータに倍率/指定値を適用し、Cadexで再生成（STEP/STL/MJCF/学習タスク）、
                      全部品の干渉と各関節の可動域スイープを再検証（scripts/build_robot.py）
  2. train_variant    --train local: 新しい機体の学習タスクで PPO を学習（CPU。GPUは vast.ai の手順を使う）
  3. evaluate_*       基準と新機体を、同じシード・同じ成功条件・同じ指標で評価（evaluation/evaluate_variant.py）
  4. compare          artifacts/compare/base_vs_<variant>.json / .md

学習済み方策がある場合は --policy-base / --policy-variant で渡す（無ければ固定歩容だけを比べる）。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))


def parse_scale(text: str) -> tuple[list[str], float]:
    names, factor = text.split("=", 1)
    return [n.strip() for n in names.split(",") if n.strip()], float(factor)


def resolve_sets(base: dict, *, scales: list[tuple[list[str], float]], sets: dict[str, float]) -> dict[str, float]:
    out: dict[str, float] = {}
    for names, factor in scales:
        for n in names:
            if n not in base:
                raise KeyError(f"基準機体に {n!r} というパラメータは無い")
            out[n] = float(base[n]) * factor
    for n, v in sets.items():
        if n not in base:
            raise KeyError(f"基準機体に {n!r} というパラメータは無い")
        out[n] = float(v)
    return out


def plan_steps(variant: str, *, train: str, policy_base: Path | None) -> list[tuple[str, str]]:
    steps = [("build", f"Cadexで {variant} を再生成し、干渉と可動域を再検証")]
    if train == "local":
        steps.append(("train_variant", f"{variant} の学習タスクで PPO（CPU）"))
    steps += [("evaluate_base", "基準機体を評価"), ("evaluate_variant", f"{variant} を評価"),
              ("compare", "比較表を書く")]
    return steps


def train_local(variant: str, iterations: int, envs: int, hidden: list[int]) -> Path:
    from q8.paths import ARTIFACTS, MJCF_DIR, PROJECT, TRAINER

    from q8.training import cut_task_path

    export_task = next((MJCF_DIR / variant).glob("*-task.json"))
    cut = cut_task_path(MJCF_DIR / variant)
    task = cut if cut.exists() else export_task  # カットモデルがあればそれで学習（約9倍速い）
    out = ARTIFACTS / "runs" / f"{variant}-local"
    out.mkdir(parents=True, exist_ok=True)
    policy = out / "walk.cxpolicy"
    cmd = [str(PROJECT / ".venv" / "bin" / "python"), str(TRAINER), str(task), "--out", str(policy),
           "--envs", str(envs), "--iterations", str(iterations), "--seed", "0", "--hidden", *map(str, hidden),
           "--checkpoint-every", "50", "--progress", str(out / "progress.json"),
           "--action-filter-alpha", "0.6", "--no-load-speed", "10.47"]
    if task == cut:
        cmd += ["--cad-model", str(next((MJCF_DIR / variant).glob("*-model.xml")))]
    with (out / "receipt.json").open("w") as receipt, (out / "stderr.log").open("w") as err:
        subprocess.run(cmd, check=True, stdout=receipt, stderr=err, env={**__import__("os").environ, "JAX_PLATFORMS": "cpu"})
    return policy


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--scale", action="append", default=[], help="名前[,名前...]=倍率（基準機体の値に掛ける）")
    ap.add_argument("--set", action="append", default=[], help="名前=値")
    ap.add_argument("--train", choices=["none", "local"], default="none")
    ap.add_argument("--iterations", type=int, default=300)
    ap.add_argument("--envs", type=int, default=256)
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 64])
    ap.add_argument("--policy-base", type=Path, default=None)
    ap.add_argument("--policy-variant", type=Path, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    from q8 import cadex
    from q8.compare import compare, to_markdown
    from q8.paths import ARTIFACTS, cadex_project

    base_params = cadex.params(cadex_project("base"))
    sets = resolve_sets(base_params, scales=[parse_scale(s) for s in args.scale],
                        sets={k: float(v) for k, v in (s.split("=", 1) for s in args.set)})
    steps = plan_steps(args.variant, train=args.train, policy_base=args.policy_base)
    print(json.dumps({"variant": args.variant, "set": sets, "steps": steps}, ensure_ascii=False, indent=2))
    if args.dry_run:
        return 0
    # 関節制限など、基準機体で決めた値は引き継ぐ（変更したものだけ上書き）
    carry = {k: v for k, v in base_params.items() if k.endswith(("_min", "_max"))}
    build = [sys.executable, str(ROOT / "scripts" / "build_robot.py"), "--variant", args.variant]
    for k, v in {**carry, **sets}.items():
        build += ["--set", f"{k}={v}"]
    subprocess.run(build, check=True)
    policy_variant = args.policy_variant
    if args.train == "local":
        policy_variant = train_local(args.variant, args.iterations, args.envs, args.hidden)
    from evaluate_variant import evaluate_variant

    base = evaluate_variant("base", policy=args.policy_base, video=False)
    var = evaluate_variant(args.variant, policy=policy_variant, video=False)
    c = compare({"design": base["design"], "fixed_gait": base["fixed_gait"], "policy": base["policy"]},
                {"design": var["design"], "fixed_gait": var["fixed_gait"], "policy": var["policy"]},
                labels=("base", args.variant))
    c["set"] = sets
    out = ARTIFACTS / "compare"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"base_vs_{args.variant}.json").write_text(json.dumps(c, ensure_ascii=False, indent=2))
    (out / f"base_vs_{args.variant}.md").write_text(to_markdown(c))
    print(to_markdown(c))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
