"""機体を Cadex で生成し、製造データ・MJCF・検証レポートを書き出す（パラメータ違いのバリアントにも使う）。

  1. Cadex プロジェクト（~/cadex-projects/quadruped8[-<variant>]）へ設計スクリプトを書き、パラメータを適用
  2. cadex clearance（全部品の組の干渉）と clearance --sweep（各関節を可動域いっぱいに動かした干渉）
  3. cadex export: STEP / STL（各部品）/ MJCF / 学習タスクJSON
  4. STLをバイナリ化して robot/meshes/<variant>/ へ、MJCFと学習タスクを robot/mjcf/<variant>/ へ
  5. 見た目用MJCF、質量・重心・保持トルク、干渉、印刷可否、BOM を artifacts/<variant>/design_report.json へ

使い方:
  uv run python scripts/build_robot.py                          # 基準機体
  uv run python scripts/build_robot.py --variant leg120 --set thigh_length=54 --set shin_length=60
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from q8 import cadex  # noqa: E402
from q8.clearance import summarise  # noqa: E402
from q8.manufacturing import bom, printability  # noqa: E402
from q8.paths import ARTIFACTS, DESIGN_SCRIPT, MESHES, MJCF_DIR, PROJECT, cadex_project  # noqa: E402
from q8.physics import static_report  # noqa: E402
from q8.render import render_views  # noqa: E402
from q8.stl import read_stl, write_binary_stl  # noqa: E402
from q8.training import make_training_cut  # noqa: E402
from q8.visual import add_visual_meshes  # noqa: E402

LEGS = ("lf", "rf", "lh", "rh")


def intended_contacts() -> set[frozenset]:
    pairs = set()
    for leg in LEGS:
        for j in ("hip", "knee"):
            pairs.add(frozenset((f"c_{j}_servo_{leg}", f"c_{j}_horn_{leg}")))
    return pairs


def parse_sets(items: list[str]) -> dict[str, float]:
    out = {}
    for item in items:
        k, v = item.split("=", 1)
        out[k.strip()] = float(v)
    return out


def _json_default(o):
    return o.item() if hasattr(o, "item") else str(o)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="base")
    ap.add_argument("--set", action="append", default=[], help="パラメータ k=v（複数可）")
    ap.add_argument("--skip-build", action="store_true", help="Cadexのビルドと干渉計測を省き、受理済みの出力を使う")
    ap.add_argument("--no-render", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    project = cadex_project(args.variant)
    sets = parse_sets(args.set)
    timings = {}
    if not args.skip_build:
        # パラメータの値はプロジェクトに保存され、スクリプトを書き直しても残る（docs/CLI.md）。
        cadex.write_script(project, DESIGN_SCRIPT, sets=sets, fit=False)
        timings["build_s"] = round(time.time() - t0, 1)
        t1 = time.time()
        cadex.clearance(project)
        timings["clearance_s"] = round(time.time() - t1, 1)
    export_dir = ARTIFACTS / "tmp" / f"export-{args.variant}"
    if export_dir.exists():
        shutil.rmtree(export_dir)
    cadex.export(project, export_dir)
    params = cadex.params(project)

    # 製造データ: STEP はそのまま、STL はバイナリ化（スライサーも MuJoCo も読める）
    mesh_out = MESHES / args.variant
    if mesh_out.exists():
        shutil.rmtree(mesh_out)
    mesh_out.mkdir(parents=True)
    for f in sorted(export_dir.iterdir()):
        if f.suffix == ".stl":
            write_binary_stl(read_stl(f), mesh_out / f.name)
        elif f.suffix == ".step":
            shutil.copy2(f, mesh_out / f.name)
    # MJCF と学習タスク（モデルとタスクは同じ場所に並べる。タスクはモデルを相対パスとsha256で参照する）
    mjcf_out = MJCF_DIR / args.variant
    if mjcf_out.exists():
        shutil.rmtree(mjcf_out)
    mjcf_out.mkdir(parents=True)
    for f in sorted(export_dir.iterdir()):
        if f.suffix in (".xml", ".json"):
            shutil.copy2(f, mjcf_out / f.name)
    model_xml = next(mjcf_out.glob("*-model.xml"))
    # 学習用カットモデル（CPUで約9倍速い。評価はエンジンのモデルで行う）
    cut_info = None
    for task_json in mjcf_out.glob("*-task.json"):
        cut_info = make_training_cut(project, task_json, mjcf_out)
    visual_xml = add_visual_meshes(model_xml, mesh_out, mjcf_out / "visual.xml",
                                   mesh_prefix=f"../../meshes/{args.variant}")

    # 検証
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    stall = float(max(model.actuator_forcerange[:, 1]))
    rated = stall * float(params.get("rated_fraction", 0.4))
    report = {
        "variant": args.variant,
        "cadex_project": str(project),
        "accepted_revision": json.loads((project / "script.json").read_text())["accepted_revision"],
        "design_script": str(DESIGN_SCRIPT.relative_to(PROJECT)),
        "params": params,
        "units": "mm, deg, g, N*mm（表記のないものはSI）",
        "servo": {"stall_torque_nmm": round(stall * 1000, 1), "rated_torque_nmm_estimate": round(rated * 1000, 1),
                  "source": "CadexCatalog.py の SERVOS（TowerPro公表値 4.8 V）。連続定格は失速の40%とした推定"},
        "physics": static_report(model, stall, rated),
        "clearance": summarise(project, intended_contacts()),
        "printability": printability(mesh_out, params),
        "joints": {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j): [round(float(v) * 57.29578, 2) for v in model.jnt_range[j]]
                   for j in range(1, model.njnt)},
        "timings_s": timings,
        "training_cut": cut_info,
        "files": {"meshes": str(mesh_out.relative_to(PROJECT)), "mjcf": str(model_xml.relative_to(PROJECT)),
                  "visual_mjcf": str(visual_xml.relative_to(PROJECT)),
                  "task": [str(p.relative_to(PROJECT)) for p in mjcf_out.glob("*-task.json")]},
    }
    report["bom"] = bom(cadex.inventory(project), report["printability"], PROJECT / "configs" / "bom_prices.json",
                        float(params.get("pla_density", 800)))
    phys = report["physics"]
    report["verdict"] = {
        "self_collision_free_over_ranges": report["clearance"]["self_collision_free"],
        "printable_on_home_fdm": all(report["printability"]["checks"][k] for k in
                                     ("all_parts_fit_bed", "wall_at_least_4_perimeters", "link_plate_at_least_5mm")),
        "com_inside_support_polygon": phys["support_margin_mm"] > 10.0,
        "stands_in_sim": phys["sim_hold"]["tilt_deg"] < 3.0 and phys["sim_hold"]["body_drop_mm"] < 10.0,
        "trot_static_torque_within_rated": phys["static_estimate"]["2脚支持（トロット）"]["rated_ratio"] <= 1.0,
        "within_budget_jpy_10000": report["bom"]["within_budget"],
    }
    out_dir = ARTIFACTS / args.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "design_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=_json_default))
    if not args.no_render:
        vm = mujoco.MjModel.from_xml_path(str(visual_xml))
        vd = mujoco.MjData(vm)
        mujoco.mj_resetDataKeyframe(vm, vd, 0)
        mujoco.mj_forward(vm, vd)
        render_views(vm, vd, out_dir / "renders")
        render_views(vm, vd, out_dir / "renders", show_collision=True)
    print(json.dumps(report["verdict"], ensure_ascii=False, indent=2, default=_json_default))
    print(f"report: {out_dir / 'design_report.json'}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
