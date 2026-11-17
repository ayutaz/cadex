"""Cadexが書いた MJCF から URDF を作る。

使い方: .venv/bin/python scripts/mjcf_to_urdf.py --variant base
  入力: robot/mjcf/<variant>/model-model.xml
  見た目: robot/meshes/<variant>/*.stl（無ければ見た目なしで書く。STL は scripts/build_robot.py が作る）
  出力: robot/urdf/<variant>/quadruped8.urdf
変換の前提と制約は docs/urdf.md。
"""

from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from q8.paths import MESHES, MJCF_DIR, URDF_DIR  # noqa: E402
from q8.urdf import MG90S_VELOCITY, write_urdf  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="base", help="機体バリアント（既定 base）")
    ap.add_argument("--mjcf", type=Path, help="入力 MJCF（既定 robot/mjcf/<variant>/model-model.xml）")
    ap.add_argument("--meshes", type=Path, help="STL のディレクトリ（既定 robot/meshes/<variant>）")
    ap.add_argument("--out", type=Path, help="出力 URDF（既定 robot/urdf/<variant>/quadruped8.urdf）")
    ap.add_argument("--velocity", type=float, default=MG90S_VELOCITY,
                    help=f"関節の速度上限 [rad/s]（既定 MG90S 4.8 V の 0.10 s/60° = {MG90S_VELOCITY:.4f}）")
    ap.add_argument("--no-mujoco-tag", action="store_true",
                    help="MuJoCo 用の <mujoco><compiler/></mujoco> 要素を入れない")
    args = ap.parse_args()

    mjcf = args.mjcf or MJCF_DIR / args.variant / "model-model.xml"
    meshes = args.meshes or MESHES / args.variant
    out = args.out or URDF_DIR / args.variant / "quadruped8.urdf"
    if not mjcf.exists():
        print(f"MJCF がない: {mjcf}", file=sys.stderr)
        return 1
    if not meshes.is_dir():
        print(f"警告: STL のディレクトリがない（見た目なしで書く）: {meshes}", file=sys.stderr)
        meshes = None

    write_urdf(mjcf, out, mesh_dir=meshes, velocity=args.velocity, robot_name="quadruped8",
               mujoco_tag=not args.no_mujoco_tag)

    root = ET.parse(out).getroot()
    joints = root.findall("joint")
    kinds = {k: sum(j.get("type") == k for j in joints) for k in ("revolute", "continuous", "fixed")}
    nvis = len(root.findall("link/visual"))
    nlink = len(root.findall("link"))
    mass = sum(float(e.get("value")) for e in root.iter("mass"))
    print(f"{out}")
    print(f"  リンク {nlink}、関節 revolute {kinds['revolute']} / continuous {kinds['continuous']} / "
          f"fixed {kinds['fixed']}、見た目 {nvis}、総質量 {mass * 1000:.3f} g")
    if meshes is not None and nvis < nlink:
        print(f"  警告: STL が見つからないリンクが {nlink - nvis} 個ある", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
