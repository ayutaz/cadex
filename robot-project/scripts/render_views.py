"""出力済みのMJCF（見た目用メッシュ付き）を、立位の初期姿勢で複数視点から描画する。

使い方: uv run python scripts/render_views.py robot/mjcf/base/visual.xml artifacts/base/renders
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402

from q8.render import render_views  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model")
    ap.add_argument("out_dir")
    ap.add_argument("--collision", action="store_true", help="衝突形状も重ねて描く")
    args = ap.parse_args()
    m = mujoco.MjModel.from_xml_path(args.model)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, 0)
    mujoco.mj_forward(m, d)
    for p in render_views(m, d, Path(args.out_dir), show_collision=args.collision):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
