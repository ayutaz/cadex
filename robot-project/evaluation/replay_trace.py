"""評価トレース（cadex evaluate の seed-*-trace.json、または固定歩容の記録）を、
CADメッシュ付きの MuJoCo モデルで再生して動画にする。物理は解かず、記録された姿勢をそのまま置く。

使い方:
  uv run python evaluation/replay_trace.py robot/mjcf/base/visual.xml \
      artifacts/taskv3/evaluation/policy/seed-2101-trace.json artifacts/videos/policy-seed2101.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from q8.render import VideoWriter, camera  # noqa: E402


def qpos_from_frame(model: mujoco.MjModel, frame: dict, channels: list[dict]) -> np.ndarray:
    """フレームの胴体の配置（mm, xyzw）と関節角（deg）から qpos を作る。"""
    joints = frame["joint_positions"]
    base = frame["component_placements"]["c_chassis"]
    qpos = model.qpos0.copy()
    x, y, z, w = base["rotation_xyzw"]
    # 胴体のボディ座標系はワールド原点（立位）にあるので、free joint の位置 = 配置の位置
    qpos[0:3] = np.array(base["position_mm"]) / 1000.0
    qpos[3:7] = [w, x, y, z]
    for ch, deg in zip(channels, joints):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, ch["joint"])
        qpos[model.jnt_qposadr[jid]] = math.radians(deg)
    return qpos


def replay(model_xml: Path, trace_path: Path, out: Path, *, width: int = 768, height: int = 480) -> Path:
    model = mujoco.MjModel.from_xml_path(str(model_xml))
    data = mujoco.MjData(model)
    trace = json.loads(trace_path.read_text())
    frames = [f for f in trace["frames"] if f.get("frame_kind") == "solver_output" and "joint_positions" in f]
    hz = int(trace.get("dynamics", {}).get("control_hz") or 50)
    renderer = mujoco.Renderer(model, height, width)
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]
    writer = VideoWriter(out, width, height, hz)
    for f in frames:
        data.qpos[:] = qpos_from_frame(model, f, trace["actuator_channels"])
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera(150.0, -18.0, 0.45, data.subtree_com[1] + np.array([0, 0, -0.02])),
                              scene_option=opt)
        writer.write(renderer.render())
    renderer.close()
    return writer.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", type=Path)
    ap.add_argument("trace", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    print(replay(args.model, args.trace, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
