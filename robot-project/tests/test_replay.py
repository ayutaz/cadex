"""評価トレース（cadex evaluate が書く seed-*-trace.json）を MuJoCo の姿勢に戻して再生する。"""

import math
import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from replay_trace import qpos_from_frame  # noqa: E402

from q8.evaluation import record_episode  # noqa: E402


def test_frame_is_turned_back_into_the_same_pose(base_model):
    ep = record_episode(base_model, target_fn=lambda t: {"hip_lf": math.radians(10 * math.sin(6 * t))},
                        seconds=0.5, control_hz=50, seed=3, command_mm_s=100.0)
    frame = ep["trace"]["frames"][-1]
    data = mujoco.MjData(base_model)
    data.qpos[:] = qpos_from_frame(base_model, frame, ep["trace"]["actuator_channels"])
    mujoco.mj_forward(base_model, data)
    foot = mujoco.mj_name2id(base_model, mujoco.mjtObj.mjOBJ_BODY, "c_foot_lf")
    want = np.array(frame["component_placements"]["c_foot_lf"]["position_mm"]) / 1000.0
    assert np.allclose(data.xpos[foot], want, atol=1e-6)


def test_frame_without_joint_positions_is_refused(base_model):
    with pytest.raises(KeyError):
        qpos_from_frame(base_model, {"component_placements": {}}, [])
