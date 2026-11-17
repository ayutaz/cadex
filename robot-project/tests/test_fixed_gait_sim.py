"""固定モーションを実際に MuJoCo で動かす結合テスト（出力済みの基準機体の MJCF を使う）。"""

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "simulation"))

from fixed_gait import joint_direction_check, run_scene  # noqa: E402
from q8.gait import TrotParams, trot_targets  # noqa: E402
from q8.robot import JOINTS, leg_geometry  # noqa: E402


@pytest.fixture(scope="module")
def stall(base_model):
    return float(base_model.actuator_forcerange[0, 1])


def test_every_joint_moves_the_way_it_is_commanded(base_model):
    result = joint_direction_check(base_model)
    assert set(result) == set(JOINTS)
    assert all(v["ok"] for v in result.values()), result


def test_standing_pose_holds(base_model, stall, tmp_path):
    r = run_scene(base_model, "stand", lambda t: {n: 0.0 for n in JOINTS}, 2.0, tmp_path, False, stall)
    assert r["max_tilt_deg"] < 2.0
    assert r["max_base_drop_mm"] < 5.0
    assert r["joint_limit_violation_deg"] < 0.5
    assert not r["non_foot_floor_contacts_mm"]


def test_fixed_trot_walks_forward_without_violations(base_model, stall, tmp_path):
    geo = leg_geometry(base_model)
    p = TrotParams(period_s=0.5, stride_m=0.04, lift_m=0.008)
    r = run_scene(base_model, "trot", lambda t: trot_targets(t, geo, p), 5.0, tmp_path, False, stall)
    assert r["mean_speed_x_mm_s"] > 30.0
    assert r["max_tilt_deg"] < 15.0
    assert r["joint_limit_violation_deg"] < 1.0
    assert not r["non_foot_floor_contacts_mm"]
    assert abs(r["final_yaw_deg"]) < 15.0


@pytest.mark.parametrize("turn,sign", [(0.4, 1.0), (-0.4, -1.0)])
def test_turn_command_turns_the_commanded_way(base_model, stall, tmp_path, turn, sign):
    """turn > 0 は左の歩幅を縮める → 左旋回（ヨー角が正）。"""
    geo = leg_geometry(base_model)
    p = TrotParams(period_s=0.5, stride_m=0.04, lift_m=0.008, turn=turn)
    r = run_scene(base_model, "turn", lambda t: trot_targets(t, geo, p), 5.0, tmp_path, False, stall)
    assert sign * r["final_yaw_deg"] > 10.0, r["final_yaw_deg"]
