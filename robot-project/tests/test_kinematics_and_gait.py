import math

import pytest

from q8.gait import TrotParams, foot_offset, trot_targets
from q8.robot import JOINTS, LegGeometry, leg_geometry

GEO = LegGeometry(thigh=0.045, shin=0.050, stand_hip=math.radians(30), stand_shin=-math.asin(0.045 * 0.5 / 0.050))


def test_standing_foot_is_under_the_hip():
    x, z = GEO.foot(0.0, 0.0)
    assert abs(x) < 1e-9
    assert z < -0.08


@pytest.mark.parametrize("dx,dz", [(0.0, 0.0), (0.015, 0.0), (-0.02, 0.008), (0.01, 0.012)])
def test_ik_inverts_fk(dx, dz):
    x0, z0 = GEO.foot(0.0, 0.0)
    qh, qk = GEO.ik(x0 + dx, z0 + dz)
    x, z = GEO.foot(qh, qk)
    assert abs(x - (x0 + dx)) < 1e-9 and abs(z - (z0 + dz)) < 1e-9


def test_ik_of_standing_pose_is_zero():
    qh, qk = GEO.ik(*GEO.foot(0.0, 0.0))
    assert abs(qh) < 1e-9 and abs(qk) < 1e-9


def test_foot_offset_is_continuous_and_lifts_only_in_swing():
    stride, lift, duty = 0.04, 0.01, 0.5
    prev = foot_offset(0.0, stride, lift, duty)
    assert prev == (stride / 2, 0.0)
    for i in range(1, 401):
        cur = foot_offset(i / 400, stride, lift, duty)
        assert abs(cur[0] - prev[0]) < stride / 50 and abs(cur[1] - prev[1]) < lift / 20
        if i / 400 < duty:
            assert cur[1] == 0.0
        prev = cur
    assert foot_offset(0.25, stride, lift, duty)[0] == pytest.approx(0.0)       # 立脚の中間
    assert foot_offset(0.75, stride, lift, duty)[1] == pytest.approx(lift)      # 遊脚の頂点


def test_trot_starts_from_standing_and_pairs_diagonals():
    p = TrotParams(ramp_s=1.0)
    t0 = trot_targets(0.0, GEO, p)
    assert set(t0) == set(JOINTS)
    assert all(abs(v) < 1e-9 for v in t0.values())
    t = trot_targets(2.13, GEO, p)
    assert t["hip_lf"] == pytest.approx(t["hip_rh"]) and t["hip_rf"] == pytest.approx(t["hip_lh"])
    assert t["hip_lf"] != pytest.approx(t["hip_rf"])


def test_positive_turn_shortens_the_left_stride():
    p = TrotParams(ramp_s=0.0, turn=0.5, stride_m=0.04)
    # 立脚の始め（phase 0）は足先が前方 +stride/2。左（lf）は歩幅が短いので前に出る量が小さい。
    t = trot_targets(0.0, GEO, p)
    x0, _ = GEO.foot(0.0, 0.0)
    xl, _ = GEO.foot(t["hip_lf"], t["knee_lf"])
    xr, _ = GEO.foot(t["hip_rh"], t["knee_rh"])
    assert (xl - x0) == pytest.approx(0.04 * 0.5 / 2)
    assert (xr - x0) == pytest.approx(0.04 * 1.5 / 2)


def test_leg_geometry_from_exported_model(base_model):
    geo = leg_geometry(base_model)
    assert geo.thigh == pytest.approx(0.045, abs=1e-4)
    assert geo.shin == pytest.approx(0.050, abs=1e-4)
    assert abs(geo.foot(0.0, 0.0)[0]) < 1e-4  # 立位で足先は股軸の真下


def test_unwrapped_yaw_accumulates_past_180():
    from q8.robot import YawTracker

    tr = YawTracker()
    for deg in list(range(0, 180, 10)) + list(range(-180, -90, 10)):
        tr.update(deg)
    assert tr.total == pytest.approx(260.0)
