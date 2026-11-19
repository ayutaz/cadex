"""1本の脚を上げたとき、重心を残る2本の足を結ぶ線にどこまで寄せられるか（q8.tripod_reach）の検査。

比べる値は、調査の報告（.superpowers/sdd/2026-11-18-tripod-and-tail/diag-tripod-report.md）の表から写した。
- 立位の表: 重心と支持線の距離 lf 19.8、rf 31.9、hc 43.2 mm。どれも重心が上げた足の側に出る（この関数では負）。
- 探索の表（胴体の傾き 10° 以内）: lf +1.9、rf −13.3、hc +13.1 mm。乱数 2 万通りで見つけた最良の値なので、
  本当の最良はこれ以上になりうる（報告のとおり）。
tri-r2-free-dr と tri-r2-wave-dr は、カットモデルの形が同じ（違うのはアクチュエータの行だけ）。
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import pytest

from q8 import tripod_reach

ROOT = Path(__file__).resolve().parents[1]
WAVE = ROOT / "artifacts" / "tri-r2-wave-dr"
DYN = ROOT / "artifacts" / "tri-r2-dyn-dr"
LEGS = ("lf", "rf", "hc")

STAND_DIAG = {"lf": -19.8, "rf": -31.9, "hc": -43.2}
BEST_DIAG_TILT10 = {"lf": 1.9, "rf": -13.3, "hc": 13.1}


def _need(bundle: Path) -> None:
    if not (bundle / "cut" / "cut.xml").exists():
        pytest.skip(f"{bundle} の DR バンドルが無い")


def _stand(bundle: Path):
    model = mujoco.MjModel.from_xml_path(str(tripod_reach.bundle_xml(bundle)))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


@pytest.fixture(scope="module")
def best_wave() -> dict[str, dict]:
    _need(WAVE)
    return {leg: tripod_reach.best_lift_margin(WAVE, leg) for leg in LEGS}


@pytest.fixture(scope="module")
def best_dyn() -> dict[str, dict]:
    _need(DYN)
    return {leg: tripod_reach.best_lift_margin(DYN, leg) for leg in LEGS}


def test_stand_margins_match_diag():
    _need(WAVE)
    model, data = _stand(WAVE)
    for leg in LEGS:
        assert tripod_reach.lift_margin(model, data, leg) == pytest.approx(STAND_DIAG[leg], abs=0.5), leg


def test_best_matches_diag_search(best_wave):
    """傾き 10° 以内の探索は、調査の乱数探索の値（見つかった下限）に届く。"""
    for leg in LEGS:
        got = best_wave[leg]
        assert got["feasible"], leg
        assert got["margin_mm"] >= BEST_DIAG_TILT10[leg] - 0.5, (leg, got)


def test_best_not_worse_than_stand(best_wave, best_dyn):
    for bundle, best in ((WAVE, best_wave), (DYN, best_dyn)):
        model, data = _stand(bundle)
        for leg in LEGS:
            assert best[leg]["margin_mm"] >= tripod_reach.lift_margin(model, data, leg), (bundle.name, leg)


def test_best_pose_meets_conditions(best_dyn):
    """返した関節角と姿勢を MuJoCo に置くと、支えの2本は床に着き、上げた足は 10 mm 以上浮き、余裕も同じになる。"""
    model = mujoco.MjModel.from_xml_path(str(tripod_reach.bundle_xml(DYN)))
    data = mujoco.MjData(model)
    for leg in LEGS:
        got = best_dyn[leg]
        feet = tripod_reach.place(model, data, leg, got["joint_deg"], got["phi_deg"])
        radius_mm = {k: v["radius_mm"] for k, v in feet.items()}
        z_mm = {k: v["z_mm"] for k, v in feet.items()}
        for other in LEGS:
            if other != leg:
                assert abs(z_mm[other] - radius_mm[other]) <= 1.0, (leg, other, z_mm)
        assert z_mm[leg] - radius_mm[leg] >= 10.0 - 1e-6, (leg, z_mm)
        assert tripod_reach.lift_margin(model, data, leg) == pytest.approx(got["margin_mm"], abs=1e-6)
        assert got["tilt_deg"] <= 10.0 + 1e-6


def test_pitch_only_cannot_lift_all(best_dyn):
    """ロールが無い tri-r2-dyn では、3本のどれでも静的に上げられる姿勢は無い（3本の最小が負）。"""
    assert min(best_dyn[leg]["margin_mm"] for leg in LEGS) < 0


@pytest.mark.parametrize("variant", ["tri3-r8", "tri3-r9"])
def test_roll_bundle(variant):
    """ロールの関節がある機体でも動き、``roll_range_deg`` でロールの範囲を置き換える。"""
    bundle = ROOT / "artifacts" / f"{variant}-dr"
    _need(bundle)
    got = tripod_reach.best_lift_margin(bundle, "rf", roll_range_deg=5.0, samples=2000)
    rolls = {k: v for k, v in got["joint_deg"].items() if k.startswith("roll_")}
    assert rolls, got["joint_deg"]
    assert all(abs(v) <= 5.0 + 1e-6 for v in rolls.values()), rolls
