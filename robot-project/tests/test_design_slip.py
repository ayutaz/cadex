"""足の滑りの罰（slip_cost）: すり足（足を上げずに滑らせて進む）に点が入らないようにする。

slip_cost=0（既定）では報酬の項も観測も足さず、今の出力（四足の既定、mini-r2-p7-eff、tri-r2、tri-r2-wave の
基準ログ）を1バイトも変えない。0 より大きいと、足ごとに足の重心の速度の観測 fv_<脚>（centre_of_mass_velocity。
足は MJCF の葉のボディなので、足そのものの重心の速度）を足し、報酬の項 foot_slip（重み -slip_cost）を足す。
foot_slip は「接地している足の、床に沿った速さ（100 mm/s で正規化）の2乗」の脚の平均（浮いた足は 0 と数える）。
偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from test_design_battery import run_design
from test_design_tail import DATA, MINI_R2_P7_EFF, of_kind, signature
from test_design_tripod import evaluate

TRI_R2 = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=0.0)
TRI_R2_WAVE = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=1.0)
QUAD_LEGS = ("lf", "rf", "lh", "rh")
TRI_LEGS = ("lf", "rf", "hc")
ROOT = Path(__file__).resolve().parents[1]


def _reward(log, label):
    found = [t for t in of_kind(log, "assembly.reward") if t.kwargs.get("label") == label]
    return found


def _obs(log, name):
    return [t for t in of_kind(log, "assembly.observation") if t.kwargs.get("name") == name]


@pytest.mark.parametrize("name, overrides", [
    ("default", {}), ("mini-r2-p7-eff", MINI_R2_P7_EFF), ("tri-r2", TRI_R2), ("tri-r2-wave", TRI_R2_WAVE)])
def test_slip_cost_zero_unchanged(name, overrides):
    baseline = json.loads((DATA / f"design_log_{name}.json").read_text())
    _, log = run_design(**overrides)
    assert signature(log) == baseline
    _, log0 = run_design(**overrides, slip_cost=0.0)
    assert signature(log0) == baseline
    assert not _reward(log0, "foot_slip")


@pytest.mark.parametrize("overrides, legs", [({}, QUAD_LEGS), (MINI_R2_P7_EFF, QUAD_LEGS),
                                             (TRI_R2, TRI_LEGS), (TRI_R2_WAVE, TRI_LEGS)])
def test_slip_term_present(overrides, legs):
    result, log = run_design(**overrides, slip_cost=1.0)
    (term,) = _reward(log, "foot_slip")
    assert term.kwargs["weight"] == -1.0
    # 足ごとの速度の観測（足の重心の速度、privileged）
    for leg in legs:
        (o,) = _obs(log, "fv_" + leg)
        assert o.args[0] is result["c_foot_" + leg]
        assert o.args[1] == "centre_of_mass_velocity"
        assert o.kwargs["role"] == "privileged"
    # 観測の後に足した観測は、報酬と同じ脚の名前（LEG_NAMES）だけ
    assert sorted(t.kwargs["name"] for t in of_kind(log, "assembly.observation")
                  if t.kwargs["name"].startswith("fv_")) == sorted("fv_" + leg for leg in legs)

    expr = term.args[0]
    n = len(legs)

    def f(down, vx=0.0, vy=0.0, vz=0.0):
        values = {}
        for leg in legs:
            values["ft_%s_down" % leg] = down.get(leg, 0.0)
            values["fv_%s_x" % leg] = vx
            values["fv_%s_y" % leg] = vy
            values["fv_%s_z" % leg] = vz
        return evaluate(expr, **values)

    alldown = {leg: 1.0 for leg in legs}
    assert f(alldown) == pytest.approx(0.0)                          # 止まっていれば 0
    assert f(alldown, vx=100.0) == pytest.approx(1.0)                 # 全足が 100 mm/s で滑れば 1
    assert f(alldown, vx=60.0, vy=80.0) == pytest.approx(1.0)         # 床に沿った速さ（x と y）
    assert f(alldown, vx=200.0) == pytest.approx(4.0)                 # 2乗
    assert f(alldown, vz=300.0) == pytest.approx(0.0)                 # 上下の速さは数えない
    assert f({}, vx=100.0) == pytest.approx(0.0)                      # 浮いた足は数えない
    assert f({legs[0]: 1.0}, vx=100.0) == pytest.approx(1.0 / n)      # 脚の平均


def test_slip_cost_scales_the_weight():
    _, log = run_design(**TRI_R2_WAVE, slip_cost=2.5)
    (term,) = _reward(log, "foot_slip")
    assert term.kwargs["weight"] == -2.5


@pytest.mark.parametrize("bad", [-0.5, -1e-9, 5.5])
def test_slip_cost_rejects_negative(bad):
    with pytest.raises(ValueError):
        run_design(**TRI_R2_WAVE, slip_cost=bad)


# --- MuJoCo で、すり足なら正、立ったままなら約 0 ---

def _saw(t, period, rise):
    ph = (t % period) / period
    r = rise / period
    return (2 * ph / r - 1) if ph < r else (1 - 2 * (ph - r) / (1 - r))


def _replay_slip(bundle: Path, expr: str, legs, motion: str, seconds=4.0, skip=1.0):
    """バンドルの cut.xml を MuJoCo で再生し、foot_slip の式の値（1ステップあたりの平均）を返す。
    fv は足のボディの subtree_linvel（足は葉のボディなので足の重心の速度）、接地は足の形状と床の接触。"""
    mujoco = pytest.importorskip("mujoco")
    task = json.loads((bundle / "walk-task.json").read_text())
    m = mujoco.MjModel.from_xml_path(str(bundle / "cut.xml"))
    d = mujoco.MjData(m)
    acts = task["actions"]
    ep = task["episode"]
    dt, nsub = ep["control_interval_s"], ep["solver_steps_per_action"]
    foot = {leg: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "c_foot_" + leg) for leg in legs}
    assert min(foot.values()) >= 0
    foot_geoms = {leg: {g for g in range(m.ngeom) if m.geom_bodyid[g] == b} for leg, b in foot.items()}
    floor = {g for g in range(m.ngeom) if m.geom_bodyid[g] == 0}
    kf = ep.get("reset_keyframe")
    k = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, kf) if kf else 0
    mujoco.mj_resetDataKeyframe(m, d, max(k, 0))
    d.qpos[2] += 0.005
    mujoco.mj_forward(m, d)
    d.ctrl[:] = 0.0
    period, rise, amp = 0.5, 0.375, 10.0
    vals = []
    for step in range(int(seconds / dt)):
        t = step * dt
        for i, a in enumerate(acts):
            kind, leg = a["joint"].split("_", 1)
            q = 0.0
            if motion == "shuffle" and kind == "hip":
                # 股を脚ごとに 1/脚数 周期ずらして掃く。膝は 0 のまま（足を床に付けたまま滑らせる）
                q = amp * _saw(t + legs.index(leg) * period / len(legs), period, rise)
            d.ctrl[i] = min(max(q, a["low"]), a["high"]) * a["scale"]
        for _ in range(nsub):
            mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        mujoco.mj_subtreeVel(m, d)
        if t < skip:
            continue
        values = {}
        for leg in legs:
            values["ft_%s_down" % leg] = 0.0
            for ax, v in zip("xyz", d.subtree_linvel[foot[leg]] * 1000.0):
                values["fv_%s_%s" % (leg, ax)] = float(v)
        for c in d.contact[:d.ncon]:
            gs = {c.geom1, c.geom2}
            for leg in legs:
                if gs & foot_geoms[leg] and gs & floor:
                    values["ft_%s_down" % leg] = 1.0
        vals.append(evaluate(expr, **values))
    return sum(vals) / len(vals)


@pytest.mark.parametrize("bundle, overrides, legs", [
    ("tri-r2-dyn-dr", TRI_R2_WAVE, TRI_LEGS), ("mini-r2-p7-eff-dr", MINI_R2_P7_EFF, QUAD_LEGS)])
def test_slip_term_in_mujoco_positive_for_shuffle_zero_for_standing(bundle, overrides, legs):
    path = ROOT / "artifacts" / bundle / "cut"
    if not (path / "cut.xml").exists():
        pytest.skip("バンドルが無い: %s" % path)
    _, log = run_design(**overrides, slip_cost=1.0)
    (term,) = _reward(log, "foot_slip")
    stand = _replay_slip(path, term.args[0], legs, "stand")
    shuffle = _replay_slip(path, term.args[0], legs, "shuffle")
    assert stand < 0.01
    assert shuffle > 0.05
    assert shuffle > 20 * max(stand, 1e-6)
