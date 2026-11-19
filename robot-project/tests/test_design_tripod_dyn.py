"""3本足の「動的な一歩」の直し（tri-r2-dyn）: hc の足先を重心の後ろへ（hc_foot_center・hc_foot_y）、
追従の速度を体全体の重心で読む（vel_source）、波の遊脚を短くする（wave_swing）。

3つとも既定値では、今の出力（四足の既定、mini-r2-p7-eff、tri-r2、tri-r2-wave の基準ログ）を変えない。
偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。
"""

from __future__ import annotations

import json
import math

import pytest

from test_design_battery import run_design
from test_design_tail import DATA, MINI_R2_P7_EFF, of_kind, rewards, signature
from test_design_tripod import evaluate

TRI_R2 = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=0.0)
TRI_R2_WAVE = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=1.0)
NEW_DEFAULTS = dict(hc_foot_center=0.0, vel_source=0.0, wave_swing=0.5)
LEGS = ("lf", "rf", "hc")


@pytest.mark.parametrize("name, overrides", [
    ("default", {}), ("mini-r2-p7-eff", MINI_R2_P7_EFF), ("tri-r2", TRI_R2), ("tri-r2-wave", TRI_R2_WAVE)])
def test_新しい_params_の既定値では出力が変わらない(name, overrides):
    baseline = json.loads((DATA / f"design_log_{name}.json").read_text())
    _, log = run_design(**overrides)
    assert signature(log) == baseline
    _, log0 = run_design(**overrides, **NEW_DEFAULTS)
    assert signature(log0) == baseline


TRI_R2_DYN = dict(TRI_R2_WAVE, hc_foot_center=1.0, vel_source=1.0, wave_swing=0.8)


def test_tri_r2_dyn_の出力は基準ログと同じ():
    # 基準ログ design_log_tri-r2-dyn.json は、ロール（leg_roll）を足す前の設計スクリプト（コミット 9c00eebdc の版）の出力
    baseline = json.loads((DATA / "design_log_tri-r2-dyn.json").read_text())
    _, log = run_design(**TRI_R2_DYN)
    assert signature(log) == baseline


# --- 1. hc の足先を重心の後ろへ ---

def _foot(log, leg):
    (s,) = [t for t in of_kind(log, "part.sphere") if t.label == "foot_" + leg]
    return s.kwargs["center"]


def _servo_origin(log, label):
    (s,) = [t for t in of_kind(log, "lib.servo") if t.label == label]
    return list(s.origin)


def _collision(log, label):
    (c,) = [t for t in of_kind(log, "assembly.collision") if t.label == label]
    return c.kwargs["offset"]["position"]


@pytest.mark.parametrize("foot_y", [-1.0, -3.5])
def test_hc_foot_center_1_は_hc_の脚をまるごと左右にずらし_足先を_hc_foot_y_に置く(foot_y):
    _, log0 = run_design(**TRI_R2_WAVE)
    _, log1 = run_design(**TRI_R2_WAVE, hc_foot_center=1.0, hc_foot_y=foot_y)
    f0, f1 = _foot(log0, "hc"), _foot(log1, "hc")
    assert f0[1] > 10.0                                  # 今は +11 mm
    assert f1[1] == pytest.approx(foot_y)
    assert f1[0] == pytest.approx(f0[0]) and f1[2] == pytest.approx(f0[2])
    dy = f1[1] - f0[1]
    # 股・膝のサーボ、上腿・下腿の衝突形状、台の衝突の箱が同じだけずれる
    for label in ("hip_servo_hc", "knee_servo_hc"):
        o0, o1 = _servo_origin(log0, label), _servo_origin(log1, label)
        assert o1[1] - o0[1] == pytest.approx(dy)
        assert o1[0] == pytest.approx(o0[0]) and o1[2] == pytest.approx(o0[2])
    for label in ("thigh_cap_hc", "hc_mount_box"):
        assert _collision(log1, label)[1] - _collision(log0, label)[1] == pytest.approx(dy)
    # 前の2本は変わらない
    for leg in ("lf", "rf"):
        assert _foot(log1, leg) == _foot(log0, leg)
        assert _servo_origin(log1, "hip_servo_" + leg) == _servo_origin(log0, "hip_servo_" + leg)


def _boxes_positive(log):
    bad = [t.kwargs.get("origin") for t in of_kind(log, "part.box") if min(t.args[:3]) <= 0.0]
    assert not bad, bad


@pytest.mark.parametrize("foot_y", [-10.0, -4.0, -1.0, 2.0, 11.0])
def test_hc_foot_center_1_でも箱の寸法はすべて正(foot_y):
    # 足先を中央へ寄せると、股サーボのリード線の逃げが胴体の側壁の外（y < -(W2 - w)）に出る。
    # 胴体の空洞への通り道（幅 = 逃げの上限 + (W2 - w)）が 0 以下なら作らない（実ビルドでは幅が負の箱で落ちた）
    _, log = run_design(**TRI_R2_WAVE, hc_foot_center=1.0, hc_foot_y=foot_y)
    _boxes_positive(log)


def test_hc_foot_center_1_の台は胴体に壁1枚ぶん以上重なる():
    # 台（股サーボのケースを包む箱）の +Y の面 = 股サーボのケース上面 HC_YB。胴体の側面 -W2 より壁 1枚ぶん内側まで
    # 届かないと、台が胴体から離れる（実ビルドで、足先 y = -13 mm のとき part.fuse が2つの立体になって落ちた）
    _, log = run_design(**TRI_R2_WAVE, hc_foot_center=1.0, hc_foot_y=-10.0)
    assert _collision(log, "hc_mount_box")[1] < 0.0
    with pytest.raises(ValueError):
        run_design(**TRI_R2_WAVE, hc_foot_center=1.0, hc_foot_y=-10.5)


def test_hc_foot_y_の既定は実測で差が最小の_m10():
    _, log = run_design(**TRI_R2_WAVE, hc_foot_center=1.0)
    assert _foot(log, "hc")[1] == pytest.approx(-10.0)


def test_hc_foot_center_1_は3本足だけ():
    with pytest.raises(ValueError):
        run_design(**MINI_R2_P7_EFF, hc_foot_center=1.0)
    with pytest.raises(ValueError):
        run_design(**TRI_R2_WAVE, hc_foot_center=0.5)


# --- 2. 追従の速度を体全体の重心で読む ---

def _vel(log):
    (o,) = [t for t in of_kind(log, "assembly.observation") if t.kwargs.get("name") == "vel"]
    return o


@pytest.mark.parametrize("source, kind", [(0.0, "component_linear_velocity"), (1.0, "centre_of_mass_velocity")])
def test_vel_source_で速度の観測の種類を切り替える(source, kind):
    result, log = run_design(**TRI_R2_WAVE, vel_source=source)
    o = _vel(log)
    assert o.args[0] is result["c_chassis"]               # 胴体（MJCF の根のボディ＝体全体の部分木）
    assert o.args[1] == kind
    assert o.kwargs["role"] == "privileged"
    r = rewards(log)
    assert "vel_x" in r["track_speed"] and "vel_y" in r["lateral"]   # 名前（vel_x・vel_y）と式は同じ


def test_vel_source_は0か1():
    with pytest.raises(ValueError):
        run_design(**TRI_R2_WAVE, vel_source=2.0)


# --- 3. 素早い一歩（波の遊脚を短くする） ---

def _contact(log):
    contact = rewards(log)["gait_contact"]

    def f(theta, down):
        return evaluate(contact, clock_sin=math.sin(theta), clock_cos=math.cos(theta),
                        **{"ft_%s_down" % k: down[i] for i, k in enumerate(LEGS)})
    return f


@pytest.mark.parametrize("swing", [0.5, 0.8])
def test_wave_swing_は遊脚の位相の幅を決め_立っているだけなら0(swing):
    _, log = run_design(**TRI_R2_WAVE, wave_swing=swing)
    f = _contact(log)
    allv = [f(math.radians(d / 10.0), (1.0, 1.0, 1.0)) for d in range(3600)]
    assert max(abs(v) for v in allv) < 1e-9
    # 脚 k だけを上げたときの報酬が最大の半分を超える位相は、cos(θ - k·120°) < -swing の範囲（周期の acos(swing)/π）。
    # 平均を引く形なので、小さな正の値はその外（周期の 0.35 ほど）まで続く
    expect = math.acos(swing) / math.pi
    for k in range(3):
        lifted = tuple(0.0 if i == k else 1.0 for i in range(3))
        vals = [f(math.radians(d / 10.0), lifted) for d in range(3600)]
        half = sum(v > 0.5 * max(vals) for v in vals) / 3600.0
        assert half == pytest.approx(expect, abs=0.03)
        assert sum(v > 0 for v in vals) / 3600.0 < 0.4
        assert f(math.radians(k * 120.0 + 180.0), lifted) > 0.5
        assert f(math.radians(k * 120.0), lifted) < 0.0
    # 足上げも、その遊脚の位相で払う
    lift = rewards(log)["swing_lift"]
    low = {"fc_%s_z" % k: 6.0 for k in LEGS}
    for k, leg in enumerate(LEGS):
        at = lambda d: evaluate(lift, clock_sin=math.sin(math.radians(d)), clock_cos=math.cos(math.radians(d)),
                                **{**low, "fc_%s_z" % leg: 16.0})
        assert at(k * 120.0 + 180.0) > 0.5
        assert at(k * 120.0) < 0.05
        # 足上げの重み（0.5 - 0.5·立脚の度合い）が 0.5 を超えるのは、ちょうど遊脚の幅
        w = [0.5 - 0.5 * math.tanh(5 * (math.cos(math.radians(d / 10.0)) + swing)) for d in range(3600)]
        assert sum(v > 0.5 for v in w) / 3600.0 == pytest.approx(expect, abs=0.002)


def test_wave_swing_08_の遊脚は周期_07_s_で約_0143_s():
    assert 0.7 * math.acos(0.8) / math.pi == pytest.approx(0.143, abs=0.001)
    _, log = run_design(**TRI_R2_WAVE, wave_swing=0.8)
    assert "tanh(5*(clock_cos + 0.8))" in rewards(log)["gait_contact"]


def test_wave_swing_は波でだけ変えられる():
    with pytest.raises(ValueError):
        run_design(**TRI_R2, wave_swing=0.8)


# --- tri-r2-dyn の値 ---

def test_tri_r2_dyn_は3つの直しを全部使う():
    result, log = run_design(**TRI_R2_WAVE, hc_foot_center=1.0, vel_source=1.0, wave_swing=0.8)
    assert _vel(log).args[1] == "centre_of_mass_velocity"
    assert "+ 0.8))" in rewards(log)["gait_contact"]
    assert _foot(log, "hc")[1] == pytest.approx(-10.0)
