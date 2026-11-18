"""設計スクリプト robot/design/quadruped8.py の3本足（rear_legs=1）と、その歩き方の報酬（gait_mode）。

偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。形状は作らないので、
確かめるのは宣言だけ。3本足の後ろの脚 hc は、前脚と同じ2関節の脚を胴体の後端の中央に1本置いたもの。
"""

from __future__ import annotations

import json
import math
import re

import pytest

from test_design_battery import run_design
from test_design_tail import DATA, MINI_R2_P7_EFF, actuators, joint, of_kind, rewards, signature, task

BL, R, G, T = 110.0, 12.5, 1.0, 6.5   # 既定の body_length・hub_radius・joint_gap・link_thickness


def evaluate(expr: str, **values) -> float:
    """報酬の式（Cadex の式。^ は累乗）を Python で計算する。"""
    py = expr.replace("^", "**")
    names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", py)) - {"tanh", "exp", "cos", "sin"}
    missing = names - set(values)
    assert not missing, missing
    return eval(py, {"__builtins__": {}, "tanh": math.tanh, "exp": math.exp, "cos": math.cos,
                     "sin": math.sin}, values)


@pytest.mark.parametrize("name, overrides", [("default", {}), ("mini-r2-p7-eff", MINI_R2_P7_EFF)])
def test_既定の_rear_legs_と_gait_mode_では出力が変わらない(name, overrides):
    baseline = json.loads((DATA / f"design_log_{name}.json").read_text())
    _, log = run_design(**overrides, rear_legs=2.0, gait_mode=0.0)
    assert signature(log) == baseline


def test_rear_legs_1_は後ろ中央の1本():
    result, log = run_design(rear_legs=1.0)
    acts = actuators(log)
    assert acts == ["hip_drive_lf", "knee_drive_lf", "hip_drive_rf", "knee_drive_rf", "hip_drive_hc", "knee_drive_hc"]
    assert [a.label for a in task(log).kwargs["actions"]] == acts
    assert not [t for t in log if t.label and (t.label.endswith("_lh") or t.label.endswith("_rh"))]
    hip = joint(log, "hip_hc")
    pos = hip.args[1].kwargs["offset"]["position"]
    assert pos[0] == pytest.approx(-(BL / 2.0 + R + G))
    assert abs(pos[1]) <= 0.5
    # 上腿の板の中央面が y = 0（衝突形状のカプセルの中心）
    (cap,) = [t for t in of_kind(log, "assembly.collision") if t.label == "thigh_cap_hc"]
    assert abs(cap.kwargs["offset"]["position"][1]) <= 0.5
    # 股のサーボは胴体（シャーシ）に溶接。可動域は後脚のもの
    assert result["weld_hip_servo_hc"].args[1].args[0] is result["c_chassis"]
    assert hip.kwargs["angle_limits_degrees"] == [-35.0, 45.0]
    assert joint(log, "knee_hc").kwargs["angle_limits_degrees"] == [-60.0, 45.0]
    # 股のサーボの軸は +Y、ケース上面は上腿の内面（-Y 側）の joint_gap 手前
    servos = {t.label: t for t in of_kind(log, "lib.servo")}
    assert "hip_servo_hc" in servos and "knee_servo_hc" in servos
    # 観測とランダム化も3脚ぶん
    obs = {t.kwargs["name"] for t in of_kind(log, "assembly.observation")}
    for k in ("q_hip_hc", "q_knee_hc", "ft_hc", "fc_hc", "c_hip_hc", "tau_knee_hc"):
        assert k in obs
    assert not {"q_hip_lh", "ft_rh"} & obs
    rand = [(t.args[0].label, t.args[1]) for t in of_kind(log, "assembly.randomise")]
    assert ("hip_hc", "damping") in rand and ("knee_hc", "friction_loss") in rand
    # 報酬の平均は6で割る
    assert rewards(log)["effort"].endswith(") / 6")


def test_rear_legs_1_の股のサーボは胴体の後端に足した台に入る():
    _, log = run_design(rear_legs=1.0)
    # 胴体の箱と台の箱をつないだもの（part.fuse）から、股のサーボの収納部を切る
    fuses = [t for t in of_kind(log, "part.fuse") if t.args and isinstance(t.args[0], list)
             and t.args[0] and t.args[0][0].kind == "part.box" and t.args[0][0].args[0] == BL]
    assert len(fuses) == 1
    mount = fuses[0].args[0][1]
    x0 = mount.kwargs["origin"][0]
    assert x0 < -(BL / 2.0 + R + G)                 # 股軸より後ろまで
    assert x0 + mount.args[0] > -BL / 2.0           # 胴体の後端に重なる


def test_しっぽと3本足は一緒に使えない():
    with pytest.raises(ValueError):
        run_design(tail_dof=1.0, rear_legs=1.0)
    with pytest.raises(ValueError):
        run_design(tail_dof=2.0, rear_legs=1.0)


def test_gait_mode_1_は4脚では使えない():
    with pytest.raises(ValueError):
        run_design(gait_mode=1.0)


def test_gait_mode_0_は接地のリズムの報酬が無い():
    _, log = run_design(rear_legs=1.0, gait_mode=0.0)
    r = rewards(log)
    assert "trot_contact" not in r
    assert "gait_contact" not in r
    lift = r["swing_lift"]
    for leg in ("lf", "rf", "hc"):
        assert "fc_%s_z" % leg in lift
    assert "clock" not in lift                      # 位相で重みを付けない
    # 足上げの項は、ほかの2本がどちらも接地しているときだけ払う（3本を同時に浮かせる＝跳ねると 0）
    legs = ("lf", "rf", "hc")
    up = {"fc_%s_z" % k: 16.0 for k in legs}
    assert evaluate(lift, **up, **{"ft_%s_down" % k: 0.0 for k in legs}) == pytest.approx(0.0, abs=1e-12)
    # 1本だけ浮かせると、その足の項だけが正（ほかの足が少し上がっていても、その項は 0）
    one = []
    for leg in legs:
        z = {"fc_%s_z" % k: (16.0 if k == leg else 7.0) for k in legs}
        down = {"ft_%s_down" % k: (0.0 if k == leg else 1.0) for k in legs}
        one.append(evaluate(lift, **z, **down))
        assert one[-1] == pytest.approx(math.tanh((16.0 - 6.0) / 10.0))
    assert one == pytest.approx([one[0]] * 3)       # どの足でも同じ
    # 2本を同時に浮かせても払わない
    z = {"fc_%s_z" % k: 16.0 for k in legs}
    assert evaluate(lift, **z, ft_lf_down=0.0, ft_rf_down=0.0, ft_hc_down=1.0) == pytest.approx(0.0, abs=1e-12)
    # 位相クロックは方策の入力として残す（4脚と同じ入力の並び）
    goals = [g.args[0] for g in task(log).kwargs["goals"]]
    assert goals == ["cmd", "clock"]


def _argmax_deg(f) -> float:
    best = max(range(360), key=lambda d: f(math.radians(d)))
    return float(best)


def test_gait_mode_1_は1_3周期ずつずらす():
    _, log = run_design(rear_legs=1.0, gait_mode=1.0)
    r = rewards(log)
    assert "trot_contact" not in r
    contact = r["gait_contact"]
    assert "clock_sin" in contact and "clock_cos" in contact
    legs = ("lf", "rf", "hc")

    def contact_of(theta, down):
        return evaluate(contact, clock_sin=math.sin(theta), clock_cos=math.cos(theta),
                        **{"ft_%s_down" % k: down[i] for i, k in enumerate(legs)})

    # 全脚が接地したままなら、どの時刻も 0（1周期の平均も 0）。四足のトロットの式と同じく、立っているだけでは得をしない
    allv = [contact_of(math.radians(d / 10.0), (1.0, 1.0, 1.0)) for d in range(3600)]
    assert sum(allv) / len(allv) == pytest.approx(0.0, abs=1e-6)
    assert max(abs(v) for v in allv) < 1e-9
    for k, leg in enumerate(legs):
        lifted = tuple(0.0 if i == k else 1.0 for i in range(3))
        def lift_one(theta, lifted=lifted):
            return contact_of(theta, lifted)
        # 脚 k だけを上げたときの報酬が最も大きいのは、脚 k の遊脚の中心 k·120° + 180°（正）
        best = _argmax_deg(lift_one)
        assert best == pytest.approx((k * 120.0 + 180.0) % 360.0, abs=1.0)
        assert lift_one(math.radians(best)) > 0.5
        # 立脚の中心 k·120° で上げると罰（負）
        assert lift_one(math.radians(k * 120.0)) < 0.0
        # 正になる位相は、遊脚の 1/3 周期のまわりだけ
        frac = sum(lift_one(math.radians(d)) > 0 for d in range(360)) / 360.0
        assert 0.3 < frac < 0.4
    # 足上げは遊脚の位相で払う
    lift = r["swing_lift"]
    low = {"fc_%s_z" % k: 6.0 for k in legs}
    for k, leg in enumerate(legs):
        def lift_of(theta, leg=leg):
            return evaluate(lift, clock_sin=math.sin(theta), clock_cos=math.cos(theta),
                            **{**low, "fc_%s_z" % leg: 16.0})
        assert _argmax_deg(lift_of) == pytest.approx((k * 120.0 + 180.0) % 360.0, abs=1.0)


def test_3本足の成功条件の足は3つ():
    result, log = run_design(rear_legs=1.0)
    (spec,) = of_kind(log, "assembly.success")
    feet = spec.kwargs["feet"]
    assert [f.label for f in feet] == ["foot_lf", "foot_rf", "foot_hc"]
    assert feet[2] is result["c_foot_hc"]


def test_3本足は_mini_r2_の値でも作れる():
    result, log = run_design(**MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=1.0)
    assert len(actuators(log)) == 6
    hip = joint(log, "hip_hc").args[1].kwargs["offset"]["position"]
    assert hip[0] == pytest.approx(-(96.0 / 2.0 + R + G))
    (cap,) = [t for t in of_kind(log, "assembly.collision") if t.label == "thigh_cap_hc"]
    assert abs(cap.kwargs["offset"]["position"][1]) <= 0.5
