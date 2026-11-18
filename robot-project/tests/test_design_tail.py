"""設計スクリプト robot/design/quadruped8.py のしっぽ（tail_dof、tail_length、tail_tip_mass_g）。

偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。形状は作らないので、
確かめるのは宣言（部品・関節・アクチュエータ・観測・報酬・ランダム化）だけ。形と干渉は Cadex の実ビルドで見る。

基準のログ tests/data/design_log_*.json は、しっぽを足す前の設計スクリプト（コミット 6ca525377）を
run_design で実行した宣言の列（種類・ラベル・引数の repr）。既定値ではこれと1つも違わない。
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from test_design_battery import run_design

DATA = Path(__file__).resolve().parent / "data"

# mini-r2-p7-eff の params のうち、既定値と違うもの（artifacts/mini-r2-p7-eff/design_report.json）
MINI_R2_P7_EFF = dict(body_length=96.0, body_width=62.0, hip_x=40.0, thigh_length=40.0, shin_length=44.0,
                      battery_mass_g=0.0, cmd_min=30.0, cmd_max=120.0, gait_period=0.7, w_trot=1.0,
                      w_swing=1.5, w_effort_rated=1.0, leg_rev=2.0)


def signature(log) -> list[list]:
    return [[t.kind, t.label, repr(t.args), repr(t.kwargs)] for t in log]


def of_kind(log, kind):
    return [t for t in log if t.kind == kind]


def actuators(log):
    return [t.kwargs["label"] for t in of_kind(log, "assembly.actuator")]


def joint(log, label):
    js = [t for t in of_kind(log, "assembly.joint") if t.label == label]
    assert len(js) == 1, label
    return js[0]


def rewards(log) -> dict:
    return {t.kwargs["label"]: t.args[0] for t in of_kind(log, "assembly.reward")}


def task(log):
    (t,) = of_kind(log, "assembly.task")
    return t


def connector_z_axis(conn) -> tuple[float, float, float]:
    """コネクタの offset の回転で、コネクタの +Z が向くワールドの向き（回転なしなら +Z）。"""
    off = conn.kwargs.get("offset", {})
    ang = math.radians(off.get("angle_degrees", 0.0))
    ax = off.get("axis", [0, 0, 1])
    n = math.sqrt(sum(a * a for a in ax))
    kx, ky, kz = (a / n for a in ax)
    # ロドリゲスの回転公式で (0, 0, 1) を回す
    vx, vy, vz = 0.0, 0.0, 1.0
    c, s = math.cos(ang), math.sin(ang)
    dot = kx * vx + ky * vy + kz * vz
    cx, cy, cz = ky * vz - kz * vy, kz * vx - kx * vz, kx * vy - ky * vx
    return tuple(round(v * c + cr * s + k * dot * (1 - c), 9) + 0.0
                 for v, cr, k in ((vx, cx, kx), (vy, cy, ky), (vz, cz, kz)))


@pytest.mark.parametrize("name, overrides", [("default", {}), ("mini-r2-p7-eff", MINI_R2_P7_EFF)])
def test_既定値では出力が変わらない(name, overrides):
    baseline = json.loads((DATA / f"design_log_{name}.json").read_text())
    _, log = run_design(**overrides)
    assert signature(log) == baseline
    _, log0 = run_design(**overrides, tail_dof=0.0)
    assert signature(log0) == baseline


def test_tail_dof_1_は左右の関節を1つ足す():
    result, log = run_design(tail_dof=1.0)
    acts = actuators(log)
    assert len(acts) == 9
    assert acts[-1] == "tail_yaw_drive"
    assert [a.label for a in task(log).kwargs["actions"]][-1] == "tail_yaw_drive"
    j = joint(log, "tail_yaw")
    assert j.args[0] == "revolute"
    for conn in j.args[1:3]:
        assert connector_z_axis(conn) == (0.0, 0.0, 1.0)
    # 付け根: 胴体（シャーシ）としっぽの1本目のリンクの間。可動域は ±45°（仮）
    assert j.args[1].args[0] is result["c_chassis"]
    assert j.args[2].args[0] is result["c_tail_link_1"]
    assert j.kwargs["angle_limits_degrees"] == [-45.0, 45.0]
    # 付け根の軸は胴体の後端より後ろ、左右の中央
    pos = j.args[1].kwargs["offset"]["position"]
    assert pos[0] < -110.0 / 2.0
    assert pos[1] == 0.0
    assert "tail_pitch" not in [t.label for t in of_kind(log, "assembly.joint")]
    # サーボは胴体に溶接、ホーンはリンクに溶接。サーボとホーンの接触は意図したもの
    weld = result["weld_tail_yaw_servo"]          # assembly.joint("fixed", 胴体側, サーボ側)
    assert weld.args[0] == "fixed"
    assert weld.args[1].args[0] is result["c_chassis"]
    assert weld.args[2].args[0] is result["c_tail_yaw_servo"]
    assert result["weld_tail_yaw_horn"].args[1].args[0] is result["c_tail_link_1"]
    asm = result["asm"]
    assert (result["c_tail_yaw_servo"], result["c_tail_yaw_horn"]) in asm.kwargs["contacts"]
    for key in ("c_tail_yaw_servo", "c_tail_yaw_horn", "c_tail_link_1", "c_tail_tip"):
        assert result[key] in asm.args[0]


def test_tail_dof_2_は左右と上下の2つ():
    result, log = run_design(tail_dof=2.0)
    acts = actuators(log)
    assert len(acts) == 10
    assert acts[-2:] == ["tail_yaw_drive", "tail_pitch_drive"]
    assert [a.label for a in task(log).kwargs["actions"]][-2:] == ["tail_yaw_drive", "tail_pitch_drive"]
    j = joint(log, "tail_pitch")
    for conn in j.args[1:3]:
        assert connector_z_axis(conn) == (0.0, 1.0, 0.0)
    assert j.args[1].args[0] is result["c_tail_link_1"]
    assert j.args[2].args[0] is result["c_tail_link_2"]
    assert j.kwargs["angle_limits_degrees"] == [-30.0, 45.0]
    # 上下の軸は左右の軸より後ろ
    yaw_x = joint(log, "tail_yaw").args[1].kwargs["offset"]["position"][0]
    assert j.args[1].kwargs["offset"]["position"][0] < yaw_x
    # 上下のサーボは1本目のリンクに、ホーンは2本目のリンクに溶接
    assert result["weld_tail_pitch_servo"].args[1].args[0] is result["c_tail_link_1"]
    assert result["weld_tail_pitch_horn"].args[1].args[0] is result["c_tail_link_2"]
    assert result["weld_tail_tip"].args[1].args[0] is result["c_tail_link_2"]
    assert (result["c_tail_pitch_servo"], result["c_tail_pitch_horn"]) in result["asm"].kwargs["contacts"]


@pytest.mark.parametrize("dof", [1.0, 2.0])
def test_しっぽの観測とランダム化(dof):
    _, log = run_design(tail_dof=dof)
    obs = {t.kwargs["name"]: t for t in of_kind(log, "assembly.observation")}
    keys = ["tail_yaw"] + (["tail_pitch"] if dof > 1.5 else [])
    for k in keys:
        assert obs["c_" + k].args[1] == "command"
        assert obs["c_" + k].kwargs.get("role") is None                # 方策が読む（前回の指令）
        assert obs["tau_" + k].kwargs["role"] == "privileged"
    assert ("c_tail_pitch" in obs) == (dof > 1.5)
    rand = [(t.args[0].label, t.args[1]) for t in of_kind(log, "assembly.randomise")]
    for k in keys:
        assert (k, "damping") in rand
        assert (k, "friction_loss") in rand
    assert ("tail_tip", "mass") in rand


@pytest.mark.parametrize("dof", [1.0, 2.0])
def test_しっぽの重さと長さは_params_で変わる(dof):
    def tip(**kw):
        result, log = run_design(tail_dof=dof, **kw)
        (body,) = [t for t in of_kind(log, "assembly.body") if t.args[0] is result["c_tail_tip"]]
        return result["tail_tip"], body

    shape60, body60 = tip()
    shape90, _ = tip(tail_length=90.0)
    assert shape60.kind == "part.sphere"
    assert shape90.kwargs["center"][0] == pytest.approx(shape60.kwargs["center"][0] - 30.0)
    assert shape90.kwargs["center"][1:] == pytest.approx(shape60.kwargs["center"][1:])
    # 先端のおもりの質量（g）= 密度 × 球の体積
    for mass in (5.0, 12.0):
        shape, body = tip(tail_tip_mass_g=mass)
        vol_m3 = 4.0 / 3.0 * math.pi * shape.args[0] ** 3 * 1e-9
        assert body.kwargs["density_kg_m3"] * vol_m3 * 1000.0 == pytest.approx(mass)
    # 付け根の位置は長さで変わらない
    _, log60 = run_design(tail_dof=dof)
    _, log90 = run_design(tail_dof=dof, tail_length=90.0)
    assert joint(log60, "tail_yaw").args[1].kwargs == joint(log90, "tail_yaw").args[1].kwargs


@pytest.mark.parametrize("dof, n", [(0.0, 8), (1.0, 9), (2.0, 10)])
def test_報酬の_8_で割る平均はアクチュエータの数で割る(dof, n):
    _, log = run_design(tail_dof=dof)
    r = rewards(log)
    for label in ("action_rate", "effort", "effort_rated"):
        assert r[label].endswith(") / %d" % n), r[label]
    assert r["action_rate"].count("_rate / 600") == n
    if dof > 1.5:
        assert "c_tail_pitch_rate" in r["action_rate"]
        assert "tau_tail_pitch" in r["effort"]


def test_しっぽは脚と同じ式のアクチュエータ():
    _, log = run_design(tail_dof=2.0)
    acts = {t.kwargs["label"]: t.kwargs for t in of_kind(log, "assembly.actuator")}
    leg = dict(acts["hip_drive_lf"])
    for k in ("tail_yaw_drive", "tail_pitch_drive"):
        a = dict(acts[k])
        for key in ("kind", "control_deg", "stiffness_nmm_per_deg", "damping_nmms_per_deg",
                    "torque_limit_nmm", "rated_torque_nmm"):
            assert a[key] == leg[key], key
    assert acts["tail_yaw_drive"]["command_limits_degrees"] == [-45.0, 45.0]
    assert acts["tail_pitch_drive"]["command_limits_degrees"] == [-30.0, 45.0]
    dyn = {t.kwargs["label"]: {k: v for k, v in t.kwargs.items() if k != "label"}
           for t in of_kind(log, "assembly.joint_dynamics")}
    assert dyn["tail_yaw_dyn"] == dyn["hip_dyn_lf"]
    assert dyn["tail_pitch_dyn"] == dyn["hip_dyn_lf"]


def test_しっぽの先端のおもりが0なら作らない():
    result, log = run_design(tail_dof=1.0, tail_tip_mass_g=0.0)
    assert "c_tail_tip" not in result
    assert all(t.label != "tail_tip" for t in log)
    assert "tail_tip" not in [t.args[0].label for t in of_kind(log, "assembly.randomise")]


# --- 実ビルド（Cadex の clearance）で見つかった形の不具合（Task 4） ---
# サーボの寸法（tests/test_design_battery.py の SERVO_SPEC）から、スクリプトと同じ式でサーボの外形を出す。
# ロール 180 度・軸の向き ±Y または +Z のサーボのワールドの x の範囲は、ローカルの x の範囲を反転して軸の x に足す。
_S = {"CASE_L": 22.8, "FRONT": 8.75, "TAB_HALF": 32.5 / 2.0}
_CASE_CX = _S["FRONT"] - _S["CASE_L"] / 2.0


def _servo_x_roll180(axis_x: float) -> dict:
    """ロール 180 度のサーボの、ケースとタブ（耳）の x の範囲。"""
    return {"case": (axis_x - _S["FRONT"], axis_x + (_S["CASE_L"] - _S["FRONT"])),
            "tabs": (axis_x - (_S["TAB_HALF"] + _CASE_CX), axis_x + (_S["TAB_HALF"] - _CASE_CX))}


def _servo(log, label):
    (sv,) = [t for t in of_kind(log, "lib.servo") if t.label == label]
    return sv


def _boxes_of(shape) -> list:
    """part.fuse / part.cut の木から、材料になる箱（part.box）を集める（cut は最初の引数だけたどる）。"""
    if shape.kind == "part.box":
        return [shape]
    if shape.kind == "part.fuse":
        return [b for s in shape.args[0] for b in _boxes_of(s)]
    if shape.kind == "part.cut":
        return _boxes_of(shape.args[0])
    return []


def _extent(box, axis: int) -> tuple[float, float]:
    lo = box.kwargs["origin"][axis]
    return lo, lo + box.args[axis]


@pytest.mark.parametrize("overrides", [dict(tail_dof=1.0), dict(tail_dof=2.0), dict(rear_legs=1.0)])
def test_胴体を上下に分ける箱は後端の台まで覆う(overrides):
    # 前は分ける箱が胴体の箱（長さ BL + 2）の大きさのままで、しっぽ・3本足の台のはみ出した部分が
    # chassis と cover の両方に残った（mini-r2-tail1 の実ビルドで重なり 4738 mm³）
    _, log = run_design(**MINI_R2_P7_EFF, **overrides)
    (chassis,) = [t for t in of_kind(log, "part.cut") if t.label == "chassis"]
    (cover,) = [t for t in of_kind(log, "part.cut") if t.label == "cover"]
    hull_boxes = _boxes_of(chassis.args[0])
    assert len(hull_boxes) >= 2  # 胴体の箱と台
    for half in (chassis.args[1], cover.args[1]):
        for axis in (0, 1):
            lo, hi = _extent(half, axis)
            assert lo < min(_extent(b, axis)[0] for b in hull_boxes), (axis, half.kwargs)
            assert hi > max(_extent(b, axis)[1] for b in hull_boxes), (axis, half.kwargs)


@pytest.mark.parametrize("dof", [1.0, 2.0])
def test_しっぽの左右のサーボは後ろの股サーボから離れる(dof):
    # 前はしっぽのサーボのタブの先が胴体の中へ入り、後ろの股サーボ（lh・rh）と 42 mm³ ずつ重なった
    c, w = 0.5, 2.0
    _, log = run_design(**MINI_R2_P7_EFF, tail_dof=dof, clearance=c, wall=w)
    tail = _servo_x_roll180(_servo(log, "tail_yaw_servo").origin[0])
    for leg in ("lh", "rh"):
        hip = _servo_x_roll180(_servo(log, "hip_servo_" + leg).origin[0])
        # しっぽのサーボの前の端（タブの先）から、股サーボのケースの後ろの面まで、両方のすきまと壁1枚
        assert tail["tabs"][1] + c + w + c <= hip["case"][0] + 1e-9, (tail, hip)


def test_しっぽの上下のサーボは左右の軸と一緒に動く():
    # 左右の軸を後ろへずらしても、上下のサーボ（1本目のリンクの上）は左右の軸からの距離を保つ
    _, log = run_design(**MINI_R2_P7_EFF, tail_dof=2.0, joint_gap=1.0, clearance=0.5, wall=2.0)
    ty = _servo(log, "tail_yaw_servo").origin[0]
    tp = _servo(log, "tail_pitch_servo").origin[0]
    h_lead = (_S["CASE_L"] - _S["FRONT"]) + 0.5 + 6.0 + 2.0
    assert tp - ty == pytest.approx((_S["CASE_L"] - _S["FRONT"]) - 1.0 - h_lead)


def _body_tools(log) -> list:
    (chassis,) = [t for t in of_kind(log, "part.cut") if t.label == "chassis"]
    body_cut = chassis.args[0]
    return [t for t in body_cut.args[1] if t.kind == "part.box"]


def _lead_channel(log, x_from, x_to):
    """胴体から切る箱のうち、x_from から x_to まで（両端を含む）を通すもの。"""
    return [b for b in _body_tools(log) if _extent(b, 0)[0] <= x_from and _extent(b, 0)[1] >= x_to]


@pytest.mark.parametrize("dof", [1.0, 2.0])
def test_しっぽのサーボのリード線は股サーボの収納部へ抜ける(dof):
    # 軸を後ろへずらしたので、リード線の逃げ（ケースの前の面から 6 mm）だけでは胴体の中の収納部に届かない
    c = 0.5
    _, log = run_design(**MINI_R2_P7_EFF, tail_dof=dof, clearance=c, wall=2.0)
    tail = _servo_x_roll180(_servo(log, "tail_yaw_servo").origin[0])
    hip = _servo_x_roll180(_servo(log, "hip_servo_lh").origin[0])
    lead_end = tail["case"][1] + c + 6.0
    assert lead_end < hip["case"][0] - c  # 逃げだけでは届かない
    (ch,) = _lead_channel(log, lead_end, hip["case"][0] - c)
    # 通り道は左右の中央（しっぽのサーボの幅）で、後ろの股サーボの収納部（|y| >= 胴体の半幅 - ケースの高さ - すきま）に重なる
    y0, y1 = _extent(ch, 1)
    assert y0 < -(31.0 - 28.4 - c) and y1 > 31.0 - 28.4 - c
