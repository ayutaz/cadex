"""設計スクリプト robot/design/quadruped8.py の脚の組立の版（leg_rev）。

偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。形状は作らないので、
確かめるのは板の位置と、ホーンの凹みを切る道具の位置だけ。実際に組めるかは STL から判定する
（tests/test_assembly_guide.py の mini-r2）。

- leg_rev=1（既定）: mini までの形。base・leg120・mini などの生成物と同じ寸法になる。
- leg_rev=2（mini-r2）: ホーンの凹みをサーボ側の面に開け、上腿・下腿の内面をホーンの腕の下に置く。
"""

from __future__ import annotations

import pytest

from test_design_battery import SERVO_SPEC, run_design

HUB_H = 2.5      # 偽物のホーンのハブの高さ
ARM_T = 1.5      # 偽物のホーンの腕の厚み
W2 = 64.0 / 2.0  # 既定の body_width の半分
G, T, C = 1.0, 6.5, 0.5
HORN_TOP = SERVO_SPEC["spline_height_mm"] + HUB_H  # サーボのケース上面からホーンの外面まで


def _beam(result, leg):
    # thigh = cut(transform(fuse([beam, hub, kcap, housing])), tools)
    return result["thigh_" + leg].args[0].args[0].args[0][0]


def _shin_hub(result, leg):
    return result["shin_" + leg].args[0].args[0].args[0][0]


def _inner_y(thing):
    """左の脚（+Y）の板の内面の y。"""
    return thing.kwargs["origin"][1]


def _boxes_between(tools, lo, hi):
    """y の範囲が [lo, hi] を覆う道具（左の脚、+Y）。"""
    out = []
    for t in tools:
        if t.kind not in ("part.box", "part.cylinder"):
            continue
        y0 = t.kwargs["origin"][1]
        dy = t.args[1] if t.kind == "part.box" else t.args[1]
        if y0 <= lo + 1e-9 and y0 + dy >= hi - 1e-9:
            out.append((y0, y0 + dy, t))
    return out


def _capsule_y(log, label):
    col = [t for t in log if t.kind == "assembly.collision" and t.kwargs.get("label") == label]
    assert len(col) == 1
    return col[0].kwargs["offset"]["position"][1]


def test_既定の脚は_mini_までと同じ寸法():
    result, log = run_design()
    assert _inner_y(_beam(result, "lf")) == pytest.approx(W2 + G)
    y_kt = W2 + G + T
    assert _inner_y(_shin_hub(result, "lf")) == pytest.approx(y_kt + G)
    assert result["foot_lf"].kwargs["center"][1] == pytest.approx(y_kt + G + T / 2.0)
    assert _capsule_y(log, "thigh_cap_lf") == pytest.approx(W2 + G + T / 2.0)
    # 上腿・下腿を切る道具は v1 の4つ（収納部・ホーン・スプラインの逃げ・ドライバーの穴）のまま
    assert len(result["thigh_lf"].args[1]) == 4
    assert len(result["shin_lf"].args[1]) == 4


def test_既定と_leg_rev_1_は同じ宣言():
    _, a = run_design()
    _, b = run_design(leg_rev=1.0)
    assert [(t.kind, t.label, repr(t.args), repr(t.kwargs)) for t in a] == \
           [(t.kind, t.label, repr(t.args), repr(t.kwargs)) for t in b]


@pytest.mark.parametrize("leg", ["lf", "lh"])
def test_leg_rev_2_は上腿と下腿の内面をホーンの腕の下に置く(leg):
    result, log = run_design(leg_rev=2.0)
    seat = HORN_TOP - (ARM_T + C)  # 凹みの深さ = 腕の厚み + すきま
    y_th_in = W2 + seat
    assert _inner_y(_beam(result, leg)) == pytest.approx(y_th_in)
    y_kt = y_th_in + T
    assert _inner_y(_shin_hub(result, leg)) == pytest.approx(y_kt + seat)
    assert result["foot_" + leg].kwargs["center"][1] == pytest.approx(y_kt + seat + T / 2.0)
    # 衝突形状も板と一緒に動く
    assert _capsule_y(log, "thigh_cap_" + leg) == pytest.approx(y_th_in + T / 2.0)


def test_leg_rev_2_のホーンの凹みはサーボ側の面に開き底はホーンの外面():
    result, _ = run_design(leg_rev=2.0)
    seat = HORN_TOP - (ARM_T + C)
    # 上腿: 股ホーンの外面 W2 + HORN_TOP が底。内面 W2 + seat より内側まで抜ける
    pockets = [p for p in _boxes_between(result["thigh_lf"].args[1], W2 + seat, W2 + HORN_TOP)
               if p[1] == pytest.approx(W2 + HORN_TOP)]
    assert len(pockets) >= 3  # 十字の腕2本 + ハブ
    for y0, _, _ in pockets:
        assert y0 < W2 + seat
    # 下腿: 膝ホーンの外面 y_kt + HORN_TOP が底
    y_kt = W2 + seat + T
    pockets = [p for p in _boxes_between(result["shin_lf"].args[1], y_kt + seat, y_kt + HORN_TOP)
               if p[1] == pytest.approx(y_kt + HORN_TOP)]
    assert len(pockets) >= 3
    for y0, _, _ in pockets:
        assert y0 < y_kt + seat


def test_leg_rev_2_でも報酬とタスクは変わらない():
    def tasks(log):
        return [(t.kind, t.label, repr(t.args), repr(t.kwargs)) for t in log
                if t.kind in ("assembly.reward", "assembly.termination", "assembly.goal", "assembly.randomise")]
    _, a = run_design()
    _, b = run_design(leg_rev=2.0)
    assert tasks(a) == tasks(b)
