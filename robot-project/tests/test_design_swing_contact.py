"""遊脚の位相で接地していることへの脚ごとの罰（swing_contact_cost）: 3本足の波（gait_mode=1）で、
後ろの足 hc を引きずる（遊脚の位相でも床に付けたまま）動きに、脚ごとに罰を入れる。

tri3-r9 は 12 回の学習のどれでも hc を引きずった（歩数 0）。足上げの報酬 swing_lift は3脚の和なので前の2本で
稼げ、遊脚の位相の接地への罰 gait_contact（重み w_trot）は速さの点に負けていた。

swing_contact_cost=0（既定）では項を足さず、今の出力（基準ログ5つ、tri3-r9）を1バイトも変えない。
0 より大きいと、報酬の項 swing_contact（重み -swing_contact_cost）を1つ足す。式は
Σ_k (0.5 - 0.5·st_k)·ft_k_down で、st_k は gait_contact と同じ tanh(5*(WAVE_PHASE[k] + wave_swing))。脚で割らない。
偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。
"""

from __future__ import annotations

import json
import math

import pytest

from test_design_battery import run_design
from test_design_tail import DATA, MINI_R2_P7_EFF, of_kind, signature
from test_design_tripod import evaluate

TRI_R2 = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=0.0)
TRI_R2_WAVE = dict(MINI_R2_P7_EFF, rear_legs=1.0, gait_mode=1.0)
TRI_R2_DYN = dict(TRI_R2_WAVE, hc_foot_center=1.0, vel_source=1.0, wave_swing=0.8)
TRI3_R9 = dict(TRI_R2_WAVE, hc_foot_center=1.0, vel_source=1.0, leg_roll=2.0)
TRI_LEGS = ("lf", "rf", "hc")


def _reward(log, label):
    return [t for t in of_kind(log, "assembly.reward") if t.kwargs.get("label") == label]


def _rewards(log):
    return of_kind(log, "assembly.reward")


# --- 0（既定）では今と同じ ---

@pytest.mark.parametrize("name, overrides", [
    ("default", {}), ("mini-r2-p7-eff", MINI_R2_P7_EFF), ("tri-r2", TRI_R2), ("tri-r2-wave", TRI_R2_WAVE),
    ("tri-r2-dyn", TRI_R2_DYN)])
def test_swing_contact_cost_zero_unchanged(name, overrides):
    baseline = json.loads((DATA / f"design_log_{name}.json").read_text())
    _, log0 = run_design(**overrides, swing_contact_cost=0.0)
    assert signature(log0) == baseline
    assert not _reward(log0, "swing_contact")


def test_swing_contact_cost_zero_unchanged_tri3_r9():
    _, log = run_design(**TRI3_R9)
    _, log0 = run_design(**TRI3_R9, swing_contact_cost=0.0)
    assert signature(log0) == signature(log)
    assert not _reward(log0, "swing_contact")


# --- 0 より大きいと項が1つ増える ---

def _clock(theta):
    return dict(clock_cos=math.cos(theta), clock_sin=math.sin(theta))


def _reference(theta, down, wave_swing):
    """Σ_k (0.5 - 0.5·tanh(5*(cos(θ - k·120°) + wave_swing)))·ft_k_down"""
    total = 0.0
    for k, leg in enumerate(TRI_LEGS):
        st = math.tanh(5.0 * (math.cos(theta - k * 2.0 * math.pi / 3.0) + wave_swing))
        total += (0.5 - 0.5 * st) * down.get(leg, 0.0)
    return total


def _value(expr, theta, down):
    values = _clock(theta)
    for leg in TRI_LEGS:
        values["ft_%s_down" % leg] = down.get(leg, 0.0)
    return evaluate(expr, **values)


@pytest.mark.parametrize("overrides", [TRI_R2_WAVE, TRI_R2_DYN, TRI3_R9])
def test_swing_contact_adds_one_term(overrides):
    _, base = run_design(**overrides)
    _, log = run_design(**overrides, swing_contact_cost=1.0)
    assert len(_rewards(log)) == len(_rewards(base)) + 1
    (term,) = _reward(log, "swing_contact")
    assert term.kwargs["weight"] == -1.0
    # 項を1つ足しただけで、ほかの宣言は変わらない（task の報酬の列は、末尾に swing_contact が1つ増える）
    assert signature([t for t in log if t is not term and t.kind != "assembly.task"]) == \
        signature([t for t in base if t.kind != "assembly.task"])
    (task,) = of_kind(log, "assembly.task")
    (task0,) = of_kind(base, "assembly.task")
    assert [r.label for r in task.kwargs["reward"]] == [r.label for r in task0.kwargs["reward"]] + ["swing_contact"]


def test_swing_contact_cost_scales_the_weight():
    _, log = run_design(**TRI3_R9, swing_contact_cost=2.5)
    (term,) = _reward(log, "swing_contact")
    assert term.kwargs["weight"] == -2.5


@pytest.mark.parametrize("overrides", [TRI_R2_WAVE, TRI_R2_DYN, TRI3_R9])
def test_swing_contact_formula(overrides):
    """式に数値を入れて、Σ_k (0.5 - 0.5·st_k)·ft_k_down（脚で割らない）と一致する。"""
    _, log = run_design(**overrides, swing_contact_cost=1.0)
    (term,) = _reward(log, "swing_contact")
    expr = term.args[0]
    ws = overrides.get("wave_swing", 0.5)
    combos = [{}, {"lf": 1.0}, {"rf": 1.0}, {"hc": 1.0}, {"lf": 1.0, "rf": 1.0, "hc": 1.0},
              {"lf": 1.0, "hc": 1.0}, {"rf": 0.5, "hc": 1.0}]
    for i in range(24):
        theta = i * 2.0 * math.pi / 24.0
        for down in combos:
            assert _value(expr, theta, down) == pytest.approx(_reference(theta, down, ws), abs=1e-5)


@pytest.mark.parametrize("ws", [0.5, 0.8])
def test_swing_contact_uses_gait_contact_stance(ws):
    """立脚の度合い st_k の文字列は gait_contact のものと同じ（wave_swing を変えても揃う）。"""
    _, log = run_design(**dict(TRI3_R9, wave_swing=ws), swing_contact_cost=1.0)
    (term,) = _reward(log, "swing_contact")
    (gait,) = _reward(log, "gait_contact")
    r3 = "%.6f" % (math.sqrt(3.0) / 2.0)
    phases = ["clock_cos", "(-0.5*clock_cos + %s*clock_sin)" % r3, "(-0.5*clock_cos - %s*clock_sin)" % r3]
    for phase, leg in zip(phases, TRI_LEGS):
        st = "tanh(5*(%s + %g))" % (phase, ws)
        assert st in gait.args[0]
        assert "(0.5 - 0.5*%s) * ft_%s_down" % (st, leg) in term.args[0]
    assert "/" not in term.args[0]  # 脚で割らない


# --- 落ちる場合 ---

@pytest.mark.parametrize("overrides", [{}, MINI_R2_P7_EFF, TRI_R2])
def test_swing_contact_rejects_non_wave(overrides):
    # 四足（既定・mini-r2-p7-eff）と、3本足でも波でないもの（tri-r2、gait_mode=0）
    with pytest.raises(ValueError, match="swing_contact_cost"):
        run_design(**overrides, swing_contact_cost=1.0)


@pytest.mark.parametrize("bad", [-0.5, -1e-9, 5.5])
def test_swing_contact_rejects_out_of_range(bad):
    with pytest.raises(ValueError, match="swing_contact_cost"):
        run_design(**TRI3_R9, swing_contact_cost=bad)


# --- 値: 遊脚の位相で接地していれば正、立脚なら 0 に近い ---

@pytest.mark.parametrize("k, leg", list(enumerate(TRI_LEGS)))
def test_swing_contact_positive_in_swing_zero_in_stance(k, leg):
    _, log = run_design(**TRI3_R9, swing_contact_cost=1.0)
    (term,) = _reward(log, "swing_contact")
    expr = term.args[0]
    # 脚 k の位相は cos(θ - k·120°)。θ = k·120° + 180° で遊脚の中心（位相 -1）、θ = k·120° で立脚の中心（位相 +1）
    swing = k * 2.0 * math.pi / 3.0 + math.pi
    stance = k * 2.0 * math.pi / 3.0
    assert _value(expr, swing, {leg: 1.0}) > 0.95       # 遊脚の位相で接地: ほぼ 1
    assert _value(expr, swing, {}) == pytest.approx(0.0)  # 遊脚の位相で浮いていれば 0
    assert _value(expr, stance, {leg: 1.0}) < 1e-3       # 立脚の位相で接地: ほぼ 0
    # 全脚が接地したままでも、どこかの脚が遊脚の位相にあるので罰が入る（和のまま、脚で割らない）
    assert _value(expr, swing, {l: 1.0 for l in TRI_LEGS}) > 0.95
