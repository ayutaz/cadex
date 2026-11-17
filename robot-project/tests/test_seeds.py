"""学習の乱数（学習シード）を変えた複数回の評価を、機体ごとにまとめて比べる。

強化学習の結果は学習シードで大きくぶれる。1回ずつの比較では、差が形状によるものか乱数によるものか
分からない。各回の評価（evaluate_variant の summary.json）の値を学習シードの方向に集め、平均・
標準偏差・最小・最大で比べる。
"""

from __future__ import annotations

import pytest

from q8 import seeds


def _summary(variant, pass_rate, speed, slip):
    return {"variant": variant,
            "policy": {"pass_rate": pass_rate,
                       "metrics": {"mean_forward_speed_mm_s": {"mean": speed, "min": speed, "max": speed, "n": 10},
                                   "slip_share_max": {"mean": slip, "min": slip, "max": slip, "n": 10},
                                   "falls": None}}}


BASE = [_summary("base", 0.5, 100.0, 0.8), _summary("base", 0.3, 90.0, 0.9), _summary("base", 0.7, 110.0, 0.7)]
LEG = [_summary("leg120", 1.0, 99.0, 0.2), _summary("leg120", 0.9, 97.0, 0.1), _summary("leg120", 1.0, 101.0, 0.3)]


def test_across_collects_each_run_and_its_spread():
    a = seeds.across(BASE)
    assert a["n"] == 3
    assert a["pass_rate"]["values"] == [0.5, 0.3, 0.7]
    assert a["pass_rate"]["mean"] == pytest.approx(0.5)
    assert a["pass_rate"]["std"] == pytest.approx(0.2)  # 標本標準偏差（n-1）
    assert (a["pass_rate"]["min"], a["pass_rate"]["max"]) == (0.3, 0.7)
    speed = a["metrics"]["mean_forward_speed_mm_s"]
    assert speed["values"] == [100.0, 90.0, 110.0] and speed["mean"] == pytest.approx(100.0)


def test_across_skips_missing_metrics_and_rejects_mixed_variants():
    a = seeds.across(BASE)
    assert a["metrics"]["falls"] is None
    with pytest.raises(ValueError):
        seeds.across([BASE[0], LEG[0]])
    with pytest.raises(ValueError):
        seeds.across([])


def test_a_single_run_has_no_spread():
    a = seeds.across(BASE[:1])
    assert a["pass_rate"]["std"] is None


def test_compare_says_whether_the_ranges_overlap():
    c = seeds.compare(seeds.across(BASE), seeds.across(LEG), labels=("base", "leg120"))
    pass_row = c["pass_rate"]
    assert pass_row["difference"] == pytest.approx(0.9667 - 0.5, abs=1e-3)
    assert pass_row["separated"] is True       # 0.3〜0.7 と 0.9〜1.0 は重ならない
    speed = c["metrics"]["mean_forward_speed_mm_s"]
    assert speed["separated"] is False          # 90〜110 と 97〜101 は重なる
    assert c["labels"] == ["base", "leg120"] and c["n"] == [3, 3]


def test_markdown_lists_mean_spread_and_verdict():
    md = seeds.to_markdown(seeds.compare(seeds.across(BASE), seeds.across(LEG), labels=("base", "leg120")))
    assert "| 合格率 |" in md
    assert "0.50 ± 0.20" in md
    assert "重ならない" in md and "重なる" in md
    assert "学習シード 3 回" in md
