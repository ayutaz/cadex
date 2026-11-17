import pytest

from q8.compare import KEY_METRICS, compare, summarise_seeds, to_markdown


def _seeds(values, passes):
    return {"seeds": [{"seed": i, "pass": p, "metrics": {"mean_forward_speed_mm_s": v, "max_tilt_deg": None if v is None else 2.0 * v,
                                                         "falls": 0.0, "completed": 1.0}}
                      for i, (v, p) in enumerate(zip(values, passes))]}


def test_summary_is_mean_min_max_and_pass_rate():
    s = summarise_seeds(_seeds([10.0, 20.0, None], [True, False, True]), ["mean_forward_speed_mm_s"])
    assert s["pass_rate"] == pytest.approx(2 / 3)
    assert s["metrics"]["mean_forward_speed_mm_s"] == {"mean": 15.0, "min": 10.0, "max": 20.0, "n": 2}


def test_compare_reports_relative_change_per_metric():
    base = {"design": {"total_mass_g": 200.0, "com_height_mm": 70.0},
            "fixed_gait": _seeds([40.0, 60.0], [True, True])}
    var = {"design": {"total_mass_g": 220.0, "com_height_mm": 84.0},
           "fixed_gait": _seeds([60.0, 60.0], [True, False])}
    c = compare(base, var, labels=("base", "leg120"))
    assert c["design"]["total_mass_g"] == {"base": 200.0, "leg120": 220.0, "change": pytest.approx(0.10)}
    fg = c["fixed_gait"]
    assert fg["pass_rate"] == {"base": 1.0, "leg120": 0.5, "change": pytest.approx(-0.5)}
    assert fg["mean_forward_speed_mm_s"]["leg120"] == 60.0
    assert fg["mean_forward_speed_mm_s"]["change"] == pytest.approx(0.2)


def test_missing_section_is_reported_not_invented():
    c = compare({"design": {"total_mass_g": 1.0}}, {"design": {"total_mass_g": 2.0}}, labels=("a", "b"))
    assert c["policy"] is None
    md = to_markdown(c)
    assert "学習済み方策" in md and "未評価" in md


def test_key_metrics_cover_the_requested_evaluation():
    for name in ("mean_forward_speed_mm_s", "speed_error_rms_mm_s", "max_tilt_deg", "falls",
                 "effort_rms_rated_ratio_max", "cost_of_transport", "max_heading_deg"):
        assert name in KEY_METRICS
