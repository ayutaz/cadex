import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from evaluate_checkpoints import checkpoints, row_for  # noqa: E402


def test_checkpoints_are_sorted_by_iteration_and_final_comes_last(tmp_path):
    for name in ("walk.000200.cxpolicy", "walk.000050.cxpolicy", "walk.best.cxpolicy", "walk.cxpolicy", "x.txt"):
        (tmp_path / name).write_text("")
    found = checkpoints(tmp_path)
    assert [label for label, _ in found] == ["50", "200", "final"]


def test_row_summarises_one_evaluation():
    ev = {"seeds": [{"seed": 1, "pass": True, "failing": [], "metrics": {"speed_ratio": 1.0, "max_tilt_deg": 10.0}},
                    {"seed": 2, "pass": False, "failing": ["rated"], "metrics": {"speed_ratio": 0.5, "max_tilt_deg": 20.0}}]}
    row = row_for("100", ev, keys=["speed_ratio", "max_tilt_deg"])
    assert row == {"checkpoint": "100", "pass_rate": 0.5, "failing": {"rated": 1},
                   "speed_ratio": 0.75, "max_tilt_deg": 15.0}
