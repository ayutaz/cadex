"""設計スクリプト robot/design/quadruped8.py の報酬の重み（トルクの罰）を params で変えられること。

偽物の API でスクリプトを実行する（tests/test_design_battery.py の run_design）。
既定値では、params にする前の固定値（effort −0.5、effort_rated −0.3）と同じ報酬になる。
"""

from __future__ import annotations

import pytest

from test_design_battery import run_design


def _rewards(log) -> dict:
    return {t.kwargs["label"]: t for t in log if t.kind == "assembly.reward"}


def test_トルクの罰の重みの既定値は_params_にする前と同じ():
    _, log = run_design()
    rewards = _rewards(log)
    assert rewards["effort"].kwargs["weight"] == -0.5
    assert rewards["effort_rated"].kwargs["weight"] == -0.3


@pytest.mark.parametrize("w_effort, w_effort_rated", [(0.5, 1.0), (1.0, 0.3), (0.0, 2.0)])
def test_トルクの罰の重みは_params_で変えられ符号は罰のまま(w_effort, w_effort_rated):
    _, base_log = run_design()
    _, log = run_design(w_effort=w_effort, w_effort_rated=w_effort_rated)
    rewards = _rewards(log)
    assert rewards["effort"].kwargs["weight"] == -w_effort
    assert rewards["effort_rated"].kwargs["weight"] == -w_effort_rated
    # 式と、ほかの報酬は変わらない
    base = _rewards(base_log)
    assert [t.args for t in rewards.values()] == [t.args for t in base.values()]
    others = {k: t.kwargs for k, t in rewards.items() if k not in ("effort", "effort_rated")}
    assert others == {k: t.kwargs for k, t in base.items() if k not in ("effort", "effort_rated")}
