"""固定歩容を Cadex と同じ指標（CadexEvaluation）で評価する仕組みのテスト。"""

import json

import pytest

from q8.evaluation import (cadex_modules, evaluate_fixed_gait, metrics_table, record_episode,
                           success_spec)


def test_cadex_evaluation_modules_load_without_the_engine():
    dyn, ev = cadex_modules()
    assert hasattr(dyn, "evaluation_rig") and hasattr(ev, "measure")


def test_recorded_episode_is_a_cadex_trace(base_model):
    """1秒の立位を記録すると、CadexEvaluation.read_trace が読める形式になる。"""
    ep = record_episode(base_model, target_fn=lambda t: {}, seconds=1.0, control_hz=50, seed=1, command_mm_s=0.0)
    raw = ep["trace"]
    frames = [f for f in raw["frames"] if f["frame_kind"] == "solver_output"]
    assert len(frames) == 51
    assert frames[1]["nominal_time_s"] == pytest.approx(0.02)
    assert set(frames[0]["component_placements"]) >= {"c_chassis", "c_foot_lf"}
    assert len(frames[3]["joint_positions"]) == 8 and len(raw["actuator_channels"]) == 8
    _, ev = cadex_modules()
    read = ev.read_trace(raw)
    assert len(read["samples"]) == 51 and read["drives"] is not None
    assert read["episode"]["truncated"] is True


def test_standing_still_passes_upright_and_fails_walking(base_model):
    task = {"success": {"predicates": [{"id": "upright", "metric": "max_tilt_deg", "max": 25},
                                       {"id": "walks", "metric": "steps_min", "min": 4}]}}
    res = evaluate_fixed_gait(base_model, task, target_fn=lambda t: {}, seeds=[1], seconds=3.0,
                              command_mm_s=100.0, feet=["c_foot_lf", "c_foot_rf", "c_foot_lh", "c_foot_rh"])
    seed = res["seeds"][0]
    by = {p["id"]: p for p in seed["predicates"]}
    assert by["upright"]["pass"] is True
    assert by["walks"]["pass"] is False
    assert seed["metrics"]["completed"] == 1.0
    assert seed["metrics"]["max_tilt_deg"] < 5


def test_success_spec_is_read_from_the_exported_task(base_model_xml):
    task = json.loads(next(base_model_xml.parent.glob("*-task.json")).read_text())
    spec = success_spec(task)
    assert {p["metric"] for p in spec["predicates"]} >= {"completed", "max_tilt_deg", "steps_min"}
    assert spec["seeds"] and spec["feet"]


def test_metrics_table_flattens_seeds():
    res = {"seeds": [{"seed": 1, "pass": True, "metrics": {"a": 1.0, "b": 2.0}},
                     {"seed": 2, "pass": False, "metrics": {"a": 3.0, "b": None}}]}
    rows = metrics_table(res, label="x", keys=["a", "b"])
    assert rows == [{"label": "x", "seed": 1, "pass": True, "a": 1.0, "b": 2.0},
                    {"label": "x", "seed": 2, "pass": False, "a": 3.0, "b": None}]
