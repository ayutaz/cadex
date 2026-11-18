"""1機体の評価（固定歩容 + 学習済み方策）をファイルに書き出す流れの結合テスト。"""

import csv
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))

from evaluate_variant import evaluate_variant, fixed_gait_target, read_cadex_evaluation  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def base_ready(base_model_xml):
    if not (ROOT / "artifacts" / "base" / "design_report.json").exists():
        pytest.skip("artifacts/base/design_report.json が無い")
    return base_model_xml


def test_fixed_gait_is_evaluated_on_the_task_seeds(base_ready, tmp_path):
    out = evaluate_variant("base", policy=None, out_dir=tmp_path, seconds=4.0)
    summary = json.loads((tmp_path / "summary.json").read_text())
    task = json.loads(next(base_ready.parent.glob("*-task.json")).read_text())
    assert [s["seed"] for s in out["fixed_gait"]["seeds"]] == task["success"]["seeds"]
    assert summary["design"]["total_mass_g"] > 100
    assert summary["policy"] is None
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert len(rows) == len(task["success"]["seeds"]) and rows[0]["label"] == "fixed_gait"
    speeds = [s["metrics"]["mean_forward_speed_mm_s"] for s in out["fixed_gait"]["seeds"]]
    assert min(speeds) > 20.0  # 開ループのトロットでも前進する


def test_fixed_gait_target_matches_trot_parameters(base_model):
    fn, nominal = fixed_gait_target(base_model, period_s=0.5, stride_m=0.04, lift_m=0.008)
    assert nominal == pytest.approx(160.0)
    t = fn(0.0)
    assert set(t) == {"hip_lf", "knee_lf", "hip_rf", "knee_rf", "hip_lh", "knee_lh", "hip_rh", "knee_rh"}


def test_reads_an_engine_evaluation(tmp_path):
    (tmp_path / "evaluation.json").write_text(json.dumps({
        "verdict": "fail", "seeds": [{"seed": 1, "pass": False, "metrics": {"falls": 0.0}, "predicates": []}],
        "wall_time_s": 3.0}))
    ev = read_cadex_evaluation(tmp_path)
    assert ev["seeds"][0]["seed"] == 1 and ev["verdict"] == "fail"


def test_四足でない機体では固定歩容を飛ばして理由を残す(tmp_path):
    # 固定歩容は四足のトロット（lf/rf/lh/rh）の目標しか作らない。3本足で回すと hc が 0 に固定されたまま
    # 四足のトロットとして黙って記録されるので、評価せずに飛ばしたことを出力に残す（final-review I2）。
    if not list((ROOT / "robot" / "mjcf" / "tri-r2").glob("*-model.xml")):
        pytest.skip("robot/mjcf/tri-r2 の MJCF が無い")
    out = evaluate_variant("tri-r2", policy=None, out_dir=tmp_path, seconds=1.0)
    reason = out["fixed_gait"]["skipped"]
    assert "四足" in reason
    assert "seeds" not in out["fixed_gait"]
    assert json.loads((tmp_path / "fixed_gait.json").read_text())["skipped"] == reason
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["fixed_gait"] == {"skipped": reason}
    rows = list(csv.DictReader((tmp_path / "metrics.csv").open()))
    assert not any(r["label"] == "fixed_gait" for r in rows)


def test_飛ばした固定歩容は比較で未評価になる():
    from q8.compare import compare

    skipped = {"skipped": "四足でないため"}
    c = compare({"design": {}, "fixed_gait": skipped, "policy": None},
                {"design": {}, "fixed_gait": skipped, "policy": None})
    assert c["fixed_gait"] is None
