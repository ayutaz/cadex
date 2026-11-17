"""学習用カットモデル（溶接部品を統合し床接触だけを残した速いモデル）と、タスクの付け替え。"""

import json

import pytest

from q8.training import CUT_DIR_NAME, cut_task_path, make_training_cut


def test_cut_task_lives_beside_the_export(tmp_path):
    assert cut_task_path(tmp_path) == tmp_path / CUT_DIR_NAME / "walk-task.json"


@pytest.mark.slow
def test_make_cut_for_the_base_project(base_model_xml, tmp_path):
    from q8.paths import cadex_project

    if not cadex_project("base").exists():
        pytest.skip("Cadexプロジェクトが無い")
    info = make_training_cut(cadex_project("base"), base_model_xml.parent / "walk-task.json", tmp_path)
    task = json.loads(cut_task_path(tmp_path).read_text())
    assert task["model"]["path"].endswith("cut.xml") if isinstance(task["model"], dict) else "cut.xml" in str(task["model"])
    assert info["fidelity_pass"] is True
    assert info["dropped_randomisation"]  # 統合されたボディのランダム化は落ちる（記録として返す）


def test_archive_run_keeps_only_small_reproducibility_files(tmp_path):
    from q8.training import archive_run

    run = tmp_path / "run"
    (run / "eval-100").mkdir(parents=True)
    (run / "eval-100" / "seed-1-trace.json").write_text("x" * 1000)
    for name in ("walk.cxpolicy", "walk.000100.cxpolicy", "walk.best.cxpolicy", "progress.json", "receipt.json",
                 "checkpoints.json", "checkpoints.csv", "stderr.log", "walk.000100.rollout-trace.json"):
        (run / name).write_text("{}")
    (run / "eval-final").mkdir()
    (run / "eval-final" / "evaluation.json").write_text("{}")
    dest = archive_run(run, tmp_path / "keep")
    kept = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert kept == ["checkpoints.csv", "checkpoints.json", "eval-final/evaluation.json", "progress.json",
                    "receipt.json", "stderr.log", "walk.000100.cxpolicy", "walk.best.cxpolicy", "walk.cxpolicy"]


def test_archive_run_also_keeps_a_vastai_run_without_its_address(tmp_path):
    # vast.ai から pull した回: 受領書は stdout.log（警告の行が混ざる）、ログは train.log、
    # 起動の記録は run.json（インスタンスの接続先と pid を含むので、それだけ落とす）
    from q8.training import archive_run

    run = tmp_path / "run"
    run.mkdir()
    for name in ("walk.cxpolicy", "walk.000100.cxpolicy", "progress.json", "metrics.csv", "train.log",
                 "walk.000100.rollout-trace.json"):
        (run / name).write_text("{}")
    (run / "stdout.log").write_text("Failed to import warp: No module named 'warp'\n"
                                    '{"sha256": "ab", "device": "gpu"}\n')
    (run / "run.json").write_text(json.dumps({
        "run_id": "full-s1", "remote_dir": "/workspace/runs/full-s1", "pid": 42, "host": "203.0.113.7",
        "port": 40022, "argv": ["--seed", "1"], "task_sha256": "cd", "device": "gpu"}))
    dest = archive_run(run, tmp_path / "keep")
    kept = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert kept == ["metrics.csv", "progress.json", "receipt.json", "run.json", "train.log",
                    "walk.000100.cxpolicy", "walk.cxpolicy"]
    assert json.loads((dest / "receipt.json").read_text()) == {"sha256": "ab", "device": "gpu"}
    record = json.loads((dest / "run.json").read_text())
    assert record["argv"] == ["--seed", "1"] and record["task_sha256"] == "cd"
    assert not {"host", "port", "pid"} & set(record)
    assert "203.0.113.7" not in (dest / "run.json").read_text()
