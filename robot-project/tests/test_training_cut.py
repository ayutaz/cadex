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
