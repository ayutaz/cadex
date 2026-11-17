"""vast.ai 学習基盤: progress.json → CSV / TensorBoard 変換。"""

from __future__ import annotations

import csv
import json

import pytest

from vastai import metrics


def _progress() -> dict:
    # cadex_train.py が書く cadex-training-progress-v1 の一部（*_curve は [iteration, value]）
    return {
        "schema": "cadex-training-progress-v1",
        "state": "training",
        "iteration": 2,
        "total": 10,
        "curve": [[0, -1.0], [1, -0.5], [2, 0.25]],
        "loss_curve": [[0, 3.0], [1, 2.0], [2, 1.5]],
        "episode_steps_curve": [[0, 10.0], [2, 30.0]],  # 間引かれて 1 が無い
        "action_std_curve": [[0, 0.69], [1, 0.6], [2, 0.55]],
        "terrain_levels_curve": [[0, [4, 0]], [1, [3, 1]], [2, [2, 2]]],
        "checkpoints": [],
    }


def test_progress_を反復ごとの行に直す():
    columns, rows = metrics.progress_rows(_progress())
    assert columns[:5] == ["iteration", "reward_per_step", "loss", "episode_steps", "action_std"]
    assert "terrain_levels" in columns
    assert [row["iteration"] for row in rows] == [0, 1, 2]
    assert rows[0]["reward_per_step"] == -1.0
    assert rows[1]["episode_steps"] is None  # 欠けた点は空
    assert rows[2]["terrain_levels"] == [2.0, 2.0]


def test_csv_に書く(tmp_path):
    out = tmp_path / "metrics.csv"
    metrics.write_csv(_progress(), out)
    with out.open(newline="", encoding="utf-8") as handle:
        read = list(csv.DictReader(handle))
    assert len(read) == 3
    assert read[0]["reward_per_step"] == "-1.0"
    assert read[1]["episode_steps"] == ""
    assert read[2]["terrain_levels"] == "2.0;2.0"


def test_csv_はファイルからも読める(tmp_path):
    source = tmp_path / "progress.json"
    source.write_text(json.dumps(_progress()), encoding="utf-8")
    out = tmp_path / "m.csv"
    metrics.write_csv(metrics.load_progress(source), out)
    assert out.read_text(encoding="utf-8").startswith("iteration,reward_per_step")


def test_曲線が無い進捗は見出しだけ(tmp_path):
    out = tmp_path / "empty.csv"
    metrics.write_csv({"state": "starting", "iteration": -1}, out)
    assert out.read_text(encoding="utf-8").strip() == (
        "iteration,reward_per_step,loss,episode_steps,action_std")


def test_crc32c_の既知値():
    assert metrics.crc32c(b"123456789") == 0xE3069283
    assert metrics.crc32c(b"") == 0


def test_tensorboard_のイベントファイルを依存なしで書いて読み戻せる(tmp_path):
    path = metrics.write_tensorboard(_progress(), tmp_path / "tb")
    assert path.name.startswith("events.out.tfevents.")
    events = metrics.read_scalar_events(path)
    assert ("train/reward_per_step", 2, pytest.approx(0.25)) in events
    assert ("train/loss", 0, pytest.approx(3.0)) in events
    # 欠けた点とリスト値は書かない
    assert not any(tag == "train/episode_steps" and step == 1 for tag, step, _ in events)
    assert not any(tag == "train/terrain_levels" for tag, _, _ in events)


def test_tensorboard_が入っていれば本家の読み手でも読める(tmp_path):
    event_accumulator = pytest.importorskip(
        "tensorboard.backend.event_processing.event_accumulator")
    metrics.write_tensorboard(_progress(), tmp_path / "tb")
    acc = event_accumulator.EventAccumulator(str(tmp_path / "tb"))
    acc.Reload()
    values = [event.value for event in acc.Scalars("train/reward_per_step")]
    assert values == pytest.approx([-1.0, -0.5, 0.25])
