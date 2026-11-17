"""scripts/compare_seeds.py: 学習の乱数違いの複数回を評価し、二機体を比べる。"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import compare_seeds  # noqa: E402


def _fake_evaluate(calls):
    def evaluate(variant, *, policy, out_dir, video):
        calls.append((variant, Path(policy).parent.name, Path(out_dir)))
        rate = {"base": 0.5, "leg120": 1.0}[variant]
        summary = {"variant": variant, "policy": {"pass_rate": rate, "metrics": {
            "slip_share_max": {"mean": 0.5, "min": 0.5, "max": 0.5, "n": 10}}}}
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "summary.json").write_text(json.dumps(summary))
        # 本物の evaluate_variant の戻り値は summary.json と形が違う（variant を含まない）
        return {"fixed_gait": None, "policy": {}}
    return evaluate


def test_parse_run_needs_a_variant_and_a_run_folder(tmp_path):
    run = tmp_path / "full-s0"
    run.mkdir()
    (run / "walk.cxpolicy").write_bytes(b"x")
    assert compare_seeds.parse_run(f"base={run}") == ("base", run)
    with pytest.raises(ValueError):
        compare_seeds.parse_run(str(run))
    with pytest.raises(FileNotFoundError):
        compare_seeds.parse_run(f"base={tmp_path / 'missing'}")


def test_each_run_is_evaluated_into_its_own_folder_and_compared(tmp_path):
    runs = []
    for variant in ("base", "leg120"):
        for seed in range(3):
            d = tmp_path / "training" / f"{variant}-s{seed}"
            d.mkdir(parents=True)
            (d / "walk.cxpolicy").write_bytes(b"x")
            runs.append((variant, d))
    calls = []
    result = compare_seeds.run(runs, artifacts=tmp_path / "artifacts", evaluate=_fake_evaluate(calls))
    assert len(calls) == 6
    assert {c[2] for c in calls} == {tmp_path / "artifacts" / v / "evaluation-seeds" / f"{v}-s{s}"
                                     for v in ("base", "leg120") for s in range(3)}
    assert result["labels"] == ["base", "leg120"] and result["n"] == [3, 3]
    md = (tmp_path / "artifacts" / "compare" / "seeds_base_vs_leg120.md").read_text()
    assert "学習シード 3 回" in md
    saved = json.loads((tmp_path / "artifacts" / "compare" / "seeds_base_vs_leg120.json").read_text())
    assert saved["runs"]["base"] == ["base-s0", "base-s1", "base-s2"]


def test_an_existing_summary_is_reused(tmp_path):
    d = tmp_path / "training" / "base-s0"
    d.mkdir(parents=True)
    (d / "walk.cxpolicy").write_bytes(b"x")
    e = tmp_path / "training" / "leg-s0"
    e.mkdir(parents=True)
    (e / "walk.cxpolicy").write_bytes(b"x")
    calls = []
    compare_seeds.run([("base", d), ("leg120", e)], artifacts=tmp_path / "a", evaluate=_fake_evaluate(calls))
    compare_seeds.run([("base", d), ("leg120", e)], artifacts=tmp_path / "a", evaluate=_fake_evaluate(calls))
    assert len(calls) == 2  # 2回目は評価し直さない


def test_two_variants_are_required(tmp_path):
    d = tmp_path / "base-s0"
    d.mkdir()
    (d / "walk.cxpolicy").write_bytes(b"x")
    with pytest.raises(ValueError):
        compare_seeds.run([("base", d)], artifacts=tmp_path / "a", evaluate=_fake_evaluate([]))
