"""vast.ai 学習基盤: Dockerfile と設定とスクリプトが同じ配置を前提にしていること。"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from vastai import config as cfg

ROOT = Path(__file__).resolve().parents[1]
CADEX = ROOT.parent
DOCKERFILE = ROOT / "vastai" / "Dockerfile"


def _copy_sources() -> list[str]:
    text = DOCKERFILE.read_text(encoding="utf-8").replace("\\\n", " ")
    sources = []
    for line in text.splitlines():
        match = re.match(r"\s*COPY\s+(.+)", line)
        if match:
            words = [w for w in match.group(1).split() if not w.startswith("--")]
            sources += words[:-1]
    return sources


def test_COPY_元は_Cadex_リポジトリに実在する():
    if not (CADEX / "training" / "cadex_train.py").exists():
        pytest.skip("Cadex のチェックアウトの中ではない")
    sources = _copy_sources()
    assert "training/cadex_train.py" in sources
    assert "training/cadex_goto.py" in sources
    for source in sources:
        assert (CADEX / source).is_file(), source


def test_CUDA12_の公式イメージと_jax_cuda12_のピン():
    text = DOCKERFILE.read_text(encoding="utf-8")
    assert re.search(r"^FROM nvidia/cuda:12\.\S+", text, re.M)
    assert '"jax[cuda12]==0.7.2"' in text


def test_設定のリモート配置は_Dockerfile_と一致する():
    text = DOCKERFILE.read_text(encoding="utf-8")
    for path in sorted((ROOT / "configs").glob("train_*.json")):
        remote = cfg.load_config(path)["remote"]
        assert f"python3 -m venv {Path(remote['python']).parent.parent}" in text, path
        assert f"{remote['trainer_root']}/training/" in text, path
