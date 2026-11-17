"""共通のフィクスチャ。リポジトリ内の生成物（MJCFなど）が無い環境では、それを使うテストを飛ばす。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BASE_MJCF_DIR = ROOT / "robot" / "mjcf" / "base"


@pytest.fixture(scope="session")
def base_model_xml() -> Path:
    found = sorted(BASE_MJCF_DIR.glob("*-model.xml"))
    if not found:
        pytest.skip("robot/mjcf/base の MJCF が無い（scripts/build_robot.py で生成する）")
    return found[0]


@pytest.fixture(scope="session")
def base_model(base_model_xml):
    import mujoco

    return mujoco.MjModel.from_xml_path(str(base_model_xml))
