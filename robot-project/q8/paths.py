"""パスの取り決め。環境変数で上書きできる。"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]          # robot-project/
CADEX_REPO = Path(os.environ.get("CADEX_REPO", PROJECT.parent)).resolve()
CADEX_CLI = CADEX_REPO / "cadex"
TRAINER = CADEX_REPO / "training" / "cadex_train.py"
# Cadexのプロジェクトストア（ダッシュボードが既定で表示する ~/cadex-projects の下）
CADEX_PROJECTS = Path(os.environ.get("CADEX_PROJECTS", Path.home() / "cadex-projects"))
DESIGN_SCRIPT = PROJECT / "robot" / "design" / "quadruped8.py"
MESHES = PROJECT / "robot" / "meshes"
MJCF_DIR = PROJECT / "robot" / "mjcf"
URDF_DIR = PROJECT / "robot" / "urdf"
ARTIFACTS = PROJECT / "artifacts"


def cadex_project(variant: str = "base") -> Path:
    """機体バリアントごとのCadexプロジェクトのルート。"""
    name = "quadruped8" if variant == "base" else f"quadruped8-{variant}"
    return CADEX_PROJECTS / name
