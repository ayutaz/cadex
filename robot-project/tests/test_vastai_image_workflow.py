"""vast.ai 用の学習イメージを GitHub Actions でビルドし、GHCR（公開）に置くワークフロー。

手元（Apple Silicon）で linux/amd64 の CUDA 版 JAX を含むイメージを作るのは重いので、
GitHub の x86 ランナーでビルドする。認証は GITHUB_TOKEN だけで、個人のトークンは使わない。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
CADEX = ROOT.parent
WORKFLOW = CADEX / ".github" / "workflows" / "cadex-train-image.yml"
DOCKERFILE = "robot-project/vastai/Dockerfile"


@pytest.fixture(scope="module")
def wf() -> dict:
    if not WORKFLOW.exists():
        pytest.fail(f"{WORKFLOW} が無い")
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def _triggers(wf: dict) -> dict:
    return wf.get("on", wf.get(True))  # PyYAML は on を True と読む


def _steps(wf: dict) -> list[dict]:
    (job,) = wf["jobs"].values()
    return job["steps"]


def _step(wf: dict, action: str) -> dict:
    found = [s for s in _steps(wf) if str(s.get("uses", "")).startswith(action + "@")]
    assert len(found) == 1, f"{action} の step が {len(found)} 個"
    return found[0]


def _copy_sources() -> list[str]:
    text = (CADEX / DOCKERFILE).read_text(encoding="utf-8").replace("\\\n", " ")
    out: list[str] = []
    for line in text.splitlines():
        m = re.match(r"\s*COPY\s+(.+)", line)
        if m:
            words = [w for w in m.group(1).split() if not w.startswith("--")]
            out += words[:-1]
    return out


def test_runs_by_hand_and_when_an_input_of_the_image_changes(wf):
    on = _triggers(wf)
    assert "workflow_dispatch" in on
    paths = set(on["push"]["paths"])
    # イメージに入るもの（COPY 元）と、Dockerfile・ワークフロー自身が変わったら作り直す
    assert set(_copy_sources()) | {DOCKERFILE, ".github/workflows/cadex-train-image.yml"} <= paths


def test_only_writes_packages_with_the_repository_token(wf):
    assert wf["permissions"] == {"contents": "read", "packages": "write"}
    login = _step(wf, "docker/login-action")
    assert login["with"]["registry"] == "ghcr.io"
    assert login["with"]["password"] == "${{ secrets.GITHUB_TOKEN }}"


def test_builds_the_vastai_dockerfile_for_x86_from_the_cadex_root(wf):
    build = _step(wf, "docker/build-push-action")["with"]
    assert build["file"] == DOCKERFILE
    assert build["context"] == "."
    assert build["platforms"] == "linux/amd64"
    assert build["push"] is True


def test_image_is_named_cadex_train_and_tagged_by_commit(wf):
    meta = _step(wf, "docker/metadata-action")["with"]
    assert meta["images"] == "ghcr.io/${{ github.repository_owner }}/cadex-train"
    tags = meta["tags"]
    assert "type=sha" in tags and "type=raw,value=latest" in tags


def test_image_points_at_its_source_for_the_lgpl(wf):
    labels = _step(wf, "docker/metadata-action")["with"]["labels"]
    assert "org.opencontainers.image.source=https://github.com/${{ github.repository }}" in labels
    assert "org.opencontainers.image.licenses=LGPL-2.1-or-later" in labels


def test_every_action_is_pinned_to_a_major_version(wf):
    for s in _steps(wf):
        if "uses" in s:
            assert re.fullmatch(r"[\w.-]+/[\w.-]+@v\d+", s["uses"]), s["uses"]


@pytest.mark.parametrize("name", ["train_smoke.json", "train_full.json"])
def test_configs_use_a_commit_tag_of_the_published_image(name):
    import json

    image = json.loads((ROOT / "configs" / name).read_text())["instance"]["image"]
    # 再現できるよう sha タグ（latest や仮の値ではない）
    assert re.fullmatch(r"ghcr\.io/ayutaz/cadex-train:sha-[0-9a-f]{7}", image), image
