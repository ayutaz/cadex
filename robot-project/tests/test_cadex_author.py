"""Cadex のプロジェクト（~/cadex-projects/<name>）の git の作者を、cadex リポジトリの作者にそろえる。

Cadex は初回の実行でプロジェクトを git init し、その場で自動コミットする。プロジェクトに
ローカル設定が無いと、PC 全体の設定（別の名前やメール）で記録されてしまう。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

import q8.cadex as cx


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def source(tmp_path) -> Path:
    repo = tmp_path / "cadex"
    repo.mkdir()
    _git("-C", str(repo), "init", "-q")
    _git("-C", str(repo), "config", "user.name", "tester")
    _git("-C", str(repo), "config", "user.email", "tester@example.com")
    return repo


def test_author_env_comes_from_the_source_repository(source):
    env = cx.author_env(source)
    assert env == {"GIT_AUTHOR_NAME": "tester", "GIT_AUTHOR_EMAIL": "tester@example.com",
                   "GIT_COMMITTER_NAME": "tester", "GIT_COMMITTER_EMAIL": "tester@example.com"}


def test_adopt_author_sets_the_project_local_config(source, tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    _git("-C", str(project), "init", "-q")
    cx.adopt_author(project, source)
    assert _git("-C", str(project), "config", "--local", "user.name") == "tester"
    assert _git("-C", str(project), "config", "--local", "user.email") == "tester@example.com"


def test_adopt_author_leaves_a_folder_without_git_alone(source, tmp_path):
    project = tmp_path / "plain"
    project.mkdir()
    cx.adopt_author(project, source)
    assert not (project / ".git").exists()


def test_run_gives_cadex_the_author_and_fixes_the_new_project(source, tmp_path, monkeypatch):
    project = tmp_path / "quadruped8-new"
    seen = {}

    def fake_call(cmd, env, timeout):
        # Cadex の初回実行: プロジェクトを作り git init して自動コミットする
        seen["env"] = env
        project.mkdir(exist_ok=True)
        subprocess.run(["git", "-C", str(project), "init", "-q"], check=True, env=env)
        (project / "x.txt").write_text("x")
        subprocess.run(["git", "-C", str(project), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(project), "-c", "user.name=cadex", "-c", "user.email=cadex@localhost",
                        "commit", "-q", "-m", "first"], check=True, env=env)
        return subprocess.CompletedProcess(cmd, 0, stdout=json.dumps({"ok": True}), stderr="")

    monkeypatch.setattr(cx, "_call", fake_call)
    monkeypatch.setattr(cx, "CADEX_REPO", source)
    cx.run(project, "inspect")
    assert seen["env"]["GIT_AUTHOR_EMAIL"] == "tester@example.com"
    # 最初の自動コミットから作者が正しい（-c user.* より環境変数が勝つ）
    assert _git("-C", str(project), "log", "-1", "--format=%an <%ae> | %cn <%ce>") == \
        "tester <tester@example.com> | tester <tester@example.com>"
    assert _git("-C", str(project), "config", "--local", "user.email") == "tester@example.com"
