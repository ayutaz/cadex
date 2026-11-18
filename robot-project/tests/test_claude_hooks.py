"""Claude Code の PreToolUse hook (.claude/hooks/guard_bash.py) のテスト。

スクリプトを subprocess で呼び、hook の JSON を標準入力に渡して結果を確かめる。
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK = Path(__file__).resolve().parents[2] / ".claude" / "hooks" / "guard_bash.py"
SETTINGS = Path(__file__).resolve().parents[2] / ".claude" / "settings.json"


def run_hook(command, tool_name="Bash"):
    payload = {"tool_name": tool_name, "tool_input": {"command": command}}
    proc = subprocess.run(
        [sys.executable, str(HOOK)],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=20,
    )
    return proc


def assert_denied(command):
    proc = run_hook(command)
    assert proc.returncode == 0, proc.stderr
    out = json.loads(proc.stdout)
    spec = out["hookSpecificOutput"]
    assert spec["hookEventName"] == "PreToolUse"
    assert spec["permissionDecision"] == "deny"
    assert spec["permissionDecisionReason"]
    return spec["permissionDecisionReason"]


def assert_allowed(command):
    proc = run_hook(command)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


DENY_ENV = [
    "cat .env",
    "cat robot-project/.env",
    "head .env",
    "tail -n 5 .env",
    "less .env",
    "more .env",
    "bat .env",
    "nl .env",
    "sed -n 1p .env",
    "grep KEY .env",
    "awk 1 .env",
    "printenv VAST_API_KEY",
    "echo $VAST_API_KEY",
    "env | grep -i key",
    "cd robot-project && cat .env",
    "echo ok; cat ./.env",
]

ALLOW_ENV = [
    "test -f .env",
    "ls -la .env",
    "grep -c VAST_API_KEY .env",
    "grep -q VAST_API_KEY .env",
    "git check-ignore .env",
    "uv run python -m vastai instances",
    "cat .env.example",
    "cat robot-project/.env.example",
    "ls",
    "git status",
    "uv run pytest -q",
    "echo hello",
    "grep -rn environment docs",
    "cat README.md",
    "env FOO=1 uv run pytest -q",
    "printenv HOME",
    "echo $HOME",
]

DENY_PUSH = [
    "git push origin main",
    "git push origin HEAD:main",
    "git push -f origin main",
    "git push origin master",
    "git push --force origin main",
    "cd x && git push origin main",
]

ALLOW_PUSH = [
    "git push origin quadruped-8dof",
    "git push -q origin quadruped-8dof",
    "git push -u origin quadruped-8dof",
    "git pull origin main",
    "git checkout main",
    "git log main..HEAD",
    "git push origin feature/main-fix",
]

DENY_ZSH = [
    "echo $F6:textfile=",
    "echo $S:t",
    "cp a $NAME:r.txt",
    "ls $DIR:h",
]

ALLOW_ZSH = [
    "echo ${F6}:textfile=",
    "export PATH=$PATH:/usr/bin",
    "echo $HOST:22",
    "echo '$X:y'",
    "curl https://example.com/a",
    "echo ${S}:t",
    "echo $1:foo",
]


def test_script_exists():
    assert HOOK.is_file()


@pytest.mark.parametrize("command", DENY_ENV)
def test_deny_env_dump(command):
    assert_denied(command)


@pytest.mark.parametrize("command", ALLOW_ENV)
def test_allow_env_safe(command):
    assert_allowed(command)


@pytest.mark.parametrize("command", DENY_PUSH)
def test_deny_push_main(command):
    assert_denied(command)


@pytest.mark.parametrize("command", ALLOW_PUSH)
def test_allow_push_other(command):
    assert_allowed(command)


@pytest.mark.parametrize("command", DENY_ZSH)
def test_deny_zsh_modifier(command):
    reason = assert_denied(command)
    assert "波括弧" in reason
    assert "${" in reason


@pytest.mark.parametrize("command", ALLOW_ZSH)
def test_allow_zsh_ok(command):
    assert_allowed(command)


def test_non_bash_tool_ignored():
    assert run_hook("cat .env", tool_name="Read").stdout.strip() == ""


@pytest.mark.parametrize("stdin", ["", "not json", "[]", '{"tool_input": null}'])
def test_bad_input_is_harmless(stdin):
    proc = subprocess.run(
        [sys.executable, str(HOOK)], input=stdin, capture_output=True, text=True, timeout=20
    )
    assert proc.returncode == 0
    assert proc.stdout.strip() == ""


def test_settings_registers_hook():
    cfg = json.loads(SETTINGS.read_text())
    entries = cfg["hooks"]["PreToolUse"]
    bash = [e for e in entries if e["matcher"] == "Bash"]
    assert bash
    cmds = [h["command"] for e in bash for h in e["hooks"]]
    assert any("guard_bash.py" in c and "python3" in c and "$CLAUDE_PROJECT_DIR" in c for c in cmds)


def test_settings_denies_reading_env_with_the_read_tool():
    # hook は Bash しか見ないので、Read ツールで .env を開く経路は permissions の deny で止める（final-review I4）。
    # .env.example は鍵を持たないので止めない。
    deny = json.loads(SETTINGS.read_text())["permissions"]["deny"]
    assert "Read(**/.env)" in deny
    assert "Read(.env)" in deny
    assert not any("example" in rule or ".env.*" in rule or ".env*" in rule for rule in deny)
