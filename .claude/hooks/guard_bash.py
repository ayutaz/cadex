#!/usr/bin/env python3
"""Claude Code の PreToolUse hook (Bash 用)。次の3つを拒否する。

1. .env の中身（API キーなど）を画面に出すコマンド
2. main / master への git push
3. zsh の変数の修飾子の事故（$NAME:t のように、$NAME の直後にコロンと英字が続く形）

標準ライブラリだけで動く。拒否しないときは何も出力せず、終了コード 0。
"""

import json
import re
import shlex
import sys

READERS = {
    "cat", "tac", "head", "tail", "less", "more", "bat", "nl", "sed", "grep",
    "egrep", "fgrep", "rg", "awk", "gawk", "cut", "sort", "strings", "xxd",
    "od", "hexdump", "base64", "paste", "uniq", "column", "view", "vi", "vim",
    "nano", "open", "source", ".",
}
SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)", re.I)
VAR_REF = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)")
ZSH_MODIFIER = re.compile(r"\$([A-Za-z_][A-Za-z0-9_]*):[A-Za-z]")
ENV_FILE = re.compile(r"^\.env(\.[A-Za-z0-9_-]+)?$")
ENV_SAFE_SUFFIX = {"example", "sample", "template", "dist"}


def deny(reason):
    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": reason,
        }
    }, ensure_ascii=False))
    sys.exit(0)


def strip_single_quotes(text):
    return re.sub(r"'[^']*'", "''", text)


def split_segments(command):
    """; && || | & と改行で分ける。クォートの中では分けない。"""
    segments, buf, quote, i = [], [], None, 0
    while i < len(command):
        c = command[i]
        if quote:
            buf.append(c)
            if c == "\\" and quote == '"' and i + 1 < len(command):
                buf.append(command[i + 1])
                i += 1
            elif c == quote:
                quote = None
        elif c in "'\"":
            quote = c
            buf.append(c)
        elif c == "\\" and i + 1 < len(command):
            buf.append(c + command[i + 1])
            i += 1
        elif c in ";|&\n":
            segments.append("".join(buf))
            buf = []
        else:
            buf.append(c)
        i += 1
    segments.append("".join(buf))
    return [s.strip() for s in segments if s.strip()]


def tokenize(segment):
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    # 先頭の VAR=value を飛ばす
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    return tokens


def is_env_file(token):
    base = token.rsplit("/", 1)[-1]
    m = ENV_FILE.match(base)
    if not m:
        return False
    suffix = (m.group(1) or "").lstrip(".")
    return suffix not in ENV_SAFE_SUFFIX


def grep_is_quiet(args):
    for a in args:
        if a in ("--count", "--quiet", "--silent", "--files-with-matches",
                 "--files-without-match"):
            return True
        if re.match(r"^-[A-Za-z]+$", a) and re.search(r"[cqlL]", a):
            return True
    return False


ENV_REASON = (
    ".env の中身を画面に出すコマンドは拒否します。API キーが会話に残るためです。"
    "存在の確認は `test -f .env`、キーの有無は `grep -c KEY .env` か `grep -q KEY .env` を使ってください。"
)


def check_env(segment, tokens):
    if not tokens:
        return
    cmd = tokens[0].rsplit("/", 1)[-1]
    args = tokens[1:]
    if cmd in READERS and any(is_env_file(a.lstrip("<")) for a in args):
        if cmd in ("grep", "egrep", "fgrep", "rg") and grep_is_quiet(args):
            return
        deny(ENV_REASON)
    if cmd == "printenv":
        names = [a for a in args if not a.startswith("-")]
        if not names or any(SECRET_NAME.search(n) for n in names):
            deny("printenv で環境変数の値を出すコマンドは拒否します。秘密の値が会話に残るためです。"
                 "設定の有無だけなら `test -n \"$NAME\"` を使ってください。")
    if cmd == "env":
        rest = [a for a in args if not a.startswith("-")]
        if not rest:
            deny("環境変数の一覧を出す `env` は拒否します。秘密の値が会話に残るためです。")
    if cmd in ("echo", "printf", "print"):
        for name in VAR_REF.findall(strip_single_quotes(segment)):
            if SECRET_NAME.search(name):
                deny(f"秘密らしい変数 ${name} の値を画面に出すコマンドは拒否します。"
                     "設定の有無だけなら `test -n \"$NAME\"` を使ってください。")


def check_push(tokens):
    if not tokens or tokens[0].rsplit("/", 1)[-1] != "git" or "push" not in tokens:
        return
    after = tokens[tokens.index("push") + 1:]
    for a in after:
        if a.startswith("-"):
            continue
        ref = a.lstrip("+")
        dst = ref.rsplit(":", 1)[-1]
        if dst.startswith("refs/heads/"):
            dst = dst[len("refs/heads/"):]
        if dst in ("main", "master"):
            deny("main / master への push は拒否します。作業ブランチ（例: quadruped-8dof）へ push してください。")


def main():
    try:
        data = json.load(sys.stdin)
    except Exception:
        return
    if not isinstance(data, dict) or data.get("tool_name") != "Bash":
        return
    tool_input = data.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str) or not command.strip():
        return

    m = ZSH_MODIFIER.search(strip_single_quotes(command))
    if m:
        name = m.group(1)
        deny(f"zsh では `${name}:` の直後が英字だと変数の修飾子になり、意図しない結果になります。"
             f"`${{{name}}}` のように波括弧で囲んでください（例: `${{{name}}}:textfile=`）。")

    for segment in split_segments(command):
        tokens = tokenize(segment)
        check_env(segment, tokens)
        check_push(tokens)


if __name__ == "__main__":
    main()
