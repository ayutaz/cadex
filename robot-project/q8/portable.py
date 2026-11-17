"""成果物からマシン固有の絶対パスを取り除く。

Cadex（`cadex evaluate`、学習器、カット処理）は受け取ったパスを絶対パスにして記録する。
robot-project は公開リポジトリに入るので、成果物を書く・写す箇所で、既知の根を置き換え文字にする。

- robot-project       → ``<robot-project>``
- Cadex のチェックアウト → ``<cadex>``
- Cadex のプロジェクトストア → ``~/cadex-projects``
- ホーム              → ``~``

置き換えは記録用で、元に戻す必要のある値（読み込み直すパス）には使っていない。
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .paths import CADEX_PROJECTS, CADEX_REPO, PROJECT

Roots = Sequence[tuple[Path, str]]

CXPOLICY_MAGIC = b"CXPOLICY1\n"
#: scrub_tree が書き換える拡張子（メッシュや動画、トレースの .stl/.mp4 などは触らない）
TEXT_SUFFIXES = frozenset({".json", ".csv", ".log", ".md", ".txt", ".xml"})


def default_roots() -> list[tuple[Path, str]]:
    return [(PROJECT, "<robot-project>"), (CADEX_REPO, "<cadex>"),
            (CADEX_PROJECTS, "~/cadex-projects"), (Path.home(), "~")]


def _pattern(roots: Roots) -> tuple[re.Pattern[str], dict[str, str]]:
    table = {str(r).rstrip("/"): s for r, s in roots}
    keys = sorted(table, key=len, reverse=True)  # 長い根を先に試す
    # 根の直後がパスの続き（英数字・.・-）なら別の名前なので置き換えない
    return re.compile("|".join(f"(?:{re.escape(k)})(?![\\w.-])" for k in keys)), table


def portable_text(text: str, roots: Roots | None = None) -> str:
    pattern, table = _pattern(roots if roots is not None else default_roots())
    return pattern.sub(lambda m: table[m.group(0)], text)


def portable(obj: Any, roots: Roots | None = None) -> Any:
    """dict / list / str を再帰的にたどり、キーと値の文字列を置き換えた複製を返す。"""
    roots = roots if roots is not None else default_roots()
    if isinstance(obj, str):
        return portable_text(obj, roots)
    if isinstance(obj, dict):
        return {portable(k, roots) if isinstance(k, str) else k: portable(v, roots) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [portable(v, roots) for v in obj]
    return obj


def write_json(path: Path, obj: Any, *, roots: Roots | None = None, **dump_kw: Any) -> Path:
    """json.dumps（ensure_ascii=False）で書き、書く前に根を置き換える。"""
    dump_kw.setdefault("ensure_ascii", False)
    text = json.dumps(obj, **dump_kw)
    path = Path(path)
    path.write_text(portable_text(text, roots))
    return path


def _canonical(header: dict) -> bytes:
    # トレーナーの encode_policy と同じ並びと区切り
    return json.dumps(header, ensure_ascii=True, sort_keys=True, separators=(",", ":"),
                      allow_nan=False).encode("ascii")


def _scrub_cxpolicy(blob: bytes, roots: Roots | None) -> bytes:
    if not blob.startswith(CXPOLICY_MAGIC):
        raise ValueError(".cxpolicy ではない")
    start = len(CXPOLICY_MAGIC) + 8
    n = int.from_bytes(blob[len(CXPOLICY_MAGIC):start], "little")
    header = json.loads(blob[start:start + n])
    payload = _canonical(portable(header, roots))
    return CXPOLICY_MAGIC + len(payload).to_bytes(8, "little") + payload + blob[start + n:]


def scrub_file(path: Path, roots: Roots | None = None) -> bool:
    """ファイルをその場で書き換える。変わったら True。.cxpolicy はヘッダだけ直し、重みはそのまま。"""
    path = Path(path)
    blob = path.read_bytes()
    if path.suffix == ".cxpolicy":
        if not blob.startswith(CXPOLICY_MAGIC):
            return False  # 方策の形式でないものは触らない（壊さない）
        new = _scrub_cxpolicy(blob, roots)
    else:
        new = portable_text(blob.decode("utf-8"), roots).encode("utf-8")
    if new == blob:
        return False
    path.write_bytes(new)
    return True


def scrub_tree(root: Path, roots: Roots | None = None,
               suffixes: Iterable[str] = TEXT_SUFFIXES | {".cxpolicy"}) -> list[Path]:
    """root の下の対象ファイルを書き換え、変わったファイルを返す。"""
    wanted = set(suffixes)
    return [p for p in sorted(Path(root).rglob("*"))
            if p.is_file() and p.suffix in wanted and scrub_file(p, roots)]


def json_from_stdout(text: str) -> dict:
    """標準出力から JSON オブジェクトを取り出す。先頭に混ざる警告の行は飛ばす。"""
    lines = text.splitlines()
    for i, line in enumerate(lines):
        if line.lstrip().startswith("{"):
            try:
                value = json.loads("\n".join(lines[i:]))
            except ValueError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError("標準出力に JSON オブジェクトが無い")
