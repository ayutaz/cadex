"""Cadex CLI の薄いラッパー。`./cadex ... --json` を呼び、エンベロープ（JSON）を返す。

Cadexの内部APIには触れず、公開された CLI の契約（docs/CLI.md §3 の --json エンベロープ）だけを使う。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .paths import CADEX_CLI


class CadexError(RuntimeError):
    def __init__(self, message: str, envelope: dict | None = None, stderr: str = ""):
        super().__init__(message)
        self.envelope = envelope or {}
        self.stderr = stderr


def run(project: Path, *args: str, engine_timeout: int | None = None, timeout: float = 3600.0) -> dict:
    cmd = [str(CADEX_CLI), "--project", str(project), "--wait", "--json"]
    if engine_timeout:
        cmd += ["--engine-timeout", str(engine_timeout)]
    cmd += list(args)
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    try:
        env = json.loads(proc.stdout) if proc.stdout.strip() else {}
    except json.JSONDecodeError:
        env = {}
    if proc.returncode != 0 or not env.get("ok", False):
        detail = {k: env.get(k) for k in ("failure_code", "observed", "retry", "notes") if k in env}
        raise CadexError(f"cadex {' '.join(args)} が失敗 (exit {proc.returncode}): "
                         f"{json.dumps(detail, ensure_ascii=False)[:4000]}\n{proc.stderr[-2000:]}",
                         env, proc.stderr)
    return env


def write_script(project: Path, script: Path, sets: dict[str, float] | None = None, fit: bool = False) -> dict:
    env = run(project, "script", "--set", str(script), "--replace")
    if sets:
        args = ["params"]
        for k, v in sets.items():
            args += ["--set", f"{k}={v}"]
        if fit:
            args.append("--fit")
        env = run(project, *args, engine_timeout=1800)
    return env


def clearance(project: Path) -> None:
    run(project, "clearance", engine_timeout=1800)
    run(project, "clearance", "--sweep")


def export(project: Path, out: Path, formats: str = "step,stl") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    return run(project, "--format", formats, "export", "--out", str(out))


def inventory(project: Path) -> list[dict]:
    run(project, "inventory")
    rows = []
    for line in (project / "docs" / "inventory.md").read_text().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [c.strip().strip("`") for c in line.strip("|").split("|")]
        catalog = cells[2]
        family, sku = (catalog.split(" ", 1) + [""])[:2] if catalog != "—" else ("", "")
        rows.append({"component": cells[0], "output": cells[1], "family": family, "sku": sku.strip("`"),
                     "appearance": cells[3], "volume_mm3": float(cells[5])})
    return rows


def params(project: Path) -> dict:
    data = json.loads((project / "script.json").read_text())
    values = {s["name"]: s.get("default") for s in data.get("param_specs", []) if "name" in s}
    values.update({k: v for k, v in data.get("param_values", {}).items() if k in values})
    return values
