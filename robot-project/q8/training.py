"""学習用カットモデルの作成（Cadex の training/cadex_training_model.py cut と cadex_relay.py を呼ぶ）。

カットモデルは、溶接部品を1つのボディに統合し、床との接触だけを残した学習専用のMJCF。
Cadex のカット処理は、エンジンのモデルとの軌道差（5 mm以内）と立位を自ら検査し、通らなければ拒否する。
本機ではCPUで約9倍速い（1,032 → 9,180 env steps/s、256環境）。評価は常にエンジンのモデルで行う。
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .paths import CADEX_REPO, PROJECT

CUT_DIR_NAME = "cut"


def cut_task_path(mjcf_dir: Path) -> Path:
    return mjcf_dir / CUT_DIR_NAME / "walk-task.json"


def _python() -> str:
    return str(PROJECT / ".venv" / "bin" / "python")


def make_training_cut(cadex_project: Path, export_task: Path, mjcf_dir: Path) -> dict:
    env = {**os.environ, "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS", "cpu")}
    out = mjcf_dir / CUT_DIR_NAME
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        cut = subprocess.run([_python(), str(CADEX_REPO / "training" / "cadex_training_model.py"), "cut",
                              "--project", str(cadex_project), "--out", str(tmp / "cut.xml")],
                             capture_output=True, text=True, env=env)
        if cut.returncode != 0:
            raise RuntimeError(f"cut に失敗: {cut.stderr[-2000:]}")
        report = json.loads(cut.stdout)
        relay = subprocess.run([_python(), str(CADEX_REPO / "training" / "cadex_relay.py"), str(export_task),
                                str(tmp / "cut.xml"), "--out", str(tmp / "relay")],
                               capture_output=True, text=True, env=env)
        if relay.returncode != 0:
            raise RuntimeError(f"relay に失敗: {relay.stderr[-2000:]}")
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        for f in (tmp / "relay" / "outputs").iterdir():
            shutil.copy2(f, out / f.name)
        shutil.copy2(tmp / "relay" / "relay.json", out / "relay.json")
        (out / "cut_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    dropped = [line.strip() for line in relay.stdout.splitlines() + relay.stderr.splitlines()
               if line.startswith("randomisation dropped")]
    return {"fidelity_pass": bool(report.get("ok")) and not report.get("problems"),
            "options": report.get("options"), "dropped_randomisation": dropped, "task": str(cut_task_path(mjcf_dir))}
