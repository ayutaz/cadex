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
from .portable import Roots, json_from_stdout, scrub_tree, write_json

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
        write_json(out / "cut_report.json", report, indent=2)
        scrub_tree(out)  # Cadex の付け替えは絶対パスを記録する
    dropped = [line.strip() for line in relay.stdout.splitlines() + relay.stderr.splitlines()
               if line.startswith("randomisation dropped")]
    return {"fidelity_pass": bool(report.get("ok")) and not report.get("problems"),
            "options": report.get("options"), "dropped_randomisation": dropped, "task": str(cut_task_path(mjcf_dir))}


ARCHIVE_SUFFIXES = (".cxpolicy",)
ARCHIVE_NAMES = ("progress.json", "receipt.json", "checkpoints.json", "checkpoints.csv", "stderr.log",
                 "train.log", "metrics.csv")
#: vast.ai の起動記録（run.json）から落とす項目。インスタンスの接続先と pid は、公開しないうえ再現にも要らない。
RUN_RECORD_DROP = ("host", "port", "pid")


def save_receipt(stdout: str, path: Path, roots: Roots | None = None) -> Path:
    """学習器の標準出力（受領書の JSON）を保存する。前に混ざる警告の行は落とし、根を置き換える。"""
    return write_json(path, json_from_stdout(stdout), roots=roots)


def archive_run(run_dir: Path, dest: Path, roots: Roots | None = None) -> Path:
    """学習結果のうち、再現と比較に必要な小さいファイルだけを dest に写す（トレースや動画は除く）。

    写したファイルからは、マシン固有の絶対パスを取り除く（q8.portable）。
    """
    dest.mkdir(parents=True, exist_ok=True)
    for p in run_dir.iterdir():
        if p.is_file() and (p.suffix in ARCHIVE_SUFFIXES or p.name in ARCHIVE_NAMES):
            if p.name == "receipt.json":
                save_receipt(p.read_text(), dest / p.name, roots)
            else:
                shutil.copy2(p, dest / p.name)
    # vast.ai から pull した回: 受領書は stdout.log、起動の記録は run.json
    stdout = run_dir / "stdout.log"
    if stdout.exists() and not (run_dir / "receipt.json").exists():
        save_receipt(stdout.read_text(), dest / "receipt.json", roots)
    record = run_dir / "run.json"
    if record.exists():
        data = json.loads(record.read_text(encoding="utf-8"))
        write_json(dest / "run.json", {k: v for k, v in data.items() if k not in RUN_RECORD_DROP},
                   roots=roots, indent=2)
    final_eval = run_dir / "eval-final" / "evaluation.json"
    if final_eval.exists():
        (dest / "eval-final").mkdir(exist_ok=True)
        shutil.copy2(final_eval, dest / "eval-final" / "evaluation.json")
    scrub_tree(dest, roots)
    return dest
