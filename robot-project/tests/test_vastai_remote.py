"""vast.ai 学習基盤: SSH/rsync のコマンド組み立て、起動、回収と検証、再開。

subprocess は偽物の runner で差し替える。ssh も rsync も実際には走らない。
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from pathlib import Path

import pytest

from vastai import remote
from vastai.config import SSHTarget

ROOT = Path(__file__).resolve().parents[1]
REAL_TASK = ROOT / "robot" / "mjcf" / "base" / "walk-task.json"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class FakeRunner:
    """呼ばれた argv を記録し、handler があればそれに応答させる。"""

    def __init__(self, handler=None):
        self.calls: list[list[str]] = []
        self.handler = handler

    def __call__(self, argv, input=None):
        argv = [str(item) for item in argv]
        self.calls.append(argv)
        if self.handler is not None:
            result = self.handler(argv)
            if result is not None:
                return result
        return subprocess.CompletedProcess(argv, 0, "", "")

    def remote_commands(self) -> list[str]:
        return [call[-1] for call in self.calls if call[0] == "ssh"]


def ok(argv, stdout=""):
    return subprocess.CompletedProcess(argv, 0, stdout, "")


TARGET = SSHTarget(host="ssh5.vast.ai", port=41234, user="root", key="")
REMOTE = {"work_dir": "/workspace/runs", "trainer_root": "/opt/cadex",
          "python": "/opt/venv/bin/python"}


def make_bundle(tmp_path: Path, *, nested: bool = False) -> Path:
    model = b"<mujoco model='m'/>"
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    (outputs / "walk-model.xml").write_bytes(model)
    task = {"schema": "cadex-training-task-v1",
            "model": {"path": "outputs/walk-model.xml", "sha256": sha(model)}}
    task_path = outputs / "walk-task.json"
    task_path.write_text(json.dumps(task), encoding="utf-8")
    return task_path


def config(task: Path, **trainer) -> dict:
    return {
        "name": "smoke",
        "bundle": str(task),
        "device": "gpu",
        "env": {"JAX_PLATFORMS": "cuda"},
        "policy_name": "walk",
        "trainer": {"seed": 0, "iterations": 10, "envs": 64, "checkpoint_every": 2, **trainer},
        "resume": {"critic_warmup": 3, "actor_ramp": 2},
        "remote": REMOTE,
    }


# --- コマンド組み立て -------------------------------------------------------

def test_ssh_argv():
    assert TARGET.ssh_argv("echo hi") == [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        "-p", "41234", "root@ssh5.vast.ai", "echo hi"]


def test_ssh_argv_は鍵を_i_で渡す(tmp_path):
    key = tmp_path / "id"
    key.write_text("x")
    target = SSHTarget(host="h", port=22, user="root", key=str(key))
    assert target.ssh_argv("true")[:14] == [
        "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
        "-p", "22", "-i"]


def test_接続の止まった_ssh_を待ち続けない():
    # 起動直後の台への確認の ssh が、応答のないまま7分止まり、sweep の台が動かなくなった（2026-11-17）
    argv = TARGET.ssh_argv("true")
    options = [argv[i + 1] for i, a in enumerate(argv) if a == "-o"]
    assert "ConnectTimeout=20" in options
    assert "ServerAliveInterval=15" in options and "ServerAliveCountMax=4" in options


def test_確認の重いコマンドはリモートで時間を区切る(tmp_path):
    # GPU の壊れた台では nvidia-smi や jax の初期化が返らないことがある
    runner = FakeRunner()
    session = remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs")
    session.check("/opt/cadex", "/opt/venv/bin/python")
    command = runner.remote_commands()[-1]
    assert "timeout 60 nvidia-smi" in command
    assert "timeout 180 /opt/venv/bin/python" in command


def test_rsync_の送りと受け():
    up = TARGET.rsync_up(["a.json", "b.xml"], "/workspace/runs/x/")
    assert up == ["rsync", "-rt", "-e",
                  ("ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "
                   "-o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 41234"),
                  "a.json", "b.xml", "root@ssh5.vast.ai:/workspace/runs/x/"]
    down = TARGET.rsync_down("/workspace/runs/x/", "local/", includes=["*.cxpolicy"])
    assert down[:4] == ["rsync", "-rt", "-e",
                        ("ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -o ConnectTimeout=20 "
                   "-o ServerAliveInterval=15 -o ServerAliveCountMax=4 -p 41234")]
    assert "--include=*.cxpolicy" in down and down[-3] == "--exclude=*"
    assert down[-2:] == ["root@ssh5.vast.ai:/workspace/runs/x/", "local/"]


def test_タスクのペアを解決しモデルの_sha256_を確かめる(tmp_path):
    task = make_bundle(tmp_path)
    bundle = remote.resolve_bundle(task)
    assert bundle.model.name == "walk-model.xml"
    assert bundle.task_sha256 == sha(task.read_bytes())


def test_同梱のロボットのタスクも解決できる():
    if not REAL_TASK.exists():
        pytest.skip("robot/mjcf/base/walk-task.json が無い")
    bundle = remote.resolve_bundle(REAL_TASK)
    assert bundle.model.exists()


def test_モデルが書き換わっていれば送る前に止める(tmp_path):
    task = make_bundle(tmp_path)
    (task.parent / "walk-model.xml").write_bytes(b"changed")
    with pytest.raises(remote.RemoteError, match="sha256"):
        remote.resolve_bundle(task)


def test_起動コマンドは_nohup_setsid_で切り離し_pid_を自分で書く():
    command = remote.launch_command(
        run_dir="/workspace/runs/smoke-1", python="/opt/venv/bin/python",
        trainer="/opt/cadex/training/cadex_train.py", task_name="walk-task.json",
        out_name="walk.cxpolicy", argv=["--iterations", "10", "--label", "a b"],
        env={"JAX_PLATFORMS": "cuda"})
    assert command.startswith("mkdir -p /workspace/runs/smoke-1 && cd /workspace/runs/smoke-1 && ")
    assert "(nohup setsid sh -c " in command
    assert command.rstrip().endswith("< /dev/null > /dev/null 2>&1 &)")
    inner = shlex.split(command.split("sh -c ", 1)[1])[0]
    assert inner.startswith("echo $$ > train.pid; exec env ")
    assert "JAX_PLATFORMS=cuda" in inner
    assert "CADEX_RUN_DRIVER=vastai" in inner
    words = shlex.split(inner.split("exec ", 1)[1])
    assert words[words.index("/opt/cadex/training/cadex_train.py") + 1] == \
        "/workspace/runs/smoke-1/walk-task.json"
    assert words[words.index("--out") + 1] == "/workspace/runs/smoke-1/walk.cxpolicy"
    assert words[words.index("--progress") + 1] == "/workspace/runs/smoke-1/progress.json"
    assert "a b" in words  # 空白を含む値も崩れない
    assert words[-4:] == [">", "stdout.log", "2>", "train.log"]


def test_起動コマンドは学習の終わりを待たずに戻る(tmp_path):
    # ssh は、リモートのシェルとその子が出力をつかんでいるあいだ戻らない。
    # `a && b && c &` だと & が && の全体にかかり、全体を包むサブシェルが学習の終わりまで
    # ssh の出力をつかむ（vast.ai で launch が戻らなかった）。パイプで ssh の出力に見立てて確かめる。
    import sys
    import time

    trainer = tmp_path / "slow_trainer.py"
    trainer.write_text("import time\ntime.sleep(8)\n")
    command = remote.launch_command(
        run_dir=str(tmp_path / "run"), python=sys.executable, trainer=str(trainer),
        task_name="walk-task.json", out_name="walk.cxpolicy", argv=[], env={})
    # macOS には setsid が無い。Linux の setsid は、プロセスグループの長でなければ fork せずに
    # exec する（今回の不具合はこの場合に起きる）。同じ動きの代用品を PATH の先頭に置く。
    import os
    import shutil

    env = dict(os.environ)
    if shutil.which("setsid") is None:
        shim = tmp_path / "bin"
        shim.mkdir()
        (shim / "setsid").write_text('#!/bin/sh\nexec "$@"\n')
        (shim / "setsid").chmod(0o755)
        env["PATH"] = f"{shim}:{env['PATH']}"
    start = time.monotonic()
    subprocess.run(["bash", "-c", command], capture_output=True, timeout=30, env=env)
    assert time.monotonic() - start < 4
    for _ in range(50):
        if (tmp_path / "run" / "train.pid").exists():
            break
        time.sleep(0.1)
    assert (tmp_path / "run" / "train.pid").read_text().strip().isdigit()


# --- 起動 -------------------------------------------------------------------

def test_launch_はペアを送り切り離して起動し記録を残す(tmp_path):
    task = make_bundle(tmp_path)

    def handler(argv):
        if argv[0] == "ssh" and "cat " in argv[-1] and "train.pid" in argv[-1]:
            return ok(argv, "4242\n")
        return None

    runner = FakeRunner(handler)
    session = remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs",
                             sleep=lambda s: None)
    result = session.launch(config(task), run_id="smoke-1")

    rsyncs = [c for c in runner.calls if c[0] == "rsync"]
    assert len(rsyncs) == 1
    assert rsyncs[0][-3:] == [str(task), str(task.parent / "walk-model.xml"),
                              "root@ssh5.vast.ai:/workspace/runs/smoke-1/"]
    commands = runner.remote_commands()
    assert commands[0] == "mkdir -p /workspace/runs/smoke-1"
    assert any("nohup setsid" in c for c in commands)
    assert result["pid"] == 4242
    record = json.loads((tmp_path / "runs" / "smoke-1" / "run.json").read_text())
    assert record["remote_dir"] == "/workspace/runs/smoke-1"
    assert record["policy_name"] == "walk"
    assert record["total_iterations"] == 10
    assert record["task_sha256"] == sha(task.read_bytes())


def test_launch_の_dry_run_は何も実行しない(tmp_path):
    task = make_bundle(tmp_path)
    runner = FakeRunner()
    session = remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs")
    result = session.launch(config(task), run_id="smoke-1", dry_run=True)
    assert runner.calls == []
    assert any(c[0] == "rsync" for c in result["commands"])
    assert not (tmp_path / "runs" / "smoke-1").exists()


# --- 回収と検証 -------------------------------------------------------------

def make_pulled(dest: Path, *, total=10, state="done") -> tuple[dict, dict]:
    """回収済みのディレクトリを作り、(progress, receipt) を返す。"""

    dest.mkdir(parents=True, exist_ok=True)
    blobs = {"walk.000002.cxpolicy": b"c2", "walk.000004.cxpolicy": b"c4",
             "walk.best.cxpolicy": b"best", "walk.cxpolicy": b"final"}
    for name, blob in blobs.items():
        (dest / name).write_bytes(blob)
    progress = {
        "schema": "cadex-training-progress-v1", "state": state,
        "iteration": 4 if state != "done" else total - 1, "total": total,
        "wall_time_s": 50.0,
        "checkpoints": [
            {"tag": "000002", "iteration": 1, "path": "walk.000002.cxpolicy",
             "sha256": sha(b"c2"), "reward_per_step": 0.1},
            {"tag": "best", "iteration": 3, "path": "walk.best.cxpolicy",
             "sha256": sha(b"best"), "reward_per_step": 0.3},
            {"tag": "000004", "iteration": 3, "path": "walk.000004.cxpolicy",
             "sha256": sha(b"c4"), "reward_per_step": 0.3},
        ],
    }
    (dest / "progress.json").write_text(json.dumps(progress), encoding="utf-8")
    receipt = {"sha256": sha(b"final"), "device": "gpu", "out": "walk.cxpolicy"}
    return progress, receipt


def remote_sums(dest: Path) -> dict[str, str]:
    return {p.name: sha(p.read_bytes()) for p in dest.glob("*.cxpolicy")}


def test_検証は全ファイルの_sha256_を三者で突き合わせる(tmp_path):
    progress, receipt = make_pulled(tmp_path)
    report = remote.verify_run(tmp_path, progress, receipt, remote_sums(tmp_path),
                               policy_name="walk", expect_device="gpu")
    assert {row["file"] for row in report} == {
        "walk.000002.cxpolicy", "walk.000004.cxpolicy", "walk.best.cxpolicy", "walk.cxpolicy"}
    assert all(row["ok"] for row in report)


def test_手元のファイルが壊れていれば検証が失敗する(tmp_path):
    progress, receipt = make_pulled(tmp_path)
    sums = remote_sums(tmp_path)
    (tmp_path / "walk.000004.cxpolicy").write_bytes(b"truncated")
    with pytest.raises(remote.VerificationError, match="walk.000004.cxpolicy"):
        remote.verify_run(tmp_path, progress, receipt, sums,
                          policy_name="walk", expect_device="gpu")


def test_最終ポリシーが受領書と違えば失敗する(tmp_path):
    progress, receipt = make_pulled(tmp_path)
    receipt["sha256"] = "0" * 64
    with pytest.raises(remote.VerificationError, match="walk.cxpolicy"):
        remote.verify_run(tmp_path, progress, receipt, remote_sums(tmp_path),
                          policy_name="walk", expect_device="gpu")


def test_GPU_のはずが_CPU_で走っていれば失敗する(tmp_path):
    progress, receipt = make_pulled(tmp_path)
    receipt["device"] = "cpu"
    with pytest.raises(remote.VerificationError, match="cpu"):
        remote.verify_run(tmp_path, progress, receipt, remote_sums(tmp_path),
                          policy_name="walk", expect_device="gpu")


def test_受領書は_stdout_の最後の_JSON_行():
    text = "Failed to import warp: No module named 'warp'\n" \
           '{"sha256": "ab", "device": "gpu"}\n'
    assert remote.parse_receipt(text) == {"sha256": "ab", "device": "gpu"}
    assert remote.parse_receipt("warn only\n") is None


def test_リモートの_sha256sum_出力を読む():
    text = f"{'a' * 64}  walk.cxpolicy\n{'b' * 64}  walk.000002.cxpolicy\n"
    assert remote.parse_sha256sum(text) == {
        "walk.cxpolicy": "a" * 64, "walk.000002.cxpolicy": "b" * 64}


def test_pull_は回収してから検証する(tmp_path):
    staging = tmp_path / "remote-copy"
    progress, receipt = make_pulled(staging)
    (staging / "stdout.log").write_text("noise\n" + json.dumps(receipt) + "\n")

    def handler(argv):
        if argv[0] == "rsync":
            dest = Path(argv[-1])
            dest.mkdir(parents=True, exist_ok=True)
            for item in staging.iterdir():
                (dest / item.name).write_bytes(item.read_bytes())
            return ok(argv)
        if argv[0] == "ssh" and "sha256sum" in argv[-1]:
            return ok(argv, "".join(f"{digest}  {name}\n"
                                    for name, digest in remote_sums(staging).items()))
        return None

    runner = FakeRunner(handler)
    local_root = tmp_path / "runs"
    record_dir = local_root / "smoke-1"
    record_dir.mkdir(parents=True)
    (record_dir / "run.json").write_text(json.dumps({
        "run_id": "smoke-1", "remote_dir": "/workspace/runs/smoke-1",
        "policy_name": "walk", "device": "gpu", "total_iterations": 10}))
    session = remote.Session(TARGET, runner=runner, local_root=local_root)
    report = session.pull("smoke-1")
    assert all(row["ok"] for row in report)
    assert (record_dir / "walk.cxpolicy").read_bytes() == b"final"
    assert (record_dir / "metrics.csv").exists()


# --- 再開 -------------------------------------------------------------------

def test_最新チェックポイントは番号の最大で_best_ではない():
    progress = {"checkpoints": [
        {"tag": "000100", "iteration": 99, "path": "walk.000100.cxpolicy", "sha256": "1"},
        {"tag": "best", "iteration": 250, "path": "walk.best.cxpolicy", "sha256": "b"},
        {"tag": "000300", "iteration": 299, "path": "walk.000300.cxpolicy", "sha256": "3"},
        {"tag": "000200", "iteration": 199, "path": "walk.000200.cxpolicy", "sha256": "2"},
    ]}
    latest = remote.latest_checkpoint(progress)
    assert latest["path"] == "walk.000300.cxpolicy"
    assert latest["completed"] == 300


def test_進捗が無ければファイル名から最新を選ぶ(tmp_path):
    for name in ("walk.000100.cxpolicy", "walk.000900.cxpolicy", "walk.best.cxpolicy",
                 "walk.000050.cxpolicy", "walk.cxpolicy"):
        (tmp_path / name).write_bytes(b"x")
    latest = remote.latest_checkpoint(None, directory=tmp_path, policy_name="walk")
    assert latest["path"] == "walk.000900.cxpolicy"
    assert latest["completed"] == 900


def test_チェックポイントが一つも無ければ_None():
    assert remote.latest_checkpoint({"checkpoints": []}) is None


def test_再開の引数は残り反復と_warm_start():
    progress = {"state": "failed", "total": 1000, "checkpoints": [
        {"tag": "000300", "iteration": 299, "path": "walk.000300.cxpolicy", "sha256": "3"}]}
    plan = remote.resume_plan(progress, {"critic_warmup": 20, "actor_ramp": 10})
    assert plan["checkpoint"] == "walk.000300.cxpolicy"
    assert plan["iterations"] == 700
    assert plan["argv"] == ["--critic-warmup", "20", "--actor-ramp", "10"]


def test_終わった学習は再開しない():
    progress = {"state": "done", "total": 10, "checkpoints": [
        {"tag": "000004", "iteration": 3, "path": "walk.000004.cxpolicy", "sha256": "4"}]}
    with pytest.raises(remote.RemoteError, match="終わ"):
        remote.resume_plan(progress, {})


def test_resume_は最新チェックポイントを_warm_に送り_init_from_で起動する(tmp_path):
    task = make_bundle(tmp_path)
    local_root = tmp_path / "runs"
    old = local_root / "smoke-1"
    progress, _ = make_pulled(old, state="failed")
    (old / "run.json").write_text(json.dumps({
        "run_id": "smoke-1", "remote_dir": "/workspace/runs/smoke-1",
        "policy_name": "walk", "device": "gpu", "total_iterations": 10}))

    def handler(argv):
        if argv[0] == "ssh" and "cat " in argv[-1] and "train.pid" in argv[-1]:
            return ok(argv, "77\n")
        if argv[0] == "ssh" and "sha256sum" in argv[-1]:
            return ok(argv, "".join(f"{d}  {n}\n" for n, d in remote_sums(old).items()))
        return None

    runner = FakeRunner(handler)
    session = remote.Session(TARGET, runner=runner, local_root=local_root,
                             sleep=lambda s: None)
    result = session.resume("smoke-1", config(task), run_id="smoke-2", pull_first=False)

    warm_rsync = [c for c in runner.calls if c[0] == "rsync" and c[-1].endswith("/warm/")]
    assert len(warm_rsync) == 1
    assert warm_rsync[0][-2] == str(old / "walk.000004.cxpolicy")
    launch = next(c for c in runner.remote_commands() if "nohup setsid" in c)
    inner = shlex.split(launch.split("sh -c ", 1)[1])[0]
    words = shlex.split(inner.split("exec ", 1)[1])
    assert words[words.index("--init-from") + 1] == \
        "/workspace/runs/smoke-2/warm/walk.000004.cxpolicy"
    assert words[words.index("--iterations") + 1] == "6"  # 10 - 4
    assert words[words.index("--critic-warmup") + 1] == "3"
    assert words.count("--iterations") == 1
    assert result["resumed_from"]["checkpoint"] == "walk.000004.cxpolicy"
    record = json.loads((local_root / "smoke-2" / "run.json").read_text())
    assert record["resumed_from"]["run_id"] == "smoke-1"


def test_resume_は手元のチェックポイントが壊れていれば送らない(tmp_path):
    task = make_bundle(tmp_path)
    local_root = tmp_path / "runs"
    old = local_root / "smoke-1"
    make_pulled(old, state="failed")
    (old / "walk.000004.cxpolicy").write_bytes(b"broken")
    (old / "run.json").write_text(json.dumps({
        "run_id": "smoke-1", "remote_dir": "/workspace/runs/smoke-1",
        "policy_name": "walk", "device": "gpu", "total_iterations": 10}))
    runner = FakeRunner()
    session = remote.Session(TARGET, runner=runner, local_root=local_root)
    with pytest.raises(remote.VerificationError):
        session.resume("smoke-1", config(task), run_id="smoke-2", pull_first=False)
    assert not any(c[0] == "rsync" for c in runner.calls)


# --- 停止 -------------------------------------------------------------------

def test_stop_は_TERM_を送り止まったことを確かめる(tmp_path):
    local_root = tmp_path / "runs"
    (local_root / "smoke-1").mkdir(parents=True)
    (local_root / "smoke-1" / "run.json").write_text(json.dumps(
        {"run_id": "smoke-1", "remote_dir": "/workspace/runs/smoke-1"}))

    def handler(argv):
        if argv[0] == "ssh" and "kill -TERM" in argv[-1]:
            return ok(argv, "stopped\n")
        if argv[0] == "ssh" and "kill -0" in argv[-1]:
            return ok(argv, "no\n")
        return None

    runner = FakeRunner(handler)
    session = remote.Session(TARGET, runner=runner, local_root=local_root,
                             sleep=lambda s: None)
    assert session.stop("smoke-1") == "stopped"
    assert any("kill -TERM" in c and "/workspace/runs/smoke-1/train.pid" in c
               for c in runner.remote_commands())


# --- 複数のインスタンス -----------------------------------------------------
# 学習を複数のインスタンスで並べると、.env（や環境変数）の接続先は1つしか指せない。既にある回の
# status / pull / stop は、その回を起動した接続先（run.json の host と port）に送る。


def _recorded(tmp_path, host="ssh9.vast.ai", port=50001):
    directory = tmp_path / "runs" / "full-s1"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({
        "run_id": "full-s1", "remote_dir": "/workspace/runs/full-s1", "pid": 7,
        "host": host, "port": port, "device": "gpu", "policy_name": "walk"}), encoding="utf-8")
    return directory


def _addresses(runner):
    return {(call[call.index("-p") + 1] if "-p" in call else None,
             next(item for item in call if "@" in item).split("@")[1].split(":")[0])
            for call in runner.calls if call[0] == "ssh"}


def test_status_は記録された接続先に送る(tmp_path):
    _recorded(tmp_path)
    runner = FakeRunner(lambda argv: ok(argv, json.dumps({"state": "training", "iteration": 3})))
    session = remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs")
    session.progress("full-s1")
    session.alive("full-s1")
    assert _addresses(runner) == {("50001", "ssh9.vast.ai")}


def test_stop_は記録された接続先に送る(tmp_path):
    _recorded(tmp_path)
    runner = FakeRunner(lambda argv: ok(argv, "gone"))
    session = remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs", sleep=lambda s: None)
    assert session.stop("full-s1") == "gone"
    assert _addresses(runner) == {("50001", "ssh9.vast.ai")}


def test_pull_は記録された接続先から受け取る(tmp_path):
    _recorded(tmp_path)
    runner = FakeRunner()
    session = remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs")
    with pytest.raises(remote.VerificationError):
        session.pull("full-s1")  # 中身が無いので検証は落ちる。送り先だけを見る
    rsync = [call for call in runner.calls if call[0] == "rsync"]
    assert rsync and all("root@ssh9.vast.ai:/workspace/runs/full-s1/" in call for call in rsync)
    assert all("-p 50001" in call[call.index("-e") + 1] for call in rsync)
    assert _addresses(runner) <= {("50001", "ssh9.vast.ai")}


def test_接続先の記録が無い古い回は設定の接続先を使う(tmp_path):
    directory = tmp_path / "runs" / "old"
    directory.mkdir(parents=True)
    (directory / "run.json").write_text(json.dumps({"run_id": "old", "remote_dir": "/w/old"}))
    runner = FakeRunner(lambda argv: ok(argv, json.dumps({"state": "done"})))
    remote.Session(TARGET, runner=runner, local_root=tmp_path / "runs").progress("old")
    assert _addresses(runner) == {("41234", "ssh5.vast.ai")}


def test_回収した方策が一つも無ければ成功と報告しない(tmp_path):
    with pytest.raises(remote.VerificationError):
        remote.verify_run(tmp_path, None, None, {}, policy_name="walk")
