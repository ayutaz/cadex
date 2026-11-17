"""借りたインスタンスでの学習: 送る、切り離して起動、進捗、回収と検証、再開、停止。

ここで呼ぶのは ``ssh`` と ``rsync`` だけ（subprocess）。インスタンスを作ったり
消したりはしない（それは offers.py）。

リモートの 1 回の学習は ``<work_dir>/<run_id>/`` に置く::

    walk-task.json  model-model.xml      送ったタスクのペア（同じディレクトリ）
    warm/<checkpoint>.cxpolicy           再開時の warm start 元
    train.pid                            学習プロセス自身の pid
    stdout.log                           cadex_train の stdout（最後の行が受領書 JSON）
    train.log                            cadex_train の stderr（反復ごとの報酬など）
    progress.json                        cadex_train が毎反復書き直す進捗
    walk.000100.cxpolicy ...             --checkpoint-every のチェックポイント
    walk.best.cxpolicy                   それまでの最良
    walk.cxpolicy                        最終ポリシー

手元には ``artifacts/runs/vastai/<run_id>/`` に同じものを回収し、``run.json``
（この学習の記録）と ``metrics.csv`` を足す。
"""

from __future__ import annotations

import hashlib
import json
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from . import config as cfg
from . import metrics

Runner = Callable[..., subprocess.CompletedProcess]

#: 回収するファイル（rsync の include）
PULL_INCLUDES = ["progress.json", "stdout.log", "train.log", "*.cxpolicy",
                 "*.rollout-trace.json", "view/", "view/***"]

NUMBERED = re.compile(r"^(?P<stem>.+)\.(?P<tag>\d{6})\.cxpolicy$")


class RemoteError(RuntimeError):
    """リモート操作の失敗。メッセージはそのまま利用者に見せる。"""


class VerificationError(RemoteError):
    """回収したファイルの sha256 が合わない、または想定と違う装置で学習した。"""


def run_subprocess(argv: list[str], input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(argv, input=input, capture_output=True, text=True, check=False)


def sha256_file(path: str | Path) -> str:
    """``shasum -a 256 <path>`` と同じ値。"""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --- タスクのペア -----------------------------------------------------------

@dataclass(frozen=True)
class Bundle:
    task: Path
    model: Path
    task_sha256: str
    model_sha256: str


def resolve_bundle(task_path: str | Path) -> Bundle:
    """cadex_train.load_bundle と同じ規則でモデルを探し、sha256 を確かめる。

    モデルはタスクの ``model.path``（プロジェクト根からの相対）で、タスクの
    親の親か、タスクと同じディレクトリに置かれている。リモートでは両方を
    同じディレクトリに並べる。
    """

    task_path = Path(task_path)
    try:
        task = json.loads(task_path.read_text(encoding="utf-8"))
        reference = task["model"]
        relative = Path(str(reference["path"]))
        expected = str(reference["sha256"])
    except FileNotFoundError:
        raise RemoteError(f"タスクがありません: {task_path}") from None
    except (KeyError, TypeError, ValueError) as error:
        raise RemoteError(f"{task_path} は学習タスク（model.path / model.sha256）として読めません: "
                          f"{error}") from None
    model = task_path.resolve().parent.parent / relative
    if not model.exists():
        model = task_path.parent / relative.name
    if not model.exists():
        raise RemoteError(f"タスクが参照するモデル {relative} が {task_path.parent} にありません")
    actual = sha256_file(model)
    if actual != expected:
        raise RemoteError(
            f"{model} の sha256 がタスクの記録と違います（{actual[:12]} ≠ {expected[:12]}）。"
            "タスクとモデルを同じエクスポートから取り直してください")
    return Bundle(task=task_path, model=model, task_sha256=sha256_file(task_path),
                  model_sha256=actual)


# --- コマンド ---------------------------------------------------------------

def launch_command(*, run_dir: str, python: str, trainer: str, task_name: str,
                   out_name: str, argv: list[str], env: Mapping[str, str]) -> str:
    """リモートで実行する 1 行。ssh が切れても学習が続くよう切り離す。

    * 内側の sh が自分の pid を train.pid に書いてから ``exec`` するので、
      記録される pid は学習プロセスそのもの（remote_train.sh と同じ理由）。
    * 3 つの記述子をすべて付け替えるので ssh はすぐ戻る。
    """

    q = shlex.quote
    variables = {**env, "CADEX_RUN_DRIVER": "vastai"}
    words = ["env", *[f"{key}={value}" for key, value in variables.items()],
             python, trainer, f"{run_dir}/{task_name}",
             "--out", f"{run_dir}/{out_name}",
             "--progress", f"{run_dir}/progress.json", *argv]
    inner = ("echo $$ > train.pid; exec " + " ".join(q(w) for w in words)
             + " > stdout.log 2> train.log")
    return (f"mkdir -p {q(run_dir)} && cd {q(run_dir)} && "
            f"nohup setsid sh -c {q(inner)} < /dev/null > /dev/null 2>&1 &")


def parse_receipt(text: str) -> dict[str, Any] | None:
    """cadex_train の stdout の最後の JSON 行（受領書）。warp の警告などが前に混ざる。"""

    for line in reversed(text.splitlines()):
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def parse_sha256sum(text: str) -> dict[str, str]:
    sums = {}
    for line in text.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts) == 2 and len(parts[0]) == 64:
            sums[parts[1].lstrip("*").strip()] = parts[0]
    return sums


# --- 検証 -------------------------------------------------------------------

def verify_run(directory: str | Path, progress: Mapping[str, Any] | None,
               receipt: Mapping[str, Any] | None, remote_sums: Mapping[str, str],
               *, policy_name: str, expect_device: str | None = "gpu") -> list[dict[str, Any]]:
    """回収したポリシーを sha256 で三者（手元・リモート・学習器の記録）突き合わせる。

    学習器の記録は、チェックポイントなら progress.json の ``checkpoints[].sha256``、
    最終ポリシーなら受領書の ``sha256``。一つでも合わなければ VerificationError。
    """

    directory = Path(directory)
    claimed: dict[str, str] = {}
    for entry in (progress or {}).get("checkpoints") or []:
        claimed[str(entry["path"])] = str(entry["sha256"])
    final = f"{policy_name}.cxpolicy"
    if receipt and receipt.get("sha256"):
        claimed[final] = str(receipt["sha256"])

    report, problems = [], []
    names = sorted(set(claimed) | {p.name for p in directory.glob("*.cxpolicy")})
    for name in names:
        path = directory / name
        local = sha256_file(path) if path.exists() else None
        row = {"file": name, "local": local, "remote": remote_sums.get(name),
               "trainer": claimed.get(name)}
        expected = {value for value in (row["remote"], row["trainer"]) if value}
        row["ok"] = local is not None and bool(expected) and expected == {local}
        if not row["ok"]:
            if local is None:
                problems.append(f"{name}: 手元にありません")
            elif not expected:
                problems.append(f"{name}: 照合先（リモートの sha256sum も学習器の記録も）がありません")
            else:
                problems.append(
                    f"{name}: 手元 {local[:12]} / リモート {str(row['remote'])[:12]} / "
                    f"学習器 {str(row['trainer'])[:12]} が一致しません")
        report.append(row)

    if receipt and expect_device and receipt.get("device") not in (None, expect_device):
        problems.append(
            f"学習は device '{receipt.get('device')}' で走りました（想定は '{expect_device}'）。"
            "GPU を使うはずなら CUDA 版 jax が読めていません: `python -m vastai check`")
    if problems:
        raise VerificationError("検証に失敗しました:\n  " + "\n  ".join(problems))
    return report


# --- 再開 -------------------------------------------------------------------

def latest_checkpoint(progress: Mapping[str, Any] | None, *, directory: str | Path | None = None,
                      policy_name: str | None = None) -> dict[str, Any] | None:
    """番号付きチェックポイントのうち一番進んだもの（``best`` は選ばない）。

    ``completed`` はそのチェックポイントまでに終わった反復数（タグの数字）。
    progress.json が無ければディレクトリのファイル名から選ぶ。
    """

    candidates: list[dict[str, Any]] = []
    for entry in (progress or {}).get("checkpoints") or []:
        tag = str(entry.get("tag", ""))
        if tag.isdigit():
            candidates.append({**entry, "completed": int(tag)})
    if not candidates and directory is not None:
        for path in Path(directory).glob("*.cxpolicy"):
            match = NUMBERED.match(path.name)
            if match and (policy_name is None or match["stem"] == policy_name):
                candidates.append({"tag": match["tag"], "path": path.name,
                                   "completed": int(match["tag"]), "sha256": None})
    if not candidates:
        return None
    return max(candidates, key=lambda entry: entry["completed"])


def resume_plan(progress: Mapping[str, Any] | None, resume: Mapping[str, Any], *,
                total: int | None = None, directory: str | Path | None = None,
                policy_name: str | None = None) -> dict[str, Any]:
    """最新チェックポイントから warm start する引数。

    cadex_train の --init-from が運ぶのは actor と探索幅と正規化だけで、critic と
    optimizer は新しく始まる（optimizer 状態まで戻る厳密な再開ではない）。
    そのため --critic-warmup（ADR-740）と --actor-ramp（ADR-984）を付ける。
    """

    progress = progress or {}
    if progress.get("state") == "done":
        raise RemoteError("この学習は終わっています（state=done）。再開するものがありません")
    latest = latest_checkpoint(progress, directory=directory, policy_name=policy_name)
    if latest is None:
        raise RemoteError("チェックポイントがありません（--checkpoint-every を 0 より大きく）。"
                          "最初からやり直してください")
    total = int(total if total is not None else progress.get("total") or 0)
    remaining = total - int(latest["completed"])
    if remaining <= 0:
        raise RemoteError(f"残りの反復がありません（全 {total}、済み {latest['completed']}）")
    argv: list[str] = []
    if int(resume.get("critic_warmup", 0) or 0) > 0:
        argv += ["--critic-warmup", str(int(resume["critic_warmup"]))]
    if int(resume.get("actor_ramp", 0) or 0) > 0:
        argv += ["--actor-ramp", str(int(resume["actor_ramp"]))]
    return {"checkpoint": latest["path"], "sha256": latest.get("sha256"),
            "completed": int(latest["completed"]), "iterations": remaining, "argv": argv}


# --- セッション -------------------------------------------------------------

def _replace_flag(argv: list[str], flag: str, value: str) -> list[str]:
    out, skip = [], False
    for item in argv:
        if skip:
            skip = False
            continue
        if item == flag:
            skip = True
            continue
        if item.startswith(flag + "="):
            continue
        out.append(item)
    return out + [flag, value]


class Session:
    """一つのインスタンス（SSH の接続先）に対する操作。"""

    def __init__(self, target: cfg.SSHTarget, *, runner: Runner = run_subprocess,
                 local_root: str | Path, sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.time):
        self.target = target
        self.runner = runner
        self.local_root = Path(local_root)
        self.sleep = sleep
        self.clock = clock

    # 下回り
    def _run(self, argv: list[str], *, what: str) -> subprocess.CompletedProcess:
        result = self.runner(argv)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip().splitlines()
            raise RemoteError(f"{what}に失敗しました（終了コード {result.returncode}）"
                              + (f": {detail[-1]}" if detail else ""))
        return result

    def ssh(self, command: str, *, what: str = "ssh") -> str:
        return self._run(self.target.ssh_argv(command), what=what).stdout

    def record_dir(self, run_id: str) -> Path:
        return self.local_root / run_id

    def load_record(self, run_id: str) -> dict[str, Any]:
        path = self.record_dir(run_id) / "run.json"
        if not path.exists():
            raise RemoteError(f"{path} がありません（launch した手元でなければ run_id が分かりません）")
        return json.loads(path.read_text(encoding="utf-8"))

    def _save_record(self, run_id: str, record: Mapping[str, Any]) -> None:
        directory = self.record_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / "run.json.partial"
        temporary.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(directory / "run.json")

    def new_run_id(self, name: str) -> str:
        return f"{name}-{time.strftime('%Y%m%d-%H%M%S', time.localtime(self.clock()))}"

    # 確認
    def check(self, trainer_root: str, python: str) -> str:
        """GPU、CUDA 版 jax、ピン留めの版、学習器の sha256 を一度に見る（読むだけ）。"""

        q = shlex.quote
        probe = (
            "import importlib.metadata as m\n"
            "for n in ('jax','jaxlib','jax-cuda12-plugin','mujoco','mujoco-mjx','numpy',"
            "'brax','flax','optax'):\n"
            "    try: print('pkg', n, m.version(n))\n"
            "    except Exception: print('pkg', n, '-')\n"
            "import jax; print('backend', jax.default_backend()); print('devices', jax.devices())\n")
        command = (
            "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader "
            "2>&1 | sed 's/^/gpu /'; "
            f"sha256sum {q(trainer_root + '/training/cadex_train.py')} | sed 's/^/trainer /'; "
            f"cat {q(trainer_root + '/TRAINER_SHA256')} 2>/dev/null | sed 's/^/image /'; "
            f"JAX_PLATFORMS=cuda {q(python)} -c {q(probe)} 2>&1 | tail -n 14; "
            "df -h /workspace 2>/dev/null | tail -n 1 | sed 's/^/disk /'")
        return self.ssh(command, what="インスタンスの確認")

    # 起動
    def launch(self, config: Mapping[str, Any], *, run_id: str | None = None,
               warm: Path | None = None, extra_argv: list[str] | None = None,
               iterations: int | None = None, resumed_from: Mapping[str, Any] | None = None,
               dry_run: bool = False) -> dict[str, Any]:
        bundle = resolve_bundle(config["bundle"])
        remote = {**cfg.DEFAULT_REMOTE, **config.get("remote", {})}
        run_id = run_id or self.new_run_id(str(config["name"]))
        run_dir = f"{remote['work_dir'].rstrip('/')}/{run_id}"
        policy_name = str(config.get("policy_name") or "policy")
        argv = cfg.trainer_argv(config["trainer"])
        extra = list(extra_argv or [])
        for index in range(0, len(extra) - 1, 2):  # (フラグ, 値) の組で上書き
            argv = _replace_flag(argv, extra[index], extra[index + 1])
        total = int(config["trainer"].get("iterations", 200))
        if iterations is not None:
            argv = _replace_flag(argv, "--iterations", str(int(iterations)))
            total = int(iterations)
        if warm is not None:
            argv += ["--init-from", f"{run_dir}/warm/{warm.name}"]

        commands: list[list[str]] = [
            self.target.ssh_argv(f"mkdir -p {shlex.quote(run_dir)}"),
            self.target.rsync_up([str(bundle.task), str(bundle.model)], run_dir + "/"),
        ]
        if warm is not None:
            commands.append(self.target.ssh_argv(f"mkdir -p {shlex.quote(run_dir + '/warm')}"))
            commands.append(self.target.rsync_up([str(warm)], run_dir + "/warm/"))
        launch = launch_command(
            run_dir=run_dir, python=remote["python"],
            trainer=f"{remote['trainer_root'].rstrip('/')}/training/cadex_train.py",
            task_name=bundle.task.name, out_name=f"{policy_name}.cxpolicy",
            argv=argv, env=config.get("env", {}))
        commands.append(self.target.ssh_argv(launch))
        result: dict[str, Any] = {"run_id": run_id, "remote_dir": run_dir, "commands": commands}
        if dry_run:
            return result

        for command in commands:
            self._run(command, what="送付と起動")
        pid = self._wait_pid(run_dir)
        record = {
            "run_id": run_id, "remote_dir": run_dir, "pid": pid,
            "host": self.target.host, "port": self.target.port,
            "config": config.get("path", ""), "config_name": config["name"],
            "device": config.get("device", "gpu"), "policy_name": policy_name,
            "total_iterations": total, "argv": argv,
            "task": str(bundle.task), "task_sha256": bundle.task_sha256,
            "model_sha256": bundle.model_sha256, "started_at": self.clock(),
        }
        if resumed_from:
            record["resumed_from"] = dict(resumed_from)
        self._save_record(run_id, record)
        result["pid"] = pid
        return result

    def _wait_pid(self, run_dir: str, attempts: int = 20) -> int:
        for _ in range(attempts):
            text = self.runner(self.target.ssh_argv(
                f"cat {shlex.quote(run_dir + '/train.pid')} 2>/dev/null")).stdout.strip()
            if text.isdigit():
                return int(text)
            self.sleep(1.0)
        raise RemoteError(f"起動したはずですが {run_dir}/train.pid が出ません。"
                          f"リモートで {run_dir}/train.log を見てください")

    # 進捗
    def progress(self, run_id: str) -> dict[str, Any]:
        record = self.load_record(run_id)
        text = self.ssh(f"cat {shlex.quote(record['remote_dir'] + '/progress.json')}",
                        what="progress.json の取得")
        progress = json.loads(text)
        directory = self.record_dir(run_id)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "progress.json").write_text(text, encoding="utf-8")
        return progress

    def alive(self, run_id: str) -> bool:
        record = self.load_record(run_id)
        pid_file = shlex.quote(record["remote_dir"] + "/train.pid")
        answer = self.runner(self.target.ssh_argv(
            f"kill -0 $(cat {pid_file}) 2>/dev/null && echo yes || echo no")).stdout.strip()
        return answer == "yes"

    # 回収
    def remote_sums(self, remote_dir: str) -> dict[str, str]:
        text = self.runner(self.target.ssh_argv(
            f"cd {shlex.quote(remote_dir)} && sha256sum *.cxpolicy 2>/dev/null")).stdout
        return parse_sha256sum(text)

    def pull(self, run_id: str, *, allow_cpu: bool = False) -> list[dict[str, Any]]:
        """チェックポイント・最終ポリシー・進捗・ログを回収し、sha256 で検証する。"""

        record = self.load_record(run_id)
        directory = self.record_dir(run_id)
        self._run(self.target.rsync_down(record["remote_dir"] + "/", str(directory) + "/",
                                         includes=PULL_INCLUDES), what="回収（rsync）")
        progress_path = directory / "progress.json"
        progress = metrics.load_progress(progress_path) if progress_path.exists() else None
        stdout = directory / "stdout.log"
        receipt = parse_receipt(stdout.read_text(encoding="utf-8")) if stdout.exists() else None
        if progress:
            metrics.write_csv(progress, directory / "metrics.csv")
        expect = None if allow_cpu else ("gpu" if record.get("device", "gpu") == "gpu" else None)
        return verify_run(directory, progress, receipt, self.remote_sums(record["remote_dir"]),
                          policy_name=record.get("policy_name", "policy"), expect_device=expect)

    # 再開
    def resume(self, source_run_id: str, config: Mapping[str, Any], *,
               run_id: str | None = None, pull_first: bool = True,
               dry_run: bool = False) -> dict[str, Any]:
        """``source_run_id`` の最新チェックポイントから新しい run として続ける。

        チェックポイントは手元の回収済みコピーを sha256 で確かめてから送るので、
        元のインスタンスが消えていても（``pull_first=False``）新しいインスタンスで続けられる。
        """

        record = self.load_record(source_run_id)
        directory = self.record_dir(source_run_id)
        if pull_first:
            try:
                self.pull(source_run_id, allow_cpu=True)
            except VerificationError:
                raise
            except RemoteError as error:
                raise RemoteError(f"{error}\n元のインスタンスに届かないなら --no-pull で、"
                                  "手元に回収済みのチェックポイントから再開できます") from None
        progress_path = directory / "progress.json"
        progress = metrics.load_progress(progress_path) if progress_path.exists() else None
        policy_name = record.get("policy_name") or config.get("policy_name")
        plan = resume_plan(progress, config.get("resume", {}),
                           total=record.get("total_iterations"), directory=directory,
                           policy_name=policy_name)
        checkpoint = directory / plan["checkpoint"]
        if not checkpoint.exists():
            raise VerificationError(f"{checkpoint} が手元にありません。先に pull してください")
        if plan["sha256"] and sha256_file(checkpoint) != plan["sha256"]:
            raise VerificationError(
                f"{checkpoint.name} の sha256 が progress.json の記録と違います。"
                "壊れたチェックポイントは送りません（pull し直してください）")
        resumed_from = {"run_id": source_run_id, "checkpoint": plan["checkpoint"],
                        "sha256": sha256_file(checkpoint), "completed": plan["completed"]}
        result = self.launch(config, run_id=run_id, warm=checkpoint, extra_argv=plan["argv"],
                             iterations=plan["iterations"], resumed_from=resumed_from,
                             dry_run=dry_run)
        result["resumed_from"] = resumed_from
        return result

    def stop(self, run_id: str) -> str:
        """TERM を送り、止まったことを確かめる。書かれたチェックポイントは残る。"""

        record = self.load_record(run_id)
        pid_file = shlex.quote(record["remote_dir"] + "/train.pid")
        answer = self.ssh(
            f"if [ -f {pid_file} ]; then kill -TERM $(cat {pid_file}) 2>/dev/null "
            "&& echo stopped || echo gone; else echo nopid; fi", what="停止").strip()
        if answer == "stopped":
            self.sleep(2.0)
            if self.alive(run_id):
                raise RemoteError(f"TERM を送りましたがまだ動いています。リモートで "
                                  f"kill -KILL $(cat {record['remote_dir']}/train.pid)")
        elif answer == "nopid":
            raise RemoteError(f"{record['remote_dir']}/train.pid がありません")
        return answer
