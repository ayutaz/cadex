"""vast.ai 学習基盤: コマンドラインの入口。課金操作が既定で走らないことを確かめる。"""

from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path


from vastai import cli

ROOT = Path(__file__).resolve().parents[1]
SMOKE = str(ROOT / "configs" / "train_smoke.json")
FULL = str(ROOT / "configs" / "train_full.json")


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, payload):
        self.payload = payload
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return FakeResponse(json.dumps(self.payload).encode())


class ForbiddenRunner:
    def __init__(self):
        self.calls = []

    def __call__(self, argv, input=None):
        self.calls.append(argv)
        raise AssertionError(f"実行してはいけない: {argv}")


OFFERS = {"offers": [{"id": 222, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.35,
                      "reliability2": 0.99, "cuda_max_good": 12.4, "gpu_ram": 24564,
                      "disk_space": 60.0, "inet_down": 500.0, "geolocation": "SE"}]}


def run(argv, *, environ=None, opener=None, runner=None):
    out = io.StringIO()
    code = cli.main(argv, environ=environ or {}, opener=opener, runner=runner,
                    stdout=out, env_file=Path("/nonexistent/.env"))
    return code, out.getvalue()


def test_offers_は料金表と推定総額を出す():
    opener = FakeOpener(OFFERS)
    code, out = run(["offers", "--config", SMOKE], opener=opener)
    assert code == 0
    assert "RTX 4090" in out and "0.350" in out
    assert "推定" in out
    assert [r.get_method() for r in opener.requests] == ["POST"]


def test_create_は既定で計画を出すだけ():
    opener = FakeOpener({"success": True, "new_contract": 1})
    code, out = run(["create", "--config", SMOKE, "--offer", "222"],
                    environ={"VAST_API_KEY": "k"}, opener=opener)
    assert code == 0
    assert "dry-run" in out
    assert opener.requests == []


def test_create_は_yes_でも_API_キーが無ければ送らない():
    opener = FakeOpener({"success": True, "new_contract": 1})
    code, out = run(["create", "--config", SMOKE, "--offer", "222", "--yes"], opener=opener)
    assert code == 2
    assert "VAST_API_KEY" in out
    assert opener.requests == []


def test_create_は_yes_と_API_キーで初めて作る():
    opener = FakeOpener({"success": True, "new_contract": 555})
    code, out = run(["create", "--config", SMOKE, "--offer", "222", "--yes",
                     "--image", "ghcr.io/example/cadex-train:1"],
                    environ={"VAST_API_KEY": "k"}, opener=opener)
    assert code == 0
    assert "555" in out
    assert [r.get_method() for r in opener.requests] == ["PUT"]


def test_destroy_も既定では送らない():
    opener = FakeOpener({"success": True})
    code, out = run(["destroy", "555"], environ={"VAST_API_KEY": "k"}, opener=opener)
    assert code == 0 and opener.requests == []
    assert "dry-run" in out


def test_launch_の_dry_run_は_ssh_も_rsync_も走らせない():
    runner = ForbiddenRunner()
    code, out = run(["launch", "--config", SMOKE, "--dry-run", "--run-id", "smoke-x"],
                    environ={"VAST_SSH_HOST": "ssh5.vast.ai", "VAST_SSH_PORT": "41234"},
                    runner=runner)
    assert code == 0, out
    assert runner.calls == []
    assert "rsync" in out and "nohup setsid" in out


def test_launch_は_device_cpu_で_JAX_PLATFORMS_cpu():
    code, out = run(["launch", "--config", SMOKE, "--dry-run", "--device", "cpu",
                     "--run-id", "smoke-x"],
                    environ={"VAST_SSH_HOST": "h"}, runner=ForbiddenRunner())
    assert code == 0, out
    assert "JAX_PLATFORMS=cpu" in out


def test_estimate_は時間と金額を出す():
    code, out = run(["estimate", "--config", FULL, "--seconds-per-iteration", "2.0",
                     "--dph", "0.40"])
    assert code == 0
    assert "時間" in out and "$" in out


def test_estimate_は_smoke_の進捗から一反復の秒数を取る(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"iteration": 9, "wall_time_s": 30.0}))
    code, out = run(["estimate", "--config", FULL, "--from-progress", str(progress),
                     "--dph", "0.40"])
    assert code == 0
    assert "3.00" in out  # 秒/反復


def test_metrics_は_CSV_を書く(tmp_path):
    progress = tmp_path / "progress.json"
    progress.write_text(json.dumps({"curve": [[0, 1.0], [1, 2.0]],
                                    "loss_curve": [[0, 0.5], [1, 0.4]]}))
    out_csv = tmp_path / "m.csv"
    code, _ = run(["metrics", str(progress), "--csv", str(out_csv),
                   "--tensorboard", str(tmp_path / "tb")])
    assert code == 0
    assert out_csv.read_text().splitlines()[1].startswith("0,1.0,0.5")
    assert list((tmp_path / "tb").glob("events.out.tfevents.*"))


def test_SSH_の接続先が無ければ説明して終わる():
    code, out = run(["status", "smoke-x"], runner=ForbiddenRunner())
    assert code == 2
    assert "VAST_SSH_HOST" in out


def test_python_m_vastai_help_が動く():
    result = subprocess.run(
        [__import__("sys").executable, "-m", "vastai", "--help"],
        cwd=ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 0
    assert "offers" in result.stdout and "resume" in result.stdout


def test_launch_の_seed_は学習の乱数だけを差し替える():
    code, out = run(["launch", "--config", FULL, "--dry-run", "--seed", "2",
                     "--run-id", "full-s2"],
                    environ={"VAST_SSH_HOST": "h"}, runner=ForbiddenRunner())
    assert code == 0, out
    assert "--seed 2" in out
    assert "--seed 0" not in out


def test_status_watch_は終わった直後にプロセスが消えても成功と扱う(tmp_path, monkeypatch):
    # 進捗を読んだ時点では training、その直後に学習が終わってプロセスが消えた。
    # 「落ちた」と誤って失敗の終了コードを返していた（2026-11-17、full-leg120-s1）
    record = tmp_path / "full-x"
    record.mkdir()
    (record / "run.json").write_text(json.dumps({"run_id": "full-x", "remote_dir": "/w/full-x",
                                              "host": "h", "port": 22}))
    monkeypatch.setattr(cli, "LOCAL_RUNS", tmp_path)
    monkeypatch.setattr(cli.time, "sleep", lambda s: None)
    states = iter(["training", "done"])

    def handler(argv):
        command = argv[-1]
        if "progress.json" in command:
            return subprocess.CompletedProcess(argv, 0, json.dumps(
                {"state": next(states), "iteration": 999, "total": 1000}), "")
        if "kill -0" in command:
            return subprocess.CompletedProcess(argv, 0, "no\n", "")
        return subprocess.CompletedProcess(argv, 0, "", "")

    calls = []

    def runner(argv, input=None):
        calls.append(argv)
        return handler(argv)

    code, out = run(["status", "full-x", "--watch", "--interval", "0"],
                    environ={"VAST_SSH_HOST": "h"}, runner=runner)
    assert code == 0, out
    assert "終わりました" in out


class BundlesAndAsks:
    """POST /bundles には OFFERS を、PUT には成功を返す。"""

    def __init__(self):
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if request.get_method() == "POST":
            return FakeResponse(json.dumps(OFFERS).encode())
        return FakeResponse(json.dumps({"success": True, "new_contract": 777}).encode())


def test_create_count_は既定で借りるはずのオファーを出すだけ():
    opener = BundlesAndAsks()
    code, out = run(["create", "--config", FULL, "--count", "1"], environ={"VAST_API_KEY": "k"},
                    opener=opener)
    assert code == 0, out
    assert "dry-run" in out and "222" in out and "RTX 4090" in out
    assert [r.get_method() for r in opener.requests] == ["POST"]


def test_create_count_は_yes_で検索の直後に借りる():
    opener = BundlesAndAsks()
    code, out = run(["create", "--config", FULL, "--count", "1", "--yes", "--exclude-machine", "5"],
                    environ={"VAST_API_KEY": "k"}, opener=opener)
    assert code == 0, out
    assert "777" in out and "destroy" in out
    assert [r.get_method() for r in opener.requests] == ["POST", "PUT"]


def test_create_は_offer_か_count_のどちらか一方():
    import pytest

    with pytest.raises(SystemExit):
        run(["create", "--config", FULL], environ={"VAST_API_KEY": "k"}, opener=BundlesAndAsks())
    with pytest.raises(SystemExit):
        run(["create", "--config", FULL, "--offer", "1", "--count", "2"], environ={"VAST_API_KEY": "k"},
            opener=BundlesAndAsks())
