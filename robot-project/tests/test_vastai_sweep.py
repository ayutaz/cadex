"""複数のインスタンスで、学習の待ち行列を回す（python -m vastai sweep）。

2026-11-17 の GPU 学習（二機体 × 学習シード5回）で手で行ったことを、テスト付きのコマンドにしたもの。
各台は、起動を待ち、GPU を確かめ、共有の待ち行列から1回ずつ取って 起動 → 監視 → 回収 する。
"""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from vastai import sweep

ROOT = Path(__file__).resolve().parents[1]
FULL = str(ROOT / "configs" / "train_full.json")
LEG = str(ROOT / "configs" / "train_full_leg120.json")


def _row(iid, *, label="cadex-q8-full-20261117-1956", status="running", host=None):
    return {"id": iid, "label": label, "actual_status": status,
            "direct_ssh": (host or f"h{iid}", 40000 + iid) if status == "running" else None,
            "proxy_ssh": None}


class FakeApi:
    def __init__(self, rows):
        self.rows = {r["id"]: r for r in rows}
        self.destroyed = []
        self.lock = threading.Lock()

    def list_instances(self):
        with self.lock:
            return [dict(r) for r in self.rows.values()]

    def destroy(self, iid):
        with self.lock:
            self.destroyed.append(iid)
            self.rows.pop(iid, None)


class FakeSession:
    """接続先ごとの振る舞い: gpu=False なら check が cpu、fail_runs の回は学習が failed になる。"""

    log: list = []
    lock = threading.Lock()

    def __init__(self, target, *, gpu=True, fail_runs=(), vanish_runs=(), refuse_checks=0):
        self.target = target
        self.gpu = gpu
        self.refuse_checks = refuse_checks
        self.fail_runs = set(fail_runs)
        self.vanish_runs = set(vanish_runs)
        self.reads = {}

    def _note(self, *event):
        with FakeSession.lock:
            FakeSession.log.append((self.target.host, *event))

    def check(self, trainer_root, python):
        self._note("check")
        if self.refuse_checks:
            self.refuse_checks -= 1
            raise sweep.remote.RemoteError("インスタンスの確認に失敗しました（終了コード 255）: "
                                           "ssh: connect to host h1 port 40001: Connection refused")
        return "backend gpu\n" if self.gpu else "backend cpu\n"

    def launch(self, config, *, run_id, extra_argv, dry_run=False):
        self._note("launch", run_id, tuple(extra_argv), config["name"])
        return {"run_id": run_id}

    def progress(self, run_id):
        n = self.reads[run_id] = self.reads.get(run_id, 0) + 1
        if run_id in self.fail_runs:
            return {"state": "failed", "error": "RuntimeError: cuda"}
        if run_id in self.vanish_runs and n == 1:
            return {"state": "training", "iteration": 999}
        return {"state": "done" if n >= 2 or run_id in self.vanish_runs else "training"}

    def alive(self, run_id):
        return run_id not in self.vanish_runs

    def pull(self, run_id):
        self._note("pull", run_id)
        return [{"file": "walk.cxpolicy"}]


@pytest.fixture(autouse=True)
def _clear_log():
    FakeSession.log = []


class Clock:
    def __init__(self):
        self.now = 0.0
        self.lock = threading.Lock()

    def __call__(self):
        with self.lock:
            return self.now

    def sleep(self, seconds):
        with self.lock:
            self.now += max(seconds, 1.0)


def _sweep(items, api, factory, **kw):
    clock = Clock()
    return sweep.Sweep(items, api=api, session_factory=factory, sleep=clock.sleep, clock=clock,
                       log=lambda *a: None, poll_s=30, **kw)


def _events(kind):
    return [e for e in FakeSession.log if e[1] == kind]


def test_parse_seeds():
    assert sweep.parse_seeds("0-4") == [0, 1, 2, 3, 4]
    assert sweep.parse_seeds("0,2,5") == [0, 2, 5]
    assert sweep.parse_seeds("3") == [3]
    with pytest.raises(ValueError):
        sweep.parse_seeds("4-1")


def test_plan_names_each_run_by_config_and_seed():
    items = sweep.plan([FULL, LEG], [0, 1])
    assert [i.run_id for i in items] == ["full-s0", "full-s1", "full-leg120-s0", "full-leg120-s1"]
    assert [i.seed for i in items] == [0, 1, 0, 1]


def test_every_run_is_trained_once_with_its_seed_and_pulled():
    items = sweep.plan([FULL, LEG], [0, 1])
    api = FakeApi([_row(1), _row(2)])
    result = _sweep(items, api, lambda t: FakeSession(t), destroy=True).run([1, 2])
    launched = {e[2]: e[3] for e in _events("launch")}
    assert launched == {"full-s0": ("--seed", "0"), "full-s1": ("--seed", "1"),
                        "full-leg120-s0": ("--seed", "0"), "full-leg120-s1": ("--seed", "1")}
    assert sorted(e[2] for e in _events("pull")) == sorted(launched)
    assert sorted(result["done"]) == sorted(launched) and result["failed"] == [] and result["left"] == []
    assert sorted(api.destroyed) == [1, 2]   # 列が空になった台は破棄


def test_an_instance_without_gpu_is_destroyed_and_the_others_do_its_share():
    items = sweep.plan([FULL], [0, 1, 2])
    api = FakeApi([_row(1, host="bad"), _row(2)])
    factory = lambda t: FakeSession(t, gpu=t.host != "bad")  # noqa: E731
    result = _sweep(items, api, factory, destroy=True).run([1, 2])
    assert {e[0] for e in _events("launch")} == {"h2"}
    assert sorted(result["done"]) == ["full-s0", "full-s1", "full-s2"]
    assert "GPU" in result["instances"][1]
    assert 1 in api.destroyed


def test_a_failed_run_goes_back_to_the_queue_and_its_instance_is_retired():
    items = sweep.plan([FULL], [0, 1])
    api = FakeApi([_row(1, host="broken"), _row(2)])
    # 壊れた台ではどの回も失敗する。失敗した回は列に戻り、もう1台が回す
    factory = lambda t: FakeSession(t, fail_runs={"full-s0", "full-s1"} if t.host == "broken" else ())  # noqa: E731
    result = _sweep(items, api, factory, destroy=True).run([1, 2])
    assert sorted(result["done"]) == ["full-s0", "full-s1"]
    assert 1 in api.destroyed
    assert len([e for e in _events("launch") if e[0] == "broken"]) <= 1   # 失敗した台は退く


def test_a_run_that_fails_everywhere_stops_after_the_attempt_limit():
    items = sweep.plan([FULL], [0])
    api = FakeApi([_row(1), _row(2), _row(3)])
    result = _sweep(items, api, lambda t: FakeSession(t, fail_runs={"full-s0"}), destroy=True,
                    max_attempts=2).run([1, 2, 3])
    assert len(_events("launch")) == 2
    assert result["failed"] == ["full-s0"] and result["done"] == []


def test_instances_of_other_work_are_never_touched():
    items = sweep.plan([FULL], [0])
    api = FakeApi([_row(60809472, label="s2-k1-l40s"), _row(2)])
    result = _sweep(items, api, lambda t: FakeSession(t), destroy=True).run([60809472, 2])
    assert all(e[0] != "h60809472" for e in FakeSession.log)   # check も launch もしない
    assert 60809472 not in api.destroyed
    assert "触らない" in result["instances"][60809472]
    assert result["done"] == ["full-s0"]


def test_an_instance_that_never_boots_is_destroyed_after_the_timeout():
    items = sweep.plan([FULL], [0])
    api = FakeApi([_row(1, status="loading"), _row(2)])
    result = _sweep(items, api, lambda t: FakeSession(t), destroy=True, boot_timeout_s=600).run([1, 2])
    assert 1 in api.destroyed
    assert result["done"] == ["full-s0"]


def test_finishing_just_before_the_process_check_counts_as_done():
    items = sweep.plan([FULL], [0])
    api = FakeApi([_row(1)])
    result = _sweep(items, api, lambda t: FakeSession(t, vanish_runs={"full-s0"}), destroy=True).run([1])
    assert result["done"] == ["full-s0"]


def test_without_destroy_nothing_is_destroyed():
    items = sweep.plan([FULL], [0, 1])
    api = FakeApi([_row(1, host="bad"), _row(2)])
    factory = lambda t: FakeSession(t, gpu=t.host != "bad")  # noqa: E731
    result = _sweep(items, api, factory, destroy=False).run([1, 2])
    assert api.destroyed == []
    assert sorted(result["done"]) == ["full-s0", "full-s1"]


def test_ssh_refused_just_after_boot_is_retried_instead_of_destroying_the_instance():
    # running になった直後は sshd がまだ受け付けないことがある。1回の拒否で破棄していた（2026-11-17）
    items = sweep.plan([FULL], [0])
    api = FakeApi([_row(1)])
    result = _sweep(items, api, lambda t: FakeSession(t, refuse_checks=3), destroy=True).run([1])
    assert result["done"] == ["full-s0"]
    assert [e[0] for e in _events("launch")] == ["h1"]


def test_an_instance_that_keeps_refusing_ssh_is_retired_after_the_grace():
    items = sweep.plan([FULL], [0])
    api = FakeApi([_row(1, host="dead"), _row(2)])
    factory = lambda t: FakeSession(t, refuse_checks=10_000 if t.host == "dead" else 0)  # noqa: E731
    result = _sweep(items, api, factory, destroy=True).run([1, 2])
    assert 1 in api.destroyed and "確認に失敗" in result["instances"][1]
    assert result["done"] == ["full-s0"]
    assert len([e for e in _events("check") if e[0] == "dead"]) < 100   # 猶予のあとは諦める
