"""複数のインスタンスで、学習の待ち行列を回す（``python -m vastai sweep``）。

各台は次の順に動く。台ごとに1本のスレッド。

1. 起動を待つ。label が ``cadex-q8-`` で始まらない台（他の作業のもの）には、確認も起動も破棄もしない。
   作った直後の台は一覧に出ないことがあるので、少し待つ。``boot_timeout_s`` までに起動しなければ退く。
2. GPU を確かめる（``check`` に ``backend gpu`` が出る）。出なければ退く。
3. 共有の待ち行列から1回ずつ取り、起動 → 監視 → 回収する。学習の乱数は ``--seed`` で渡す。
   学習に失敗した回は列に戻し（``max_attempts`` まで）、その台は退く（GPU が壊れていることがあった）。
4. 列が空で、他の台で走っている回も無くなったら退く。

``destroy`` が真なら、退いた台を破棄する（課金を止める）。破棄は ``offers.destroy_instance`` を通るので、
このプロジェクトの label でない台には送られない。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from . import config as cfg
from . import offers, remote

#: 作った直後の台が一覧に出るまで待つ秒数
LIST_GRACE_S = 300.0


@dataclass
class Item:
    config: str
    seed: int
    run_id: str
    attempts: int = 0


def parse_seeds(text: str) -> list[int]:
    """``0-4`` / ``0,2,5`` / ``3`` → 学習の乱数の並び。"""

    seeds: list[int] = []
    for part in str(text).split(","):
        part = part.strip()
        if "-" in part:
            low, high = (int(v) for v in part.split("-", 1))
            if high < low:
                raise ValueError(f"乱数の範囲が逆です: {part}")
            seeds.extend(range(low, high + 1))
        elif part:
            seeds.append(int(part))
    if not seeds:
        raise ValueError("学習の乱数がありません")
    return seeds


def plan(configs: Iterable[str], seeds: Iterable[int],
         load: Callable[[str], Mapping[str, Any]] = cfg.load_config) -> list[Item]:
    """設定 × 乱数 → 回の並び。run_id は ``<設定の name>-s<乱数>``。"""

    seeds = list(seeds)
    items = []
    for path in configs:
        name = str(load(path)["name"])
        items.extend(Item(config=str(path), seed=int(s), run_id=f"{name}-s{int(s)}") for s in seeds)
    return items


class VastApi:
    """本物の vast.ai（一覧と破棄）。"""

    def __init__(self, api_key: str, opener: Any = None):
        self.api_key = api_key
        self.opener = opener

    def list_instances(self) -> list[dict[str, Any]]:
        return offers.list_instances(api_key=self.api_key, opener=self.opener)

    def destroy(self, instance_id: int) -> None:
        result = offers.destroy_instance(instance_id, api_key=self.api_key, yes=True, opener=self.opener)
        if not result["executed"]:
            raise offers.VastError(result["reason"])


class Sweep:
    def __init__(self, items: Iterable[Item], *, api: Any, session_factory: Callable[[cfg.SSHTarget], Any],
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 log: Callable[..., None] = print, destroy: bool = False, boot_timeout_s: float = 1200.0,
                 poll_s: float = 30.0, max_attempts: int = 2, ssh_user: str = "root", ssh_key: str = "",
                 load: Callable[[str], Mapping[str, Any]] = cfg.load_config):
        self.queue = list(items)
        self.api = api
        self.session_factory = session_factory
        self.sleep = sleep
        self.clock = clock
        self.log = log
        self.destroy = destroy
        self.boot_timeout_s = boot_timeout_s
        self.poll_s = poll_s
        self.max_attempts = max_attempts
        self.ssh_user = ssh_user
        self.ssh_key = ssh_key
        self.load = load
        self.lock = threading.Lock()
        self.inflight = 0
        self.done: list[str] = []
        self.failed: list[str] = []
        self.instances: dict[int, str] = {}

    # --- 待ち行列 -----------------------------------------------------------
    def _finished(self) -> bool:
        with self.lock:
            return not self.queue and self.inflight == 0

    def _take(self) -> Item | None:
        """次の回。列が空でも他の台で走っている回があれば、戻ってくるかもしれないので待つ。"""

        while True:
            with self.lock:
                if self.queue:
                    self.inflight += 1
                    return self.queue.pop(0)
                if self.inflight == 0:
                    return None
            self.sleep(self.poll_s)

    def _settle(self, item: Item, ok: bool) -> None:
        with self.lock:
            self.inflight -= 1
            if ok:
                self.done.append(item.run_id)
                return
            item.attempts += 1
            if item.attempts < self.max_attempts:
                self.queue.append(item)
            else:
                self.failed.append(item.run_id)

    # --- 1台 ---------------------------------------------------------------
    def _retire(self, iid: int, reason: str) -> None:
        with self.lock:
            self.instances[iid] = reason
        self.log(f"[{iid}] 退く: {reason}")
        if self.destroy:
            try:
                self.api.destroy(iid)
                self.log(f"[{iid}] 破棄しました")
            except Exception as error:  # noqa: BLE001 — 破棄の失敗は伝えて続ける
                self.log(f"[{iid}] 破棄に失敗: {error}。コンソールで確かめてください")

    def _wait_boot(self, iid: int) -> cfg.SSHTarget | str:
        start = self.clock()
        while True:
            row = next((r for r in self.api.list_instances() if str(r.get("id")) == str(iid)), None)
            elapsed = self.clock() - start
            if row is not None and not offers.is_owned(row):
                return "other"
            if row is None and elapsed > LIST_GRACE_S:
                return "一覧に無い（破棄済み）"
            if row is not None and row.get("actual_status") == "running" and row.get("direct_ssh"):
                host, port = row["direct_ssh"]
                return cfg.SSHTarget(host=str(host), port=int(port), user=self.ssh_user, key=self.ssh_key)
            if self._finished():
                return "起動前に列が空になった"
            if elapsed > self.boot_timeout_s:
                return f"{self.boot_timeout_s / 60:.0f} 分たっても起動しない"
            self.sleep(self.poll_s)

    def _train(self, session: Any, item: Item) -> bool:
        try:
            loaded = self.load(item.config)
            session.launch(loaded, run_id=item.run_id, extra_argv=["--seed", str(item.seed)])
            while True:
                progress = session.progress(item.run_id)
                state = progress.get("state")
                if state not in ("done", "failed") and not session.alive(item.run_id):
                    progress = session.progress(item.run_id)  # 終わった直後に消えたのかもしれない
                    state = progress.get("state") if progress.get("state") in ("done", "failed") else "failed"
                if state == "done":
                    session.pull(item.run_id)
                    return True
                if state == "failed":
                    self.log(f"{item.run_id} が失敗: {progress.get('error') or 'プロセスが居ない'}")
                    return False
                self.sleep(self.poll_s)
        except (remote.RemoteError, remote.VerificationError, cfg.ConfigError) as error:
            self.log(f"{item.run_id} が失敗: {error}")
            return False

    def _worker(self, iid: int) -> None:
        target = self._wait_boot(iid)
        if target == "other":
            with self.lock:
                self.instances[iid] = "他の作業のインスタンス（label が cadex-q8- で始まらない）。触らない"
            self.log(f"[{iid}] 他の作業のインスタンスなので触らない")
            return
        if isinstance(target, str):
            self._retire(iid, target)
            return
        session = self.session_factory(target)
        with self.lock:
            sample = self.queue[0].config if self.queue else None
        where = {**cfg.DEFAULT_REMOTE, **((self.load(sample).get("remote") or {}) if sample else {})}
        try:
            report = session.check(where["trainer_root"], where["python"])
        except remote.RemoteError as error:
            self._retire(iid, f"確認に失敗: {error}")
            return
        if "backend gpu" not in report:
            self._retire(iid, "GPU が使えない（backend gpu が出ない）")
            return
        self.log(f"[{iid}] 起動済み・GPU 確認済み（{target.host}:{target.port}）")
        while True:
            item = self._take()
            if item is None:
                self._retire(iid, "列が空")
                return
            self.log(f"[{iid}] {item.run_id} を起動（乱数 {item.seed}）")
            ok = self._train(session, item)
            self._settle(item, ok)
            if ok:
                self.log(f"[{iid}] {item.run_id} 完了・回収済み")
            else:
                self._retire(iid, f"{item.run_id} の学習に失敗した台")
                return

    def run(self, instance_ids: Iterable[int]) -> dict[str, Any]:
        threads = [threading.Thread(target=self._worker, args=(int(i),), daemon=True) for i in instance_ids]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        with self.lock:
            return {"done": list(self.done), "failed": list(self.failed),
                    "left": [i.run_id for i in self.queue], "instances": dict(self.instances)}
