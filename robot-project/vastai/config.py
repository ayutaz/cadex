"""学習設定ファイル（configs/train_*.json）と接続設定（.env / 環境変数）。

学習設定の ``trainer`` は Cadex の ``training/cadex_train.py`` のフラグそのもの
（キーの ``_`` を ``-`` に直して ``--`` を付ける）。推測でフラグ表を持たず、
:func:`validate_with_trainer` が本物の ``cadex_train.arguments()`` に通して確かめる。
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib.util
import io
import json
import os
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

#: このツールが自分で決めるフラグ。設定ファイルに書かれていたら弾く。
#: bundle/out/progress は送り先のパスで決まり、init_from 系は resume が付ける。
RESERVED_KEYS = frozenset({
    "bundle", "out", "progress", "view", "init_from", "init_from_task_change",
    "init_from_base", "init_from_parent_task", "family", "teacher",
})

DEVICES = {"gpu": "cuda", "cpu": "cpu"}

#: リモートの既定の配置。Dockerfile と同じ値。
DEFAULT_REMOTE = {
    "work_dir": "/workspace/runs",
    "trainer_root": "/opt/cadex",
    "python": "/opt/venv/bin/python",
}


class ConfigError(ValueError):
    """設定の誤り。メッセージはそのまま利用者に見せる。"""


# --- 学習設定 ---------------------------------------------------------------

def _merge(base: dict, override: Mapping) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: str | Path, device: str | None = None) -> dict[str, Any]:
    """設定を読み、``device``（省略時は設定の値）の上書きを当てて返す。

    返す辞書には ``env``（``JAX_PLATFORMS`` を含むリモートの環境変数）、
    ``remote``（既定値で補った配置）、``path`` が入る。
    """

    path = Path(path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ConfigError(f"設定ファイルがありません: {path}") from None
    except json.JSONDecodeError as error:
        raise ConfigError(f"{path} は JSON として読めません: {error}") from None
    if not isinstance(raw, dict):
        raise ConfigError(f"{path} の中身はオブジェクトでなければなりません")
    for key in ("name", "bundle", "trainer"):
        if key not in raw:
            raise ConfigError(f"{path}: 必須キー {key!r} がありません")
    if not isinstance(raw["trainer"], dict):
        raise ConfigError(f"{path}: 'trainer' はオブジェクトでなければなりません")

    chosen = device or raw.get("device", "gpu")
    if chosen not in DEVICES:
        raise ConfigError(f"device は {sorted(DEVICES)} のどれかです（{chosen!r} が指定された）")

    loaded = copy.deepcopy(raw)
    overrides = raw.get("device_overrides", {}).get(chosen, {})
    loaded = _merge(loaded, overrides)
    loaded["device"] = chosen
    loaded["env"] = {**loaded.get("env", {}), "JAX_PLATFORMS": DEVICES[chosen]}
    # device_overrides.<device>.env は JAX_PLATFORMS 以外を足せる
    loaded["env"].update({k: v for k, v in overrides.get("env", {}).items()
                          if k != "JAX_PLATFORMS"})
    loaded["remote"] = {**DEFAULT_REMOTE, **loaded.get("remote", {})}
    loaded.setdefault("resume", {})
    loaded.setdefault("policy_name", Path(str(raw["bundle"])).name.removesuffix(".json")
                      .removesuffix("-task"))
    loaded["path"] = str(path)
    _check_trainer(loaded["trainer"], path)
    return loaded


def _check_trainer(trainer: Mapping[str, Any], path: Path) -> None:
    trainer_argv(trainer)  # 予約キーを弾く
    iterations = trainer.get("iterations", 200)
    if not isinstance(iterations, int) or iterations <= 0:
        raise ConfigError(f"{path}: trainer.iterations は正の整数です")
    envs, minibatches = trainer.get("envs", 256), trainer.get("minibatches")
    if minibatches and envs % minibatches:
        raise ConfigError(
            f"{path}: trainer.envs ({envs}) が trainer.minibatches ({minibatches}) "
            "で割り切れません（cadex_train.py の --minibatches の制約）")


def trainer_argv(trainer: Mapping[str, Any]) -> list[str]:
    """``{"learning_rate": 3e-4, "terrain_curriculum": True}`` → フラグの並び。

    True はフラグだけ、False と None は省く、リストは nargs として並べる。
    """

    argv: list[str] = []
    for key, value in trainer.items():
        if key in RESERVED_KEYS:
            raise ConfigError(
                f"trainer.{key} はこのツールが決めるので設定に書けません"
                "（送り先のパスと resume が付けます）")
        flag = "--" + key.replace("_", "-")
        if value is None or value is False:
            continue
        if value is True:
            argv.append(flag)
        elif isinstance(value, (list, tuple)):
            argv += [flag, *(str(item) for item in value)]
        else:
            argv += [flag, str(value)]
    return argv


def validate_with_trainer(argv: list[str], trainer_path: str | Path) -> None:
    """``argv`` を本物の ``cadex_train.arguments()`` に通す。弾かれたら ConfigError。

    cadex_train.py はモジュールの最上位で標準ライブラリしか import しないので、
    jax の無い手元でも読める。
    """

    trainer_path = Path(trainer_path)
    spec = importlib.util.spec_from_file_location("_cadex_train_for_validation", trainer_path)
    if spec is None or spec.loader is None:
        raise ConfigError(f"{trainer_path} を読み込めません")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # argparse は既定で前方一致の省略形を受け付ける（--iteration が --iterations に
    # なる）。設定の綴り違いを見逃さないよう、この読み込みだけ省略形を禁じる。
    class _Strict(argparse.ArgumentParser):
        def __init__(self, *args, **kwargs):
            kwargs["allow_abbrev"] = False
            super().__init__(*args, **kwargs)

    module.argparse = types.SimpleNamespace(**{**vars(argparse), "ArgumentParser": _Strict})
    stderr = io.StringIO()
    try:
        with contextlib.redirect_stderr(stderr):
            module.arguments(["bundle-task.json", "--out", "policy.cxpolicy", *argv])
    except SystemExit:
        message = stderr.getvalue().strip().splitlines()
        raise ConfigError(
            "cadex_train.py がフラグを受け付けません: "
            + (message[-1] if message else " ".join(argv))) from None


# --- 接続設定 ---------------------------------------------------------------

def read_env_file(path: str | Path | None) -> dict[str, str]:
    """``KEY=value`` を字面どおりに読む。シェルとして source しない（展開も実行もしない）。"""

    values: dict[str, str] = {}
    if path is None:
        return values
    path = Path(path)
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        name = name.strip().removeprefix("export ").strip()
        value = value.strip()
        if value[:1] in ("'", '"'):
            quote = value[0]
            end = value.find(quote, 1)
            value = value[1:end] if end > 0 else value[1:]
        elif " #" in value:
            value = value.split(" #", 1)[0].rstrip()
        values[name] = value
    return values


@dataclass(frozen=True)
class SSHTarget:
    host: str
    port: int = 22
    user: str = "root"
    key: str = ""

    @property
    def address(self) -> str:
        return f"{self.user}@{self.host}" if self.user else self.host

    def _options(self) -> list[str]:
        # vast.ai はインスタンスごとにホスト鍵が変わる。初見は受け入れ、変化は拒む。
        options = ["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                   "-p", str(self.port)]
        if self.key:
            options += ["-i", str(Path(self.key).expanduser())]
        return options

    def ssh_argv(self, command: str) -> list[str]:
        return ["ssh", *self._options(), self.address, command]

    def rsync_transport(self) -> str:
        return " ".join(["ssh", *self._options()])

    def rsync_up(self, sources: list[str], dest: str) -> list[str]:
        return ["rsync", "-rt", "-e", self.rsync_transport(),
                *[str(s) for s in sources], f"{self.address}:{dest}"]

    def rsync_down(self, source: str, dest: str, includes: list[str] | None = None) -> list[str]:
        filters: list[str] = []
        if includes:
            filters = [f"--include={pattern}" for pattern in includes] + ["--exclude=*"]
        return ["rsync", "-rt", "-e", self.rsync_transport(), *filters,
                f"{self.address}:{source}", str(dest)]


def _lookup(environ: Mapping[str, str], file_values: Mapping[str, str], name: str) -> str:
    if name in environ:
        return environ[name]
    return file_values.get(name, "")


def ssh_settings(environ: Mapping[str, str] | None = None,
                 env_file: str | Path | None = None) -> SSHTarget:
    """``VAST_SSH_HOST`` / ``VAST_SSH_PORT`` / ``VAST_SSH_USER`` / ``VAST_SSH_KEY``。

    環境変数が .env に勝つ。
    """

    environ = os.environ if environ is None else environ
    file_values = read_env_file(env_file)
    host = _lookup(environ, file_values, "VAST_SSH_HOST").strip()
    if not host:
        raise ConfigError(
            "VAST_SSH_HOST が設定されていません。robot-project/.env か環境変数に、"
            "`python -m vastai instances` で表示された接続先を書いてください"
            "（VAST_SSH_HOST / VAST_SSH_PORT / VAST_SSH_USER）。")
    port_text = _lookup(environ, file_values, "VAST_SSH_PORT").strip() or "22"
    try:
        port = int(port_text)
    except ValueError:
        raise ConfigError(f"VAST_SSH_PORT は整数です（{port_text!r}）") from None
    user = _lookup(environ, file_values, "VAST_SSH_USER").strip() or "root"
    key = _lookup(environ, file_values, "VAST_SSH_KEY").strip()
    return SSHTarget(host=host, port=port, user=user, key=key)


def api_key(environ: Mapping[str, str] | None = None,
            env_file: str | Path | None = None) -> str:
    environ = os.environ if environ is None else environ
    return _lookup(environ, read_env_file(env_file), "VAST_API_KEY").strip()
