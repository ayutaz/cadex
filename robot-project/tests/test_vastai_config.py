"""vast.ai 学習基盤: 学習設定ファイルと接続設定の検証。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vastai import config as cfg

ROOT = Path(__file__).resolve().parents[1]
TRAINER = ROOT.parent / "training" / "cadex_train.py"
CONFIGS = sorted((ROOT / "configs").glob("train_*.json"))


def _write(tmp_path: Path, data: dict) -> Path:
    path = tmp_path / "train_x.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _minimal(**over) -> dict:
    data = {
        "name": "x",
        "bundle": "robot/mjcf/base/walk-task.json",
        "device": "gpu",
        "trainer": {"seed": 0, "iterations": 4, "envs": 64},
    }
    data.update(over)
    return data


def test_smoke_と_full_の両方がある():
    names = {path.stem for path in CONFIGS}
    assert {"train_smoke", "train_full"} <= names


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_同梱の設定は読み込める(path):
    loaded = cfg.load_config(path)
    assert loaded["device"] in ("gpu", "cpu")
    assert loaded["env"]["JAX_PLATFORMS"] in ("cuda", "cpu")
    assert loaded["trainer"]["iterations"] > 0


@pytest.mark.skipif(not TRAINER.exists(), reason="Cadex の training/cadex_train.py が無い")
@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
@pytest.mark.parametrize("device", ["gpu", "cpu"])
def test_同梱の設定のフラグは本物の_cadex_train_が受け付ける(path, device):
    loaded = cfg.load_config(path, device=device)
    argv = cfg.trainer_argv(loaded["trainer"])
    cfg.validate_with_trainer(argv, TRAINER)  # 例外が出なければよい


def test_smoke_は_full_より小さい():
    smoke = cfg.load_config(ROOT / "configs" / "train_smoke.json")
    full = cfg.load_config(ROOT / "configs" / "train_full.json")
    assert smoke["trainer"]["iterations"] < full["trainer"]["iterations"]
    assert smoke["trainer"]["envs"] <= full["trainer"]["envs"]
    assert full["trainer"].get("checkpoint_every", 0) > 0


def test_trainer_argv_の変換規則():
    argv = cfg.trainer_argv({
        "iterations": 4,
        "terrain_curriculum": True,
        "stop_on_collapse": False,
        "hidden": [64, 32],
        "initial_std": None,
        "learning_rate": 0.0003,
    })
    assert argv == [
        "--iterations", "4",
        "--terrain-curriculum",
        "--hidden", "64", "32",
        "--learning-rate", "0.0003",
    ]


@pytest.mark.parametrize("key", ["out", "progress", "init_from", "bundle", "view"])
def test_ツールが管理するフラグは設定に書けない(key):
    with pytest.raises(cfg.ConfigError, match=key):
        cfg.trainer_argv({key: "x"})


@pytest.mark.skipif(not TRAINER.exists(), reason="Cadex の training/cadex_train.py が無い")
def test_存在しないフラグは本物のパーサで弾かれる():
    with pytest.raises(cfg.ConfigError, match="iteration"):
        cfg.validate_with_trainer(["--iteration", "3"], TRAINER)


def test_device_の切り替えで_JAX_PLATFORMS_と上書きが効く(tmp_path):
    path = _write(tmp_path, _minimal(
        trainer={"seed": 0, "iterations": 4, "envs": 4096, "minibatches": 32},
        device_overrides={"cpu": {"trainer": {"envs": 64, "minibatches": 8},
                                  "env": {"XLA_FLAGS": "--xla_cpu_multi_thread_eigen=true"}}},
    ))
    gpu = cfg.load_config(path)
    cpu = cfg.load_config(path, device="cpu")
    assert gpu["env"]["JAX_PLATFORMS"] == "cuda"
    assert gpu["trainer"]["envs"] == 4096
    assert cpu["env"]["JAX_PLATFORMS"] == "cpu"
    assert cpu["env"]["XLA_FLAGS"] == "--xla_cpu_multi_thread_eigen=true"
    assert cpu["trainer"]["envs"] == 64
    assert cpu["trainer"]["minibatches"] == 8
    assert cpu["trainer"]["iterations"] == 4  # 上書きしないキーは残る


def test_不正な_device_は弾く(tmp_path):
    with pytest.raises(cfg.ConfigError, match="device"):
        cfg.load_config(_write(tmp_path, _minimal(device="tpu")))


@pytest.mark.parametrize("missing", ["name", "bundle", "trainer"])
def test_必須キーが無ければ弾く(tmp_path, missing):
    data = _minimal()
    del data[missing]
    with pytest.raises(cfg.ConfigError, match=missing):
        cfg.load_config(_write(tmp_path, data))


def test_envs_が_minibatches_で割り切れなければ弾く(tmp_path):
    with pytest.raises(cfg.ConfigError, match="minibatches"):
        cfg.load_config(_write(tmp_path, _minimal(
            trainer={"iterations": 4, "envs": 100, "minibatches": 32})))


def test_env_ファイルは字面どおりに読む(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# コメント\n"
        "VAST_SSH_HOST=ssh5.vast.ai\n"
        "VAST_SSH_PORT='41234'\n"
        'VAST_SSH_USER="root"  # 末尾コメント\n'
        "VAST_SSH_KEY=~/.ssh/vast\n"
        "OTHER=$(rm -rf /)\n",
        encoding="utf-8",
    )
    values = cfg.read_env_file(env)
    assert values["VAST_SSH_HOST"] == "ssh5.vast.ai"
    assert values["VAST_SSH_PORT"] == "41234"
    assert values["VAST_SSH_USER"] == "root"
    assert values["VAST_SSH_KEY"] == "~/.ssh/vast"
    assert values["OTHER"] == "$(rm -rf /)"  # 実行しない


def test_ssh_設定は環境変数が_env_ファイルに勝つ(tmp_path):
    env = tmp_path / ".env"
    env.write_text("VAST_SSH_HOST=from-file\nVAST_SSH_PORT=22\n", encoding="utf-8")
    target = cfg.ssh_settings({"VAST_SSH_HOST": "from-env", "VAST_SSH_PORT": "41234"}, env)
    assert target.host == "from-env"
    assert target.port == 41234
    assert target.user == "root"  # 既定


def test_ssh_ホストが無ければ説明つきで弾く(tmp_path):
    with pytest.raises(cfg.ConfigError, match="VAST_SSH_HOST"):
        cfg.ssh_settings({}, tmp_path / "missing.env")


def test_api_key_は環境変数か_env_から読み無ければ空():
    assert cfg.api_key({"VAST_API_KEY": " abc "}, None) == "abc"
    assert cfg.api_key({}, None) == ""


#: 一反復の時間を決める値。smoke の実測を本番の見積りに使うので、そろえる。
TIMING_KEYS = ("envs", "unroll", "epochs", "minibatches", "hidden")


def test_smoke_は本番と同じ大きさの一反復で時間を測る():
    smoke = cfg.load_config(ROOT / "configs" / "train_smoke.json")
    full = cfg.load_config(ROOT / "configs" / "train_full.json")
    for key in TIMING_KEYS:
        assert smoke["trainer"].get(key) == full["trainer"].get(key), key
    assert smoke["offer"]["gpu_name"] == full["offer"]["gpu_name"]  # 同じ GPU で測る


#: 基準機体（train_full.json）と比べる派生機体の本番設定。機体（bundle）以外は同じにする。
VARIANT_CONFIGS = {"leg120": "train_full_leg120.json", "mini": "train_full_mini.json"}


@pytest.mark.parametrize("variant", sorted(VARIANT_CONFIGS))
def test_派生機体の本番設定は基準機体と機体だけが違う(variant):
    base = json.loads((ROOT / "configs" / "train_full.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / VARIANT_CONFIGS[variant]).read_text(encoding="utf-8"))
    assert other["bundle"] == f"robot/mjcf/{variant}/cut/walk-task.json"
    assert (ROOT / other["bundle"]).exists()
    assert base["bundle"] == "robot/mjcf/base/cut/walk-task.json"
    assert other["name"] == f"full-{variant}"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in base.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in base["trainer"].items() if k != "label"}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in base["instance"].items() if k != "label"}
    assert other["instance"]["label"] == f"q8-full-{variant}"


@pytest.mark.parametrize("variant", sorted(VARIANT_CONFIGS))
def test_比べる機体のタスクは報酬と成功条件が同じ(variant):
    # 形状の差だけを比べるため、機体のモデル以外（報酬・成功条件・指令）は同じ。
    # success.scale は機体の寸法と質量から決まる値（腰の高さ・重心の高さ・質量）なので除く。
    def task(name):
        data = json.loads((ROOT / "robot" / "mjcf" / name / "cut" / "walk-task.json")
                          .read_text(encoding="utf-8"))
        out = {k: data[k] for k in ("reward", "success", "commands", "episode", "termination")}
        out["success"] = {k: v for k, v in out["success"].items() if k != "scale"}
        return out
    assert task("base") == task(variant)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.stem)
def test_イメージを速く取れるよう回線の速い台だけを探す(path):
    # 学習イメージは数 GB。200 Mbps 級の台では取得に 30 分以上かかり、終わらない台もあった（2026-11-17）
    offer = json.loads(path.read_text(encoding="utf-8"))["offer"]
    assert offer["min_inet_down_mbps"] >= 1000
