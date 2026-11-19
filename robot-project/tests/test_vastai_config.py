"""vast.ai 学習基盤: 学習設定ファイルと接続設定の検証。"""

from __future__ import annotations

import json
import math
import re
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


#: 小型機（mini）の再調整の案。機体の形は mini と同じで、タスクの指令速度と報酬の値だけを変える
#: （docs/experiment-results.md §7）。値は設計スクリプトの params（cmd_min/cmd_max/w_trot/w_swing/gait_lift）。
MINI_TUNING = {
    "mini-cmd": {"cmd": (30.0, 120.0), "w_trot": 0.5, "w_swing": 0.5, "gait_lift": 10.0},
    "mini-v3": {"cmd": (30.0, 120.0), "w_trot": 1.0, "w_swing": 1.5, "gait_lift": 10.0},
    "mini-lift": {"cmd": (30.0, 120.0), "w_trot": 1.0, "w_swing": 1.5, "gait_lift": 7.5},
}
#: 意図して変える項目を持つ報酬の label
TUNED_REWARDS = ("trot_contact", "swing_lift")


def _task(name: str) -> dict:
    return json.loads((ROOT / "robot" / "mjcf" / name / "cut" / "walk-task.json").read_text(encoding="utf-8"))


def _cmd_goal(goals: list[dict]) -> dict:
    return next(g for g in goals if g["name"] == "cmd")


def _lift_mm(task: dict) -> set[float]:
    expr = next(r for r in task["reward"] if r["label"] == "swing_lift")["expression"]
    return {float(v) for v in re.findall(r"tanh\(\(fc_\w+_z - [\d.]+\) / ([\d.]+)\)", expr)}


def _without_tuning(task: dict) -> dict:
    """意図して変えた項目（指令速度の範囲、二つの報酬の重みと足上げ高さ）を除いたタスク。"""
    out = json.loads(json.dumps({k: task[k] for k in
                                 ("model", "reward", "success", "commands", "episode", "termination",
                                  "randomisation", "reset_variation", "observations", "actions", "goal")}))
    out["success"].pop("scale", None)
    for goals in (out["goal"], out["success"]["goal"]):
        for key in ("low", "high", "nominal"):
            _cmd_goal(goals)[key] = None
    for reward in out["reward"]:
        if reward["label"] in TUNED_REWARDS:
            reward["weight"] = None
        if reward["label"] == "swing_lift":
            reward["expression"] = re.sub(r"/ [\d.]+\)", "/ LIFT)", reward["expression"])
    return out


@pytest.mark.parametrize("variant", sorted(MINI_TUNING))
def test_小型機の再調整の本番設定は_mini_と機体だけが違う(variant):
    mini = json.loads((ROOT / "configs" / "train_full_mini.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}.json").read_text(encoding="utf-8"))
    assert other["bundle"] == f"robot/mjcf/{variant}/cut/walk-task.json"
    assert (ROOT / other["bundle"]).exists()
    assert other["name"] == f"full-{variant}"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in mini.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in mini["trainer"].items() if k != "label"}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in mini["instance"].items() if k != "label"}
    assert other["instance"]["label"] == f"q8-full-{variant}"


@pytest.mark.parametrize("variant", sorted(MINI_TUNING))
def test_小型機の再調整のタスクは意図した項目だけが_mini_と違う(variant):
    mini, other = _task("mini"), _task(variant)
    # 機体のモデルは同じ（sha256 まで）。意図した項目を除けば、報酬・成功条件・観測も同じ。
    assert _without_tuning(other) == _without_tuning(mini)
    want = MINI_TUNING[variant]
    for goals in (other["goal"], other["success"]["goal"]):
        assert (_cmd_goal(goals)["low"], _cmd_goal(goals)["high"]) == want["cmd"]
    weights = {r["label"]: r["weight"] for r in other["reward"]}
    assert (weights["trot_contact"], weights["swing_lift"]) == (want["w_trot"], want["w_swing"])
    assert _lift_mm(other) == {want["gait_lift"]}
    # 元の mini は大きい機体と同じ値（指令 40〜160 mm/s、重み 0.5・0.5、足上げ 10 mm）
    assert (_cmd_goal(mini["goal"])["low"], _cmd_goal(mini["goal"])["high"]) == (40.0, 160.0)
    assert _lift_mm(mini) == {10.0}


#: mini-v3 から、トルクの罰を減らすための案。mini-v3 とは、歩容の周期（gait_period）か
#: トルクの罰の重み（w_effort/w_effort_rated）だけが違う（docs/experiment-results.md §8）。
MINI_V3_TUNING = {
    "mini-p6": {"period_s": 0.6, "w_effort": 0.5, "w_effort_rated": 0.3},
    "mini-p7": {"period_s": 0.7, "w_effort": 0.5, "w_effort_rated": 0.3},
    "mini-eff": {"period_s": 0.5, "w_effort": 0.5, "w_effort_rated": 1.0},
    # 上の3案の結果（§8）から、合格率の高い2つ（周期 0.7、罰 1.0）を合わせた案
    "mini-p7-eff": {"period_s": 0.7, "w_effort": 0.5, "w_effort_rated": 1.0},
}


def _clock_goal(goals: list[dict]) -> dict:
    return next(g for g in goals if g["name"] == "clock")


def _without_v3_tuning(task: dict) -> dict:
    """意図して変えた項目（位相クロックの周期、トルクの罰の重み）を除いたタスク。"""
    out = json.loads(json.dumps({k: task[k] for k in
                                 ("model", "reward", "success", "commands", "episode", "termination",
                                  "randomisation", "reset_variation", "observations", "actions", "goal")}))
    out["success"].pop("scale", None)
    for goals in (out["goal"], out["success"]["goal"]):
        for key in ("period_s", "radians_per_step"):
            _clock_goal(goals)[key] = None
    for reward in out["reward"]:
        if reward["label"] in ("effort", "effort_rated"):
            reward["weight"] = None
    return out


@pytest.mark.parametrize("variant", sorted(MINI_V3_TUNING))
def test_mini_v3_からの案の本番設定は_mini_v3_と機体だけが違う(variant):
    v3 = json.loads((ROOT / "configs" / "train_full_mini-v3.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}.json").read_text(encoding="utf-8"))
    assert other["bundle"] == f"robot/mjcf/{variant}/cut/walk-task.json"
    assert (ROOT / other["bundle"]).exists()
    assert other["name"] == f"full-{variant}"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in v3.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in v3["trainer"].items() if k != "label"}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in v3["instance"].items() if k != "label"}
    assert other["instance"]["label"] == f"q8-full-{variant}"


@pytest.mark.parametrize("variant", sorted(MINI_V3_TUNING))
def test_mini_v3_からの案のタスクは意図した項目だけが_mini_v3_と違う(variant):
    v3, other = _task("mini-v3"), _task(variant)
    # 機体のモデル（sha256 まで）、指令速度、成功条件、ほかの報酬は mini-v3 と同じ
    assert _without_v3_tuning(other) == _without_v3_tuning(v3)
    want = MINI_V3_TUNING[variant]
    for goals in (other["goal"], other["success"]["goal"]):
        clock = _clock_goal(goals)
        assert clock["period_s"] == pytest.approx(want["period_s"])
        # 50 Hz の制御で1周期に 2π 進む
        assert clock["radians_per_step"] == pytest.approx(2 * math.pi / (50 * want["period_s"]))
    weights = {r["label"]: r["weight"] for r in other["reward"]}
    assert (weights["effort"], weights["effort_rated"]) == (-want["w_effort"], -want["w_effort_rated"])
    assert _clock_goal(v3["goal"])["period_s"] == pytest.approx(0.5)
    assert {r["label"]: r["weight"] for r in v3["reward"]}["effort_rated"] == -0.3


#: 脚を組めるように直した mini-r2（leg_rev=2）に、mini の再調整で良かった報酬の案を載せた機体。
#: 機体のモデルは mini-r2 と同じで、タスク（指令・報酬・成功条件）は対応する mini の案と同じ
#: （docs/experiment-results.md §9）。
MINI_R2_TUNING = {"mini-r2-eff": "mini-eff", "mini-r2-p7-eff": "mini-p7-eff"}


def _task_without_model(task: dict) -> dict:
    """機体の形で決まる項目（モデル、成功判定の scale）と名前（label）を除いたタスク。"""
    out = json.loads(json.dumps({k: v for k, v in task.items() if k not in ("model", "label")}))
    out["success"].pop("scale", None)
    return out


@pytest.mark.parametrize("variant", sorted(MINI_R2_TUNING))
def test_mini_r2_の案の本番設定は対応する_mini_の案と機体だけが違う(variant):
    ref = json.loads((ROOT / "configs" / f"train_full_{MINI_R2_TUNING[variant]}.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}.json").read_text(encoding="utf-8"))
    assert other["bundle"] == f"robot/mjcf/{variant}/cut/walk-task.json"
    assert (ROOT / other["bundle"]).exists()
    assert other["name"] == f"full-{variant}"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in ref.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in ref["trainer"].items() if k != "label"}
    assert other["trainer"]["label"] == f"q8 vast.ai full {variant}"
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}
    assert other["instance"]["label"] == f"q8-full-{variant}"


@pytest.mark.parametrize("variant", sorted(MINI_R2_TUNING))
def test_mini_r2_の案は機体のモデルが_mini_r2_で報酬が対応する_mini_の案と同じ(variant):
    other, ref, r2 = _task(variant), _task(MINI_R2_TUNING[variant]), _task("mini-r2")
    # 機体のモデル（sha256 まで）は mini-r2 と同じで、mini の案とは違う（足先の幅が違う）
    assert other["model"] == r2["model"]
    assert other["model"]["sha256"] != ref["model"]["sha256"]
    # 指令・報酬・成功条件・観測・行動・乱択は、対応する mini の案と同じ
    assert _task_without_model(other) == _task_without_model(ref)
    # mini-r2 は既定の報酬（指令 40〜160 mm/s）のままなので、タスクは違う
    assert _task_without_model(other) != _task_without_model(r2)


#: 実機との差に強くする案（段階A-2）。機体のモデルとタスクは mini-r2-p7-eff に DR を足したもの
#: （q8/dr_bundle.py、docs/experiment-results.md §11）。学習設定は、名前・機体（bundle）と、
#: 学習器のランダム化のフラグ（足の摩擦とサーボのトルク）だけが mini-r2-p7-eff と違う。
DR_TUNING = {"mini-r2-p7-eff-dr": "mini-r2-p7-eff"}
DR_FLAGS = {"randomise_friction": [0.35, 1.25], "randomise_drive_torque": [0.8, 1.1]}


@pytest.mark.parametrize("variant", sorted(DR_TUNING))
def test_DR_の案の本番設定は元の案と機体とランダム化のフラグだけが違う(variant):
    ref = json.loads((ROOT / "configs" / f"train_full_{DR_TUNING[variant]}.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}.json").read_text(encoding="utf-8"))
    assert other["bundle"] == f"artifacts/{variant}/cut/walk-task.json"
    assert (ROOT / other["bundle"]).exists()
    assert other["name"] == f"full-{variant}"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in ref.items() if k not in ignore}
    trainer = {k: v for k, v in other["trainer"].items() if k not in ("label", *DR_FLAGS)}
    assert trainer == {k: v for k, v in ref["trainer"].items() if k != "label"}
    assert {k: other["trainer"][k] for k in DR_FLAGS} == DR_FLAGS
    assert other["trainer"]["label"] == f"q8 vast.ai full {variant}"
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}
    assert other["instance"]["label"] == f"q8-full-{variant}"


@pytest.mark.parametrize("variant", sorted(DR_TUNING))
def test_DR_の案のタスクは元の案に_DR_を足しただけ(variant):
    ref = _task(DR_TUNING[variant])
    other = json.loads((ROOT / "artifacts" / variant / "cut" / "walk-task.json").read_text(encoding="utf-8"))
    ignore = {"model", "randomisation", "disturbance"}
    assert {k: v for k, v in other.items() if k not in ignore} == {k: v for k, v in ref.items() if k not in ignore}
    # _task は機体の cut のタスク。DR の案はそれに足すだけで、元のランダム化を消さない
    assert other["randomisation"][:len(ref["randomisation"])] == ref["randomisation"]
    assert other["model"]["sha256"] != ref["model"]["sha256"]


def test_DR_の反復2000回の案は_DR_の案と反復回数と名前だけが違う():
    ref = json.loads((ROOT / "configs" / "train_full_mini-r2-p7-eff-dr.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / "train_full_mini-r2-p7-eff-dr-2k.json").read_text(encoding="utf-8"))
    assert other["name"] == "full-mini-r2-p7-eff-dr-2k"
    assert other["bundle"] == ref["bundle"]
    assert other["trainer"]["iterations"] == 2000
    assert other["trainer"]["label"] == "q8 vast.ai full mini-r2-p7-eff-dr-2k"
    assert other["instance"]["label"] == "q8-full-mini-r2-p7-eff-dr-2k"
    ignore = {"_comment", "name", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == {k: v for k, v in ref.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k not in ("label", "iterations")} == \
        {k: v for k, v in ref["trainer"].items() if k not in ("label", "iterations")}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}


#: しっぽ付き四足と3本足の案（計画 2026-11-18-tripod-and-tail、Task 5）。学習設定は mini-r2-p7-eff-dr と
#: 名前・機体（bundle）だけが違う。バンドルは各機体のカットモデルに q8/dr_bundle.py で同じ DR を足したもの
#: （artifacts/<案>/cut/）。値は、DR を足す元の機体（robot/mjcf/<機体>/cut/）。tri-r2-free は tri-r2
#: （gait_mode=0、歩き方を決めない）、tri-r2-wave は gait_mode=1（波）。
TAIL_TRI_DR = {
    "mini-r2-tail1-dr": "mini-r2-tail1",
    "mini-r2-tail2-dr": "mini-r2-tail2",
    "tri-r2-free-dr": "tri-r2",
    "tri-r2-wave-dr": "tri-r2-wave",
}
#: 3本足の直し（tri-r2-dyn: hc の足先を重心の後ろへ、速度を重心で、波の遊脚を短く）。学習設定は tri-r2-wave-dr と
#: entropy も違うので TAIL_TRI_DR には入れない（test_tri_r2_dyn_dr_は_wave_dr_と…）
#: 横に開く関節（leg_roll）を足した3本足（計画 2026-11-19-tripod-roll、Task 5）。-slip は同じ形に
#: 接地した足の滑りの罰（slip_cost=5）を足した機体。
TRI3_DR = {
    "tri3-r8-dr": "tri3-r8",
    "tri3-r8-slip-dr": "tri3-r8-slip",
    "tri3-r9-dr": "tri3-r9",
    "tri3-r9-slip-dr": "tri3-r9-slip",
}
DR_SOURCE = {**TAIL_TRI_DR, "tri-r2-dyn-dr": "tri-r2-dyn", **TRI3_DR}
#: バンドルを作り終えた案（機体の形は ef0451348 の版。tri-r2-dyn-dr は hc_foot_center などを足した版、
#: tri3-* は leg_roll を足した 695920bb1 の版）
TAIL_TRI_DR_BUILT = tuple(sorted(TAIL_TRI_DR)) + ("tri-r2-dyn-dr",) + tuple(sorted(TRI3_DR))


@pytest.mark.parametrize("variant", sorted(TAIL_TRI_DR))
def test_しっぽと3本足の設定は_dr_の設定と機体だけ違う(variant):
    ref = json.loads((ROOT / "configs" / "train_full_mini-r2-p7-eff-dr.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}.json").read_text(encoding="utf-8"))
    assert other["bundle"] == f"artifacts/{variant}/cut/walk-task.json"
    assert other["name"] == f"full-{variant}"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in ref.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in ref["trainer"].items() if k != "label"}
    assert other["trainer"]["label"] == f"q8 vast.ai full {variant}"
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}
    assert other["instance"]["label"] == f"q8-full-{variant}"


def test_tri_r2_dyn_dr_は_wave_dr_と機体_名前_ラベル_entropy_だけが違う():
    # 3本足の直し（後ろ足の足先を真ん中に、速さを重心で測る、波の遊脚を短く）。バンドルは tri-dyn が作る
    ref = json.loads((ROOT / "configs" / "train_full_tri-r2-wave-dr.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / "train_full_tri-r2-dyn-dr.json").read_text(encoding="utf-8"))
    assert other["name"] == "full-tri-r2-dyn-dr"
    assert other["bundle"] == "artifacts/tri-r2-dyn-dr/cut/walk-task.json"
    assert other["trainer"]["label"] == "q8 vast.ai full tri-r2-dyn-dr"
    assert other["instance"]["label"] == "q8-full-tri-r2-dyn-dr"
    assert other["trainer"]["entropy"] == 0.02
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in ref.items() if k not in ignore}
    skip = {"label", "entropy"}
    assert {k: v for k, v in other["trainer"].items() if k not in skip} == \
        {k: v for k, v in ref["trainer"].items() if k not in skip}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}


@pytest.mark.parametrize("variant", ["tri3-r8", "tri3-r9"])
def test_tri3_configs_match_dyn(variant):
    # 横に開く関節を足した3本足。tri-r2-dyn-dr と、機体・名前・ラベル・バンドルのパスだけが違う
    ref = json.loads((ROOT / "configs" / "train_full_tri-r2-dyn-dr.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}-dr.json").read_text(encoding="utf-8"))
    assert other["name"] == f"full-{variant}-dr"
    assert other["bundle"] == f"artifacts/{variant}-dr/cut/walk-task.json"
    assert other["trainer"]["label"] == f"q8 vast.ai full {variant}-dr"
    assert other["instance"]["label"] == f"q8-full-{variant}-dr"
    assert other["trainer"]["iterations"] == 1000
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in ref.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in ref["trainer"].items() if k != "label"}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}


@pytest.mark.parametrize("variant", ["tri3-r8", "tri3-r9"])
def test_tri3_slip_configs_match_tri3(variant):
    # 滑りの罰を足した版。train_full_<機体>-dr.json と、名前・ラベル・バンドルのパスだけが違う
    ref = json.loads((ROOT / "configs" / f"train_full_{variant}-dr.json").read_text(encoding="utf-8"))
    other = json.loads((ROOT / "configs" / f"train_full_{variant}-slip-dr.json").read_text(encoding="utf-8"))
    assert other["name"] == f"full-{variant}-slip-dr"
    assert other["bundle"] == f"artifacts/{variant}-slip-dr/cut/walk-task.json"
    assert other["trainer"]["label"] == f"q8 vast.ai full {variant}-slip-dr"
    assert other["instance"]["label"] == f"q8-full-{variant}-slip-dr"
    ignore = {"_comment", "name", "bundle", "trainer", "instance"}
    assert {k: v for k, v in other.items() if k not in ignore} == \
        {k: v for k, v in ref.items() if k not in ignore}
    assert {k: v for k, v in other["trainer"].items() if k != "label"} == \
        {k: v for k, v in ref["trainer"].items() if k != "label"}
    assert {k: v for k, v in other["instance"].items() if k != "label"} == \
        {k: v for k, v in ref["instance"].items() if k != "label"}


@pytest.mark.parametrize("variant", sorted(TRI3_DR))
def test_tri3_bundles_carry_slip_cost_only_on_slip(variant):
    # -slip のバンドルだけが滑りの罰 foot_slip（重み -5）を持つ。罰の無い版には項が無い
    task = json.loads((ROOT / "artifacts" / variant / "cut" / "walk-task.json").read_text(encoding="utf-8"))
    slip = [r for r in task["reward"] if r["label"] == "foot_slip"]
    if "-slip-" in variant:
        assert len(slip) == 1 and slip[0]["weight"] == -5.0
    else:
        assert slip == []


@pytest.mark.parametrize("variant", TAIL_TRI_DR_BUILT)
def test_しっぽと3本足の_dr_のバンドルは今の機体に_dr_を足したもの(variant, tmp_path):
    from q8 import dr_bundle

    source = ROOT / "robot" / "mjcf" / DR_SOURCE[variant] / "cut"
    out = ROOT / "artifacts" / variant / "cut"
    # 今の機体のカットモデルから作り直すと、置いてあるバンドルと1バイトも違わない（機体を作り直したら作り直す）
    dr_bundle.build(source, tmp_path)
    for name in ("cut.xml", "walk-task.json", "dr.json"):
        assert (tmp_path / name).read_bytes() == (out / name).read_bytes(), name
    ref = _task(DR_SOURCE[variant])
    other = json.loads((out / "walk-task.json").read_text(encoding="utf-8"))
    ignore = {"model", "randomisation", "disturbance"}
    assert {k: v for k, v in other.items() if k not in ignore} == {k: v for k, v in ref.items() if k not in ignore}
    assert other["randomisation"][:len(ref["randomisation"])] == ref["randomisation"]
    # サーボの剛性のランダム化は、全部のアクチュエータに1つずつ（3本足は 6、しっぽ付きは 9・10）
    kp = [r for r in other["randomisation"] if r["label"].startswith("dr_servo_kp_")]
    assert len(kp) == len(ref["actions"])
