"""実機との差に強くする学習バンドル（q8.dr_bundle）の検査。

元のカットモデル（mini-r2-p7-eff）に、サーボの応答の遅れ・剛性・重心のずれ・押しのランダム化を足す。
既存の番号（body・joint・sensor・actuator）を変えず、公称の値では元のモデルと同じに立つこと。
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import mujoco
import numpy as np
import pytest

from q8 import dr_bundle

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "robot" / "mjcf" / "mini-r2-p7-eff" / "cut"
SHIPPED = ROOT / "artifacts" / "mini-r2-p7-eff-dr" / "cut"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("dr") / "cut"
    dr_bundle.build(SOURCE, out)
    return out


def _model(path: Path) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(str(path))


def _task(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _entry(task: dict, label: str) -> dict:
    found = [e for e in task["randomisation"] if e["label"] == label]
    assert len(found) == 1, label
    return found[0]


def test_出力は_cut_xml_と_walk_task_と_dr_json(built):
    assert {p.name for p in built.iterdir()} == {"cut.xml", "walk-task.json", "dr.json"}


def test_タスクは新しいモデルを相対パスと_sha256_で参照する(built):
    task = _task(built / "walk-task.json")
    assert task["model"]["path"] == "outputs/cut.xml"
    assert task["model"]["sha256"] == hashlib.sha256((built / "cut.xml").read_bytes()).hexdigest()
    assert task["model"]["bytes"] == len((built / "cut.xml").read_bytes())


def test_既存の番号は変わらず_payload_は末尾の_body(built):
    old, new = _model(SOURCE / "cut.xml"), _model(built / "cut.xml")
    for size in ("nq", "nv", "nu", "nsensor", "nsensordata", "njnt", "ngeom", "nsite"):
        assert getattr(new, size) == getattr(old, size), size
    assert new.nbody == old.nbody + 1
    for b in range(old.nbody):
        assert mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_BODY, b) == \
            mujoco.mj_id2name(old, mujoco.mjtObj.mjOBJ_BODY, b)
    payload = old.nbody
    assert mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_BODY, payload) == "dr_payload"
    assert new.body_parentid[payload] == 1  # c_chassis
    assert new.body_jntnum[payload] == 0


def test_総質量は変わらず_payload_は_20_g(built):
    old, new = _model(SOURCE / "cut.xml"), _model(built / "cut.xml")
    assert new.body_subtreemass[0] == pytest.approx(old.body_subtreemass[0], abs=1e-9)
    assert new.body_mass[old.nbody] == pytest.approx(0.020)
    assert new.body_mass[1] == pytest.approx(old.body_mass[1] - 0.020)


def test_アクチュエータは一次遅れ_15_ms(built):
    new = _model(built / "cut.xml")
    assert all(int(t) == int(mujoco.mjtDyn.mjDYN_FILTEREXACT) for t in new.actuator_dyntype)
    assert np.allclose(new.actuator_dynprm[:, 0], 0.015)
    assert new.na == new.nu


def test_遅れのランダム化は8つのアクチュエータに共通の係数(built):
    entry = _entry(_task(built / "walk-task.json"), "dr_servo_lag")
    assert (entry["low"], entry["high"], entry["mode"]) == (0.1, 2.0, "scale")
    width = mujoco.mjNDYN
    assert entry["fields"] == [{"field": "actuator_dynprm", "index": a * width} for a in range(8)]


def test_剛性のランダム化はアクチュエータごとに_gain_と_bias_を同じ係数で(built):
    task = _task(built / "walk-task.json")
    new = _model(built / "cut.xml")
    for a in range(8):
        name = mujoco.mj_id2name(new, mujoco.mjtObj.mjOBJ_ACTUATOR, a)
        entry = _entry(task, f"dr_servo_kp_{name.split('/')[0]}")
        assert (entry["low"], entry["high"]) == (0.7, 1.3)
        assert entry["fields"] == [{"field": "actuator_gainprm", "index": a * mujoco.mjNGAIN + 0},
                                   {"field": "actuator_biasprm", "index": a * mujoco.mjNBIAS + 1}]


def _apply(model, task, labels_factors):
    """タスクのエントリを、指定の係数でエンジンと同じ規則（flat に掛ける）で当てる。"""
    for label, factor in labels_factors.items():
        for field in _entry(task, label)["fields"]:
            getattr(model, field["field"]).flat[field["index"]] *= factor
    mujoco.mj_setConst(model, mujoco.MjData(model))


def test_剛性の係数を掛けても位置サーボのまま(built):
    task, model = _task(built / "walk-task.json"), _model(built / "cut.xml")
    _apply(model, task, {"dr_servo_kp_hip_lf": 0.7})
    assert model.actuator_gainprm[0, 0] == pytest.approx(-model.actuator_biasprm[0, 1])
    assert model.actuator_gainprm[0, 0] == pytest.approx(0.7 * 2.02277)


@pytest.mark.parametrize("fx,fy", [(1.0, 1.0), (-1.0, -1.0), (1.0, -1.0)])
def test_重心のずれは_x_約_2_7_mm_y_約_1_8_mm(built, fx, fy):
    task = _task(built / "walk-task.json")

    def com(factor_x, factor_y):
        model = _model(built / "cut.xml")
        _apply(model, task, {"dr_com_x": factor_x, "dr_com_y": factor_y})
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        return data.subtree_com[1].copy()

    shift = (com(fx, fy) - com(0.0, 0.0)) * 1000.0
    assert shift[0] == pytest.approx(fx * 2.73, abs=0.02)
    assert shift[1] == pytest.approx(fy * 1.82, abs=0.02)
    assert abs(shift[2]) < 1e-6


def test_公称の_payload_は胴体の重心の上(built):
    old, new = _model(SOURCE / "cut.xml"), _model(built / "cut.xml")
    d_old, d_new = mujoco.MjData(old), mujoco.MjData(new)
    mujoco.mj_forward(old, d_old)
    mujoco.mj_forward(new, d_new)
    # 公称（係数 0 にあたる位置）は payload の ipos を x・y とも 0 にしたとき
    model = _model(built / "cut.xml")
    _apply(model, _task(built / "walk-task.json"), {"dr_com_x": 0.0, "dr_com_y": 0.0})
    d = mujoco.MjData(model)
    mujoco.mj_forward(model, d)
    assert np.allclose(d.subtree_com[0], d_old.subtree_com[0], atol=1e-6)


def _stand(model, seconds=0.5):
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    for _ in range(int(round(seconds / model.opt.timestep))):
        mujoco.mj_step(model, data)
    return data


def test_公称では元のモデルとほぼ同じに立つ(built):
    old = _model(SOURCE / "cut.xml")
    new = _model(built / "cut.xml")
    _apply(new, _task(built / "walk-task.json"), {"dr_com_x": 0.0, "dr_com_y": 0.0})
    a, b = _stand(old), _stand(new)
    assert abs(a.qpos[2] - b.qpos[2]) < 0.001
    tilt = math.degrees(2 * math.asin(min(1.0, math.hypot(b.qpos[4], b.qpos[5]))))
    assert tilt < 1.0
    assert np.allclose(a.qpos[7:], b.qpos[7:], atol=math.radians(0.5))


def test_押しは胴体への水平の2つで_success_には無い(built):
    task = _task(built / "walk-task.json")
    pushes = task["disturbance"]
    assert [p["label"] for p in pushes] == ["dr_push_early", "dr_push_late"]
    for p in pushes:
        assert (p["body"], p["body_id"], p["direction"]) == ("c_chassis", 1, "horizontal")
        assert (p["newtons_low"], p["newtons_high"], p["duration_s"]) == (0.0, 1.5, 0.1)
        assert (p["azimuth_low_rad"], p["azimuth_high_rad"]) == (0.0, 2.0 * math.pi)
        assert p["sustained"] is False
        assert (p["applied_at"], p["frame"]) == ("centre_of_mass", "world")
    assert (pushes[0]["at_low_s"], pushes[0]["at_high_s"]) == (1.0, 3.5)
    assert (pushes[1]["at_low_s"], pushes[1]["at_high_s"]) == (4.0, 7.0)
    episode = task["episode"]["episode_seconds"]
    assert all(p["at_high_s"] + p["duration_s"] <= episode for p in pushes)
    assert task["success"]["disturbance"] == []


def test_元のタスクのランダム化とリセットはそのまま(built):
    src, new = _task(SOURCE / "walk-task.json"), _task(built / "walk-task.json")
    assert new["randomisation"][:len(src["randomisation"])] == src["randomisation"]
    assert new["reset_variation"] == src["reset_variation"]
    ignore = {"model", "randomisation", "disturbance"}
    assert {k: v for k, v in new.items() if k not in ignore} == \
        {k: v for k, v in src.items() if k not in ignore}


def test_dr_json_は範囲と実際の数値を書く(built):
    dr = json.loads((built / "dr.json").read_text(encoding="utf-8"))
    assert dr["servo_lag_ms"] == pytest.approx([1.5, 30.0])
    assert dr["com_shift_mm"]["x"] == pytest.approx([-2.73, 2.73], abs=0.02)
    assert dr["com_shift_mm"]["y"] == pytest.approx([-1.82, 1.82], abs=0.02)
    assert dr["source"]["task_sha256"] == hashlib.sha256((SOURCE / "walk-task.json").read_bytes()).hexdigest()
    assert "/Users/" not in (built / "dr.json").read_text(encoding="utf-8")


def test_置いてあるバンドルは今の関数の出力と同じ(built):
    for name in ("cut.xml", "walk-task.json", "dr.json"):
        assert (SHIPPED / name).read_bytes() == (built / name).read_bytes(), name
