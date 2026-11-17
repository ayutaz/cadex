"""設計スクリプト robot/design/quadruped8.py の電池の有無（battery_mass_g）。

Cadex のエンジンは使わない。xscript が受け取るグローバル（params, num, part, assembly, lib, math）を
呼び出しを記録するだけの偽物に差し替えて、スクリプトを Python として実行する。形状は作らないので、
確かめられるのは「どの部品・コンポーネント・溶接・質量・ランダム化を宣言したか」だけ。
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace

import pytest

DESIGN = Path(__file__).resolve().parents[1] / "robot" / "design" / "quadruped8.py"

# CadexCatalog.py の MG90S / SG90 の値のうち、スクリプトが読むもの（数値は計算が通れば何でもよい）
SERVO_SPEC = {
    "body_length_mm": 22.8, "body_width_mm": 12.2, "case_height_mm": 28.4, "overall_tab_length_mm": 32.5,
    "shaft_offset_from_front_mm": 8.75, "spline_dia_mm": 4.9, "spline_height_mm": 4.0,
    "stall_torque": [{"volts": 4.8, "kg_cm": 1.8}], "speed": [{"volts": 4.8, "s_per_60_deg": 0.10}],
    "effective_density_kg_m3": 1500.0,
}


class Thing:
    """宣言された部品・コンポーネント・関節など。label と、作ったときの引数を持つ。"""

    def __init__(self, kind: str, args: tuple, kwargs: dict, spec: dict | None = None):
        self.kind = kind
        self.args = args
        self.kwargs = kwargs
        self.label = kwargs.get("label")
        self.spec = spec or {}

    def __repr__(self) -> str:
        return f"<{self.kind} {self.label}>"


class Domain:
    """part.* / assembly.* の偽物。どの関数を呼んでも Thing を返し、呼び出しを記録する。"""

    def __init__(self, name: str, log: list[Thing]):
        self._name = name
        self._log = log

    def __getattr__(self, fn: str):
        def call(*args, **kwargs):
            th = Thing(f"{self._name}.{fn}", args, kwargs)
            self._log.append(th)
            return th
        return call


class FakeServo:
    def __init__(self, log, label):
        self.spec = dict(SERVO_SPEC)
        self.body = Thing("lib.servo", (), {"label": label})
        self._log = log
        log.append(self.body)

    def bay(self, **kwargs):
        return Thing("servo.bay", (), kwargs)

    def horn(self, kind, label=None):
        h = SimpleNamespace(spec={"hub_height_mm": 2.5, "density_kg_m3": 1200.0},
                            body=Thing("servo.horn", (kind,), {"label": label}))
        self._log.append(h.body)
        return h


class FakeLib:
    def __init__(self, log):
        self._log = log

    def servo(self, sku, origin=None, direction=None, roll_degrees=0.0, label=None):
        return FakeServo(self._log, label)

    def bolt(self, size, length, origin=None, direction=None, label=None):
        body = Thing("lib.bolt", (size, length), {"label": label})
        self._log.append(body)
        return SimpleNamespace(spec={"head_dia_mm": 3.8, "density_kg_m3": 7900.0}, body=body)

    def clearance_hole(self, size):
        return 2.4

    def tap_drill(self, size):
        return 1.6


def run_design(**overrides) -> tuple[dict, list[Thing]]:
    """設計スクリプトを偽物の API で実行し、result と宣言の記録を返す。"""
    log: list[Thing] = []

    def params(**specs):
        unknown = set(overrides) - set(specs)
        assert not unknown, f"スクリプトに無いパラメータ: {unknown}"
        return SimpleNamespace(**{k: overrides.get(k, v) for k, v in specs.items()})

    def num(default, **kwargs):
        return default

    env = {"__builtins__": __builtins__, "math": math, "params": params, "num": num,
           "part": Domain("part", log), "assembly": Domain("assembly", log), "lib": FakeLib(log)}
    exec(compile(DESIGN.read_text(), str(DESIGN), "exec"), env)
    return env["result"], log


def labels(log: list[Thing], kind: str) -> list:
    return [t.label for t in log if t.kind == kind]


def bodies_of(log: list[Thing]) -> list:
    return [t.args[0] for t in log if t.kind == "assembly.body"]


def test_default_design_carries_the_battery_standin():
    result, log = run_design()
    assert "battery_standin" in labels(log, "part.box")
    assert "battery" in labels(log, "assembly.component")
    assert "rand_battery_mass" in labels(log, "assembly.randomise")
    assert {"battery", "c_bat", "weld_battery"} <= set(result)
    assert result["c_bat"] in bodies_of(log)
    assert result["c_bat"] in result["asm"].args[0]           # アセンブリのコンポーネント一覧に入る


@pytest.mark.parametrize("mass", [30.0, 1.0, 120.0])
def test_battery_mass_above_zero_sets_the_standin_density(mass):
    result, log = run_design(battery_mass_g=mass)
    body = [t for t in log if t.kind == "assembly.body" and t.args[0] is result["c_bat"]]
    assert len(body) == 1
    vol_m3 = 40.0 * 26.0 * 8.0 * 1e-9
    assert body[0].kwargs["density_kg_m3"] * vol_m3 * 1000.0 == pytest.approx(mass)


def test_zero_battery_mass_declares_no_battery_at_all():
    result, log = run_design(battery_mass_g=0.0)
    assert [t for t in log if "bat" in str(t.label)] == []
    assert not ({"battery", "c_bat", "weld_battery"} & set(result))
    components = result["asm"].args[0]
    assert all(c.label != "battery" for c in components)
    assert all(b.label != "battery" for b in bodies_of(log))


def test_zero_battery_mass_changes_nothing_else():
    with_bat, log_with = run_design(battery_mass_g=30.0)
    without, log_without = run_design(battery_mass_g=0.0)
    removed = {"battery", "c_bat", "weld_battery"}
    assert set(with_bat) - set(without) == removed
    assert set(without) <= set(with_bat)
    # 宣言の列から電池の4つ（部品・コンポーネント・溶接・質量）とランダム化を除くと、ほかは同じ並び
    def signature(log, drop):
        return [(t.kind, t.label) for t in log if t not in drop]
    bat = with_bat["c_bat"]
    weld = with_bat["weld_battery"]
    drop = {with_bat["battery"], bat, weld, *weld.args}     # 溶接の2つのコネクタも
    drop |= {t for t in log_with if t.kind in ("assembly.body", "assembly.randomise") and t.args and t.args[0] is bat}
    assert signature(log_with, drop) == signature(log_without, set())
    assert len(result_components := with_bat["asm"].args[0]) == len(without["asm"].args[0]) + 1
    assert bat in result_components
