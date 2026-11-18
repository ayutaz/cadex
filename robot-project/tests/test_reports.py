import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest

from q8.clearance import summarise
from q8.manufacturing import bom, printability
from q8.stl import write_binary_stl
from q8.visual import add_visual_meshes, stl_for_body

STATIC_MD = """# Clearance and intersection
| first | second | distance (mm) | common volume (mm³) | verdict | detail |
|---|---|---|---|---|---|
| c_a (a; uncatalogued) | c_b (b; uncatalogued) | 3.0 | 0.0 | clear | — |
| c_servo (servo; servo/mg90s) | c_horn (horn; servo_horn/mg90s-cross) | 0.0 | 0.0 | below clearance | — |
| c_cover (cover; uncatalogued) | c_bolt (bolt; bolt/m2) | 0.0 | 5.6 | threaded | — |
"""


def _sweep_md(pairs):
    body = {"status": "complete", "joints": [{"joint": "hip", "status": "complete", "range_degrees": [-10, 10],
                                              "step": 5, "sample_count": 5, "pairs": pairs}]}
    return "# Swept\n```json\n" + json.dumps(body) + "\n```\n"


@pytest.fixture
def project(tmp_path):
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "clearance.md").write_text(STATIC_MD)
    return tmp_path


def test_intended_contacts_and_threads_are_not_problems(project):
    (project / "docs" / "clearance-sweep.md").write_text(_sweep_md([]))
    s = summarise(project, {frozenset(("c_servo", "c_horn"))})
    assert s["static_pairs"] == 3
    assert s["static_problems"] == []
    assert s["self_collision_free"] is True


def test_a_swept_collision_is_reported_with_its_first_contact(project):
    pairs = [{"first": "c_body", "second": "c_leg", "minimum_distance_mm": 0.0,
              "maximum_common_volume_mm3": 12.0, "first_contact_degrees": 10.0},
             {"first": "c_x", "second": "c_y", "minimum_distance_mm": 2.0, "maximum_common_volume_mm3": 0.0}]
    (project / "docs" / "clearance-sweep.md").write_text(_sweep_md(pairs))
    s = summarise(project, {frozenset(("c_servo", "c_horn"))})
    assert s["self_collision_free"] is False
    assert s["joints"][0]["contacts"] == [{"first": "c_body", "second": "c_leg", "minimum_distance_mm": 0.0,
                                           "maximum_common_volume_mm3": 12.0, "first_contact_degrees": 10.0}]


def _box(path, size):
    sx, sy, sz = size
    v = np.array([[0, 0, 0], [sx, 0, 0], [sx, sy, 0], [0, sy, 0], [0, 0, sz], [sx, 0, sz], [sx, sy, sz], [0, sy, sz]])
    f = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4), (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    write_binary_stl(np.array([[v[a], v[b], v[c]] for a, b, c in f], float), path)


def test_printability_and_bom(tmp_path):
    _box(tmp_path / "chassis.stl", (100, 60, 10))       # 60 cm3
    _box(tmp_path / "foot_lf.stl", (10, 10, 10))        # 1 cm3, TPU
    _box(tmp_path / "hip_servo_lf.stl", (20, 10, 20))   # 購入品なので印刷部品に数えない
    _box(tmp_path / "huge.stl", (10, 10, 10))
    pr = printability(tmp_path, {"wall": 2.0, "link_thickness": 6.5, "clearance": 0.5})
    assert set(pr["parts"]) == {"chassis", "foot_lf"}
    assert pr["parts"]["foot_lf"]["material"] == "TPU"
    assert pr["checks"]["all_parts_fit_bed"] and pr["checks"]["wall_at_least_4_perimeters"]
    inv = [{"component": f"c_hip_servo_{i}", "output": f"hip_servo_{i}", "family": "servo", "sku": "mg90s",
            "appearance": "mechanism", "volume_mm3": 1.0} for i in range(8)]
    inv += [{"component": "c_chassis", "output": "chassis", "family": "", "sku": "", "appearance": "shell", "volume_mm3": 1.0},
            {"component": "c_esp", "output": "esp32", "family": "", "sku": "", "appearance": "mechanism", "volume_mm3": 1.0}]
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"items": {
        "servo/mg90s": {"name": "MG90S", "unit_price": 450, "status": "推定"},
        "standin/esp32s3": {"name": "ESP32-S3", "unit_price": 1800, "status": "推定"},
        "extra/self_tap_m2x6": {"name": "ネジ", "unit_price": 5, "status": "推定"},
        "extra/wiring": {"name": "配線", "unit_price": 500, "status": "推定"},
        "filament/pla": {"name": "PLA", "unit_price_per_g": 2.5, "status": "推定"},
        "filament/tpu": {"name": "TPU", "unit_price_per_g": 4.0, "status": "推定"}}}))
    b = bom(inv, pr, prices, pla_density=1000.0)
    by = {l["item"]: l for l in b["purchased"]}
    assert by["servo/mg90s"]["qty"] == 8 and by["servo/mg90s"]["cost_jpy"] == 3600
    assert by["extra/self_tap_m2x6"]["qty"] == 16
    assert by["filament/pla"]["cost_jpy"] == 150           # 60 cm3 × 1.0 g/cm3 × 2.5 円
    assert b["estimated_total_jpy"] == 3600 + 1800 + 80 + 500 + 150 + round(1.2 * 4.0)
    assert b["within_budget"]


def test_visual_meshes_attach_to_matching_bodies(tmp_path):
    model = tmp_path / "m.xml"
    model.write_text("""<mujoco><worldbody><geom name="environment/floor" type="plane" size="0 0 0.1"/>
      <body name="c_chassis"><freejoint/><geom name="c_chassis/collision0" type="box" size=".01 .01 .01"/>
        <body name="c_esp"><inertial pos="0 0 0" mass="0.01" diaginertia="1e-6 1e-6 1e-6"/></body></body>
      </worldbody></mujoco>""")
    _box(tmp_path / "chassis.stl", (10, 10, 10))
    _box(tmp_path / "esp32.stl", (10, 10, 2))
    out = add_visual_meshes(model, tmp_path, tmp_path / "v.xml")
    root = ET.parse(out).getroot()
    geoms = {g.get("name"): g for g in root.iter("geom")}
    assert "vis/chassis" in geoms and "vis/esp32" in geoms
    assert geoms["vis/chassis"].get("contype") == "0"
    assert geoms["c_chassis/collision0"].get("group") == "3"
    assert stl_for_body("c_bat") == "battery" and stl_for_body("world") is None
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(out))
    assert m.nmesh == 2


def test_bom_has_a_battery_line_only_when_the_design_carries_one(tmp_path):
    pr = {"parts": {}}
    prices = tmp_path / "prices.json"
    prices.write_text(json.dumps({"items": {
        "standin/battery": {"name": "電池", "unit_price": 0, "status": "未定"},
        "filament/pla": {"name": "PLA", "unit_price_per_g": 2.5, "status": "推定"},
        "filament/tpu": {"name": "TPU", "unit_price_per_g": 4.0, "status": "推定"}}}))
    bat = {"component": "c_bat", "output": "battery", "family": "", "sku": "", "appearance": "mechanism",
           "volume_mm3": 8320.0}
    with_bat = {l["item"]: l for l in bom([bat], pr, prices, pla_density=800.0)["purchased"]}
    assert with_bat["standin/battery"]["qty"] == 1
    without = {l["item"] for l in bom([], pr, prices, pla_density=800.0)["purchased"]}
    assert "standin/battery" not in without          # 電池なし（battery_mass_g=0）の機体は行を出さない


def test_しっぽのリンクは印刷部品で先端のおもりは購入品(tmp_path):
    # 前は tail_link_* が印刷部品に入らず、部品表に値段の分からない代用品（standin/tail_link_1）として載った
    _box(tmp_path / "tail_link_1.stl", (80, 25, 6.5))
    _box(tmp_path / "tail_link_2.stl", (70, 6.5, 25))
    _box(tmp_path / "tail_tip.stl", (10, 10, 10))
    pr = printability(tmp_path, {"wall": 2.0, "link_thickness": 6.5, "clearance": 0.5})
    assert set(pr["parts"]) == {"tail_link_1", "tail_link_2"}
    assert {p["material"] for p in pr["parts"].values()} == {"PLA"}
    inv = [{"component": "c_" + n, "output": n, "family": "", "sku": "", "appearance": "shell", "volume_mm3": 1.0}
           for n in ("tail_link_1", "tail_link_2", "tail_tip")]
    prices = json.loads((Path(__file__).resolve().parents[1] / "configs" / "bom_prices.json").read_text())
    path = tmp_path / "prices.json"
    path.write_text(json.dumps(prices))
    by = {l["item"]: l for l in bom(inv, pr, path, pla_density=800.0)["purchased"]}
    assert not {"standin/tail_link_1", "standin/tail_link_2"} & set(by)
    assert by["standin/tail_tip"]["unit_price_jpy"] is not None
