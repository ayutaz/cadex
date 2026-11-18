"""scripts/assembly_guide.py: 印刷の向き・張り出し・プレート割り・組立の段階図（docs/assembly-guide.md 用）。"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import assembly_guide as ag  # noqa: E402
from q8.stl import write_binary_stl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MINI_MESHES = ROOT / "robot" / "meshes" / "mini"


def _box(lo, hi):
    """軸に沿った直方体の三角形 (12, 3, 3)。外向きの巻き順。"""
    x0, y0, z0 = lo
    x1, y1, z1 = hi
    v = np.array([[x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
                  [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1]], float)
    faces = [(0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
             (1, 2, 6), (1, 6, 5), (2, 3, 7), (2, 7, 6), (3, 0, 4), (3, 4, 7)]
    return np.array([[v[a], v[b], v[c]] for a, b, c in faces])


def _tee():
    """T 字（下に細い柱、上に横長の板）。板の下面の両脇が張り出しになる。"""
    stem = _box((-5, -5, 0), (5, 5, 30))
    left = _box((-25, -5, 20), (-5, 5, 30))
    right = _box((5, -5, 20), (25, 5, 30))
    return np.concatenate([stem, left, right])


@pytest.mark.parametrize("down", ag.AXES)
def test_rotation_puts_the_chosen_face_on_the_bed(down):
    r = ag.rotation_to_bed(down)
    np.testing.assert_allclose(r @ np.asarray(down, float), [0.0, 0.0, -1.0], atol=1e-12)
    assert np.linalg.det(r) == pytest.approx(1.0)


def test_a_box_on_its_face_has_no_overhang():
    rep = ag.overhang(_box((0, 0, 0), (10, 20, 30)), (0, 0, -1))
    assert rep["bed_contact_mm2"] == pytest.approx(200.0)
    assert rep["overhang_mm2"] == pytest.approx(0.0)
    assert rep["height_mm"] == pytest.approx(30.0)
    assert rep["footprint_mm"] == pytest.approx([10.0, 20.0])


def test_a_tee_overhangs_when_upright_and_not_when_upside_down():
    upright = ag.overhang(_tee(), (0, 0, -1))
    # 板の下面 50x10 から柱の断面 10x10 を除いた 400 mm² が宙に浮く（水平なので橋渡し）
    assert upright["overhang_mm2"] == pytest.approx(400.0)
    assert upright["bridge_mm2"] == pytest.approx(400.0)
    flipped = ag.overhang(_tee(), (0, 0, 1))
    assert flipped["overhang_mm2"] == pytest.approx(0.0)
    assert flipped["bed_contact_mm2"] == pytest.approx(500.0)


def test_best_orientation_picks_the_least_overhang():
    # 横に寝かせると張り出しが無く、ベッドに接する面（T の形 700 mm²）が最も広い
    best = ag.best_orientation(_tee())
    assert best["down"] in {(0, 1, 0), (0, -1, 0)}
    assert best["overhang_mm2"] == pytest.approx(0.0)
    assert best["bed_contact_mm2"] == pytest.approx(700.0)


def test_inward_winding_is_corrected():
    tris = _box((0, 0, 0), (10, 10, 10))[:, ::-1]  # 巻き順を逆にした（法線が内向き）
    rep = ag.overhang(tris, (0, 0, -1))
    assert rep["bed_contact_mm2"] == pytest.approx(100.0)
    assert rep["overhang_mm2"] == pytest.approx(0.0)


def _cup(closed_top=False):
    """10 x 10 の穴を持つ受け（底は z=0..2、壁の厚さ 2）。closed_top なら上に蓋（z=12..14）を載せる。"""
    parts = [_box((0, 0, 0), (14, 14, 2)), _box((0, 0, 2), (2, 14, 12)), _box((12, 0, 2), (14, 14, 12)),
             _box((2, 0, 2), (12, 2, 12)), _box((2, 12, 2), (12, 14, 12))]
    if closed_top:
        parts.append(_box((0, 0, 12), (14, 14, 14)))
    return np.concatenate(parts)


def test_blocked_fraction_sees_an_open_and_a_closed_pocket():
    insert = _box((3, 3, 2), (11, 11, 10))  # 受けの穴の底に載った箱
    # 上から下ろす（-Z へ動かす）なら入る。下から押し上げるなら底に当たる
    assert ag.blocked_fraction(_cup(), insert, (0, 0, -1)) == pytest.approx(0.0)
    assert ag.blocked_fraction(_cup(), insert, (0, 0, 1)) == pytest.approx(1.0)
    # 蓋があれば、どちらからも入らない
    assert ag.blocked_fraction(_cup(closed_top=True), insert, (0, 0, -1)) == pytest.approx(1.0)


def test_enclosed_void_finds_a_sealed_room_but_not_an_open_cup():
    insert = _box((3, 3, 2), (11, 11, 6))  # 穴の底に置いた低い箱。上に 6 mm の空間が残る
    assert ag.enclosed_void_mm3(_cup(), insert, h=0.5) == pytest.approx(0.0)
    # 蓋をすると、箱の上の空間（約 10 x 10 x 6 = 600 mm³ から箱の分を除く）が閉じ込められる
    assert ag.enclosed_void_mm3(_cup(closed_top=True), insert, h=0.5) > 300.0


def test_pack_plate_places_items_inside_the_bed_without_overlap():
    items = [("a", (96.0, 62.0)), ("b", (96.0, 62.0))]
    placed = ag.pack_plate(items, bed=(180.0, 180.0), margin=5.0, gap=5.0)
    assert [p["name"] for p in placed] == ["a", "b"]
    rects = [(p["x"], p["y"], p["x"] + p["w"], p["y"] + p["d"]) for p in placed]
    for x0, y0, x1, y1 in rects:
        assert 5.0 <= x0 and x1 <= 175.0 and 5.0 <= y0 and y1 <= 175.0
    (a0, b0, a1, b1), (c0, d0, c1, d1) = rects
    assert a1 + 5.0 <= c0 or c1 + 5.0 <= a0 or b1 + 5.0 <= d0 or d1 + 5.0 <= b0


def test_pack_plate_refuses_what_does_not_fit():
    with pytest.raises(ValueError):
        ag.pack_plate([("big", (200.0, 10.0))], bed=(180.0, 180.0), margin=5.0, gap=5.0)
    with pytest.raises(ValueError):
        ag.pack_plate([(str(i), (80.0, 80.0)) for i in range(5)], bed=(180.0, 180.0), margin=5.0, gap=5.0)


def test_plates_print_every_printed_part_once():
    names = [n for plate in ag.PLATES for n in plate["parts"]]
    assert sorted(names) == sorted(ag.PRINTED_PARTS)
    assert len(ag.PRINTED_PARTS) == 14


def test_steps_add_parts_in_order_and_cover_the_whole_robot():
    seen: set[str] = set()
    for step in ag.STEPS:
        assert step["new"], step["key"]
        assert set(step["new"]) <= set(step["shown"]), step["key"]
        assert set(step["shown"]) <= set(ag.ALL_PARTS), step["key"]
        if not step.get("detail"):
            # 全体図の段は、それまでに足した部品をすべて見せる（部品が消えない）
            assert seen <= set(step["shown"]), step["key"]
            seen |= set(step["shown"])
    assert seen == set(ag.ALL_PARTS) - {"imu"}  # IMU は方策を載せる段階で買うので、組立図には出さない


def test_scene_xml_loads_in_mujoco(tmp_path):
    import mujoco

    for name in ("p", "q"):
        write_binary_stl(_box((0, 0, 0), (10, 10, 10)), tmp_path / f"{name}.stl")
    xml = ag.scene_xml({"p": tmp_path / "p.stl", "q": tmp_path / "q.stl"},
                       colours={"p": ag.COLOUR_DONE, "q": ag.COLOUR_NEW}, size=(320, 240))
    m = mujoco.MjModel.from_xml_string(xml)
    assert m.nmesh == 2
    assert m.vis.global_.offwidth == 320


def test_png_round_trips_through_the_row_filters(tmp_path):
    img = np.zeros((60, 80, 3), np.uint8)
    img[:, :, 0] = np.arange(80, dtype=np.uint8)[None, :] * 3
    img[:, :, 1] = np.arange(60, dtype=np.uint8)[:, None] * 4
    path = tmp_path / "x.png"
    ag.save_png(img, path)
    raw = path.read_bytes()
    assert raw[:8] == b"\x89PNG\r\n\x1a\n"
    import zlib
    import struct

    # IDAT を伸張すると元の画素に戻る（Paeth などのフィルタを正しく掛け外しできる）
    pos, idat = 8, b""
    while pos < len(raw):
        n = struct.unpack(">I", raw[pos:pos + 4])[0]
        tag = raw[pos + 4:pos + 8]
        if tag == b"IDAT":
            idat += raw[pos + 8:pos + 8 + n]
        pos += 12 + n
    data = zlib.decompress(idat)
    assert ag.unfilter_png(data, 80, 60).tolist() == img.tolist()


@pytest.mark.skipif(not (MINI_MESHES / "chassis.stl").exists(),
                    reason="robot/meshes/mini の STL が無い（scripts/build_robot.py --variant mini で生成する）")
def test_print_plan_for_mini(tmp_path):
    plan = ag.print_plan(MINI_MESHES)
    json.dumps(plan)  # JSON に書ける
    parts = plan["parts"]
    assert set(parts) == set(ag.PRINTED_PARTS)
    # 胴体の下半分は底面、上腿・下腿は外側の板面を下にするのが張り出し最小
    assert parts["chassis"]["down"] == [0, 0, -1]
    assert parts["thigh_lf"]["down"] == [0, 1, 0]
    assert parts["thigh_rf"]["down"] == [0, -1, 0]
    for plate in plan["plates"]:
        assert plate["filament_g"] > 0
        for p in plate["placed"]:
            assert p["x"] + p["w"] <= 180.0 and p["y"] + p["d"] <= 180.0
    fits = {(c["part"], c["insert"], tuple(c["direction"])): c["blocked_fraction"] for c in plan["fit_checks"]}
    # 股サーボは chassis に上から入り、cover は上から被せられる
    assert fits[("chassis", "hip_servo_lf", (0, 0, -1))] == pytest.approx(0.0)
    assert fits[("cover", "hip_servo_lf", (0, 0, 1))] == pytest.approx(0.0)
    assert {k[:2] for k in fits} >= {("thigh_lf", "knee_servo_lf"), ("thigh_lf", "hip_horn_lf"),
                                     ("shin_lf", "knee_horn_lf")}
    voids = {(v["part"], v["insert"]): v["enclosed_void_mm3"] for v in plan["enclosed_voids"]}
    # 股サーボの線は胴体の空洞へ抜ける（ネジ穴の底などの小さな隅、数 mm³ は残る）
    assert voids[("chassis+cover", "hip_servo_lf")] < 50.0
    assert ("thigh_lf", "knee_servo_lf") in voids
    total = sum(p["filament_g"] for p in plan["plates"])
    assert total == pytest.approx(0.8 * sum(parts[n]["volume_cm3"] for n in parts), rel=1e-6)
