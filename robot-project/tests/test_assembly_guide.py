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
MINI_R2_MESHES = ROOT / "robot" / "meshes" / "mini-r2"


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


def test_knee_servo_is_drawn_from_both_sides_of_the_thigh():
    # leg_rev 2 の膝サーボは上腿の内側（左脚では -Y、胴体の側）から差し込むので、内側から見た図も要る。
    # MuJoCo の方位角 270 は +Y（左脚の外側）から、0〜180 は -Y（左脚の内側）から見る。
    knee = [s for s in ag.STEPS if s["new"] == ["knee_servo_lf"]]
    azimuths = sorted(s["view"][0] for s in knee)
    assert len(knee) == 2, [s["key"] for s in knee]
    assert 0.0 < azimuths[0] < 180.0 and 180.0 < azimuths[1] < 360.0
    assert all(s.get("detail") and set(s["shown"]) == {"thigh_lf", "knee_servo_lf"} for s in knee)


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
    # mini の脚は組めない（docs/assembly-guide.md §0）: 胴体と股サーボだけが通る
    verdict = ag.assembly_verdict(plan)
    assert verdict == {"hip_servo_into_chassis": True, "cover_closes": True, "hip_lead_exits": True,
                       "knee_servo_into_thigh": False, "hip_horn_into_thigh": False,
                       "knee_horn_into_shin": False, "knee_lead_exits": False}
    assert plan["assembly_verdict"] == verdict


def _plan(fits, voids):
    return {"fit_checks": [{"part": p, "insert": i, "direction": list(d), "blocked_fraction": f}
                           for (p, i, d), f in fits.items()],
            "enclosed_voids": [{"part": p, "insert": i, "enclosed_void_mm3": v} for (p, i), v in voids.items()]}


def test_assembly_verdict_needs_one_open_direction_and_no_sealed_room():
    fits = {(c["part"], c["insert"], tuple(c["direction"])): 0.0 for c in ag.FIT_CHECKS}
    voids = {(c["part"], c["insert"]): 0.0 for c in ag.VOID_CHECKS}
    assert all(ag.assembly_verdict(_plan(fits, voids)).values())
    # 膝サーボは外側か内側のどちらか一方から入れば組める
    fits[("thigh_lf", "knee_servo_lf", (0, -1, 0))] = 0.24
    assert ag.assembly_verdict(_plan(fits, voids))["knee_servo_into_thigh"] is True
    fits[("thigh_lf", "knee_servo_lf", (0, 1, 0))] = 0.02
    assert ag.assembly_verdict(_plan(fits, voids))["knee_servo_into_thigh"] is False
    # 閉じた空間が残れば、リード線は出られない
    voids[("thigh_lf", "knee_servo_lf")] = 1035.0
    assert ag.assembly_verdict(_plan(fits, voids))["knee_lead_exits"] is False


@pytest.mark.skipif(not (MINI_R2_MESHES / "chassis.stl").exists(),
                    reason="robot/meshes/mini-r2 の STL が無い（scripts/build_robot.py --variant mini-r2 で生成する）")
def test_print_plan_for_mini_r2_assembles():
    plan = ag.print_plan(MINI_R2_MESHES)
    verdict = ag.assembly_verdict(plan)
    assert all(verdict.values()), verdict
    fits = {(c["part"], c["insert"], tuple(c["direction"])): c["blocked_fraction"] for c in plan["fit_checks"]}
    # 膝サーボは上腿の内側から、ホーンは上腿・下腿のサーボ側（内側）の面から入る
    assert fits[("thigh_lf", "knee_servo_lf", (0, 1, 0))] == pytest.approx(0.0)
    assert fits[("thigh_lf", "hip_horn_lf", (0, 1, 0))] == pytest.approx(0.0)
    assert fits[("shin_lf", "knee_horn_lf", (0, 1, 0))] == pytest.approx(0.0)
    voids = {(v["part"], v["insert"]): v["enclosed_void_mm3"] for v in plan["enclosed_voids"]}
    assert voids[("thigh_lf", "knee_servo_lf")] < 50.0
    # 印刷部品はベッドに載り、上腿・下腿はサポート無し（張り出しは橋渡しだけ）で刷れる向きがある
    for name in ("thigh_lf", "thigh_rf", "shin_lf", "shin_rf"):
        part = plan["parts"][name]
        assert part["overhang_mm2"] == pytest.approx(part["bridge_mm2"]), name



# --- 脚と関節をモデルから読む（3本足・しっぽ付きの機体） ---

def _check_layout(lay):
    """部品・プレート・組立の段の決まり（4脚の定数で確かめているのと同じこと）。"""
    names = [n for plate in lay["plates"] for n in plate["parts"]]
    assert sorted(names) == sorted(lay["printed"])
    seen: set[str] = set()
    for step in lay["steps"]:
        assert step["new"], step["key"]
        assert set(step["new"]) <= set(step["shown"]), step["key"]
        assert set(step["shown"]) <= set(lay["all"]), step["key"]
        if not step.get("detail"):
            assert seen <= set(step["shown"]), step["key"]
            seen |= set(step["shown"])
    assert seen == set(lay["all"]) - {"imu"}
    keys = {(c["part"], c["insert"]) for c in lay["fit_checks"]}
    assert set(lay["verdict_fits"].values()) <= keys
    assert set(lay["verdict_voids"].values()) <= {(c["part"], c["insert"]) for c in lay["void_checks"]}


# 4脚の値を、layout() を入れる前（コミット 75a3bb623）の定数から写して固定する。
# 部品・プレート・差し込み・リード線・判定のキーは、Git にある mini-r2 の印刷計画（そのときの定数で作った出力）と比べる。
OLD_STEPS = [  # (key, title, detail, new)
    ("step1_hip_servos", "股サーボ 4個を chassis に入れる", False,
     ["hip_servo_lf", "hip_servo_rf", "hip_servo_lh", "hip_servo_rh"]),
    ("step2_cover", "cover を被せて M2x10 を 2本締める", False, ["cover", "body_bolt_0", "body_bolt_1"]),
    ("step3_knee_servo", "膝サーボを上腿に入れる（左前脚、外側から見た図）", True, ["knee_servo_lf"]),
    ("step3b_knee_servo_inside", "膝サーボを上腿に入れる（左前脚、内側から見た図）", True, ["knee_servo_lf"]),
    ("step4_hip_horn", "股ホーンを上腿に留め、股サーボの軸に差す（左前脚）", True, ["hip_horn_lf", "thigh_lf"]),
    ("step5_thighs", "4本の上腿を付けた状態", False,
     ["hip_horn_lf", "hip_horn_rf", "hip_horn_lh", "hip_horn_rh", "thigh_lf", "thigh_rf", "thigh_lh", "thigh_rh",
      "knee_servo_lf", "knee_servo_rf", "knee_servo_lh", "knee_servo_rh"]),
    ("step6_shins", "膝ホーンと下腿を付ける", False,
     ["knee_horn_lf", "knee_horn_rf", "knee_horn_lh", "knee_horn_rh", "shin_lf", "shin_rf", "shin_lh", "shin_rh"]),
    ("step7_feet", "足先を付ける（滑り止めシートは描いていない）", False, ["foot_lf", "foot_rf", "foot_lh", "foot_rh"]),
    ("step8_esp32", "完成形（ESP32-S3 を cover 上面のポケットに載せた場合）", False, ["esp32"]),
    ("stand_side", "立った姿勢を左から見た図（ホーンを付ける角度の目安）", True,
     ["thigh_lf", "thigh_rf", "thigh_lh", "thigh_rh", "shin_lf", "shin_rf", "shin_lh", "shin_rh"]),
]
MINI_R2_PLAN = ROOT / "artifacts" / "mini-r2" / "assembly" / "print_plan.json"


def test_layout_の既定は_layout_を入れる前の4脚と同じ():
    lay = ag.layout()
    plan = json.loads(MINI_R2_PLAN.read_text())
    assert lay["printed"] == list(plan["parts"])
    assert len(lay["printed"]) == 14
    assert [(p["key"], p["title"], p["parts"]) for p in lay["plates"]] == \
        [(p["key"], p["title"], p["parts"]) for p in plan["plates"]]
    assert [(c["part"], c["insert"], list(c["direction"]), c["label"]) for c in lay["fit_checks"]] == \
        [(c["part"], c["insert"], c["direction"], c["label"]) for c in plan["fit_checks"]]
    assert [(c["part"], c["insert"], c["label"]) for c in lay["void_checks"]] == \
        [(c["part"], c["insert"], c["label"]) for c in plan["enclosed_voids"]]
    assert list(lay["verdict_fits"]) + list(lay["verdict_voids"]) == list(plan["assembly_verdict"])
    assert lay["verdict_fits"] == {
        "hip_servo_into_chassis": ("chassis", "hip_servo_lf"), "cover_closes": ("cover", "hip_servo_lf"),
        "knee_servo_into_thigh": ("thigh_lf", "knee_servo_lf"), "hip_horn_into_thigh": ("thigh_lf", "hip_horn_lf"),
        "knee_horn_into_shin": ("shin_lf", "knee_horn_lf")}
    assert lay["verdict_voids"] == {"hip_lead_exits": ("chassis+cover", "hip_servo_lf"),
                                    "knee_lead_exits": ("thigh_lf", "knee_servo_lf")}
    assert [(s["key"], s["title"], bool(s.get("detail")), s["new"]) for s in lay["steps"]] == OLD_STEPS
    assert lay["all"] == lay["printed"] + [f"{k}_{g}" for k in ("hip_servo", "knee_servo", "hip_horn", "knee_horn")
                                           for g in ("lf", "rf", "lh", "rh")] + \
        ["body_bolt_0", "body_bolt_1", "esp32", "imu"]
    # モジュールの定数は既定の layout と同じもの
    assert (ag.PRINTED_PARTS, ag.ALL_PARTS, ag.PLATES, ag.STEPS) == (lay["printed"], lay["all"], lay["plates"],
                                                                     lay["steps"])
    _check_layout(lay)


def test_layout_は3本足の後ろの脚の部品と差し込みを並べる():
    lay = ag.layout(["lf", "rf", "hc"])
    _check_layout(lay)
    assert len(lay["printed"]) == 2 + 3 * 3
    assert {"thigh_hc", "shin_hc", "foot_hc", "hip_servo_hc", "knee_horn_hc"} <= set(lay["all"])
    assert not {"thigh_lh", "hip_servo_rh"} & set(lay["all"])
    # 後ろの脚 hc は左前脚と別の場所（胴体の後端の台）に付くので、差し込みとリード線を別に確かめる
    assert {"hc_hip_servo_into_chassis", "hc_cover_closes", "hc_knee_servo_into_thigh", "hc_hip_horn_into_thigh",
            "hc_knee_horn_into_shin"} <= set(lay["verdict_fits"])
    assert {"hc_hip_lead_exits", "hc_knee_lead_exits"} <= set(lay["verdict_voids"])
    assert "上腿 3本" in [p["title"] for p in lay["plates"]]


@pytest.mark.parametrize("tail", [["tail_yaw"], ["tail_yaw", "tail_pitch"]])
def test_layout_はしっぽの部品と差し込みを並べる(tail):
    lay = ag.layout(list(ag.LEGS), tail)
    _check_layout(lay)
    assert "tail_link_1" in lay["printed"] and "tail_tip" not in lay["printed"]
    assert {"tail_yaw_servo", "tail_yaw_horn", "tail_tip"} <= set(lay["all"])
    assert {"tail_yaw_servo_into_body", "tail_yaw_horn_into_link"} <= set(lay["verdict_fits"])
    assert "tail_yaw_lead_exits" in lay["verdict_voids"]
    if "tail_pitch" in tail:
        assert {"tail_link_2", "tail_pitch_servo", "tail_pitch_horn"} <= set(lay["all"])
        assert {"tail_pitch_servo_into_link", "tail_pitch_horn_into_link"} <= set(lay["verdict_fits"])
    else:
        assert "tail_link_2" not in lay["all"]


def test_assembly_verdict_は_layout_の判定を使う():
    lay = ag.layout(["lf", "rf", "hc"])
    fits = {(c["part"], c["insert"], tuple(c["direction"])): 0.0 for c in lay["fit_checks"]}
    voids = {(c["part"], c["insert"]): 0.0 for c in lay["void_checks"]}
    verdict = ag.assembly_verdict(_plan(fits, voids), lay)
    assert set(verdict) == set(lay["verdict_fits"]) | set(lay["verdict_voids"]) and all(verdict.values())
    voids[("chassis+cover", "hip_servo_hc")] = 900.0
    assert ag.assembly_verdict(_plan(fits, voids), lay)["hc_hip_lead_exits"] is False


@pytest.mark.skipif(not (ROOT / "robot" / "mjcf" / "tri-r2" / "model-model.xml").exists(),
                    reason="robot/mjcf/tri-r2 が無い（scripts/build_robot.py --variant tri-r2 で生成する）")
def test_read_layout_はモデルの脚としっぽを読む():
    assert ag.read_layout("tri-r2")["legs"] == ["lf", "rf", "hc"]
    lay = ag.read_layout("mini-r2-p7-eff")
    assert lay["legs"] == list(ag.LEGS) and lay["tail"] == []
    if (ROOT / "robot" / "mjcf" / "mini-r2-tail2" / "model-model.xml").exists():
        assert ag.read_layout("mini-r2-tail2")["tail"] == ["tail_yaw", "tail_pitch"]


@pytest.mark.parametrize("variant", ["mini-r2-tail2", "tri-r2"])  # tail1 の部品は tail2 に含まれる
def test_print_plan_は3本足としっぽ付きの機体でも作れる(variant):
    mesh_dir = ROOT / "robot" / "meshes" / variant
    if not (mesh_dir / "chassis.stl").exists():
        pytest.skip(f"robot/meshes/{variant} の STL が無い")
    lay = ag.read_layout(variant)
    plan = ag.print_plan(mesh_dir, lay)
    json.dumps(plan)
    assert set(plan["parts"]) == set(lay["printed"])
    assert set(plan["assembly_verdict"]) == set(lay["verdict_fits"]) | set(lay["verdict_voids"])
    if variant == "tri-r2":
        # 3本足の後ろの脚（股サーボは胴体の後端の台、線は通り道から空洞へ）も組める（2026-11-18 の実ビルド）
        assert all(plan["assembly_verdict"].values()), plan["assembly_verdict"]
