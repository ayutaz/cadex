"""Cadexが書いたMJCFに、出力したSTLを見た目用メッシュとして足す。

CadexのMJCFは衝突用の単純形状（箱・カプセル・球）だけを持つ（docs/XSCRIPT.md）。
このロボットの部品はワールド座標の立位姿勢で作ってあり、MJCFの各ボディの座標系は
ワールド原点にある。そのため、STL（mm）を 0.001 倍してそのボディに付ければ正しく重なる。
追加する形状は contype=0 / conaffinity=0 なので、物理挙動は一切変わらない。
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

# MJCFのボディ名（"c_" + 出力キー）と STL のファイル名の対応で、名前が違うもの
ALIASES = {"esp": "esp32", "bat": "battery"}

COLOURS = {
    "shell": "0.86 0.86 0.84 1",      # 印刷部品
    "mechanism": "0.18 0.19 0.21 1",  # サーボ・基板など購入品
    "accent": "0.95 0.45 0.10 1",     # 足先・IMU
}


def stl_for_body(body: str) -> str | None:
    if not body.startswith("c_"):
        return None
    key = body[2:]
    return ALIASES.get(key, key)


def role_of(stem: str) -> str:
    if stem.startswith(("foot_", "imu")):
        return "accent"
    if stem.startswith(("hip_servo", "knee_servo", "hip_horn", "knee_horn", "esp32", "battery")):
        return "mechanism"
    return "shell"


def add_visual_meshes(model_xml: Path, mesh_dir: Path, out_xml: Path,
                      mesh_prefix: str | None = None) -> Path:
    """model_xml に mesh_dir のSTLを足した MJCF を out_xml に書く。

    mesh_prefix は out_xml から見たSTLへの相対パス（省略時は絶対パス）。
    """
    tree = ET.parse(model_xml)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        root.insert(1, asset)
    # 見た目の材質（衝突形状は半透明にして、どちらも見えるようにする）
    for name, rgba in COLOURS.items():
        ET.SubElement(asset, "material", name="vis_" + name, rgba=rgba, specular="0.3", shininess="0.4")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", width="512", height="512",
                  rgb1="0.20 0.22 0.25", rgb2="0.15 0.16 0.18")
    ET.SubElement(asset, "material", name="floor_mat", texture="grid", texrepeat="20 20", reflectance="0.05")
    used = 0
    for body in root.iter("body"):
        stem = stl_for_body(body.get("name", ""))
        if stem is None:
            continue
        stl = mesh_dir / f"{stem}.stl"
        if not stl.exists():
            continue
        file_ref = (f"{mesh_prefix.rstrip('/')}/{stl.name}" if mesh_prefix is not None else str(stl.resolve()))
        ET.SubElement(asset, "mesh", name="vis_" + stem, file=file_ref, scale="0.001 0.001 0.001")
        geom = ET.Element("geom", name="vis/" + stem, type="mesh", mesh="vis_" + stem,
                          material="vis_" + role_of(stem), contype="0", conaffinity="0", group="1",
                          density="0", mass="0")
        # inertial の直後に挿入（MJCFは順序に寛容だが、読みやすさのため）
        body.insert(1, geom)
        used += 1
    # 衝突形状は group 3（既定で非表示）へ移す
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if "/collision" in name:
            geom.set("group", "3")
        if name == "environment/floor":
            geom.set("material", "floor_mat")
            geom.set("group", "0")
    vis = root.find("visual")
    if vis is None:
        vis = ET.Element("visual")
        root.insert(1, vis)
    ET.SubElement(vis, "global", offwidth="1280", offheight="720")
    ET.SubElement(vis, "headlight", ambient="0.45 0.45 0.45", diffuse="0.6 0.6 0.6")
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="unicode")
    if used == 0:
        raise RuntimeError(f"{mesh_dir} に対応するSTLが見つからない")
    return out_xml
