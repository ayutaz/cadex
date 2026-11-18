# SPDX-License-Identifier: LGPL-2.1-or-later
# quadruped8: MG90S を8個使う、FDMで印刷できる8自由度の四足歩行ロボット。
#
# 座標系: X が前方、Y が左、Z が上。単位は mm、度、kg/m^3。
# 各脚は2自由度（股ピッチ・膝ピッチ）。8個の関節軸はすべてワールドの +Y に平行。
# 関節角0 = この形状を組んだ姿勢 = 立位姿勢。正の角度は脚を後ろへ振る向き。
# 部品はすべてワールド座標の立位姿勢で作り、コンポーネントの配置は恒等変換にする。
# 関節の2つのコネクタには同じワールドフレームを与える（docs/MUJOCO.md §7 の手順2）。

p = params(
    body_length=num(110.0, unit="mm", min=96.0, max=160.0, step=1.0, label="胴体の長さ"),
    body_width=num(64.0, unit="mm", min=62.0, max=90.0, step=1.0, label="胴体の幅（左右の股サーボのケース上面の間隔）"),
    body_height=num(24.0, unit="mm", min=20.0, max=40.0, step=0.5, label="胴体の高さ"),
    hip_x=num(42.0, unit="mm", min=36.0, max=70.0, step=0.5, label="股軸の前後位置（中心からの距離）"),
    thigh_length=num(45.0, unit="mm", min=35.0, max=80.0, step=0.5, label="上腿の長さ（股軸から膝軸まで）"),
    shin_length=num(50.0, unit="mm", min=35.0, max=90.0, step=0.5, label="下腿の長さ（膝軸から足先の中心まで）"),
    stand_hip_deg=num(30.0, unit="deg", min=10.0, max=45.0, step=1.0, label="立位での上腿の後傾角"),
    wall=num(2.0, unit="mm", min=1.6, max=3.2, step=0.1, label="印刷部品の肉厚"),
    link_thickness=num(6.5, unit="mm", min=5.0, max=10.0, step=0.5, label="脚リンクの板厚"),
    clearance=num(0.5, unit="mm", min=0.2, max=1.0, step=0.05, label="サーボ収納部のすきま"),
    joint_gap=num(1.0, unit="mm", min=0.5, max=3.0, step=0.1, label="回転部品どうしのすきま"),
    hub_radius=num(12.5, unit="mm", min=11.0, max=18.0, step=0.5, label="関節ハブの半径（十字ホーン腕長10.2 + 肉厚）"),
    screw_hole=num(2.2, unit="mm", min=1.6, max=3.0, step=0.1, label="ホーン固定ネジ用の穴径（M2）"),
    foot_radius=num(6.0, unit="mm", min=4.0, max=10.0, step=0.5, label="足先（TPU）の半径"),
    pla_density=num(800.0, unit="kg/m3", min=300.0, max=1250.0, step=10.0,
                    label="印刷部品の実効密度（推定: PLA 1240 kg/m3、充填率約30%、外周4本）"),
    tpu_density=num(1200.0, unit="kg/m3", min=600.0, max=1250.0, step=10.0, label="足先の密度（TPU 95A、充填率100%）"),
    battery_mass_g=num(30.0, unit="g", min=0.0, max=120.0, step=1.0, label="電池の質量（仮置き。0なら電池を載せない）"),
    # 関節の可動域。clearance スイープ（5度刻み）で測った最初の接触から2ステップ以上手前に置く。
    # 前脚の股 +30 度でシャーシと上腿が接触（+25 度まで干渉なし）→ 上限 +20。
    # 後脚の股 -45 度で前脚の上腿と後脚の足先が接触（-40 度まで干渉なし）→ 下限 -35。
    front_hip_min=num(-45.0, unit="deg", min=-80.0, max=0.0, step=1.0, label="前脚の股関節の下限（前へ振る）"),
    front_hip_max=num(20.0, unit="deg", min=0.0, max=80.0, step=1.0, label="前脚の股関節の上限（後ろへ振る）"),
    rear_hip_min=num(-35.0, unit="deg", min=-80.0, max=0.0, step=1.0, label="後脚の股関節の下限（前へ振る）"),
    rear_hip_max=num(45.0, unit="deg", min=0.0, max=80.0, step=1.0, label="後脚の股関節の上限（後ろへ振る）"),
    # 膝は ±60 度で干渉なし。上限は脚が伸び切る特異姿勢（+57 度付近）の手前。
    knee_min=num(-60.0, unit="deg", min=-90.0, max=0.0, step=1.0, label="膝関節の下限（曲げる）"),
    knee_max=num(45.0, unit="deg", min=0.0, max=90.0, step=1.0, label="膝関節の上限（伸ばす）"),
    servo_sg90=num(0.0, min=0.0, max=1.0, step=1.0, label="1ならSG90、0ならMG90S"),
    rated_fraction=num(0.4, min=0.2, max=1.0, step=0.05,
                       label="連続定格トルク / 失速トルク（推定値、ADR-099の判断に倣う）"),
    friction_loss=num(2.0, unit="N*mm", min=0.0, max=20.0, step=0.5,
                      label="関節の静止摩擦（ギア列、推定値。ランダム化の対象）"),
    policy_on=num(0.0, min=0.0, max=1.0, step=1.0, label="1で学習済みポリシーを宣言して再生する"),
    cmd_min=num(40.0, unit="mm/s", min=0.0, max=300.0, step=5.0, label="指令する前進速度の下限"),
    cmd_max=num(160.0, unit="mm/s", min=10.0, max=400.0, step=5.0, label="指令する前進速度の上限"),
    gait_period=num(0.5, unit="s", min=0.25, max=1.0, step=0.05, label="トロットの周期（位相クロック）"),
    w_trot=num(0.5, min=0.0, max=5.0, step=0.1, label="報酬の重み: 対角ペアの接地位相（v3で1.0）"),
    w_swing=num(0.5, min=0.0, max=5.0, step=0.1, label="報酬の重み: 遊脚の足上げ（v3で1.5）"),
    gait_lift=num(10.0, unit="mm", min=3.0, max=30.0, step=1.0, label="遊脚で払う足上げ高さ"),
    w_effort=num(0.5, min=0.0, max=5.0, step=0.1, label="報酬の罰の重み: トルク（失速トルクで正規化した2乗）"),
    w_effort_rated=num(0.3, min=0.0, max=5.0, step=0.1, label="報酬の罰の重み: 推定連続定格を超えるトルク（4乗）"),
    armature=num(20.0, unit="kg*mm2", min=0.0, max=200.0, step=1.0,
                 label="サーボのロータ慣性（出力軸換算、推定値）"),
)

SKU = "sg90" if p.servo_sg90 >= 0.5 else "mg90s"
VOLT = 4.8  # 失速トルクと速度の両方がデータシートにある電圧

L1 = p.thigh_length
L2 = p.shin_length
W2 = p.body_width / 2.0
t = p.link_thickness
c = p.clearance
g = p.joint_gap
w = p.wall
R = p.hub_radius
th = p.stand_hip_deg
th_r = math.radians(th)
# 足先が股軸の真下に来るように、下腿の絶対角を決める。
psi_r = -math.asin(L1 * math.sin(th_r) / L2)
psi = math.degrees(psi_r)
knee_stand = psi - th  # 参考値（関節角の0はこの姿勢）
# 足先の球の下端が z=0 に来る股軸の高さ。
ZH = L1 * math.cos(th_r) + L2 * math.cos(psi_r) + p.foot_radius

# --- サーボ寸法（カタログ値、MG90Sの一部は推定値。spec['approximate'] を参照） ---
probe = lib.servo(SKU)
S = probe.spec
CASE_H = S["case_height_mm"]
CASE_W = S["body_width_mm"]
FRONT = S["shaft_offset_from_front_mm"]
CASE_L = S["body_length_mm"]
TAB_HALF = S["overall_tab_length_mm"] / 2.0
CASE_CX = FRONT - CASE_L / 2.0  # 軸から見たケース中心（サーボのローカルX）
SPLINE_R = S["spline_dia_mm"] / 2.0
SPLINE_H = S["spline_height_mm"]
LEAD = 6.0
STALL = [e["kg_cm"] for e in S["stall_torque"] if abs(e["volts"] - VOLT) < 1e-6][0] * 98.0665
SPEED = [e["s_per_60_deg"] for e in S["speed"] if abs(e["volts"] - VOLT) < 1e-6][0]

# --- 胴体（下側シャーシ + 上側カバー）。股軸の高さで上下に分割する ---
z_bot = ZH - CASE_W / 2.0 - c - w
z_top = z_bot + p.body_height
BL = p.body_length
hull = part.box(BL, 2.0 * W2, p.body_height, origin=[-BL / 2.0, -W2, z_bot])

hip_servos = {}
bays = []
for name, sx, sy in (("lf", 1, 1), ("rf", 1, -1), ("lh", -1, 1), ("rh", -1, -1)):
    sv = lib.servo(SKU, origin=(sx * p.hip_x, sy * W2, ZH), direction=(0, sy, 0),
                   roll_degrees=(0.0 if sx > 0 else 180.0), label="hip_servo_" + name)
    hip_servos[name] = sv
    bays.append(sv.bay(clearance=c, lead_room=LEAD, ledge=4.0))

# 中央の空洞: 前後の股サーボの配線スペースまで達し、電池・IMUを収める。
cav_x = p.hip_x - (CASE_L - FRONT) - c - LEAD
cavity = part.box(2.0 * cav_x, 2.0 * (W2 - w), p.body_height - 2.0 * w,
                  origin=[-cav_x, -(W2 - w), z_bot + w])

# 制御基板 ESP32-S3-DevKitC-1 の代用品（寸法は推定: 63 x 25.5 x 4 mm）。上面の浅いポケットに載る。
ESP_L, ESP_W, ESP_H = 63.0, 25.5, 4.0
esp_pocket = part.box(ESP_L + 2 * c, ESP_W + 2 * c, 2.0 + c,
                      origin=[-ESP_L / 2.0 - c, -ESP_W / 2.0 - c, z_top - 2.0])
# 上下の締結: 左右の股サーボの間の壁（幅 約6 mm）に M2x10 を前後2本。
# カバーは座ぐり + 通し穴、シャーシは下穴（タップ用）。サーボのタブは上下で挟んで固定する。
BOLT_LEN = 10.0
THREAD_IN = 5.0
z_seat = ZH + BOLT_LEN - THREAD_IN
head = lib.bolt("m2", BOLT_LEN).spec
bolts = []
bolt_holes = []
for i, bx in enumerate((p.hip_x + CASE_CX, -(p.hip_x + CASE_CX))):
    bolts.append(lib.bolt("m2", BOLT_LEN, origin=(bx, 0.0, z_seat), direction=(0, 0, 1), label="body_bolt_%d" % i))
    bolt_holes += [
        part.cylinder(head["head_dia_mm"] / 2.0 + 0.2, z_top - z_seat + 1.0, origin=[bx, 0.0, z_seat]),
        part.cylinder(lib.clearance_hole("m2") / 2.0, z_seat - ZH + 0.01, origin=[bx, 0.0, ZH]),
        part.cylinder(lib.tap_drill("m2") / 2.0, THREAD_IN + 1.0, origin=[bx, 0.0, ZH - THREAD_IN - 1.0]),
    ]
body_cut = part.cut(hull, bays + [cavity, esp_pocket] + bolt_holes)
upper_half = part.box(BL + 2, 2.0 * W2 + 2, p.body_height, origin=[-BL / 2.0 - 1, -W2 - 1, ZH])
lower_half = part.box(BL + 2, 2.0 * W2 + 2, p.body_height, origin=[-BL / 2.0 - 1, -W2 - 1, ZH - p.body_height])
chassis = part.cut(body_cut, upper_half, label="chassis")
cover = part.cut(body_cut, lower_half, label="cover")

esp32 = part.box(ESP_L, ESP_W, ESP_H, origin=[-ESP_L / 2.0, -ESP_W / 2.0, z_top - 2.0], label="esp32s3_standin")
# IMU（GY-521 / MPU-6050 の代用品、寸法は推定: 21 x 16 x 3 mm）。空洞の床、+Y寄り。
IMU_L, IMU_W, IMU_H = 21.0, 16.0, 3.0
imu_y0 = W2 - w - IMU_W - 0.5
imu = part.box(IMU_L, IMU_W, IMU_H, origin=[-IMU_L / 2.0, imu_y0, z_bot + w], label="imu_standin")
# 電池の代用品（40 x 26 x 8 mm、質量はパラメータ）。空洞の床の中央。
# battery_mass_g が 0 なら電池を載せない: 部品・コンポーネント・溶接・質量・ランダム化をどれも作らない
# （胴体を小さくした機体では、この箱がシャーシ・カバー・IMU とぶつかる）。
HAS_BAT = p.battery_mass_g > 0.0
BAT = (40.0, 26.0, 8.0)
if HAS_BAT:
    battery = part.box(BAT[0], BAT[1], BAT[2], origin=[-BAT[0] / 2.0, -BAT[1] / 2.0, z_bot + w], label="battery_standin")
    bat_density = max(p.battery_mass_g, 0.1) * 1.0e6 / (BAT[0] * BAT[1] * BAT[2])


def rot_y(shape, deg, pivot):
    return part.transform(shape, rotation_axis=[0, 1, 0], rotation_degrees=deg, pivot=pivot)


def y_slab(y_in, thick, sy):
    # 内側の面 y_in（絶対値）から外へ thick の範囲を、符号 sy 側で返す: (下端, 厚み)
    lo = y_in if sy > 0 else -(y_in + thick)
    return lo, thick


result = {"chassis": chassis, "cover": cover, "esp32": esp32, "imu": imu}
if HAS_BAT:
    result["battery"] = battery
legs = {}
for name, sx, sy in (("lf", 1, 1), ("rf", 1, -1), ("lh", -1, 1), ("rh", -1, -1)):
    hip = [sx * p.hip_x, 0.0, ZH]
    hx = hip[0]
    hsv = hip_servos[name]
    hhorn = hsv.horn("cross", label="hip_horn_" + name)
    # 上腿の板: 胴体の側面から joint_gap 離して外側に置く。
    y_th_in = W2 + g
    y_lo, _ = y_slab(y_th_in, t, sy)
    y_kt = sy * (y_th_in + t)  # 膝サーボのケース上面 = 上腿の外面
    kx = hx - L1 * math.sin(th_r)
    kz = ZH - L1 * math.cos(th_r)
    # 上腿は「真下に伸びた姿勢」で作り、股軸まわりに th 回す。
    hw = CASE_W / 2.0 + c + w  # 膝サーボを包むハウジングの半幅
    beam = part.box(2.0 * hw, t, L1, origin=[hx - hw, y_lo, ZH - L1])
    hub = part.cylinder(R, t, origin=[hx, y_lo, ZH], direction=[0, 1, 0])
    kcap = part.cylinder(hw, t, origin=[hx, y_lo, ZH - L1], direction=[0, 1, 0])
    # 膝サーボのケースは膝から足先側へ伸ばす（サーボのローカル +X が股側）。
    # 股側へ伸ばすと、前脚を後ろへ振ったときにハウジングが胴体の底へ潜る（v1のスイープで +20° で接触）。
    h_top = (TAB_HALF + CASE_CX) + c + w      # 膝軸から股側へ（タブの先端まで）
    h_bot = (CASE_L - FRONT) + c + LEAD + w   # 膝軸から足先側へ（ケース後端 + リード線の余裕）
    depth = CASE_H + c + w                    # 膝サーボのケースの奥行き（内側へ）
    h_y_lo = (abs(y_kt) - depth) if sy > 0 else -abs(y_kt)
    housing = part.box(2.0 * hw, depth, h_top + h_bot,
                       origin=[hx - hw, h_y_lo, ZH - L1 - h_bot])
    thigh_raw = rot_y(part.fuse([beam, hub, kcap, housing]), th, hip)
    # ロール角: 左はサーボのローカル +X が股側（-d1）を向くよう th-90、右は 90-th。
    ksv = lib.servo(SKU, origin=(kx, y_kt, kz), direction=(0, sy, 0),
                    roll_degrees=((th - 90.0) if sy > 0 else (90.0 - th)), label="knee_servo_" + name)
    khorn = ksv.horn("cross", label="knee_horn_" + name)
    # スプラインとホーンの軸穴の逃げ（ホーンのハブ高さ分まで抜かないと、軸穴に材料が残る）
    HUB_H = hhorn.spec["hub_height_mm"]
    spline_gap = part.cylinder(SPLINE_R + c, SPLINE_H + HUB_H, origin=[hx, sy * W2, ZH], direction=[0, sy, 0])
    hip_access = part.cylinder(p.screw_hole / 2.0 + 0.8, t + 2.0, origin=[hx, sy * (y_th_in - 1.0), ZH], direction=[0, sy, 0])
    thigh = part.cut(thigh_raw, [ksv.bay(clearance=c, lead_room=LEAD, ledge=4.0), hhorn.body, spline_gap, hip_access],
                     label="thigh_" + name)

    # 下腿: 膝ホーンの外側の板。真下に伸びた姿勢で作り、膝軸まわりに psi 回す。
    y_sh_in = abs(y_kt) + g
    ys_lo, _ = y_slab(y_sh_in, t, sy)
    knee = [kx, 0.0, kz]
    s_hub = part.cylinder(R, t, origin=[kx, ys_lo, kz], direction=[0, 1, 0])
    s_beam = part.box(10.0, t, L2, origin=[kx - 5.0, ys_lo, kz - L2])
    s_tip = part.cylinder(5.0, t, origin=[kx, ys_lo, kz - L2], direction=[0, 1, 0])
    shin_raw = rot_y(part.fuse([s_hub, s_beam, s_tip]), psi, knee)
    fx = kx - L2 * math.sin(psi_r)
    fz = kz - L2 * math.cos(psi_r)
    fy = sy * (y_sh_in + t / 2.0)
    foot = part.sphere(p.foot_radius, center=[fx, fy, fz], label="foot_" + name)
    k_spline_gap = part.cylinder(SPLINE_R + c, SPLINE_H + HUB_H, origin=[kx, y_kt, kz], direction=[0, sy, 0])
    k_access = part.cylinder(p.screw_hole / 2.0 + 0.8, t + 2.0, origin=[kx, sy * (y_sh_in - 1.0), kz], direction=[0, sy, 0])
    shin = part.cut(shin_raw, [khorn.body, k_spline_gap, k_access, foot], label="shin_" + name)

    legs[name] = {"sx": sx, "sy": sy, "hip": hip, "knee": knee, "foot": [fx, fy, fz],
                  "hsv": hsv, "hhorn": hhorn, "ksv": ksv, "khorn": khorn,
                  "thigh": thigh, "shin": shin, "foot_shape": foot}
    result["hip_servo_" + name] = hsv.body
    result["hip_horn_" + name] = hhorn.body
    result["thigh_" + name] = thigh
    result["knee_servo_" + name] = ksv.body
    result["knee_horn_" + name] = khorn.body
    result["shin_" + name] = shin
    result["foot_" + name] = foot

# --- アセンブリ ---
HINGE = {"axis": [1, 0, 0], "angle_degrees": -90}  # コネクタの +Z をワールドの +Y に向ける


def hinge_frame(point):
    return {"position": list(point), "axis": HINGE["axis"], "angle_degrees": HINGE["angle_degrees"]}


def hip_limits(sx):
    return [p.front_hip_min, p.front_hip_max] if sx > 0 else [p.rear_hip_min, p.rear_hip_max]


def weld(a, b, tag):
    j = assembly.joint("fixed", assembly.connector(a, "origin"), assembly.connector(b, "origin"), label=tag)
    result["weld_" + tag] = j
    return j


comps = []
joints = []
bodies = []
c_chassis = assembly.component(chassis, appearance="shell", role="frame", label="chassis")
c_cover = assembly.component(cover, appearance="shell", role="frame", label="cover")
c_esp = assembly.component(esp32, appearance="mechanism", role="hardware", label="esp32s3")
c_imu = assembly.component(imu, appearance="accent", role="hardware", label="imu")
c_bat = assembly.component(battery, appearance="mechanism", role="hardware", label="battery") if HAS_BAT else None
result.update({"c_chassis": c_chassis, "c_cover": c_cover, "c_esp": c_esp, "c_imu": c_imu})
comps += [c_chassis, c_cover, c_esp, c_imu]
if HAS_BAT:
    result["c_bat"] = c_bat
    comps.append(c_bat)
joints += [weld(c_chassis, c_cover, "cover"), weld(c_cover, c_esp, "esp"), weld(c_chassis, c_imu, "imu")]
if HAS_BAT:
    joints.append(weld(c_chassis, c_bat, "battery"))
contacts = []
bolt_comps = []
for i, b in enumerate(bolts):
    key = "body_bolt_%d" % i
    result[key] = b.body
    cb = assembly.component(b.body, appearance="mechanism", role="hardware", label=key)
    result["c_" + key] = cb
    comps.append(cb)
    bolt_comps.append((cb, b.spec["density_kg_m3"]))
    joints.append(weld(c_cover, cb, key))

body_box = [BL, 2.0 * W2, p.body_height]
body_mid = [0.0, 0.0, z_bot + p.body_height / 2.0]
bodies += [
    assembly.body(c_chassis, density_kg_m3=p.pla_density,
                  collision=assembly.collision("box", size_mm=body_box, offset={"position": body_mid}, label="body_box")),
    assembly.body(c_cover, density_kg_m3=p.pla_density),
    assembly.body(c_esp, density_kg_m3=9.0e-3 / (ESP_L * ESP_W * ESP_H * 1e-9)),   # 約9 g（推定）
    assembly.body(c_imu, density_kg_m3=2.0e-3 / (IMU_L * IMU_W * IMU_H * 1e-9)),   # 約2 g（推定）
]
if HAS_BAT:
    bodies.append(assembly.body(c_bat, density_kg_m3=bat_density))
bodies += [assembly.body(cb, density_kg_m3=dens) for cb, dens in bolt_comps]

drives = []
drive_by = {}
dynamics = []
hip_joints = {}
knee_joints = {}
feet = {}
for name in ("lf", "rf", "lh", "rh"):
    L = legs[name]
    c_hsv = assembly.component(L["hsv"].body, appearance="mechanism", role="hardware", label="hip_servo_" + name)
    c_hhorn = assembly.component(L["hhorn"].body, appearance="mechanism", role="hardware", label="hip_horn_" + name)
    c_thigh = assembly.component(L["thigh"], appearance="shell", role="link", label="thigh_" + name)
    c_ksv = assembly.component(L["ksv"].body, appearance="mechanism", role="hardware", label="knee_servo_" + name)
    c_khorn = assembly.component(L["khorn"].body, appearance="mechanism", role="hardware", label="knee_horn_" + name)
    c_shin = assembly.component(L["shin"], appearance="shell", role="link", label="shin_" + name)
    c_foot = assembly.component(L["foot_shape"], appearance="accent", role="link", label="foot_" + name)
    for key, cv in (("c_hip_servo_", c_hsv), ("c_hip_horn_", c_hhorn), ("c_thigh_", c_thigh),
                    ("c_knee_servo_", c_ksv), ("c_knee_horn_", c_khorn), ("c_shin_", c_shin), ("c_foot_", c_foot)):
        result[key + name] = cv
    comps += [c_hsv, c_hhorn, c_thigh, c_ksv, c_khorn, c_shin, c_foot]
    # サーボの出力軸（スプライン先端）にホーンが載る: 意図した接触
    contacts += [(c_hsv, c_hhorn), (c_ksv, c_khorn)]
    joints += [weld(c_chassis, c_hsv, "hip_servo_" + name), weld(c_thigh, c_hhorn, "hip_horn_" + name),
               weld(c_thigh, c_ksv, "knee_servo_" + name), weld(c_shin, c_khorn, "knee_horn_" + name),
               weld(c_shin, c_foot, "foot_" + name)]
    j_hip = assembly.joint("revolute",
                           assembly.connector(c_chassis, "origin", offset=hinge_frame(L["hip"])),
                           assembly.connector(c_thigh, "origin", offset=hinge_frame(L["hip"])),
                           angle_limits_degrees=hip_limits(L["sx"]), label="hip_" + name)
    j_knee = assembly.joint("revolute",
                            assembly.connector(c_thigh, "origin", offset=hinge_frame(L["knee"])),
                            assembly.connector(c_shin, "origin", offset=hinge_frame(L["knee"])),
                            angle_limits_degrees=[p.knee_min, p.knee_max], label="knee_" + name)
    result["hip_" + name] = j_hip
    result["knee_" + name] = j_knee
    joints += [j_hip, j_knee]
    hip_joints[name] = j_hip
    knee_joints[name] = j_knee
    feet[name] = c_foot

    # 衝突形状（コンポーネント座標 = ワールド座標、立位姿勢）
    hip = L["hip"]
    knee = L["knee"]
    ft = L["foot"]
    sy = L["sy"]
    th_mid = [(hip[0] + knee[0]) / 2.0, sy * (W2 + g + t / 2.0), (hip[2] + knee[2]) / 2.0]
    sh_mid = [(knee[0] + ft[0]) / 2.0, ft[1], (knee[2] + ft[2]) / 2.0]
    bodies += [
        assembly.body(c_hsv, density_kg_m3=L["hsv"].spec["effective_density_kg_m3"]),
        assembly.body(c_hhorn, density_kg_m3=L["hhorn"].spec["density_kg_m3"]),
        assembly.body(c_thigh, density_kg_m3=p.pla_density,
                      collision=assembly.collision("capsule", radius_mm=t, length_mm=L1,
                                                   offset={"position": th_mid, "axis": [0, 1, 0], "angle_degrees": th},
                                                   contact_group=1, label="thigh_cap_" + name)),
        assembly.body(c_ksv, density_kg_m3=L["ksv"].spec["effective_density_kg_m3"]),
        assembly.body(c_khorn, density_kg_m3=L["khorn"].spec["density_kg_m3"]),
        assembly.body(c_shin, density_kg_m3=p.pla_density,
                      collision=assembly.collision("capsule", radius_mm=4.0, length_mm=max(L2 - 2.0 * p.foot_radius, 1.0),
                                                   offset={"position": sh_mid, "axis": [0, 1, 0], "angle_degrees": psi},
                                                   contact_group=1, label="shin_cap_" + name)),
        assembly.body(c_foot, density_kg_m3=p.tpu_density,
                      collision=assembly.collision("sphere", radius_mm=p.foot_radius, offset={"position": ft},
                                                   friction=0.9, contact_group=2, label="foot_ball_" + name)),
    ]
    # 角度指令型PWMサーボ: 失速トルクでクランプした位置アクチュエータ（剛性 = 5度で失速トルク）。
    for kind, j in (("hip", j_hip), ("knee", j_knee)):
        lo, hi = hip_limits(L["sx"]) if kind == "hip" else (p.knee_min, p.knee_max)
        a = assembly.actuator(j, kind="position", control_deg="0",
                              stiffness_nmm_per_deg=STALL / 5.0, damping_nmms_per_deg=STALL / 100.0,
                              torque_limit_nmm=STALL, rated_torque_nmm=STALL * p.rated_fraction,
                              command_limits_degrees=[lo, hi], label=kind + "_drive_" + name)
        drives.append(a)
        drive_by[kind + "_" + name] = a
        # 失速トルク / 無負荷速度 の粘性: 全力時にデータシートの速度で飽和する（ADR-409）。
        dynamics.append(assembly.joint_dynamics(j, damping_nmms_per_deg=STALL / (60.0 / SPEED),
                                                armature_kgmm2=p.armature, friction_loss_nmm=p.friction_loss,
                                                label=kind + "_dyn_" + name))

asm = assembly.assembly(comps, joints, sweep_step_degrees=5.0, contacts=contacts, label="quadruped8")
diag = assembly.solve(asm)
result["asm"] = asm
result["diag"] = diag

imu_sensor = assembly.sensor(c_imu, "imu", name="imu")
obs = [
    assembly.observation(c_imu, "component_orientation", name="rot", sensor=imu_sensor),
    assembly.observation(c_imu, "component_angular_velocity", name="gyro", sensor=imu_sensor),
    assembly.observation(c_chassis, "component_linear_velocity", name="vel", role="privileged"),
    assembly.observation(c_chassis, "component_position", name="pos", role="privileged"),
]
for name in ("lf", "rf", "lh", "rh"):
    obs.append(assembly.observation(hip_joints[name], "position", name="q_hip_" + name, role="privileged"))
    obs.append(assembly.observation(knee_joints[name], "position", name="q_knee_" + name, role="privileged"))
    obs.append(assembly.observation(feet[name], "contact", name="ft_" + name, role="privileged"))
    obs.append(assembly.observation(feet[name], "centre_of_mass", name="fc_" + name, role="privileged"))
for key, a in drive_by.items():
    # 前回の関節指令（コントローラ自身が知っている値なのでセンサ不要）と、サーボの出力トルク（実機では測れない）
    obs.append(assembly.observation(a, "command", name="c_" + key))
    obs.append(assembly.observation(a, "actuator_force", name="tau_" + key, role="privileged"))
model = assembly.mjcf(asm, bodies, actuators=drives, joint_dynamics=dynamics, observations=obs,
                      solver_step_s=0.002, label="quadruped8")
result["model"] = model

# --- 強化学習タスク（docs/XSCRIPT.md: assembly.task） ---
# 方策（policy）が読むのは実機で得られるものだけ:
#   IMUの姿勢 rot_q* と角速度 gyro_*、前回の関節指令 c_*（8）、目標速度 cmd、歩容の位相 ref_sin/ref_cos。
# 関節角・胴体の速度と位置・足の接地・トルクは privileged（報酬・終了条件・critic だけが読む）。
KEYS = list(drive_by)
UP = "(1 - 2*(rot_qx^2 + rot_qy^2))"
cmd = assembly.goal("cmd", kind="speed", between=[p.cmd_min, p.cmd_max], resample_seconds=4.0)
FEET = [feet["lf"], feet["rf"], feet["lh"], feet["rh"]]
# Cadex の歩容リファレンス（assembly.gait）は1脚3関節以上が前提で、2自由度の脚には使えない。
# 代わりに位相クロック（コントローラ自身のタイマー。実機でも同じものを使える）を方策に与え、
# トロットのリズムは報酬で教える: 対角ペア (LF,RH) は clock_sin > 0 で立脚、(RF,LH) は < 0 で立脚。
clock = assembly.goal("clock", kind="phase", period_seconds=p.gait_period)
STANCE_A = "(0.5 + 0.5*tanh(5*clock_sin))"   # LF, RH が立脚である度合い
STANCE_B = "(0.5 - 0.5*tanh(5*clock_sin))"   # RF, LH
FOOT_R = p.foot_radius


def total(fmt):
    return " + ".join(fmt.format(k=k) for k in KEYS)


rewards = [
    assembly.reward("1", weight=1.0, label="alive"),
    assembly.reward(UP, weight=1.0, label="upright"),
    # 目標速度への追従。転倒中は払わない（UP を掛ける）。速度はワールドX（mm/s）。
    assembly.reward(UP + " * exp(-((vel_x - cmd) / 40)^2)", weight=3.0, label="track_speed"),
    assembly.reward("(vel_y / 100)^2", weight=-1.0, label="lateral"),
    assembly.reward("(gyro_z / 60)^2", weight=-0.5, label="yaw_rate"),
    assembly.reward("(gyro_x / 100)^2 + (gyro_y / 100)^2", weight=-0.2, label="body_rock"),
    assembly.reward("(pos_z / 15)^2", weight=-0.5, label="height"),
    # トロットのリズム: 正しい組が接地していれば +、逆の組が接地していれば −（最大 2）
    assembly.reward("tanh(5*clock_sin) * ((ft_lf_down + ft_rh_down) - (ft_rf_down + ft_lh_down))",
                    weight=p.w_trot, label="trot_contact"),
    # 遊脚の足上げ（足先の中心の高さ。立位で足の半径）。gait_lift まで払う。
    assembly.reward(" + ".join(
        "%s * tanh((fc_%s_z - %.2f) / %.2f)" % (STANCE_B if leg in ("lf", "rh") else STANCE_A, leg, FOOT_R, p.gait_lift)
        for leg in ("lf", "rf", "lh", "rh")), weight=p.w_swing, label="swing_lift"),
    # 滑らかさ: 指令の変化率（deg/s）とトルク（失速トルクで正規化）
    assembly.reward("(" + total("(c_{k}_rate / 600)^2") + ") / 8", weight=-0.5, label="action_rate"),
    assembly.reward("(" + total("(tau_{k} / %.3f)^2" % STALL) + ") / 8", weight=-p.w_effort, label="effort"),
    # v2: 推定連続定格で正規化したトルクの4乗。定格を超える区間だけを強く罰する（v1は左前脚が定格の1.4〜1.5倍）。
    assembly.reward("(" + total("(tau_{k} / %.3f)^4" % (STALL * p.rated_fraction)) + ") / 8",
                    weight=-p.w_effort_rated, label="effort_rated"),
    # v2: 胴体の傾き（5度で1）。v1 は最大傾斜が約20度だった。
    assembly.reward("(rot_qx^2 + rot_qy^2) / 0.0019", weight=-0.1, label="tilt"),
]
terminations = [
    assembly.termination(UP, below=0.7, label="fell"),          # 約45度以上傾いた
    assembly.termination("pos_z", below=-30.0, label="sank"),   # 胴体が立位から30 mm以上下がった
]
randomisation = [
    assembly.randomise(c_chassis, "mass", scale=[0.85, 1.15], label="rand_chassis_mass"),
    assembly.randomise(c_cover, "mass", scale=[0.85, 1.15], label="rand_cover_mass"),
]
if HAS_BAT:
    randomisation.append(assembly.randomise(c_bat, "mass", scale=[0.6, 1.4], label="rand_battery_mass"))  # 重心位置も動く
for name in ("lf", "rf", "lh", "rh"):
    for j in (hip_joints[name], knee_joints[name]):
        randomisation.append(assembly.randomise(j, "damping", scale=[0.7, 1.3]))      # サーボ速度のばらつき
        randomisation.append(assembly.randomise(j, "friction_loss", scale=[0.5, 2.0]))
reset = [assembly.reset_variation(c_chassis, tilt_degrees=[0.0, 3.0], height_mm=[4.0, 6.0])]  # 3度傾けると足が3.2 mm沈むため持ち上げる
spec = assembly.success(
    [
        {"id": "completes", "metric": "completed", "min": 1},
        {"id": "upright", "metric": "max_tilt_deg", "max": 25},
        {"id": "walks", "metric": "steps_min", "min": 4},
        {"id": "speed", "metric": "speed_ratio", "min": 0.5, "max": 1.5},
        {"id": "rated", "metric": "effort_rms_rated_ratio_max", "max": 1.0},
    ],
    seeds=[2101, 2102, 2103, 2104, 2105, 2106, 2107, 2108, 2109, 2110],
    feet=FEET, episode_seconds=8.0,
)
task = assembly.task(model, actions=[drive_by[k] for k in KEYS], reward=rewards, termination=terminations,
                     randomisation=randomisation, reset_variation=reset, goals=[cmd, clock], success=spec,
                     episode_seconds=8.0, control_hz=50, label="walk")
result["walk"] = task
if p.policy_on >= 0.5:
    policy = assembly.policy(task, weights="walk.cxpolicy", sha256="0" * 64)
    result["policy"] = policy
    result["run"] = assembly.rollout(policy, frames_per_second=50, seed=2101)
