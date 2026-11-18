# SPDX-License-Identifier: LGPL-2.1-or-later
# quadruped8: MG90S を8個使う、FDMで印刷できる8自由度の四足歩行ロボット。
#
# 座標系: X が前方、Y が左、Z が上。単位は mm、度、kg/m^3。
# 各脚は2自由度（股ピッチ・膝ピッチ）。8個の関節軸はすべてワールドの +Y に平行。
# tail_dof を 1・2 にすると、胴体の後端にしっぽ（左右の軸 +Z、その先で上下の軸 +Y）のサーボが足される。
# rear_legs を 1 にすると、後ろの2本の代わりに胴体の後端の中央に1本（hc）を置いた3本足になる（6自由度）。
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
    # 1 は mini までの形（base・leg120・mini などの生成物はこの形）。STL で調べると脚が組めない
    # （docs/assembly-guide.md §0: 膝サーボのタブが閉じた溝に入る、ホーンの凹みが板の中に閉じている、
    # 膝サーボのリード線の出口が無い）。2 はそれを直した形（mini-r2）。
    leg_rev=num(1.0, min=1.0, max=2.0, step=1.0,
                label="脚の組立の版（1: mini まで、2: 膝サーボを上腿の内側から入れ、ホーンの凹みをサーボ側の面に開ける）"),
    # しっぽ（胴体の後端の中央）。0（既定）ならしっぽの部品・関節・アクチュエータをどれも作らず、出力は変わらない。
    tail_dof=num(0.0, min=0.0, max=2.0, step=1.0, label="しっぽの関節の数（0: なし、1: 左右、2: 左右 + その先で上下）"),
    tail_length=num(60.0, unit="mm", min=30.0, max=100.0, step=1.0,
                    label="しっぽの長さ（左右の軸から先端のおもりの中心まで、前後方向）"),
    tail_tip_mass_g=num(5.0, unit="g", min=0.0, max=20.0, step=0.5, label="しっぽの先端のおもりの質量（0 なら付けない）"),
    # しっぽの可動域は仮の値（clearance スイープで決め直す）。
    tail_yaw_limit=num(45.0, unit="deg", min=5.0, max=90.0, step=1.0, label="しっぽの左右の可動域（±）"),
    tail_pitch_min=num(-30.0, unit="deg", min=-80.0, max=0.0, step=1.0, label="しっぽの上下の下限（下げる）"),
    tail_pitch_max=num(45.0, unit="deg", min=0.0, max=80.0, step=1.0, label="しっぽの上下の上限（上げる）"),
    # 3本足（rear_legs=1）: 前の2本はそのまま、後ろは胴体の後端の中央に1本（脚の名前 hc）。しっぽとは一緒に使えない。
    rear_legs=num(2.0, min=1.0, max=2.0, step=1.0, label="後ろの脚の数（2: 四足、1: 後ろ中央の1本の3本足）"),
    # 3本足の歩き方の報酬。0: 決めない（接地のリズムの報酬なし、足上げは全脚に払う）、
    # 1: 波（lf → rf → hc の順に 1/3 周期ずつ立脚をずらす）。四足はトロットで、1 は使えない。
    gait_mode=num(0.0, min=0.0, max=1.0, step=1.0, label="3本足の歩き方の報酬（0: 決めない、1: 波）"),
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
R2 = p.leg_rev >= 1.5
STALL = [e["kg_cm"] for e in S["stall_torque"] if abs(e["volts"] - VOLT) < 1e-6][0] * 98.0665
SPEED = [e["s_per_60_deg"] for e in S["speed"] if abs(e["volts"] - VOLT) < 1e-6][0]



def horn_seat_of(hs):
    """leg_rev 2: サーボのケース上面から、ホーンを受けるリンクの内面までの距離（hs はホーンの spec）。

    ホーン（CAD はスプラインの先端に載る形。腕はハブの外側の端にある）の外面から、
    腕の厚み + すきま だけサーボ側にリンクの内面を置き、凹みをその面に開ける。
    実物のホーンはスプラインにかぶさるので CAD よりサーボ側に寄る（最大でスプラインの高さ、推定）。
    そのときはリンクがホーンと一緒に寄るだけで、凹みの形は変わらない。
    """
    return max(g, SPLINE_H + hs["hub_height_mm"] - (hs["arm_thickness_mm"] + c))


# --- 脚の数（rear_legs）と歩き方の報酬（gait_mode） ---
REAR_LEGS = int(round(p.rear_legs))
if REAR_LEGS not in (1, 2):
    raise ValueError("rear_legs は 1 か 2: %r" % p.rear_legs)
TRI = REAR_LEGS == 1
GAIT_MODE = int(round(p.gait_mode))
if GAIT_MODE not in (0, 1):
    raise ValueError("gait_mode は 0 か 1: %r" % p.gait_mode)
if GAIT_MODE == 1 and not TRI:
    raise ValueError("gait_mode=1（波）は3本足（rear_legs=1）だけ。四足の歩き方はトロット")
if TRI and p.tail_dof >= 0.5:
    raise ValueError("しっぽ（tail_dof > 0）と3本足（rear_legs=1）は一緒に使えない")

# --- 胴体（下側シャーシ + 上側カバー）。股軸の高さで上下に分割する ---
z_bot = ZH - CASE_W / 2.0 - c - w
z_top = z_bot + p.body_height
BL = p.body_length
hull = part.box(BL, 2.0 * W2, p.body_height, origin=[-BL / 2.0, -W2, z_bot])

# 脚ごとの (名前, 前後の符号 sx, 左右の符号 sy, 股軸の x, 股サーボのケース上面の |y|)。
# 左右の符号 sy は、股サーボの軸と脚の板が並ぶ向き（+Y なら、股サーボ → 上腿 → 膝サーボ → 下腿の順に +Y へ）。
# 3本足の後ろの脚 hc は、左の脚（sy=+1）と同じ組み方で、上腿の板の中央面を y = 0 に置く。
# 股軸は胴体の後端からハブの半径 + joint_gap 後ろ。股サーボは胴体の後端に足した台に入る。
if TRI:
    HC_X = -(BL / 2.0 + R + g)
    seat0 = horn_seat_of(probe.horn("cross").spec) if R2 else g
    HC_YB = -t / 2.0 - seat0           # 上腿の内面 = -t/2 になる、股サーボのケース上面の y
    LEG_SPECS = [("lf", 1, 1, p.hip_x, W2), ("rf", 1, -1, p.hip_x, W2), ("hc", -1, 1, HC_X, HC_YB)]
else:
    LEG_SPECS = [("lf", 1, 1, p.hip_x, W2), ("rf", 1, -1, p.hip_x, W2),
                 ("lh", -1, 1, -p.hip_x, W2), ("rh", -1, -1, -p.hip_x, W2)]
LEG_NAMES = [spec[0] for spec in LEG_SPECS]
LEG_YB = {spec[0]: spec[4] for spec in LEG_SPECS}
hip_servos = {}
bays = []
LEG_X = {spec[0]: spec[3] for spec in LEG_SPECS}
for name, sx, sy, hx_leg, yb in LEG_SPECS:
    sv = lib.servo(SKU, origin=(hx_leg, sy * yb, ZH), direction=(0, sy, 0),
                   roll_degrees=(0.0 if sx > 0 else 180.0), label="hip_servo_" + name)
    hip_servos[name] = sv
    bays.append(sv.bay(clearance=c, lead_room=LEAD, ledge=4.0))

# --- しっぽの付け根（tail_dof >= 1） ---
# 左右のサーボ: 出力軸はワールドの +Z（上）、ケースの上面（軸の基準点）は胴体の上面と同じ高さ、左右の中央。
# ロール 180 度でケースは軸から前（胴体の側）へ伸び、リード線の逃げは前（胴体の側）へ向く。
# 軸の前後の位置: ケースの前端が胴体の後端に来る位置か、タブの先が後ろの股サーボのケースから
# すきま 2つと壁 1枚ぶん離れる位置の、後ろのほう（mini-r2 の値では後者。前者だとタブが股サーボに重なる）。
# リード線の逃げの先から後ろの股サーボの収納部まで、通り道を開ける（線は股サーボの収納部から胴体の空洞へ）。
# サーボは胴体の後端に足した台（箱）の収納部に入る。台は胴体と一体（シャーシとカバーに分かれる）。
TAIL_DOF = int(round(p.tail_dof))
if TAIL_DOF not in (0, 1, 2):
    raise ValueError("tail_dof は 0・1・2 のどれか: %r" % p.tail_dof)
TAIL = TAIL_DOF > 0
# サーボを包む台・ハウジングの、軸から両側への長さ（膝サーボのハウジングと同じ式）
H_SHAFT = (TAB_HALF + CASE_CX) + c + w      # 軸の側（サーボのローカル +X、タブの先端まで）
H_LEAD = (CASE_L - FRONT) + c + LEAD + w    # ケース後端の側（ローカル -X、リード線の余裕まで）
HW = CASE_W / 2.0 + c + w                   # 台・ハウジングの半幅
if TAIL:
    rear_hip_case = -(p.hip_x + FRONT)                       # 後ろの股サーボ（ロール 180 度）のケースの後ろの面
    ty_x = min(-BL / 2.0 - (CASE_L - FRONT), rear_hip_case - 2.0 * c - w - (TAB_HALF - CASE_CX))
    TY = [ty_x, 0.0, z_top]   # 左右の軸の基準点
    tail_yaw_sv = lib.servo(SKU, origin=tuple(TY), direction=(0, 0, 1), roll_degrees=180.0, label="tail_yaw_servo")
    bays.append(tail_yaw_sv.bay(clearance=c, lead_room=LEAD, ledge=4.0))
    # リード線の通り道: 逃げ（ケースの前の面から LEAD、タブの下 4 mm より下）の先から、股サーボの収納部へ 1 mm 入るまで。
    # 床（胴体の底の肉厚）は残す。
    lead_x0 = TY[0] + (CASE_L - FRONT) + c + LEAD - 1.0
    lead_x1 = rear_hip_case - c + 1.0
    if lead_x1 > lead_x0:
        lead_z0 = z_bot + w
        lead_z1 = z_top + S["mount_hole_z_mm"] - c - 4.0
        bays.append(part.box(lead_x1 - lead_x0, CASE_W + 2.0 * c, lead_z1 - lead_z0,
                             origin=[lead_x0, -(CASE_W / 2.0 + c), lead_z0]))
    # 前端は胴体の後端から肉厚ぶん中へ重ねる（胴体の箱と確実につながる）
    tail_mount = part.box((-BL / 2.0 + w) - (TY[0] - H_SHAFT), 2.0 * HW, p.body_height,
                          origin=[TY[0] - H_SHAFT, -HW, z_bot])
    hull = part.fuse([hull, tail_mount])
if TRI:
    # 3本足の後ろの脚の股サーボの台: ケース（ケース上面 y = HC_YB から -Y へ CASE_H）を包み、+Y の面は開ける。
    # ロール 180 度でケースは軸から前へ伸び、リード線の逃げは胴体の後ろの壁へ入る。前端は胴体へ肉厚ぶん重ねる。
    hc_depth = CASE_H + c + w
    hc_mount = part.box((-BL / 2.0 + w) - (HC_X - H_SHAFT), hc_depth, p.body_height,
                        origin=[HC_X - H_SHAFT, HC_YB - hc_depth, z_bot])
    hull = part.fuse([hull, hc_mount])

# 中央の空洞: 前後の股サーボの配線スペースまで達し、電池・IMUを収める。
cav_x = p.hip_x - (CASE_L - FRONT) - c - LEAD
cavity = part.box(2.0 * cav_x, 2.0 * (W2 - w), p.body_height - 2.0 * w,
                  origin=[-cav_x, -(W2 - w), z_bot + w])
if TRI:
    # 後ろの股サーボ hc のリード線の通り道: 逃げ（ケースの前の面から LEAD）の先から胴体の空洞へ 1 mm 入るまで。
    # 四足の後ろの股サーボは逃げが空洞に届くが、hc は股軸が胴体の後端より後ろなので届かない。
    # 左右は胴体の側面の壁の内側から、タブの下 4 mm まで（逃げと同じ上限）。高さはサーボの収納部の幅。
    hc_lead_x0 = HC_X + (CASE_L - FRONT) + c + LEAD - 1.0
    hc_lead_y1 = HC_YB + S["mount_hole_z_mm"] - c - 4.0
    bays.append(part.box(-cav_x + 1.0 - hc_lead_x0, hc_lead_y1 + (W2 - w), CASE_W + 2.0 * c,
                         origin=[hc_lead_x0, -(W2 - w), ZH - CASE_W / 2.0 - c]))

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
if TAIL or TRI:
    # しっぽ・3本足の台は胴体の箱の外へはみ出すので、分ける箱を台まで広げる（はみ出しが両方に残らないように）
    sx0, sy0 = -BL / 2.0, -W2
    if TAIL:
        sx0 = min(sx0, TY[0] - H_SHAFT)
    if TRI:
        sx0 = min(sx0, HC_X - H_SHAFT)
        sy0 = min(sy0, HC_YB - hc_depth)
    split = (BL / 2.0 - sx0 + 2.0, W2 - sy0 + 2.0)
    upper_half = part.box(split[0], split[1], p.body_height, origin=[sx0 - 1.0, sy0 - 1.0, ZH])
    lower_half = part.box(split[0], split[1], p.body_height, origin=[sx0 - 1.0, sy0 - 1.0, ZH - p.body_height])
else:
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


def horn_seat(horn):
    """leg_rev 2: サーボのケース上面から、ホーンを受けるリンクの内面までの距離（horn_seat_of）。"""
    return horn_seat_of(horn.spec)


def horn_pocket(horn, axis_xz, y_floor, y_face, sy):
    """leg_rev 2: 十字ホーンが収まる凹み（すきま込み）。底は y_floor（ホーンの外面）、y_face（リンクの内面）の
    外まで抜ける（絶対値、符号 sy 側）。ホーンの腕はワールドの X と Z に沿う（lib のホーンはサーボのロールに依らない）。
    腕の外側の穴（半径が最大の穴）2つに、ホーン側から締めるタッピングネジの下穴を通す。
    """
    hs = horn.spec
    x, z = axis_xz
    reach = hs["arm_reach_mm"] + c
    aw = hs["arm_width_mm"] / 2.0 + c
    lo = y_face - 1.0
    y0, dy = y_slab(lo, y_floor - lo, sy)
    tools = [part.box(2.0 * reach, dy, 2.0 * aw, origin=[x - reach, y0, z - aw]),
             part.box(2.0 * aw, dy, 2.0 * reach, origin=[x - aw, y0, z - reach]),
             part.cylinder(hs["hub_dia_mm"] / 2.0 + c, dy, origin=[x, y0, z], direction=[0, 1, 0])]
    r_hole = max(hs["hole_radii_mm"])
    yh, dyh = y_slab(lo, t + 2.0, sy)
    for dx in (-r_hole, r_hole):
        tools.append(part.cylinder(lib.tap_drill("m2") / 2.0, dyh, origin=[x + dx, yh, z], direction=[0, 1, 0]))
    return tools


def knee_bay_opening(hx, y_kt_abs, depth, sy):
    """leg_rev 2: 膝サーボを上腿の内側から差し込む通り道（上腿を真下に伸ばした姿勢で作る）。

    v1 の収納部は、タブ（ケースの両端の耳）がハウジングの中の閉じた溝に入り、外からも内からも入らなかった。
    ハウジングの内側の壁を、ケース・タブ・リード線の逃げの外形で抜き、タブはその上の材料（外側の板）に当てて、
    タブの穴から内側へタッピングネジ（M2）で留める。リード線はこの開口から胴体の側へ出る。
    この姿勢でサーボのローカル +X（股側）はワールドの +Z、膝軸は z = ZH - L1。
    """
    kz0 = ZH - L1
    half_w = CASE_W / 2.0 + c
    open_lo = y_kt_abs - depth - 1.0                   # ハウジングの内側の面より内
    tab_top = S["mount_hole_z_mm"] + S["tab_thickness_mm"] + c   # ケース上面からタブの上面（すきま込み、負）
    tab_lo = kz0 + CASE_CX - TAB_HALF - c
    tab_hi = kz0 + CASE_CX + TAB_HALF + c
    lead_lo = kz0 + (FRONT - CASE_L) - c - LEAD       # ケース後端の先のリード線の逃げの端
    y0, dy = y_slab(open_lo, (y_kt_abs + tab_top) - open_lo, sy)
    y1, dy1 = y_slab(open_lo, (y_kt_abs - CASE_H - c + 0.01) - open_lo, sy)
    tools = [part.box(2.0 * half_w, dy, tab_hi - tab_lo, origin=[hx - half_w, y0, tab_lo]),
             part.box(2.0 * half_w, dy1, tab_hi - lead_lo, origin=[hx - half_w, y1, lead_lo])]
    # タブのネジの下穴: タブの上面から外へ 5 mm（外面に 2 mm 残す）
    yp, dyp = y_slab(y_kt_abs + tab_top - 0.01, 5.0, sy)
    for side in (-1.0, 1.0):
        zc = kz0 + CASE_CX + side * S["hole_spacing_mm"] / 2.0
        tools.append(part.cylinder(lib.tap_drill("m2") / 2.0, dyp, origin=[hx, yp, zc], direction=[0, 1, 0]))
    return tools


result = {"chassis": chassis, "cover": cover, "esp32": esp32, "imu": imu}
if HAS_BAT:
    result["battery"] = battery
legs = {}
for name, sx, sy, _, yb in LEG_SPECS:
    hip = [LEG_X[name], 0.0, ZH]
    hx = hip[0]
    hsv = hip_servos[name]
    hhorn = hsv.horn("cross", label="hip_horn_" + name)
    # 上腿の板: 股サーボのケース上面（胴体の側面、3本足の hc は台の面）から joint_gap 離して外側に置く
    # （leg_rev 2 はホーンの腕の下に内面を置く）。
    y_th_in = yb + (horn_seat(hhorn) if R2 else g)
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
    spline_gap = part.cylinder(SPLINE_R + c, SPLINE_H + HUB_H, origin=[hx, sy * yb, ZH], direction=[0, sy, 0])
    hip_access = part.cylinder(p.screw_hole / 2.0 + 0.8, t + 2.0, origin=[hx, sy * (y_th_in - 1.0), ZH], direction=[0, sy, 0])
    thigh_tools = [ksv.bay(clearance=c, lead_room=LEAD, ledge=4.0), hhorn.body, spline_gap, hip_access]
    if R2:
        thigh_tools += horn_pocket(hhorn, (hx, ZH), yb + SPLINE_H + HUB_H, y_th_in, sy)
        thigh_tools += [rot_y(tool, th, hip) for tool in knee_bay_opening(hx, abs(y_kt), depth, sy)]
    thigh = part.cut(thigh_raw, thigh_tools, label="thigh_" + name)

    # 下腿: 膝ホーンの外側の板。真下に伸びた姿勢で作り、膝軸まわりに psi 回す。
    y_sh_in = abs(y_kt) + (horn_seat(khorn) if R2 else g)
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
    shin_tools = [khorn.body, k_spline_gap, k_access, foot]
    if R2:
        shin_tools += horn_pocket(khorn, (kx, kz), abs(y_kt) + SPLINE_H + HUB_H, y_sh_in, sy)
    shin = part.cut(shin_raw, shin_tools, label="shin_" + name)

    legs[name] = {"sx": sx, "sy": sy, "hip": hip, "knee": knee, "foot": [fx, fy, fz], "y_th_in": y_th_in,
                  "hsv": hsv, "hhorn": hhorn, "ksv": ksv, "khorn": khorn,
                  "thigh": thigh, "shin": shin, "foot_shape": foot}
    result["hip_servo_" + name] = hsv.body
    result["hip_horn_" + name] = hhorn.body
    result["thigh_" + name] = thigh
    result["knee_servo_" + name] = ksv.body
    result["knee_horn_" + name] = khorn.body
    result["shin_" + name] = shin
    result["foot_" + name] = foot

# --- しっぽ（tail_dof >= 1） ---
# 1本目のリンク: 左右のホーンの上の水平な板（下面は胴体の上面から joint_gap、ホーンは板の中に切り込む）。
#   tail_dof=1 では、これがしっぽ。後ろへ伸び、先端におもりの球が載る。
#   tail_dof=2 では、上下のサーボを載せるハウジングを持つ短い台。上下のサーボの軸はワールドの +Y で、
#   ケースは左右の中央。ロール 180 度でケースは軸から前へ伸びる。ハウジングの前端は胴体の後端の後ろ。
# 2本目のリンク（tail_dof=2）: 上下のホーンの外（+Y）の縦の板。後ろへ伸び、先端におもりが載る。
#   板は +Y 側にあり、しっぽの中心面は y = 0 から外れる（左右に非対称）。
TIP_R = 5.0
HAS_TIP = TAIL and p.tail_tip_mass_g > 0.0
if TAIL:
    tz0 = z_top + g                        # 1本目のリンクの下面
    x_tip = TY[0] - p.tail_length          # 先端のおもりの中心
    ty_horn = tail_yaw_sv.horn("cross", label="tail_yaw_horn")
    T_HUB_H = ty_horn.spec["hub_height_mm"]
    tl1_parts = [part.cylinder(R, t, origin=[TY[0], 0.0, tz0])]
    tl1_tools = [ty_horn.body,
                 part.cylinder(SPLINE_R + c, SPLINE_H + T_HUB_H, origin=list(TY)),
                 part.cylinder(p.screw_hole / 2.0 + 0.8, t + 2.0, origin=[TY[0], 0.0, tz0 - 1.0])]
    if TAIL_DOF == 1:
        tip_center = [x_tip, 0.0, tz0 + t / 2.0]
        tl1_parts += [part.box(TY[0] - x_tip, 10.0, t, origin=[x_tip, -5.0, tz0]),
                      part.cylinder(TIP_R + 1.0, t, origin=[x_tip, 0.0, tz0])]
    else:
        # 上下のサーボ: ケース（高さ CASE_H）を y = 0 の中央に。軸の高さはケースが1本目の板の上に載る位置。
        TP_Y = CASE_H / 2.0                                  # 上下のサーボのケース上面（+Y）
        # 前後: ハウジングの前端が、左右のサーボのケースの前の面から joint_gap 後ろ（左右の軸と一緒に動く）
        TP = [TY[0] + (CASE_L - FRONT) - g - H_LEAD, TP_Y, tz0 + t + c + CASE_W / 2.0]
        tail_pitch_sv = lib.servo(SKU, origin=tuple(TP), direction=(0, 1, 0), roll_degrees=180.0,
                                  label="tail_pitch_servo")
        tp_horn = tail_pitch_sv.horn("cross", label="tail_pitch_horn")
        depth_tp = CASE_H + c + w
        tl1_parts.append(part.box(H_SHAFT + H_LEAD, depth_tp, (TP[2] + HW) - tz0,
                                  origin=[TP[0] - H_SHAFT, TP_Y - depth_tp, tz0]))
        tl1_tools.append(tail_pitch_sv.bay(clearance=c, lead_room=LEAD, ledge=4.0))
        # 2本目のリンク（上下）。上下のホーンの外に置く（ホーンは板の中に切り込む）。
        y_l2 = TP_Y + g
        tip_center = [x_tip, y_l2 + t / 2.0, TP[2]]
        tl2_parts = [part.cylinder(R, t, origin=[TP[0], y_l2, TP[2]], direction=[0, 1, 0]),
                     part.box(TP[0] - x_tip, t, 10.0, origin=[x_tip, y_l2, TP[2] - 5.0]),
                     part.cylinder(TIP_R + 1.0, t, origin=[x_tip, y_l2, TP[2]], direction=[0, 1, 0])]
        tl2_tools = [tp_horn.body,
                     part.cylinder(SPLINE_R + c, SPLINE_H + T_HUB_H, origin=list(TP), direction=[0, 1, 0]),
                     part.cylinder(p.screw_hole / 2.0 + 0.8, t + 2.0, origin=[TP[0], y_l2 - 1.0, TP[2]],
                                   direction=[0, 1, 0])]
    if HAS_TIP:
        tail_tip = part.sphere(TIP_R, center=tip_center, label="tail_tip")
        tip_density = p.tail_tip_mass_g * 1.0e6 / (4.0 / 3.0 * math.pi * TIP_R ** 3)
        (tl1_tools if TAIL_DOF == 1 else tl2_tools).append(tail_tip)
        result["tail_tip"] = tail_tip
    tail_link_1 = part.cut(part.fuse(tl1_parts), tl1_tools, label="tail_link_1")
    result["tail_yaw_servo"] = tail_yaw_sv.body
    result["tail_yaw_horn"] = ty_horn.body
    result["tail_link_1"] = tail_link_1
    if TAIL_DOF == 2:
        tail_link_2 = part.cut(part.fuse(tl2_parts), tl2_tools, label="tail_link_2")
        result["tail_pitch_servo"] = tail_pitch_sv.body
        result["tail_pitch_horn"] = tp_horn.body
        result["tail_link_2"] = tail_link_2

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
for name in LEG_NAMES:
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
    th_mid = [(hip[0] + knee[0]) / 2.0, sy * (L["y_th_in"] + t / 2.0), (hip[2] + knee[2]) / 2.0]
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


def servo_drive(j, lo, hi, key):
    """角度指令型PWMサーボのアクチュエータと関節の動特性（脚と同じ式）。"""
    a = assembly.actuator(j, kind="position", control_deg="0",
                          stiffness_nmm_per_deg=STALL / 5.0, damping_nmms_per_deg=STALL / 100.0,
                          torque_limit_nmm=STALL, rated_torque_nmm=STALL * p.rated_fraction,
                          command_limits_degrees=[lo, hi], label=key + "_drive")
    drives.append(a)
    drive_by[key] = a
    dynamics.append(assembly.joint_dynamics(j, damping_nmms_per_deg=STALL / (60.0 / SPEED),
                                            armature_kgmm2=p.armature, friction_loss_nmm=p.friction_loss,
                                            label=key + "_dyn"))
    return a


tail_joints = []
if TAIL:
    # しっぽ: 左右のサーボは胴体に溶接。アクチュエータは脚のあと（task の actions の最後）に、左右・上下の順。
    def tail_comp(shape, key, appearance, role):
        cv = assembly.component(shape, appearance=appearance, role=role, label=key)
        result["c_" + key] = cv
        comps.append(cv)
        return cv

    c_tysv = tail_comp(tail_yaw_sv.body, "tail_yaw_servo", "mechanism", "hardware")
    c_tyhorn = tail_comp(ty_horn.body, "tail_yaw_horn", "mechanism", "hardware")
    c_tl1 = tail_comp(tail_link_1, "tail_link_1", "shell", "link")
    contacts.append((c_tysv, c_tyhorn))
    joints += [weld(c_chassis, c_tysv, "tail_yaw_servo"), weld(c_tl1, c_tyhorn, "tail_yaw_horn")]
    yaw_lim = [-p.tail_yaw_limit, p.tail_yaw_limit]
    j_ty = assembly.joint("revolute",
                          assembly.connector(c_chassis, "origin", offset={"position": list(TY)}),
                          assembly.connector(c_tl1, "origin", offset={"position": list(TY)}),
                          angle_limits_degrees=yaw_lim, label="tail_yaw")
    result["tail_yaw"] = j_ty
    joints.append(j_ty)
    bodies += [assembly.body(c_tysv, density_kg_m3=tail_yaw_sv.spec["effective_density_kg_m3"]),
               assembly.body(c_tyhorn, density_kg_m3=ty_horn.spec["density_kg_m3"])]
    tail_joints.append(j_ty)
    servo_drive(j_ty, yaw_lim[0], yaw_lim[1], "tail_yaw")
    c_tail_end = c_tl1
    if TAIL_DOF == 1:
        # しっぽの板の衝突形状: 付け根のハブの後ろから先端のおもりの手前まで（ワールドの X に沿う）
        cap_len = max(p.tail_length - R - TIP_R, 1.0)
        bodies.append(assembly.body(c_tl1, density_kg_m3=p.pla_density,
                                    collision=assembly.collision(
                                        "capsule", radius_mm=4.0, length_mm=cap_len,
                                        offset={"position": [TY[0] - R - cap_len / 2.0, 0.0, tz0 + t / 2.0],
                                                "axis": [0, 1, 0], "angle_degrees": 90.0},
                                        contact_group=1, label="tail_cap_1")))
    else:
        bodies.append(assembly.body(c_tl1, density_kg_m3=p.pla_density))
        c_tpsv = tail_comp(tail_pitch_sv.body, "tail_pitch_servo", "mechanism", "hardware")
        c_tphorn = tail_comp(tp_horn.body, "tail_pitch_horn", "mechanism", "hardware")
        c_tl2 = tail_comp(tail_link_2, "tail_link_2", "shell", "link")
        contacts.append((c_tpsv, c_tphorn))
        joints += [weld(c_tl1, c_tpsv, "tail_pitch_servo"), weld(c_tl2, c_tphorn, "tail_pitch_horn")]
        pitch_lim = [p.tail_pitch_min, p.tail_pitch_max]
        j_tp = assembly.joint("revolute",
                              assembly.connector(c_tl1, "origin", offset=hinge_frame(TP)),
                              assembly.connector(c_tl2, "origin", offset=hinge_frame(TP)),
                              angle_limits_degrees=pitch_lim, label="tail_pitch")
        result["tail_pitch"] = j_tp
        joints.append(j_tp)
        cap_len = max(TP[0] - x_tip - R - TIP_R, 1.0)
        bodies += [
            assembly.body(c_tpsv, density_kg_m3=tail_pitch_sv.spec["effective_density_kg_m3"]),
            assembly.body(c_tphorn, density_kg_m3=tp_horn.spec["density_kg_m3"]),
            assembly.body(c_tl2, density_kg_m3=p.pla_density,
                          collision=assembly.collision(
                              "capsule", radius_mm=4.0, length_mm=cap_len,
                              offset={"position": [TP[0] - R - cap_len / 2.0, y_l2 + t / 2.0, TP[2]],
                                      "axis": [0, 1, 0], "angle_degrees": 90.0},
                              contact_group=1, label="tail_cap_2")),
        ]
        tail_joints.append(j_tp)
        servo_drive(j_tp, pitch_lim[0], pitch_lim[1], "tail_pitch")
        c_tail_end = c_tl2
    if HAS_TIP:
        c_tip = tail_comp(tail_tip, "tail_tip", "accent", "link")
        joints.append(weld(c_tail_end, c_tip, "tail_tip"))
        bodies.append(assembly.body(c_tip, density_kg_m3=tip_density,
                                    collision=assembly.collision("sphere", radius_mm=TIP_R,
                                                                 offset={"position": list(tip_center)},
                                                                 contact_group=1, label="tail_tip_ball")))

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
for name in LEG_NAMES:
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
#   IMUの姿勢 rot_q* と角速度 gyro_*、前回の関節指令 c_*（アクチュエータの数: 脚 8 + しっぽ）、目標速度 cmd、歩容の位相 ref_sin/ref_cos。
# 関節角・胴体の速度と位置・足の接地・トルクは privileged（報酬・終了条件・critic だけが読む）。
KEYS = list(drive_by)
UP = "(1 - 2*(rot_qx^2 + rot_qy^2))"
cmd = assembly.goal("cmd", kind="speed", between=[p.cmd_min, p.cmd_max], resample_seconds=4.0)
FEET = [feet[name] for name in LEG_NAMES]
# Cadex の歩容リファレンス（assembly.gait）は1脚3関節以上が前提で、2自由度の脚には使えない。
# 代わりに位相クロック（コントローラ自身のタイマー。実機でも同じものを使える）を方策に与え、
# トロットのリズムは報酬で教える: 対角ペア (LF,RH) は clock_sin > 0 で立脚、(RF,LH) は < 0 で立脚。
clock = assembly.goal("clock", kind="phase", period_seconds=p.gait_period)
STANCE_A = "(0.5 + 0.5*tanh(5*clock_sin))"   # LF, RH が立脚である度合い
STANCE_B = "(0.5 - 0.5*tanh(5*clock_sin))"   # RF, LH
FOOT_R = p.foot_radius


def total(fmt):
    return " + ".join(fmt.format(k=k) for k in KEYS)


def lift_term(leg):
    """足上げ（足先の中心の高さ。立位で足の半径）。gait_lift まで払う。"""
    return "tanh((fc_%s_z - %.2f) / %.2f)" % (leg, FOOT_R, p.gait_lift)


# 3本足の波（gait_mode=1）: 脚 k（lf, rf, hc の順に 0, 1, 2）の位相は cos(θ - k·120°)。θ は位相クロックの角で、
# clock_sin = sin θ、clock_cos = cos θ。cos(θ - φ) = cos θ cos φ + sin θ sin φ。
# 位相が -0.5 より上（周期の 2/3）を立脚、下（1/3）を遊脚にする。どの時刻も遊脚は1本で、
# 遊脚の中心は lf → rf → hc の順に 1/3 周期ずつずれる。
WAVE_PHASE = [
    "clock_cos",
    "(-0.5*clock_cos + %.6f*clock_sin)" % (math.sqrt(3.0) / 2.0),
    "(-0.5*clock_cos - %.6f*clock_sin)" % (math.sqrt(3.0) / 2.0),
]


def gait_rewards():
    if not TRI:
        return [
            # トロットのリズム: 正しい組が接地していれば +、逆の組が接地していれば −（最大 2）
            assembly.reward("tanh(5*clock_sin) * ((ft_lf_down + ft_rh_down) - (ft_rf_down + ft_lh_down))",
                            weight=p.w_trot, label="trot_contact"),
            # 遊脚の足上げ（足先の中心の高さ。立位で足の半径）。gait_lift まで払う。
            assembly.reward(" + ".join(
                "%s * tanh((fc_%s_z - %.2f) / %.2f)" % (STANCE_B if leg in ("lf", "rh") else STANCE_A, leg, FOOT_R, p.gait_lift)
                for leg in ("lf", "rf", "lh", "rh")), weight=p.w_swing, label="swing_lift"),
        ]
    if GAIT_MODE == 0:
        # 決めない: 接地のリズムの報酬は無く、足上げはどの脚にも位相によらず払う。
        # ただし脚 k の足上げは、ほかの2本がどちらも接地しているときだけ払う（3本を同時に浮かせる＝跳ねると 0）。
        return [assembly.reward(" + ".join(
            "%s * %s" % (lift_term(leg), " * ".join("ft_%s_down" % o for o in LEG_NAMES if o != leg))
            for leg in LEG_NAMES), weight=p.w_swing, label="swing_lift")]
    stance = ["tanh(5*(%s + 0.5))" % ph for ph in WAVE_PHASE]       # 立脚で +、遊脚で −
    # 立脚の度合いの3脚の平均を引く: 全脚が接地したままなら、どの時刻も 0（四足のトロットの式と同じ意味）。
    # 脚 k を上げたときの報酬は「平均 - 脚 k の度合い」で、脚 k の遊脚の位相で正、立脚の位相で負。
    mean_st = "((%s) / 3)" % " + ".join(stance)
    return [
        # 立脚の位相で接地していれば +、遊脚の位相で接地していれば −（重みは四足のトロットと同じ w_trot）
        assembly.reward(" + ".join("(%s - %s) * ft_%s_down" % (st, mean_st, leg) for st, leg in zip(stance, LEG_NAMES)),
                        weight=p.w_trot, label="gait_contact"),
        assembly.reward(" + ".join("(0.5 - 0.5*%s) * %s" % (st, lift_term(leg)) for st, leg in zip(stance, LEG_NAMES)),
                        weight=p.w_swing, label="swing_lift"),
    ]


rewards = [
    assembly.reward("1", weight=1.0, label="alive"),
    assembly.reward(UP, weight=1.0, label="upright"),
    # 目標速度への追従。転倒中は払わない（UP を掛ける）。速度はワールドX（mm/s）。
    assembly.reward(UP + " * exp(-((vel_x - cmd) / 40)^2)", weight=3.0, label="track_speed"),
    assembly.reward("(vel_y / 100)^2", weight=-1.0, label="lateral"),
    assembly.reward("(gyro_z / 60)^2", weight=-0.5, label="yaw_rate"),
    assembly.reward("(gyro_x / 100)^2 + (gyro_y / 100)^2", weight=-0.2, label="body_rock"),
    assembly.reward("(pos_z / 15)^2", weight=-0.5, label="height"),
] + gait_rewards() + [
    # 滑らかさ: 指令の変化率（deg/s）とトルク（失速トルクで正規化）
    assembly.reward("(" + total("(c_{k}_rate / 600)^2") + ") / %d" % len(KEYS), weight=-0.5, label="action_rate"),
    assembly.reward("(" + total("(tau_{k} / %.3f)^2" % STALL) + ") / %d" % len(KEYS), weight=-p.w_effort, label="effort"),
    # v2: 推定連続定格で正規化したトルクの4乗。定格を超える区間だけを強く罰する（v1は左前脚が定格の1.4〜1.5倍）。
    assembly.reward("(" + total("(tau_{k} / %.3f)^4" % (STALL * p.rated_fraction)) + ") / %d" % len(KEYS),
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
for name in LEG_NAMES:
    for j in (hip_joints[name], knee_joints[name]):
        randomisation.append(assembly.randomise(j, "damping", scale=[0.7, 1.3]))      # サーボ速度のばらつき
        randomisation.append(assembly.randomise(j, "friction_loss", scale=[0.5, 2.0]))
for j in tail_joints:
    randomisation.append(assembly.randomise(j, "damping", scale=[0.7, 1.3]))
    randomisation.append(assembly.randomise(j, "friction_loss", scale=[0.5, 2.0]))
if HAS_TIP:
    randomisation.append(assembly.randomise(c_tip, "mass", scale=[0.8, 1.2], label="rand_tail_tip_mass"))
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
