# 強化学習環境（Phase 5）

最終確認日: 2026-11-18

学習器は Cadex 同梱の `training/cadex_train.py`（Brax PPO、MuJoCo MJX、JAX）をそのまま使う。
新しく書いたのは、タスクの定義（設計スクリプトの `assembly.task`）と、それを回す周辺スクリプトだけ。
GPU（vast.ai）での実行手順は [training-vastai.md](training-vastai.md)。結果は [experiment-results.md](experiment-results.md)。

**今の状態（2026-11-18）**: 作る機体は、小型機 mini の脚を組めるように直した `mini-r2`。学習タスクは、mini の再調整
（experiment-results.md §6〜§8）で良かった報酬を載せた **`mini-r2-p7-eff`**（指令 30〜120 mm/s、接地位相 1.0、足上げ 1.5、
定格を超えるトルクの罰 1.0、歩容の周期 0.7 s）に決めた。GPU で学習シード5回学習し、合格率 0.90 ± 0.22、RMSトルク / 推定連続定格
0.75 ± 0.10（§9）。周期 0.5 s の `mini-r2-eff` は合格率 0.50 で、mini-eff（0.88）より悪かった。実機に載せる候補は
`artifacts/training/full-mini-r2-p7-eff-s3/walk.cxpolicy`（合格 10/10）。実機では未確認。

**実機との差に強くする（段階A-2、2026-11-18、experiment-results.md §11）**: s3 は、足の摩擦が 0.3〜0.5 に下がると滑って歩数が
数えられず（合格率 0.00〜0.40）、実機を想定した幅で全部を変えると 0.58 だった。足の摩擦・サーボの遅れと剛性とトルク・重心・押しを
学習に入れた **`mini-r2-p7-eff-dr`**（§5.2）は、普通の評価で5回とも 10/10、トルク比 0.61、滑り 0.27、実機を想定した幅で 0.84 ± 0.02
（元の案 0.60 ± 0.12、幅は重ならない）。押しへの強さはほとんど変わらなかった。反復を 2000 にした案は、はっきりとは良くならなかった。
**実機の候補を `artifacts/training/full-mini-r2-p7-eff-dr-s4/walk.cxpolicy` に替えた**
（ESP32 向けの書き出しは `deploy/esp32/exports/full-mini-r2-p7-eff-dr-s4/`）。実機では未確認。

## 1. タスクの在りか

タスクは CAD と同じ設計スクリプト `robot/design/quadruped8.py` の末尾に書いてある。
`cadex export` がモデル（`model-model.xml`）とタスク（`walk-task.json`）を対で書き出す。
タスクはモデルを相対パスと sha256 で参照するので、2つは同じ場所に置く（`robot/mjcf/<variant>/`）。

## 2. 観測（Observation）

| チャンネル | 由来 | 方策が読むか | 実機での取得方法 |
|---|---|---|---|
| `rot_qw..qz` | IMUの姿勢（クォータニオン） | 読む | IMU（MPU-6050のDMP、ICM-42688+姿勢推定など） |
| `gyro_x..z` | IMUの角速度 | 読む | 同上（※Cadexはワールド座標で出す。実機ではジャイロ値を姿勢で回して合わせる） |
| `c_<関節>`（8） | 前回の関節指令 | 読む | コントローラ自身が持つ値 |
| `cmd` | 目標前進速度 [mm/s] | 読む | 上位の指令 |
| `clock_sin`, `clock_cos` | 歩行位相（周期0.5 s） | 読む | コントローラのタイマー |
| `vel_*`, `pos_*` | 胴体の速度・位置 | 読まない（報酬・終了のみ） | 実機に無い |
| `q_<関節>` | 関節角 | 読まない | PWMサーボは角度を返さない |
| `ft_*`, `fc_*` | 足の接地・足先の高さ | 読まない | 実機に無い |
| `tau_*` | サーボの出力トルク | 読まない | 実機に無い |

**角速度の座標系（実機との差の要因）**: Cadex の `component_angular_velocity` はワールド座標で出す。`frame=` の指定は `component_position` でしか受け付けられないため、機体座標のジャイロ値にはできなかった。実機では、ジャイロ値を推定姿勢で回してワールド座標に直す（`deploy/esp32` の `gyro_world`）。ただし MPU-6050 には磁気センサが無く、ヨー角が漂流する。その誤差は角速度の x・y 成分にも入る。対策は2つ: 学習時に初期ヨーをばらつかせる（Cadexの `reset_variation` にヨーの項は無いので、トレーナー側の対応が要る）、またはヨーを含まない観測（重力方向ベクトル）を Cadex に足す。どちらも未着手。

実機に無い値を方策に入れると、`cadex train` がそのチャンネルを「接地されていない」として拒否する（ADR-408）。

## 3. 行動（Action）

8関節の目標角（度）。各関節の可動域（`command_limits_degrees`）に収まるよう、`tanh` で写像される。
シミュレーション上のサーボは位置アクチュエータで、次のように与えた。

- 剛性: 失速トルク / 5°
- トルク制限: 失速トルク 176.5 N·mm（MG90S、4.8 V）
- 関節の粘性: 失速トルク / 無負荷速度。全力で回したとき、データシートの速度 600°/s で頭打ちになる

学習時は次のトレーナーフラグも付ける。

- `--no-load-speed 10.47`（rad/s）: トルク–速度の直線
- `--action-filter-alpha 0.6`: 指令の一次遅れで、応答遅延の近似

## 4. 報酬（Reward）

| ラベル | 式（要約） | 重み | ねらい |
|---|---|---|---|
| alive | 1 | +1.0 | 転倒せず続けること |
| upright | 1−2(qx²+qy²) | +1.0 | 胴体を水平に保つ |
| track_speed | upright × exp(−((vel_x−cmd)/40)²) | +3.0 | 目標速度への追従（転倒中は払わない） |
| lateral | (vel_y/100)² | −1.0 | 横すべり |
| yaw_rate | (gyro_z/60)² | −0.5 | 向きの変化 |
| body_rock | (gyro_x/100)²+(gyro_y/100)² | −0.2 | 胴体の揺れ |
| height | (pos_z/15)² | −0.5 | 胴体の高さの変化 |
| trot_contact | tanh(5·clock_sin)·((LF+RH接地)−(RF+LH接地)) | +0.5 | 対角ペアを交互に接地させる |
| swing_lift | 遊脚の足先の高さ（tanh、10 mmまで） | +0.5 | すり足を防ぐ |
| action_rate | Σ(指令の変化率/600)²/8 | −0.5 | 滑らかさ |
| effort | Σ(トルク/失速トルク)²/8 | −0.5（`w_effort`） | サーボの負荷 |
| effort_rated | Σ(トルク/推定連続定格)⁴/8 | −0.3（`w_effort_rated`） | 推定連続定格（失速の40%）を超える区間を強く罰する |
| tilt | (qx²+qy²)/0.0019 | −0.1 | 胴体の傾き（5度で1） |

**Cadexの歩容リファレンス（`assembly.gait`）は使えなかった。** 1脚に3関節以上が必要で、「leg LF has 2 joints of its own; placing a foot in space takes at least 3」として拒否された。
代わりに位相クロックを方策に与え、トロットのリズムを報酬で教えている。

終了条件は2つ。

- 傾きが約45°を超えた（upright < 0.7）
- 胴体が立位より30 mm以上下がった

## 5. ドメインランダム化

### 5.1 元のタスク（`robot/mjcf/<variant>/walk-task.json`、どの案にもある）

| 対象 | どこで | 範囲 |
|---|---|---|
| 胴体・カバーの質量 | タスク（`assembly.randomise`） | ×0.85〜1.15 |
| 電池の質量（重心位置も動く。`battery_mass_g=0` の機体には無い） | タスク | ×0.6〜1.4 |
| 関節の粘性（サーボ速度のばらつき） | タスク | ×0.7〜1.3 |
| 関節の静止摩擦 | タスク | ×0.5〜2.0 |
| 初期姿勢の傾き・持ち上げ | タスク（`reset_variation`） | 0〜3°、4〜6 mm |
| 応答遅延 | `--action-filter-alpha`（指令の一次遅れ。実機のコントローラでも同じ式を回す） | 固定0.6 |

学習用カットモデルでは、胴体に統合されたカバーと電池の質量ランダム化が落ちる（`cadex_relay.py` が報告する）。
胴体の質量ランダム化は残る。

### 5.2 実機との差に強くする案（段階A-2、`mini-r2-p7-eff-dr`）

実機（PLA の脚＋ゴム足、MG90S ×8、ESP32-S3、20 ms 周期）との差を、学習に入れた案。設計スクリプトは変えず、
mini-r2-p7-eff のカットモデルとタスクに足したバンドルを `q8/dr_bundle.py`（`scripts/make_dr_bundle.py`）で作る
（`artifacts/mini-r2-p7-eff-dr/cut/`、中身の記録は `dr.json`）。学習設定は `configs/train_full_mini-r2-p7-eff-dr.json`
（反復 1000）と `-dr-2k.json`（反復 2000）。5.1 のランダム化はそのまま残し、次を足した。

| 対象 | どこで | 範囲 |
|---|---|---|
| 足と床の摩擦 | 学習器 `--randomise-friction 0.35 1.25`（全 geom の摩擦に共通の係数。床が 1.0 で接触は大きい方をとるので μ 0.35〜1.25） | PLA の足（約 0.3）〜ゴム足 |
| サーボのトルク | 学習器 `--randomise-drive-torque 0.8 1.1` | 個体差と電圧 |
| サーボの応答の遅れ | カットモデルのアクチュエータを `dyntype="filterexact"`（一次遅れ）にし、時定数を `actuator_dynprm` のランダム化で | 1.5〜30 ms（8関節に共通） |
| サーボの剛性 | `actuator_gainprm[a,0]` と `biasprm[a,1]` に同じ係数（関節ごと） | ×0.7〜1.3 |
| 胴体の重心 | 胴体の 20 g を子の body `dr_payload` に移し、その重心の x・y に ×−1〜1（総質量は同じ） | x ±2.7 mm、y ±1.8 mm |
| 押し | タスクの `disturbance`（胴体の重心、水平・全周、0.1 s） | 0〜1.5 N を 2回（開始 1.0〜3.5 s と 4.0〜7.0 s） |

学習に入れられなかったもの（Cadex の学習器に機能が無く、学習器は変えない方針のため）:

- **純粋な遅れ**と**制御周期のゆらぎ**: 一次遅れ（上の表）で近似しただけ。
- **バックラッシュ（ガタ）**: 剛性のランダム化で、関節の柔らかさだけを近似した。
- **IMU のノイズ**: 学習器の観測ノイズは追跡器と荷重センサーにしか無い。
- 学習器はエピソードの初めに一次遅れの状態（`act`）を戻さない。新しいエピソードの最初の数十 ms には、前の指令の名残がある。

純粋な遅れ・周期のゆらぎ・ガタは、評価（`q8/robustness.py`、[experiment-results.md](experiment-results.md) §11）で入れて確かめる。

## 6. 学習用カットモデル

`training/cadex_training_model.py cut` は、溶接部品を1つのボディに統合し、床接触だけを残したモデルを作る。
エンジンのモデルとの軌道差が5 mm以内で、立てることを自ら検査する。本機では1.4 mmで合格した。
`cadex_relay.py` がタスクをこのモデルへ名前で付け替える。

CPU（M4 Max、256環境）での速度は次のとおり。

| モデル | env steps/s |
|---|---|
| エンジンのモデル | 1,032 |
| カットモデル | 9,180（約8.9倍） |

`scripts/build_robot.py` が、どの機体についてもカットモデルを `robot/mjcf/<variant>/cut/` に作る。
学習はカットモデルで行い、`--cad-model` にエンジンのモデルを渡す。評価は必ずエンジンのモデルで、`cadex evaluate` を使う。

## 7. 実行

```bash
cd robot-project
# CPU（パイプラインの検証用）
JAX_PLATFORMS=cpu uv run python ../training/cadex_train.py robot/mjcf/base/cut/walk-task.json \
  --cad-model robot/mjcf/base/model-model.xml --out artifacts/runs/cpu-cut/walk.cxpolicy \
  --envs 512 --iterations 1000 --hidden 128 64 --checkpoint-every 100 \
  --action-filter-alpha 0.6 --no-load-speed 10.47 --progress artifacts/runs/cpu-cut/progress.json
# 評価（固定歩容と同じシード・同じ成功条件・同じ指標）
uv run python evaluation/evaluate_variant.py --variant base --policy artifacts/runs/cpu-cut/walk.cxpolicy
```

隠れ層は既定の 512-256-128 ではなく 128-64 にしている。ESP32-S3 で推論するため（[esp32-deployment.md](esp32-deployment.md)）。
学習の結果は [experiment-results.md](experiment-results.md) にまとめる。

本番の学習は vast.ai の GPU（RTX 4090）で、`configs/train_full*.json`（8192 環境、1000 反復）を使って回す。
複数台を借りて設定 × 学習シードを並べる `python -m vastai create --count N` と `sweep` の手順は
[training-vastai.md](training-vastai.md)。学習シード違いの複数回は `scripts/compare_seeds.py` で評価し、
平均 ± 標準偏差（最小〜最大）で比べる。学習だけを変えた案（DR など）は `--as <案>=<機体>` で元の機体のモデルとタスクで評価する。
実機との差の評価（摩擦・遅れ・ガタ・押しなど）は `evaluation/evaluate_robustness.py`（1つの方策）と
`scripts/compare_robustness.py --variant <機体> <案>=<回> …`（2案を学習シード方向に）。

## 8. 小型機 mini の再調整

mini は、base・leg120 と同じタスク（指令 40〜160 mm/s、報酬 v2）のままでは、足を上げずに滑って進んだ
（experiment-results.md §6）。タスクの値（設計スクリプトの params）だけを変えた案を作り、GPU で学習シード違いの複数回で比べた。
どの案も機体の形は mini と同じで、学習設定は機体（`bundle`）と名前以外は元の案と同じ。差が意図した項目だけであることは
`tests/test_vastai_config.py`（`MINI_TUNING`、`MINI_V3_TUNING`）で固定している。

| 案 | 元の案からの差（`artifacts/<案>/design_report.json` の `params`） | 合格率 | RMSトルク / 推定定格 |
|---|---|---|---|
| mini | — | 0.24 ± 0.36（5回） | 0.67 |
| `mini-cmd` | mini の指令速度 `cmd_min`/`cmd_max` を 40/160 から 30/120 mm/s に縮める | 0.03 ± 0.06（3回） | 0.64 |
| `mini-v3` | `mini-cmd` に加え、v3 の報酬の重み（`w_trot` 0.5→1.0、`w_swing` 0.5→1.5） | 0.48 ± 0.13（5回） | 1.06 |
| `mini-lift` | `mini-v3` に加え、足上げ高さ `gait_lift` を 10 から 7.5 mm に縮める | 0.40 ± 0.00（3回） | 1.10 |
| `mini-p6` | `mini-v3` の周期 `gait_period` を 0.5 から 0.6 s に延ばす | 0.30 ± 0.10（3回） | 1.11 |
| `mini-p7` | `mini-v3` の周期を 0.7 s に延ばす | 0.57 ± 0.40（3回） | 0.99 |
| `mini-eff` | `mini-v3` の定格を超えるトルクの罰 `w_effort_rated` を 0.3 から 1.0 に強める | 0.88 ± 0.18（5回） | 0.80 |
| **`mini-p7-eff`** | `mini-v3` の周期を 0.7 s に、`w_effort_rated` を 1.0 に | **0.97 ± 0.06（3回）** | 0.76 |

- 足を上げて歩くようになったのは、報酬の重みを v3 にしてから（mini-v3。experiment-results.md §7）。
  ただし、トルクが推定連続定格を超え、不合格はすべてそのため。
- トルクの罰を強めると（mini-eff）、歩き方を保ったままトルクが定格の内側に寄り、合格率が 0.88 に上がった。
  周期 0.7 s と合わせた mini-p7-eff は3回で 0.97 で、今の第一候補（experiment-results.md §8）。滑りは mini-eff より多い。
- **残り**: mini-p7-eff を学習シード 3・4 で5回に増やして確かめる。推定連続定格を実機で確かめる。
- 周期を延ばすと、評価の「歩数」の条件（1歩の前進が腰の高さの 0.15 倍以上、4歩以上）はゆるむ方向に効く（1歩が長くなり、
  回数は減る）。今回の案では歩数の条件で落ちた評価は無い（experiment-results.md §8「比べ方は公平か」）。
