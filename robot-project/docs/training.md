# 強化学習環境（Phase 5）

最終確認日: 2026-11-17

学習器は Cadex 同梱の `training/cadex_train.py`（Brax PPO、MuJoCo MJX、JAX）をそのまま使う。
新しく書いたのは、タスクの定義（設計スクリプトの `assembly.task`）と、それを回す周辺スクリプトだけ。
GPU（vast.ai）での実行手順は [training-vastai.md](training-vastai.md)。結果は [experiment-results.md](experiment-results.md)。

**今の状態（2026-11-17）**: 作る機体は小型機 `mini`。mini を GPU で学習シード5回学習した結果は、転ばず指令速度で進むが、
足を上げずに滑る（合格率 0.24、滑り 0.83。experiment-results.md §6）。指令速度や報酬の重みを小型機に合わせる再調整中で、
結果は未確定（§8）。

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
| effort | Σ(トルク/失速トルク)²/8 | −0.5 | サーボの負荷 |

**Cadexの歩容リファレンス（`assembly.gait`）は使えなかった。** 1脚に3関節以上が必要で、「leg LF has 2 joints of its own; placing a foot in space takes at least 3」として拒否された。
代わりに位相クロックを方策に与え、トロットのリズムを報酬で教えている。

終了条件は2つ。

- 傾きが約45°を超えた（upright < 0.7）
- 胴体が立位より30 mm以上下がった

## 5. ドメインランダム化

| 対象 | どこで | 範囲 |
|---|---|---|
| 胴体・カバーの質量 | タスク（`assembly.randomise`） | ×0.85〜1.15 |
| 電池の質量（重心位置も動く。`battery_mass_g=0` の機体には無い） | タスク | ×0.6〜1.4 |
| 関節の粘性（サーボ速度のばらつき） | タスク | ×0.7〜1.3 |
| 関節の静止摩擦 | タスク | ×0.5〜2.0 |
| 初期姿勢の傾き・持ち上げ | タスク（`reset_variation`） | 0〜3°、4〜6 mm |
| 床の摩擦 | トレーナー `--randomise-friction` | 段階2で使う |
| サーボトルク | トレーナー `--randomise-drive-torque` | 段階2で使う |
| 接触の硬さ | トレーナー `--randomise-contact-timeconst` | 段階2で使う |
| 応答遅延 | `--action-filter-alpha`（一次遅れ）で近似 | 固定0.6 |
| IMUノイズ | **Cadexのトレーナーに機能が無い**（未対応） | — |
| 関節のガタ（バックラッシュ） | **未対応**（静止摩擦のランダム化で一部だけ近似） | — |

学習用カットモデルでは、胴体に統合されたカバーと電池の質量ランダム化が落ちる（`cadex_relay.py` が報告する）。
胴体の質量ランダム化は残る。

段階的に難しくする方針は次のとおり。

1. 段階1: 上の「タスク」欄のランダム化だけ（実施済み、CPU）
2. 段階2: トレーナーの摩擦・トルク・接触のランダム化を追加
3. 段階3: `assembly.disturbance`（押し）と旋回指令（`yaw_rate` ゴール）を追加

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
平均 ± 標準偏差（最小〜最大）で比べる。

## 8. 小型機 mini の再調整（作業中）

mini は、base・leg120 と同じタスク（指令 40〜160 mm/s、報酬 v2）のままでは、足を上げずに滑って進んだ
（experiment-results.md §6）。小型機に合わせた3案を作り、GPU で比べている最中で、**結果は未確定**。

| 案 | 機体 / 設定 | mini からの差（`artifacts/<案>/design_report.json` の `params`） |
|---|---|---|
| `mini-cmd` | `robot/mjcf/mini-cmd/`、`configs/train_full_mini-cmd.json` | 指令速度 `cmd_min`/`cmd_max` を 40/160 から 30/120 mm/s に縮める |
| `mini-v3` | `robot/mjcf/mini-v3/`、`configs/train_full_mini-v3.json` | `mini-cmd` に加え、v3 の報酬の重み（`w_trot` 0.5→1.0、`w_swing` 0.5→1.5） |
| `mini-lift` | `robot/mjcf/mini-lift/`、`configs/train_full_mini-lift.json` | `mini-v3` に加え、足上げ高さ `gait_lift` を 10 から 7.5 mm に縮める |

3案とも機体の形は mini と同じ。学習設定は、機体（`bundle`）と名前以外は `train_full_mini.json` と同じ。
mini とタスクの差が意図した項目だけであることはテストで固定している。
