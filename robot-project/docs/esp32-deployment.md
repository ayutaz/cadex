# ESP32-S3 へのポリシー配備（技術検証）

最終確認日: 2026-11-17。対象: MG90S × 8 の 8 自由度四足、制御基板 ESP32-S3、IMU（MPU-6050 相当）。

作る機体は小型機 `mini`（電池なし、外部 5V 給電）です。この文書の書き出しの確認は、基準機体 `base` の方策で行いました。
**mini の学習済み方策はまだありません**（再調整の結果待ち。下の「今の状態」）。

この文書では、**確認したこと**（コードを読んだ、またはテストで確かめた）と、
**推定**（実機で測っていない値）を分けて書く。推定には必ず「推定」と書く。

## 1. 結論

- `.cxpolicy` の形式は Cadex 本体のコードを読んで確定させた（§2）。参照実装
  `deploy/esp32/cxpolicy_io.py` は、エンジンの `CadexDynamics.policy_forward` と
  誤差 1e-9 以内で一致し、合成ファイルはトレーナーの witness 検査を通る（テストで確認）。
- **推奨する層構成は 20-128-64-8 か 20-64-64-8 で、手書き C の float32 推論。**
  見積りでは 1 回 0.1〜0.5 ms（推定）で、50 Hz（20 ms 周期）の数 % にとどまる。重みは
  45 KiB 以下で内部 SRAM に載る。float32 ならエンジンとの差は相対 1e-5 未満（ホストで確認）で、
  量子化による劣化は無い。
- int8 量子化は、512-256-128 のように大きいネットワークを載せるときだけ要る
  （float32 では 688 KiB あり、SRAM に載らない）。合成ポリシーでの誤差は、平均 0.3〜0.4°、
  最大 1.6〜3.7°（§5）。
- トレーナーには `--hidden` があり、小さい actor（実機に載るネットワーク）を指定できる。
  critic は `--value-hidden` で別に大きくできる。小さい actor（26-128-64-8）でこの四足は、
  シミュレーションでは歩けた（GPU で学習、二機体 × 学習シード5回の合格率は 0.92 と 1.00。
  [experiment-results.md](experiment-results.md) §5、本書 §12）。実機では未確認。

### 今の状態（2026-11-17）

- 書き出しの道具（`cxpolicy_io`、`quantize`、`export_c`、`budget`）と、ホストでの C の照合は、base の方策で通した（§12）。
- 作る機体 mini の方策は、まだ ESP32 に載せられる段階にない。GPU で学習シード5回学習した mini の方策は、
  転ばず指令速度で進むが、足を上げずに滑る（合格率 0.24、滑り 0.83。[experiment-results.md](experiment-results.md) §6）。
  指令速度や報酬の重みを小型機に合わせる再調整（`mini-cmd`、`mini-v3`、`mini-lift`）を GPU で比べている最中で、結果は未確定。
- `export_c --task` はタスクバンドルの sha256 を照合するので、mini の方策は mini（または採用した再調整の案）の
  `robot/mjcf/<variant>/cut/walk-task.json` と組で書き出す。base の書き出し（`exports/full-base-s0/`）は mini には使えない。
- ESP32-S3 のボードは依頼者の手元にあるが、**型番・寸法・USB 端子は未確認**。設計は 63 × 25.5 mm の
  ESP32-S3-DevKitC-1 を想定している（§8 の電源と配線、§10 のリスク）。
- ファームウェアは ESP-IDF でまだビルドしていない。

### 実物での確認（2026-11-17、`artifacts/runs/cpu-a/walk.000050.cxpolicy`）

学習 50 反復のチェックポイントを使った。歩行の質はまだ問わない段階のもの。

- 層構成は 26-128-64-8（swish）。witness を `cxpolicy_io check` で照合し、最大相対誤差は 1.5e-7
  （許容 1e-4）だった。
- 学習に使ったタスクバンドルは、sha256 が一致する `artifacts/tmp/export-base/walk-task.json`
  （= `robot/mjcf/base/walk-task.json`）。これを `export_c --task` に渡すと、26 チャネルすべての取り出し元が
  自動で決まる。sha256 が違うバンドル（`leg120` など）を渡すと止まる。
- 入力（26）: `rot_q{w,x,y,z}`（IMU 姿勢）、`gyro_{x,y,z}`（deg/s）、`cmd`（目標速度のゴール。
  40〜160 mm/s）、`clock_{sin,cos}`（位相ゴール。周期 0.5 s、1 ステップ 0.2513 rad なので制御周期は 20 ms）、
  関節ごとの `c_<j>` と `c_<j>_rate`（前回指令と、その変化率）。privileged のチャネル（並進速度、位置、
  関節角、接地、トルク）は入っていない。
- **角速度の行に `frame` が無いので、学習時の角速度は世界座標**になっている。実機ではジャイロ（機体座標）を
  推定した姿勢で回して世界座標に直す（`OBS_GYRO_WORLD`、`--task` を渡すと自動で選ばれる）。
  その結果、x と y の成分がヨー推定に依存する（§7 のヨーの漂流がそのまま効く）。学習側で `frame` を
  IMU のボディにすれば、この依存はなくなる。
- `action_filter_alpha` は 0.6、`command_slew_deg` は 0。どちらも `policy_weights.h` に入る。
- ホストの C で照合した結果: float32 版の参照との差は行動範囲の 1e-5 未満。int8 版は tanh 手前の値が
  ビット一致（`tests/test_esp32_export_c.py::test_real_policy_end_to_end`）。
- int8 の誤差（較正 2048、評価 2048。標本は witness と正規化統計からの合成）:

  | 方式 | 最大 [°] | 平均 [°] | 99%点 [°] |
  |---|---:|---:|---:|
  | float32 | 0.000 | 0.000 | 0.000 |
  | 重みだけ int8 | 0.57 | 0.06 | 0.27 |
  | 重み・活性値 int8 + LUT | 2.93 | 0.40 | 1.63 |
  | float、hard-swish に置換 | 9.81 | 1.39 | 5.54 |
  | float、32 区間の折れ線 | 1.39 | 0.22 | 0.82 |

- 見積り（`budget 26-128-64-8`）: MAC 12,032。重みは float32 で 47.8 KiB、int8 で 13.8 KiB。
  推論は float32 の手書き C で 186〜505 µs、int8 の手書き C で 89〜235 µs（推定）。

```bash
.venv/bin/python -m deploy.esp32.export_c artifacts/runs/cpu-a/walk.000050.cxpolicy \
    --task artifacts/tmp/export-base/walk-task.json --no-int8 \
    --initial-command <keyframe の 8 関節[deg]> --out deploy/esp32/firmware/main/policy_weights.h
```

## 2. `.cxpolicy` の形式（コードを読んで確認）

出典は `training/cadex_train.py`（`encode_policy`, `flat_parameters`, `snapshot`,
`policy_header`）と `src/Mod/cadex/CadexDynamics.py`（`decode_policy`, `policy_forward`,
`policy_swish`, `recorded_command_filter`, 再生中の `_actions`）。

```
b"CXPOLICY1\n" | u64 LE のヘッダ長 | 正準 JSON（キー整列・空白なし・ASCII・NaN 禁止） | float32 LE の重み
```

- 重みは層の順に、重み行列（入力, 出力）を行優先で並べ、続けてバイアス（出力）を置く。
- 最終層は、Brax の出力（平均と分散のぶんで 2 倍の幅がある）のうち**決定的な半分（平均）だけ**を
  保存している。探索の幅は `exploration.log_std` にあるが、再生には使わない。
- ヘッダの主なキー:
  - `observations`: 入力チャネル名の順序
  - `actions`: `actuator, index, unit, low, high, scale, ...`。サーボの `unit` は `deg`
  - `network`: `kind: "mlp"`, `layers`, `activation`（既定 `swish`）, `output: "tanh"`,
    `output_scale`, `output_bias`
  - `normaliser`: `mean`, `std`, `count`。std が 0 のチャネルは、トレーナーが 1 に置き換えて書く
  - `training`: `action_filter_alpha`, `command_slew_deg` ほか
  - `evaluation`: witness（観測とトレーナーが記録した行動、8〜256 組）
- 順伝播（エンジンと同じ。float64）:
  ```
  x = (obs - mean) / std
  隠れ層: x = swish(x W + b)        swish(x) = x * sigmoid(x)
  最終層: z = x W + b
  action = tanh(z) * output_scale + output_bias      （タスクの単位。サーボは度）
  ```
- 再生時の指令フィルタ（`training` の値。**実機でも必ず掛ける**）: 行動範囲で clamp し、
  次に一次遅れ `alpha*今回 + (1-alpha)*前回` を掛け、最後に前回からの変化を ±`command_slew_deg`
  に制限する。最初の 1 回は clamp だけ。
- 観測値は MuJoCo の `sensordata` に、バンドルの各行の `scale` を掛けたもの
  （角度は deg、角速度は deg/s、長さは mm）。実機でも同じ単位で入力する。

## 3. ネットワークの大きさとトレーナー

- `training/cadex_train.py --hidden W1 W2 ...` で actor の隠れ層を指定する（既定は `512 256 128`）。
  `--value-hidden` を省くと、critic も actor と同じ幅になる。**実機に載るのは actor だけ**なので、
  次の形を推奨する:
  ```
  --hidden 128 64 --value-hidden 512 256 128        # まずこれ
  --hidden 64 64  --value-hidden 512 256 128        # 余裕を見るならこちら
  ```
- `--activation` は `swish`（既定）か `tanh`。`tanh` なら C 側は `tanhf` になるが、計算量はほぼ同じ。
- 大きい方針（512-256-128）を先に学習しておき、小さい actor に蒸留する道もある（`--teacher`、ADR-825）。
  教師の検査は観測チャネルとアクション表の一致だけで、層の形が同じである必要はない
  （`cadex_train.py` 1280〜1350 行付近で確認）。
- warm start（`--init-from`）は層の形が一致していることを要求する。小さい actor は最初から学習するか、
  蒸留で作る。
- `--hidden 128 64` は GPU 学習で歩けた（[experiment-results.md](experiment-results.md) §5）。
  未検証: `--hidden 64 64` でも届くか。`configs/train_full.json` の `hidden` だけを変えた設定を作り、
  `python -m vastai sweep` で並べて回して比べること。

## 4. 計算量・メモリ・推論時間（時間は推定）

`python -m deploy.esp32.budget` の出力:

| 層構成 | MAC | 重み f32 | 重み int8 | f32 手書きC [µs] | int8 手書きC [µs] | int8 SIMD [µs] |
|---|---:|---:|---:|---:|---:|---:|
| 20-512-256-128-8 | 175,104 | 687.5K* | 178.8K | 19,438〜39,983 | 1,137〜3,025 | 152〜471 |
| 20-256-128-8 | 38,912 | 153.5K | 41.6K | 489〜1,304 | 264〜702 | 45〜134 |
| 20-128-128-8 | 19,968 | 79.0K | 22.1K | 278〜749 | 140〜373 | 28〜82 |
| **20-128-64-8** | 11,264 | 44.8K | 13.1K | **179〜488** | 83〜221 | 20〜57 |
| **20-64-64-8** | 5,888 | 23.5K | 7.3K | **107〜296** | 47〜125 | 14〜39 |
| 20-32-32-8 | 1,920 | 7.8K | 2.9K | 48〜133 | 20〜52 | 9〜24 |

`*` は、内部 SRAM の重み予算 256 KiB に載らないことを示す。この場合、毎周期すべての重みを flash から
読み直す時間を足してある。

前提（すべて推定。根拠は `deploy/esp32/budget.py` の `ASSUMPTIONS`）:

- 240 MHz の Xtensa LX7。手書き C の float32 は 1 MAC あたり 2〜5 サイクル（ロード 2 回と madd.s、
  ループ制御を含む）。int8 のスカラー版は 1.5〜4 サイクル。PIE SIMD（16 レーン int8）を使う
  ESP-NN / ESP-DL は 0.15〜0.5 サイクル。
- swish は `expf` を使うので、隠れニューロン 1 個あたり 100〜300 サイクル。int8 版は LUT を使い、10〜25 サイクル。
- flash は QSPI 80 MHz で、実効 20〜40 MB/s。
- **実測方法**: `firmware/main/main.c` が `esp_cpu_get_cycle_count()` で推論の平均と最大の
  サイクル数を毎秒ログに出す。書き込めばこの表を実測値で置き換えられる。

512-256-128 の float32 は flash の読み出しだけで 20 ms 周期を使い切る見込み。これを載せるなら
int8 にするか、PSRAM 付きモジュールで重みを PSRAM に置く必要がある（PSRAM の帯域は未見積り）。

### ランタイムの比較

| 方式 | 長所 | 短所 | この用途 |
|---|---|---|---|
| **手書き C（本実装 `firmware/policy.c`）** | 依存なし。式がエンジンと 1 対 1 で、ホストで照合できる。float32 ならエンジンとの差は相対 1e-5 未満 | SIMD を使わない | **≤128 幅なら第一候補** |
| ESP-NN | Espressif の int8 最適化カーネル（S3 は PIE のアセンブリ）。全結合層の関数を直接呼べる | 量子化は TFLite 流（ゼロ点つき活性値、固定小数点の倍率とシフト）。チャネルごとの倍率に対応するかは版を要確認 | 大きいネットワークの int8 を速くしたいとき。本実装の倍率を (mult, shift) に変換すれば使える |
| ESP-DL | S3 向けの int8/int16 推論。量子化ツール（ESP-PPQ）つき | ONNX への書き出しと変換が要る。6k パラメータの MLP には大げさ | 将来、CNN などに広げるとき |
| TFLite Micro（esp-tflite-micro） | 標準的。S3 では ESP-NN が下で動く | .tflite への変換（Keras などを経由）、インタプリタと arena のぶんのコード・RAM。swish は MUL + LOGISTIC で表す | 既存の TF 資産があるときだけ |

## 5. 量子化（`deploy/esp32/quantize.py`）

### 手順

1. 較正用の観測を集める。実機の分布に近いもの（シミュレーションのロールアウトで記録した観測、
   N × 観測数）を `.npy` に保存する。無いあいだは、ファイル内の witness と正規化統計
   N(mean, std) から作る合成標本を使う。
2. `python -m deploy.esp32.quantize POLICY --samples obs.npy --json report.json` を実行する。
   前半の標本で較正し、後半で float64 参照との誤差を表にする。
3. `python -m deploy.esp32.export_c POLICY --samples obs.npy --out firmware/main/policy_weights.h`
   を実行する。

### 方式

- 重み: 対称 int8（-127..127）。出力チャネルごと（既定）か層ごとのスケール（`--per-layer`）。
- 活性値: 対称 int8。スケールは較正標本での最大絶対値 / 127。百分位点でのクリップ
  （`--percentile`）も試したが、合成ポリシーでは最大誤差が悪化した（99.9% で約 10°）。そのため既定は最大値。
- バイアス: int32。スケールは「入力スケール × 重みスケール」。
- 隠れ層の活性化: int32 の累積に float32 の倍率を掛けて int8 の前活性化に戻し、
  256 要素の LUT で活性化後の int8 にする。LUT は量子化の格子点上で swish の真値を持つので、
  活性化の近似誤差は出ない。
- 最終層: 出力 8 個だけを float に戻し、`tanhf(z) * scale + bias` を計算する。
- C の `policy_forward_i8` と Python の `forward_int8` は同じ float32 演算になるよう書いてある。
  ホストの clang でコンパイルした C と比べて、tanh の手前の値が**ビット一致**する（テストで確認）。

### 誤差（合成ポリシー、乱数重み。float64 参照との差。単位は度）

実物のポリシーでは値が変わる。実物が届いたら、同じコマンドで測り直すこと。

| 方式 | 20-512-256-128-8 最大 / 平均 | 20-128-64-8 最大 / 平均 | 20-64-64-8 最大 / 平均 |
|---|---|---|---|
| float32（C の float 版） | 0.000 / 0.000 | 0.000 / 0.000 | 0.000 / 0.000 |
| 重みだけ int8 | 0.62 / 0.08 | 0.68 / 0.10 | 0.80 / 0.11 |
| 重み・活性値 int8 + LUT（C の int8 版） | 1.59 / 0.30 | 2.25 / 0.39 | 3.67 / 0.41 |
| 同上、重みスケールを層ごと | 1.82 / 0.31 | 2.68 / 0.41 | 3.58 / 0.43 |
| float、swish → hard-swish | 6.73 / 1.13 | 10.60 / 1.77 | 9.17 / 1.68 |
| float、swish → 32 区間の折れ線 | 1.69 / 0.49 | 1.74 / 0.46 | 1.46 / 0.42 |
| float、swish → 8 区間の折れ線 | 12.9 / 3.7 | 13.6 / 3.6 | 11.8 / 3.3 |

この表から次のことが言える。

- 誤差の大部分は出力の写像 `tanh(z) × 45°` で拡大される。z が 0.05 ずれると約 2° ずれる。
- swish を、学習しなおさずに hard-swish に置き換えるのは不可（最大 6〜11°）。折れ線で近似するなら、
  32 区間でも最大 1.5° 程度の誤差が出る。補間誤差の理論値は h²/8·max|swish''| ≈ 0.016。
  float 版では `expf` をそのまま使えばよい（小さいネットワークなら時間は問題にならない）。
- int8 版では LUT を使うので、活性化の近似は問題にならない。
- MG90S の不感帯は数 µs〜10 µs 程度とされる（推定、0.5〜1° 相当）。平均 0.3〜0.4° はその内側に
  収まる。最大 2〜4° を許せるかは、実機で歩かせて判断する。

## 6. 実機で観測できる入力の定義

ポリシーへの入力には、**実機で測れるものだけ**を入れる。Cadex の観測行のうち、
`role: "privileged"` のものは actor に入らない（ADR-408）。報酬・終了条件・critic だけが読む。
policy の行には `grounded_sensor` で「実機のどのセンサが測るか」を宣言でき、
`CadexDynamics.ungrounded_policy_channels` で、宣言の無いチャネルを洗い出せる。

| チャネル | 次元 | 実機での出どころ | 単位（バンドル） |
|---|---:|---|---|
| 姿勢クォータニオン wxyz | 4 | IMU の姿勢推定（相補フィルタ、Madgwick など） | 無次元 |
| 角速度 | 3 | IMU のジャイロ（機体座標） | deg/s |
| 前回の関節指令 | 8 | 自分が出した指令（フィルタ後） | deg |
| （任意）指令の変化率 | 8 | (今回 − 前回) / 周期 | deg/s |
| 目標速度などのゴール | 2〜3 | 操縦入力 | バンドルのゴールの単位 |
| 歩行位相 sin/cos | 2 | 内部クロック | 無次元 |

次のものは**入れない**（シミュレーションの内部状態で、実機では測れないため）:
胴体の並進速度・高さ・位置、関節角・関節角速度（MG90S には位置の帰還が無い）、関節トルク、
足の接地、地形の高さ。これらは privileged にして、報酬と critic だけに使う。

前回の関節指令は、Cadex の `assembly.observation(actuator, "command", name=...)` で作る
（ADR-696）。この観測は `hip` と `hip_rate` の 2 チャネルになる。

`export_c.py` は各チャネルの取り出し元
（`quat / gyro / gyro_world / gravity / cmd / cmd_rate / goal / phase_sin / phase_cos / const`）と倍率を決め、
`policy_weights.h` に書く。決め方の優先順は次のとおり。

1. `--layout` で明示したもの
2. `--task` のタスクバンドルにある観測行の種類・`scale`・`frame`、ゴール、コマンド行
3. 名前からの推定

タスクが privileged と宣言したチャネルをポリシーが読んでいれば、そこで止まる。名前で決められないチャネルがあれば止まる。その場合は
`--layout-template` で雛形を出して埋め、`--layout` で渡す。

## 7. IMU の座標系と単位の合わせ方

- MuJoCo の世界座標は Z 上向きで、重力は −Z。`component_orientation`（FRAMEQUAT）は、
  IMU を付けたボディの「機体→世界」の回転を **wxyz** で表したもの。
- 角速度は `component_angular_velocity`（FRAMEANGVEL）。**`frame` を指定しないと世界座標になり、
  実機のジャイロ（機体座標）と一致しない**。タスクで角速度の行を作るときは、`frame` に IMU のボディを指定する
  （エンジンは `reftype = mjOBJ_XBODY` を設定する）。
- 単位: MPU-6050 の生値は LSB/(°/s)（±250 dps なら 131、±500 dps なら 65.5）。ファームウェアの
  `imu_sample_t.gyro_rad_s` には rad/s で入れ、`policy_obs_gain`（57.2958）で deg/s にする。
- 取り付け向き: センサ軸を機体軸に写す回転行列 R_mount（符号つきの軸の並べ替え）を決め、
  ジャイロにも加速度にも掛ける。確認の手順:
  1. 水平に静置する。加速度は機体の +Z 軸で +1 g を示すこと（比力なので上向きが正）。
  2. 機体を手で +X 軸（前）まわりに右ねじの向きで回す。gx が正になること。Y、Z 軸も同様に確かめる。
  3. シミュレーションで同じ姿勢をとらせ、観測値（クォータニオンと角速度）の符号を突き合わせる。
- **ヨー角**: MPU-6050 には磁気センサが無いので、ヨーは積分で漂流する。クォータニオンを
  そのまま入れる場合は、起動時の向きをヨー 0 と定義する。そのうえで、学習時に初期ヨーをランダムに
  振っていないと、時間とともにずれたヨーが、学習時に見たことのない入力になる（リスク）。
  対策の候補: (a) 学習で初期ヨーをランダムに振る、(b) ヨーに依らない量（機体座標の重力方向。
  `control.c` には `OBS_GRAVITY` として実装済み）で学習する。(b) には、Cadex 側に重力方向の観測種別が要る。
  現在の観測種別には無い（`OBSERVATION_KINDS` で確認）。エンジンへの追加は ADR が要る変更になる。

## 8. サーボの角度とパルス幅

```
pulse_us = center_us + sign * (angle_deg + trim_deg) * us_per_deg    →  [min_us, max_us] に収める
duty     = round(pulse_us * 2^14 / 20000)                             （LEDC 14 bit、50 Hz）
```

- **MG90S の 500〜2400 µs・約 180° は推定値**。カタログ値も個体差もあり、この範囲で使えるとは限らない。
  既定値は center 1450 µs、`us_per_deg` = 1900/180 ≈ 10.56。
- 関節角 0° は、Cadex の関節の基準（組立の零姿勢）を指す。ホーンの取り付け位相のずれは、関節ごとに
  `trim_deg` と `sign` で吸収する。較正は分度器か治具で、少なくとも 3 点（-45°, 0°, +45°）を測り、
  `us_per_deg` も個体ごとに合わせる。結果は `--servo servo.json` で渡す
  （`{"default": {...}, "joints": {"hip_lf": {"trim_deg": 3.0, "sign": -1}}}`）。
- MG90S はアナログサーボで、50 Hz の PWM が前提。制御を 50 Hz で回せば、1 フレームごとに新しい指令になる。
  25 Hz なら、同じ指令が 2 フレーム続く。
- 起動時は、`policy_initial_command`（既定は行動範囲の中央）へいきなり動く。学習時のリセット姿勢
  （keyframe）に合わせて `--initial-command` で指定し、電源投入の手順（低速で姿勢に入れるなど）を決めること。

### 電源と配線（mini、最初の構成）

mini は電池を載せず、外部の 5V 電源で動かす。実機で確かめたことはない。

- **電源**: スマホの充電器か、5V 4A の AC アダプタ（ブレッドボード用 DC ジャックと組み合わせる）。
  5V は MG90S の動作電圧 4.8〜6.6 V の範囲内。充電器の出力電流と、サーボ8個が同時に使う電流は**未確認**。
  USB-PD で 9V・12V を出す充電器もあるので、サーボにつなぐのは 5V だけにする。
  DC ジャックのキットの定格は 3A（[shopping-list.md](shopping-list.md) §3）。
- **サーボの電源を ESP32 から取らない。** サーボの +（赤）は 5V の電源から直接取る。ESP32 ボードの 5V ピン経由で
  取ると、ボード上の配線やパソコンの USB ポートに大きな電流が流れ、電圧が下がって ESP32 が再起動しうる。
- **GND を共通にする。** サーボの電源の GND と ESP32 の GND をつなぐ。つながないと、PWM の信号の基準がずれて
  サーボが正しく動かない。
- **ESP32 の電源**: 書き込み・ログの確認中は USB（パソコン）から。サーボの 5V 電源とは GND だけを共通にする。
  外部 5V から ESP32 にも給電する方法（ボードの 5V ピンに入れてよいか、USB と同時につないでよいか）は、
  ボードの型番が分かってから回路図で確かめる（未確認）。
- **信号線**: ファームウェアのサーボの GPIO は仮に `{4, 5, 6, 7, 15, 16, 17, 18}`（`firmware/main/main.c` の `SERVO_GPIO`、TODO）。
  手元のボードでこのピンが使えるか（ブートに使うピンや、PSRAM・USB に取られたピンでないか）は、型番が分かってから確かめる。
  IMU の I²C は未実装（`main.c` の TODO）。
- 3.3V の GPIO で 5V 給電の MG90S を駆動できるか（信号の電圧）は実機で確かめていない。

## 9. ファイルと使い方

```
deploy/esp32/cxpolicy_io.py   .cxpolicy の読み書き、参照順伝播、witness 検査、指令フィルタ、合成ファイル
deploy/esp32/quantize.py      int8 量子化、較正、誤差評価、swish 近似の比較
deploy/esp32/export_c.py      policy_weights.h の生成、観測レイアウト、サーボ換算
deploy/esp32/budget.py        MAC 数、重みサイズ、推論時間の見積り
deploy/esp32/firmware/        ESP-IDF プロジェクト（policy.c/h, control.c/h, main/main.c）
deploy/esp32/firmware/host/   ホスト検証用ハーネス（テストが clang でビルドする）
tests/test_esp32_*.py         テスト（Cadex 本体の decode_policy / policy_forward / witness_disagreement との照合を含む）
```

```bash
cd robot-project
.venv/bin/python -m deploy.esp32.cxpolicy_io info  artifacts/<run>/walk.cxpolicy
.venv/bin/python -m deploy.esp32.cxpolicy_io check artifacts/<run>/walk.cxpolicy   # witness 一致を確認
.venv/bin/python -m deploy.esp32.quantize  artifacts/<run>/walk.cxpolicy --samples obs.npy
.venv/bin/python -m deploy.esp32.export_c  artifacts/<run>/walk.cxpolicy --layout-template > layout.json
.venv/bin/python -m deploy.esp32.export_c  artifacts/<run>/walk.cxpolicy --layout layout.json \
    --servo servo.json --initial-command ... --out deploy/esp32/firmware/main/policy_weights.h
.venv/bin/python -m deploy.esp32.budget 20-128-64-8
# ESP-IDF v5: cd deploy/esp32/firmware && idf.py set-target esp32s3 && idf.py build flash monitor
```

同梱の `firmware/main/policy_weights.h` は、合成ポリシー（乱数重み）から作った見本。
`POLICY_SYNTHETIC` が 1 のあいだ、`main.c` はサーボに出力しない。

## 10. リスク

- 小さい actor（26-128-64-8）は、シミュレーションでは歩けた（§12）。実機では未確認。
- ヨーの漂流（§7）。
- 角速度の座標系。実物のタスク（`base/walk-task.json`）は `frame` が無く、世界座標で学習している。
  `OBS_GYRO_WORLD` で再現できるが、ヨー推定の誤差が角速度の x と y 成分にも入る。学習側で `frame` を
  指定することを勧める。
- MG90S の角度とパルス幅の関係、不感帯、速度（カタログ 0.1 s/60°）、ギアのガタは推定または未測定。
  シミュレーションのアクチュエータモデル（`--no-load-speed` などのトルク・速度曲線）が実機と合わなければ、
  方針が実機で通用しない。
- 制御周期の揺らぎ。`vTaskDelayUntil` と FreeRTOS 1 kHz tick を前提にしている。I2C の読み取り時間は未測定。
- 起動時の指令は `.cxpolicy` に入っていない。`export_c --task` を渡すと、学習のリセットと同じ値
  （リセットのキーフレームの ctrl、`CadexDynamics.reset_command` と同じ式）が自動で入る。この機体では全関節 0°。
- `main.c` は ESP-IDF でビルドしていない。スタブヘッダでの構文・型検査だけを行った。
- int8 の誤差は、実物の方策で合成ポリシーより大きく出た（最大 4.6°、§12）。float32 を使う。
- **作る機体 mini の方策がまだ無い。** mini の学習は再調整中で、ESP32 に載せる方策は結果待ち。
- **手元の ESP32-S3 の型番・寸法が未確認。** 設計は 63 × 25.5 mm の DevKitC-1 を想定している。違う場合は、
  カバー上面のポケットに収まらない、サーボの GPIO（§8）が使えない、USB 端子が違う、などが起こりうる。
- **電源（§8）**: 充電器・AC アダプタの電流が足りないと、ESP32 の再起動やサーボの震えが起こる。
  サーボ8個の消費電流は実測していない。外部電源のコードの重さと引っ張りはシミュレーションに入っていない。

## 11. 次の手順

1. （済、§12）26-128-64-8 を GPU で学習し、`check` → `quantize` → `export_c` を通した（base の方策）。
   残り: `--hidden 64 64` との比較。ロールアウトの観測を `.npy` に記録し、int8 の較正と評価に使う（float32 なら不要）。
2. **手元の ESP32-S3 の型番と寸法を確かめる。** ボードの印字か購入記録から型番を調べ、寸法・USB 端子・ピン配置を確認する。
   63 × 25.5 mm と違えば、カバー上面のポケットの寸法を設計スクリプトで合わせ、`SERVO_GPIO`（§8）を見直す。
3. **電源を用意する。** 充電器の出力（5V で何 A か）を確かめ、3A 未満なら 5V 4A の AC アダプタを使う。
   サーボの電源を 5V の電源から直接取り、GND を ESP32 と共通にする（§8）。
4. ESP-IDF でビルドし、推論サイクル数を実測して §4 の表を置き換える。
5. サーボを1個ずつ動かして確かめたうえで較正し、`servo.json` を作る（§8）。
6. **mini の再調整の結果を待つ。** 合格した方策ができたら、その機体のタスクバンドルと組で `check` → `export_c` を通し、
   `servo.json` を渡して `firmware/main/policy_weights.h` を作る（置いた時点で `POLICY_SYNTHETIC` が 0 になり、サーボに出力する）。
7. IMU ドライバと姿勢推定を実装し、§7 の手順で軸を合わせる（IMU は方策を載せる段階で買う）。
8. HIL（hardware-in-the-loop）: シミュレーションの観測列を UART で基板に送り、基板の出力が
   参照実装と一致することを確かめる。そのあとで吊り下げ試験、接地試験の順に進める。

## 12. GPU で学習した方策の書き出し（2026-11-17）

基準機体の GPU 学習（[experiment-results.md](experiment-results.md) §5、学習シード5回）から
`artifacts/training/full-base-s0/walk.cxpolicy` を選んだ。選び方: 合格 10/10 かつ全脚 14 歩以上の回
（シード 0 と 4）のうち、向きのずれの小さい方（4.1° と 6.9°）。

- 層構成 26-128-64-8（swish）。`cadex evaluate` で 10/10、平均前進速度 99.6 mm/s、追従誤差 RMS 4.3 mm/s。
- witness の照合: 最大相対誤差 1.1e-7（許容 1e-4）。
- 書き出し: `deploy/esp32/exports/full-base-s0/policy_weights.h`（`export_c --task robot/mjcf/base/cut/walk-task.json`）。
  初期指令は学習のリセットと同じ全関節 0°。サーボの換算は既定値のまま（`servo.json` が無い）なので、
  `firmware/main/` には置いていない。
- ホストの C での照合: GPU 学習の10個の方策すべてで、float32 版は参照との差が行動範囲の 1e-5 未満、
  int8 版は tanh 手前の値がビット一致（`tests/test_esp32_export_c.py::test_real_policy_end_to_end`）。
- 量子化の誤差（`exports/full-base-s0/quantize-report.json`、標本は正規化統計からの合成 4096 個。単位は度）:

| 方式 | 最大 | 平均 | 99%点 |
|---|---|---|---|
| float32（C の float 版） | 0.000 | 0.000 | 0.000 |
| 重みだけ int8 | 1.24 | 0.11 | 0.50 |
| 重み・活性値 int8 + LUT（C の int8 版） | 4.65 | 0.52 | 2.23 |

  合成ポリシー（§5）より誤差が大きい。重みは float32 で 47.8 KB と内部 SRAM に載るので、**float32 を使う**。
- 未確認: ESP-IDF でのビルドと推論時間の実測、IMU とサーボの較正、実機での歩行。
- この書き出しは base の方策なので、作る機体 mini には使わない（mini の方策は再調整の結果待ち）。
