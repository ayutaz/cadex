# quadruped8: Cadex で設計する8自由度四足歩行ロボット

最終確認日: 2026-11-18

MG90S マイクロサーボを8個使い、家庭用FDMプリンタで印刷できる小型の四足歩行ロボットを、
Cadex（このリポジトリの本体。OCCT 上のヘッドレスCADエンジン）で設計し、MuJoCo で動かし、
強化学習で歩き方を学ばせ、最終的に ESP32-S3 へ移植するためのプロジェクトです。

`robot-project/` は Cadex の **外側** にある利用者側のプロジェクトです。Cadex 本体
（エンジン・CLI・ダッシュボード・`training/` のトレーナー）は呼び出すだけで、改変していません。

## 今の方針と状態（2026-11-18）

まず「小さく、簡単に、安く」1台作ります。作る機体は小型機 **`mini-r2`** です。

**実機はまだ製作していません。** ここに書いた結果は、すべてシミュレーションとパソコン上のテストで出したものです。

| 項目 | 状態 |
|---|---|
| 作る機体 | `mini-r2`（8自由度、MG90S × 8、電池なし。胴体 96 × 62 mm、上腿 40 mm、下腿 44 mm、`hip_x` 40 mm、220.1 g。設計レポートの判定はすべて合格: `artifacts/mini-r2/design_report.json`）。前の小型機 `mini` の脚を組めるように直したもの（下の「mini-r2 にした理由」） |
| 歩き方（学習） | 実機の候補は `artifacts/training/full-mini-r2-p7-eff-dr-s4/walk.cxpolicy`。シミュレーションの普通の評価で 10/10 合格、実機との差を想定した評価で合格率 0.86（[docs/experiment-results.md](docs/experiment-results.md) §11） |
| ESP32 向けの書き出し | `deploy/esp32/exports/full-mini-r2-p7-eff-dr-s4/`（float32）。ファームウェア（`deploy/esp32/firmware/main/`）にはまだ入れていない |
| ファームウェア | 何も指定しないと **サーボの試験・較正モード** で起動し、方策（学習した歩き方）は動かさない（[docs/servo-calibration.md](docs/servo-calibration.md)）。ESP-IDF（ESP32 の開発環境）は未インストールで、ビルドも実機での動作も未確認 |
| 電源 | 最初は電池を載せず、スマホの充電器か 5V 4A の AC アダプタから給電（[docs/esp32-deployment.md](docs/esp32-deployment.md) §8） |
| 3D プリンタ | Bambu Lab A1 mini を購入予定（造形 180 × 180 × 180 mm）。印刷と組立の手順は [docs/assembly-guide.md](docs/assembly-guide.md)（mini-r2 用） |
| 購入部品 | [docs/shopping-list.md](docs/shopping-list.md)（最小構成 ¥4,998: MG90S 10個と PLA 1 kg）。まだ買っていない |
| 制御基板 | ESP32-S3 は手元にある（型番・寸法は未確認。設計は 63 × 25.5 mm の DevKitC-1 を想定） |
| GPU の費用 | vast.ai の累計 約 $10.5（予算 $100。表示の $/h と借りた時間からの概算で、請求額は未確認） |

### 段階

| 段階 | 中身 | 状態 |
|---|---|---|
| A | シミュレーションでの歩き方（設計、学習、評価。実機との差に強くする学習を含む） | 完了 |
| B | 印刷データと組立手順（[docs/assembly-guide.md](docs/assembly-guide.md)） | 完了 |
| C の準備 | サーボの試験・較正の道具（ファームウェアの較正モードと `deploy/esp32/servo_calibration.py`） | 完了（パソコン上のテストだけ） |

残りの作業（この順に進める予定）:

1. 部品の購入（[docs/shopping-list.md](docs/shopping-list.md)）
2. ESP-IDF の導入とファームウェアのビルド
3. サーボの較正（[docs/servo-calibration.md](docs/servo-calibration.md)）
4. IMU（傾きセンサー）の購入と実装（方策が胴体の傾きを入力に使うため。[docs/esp32-deployment.md](docs/esp32-deployment.md) §7）
5. 実機での試験

### 機体の候補（形）

どれも同じ設計スクリプト `robot/design/quadruped8.py` から、パラメータだけを変えて作ります。

| 機体 | 胴体 | 上腿 / 下腿 | 電池 | 総質量 | 役割 |
|---|---|---|---|---|---|
| `base` | 110 × 64 mm | 45 / 50 mm | 30 g の仮置き | 275.2 g | 基準機体。設計・学習・ESP32 への書き出しの手順を最初に通した（GPU 学習で合格率 0.92） |
| `leg120` | 110 × 64 mm | 54 / 60 mm | 30 g の仮置き | 280.5 g | 脚を20%長くした派生機体（GPU 学習で合格率 1.00） |
| `mini` | 96 × 62 mm | 40 / 44 mm | なし | 228.4 g | 前の小型機。脚が組めない形だったので作らない（mini-r2 に置き換えた） |
| **`mini-r2`** | **96 × 62 mm** | **40 / 44 mm** | **なし** | **220.1 g** | **作る機体**。mini の脚を組めるように直したもの（`leg_rev=2`） |

値は各機体の `artifacts/<variant>/design_report.json` から取りました。

### mini-r2 で学んだ歩き方の案

形は mini-r2 のまま、学習のしかた（報酬やランダム化）だけを変えた案です。どちらも学習の乱数を変えて5回ずつ学習しました。

| 案 | 何を変えたか | 普通の評価の合格率 | 実機との差を想定した評価の合格率 | トルク / 推定定格 | 滑り |
|---|---|---|---|---|---|
| `mini-r2-p7-eff` | 歩く周期を 0.7 秒に延ばし、推定定格を超えるトルクの罰を 1.0 に強めた | 0.90（5回の平均） | 0.60 | 0.75 | 0.38 |
| **`mini-r2-p7-eff-dr`** | 上に、実機との差（足の摩擦、サーボの遅れ・剛性・トルク、重心のずれ、押し）のランダム化を足した | **1.00（5回とも 10/10）** | **0.84** | **0.61** | **0.27** |

数字は普通の評価のもの（「実機との差」の列だけ別）。詳しくは [docs/experiment-results.md](docs/experiment-results.md) §9・§11、
ランダム化の中身は [docs/training.md](docs/training.md) §5.2。実機の候補は、dr の5回（どれも普通の評価で 10/10）から、実機との差の評価と向きのずれで選んだ **乱数 4 の回**です。

### mini-r2 にした理由

- 依頼者は初心者で、まず小さく・簡単に・安く作りたい。電池と充電・降圧の回路を後回しにし、外部給電で始める。
- 胴体 96 × 62 mm は、MG90S 8個が収まる設計スクリプトの下限（`body_length` の min 96、`body_width` の min 62）。
- 組立の手順書を作る中で、前の `mini` は STL では脚が組めないと分かった（膝サーボのタブが閉じた溝に入る、ホーンの凹みが板の中に閉じている、
  膝サーボのリード線の出口が無い）。`mini-r2` はこれを直した形で、ほかの寸法は mini と同じ。
  - 膝サーボは上腿の内側（胴体の側）から差し込み、取付タブをネジで留める。
  - 十字ホーンの凹みは、上腿・下腿のサーボ側の面にある。
  - 膝サーボのリード線を出す口がある。
  - そのぶん足先は mini より左右に 7 mm ずつ外へ出た。
- 組立の要所7項目の判定（`artifacts/mini-r2/assembly/print_plan.json` の `assembly_verdict`）はすべて合格（CAD 上の確認で、実物では未確認）。
- サーボは MG90S のまま。SG90 版も設計の判定は通った（試算 194 g、部品代 約1,200円安い。設計レポートは残していない）が、
  Amazon の10個セットでは MG90S のほうが安いことがあり、樹脂ギアの SG90 は欠けやすい。

## 目的

1. 設計スクリプト1本（xscript）から、印刷できる部品・購入部品表・物理モデルを一貫して作る。
2. 干渉・質量・保持トルク・印刷可否・組み立てられるか・予算を、数値で検証してから製作に進む。
3. 学習なしの固定歩容と、学習した方策（`.cxpolicy`）を、同じ指標で評価する。実機との差（摩擦・遅れ・ガタ・押しなど）を入れた評価もする。
4. 寸法を変えた派生機体（脚20%増しの `leg120`、小型機の `mini`・`mini-r2`）を1コマンドで作り直し、比較する。
5. 学習済み方策を ESP32-S3 で動かせる形に変換し、サーボを較正して実機で歩かせる（書き出しと較正の道具までできた。実機は未確認）。

## ディレクトリ構成

```
robot-project/
  pyproject.toml, uv.lock   学習・評価用の Python 環境（uv、Python 3.13）
  robot/
    design/quadruped8.py    設計スクリプト（Cadex の xscript）。すべての形状・関節・学習タスクの源
    meshes/<variant>/       部品ごとの STEP（コミット対象）と STL（再生成できるので .gitignore）
    mjcf/<variant>/         Cadex が出力した MJCF（model-model.xml）、見た目用 MJCF（visual.xml）、学習タスク（walk-task.json）
    urdf/<variant>/         MJCF から変換した URDF（docs/urdf.md）
  q8/                       Cadex の外側の道具一式（CLI ラッパー、運動学、歩容、評価、製造、描画、URDF 変換）
  scripts/
    build_robot.py          機体を生成・検証し、artifacts/<variant>/design_report.json を書く
    redesign.py             寸法変更 → 再生成 → 再検証 →（再学習）→ 評価 → 比較を1コマンドで
    assembly_guide.py       印刷の向き・プレート割り・差し込めるかを判定し、組立の段階図を描く
    compare_seeds.py        学習の乱数違いの複数回を評価し、二つの案を比べる
    compare_robustness.py   同じく、実機との差の評価で二つの案を比べる
    make_dr_bundle.py       実機との差に強くする学習用のモデルとタスクを作る（q8/dr_bundle.py）
    render_views.py         MJCF を複数視点で静止画に描く
    mjcf_to_urdf.py         MJCF → URDF
  simulation/fixed_gait.py  学習なしの固定モーション（立位・関節確認・トロット・旋回）を MuJoCo で再生
  evaluation/
    evaluate_variant.py     固定歩容と学習済み方策を同じ成功条件・シード・指標で評価
    evaluate_robustness.py  方策を、実機との差（摩擦・遅れ・ゆらぎ・ガタ・剛性・質量・重心・押し）を入れて評価（q8/robustness.py）
  configs/                  BOM の価格表（bom_prices.json）、学習設定（train_smoke.json, train_full.json, train_full_<variant>.json）
  vastai/                   vast.ai の GPU で学習を回す道具（借りる・送る・起動・監視・回収・並列の sweep）
  deploy/esp32/             .cxpolicy を ESP32-S3 用の C コードへ変換する道具、サーボの較正の道具（servo_calibration.py）、
                            ファームウェア（既定は試験・較正モード。ESP-IDF で未ビルド）
  tests/                    pytest
  docs/                     ドキュメント（下記）
  artifacts/                生成物（下記）
```

Cadex のプロジェクトストア（Cadex が設計スクリプト・ビルド結果・検証レポートを置く場所）は
リポジトリの外の `~/cadex-projects/quadruped8`（派生機体は `~/cadex-projects/quadruped8-<variant>`）にあり、
Cadex 自身がそこで独自の git 履歴を持ちます。詳しくは [docs/architecture.md](docs/architecture.md)。

## セットアップ

前提: macOS（Apple Silicon）で確認しました。Cadex 本体は Linux x86-64 が公式の検証環境です。

### 1. Cadex エンジンをビルドする（リポジトリのルートで）

```bash
# pixi が未導入なら https://pixi.sh の手順で入れる
cd /path/to/cadex
pixi run setup-engine && pixi run build-engine   # 初回は約15分（M4 Max で実測）
pixi run test-engine                             # エンジンのテスト（任意）
```

### 2. robot-project の Python 環境を作る

```bash
cd robot-project
uv sync          # .venv に Python 3.13 と brax / jax / mujoco 3.10 / mujoco-mjx などを入れる
```

パスは環境変数で変えられます（`q8/paths.py`）。

| 変数 | 既定値 | 意味 |
|---|---|---|
| `CADEX_REPO` | `robot-project` の親（このリポジトリ） | `./cadex` CLI と `training/cadex_train.py` の場所 |
| `CADEX_PROJECTS` | `~/cadex-projects` | Cadex のプロジェクトストア |

動画（MP4。`simulation/fixed_gait.py --video` など）の書き出しには `ffmpeg` が必要です（macOS なら `brew install ffmpeg`）。

vast.ai を使う場合は `.env.example` を `.env` に写して値を入れます（`.env` はコミットしない）。

## 主要コマンド

すべて `robot-project/` で実行します。

```bash
cd robot-project
uv sync                                     # 学習・評価用の環境（Python 3.13）
uv run python scripts/build_robot.py --variant mini-r2 --set body_length=96 --set body_width=62 \
  --set battery_mass_g=0 --set hip_x=40 --set thigh_length=40 --set shin_length=44 --set leg_rev=2   # 作る機体 mini-r2
uv run python scripts/assembly_guide.py --variant mini-r2        # 印刷の計画と組立の図
uv run python evaluation/evaluate_robustness.py --variant mini-r2-p7-eff \
  --policy artifacts/training/full-mini-r2-p7-eff-dr-s4/walk.cxpolicy \
  --out artifacts/mini-r2-p7-eff-dr/robustness/full-mini-r2-p7-eff-dr-s4   # 実機との差を入れた評価
uv run python -m deploy.esp32.servo_calibration nominal --out cal.json  # 較正の名目値（未較正）
uv run pytest -q
```

| コマンド | 何をするか | 主な出力 |
|---|---|---|
| `scripts/build_robot.py [--variant V] [--set k=v ...]` | 設計スクリプトを Cadex プロジェクトへ書き、パラメータを適用してビルド → 全部品の干渉と関節スイープの干渉を計測 → STEP/STL/MJCF/学習タスクを出力 → 質量・重心・保持トルク・印刷可否・BOM をまとめる | `robot/meshes/V/`、`robot/mjcf/V/`、`artifacts/V/design_report.json`、`artifacts/V/renders/` |
| `simulation/fixed_gait.py MODEL OUT [--video]` | 学習なしの固定モーションを再生して判定する | `OUT/fixed_gait.json`、`OUT/<場面>.csv`、`OUT/<場面>.mp4` |
| `scripts/redesign.py --variant V --scale a,b=倍率` | 基準機体の値に倍率を掛けた派生機体を作り、検証・評価して基準と比べる（`--train local` で CPU 学習も） | `artifacts/V/`、`artifacts/compare/base_vs_V.{md,json}` |
| `evaluation/evaluate_variant.py --variant V [--policy P]` | 固定歩容（MuJoCo 上で開ループ）と学習済み方策（`cadex evaluate`）を同じ10シードで評価 | `artifacts/V/evaluation/` |
| `evaluation/evaluate_robustness.py --variant V --policy P --out D [--suites ...] [--draws N]` | 方策を、実機との差を入れて評価する。組は4つ: `nominal`（何も変えない。`cadex evaluate` と同じ数字）、`sweep`（差を1つずつ）、`random`（実機を想定した幅から全部を一度に）、`push`（横から押す） | `D/robustness.json`、`D/summary.md` |
| `scripts/assembly_guide.py [--variant V]` | 印刷の向き・張り出し・プレート割り・差し込めるかを STL から判定し、組立の段階図を描く（[docs/assembly-guide.md](docs/assembly-guide.md)） | `artifacts/V/assembly/` |
| `scripts/render_views.py MODEL OUT [--collision]` | 静止画を描く | PNG |
| `scripts/mjcf_to_urdf.py --variant V` | MJCF から URDF を作る | `robot/urdf/V/quadruped8.urdf` |
| `python -m vastai create --count N` / `sweep` | vast.ai の GPU を検索の直後に N 台借り、設定 × 学習の乱数の回を並べて回す（手順は [docs/training-vastai.md](docs/training-vastai.md)） | `artifacts/runs/vastai/<run>/` |
| `scripts/compare_seeds.py A=<回> ... B=<回> ... [--as 案=機体]` | 学習の乱数違いの複数回を評価し、二つの案を 平均 ± 標準偏差（最小〜最大）で比べる。`--as` は、学習だけを変えた案（例: `mini-r2-p7-eff-dr=mini-r2-p7-eff`）を元の機体のモデルとタスクで評価する | `artifacts/<案>/evaluation-seeds/<回>/`、`artifacts/compare/seeds_A_vs_B.{md,json}` |
| `scripts/compare_robustness.py --variant V A=<回> ... B=<回> ...` | 二つの案の各回を実機との差の評価にかけ（評価済みの回は飛ばす）、学習の乱数方向にまとめて比べる | `artifacts/<案>/robustness/<回>/`、`artifacts/compare/robust_A_vs_B.{md,json}` |
| `scripts/make_dr_bundle.py` | mini-r2-p7-eff の学習用カットモデルとタスクに、サーボの一次遅れ・剛性・重心のずれ・押しのランダム化を足す（学習器は変えない） | `artifacts/mini-r2-p7-eff-dr/cut/{cut.xml, walk-task.json, dr.json}` |
| `python -m deploy.esp32.servo_calibration <命令>` | サーボの較正の結果を扱う。`nominal`（名目値の JSON）、`read`（ESP32 からシリアルで受け取る）、`parse`（貼り付けた出力から）、`check`（検査）、`servo-json`（`export_c --servo` 用）、`header`（ファームウェアの `servo_map.h`）。手順は [docs/servo-calibration.md](docs/servo-calibration.md) | 較正の JSON、`servo.json`、`servo_map.h` |
| `python -m deploy.esp32.export_c ...` | `.cxpolicy` を ESP32 用の C ヘッダに書き出す（手順は [docs/esp32-deployment.md](docs/esp32-deployment.md)） | `policy_weights.h` など |

Cadex を直接使う場合（例）:

```bash
cd ..   # リポジトリのルート
./cadex params --project ~/cadex-projects/quadruped8 --set thigh_length=50 --out /tmp/q8-thigh50
./cadex --project ~/cadex-projects/quadruped8 clearance --sweep
```

## 成果物の場所

| 場所 | 中身 |
|---|---|
| `artifacts/<variant>/design_report.json` | パラメータ、質量・重心・支持余裕、静的トルク見積り、MuJoCo での立位保持、干渉、印刷可否、BOM、判定 |
| `artifacts/<variant>/renders/` | 5視点の静止画（`*_collision.png` は衝突形状を重ねたもの） |
| `artifacts/base/fixed_gait/` | 固定モーションの判定（`fixed_gait.json`）、時系列 CSV、動画（`stand.mp4`、`trot.mp4`、`turn.mp4`、`joint_sweep_hip_lf.mp4`） |
| `artifacts/<variant>/evaluation/` | 評価の要約（`summary.json`）、シードごとの指標（`metrics.csv`）、固定歩容の詳細（`fixed_gait.json`） |
| `artifacts/<variant>/evaluation-seeds/` | GPU 学習の学習シードごとの評価 |
| `artifacts/<案>/robustness/<run>/` | 実機との差を入れた評価（`robustness.json`、`summary.md`） |
| `artifacts/mini-r2/assembly/` | 印刷の計画（`print_plan.json`。組立の判定 `assembly_verdict` を含む）と組立の段階図 |
| `artifacts/mini-r2-p7-eff-dr/cut/` | 実機との差に強くする学習用のモデルとタスク（中身の記録は `dr.json`） |
| `artifacts/compare/` | 案どうしの比較（`seeds_<a>_vs_<b>.md` は普通の評価、`robust_<a>_vs_<b>.md` は実機との差の評価。例: `seeds_mini-r2-p7-eff_vs_mini-r2-p7-eff-dr.md`） |
| `artifacts/training/<run>/` | GPU 学習の記録と方策。**実機の候補は `full-mini-r2-p7-eff-dr-s4/walk.cxpolicy`** |
| `deploy/esp32/exports/<run>/` | ESP32 向けに書き出した方策（`policy_weights.h`、`quantize-report.json`）。実機の候補は `full-mini-r2-p7-eff-dr-s4/`。`firmware/main/` には置いていない |
| `artifacts/runs/<run>/` | 学習の実行記録と `.cxpolicy`（.gitignore 対象） |
| `robot/meshes/<variant>/*.step` | 部品ごとの STEP（製造・CAD 連携用） |
| `robot/meshes/<variant>/*.stl` | 部品ごとのバイナリ STL（`build_robot.py` で再生成。1機体 約19 MB のためコミットしない） |
| `robot/mjcf/<variant>/` | `model-model.xml`（Cadex の出力そのまま）、`visual.xml`（見た目用メッシュ付き。物理は同じ）、`walk-task.json`（学習タスク） |

## テスト

```bash
cd robot-project
uv run pytest -q
```

2026-11-18 の結果は **553 passed, 1 skipped**（skip は `tensorboard` が入っていないときの `test_vastai_metrics.py` の1件）。
`test_fixed_gait_sim.py` など MuJoCo を動かすテスト、実機との差の評価が何も変えない条件で `cadex evaluate` と同じ数字になることを
確かめるテスト（`test_robustness.py`）、ファームウェアの較正のロジックをパソコンの clang でビルドして動かすテスト
（`test_servo_cal_c.py`）を含みます。**どれもパソコン上の確認で、ESP32 や実物のサーボでは確かめていません。** Cadex 本体のテストは
リポジトリのルートで `pixi run test-engine` と `pixi run python -m pytest cli/tests` です
（結果は [docs/cadex-capabilities.md](docs/cadex-capabilities.md)）。

## ドキュメント

| 文書 | 内容 | 状態 |
|---|---|---|
| [docs/cadex-capabilities.md](docs/cadex-capabilities.md) | Cadex で何ができて何ができないか（確認済み・未確認の区別） | あり |
| [docs/architecture.md](docs/architecture.md) | 全体構成とデータの流れ | あり |
| [docs/hardware-spec.md](docs/hardware-spec.md) | 機体仕様、座標系、関節、サーボ、確認済みの値と推定値 | あり |
| [docs/manufacturing.md](docs/manufacturing.md) | 印刷部品・購入部品・BOM・組立手順の案・未検証事項 | あり |
| [docs/shopping-list.md](docs/shopping-list.md) | 初心者向けの購入部品の一覧（最小構成 約 ¥5,000・後で買う物・注意・工具・価格の出典） | あり |
| [docs/assembly-guide.md](docs/assembly-guide.md) | 初心者向けの mini-r2 の印刷と組立の手順書（図付き。A1 mini の設定、配線図、mini から直した3か所） | あり |
| [docs/servo-calibration.md](docs/servo-calibration.md) | サーボの試験と較正の手順（ファームウェアの試験・較正モード、安全のきまり、結果の受け取りと検査） | あり（パソコン上のテストだけ） |
| [docs/simulation.md](docs/simulation.md) | MuJoCo モデルの中身、固定モーションの結果、Sim2Real の差 | あり |
| [docs/training.md](docs/training.md) | 強化学習の設定と手順（実機との差に強くするランダム化は §5.2） | あり |
| [docs/training-vastai.md](docs/training-vastai.md) | vast.ai の GPU での学習 | あり |
| [docs/experiment-results.md](docs/experiment-results.md) | 実験結果（固定歩容、CPU 学習、GPU 学習: base・leg120・mini とその再調整・mini-r2、実機との差の評価とランダム化の学習） | あり |
| [docs/esp32-deployment.md](docs/esp32-deployment.md) | ESP32-S3 への移植、方策の書き出し、電源と配線の注意 | あり |
| [docs/urdf.md](docs/urdf.md) | URDF への変換 | あり |

Cadex 本体の文書: [`../docs/XSCRIPT.md`](../docs/XSCRIPT.md)（設計スクリプト）、
[`../docs/CLI.md`](../docs/CLI.md)（CLI）、[`../docs/MUJOCO.md`](../docs/MUJOCO.md)（動力学と学習）、
[`../training/RECIPE.md`](../training/RECIPE.md)（学習の手順）。

## ライセンス

設計スクリプトは Cadex に合わせて `LGPL-2.1-or-later` です（ファイル冒頭の SPDX ヘッダ）。
