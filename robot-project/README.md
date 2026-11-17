# quadruped8: Cadex で設計する8自由度四足歩行ロボット

最終確認日: 2026-11-17

MG90S マイクロサーボを8個使い、家庭用FDMプリンタで印刷できる小型の四足歩行ロボットを、
Cadex（このリポジトリの本体。OCCT 上のヘッドレスCADエンジン）で設計し、MuJoCo で動かし、
強化学習で歩容を学ばせ、最終的に ESP32-S3 へ移植するためのプロジェクトです。

`robot-project/` は Cadex の **外側** にある利用者側のプロジェクトです。Cadex 本体
（エンジン・CLI・ダッシュボード・`training/` のトレーナー）は呼び出すだけで、改変していません。

## 今の方針と状態（2026-11-17）

まず「小さく、簡単に、安く」1台作ります。作る機体は小型機 **`mini`** です。**実機はまだ製作していません。**

| 項目 | 状態 |
|---|---|
| 作る機体 | `mini`（8自由度、MG90S × 8、電池なし。胴体 96 × 62 mm、上腿 40 mm、下腿 44 mm、228.4 g。設計レポートの判定はすべて合格: `artifacts/mini/design_report.json`） |
| 電源 | 最初は電池を載せず、スマホの充電器か 5V 4A の AC アダプタから給電（[docs/esp32-deployment.md](docs/esp32-deployment.md) §8） |
| 3D プリンタ | Bambu Lab A1 mini を購入予定（造形 180 × 180 × 180 mm）。足先は PLA で印刷して滑り止めシートを貼る（[docs/manufacturing.md](docs/manufacturing.md)） |
| 購入部品 | [docs/shopping-list.md](docs/shopping-list.md)（最小構成 ¥4,998: MG90S 10個と PLA 1 kg） |
| 制御基板 | ESP32-S3 は手元にある（型番・寸法は未確認。設計は 63 × 25.5 mm の DevKitC-1 を想定） |
| 学習 | mini を GPU で学習シード5回学習した結果、転ばず指令速度で進むが、足を上げずに滑る（合格率 0.24、滑り 0.83。[docs/experiment-results.md](docs/experiment-results.md) §6）。指令速度や報酬の重みを小型機に合わせる再調整中（結果は未確定） |

### 機体の候補（base / leg120 / mini）

3つとも同じ設計スクリプト `robot/design/quadruped8.py` から、パラメータだけを変えて作ります。

| 機体 | 胴体 | 上腿 / 下腿 | 電池 | 総質量 | 役割 |
|---|---|---|---|---|---|
| `base` | 110 × 64 mm | 45 / 50 mm | 30 g の仮置き | 275.2 g | 基準機体。設計・学習・ESP32 への書き出しの手順を最初に通した |
| `leg120` | 110 × 64 mm | 54 / 60 mm | 30 g の仮置き | 280.5 g | 脚を20%長くした派生機体。GPU 学習で学習シード5回とも合格（合格率 1.00） |
| **`mini`** | **96 × 62 mm** | **40 / 44 mm** | **なし** | **228.4 g** | **作る機体**。base より小さく軽い |

値は各機体の `artifacts/<variant>/design_report.json` から取りました。

mini にした理由:

- 依頼者は初心者で、まず小さく・簡単に・安く作りたい。電池と充電・降圧の回路を後回しにし、外部給電で始める。
- 胴体 96 × 62 mm は、MG90S 8個が収まる設計スクリプトの下限（`body_length` の min 96、`body_width` の min 62）。
- SG90 版も設計の判定は通った（依頼者との検討での試算: 194 g、部品代 約1,200円安い。設計レポートはリポジトリに残していない）。
  ただし Amazon の10個セットでは MG90S のほうが安いことがあり、樹脂ギアの SG90 は欠けやすいので、MG90S のままにした。
- 学習では、leg120 のほうが今は良く歩く（合格率 1.00 と 0.24）。mini は小型機に合わせた学習の再調整中。

## 目的

1. 設計スクリプト1本（xscript）から、印刷できる部品・購入部品表・物理モデルを一貫して作る。
2. 干渉・質量・保持トルク・印刷可否・予算を、数値で検証してから製作に進む。
3. 学習なしの固定歩容と、学習した方策（`.cxpolicy`）を、同じ指標で評価する。
4. 寸法を変えた派生機体（脚20%増しの `leg120`、小型機の `mini`）を1コマンドで作り直し、比較する。
5. 学習済み方策を ESP32-S3 で動かせる形に変換する（基準機体の方策で書き出しまで確認済み。mini の方策は再調整の結果待ち）。

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
    render_views.py         MJCF を複数視点で静止画に描く
    mjcf_to_urdf.py         MJCF → URDF
  simulation/fixed_gait.py  学習なしの固定モーション（立位・関節確認・トロット・旋回）を MuJoCo で再生
  evaluation/evaluate_variant.py  固定歩容と学習済み方策を同じ成功条件・シード・指標で評価
  configs/                  BOM の価格表（bom_prices.json）、学習設定（train_smoke.json, train_full.json, train_full_<variant>.json）
  vastai/                   vast.ai の GPU で学習を回す道具（借りる・送る・起動・監視・回収・並列の sweep）
  deploy/esp32/             .cxpolicy を ESP32-S3 用の C コードへ変換する道具とファームウェア（ファームウェアは ESP-IDF で未ビルド）
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
uv run python scripts/build_robot.py        # 基準機体を生成・検証（artifacts/base/design_report.json）
uv run python scripts/build_robot.py --variant mini --set body_length=96 --set body_width=62 \
  --set battery_mass_g=0 --set hip_x=40 --set thigh_length=40 --set shin_length=44   # 作る機体 mini
uv run python simulation/fixed_gait.py robot/mjcf/base/visual.xml artifacts/base/fixed_gait --video
uv run python scripts/redesign.py --variant leg120 --scale thigh_length,shin_length=1.2
uv run python evaluation/evaluate_variant.py --variant base --policy <path>.cxpolicy
uv run python -m pytest tests -q
```

| コマンド | 何をするか | 主な出力 |
|---|---|---|
| `scripts/build_robot.py [--variant V] [--set k=v ...]` | 設計スクリプトを Cadex プロジェクトへ書き、パラメータを適用してビルド → 全部品の干渉と関節スイープの干渉を計測 → STEP/STL/MJCF/学習タスクを出力 → 質量・重心・保持トルク・印刷可否・BOM をまとめる | `robot/meshes/V/`、`robot/mjcf/V/`、`artifacts/V/design_report.json`、`artifacts/V/renders/` |
| `simulation/fixed_gait.py MODEL OUT [--video]` | 学習なしの固定モーションを再生して判定する | `OUT/fixed_gait.json`、`OUT/<場面>.csv`、`OUT/<場面>.mp4` |
| `scripts/redesign.py --variant V --scale a,b=倍率` | 基準機体の値に倍率を掛けた派生機体を作り、検証・評価して基準と比べる（`--train local` で CPU 学習も） | `artifacts/V/`、`artifacts/compare/base_vs_V.{md,json}` |
| `evaluation/evaluate_variant.py --variant V [--policy P]` | 固定歩容（MuJoCo 上で開ループ）と学習済み方策（`cadex evaluate`）を同じ10シードで評価 | `artifacts/V/evaluation/` |
| `scripts/render_views.py MODEL OUT [--collision]` | 静止画を描く | PNG |
| `scripts/mjcf_to_urdf.py --variant V` | MJCF から URDF を作る | `robot/urdf/V/quadruped8.urdf` |
| `python -m vastai create --count N` / `sweep` | vast.ai の GPU を検索の直後に N 台借り、設定 × 学習の乱数の回を並べて回す（手順は [docs/training-vastai.md](docs/training-vastai.md)） | `artifacts/runs/vastai/<run>/` |
| `scripts/compare_seeds.py A=<回> ... B=<回> ...` | 学習の乱数違いの複数回を評価し、二機体を 平均 ± 標準偏差（最小〜最大）で比べる | `artifacts/compare/seeds_A_vs_B.{md,json}` |

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
| `artifacts/compare/` | 機体どうしの比較（`base_vs_leg120.md`、`seeds_base_vs_leg120.md`、`seeds_leg120_vs_mini.md`） |
| `artifacts/training/<run>/` | GPU 学習の記録と方策（例: `full-mini-s0`） |
| `artifacts/runs/<run>/` | 学習の実行記録と `.cxpolicy`（.gitignore 対象） |
| `robot/meshes/<variant>/*.step` | 部品ごとの STEP（製造・CAD 連携用） |
| `robot/meshes/<variant>/*.stl` | 部品ごとのバイナリ STL（`build_robot.py` で再生成。1機体 約19 MB のためコミットしない） |
| `robot/mjcf/<variant>/` | `model-model.xml`（Cadex の出力そのまま）、`visual.xml`（見た目用メッシュ付き。物理は同じ）、`walk-task.json`（学習タスク） |

## テスト

```bash
cd robot-project
uv run python -m pytest tests -q
```

`test_fixed_gait_sim.py` など MuJoCo を動かすテストを含みます。Cadex 本体のテストは
リポジトリのルートで `pixi run test-engine` と `pixi run python -m pytest cli/tests` です
（結果は [docs/cadex-capabilities.md](docs/cadex-capabilities.md)）。

## ドキュメント

| 文書 | 内容 | 状態 |
|---|---|---|
| [docs/cadex-capabilities.md](docs/cadex-capabilities.md) | Cadex で何ができて何ができないか（確認済み・未確認の区別） | あり |
| [docs/architecture.md](docs/architecture.md) | 全体構成とデータの流れ | あり |
| [docs/hardware-spec.md](docs/hardware-spec.md) | 機体仕様、座標系、関節、サーボ、確認済みの値と推定値 | あり |
| [docs/manufacturing.md](docs/manufacturing.md) | 印刷部品・購入部品・BOM・組立手順の案・未検証事項 | あり |
| [docs/shopping-list.md](docs/shopping-list.md) | 初心者向けの購入部品の一覧（mini 用。最小構成・後で買う物・注意・工具・価格の出典） | あり |
| [docs/simulation.md](docs/simulation.md) | MuJoCo モデルの中身、固定モーションの結果、Sim2Real の差 | あり |
| [docs/training.md](docs/training.md) | 強化学習の設定と手順 | あり |
| [docs/training-vastai.md](docs/training-vastai.md) | vast.ai の GPU での学習 | あり |
| [docs/experiment-results.md](docs/experiment-results.md) | 実験結果（固定歩容、CPU 学習、GPU 学習: base・leg120・mini） | あり |
| [docs/esp32-deployment.md](docs/esp32-deployment.md) | ESP32-S3 への移植、電源と配線の注意 | あり |
| [docs/urdf.md](docs/urdf.md) | URDF への変換 | あり |

Cadex 本体の文書: [`../docs/XSCRIPT.md`](../docs/XSCRIPT.md)（設計スクリプト）、
[`../docs/CLI.md`](../docs/CLI.md)（CLI）、[`../docs/MUJOCO.md`](../docs/MUJOCO.md)（動力学と学習）、
[`../training/RECIPE.md`](../training/RECIPE.md)（学習の手順）。

## ライセンス

設計スクリプトは Cadex に合わせて `LGPL-2.1-or-later` です（ファイル冒頭の SPDX ヘッダ）。
