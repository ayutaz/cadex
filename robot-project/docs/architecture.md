# 全体構成とデータの流れ

最終確認日: 2026-11-18

## 2つの層

このリポジトリには、**Cadex 本体**と、それを使う利用者側のプロジェクト **robot-project** が同居しています。
robot-project は Cadex 本体を一切変更していません。Cadex へは公開された CLI（`./cadex ... --json`）と、
`training/cadex_train.py` をサブプロセスとして呼ぶことでのみ触れます（`q8/cadex.py`、`scripts/redesign.py`）。
例外として、固定歩容の評価（`q8/evaluation.py`）は Cadex の評価指標と同じ計算を使うため、
`src/Mod/cadex` の `CadexDynamics` と `CadexEvaluation`（純 Python + mujoco）を読み込みます（読むだけで変更はしない）。

### Cadex 本体（リポジトリのルート）

| 部分 | 場所 | 役割 |
|---|---|---|
| エンジン | `src/Mod/cadex/` | FreeCAD のフォーク（OCCT）。`cadexd` がプロジェクトごとに起動し、設計スクリプトをサンドボックスの `FreeCADCmd` ワーカーで実行して、検証済みの形状だけを受理する。アセンブリ・MJCF 出力・学習タスク・方策の再生と評価も担う |
| CLI | `./cadex`、`cli/cadex_cli/` | `script`、`params`、`clearance`、`export`、`inventory`、`evaluate` など。`--json` で機械可読な結果を返す |
| ダッシュボード | `pixi run app` | 読み取り専用の閲覧画面（今回は未確認） |
| エージェント連携 | `./cadex mcp` | MCP サーバー（今回は未使用。同じ処理を CLI で実行） |
| トレーナー | `training/cadex_train.py` | Brax PPO（MJX/JAX）。エンジンの外にあり、学習タスク JSON を読んで `.cxpolicy` を書く |

### robot-project（このディレクトリ）

| 部分 | 場所 | 役割 |
|---|---|---|
| 設計スクリプト | `robot/design/quadruped8.py` | 形状・関節・物理パラメータ・センサー・学習タスク（報酬・終了条件・ランダム化・成功条件）をすべて宣言する xscript |
| 生成と検証 | `scripts/build_robot.py`、`q8/cadex.py`、`q8/clearance.py`、`q8/physics.py`、`q8/manufacturing.py` | Cadex を呼んで生成・干渉計測・出力し、質量・保持トルク・印刷可否・BOM をまとめる |
| シミュレーション | `simulation/fixed_gait.py`、`q8/gait.py`、`q8/robot.py` | 逆運動学による固定歩容を MuJoCo で再生 |
| 評価 | `evaluation/evaluate_variant.py`、`q8/evaluation.py`、`q8/compare.py`、`q8/seeds.py`、`scripts/compare_seeds.py` | 固定歩容と学習済み方策を同じ指標で評価し、機体どうしを学習シード方向に比べる |
| 実機との差の評価 | `evaluation/evaluate_robustness.py`、`q8/robustness.py`、`scripts/compare_robustness.py` | エンジンの評価（`CadexDynamics.evaluate_success`、`cadex evaluate` と同じ関数）をそのまま使い、摩擦・サーボの剛性とトルク・胴体の質量と重心をモデルに、サーボの遅れ・制御周期のゆらぎ・ガタ・横からの押しを 2 ms ごとに入れて評価する |
| ランダム化の学習 | `q8/dr_bundle.py`、`scripts/make_dr_bundle.py`、`configs/train_full_mini-r2-p7-eff-dr*.json` | 学習用カットモデルとタスクに、サーボの一次遅れ・剛性・重心のずれ・押しのランダム化を足す（学習器は変えない）。足の摩擦とトルクは学習器のフラグで |
| 学習基盤 | `configs/train_*.json`、`vastai/` | 学習設定と、vast.ai の GPU で学習を回す道具（複数台で待ち行列を回す `sweep` を含む） |
| 組立 | `scripts/assembly_guide.py` | 印刷の向き・プレート割り・差し込めるかを STL から判定し、組立の段階図を描く |
| 移植 | `deploy/esp32/`（`cxpolicy_io`、`quantize`、`export_c`、`budget`） | `.cxpolicy` を ESP32-S3 用の C コードにする（ホストでの照合まで確認。ファームウェアは ESP-IDF で未ビルド） |
| サーボの試験・較正 | `deploy/esp32/firmware/servo_cal.c`・`servo_console.c`・`servo_hal.h`、`deploy/esp32/servo_calibration.py` | ファームウェアは既定でこのモードで起動し、方策を動かさない（`Q8_RUN_POLICY=1` のときだけ方策の制御ループ）。シリアルのコマンドで関節ごとに中立・向き・範囲を測って NVS に保存し、パソコン側で受け取って検査し、`servo.json` と `servo_map.h` を作る（ホストのテストだけで確認） |
| 変換 | `q8/urdf.py`、`scripts/mjcf_to_urdf.py` | MJCF → URDF |
| 描画 | `q8/render.py`、`q8/visual.py`、`scripts/render_views.py` | 静止画・動画 |

## データの流れ

```
robot/design/quadruped8.py（xscript）
        │  ./cadex script --set / params --set
        ▼
~/cadex-projects/quadruped8[-<variant>]   ← Cadex のプロジェクトストア（独自の git 履歴）
        │  cadexd → FreeCADCmd ワーカー（サンドボックス）で実行、OCCT の BREP を受理
        │  ./cadex clearance / clearance --sweep（干渉）、./cadex inventory（部品一覧）
        │  ./cadex export
        ▼
STEP / STL（ASCII）/ MJCF（model-model.xml）/ 学習タスク（walk-task.json）
        │  build_robot.py: STL をバイナリ化、見た目用 MJCF（visual.xml）、学習用カットモデル（cut/）を追加
        ▼
robot/meshes/<variant>/、robot/mjcf/<variant>/、artifacts/<variant>/design_report.json
        │
        ├─ scripts/assembly_guide.py ─────── 印刷の計画と組立の段階図 → artifacts/<variant>/assembly/
        │
        ├─ simulation/fixed_gait.py ──────── MuJoCo（CPU）で固定歩容 → artifacts/base/fixed_gait/
        │
        ├─ scripts/make_dr_bundle.py ─────── cut/ にランダム化を足す → artifacts/mini-r2-p7-eff-dr/cut/
        │        │                          （サーボの一次遅れ・剛性・重心のずれ・押し）
        │        ▼
        ├─ training/cadex_train.py ───────── MuJoCo MJX / JAX（手元の CPU か vast.ai の GPU）で PPO
        │        ▼                          （dr の案は --randomise-friction と --randomise-drive-torque も）
        │   .cxpolicy（重み・観測の正規化）→ artifacts/runs/<run>/、artifacts/training/<run>/
        │        │
        │        ├─ ./cadex evaluate ──────── 成功条件・固定シード・動画 → artifacts/<variant>/evaluation/policy/
        │        ├─ evaluation/evaluate_robustness.py
        │        │      エンジンの評価に実機との差（摩擦・遅れ・ゆらぎ・ガタ・剛性・質量・重心・押し）を重ねる
        │        │      → artifacts/<案>/robustness/<run>/（scripts/compare_robustness.py で案どうしを比べる）
        │        └─ deploy/esp32/export_c ─── C コード（float32 か int8）→ deploy/esp32/exports/<run>/
        │                 │                   （servo.json でパルス幅に換算）
        │                 ▼
        │           ESP32-S3 のファームウェア（実機は未確認。方策の制御ループは Q8_RUN_POLICY=1 のときだけ）
        │                 ▲
        │           servo.json、servo_map.h
        │                 │  deploy/esp32/servo_calibration.py（read / parse → check → servo-json / header）
        │           較正の JSON ← ファームウェアの試験・較正モード（既定）で、関節ごとに中立・向き・範囲を測る
        │
        └─ evaluation/evaluate_variant.py ── 固定歩容と方策を同じ CadexEvaluation の指標で並べる
```

### 各段階の入出力

| 段階 | 入力 | 出力 | 実行するもの |
|---|---|---|---|
| 設計 | `quadruped8.py`、パラメータ | Cadex プロジェクト内の受理済みリビジョン | `cadexd` と `FreeCADCmd`（Cadex） |
| 検証 | 受理済みリビジョン | 干渉レポート（Markdown）、部品一覧 | `cadex clearance`、`cadex inventory` |
| 出力 | 受理済みリビジョン | STEP、STL、MJCF、学習タスク JSON | `cadex export` |
| 物理チェック | MJCF | 質量・重心・支持余裕・保持トルク | `q8/physics.py`（MuJoCo） |
| 固定歩容 | `visual.xml` | 判定 JSON、CSV、MP4 | `simulation/fixed_gait.py`（MuJoCo） |
| 学習 | `walk-task.json` | `.cxpolicy` | `training/cadex_train.py`（Brax PPO、MJX） |
| 評価 | MJCF と `.cxpolicy` | `evaluation.json`、トレース、動画 | `cadex evaluate`（エンジンが方策を再生） |
| ランダム化の学習 | 学習用カットモデルとタスク（`robot/mjcf/mini-r2-p7-eff/cut/`） | ランダム化を足したバンドル（`cut.xml`、`walk-task.json`、`dr.json`）→ 学習して `.cxpolicy` | `scripts/make_dr_bundle.py`（`q8/dr_bundle.py`）、`training/cadex_train.py` |
| 実機との差の評価 | 機体のモデルとタスク、`.cxpolicy` | `robustness.json`、`summary.md`（組は nominal・sweep・random・push） | `evaluation/evaluate_robustness.py`（`q8/robustness.py`。エンジンの評価の関数をそのまま使う） |
| 移植 | `.cxpolicy`、タスク、`servo.json` | C のヘッダ（`policy_weights.h`）とファームウェア | `deploy/esp32/`（ホストでの照合まで。実機は未確認） |
| サーボの試験・較正 | ファームウェアの試験・較正モードの `dump` の出力（`CAL ...` の行） | 較正の JSON、`servo.json`（`export_c --servo` 用）、`servo_map.h` | `deploy/esp32/firmware/`（既定のモード）、`deploy/esp32/servo_calibration.py`（ホストのテストだけで確認） |

## Cadex のプロジェクトストア

`scripts/build_robot.py` は、機体ごとに Cadex のプロジェクトを使います（`q8/paths.py` の `cadex_project`）。

| 機体 | プロジェクト |
|---|---|
| base（基準機体） | `~/cadex-projects/quadruped8` |
| その他（leg120、mini、mini-r2、学習タスクを変えた案 mini-v3 / mini-r2-p7-eff など） | `~/cadex-projects/quadruped8-<variant>` |

## 機体（variant）

機体はすべて同じ設計スクリプトから、`scripts/build_robot.py --variant V --set k=v ...`（または
`scripts/redesign.py --scale`）でパラメータだけを変えて作ります。生成物は機体ごとに
`robot/meshes/V/`、`robot/mjcf/V/`、`artifacts/V/` に分かれます。

| 機体 | 何か |
|---|---|
| `base` | 基準機体（胴体 110 × 64 mm、電池 30 g の仮置き） |
| `leg120` | base の上腿・下腿を 1.2 倍 |
| `mini` | 前の小型機。胴体 96 × 62 mm、上腿 40 mm、下腿 44 mm、`hip_x` 40 mm、電池なし（`battery_mass_g=0`）。STL では脚が組めないと分かり、mini-r2 に置き換えた |
| `mini-r2` | **作る機体**。mini の6つの値に `leg_rev=2` を足し、脚を組めるように直した形（膝サーボを内側から差し込む、ホーンの凹みをサーボ側の面に、リード線の口）。220.1 g。理由は [README](../README.md)、組立は [assembly-guide.md](assembly-guide.md) |
| `mini-cmd`、`mini-v3`、`mini-lift`、`mini-p6`、`mini-p7`、`mini-eff`、`mini-p7-eff` | mini と同じ形で、学習タスク（指令速度・報酬の重み・周期・トルクの罰）だけを変えた再調整の案。比べ終えた（[experiment-results.md](experiment-results.md) §7・§8） |
| `mini-r2-eff`、`mini-r2-p7-eff` | mini-r2 の形に、mini-eff・mini-p7-eff と同じ学習タスクを載せた案。mini-r2-p7-eff が良かった（§9） |
| `mini-r2-p7-eff-dr`（と `-dr-2k`） | `build_robot.py` の機体ではなく、mini-r2-p7-eff の学習用カットモデルにランダム化を足した学習だけの案（`artifacts/mini-r2-p7-eff-dr/cut/`）。評価は元の mini-r2-p7-eff のモデルとタスクで行う（`compare_seeds.py --as`）。実機の候補はこの案の乱数 4 の回（§11） |
| `taskv2`、`taskv3` | base の形で、報酬を見直したときの学習タスク（[experiment-results.md](experiment-results.md)） |

プロジェクトの中には、Cadex が管理する `script.py`（設計スクリプトの写し）、`script.json`（パラメータ）、
`docs/`（干渉・部品一覧などのレポート）、`review/` などがあり、Cadex 自身が操作ごとに git へコミットします
（例: `cadex clearance → docs/clearance.md`、`cadex export → ...`）。この git は robot-project のリポジトリとは
別物です。設計の正本は `robot-project/robot/design/quadruped8.py` で、プロジェクトストアはそこから
いつでも作り直せます。`design_report.json` の `accepted_revision` が、出力に使った Cadex のリビジョンです。

場所は環境変数 `CADEX_PROJECTS` で変えられます。ダッシュボード（`pixi run app`）は既定で
`~/cadex-projects` を表示するので、ここに置くとダッシュボードからも見えるはずです（未確認）。

## 設計上の取り決め

- **設計スクリプトが唯一の正本。** 形状だけでなく、関節制限・アクチュエータ・センサー・報酬・成功条件まで
  `quadruped8.py` に書く。派生機体はパラメータを変えるだけで作る（`scripts/redesign.py`）。
- **Cadex を改変しない。** 足りない機能（URDF、BOM の価格、固定歩容の評価、STL のバイナリ化、
  見た目用 MJCF）は robot-project 側で補う。
- **生成物は再生成できる。** STL・学習の実行記録・一時ファイルは `.gitignore` 対象。STEP、MJCF、
  学習タスク、`design_report.json` はコミットする。
- **成果物にマシン固有の絶対パスを書かない。** Cadex（`cadex evaluate`、学習器、カット処理）は、
  受け取ったパスを絶対パスにして記録する。そこで、robot-project に成果物を書く・写す箇所
  （`build_robot.py` の `design_report.json`、学習用カットモデル、`archive_run` による学習結果の保存、
  `evaluate_variant.py` の評価結果）で `q8/portable.py` が既知の根を置き換える。
  - robot-project → `<robot-project>`、Cadex のチェックアウト → `<cadex>`、
    `~/cadex-projects` → `~/cadex-projects`、ホーム → `~`
  - `.cxpolicy` はヘッダだけを書き換え、重みのバイト列は変えない。
  - 学習器の標準出力に混ざる警告の行は落とし、`receipt.json` を JSON として保存する。
  - `tests/test_portable.py` は、追跡中のファイルにこのマシンのホームのパスがあると失敗する。
    手で置いたファイルが引っかかったら `uv run python -c "from q8.portable import scrub_tree; scrub_tree('<dir>')"` で直す。
