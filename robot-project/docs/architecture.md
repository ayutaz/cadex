# 全体構成とデータの流れ

最終確認日: 2026-11-17

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
| 評価 | `evaluation/evaluate_variant.py`、`q8/evaluation.py`、`q8/compare.py` | 固定歩容と学習済み方策を同じ指標で評価し、機体どうしを比べる |
| 学習基盤 | `configs/train_*.json`、`vastai/` | 学習設定と、vast.ai の GPU で学習を回す道具（作業中） |
| 移植 | `deploy/esp32/` | `.cxpolicy` を量子化して ESP32-S3 用の C コードにする（作業中） |
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
        │  build_robot.py: STL をバイナリ化、見た目用 MJCF（visual.xml）を追加
        ▼
robot/meshes/<variant>/、robot/mjcf/<variant>/、artifacts/<variant>/design_report.json
        │
        ├─ simulation/fixed_gait.py ──────── MuJoCo（CPU）で固定歩容 → artifacts/base/fixed_gait/
        │
        ├─ training/cadex_train.py ───────── MuJoCo MJX / JAX（CPU、将来は vast.ai の GPU）で PPO
        │        ▼
        │   .cxpolicy（重み・観測の正規化）→ artifacts/runs/<run>/
        │        │
        │        ├─ ./cadex evaluate ──────── 成功条件・固定シード・動画 → artifacts/<variant>/evaluation/policy/
        │        └─ deploy/esp32/ ─────────── 量子化 → C コード → ESP32-S3（作業中）
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
| 移植 | `.cxpolicy` | C のソースとファームウェア | `deploy/esp32/`（作業中） |

## Cadex のプロジェクトストア

`scripts/build_robot.py` は、機体ごとに Cadex のプロジェクトを使います（`q8/paths.py` の `cadex_project`）。

| 機体 | プロジェクト |
|---|---|
| base（基準機体） | `~/cadex-projects/quadruped8` |
| その他（例: leg120） | `~/cadex-projects/quadruped8-<variant>` |

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
