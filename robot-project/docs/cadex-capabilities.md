# Cadex でできること・できないこと（quadruped8 で確かめた結果）

最終確認日: 2026-11-17

この文書は、quadruped8 を作る過程で Cadex の各機能を実際に動かして確かめた結果をまとめたものです。
Cadex 本体の文書（`../../README.md`、`../../docs/*.md`）が「何ができるか」を書いているのに対し、
ここでは「この環境で実際に動いたか」を区別して書きます。

## 状態の区分

| 区分 | 意味 |
|---|---|
| **実装済み・実動作確認済み** | 今回、実際に動かして期待どおりの出力を得た |
| **実装済み・未確認** | Cadex にコードと文書はあるが、今回は動かしていない |
| **部分実装** | 動くが、この機体では前提が合わない、または足りない部分を自作で補った |
| **未実装** | Cadex に機能がない（必要なら自作した、または作成中） |

## 確認した環境

| 項目 | 内容 |
|---|---|
| OS / CPU | macOS 26.2 / Apple M4 Max（arm64、16コア、64 GB） |
| GPU | 手元に CUDA GPU はない。本番の学習は vast.ai で借りた RTX 4090 で実行（[training-vastai.md](training-vastai.md)） |
| pixi | `~/.pixi` に導入（シェル設定は変更していない） |
| Cadex のビルド | `pixi run setup-engine && pixi run build-engine` 成功（約15分） |
| robot-project の環境 | uv、Python 3.13、mujoco 3.10.0、mujoco-mjx 3.10.0、brax 0.14.2、jax 0.7.2（`pyproject.toml`） |

## 機能ごとの状態

### CAD 生成と寸法変更

| 機能 | 状態 | 確かめたこと |
|---|---|---|
| xscript（Python の設計スクリプト）による形状生成 | 実装済み・実動作確認済み | `robot/design/quadruped8.py` を Cadex のサンドボックス（`FreeCADCmd` ワーカー）で実行し、OCCT の厳密形状として受理された |
| 宣言パラメータ（`params(...)`）と寸法変更・再生成 | 実装済み・実動作確認済み | `./cadex script --set`、`./cadex params --set k=v` で再生成。派生機体 leg120・mini もパラメータだけで作った。ビルドの所要は `design_report.json` の `timings_s` で base 3.2秒、leg120 6.3秒、mini 10.3秒（干渉計測は別に約20〜31秒） |
| 部品ライブラリ（`lib.servo`、`lib.bolt`、ホーン、収納部 `bay`） | 実装済み・実動作確認済み | MG90S のケース・タブ・スプライン、十字ホーン、M2 ボルトと下穴・通し穴をカタログ値から生成 |
| MCP 経由の操作（`cadex mcp`） | 実装済み・未確認 | ツール面は存在する。今回は同じ処理を CLI（`./cadex ... --json`）から呼び、MCP クライアントからは操作していない |

### 出力

| 形式 | 状態 | 補足 |
|---|---|---|
| STEP | 実装済み・実動作確認済み | 部品ごとに出力（`robot/meshes/<variant>/*.step`） |
| STL | 実装済み・実動作確認済み | Cadex は **ASCII STL** を書く。MuJoCo はバイナリ STL しか読めないため、`q8/stl.py` でバイナリに直して保存 |
| MJCF | 実装済み・実動作確認済み | MuJoCo 3.10 で読み込み・立位保持を確認（[simulation.md](simulation.md)） |
| 学習タスク JSON（`walk-task.json`） | 実装済み・実動作確認済み | `training/cadex_train.py` が読み込んで学習が回った |
| BOM（部品表） | 部分実装 | Cadex の `inventory`（部品とカタログ品番の一覧）はある。価格・フィラメント量・予算判定は `q8/manufacturing.py` で自作 |
| URDF | 未実装 | Cadex に URDF 出力はない。MJCF から変換する `q8/urdf.py` を別途作成（[urdf.md](urdf.md)） |

### アセンブリと検証

| 機能 | 状態 | 確かめたこと |
|---|---|---|
| revolute / fixed 関節、関節制限 | 実装済み・実動作確認済み | 8個の回転関節（股・膝×4脚）と、サーボ・ホーン・電装品の固定関節 |
| 静的な干渉計測（`cadex clearance`） | 実装済み・実動作確認済み | OCCT の厳密形状で base は 595 組、電池の無い mini は 561 組を計測し、問題なし |
| 可動域スイープ（`sweep_step_degrees`、`clearance --sweep`） | 実装済み・実動作確認済み | 5度刻みで全関節を可動域いっぱいに動かし、全組の干渉を計測。下の「スイープで見つけた問題」を参照 |
| 質量・慣性 | 実装済み・実動作確認済み | OCCT 形状の体積×密度から計算。総質量は base 275.2 g（サーボ 107.2 g、PLA 120.0 g、電池の仮置き 30.0 g ほか）、mini 228.4 g（電池なし） |
| 計測結果の機械可読な出力 | 部分実装 | 干渉の結果は Cadex プロジェクトの `docs/clearance.md` などの Markdown に書かれる。`q8/clearance.py` で読み取って JSON にしている |

**スイープで見つけた問題（v1 → 現行）**: 最初の版では、前脚の股を +20° 振ると上腿（膝サーボのハウジング）が
胴体の底に食い込んだ。膝サーボの向きを反転してケースを足先側へ伸ばす形にしたところ、+25° まで干渉なし、
+30° で接触に改善した。そこから2ステップ（10°）手前に関節制限を置き、前脚の股 [-45°, +20°]、
後脚の股 [-35°, +45°]、膝 [-60°, +45°] とした。この範囲で全関節「干渉なし」を確認済み
（`artifacts/base/design_report.json` の `clearance.joints`）。

### 物理モデル（MJCF）

| 機能 | 状態 | 確かめたこと |
|---|---|---|
| 位置サーボ（トルク制限付き） | 実装済み・実動作確認済み | 剛性・減衰・トルク上限が MJCF の `<general>` に出る |
| 関節の粘性・アーマチュア・摩擦（`joint_dynamics`） | 実装済み・実動作確認済み | MJCF の `damping`・`armature`・`frictionloss` に出る |
| IMU センサー、接地の観測、キーフレーム | 実装済み・実動作確認済み | `framequat`・`frameangvel`・`touch` センサーと `solved` キーフレーム |
| 衝突形状 | 実装済み・実動作確認済み | box / capsule / sphere の単純形状のみ。見た目用のメッシュは含まれないので、`q8/visual.py` で STL を見た目専用（`contype=0`）に追加した `visual.xml` を作る |

### シミュレーション・学習・評価

| 機能 | 状態 | 確かめたこと |
|---|---|---|
| MuJoCo での物理シミュレーション | 実装済み・実動作確認済み | 接触・重力・アクチュエータの動作を確認。固定トロットで約50 mm/s 前進、左右の歩幅差による旋回で約33°/s |
| 歩容リファレンス（`assembly.gait`） | 部分実装 | 1脚3関節以上が前提で、2自由度の脚には使えない。代わりに位相クロック（`assembly.goal(kind="phase")`）と接地位相の報酬で代替 |
| 学習（`training/cadex_train.py`、Brax PPO on MJX/JAX） | 実装済み・実動作確認済み | CPU で学習が回ることを確認（256環境で約360ステップ/秒、1反復約14秒）。GPU でも回した（次の行） |
| GPU での学習 | 実装済み・実動作確認済み | vast.ai の RTX 4090 で、base・leg120・mini を学習シード5回ずつ学習した（8192 環境、1000 反復。[experiment-results.md](experiment-results.md) §5・§6）。base と leg120 は合格率 0.92 と 1.00、mini は 0.24（再調整中） |
| 学習時のセンサーノイズ・通信遅延のランダム化 | 未実装 | Cadex のトレーナーにこの機能はない（[simulation.md](simulation.md) の Sim2Real） |
| 方策の評価（`cadex evaluate`） | 実装済み・実動作確認済み | 成功条件・固定シード・動画つきで評価できた |
| 固定歩容の評価 | 部分実装 | `cadex evaluate` は方策ファイルしか再生できない。固定歩容は `q8/evaluation.py` で MuJoCo 上で再生し、同じ `CadexEvaluation` の指標で評価する仕組みを自作 |
| 方策ファイル（`.cxpolicy`） | 実装済み・実動作確認済み | Cadex 独自形式（重みと観測の正規化を含む）。ESP32 向けの C コードへの書き出しは base の方策で確認済み、実機は未確認（[esp32-deployment.md](esp32-deployment.md)） |

### ダッシュボード

| 機能 | 状態 | 補足 |
|---|---|---|
| `pixi run app`（読み取り専用の閲覧画面） | 実装済み・未確認 | 起動とブラウザでの表示は、まだ確かめていない |

## テスト結果

| スイート | 結果 | 補足 |
|---|---|---|
| `pixi run test-engine` | 162 passed / 2 failed / 3 skipped | 失敗2件は今回の変更と無関係の既存の問題（下記） |
| `pixi run python -m pytest cli/tests` | 51 passed / 5 skipped | |

失敗2件の内容:

1. **ライセンス検査**: `training/` の4つのテストファイルに SPDX ヘッダがない。
2. **`test_static_fit_processes...`**: テストが `FreeCADCmd` に `-c` で長い複数行のプログラムを渡すと、
   macOS ではプロセスが即座に終了する。同じ処理を `exec` 経由で動かすと並列の fit は正常に動き、
   直列の結果と一致した。テストの起動方法と macOS の組み合わせの問題で、fit の機能自体の不具合ではない。

## Cadex の README の記述と実際の違い

| README・文書の記述 | 実際に確かめたこと |
|---|---|
| macOS arm64 は CI を通るが手では試していない | 手元の macOS 26.2 / M4 Max でビルドとエンジンのテストが通った（上の既存の2件を除く） |
| テスト件数（`AGENTS.md`: エンジン 158 passed / 3 skipped、CLI 53 passed、2026-10-28 計測） | この環境ではエンジン 162 passed / 2 failed / 3 skipped、CLI 51 passed / 5 skipped（2026-11-17 には測り直していない） |
| アセンブリは歩容リファレンス（gait references）を持つ | 1脚3関節以上が前提で、2自由度脚の quadruped8 では使えない |
| export で STL を書く | ASCII STL。MuJoCo で使うにはバイナリへの変換が要る |
| MJCF を正確な慣性つきで書く | そのとおり。ただし衝突形状は単純形状だけで、見た目のメッシュは入らない |
| トレーナーは CPU で小さな機構、GPU で歩容 | CPU でも歩行タスクの学習は回るが遅い（1反復約14秒）。GPU（RTX 4090）では1回の学習（8192 環境、1000 反復）が約 480〜550 秒（[experiment-results.md](experiment-results.md) §5・§6） |
| URDF | README に記述はなく、機能もない |
