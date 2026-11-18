# 3本足ロボットとしっぽ付き四足（シミュレーション）実装計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** mini-r2 を元に、しっぽ付き四足（しっぽ 1 関節・2 関節）と 3本足ロボットを設計し、実機との差に強くする学習（dr）で GPU 学習して、mini-r2-p7-eff-dr と比べる。シミュレーションだけで、印刷と実機は範囲外。

**Architecture:** 設計スクリプト `robot/design/quadruped8.py` に `tail_dof`（0/1/2）と `rear_legs`（2/1）の params を足し、既定値では今の出力とまったく同じにする。脚の数・関節の数を決め打ちしている q8 の各所を、モデルから脚と関節を読む形に直す。学習・評価・vast.ai・dr バンドルは今の仕組みをそのまま使う。

**Tech Stack:** Cadex xscript（assembly ドメイン）、MuJoCo、Cadex の PPO 学習器（training/cadex_train.py）、vast.ai（RTX 4090）、uv、pytest。

**Spec:** この計画の「Goal」と下の「設計の決定」。元の提案は会話の中（案1: 3本足、案2: しっぽ付き四足。シミュレーションで並べて試す）。

## 設計の決定

| 項目 | 決めたこと |
|---|---|
| しっぽの付け根 | 胴体の後端の中央。`x = -(BL/2)` の後ろにサーボ 1個を胴体へ固定 |
| しっぽ 1 関節（`tail_dof=1`） | 左右に振る（軸はワールド +Z）。可動域 ±45°（clearance スイープで決め直す） |
| しっぽ 2 関節（`tail_dof=2`） | 付け根で左右（+Z）、その先で上下（+Y）。上下の可動域 −30〜+45° |
| しっぽの寸法 | 新しい params `tail_length`（既定 60 mm、範囲 30〜100）。先端に `tail_tip_mass_g`（既定 5 g、0〜20）のおもり |
| 3本足の配置（`rear_legs=1`） | 前の2本は今と同じ。後ろは胴体中央の1本（名前 `hc`）。股軸は `x = -(BL/2 + hub_radius + joint_gap)`、上腿の板の中央面を `y = 0` に置く。サーボは胴体の後端から出した台に固定 |
| 3本足の歩き方の報酬 | params `gait_mode`: 0 = 決めない（接地のリズムの報酬なし、足上げは全脚に払う）、1 = 波（lf → rf → hc の順に 1/3 周期ずつ立脚をずらす）。両方を学習して比べる |
| 組み合わせ | `tail_dof > 0` と `rear_legs = 1` は同時に使わない（スクリプトが `ValueError` で止まる） |
| 既定値 | `tail_dof=0`、`rear_legs=2`、`gait_mode=0` では、今の mini-r2 系の出力（MJCF・タスク・sha256）が1バイトも変わらない |
| 機体の名前 | `mini-r2-tail1`、`mini-r2-tail2`、`tri-r2`（mini-r2 の値に上の params を足す）。報酬は mini-r2-p7-eff と同じ（周期 0.7 s、`w_effort_rated` 1.0、指令 30〜120 mm/s） |

## Global Constraints

- Python の環境は uv で管理し、Cadex 本体（robot-project の外）は変えない。
- 実装は TDD で行う。テストを先に書き、失敗を確かめてから実装する。
- 文書と報告は日本語。確かめていない成果を、成功と書かない。
- GPU：この計画で $8 まで、累計 $20 で止める（予算は $100、今の累計は約 $10.5）。
- vast.ai は `.claude/skills/q8-vastai-gpu-training/SKILL.md` に従う。ラベルは `cadex-q8-<kind>-<YYYYMMDD-HHMM>`。`cadex-q8-` で始まらない台には触らない。終わったら `instances` で台がゼロになったことを確かめる。
- `.env` の中身を表示しない。
- コミットの作者は ayutaz。push は `quadruped-8dof` にだけ行う。生成物の扱いは `.claude/skills/q8-artifacts-and-publishing/SKILL.md` に従う。
- 既存の書き出し（`deploy/esp32/exports/full-mini-r2-p7-eff-dr-s4/` など）の task.sha256 が合わなくならないこと。
- ESP32・印刷・実機はこの計画の範囲外。

## Review Focus

1. 既定の params で、mini-r2-p7-eff のタスクバンドルの sha256 が変わらないこと → Task 2 のテスト `test_既定値では出力が変わらない`。
2. `tail_dof=1` と `rear_legs=1` を一緒に指定 → 黙って変な形を作らず `ValueError` → Task 3 のテスト。
3. 3本足で立った姿勢の重心が、3つの足先の三角形の外にある → `design_report.json` の判定 `com_inside_support_polygon` が false になり、Task 4 で止まって報告する → Task 1 の `static_report` の3脚のテスト。
4. アクチュエータが 8 個でない（しっぽで 9・10 個、3本足で 6 個）→ dr バンドル・評価・押しの評価が全部のアクチュエータに効く → Task 1 のテスト。
5. 8 関節を前提にした ESP32 の書き出しに 6・10 関節の方策を渡す → 黙って 8 個だけ書かずに止まる → Task 1 のテスト `test_関節の数が違う方策は書き出しで止まる`。

---

### Task 1: 脚と関節の数をモデルから読む

**Files:**
- Modify: `q8/robot.py`（`LEGS`・`JOINTS` の決め打ちの代わりに、モデルから読む関数を足す。今の定数は残す）
- Modify: `q8/physics.py:54-100`（`static_report` を N 脚にする）
- Modify: `q8/evaluation.py:23-56`（`FEET` と `JOINTS` をモデルから）
- Modify: `q8/dr_bundle.py:80-82`（`<general>` の数を `model.nu` と合わせる）
- Modify: `scripts/build_robot.py:38-45`（`intended_contacts` をモデルの脚としっぽから）
- Modify: `deploy/esp32/cxpolicy_io.py:52-54`（関節の数が合わなければ止まる）
- Test: `tests/test_model_layout.py`（新規）

**Interfaces:**
- Produces: `q8.robot.model_legs(model: mujoco.MjModel) -> list[str]` は、`hip_<leg>` の関節名から脚の名前を出す。今の機体は `["lf","rf","lh","rh"]`、3本足は `["lf","rf","hc"]`。
- Produces: `q8.robot.model_joints(model) -> list[str]` は、アクチュエータの順の関節名を出す（しっぽは `tail_yaw`、`tail_pitch`）。
- Produces: `q8.robot.model_feet(model) -> list[str]` は `["c_foot_<leg>", ...]` を出す。

- [ ] **Step 1: 失敗するテストを書く。** `tests/test_model_layout.py` に、小さい MJCF の文字列で作った 3脚・6 アクチュエータのモデルと、4脚＋しっぽ 2・10 アクチュエータのモデルを用意し、次を書く。
  - `test_model_legs_は関節名から脚を出す`: 3本足で `["lf","rf","hc"]`、しっぽ付きで `["lf","rf","lh","rh"]`（しっぽは脚に入らない）。
  - `test_static_report_は3脚で支持多角形を三角形にする`: 重心が三角形の外のモデルで `support_margin_mm < 0`。
  - `test_dr_bundle_は8個でないアクチュエータにも遅れを入れる`: 10 個のモデルで `dr_model_xml` が 10 個すべてに `dyntype="filterexact"` を入れる。
  - `test_関節の数が違う方策は書き出しで止まる`: 10 行動の方策を `cxpolicy_io` に渡すと、`ValueError` で「8」と「10」がメッセージに入る。
  - `test_今の機体では今と同じ`: `robot/mjcf/mini-r2-p7-eff/model-model.xml` で、`model_legs`・`model_joints` が今の `LEGS`・`JOINTS` と同じ。`static_report` の結果が `artifacts/mini-r2-p7-eff/design_report.json` の `physics` と一致する（小数の丸め内）。
- [ ] **Step 2: 実行して失敗を確かめる。** `uv run pytest tests/test_model_layout.py -v` で、`model_legs` が無いために FAIL。
- [ ] **Step 3: 実装する。** 上の Interfaces の3関数を作り、physics・evaluation・dr_bundle・build_robot をそれらで書き直す。`static_report` の「2脚支持（トロット）」の行は、3脚では「2脚支持（1脚を上げた最悪）」にする。
- [ ] **Step 4: 全体のテストを回す。** `uv run pytest -q` で、全部 PASS（今は 553 passed, 1 skipped）。
- [ ] **Step 5: コミットする。**

### Task 2: しっぽ（`tail_dof`）

**Files:**
- Modify: `robot/design/quadruped8.py`（params に `tail_dof`・`tail_length`・`tail_tip_mass_g` を足す。部品・関節・アクチュエータ・観測・ランダム化）
- Test: `tests/test_design_tail.py`（新規。`tests/test_design_battery.py` の `run_design` を使う）

**Interfaces:**
- Produces: アクチュエータのラベル `tail_yaw_drive`（`tail_dof>=1`）と `tail_pitch_drive`（`tail_dof==2`）。task の `actions` の最後に、この順で並ぶ。観測 `c_tail_yaw`・`c_tail_pitch`（指令）と `tau_tail_*`（privileged）。
- Produces: 関節のラベル `tail_yaw` と `tail_pitch`。部品のラベル `tail_servo_yaw`、`tail_link_1`、`tail_tip` など。

- [ ] **Step 1: 失敗するテストを書く。**
  - `test_既定値では出力が変わらない`: `run_design()` と、params を足す前のログ（このテストで保存した基準）が同じ。
  - `test_tail_dof_1_は左右の関節を1つ足す`: アクチュエータが 9 個で、最後が `tail_yaw_drive`。関節の軸は +Z。
  - `test_tail_dof_2_は左右と上下の2つ`: 10 個で、最後の2つが `tail_yaw_drive`、`tail_pitch_drive`。
  - `test_しっぽの重さと長さは_params_で変わる`: `tail_length=90` で先端の位置が x 方向に 30 mm 後ろへ動く。
  - `test_報酬の_8_で割る平均はアクチュエータの数で割る`: `tail_dof=2` で `action_rate` などの式の割る数が 10。
- [ ] **Step 2: `uv run pytest tests/test_design_tail.py -v` で FAIL を確かめる。**
- [ ] **Step 3: 実装する。** しっぽのサーボは胴体に溶接し（`weld`）、関節は `assembly.joint("revolute", ...)`。アクチュエータは脚と同じ式（失速トルクでクランプ、剛性 = 5° で失速）。報酬の `/ 8` は `/ len(KEYS)` にする。
- [ ] **Step 4: `uv run pytest -q` で全部 PASS を確かめる。**
- [ ] **Step 5: コミットする。**

### Task 3: 3本足（`rear_legs`、`gait_mode`）

**Files:**
- Modify: `robot/design/quadruped8.py`
- Test: `tests/test_design_tripod.py`（新規）

**Interfaces:**
- Consumes: Task 2 の `len(KEYS)` で割る報酬。
- Produces: 脚の名前 `hc`。アクチュエータ `hip_drive_hc`・`knee_drive_hc`。actions は `lf, rf, hc` の順に hip・knee。`FEET` は3つ。

- [ ] **Step 1: 失敗するテストを書く。**
  - `test_rear_legs_1_は後ろ中央の1本`: アクチュエータが6個。`hc` の股軸の y が 0 ± 0.5 mm、x が `-(BL/2 + hub_radius + joint_gap)`。
  - `test_しっぽと3本足は一緒に使えない`: `run_design(tail_dof=1, rear_legs=1)` が `ValueError`。
  - `test_gait_mode_0_は接地のリズムの報酬が無い`: 報酬のラベルに `trot_contact` が無く、`swing_lift` は3脚すべての足に払う。
  - `test_gait_mode_1_は1_3周期ずつずらす`: `gait_contact` の式に、`clock_sin`・`clock_cos` で作る 3 相（0°、120°、240°）が入る。
  - `test_3本足の成功条件の足は3つ`: `success` の `feet` が3つ。
- [ ] **Step 2: `uv run pytest tests/test_design_tripod.py -v` で FAIL を確かめる。**
- [ ] **Step 3: 実装する。** 後ろの脚は前脚と同じ2関節の脚の作り方を使う（`sx=-1`、`sy=0`）。サーボの台は胴体の後端に足す箱。3相の位相は `cos(θ - k·120°)` を `clock_sin`・`clock_cos` から作る。
- [ ] **Step 4: `uv run pytest -q` で全部 PASS を確かめる。**
- [ ] **Step 5: コミットする。**

### Task 4: 3つの機体を作り、形を確かめる

**Files:**
- Create: `artifacts/mini-r2-tail1/`、`artifacts/mini-r2-tail2/`、`artifacts/tri-r2/`（design_report.json、renders/）、`robot/mjcf/<variant>/`、`robot/meshes/<variant>/`
- Modify: `robot/design/quadruped8.py`（clearance スイープの結果で可動域の既定値を直す場合だけ）

- [ ] **Step 1: 作る。** mini-r2 の値（`artifacts/mini-r2/design_report.json` の params）に、`tail_dof=1`／`tail_dof=2`／`rear_legs=1` と、mini-r2-p7-eff の報酬の値を足して `scripts/build_robot.py --variant <name> --set k=v ...` を3回実行する（zsh では引数を1つずつ書く）。
- [ ] **Step 2: 判定を確かめる。** 各 `design_report.json` の `verdict` が全部 true か。false があれば、clearance の最初の接触から 2 ステップ手前に可動域を直して作り直す。3本足で `com_inside_support_polygon` が false なら、作り直さずに止まって報告する（股の前後位置か後脚の位置を変える案を出す）。
- [ ] **Step 3: 組めるかを確かめる。** `uv run python scripts/assembly_guide.py --variant <name>` の判定が合格か。しっぽ・後脚のサーボの差し込みも含む。
- [ ] **Step 4: 画像を利用者に見せる。** 3機体の `renders/` の画像を報告に載せる。
- [ ] **Step 5: コミットする。**

### Task 5: 学習設定と GPU 学習

**Files:**
- Create: `configs/train_full_mini-r2-tail1-dr.json`、`configs/train_full_mini-r2-tail2-dr.json`、`configs/train_full_tri-r2-free-dr.json`、`configs/train_full_tri-r2-wave-dr.json`（`gait_mode` 0 と 1）
- Create: `artifacts/<variant>-dr/cut/`（`q8/dr_bundle.py` で作る）
- Modify: `tests/test_vastai_config.py`（4つの設定が `train_full_mini-r2-p7-eff-dr.json` とバンドルの場所・名前だけ違うことを固定）

- [ ] **Step 1: 失敗するテストを書く。** `test_しっぽと3本足の設定は_dr_の設定と機体だけ違う`。
- [ ] **Step 2: FAIL を確かめ、設定とバンドルを作り、PASS を確かめる。**
- [ ] **Step 3: 学習する。** 4設定 × 学習シード 0〜2（12回）を `vastai create --count 12 --yes` → `sweep --destroy-when-done` で回す。見込みは約 $3〜4。
- [ ] **Step 4: 追加で学習する。** しっぽの良い方と 3本足の良い方に、シード 3〜4 を足して5回にする（4回、約 $1.5）。累計が $20 を超えそうなら、ここを飛ばす。
- [ ] **Step 5: 台がゼロであることを確かめ、学習結果を `q8.training.archive_run` で残してコミットする。**

### Task 6: 評価・比較・動画・文書

**Files:**
- Create: `artifacts/compare/seeds_*`、`artifacts/<variant>/robustness/`、`artifacts/videos/<variant>-*.mp4`
- Modify: `docs/experiment-results.md`（§12 を新設）、`README.md`（機体の表）、`docs/training.md`（今の状態）

- [ ] **Step 1: 評価する。** 各回を `scripts/compare_seeds.py --as` で普通に評価し、`scripts/compare_robustness.py` で実機との差と押しを評価する。
- [ ] **Step 2: 比べる。** 比べる相手は mini-r2-p7-eff-dr（5回）。項目は、合格率、トルク比、滑り、速度比、最大傾斜、実機を想定した幅の合格率、押し 2.0 N で倒れず戻った割合、機体の重さ。平均 ± 標準偏差（最小〜最大）。
- [ ] **Step 3: 動画を作る。** 各機体の最も良い回の評価トレースを `evaluation/replay_trace.py` で再生し、mini-r2 と並べた比較動画を作る。
- [ ] **Step 4: 文書に書く。** experiment-results §12 に結果・費用・確かめていないことを書き、README と training.md を更新する。
- [ ] **Step 5: `uv run pytest -q` で全部 PASS を確かめ、コミットして push する。**
