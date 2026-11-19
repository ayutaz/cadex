# 横に開く関節を足した3本足（tri3）計画

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 担当への分け方は `.claude/skills/parallel-orchestration/SKILL.md` に従う。

**Goal:** 脚の付け根に横へ開く関節（ロール）を足した3本足を2種類つくり、シミュレーションで歩き方を学習して、合格するかを確かめる。

**Architecture:**
- 2種類の機体は、どちらも `robot/design/quadruped8.py` の新しい param `leg_roll` で作る。既定値では、今の出力を1バイトも変えない。
- 学習の前に、新しいモジュール `q8/tripod_reach.py` で確かめる。確かめるのは、どの脚を上げるときも、残る2本の足を結ぶ線の上に重心を置けるか、の1点。
- 学習は、2種類 × シード3つを、GPU で同時に回す。

**Tech Stack:** Cadex xscript、MuJoCo、`training/cadex_train.py`（PPO）、vast.ai RTX 4090、pytest。

**Spec:** 別の仕様書は無い。下の「設計の判断」がこの計画の仕様。根拠は、次の2つの報告。
- `.superpowers/sdd/2026-11-18-tripod-and-tail/diag-tripod-report.md`
- `.superpowers/sdd/2026-11-18-tripod-and-tail/tri-dyn-report.md`

## 設計の判断（なぜ横の関節か）

- **今の3本足が歩けない理由**：1本を上げると、残りは2本の足になり、支えは2点を結ぶ線だけになる。
  - 四足のトロットも、支えは対角の2本の線だけになる。それでも歩けるのは、その線が重心の近くを通るからである。
  - 今の3本足は前後に振る関節だけなので、足を横へ動かせない。そのため、右前を上げると、残りの線が重心から 4〜13 mm 離れてしまう（tri-r2-dyn は −4.4 mm）。
- **直し方**：付け根にロール（前後の軸まわりに脚を横へ開く関節）を足す。
  - 足を左右へ運べるようになり、上げる脚ごとに、残る2本の線を重心の真下へ持ってこられる。
  - つまり、3本足の一歩一歩を、トロットと同じ条件にする。
- **2種類を同時に試す**：

  | 機体 | ロールを付ける脚 | 関節（サーボ）の数 | 基板 |
  |---|---|---|---|
  | **tri3-r8** | 前の2本だけ | 8 | 今の ESP32-S3 の8チャンネルにそのまま収まる（関節の名前の対応は直す） |
  | **tri3-r9** | 3本とも | 9 | ESP32-S3 の LEDC は8チャンネルなので、PWM 基板（PCA9685）か MCPWM が要る |

- **ほかは tri-r2-dyn と同じ**：後ろ足は真ん中（`hc_foot_center=1`）、速度は重心で測る（`vel_source=1`）、波の歩き方（`gait_mode=1`）、周期 0.7 s。
  - 遊脚のしきい値は既定の 0.5 に戻す。tri-r2-dyn の 0.8（素早い一歩）は、すり足に落ちたため。
- **足の裏（パッド）は足さない**（YAGNI）。足首が無いと、平らな裏は脚の傾きで角だけが当たり、支えは広がらない。ロールで足りなかったときの次の手として、文書に残す。
- **実機の書き出し（ESP32）は、この計画では扱わない。**
  - `deploy/esp32` は8関節で決め打ちしている（`cx.require_joint_count`）。今の決まりどおり、tri3 の方策は書き出しで断られる。
  - 合格した機体が決まってから、別の計画で次の2つを直す。
    - 関節の数を決め打ちしない書き出し。
    - 9個のときの PWM 基板。

## Global Constraints

- Python の新しい環境は uv。Cadex 本体は公式の pixi。既存の依存とビルドを壊さない。Cadex 本体（`src/`）は変えない。
- 実装は全部 TDD（先にテストを書いて、失敗を確かめる）。
- 新しい params の既定値では、出力を1バイトも変えない。
  - `tests/data/design_log_{default,mini-r2-p7-eff,tri-r2,tri-r2-wave}.json` で守る。
  - Task 1 で `design_log_tri-r2-dyn.json` を、変更の前に足す。
- 新しい param の名前と値：
  - `leg_roll`：0 は無し（既定）、1 は前の2本、2 は3本とも。3本足（`rear_legs=1`）でだけ使える。
  - `roll_min`、`roll_max`：既定は −20°、+20°。正の向きは、脚を外へ開く向き。
- 関節の名前は `roll_<脚>`、`hip_<脚>`、`knee_<脚>`。脚の中の順は roll → hip → knee、脚の順は lf → rf → hc。
- 機体の名前（`--variant`）は `tri3-r8` と `tri3-r9`。DR バンドルは `artifacts/<機体>-dr/cut`、学習設定は `configs/train_full_<機体>-dr.json`。
- GPU：
  - このプロジェクトの台は、ラベルが `cadex-q8-` で始まるものだけ。ほかの台（61268502 など）には触らない。
  - この計画の予算は $5。累計（今は約 $13.4、概算）が $18 を超えそうなら止めて報告する。
  - sweep には `--destroy-when-done` を付ける。
- API キーや .env は表示しない、コミットしない。学習結果は `q8-artifacts-and-publishing` の決まりで残す。
- コミットの作者は ayutaz。push は quadruped-8dof へ、リーダーだけが行う。
- ディスク全体の検索（`find /`、`find ~`）をしない。時間のかかるコマンドには timeout を付ける。区切りごとに進捗を送る。
- 検証していない成果を、成功と報告しない。合格しなくても、そのとおりに書く。

## Review Focus

1. **ロールの関節の軸と符号を取り違える。** 左右の脚で「外へ開く」が逆になると、片側の脚が胴体にめり込む。
   - Task 1 に `test_roll_positive_opens_outward` を置く。両側の脚で、+10° にしたら足先の |y| が増えることを確かめる。
2. **ロールのサーボと台が、胴体や隣の脚とぶつかる、または台が胴体から浮く。** tri-r2-dyn では、台の重なりが 2.25 mm しか無かった。
   - Task 1 に `test_roll_mount_overlap` を置く。実ビルドで、台と胴体の重なりが 4 mm 以上あることを確かめる。
   - 印刷の都合（A1 mini の 180 mm に収まるか）は、Task 5 の報告に書く。
3. **重心の線に届かない。** ロールの範囲 ±20° では、足が重心の下まで届かないかもしれない。
   - Task 5 で `best_lift_margin` を3本の脚それぞれに対して出す。
   - 1本でも −2 mm より悪ければ、GPU に進まない。範囲（最大 ±30°）を広げて測り直し、それでも届かなければ報告して止める。
4. **3本足の評価が、四足向けの決まりで落ちる。** 例えば、比較表の列が関節8個を前提にしている。
   - Task 7 で、tri3 の評価が `compare_seeds.py` を通ることを、tri-r2-dyn と同じ手順で確かめる。
5. **300 反復の判定を見逃す。** 前回は、見張りの条件の書き誤りで見逃した。
   - Task 6 では、`progress.json` の反復が 300 以上になるまで待つ Monitor を置く。
   - 300 になったら、報酬の成分 `gait_contact` と、足を上げる項を、tri-r2-dyn の同じ時点と比べる。

---

## 担当への分け方（待ちの表）

| 塊 | 待つ結果 | 触るファイル | 担当 |
|---|---|---|---|
| Task 1 機体の param | なし | `robot/design/quadruped8.py`、`tests/test_design_tripod_roll.py`、`tests/data/design_log_tri-r2-dyn.json` | roll-design |
| Task 2 重心の線の計算 | なし（tri-r2-dyn のバンドルで試す） | `q8/tripod_reach.py`、`tests/test_tripod_reach.py` | reach |
| Task 3 学習設定 | なし | `configs/train_full_tri3-r{8,9}-dr.json`、`tests/test_vastai_config.py` | cfg |
| Task 4 動画の素材と文書の枠 | なし | scratchpad、`docs/experiment-results.md`（§14 の枠だけ） | prep |
| Task 5 2種類の実ビルドと確認、バンドル | Task 1、2、3 | `artifacts/tri3-*`、`robot/meshes/tri3-*`、`tests/test_vastai_config.py`（バンドルの一覧） | build（tri3-r8 と tri3-r9 で1人ずつ。共有のテストの一覧は r8 の担当が持つ） |
| Task 6 GPU 学習 | Task 5 | `artifacts/runs/vastai/full-tri3-*` | gpu |
| Task 7 評価 | 回収された回ごと | その回の評価の出力 | eval-<回>（回ごとに新しい担当） |
| Task 8 比較・動画・文書の数値 | Task 7 の全部 | `docs/experiment-results.md` §14、動画 | doc-num、video |

Task 1〜4 は、最初の応答で同時に起動し、同じ応答で15分ごとの見張り（CronCreate）を置く。

---

### Task 1: 横の関節 `leg_roll` を設計スクリプトに足す

**Files:**
- Modify: `robot/design/quadruped8.py`
  - params（12〜50 行付近）
  - 3本足の検査（140〜165 行）
  - `LEG_SPECS` と股サーボの組み方（175〜200 行）
  - 関節とコンポーネント（620〜660 行）
  - `hip_limits` と `servo_drive`（531、678 行）
- Create: `tests/test_design_tripod_roll.py`
- Create: `tests/data/design_log_tri-r2-dyn.json`（変更の前に、HEAD の出力で作る）

**Interfaces:**
- Produces:
  - `--variant tri3-r8`（`--set rear_legs=1 --set leg_roll=1 --set gait_mode=1 --set hc_foot_center=1 --set vel_source=1`）
  - `--variant tri3-r9`（同じで、`leg_roll=2`）
  - 関節名は Global Constraints のとおり。ロールの関節の軸はワールドの x（前後）。
- 機械の組み方（決めておくこと）：
  - ロールのサーボは胴体に付け、軸を前後に向ける（`lib.servo(SKU, direction=(1,0,0) か (-1,0,0))`）。
  - その出力の台（新しいコンポーネント `c_rollbr_<脚>`）が、股のサーボを持つ。
  - 連なりは、胴体 →roll→ 台 →hip→ 上腿 →knee→ 下腿。
  - 台の当たり判定は `box_collision`。

- [ ] **Step 1:** HEAD のスクリプトで `tri-r2-dyn` の design log を作り、`tests/data/design_log_tri-r2-dyn.json` に置く。
  - 既定値の比較テストの対象（`test_design_tripod_dyn.py` の基準の一覧）に足す。
  - テストが通ることを確かめる。
- [ ] **Step 2:** 失敗するテストを書く（`tests/test_design_tripod_roll.py`）。
  - `test_leg_roll_rejects_bad_values`：`leg_roll=3` と `leg_roll=1, rear_legs=2` が、`ValueError` で `"leg_roll"` を含む。
  - `test_default_unchanged`：5つの基準ログ（default、mini-r2-p7-eff、tri-r2、tri-r2-wave、tri-r2-dyn）と1バイト一致する。
  - `test_joint_order_r8`：`model_joints` が `["roll_lf","hip_lf","knee_lf","roll_rf","hip_rf","knee_rf","hip_hc","knee_hc"]`。
  - `test_joint_order_r9`：上に、`"roll_hc"` を `"hip_hc"` の前に入れたもの（9個）。
  - `test_roll_axis_is_forward`：ロールの関節の軸と (1,0,0) の内積の絶対値が 0.99 以上。
  - `test_roll_positive_opens_outward`：各脚のロールを +10° にすると、足先の |y| が 3 mm 以上増える（MuJoCo の順運動学）。
  - `test_roll_limits`：関節の範囲が `roll_min`〜`roll_max`（既定 ±20°）。
  - 実ビルドのテスト（エンジンが要る。無ければ skip）：
    - `test_roll_build_verdicts`：tri3-r8 と tri3-r9 で、`verdict` と `assembly_verdict` が全部 true。
    - `test_roll_mount_overlap`：ロールの台と胴体の重なりが 4 mm 以上。
- [ ] **Step 3:** テストを回し、新しいテストが落ちることを確かめる。
  - Run: `uv run pytest tests/test_design_tripod_roll.py -q`
  - Expected: FAIL（`leg_roll` が無い）
- [ ] **Step 4:** `quadruped8.py` に、`leg_roll`、`roll_min`、`roll_max` と、上の組み方を入れる。
  - 胴体の幅 `W2` は変えない。ロールの台は胴体の側面の外に足す。
  - 質量と部品表（BOM）にサーボの数が反映されることを確かめる。
- [ ] **Step 5:** テストを回し、全部通ることを確かめる。
  - Run: `uv run pytest tests -q -x --timeout 900`
  - Expected: 全部 PASS
- [ ] **Step 6:** パスを指定して `git add` し、コミットする。

### Task 2: 重心の線に届くかを計算する `q8/tripod_reach.py`

**Files:**
- Create: `q8/tripod_reach.py`
- Create: `tests/test_tripod_reach.py`

**Interfaces:**
- Produces:
  - `lift_margin(model, data, lift: str) -> float`
    - 今の姿勢で、`lift` 以外の2本の足先を結ぶ線から重心（`subtree_com[0]` を床へ写したもの）までの距離 [mm]。
    - 重心が、2本の線と上げる足の反対側にあれば正。
  - `best_lift_margin(bundle_dir: Path, lift: str, clear_mm: float = 10.0, roll_range_deg: float | None = None) -> dict`
    - 返り値：`{"margin_mm": float, "joint_deg": dict[str, float], "feasible": bool}`
    - 条件：残りの2本は床に着いたまま（足先の z が足先の半径 ± 1 mm）、上げる足は `clear_mm` 以上上がる。その条件のもとで、関節の範囲の中で `margin` を最大にする。
    - `roll_range_deg` を渡すと、ロールの範囲を ±その値に置き換える。
    - 解き方：関節ごとに 5° 刻みの格子で粗く探し、上位10点から `scipy.optimize.minimize`（Nelder-Mead）で詰める。scipy が venv に無ければ、格子を 1° 刻みで詰める。
- Consumes：DR バンドルの MJCF（`<bundle>/…/model.xml`。場所は `q8/dr_bundle.py` に合わせる）、`q8.robot.model_feet`。

- [ ] **Step 1:** 失敗するテストを書く。
  - `test_stand_margins_match_diag`：tri-r2-wave-dr のバンドル（payload は基準の位置）を立位にしたとき、`lift_margin` が調査の値と ±0.5 mm で一致する。
    - 調査の値は、`diag-tripod-report.md` の表から写す（lf +1.9、rf −13.3、hc +13.1 mm）。
    - 値の定義が合わないときは、報告の定義に合わせて関数を直す。テストの値は変えない。
  - `test_best_not_worse_than_stand`：どの脚でも、`best_lift_margin(...)["margin_mm"]` が立位の `lift_margin` 以上。
  - `test_pitch_only_cannot_lift_rf`：tri-r2-dyn-dr で、`best_lift_margin(…, "rf")["margin_mm"] < 0`。
- [ ] **Step 2:** テストを回し、落ちることを確かめる。
  - Run: `uv run pytest tests/test_tripod_reach.py -q`
  - Expected: FAIL（モジュールが無い）
- [ ] **Step 3:** 上の2つの関数を実装する。
- [ ] **Step 4:** テストを回し、通ることを確かめる。
- [ ] **Step 5:** tri-r2-dyn-dr での3本の脚の `best_lift_margin` を、報告に書く（tri3 との比べる相手になる）。
- [ ] **Step 6:** コミットする。

### Task 3: 学習設定 `train_full_tri3-r{8,9}-dr.json`

**Files:**
- Create: `configs/train_full_tri3-r8-dr.json`、`configs/train_full_tri3-r9-dr.json`
- Modify: `tests/test_vastai_config.py`（設定の一覧だけ。バンドルの一覧 `TAIL_TRI_DR_BUILT` には、Task 5 が足す）

**Interfaces:**
- 中身は `train_full_tri-r2-dyn-dr.json` を写す。違いは、機体・名前・ラベル・バンドルのパスの4つだけ。
- エントロピーなど、学習の数値は同じにする（1000 反復）。

- [ ] **Step 1:** 失敗するテストを書く。
  - `test_tri3_configs_match_dyn`：2つの設定が、上の4つ以外で `train_full_tri-r2-dyn-dr.json` と一致する。
- [ ] **Step 2:** テストを回し、落ちることを確かめる。
- [ ] **Step 3:** 2つの設定を作る。
- [ ] **Step 4:** テストを回し、通ることを確かめる。
- [ ] **Step 5:** コミットする。

### Task 4: 動画の素材と文書の枠

**Files:**
- scratchpad の `video-tri3/` に、カメラ・文字・カードを置く（`q8-sim-video` に従う）。
- Modify: `docs/experiment-results.md`
  - §13 の後ろに、§14「横の関節を足した3本足（tri3）」の見出しを足す。
  - 中身は、設計の判断の要約と、数値が空欄の比較表。
  - 比べる相手は、mini-r2-p7-eff-dr、tri-r2-dyn-dr、tri3-r8-dr、tri3-r9-dr。

- [ ] **Step 1:** 文書の枠を書き、文書のテスト（`tests` の中で docs を読むもの）を回す。
- [ ] **Step 2:** コミットする（docs だけ）。
- [ ] **Step 3:** 動画の素材を scratchpad に作る。
  - 3つを並べる構成：mini-r2、tri3-r8、tri3-r9。
  - 最後のカードは付けない。

### Task 5: tri3-r8 と tri3-r9 を作って確かめ、DR バンドルを作る（機体ごとに1人）

**Files:**
- `artifacts/tri3-r{8,9}/`
- `robot/meshes/tri3-r{8,9}/`
- `artifacts/tri3-r{8,9}-dr/cut`
- `tests/test_vastai_config.py` の `TAIL_TRI_DR_BUILT`（r8 の担当が、2つとも足す）

**Interfaces:**
- Consumes：
  - Task 1 の `--variant` と params。
  - Task 2 の `best_lift_margin`。
  - Task 3 の設定。
- DR は `q8/dr_bundle.py` で、mini-r2-p7-eff-dr と同じものを足す。

- [ ] **Step 1:** 実ビルドする。
  - Run: `uv run python scripts/build_robot.py --variant tri3-r8 --set rear_legs=1 --set leg_roll=1 --set gait_mode=1 --set hc_foot_center=1 --set vel_source=1`（r9 は `leg_roll=2`）
  - verdict、質量、BOM、衝突形状を報告に書く。
- [ ] **Step 2:** DR バンドルを作る（`scripts/make_dr_bundle.py`）。
- [ ] **Step 3:** 3本の脚それぞれで `best_lift_margin` を出す。
  - **判定**：3本とも −2 mm 以上なら、GPU に進む。
  - 届かない脚があれば、`roll_range_deg=30` で測り直す。それで届けば、`roll_min/max` を ±30° にして Step 1 からやり直す。それでも届かなければ、GPU に進まずに報告する。
- [ ] **Step 4:** ずるい取り方を確かめる（MuJoCo、GPU は要らない）。
  - 立たせたまま、胴体を揺する、すり足の3つの動きで、`track_speed` と `gait_contact` に入る点を、tri-r2-dyn の値と並べる。
- [ ] **Step 5:** `TAIL_TRI_DR_BUILT` に `tri3-r8-dr` と `tri3-r9-dr` を足す（r8 の担当だけ）。
  - バンドルの1バイト一致のテストを回す。
- [ ] **Step 6:** コミットする。r9 の担当は、自分のフォルダだけを `git add` する。

### Task 6: GPU で学習（2種類 × シード 0〜2）

`q8-vastai-gpu-training` に従う。

- [ ] **Step 1:** RTX 4090 を7台（6回＋予備1台）借り、ラベル `cadex-q8-full-tri3-*` を付ける。
  - 2つの設定 × シード 0〜2 を、`sweep --destroy-when-done` で同時に始める。
  - 費用の見込みは $1.5〜2.5（概算）。
- [ ] **Step 2:** `progress.json` の反復が 300 以上になるまで待つ Monitor を置く（Review Focus 5）。
  - 300 で、`gait_contact` と足を上げる項を、tri-r2-dyn の同じ時点と比べる。
  - **両方の機体で**伸びていなければ、sweep を止めて台を破棄し、報告する。
  - 片方だけ悪ければ、その機体の回だけを止める。
- [ ] **Step 3:** 回が終わるたびに `archive_run` し、リーダーに1行送る。
  - リーダーは、その回の評価の担当（Task 7）を起動する。
- [ ] **Step 4:** 全部の回が終わったら、`instances` で自分の台がゼロであることを確かめ、費用を書く。

### Task 7: 評価（回ごとに新しい担当）

- [ ] **Step 1:** その回を評価する。評価のシード（10個）、合格率、各脚の歩数・滞空・振り、滑り、速度比、トルク比を出す。
- [ ] **Step 2:** 結果のファイルを、`q8-artifacts-and-publishing` のとおりに置く。
- [ ] **Step 3:** リーダーに2行で送る。内容は、合格率と、rf の歩数が tri-r2-dyn より伸びたか。

### Task 8: 比較・文書の数値・動画の仕上げ

- [ ] **Step 1:** `scripts/compare_seeds.py` で、tri3-r8-dr と tri3-r9-dr を、mini-r2-p7-eff-dr（5回）と tri-r2-dyn-dr（3回）の両方と比べる。
- [ ] **Step 2:** §14 の表の数値を埋め、次を書く。
  - 結論
  - 費用（概算と明記）
  - 確かめていないこと（実機、印刷、ESP32 の書き出しは次の計画）
- [ ] **Step 3:** 動画を仕上げる（Task 4 の素材と評価トレース）。
- [ ] **Step 4:** テストを回して、コミットする。リーダーが全体のレビューのあと push する。

## 成功の判定

| 結果 | 判断 |
|---|---|
| どちらかの機体で、3回の平均の合格率が 0.6 以上 | 3本足で歩けた。その機体で、実機の計画（ESP32 の書き出し、部品）に進むかを利用者に聞く |
| 合格率は 0.6 未満だが、rf と hc の歩数が 4 以上 | 歩き始めている。報酬か学習の長さを直す次の手を、報告に書く |
| rf が上がらない | 横の関節でも足りない。足の裏（パッド）と足首など、次の手を報告に書く |

## 費用と時間の見込み（概算）

- **GPU**：$1.5〜2.5。累計は約 $15〜16。
- **部品**（実機にする場合）：
  - サーボ +2個（r8）か +3個（r9）。MG90S が1個約 450 円（推定）。
  - r9 は PWM 基板（PCA9685、約 600 円、推定）が要る。
- **時間**：
  - Task 1〜4 を同時に進めて、約1時間。Task 1 が一番長い。
  - Task 5 が約30分。
  - 学習と評価が約40分。
  - 合わせて約2〜2.5時間。
