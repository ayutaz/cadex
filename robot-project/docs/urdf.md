# URDF の書き出し

最終確認日: 2026-11-18

Cadex には URDF 出力がないので、Cadex が書いた MJCF から変換する（`q8/urdf.py`）。

いまリポジトリにある URDF は基準機体の `robot/urdf/base/quadruped8.urdf` だけ。作る機体 `mini-r2`（mini の脚を
組めるように直した形、`leg_rev=2`）の URDF は、リポジトリにはまだ置いていない。`--variant mini-r2` で、
`robot/mjcf/mini-r2/model-model.xml` から同じ手順で作れる（mini-r2 には電池のリンク `c_bat` が無い）。
2026-11-18 に作業用の場所へ書き出して確かめた: リンク 34、関節は revolute 8・fixed 25、総質量 220.147 g。
`tests/test_urdf.py` と同じ比べ方（MuJoCo で読み直し、立位とランダムな関節角3組）で、元の MJCF との差は最大 4.2e-7 だった。

## 使い方

```bash
cd robot-project
.venv/bin/python scripts/mjcf_to_urdf.py --variant base
# 入力  robot/mjcf/base/model-model.xml
# 見た目 robot/meshes/base/*.stl（scripts/build_robot.py が作る。無ければ見た目なしで書く）
# 出力  robot/urdf/base/quadruped8.urdf
```

主なオプション: `--velocity`（関節の速度上限 [rad/s]。既定は MG90S の 4.8 V・0.10 s/60° から
10.472）、`--out`、`--mjcf`、`--meshes`、`--no-mujoco-tag`。

テスト: `.venv/bin/python -m pytest tests/test_urdf.py -q`。生成した URDF を MuJoCo で読み直し、
立位とランダムな関節角で、関節のワールド位置・軸、各リンクの質量・重心・慣性、全体の重心が
元の MJCF と一致することを確かめる（許容 0.1 mm。実測の誤差は 1e-17 m 程度）。

## 変換の中身

- リンク = MJCF のボディ。ワールド直下のボディ（`c_chassis`、free 関節）がルートの `base_link`。
  `base_link` の原点は `c_chassis` の原点（立位で機体の真下の床の高さ）のまま。
- hinge → `revolute`（range があるとき。無ければ `continuous`）。関節の原点 = 関節の `pos`。
  子リンクの inertial / visual / collision の原点はその分だけずらす。
  - `limit`: lower/upper = MJCF の range [rad]、effort = アクチュエータの forcerange × gear
    （力の上限が無ければ 0 とコメント）、velocity = 引数。
  - `dynamics`: damping = MJCF の damping、friction = frictionloss。
- 関節のないボディ → `fixed` 関節（名前は `<子リンク>_fixed`）。リンクのフレームはボディのフレームのまま。
- 慣性: 主慣性（diaginertia）を iquat で回し、リンクの軸で表したテンソル（ixx..izz）。
- visual: STL を相対パス（`../../meshes/<variant>/<部品>.stl`）で参照し、`scale="0.001 0.001 0.001"`（mm→m）。
  ボディ名 `c_<key>` と `<key>.stl` が対応する（`c_esp`→`esp32.stl`、`c_bat`→`battery.stl`、`q8/visual.py`）。
- collision: 接触しうる box / sphere / cylinder をそのまま、capsule は下記の近似で。床は入れない。
- MJCF は MuJoCo でコンパイルしてから読むので、既定値・`fromto`・`euler` なども MuJoCo と同じ解釈になる。

## 前提と制約

- **capsule は URDF 標準に無い**。同じ半径・長さの `cylinder` と両端の `sphere` 2 個、計 3 要素で
  近似する（形状は同じ。ただし衝突要素が増える）。元が capsule だったことは URDF のコメントに残る。
- **URDF に入らないもの**: アクチュエータ（位置制御のゲイン・バイアス。力の上限だけ effort に残る）、
  センサ（IMU・関節角・足裏の接触）、接触パラメータ（solref・friction・contype/conaffinity・
  contact exclude）、armature、キーフレーム、積分器などの option。シミュレーションには元の MJCF を使う。
- 対応する関節は hinge と、ルートの free だけ。ball・slide、1 ボディに複数関節、ref ≠ 0 は
  `NotImplementedError`。ワールド直下のボディは 1 つである必要がある。
- URDF には MuJoCo 用の拡張要素 `<mujoco><compiler discardvisual="false" fusestatic="false"/></mujoco>`
  を入れる（MuJoCo は既定で URDF の固定リンクを親に統合し、見た目を捨てるため）。ROS などほかの
  パーサは無視する。不要なら `--no-mujoco-tag`。
- ROS の KDL はルートリンクに慣性があると警告する。気になるなら、慣性のない空のリンクを
  `base_link` の親に足す。
- MJCF を作り直したら（`scripts/build_robot.py`）、このスクリプトも実行し直す。URDF は手で編集しない。
