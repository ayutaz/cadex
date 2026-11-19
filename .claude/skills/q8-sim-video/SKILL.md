---
name: q8-sim-video
description: robot-project（8自由度四足）で、学習した歩き方の動画を作るとき（確認用、X（Twitter）などに投稿する用）、評価トレース seed-*-trace.json から mp4 を作るとき、ffmpeg でタイトル・カード・日本語の文字を重ねるときに使う。
---

# 歩き方の動画の作り方（robot-project）

## 原則

動画は**シミュレーションの再生**であって、実機ではない。見た人に実機と誤解させない表示を必ず入れる。
物理は解かない。`cadex evaluate` が記録したトレースを再生して描く。投稿は利用者がする（こちらはしない）。
作業用のスクリプトと素材は scratchpad に置く。

## 必ず守る（利用者の指示。2回言われた）

- **最後のカード（まとめ・結論・「実機はこれから」などのページ）は付けない。** 動画は最後の画（ロボットの映像）で終わる。
  構成は「タイトルのカード → 画 → 画 → …」まで。まとめの文は、字幕か投稿の本文に回す。
- **機械学習の用語は英語のまま書く。** 例: reward、penalty、policy、PPO、iterations、swing / stance phase、
  duty factor、contact、steps、success。無理に日本語へ言い換えない（「罰」「方策」などにしない）。

## 1. 素材（画ごとの mp4）

- トレース: `artifacts/<variant>/evaluation-seeds/<run>/policy/seed-*-trace.json`（Git の追跡外）。
- 再生の部品は `evaluation/replay_trace.py` の `qpos_from_frame(model, frame, trace["actuator_channels"])`、
  描画は `q8/render.py` の `VideoWriter(path, w, h, fps)` と `camera(azimuth, elevation, distance, lookat)`。
  `replay()` は参考になるが1画しか作れないので、同じ流れを自分のスクリプトで書く。
  フレームは `frame_kind == "solver_output"` かつ `joint_positions` があるものだけ。
- モデルは `robot/mjcf/<variant>/visual.xml`（CAD メッシュ付き）。**学習だけの派生（`-dr`）は、元の機体のモデルを使う**
  （例: `mini-r2-p7-eff-dr` は `mini-r2-p7-eff` の visual.xml）。
- `opt.geomgroup[:] = [1, 1, 1, 0, 0, 0]` で見た目の幾何だけ出す。
- カメラは方位・仰角・距離を変えた複数の画（横、斜め、前、上など）を用意する。
  注視点は胴体の重心（`data.subtree_com[1]`）を指数平滑で追う（`0.9*前 + 0.1*今`）。
- 描画は 1280x720、50fps（トレースの `control_hz`）。

## 2. 仕上げ（ffmpeg 8）

- 各画を `fps=30` にそろえる。スローの画は `setpts=2*PTS`。
- 画は concat でつなぐ。
- タイトルのカードは `color` の lavfi に `drawtext` を重ねて作る。最後のカードは作らない（上の「必ず守る」）。
- 日本語のフォントは `/System/Library/Fonts/ヒラギノ角ゴシック W6.ttc`（太）や `W3.ttc`（細）。
  **作業フォルダに ASCII の名前で写してから使う**（`cp "…W6.ttc" $WORK/font-w6.ttc`）。
- 文字は `textfile=` で渡し、**絶対パス**にする。
- 書き出し: H.264 High、`yuv420p`、`-movflags +faststart`、`crf 20`。

```bash
WORK=/絶対パス/scratchpad/video
F6=$WORK/font-w6.ttc
ffmpeg -y -f lavfi -i color=c=0x101418:s=1280x720:d=3:r=30 \
  -vf "drawtext=fontfile=${F6}:textfile=${WORK}/title.txt:fontcolor=white:fontsize=56:x=(w-text_w)/2:y=(h-text_h)/2" \
  -c:v libx264 -profile:v high -pix_fmt yuv420p ${WORK}/title.mp4
```

変数のあとに `:` と英字が続くときは、必ず `${変数}` と波括弧で囲む（下の「よくある失敗」）。

## 3. 誤解させない表示（必須）

- 全編の左上に「シミュレーション（MuJoCo）」を出し続ける（`drawtext` を画に重ねる）。
- 成績の文字（速度、歩数、転倒なしなど）にも「（シミュレーション）」を付ける。
- 実機で動いていないことは、全編の左上の表示と、タイトルのカードの文字で示す（最後のカードは付けない）。

## 4. X への投稿条件

2分20秒以内、512 MB 以内、MP4。約39秒・3.8 MB で収まった。書き出したら `ffprobe` で確かめる。

## 5. 確認

- コマ抜きで 2x2 などに並べた画像（`-vf "fps=1/5,tile=2x2"`）を作り、**目で見る**（文字の欠け、重なり、機体が画外に出ていないか）。
- `ffprobe -v error -show_entries stream=codec_name,profile,pix_fmt,width,height:format=duration,size -of default=nw=1 <mp4>`
  で、長さ・解像度・コーデック・サイズを確かめる。

## 6. 置き場所とコミット

- `artifacts/videos/<variant>-<内容>.mp4`（例: `mini-r2-dr-s4-x-post.mp4`）。
- コミットするかは**利用者に聞く**（動画は Git から外すこともある）。`*.webm` は追跡外。
  残す手順は `q8-artifacts-and-publishing`。投稿はしない。

## よくある失敗

| 失敗 | 正しくは |
|---|---|
| zsh で変数の直後に `:textfile=` と続けると、`:t` が変数の修飾子（tail）と解釈され、drawtext が「Either text, a valid file…」で落ちる | `${F6}:textfile=` と波括弧で囲む |
| `textfile=` を相対パスにすると読めない | 絶対パスにする |
| 日本語・空白のあるフォントのパスを `fontfile=` に直接書く | ASCII の名前に写してから使う |
| `-dr` 派生の visual.xml を探して見つからない | 元の機体の `robot/mjcf/<variant>/visual.xml` を使う |
| 置き場所を `artifacts/<variant>/video/` と推測する | `artifacts/videos/` |
| 「シミュレーション」の表示を冒頭だけにする、または成績だけに書く | 全編の左上と、成績の文字と、タイトルのカード |
| 最後にまとめのカードを付ける | 付けない。最後の画で終わる（2回指示された） |
| 「罰」「方策」「報酬」など、機械学習の用語を日本語に言い換える | penalty、policy、reward など英語のまま |
| 字幕に半角の `%` を書くと、drawtext が「Stray %」で文字を落とす（`%%` でも直らなかった） | 全角の「％」を使う |
| 「後ろから」の画を方位 200° で撮る（前寄りの斜めになる） | 方位 0° 付近が後ろ（機体は +x へ進む）、90° が横、180° が前。描いた画を見て確かめる |
| 見ずに「できた」と報告する | コマ抜きの画像を見て、ffprobe を通す |
