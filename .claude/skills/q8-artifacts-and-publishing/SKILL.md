---
name: q8-artifacts-and-publishing
description: robot-project（8自由度四足）の学習結果・評価結果・設計レポート・派生機体の出力を Git に残すとき、artifacts/runs の結果をコミットしたいと言われたとき、ayutaz/cadex（公開 fork）や ~/cadex-projects のリポジトリにコミット・push するときに使う。
---

# 成果物の残し方と公開（robot-project）

## 原則

`ayutaz/cadex` は**公開**リポジトリ。このマシンのユーザー名やフォルダ構成、秘密情報、巨大なファイルを
一度でも push すると、履歴から消すのは難しい。成果物は「再現と比較に要る小さいもの」だけを、
決まった場所へ、決まった関数を通して残す。

## 何をどこに残すか

| 成果物 | 残し方 | 置き場所（コミットされる） |
|---|---|---|
| 学習の実行（vast.ai は `artifacts/runs/vastai/<run>/`、手元は `artifacts/runs/<run>/`。Git の追跡外） | **`archive_run` で写す**（下のコマンド） | `artifacts/training/<run>/` |
| 学習シード違いの各回の評価 | `scripts/compare_seeds.py`（評価もこれが行う） | `artifacts/<variant>/evaluation-seeds/<run>/` |
| 学習シード違いの比較表 | `scripts/compare_seeds.py` | `artifacts/compare/seeds_<a>_vs_<b>.{md,json}` |
| 1つの方策の評価（`cadex evaluate`） | `evaluation/evaluate_variant.py`、または `scripts/redesign.py` | `artifacts/<variant>/evaluation/` |
| 設計レポート・レンダー | `scripts/build_robot.py` | `artifacts/<variant>/design_report.json`、`renders/` |
| 機体の比較表（1回ずつ） | `scripts/redesign.py` | `artifacts/compare/<a>_vs_<b>.{md,json}` |
| STEP・MJCF・学習タスク・カットモデル | `scripts/build_robot.py` | `robot/meshes/<v>/*.step`、`robot/mjcf/<v>/` |
| 学習設定 | 手で作る（下の「新しい派生機体」） | `configs/train_full_<variant>.json` |
| 動画 | 再生スクリプトの出力 | `artifacts/videos/*.mp4` |

追跡外のもの（コミットしない）:
- `artifacts/runs/`
- `seed-*-trace.json`（1評価で約29 MB）
- `*.webm`
- `robot/meshes/*/*.stl`
- `artifacts/tmp/`
- `.env`、`*.pem`

`git add -f` で押し込まない。

学習の実行を残すコマンド。方策・progress・receipt・checkpoints・最終評価・stderr を写し、
絶対パスを置き換え、receipt に混ざった警告の行を落とす。

```bash
cd robot-project
# vast.ai の回（sweep の run は <設定の name>-s<シード>、例 full-mini-s0）
uv run python -c "from pathlib import Path; from q8.training import archive_run; \
print(archive_run(Path('artifacts/runs/vastai/<run>'), Path('artifacts/training/<run>')))"
```

## 新しい派生機体を作って、学習シード違いで比べるまで

例: mini を少し変えた `mini-x` を5回学習し、mini と比べる（cwd は `robot-project`）。

1. **機体を作る**。`build_robot.py` は base の値に `--set` を重ねるので、**元の機体の値を全部渡す**
   （mini の値は `artifacts/mini/design_report.json` の `params`、または `docs/experiment-results.md`）。
   出力は `artifacts/mini-x/`、`robot/meshes/mini-x/`、`robot/mjcf/mini-x/`、
   Cadex プロジェクト `~/cadex-projects/quadruped8-mini-x`。
   ```bash
   uv run python scripts/build_robot.py --variant mini-x --set body_length=96 --set body_width=62 \
     --set battery_mass_g=0 --set hip_x=40 --set thigh_length=40 --set shin_length=44 --set <変える値>
   ```
   **`--set` は1つずつ書き下す。** zsh では `$args` が単語に分かれないので、ループで文字列に
   組み立てると1つの引数で渡り、名前に空白の入ったフォルダが `~/cadex-projects` や `artifacts/` に
   できた（2026-11-17）。できたら消して作り直す。
2. **学習設定を作る**。比べる相手の設定（`configs/train_full_mini.json`）を写して
   `configs/train_full_mini-x.json` にし、`name`=`full-mini-x`、`bundle`=`robot/mjcf/mini-x/cut/walk-task.json`、
   `trainer.label`、`instance.label`=`q8-full-mini-x`、`_comment` だけを変える。
   `tests/test_vastai_config.py` で、差が意図した項目だけであることをテストで固定する。
   base と比べる形状違いは `VARIANT_CONFIGS`、mini と形が同じでタスクだけ違うものは `MINI_TUNING` に足す。
   どちらにも当てはまらない（例: mini の形を変える）なら、同じ形のテストを1つ足す（TDD: 先にテストを書く）。
3. **学習する**: `q8-vastai-gpu-training` の `create --count 6`（5回＋予備1〜2台）→
   `sweep --config configs/train_full_mini-x.json --seeds 0-4 --instance …`。
   回は `artifacts/runs/vastai/full-mini-x-s0` … `-s4` に回収される。
4. **残す**: 5回とも `archive_run` で `artifacts/training/full-mini-x-s<N>/` に写す。
5. **比べる**: 評価もこのスクリプトが行う（評価済みの回は飛ばす）。最初に書いた機体が `<a>` になる。
   ```bash
   uv run python scripts/compare_seeds.py \
     mini=artifacts/training/full-mini-s0 ... mini=artifacts/training/full-mini-s4 \
     mini-x=artifacts/training/full-mini-x-s0 ... mini-x=artifacts/training/full-mini-x-s4
   ```
   → `artifacts/mini-x/evaluation-seeds/full-mini-x-s<N>/`、`artifacts/compare/seeds_mini_vs_mini-x.{md,json}`
6. **コミットする**（下の確認のあと）: `robot-project/` の `configs/train_full_mini-x.json`、
   `tests/test_vastai_config.py`、`artifacts/mini-x/`、`robot/meshes/mini-x/*.step`、`robot/mjcf/mini-x/`、
   `artifacts/training/full-mini-x-s*/`、`artifacts/compare/seeds_mini_vs_mini-x.*`。
   機体と設定（学習前）と、学習結果と比較（学習後）は、別のコミットにしてよい。
   Cadex プロジェクトは作者を確かめてから `gh repo create ayutaz/quadruped8-mini-x --private …`。

## 絶対パス

Cadex・学習器・`cadex evaluate` は、絶対パス（`/Users/<名前>/…`）を記録する。上の表のスクリプトと
`archive_run` は、`q8/portable.py` で自動的に置き換える（`<robot-project>`、`<cadex>`、
`~/cadex-projects`、`~`）。

**それ以外の方法で置いたファイル**（手で写した、別のツールで作った）は、コミット前に置き換える。

```bash
uv run python -c "from q8.portable import scrub_tree; print(scrub_tree('<dir>'))"
```

`tests/test_portable.py::test_tracked_files_have_no_home_path` は、追跡中のファイルにホームの
パスがあると失敗する。**`git add` の後、コミットの前に `uv run pytest -q` を回す**
（このテストは、追跡済みのファイルしか見ないため）。

## コミットと push の前の確認

```bash
cd /Users/s19447/Documents/cadex
git config user.name; git config user.email          # ayutaz / ka1357amnbpdr@gmail.com
(cd robot-project && uv run pytest -q)                # 絶対パスの検査を含む
git diff --cached --stat                              # 追跡外のものが混ざっていないか
git diff --cached --name-only | xargs -I{} du -k {} 2>/dev/null | sort -n | tail -3   # 大きいファイル
git log --format='%h %an <%ae>' origin/quadruped-8dof..HEAD    # push するコミットの作者
```

- 1ファイルが数 MB を超えたら、残す理由を確かめる。GitHub は 50 MB で警告し、100 MB で拒否する。
- 秘密情報（API キー、トークン、SSH 鍵、vast.ai の接続先）が入っていたらコミットしない。

## リポジトリと push 先

| リポジトリ | 公開範囲 | push |
|---|---|---|
| `ayutaz/cadex`（ブランチ `quadruped-8dof`） | 公開 | `git push origin quadruped-8dof`。main には push しない |
| `ayutaz/quadruped8`、`quadruped8-<variant>`（`~/cadex-projects/` の Cadex プロジェクト） | 非公開 | 各ディレクトリで `git push` |

- `~/cadex-projects/*` は、Cadex が設計プロジェクトごとに作り、自動でコミットする git。
  `ayutaz/cadex` には入れない。
- **作者**: robot-project のスクリプト（`q8.cadex.run` を通るもの）で Cadex を動かすと、
  cadex リポジトリの作者（ayutaz）が環境変数で渡り、プロジェクトのローカル設定もそろう。
  そのため、新しいプロジェクトでも最初の自動コミットから ayutaz になる。
  ただし、`./cadex` を直接叩いた場合やダッシュボードから作った場合は、PC 全体の設定
  （勤務先の名前とメール）で記録される。push の前に必ず作者を確かめる。

  ```bash
  git -C ~/cadex-projects/<name> log --format='%an <%ae>' | sort | uniq -c
  ```

  ayutaz 以外が混ざっていたら push しない。直すには履歴の書き換えが要るので、依頼者の了承を取る。
- 新しい派生機体のプロジェクトを GitHub に上げるときは、次のコマンドを使う。
  `gh repo create ayutaz/<name> --private --source ~/cadex-projects/<name> --remote origin --push`
- 強制 push と履歴の書き換えは、依頼者の明示の了承があるときだけ。

## よくある間違い

| 間違い | 正しくは |
|---|---|
| `artifacts/runs/<run>/` を「まるごとコミット」しようとする | `archive_run` で `artifacts/training/<run>/` に写す |
| 「方策の置き場所は別に決める」と保留する | 置き場所は `artifacts/training/<run>/` と決まっている |
| 派生機体を、変える値の `--set` だけで作る | base に重なるので、元の機体の値を全部渡す |
| `--set` をシェルのループで組み立てる | 1つずつ書き下す（zsh では渡らない） |
| 学習シード違いの評価を `artifacts/<variant>/evaluation/` に置く | `compare_seeds.py` に任せ、`evaluation-seeds/<run>/` に置く |
| 絶対パスを grep だけで探す | `uv run pytest -q`（ホームのパスの検査）を回す |
| 手で写した JSON をそのままコミットする | `scrub_tree` を通してからコミットする |
| 新しい Cadex プロジェクトを、作者を確かめずに push する | `git log` で全コミットの作者が ayutaz か確かめてから push する |
