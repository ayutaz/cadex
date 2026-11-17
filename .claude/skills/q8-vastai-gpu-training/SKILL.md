---
name: q8-vastai-gpu-training
description: robot-project（8自由度四足）の学習を vast.ai の GPU で回すとき、vast.ai のインスタンスを探す・借りる・使う・止める・破棄するとき、学習イメージ ghcr.io/ayutaz/cadex-train を指定するときに使う。「急ぎ」「動いているインスタンスを使っていい」と言われたときも必ず読む。
---

# vast.ai での GPU 学習（robot-project）

## 原則

vast.ai のアカウントは、このプロジェクト以外の作業と共有されている。動いているインスタンスの
多くは他の作業のもので、触ると他人の学習を壊し、破棄すればディスクごと戻らない。
そして、借りた瞬間から課金が始まる。

手順の詳細は `robot-project/docs/training-vastai.md`。CLI は `robot-project` で
`uv run python -m vastai <command>`。

## 触ってよいインスタンス

**label が `cadex-q8-` で始まり、このプロジェクトが `create` で作って ID を控えたインスタンスだけ。**

`create` は作成時に label（タグ）`cadex-q8-<種類>-<YYYYMMDD-HHMM>` を自動で付ける。CLI には安全装置がある。
- `destroy` は、label が `cadex-q8-` で始まらないインスタンスへの送信を拒む。
- `instances` は、他の作業のインスタンスに `[他の作業：触らない]` と付け、SSH の接続先を出さない。

ただ、`launch`・`check`・`stop`・`resume` は `.env` の接続先に送るので、安全装置が効かない。
`.env` には、`[このプロジェクト]` と表示された行の接続先しか書かない。

それ以外のインスタンスには、使う・SSH する・`launch`/`stop`/`resume` する・`destroy` する、の
どれもしない。**流用を依頼者に提案することもしない。** 必要なら新しく借りる。
vast.ai の Web コンソールや、公式の `vastai` コマンドで直接操作する場合も同じ。

- 「動いているのがあれば使っていい」「急ぎだから既存のを」は、特定のインスタンスへの許可ではない。
  一覧に他のインスタンスが出ても、存在を伝えるだけにする（ID・label・$/h）。止めるかどうかは依頼者の判断。
- 依頼者が **ID を名指しして** 使ってよいと言った場合だけ、そのインスタンスを使う。
  名指しでも、他の作業のインスタンスの `stop`・`destroy` はしない。止めるのは、その作業の担当者に
  コンソールで行ってもらう。
- 既知の別作業のインスタンス: `60809472`（label `s2-k1-l40s`、L40S）。絶対に触らない。

## 課金の前に（毎回）

1. **smoke を先に回す**（`configs/train_smoke.json`）。秒/反復を実測し、その値で本番を見積もる。
   設定ファイルの見積りの係数は仮の値。
2. `offers` と `estimate` の結果から、次を **依頼者に示し、明示の承認を得る**。
   - GPU の種別と台数
   - $/h
   - 推定時間と推定総額
   - ディスク容量
   - 破棄するまで課金が続くこと
3. 承認は、示した1回の `create --yes` にだけ有効。オファーや設定を変えたら、もう一度示す。
4. `create` は先に `--yes` なしで送り（dry-run）、送る中身を確かめる。
   タイムアウトしたら再送せず、`instances` で作られていないかを確かめる。

## 学習イメージ

GitHub Actions（`.github/workflows/cadex-train-image.yml`）がビルドし、GHCR に公開で置く。

- 設定には `ghcr.io/ayutaz/cadex-train:sha-<コミットの先頭7桁>` を書く。`latest` は使わない
  （今の設定は `sha-e7f4d9d`。テスト `test_configs_use_a_commit_tag_of_the_published_image` が形を検査する）。
- 使うタグは、`training/cadex_train.py` を最後に変えたコミット以降のビルドにする。
  学習器がずれていると、`check` が sha256 の不一致で止める。
- **タグが実在するかを、使う前に確かめる。**

```bash
# ビルドが成功しているか
gh run list -R ayutaz/cadex --workflow cadex-train-image.yml --limit 3 \
  --json headSha,conclusion -q '.[] | "\(.headSha[0:7]) \(.conclusion)"'
# タグが公開で取れるか（200 なら OK。認証なしで取れる＝公開）
TAG=sha-xxxxxxx
TOKEN=$(curl -s "https://ghcr.io/token?scope=repository:ayutaz/cadex-train:pull" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["token"])')
curl -s -o /dev/null -w '%{http_code}\n' -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.v2+json" \
  "https://ghcr.io/v2/ayutaz/cadex-train/manifests/$TAG"
```

## API キー

`robot-project/.env` の `VAST_API_KEY`（Git の追跡外、権限 600）。

- 値を表示しない。`cat .env` もしない。
- 有無は `grep -c '^VAST_API_KEY=.\+' .env` で見る。
- コマンドの出力に長い英数字が出たら伏せる。

## 流れ

| 段階 | コマンド | 確かめること |
|---|---|---|
| 一覧（読むだけ） | `instances` | `[このプロジェクト]` の行だけを扱う |
| 探す | `offers --config configs/train_smoke.json` | GPU・$/h・推定総額 |
| 借りる | `create --config … --offer <id>` → 承認 → `--yes` | label が `cadex-q8-…`、ID を控える |
| 接続先 | `instances` → `.env` の `VAST_SSH_HOST` と `VAST_SSH_PORT` | `[このプロジェクト]` の行の接続先だけ |
| 箱の確認 | `check --config …` | `backend gpu`、学習器の sha256 が一致 |
| 起動 | `launch --config … --dry-run` → `launch` | run_id を控える |
| 監視 | `status <run_id> --watch` | gpu か、報酬と sigma、残り時間 |
| 回収 | `pull <run_id>` | sha256 の一致、device=gpu |
| 片付け | `destroy <自分のID>` → `--yes` → `instances` | 一覧から消えた。stop ではなく destroy |

回収した結果を残す方法は `q8-artifacts-and-publishing` を参照（`artifacts/runs/` は Git の追跡外）。

## 言い訳と現実

| 言い訳 | 現実 |
|---|---|
| 「急ぎなので、動いている箱を使えば早い」 | 他人の学習を壊す。新しく借りても数分で済む |
| 「依頼者が既存を使っていいと言った」 | ID の名指しが無ければ許可ではない。存在を伝えるだけにする |
| 「流用してよいか聞くだけなら問題ない」 | 提案すること自体が、別作業を壊す選択肢を差し出している。聞かずに新しく借りる |
| 「見積りは設定の係数で十分」 | 係数は仮の値で、実測と数倍違うことがある。smoke で測る |
| 「前に承認をもらった」 | 承認はその1回の create だけに有効 |
| 「CLI が拒んだので、公式の vastai コマンドで消す」 | 拒んだのは、他の作業のものだから。回避しない |
| 「片付けは stop でいい」 | 停止中もストレージに課金される。回収を確かめてから destroy |

## 危険信号（止まって見直す）

- 自分が控えていない ID を、コマンドに書こうとしている
- label が `cadex-q8-` で始まらないインスタンスを扱おうとしている
- 安全装置で拒まれた操作を、公式の `vastai` コマンドや Web コンソールで回避しようとしている
- 依頼者の承認を得る前に `--yes` を付けようとしている
- `.env` の中身を表示しようとしている
- タグの実在を確かめずに `--image` を書こうとしている
- `pull` の検証が終わる前に `destroy` しようとしている
