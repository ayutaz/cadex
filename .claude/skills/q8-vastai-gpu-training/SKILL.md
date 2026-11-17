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

ただ、`launch`・`check`・`resume` は、環境変数 `VAST_SSH_HOST`/`VAST_SSH_PORT`（`.env` より優先）か
`.env` の接続先に送るので、安全装置が効かない。そこには `[このプロジェクト]` と表示された行の
接続先しか入れない。複数台を使うときは、`.env` を書き換えずに環境変数で台ごとに渡す。
`status`・`pull`・`stop` は、その回を起動した接続先（`run.json` の `host`/`port`）に送る。
`sweep` は、label が `cadex-q8-` でない台には何もしない。

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

1. **見積りは実測の秒/反復で出す。smoke では出さない。** 10 反復の smoke は時間のほとんどが
   JIT コンパイルで、実際の約7倍に出た（4.1 対 0.55 秒/反復）。
   - 今の設定（8192 envs、128-64、RTX 4090）の実測は約 0.55 秒/反復＝1000 反復で約 550 秒。
     設定の `estimate.seconds_per_iteration` は 0.6。
   - 設定や GPU が変わったら、本番の最初の数十反復の `progress.json` で `estimate --from-progress` する。
   - smoke を回すのは、イメージか学習器か送り方を変えたときだけ（経路と GPU の確認）。
     本番の前に毎回はさまない。
2. 借りる前に、次を **依頼者に示す**。
   - GPU の種別と台数、$/h、推定時間と推定総額、ディスク容量、破棄するまで課金が続くこと
3. 承認の取り方は、依頼者の言葉で決まる。
   - **予算の上限と GPU 選びの委任がある**（例「$100 まで使っていい、GPU は任せる」）:
     その範囲の `create --yes` は承認済み。2 を報告として伝えて進めてよい。
     ただし、これまでの分を含めた累計が予算を超えそうなら、止まって聞く。
   - それ以外: 明示の承認を得る。承認は示した1回の `create --yes` にだけ有効。
     オファーや設定を変えたら、もう一度示す。
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

複数回（学習シード違い、機体違い）は、**1回1台で並べる**。1回約 550 秒なので、10回なら
10台で約30分、1台なら約2時間。費用は台数でほとんど変わらない（台×時間が同じ）。

| 段階 | コマンド | 確かめること |
|---|---|---|
| 一覧（読むだけ） | `instances` | `[このプロジェクト]` の行だけを扱う |
| 探して借りる | `create --config configs/train_full.json --count N`（dry-run）→ `--yes` | 検索の直後に安い順に借りる。label が `cadex-q8-…`、ID を控える |
| 回す（複数回） | `sweep --config A --config B --seeds 0-4 --instance <ID> …`（`--dry-run` で計画）→ `--destroy-when-done` | 起動待ち・GPU 確認・起動・監視・回収・失敗の再投入・破棄まで行う。数十分かかるので、ログをファイルに出して裏で回す |
| 1回だけ | `VAST_SSH_HOST=… VAST_SSH_PORT=… check` → `launch [--seed N --run-id …]` → `status <run_id> --watch` → `pull <run_id>` | `backend gpu`、学習器の sha256、受領書の device=gpu、sha256 の三者一致 |
| 残す | `q8.training.archive_run(artifacts/runs/vastai/<run>, artifacts/training/<run>)` | 接続先と pid は落ちる。比べるなら `scripts/compare_seeds.py` |
| 片付け | `instances` で自分の ID が消えたか。残っていれば `destroy <自分のID>` → `--yes` | stop ではなく destroy |

借りるときの注意（2026-11-17 に実際に起きたこと）:

- **`offers` の表を見てから `--offer <id>` で借りると、id が無効になっていることが多い**（HTTP 400
  `no_such_ask`）。`--count` を使う。
- **回線は 1 Gbps 以上**（設定の `min_inet_down_mbps`）。遅い台はイメージの取得が30分以上終わらなかった。
  表示の速さは保証ではないので、急ぐときは1〜2台多めに借り、起動の早い台を使う（`sweep` は
  起動しない台を `--boot-timeout` で退かせ、余った台も列が空になれば退かせる）。
- **GPU が壊れた台がある**（`check` で `cuInit` が `CUDA_ERROR_UNKNOWN`）。`sweep` は退かせる。
  借り直すときは `--exclude-machine <machine_id>` で同じマシンを外す。
- 作った直後の台は、数十秒ほど一覧に出ないことがある。消えたと決めつけない。
  一覧に出ても、しばらくは `actual_status` が空（null）。

`sweep` を回している間の注意（2026-11-17、6台で初めて回したとき。詳細は `training-vastai.md` §6.1）:

- **running になった直後の台の `check` が Connection refused になるのは普通。** sshd がまだ立ち上がって
  いないため。`sweep` は5分まで間をおいて繰り返すので、手で `check` し直したり destroy したりしない。
  5分たっても繋がらなければ、`sweep` がその台を退かせる。
- **ある台のログが長く（10分以上）進まないときは、その台の手元の ssh が止まっていることを疑う。**
  ssh には ConnectTimeout・ServerAlive を付け、`check` のリモートのコマンドは `timeout` で区切ったが、
  それでも止まった台は `sweep` が自分では退かせない（1回の学習は約 500 秒なので、`status` が出ないまま
  10分たっていればおかしい）。
  手元の Mac でその台のポートに繋いでいる ssh だけを止める。`sweep` がその台を退かせ、回は列に戻る。
  ```bash
  pgrep -fl "ssh .*-p <その台のポート> .*<その台のホスト>"   # その台の接続先だけに絞る。確かめてから
  kill <そのpid>
  ```
  止めてよいのは、自分の `sweep` が自分の台に張った手元の ssh プロセスだけ。リモートの台や、
  他の作業のインスタンスには何もしない。`pkill ssh` のようにまとめて止めない。
- **20分たっても起動しない台（loading のまま）は、`--boot-timeout` で退いて破棄される。** 6台借りて
  1台がそうなった。予備を1〜2台多めに借りるのはこのため。
- **仕事の無くなった台は、すぐには破棄されない。** 失敗した回が列に戻るのに備えて、最後の回が
  終わるまで待ってから破棄される。そのぶん課金が少し増えるが、設計どおりなので手で消さない。
- 実測: 小型機（mini）の1回は 476〜514 秒（RTX 4090）。5回を6台で約23分（作成から最後の破棄まで）。

回収した結果を Git に残す方法は `q8-artifacts-and-publishing` を参照（`artifacts/runs/` は Git の追跡外）。

## 言い訳と現実

| 言い訳 | 現実 |
|---|---|
| 「急ぎなので、動いている箱を使えば早い」 | 他人の学習を壊す。新しく借りても数分で済む |
| 「依頼者が既存を使っていいと言った」 | ID の名指しが無ければ許可ではない。存在を伝えるだけにする |
| 「流用してよいか聞くだけなら問題ない」 | 提案すること自体が、別作業を壊す選択肢を差し出している。聞かずに新しく借りる |
| 「見積りのために smoke を先に回す」 | smoke はほぼコンパイルの時間で、約7倍に出る。本番の実測（約 0.55 秒/反復）か、本番の最初の数十反復で見積もる |
| 「予算を任されたが、念のため create の前に承認を取り直す」 | 委任された範囲なら承認済み。構成と見積りは伝えて進める。累計が予算を超えそうなときだけ聞く |
| 「1台で順に回せば管理が楽」 | 費用は同じで時間が台数分の1になる。`sweep` が管理する |
| 「表で選んだオファーの id で借りる」 | 数分で無効になる。`create --count` で検索の直後に借りる |
| 「前に承認をもらった」 | 承認はその1回の create だけに有効 |
| 「CLI が拒んだので、公式の vastai コマンドで消す」 | 拒んだのは、他の作業のものだから。回避しない |
| 「`sweep` が全部面倒を見るので、ログが止まっても待つ」 | 手元の ssh が止まると、その台のスレッドは動かないまま。その台のポートの ssh だけを止めて退かせる |
| 「起動直後に Connection refused だったので、壊れた台として消す」 | 起動直後は普通。`sweep` が5分まで再試行する |
| 「片付けは stop でいい」 | 停止中もストレージに課金される。回収を確かめてから destroy |

## 危険信号（止まって見直す）

- 自分が控えていない ID を、コマンドに書こうとしている
- label が `cadex-q8-` で始まらないインスタンスを扱おうとしている
- 安全装置で拒まれた操作を、公式の `vastai` コマンドや Web コンソールで回避しようとしている
- 依頼者の承認を得る前に `--yes` を付けようとしている
- `.env` の中身を表示しようとしている
- タグの実在を確かめずに `--image` を書こうとしている
- `pull` の検証が終わる前に `destroy` しようとしている
- 本番の見積りに smoke の時間を使おうとしている
- 複数台を使うのに `.env` の接続先を書き換えようとしている（環境変数で台ごとに渡す）
