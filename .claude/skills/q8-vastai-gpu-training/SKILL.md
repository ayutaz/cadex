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
5. **学習用バンドルが、今の機体の形から作ったものかを確かめる。** 形は
   `robot/design/` → 実ビルド → `robot/mjcf/<機体>/cut/` → `q8/dr_bundle.py` → `artifacts/<案>/cut/`
   の順に流れる。形を直したのにバンドルを作り直さないと、古い形で全部の回を学習してしまう
   （2026-11-18 に、作り直しが2回要った）。「すぐ回して」と言われても、借りる前に毎回見る。
   ```bash
   # 上流ほど古く、下流ほど新しいはず。下流の時刻が上流より前なら、そこから作り直す
   git log -1 --format='%ci %h' -- robot/design/              # 形の元
   git log -1 --format='%ci %h' -- robot/mjcf/<機体>/cut       # 実ビルドしたカットモデル
   git log -1 --format='%ci %h' -- artifacts/<案>/cut          # DR バンドル
   git status --short -- robot/design robot/mjcf/<機体> artifacts/<案>   # 未コミットの作り直し・直しかけ
   uv run pytest tests/test_vastai_config.py -k "今の機体に" -q   # バンドル＝今のカットモデル＋DR（1バイト一致）
   grep '"bundle"' configs/train_full_<案>.json                # 設定が artifacts/<案>/cut/walk-task.json を指すか
   ```
   一致のテストは、カットモデルとバンドルを比べるだけ。カットモデル自体が古ければ通ってしまうので、
   時刻の比べ（`robot/design/` より後に実ビルドしたか）を省かない。案がテストの一覧
   （`TAIL_TRI_DR_BUILT` など）に無ければ、テストは何も見ていない。作り直し方は `q8-body-variants` を参照。
   設計スクリプト（`robot/design/quadruped8.py`）は全部の機体で1つなので、直しがどの機体に効くかは、
   差分を読んで推測せずに確かめる。その機体を作り直し、`robot/mjcf/<機体>/cut/` の各ファイルを前の版と
   `cmp` で比べる。1バイト同じならバンドルはそのままでよく、違えばバンドルも作り直す。

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

複数回（学習シード違い、機体違い）は、**1回1台で並べ、予備を1台足す：N 回なら `create --count N+1`。**
1回約 10〜22 分（RTX 4090）。費用は同時に回しても順に回しても、ほぼ同じ（台数×時間）で、待ち時間だけが短くなる。
予備は、鍵が通らない台・起動しない台・GPU が壊れた台の回を引き受ける（`sweep` が自動で外して破棄し、
回を列に戻す）。予備の数は N+1 に固定し、その場の判断で2台以上にしない。

- **N は、1回の `sweep` で回す回の合計（機体×シード）。** 予備は全体で+1。設定ごとに+1 にしない
  （4機体×3シードなら `--count 13`）。
- **機体が複数でも、`create --config` はどれか1つの設定で借りればよい。** `create` が設定から使うのは
  台の条件（`offer` と、label 以外の `instance`：イメージ・ディスク）だけで、バンドルは `sweep` が回ごとに
  送る。2026-11-18 は4機体を `train_full_tri-r2-free-dr.json` 1つで借りた。設定どうしで `offer` や
  `instance` が違うときは、それぞれの設定で借りる。
- **予算の累計は、借りるたびに数える。** ledger か報告に毎回「累計 約$X（概算）」と書き、
  前回の値に今回の概算を足す。累計が予算を超えそうなら、借りる前に止まって聞く。

実績（2026-11-17〜18、RTX 4090）: 12回＋予備1台で約 $2.1。13台のうち2台は SSH の鍵が通らず
（`Permission denied (publickey)`）、`sweep` が外して破棄し、予備ともう1台が回を引き受けた。
金額は `$/h × 台 × 時間` の概算で、vast.ai の請求額とは突き合わせていない。依頼者には「概算」と書いて伝える。

| 段階 | コマンド | 確かめること |
|---|---|---|
| 一覧（読むだけ） | `instances` | `[このプロジェクト]` の行だけを扱う |
| 探して借りる | `create --config configs/train_full_<機体>.json --count <回数+1>`（dry-run）→ `--yes` | 検索の直後に安い順に借りる。label が `cadex-q8-…`、ID を控える |
| 回す（複数回） | `sweep --config configs/train_full_<機体>.json [--config …] --seeds 0-4 --instance <ID1> --instance <ID2> …`（`--dry-run` で計画）→ `--destroy-when-done` | 起動待ち・GPU 確認・起動・監視・回収・失敗の再投入・破棄まで行う。回の名前は `<設定の name>-s<シード>`（例 `full-mini-s0`）。数十分かかるので、ログをファイルに出して裏で回す |
| 1回だけ | `VAST_SSH_HOST=… VAST_SSH_PORT=… check` → `launch [--seed N --run-id …]` → `status <run_id> --watch` → `pull <run_id>` | `backend gpu`、学習器の sha256、受領書の device=gpu、sha256 の三者一致 |
| 残す | `q8.training.archive_run(artifacts/runs/vastai/<run>, artifacts/training/<run>)` | 接続先と pid は落ちる。比べるなら `scripts/compare_seeds.py`（評価も行う） |
| 片付け | `instances` で自分の ID が消えたか。残っていれば `destroy <自分のID>` → `--yes` | stop ではなく destroy |

借りるときの注意（2026-11-17 に実際に起きたこと）:

- **`offers` の表を見てから `--offer <id>` で借りると、id が無効になっていることが多い**（HTTP 400
  `no_such_ask`）。`--count` を使う。
- **回線は 1 Gbps 以上**（設定の `min_inet_down_mbps`）。遅い台はイメージの取得が30分以上終わらなかった。
  表示の速さは保証ではない。遅い台は予備（N+1 の1台）が埋める（`sweep` は
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
  探すときは、推測した形で絞らない。先に `sweep` が実際に張っているコマンド行を見て、その形で絞る
  （`ssh` のほか、回収の `rsync -e ssh …` もある）。
  ```bash
  pgrep -fl ssh                                # まず実際のコマンド行を見る（オプションの並び・user@host の形）
  pgrep -fl "<上で見た、その台のポートとホストを含む形>"   # その台の接続先だけに絞る。確かめてから
  kill <そのpid>
  ```
  止めてよいのは、自分の `sweep` が自分の台に張った手元の ssh プロセスだけ。リモートの台や、
  他の作業のインスタンスには何もしない。`pkill ssh` のようにまとめて止めない。
- **20分たっても起動しない台（loading のまま）は、`--boot-timeout` で退いて破棄される。** 6台借りて
  1台がそうなった。予備を1台足すのはこのため。
- **SSH の鍵が通らない台がある**（`Permission denied (publickey)`。13台のうち2台）。`sweep` が退かせて
  破棄するので、手で消したり鍵を直しに行ったりしない。
- **仕事の無くなった台は、すぐには破棄されない。** 失敗した回が列に戻るのに備えて、最後の回が
  終わるまで待ってから破棄される。そのぶん課金が少し増えるが、設計どおりなので手で消さない。
- 実測: 小型機（mini）の1回は 476〜514 秒（RTX 4090）。5回を6台で約23分（作成から最後の破棄まで）。

### 終わりを待つ：知らせだけに頼らない

バックグラウンドの `sweep` の「終わりの知らせ」は、届かないことがある。2026-11-18 に2回、終わっているのに
35〜45 分止まった（17:45 に終わったのに、評価が始まったのは 18:27）。待ちは次の形にする。

1. `sweep` はログをファイルに出して裏で走らせる（`... > sweep.log 2>&1`）。終わると最後に
   `完了 N:`・`失敗 N:`・`未着手 N:` の3行が出る。
2. **15分ごとに自分が起きる仕掛けを、走らせた直後に置く。** どちらか。
   - Monitor ツール（`timeout_ms: 900000`）。期限で切れたこと自体が15分ごとの起床になる。切れたら下の4つを見て、また置く。
     ```bash
     tail -n0 -F sweep.log | grep -E --line-buffered '完了・回収済み|が失敗|退く|破棄に失敗|^未着手 |Traceback'
     ```
   - `run_in_background` の Bash で `sleep 900; tail -n 5 sweep.log`。終わると自分が起きる。起きたら、また置く。
3. 起きたら（知らせの有無にかかわらず）、次の4つを見る。
   - `sweep.log` の末尾（`未着手 N:` の行があれば終わっている。各台の最後の行の時刻）
   - `uv run python -m vastai instances`（自分の台が残っているか、`[このプロジェクト]` の行だけ）
   - `artifacts/runs/vastai/<run>/` の回収（回の数だけそろったか）
   - 成果物の更新時刻（`ls -lt artifacts/runs/vastai/ | head`）
   終わっていれば、知らせを待たずに次（回収の確かめ・評価・`instances` で破棄の確認）に進む。
   1台のログが10分以上進まないなら、上の「手元の ssh が止まっている」の手当てをする。
4. **リーダー（担当に待ちを任せた側）は、CronCreate で15分ごとの見張りを置く。** 見張りは、担当の
   最後の報告と `sweep.log` の末尾を見て、終わっているのに担当が動いていなければ、SendMessage で
   担当を再開させる（「sweep は終わっている。評価に進んで」）。sweep が全部終わり、評価まで済んだら見張りを消す（CronDelete）。

学習設定は機体ごとに `configs/train_full_<機体>.json`（base は `train_full.json`）。新しい機体の設定の作り方、
回収した結果を Git に残す方法と比べ方は `q8-artifacts-and-publishing` を参照（`artifacts/runs/` は Git の追跡外）。

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
| 「終わったら知らせが来るので、それまで待つ」 | 知らせは届かないことがある（35〜45 分止まった）。15分ごとに起きる仕掛けを置き、自分で見に行く |
| 「念のため予備を2〜3台」 | 予備は1台（N+1）。鍵や起動の失敗は `sweep` が予備に回す |
| 「機体ごとの設定で、それぞれ +1 台」 | N は sweep 全体の回の合計。予備は全体で1台 |
| 「形を直したのは別の担当で、バンドルは前に作ってある」 | 古い形で全部の回を学習しうる。借りる前に、形の元・カットモデル・バンドルの順に時刻を比べる |
| 「1バイト一致のテストが通ったので、バンドルは新しい」 | テストはカットモデルとバンドルを比べるだけ。カットモデルが古ければ通る。`robot/design/` との時刻も見る |
| 「費用は $X だった」 | `$/h × 台 × 時間` の概算。請求額と突き合わせていなければ「概算」と書く |

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
- `sweep` を裏で走らせたのに、15分ごとに起きる仕掛け（Monitor か `sleep 900`）を置かずに待とうとしている
- `create --count` に回数ちょうど、または回数+2 以上を入れようとしている（回数+1）
- 形を直すコミットの後なのに、バンドルの時刻を見ずに借りようとしている
