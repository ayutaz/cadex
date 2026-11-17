# vast.ai で強化学習を回す（cadex_train.py）

確認日: 2026-11-17。対象は Cadex の `training/cadex_train.py`（Brax PPO on MJX/JAX）。

手元の Mac（GPU なし）でロボットを設計し、Cadex が書き出した学習タスクの
ペア（`*-task.json` と `*-model.xml`）を vast.ai で借りた GPU に送って学習し、
できた `.cxpolicy` を持ち帰る。そのための道具が `robot-project/vastai/` にある。

```
手元（robot-project/）                         vast.ai のインスタンス（自作イメージ）
  configs/train_*.json ──┐                       /opt/cadex/training/cadex_train.py
  robot/mjcf/base/       │  rsync over SSH       /opt/venv（ピン留めの依存 + jax[cuda12]）
    walk-task.json ──────┼──────────────────▶   /workspace/runs/<run_id>/
    model-model.xml ─────┘                         walk-task.json, model-model.xml
                                                   progress.json（毎反復）
  artifacts/runs/vastai/<run_id>/  ◀──────────     walk.000100.cxpolicy …（チェックポイント）
    run.json, progress.json, metrics.csv           walk.cxpolicy（最終）
    *.cxpolicy（sha256 で検証済み）                stdout.log（受領書）, train.log
```

## ファイル

| ファイル | 役割 |
|---|---|
| `vastai/Dockerfile` | CUDA 12 の公式イメージ + Python 3.12 + ピン留めの依存 + `jax[cuda12]==0.7.2` + 学習器 |
| `vastai/cli.py`（`python -m vastai`） | 下のすべてのコマンドの入口 |
| `vastai/offers.py` | vast.ai REST API: オファー検索・見積り・作成・一覧・破棄 |
| `vastai/remote.py` | SSH/rsync: 送付・起動・進捗・回収と sha256 検証・再開・停止 |
| `vastai/metrics.py` | `progress.json` → CSV / TensorBoard イベントファイル（依存なし） |
| `vastai/config.py` | 学習設定と接続設定の読み込み・検証 |
| `configs/train_smoke.json` | 小規模のパイプライン検証用（10 反復）。一反復の大きさ（envs・unroll・epochs・minibatches・hidden）と GPU の候補は本番と同じにしてあり、秒/反復の実測をそのまま本番の見積りに使える（テストで固定） |
| `configs/train_full.json` | 本番用、基準機体（1000 反復、8192 envs。環境数などは Cadex `training/RECIPE.md` §A の既定） |
| `configs/train_full_leg120.json` | 本番用、脚20%増し（leg120）。機体（`bundle`）以外は `train_full.json` と同じ（テストで固定） |

コマンドはすべて `robot-project/` で実行する:

```bash
cd robot-project
.venv/bin/python -m vastai --help
```

**課金が始まる操作は `create` だけ**、インスタンスを消すのは `destroy` だけで、
どちらも `--yes` と `VAST_API_KEY` の両方が揃った時しか送らない。それ以外は
送るはずの内容を表示して終わる（dry-run が既定）。`offers` と `instances` は読むだけ。

**vast.ai のアカウントは他の作業と共有している。このプロジェクトが作ったインスタンスにしか触らない。**

- `create` は、作成時に label（タグ）`cadex-q8-<種類>-<YYYYMMDD-HHMM>` を自動で付ける。
  種類は設定の `instance.label` から取る（`q8-smoke` → `smoke`）。
- `destroy` は、送る前に一覧で label を確かめ、`cadex-q8-` で始まらなければ送らない。
- `instances` は、他の作業のインスタンスに `[他の作業：触らない]` と付け、SSH の接続先を出さない。
  `.env` に書けるのは、このプロジェクトのインスタンスの接続先だけになる。

## 1. 準備

### API キーの置き方

1. vast.ai のコンソール（Account → API Keys）でキーを作る。
   できればこの用途専用のキーにする。
2. `robot-project/.env` に書く（`.gitignore` 済み。`.env.example` が雛形）:

   ```
   VAST_API_KEY=xxxxxxxx
   ```

   または端末で `export VAST_API_KEY=...`。環境変数が `.env` より優先される。
3. `.env` は `KEY=value` を字面どおりに読むだけで、シェルとして実行しない。
   キーは HTTP の `Authorization` 見出しにだけ載せ、表示も保存もしない。
   チャットやコミットに貼らないこと。

### SSH 鍵

vast.ai のコンソール（Account → SSH Keys）に公開鍵を登録しておく。作成時に
インスタンスへ配られる。暗号化した鍵は `ssh-add` しておく（`BatchMode=yes` で
繋ぐので、パスフレーズの入力はできない）。鍵を明示するなら `.env` に
`VAST_SSH_KEY=~/.ssh/vast_ed25519`。

### イメージを作る

イメージは **GitHub Actions がビルドし、GHCR に公開で置く**
（`.github/workflows/cadex-train-image.yml`）。名前は `ghcr.io/<owner>/cadex-train` で、
fork（ayutaz/cadex）では `ghcr.io/ayutaz/cadex-train`。

- 次のどれかが変わったブランチを push すると、自動で作り直す。手で走らせるなら
  Actions の画面か `gh workflow run cadex-train-image.yml --ref <ブランチ>`。
  - `vastai/Dockerfile`、このワークフロー
  - Dockerfile が COPY するファイル（`training/cadex_train.py`・`cadex_goto.py`・
    `requirements.txt`、`src/Mod/cadex/` の純 Python 3 ファイル）
- タグは `sha-<コミットの先頭7桁>` と `latest`。学習の設定には、再現できるよう
  `sha-…` のほうを書く。
- 手元（Apple Silicon）では作らない。linux/amd64 の CUDA 版 JAX を含むイメージを
  エミュレーションで作るのは重い。どうしても手元で作るなら、Cadex リポジトリの根で
  `docker build --platform linux/amd64 -f robot-project/vastai/Dockerfile -t <名前> .`。
- 公開イメージなので、vast.ai から取ってくるときにログイン情報は要らない。
  中身は学習器の5ファイルと公開ライブラリだけで、秘密情報は入らない。
- COPY するファイルは LGPL-2.1-or-later。ソースの場所はイメージのラベル
  `org.opencontainers.image.source` に残し、ソースはこの公開リポジトリにある。
- リポジトリ全体を clone せず必要なファイルだけを COPY する理由は、Dockerfile 冒頭の
  コメントにある（学習器は標準ライブラリ以外を最上位で import せず、FreeCAD もビルドも要らない）。
- **学習器を変えたらイメージを作り直す。** 走った学習器の sha256 はイメージの
  `/opt/cadex/TRAINER_SHA256` とポリシーの `trainer_sha256` に残り、
  `python -m vastai check` が手元の `training/cadex_train.py` と比べる。
- 設定の `instance.image` にイメージ名を書く（`<your-registry>/…` のままだと
  `create` は送らない）。

## 2. 料金の目安を出し、オファーを選ぶ

```bash
.venv/bin/python -m vastai offers --config configs/train_smoke.json
```

条件（設定の `offer` 節: GPU 名、台数、信頼度、CUDA の版、ディスク、回線、
上限 $/h、verified）で検索し、安い順に GPU 種別・VRAM・$/h・**推定総額**・
信頼度・CUDA・下り回線・場所を表示する。推定総額は

```
推定総額 = $/h × 借りる時間 + ディスク GB × storage_cost($/GB/月) × 借りる時間 / 730
借りる時間 = (反復数 × 秒/反復 / 3600 + 準備分 / 60) × 余裕
```

（設定の `estimate` 節: `seconds_per_iteration`, `setup_minutes`, `margin`）。
転送料と、停止したまま置いておくディスク代は含まない。

選び方の目安:

- `cuda_max_good`（ドライバが扱える CUDA）が 12.4 以上。`jax[cuda12]` が動く。
- 信頼度（`reliability2`）0.98 以上、verified。途中で落ちると借りた時間が無駄になる。
- VRAM: 8192 envs の本番は 24 GB 級（RTX 4090）以上を目安に。足りなければ
  `envs` を下げる（`minibatches` で割り切れる値に）。
- 下り回線: イメージの取得に効く。準備時間（`setup_minutes`）に反映する。

## 3. 時間とコストの見積り方

一反復の秒数はロボットと envs と GPU でまるで違う（Cadex `training/RECIPE.md` §B:
RTX 5090 で Go1 の 1260 反復が 333 秒＝約 0.26 秒/反復、別の機体では 8192 envs で
14.1 秒/反復）。**推測せず、smoke で実測してから本番を見積もる**:

1. 安いオファーで smoke（10 反復）を回して回収する（下の 4〜7）。
2. 実測から本番を見積もる:

   ```bash
   .venv/bin/python -m vastai estimate --config configs/train_full.json \
       --from-progress artifacts/runs/vastai/<smoke の run_id>/progress.json --dph 0.45
   ```

   `progress.json` の `wall_time_s / 反復数` を秒/反復とする。今の smoke は本番と同じ envs
   なので拡大しない（envs が違う設定で測った場合は比例で拡大し、上限の目安になる）。
   最初の反復は JIT コンパイルを含むので、反復の少ない smoke では多めに出る。
3. 手で数を入れるなら `--seconds-per-iteration 0.5 --dph 0.45`。
4. 納得したら `configs/train_full.json` の `estimate.seconds_per_iteration` を実測値に直す。

   実測（2026-11-17、RTX 4090 1台、8192 envs、学習用カットモデル、128-64）:
   - smoke（10 反復）: 1回目 183 秒、コンパイルのキャッシュが効いた2回目 40.8 秒。
   - 本番（1000 反復）: **550 秒＝約 0.55 秒/反復**（コンパイル込み）。学習中の GPU 使用率は 96%。
   - **10 反復の smoke は、時間のほとんどがコンパイルなので、本番の見積りには使えない**
     （smoke から出した 4.1 秒/反復は、実際の約7倍だった）。見積りには、本番の最初の数十反復の
     `progress.json` を使う。
   - 設定の `seconds_per_iteration` は 0.6 にしてある。

## 4. 借りる（課金開始）

```bash
# まず dry-run（何も送らない。送るはずの PUT の中身が出る）
.venv/bin/python -m vastai create --config configs/train_smoke.json --offer <id>
# 本当に借りる
.venv/bin/python -m vastai create --config configs/train_smoke.json --offer <id> --yes
```

作成要求がタイムアウトしたら、再送する前にコンソールで作られていないか確かめる
（応答が無くても作られていることがある）。

作ったインスタンスには label `cadex-q8-<種類>-<日時>` が付く（dry-run でも送る label が見える）。
コンソールでもこの label で、このプロジェクトのものと見分けられる。

## 5. 接続先を設定し、箱を確かめる

```bash
.venv/bin/python -m vastai instances
#   12345678  running  $0.420/h  q8-smoke
#       VAST_SSH_HOST=ssh5.vast.ai VAST_SSH_PORT=41234  （プロキシ）
#       VAST_SSH_HOST=203.0.113.7 VAST_SSH_PORT=40022  （直結）
```

表示された値を `robot-project/.env` に書く（`VAST_SSH_USER` は既定 `root`）。
直結の方が rsync は速い。ホスト鍵は初回だけ受け入れ（`accept-new`）、変わったら拒む。

```bash
.venv/bin/python -m vastai check --config configs/train_smoke.json
```

GPU 名・ドライバ、ピン留めの版、`jax.default_backend()`、イメージの学習器の
sha256 を一度に見る。`backend gpu` が出ない、または学習器が手元と違えば失敗する。

## 6. 起動

```bash
# 実行するコマンドを確かめる（ssh も rsync も走らない）
.venv/bin/python -m vastai launch --config configs/train_smoke.json --dry-run
# 起動
.venv/bin/python -m vastai launch --config configs/train_smoke.json
#   起動しました: run smoke-20261117-153000（pid 4242、/workspace/runs/smoke-20261117-153000）
```

やること:

1. 設定の `trainer` を本物の `cadex_train.arguments()` に通して検証する
   （綴り違いは省略形一致も含めて弾く）。
2. タスクのモデルの sha256 がタスクの記録と合うか確かめる。
3. タスクとモデルを `/workspace/runs/<run_id>/` に並べて rsync で送る。
4. `nohup setsid` で切り離して起動する。学習プロセス自身の pid を `train.pid` に
   書くので、ssh が切れても、Mac を閉じても学習は続く。
5. 手元の `artifacts/runs/vastai/<run_id>/run.json` に記録を残す（以後の
   `status`/`pull`/`resume`/`stop` はこれを読む。消さないこと）。

複数のインスタンスで並べるときは、起動のたびに接続先を環境変数で渡す（環境変数は `.env` に勝つ）。
`status`・`pull`・`stop` は、その回を起動した接続先（`run.json` の `host` と `port`）に送るので、
`.env` がどの台を指していても取り違えない。

```bash
VAST_SSH_HOST=<[このプロジェクト] の行のホスト> VAST_SSH_PORT=<ポート> \
  .venv/bin/python -m vastai launch --config configs/train_full.json --seed 1 --run-id full-base-s1
```

同じ設定を学習の乱数だけ変えて複数回回すときは `--seed` を付ける（設定の `trainer.seed` を
上書きする。run_id は区別できる名前を `--run-id` で付ける）。

```bash
.venv/bin/python -m vastai launch --config configs/train_full_leg120.json --seed 1 --run-id full-leg120-s1
```

起動時に `--out <run_dir>/<policy_name>.cxpolicy` と `--progress <run_dir>/progress.json`
を付け、`JAX_PLATFORMS` と `CADEX_JAX_CACHE` を明示的に渡す（vast.ai の SSH では
イメージの `ENV` が引き継がれないことがあるため）。

## 7. 監視

```bash
.venv/bin/python -m vastai status <run_id>            # 1 回
.venv/bin/python -m vastai status <run_id> --watch    # 終わるまで 30 秒ごと
```

`progress.json`（学習器が毎反復書き直す）だけを読み、状態・反復・報酬/step・
エピソード長・探索幅 σ・最良とその反復・経過・残り・device・チェックポイント数を
1 行で出す。学習器の stderr は解析しない。プロセスが居なくなれば知らせる。

曲線を表やグラフで見る:

```bash
.venv/bin/python -m vastai metrics artifacts/runs/vastai/<run_id>/progress.json \
    --csv artifacts/runs/vastai/<run_id>/metrics.csv --tensorboard artifacts/runs/vastai/<run_id>/tb
tensorboard --logdir artifacts/runs/vastai   # tensorboard を入れていれば
```

CSV の列は `iteration, reward_per_step, loss, episode_steps, action_std` と、
あれば地形レベルやゲートなどの `*_curve`。学習器は曲線を最大点数に間引くので、
長い学習では全反復は残らない（最初と最後は必ず残る）。欠けた点は空欄。
TensorBoard のイベントファイルは標準ライブラリだけで書く（tensorboard は読む時だけ要る）。

## 8. 回収と検証

```bash
.venv/bin/python -m vastai pull <run_id>
```

`progress.json`・`stdout.log`・`train.log`・すべての `.cxpolicy`・`view/` を
rsync で `artifacts/runs/vastai/<run_id>/` に持ち帰り、`metrics.csv` を書き、
各ポリシーの sha256 を **三者で突き合わせる**:

- 手元のファイル（`shasum -a 256` と同じ値を hashlib で計算）
- リモートの `sha256sum`
- 学習器の記録（チェックポイントは `progress.json` の `checkpoints[].sha256`、
  最終ポリシーは `stdout.log` 最後の行の受領書 JSON の `sha256`）

一つでも違えば失敗する。受領書の `device` が `gpu` でなければ（CUDA 版 jax が
読めず CPU に落ちた）それも失敗にする（`--allow-cpu` で受け入れる）。
手で確かめるなら:

```bash
shasum -a 256 artifacts/runs/vastai/<run_id>/walk.cxpolicy
```

チェックポイントの選び方は Cadex の方針どおり、学習器の報酬ではなく
`cadex evaluate` の判定で選ぶ（`training/SETUP.md`「Choosing among the checkpoints」）。

## 9. 中断からの再開

```bash
.venv/bin/python -m vastai resume --config configs/train_full.json <中断した run_id>
# 元のインスタンスが消えていて、手元に回収済みなら
.venv/bin/python -m vastai resume --config configs/train_full.json <run_id> --no-pull
```

1. （`--no-pull` でなければ）回収し直して検証する。
2. 番号付きチェックポイント（`<名前>.NNNNNN.cxpolicy`）の **番号が最大のもの** を選ぶ
   （`best` は選ばない）。`progress.json` が無ければファイル名から選ぶ。
3. その sha256 を `progress.json` の記録と照合し、壊れていれば送らない。
4. 新しい run_id で、チェックポイントを `warm/` に送り、
   `--init-from <run_dir>/warm/<file> --iterations <残り> --critic-warmup N --actor-ramp M`
   で起動する（N, M は設定の `resume` 節）。

注意:

- **厳密な再開ではない。** `--init-from` が運ぶのは actor・探索幅・観測の正規化だけで、
  critic と optimizer は新しく始まる（Cadex ADR-740）。だから critic を先に温める
  `--critic-warmup` を付ける。`--actor-ramp`（ADR-984）は actor の更新を徐々に戻す。
- 残り反復 = 元の反復数 − チェックポイントの番号。新しい run のチェックポイントの
  番号は 0 から数え直す。
- `--init-from` はタスクとモデルの digest が同じであることを要求する。
  タスクを作り直した後の再開はできない（カリキュラムの段は Cadex の
  `--init-from-task-change` の仕事で、この道具は扱わない）。
- 中断の多い（interruptible/spot）オファーは使っていない。使うなら
  `checkpoint_every` を短くする。

## 10. 止める・終える・課金の停止を確かめる

| 操作 | 何が止まるか | 課金 |
|---|---|---|
| `python -m vastai stop <run_id>` | 学習プロセスだけ（TERM、止まったことを確かめる） | インスタンスは **課金が続く** |
| vast.ai コンソールの Stop | インスタンス（GPU） | GPU 代は止まる。データは保持される（ドキュメントの記載）。**停止中もストレージ料金がかかる可能性がある**（未確認） |
| `python -m vastai destroy <instance_id> --yes` | インスタンスとディスク | 止まる（ディスクの中身も消える） |

終わる時の手順:

1. `pull` で回収し、検証が通ったことを確かめる（**destroy の前に**）。
2. `.venv/bin/python -m vastai destroy <instance_id>`（dry-run で対象を確認）→ `--yes`。
3. `.venv/bin/python -m vastai instances` で一覧から消えたことを確かめる。
4. vast.ai コンソールの Billing でも課金が止まったことを確かめる。

停止中もストレージ料金がかかる可能性があるので、**学習が終わって回収したら Stop ではなく
destroy を推奨する**。この道具に自動停止（時間上限で落とす仕組み）は無い。放置すれば課金は続く。

## 11. GPU / CPU の切り替え

設定の `device`（`gpu` / `cpu`）か、コマンドの `--device` で切り替える。

- `gpu` → `JAX_PLATFORMS=cuda`、`cpu` → `JAX_PLATFORMS=cpu` を起動時に渡す。
  `cuda` を明示するので、CUDA 版 jax が読めなければ黙って CPU に落ちずに失敗する。
- `device_overrides.<device>` で、その装置の時だけ `trainer` や `env` を上書きする
  （例: CPU では `envs` を 64 に、`minibatches` を割り切れる 8 に）。
- 手元の Mac で CPU のまま動作確認する時は、同じ設定から引数を作って直接回せる:

  ```bash
  JAX_PLATFORMS=cpu .venv/bin/python ../training/cadex_train.py robot/mjcf/base/walk-task.json \
      --out /tmp/walk.cxpolicy --iterations 4 --envs 64 --minibatches 8 --checkpoint-every 2
  ```

## 12. 学習設定（configs/train_*.json）

```jsonc
{
  "name": "smoke",                       // run_id の頭
  "bundle": "robot/mjcf/base/walk-task.json",  // モデルはタスクと同じ場所か親の親
  "policy_name": "walk",                 // walk.cxpolicy / walk.000005.cxpolicy
  "device": "gpu",
  "env": {"CADEX_JAX_CACHE": "/workspace/cache/jax"},
  "trainer": { "iterations": 10, "envs": 8192, "minibatches": 32, "checkpoint_every": 5, ... },
  "device_overrides": {"cpu": {"trainer": {"envs": 64}}},
  "resume": {"critic_warmup": 2, "actor_ramp": 0},
  "offer": { "gpu_name": ["RTX 4090", "RTX 5090"], "max_dph": 0.6, ... },
  "instance": {"image": "ghcr.io/ayutaz/cadex-train:sha-e7f4d9d", "disk_gb": 30, "label": "q8-smoke"},
  "estimate": {"seconds_per_iteration": 3.0, "setup_minutes": 15, "margin": 1.3},
  "remote": {"work_dir": "/workspace/runs", "trainer_root": "/opt/cadex", "python": "/opt/venv/bin/python"}
}
```

`trainer` のキーは `cadex_train.py` のフラグそのもの（`_` を `-` に直して `--` を
付ける。`true` はフラグだけ、`false`/`null` は省く、配列は並べる）。
`bundle`・`out`・`progress`・`init_from` 系はこの道具が決めるので書けない。
フラグの一覧は `python ../training/cadex_train.py --help`。

## 13. 注意点

- **GPU に乗ったかは受領書で確かめる。** CPU に落ちても学習は「成功」し、時間と
  お金だけが溶ける。`check` と `pull` がこれを失敗として扱う。
- **イメージの学習器と手元の学習器を揃える。** 古い学習器は新しいタスクの欄を
  黙って無視し、それでもタスクのアルゴリズム名をポリシーに記録する
  （`training/remote_train.sh` の注記）。
- **mujoco の版はタスクの `mujoco_version`（3.10.0）と一致させる。** Dockerfile が
  ピン留めを確かめる。
- run_id は使い捨て。同じ run_id で起動し直さない。
- `artifacts/runs/vastai/` は `.gitignore` 済み（`artifacts/runs/`）。残したい
  ポリシーは Cadex の手順（`put_asset` / `assembly.policy` に sha256 を書く）で取り込む。
- vast.ai の REST API は公式ドキュメントの調査に合わせてある（検索 `POST /api/v0/bundles`、
  作成 `PUT /api/v0/asks/{id}`、一覧 `GET /api/v1/instances`、破棄
  `DELETE /api/v0/instances/{id}`。認証は `Authorization: Bearer`）。ただし
  **実アカウントでは一度も試していない**。次の点は未確認で、`vastai/offers.py` に
  `UNCONFIRMED` と注記してある:
  - 作成が `PUT /asks/{id}` か `POST /asks/{id}/` か。各パスの末尾スラッシュの要否。
  - 検索の `type` が `"on-demand"` か `"ondemand"` か（設定の `offer.type` で変えられる）。
  - `gpu_name` を空白区切り（`"RTX 4090"`）で送ってよいか。
  - `runtype` の綴り（既定 `"ssh_direc ssh_proxy"`。設定の `instance.runtype` で変えられる）。
  - 直結の SSH ポートが `ports["22/tcp"][0].HostPort` に入るか（`instances` は
    プロキシと直結の両方を表示する）。
  - 停止中のストレージ課金の有無。
  最初は必ず dry-run と `instances` の表示で中身を確かめる。
