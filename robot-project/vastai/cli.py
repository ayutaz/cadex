"""``python -m vastai <command>``: vast.ai で cadex_train.py を回す入口。

課金が始まる ``create`` と、ディスクごと消す ``destroy`` は、``--yes`` と
``VAST_API_KEY`` の両方が揃った時だけ実行する。それ以外は計画を表示して終わる。
手順は docs/training-vastai.md。
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence, TextIO

from . import config as cfg
from . import metrics, offers, remote

ROOT = Path(__file__).resolve().parents[1]           # robot-project/
TRAINER = ROOT.parent / "training" / "cadex_train.py"  # Cadex 本体の学習器
LOCAL_RUNS = ROOT / "artifacts" / "runs" / "vastai"
ENV_FILE = ROOT / ".env"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m vastai",
        description="vast.ai の GPU インスタンスで Cadex の cadex_train.py（Brax PPO on MJX）を回す。"
                    " create / destroy 以外は課金に関わる操作をしない。")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def with_config(p: argparse.ArgumentParser) -> None:
        p.add_argument("--config", required=True, help="configs/train_*.json")
        p.add_argument("--device", choices=sorted(cfg.DEVICES), default=None,
                       help="設定の device を上書き（gpu→JAX_PLATFORMS=cuda, cpu→cpu）")

    p = sub.add_parser("offers", help="オファーを検索し、GPU 種別・$/h・推定総額を表示（読むだけ）")
    with_config(p)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--seconds-per-iteration", type=float, default=None)

    p = sub.add_parser("estimate", help="学習時間と金額の見積り（通信しない）")
    with_config(p)
    p.add_argument("--seconds-per-iteration", type=float, default=None,
                   help="一反復の秒数（既定は設定の estimate.seconds_per_iteration）")
    p.add_argument("--from-progress", default=None,
                   help="smoke などの progress.json から一反復の秒数を実測値で取る")
    p.add_argument("--dph", type=float, default=None, help="時間単価 $/h（offers の表から）")

    p = sub.add_parser("create", help="オファーを借りる（--yes と VAST_API_KEY が無ければ dry-run）")
    with_config(p)
    p.add_argument("--offer", type=int, required=True, help="offers の表の id")
    p.add_argument("--image", default=None, help="設定の instance.image を上書き")
    p.add_argument("--yes", action="store_true", help="本当に作る（課金が始まる）")

    sub.add_parser("instances", help="自分のインスタンスと SSH 接続先の一覧（VAST_API_KEY が要る）")

    p = sub.add_parser("destroy", help="インスタンスを破棄して課金を止める（--yes が無ければ dry-run）")
    p.add_argument("instance_id", type=int)
    p.add_argument("--yes", action="store_true")

    p = sub.add_parser("check", help="SSH 先の GPU・CUDA 版 jax・依存の版・学習器の sha256 を確認")
    with_config(p)

    p = sub.add_parser("launch", help="タスクのペアを送り、cadex_train.py を切り離して起動")
    with_config(p)
    p.add_argument("--run-id", default=None)
    p.add_argument("--seed", type=int, default=None,
                   help="学習の乱数（設定の trainer.seed を上書き）。同じ設定を乱数違いで複数回回すとき")
    p.add_argument("--dry-run", action="store_true", help="実行するコマンドを表示するだけ")

    p = sub.add_parser("status", help="progress.json を取ってきて 1 行で表示")
    p.add_argument("run_id")
    p.add_argument("--watch", action="store_true", help="終わるまで繰り返す")
    p.add_argument("--interval", type=float, default=30.0)

    p = sub.add_parser("pull", help="チェックポイントと最終ポリシーを回収し sha256 で検証")
    p.add_argument("run_id")
    p.add_argument("--allow-cpu", action="store_true", help="CPU で走った学習も受け入れる")

    p = sub.add_parser("resume", help="最新チェックポイントから warm start で続きを起動")
    with_config(p)
    p.add_argument("run_id", help="中断した run の id")
    p.add_argument("--run-id", dest="new_run_id", default=None, help="新しい run の id")
    p.add_argument("--no-pull", action="store_true",
                   help="回収し直さず、手元のチェックポイントを使う（元のインスタンスが無い時）")
    p.add_argument("--dry-run", action="store_true")

    p = sub.add_parser("stop", help="学習プロセスに TERM を送って止める（インスタンスは止めない）")
    p.add_argument("run_id")

    p = sub.add_parser("metrics", help="progress.json を CSV（と TensorBoard）に変換")
    p.add_argument("progress")
    p.add_argument("--csv", required=True)
    p.add_argument("--tensorboard", default=None, help="イベントファイルを書くディレクトリ")
    return parser


def _load(args: argparse.Namespace) -> dict[str, Any]:
    loaded = cfg.load_config(args.config, device=args.device)
    bundle = Path(loaded["bundle"])
    if not bundle.is_absolute():
        loaded["bundle"] = str(ROOT / bundle)
    return loaded


def _validate(loaded: Mapping[str, Any], out: TextIO) -> None:
    if TRAINER.exists():
        cfg.validate_with_trainer(cfg.trainer_argv(loaded["trainer"]), TRAINER)
    else:
        print(f"注意: {TRAINER} が無いので、フラグを本物の cadex_train.py で確かめていません", file=out)


def _hours(loaded: Mapping[str, Any], seconds: float | None = None) -> tuple[float, float]:
    estimate = loaded.get("estimate", {})
    spi = seconds if seconds is not None else float(estimate.get("seconds_per_iteration", 2.0))
    hours = offers.estimate_hours(
        iterations=int(loaded["trainer"]["iterations"]), seconds_per_iteration=spi,
        setup_minutes=float(estimate.get("setup_minutes", 15.0)),
        margin=float(estimate.get("margin", 1.3)))
    return hours, spi


def _status_line(progress: Mapping[str, Any]) -> str:
    def number(key: str, fmt: str) -> str:
        value = progress.get(key)
        return fmt % value if isinstance(value, (int, float)) else "-"

    return (f"{progress.get('state', '-'):<8} {int(progress.get('iteration', -1)) + 1:>5}/"
            f"{progress.get('total', 0):<5} 報酬/step {number('reward_per_step', '%+.5g')}  "
            f"episode {number('episode_steps', '%.1f')}  sigma {number('action_std', '%.3f')}  "
            f"best {number('best_reward_per_step', '%+.5g')}@{progress.get('best_iteration', -1)}  "
            f"経過 {number('wall_time_s', '%.0f')}s  残り {number('eta_s', '%.0f')}s  "
            f"device {progress.get('device') or '-'}  "
            f"checkpoint {len(progress.get('checkpoints') or [])}")


def main(argv: Sequence[str] | None = None, *, environ: Mapping[str, str] | None = None,
         opener: Any = None, runner: Any = None, stdout: TextIO | None = None,
         env_file: str | Path | None = None) -> int:
    out = stdout or sys.stdout
    environ = os.environ if environ is None else environ
    env_file = ENV_FILE if env_file is None else Path(env_file)
    args = _parser().parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        return _dispatch(args, environ=environ, opener=opener, runner=runner, out=out,
                         env_file=env_file)
    except cfg.ConfigError as error:
        print(f"設定エラー: {error}", file=out)
        return 2
    except remote.VerificationError as error:
        print(f"検証エラー: {error}", file=out)
        return 3
    except (remote.RemoteError, offers.VastError) as error:
        print(f"エラー: {error}", file=out)
        return 1


def _session(environ, env_file, runner) -> remote.Session:
    target = cfg.ssh_settings(environ, env_file)
    return remote.Session(target, runner=runner or remote.run_subprocess, local_root=LOCAL_RUNS)


def _dispatch(args, *, environ, opener, runner, out, env_file) -> int:
    key = cfg.api_key(environ, env_file)
    command = args.command

    if command == "offers":
        loaded = _load(args)
        offer = dict(loaded.get("offer", {}))
        if args.limit:
            offer["limit"] = args.limit
        hours, spi = _hours(loaded, args.seconds_per_iteration)
        disk = float(loaded.get("instance", {}).get("disk_gb", 40))
        found = offers.search_offers(offers.build_query(offer), api_key=key, opener=opener)
        print(f"設定 {loaded['name']}: {loaded['trainer']['iterations']} 反復 × {spi:.2f} 秒/反復"
              f" → 借りる時間の推定 {hours:.2f} 時間", file=out)
        print(offers.format_offers(found, hours=hours, disk_gb=disk), file=out)
        print("借りるには: python -m vastai create --config "
              f"{args.config} --offer <id>（--yes で実行）", file=out)
        return 0

    if command == "estimate":
        loaded = _load(args)
        seconds, note = args.seconds_per_iteration, ""
        if args.from_progress:
            progress = metrics.load_progress(args.from_progress)
            measured = offers.seconds_per_iteration(progress)
            if measured is None:
                raise cfg.ConfigError(f"{args.from_progress} にまだ反復の記録がありません")
            print(f"実測: {measured:.2f} 秒/反復（{args.from_progress}、最初の JIT を含む）", file=out)
            seconds = measured
            envs = offers.measured_envs(progress)
            target_envs = int(loaded["trainer"].get("envs", 256))
            if envs and envs != target_envs:
                seconds = measured * target_envs / envs
                note = (f"（envs {envs}→{target_envs} に比例で拡大。GPU が飽和するまでは"
                        "これより速いので上限の目安）")
        hours, spi = _hours(loaded, seconds)
        estimate = loaded.get("estimate", {})
        print(f"設定 {loaded['name']}（device {loaded['device']}）: "
              f"{loaded['trainer']['iterations']} 反復 × {spi:.2f} 秒/反復{note}", file=out)
        print(f"借りる時間の推定: {hours:.2f} 時間（準備 {estimate.get('setup_minutes', 15)} 分、"
              f"余裕 ×{estimate.get('margin', 1.3)} を含む）", file=out)
        if args.dph is not None:
            print(f"推定総額: ${args.dph * hours:.2f}（${args.dph:.3f}/h、転送料と停止中のディスク代は別）",
                  file=out)
        else:
            print("金額は --dph <offers の $/h> を付けると出ます", file=out)
        return 0

    if command == "create":
        loaded = _load(args)
        instance = dict(loaded.get("instance", {}))
        if args.image:
            instance["image"] = args.image
        result = offers.create_instance(args.offer, instance, api_key=key, yes=args.yes,
                                        opener=opener)
        if not result["executed"]:
            print(result["reason"], file=out)
            print("送るはずの内容:", file=out)
            print(json.dumps({"method": result["method"], "url": result["url"],
                              "body": result["request"]}, indent=2, ensure_ascii=False), file=out)
            return 0 if result["reason"].startswith("dry-run") else 2
        print(f"インスタンス {result['instance_id']} を作りました。課金が始まっています。", file=out)
        print("SSH の接続先は: python -m vastai instances", file=out)
        print(f"終わったら必ず: python -m vastai destroy {result['instance_id']} --yes", file=out)
        return 0

    if command == "instances":
        rows = offers.list_instances(api_key=key, opener=opener)
        if not rows:
            print("インスタンスはありません（課金されているものは無い）", file=out)
        for row in rows:
            proxy = row["proxy_ssh"]
            direct = row["direct_ssh"]
            owned = offers.is_owned(row)
            mark = "このプロジェクト" if owned else "他の作業：触らない"
            print(f"{row.get('id')}  {row.get('actual_status', '-'):<10} "
                  f"${float(row.get('dph_total') or 0):.3f}/h  {row.get('label') or ''}  [{mark}]", file=out)
            if not owned:
                continue  # 他の作業の接続先は出さない
            if proxy:
                print(f"    VAST_SSH_HOST={proxy[0]} VAST_SSH_PORT={proxy[1]}  （プロキシ）", file=out)
            if direct:
                print(f"    VAST_SSH_HOST={direct[0]} VAST_SSH_PORT={direct[1]}  （直結）", file=out)
        return 0

    if command == "destroy":
        result = offers.destroy_instance(args.instance_id, api_key=key, yes=args.yes,
                                         opener=opener)
        if not result["executed"]:
            print(result["reason"], file=out)
            print(f"送るはずの内容: {result['method']} {result['url']}", file=out)
            return 0 if result["reason"].startswith("dry-run") else 2
        print(f"インスタンス {args.instance_id} を破棄しました。"
              "`python -m vastai instances` で一覧から消えたことを確かめてください", file=out)
        return 0

    if command == "metrics":
        progress = metrics.load_progress(args.progress)
        path = metrics.write_csv(progress, args.csv)
        print(f"CSV: {path}", file=out)
        if args.tensorboard:
            event = metrics.write_tensorboard(progress, args.tensorboard)
            print(f"TensorBoard: {event}（tensorboard --logdir {args.tensorboard}）", file=out)
        return 0

    # ここから下は SSH を使う
    session = _session(environ, env_file, runner)

    if command == "check":
        loaded = _load(args)
        report = session.check(loaded["remote"]["trainer_root"], loaded["remote"]["python"])
        print(report.rstrip(), file=out)
        mine = remote.sha256_file(TRAINER) if TRAINER.exists() else None
        theirs = next((line.split()[1] for line in report.splitlines()
                       if line.startswith("trainer ")), None)
        problems = []
        if mine and theirs != mine:
            problems.append(f"リモートの cadex_train.py（{str(theirs)[:12]}）が手元（{mine[:12]}）と"
                            "違います。イメージを作り直してください")
        if loaded["device"] == "gpu" and "backend gpu" not in report:
            problems.append("jax が GPU を使えていません（backend gpu が出ない）")
        for problem in problems:
            print(f"問題: {problem}", file=out)
        return 1 if problems else 0

    if command == "launch":
        loaded = _load(args)
        _validate(loaded, out)
        extra = [] if args.seed is None else ["--seed", str(args.seed)]
        result = session.launch(loaded, run_id=args.run_id, extra_argv=extra,
                                dry_run=args.dry_run)
        return _report_launch(result, args.dry_run, out)

    if command == "resume":
        loaded = _load(args)
        _validate(loaded, out)
        result = session.resume(args.run_id, loaded, run_id=args.new_run_id,
                                pull_first=not args.no_pull, dry_run=args.dry_run)
        source = result["resumed_from"]
        print(f"{source['run_id']} の {source['checkpoint']}（{source['completed']} 反復済み）"
              "から再開します", file=out)
        return _report_launch(result, args.dry_run, out)

    if command == "status":
        while True:
            progress = session.progress(args.run_id)
            print(_status_line(progress), file=out)
            if progress.get("error"):
                print(f"    error: {progress['error']}", file=out)
            if progress.get("warning"):
                print(f"    warning: {progress['warning']}", file=out)
            state = progress.get("state")
            if not args.watch or state in ("done", "failed"):
                break
            if not session.alive(args.run_id):
                # 進捗を読んだ直後に学習が終わってプロセスが消えることがある。読み直して確かめる
                progress = session.progress(args.run_id)
                state = progress.get("state")
                if state in ("done", "failed"):
                    print(_status_line(progress), file=out)
                    break
                print("学習プロセスが居ません（落ちたか止められた）。"
                      f"python -m vastai pull {args.run_id} で train.log を回収して確かめる", file=out)
                return 1
            time.sleep(args.interval)
        if state == "done":
            print(f"終わりました。回収: python -m vastai pull {args.run_id}", file=out)
        return 1 if state == "failed" else 0

    if command == "pull":
        report = session.pull(args.run_id, allow_cpu=args.allow_cpu)
        directory = session.record_dir(args.run_id)
        print(f"回収先: {directory}", file=out)
        for row in report:
            print(f"    {row['file']:<32} {row['local']}  OK", file=out)
        print("sha256 は手元・リモート・学習器の記録で一致しました（shasum -a 256 と同じ値）", file=out)
        return 0

    if command == "stop":
        answer = session.stop(args.run_id)
        print({"stopped": f"{args.run_id} を止めました。",
               "gone": f"{args.run_id} はもう動いていません。"}.get(answer, answer), file=out)
        print("インスタンス自体はまだ課金中です。回収後に destroy してください", file=out)
        return 0

    raise AssertionError(command)


def _report_launch(result: Mapping[str, Any], dry_run: bool, out: TextIO) -> int:
    if dry_run:
        print(f"dry-run: run {result['run_id']} で実行するコマンド:", file=out)
        for command in result["commands"]:
            print("  " + " ".join(shlex.quote(str(word)) for word in command), file=out)
        return 0
    print(f"起動しました: run {result['run_id']}（pid {result['pid']}、{result['remote_dir']}）", file=out)
    print(f"進捗: python -m vastai status {result['run_id']} --watch", file=out)
    print(f"回収: python -m vastai pull {result['run_id']}", file=out)
    return 0
