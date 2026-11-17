"""成果物にマシン固有の絶対パスを書き込まない（公開リポジトリにユーザー名やフォルダ構成を出さない）。

Cadex や学習器は、受け取ったパスを絶対パスにして記録する。そこで、成果物を robot-project に
書く・写す箇所で、既知の根（robot-project、cadex、~/cadex-projects、ホーム）を置き換え文字にする。
"""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from q8.portable import (
    default_roots,
    json_from_stdout,
    portable,
    portable_text,
    scrub_file,
    scrub_tree,
    write_json,
)

HOME = Path("/home/u")
ROOTS = [
    (HOME / "work" / "cadex" / "robot-project", "<robot-project>"),
    (HOME / "work" / "cadex", "<cadex>"),
    (HOME / "cadex-projects", "~/cadex-projects"),
    (HOME, "~"),
]
MAGIC = b"CXPOLICY1\n"


def _cxpolicy(header: dict, weights: list[float]) -> bytes:
    payload = json.dumps(header, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
    return MAGIC + len(payload).to_bytes(8, "little") + payload + np.asarray(weights, dtype="<f4").tobytes()


def test_the_longest_root_wins():
    assert portable_text("/home/u/work/cadex/robot-project/robot/x.xml", ROOTS) == "<robot-project>/robot/x.xml"
    assert portable_text("/home/u/work/cadex/src/Mod/cadex", ROOTS) == "<cadex>/src/Mod/cadex"
    assert portable_text("/home/u/cadex-projects/quadruped8", ROOTS) == "~/cadex-projects/quadruped8"
    assert portable_text("/home/u/.cache/x", ROOTS) == "~/.cache/x"


def test_a_root_must_end_at_a_path_boundary():
    # /home/u/work/cadex-old は <cadex> ではない（ホームだけが置き換わる）
    assert portable_text("/home/u/work/cadex-old/a", ROOTS) == "~/work/cadex-old/a"
    assert portable_text("/home/user2/a", ROOTS) == "/home/user2/a"


def test_root_alone_and_inside_a_sentence():
    assert portable_text("/home/u/work/cadex", ROOTS) == "<cadex>"
    assert portable_text('cd "/home/u/work/cadex/robot-project" && ls', ROOTS) == 'cd "<robot-project>" && ls'


def test_nested_keys_and_values_are_rewritten():
    obj = {"/home/u/cadex-projects/a": ["/home/u/work/cadex/x", 3, None, {"p": "/home/u/y"}]}
    assert portable(obj, ROOTS) == {"~/cadex-projects/a": ["<cadex>/x", 3, None, {"p": "~/y"}]}


def test_write_json_writes_no_root(tmp_path):
    out = write_json(tmp_path / "r.json", {"task": "/home/u/work/cadex/robot-project/t.json", "n": 1.5},
                     roots=ROOTS, indent=2)
    assert json.loads(out.read_text()) == {"task": "<robot-project>/t.json", "n": 1.5}


def test_write_json_passes_dump_options(tmp_path):
    out = write_json(tmp_path / "r.json", {"x": {1, 2}}, roots=ROOTS, default=sorted)
    assert json.loads(out.read_text()) == {"x": [1, 2]}


def test_scrub_cxpolicy_rewrites_the_header_and_keeps_the_weights(tmp_path):
    p = tmp_path / "walk.cxpolicy"
    w = [0.5, -1.25, 3.0]
    p.write_bytes(_cxpolicy({"training": {"cad_model": "/home/u/work/cadex/robot-project/m.xml"}}, w))
    assert scrub_file(p, ROOTS) is True
    blob = p.read_bytes()
    n = int.from_bytes(blob[len(MAGIC):len(MAGIC) + 8], "little")
    header = json.loads(blob[len(MAGIC) + 8:len(MAGIC) + 8 + n])
    assert header == {"training": {"cad_model": "<robot-project>/m.xml"}}
    assert np.frombuffer(blob[len(MAGIC) + 8 + n:], dtype="<f4").tolist() == w


def test_scrub_text_file_and_report_unchanged(tmp_path):
    log = tmp_path / "stderr.log"
    log.write_text("loading /home/u/work/cadex/robot-project/a.json\n")
    assert scrub_file(log, ROOTS) is True
    assert log.read_text() == "loading <robot-project>/a.json\n"
    assert scrub_file(log, ROOTS) is False


def test_scrub_tree_leaves_meshes_and_videos_alone(tmp_path):
    (tmp_path / "a.json").write_text('{"p": "/home/u/x"}')
    stl = tmp_path / "m.stl"
    stl.write_bytes(b"\x00/home/u/x\x00")
    changed = scrub_tree(tmp_path, ROOTS)
    assert changed == [tmp_path / "a.json"]
    assert stl.read_bytes() == b"\x00/home/u/x\x00"


def test_json_from_stdout_skips_warning_lines():
    out = "Failed to import warp: No module named 'warp'\n{\"best_iteration\": 9,\n \"bytes\": 1}\n"
    assert json_from_stdout(out) == {"best_iteration": 9, "bytes": 1}


def test_json_from_stdout_refuses_text_without_json():
    with pytest.raises(ValueError):
        json_from_stdout("only a warning\n")


def test_default_roots_cover_this_checkout_and_the_home():
    from q8.paths import CADEX_PROJECTS, CADEX_REPO, PROJECT

    roots = dict((str(r), s) for r, s in default_roots())
    assert roots[str(PROJECT)] == "<robot-project>"
    assert roots[str(CADEX_REPO)] == "<cadex>"
    assert roots[str(CADEX_PROJECTS)] == "~/cadex-projects"
    assert roots[str(Path.home())] == "~"


def test_archive_run_output_has_no_root(tmp_path):
    from q8.training import archive_run

    run = tmp_path / "run"
    (run / "eval-final").mkdir(parents=True)
    (run / "progress.json").write_text('{"task": "/home/u/work/cadex/robot-project/t.json"}')
    (run / "receipt.json").write_text("Failed to import warp\n{\"out\": \"/home/u/work/cadex/robot-project/w\"}\n")
    (run / "eval-final" / "evaluation.json").write_text('{"freecadcmd": "/home/u/work/cadex/.pixi/bin/F"}')
    (run / "walk.cxpolicy").write_bytes(_cxpolicy({"progress": "/home/u/work/cadex/robot-project/p.json"}, [1.0]))
    dest = archive_run(run, tmp_path / "dest", roots=ROOTS)
    for f in dest.rglob("*"):
        if f.is_file():
            assert b"/home/u" not in f.read_bytes(), f
    assert json.loads((dest / "receipt.json").read_text()) == {"out": "<robot-project>/w"}


def test_tracked_files_have_no_home_path():
    """リポジトリに入っている robot-project のファイルに、このマシンのホームのパスが無い。"""
    from q8.paths import PROJECT

    listed = subprocess.run(["git", "ls-files", "-z", "."], cwd=PROJECT, capture_output=True)
    if listed.returncode != 0:
        pytest.skip("git の作業ツリーではない")
    home = str(Path.home()).encode()
    bad = [name for name in listed.stdout.decode().split("\0")
           if name and (PROJECT / name).is_file() and home in (PROJECT / name).read_bytes()]
    assert bad == [], f"ホームの絶対パスを含む: {bad}（q8.portable で置き換えてから追加する）"


# ---- 成果物を書く箇所が、このマシンのホームを書き込まない ----

HOME_STR = str(Path.home())


class _Done:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def _no_home(path: Path):
    for f in ([path] if path.is_file() else path.rglob("*")):
        if f.is_file():
            assert HOME_STR.encode() not in f.read_bytes(), f


def test_save_receipt_keeps_only_the_json(tmp_path):
    from q8.training import save_receipt

    out = save_receipt(f"Failed to import warp\n{{\"out\": \"{HOME_STR}/w.cxpolicy\"}}\n", tmp_path / "receipt.json")
    assert json.loads(out.read_text()) == {"out": "~/w.cxpolicy"}


def test_make_training_cut_writes_no_home(tmp_path, monkeypatch):
    import q8.training as tr

    def fake_run(cmd, **kw):
        if "cut" in cmd:
            return _Done(stdout=json.dumps({"ok": True, "problems": [], "source": f"{HOME_STR}/p"}))
        relay = Path(cmd[cmd.index("--out") + 1])
        (relay / "outputs").mkdir(parents=True)
        (relay / "outputs" / "walk-task.json").write_text(json.dumps({"model": f"{HOME_STR}/cut.xml"}))
        (relay / "relay.json").write_text(json.dumps({"task": f"{HOME_STR}/t.json"}))
        return _Done()

    monkeypatch.setattr(tr.subprocess, "run", fake_run)
    tr.make_training_cut(tmp_path / "proj", tmp_path / "walk-task.json", tmp_path / "mjcf")
    _no_home(tmp_path / "mjcf")
    assert json.loads((tmp_path / "mjcf" / "cut" / "cut_report.json").read_text())["source"] == "~/p"


def test_cadex_evaluate_output_is_scrubbed(tmp_path, monkeypatch):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "evaluation"))
    import evaluate_variant as ev

    def fake_run(cmd, **kw):
        out = Path(cmd[cmd.index("--out") + 1])
        out.mkdir(parents=True, exist_ok=True)
        (out / "evaluation.json").write_text(json.dumps(
            {"weights": f"{HOME_STR}/w.cxpolicy", "seeds": [], "verdict": "pass"}))
        return _Done()

    monkeypatch.setattr(ev.subprocess, "run", fake_run)
    got = ev.run_cadex_evaluate("base", tmp_path / "w.cxpolicy", tmp_path / "policy", video=False)
    _no_home(tmp_path / "policy")
    assert got["seeds"] == []


def test_design_report_is_written_without_home(tmp_path):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from build_robot import write_design_report

    out = write_design_report(tmp_path, {"cadex_project": f"{HOME_STR}/cadex-projects/q", "ok": np.bool_(True)})
    _no_home(out)
    assert json.loads(out.read_text())["ok"] is True


def test_a_file_that_is_not_a_policy_is_left_alone(tmp_path):
    p = tmp_path / "walk.cxpolicy"
    p.write_bytes(b"\xff\x00not a policy /home/u/x")
    assert scrub_file(p, ROOTS) is False
    assert p.read_bytes() == b"\xff\x00not a policy /home/u/x"
