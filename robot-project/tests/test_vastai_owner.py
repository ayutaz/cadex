"""vast.ai のアカウントは他の作業と共有している。このプロジェクトが作ったインスタンスだけを扱う。

- 作成時に label（タグ）を付ける: ``cadex-q8-<種類>-<YYYYMMDD-HHMM>``
- 破棄は、label がこのプロジェクトのものと分かるインスタンスにしか送らない
- 一覧では、他の作業のインスタンスを「触らない」と示し、SSH の接続先を出さない
"""

from __future__ import annotations

import io
import json
from datetime import datetime

from vastai import cli, offers

NOW = datetime(2026, 11, 17, 19, 30)
OURS = {"id": 111, "actual_status": "running", "dph_total": 0.42, "label": "cadex-q8-smoke-20261117-1930",
        "ssh_host": "ssh5.vast.ai", "ssh_port": 41234}
OTHER = {"id": 60809472, "actual_status": "running", "dph_total": 3.26, "label": "s2-k1-l40s",
         "ssh_host": "ssh6.vast.ai", "ssh_port": 16384}
UNLABELLED = {"id": 222, "actual_status": "running", "dph_total": 0.3, "label": None}


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class RoutingOpener:
    """GET /instances には一覧を、それ以外には success を返す。送ったリクエストを記録する。"""

    def __init__(self, instances):
        self.instances = instances
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if request.get_method() == "GET":
            return Response(json.dumps({"instances": self.instances}).encode())
        return Response(json.dumps({"success": True, "new_contract": 111}).encode())


def test_run_label_names_the_project_the_kind_and_the_time():
    assert offers.run_label("q8-smoke", NOW) == "cadex-q8-smoke-20261117-1930"
    assert offers.run_label("full", NOW) == "cadex-q8-full-20261117-1930"
    assert offers.run_label("", NOW) == "cadex-q8-run-20261117-1930"


def test_owned_only_by_the_project_label():
    assert offers.is_owned(OURS)
    assert not offers.is_owned(OTHER)
    assert not offers.is_owned(UNLABELLED)
    assert not offers.is_owned({"label": "cadex-q8"})  # 接頭辞だけでは足りない
    assert not offers.is_owned({"label": "xcadex-q8-smoke"})


def test_create_sends_the_run_label():
    opener = RoutingOpener([])
    result = offers.create_instance(5, {"image": "ghcr.io/x/cadex-train:sha-abc1234", "label": "q8-smoke"},
                                    api_key="k", yes=True, opener=opener, now=NOW)
    assert json.loads(opener.requests[0].data)["label"] == "cadex-q8-smoke-20261117-1930"
    assert result["request"]["label"] == "cadex-q8-smoke-20261117-1930"


def test_create_dry_run_shows_the_label_it_would_send():
    result = offers.create_instance(5, {"image": "img", "label": "q8-full"}, api_key="k", yes=False,
                                    opener=RoutingOpener([]), now=NOW)
    assert result["request"]["label"] == "cadex-q8-full-20261117-1930"


def test_destroy_refuses_an_instance_of_another_work():
    opener = RoutingOpener([OURS, OTHER])
    result = offers.destroy_instance(OTHER["id"], api_key="k", yes=True, opener=opener)
    assert result["executed"] is False
    assert "このプロジェクト" in result["reason"]
    assert [r.get_method() for r in opener.requests] == ["GET"]  # DELETE は送らない


def test_destroy_refuses_an_unknown_instance():
    opener = RoutingOpener([OURS])
    result = offers.destroy_instance(999, api_key="k", yes=True, opener=opener)
    assert result["executed"] is False
    assert [r.get_method() for r in opener.requests] == ["GET"]


def test_destroy_sends_delete_only_for_our_instance():
    opener = RoutingOpener([OURS, OTHER])
    result = offers.destroy_instance(OURS["id"], api_key="k", yes=True, opener=opener)
    assert result["executed"] is True
    assert [r.get_method() for r in opener.requests] == ["GET", "DELETE"]
    assert opener.requests[1].full_url.endswith(f"/instances/{OURS['id']}")


def _run(argv, opener):
    out = io.StringIO()
    code = cli.main(argv, environ={"VAST_API_KEY": "k"}, opener=opener, stdout=out,
                    env_file="/nonexistent/.env")
    return code, out.getvalue()


def test_instances_marks_other_work_and_hides_its_ssh():
    code, out = _run(["instances"], RoutingOpener([OURS, OTHER]))
    assert code == 0
    ours_line, = [l for l in out.splitlines() if l.startswith(str(OURS["id"]))]
    other_line, = [l for l in out.splitlines() if l.startswith(str(OTHER["id"]))]
    assert "このプロジェクト" in ours_line
    assert "触らない" in other_line
    assert "ssh5.vast.ai" in out
    assert "ssh6.vast.ai" not in out and "16384" not in out


def test_instances_shows_a_just_created_instance_without_status():
    # 作った直後の台は actual_status が null（None）で返る。一覧が落ちて自分の台が見えなくなった（2026-11-17）
    booting = {**OURS, "id": 333, "actual_status": None, "dph_total": None, "ssh_host": None, "ssh_port": None}
    code, out = _run(["instances"], RoutingOpener([booting, OTHER]))
    assert code == 0
    line, = [l for l in out.splitlines() if l.startswith("333")]
    assert "このプロジェクト" in line and "-" in line


def test_cli_destroy_of_other_work_fails_without_deleting():
    opener = RoutingOpener([OTHER])
    code, out = _run(["destroy", str(OTHER["id"]), "--yes"], opener)
    assert code == 2
    assert "このプロジェクト" in out
    assert all(r.get_method() != "DELETE" for r in opener.requests)
