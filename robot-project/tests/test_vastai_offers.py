"""vast.ai 学習基盤: オファー検索・料金見積り・インスタンス作成の安全装置。

HTTP はすべて偽物の opener で差し替える。本物の vast.ai には一度も触れない。
"""

from __future__ import annotations

import io
import json

import pytest

from vastai import offers


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeOpener:
    def __init__(self, payload: dict):
        self.payload = payload
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        return FakeResponse(json.dumps(self.payload).encode())


OFFER_CFG = {
    "gpu_name": "RTX 4090",
    "num_gpus": 1,
    "min_reliability": 0.98,
    "min_cuda": 12.4,
    "min_disk_gb": 40,
    "min_inet_down_mbps": 200,
    "max_dph": 1.0,
    "verified": True,
    "limit": 5,
}

OFFERS = {"offers": [
    {"id": 111, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.42,
     "reliability2": 0.995, "cuda_max_good": 12.8, "gpu_ram": 24564,
     "disk_space": 120.0, "inet_down": 900.0, "geolocation": "Japan, JP",
     "storage_cost": 0.15},
    {"id": 222, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.35,
     "reliability2": 0.99, "cuda_max_good": 12.4, "gpu_ram": 24564,
     "disk_space": 60.0, "inet_down": 500.0, "geolocation": "Sweden, SE"},
]}


def test_検索条件を組み立てる():
    query = offers.build_query(OFFER_CFG)
    assert query["gpu_name"] == {"eq": "RTX 4090"}
    assert query["num_gpus"] == {"eq": 1}
    assert query["reliability2"] == {"gte": 0.98}
    assert query["cuda_max_good"] == {"gte": 12.4}
    assert query["disk_space"] == {"gte": 40}
    assert query["inet_down"] == {"gte": 200}
    assert query["dph_total"] == {"lte": 1.0}
    assert query["rentable"] == {"eq": True}
    assert query["verified"] == {"eq": True}
    assert query["order"] == [["dph_total", "asc"]]
    assert query["limit"] == 5


def test_検索は読むだけで_POST_を一度送る():
    opener = FakeOpener(OFFERS)
    found = offers.search_offers(offers.build_query(OFFER_CFG), api_key="k", opener=opener)
    assert [offer["id"] for offer in found] == [222, 111]  # 安い順に並べ直す
    (request,) = opener.requests
    assert request.get_method() == "POST"
    assert request.full_url == "https://console.vast.ai/api/v0/bundles"
    assert request.get_header("Authorization") == "Bearer k"
    assert json.loads(request.data)["gpu_name"] == {"eq": "RTX 4090"}


def test_検索は_api_key_が無くても送れるが鍵の見出しは付けない():
    opener = FakeOpener(OFFERS)
    offers.search_offers({}, api_key="", opener=opener)
    assert opener.requests[0].get_header("Authorization") is None


def test_学習時間の見積り():
    hours = offers.estimate_hours(
        iterations=1000, seconds_per_iteration=3.6, setup_minutes=15, margin=1.2)
    # (1000*3.6/3600 + 15/60) * 1.2 = (1.0 + 0.25) * 1.2
    assert hours == pytest.approx(1.5)


def test_進捗ファイルから一反復の秒数を出す():
    progress = {"iteration": 9, "wall_time_s": 50.0}
    assert offers.seconds_per_iteration(progress) == pytest.approx(5.0)
    assert offers.seconds_per_iteration({"iteration": -1, "wall_time_s": 0.0}) is None


def test_総額は時間単価とディスク代():
    offer = {"dph_total": 0.5, "storage_cost": 0.15}
    # GPU 2 時間 = 1.0 ドル、ディスク 40GB * 0.15 $/GB/月 * 2h/730h
    total = offers.estimate_cost(offer, hours=2.0, disk_gb=40)
    assert total == pytest.approx(1.0 + 40 * 0.15 * 2.0 / 730.0)
    assert offers.estimate_cost({"dph_total": 0.5}, hours=2.0, disk_gb=40) == pytest.approx(1.0)


def test_表に_GPU種別と時間単価と推定総額が出る():
    table = offers.format_offers(OFFERS["offers"], hours=1.5, disk_gb=40)
    assert "RTX 4090" in table
    assert "0.420" in table and "0.350" in table
    assert "$/h" in table
    assert f"{0.42 * 1.5 + 40 * 0.15 * 1.5 / 730:.2f}" in table


INSTANCE_CFG = {"image": "ghcr.io/example/cadex-train:latest", "disk_gb": 40,
                "label": "q8-smoke"}


def test_作成は既定で_dry_run_で何も送らない():
    opener = FakeOpener({"success": True, "new_contract": 1})
    result = offers.create_instance(222, INSTANCE_CFG, api_key="k", yes=False, opener=opener)
    assert result["executed"] is False
    assert "--yes" in result["reason"]
    assert opener.requests == []
    assert result["request"]["image"] == INSTANCE_CFG["image"]


def test_作成は_yes_があっても_API_キーが無ければ送らない():
    opener = FakeOpener({"success": True, "new_contract": 1})
    result = offers.create_instance(222, INSTANCE_CFG, api_key="", yes=True, opener=opener)
    assert result["executed"] is False
    assert "VAST_API_KEY" in result["reason"]
    assert opener.requests == []


def test_作成は_yes_と_API_キーの両方で初めて_PUT_する():
    opener = FakeOpener({"success": True, "new_contract": 9876})
    result = offers.create_instance(222, INSTANCE_CFG, api_key="k", yes=True, opener=opener)
    assert result["executed"] is True
    assert result["instance_id"] == 9876
    (request,) = opener.requests
    assert request.get_method() == "PUT"
    assert request.full_url == "https://console.vast.ai/api/v0/asks/222"
    body = json.loads(request.data)
    assert body["image"] == INSTANCE_CFG["image"]
    assert body["disk"] == 40
    assert body["runtype"] == "ssh_direc ssh_proxy"
    # 設定の label は種類として使い、作成時刻つきの「このプロジェクトの label」に付け替える
    assert body["label"].startswith("cadex-q8-smoke-")


def test_runtype_は設定で差し替えられる():
    opener = FakeOpener({"success": True, "new_contract": 1})
    offers.create_instance(222, {**INSTANCE_CFG, "runtype": "ssh_proxy"}, api_key="k",
                           yes=True, opener=opener)
    assert json.loads(opener.requests[0].data)["runtype"] == "ssh_proxy"


def test_破棄も_yes_と_API_キーが揃うまで送らない():
    # 破棄の前に一覧で「このプロジェクトのインスタンスか」を確かめる（test_vastai_owner.py）
    opener = FakeOpener({"success": True,
                         "instances": [{"id": 9876, "label": "cadex-q8-smoke-20261117-1930"}]})
    assert offers.destroy_instance(9876, api_key="k", yes=False, opener=opener)["executed"] is False
    assert offers.destroy_instance(9876, api_key="", yes=True, opener=opener)["executed"] is False
    assert opener.requests == []
    assert offers.destroy_instance(9876, api_key="k", yes=True, opener=opener)["executed"] is True
    assert [r.get_method() for r in opener.requests] == ["GET", "DELETE"]
    request = opener.requests[-1]
    assert request.full_url == "https://console.vast.ai/api/v0/instances/9876"


def test_インスタンス一覧から_SSH_接続先を取り出す():
    opener = FakeOpener({"instances": [
        {"id": 9876, "actual_status": "running", "ssh_host": "ssh5.vast.ai",
         "ssh_port": 41234, "public_ipaddr": "203.0.113.7", "dph_total": 0.42,
         "label": "q8-smoke",
         "ports": {"22/tcp": [{"HostIp": "0.0.0.0", "HostPort": "40022"}]}},
    ]})
    (row,) = offers.list_instances(api_key="k", opener=opener)
    assert opener.requests[0].get_method() == "GET"
    assert opener.requests[0].full_url == "https://console.vast.ai/api/v1/instances"
    assert row["id"] == 9876
    assert row["proxy_ssh"] == ("ssh5.vast.ai", 41234)
    assert row["direct_ssh"] == ("203.0.113.7", 40022)


def test_一覧は_API_キーが無ければ説明つきで失敗する():
    with pytest.raises(offers.VastError, match="VAST_API_KEY"):
        offers.list_instances(api_key="", opener=FakeOpener({}))


def test_HTTP_エラーは_vast_ai_が返した理由を見せ_長い英数字は伏せる():
    import urllib.error

    import pytest

    body = json.dumps({"success": False, "error": "invalid_args",
                       "msg": "disk too small for image abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH"})

    def opener(request, timeout=None):
        raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body.encode()))

    with pytest.raises(offers.VastError) as caught:
        offers.create_instance(5, {"image": "img", "label": "q8-smoke"}, api_key="k", yes=True,
                               opener=opener)
    text = str(caught.value)
    assert "HTTP 400" in text and "invalid_args" in text and "disk too small" in text
    assert "abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGH" not in text
