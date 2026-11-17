"""検索して、その場で複数台を借りる（create --count）。

オファーの id は数分で作り直され、表を見てから --offer で借りるまでの間に無効になる
（HTTP 400 no_such_ask）。検索の直後に、条件に合う安い順に借りていき、無効なものは飛ばす。
"""

from __future__ import annotations

import io
import json
import urllib.error
from datetime import datetime

from vastai import offers

NOW = datetime(2026, 11, 17, 19, 56)
INSTANCE = {"image": "ghcr.io/ayutaz/cadex-train:sha-e7f4d9d", "disk_gb": 40, "label": "q8-full"}
OFFERS = [
    {"id": 1, "machine_id": 10, "gpu_name": "RTX 4090", "dph_total": 0.30, "inet_down": 2000.0},
    {"id": 2, "machine_id": 20, "gpu_name": "RTX 4090", "dph_total": 0.40, "inet_down": 3000.0},
    {"id": 3, "machine_id": 30, "gpu_name": "RTX 4090", "dph_total": 0.50, "inet_down": 1500.0},
    {"id": 4, "machine_id": 40, "gpu_name": "RTX 4090", "dph_total": 0.60, "inet_down": 1200.0},
]


class Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class Opener:
    """POST /bundles には OFFERS を、PUT /asks/<id> には成功か no_such_ask を返す。"""

    def __init__(self, gone=()):
        self.gone = set(gone)
        self.requests = []

    def __call__(self, request, timeout=None):
        self.requests.append(request)
        if request.get_method() == "POST":
            return Response(json.dumps({"offers": OFFERS}).encode())
        ask = int(request.full_url.rsplit("/", 1)[1])
        if ask in self.gone:
            body = json.dumps({"error": "invalid_args", "msg": f"no_such_ask {ask}"}).encode()
            raise urllib.error.HTTPError(request.full_url, 400, "Bad Request", {}, io.BytesIO(body))
        return Response(json.dumps({"success": True, "new_contract": 1000 + ask}).encode())

    def puts(self):
        return [int(r.full_url.rsplit("/", 1)[1]) for r in self.requests if r.get_method() == "PUT"]


def test_dry_run_shows_the_offers_it_would_rent_and_sends_no_put():
    opener = Opener()
    result = offers.create_many({}, INSTANCE, count=2, api_key="k", yes=False, opener=opener, now=NOW)
    assert result["executed"] is False
    assert [o["id"] for o in result["chosen"]] == [1, 2]   # 安い順
    assert opener.puts() == []


def test_rents_cheapest_first_and_skips_offers_that_are_gone():
    opener = Opener(gone={1})
    result = offers.create_many({}, INSTANCE, count=2, api_key="k", yes=True, opener=opener, now=NOW)
    assert result["executed"] is True
    assert result["created"] == [1002, 1003]
    assert opener.puts() == [1, 2, 3]
    assert result["skipped"] and result["skipped"][0][0] == 1 and "no_such_ask" in result["skipped"][0][1]
    labels = {json.loads(r.data)["label"] for r in opener.requests if r.get_method() == "PUT"}
    assert labels == {"cadex-q8-full-20261117-1956"}


def test_excluded_machines_and_one_machine_per_instance():
    opener = Opener()
    result = offers.create_many({}, INSTANCE, count=3, api_key="k", yes=True, opener=opener, now=NOW,
                                exclude_machines={20})
    assert opener.puts() == [1, 3, 4]
    assert result["created"] == [1001, 1003, 1004]


def test_reports_when_fewer_offers_than_asked():
    opener = Opener(gone={1, 2, 3, 4})
    result = offers.create_many({}, INSTANCE, count=2, api_key="k", yes=True, opener=opener, now=NOW)
    assert result["created"] == []
    assert len(result["skipped"]) == 4


def test_needs_the_api_key_to_rent():
    opener = Opener()
    result = offers.create_many({}, INSTANCE, count=1, api_key="", yes=True, opener=opener, now=NOW)
    assert result["executed"] is False and "VAST_API_KEY" in result["reason"]
    assert opener.puts() == []
