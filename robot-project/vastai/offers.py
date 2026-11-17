"""vast.ai の REST API: オファー検索、料金の見積り、インスタンスの作成・一覧・破棄。

標準ライブラリ（urllib）だけで叩く。課金が発生する操作（作成）と、
インスタンスを消す操作（破棄）は ``yes=True`` と API キーの両方が揃った時だけ
送る。どちらかが欠ければ、送るはずだった中身を返して何もしない（dry-run）。

API キーはリクエストの ``Authorization: Bearer`` 見出しにだけ載せ、
表示もファイル保存もしない。

API の仕様（2026-11-17 時点の公式ドキュメントの調査による。未確認の点は UNCONFIRMED）:

* 検索  POST   /api/v0/bundles          本体はフラットな JSON、応答は ``offers``
* 作成  PUT    /api/v0/asks/{offer_id}  応答 ``{"success": true, "new_contract": id}``
  （UNCONFIRMED: ``POST /asks/{id}/`` と書かれた資料もある）
* 一覧  GET    /api/v1/instances        応答は ``instances``（一覧だけ v1）
* 破棄  DELETE /api/v0/instances/{id}
* 停止  PUT    /api/v0/instances/{id}  ``{"state": "stopped"}``（この道具は使わない）
* UNCONFIRMED: 末尾スラッシュの要否（vast の CLI は ``/bundles/`` のように付ける）。
  失敗したら HTTP の状態コードを見て API_V0 / API_V1 と各パスを直す。
"""

from __future__ import annotations

import json
import re
from datetime import datetime
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Mapping

API = API_V0 = "https://console.vast.ai/api/v0"
API_V1 = "https://console.vast.ai/api/v1"

#: 1 か月を何時間と見るか（vast.ai のストレージ単価は $/GB/月）
HOURS_PER_MONTH = 730.0

Opener = Callable[..., Any]


class VastError(RuntimeError):
    """vast.ai とのやり取りの失敗。キーやリクエスト本体は含めない。"""


def _error_detail(error: urllib.error.HTTPError) -> str:
    """vast.ai が返した理由（error と msg）。キーなどが混ざらないよう長い英数字は伏せる。"""

    try:
        data = json.loads(error.read() or b"{}")
    except (ValueError, OSError):
        return ""
    if not isinstance(data, dict):
        return ""
    parts = [str(data[k]) for k in ("error", "msg") if data.get(k)]
    if not parts:
        return ""
    text = re.sub(r"[A-Za-z0-9_\-]{32,}", "<伏せ>", " / ".join(parts))
    return f"（{text[:300]}）"


def _request(method: str, path: str, *, api_key: str, payload: Any = None,
             opener: Opener | None = None, timeout: float = 30.0, base: str = API_V0) -> Any:
    opener = opener or urllib.request.urlopen
    headers = {"Accept": "application/json", "User-Agent": "vastai-train/1.0"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(base + path, data=data, method=method, headers=headers)
    try:
        with opener(request, timeout=timeout) as response:
            body = response.read()
    except urllib.error.HTTPError as error:
        raise VastError(f"vast.ai {method} {path}: HTTP {error.code}{_error_detail(error)}") from None
    except (urllib.error.URLError, OSError) as error:
        raise VastError(f"vast.ai {method} {path}: {type(error).__name__}"
                        "（作成を再送する前に、コンソールで作られていないか確かめる）") from None
    try:
        return json.loads(body) if body else {}
    except ValueError:
        raise VastError(f"vast.ai {method} {path}: JSON でない応答") from None


# --- 検索 -------------------------------------------------------------------

def build_query(offer: Mapping[str, Any]) -> dict[str, Any]:
    """設定の ``offer`` 節 → vast.ai の検索条件（``{"field": {"op": value}}``）。"""

    query: dict[str, Any] = {
        "rentable": {"eq": True},
        "rented": {"eq": False},
        # UNCONFIRMED: 値が "on-demand" か "ondemand" か（設定の offer.type で変えられる）
        "type": offer.get("type", "on-demand"),
        "order": [["dph_total", "asc"]],
        "limit": int(offer.get("limit", 10)),
    }
    # gpu_name は応答と同じ空白区切り（"RTX 4090"）で書く。CLI は "RTX_4090" の "_" を
    # 空白に直してから送る。UNCONFIRMED: API が "_" 区切りも受け付けるか。
    names = offer.get("gpu_name")
    if isinstance(names, (list, tuple)):
        query["gpu_name"] = {"in": list(names)}
    elif names:
        query["gpu_name"] = {"eq": names}
    if "num_gpus" in offer:
        query["num_gpus"] = {"eq": int(offer["num_gpus"])}
    ranges = {
        "min_reliability": ("reliability2", "gte"),
        "min_cuda": ("cuda_max_good", "gte"),
        "min_disk_gb": ("disk_space", "gte"),
        "min_inet_down_mbps": ("inet_down", "gte"),
        "min_gpu_ram_gb": ("gpu_ram", "gte"),
        "max_dph": ("dph_total", "lte"),
    }
    for key, (field, operator) in ranges.items():
        if offer.get(key) is not None:
            value = offer[key]
            if key == "min_gpu_ram_gb":
                value = value * 1024  # gpu_ram は MB
            query[field] = {operator: value}
    if offer.get("verified") is not None:
        query["verified"] = {"eq": bool(offer["verified"])}
    if offer.get("geolocation"):
        query["geolocation"] = {"in": list(offer["geolocation"])}
    return query


def search_offers(query: Mapping[str, Any], *, api_key: str = "",
                  opener: Opener | None = None) -> list[dict[str, Any]]:
    """読むだけ。課金は発生しない。安い順に並べて返す。"""

    answer = _request("POST", "/bundles", api_key=api_key, payload=dict(query), opener=opener)
    found = list(answer.get("offers") or [])
    found.sort(key=lambda offer: float(offer.get("dph_total") or 0.0))
    return found


# --- 見積り -----------------------------------------------------------------

def estimate_hours(*, iterations: int, seconds_per_iteration: float,
                   setup_minutes: float = 15.0, margin: float = 1.3) -> float:
    """借りている時間の見積り（時間）。

    学習そのもの（反復数 × 一反復の秒数）に、イメージの取得・JIT コンパイル・
    回収の時間（setup_minutes）を足し、余裕（margin）を掛ける。
    """

    training = iterations * seconds_per_iteration / 3600.0
    return (training + setup_minutes / 60.0) * margin


def seconds_per_iteration(progress: Mapping[str, Any]) -> float | None:
    """progress.json から実測の一反復の秒数。最初の反復の JIT コンパイルを含むので多めに出る。"""

    done = int(progress.get("iteration", -1)) + 1
    wall = float(progress.get("wall_time_s") or 0.0)
    if done <= 0 or wall <= 0.0:
        return None
    return wall / done


def measured_envs(progress: Mapping[str, Any]) -> int | None:
    """progress.json の ``run.argv`` から、その学習の --envs を読む（無ければ None）。"""

    argv = list((progress.get("run") or {}).get("argv") or [])
    for index, item in enumerate(argv):
        if item == "--envs" and index + 1 < len(argv):
            try:
                return int(argv[index + 1])
            except ValueError:
                return None
        if str(item).startswith("--envs="):
            try:
                return int(str(item).split("=", 1)[1])
            except ValueError:
                return None
    return None


def estimate_cost(offer: Mapping[str, Any], *, hours: float, disk_gb: float) -> float:
    """推定総額（ドル）。``dph_total`` は GPU 代とディスク代を含む時間単価だが、
    ``storage_cost``（$/GB/月）が分かれば停止中も含めたディスク代として足す。"""

    total = float(offer.get("dph_total") or 0.0) * hours
    storage = offer.get("storage_cost")
    if storage:
        total += float(storage) * disk_gb * hours / HOURS_PER_MONTH
    return total


def format_offers(found: list[Mapping[str, Any]], *, hours: float, disk_gb: float) -> str:
    header = (f"{'id':>10}  {'GPU':<14} {'数':>2} {'VRAM':>6} {'$/h':>7} "
              f"{'推定総額$':>9} {'信頼度':>6} {'CUDA':>5} {'下りMbps':>8}  場所")
    lines = [header, "-" * len(header)]
    for offer in found:
        lines.append(
            f"{offer.get('id', '-'):>10}  {str(offer.get('gpu_name', '-')):<14} "
            f"{offer.get('num_gpus', '-'):>2} "
            f"{float(offer.get('gpu_ram') or 0) / 1024:>5.0f}G "
            f"{float(offer.get('dph_total') or 0):>7.3f} "
            f"{estimate_cost(offer, hours=hours, disk_gb=disk_gb):>9.2f} "
            f"{float(offer.get('reliability2') or offer.get('reliability') or 0):>6.3f} "
            f"{offer.get('cuda_max_good', '-'):>5} "
            f"{float(offer.get('inet_down') or 0):>8.0f}  {offer.get('geolocation', '-')}")
    lines.append(f"（推定総額 = $/h × {hours:.2f} 時間 + ディスク {disk_gb:g} GB 分。"
                 "転送料と停止中のディスク代は別）")
    return "\n".join(lines)


# --- 作成・一覧・破棄 -------------------------------------------------------

#: このプロジェクトが作ったインスタンスの label の接頭辞。アカウントは他の作業と共有しているので、
#: 破棄などはこの label のインスタンスにだけ送る（他の作業のインスタンスには触らない）。
OWNER_PREFIX = "cadex-q8"


def run_label(kind: str, now: datetime) -> str:
    """作成時に付ける label（タグ）: ``cadex-q8-<種類>-<YYYYMMDD-HHMM>``。"""
    kind = str(kind or "").strip()
    for prefix in (OWNER_PREFIX + "-", "q8-"):
        if kind.startswith(prefix):
            kind = kind[len(prefix):]
    kind = re.sub(r"[^A-Za-z0-9_.-]+", "-", kind).strip("-") or "run"
    return f"{OWNER_PREFIX}-{kind}-{now:%Y%m%d-%H%M}"


def is_owned(instance: Mapping[str, Any]) -> bool:
    """label がこのプロジェクトのもの（``cadex-q8-…``）か。"""
    label = str(instance.get("label") or "")
    return label.startswith(OWNER_PREFIX + "-") and len(label) > len(OWNER_PREFIX) + 1

def instance_request(instance: Mapping[str, Any]) -> dict[str, Any]:
    body: dict[str, Any] = {
        "image": instance.get("image", ""),
        "disk": float(instance.get("disk_gb", 40)),
        # SSH で入る（直結 + プロキシ）。この方式ではイメージの ENTRYPOINT は使われない。
        # UNCONFIRMED: 値の綴り。vast の CLI（--ssh --direct）は "ssh_direc ssh_proxy" を送るが、
        # ドキュメントには "ssh_proxy" / "ssh_direct" とある。設定の instance.runtype で変えられる。
        "runtype": instance.get("runtype", "ssh_direc ssh_proxy"),
        "label": instance.get("label", ""),
    }
    if body["disk"].is_integer():
        body["disk"] = int(body["disk"])
    if instance.get("onstart"):
        body["onstart"] = instance["onstart"]
    if instance.get("env"):
        body["env"] = dict(instance["env"])
    return body


def _guard(*, yes: bool, api_key: str, what: str) -> str:
    if not yes:
        return f"dry-run: {what}は --yes を付けた時だけ実行します"
    if not api_key:
        return (f"{what}には VAST_API_KEY が必要です（環境変数か robot-project/.env）。"
                "何も送っていません")
    return ""


def create_instance(offer_id: int, instance: Mapping[str, Any], *, api_key: str,
                    yes: bool, opener: Opener | None = None,
                    now: datetime | None = None) -> dict[str, Any]:
    """オファーを借りる（**課金が始まる**）。``yes`` と ``api_key`` が揃わなければ送らない。

    label は設定の値を種類として、``run_label`` で作成時刻つきのものに付け替える。
    """

    body = instance_request(instance)
    body["label"] = run_label(str(instance.get("label", "")), now or datetime.now())
    reason = _guard(yes=yes, api_key=api_key, what="インスタンスの作成（課金開始）")
    image = str(body["image"])
    if not reason and (not image or "<" in image):
        reason = "instance.image が未設定です（Dockerfile から作ったイメージ名を書く）"
    plan = {"executed": False, "reason": reason, "offer_id": int(offer_id),
            "method": "PUT", "url": f"{API_V0}/asks/{int(offer_id)}", "request": body}
    if reason:
        return plan
    answer = _request("PUT", f"/asks/{int(offer_id)}", api_key=api_key,
                      payload=body, opener=opener)
    if not answer.get("success"):
        raise VastError("vast.ai がインスタンスの作成を受け付けませんでした"
                        "（コンソールで作られていないか確かめる）")
    return {**plan, "executed": True, "reason": "", "instance_id": answer.get("new_contract")}


def destroy_instance(instance_id: int, *, api_key: str, yes: bool,
                     opener: Opener | None = None) -> dict[str, Any]:
    """インスタンスを破棄する（ディスクも消え、課金が止まる）。"""

    reason = _guard(yes=yes, api_key=api_key, what="インスタンスの破棄（ディスクも消える）")
    plan = {"executed": False, "reason": reason, "instance_id": int(instance_id),
            "method": "DELETE", "url": f"{API_V0}/instances/{int(instance_id)}"}
    if reason:
        return plan
    found = [row for row in list_instances(api_key=api_key, opener=opener)
             if str(row.get("id")) == str(int(instance_id))]
    if not found or not is_owned(found[0]):
        label = found[0].get("label") if found else None
        plan["reason"] = (f"インスタンス {int(instance_id)}（label: {label or 'なし'}）は"
                          f"このプロジェクトのもの（label が {OWNER_PREFIX}- で始まる）ではないので、"
                          "破棄しません。他の作業のインスタンスには触りません")
        return plan
    _request("DELETE", f"/instances/{int(instance_id)}", api_key=api_key, opener=opener)
    return {**plan, "executed": True}


def _direct_ssh(instance: Mapping[str, Any]) -> tuple[str, int] | None:
    # UNCONFIRMED: 直結の SSH ポートが ports["22/tcp"][0]["HostPort"] に入ること、
    # それが ssh_port（プロキシ側）と別の値であること。両方を表示して利用者に選ばせる。
    mapped = (instance.get("ports") or {}).get("22/tcp") or []
    ip = instance.get("public_ipaddr")
    if not mapped or not ip:
        return None
    try:
        return (str(ip).strip(), int(mapped[0]["HostPort"]))
    except (KeyError, ValueError, TypeError):
        return None


def list_instances(*, api_key: str, opener: Opener | None = None) -> list[dict[str, Any]]:
    """自分のインスタンス一覧（読むだけ）。SSH の接続先を ``proxy_ssh`` / ``direct_ssh`` に入れる。"""

    if not api_key:
        raise VastError("インスタンス一覧には VAST_API_KEY が必要です（環境変数か robot-project/.env）")
    answer = _request("GET", "/instances", api_key=api_key, opener=opener, base=API_V1)
    rows = []
    for item in answer.get("instances") or []:
        row = dict(item)
        if item.get("ssh_host") and item.get("ssh_port"):
            row["proxy_ssh"] = (str(item["ssh_host"]), int(item["ssh_port"]))
        else:
            row["proxy_ssh"] = None
        row["direct_ssh"] = _direct_ssh(item)
        rows.append(row)
    return rows


def create_many(query: Mapping[str, Any], instance: Mapping[str, Any], *, count: int, api_key: str,
                yes: bool, opener: Opener | None = None, now: datetime | None = None,
                exclude_machines: Iterable[int] = ()) -> dict[str, Any]:
    """検索して、その場で ``count`` 台を借りる（**課金が始まる**）。

    オファーの id は数分で作り直され、表を見てから ``--offer`` で借りるまでに無効になる
    （no_such_ask）。そこで検索の直後に安い順に借りていき、無効なものは飛ばす。
    1 マシンに 1 台まで。``exclude_machines`` のマシン（GPU が壊れていた、など）は使わない。
    ``yes`` と ``api_key`` が揃わなければ、借りるはずのオファーを返すだけ。
    """

    excluded = {int(m) for m in exclude_machines}
    found = [o for o in search_offers(query, api_key=api_key, opener=opener)
             if o.get("machine_id") is None or int(o["machine_id"]) not in excluded]
    unique: list[dict[str, Any]] = []
    machines: set[Any] = set()
    for offer in found:
        machine = offer.get("machine_id")
        if machine is not None and machine in machines:
            continue
        machines.add(machine)
        unique.append(offer)
    reason = _guard(yes=yes, api_key=api_key, what="インスタンスの作成（課金開始）")
    result: dict[str, Any] = {"executed": False, "reason": reason, "chosen": unique[:count],
                              "created": [], "skipped": []}
    if reason:
        return result
    when = now or datetime.now()
    for offer in unique:
        if len(result["created"]) >= count:
            break
        try:
            made = create_instance(int(offer["id"]), instance, api_key=api_key, yes=True,
                                   opener=opener, now=when)
        except VastError as error:
            result["skipped"].append((int(offer["id"]), str(error)))
            continue
        result["created"].append(made["instance_id"])
    result["executed"] = True
    return result
