"""把 Kai 的 E/R 分類結果或人工稽核結果寫回 trajectory。

覆寫檔是 JSONL，每行一筆，針對一個 segment，可以做三種事（可同時出現）：

    {"traj_id": "...", "seg_id": 12, "role": "E", "source": "classifier"}
        → 整段改成 E

    {"traj_id": "...", "seg_id": 12, "call_role": "R", "source": "classifier"}
        → 只把「呼叫參數」改判成 R（例如 bash 指令裡寫了 solver 的假設）；
          條件不含 R 時，verifier 只會看到工具名稱，看不到參數

    {"traj_id": "...", "seg_id": 12, "spans": [[0, 40, "Q"], [40, 95, "E"]], "source": "manual"}
        → 一段切成多個區間、各自標角色（例如使用者一句話裡同時有需求和觀測）
          區間必須依序、不重疊、完整覆蓋整段文字（字元位置，[start, end)）

source 只能是 "classifier"（Kai 的分類器）或 "manual"（人工稽核）。
manual 的優先權高於 classifier：同一個 segment 兩種都有時以 manual 為準。

切分「怎麼切」是 Kai 和人工稽核的判斷；這裡只負責「正確地寫回去、寫錯會報錯」：
- 覆寫對不到任何 segment → cli 會回報並以失敗結束（通常是 ID 打錯或資料檔不對）
- 寫回後重新驗證整條 trajectory（例如改完之後沒有 Q 或沒有 A，會直接報錯）

    python -m qera.cli apply-overrides data/x.jsonl overrides/kai.jsonl overrides/manual.jsonl --out data/x.v2.jsonl
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator

from . import jsonl, tokens
from .builder import _check_reasoning_provenance
from .schema import Role, Span, Trajectory

_PRIORITY = {"rule": 0, "classifier": 1, "manual": 2}


def load_overrides(paths: Iterable[str]) -> dict[tuple[str, int], dict]:
    out: dict[tuple[str, int], dict] = {}
    for p in paths:
        for o in jsonl.read(p):
            src = o.get("source", "manual")
            if src not in ("classifier", "manual"):
                raise ValueError(f"source 必須是 classifier 或 manual：{o}")
            if not any(k in o for k in ("role", "call_role", "spans")):
                raise ValueError(f"覆寫至少要有 role / call_role / spans 其中一個：{o}")
            for k in ("role", "call_role"):
                if k in o:
                    Role(o[k])  # 驗證角色名稱
            for sp in o.get("spans") or []:
                Role(sp[2])
            key = (o["traj_id"], int(o["seg_id"]))
            merged = out.setdefault(
                key, {"traj_id": key[0], "seg_id": key[1], "_sources": {}}
            )
            sources = merged["_sources"]
            for field in ("role", "spans", "call_role"):
                if field not in o:
                    continue
                # role 與 spans 代表同一份正文判斷；呼叫則獨立比較來源優先權。
                opposite = (
                    "spans" if field == "role" else "role" if field == "spans" else None
                )
                previous = sources.get(field, sources.get(opposite, "rule"))
                if _PRIORITY[src] < _PRIORITY[previous]:
                    continue
                if opposite:
                    merged.pop(opposite, None)
                    sources.pop(opposite, None)
                merged[field] = o[field]
                sources[field] = src
                if o.get("note"):
                    merged["note"] = o["note"]
            merged["source"] = max(sources.values(), key=_PRIORITY.get)
    return out


def apply(
    trajs: Iterable[Trajectory],
    overrides: dict[tuple[str, int], dict],
    stats: Counter | None = None,
) -> Iterator[Trajectory]:
    stats = stats if stats is not None else Counter()
    for t in trajs:
        changed = False
        for s in t.segments:
            o = overrides.get((t.traj_id, s.seg_id))
            if not o:
                continue
            applied = False
            for field in ("role", "spans", "call_role"):
                if field not in o:
                    continue
                src = o.get("_sources", {}).get(field, o.get("source", "manual"))
                text_field = "spans" if s.spans else "role"
                previous = s.override_sources.get(
                    "call_role" if field == "call_role" else text_field,
                    s.role_source
                    if (
                        s.call_role_override is not None
                        if field == "call_role"
                        else s.spans is not None or s.role_override is not None
                    )
                    else "rule",
                )
                if _PRIORITY[src] < _PRIORITY[previous]:
                    continue
                if field == "role":
                    s.role_override = Role(o[field])
                    s.spans = None
                    s.override_sources.pop("spans", None)
                elif field == "spans":
                    s.spans = [
                        Span(
                            start=a,
                            end=b,
                            role=Role(r),
                            n_tokens=tokens.count(s.text[a:b]),
                        )
                        for a, b, r in o[field]
                    ]
                    s.role_override = None
                    s.override_sources.pop("role", None)
                else:
                    s.call_role_override = Role(o[field])
                s.override_sources[field] = src
                applied = True
            if not applied:
                stats["lower_priority"] += 1
                continue
            s.role_source = max(s.override_sources.values(), key=_PRIORITY.get)
            if o.get("note"):
                s.note = o["note"]
            stats["applied"] += 1
            changed = True
        if changed:
            try:
                _check_reasoning_provenance(t.segments, t.solver, t.meta)
                t = t.revalidate()  # 改完重新檢查（spans 是否合法、是否仍有 Q 和 A）
            except Exception as e:
                raise ValueError(f"{t.traj_id} 套用覆寫後不合法：{e}") from e
        yield t
