"""Reuse the first valid API response for an identical endpoint/request."""

import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, JsonValue


def request_hash(endpoint, request):
    wire = json.dumps([endpoint, request], sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(wire.encode()).hexdigest()


class CacheRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", hide_input_in_errors=True)
    version: Literal[1] = 1
    request_hash: str
    raw: JsonValue


@contextmanager
def cached_response(directory, endpoint, request):
    """Hold a per-request lock through transport and validation; failed calls aren't saved."""
    key = request_hash(endpoint, request)
    if directory is None:
        yield {"key": key, "hit": False, "raw": None}
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = root / (key + ".json")
    with (root / (key + ".lock")).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = {"key": key, "hit": path.exists(), "raw": None}
        if path.exists():
            record = CacheRecord.model_validate_json(path.read_text(encoding="utf-8"))
            if record.request_hash != key:
                raise ValueError(
                    "Cached response fingerprint does not match its filename"
                )
            state["raw"] = record.raw
        yield state
        if not state["hit"] and state.get("accepted"):
            record = CacheRecord(request_hash=key, raw=state["raw"])
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=root, delete=False, encoding="utf-8"
                ) as target:
                    temporary = Path(target.name)
                    target.write(record.model_dump_json() + "\n")
                    target.flush()
                    os.fsync(target.fileno())
                os.replace(temporary, path)
            finally:
                if temporary and temporary.exists():
                    temporary.unlink()
