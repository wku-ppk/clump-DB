from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def _normalize(obj: Any):
    """Make payload deterministic: sort keys, stabilize floats, recurse."""
    if isinstance(obj, float):
        # float 안정화(원하시면 자릿수 조정 가능)
        return float(f"{obj:.8g}")
    if isinstance(obj, dict):
        return {k: _normalize(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [_normalize(x) for x in obj]
    return obj


def make_hash_id(payload: Dict[str, Any], n: int = 12) -> str:
    """payload -> canonical json -> sha256 -> hex prefix"""
    canon = _normalize(payload)
    s = json.dumps(canon, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return h[:n]