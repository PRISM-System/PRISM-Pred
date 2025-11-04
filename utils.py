import datetime
from datetime import datetime, timezone, timedelta
import uuid
from typing import Any, Dict, Optional
import json
import dataclasses

def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def rid() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"

def to_dict(x):
    if isinstance(x, dict): return x
    if hasattr(x, "model_dump"): return x.model_dump()
    if hasattr(x, "dict"): return x.dict()
    return x

def default(val, fallback):
    return fallback if val is None or val == "" else val

def coerce_time_range(tr):
    """dict|str|None → dict {start,end} 로 통일. 없으면 최근 90분."""
    def _iso(dt): return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    now = datetime.now(timezone.utc)
    if isinstance(tr, dict):
        start = tr.get("start") or _iso(now - timedelta(minutes=90))
        end   = tr.get("end")   or _iso(now)
        return {"start": start, "end": end}
    if isinstance(tr, str) and "~" in tr:
        a, b = tr.split("~", 1)
        return {"start": a.strip(), "end": b.strip()}
    return {"start": _iso(now - timedelta(minutes=90)), "end": _iso(now)}

def normalize_list(x):
    if x is None: return []
    if isinstance(x, str):
        parts = [p.strip() for p in x.split(",")]
        return [p for p in parts if p]
    if isinstance(x, (list, tuple)): return [str(t) for t in x if t is not None]
    return []

def parse_iso_z(s: str) -> Optional[datetime]:
    try:
        if s and isinstance(s, str) and s.endswith("Z"):
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except Exception:
        pass
    return None

def is_filled(x) -> bool:
    if x is None: return False
    if isinstance(x, str): return x.strip() != ""
    if isinstance(x, (list, tuple, dict)): return len(x) > 0
    return True

def to_jsonable(obj) -> Dict[str, Any]:
    """obj를 dict로 안전 직렬화 (pydantic v2/v1/dataclass/일반객체 모두 커버)"""
    if obj is None: return {}
    if isinstance(obj, dict): return obj
    if hasattr(obj, "model_dump") and callable(getattr(obj, "model_dump")): return obj.model_dump()
    if hasattr(obj, "dict") and callable(getattr(obj, "dict")): return obj.dict()
    if dataclasses.is_dataclass(obj): return dataclasses.asdict(obj)
    try:
        return json.loads(json.dumps(obj, default=lambda o: getattr(o, "__dict__", str(o))))
    except Exception:
        return {"value": str(obj)}
