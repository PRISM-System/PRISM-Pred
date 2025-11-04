# utils.py - Utility functions for PRISM-Pred
import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, List


def now_iso() -> str:
    """Return current time in ISO format"""
    return datetime.now(timezone.utc).isoformat()


def rid() -> str:
    """Generate a random UUID"""
    return str(uuid.uuid4())


def to_dict(obj: Any) -> dict:
    """Convert object to dictionary"""
    if hasattr(obj, '__dict__'):
        return obj.__dict__
    return dict(obj)


def default(val: Any, default_val: Any) -> Any:
    """Return val if not None, otherwise return default_val"""
    return val if val is not None else default_val


def normalize_list(val: Any) -> List:
    """Normalize value to a list"""
    if val is None:
        return []
    if isinstance(val, list):
        return val
    return [val]


def coerce_time_range(start: Optional[str], end: Optional[str]) -> tuple:
    """Coerce time range strings to datetime objects"""
    from datetime import datetime

    start_dt = parse_iso_z(start) if start else None
    end_dt = parse_iso_z(end) if end else None

    return start_dt, end_dt


def parse_iso_z(ts: Optional[str]) -> Optional[datetime]:
    """Parse ISO 8601 timestamp with Z suffix"""
    if not ts:
        return None

    # Remove 'Z' suffix and parse
    if ts.endswith('Z'):
        ts = ts[:-1] + '+00:00'

    try:
        return datetime.fromisoformat(ts)
    except Exception:
        return None


def to_jsonable(obj: Any) -> Any:
    """Convert object to JSON-serializable format"""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif hasattr(obj, '__dict__'):
        return {k: to_jsonable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [to_jsonable(item) for item in obj]
    else:
        return obj
