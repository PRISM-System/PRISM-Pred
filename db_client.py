# db_client.py
import os, logging
from typing import Any, Dict, List, Optional, Tuple
import requests
import pandas as pd

logger = logging.getLogger("prism_prediction.db")
DB_API_URL = os.getenv("DB_API_URL", "").rstrip("/")

class DBBridge:
    """
    prism-core의 공정 DB 툴(예: Postgres 래퍼)과 통신하는 브릿지.
    - /api/db/tables            : 테이블 목록
    - /api/db/sql (POST, JSON)  : SQL 실행 -> rows 반환 (서버 표준에 맞게 조정)
    """
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (base_url or DB_API_URL).rstrip("/")

    def list_tables(self) -> List[str]:
        if not self.base_url:
            raise RuntimeError("DB_API_URL 환경변수가 비어 있습니다.")
        url = f"{self.base_url}/tables"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        # 서버 포맷 가정: {"tables": ["a","b",...]}
        return data.get("tables", data)

    def sql(self, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if not self.base_url:
            raise RuntimeError("DB_API_URL 환경변수가 비어 있습니다.")
        url = f"{self.base_url}/sql"
        payload = {"sql": sql, "params": params or {}}
        r = requests.post(url, json=payload, timeout=30)
        r.raise_for_status()
        data = r.json()
        # 서버 포맷 가정: {"rows":[{...}, ...]}
        return data.get("rows", data)

    def fetch_timeseries(
        self,
        table: str,
        sensor_name: str,
        columns: List[str],
        start_ts: str,
        end_ts: str,
        ts_col: str = "timestamp",
        sensor_col: str = "sensor",
    ) -> pd.DataFrame:
        """
        시간구간 + 센서명으로 지정 컬럼 가져오기.
        """
        cols_sql = ", ".join([ts_col] + columns)
        sql = f"""
            SELECT {cols_sql}
            FROM {table}
            WHERE {sensor_col} = :sensor
              AND {ts_col} >= :start_ts
              AND {ts_col} <= :end_ts
            ORDER BY {ts_col} ASC
        """
        rows = self.sql(sql, params={"sensor": sensor_name, "start_ts": start_ts, "end_ts": end_ts})
        df = pd.DataFrame(rows)
        return df
