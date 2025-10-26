# module.py

from __future__ import annotations
import os, re, math, json, sys
from dataclasses import dataclass
from typing import Any, List, Optional, Dict, Union, Tuple

import pandas as pd

# ---- 공용 상수 ----
CSV_BASE = "prism_prediction/Industrial_DB_sample/dataset_v3/test_scenarios/test_data"
FEATURE_START_COL = 2  # <-- run.py의 --feature_start_col 3 과 반드시 동일!

SENSOR_TO_FILE = {
    # 자동차
    "AUTOMOTIVE_WELDING":      os.path.join(CSV_BASE, "automotive/automotive_welding_001.csv"),
    "AUTOMOTIVE_PAINTING":     os.path.join(CSV_BASE, "automotive/automotive_painting_002.csv"),
    "AUTOMOTIVE_PRESSING":     os.path.join(CSV_BASE, "automotive/automotive_press_003.csv"),
    "AUTOMOTIVE_ASSEMBLY":     os.path.join(CSV_BASE, "automotive/automotive_assembly_004.csv"),
    # 배터리
    "BATTERY_FORMATION":       os.path.join(CSV_BASE, "battery/battery_formation_001.csv"),
    "BATTERY_COATING":         os.path.join(CSV_BASE, "battery/battery_coating_002.csv"),
    "BATTERY_AGING":           os.path.join(CSV_BASE, "battery/battery_aging_003.csv"),
    "BATTERY_PRODUCTION":      os.path.join(CSV_BASE, "battery/battery_production_004.csv"),
    # 화학
    "CHEMICAL_REACTOR":        os.path.join(CSV_BASE, "chemical/chemical_reactor_001.csv"),
    "CHEMICAL_DISTILLATION":   os.path.join(CSV_BASE, "chemical/chemical_distillation_002.csv"),
    "CHEMICAL_REFINING":       os.path.join(CSV_BASE, "chemical/chemical_refining_003.csv"),
    "CHEMICAL_FULL":           os.path.join(CSV_BASE, "chemical/chemical_full_004.csv"),
    # 반도체
    "SEMICONDUCTOR_CMP":       os.path.join(CSV_BASE, "semiconductor/semiconductor_cmp_001.csv"),
    "SEMICONDUCTOR_ETCH":      os.path.join(CSV_BASE, "semiconductor/semiconductor_etch_002.csv"),
    "SEMICONDUCTOR_DEPOSITION":os.path.join(CSV_BASE, "semiconductor/semiconductor_deposition_003.csv"),
    "SEMICONDUCTOR_FULL":      os.path.join(CSV_BASE, "semiconductor/semiconductor_full_004.csv"),
    # 철강
    "STEEL_ROLLING":           os.path.join(CSV_BASE, "steel/steel_rolling_001.csv"),
    "STEEL_CONVERTER":         os.path.join(CSV_BASE, "steel/steel_converter_002.csv"),
    "STEEL_CASTING":           os.path.join(CSV_BASE, "steel/steel_casting_003.csv"),
    "STEEL_PRODUCTION":        os.path.join(CSV_BASE, "steel/steel_production_004.csv"),
}

# taskId 접두/키워드 → SENSOR_TO_FILE 키
TASK_TO_SENSOR_KEY: Dict[str, str] = {
    r"ETCH[_-]":      "SEMICONDUCTOR_ETCH",
    r"DEP[_-]":       "SEMICONDUCTOR_DEPOSITION",
    r"FAB[_-]":       "SEMICONDUCTOR_FULL",
    r"PAINT[_-]":     "AUTOMOTIVE_PAINTING",
    r"DIST[_-]":      "CHEMICAL_DISTILLATION",
    r"REFIN(ING)?[_-]": "CHEMICAL_REFINING",
    r"CMP[_-]":       "SEMICONDUCTOR_CMP",
    r"PRESS[_-]":       "AUTOMOTIVE_PRESSING",
    r"ASSEMBLY[_-]":   "AUTOMOTIVE_ASSEMBLY",
    # 필요시 계속 추가
}

def _as_float(x, default=None):
    try:
        return float(x) if x is not None else default
    except Exception:
        return default

def _to_str_list(x: Any) -> List[str]:
    if x is None: return []
    if isinstance(x, str):
        return [s.strip() for s in x.split(",") if s.strip()]
    if isinstance(x, (list, tuple)):
        return [str(s).strip() for s in x if str(s).strip()]
    return [str(x).strip()]

def _compute_pred_len(horizon_minutes: Optional[int], interval_minutes: Optional[int], default_len: int = 11) -> int:
    if horizon_minutes and interval_minutes:
        try:
            steps = int(math.floor(horizon_minutes / interval_minutes))
            return max(1, steps)
        except Exception:
            return default_len
    return default_len

@dataclass
class PredictionInputSpec:
    task_id: str
    time_range: Union[str, Dict[str, str], None]
    sensor_name: str                   # 표시용 (CSV 선택에는 사용하지 않음)
    target_cols_raw: Any
    feature_cols_raw: Any
    horizon_minutes: Optional[int]
    interval_minutes: Optional[int]
    confidence_level: Optional[float]

    # @classmethod
    # def from_body(cls, body: Any) -> "PredictionInputSpec":
    #     get = (lambda k, d=None:
    #            (getattr(body, k, None) if getattr(body, k, None) is not None
    #             else body.get(k, d) if isinstance(body, dict) else d))
    #     return cls(
    #         task_id=get("taskId", "") or "",
    #         time_range=get("timeRange"),
    #         sensor_name=str(get("sensor_name", "") or ""),
    #         target_cols_raw=get("target_cols"),
    #         feature_cols_raw=get("feature_cols"),
    #         horizon_minutes=get("prediction_horizon_minutes"),
    #         interval_minutes=get("prediction_interval_minutes"),
    #         confidence_level=_as_float(get("confidence_level"), 0.95),
    #     )
    @classmethod
    def from_body(cls, body: Any) -> "PredictionInputSpec":
        def _get(b, k, default=None):
            if isinstance(b, dict): return b.get(k, default)
            v = getattr(b, k, None); return v if v is not None else default

        def _to_list(x):
            if x is None: return []
            if isinstance(x, (list, tuple)): return [str(i) for i in x if str(i).strip()]
            if isinstance(x, str): return [p for p in (s.strip() for s in x.split(",")) if p]
            return [str(x)]

        target_cols = _get(body, "target_cols") or _get(body, "targets") or _get(body, "target")
        feature_cols = _get(body, "feature_cols") or _get(body, "features")

        return cls(
            task_id=_get(body, "taskId", "") or "",
            time_range=_get(body, "timeRange"),
            sensor_name=str(_get(body, "sensor_name", "") or ""),
            target_cols_raw=_to_list(target_cols),
            feature_cols_raw=_to_list(feature_cols),
            horizon_minutes=_get(body, "prediction_horizon_minutes"),
            interval_minutes=_get(body, "prediction_interval_minutes"),
            confidence_level=_as_float(_get(body, "confidence_level"), 0.95),
        )


def resolve_sensor_key_from_task(task_id: str) -> Optional[str]:
    if not task_id:
        return None
    for pat, key in TASK_TO_SENSOR_KEY.items():
        if re.search(pat, task_id, flags=re.I):
            return key
    return None

import logging
logger = logging.getLogger("prism_prediction")

def resolve_csv_path(spec: PredictionInputSpec) -> str:
    task_id = spec.task_id or ""
    guessed = resolve_sensor_key_from_task(task_id)
    logger.info("[RESOLVE] task_id=%s -> guessed_key=%s", task_id, guessed)

    if guessed and guessed in SENSOR_TO_FILE:
        path = SENSOR_TO_FILE[guessed]
        logger.info("[RESOLVE] key=%s -> path=%s (exists=%s)", guessed, path, os.path.exists(path))
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"Resolved key '{guessed}', but file not found: {path}")

    # --- fallback A: 환경변수 기본 키 ---
    default_key = os.getenv("PRISM_DEFAULT_SENSOR_KEY", "").strip()
    if default_key and default_key in SENSOR_TO_FILE:
        path = SENSOR_TO_FILE[default_key]
        logger.warning("[RESOLVE] FALLBACK env key=%s -> path=%s (exists=%s)", default_key, path, os.path.exists(path))
        if os.path.exists(path):
            return path

    # --- fallback B: sensor_name 힌트 ---
    sn = (spec.sensor_name or "").upper()
    hint = None
    if "CHAMBER_E" in sn:
        hint = "SEMICONDUCTOR_ETCH"
    elif "CHAMBER_D" in sn or "DEPOSITION" in sn:
        hint = "SEMICONDUCTOR_DEPOSITION"
    if hint and hint in SENSOR_TO_FILE:
        path = SENSOR_TO_FILE[hint]
        logger.warning("[RESOLVE] FALLBACK sensor hint=%s -> path=%s (exists=%s)", hint, path, os.path.exists(path))
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot resolve CSV from taskId='{task_id}'. No pattern matched in TASK_TO_SENSOR_KEY."
    )


@dataclass
class DataLoadResult:
    df: pd.DataFrame
    feature_df: pd.DataFrame
    feature_names: List[str]
    enc_in: int
    target_col: str
    target_idx_in_features: int
    pred_len: int
    offsets: List[int]
    confidence_level: Optional[float]

class DataLoader:
    def __init__(self, csv_path: str, spec: PredictionInputSpec):
        self.csv_path = csv_path
        self.spec = spec

    def load(self) -> DataLoadResult:
        if not self.csv_path or not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # run.py와 일치: feature는 3번 열부터 사용(0-based)
        feature_df = df.iloc[:, FEATURE_START_COL:]
        feature_names = list(feature_df.columns)
        enc_in = feature_df.shape[1]

        targets = _to_str_list(self.spec.target_cols_raw)
        if not targets:
            raise ValueError("target_cols(예측 목표 변수)가 비어 있습니다.")
        target_col = targets[0]  # 첫 번째만 모델에 사용
        if target_col not in feature_df.columns:
            raise ValueError(f"타깃 컬럼({target_col})이 feature 영역(columns[{FEATURE_START_COL}:])에 없습니다. 실제 컬럼들: {feature_df.columns.tolist()}")

        target_idx_in_features = int(feature_df.columns.get_loc(target_col))

        pred_len = _compute_pred_len(self.spec.horizon_minutes, self.spec.interval_minutes, default_len=11)
        offsets = list(range(1, pred_len + 1))

        return DataLoadResult(
            df=df,
            feature_df=feature_df,
            feature_names=feature_names,
            enc_in=enc_in,
            target_col=target_col,
            target_idx_in_features=target_idx_in_features,
            pred_len=pred_len,
            offsets=offsets,
            confidence_level=self.spec.confidence_level,
        )

# ---- 간단 explain/risk (데모 안전화) ----

def explain(feature_df: pd.DataFrame, target_col: str, preds) -> Dict[str, Any]:
    corr_s = feature_df.corr(numeric_only=True).get(target_col)
    important = []
    if corr_s is not None:
        important = [c for c in corr_s.abs().sort_values(ascending=False).index if c != target_col][:5]
    return {"importantFeatures": important, "method": "corr-proxy"}

def risk(feature_df: pd.DataFrame, target_col: str, preds) -> Dict[str, Any]:
    series = pd.to_numeric(feature_df[target_col], errors="coerce").dropna()
    if series.size < 5 or not preds:
        return {"riskLevel": "unknown", "exceedsThreshold": False}
    try:
        z = (float(preds[-1]) - series.mean()) / (series.std() or 1.0)
        if abs(z) > 2.0: level = "high"
        elif abs(z) > 1.0: level = "medium"
        else: level = "low"
        return {"riskLevel": level, "exceedsThreshold": level in ("medium","high")}
    except Exception:
        return {"riskLevel": "unknown", "exceedsThreshold": False}
