# main.py
from prism_prediction.models import *
#from AGI.legacy import predictor  # (현재는 직접 사용 안하지만, 패키지 초기화 시 필요하면 유지)
import logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
import uuid
import os
from fastapi import FastAPI, Query, Path, Body, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import subprocess, sys, json


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _rid() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"

logger = logging.getLogger("prism_prediction_demo")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

# --------------------------------------------------
# FastAPI
# --------------------------------------------------
app = FastAPI(title="PRISM Prediction API (demo)", version="0.0.3")

@app.get("/")
def root():
    return {"message": "PRISM Prediction API alive"}

# --------------------------------------------------
# 공용 스키마
# --------------------------------------------------
class NLRequest(BaseModel):
    query: str = Field(..., example='화학기계연마 공정 센서의 MOTOR_CURRENT 컬럼의 2024-02-04 09:00:00~09:10:00 의 값들을 예측해줘')
class NLResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        prediction: List[float]
        length: int
        note: Optional[str] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# --------------------------------------------------
# “오케스트레이션→예측” 스키마 (요청/응답)
# --------------------------------------------------
class DirectRunRequest(BaseModel):
    taskId: str
    fromAgent: Literal["orchestration","monitoring","ui","external"] = "orchestration"
    objective: Literal["prediction","forecast","예측"] = "prediction"
    timeRange: str = Field(..., example="2025-08-20 09:00:00 - 09:10:00")
    sensor_name: str = Field(..., example="CMP")
    target_cols: str = Field(..., example="MOTOR_CURRENT")
    constraints: Optional[Any] = None
    userRole: Optional[str] = "engineer"

class DirectRunResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        taskId: str
        accepted: bool
        events: List[str]
        csv_path: str
        df_info: Dict[str, int]
        features_start_col_index_1based: int
        feature_names: List[str]
        target_col: str
        target_idx_in_features: int
        enc_in: int
        pred_len: int
        modelSelected: str
        prediction: List[float]
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# --------------------------------------------------
# 유틸: 센서명 → CSV 경로 매핑
# --------------------------------------------------
CSV_BASE = "prism_prediction/Industrial_DB_sample"
SENSOR_TO_FILE = {
    "CMP": os.path.join(CSV_BASE, "SEMI_CMP_SENSORS_predict.csv"),
    # 필요 시 여기에 다른 센서도 추가 가능
}

def _notice(events: List[str], msg: str):
    logger.info(msg)
    events.append(f"[{_now_iso()}] {msg}")

# --------------------------------------------------
# predictor(run.py) 호출 유틸
# --------------------------------------------------
RUN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")

def run_predictor_with_csv(
    csv_path: str,
    feature_start_col: int,
    target_col_name: str,
    seq_len: int,
    label_len: int,
    pred_len: int,
    epochs: int,
    batch_size: int,
    models: str,
    device: str = "cpu",
    auto_eval_idx: bool = True,
) -> Dict[str, Any]:
    """
    run.py를 서브프로세스로 호출 → run.py가 저장한 outputs/summary.json을 읽어 반환.
    stdout 파싱은 버리고, 파일을 진실의 원천으로 사용.
    """
    # run.py가 있는 폴더 기준으로 실행/읽기 (상대경로 문제 예방)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runpy_path = os.path.join(script_dir, "run.py")
    out_path = os.path.join(script_dir, "outputs", "summary.json")

    # 기존 파일이 남아있으면 지워서 혼동 방지
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass

    cmd = [
        sys.executable, runpy_path,
        "--csv_path", csv_path,
        "--feature_start_col", str(feature_start_col),
        "--target_col_name", target_col_name,
        "--seq_len", str(seq_len),
        "--label_len", str(label_len),
        "--pred_len", str(pred_len),
        "--epochs", str(epochs),
        "--batch_size", str(batch_size),
        "--models", models,
        "--device", device,
    ]
    if auto_eval_idx:
        cmd.append("--auto_eval_idx")

    # run.py를 run.py 위치에서 실행 (cwd 지정)
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

    if proc.returncode != 0:
        # run.py 오류면 stderr 그대로 노출
        raise HTTPException(status_code=500, detail=f"predictor failed: {proc.stderr}")

    # 파일을 읽어 JSON 로드 (stdout은 참고만)
    if not os.path.exists(out_path):
        # 혹시 파일 생성이 안 된 예외 케이스를 대비해 stdout 일부를 에러 메시지로 남김
        tail = proc.stdout[-400:] if proc.stdout else ""
        raise HTTPException(status_code=500, detail=f"predictor did not produce {out_path}. tail={tail}")

    try:
        with open(out_path, "r") as f:
            summary = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cannot read {out_path}: {e}")

    return summary


# --------------------------------------------------
# 오케스트레이션 JSON → CSV 읽고 → predictor 실행 → (지금은) 나이브 예측값 반환
# --------------------------------------------------
@app.post(
    "/api/v1/prediction/run-direct",
    response_model=DirectRunResponse,
    summary="오케스트레이션 JSON을 받아 Industrial DB에서 피처/타깃 구성 후 예측값 반환",
    tags=["Prediction"]
)
def run_direct(body: DirectRunRequest):
    events: List[str] = []

    # 1) 수신 ACK
    _notice(events, f"요청 수신: taskId={body.taskId}, objective={body.objective}, timeRange={body.timeRange}, sensor={body.sensor_name}, target={body.target_cols}")

    # 2) CSV 경로 결정
    csv_path = SENSOR_TO_FILE.get(body.sensor_name.upper())
    if not csv_path:
        raise HTTPException(status_code=400, detail=f"지원하지 않는 sensor_name: {body.sensor_name}")
    if not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV 파일을 찾을 수 없음: {csv_path}")
    _notice(events, f"CSV 탐색 완료: {csv_path}")

    # 3) CSV 로딩
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV 로딩 실패: {e}")
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    _notice(events, f"CSV 로딩 성공: rows={n_rows}, cols={n_cols}")

    # 4) 예측에 사용할 컬럼(시간 무시): 5번째부터 끝까지 피처로 사용
    if n_cols < 13:
        raise HTTPException(status_code=400, detail="CSV 컬럼 수가 기대보다 적습니다(최소 13열 필요).")
    feature_df = df.iloc[:, 4:]   # 5번째 컬럼부터
    feature_names = list(feature_df.columns)
    enc_in = feature_df.shape[1]

    target_col = body.target_cols
    if target_col not in feature_df.columns:
        raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col})이 CSV에 없습니다. 현재 피처 가능 컬럼: {feature_names}")
    target_idx_in_features = int(feature_df.columns.get_loc(target_col))

    _notice(events, f"피처 구성 완료: enc_in={enc_in}, feature_names={feature_names}")
    _notice(events, f"타깃 확인: target={target_col}, target_idx_in_features={target_idx_in_features}")

    # 5) timeRange 알림
    _notice(events, f"timeRange={body.timeRange} 에 대한 예측을 준비 (현 데모는 전체 시퀀스를 사용)")

    # 6-a) predictor(run.py)로 베스트 모델 선정/학습 요약
    _notice(events, "predictor(run.py) 실행 시작")
    predictor_summary = run_predictor_with_csv(
        csv_path=csv_path,
        feature_start_col=5,
        target_col_name=target_col,
        seq_len=48, label_len=24, pred_len=11,
        epochs=1, batch_size=8,
        models="Autoformer,DLinear,TimesNet,LightTS",  # 필요 시 "Autoformer,DLinear,TimesNet,LightTS,SegRNN"
        device="cpu",
        auto_eval_idx=True
    )
    best_model = predictor_summary.get("best_model") or "Unknown"
    best_focus = predictor_summary.get("best_val_rmse_focus")
    _notice(events, f"predictor 완료: best_model={best_model}, best_val_rmse_focus={best_focus}")

    # 6-b) (임시) 예측값은 나이브 외삽으로 11개 생성
    series = pd.to_numeric(feature_df[target_col], errors="coerce").dropna().to_numpy()
    if series.size == 0:
        raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col})에 유효 숫자 데이터가 없습니다.")
    delta = float(series[-1] - series[-2]) if series.size >= 2 else 0.0
    pred_len = 11
    start = float(series[-1])
    preds = [round(start + (i+1)*delta, 4) for i in range(pred_len)]
    _notice(events, f"예측 수행 완료(나이브 외삽): pred_len={pred_len}, last={start}, delta={delta}")

    return {
        "code": "SUCCESS",
        "data": {
            "taskId": body.taskId,
            "accepted": True,
            "events": events,
            "csv_path": csv_path,
            "df_info": {"rows": n_rows, "cols": n_cols},
            "features_start_col_index_1based": 5,
            "feature_names": feature_names,
            "target_col": target_col,
            "target_idx_in_features": target_idx_in_features,
            "enc_in": enc_in,
            "pred_len": pred_len,
            "modelSelected": best_model,     # predictor가 고른 베스트 모델
            "prediction": preds,             # (지금은) 나이브 예측 11개
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

# --------------------------------------------------
# 기존 NL 스텁(원하면 그대로 유지)
# --------------------------------------------------
@app.post(
    "/api/v1/prediction/nl",
    response_model=NLResponse,
    summary="자연어 질의로 예측 실행(Stub: 11개 값)",
    tags=["Prediction"]
)
def predict_by_nl(body: NLRequest = Body(...)):
    logger.info(f"[nl] query={body.query}")
    preds = [214.0 + i*0.1 for i in range(11)]
    return {
        "code": "SUCCESS",
        "data": {"prediction": preds, "length": len(preds), "note": "stub response (no real model)"},
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

# --------------------------------------------------
# 로컬 실행
# --------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
