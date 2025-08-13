# prediction_main.py
import logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
import uuid

from fastapi import FastAPI, Query, Path, Body, HTTPException
from pydantic import BaseModel, Field


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _rid() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"

logger = logging.getLogger("prism_prediction_stub")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

# -------------------------
# FastAPI
# -------------------------
app = FastAPI(title="PRISM Prediction API (Stub)", version="0.0.1")

@app.get("/")
def root():
    return {"message": "PRISM Prediction API (stub) alive"}

# -------------------------
# 스키마 (요청/응답)
# -------------------------

# 1) Orchestration Request (GET)
class OrchestrationRequestResponse(BaseModel):
    taskId: str
    fromAgent: Literal["orchestration","monitoring","ui","external"] = "orchestration"
    objective: str
    timeRange: str
    class Context(BaseModel):
        process_id: str
        domain: Optional[str] = None
        constraints: Optional[List[Dict[str, str]]] = None
    context: Optional[Context] = None
    userRole: Optional[str] = None

# 2) Create Task (POST)
class CreatePredictionTaskRequest(BaseModel):
    taskObjective: str
    timeRange: str
    userRole: Optional[str] = "engineer"

class CreatePredictionTaskResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        taskId: str
        requiredData: Optional[List[str]] = None
        status: Literal["pending","initialized","failed"] = "pending"
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# 3) Fetch Data (POST)
class FetchDataRequest(BaseModel):
    dataSources: List[str]
    class Preprocessing(BaseModel):
        missing: Optional[str] = None
        scaling: Optional[str] = None
    preprocessing: Optional[Preprocessing] = None

class FetchDataResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        status: Literal["data_ready","fetched","preprocessed","failed"] = "data_ready"
        numRecords: Optional[int] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# 4) Run (POST)
class RunPredictionRequest(BaseModel):
    modelPreference: Optional[str] = "auto"

class RunPredictionResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        prediction: List[float]
        modelUsed: str
        uncertainty: Optional[float] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# 5) Explain (POST)
class ExplainPredictionRequest(BaseModel):
    method: Optional[str] = "SHAP"

class ExplainPredictionResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        importantFeatures: List[str]
        method: str
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# 6) Risk Assess (POST)
class RiskAssessRequest(BaseModel):
    thresholds: Optional[Dict[str, float]] = None
    policy: Optional[str] = None

class RiskAssessResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        riskLevel: Literal["low","medium","high","위험","주의","안전","high"] = "high"
        exceedsThreshold: bool = True
        suggestedActions: Optional[List[str]] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# 7) Result Summary (GET)
class ResultSummaryResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        summary: str
        historyComparison: Optional[str] = None
        userView: Optional[str] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata

# 8) NL → CSV → 예측(더미) (POST)
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

# -------------------------
# 엔드포인트 (전부 가짜 응답)
# -------------------------

@app.get(
    "/api/v1/prediction/orchestration-request",
    response_model=OrchestrationRequestResponse,
    summary="오케스트레이션 요청 수용(Stub)",
    tags=["Prediction"]
)
def get_orchestration_request(
    task_id: str = Query(..., alias="taskId"),
    from_agent: Literal["orchestration","monitoring","ui","external"] = Query("orchestration", alias="fromAgent"),
    objective: str = Query("30분 내 과열 위험 탐지", alias="objective"),
    time_range: str = Query("2025-07-17T13:00:00~13:30:00", alias="timeRange"),
    process_id: str = Query("SEMI_CMP_SENSORS", alias="processId"),
    domain: str = Query("온도제어", alias="domain"),
    user_role: str = Query("engineer", alias="userRole"),
):
    logger.info(f"[orchestration-request] taskId={task_id}")
    return {
        "taskId": task_id,
        "fromAgent": from_agent,
        "objective": objective,
        "timeRange": time_range,
        "context": {"process_id": process_id, "domain": domain, "constraints": [{"type":"time","value":"30분 이내"}]},
        "userRole": user_role,
    }

@app.post(
    "/api/v1/prediction/tasks",
    response_model=CreatePredictionTaskResponse,
    summary="예측 요청 초기화(Stub)",
    tags=["Prediction"]
)
def create_prediction_task(body: CreatePredictionTaskRequest = Body(...)):
    logger.info(f"[create-task] body={body}")
    return {
        "code": "SUCCESS",
        "data": {
            "taskId": f"task_{uuid.uuid4().hex[:6]}",
            "requiredData": ["sensor1","sensor5","camera1"],
            "status": "pending",
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

@app.post(
    "/api/v1/prediction/tasks/{task_id}/fetch-data",
    response_model=FetchDataResponse,
    summary="데이터 수집 및 전처리(Stub)",
    tags=["Prediction"]
)
def fetch_data_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: FetchDataRequest = Body(...)
):
    logger.info(f"[fetch-data] taskId={task_id} body={body}")
    return {
        "code": "SUCCESS",
        "data": {"status": "data_ready", "numRecords": 1280},
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

@app.post(
    "/api/v1/prediction/tasks/{task_id}/run",
    response_model=RunPredictionResponse,
    summary="최적 모델 선택 및 예측 실행(Stub)",
    tags=["Prediction"]
)
def run_prediction_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: RunPredictionRequest = Body(...)
):
    logger.info(f"[run] taskId={task_id} body={body}")
    return {
        "code": "SUCCESS",
        "data": {
            "prediction": [1.2, 1.4, 1.6],
            "modelUsed": "TS-CNN",
            "uncertainty": 0.08
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

@app.post(
    "/api/v1/prediction/tasks/{task_id}/explain",
    response_model=ExplainPredictionResponse,
    summary="예측 결과 설명(Stub)",
    tags=["Prediction"]
)
def explain_prediction_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: ExplainPredictionRequest = Body(...)
):
    logger.info(f"[explain] taskId={task_id} body={body}")
    return {
        "code": "SUCCESS",
        "data": {"importantFeatures": ["sensor1_spike","sensor5_drop"], "method": body.method or "SHAP"},
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

@app.post(
    "/api/v1/prediction/tasks/{task_id}/risk-assess",
    response_model=RiskAssessResponse,
    summary="리스크 평가(Stub)",
    tags=["Prediction"]
)
def risk_assess_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: RiskAssessRequest = Body(None)
):
    logger.info(f"[risk-assess] taskId={task_id} body={body}")
    return {
        "code": "SUCCESS",
        "data": {
            "riskLevel": "high",
            "exceedsThreshold": True,
            "suggestedActions": ["increase_cooling_flow"]
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

@app.get(
    "/api/v1/prediction/tasks/{task_id}/result-summary",
    response_model=ResultSummaryResponse,
    summary="예측 결과 요약(Stub)",
    tags=["Prediction"]
)
def get_result_summary(
    task_id: str = Path(..., description="예측 taskId")
):
    logger.info(f"[result-summary] taskId={task_id}")
    return {
        "code": "SUCCESS",
        "data": {
            "summary": "위험 감지됨 - 냉각 조치 필요",
            "historyComparison": "consistent",
            "userView": "작업자용 요약 제공됨"
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

@app.post(
    "/api/v1/prediction/nl",
    response_model=NLResponse,
    summary="자연어 질의로 예측 실행(Stub: 11개 값)",
    tags=["Prediction"]
)
def predict_by_nl(body: NLRequest = Body(...)):
    logger.info(f"[nl] query={body.query}")

    # 완전 더미: 11개의 고정 값 반환(테스트용)
    preds = [214.0 + i*0.1 for i in range(11)]

    return {
        "code": "SUCCESS",
        "data": {
            "prediction": preds,
            "length": len(preds),
            "note": "stub response (no real model)"
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

# -------------------------
# 로컬 실행
# -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prediction_main:app", host="0.0.0.0", port=8001, reload=True)
