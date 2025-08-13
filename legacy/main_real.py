# prediction_main.py
import logging
from datetime import datetime
from typing import List, Optional, Dict, Any, Literal

from fastapi import FastAPI, Query, Path, Body
from pydantic import BaseModel, Field

# 스텁(Task) 모듈: 
from prism_prediction.modules.task import (
    prediction_orchestration_request,
    prediction_create_task,
    prediction_fetch_data,
    prediction_run,
    prediction_explain,
    prediction_risk_assess,
    prediction_result_summary,
)

# -----------------------------------------------------------------------------
# 로거 설정
# -----------------------------------------------------------------------------
logger = logging.getLogger("prism_prediction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s')
handler.setFormatter(formatter)
if not logger.handlers:
    logger.addHandler(handler)

# -----------------------------------------------------------------------------
# FastAPI 앱
# -----------------------------------------------------------------------------
app = FastAPI(title="PRISM Prediction API", version="0.1.0")

@app.get("/")
def read_root():
    logger.info("Prediction root endpoint accessed.")
    return {"message": "PRISM Prediction API alive"}

# -----------------------------------------------------------------------------
# Pydantic 스키마 (요청/응답)
#  - 스텁 함수들이 반환하는 JSON 구조와 표 예시 반영
# -----------------------------------------------------------------------------

# 1) Orchestration -> Request 수용 (GET)
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
    # 확장 필드 예: modelName: Optional[str] = None

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
    method: Optional[str] = "SHAP"  # 스텁은 path의 task_id만 필요하지만 확장 여지 보유

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
    # 스텁은 파라미터 불필요하지만 확장 여지 남김
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

# -----------------------------------------------------------------------------
# 라우팅 (모니터링 main.py 스타일로 1:1 매핑)
# -----------------------------------------------------------------------------

# 1) 오케스트레이션 요청 수용
@app.get(
    "/api/v1/prediction/orchestration-request",
    response_model=OrchestrationRequestResponse,
    summary="오케스트레이션 요청 수용",
    tags=["Prediction"]
)
def get_orchestration_request(
    task_id: str = Query(..., alias="taskId", description="오케스트레이션이 전달한 taskId"),
    from_agent: Literal["orchestration","monitoring","ui","external"] = Query("orchestration", alias="fromAgent"),
    objective: str = Query("30분 내 과열 위험 탐지", alias="objective"),
    time_range: str = Query("2025-07-17T13:00:00~13:30:00", alias="timeRange"),
    process_id: str = Query("SEMI_CMP_SENSORS", alias="processId"),
    domain: str = Query("온도제어", alias="domain"),
    user_role: str = Query("engineer", alias="userRole"),
):
    logger.info(f"[orchestration-request] taskId={task_id}")
    res = prediction_orchestration_request(
        task_id=task_id,
        from_agent=from_agent,
        objective=objective,
        time_range=time_range,
        process_id=process_id,
        domain=domain,
        user_role=user_role
    )
    return res

# 2) 예측 요청 초기화
@app.post(
    "/api/v1/prediction/tasks",
    response_model=CreatePredictionTaskResponse,
    summary="예측 요청 초기화",
    tags=["Prediction"]
)
def create_prediction_task(body: CreatePredictionTaskRequest = Body(...)):
    logger.info(f"[create-task] body={body}")
    res = prediction_create_task(
        task_objective=body.taskObjective,
        time_range=body.timeRange,
        user_role=body.userRole or "engineer",
    )
    return res

# 3) 데이터 수집 및 전처리
@app.post(
    "/api/v1/prediction/tasks/{task_id}/fetch-data",
    response_model=FetchDataResponse,
    summary="데이터 수집 및 전처리",
    tags=["Prediction"]
)
def fetch_data_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: FetchDataRequest = Body(...)
):
    logger.info(f"[fetch-data] taskId={task_id} body={body}")
    res = prediction_fetch_data(
        task_id=task_id,
        data_sources=body.dataSources,
        preprocessing=(body.preprocessing.dict() if body.preprocessing else None)
    )
    return res

# 4) 최적 모델 선택 및 예측 실행
@app.post(
    "/api/v1/prediction/tasks/{task_id}/run",
    response_model=RunPredictionResponse,
    summary="최적 모델 선택 및 예측 실행",
    tags=["Prediction"]
)
def run_prediction_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: RunPredictionRequest = Body(...)
):
    logger.info(f"[run] taskId={task_id} body={body}")
    res = prediction_run(
        task_id=task_id,
        model_preference=(body.modelPreference or "auto")
    )
    return res

# 5) 예측 결과 설명
@app.post(
    "/api/v1/prediction/tasks/{task_id}/explain",
    response_model=ExplainPredictionResponse,
    summary="예측 결과 설명",
    tags=["Prediction"]
)
def explain_prediction_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: ExplainPredictionRequest = Body(...)
):
    logger.info(f"[explain] taskId={task_id} body={body}")
    res = prediction_explain(
        task_id=task_id,
        important_features=None,
        method=(body.method or "SHAP")
    )
    return res

# 6) 리스크 평가
@app.post(
    "/api/v1/prediction/tasks/{task_id}/risk-assess",
    response_model=RiskAssessResponse,
    summary="리스크 평가",
    tags=["Prediction"]
)
def risk_assess_for_task(
    task_id: str = Path(..., description="예측 taskId"),
    body: RiskAssessRequest = Body(...)
):
    logger.info(f"[risk-assess] taskId={task_id} body={body}")
    res = prediction_risk_assess(
        task_id=task_id,
        risk_level="high",
        exceeds_threshold=True,
        suggested_actions=["increase_cooling_flow"]
    )
    return res

# 7) 결과 요약
@app.get(
    "/api/v1/prediction/tasks/{task_id}/result-summary",
    response_model=ResultSummaryResponse,
    summary="예측 결과 요약",
    tags=["Prediction"]
)
def get_result_summary(
    task_id: str = Path(..., description="예측 taskId")
):
    logger.info(f"[result-summary] taskId={task_id}")
    res = prediction_result_summary(task_id=task_id)
    return res

# -----------------------------------------------------------------------------
# 로컬 실행
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("prediction_main:app", host="0.0.0.0", port=8001, reload=True)
