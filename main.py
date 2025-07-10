from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List

app = FastAPI(
    title="AI Prediction Agent",
    description="자율 제조용 예측 AI 에이전트 (명세서 기반 구현)",
    version="1.0"
)

# ------------------------
# 요청/응답 모델 정의
# ------------------------
class PredictionRequest(BaseModel):
    task_description: str
    target_variable: str
    time_horizon: str
    data_sources: List[str]
    preprocessing: Optional[str] = None
    selected_model: Optional[str] = None

class PredictionResponse(BaseModel):
    prediction_result: dict
    key_contributing_variables: List[str]
    risk_level: str
    recommended_actions: List[str]
    uncertainty: float
    confidence_interval: str

# ------------------------
# 예측 수행 함수 (mock)
# ------------------------
def mock_predict(request: PredictionRequest):
    result = {
        "target_variable": request.target_variable,
        "predicted_value": 78.5,
        "time_horizon": request.time_horizon,
        "status": "normal"
    }
    key_vars = ["Sensor #2", "Sensor #5"]
    risk = "LOW"
    actions = ["No immediate action required"]
    uncertainty = 0.03
    ci = "±5%"

    return PredictionResponse(
        prediction_result=result,
        key_contributing_variables=key_vars,
        risk_level=risk,
        recommended_actions=actions,
        uncertainty=uncertainty,
        confidence_interval=ci
    )

# ------------------------
# 라우트: 예측 요청
# ------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    response = mock_predict(request)
    return response

# ------------------------
# 라우트: 상태 조회
# ------------------------
@app.get("/status")
async def get_status():
    return JSONResponse(content={"status": "Prediction running", "progress": "60%"})

# ------------------------
# 라우트: 로그 조회
# ------------------------
@app.get("/logs")
async def get_logs():
    logs = [{"time": "2025-07-10", "message": "Prediction completed", "risk": "LOW"}]
    return logs

# ------------------------
# 라우트: 파일 업로드
# ------------------------
@app.post("/upload-data/")
async def upload_file(file: UploadFile = File(...)):
    return {"filename": file.filename, "message": "File received"}

# ------------------------
# 라우트: 시나리오 입력
# ------------------------
@app.post("/scenario/")
async def submit_scenario(description: str = Form(...)):
    return {"message": f"Scenario received: {description}"}
