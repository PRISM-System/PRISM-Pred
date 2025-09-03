import logging, os, sys, json, uuid, pathlib, base64, hashlib, subprocess
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
import pandas as pd
from fastapi import FastAPI, Body, HTTPException, Query, Path, Response, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests
from llm_io import LLMBridge
from module import explain as explain_module
from module import risk as risk_module
import re
from ui import router as ui_router, mount_static

# 공통 헬퍼
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _rid() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"

# logging
logger = logging.getLogger("prism_prediction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

# env에서 환경 로드
load_dotenv()
LLM_API_URL = os.getenv("LLM_API_URL") 
logger.info(f"ENV Loaded: LLM_API_URL={LLM_API_URL}")

# Data load (이후 DB 스키마로 변경 예정)
CSV_BASE = "prism_prediction/Industrial_DB_sample/dataset_v2"
SENSOR_TO_FILE = {
    "CMP": os.path.join(CSV_BASE, "SEMI_CMP_SENSORS.csv"),
    "CVD" : os.path.join(CSV_BASE, "SEMI_CVD_SENSORS.csv"),
    "ETCH": os.path.join(CSV_BASE, "SEMI_ETCH_SENSORS.csv"),
    "ION": os.path.join(CSV_BASE, "SEMI_ION_SENSORS.csv"),
    "PHOTO" : os.path.join(CSV_BASE, "SEMI_PHOTO_SENSORS.csv"),
                         
}

# FastAPI 설정
app = FastAPI(
    title="PRISM Prediction Agent",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "Prediction", "description": "PRISM의 예측 에이전트입니다. 오케스트레이션에서 요청받은 자연어 질의에 대한 예측 및 필요 정보들을 자연어 응답으로 반환합니다."},
        {"name": "Debug", "description": "점검용 엔드포인트"},
        {"name": "UI", "description": "웹 UI"},
    ],
)

app.include_router(ui_router)
mount_static(app)

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui") # 홈페이지 들어올 경우 ui로 가게 설정 (변경 가능)

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# PRISM_Prediction API 요청/응답 규칙
class DirectRunResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, str]

class DirectSpec(BaseModel):
    taskId: str = Field("TaskId", example="TaskId_2O25")
    timeRange: str = Field("TimeRange_to_Predict", example="2025-08-20 09:00:00~09:10:00")
    sensor_name: str = Field("CMP", example="CMP")
    target_cols: str = Field("MOTOR_CURRENT", example="MOTOR_CURRENT")
    constraints: Optional[Any] = Field(None, example=None)
    userRole: Optional[str] = Field("engineer", example="engineer")

class DirectRunRequest(DirectSpec):
    fromAgent: Literal["orchestration","monitoring","ui","external"] = Field("orchestration", example="orchestration")
    objective: Literal["prediction","forecast","예측"] = Field("prediction", example="prediction")

    # Swagger 예시 한 번 더 노출
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "taskId": "2025",
                "timeRange": "2025-08-20 09:00:00~09:10:00",
                "sensor_name": "CMP",
                "target_cols": "MOTOR_CURRENT",
                "constraints": None,
                "userRole": "engineer",
                "fromAgent": "orchestration",
                "objective": "prediction"
            }]
        }
    }

# Predictor(run.py) 실행 
RUN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")
def run_predictor_with_csv(csv_path: str, target_col_name: str) -> Dict[str, Any]:
    """
    run.py 실행 → outputs/summary.json 반환
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runpy_path = os.path.join(script_dir, "run.py")
    out_path = os.path.join(script_dir, "outputs", "summary.json")
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass

    cmd = [
        sys.executable, runpy_path,
        "--csv_path", csv_path,
        "--feature_start_col", "5",
        "--target_col_name", target_col_name,
        "--seq_len", "48", "--label_len", "24", "--pred_len", "11",
        "--epochs", "1", "--batch_size", "8",
        "--models", "Autoformer,DLinear,TimesNet,LightTS",
        "--device", "cpu",
        "--auto_eval_idx"
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"predictor failed: {proc.stderr}")

    if not os.path.exists(out_path):
        tail = proc.stdout[-400:] if proc.stdout else ""
        raise HTTPException(status_code=500, detail=f"predictor did not produce {out_path}. tail={tail}")

    try:
        with open(out_path, "r") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cannot read {out_path}: {e}")

# 로깅 헬퍼
def _notice(events: List[str], msg: str):
    logger.info(msg)
    events.append(f"[{_now_iso()}] {msg}")

# AutoControl 전달부
AUTOCONTROL_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "outputs" / "autocontrol"
AUTOCONTROL_DIR.mkdir(parents=True, exist_ok=True)

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_inline_base64(path: str, do_inline: bool, limit_mb: int = 8) -> tuple[bool, str | None, str | None]:
    try:
        size = os.path.getsize(path)
        if do_inline and size <= limit_mb * 1024 * 1024:
            with open(path, "rb") as f:
                return True, base64.b64encode(f.read()).decode("ascii"), f"inlined base64 (<= {limit_mb}MB)"
        if do_inline and size > limit_mb * 1024 * 1024:
            return False, None, f"file too large to inline (> {limit_mb}MB); metadata only"
        return False, None, None
    except Exception as e:
        return False, None, f"inline failed: {e}"

# --- 중첩 모델 해제(스키마 안정화) ---
class AutoControlSnapshotData(BaseModel):
    taskId: str
    best_model: str
    feature_names: List[str]
    input_time_range: str
    output_offsets: List[int]
    pred_len: int
    target_col: str
    prediction: List[float]
    weight_path: Optional[str] = None
    weight_filename: Optional[str] = None
    weight_size_bytes: Optional[int] = None
    weight_sha256: Optional[str] = None
    weight_inlined: Optional[bool] = None
    weight_base64: Optional[str] = None
    note: Optional[str] = None

class AutoControlSnapshotMeta(BaseModel):
    timestamp: str
    request_id: str

class AutoControlSnapshotResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: AutoControlSnapshotData
    metadata: AutoControlSnapshotMeta

# ── (1) NL → JSON 추출 ───────────────────────────
class NLRequest(BaseModel):
    query: str = Field(..., example='화학기계연마 공정 센서의 MOTOR_CURRENT 컬럼의 2024-02-04 09:00:00~09:10:00 의 값들을 예측해줘')

class NLParsedResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: DirectSpec
    metadata: Dict[str, str]

llm = LLMBridge(base_url= os.getenv("OPENAI_BASE_URL"), model= os.getenv("OPENAI_MODEL") , api_key="EMPTY")

@app.post("/api/v1/prediction/request-processing", response_model=NLParsedResponse, tags=["Prediction"])
def nl_parse(body: NLRequest):
    events: List[str] = []
    _notice(events, f"[NL] query={body.query}")
    if LLM_API_URL:
        try:
            r = requests.get(LLM_API_URL, timeout=5)
            _notice(events, f"LLM 연결 확인: status={r.status_code}")
        except Exception as e:
            _notice(events, f"LLM 호출 실패(연결 필요): {e}")
    try:
        spec = llm._extract_json_from_text(body.query)
        return {
            "code": "SUCCESS",
            "data": spec,
            "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
        }
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"NL→JSON 실패: {e}")

# ── (2~6) run-direct: CSV 로딩 → 예측 → best model 저장/반환 → 위험도 → 기여도 → 전체 응답 ──
@app.post("/api/v1/prediction/run-direct", response_model=DirectRunResponse, tags=["Prediction"])
def run_direct(body: DirectRunRequest):
    events: List[str] = []
    _notice(events, f"요청 수신: taskId={body.taskId}, objective={body.objective}, timeRange={body.timeRange}, sensor={body.sensor_name}, target={body.target_cols}")

    # CSV 경로 확인
    csv_path = SENSOR_TO_FILE.get(body.sensor_name.upper())
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found for sensor={body.sensor_name}")

    # 2) CSV 로딩
    df = pd.read_csv(csv_path)
    n_rows, n_cols = int(df.shape[0]), int(df.shape[1])
    _notice(events, f"데이터 로딩 성공: rows={n_rows}, cols={n_cols}")

    if n_cols < 13:
        raise HTTPException(status_code=400, detail="CSV 컬럼 수가 기대보다 적습니다(최소 13열 필요).")

    feature_df = df.iloc[:, 4:]
    feature_names = list(feature_df.columns)
    enc_in = feature_df.shape[1]
    target_col = body.target_cols
    if target_col not in feature_df.columns:
        raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col})이 CSV에 없습니다.")
    target_idx_in_features = int(feature_df.columns.get_loc(target_col))
    _notice(events, f"피처 구성 완료: enc_in={enc_in}, target={target_col} (idx {target_idx_in_features})")

    # 3) predictor 실행(최적 모델 선택)
    _notice(events, "predictor(run.py) 실행 시작")
    predictor_summary = run_predictor_with_csv(csv_path=csv_path, target_col_name=target_col)
    best_model = predictor_summary.get("best_model") or "Unknown"
    best_focus = predictor_summary.get("best_val_rmse_focus")
    _notice(events, f"predictor 완료: best_model={best_model}, best_val_rmse_focus={best_focus}")

    # best weight 경로 추출
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_weight_path = os.path.join(script_dir, "outputs", best_model, "ckpt.pt")
    best_weight_path = (
        predictor_summary.get("best_weight_path")
        or predictor_summary.get("ckpt_path")
        or predictor_summary.get("checkpoint")
        or default_weight_path
    )

    # 3) 예측(데모: 나이브 외삽)
    series = pd.to_numeric(feature_df[target_col], errors="coerce").dropna().to_numpy()
    if series.size < 2:
        raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col}) 데이터가 부족(>=2)합니다.")
    delta = float(series[-1] - series[-2])
    pred_len = 11
    start = float(series[-1])
    # preds = [round(start + (i+1)*delta, 4) for i in range(pred_len)]
    preds = predictor_summary.get("prediction")
    if not preds or not isinstance(preds, list):
        series = pd.to_numeric(feature_df[target_col], errors="coerce").dropna().to_numpy()
        if series.size < 2:
            raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col}) 데이터가 부족(>=2)합니다.")
        delta = float(series[-1] - series[-2])
        pred_len = 11
        start = float(series[-1])
        preds = [round(start + (i+1)*delta, 4) for i in range(pred_len)]
    else:
        pred_len = predictor_summary.get("pred_len") or len(preds)
        _notice(events, f"예측 완료: pred_len={pred_len}, last={start}, delta={delta}")

    # 3-1) AutoControl 스냅샷 저장
    AUTOCONTROL_DIR.mkdir(parents=True, exist_ok=True)
    autocontrol_payload = {
        "taskId": body.taskId,
        "best_model": best_model,
        "feature_names": feature_names,
        "input_time_range": body.timeRange,
        "output_offsets": list(range(1, pred_len + 1)),
        "pred_len": pred_len,
        "target_col": target_col,
        "prediction": preds,
        "prediction_text": ", ".join(f"{v:.6f}" for v in preds), 
        "prediction_count": len(preds),
        "weight_path": best_weight_path,
        "saved_at": _now_iso(),
    }
    save_path = AUTOCONTROL_DIR / f"best_model_{body.taskId}.json"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(autocontrol_payload, f, ensure_ascii=False, indent=2)
        _notice(events, f"AutoControl 스냅샷 저장: {save_path}")
    except Exception as e:
        _notice(events, f"AutoControl 스냅샷 저장 실패: {e}")

    # 4) 위험도 평가
    _notice(events, "위험 평가 시작")
    try:
        risk_dict = risk_module(feature_df, target_col, preds)
    except Exception as e:
        risk_dict = {"riskLevel": "unknown", "error": str(e), "exceedsThreshold": False}
    _notice(events, f"위험 평가 완료: level={risk_dict.get('riskLevel')}")

    # 5) 변수 기여도(설명)
    _notice(events, "변수 기여도 계산 시작")
    try:
        explanation = explain_module(feature_df, target_col, preds)
    except Exception as e:
        explanation = {"error": str(e)}
    _notice(events, f"설명 완료: keys={list(explanation.keys())}")

    # 6) 전체 응답 조립
    data = {
        "taskId": body.taskId,
        "timeRange": body.timeRange,
        "sensor_name": body.sensor_name,
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
        "modelSelected": best_model,
        "prediction": preds,
        "risk": risk_dict,
        "explanation": explanation,
    }
    return {"code":"SUCCESS","data":data,"metadata":{"timestamp":_now_iso(),"request_id":_rid()}}

# ── 3-1) AutoControl 활성화용 ─────────────────────
@app.get("/api/v1/prediction/activate_autocontrol/{taskId}",
         response_model=AutoControlSnapshotResponse, tags=["Prediction"])
def activate_autocontrol_by_task(taskId: str = Path(...), inline_base64: bool = Query(False)):
    snap_path = AUTOCONTROL_DIR / f"best_model_{taskId}.json"
    if not snap_path.exists():
        raise HTTPException(status_code=404, detail=f"snapshot not found for taskId={taskId}. run /api/v1/prediction/run-direct first.")
    try:
        snap = json.load(open(snap_path, "r", encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"snapshot read error: {e}")

    weight_path = snap.get("weight_path")
    meta = {
        "weight_path": weight_path,
        "weight_filename": None, "weight_size_bytes": None, "weight_sha256": None,
        "weight_inlined": False, "weight_base64": None, "note": None
    }

    if weight_path and os.path.exists(weight_path):
        try:
            meta["weight_filename"] = os.path.basename(weight_path)
            meta["weight_size_bytes"] = os.path.getsize(weight_path)
            meta["weight_sha256"] = _sha256_of_file(weight_path)
            inlined, b64, note = _maybe_inline_base64(weight_path, inline_base64, limit_mb=8)
            meta["weight_inlined"] = inlined
            meta["weight_base64"] = b64
            meta["note"] = note
        except Exception as e:
            meta["note"] = f"weight meta error: {e}"
    else:
        meta["note"] = ("weight path missing" if not weight_path else f"weight not found: {weight_path}")

    return {
        "code": "SUCCESS",
        "data": {
            "taskId": snap["taskId"],
            "best_model": snap["best_model"],
            "feature_names": snap["feature_names"],
            "input_time_range": snap["input_time_range"],
            "output_offsets": snap["output_offsets"],
            "pred_len": snap["pred_len"],
            "target_col": snap["target_col"],
            "prediction": snap["prediction"],
            **meta
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

class NarrateResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@app.post("/api/v1/prediction/narrate_from_run", response_model=NarrateResponse, tags=["Prediction"])
def narrate_from_run(body: DirectRunRequest):
    events: List[str] = []
    try:
        _notice(events, f"NARRATE(run-first) start. LLM base_url={llm.base_url}, model={llm.model}")

     
        run_result = run_direct(body)   

        payload = run_result

        def _strip_hidden_reasoning(text: str) -> str: #llm 변경 시 삭제 가능
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
            return text.strip()
       

        text = llm._narrate_en(payload)
        text = _strip_hidden_reasoning(text)

        _notice(events, "Narration OK")
        return {
            "code": "SUCCESS",
            "data": {"narration": text, "events": events},
            "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
        }

    except Exception as e:
        err = str(e)
        _notice(events, f"Narration failed: {err}")
        # run_result 생성 단계에서 실패했을 수도 있으므로 안전하게 빈 dict 전달
        fb = llm._fallback_en(locals().get("run_result", {}))
        fb = _strip_hidden_reasoning(fb)
        return {
            "code": "ERROR",
            "data": {"narration": fb, "events": events, "error": err},
            "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
        }



# ── Debug/편의: 센서/컬럼 점검 ────────────────────
@app.get("/api/v1/prediction/sensors", tags=["Debug"])
def list_sensors():
    return {"sensors": sorted(SENSOR_TO_FILE.keys())}

@app.get("/api/v1/prediction/sensors/{sensor_name}/columns", tags=["Debug"])
def list_columns(sensor_name: str):
    csv_path = SENSOR_TO_FILE.get(sensor_name.upper())
    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found for sensor={sensor_name}")
    df = pd.read_csv(csv_path, nrows=1)
    return {
        "sensor": sensor_name.upper(),
        "csv_path": csv_path,
        "columns": list(df.columns),
        "feature_columns_from_5th": list(df.columns[4:]),
    }


# ── 로컬 실행 ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
