# main.py
from prism_prediction.models import *
import logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
import uuid
import os
from fastapi import FastAPI, Body, HTTPException, Query, Path
from pydantic import BaseModel, Field
import pandas as pd
import subprocess, sys, json
from fastapi.responses import HTMLResponse, RedirectResponse
import base64, hashlib, pathlib


# ── 추가: .env 로드 & 외부 API 호출 ─────────────────────────────────────────────
from dotenv import load_dotenv
import requests

# ── module.py의 함수 직접 사용 ────────────────────────────────────────────────
# module.py 내부에 def explain(feature_df, target_col, preds) -> dict
#                 def risk(feature_df, target_col, preds) -> dict
from module import explain as explain_module
from module import risk as risk_module


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _rid() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"


# ── 로거 ───────────────────────────────────────────────────────────────────────
logger = logging.getLogger("prism_prediction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)


# ── .env 로드 & ENV 읽기 ───────────────────────────────────────────────────────
load_dotenv()
LLM_API_URL = os.getenv("LLM_API_URL")   # 예: http://147.47.39.144:8000/api/agents
DB_API_URL  = os.getenv("DB_API_URL")    # 예: http://147.47.39.144:8000/api/db
logger.info(f"ENV Loaded: LLM_API_URL={LLM_API_URL}, DB_API_URL={DB_API_URL}")


# ── FastAPI ───────────────────────────────────────────────────────────────────
app = FastAPI(title="PRISM Prediction API", version="0.0.4")

# 루트 → /ui 로 바로 연결 (예쁜 UI)
@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")

# favicon(선택) 204로 막기
@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204, content="")


# ── 공용 스키마 ────────────────────────────────────────────────────────────────
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


# “오케스트레이션→예측” 스키마
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
        # 추가: 설명/리스크/액션
        explanation: Optional[Dict[str, Any]] = None
        risk: Optional[Dict[str, Any]] = None
        suggestedActions: Optional[List[str]] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata


# ── CSV 매핑 ──────────────────────────────────────────────────────────────────
CSV_BASE = "prism_prediction/Industrial_DB_sample"
SENSOR_TO_FILE = {
    "CMP": os.path.join(CSV_BASE, "SEMI_CMP_SENSORS_predict.csv"),
    # 필요 시 추가
}

def _notice(events: List[str], msg: str):
    logger.info(msg)
    events.append(f"[{_now_iso()}] {msg}")


# ── predictor(run.py) 호출 유틸 ────────────────────────────────────────────────
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
    run.py를 서브프로세스로 호출 → outputs/summary.json 읽어 반환.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    runpy_path = os.path.join(script_dir, "run.py")
    out_path = os.path.join(script_dir, "outputs", "summary.json")

    # 이전 결과 제거
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

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"predictor failed: {proc.stderr}")

    if not os.path.exists(out_path):
        tail = proc.stdout[-400:] if proc.stdout else ""
        raise HTTPException(status_code=500, detail=f"predictor did not produce {out_path}. tail={tail}")

    try:
        with open(out_path, "r") as f:
            summary = json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"cannot read {out_path}: {e}")

    return summary


# ── 제안 액션 룰(데모) ────────────────────────────────────────────────────────
def suggest_actions_by_risk(risk_dict: Dict[str, Any]) -> List[str]:
    level = (risk_dict or {}).get("riskLevel", "unknown")
    if level == "high":
        return ["increase_cooling_flow", "reduce_platen_rotation", "issue_operator_alert"]
    if level == "medium":
        return ["monitor_closely_15min", "schedule_maintenance_check"]
    if level == "low":
        return ["no_action_required"]
    return ["review_manually"]


# ── 예측 엔드포인트(파이프라인: 예측 → 설명 → 리스크/액션) ───────────────────
@app.post(
    "/api/v1/prediction/run-direct",
    response_model=DirectRunResponse,
    summary="오케스트레이션 JSON을 받아 Industrial DB에서 피처/타깃 구성 후 예측→설명→리스크/액션까지 반환",
    tags=["Prediction"]
)
def run_direct(body: DirectRunRequest):
    events: List[str] = []

    # 1) 수신
    _notice(events, f"요청 수신: taskId={body.taskId}, objective={body.objective}, timeRange={body.timeRange}, sensor={body.sensor_name}, target={body.target_cols}")

    # 2) CSV 경로
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

    # 4) 피처/타깃
    if n_cols < 13:
        raise HTTPException(status_code=400, detail="CSV 컬럼 수가 기대보다 적습니다(최소 13열 필요).")
    feature_df = df.iloc[:, 4:]   # 5번째부터
    feature_names = list(feature_df.columns)
    enc_in = feature_df.shape[1]

    target_col = body.target_cols
    if target_col not in feature_df.columns:
        raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col})이 CSV에 없습니다. 현재 피처 가능 컬럼: {feature_names}")
    target_idx_in_features = int(feature_df.columns.get_loc(target_col))

    _notice(events, f"피처 구성 완료: enc_in={enc_in}, feature_names={feature_names}")
    _notice(events, f"타깃 확인: target={target_col}, target_idx_in_features={target_idx_in_features}")

    # 5) 참고 알림
    _notice(events, f"timeRange={body.timeRange} 에 대한 예측 준비(데모는 전체 시퀀스 사용)")

    # 6) (옵션) DB/LLM API 연동 테스트
    if DB_API_URL:
        try:
            db_test = requests.get(f"{DB_API_URL.rstrip('/')}/tables", timeout=5)
            _notice(events, f"DB 연결 확인: status={db_test.status_code}")
        except Exception as e:
            _notice(events, f"DB 호출 실패: {e}")

    if LLM_API_URL:
        try:
            llm_agents = requests.get(LLM_API_URL, timeout=5)
            _notice(events, f"LLM 연결 확인: status={llm_agents.status_code}")
        except Exception as e:
            _notice(events, f"LLM 호출 실패: {e}")

    # 7) predictor(run.py)로 베스트 모델 선정/학습 요약
    _notice(events, "predictor(run.py) 실행 시작")
    predictor_summary = run_predictor_with_csv(
        csv_path=csv_path,
        feature_start_col=5,
        target_col_name=target_col,
        seq_len=48, label_len=24, pred_len=11,
        epochs=1, batch_size=8,
        models="Autoformer,DLinear,TimesNet,LightTS",
        device="cpu",
        auto_eval_idx=True
    )
    best_model = predictor_summary.get("best_model") or "Unknown"
    best_focus = predictor_summary.get("best_val_rmse_focus")
    _notice(events, f"predictor 완료: best_model={best_model}, best_val_rmse_focus={best_focus}")
    # BEST MODEL 경로
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_weight_path = os.path.join(script_dir, "outputs", best_model, "ckpt.pt")
    best_weight_path = (
        predictor_summary.get("best_weight_path")
        or predictor_summary.get("ckpt_path")
        or predictor_summary.get("checkpoint")
        or default_weight_path
    )
        
    # 8) (임시) 예측값
    series = pd.to_numeric(feature_df[target_col], errors="coerce").dropna().to_numpy()
    if series.size == 0:
        raise HTTPException(status_code=400, detail=f"타깃 컬럼({target_col})에 유효 숫자 데이터가 없습니다.")
    delta = float(series[-1] - series[-2]) if series.size >= 2 else 0.0
    pred_len = 11
    start = float(series[-1])
    preds = [round(start + (i+1)*delta, 4) for i in range(pred_len)]
    _notice(events, f"예측 수행 완료(나이브 외삽): pred_len={pred_len}, last={start}, delta={delta}")
    # 자율 제어에 전달하고자 하는 MODEL 가중치 및 METADATA
    autocontrol_payload = {
    "taskId": body.taskId,
    "best_model": best_model,
    "feature_names": feature_names,
    "input_time_range": body.timeRange,
    "output_offsets": list(range(1, pred_len + 1)),
    "pred_len": pred_len,
    "target_col": target_col,
    "prediction": preds,
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


    # 9) 설명(모듈 함수 호출)
    _notice(events, "변수 상관관계 계산 시작")
    try:
        explanation = explain_module(feature_df, target_col, preds)  # module.py의 함수
    except Exception as e:
        explanation = {"error": str(e)}
    _notice(events, f"설명 계산 완료: keys={list(explanation.keys())}")

    # 10) 리스크(모듈 함수 호출) + 액션
    _notice(events, "위험 평가 시작")
    try:
        risk_dict = risk_module(feature_df, target_col, preds)       # module.py의 함수
    except Exception as e:
        risk_dict = {"riskLevel": "unknown", "error": str(e), "exceedsThreshold": False}
    actions = suggest_actions_by_risk(risk_dict)
    _notice(events, f"위험 평가 완료: level={risk_dict.get('riskLevel')} → actions={actions}")

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
            "modelSelected": best_model,
            "prediction": preds,
            "explanation": explanation,
            "risk": risk_dict,
            "suggestedActions": actions,
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }

###### AutoControl 전달 부분 추가

AUTOCONTROL_DIR = pathlib.Path(os.path.dirname(os.path.abspath(__file__))) / "outputs" / "autocontrol"
AUTOCONTROL_DIR.mkdir(parents=True, exist_ok=True)

def _sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _maybe_inline_base64(path: str, do_inline: bool, limit_mb: int = 8) -> tuple[bool, str | None, str | None]:
    """return (inlined, base64_str, note)"""
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
    
class AutoControlSnapshotResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    class Data(BaseModel):
        taskId: str
        best_model: str
        feature_names: List[str]
        input_time_range: str
        output_offsets: List[int]
        pred_len: int
        target_col: str
        prediction: List[float]
        # weight 메타 + (선택) base64
        weight_path: Optional[str] = None
        weight_filename: Optional[str] = None
        weight_size_bytes: Optional[int] = None
        weight_sha256: Optional[str] = None
        weight_inlined: Optional[bool] = None
        weight_base64: Optional[str] = None
        note: Optional[str] = None
    data: Data
    class Metadata(BaseModel):
        timestamp: str
        request_id: str
    metadata: Metadata


@app.get("/api/v1/prediction/activate_autocontrol/{taskId}",
         response_model=AutoControlSnapshotResponse,
         tags=["Prediction"])
def activate_autocontrol_by_task(
    taskId: str = Path(...),
    inline_base64: bool = Query(False, description="작은 weight면 base64 인라인 포함")
):
    snap_path = AUTOCONTROL_DIR / f"best_model_{taskId}.json"
    if not snap_path.exists():
        raise HTTPException(status_code=404, detail=f"snapshot not found for taskId={taskId}. run /api/v1/prediction/run-direct first.")
    try:
        snap = json.load(open(snap_path, "r", encoding="utf-8"))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"snapshot read error: {e}")

    weight_path = snap.get("weight_path")
    weight_meta = {
        "weight_path": weight_path,
        "weight_filename": None,
        "weight_size_bytes": None,
        "weight_sha256": None,
        "weight_inlined": False,
        "weight_base64": None,
        "note": None,
    }

    if weight_path and os.path.exists(weight_path):
        try:
            weight_meta["weight_filename"] = os.path.basename(weight_path)
            weight_meta["weight_size_bytes"] = os.path.getsize(weight_path)
            weight_meta["weight_sha256"] = _sha256_of_file(weight_path)
            inlined, b64, note = _maybe_inline_base64(weight_path, inline_base64, limit_mb=8)
            weight_meta["weight_inlined"] = inlined
            weight_meta["weight_base64"] = b64
            weight_meta["note"] = note
        except Exception as e:
            weight_meta["note"] = f"weight meta error: {e}"
    else:
        weight_meta["note"] = ("weight path missing" if not weight_path else f"weight not found: {weight_path}")

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
            **weight_meta
        },
        "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
    }




# ── NL 스텁 ───────────────────────────────────────────────────────────────────
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


# ── 간단 UI (/ui) ─────────────────────────────────────────────────────────────
@app.get("/ui", response_class=HTMLResponse)
def prediction_ui():
    return """
<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>PRISM Prediction UI</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-50">
  <div class="max-w-5xl mx-auto p-6">
    <header class="mb-6">
      <h1 class="text-2xl font-bold">PRISM Prediction</h1>
      <p class="text-slate-600">오케스트레이션 JSON 없이 폼으로 예측 요청하고, 결과를 표/차트로 확인해요.</p>
    </header>

    <section class="bg-white rounded-2xl shadow p-6 mb-6">
      <form id="predForm" class="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label class="block text-sm text-slate-700 mb-1">Task ID</label>
          <input name="taskId" value="1" class="w-full border rounded-xl px-3 py-2 outline-none focus:ring" />
        </div>
        <div>
          <label class="block text-sm text-slate-700 mb-1">From Agent</label>
          <select name="fromAgent" class="w-full border rounded-xl px-3 py-2">
            <option value="orchestration" selected>orchestration</option>
            <option value="monitoring">monitoring</option>
            <option value="ui">ui</option>
            <option value="external">external</option>
          </select>
        </div>
        <div>
          <label class="block text-sm text-slate-700 mb-1">Objective</label>
          <select name="objective" class="w-full border rounded-xl px-3 py-2">
            <option value="prediction" selected>prediction</option>
            <option value="forecast">forecast</option>
            <option value="예측">예측</option>
          </select>
        </div>
        <div>
          <label class="block text-sm text-slate-700 mb-1">Time Range</label>
          <input name="timeRange" value="2025-08-20 09:00:00 - 09:10:00" class="w-full border rounded-xl px-3 py-2"/>
        </div>
        <div>
          <label class="block text-sm text-slate-700 mb-1">Sensor Name</label>
          <select name="sensor_name" class="w-full border rounded-xl px-3 py-2">
            <option value="CMP" selected>CMP</option>
          </select>
        </div>
        <div>
          <label class="block text-sm text-slate-700 mb-1">Target Column</label>
          <input name="target_cols" value="MOTOR_CURRENT" class="w-full border rounded-xl px-3 py-2"/>
        </div>
        <div class="md:col-span-2">
          <label class="block text-sm text-slate-700 mb-1">User Role</label>
          <input name="userRole" value="engineer" class="w-full border rounded-xl px-3 py-2"/>
        </div>
        <div class="md:col-span-2 flex gap-3">
          <button id="runBtn" class="px-4 py-2 rounded-xl bg-indigo-600 text-white hover:bg-indigo-700 transition">예측 실행</button>
          <span id="status" class="text-sm text-slate-600"></span>
        </div>
      </form>
    </section>

    <section id="resultCard" class="hidden bg-white rounded-2xl shadow p-6 mb-6">
      <h2 class="text-lg font-semibold mb-2">결과</h2>
      <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div>
          <h3 class="font-medium mb-2">요약</h3>
          <div class="text-sm text-slate-700 space-y-1" id="summary"></div>
        </div>
        <div>
          <h3 class="font-medium mb-2">예측 차트</h3>
          <canvas id="predChart" height="140"></canvas>
        </div>
      </div>
      <div class="mt-6">
        <h3 class="font-medium mb-2">이벤트 로그</h3>
        <ul id="events" class="list-disc pl-6 text-sm text-slate-700 space-y-1"></ul>
      </div>
      <div class="mt-6">
        <h3 class="font-medium mb-2">Raw JSON</h3>
        <pre id="raw" class="text-xs bg-slate-50 border rounded-xl p-3 overflow-auto"></pre>
      </div>
    </section>
  </div>

<script>
const form = document.getElementById('predForm');
const runBtn = document.getElementById('runBtn');
const statusEl = document.getElementById('status');
const resultCard = document.getElementById('resultCard');
const summaryEl = document.getElementById('summary');
const eventsEl = document.getElementById('events');
const rawEl = document.getElementById('raw');
let chart;

form.addEventListener('submit', async (e) => {
  e.preventDefault();
  statusEl.textContent = "요청 중...";
  runBtn.disabled = true;
  resultCard.classList.add('hidden');

  const fd = new FormData(form);
  const payload = {
    taskId: fd.get('taskId'),
    fromAgent: fd.get('fromAgent'),
    objective: fd.get('objective'),
    timeRange: fd.get('timeRange'),
    sensor_name: fd.get('sensor_name'),
    target_cols: fd.get('target_cols'),
    constraints: null,
    userRole: fd.get('userRole')
  };

  try {
    const res = await fetch('/api/v1/prediction/run-direct', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    statusEl.textContent = "";
    runBtn.disabled = false;

    if (!res.ok) {
      alert('에러: ' + (data.detail || res.statusText));
      return;
    }

    const d = data.data || {};
    summaryEl.innerHTML = `
      <div><b>code</b>: ${data.code}</div>
      <div><b>modelSelected</b>: ${d.modelSelected}</div>
      <div><b>target</b>: ${d.target_col} (idx ${d.target_idx_in_features})</div>
      <div><b>pred_len</b>: ${d.pred_len}</div>
      <div><b>csv</b>: ${d.csv_path}</div>
      <div><b>df</b>: ${d.df_info?.rows} rows × ${d.df_info?.cols} cols</div>
      <div><b>risk</b>: ${d.risk ? (d.risk.riskLevel || 'unknown') : 'n/a'}</div>
      <div><b>actions</b>: ${d.suggestedActions ? d.suggestedActions.join(', ') : 'n/a'}</div>
    `;

    eventsEl.innerHTML = '';
    (d.events || []).forEach(ev => {
      const li = document.createElement('li');
      li.textContent = ev;
      eventsEl.appendChild(li);
    });

    const preds = d.prediction || [];
    const labels = preds.map((_, i) => 't+' + (i+1));
    const ctx = document.getElementById('predChart').getContext('2d');
    if (chart) chart.destroy();
    chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [{
          label: 'Prediction',
          data: preds,
          borderWidth: 2,
          tension: 0.25
        }]
      },
      options: {
        responsive: true,
        plugins: { legend: { display: true }},
        scales: { x: { grid: { display: false }}, y: { grid: { color: '#eef2ff' }}}
      }
    });

    rawEl.textContent = JSON.stringify(data, null, 2);
    resultCard.classList.remove('hidden');

  } catch (err) {
    statusEl.textContent = "";
    runBtn.disabled = false;
    alert('요청 실패: ' + err);
  }
});
</script>
</body>
</html>
    """


# ── (옵션) 2~10번 오케스트레이션 호환용 간단 엔드포인트 스텁 ───────────────
# 필요 시 활성화; 여기서는 최소한의 호환만 유지(실제 파이프라인은 run-direct에서 수행)

@app.get("/api/v1/prediction/orchestration-request", tags=["Orchestration"])
def orchestration_request(taskId: str = Query(...)):
    return {
        "taskId": taskId,
        "fromAgent": "orchestration",
        "objective": "30분 내 과열 위험 탐지",
        "timeRange": "2025-07-17T13:00:00~13:30:00",
        "context": {"process_id":"line01","domain":"온도제어","constraints":[{"type":"time","value":"30분 이내"}]},
        "userRole": "engineer"
    }

@app.post("/api/v1/prediction/tasks", tags=["Prediction"])
def init_task(body: Dict[str, Any]):
    return {
        "code":"SUCCESS",
        "data":{"taskId":"abc123","requiredData":["sensor1","sensor5","camera1"],"status":"pending"},
        "metadata":{"timestamp":_now_iso(),"request_id":_rid()}
    }

@app.post("/api/v1/prediction/tasks/{taskId}/fetch-data", tags=["Orchestration"])
def fetch_data(taskId: str = Path(...), body: Dict[str, Any] = Body(...)):
    return {
        "code":"SUCCESS",
        "data":{"status":"data_ready","numRecords":1280},
        "metadata":{"timestamp":_now_iso(),"request_id":_rid()}
    }

@app.post("/api/v1/prediction/tasks/{taskId}/run", tags=["Orchestration"])
def run_task(taskId: str = Path(...), body: Dict[str, Any] = Body(...)):
    return {
        "code":"SUCCESS",
        "data":{"prediction":[1.2,1.4,1.6],"modelUsed":"TS-CNN","uncertainty":0.08},
        "metadata":{"timestamp":_now_iso(),"request_id":_rid()}
    }

@app.post("/api/v1/prediction/tasks/{taskId}/explain", tags=["Prediction"])
def explain_task(taskId: str = Path(...), body: Dict[str, Any] = Body(None)):
    # 데모: 고정 CSV/타깃 사용
    csv_path = SENSOR_TO_FILE["CMP"]
    df = pd.read_csv(csv_path)
    feature_df = df.iloc[:,4:]
    result = explain_module(feature_df, "MOTOR_CURRENT", preds=[])
    return {"code":"SUCCESS","data":result,"metadata":{"timestamp":_now_iso(),"request_id":_rid()}}

@app.post("/api/v1/prediction/tasks/{taskId}/risk-assess", tags=["Prediction"])
def risk_task(taskId: str = Path(...), body: Dict[str, Any] = Body(None)):
    csv_path = SENSOR_TO_FILE["CMP"]
    df = pd.read_csv(csv_path)
    feature_df = df.iloc[:,4:]
    r = risk_module(feature_df, "MOTOR_CURRENT", preds=[])
    return {"code":"SUCCESS","data":r,"metadata":{"timestamp":_now_iso(),"request_id":_rid()}}

@app.get("/api/v1/prediction/tasks/{taskId}/result-summary", tags=["Prediction"])
def result_summary(taskId: str = Path(...)):
    return {"code":"SUCCESS","data":{"summary":"위험 감지됨 - 냉각 조치 필요","historyComparison":"consistent","userView":"작업자용 요약 제공됨"},
            "metadata":{"timestamp":_now_iso(),"request_id":_rid()}}

@app.post("/api/v1/prediction/tasks/{taskId}/log", tags=["Prediction"])
def task_log(taskId: str = Path(...), body: Dict[str, Any] = Body(None)):
    return {"code":"SUCCESS","data":{"processLog":["step1","step2"],"timestamp":_now_iso()},
            "metadata":{"timestamp":_now_iso(),"request_id":_rid()}}

@app.post("/api/v1/prediction/tasks/{taskId}/interrupt", tags=["Prediction"])
def task_interrupt(taskId: str = Path(...), body: Dict[str, Any] = Body(...)):
    return {"code":"SUCCESS","data":{"status":"interrupted","message":"예측 중단 완료"},
            "metadata":{"timestamp":_now_iso(),"request_id":_rid()}}


# ── 로컬 실행 ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
