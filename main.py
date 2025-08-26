# main.py (정리판)
import logging, os, sys, json, uuid, pathlib, base64, hashlib, subprocess
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, Body, HTTPException, Query, Path
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import requests

from llm_io import LLMBridge
# module.py: explain(), risk() 사용
from module import explain as explain_module
from module import risk as risk_module

# ── 유틸 ───────────────────────────────────────────
def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
def _rid() -> str:
    return f"req_{uuid.uuid4().hex[:8]}"

# ── 로거 ───────────────────────────────────────────
logger = logging.getLogger("prism_prediction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

# ── .env ───────────────────────────────────────────
load_dotenv()
LLM_API_URL = os.getenv("LLM_API_URL")   # 예: http://147.47.39.144:8000/api/agents (헬스체크용)
logger.info(f"ENV Loaded: LLM_API_URL={LLM_API_URL}")

# ── CSV 매핑 ───────────────────────────────────────
CSV_BASE = "prism_prediction/Industrial_DB_sample"
SENSOR_TO_FILE = {
    "CMP": os.path.join(CSV_BASE, "SEMI_CMP_SENSORS_predict.csv"),
}

# ── FastAPI ────────────────────────────────────────
app = FastAPI(title="PRISM Prediction API (lean)", version="1.0.0")

@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")

@app.get("/favicon.ico")
def favicon():
    return HTMLResponse(status_code=204, content="")

# ── 공용 스키마 ────────────────────────────────────
class DirectSpec(BaseModel):
    taskId: str
    timeRange: str
    sensor_name: str
    target_cols: str
    constraints: Optional[Any] = None
    userRole: Optional[str] = None

class DirectRunRequest(DirectSpec):
    fromAgent: Literal["orchestration","monitoring","ui","external"] = "orchestration"
    objective: Literal["prediction","forecast","예측"] = "prediction"

class DirectRunResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, str]

# ── predictor(run.py) 실행 ────────────────────────
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

# ── 로그 헬퍼 ─────────────────────────────────────
def _notice(events: List[str], msg: str):
    logger.info(msg)
    events.append(f"[{_now_iso()}] {msg}")

# ── AutoControl 스냅샷 ────────────────────────────
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

# ── (1) NL → JSON 추출 ───────────────────────────
class NLRequest(BaseModel):
    query: str = Field(..., example='화학기계연마 공정 센서의 MOTOR_CURRENT 컬럼의 2024-02-04 09:00:00~09:10:00 의 값들을 예측해줘')
class NLParsedResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: DirectSpec
    metadata: Dict[str, str]


llm = LLMBridge(base_url="http://147.47.39.144:8001/v1", model="Qwen/Qwen3-14B", api_key="EMPTY")


@app.post("/api/v1/prediction/nl/parse", response_model=NLParsedResponse, tags=["Prediction"])
def nl_parse(body: NLRequest):
    events: List[str] = []
    _notice(events, f"[NL] query={body.query}")
    # (선택) 연구실 서버 LLM 헬스체크
    if LLM_API_URL:
        try:
            r = requests.get(LLM_API_URL, timeout=5)
            _notice(events, f"LLM 연결 확인: status={r.status_code}")
        except Exception as e:
            _notice(events, f"LLM 호출 실패(헬스체크): {e}")
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
    preds = [round(start + (i+1)*delta, 4) for i in range(pred_len)]
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
        # 예: {"importantFeatures":[{"name":"X","importance":0.23},...], "method":"..."}
    except Exception as e:
        explanation = {"error": str(e)}
    _notice(events, f"설명 완료: keys={list(explanation.keys())}")

    # 6) 전체 응답 조립
    data = {
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
        "risk": risk_dict,
        "explanation": explanation,
        # 보고문은 별도 엔드포인트에서 생성 (아래 /narrate)
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

# ── (7) 전체 로그/결과 → 사용자 설명(LLM) ────────
from fastapi import Request

class NarrateRequest(BaseModel):
    payload: Dict[str, Any]
    tone: Optional[str] = "operator-ko"

class NarrateResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

@app.post("/api/v1/prediction/narrate", response_model=NarrateResponse, tags=["Prediction"])
def narrate_endpoint(body: NarrateRequest):
    events: List[str] = []
    try:
        _notice(events, f"NARRATE start. LLM base_url={llm.base_url}, model={llm.model}")

        payload = dict(body.payload)  # shallow copy

        # prediction/events가 과하면 축약(선택)
        if isinstance(payload.get("data"), dict):
            d = payload["data"]
            if isinstance(d.get("prediction"), list) and len(d["prediction"]) > 200:
                arr = d["prediction"]
                d["prediction_preview"] = {"head": arr[:5], "tail": arr[-5:], "length": len(arr)}
                d.pop("prediction", None)
            if isinstance(d.get("events"), list) and len(d["events"]) > 200:
                d["events"] = d["events"][-100:]

        # LLM 시도
        text = llm._narrate(payload)
        _notice(events, "Narration OK")

        return {
            "code": "SUCCESS",
            "data": {"narration": text, "events": events},
            "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
        }

    except Exception as e:
        # 실패 시 결정적 폴백 한국어 보고 반환
        err = str(e)
        _notice(events, f"Narration failed: {err}")
        return {
            "code": "ERROR",
            "data": {
                "narration": llm._fallback_ko(body.payload),
                "events": events,
                "error": err
            },
            "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
        }


# @app.post("/api/v1/prediction/narrate", response_model=NarrateResponse, tags=["Prediction"])
# def narrate_endpoint(body: NarrateRequest):
#     events: List[str] = []
#     try:
#         # 0) LLM 엔드포인트/모델 확인 로그
#         _notice(events, f"NARRATE start. LLM base_url={llm.base_url}, model={llm.model}")

#         # 1) payload를 과도하게 크지 않게 직렬화(LLM 입력 안정화)
#         #    - 큰 배열(prediction)이나 로그(events)는 길이를 줄여서 전달
#         payload = dict(body.payload)  # shallow copy
#         # prediction은 앞/뒤 5개만 보여주고 길이만 포함
#         if "data" in payload and isinstance(payload["data"], dict):
#             d = payload["data"]
#             if isinstance(d.get("prediction"), list) and len(d["prediction"]) > 20:
#                 arr = d["prediction"]
#                 d["prediction_preview"] = {"head": arr[:5], "tail": arr[-5:], "length": len(arr)}
#                 # 원본은 제거(컨텍스트 절약)
#                 d.pop("prediction", None)
#             # events도 상위 N개만
#             if isinstance(d.get("events"), list) and len(d["events"]) > 50:
#                 d["events"] = d["events"][-50:]

#         # 2) 직렬화 (ensure_ascii=False로 한글 보존)
#         compact = json.dumps(payload, ensure_ascii=False)

#         # 3) 길이 방어 (LLM 입력 과대 방지)
#         MAX_CHARS = 14000
#         if len(compact) > MAX_CHARS:
#             # 너무 클 경우 최소 요약 형태로 축소
#             keep_keys = ["taskId", "modelSelected", "target_col", "pred_len", "risk", "explanation", "suggestedActions", "df_info"]
#             slim = {}
#             d = payload.get("data", {})
#             for k in keep_keys:
#                 if k in d:
#                     slim[k] = d[k]
#             compact = json.dumps({"data": slim, "metadata": payload.get("metadata", {})}, ensure_ascii=False)
#             _notice(events, f"Payload truncated for narration: {len(compact)} chars")

#         # 4) LLM 호출
#         text = llm.narrate(json.loads(compact), tone=body.tone or "operator-ko")
#         _notice(events, "Narration OK")

#         return {
#             "code": "SUCCESS",
#             "data": {"narration": text, "events": events},
#             "metadata": {"timestamp": _now_iso(), "request_id": _rid()}
#         }

#     except Exception as e:
#         # 에러 디테일을 그대로 응답(디버그에 유용)
#         msg = f"Narration failed: {e}"
#         _notice(events, msg)
#         raise HTTPException(status_code=502, detail=msg)
    
@app.get("/ui", response_class=HTMLResponse)
def prediction_ui():
    return """
<!doctype html><html lang="ko"><head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>PRISM Prediction (lean)</title>
<script src="https://cdn.tailwindcss.com"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body class="bg-slate-50"><div class="max-w-4xl mx-auto p-6">
<h1 class="text-2xl font-bold mb-2">PRISM Prediction</h1>
<p class="text-slate-600 mb-6">NL→JSON → run-direct → Narrate</p>

<section class="bg-white rounded-2xl shadow p-4 mb-6">
  <h2 class="font-semibold mb-2">1) NL → JSON</h2>
  <textarea id="nl" class="w-full border rounded p-2" rows="3">CMP 센서의 MOTOR_CURRENT를 2025-08-20 09:00:00~09:10:00 예측해줘</textarea>
  <div class="flex gap-2 items-center mt-2">
    <button id="btnNL" class="px-3 py-2 rounded bg-indigo-600 text-white">Parse</button>
    <span id="s1" class="text-sm text-slate-500"></span>
  </div>
  <pre id="nlout" class="text-xs bg-slate-50 border rounded p-2 mt-2"></pre>
</section>

<section class="bg-white rounded-2xl shadow p-4 mb-6">
  <h2 class="font-semibold mb-2">2) run-direct</h2>
  <div class="flex gap-2 items-center">
    <button id="btnRun" class="px-3 py-2 rounded bg-emerald-600 text-white">Run</button>
    <span id="s2" class="text-sm text-slate-500"></span>
  </div>
  <pre id="runout" class="text-xs bg-slate-50 border rounded p-2 mt-2"></pre>
</section>

<section class="bg-white rounded-2xl shadow p-4">
  <h2 class="font-semibold mb-2">3) Narrate</h2>
  <div class="flex gap-2 items-center">
    <button id="btnNarr" class="px-3 py-2 rounded bg-slate-700 text-white">Narrate (operator-ko)</button>
    <span id="s3" class="text-sm text-slate-500"></span>
  </div>
  <pre id="narout" class="text-sm bg-slate-50 border rounded p-2 mt-2"></pre>
</section>

<script>
let parsed = null;   // NL→JSON 결과 (서버 응답 전체)
let spec = null;     // parsed.data (DirectSpec)
let lastRun = null;  // run-direct 응답 전체

const nlEl = document.getElementById('nl');
const nlOut = document.getElementById('nlout');
const runOut = document.getElementById('runout');
const narOut = document.getElementById('narout');
const s1 = document.getElementById('s1');
const s2 = document.getElementById('s2');
const s3 = document.getElementById('s3');

function showJSON(el, obj) {
  el.textContent = JSON.stringify(obj, null, 2);
}

document.getElementById('btnNL').onclick = async () => {
  s1.textContent = '파싱 중...';
  try {
    const q = nlEl.value;
    const r = await fetch('/api/v1/prediction/nl/parse', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({query: q})
    });
    const j = await r.json();
    if (!r.ok) {
      s1.textContent = '실패';
      showJSON(nlOut, j);
      return;
    }
    parsed = j;
    spec = parsed.data;  // DirectSpec
    s1.textContent = '완료';
    showJSON(nlOut, parsed);
  } catch (e) {
    s1.textContent = '에러';
    nlOut.textContent = String(e);
  }
};

document.getElementById('btnRun').onclick = async () => {
  if (!spec) { alert('먼저 NL→JSON 하세요'); return; }
  s2.textContent = '실행 중...';
  try {
    const payload = {
      taskId: spec.taskId,
      timeRange: spec.timeRange,
      sensor_name: spec.sensor_name,
      target_cols: spec.target_cols,
      fromAgent: 'ui',
      objective: 'prediction'
    };
    const r = await fetch('/api/v1/prediction/run-direct', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify(payload)
    });
    const j = await r.json();
    lastRun = j; // 전체 응답 보관 (code, data, metadata)
    s2.textContent = r.ok ? '완료' : '실패';
    showJSON(runOut, j);
  } catch (e) {
    s2.textContent = '에러';
    runOut.textContent = String(e);
  }
};

document.getElementById('btnNarr').onclick = async () => {
  if (!lastRun) { alert('먼저 run-direct 실행하세요'); return; }
  s3.textContent = '생성 중...';
  try {
    // 중요: 서버 스키마에 맞게 'payload'로 보냄 (이전의 result → payload 교체)
    const r = await fetch('/api/v1/prediction/narrate', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ payload: lastRun, tone: 'operator-ko' })
    });
    const j = await r.json();
    s3.textContent = r.ok ? '완료' : '실패';
    // narrate 응답은 { code, data: { narration, events }, metadata }
    narOut.textContent = j?.data?.narration || JSON.stringify(j, null, 2);
  } catch (e) {
    s3.textContent = '에러';
    narOut.textContent = String(e);
  }
};
</script>
</div></body></html>
    """

# ── 로컬 실행 ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
