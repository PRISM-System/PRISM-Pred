# main.py — Unified pipeline: NL→Spec (with defaults) → CSV load → run.py per-target → Risk/Explain → LLM narrative
import os, sys, json, uuid, base64, dataclasses, subprocess, logging
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone, timedelta
from io import StringIO
import pathlib

import pandas as pd
from fastapi import FastAPI, Body, HTTPException, Query
from fastapi.responses import RedirectResponse, Response
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# --- local utils (you said this is already created) ---
from utils import (
    now_iso, rid, to_dict, default, normalize_list,
    coerce_time_range, parse_iso_z, to_jsonable
)



# --- optional dependencies (graceful if missing) ---
try:
    from llm_io import LLMBridge
except Exception:
    LLMBridge = None  # type: ignore

try:
    from module import explain as explain_module
except Exception:
    explain_module = None  # type: ignore

try:
    from module import risk as risk_module
except Exception:
    risk_module = None  # type: ignore

try:
    from module import DataLoader, PredictionInputSpec, resolve_csv_path
except Exception:
    DataLoader = PredictionInputSpec = resolve_csv_path = None  # type: ignore

try:
    from ui import router as ui_router, mount_static
except Exception:
    ui_router = None
    def mount_static(_app): pass


# ------------------------------ Logging / Env ------------------------------
logger = logging.getLogger("prism_prediction")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('[%(asctime)s] %(levelname)s in %(module)s: %(message)s'))
if not logger.handlers:
    logger.addHandler(handler)

load_dotenv()
logger.info(f"ENV Loaded: OPENAI_MODEL={os.getenv('OPENAI_MODEL')}")

llm = None
if LLMBridge:
    try:
        llm = LLMBridge(
            base_url=os.getenv("OPENAI_BASE_URL"),
            model=os.getenv("OPENAI_MODEL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    except Exception as e:
        logger.warning(f"LLMBridge init failed: {e}")


# ------------------------------ FastAPI App ------------------------------
app = FastAPI(
    title="PRISM Prediction Agent",
    version="1.3.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    openapi_tags=[
        {"name": "Steps", "description": "모듈을 개별 스텝으로 실행"},
        {"name": "Prediction", "description": "순차 파이프라인 전체 실행(run-direct)"},
        {"name": "UI", "description": "웹 UI"},
    ],
)
if ui_router:
    app.include_router(ui_router)
mount_static(app)

# ==== Add near imports ====
import asyncio
from datetime import datetime, timezone

# ==== Add near top-level globals (logger 선언 아래쯤) ====
_last_llm_status = {"ok": None, "ts": None, "error": None, "model": None}

def _llm_ping_sync():
    global _last_llm_status
    if llm is None:
        msg = "[LLM] LLMBridge not initialized (check OPENAI_BASE_URL / OPENAI_MODEL / OPENAI_API_KEY)."
        logger.warning(msg)
        _last_llm_status = {"ok": False, "ts": datetime.now(timezone.utc).isoformat(),
                            "error": msg, "model": None}
        return
    try:
        test_body = {
            "model": llm.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
        }
        # 비차단을 위해 startup에서는 스레드로 실행 (타임아웃은 LLMBridge에 있으면 설정해도 됨)
        llm._chat(test_body)
        logger.info(f"[LLM] Connected → model={llm.model}")
        _last_llm_status = {"ok": True, "ts": datetime.now(timezone.utc).isoformat(),
                            "error": None, "model": llm.model}
    except Exception as e:
        logger.warning(f"[LLM] Connection failed: {e}")
        _last_llm_status = {"ok": False, "ts": datetime.now(timezone.utc).isoformat(),
                            "error": str(e), "model": getattr(llm, 'model', None)}

@app.on_event("startup")
async def _startup_llm_ping():
    # 서버 포트 바인딩 직후, 이벤트 루프에서 비동기로 실행 (서버 기동을 막지 않음)
    asyncio.create_task(asyncio.to_thread(_llm_ping_sync))



@app.get("/")
def root_redirect():
    return RedirectResponse(url="/ui")

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


# ------------------------------ Schemas ------------------------------
class NLToSpecRequest(BaseModel):
    body: Dict[str, Any] = Field(..., description="raw 요청 바디(래핑 포함 가능)")
    step3_summary: Optional[str] = Field(None, description="모니터링 요약(선택)")

class NLToSpecResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class DiscoverDataRequest(BaseModel):
    spec: Any

class DiscoverDataResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class PredictRequest(BaseModel):
    context: Any

class PredictResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class RiskExplainRequest(BaseModel):
    context: Any

class RiskExplainResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class NarrateRequest(BaseModel):
    context: Any
    lang: Literal["ko","en"] = "ko"

class NarrateResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]

class DirectRunResponse(BaseModel):
    code: Literal["SUCCESS","ERROR"] = "SUCCESS"
    data: Dict[str, Any]
    metadata: Dict[str, Any]


# ------------------------------ Step 1: NL → Spec (defaults & normalization) ------------------------------
@app.post("/api/v1/steps/nl-to-spec", response_model=NLToSpecResponse, tags=["Steps"])
def step_nl_to_spec(req: NLToSpecRequest):
    raw = to_dict(req.body)

    # 1) 어디서 왔든 request 블록을 '강제'로 뽑아냄
    req_block = (
        (raw.get("step_4_orchestration_to_prediction") or {}).get("request")
        or raw.get("request")
        or raw  # 최후의 보루
    )
    if not isinstance(req_block, dict):
        req_block = {}

    # 2) payload 초기화 (사용자 입력 그대로)
    payload = dict(req_block)

    # 3) 결측 보정 (필수 키들)
    from datetime import datetime, timezone
    def _nowid():
        return f"TASK_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    payload.setdefault("taskId", _nowid())
    payload.setdefault("timeRange", {"start": None, "end": None})
    payload.setdefault("sensor_name", "")
    payload.setdefault("target_cols", ["PRESSURE"])  # ← 기본 타깃
    payload.setdefault("feature_cols", [])           # ← 비어오면 다음 스텝에서 보강
    payload.setdefault("prediction_horizon_minutes", 60)
    payload.setdefault("prediction_interval_minutes", 5)
    payload.setdefault("model_type", "lstm")
    payload.setdefault("confidence_level", 0.95)
    # step3 요약 합치기(옵션)
    step3_summary = to_dict(req).get("step3_summary")
    if step3_summary:
        base = payload.get("query") or ""
        payload["query"] = (base + ("\n" if base else "") + f"[Monitoring Summary]\n{step3_summary}")

    # 4) 디버그 로그: 반환 직전 무엇을 보내는지 확인
    logger.info("\n[NL2SPEC] payload (final) = %s", json.dumps(payload, ensure_ascii=False, indent=2))

    # # 5) Spec 객체화
    # spec_obj = PredictionInputSpec.from_body(payload)
    # spec_dict = to_jsonable(spec_obj)

    # # 6) 추가 로그: 파싱 확인
    # logger.info("[NL2SPEC] parsed task_id=%s", spec_obj.task_id)
    # logger.info("[NL2SPEC] parsed target_cols_raw=%s", spec_obj.target_cols_raw)
    # logger.info("[NL2SPEC] parsed feature_cols_raw=%s", spec_obj.feature_cols_raw)
    return {"code":"SUCCESS","data": payload, "metadata":{"timestamp":now_iso(),"request_id":rid()}}

   



# ------------------------------ Step 2: Discover / Load Data ------------------------------
@app.post("/api/v1/steps/discover-data", response_model=DiscoverDataResponse, tags=["Steps"])
def step_discover_data(req: DiscoverDataRequest):
    # spec_dict = to_jsonable(req.spec)

    # # 🔧 normalize: snake_case(spec_obj dict) → camelCase(payload style)
    # # NL2SPEC에서 spec_obj를 dict로 넘기면 아래와 같은 키들이 옵니다:
    # # task_id, time_range, target_cols_raw, feature_cols_raw, horizon_minutes, interval_minutes ...
    # spec_in = dict(spec_dict)
    # if ("task_id" in spec_in) or ("target_cols_raw" in spec_in) or ("feature_cols_raw" in spec_in):
    #     spec_in = {
    #         "taskId": spec_in.get("task_id", ""),
    #         "timeRange": spec_in.get("time_range"),
    #         "sensor_name": spec_in.get("sensor_name", ""),
    #         "target_cols": spec_in.get("target_cols_raw") or spec_in.get("target_cols") or [],
    #         "feature_cols": spec_in.get("feature_cols_raw") or spec_in.get("feature_cols") or [],
    #         "prediction_horizon_minutes": spec_in.get("horizon_minutes"),
    #         "prediction_interval_minutes": spec_in.get("interval_minutes"),
    #         "confidence_level": spec_in.get("confidence_level"),
    #         "query": spec_in.get("query", ""),
    #     }
    #     logger.info("[DISCOVER] normalized spec_in(camelCase) = %s", json.dumps(spec_in, ensure_ascii=False))
    # else:
    #     # 이미 payload 스타일이면 그대로 사용
    #     spec_in = spec_dict
    spec_in = to_jsonable(req.spec) 

    # ⚠️ 시연 안전장치: target/feature 비어오면 기본 채우기
    if not normalize_list(spec_in.get("target_cols")):
        spec_in["target_cols"] = ["PRESSURE"]
    if not normalize_list(spec_in.get("feature_cols")):
        spec_in["feature_cols"] = ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE"]

    # 이후는 기존 로직 그대로
    spec = PredictionInputSpec.from_body(spec_in)

    try:
        if resolve_csv_path and spec:
            csv_path = resolve_csv_path(spec)
        else:
            csv_path = spec_in.get("csv_path") or ""
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if not csv_path or not os.path.exists(csv_path):
        raise HTTPException(status_code=404, detail=f"CSV not found: {csv_path}")

    if DataLoader and spec:
        try:
            dl = DataLoader(csv_path, spec)
            loaded = dl.load()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"데이터 로드 실패: {e}")

        df = loaded.df
        data = {
            "spec": spec_in,  # ← 정규화된 입력을 저장
            "csv_path": csv_path,
            "df_info": {"rows": int(df.shape[0]), "cols": int(df.shape[1])},
            "feature_names": loaded.feature_names,
            "enc_in": loaded.enc_in,
            "target_col": loaded.target_col,
            "target_idx_in_features": loaded.target_idx_in_features,
            "pred_len": loaded.pred_len,
            #"offsets": loaded.offsets,
            "confidence_level": loaded.confidence_level,
            "sensor_name": spec.sensor_name,
            "timeRange": spec.time_range,
            "horizon_minutes": spec.horizon_minutes,
            "interval_minutes": spec.interval_minutes,
            "requested_target_cols": normalize_list(spec_in.get("target_cols")),
            "requested_feature_cols": normalize_list(spec_in.get("feature_cols")),
            "feature_df_pickle": base64.b64encode(loaded.feature_df.to_csv(index=False).encode("utf-8")).decode("ascii"),
            "taskId": spec.task_id,
            "query": spec_in.get("query",""),
        }
    # else:
    #     # (생략) minimal path ...
    #     ...
    return {"code":"SUCCESS","data": data, "metadata":{"timestamp":now_iso(),"request_id":rid()}}


# ------------------------------ Step 3: Predict (per-target run.py) ------------------------------
RUN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run.py")

# def _run_predictor_with_csv(csv_path: str, target_col_name: str, pred_len : int) -> Dict[str, Any]:
#     script_dir = os.path.dirname(os.path.abspath(__file__))
#     runpy_path = os.path.join(script_dir, "run.py")
#     out_path = os.path.join(script_dir, "outputs", "summary.json")
#     try:
#         if os.path.exists(out_path):
#             os.remove(out_path)
#     except Exception:
#         pass

#     cmd = [
#         sys.executable, runpy_path,
#         "--csv_path", csv_path,
#         "--feature_start_col", "2",
#         "--target_col_name", target_col_name,
#         "--seq_len", "48", "--label_len", "24", "--pred_len", str(pred_len),
#         "--epochs", "1", "--batch_size", "8",
#         "--models", "Autoformer,DLinear,TimesNet,LightTS",
#         "--device", "cpu",
#         "--auto_eval_idx",
#     ]
#     proc = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)
#     if proc.returncode != 0:
#         raise HTTPException(status_code=500, detail=f"predictor failed: {proc.stderr}")

#     if not os.path.exists(out_path):
#         tail = proc.stdout[-400:] if proc.stdout else ""
#         raise HTTPException(status_code=500, detail=f"predictor did not produce {out_path}. tail={tail}")

#     with open(out_path, "r") as f:
#         return json.load(f)
def _run_predictor_with_csv(
    csv_path: str,
    target_col_name: str,
    pred_len: int,
    first_feature_name: Optional[str] = None,
    eval_channel_idx: Optional[int] = None,
) -> Dict[str, Any]:
    import pandas as _pd

    script_dir = os.path.dirname(os.path.abspath(__file__))
    runpy_path = os.path.join(script_dir, "run.py")
    out_path = os.path.join(script_dir, "outputs", "summary.json")
    try:
        if os.path.exists(out_path):
            os.remove(out_path)
    except Exception:
        pass

    # ---  CSV 헤더로 feature_start_col 자동 산정 ---
    cols = _pd.read_csv(csv_path, nrows=0).columns.tolist()
    feature_start_col = None
    if first_feature_name and first_feature_name in cols:
        feature_start_col = cols.index(first_feature_name) + 1  # 1-based
    elif target_col_name in cols:
        feature_start_col = cols.index(target_col_name) + 1
    else:
        # 안전 기본값(타임스탬프+ID 컬럼 있다고 가정)
        feature_start_col = 3

    cmd = [
        sys.executable, runpy_path,
        "--csv_path", csv_path,
        "--feature_start_col", str(feature_start_col),
        "--target_col_name", target_col_name,
        "--seq_len", "48", "--label_len", "24", "--pred_len", str(pred_len),
        "--epochs", "1", "--batch_size", "8",
        "--models", "Autoformer,DLinear,TimesNet,LightTS",
        "--device", "cpu",
        "--auto_eval_idx",
    ]
    # eval 채널이 이미 known이면 명시적으로 전달 (오토와 일치 보장)
    if eval_channel_idx is not None:
        cmd += ["--eval_channel_idx", str(int(eval_channel_idx))]

    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=script_dir)

    if proc.returncode != 0:
        raise HTTPException(status_code=500, detail=f"predictor failed: {proc.stderr}")

    if not os.path.exists(out_path):
        tail = proc.stdout[-400:] if proc.stdout else ""
        raise HTTPException(status_code=500, detail=f"predictor did not produce {out_path}. tail={tail}")

    with open(out_path, "r") as f:
        return json.load(f)


@app.post("/api/v1/steps/predict", response_model=PredictResponse, tags=["Steps"])
def step_predict(req: PredictRequest):
    ctx = to_jsonable(req.context)
    feature_df_csv = base64.b64decode(ctx["feature_df_pickle"].encode("ascii")).decode("utf-8")
    feature_df = pd.read_csv(StringIO(feature_df_csv))

    csv_path = ctx["csv_path"]
    requested_targets = normalize_list(ctx.get("requested_target_cols")) or ([ctx.get("target_col")] if ctx.get("target_col") else [])
    interval = int(ctx.get("interval_minutes") or 5)
    pred_len_hint = int(ctx.get("pred_len") or max(1, int((ctx.get("horizon_minutes") or 60) / interval)))

    predictions_by_target: Dict[str, List[float]] = {}
    best_model_by_target: Dict[str, str] = {}

    for tgt in requested_targets:
        #summary = _run_predictor_with_csv(csv_path=csv_path, target_col_name=tgt, pred_len=pred_len_hint)
        summary = _run_predictor_with_csv(csv_path=csv_path, target_col_name=tgt, pred_len=pred_len_hint, first_feature_name=(ctx.get("feature_names") or [None])[0], eval_channel_idx=ctx.get("target_idx_in_features"),
)

        best_model_by_target[tgt] = summary.get("best_model") or "Unknown"

        # fallback/length-fix
        series = pd.to_numeric(feature_df[tgt], errors="coerce").dropna().to_numpy() if tgt in feature_df.columns else None
        if series is None or series.size < 2:
            raise HTTPException(status_code=400, detail=f"타깃({tgt}) 데이터 부족(>=2 필요).")
        delta = float(series[-1] - series[-2]); start = float(series[-1])

        preds = summary.get("prediction")
        if not isinstance(preds, list) or len(preds) == 0:
            preds = [round(start + (i + 1) * delta, 6) for i in range(pred_len_hint)]
        elif len(preds) != pred_len_hint:
            preds = (preds[:pred_len_hint] if len(preds) > pred_len_hint else preds + [preds[-1]] * (pred_len_hint - len(preds)))

        predictions_by_target[tgt] = preds

    # shared timeline
    tr = ctx.get("timeRange", {})
    t0 = parse_iso_z(tr.get("end") or tr.get("start"))
    timeline = []
    if t0:
        timeline = [(t0 + timedelta(minutes=interval * (i+1))).strftime("%Y-%m-%dT%H:%M:%SZ") for i in range(len(next(iter(predictions_by_target.values()))))]

    ctx.update({
        "modelSelected": best_model_by_target,
        "prediction": predictions_by_target,
        "prediction_timestamps": timeline,
    })
    return {"code":"SUCCESS","data": ctx, "metadata":{"timestamp":now_iso(),"request_id":rid()}}


# ------------------------------ Step 4: Risk (optional) ------------------------------
@app.post("/api/v1/steps/risk", response_model=RiskExplainResponse, tags=["Steps"])
def step_risk(req: RiskExplainRequest):
    ctx = to_jsonable(req.context)
    if risk_module is None:
        ctx["risk"] = {"riskLevel": "unknown", "note": "risk module unavailable"}
        return {"code":"SUCCESS","data": ctx, "metadata":{"timestamp":now_iso(),"request_id":rid()}}

    feature_df_csv = base64.b64decode(ctx["feature_df_pickle"].encode("ascii")).decode("utf-8")
    feature_df = pd.read_csv(StringIO(feature_df_csv))
    tgt = (normalize_list(ctx.get("requested_target_cols")) or [ctx.get("target_col")])[0]
    preds = ctx["prediction"].get(tgt, [])

    try:
        risk_dict = risk_module(feature_df, tgt, preds)
    except Exception as e:
        risk_dict = {"riskLevel": "unknown", "error": str(e), "exceedsThreshold": False}

    ctx["risk"] = risk_dict
    return {"code":"SUCCESS","data": ctx, "metadata":{"timestamp":now_iso(),"request_id":rid()}}


# ------------------------------ Step 5: Explain (optional) ------------------------------
@app.post("/api/v1/steps/explain", response_model=RiskExplainResponse, tags=["Steps"])
def step_explain(req: RiskExplainRequest):
    ctx = to_jsonable(req.context)
    if explain_module is None:
        ctx["explanation"] = {"note": "explain module unavailable"}
        return {"code":"SUCCESS","data": ctx, "metadata":{"timestamp":now_iso(),"request_id":rid()}}

    feature_df_csv = base64.b64decode(ctx["feature_df_pickle"].encode("ascii")).decode("utf-8")
    feature_df = pd.read_csv(StringIO(feature_df_csv))
    tgt = (normalize_list(ctx.get("requested_target_cols")) or [ctx.get("target_col")])[0]
    preds = ctx["prediction"].get(tgt, [])

    try:
        explanation = explain_module(feature_df, tgt, preds)
    except Exception as e:
        explanation = {"error": str(e)}

    ctx["explanation"] = explanation
    return {"code":"SUCCESS","data": ctx, "metadata":{"timestamp":now_iso(),"request_id":rid()}}


# ------------------------------ Step 6: Narrative (LLM: query + predictions) ------------------------------
def _build_llm_messages_ko(ctx: Dict[str, Any]) -> list:
    system = {
        "role": "system",
        "content": (
            "너는 산업 공정 예측 리포트를 작성하는 한국어 기술 에디터다. "
            "사용자가 제공한 자연어 질의(query), 모델 예측 결과(타깃별 시계열), 시간표, 위험/설명 정보를 바탕으로 "
            "간결하고 명확한 마크다운 리포트를 작성하라. 수치/표/시각은 절대 임의 변경하지 말고, "
            "요약·강조·해석만 수행하라."
        )
    }
    query_text = (ctx.get("query") or "").strip()
    safe_ctx = {
        "taskId": ctx.get("taskId"),
        "sensor_name": ctx.get("sensor_name"),
        "timeRange": ctx.get("timeRange"),
        "requested_target_cols": ctx.get("requested_target_cols"),
        "requested_feature_cols": ctx.get("requested_feature_cols"),
        "modelSelected": ctx.get("modelSelected"),
        "prediction_timestamps": ctx.get("prediction_timestamps"),
        "prediction": ctx.get("prediction"),
        "risk": ctx.get("risk"),
        "explanation": ctx.get("explanation"),
        "confidence_level": ctx.get("confidence_level"),
        "horizon_minutes": ctx.get("horizon_minutes"),
        "interval_minutes": ctx.get("interval_minutes"),
        "df_info": ctx.get("df_info"),
    }
    user = {
        "role": "user",
        "content": (
            "### 사용자 질의 (원문)\n"
            f"{query_text if query_text else '(질의 없음)'}\n\n"
            "### 예측 컨텍스트(JSON)\n"
            + json.dumps(safe_ctx, ensure_ascii=False, indent=2)
        )
    }
    return [system, user]

@app.post("/api/v1/steps/narrate", response_model=NarrateResponse, tags=["Steps"])
def step_narrate(req: NarrateRequest):
    ctx = to_jsonable(req.context)

    if llm is None:
        fallback = (
            "### 예측 보고서\n\n"
            "(LLM 설정이 없어 간략 요약만 표시합니다.)\n\n"
            f"- 작업 ID: {ctx.get('taskId')}\n"
            f"- 타깃: {', '.join(ctx.get('requested_target_cols') or ([] if not ctx.get('target_col') else [ctx.get('target_col')]))}\n"
            f"- 모델: {ctx.get('modelSelected')}\n"
            f"- 예측 길이: {len(next(iter(ctx.get('prediction', {'_': []}).values()), []))}\n"
        )
        return {"code":"SUCCESS","data":{"result":fallback}, "metadata":{"timestamp":now_iso(),"request_id":rid()}}

    messages = _build_llm_messages_ko(ctx)  # (영문 필요 시 별도 함수 작성 가능)
    try:
        prompt = {
            "model": llm.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": 1400,
        }
        resp = llm._chat(prompt)
        report = (resp["choices"][0]["message"]["content"] or "").strip()
        if not report:
            report = "### 예측 보고서\n\n(LLM 응답이 비어 있어 기본 메시지를 표시합니다.)"
    except Exception as e:
        report = f"### 예측 보고서\n\n(LLM 호출 실패: {e})"

    return {"code":"SUCCESS","data":{"result":report}, "metadata":{"timestamp":now_iso(),"request_id":rid()}}


# ------------------------------ Orchestrated: run-direct ------------------------------
@app.post("/api/v1/prediction/run-direct", response_model=DirectRunResponse, tags=["Prediction"])
def run_direct(
    body: Any = Body(...),
    narrate: bool = Query(True, description="자연어 보고서 포함 여부"),
    lang: Literal["ko","en"] = Query("ko", description="보고서 언어"),
):
    body_dict = to_dict(body)
    step3_sum = to_dict(body).get("step_3_monitoring_to_orchestration", {}).get("response", {}).get("summary")

    # 1) NL→Spec
    s1 = step_nl_to_spec(NLToSpecRequest(body=body_dict, step3_summary=step3_sum))
    spec = to_jsonable(s1["data"])
    
    # 2) Discover Data
    s2 = step_discover_data(DiscoverDataRequest(spec=spec))
    ctx = to_jsonable(s2["data"])

    # 3) Predict
    s3 = step_predict(PredictRequest(context=ctx))
    ctx = to_jsonable(s3["data"])

    # 4) Risk (optional)
    s4 = step_risk(RiskExplainRequest(context=ctx))
    ctx = to_jsonable(s4["data"])

    # 5) Explain (optional)
    s5 = step_explain(RiskExplainRequest(context=ctx))
    ctx = to_jsonable(s5["data"])

    # 6) Narrate
    if narrate:
        s6 = step_narrate(NarrateRequest(context=ctx, lang=lang))
        safe_raw = dict(ctx)
        safe_raw.pop("feature_df_pickle", None)
        return {
            "code": "SUCCESS",
            "data": {"result": s6["data"]["result"], "raw": safe_raw},
            "metadata": {"timestamp": now_iso(), "request_id": rid()}
        }

    return {"code":"SUCCESS","data": ctx, "metadata":{"timestamp":now_iso(),"request_id":rid()}}

from fastapi import FastAPI
from fastapi import HTTPException
import sys


# ------------------------------ Local run ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
