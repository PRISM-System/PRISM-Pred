# llm_io.py (drop-in 교체판)
import os, json, requests, datetime
from typing import Any, Dict, List, Optional

_INSTITUTION_PRESETS = {
    # 필요하면 주석 해제해서 바로 쓰세요. 보안상 .env 사용을 더 권장합니다.
    # "서울대학교":      {"id": "seoul",       "pw": "seoul1234",       "user_id": "user_1111"},
    # "한양대학교":      {"id": "hanyang",     "pw": "hanyang1234",     "user_id": "user_2222"},
    "성균관대학교":    {"id": "sunkyunkwan", "pw": "sunkyunkwan1234", "user_id": "user_3333"},
    # "카이스트":        {"id": "kaist",       "pw": "kaist1234",       "user_id": "user_4444"},
}

def _is_bimatrix_base(url: Optional[str]) -> bool:
    if not url:
        return False
    u = url.rstrip("/")
    # BiMatrix SaaS or Local BiMatrix Core
    return (("grnd.bimatrix.co.kr" in u) and ("/django/agi" in u)) or \
           (u.startswith("http://147.47.39.144:8000"))

class LLMBridge:
    """
    두 모드 자동 지원:
      1) OpenAI 호환 서버:  base_url = ".../v1"  → POST {base_url}/chat/completions (Bearer)
      2) BiMatrix 서버:     base_url = "https://grnd.bimatrix.co.kr/django/agi"
                           → POST {base_url}/api/login/ (세션 로그인)
                           → POST {base_url}/llm-agent  (세션 쿠키로 호출)
      3) Local BiMatrix Core: base_url = "http://147.47.39.144:8000"
                           → POST {base_url}/api/generate (인증 없음)
    환경변수:
      - OPENAI_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL
      - BIMATRIX_BASE_URL="https://grnd.bimatrix.co.kr/django/agi" 또는 "http://147.47.39.144:8000"
      - BIMATRIX_ID, BIMATRIX_PW, BIMATRIX_VERIFY=true|false
      - INSTITUTION="성균관대학교"  # 있으면 프리셋 자격증명 우선 적용 (env가 있으면 env 우선)
    """
    def __init__(self,
                 base_url: Optional[str] = None,
                 api_key: Optional[str] = None,
                 model: Optional[str] = None,
                 verify: Optional[bool] = None,
                 institution: Optional[str] = None):
        # 1) 기본값 확보
        #   - 우선순위: 인자 > ENV > 프리셋/기본
        #   - base_url은 OPENAI_BASE_URL 또는 BIMATRIX_BASE_URL 중 하나를 넣어주세요.
        #     (BiMatrix 사용 시 BIMATRIX_BASE_URL로 지정)
        env_openai_base = os.getenv("OPENAI_BASE_URL")
        env_bimatrix_base = os.getenv("BIMATRIX_BASE_URL")
        self.base_url = (base_url or env_bimatrix_base or env_openai_base or "").rstrip("/")

        self._service_bimatrix = _is_bimatrix_base(self.base_url)
        self._local_bimatrix = self.base_url.startswith("http://147.47.39.144:8000") if self.base_url else False
        self.model = model or os.getenv("OPENAI_MODEL", "Qwen/Qwen3-14B")

        # OpenAI 호환 모드용
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self._openai_url = f"{self.base_url}/chat/completions" if not self._service_bimatrix else None

        # BiMatrix 모드용
        if self._service_bimatrix:
            # Local BiMatrix Core는 /api/generate 사용, 로그인 불필요
            if self._local_bimatrix:
                self.llm_url = f"{self.base_url}/api/generate"
                self.login_url = None
                self.verify = False
                self.username = None
                self.password = None
                self.session = requests.Session()
                self.user_id = None
            else:
                # Remote BiMatrix SaaS
                self.login_url = f"{self.base_url}/api/login/"
                # 공지에 따라 llm_agent 엔드포인트 변경됨:
                #   기존: .../django/api/llm_agent/
                #   변경: .../django/agi/llm-agent
                self.llm_url   = f"{self.base_url}/llm-agent/"
                # verify: 기본 True 권장 (없으면 env/BIMATRIX_VERIFY 참고)
                if verify is None:
                    verify_env = os.getenv("BIMATRIX_VERIFY", "true").lower()
                    self.verify = (verify_env == "true")
                else:
                    self.verify = bool(verify)
                # 자격 증명 (institution 프리셋 → ENV 순)
                self.institution = institution or os.getenv("INSTITUTION")
                user = pw = None
                if self.institution in _INSTITUTION_PRESETS:
                    user = _INSTITUTION_PRESETS[self.institution]["id"]
                    pw   = _INSTITUTION_PRESETS[self.institution]["pw"]
                # ENV가 있으면 ENV 우선
                self.username = os.getenv("BIMATRIX_ID", user or "")
                self.password = os.getenv("BIMATRIX_PW", pw or "")
                self.session  = requests.Session()
                self.user_id: Optional[str] = None
        else:
            self.verify = True  # OpenAI 모드는 보통 공인 cert 사용

    # =========================
    # 내부 공통 POST 래퍼
    # =========================
    def _chat_openai(self, payload: dict) -> dict:
        if not self._openai_url:
            raise RuntimeError("OpenAI URL is not configured.")
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(self._openai_url, json=payload, headers=headers, timeout=60, verify=self.verify)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            raise RuntimeError(
                f"LLM request failed ({r.status_code}) to {self._openai_url}: {r.text[:200]}"
            ) from e
        return r.json()

    def _ensure_bimatrix_login(self) -> None:
        if not self._service_bimatrix:
            return
        # Local BiMatrix Core는 로그인 불필요
        if self._local_bimatrix:
            return
        if not (self.username and self.password):
            raise RuntimeError("BiMatrix credentials are missing. Set INSTITUTION or BIMATRIX_ID/BIMATRIX_PW.")
        if getattr(self, "_logged_in", False):
            return
        headers = {"accept": "application/json"}
        payload = {"username": self.username, "password": self.password}
        resp = self.session.post(self.login_url, json=payload, headers=headers, timeout=60, verify=self.verify)
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"BiMatrix login failed: {e} :: {resp.text[:200]}")
        data = resp.json()
        # 기대 응답 예: {"user_id": "user_3333", ...}
        self.user_id = data.get("user_id")
        self._logged_in = True

    def _chat_bimatrix(self, payload: dict) -> dict:
        self._ensure_bimatrix_login()
        headers = {"accept": "application/json", "Content-Type": "application/json"}
        resp = self.session.post(self.llm_url, json=payload, headers=headers, timeout=60, verify=self.verify)
        try:
            resp.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"BiMatrix LLM request failed ({resp.status_code}): {resp.text[:200]}") from e
        return resp.json()

    # 외부에서 쓰는 단일 엔트리포인트 (서비스 타입 자동 분기)
    def _chat(self, payload: dict) -> dict:
        if self._service_bimatrix:
            return self._chat_bimatrix(payload)
        return self._chat_openai(payload)

    # =========================
    # 도메인 유틸
    # =========================
    def _extract_json_from_text(self, nl_query: str) -> dict:
        """(기존 그대로) NL → JSON (툴콜 강제)"""
        tool_schema = {
            "type": "function",
            "function": {
                "name": "build_direct_spec",
                "description": "Map NL prediction request into a strict schema for run-direct.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "taskId":      {"type": "string"},
                        "timeRange":   {"type": "string"},
                        "sensor_name": {"type": "string"},
                        "target_cols": {"type": "string"},
                        "constraints": {"type": "object"},
                        "userRole":    {"type": "string"}
                    },
                    "required": ["timeRange","sensor_name","target_cols"]
                }
            }
        }

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content":
                    "Return ONLY a function call to build_direct_spec. No prose, no <think>, no extra text."},
                {"role": "user", "content": nl_query}
            ],
            "tools": [tool_schema],
            "tool_choice": {"type": "function", "function": {"name": "build_direct_spec"}},
            "temperature": 0.0,
            "max_tokens": 300
        }

        resp = self._chat(payload)
        msg = resp["choices"][0]["message"]
        calls = msg.get("tool_calls") or []
        if not calls:
            txt = msg.get("content") or ""
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1:
                return json.loads(txt[s:e+1])
            raise ValueError(f"LLM did not return a tool call or JSON: {txt[:200]}")

        args_str = calls[0]["function"]["arguments"]
        spec = json.loads(args_str)
        if not spec.get("taskId"):
            spec["taskId"] = "task_" + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return spec

    def _narrate_en(self, payload: Dict[str, Any]) -> str:
        try:
            system_msg = (
                "You are a careful writer. Produce a natural English explanation using ONLY values present "
                "in the JSON provided by the user. NEVER reveal chain-of-thought."
            )
            user_msg = (
                "Rewrite this JSON into one paragraph. Use only what's present; "
                "if missing say 'not specified'.\n\n" + json.dumps(payload, ensure_ascii=False, indent=2)
            )
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
            }
            resp = self._chat(data)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_en(payload)

    def _fallback_en(self, payload: Dict[str, Any]) -> str:
        try:
            d = payload.get("data", {})
            taskId = d.get("taskId", "N/A")
            timeRange = d.get("timeRange", "N/A")
            sensor = d.get("sensor_name", "N/A")
            target = d.get("target_col", d.get("target_cols", "N/A"))
            model = d.get("modelSelected", "Unknown")
            preds = d.get("prediction", [])
            pred_len = len(preds) if isinstance(preds, list) else "N/A"
            if isinstance(preds, list) and len(preds) > 10:
                pred_text = f"first {preds[:3]} ... last {preds[-3:]} (total {len(preds)})"
            else:
                pred_text = str(preds)
            risk = d.get("risk", {})
            risk_level = risk.get("riskLevel", "unknown")
            return (
                f"This is the answer for request {taskId}. The request covered {timeRange} on sensor {sensor}, "
                f"targeting column {target}. We selected model {model}, which produced {pred_len} predictions: "
                f"{pred_text}. The risk level was assessed as {risk_level}."
            )
        except Exception as e:
            return f"Could not generate narrative fallback: {e}"

    def narrate_ko_markdown(self, payload: Dict[str, Any]) -> str:
        try:
            system_msg = (
                "너는 제조 예측 시스템의 기술 라이터다. "
                "사용자가 제공한 JSON 안의 값들만 사용해 한국어로 Markdown 리포트를 작성하라. "
                "누락된 값은 '정보 없음'으로 표기하라. 절대 내부 추론을 노출하지 마라."
            )
            user_msg = (
                "다음 JSON을 바탕으로 한국어 Markdown 리포트를 작성하라.\n\n"
                "### 예측 결과 요약\n\n### 1) 예측 개요\n\n### 2) 타깃 지표 예측\n\n"
                "### 3) 위험도 평가\n\n### 4) 변수 기여도(설명)\n\n### 5) 데이터/파이프라인 정보\n\n### 6) 결론\n\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
            )
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.0,
                "max_tokens": 1600,
            }
            resp = self._chat(data)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_ko_md(payload)

    def _fallback_ko_md(self, payload: Dict[str, Any]) -> str:
        d = payload.get("data", {})
        md = []
        def fmt(v):
            if v is None: return "정보 없음"
            if isinstance(v, float): return f"{v:.6g}"
            return str(v)
        def head_tail(arr, k=5):
            if not isinstance(arr, list) or not arr:
                return "정보 없음"
            if len(arr) <= 2*k:
                return ", ".join(fmt(x) for x in arr) + f" (총 {len(arr)}개)"
            return f"[앞 {k}] " + ", ".join(fmt(x) for x in arr[:k]) + \
                   f" / [뒤 {k}] " + ", ".join(fmt(x) for x in arr[-k:]) + \
                   f" (총 {len(arr)}개)"
        time_range = d.get("timeRange", {})
        horizon = d.get("horizon_minutes")
        interval = d.get("interval_minutes")
        model = d.get("modelSelected")
        conf = d.get("confidence_level", 0.95)
        target = d.get("target_col")
        preds = d.get("prediction", [])
        risk = d.get("risk", {}) or {}
        expl = d.get("explanation", {}) or {}
        feature_names = d.get("feature_names", [])
        events = d.get("events", [])

        md.append("### 예측 결과 요약")
        md.append(f"- 모델: **{fmt(model)}**, 신뢰수준: **{fmt(conf)}**")
        md.append(f"- 예측 기간/간격: **{fmt(horizon)}분 / {fmt(interval)}분**")
        md.append(f"- 타깃: **{fmt(target)}**")
        if isinstance(time_range, dict):
            md.append(f"- 입력 구간: **{fmt(time_range.get('start'))} ~ {fmt(time_range.get('end'))}**")
        else:
            md.append(f"- 입력 구간: **{fmt(time_range)}**")
        md.append("")
        md.append("### 1) 예측 개요")
        md.append(f"- 모델: {fmt(model)}")
        md.append(f"- 신뢰수준: {fmt(conf)}")
        md.append("")
        md.append("### 2) 타깃 지표 예측")
        md.append(f"- 타깃 컬럼: **{fmt(target)}**")
        md.append(f"- 예측값 요약: {head_tail(preds, k=5)}")
        md.append("")
        md.append("### 3) 위험도 평가")
        md.append(f"- 위험도: **{fmt(risk.get('riskLevel'))}**")
        md.append("")
        md.append("### 4) 변수 기여도(설명)")
        feats = expl.get("importantFeatures", [])
        if feats:
            md.append("- 주요 변수(최대 5개): " + ", ".join(str(x) for x in feats[:5]))
        else:
            md.append("- 주요 변수: 정보 없음")
        md.append("")
        md.append("### 5) 데이터/파이프라인 정보")
        if feature_names:
            if len(feature_names) > 8:
                md.append(f"- feature_names: {', '.join(feature_names[:5])} ... (총 {len(feature_names)}개)")
            else:
                md.append(f"- feature_names: {', '.join(feature_names)}")
        if events:
            md.append("- 처리 이벤트:")
            for e in events:
                md.append(f"  - {e}")
        md.append("")
        md.append("### 6) 결론")
        level = fmt(risk.get("riskLevel"))
        md.append(f"- 현재 상태 요약: **{level}**")
        return "\n".join(md)

    def _narrate(self, payload: Dict[str, Any]) -> str:
        try:
            system_msg = (
                "너의 임무는 제조 예측 파이프라인의 JSON 결과를 한국어로 필드 누락 없이 그대로 설명하는 것이다. "
                "절대 새로운 가정이나 수치를 만들지 마라."
            )
            user_msg = (
                "다음 JSON의 모든 항목을 한국어로 항목별로 써라. 표/코드블록 금지.\n\n" +
                json.dumps(payload, ensure_ascii=False, indent=2)
            )
            data = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg}
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
            }
            resp = self._chat(data)
            return resp["choices"][0]["message"]["content"].strip()
        except Exception:
            return self._fallback_ko(payload)

    def _fallback_ko(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("PRISM 예측 결과 상세 보고 (폴백 모드)")
        lines.append("")
        def fmt_scalar(v: Any) -> str:
            if v is None: return "정보 없음"
            if isinstance(v, float): return f"{v:.6g}"
            return str(v)
        def walk(key: str, val: Any, indent: int = 0):
            pad = "  " * indent
            bullet = "- "
            if isinstance(val, dict):
                lines.append(f"{pad}{bullet}{key}:")
                if not val:
                    lines.append(f"{pad}  (빈 객체)")
                for k, v in val.items():
                    walk(k, v, indent + 1)
            elif isinstance(val, list):
                lines.append(f"{pad}{bullet}{key}:")
                if not val:
                    lines.append(f"{pad}  (빈 리스트)")
                else:
                    if all(not isinstance(x, (dict, list)) for x in val):
                        if len(val) > 12:
                            head = ", ".join(fmt_scalar(x) for x in val[:5])
                            tail = ", ".join(fmt_scalar(x) for x in val[-5:])
                            lines.append(f"{pad}  [앞 5] {head} / [뒤 5] {tail} / 총 {len(val)}개")
                        else:
                            joined = ", ".join(fmt_scalar(x) for x in val)
                            lines.append(f"{pad}  {joined}")
                    else:
                        for i, item in enumerate(val):
                            walk(f"{key}[{i}]", item, indent + 1)
            else:
                lines.append(f"{pad}{bullet}{key}: {fmt_scalar(val)}")
        for k, v in payload.items():
            walk(k, v, 0)
        return "\n".join(lines)

    # 편의용 별칭
    def narrate(self, text: str) -> str:
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content":
                    "너는 산업 제어 및 공정 최적화 분야의 분석 전문가다. "
                    "사용자가 제공하는 제어 후보군을 공정 특성 상식으로 평가하라."},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 10000,
            "top_p": 1.0,
            "stream": False
        }
        resp = self._chat(data)
        try:
            return resp["choices"][0]["message"]["content"]
        except Exception:
            return json.dumps(resp, ensure_ascii=False, indent=2)
