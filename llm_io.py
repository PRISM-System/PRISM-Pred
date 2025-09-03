# llm_io.py
import os, json, requests, datetime
from typing import Any, Dict, List

class LLMBridge:
    def __init__(self, base_url=None, api_key=None, model=None):
        # base_url 예: "http://147.47.39.144:8001/v1" (끝에 /v1 포함해도 됨)
        self.base_url = (base_url or os.getenv("OPENAI_BASE_URL", "http://147.47.39.144:8001/v1")).rstrip("/")
        self.api_key  = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model    = model  or os.getenv("OPENAI_MODEL", "Qwen/Qwen3-14B")
        self.url      = f"{self.base_url}/chat/completions"

    # 공통 POST 래퍼
    def _chat(self, payload: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key and self.api_key != "EMPTY":
            headers["Authorization"] = f"Bearer {self.api_key}"
        r = requests.post(self.url, json=payload, headers=headers, timeout=60)
        try:
            r.raise_for_status()
        except requests.HTTPError as e:
            # 디버깅에 도움 되도록 서버 응답 앞부분 포함
            raise RuntimeError(
                f"LLM request failed ({r.status_code}) to {self.url}: {r.text[:200]}"
            ) from e
        return r.json()

    # -------------------------------
    # (1) NL → JSON (툴콜 강제)
    # -------------------------------
    def _extract_json_from_text(self, nl_query: str) -> dict:
        """
        모델이 반드시 function call로만 응답하도록 강제 → arguments가 우리의 JSON이 됨
        """
        tool_schema = {
            "type": "function",
            "function": {
                "name": "build_direct_spec",
                "description": "Map NL prediction request into a strict schema for run-direct.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "taskId":      {"type": "string", "description": "Unique task id"},
                        "timeRange":   {"type": "string", "description": "e.g. 'YYYY-MM-DD HH:MM:SS ~ YYYY-MM-DD HH:MM:SS'"},
                        "sensor_name": {"type": "string", "description": "Sensor group name, e.g. CMP"},
                        "target_cols": {"type": "string", "description": "Target column name"},
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
                    "Return ONLY a function call to build_direct_spec. "
                    "No prose, no <think>, no extra text."},
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
            # 폴백: 혹시 텍스트로 JSON이 왔으면 그걸 파싱 시도
            txt = msg.get("content") or ""
            s, e = txt.find("{"), txt.rfind("}")
            if s != -1 and e != -1:
                return json.loads(txt[s:e+1])
            raise ValueError(f"LLM did not return a tool call or JSON: {txt[:200]}")

        args_str = calls[0]["function"]["arguments"]
        spec = json.loads(args_str)

        # taskId 보정
        if not spec.get("taskId"):
            spec["taskId"] = "task_" + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
        return spec
    

    def _narrate_en(self, payload: Dict[str, Any]) -> str:
        """
        Convert JSON payload into a natural English narrative.
        Example style:
        "This is the answer for the request ... We selected model ... The predictions are ..."
        """
        try:
            system_msg = (
            "You are a careful writer. Produce a natural English explanation using ONLY values present "
            "in the JSON provided by the user. "
            "NEVER reveal chain-of-thought, internal reasoning, analysis notes, or any <think> tags. "
            "Output ONLY the final narrative text (no headers, no bullets, no tags, no code blocks). "
            "If a field is missing in the JSON, write 'not specified' rather than guessing. "
            "When describing predictions, list ALL values exactly as provided; "
            "Keep it concise and fluent."
            )

            user_msg = (
            "Rewrite this JSON result into a single, smooth paragraph with complete sentences. "
            "Do not invent information. Use only what is present in the JSON. "
            "If a field is missing, say 'not specified'.\n\n"
            f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
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
        """
        Deterministic fallback: generate a plain narrative without LLM.
        """
        try:
            d = payload.get("data", {})
            taskId = d.get("taskId", "N/A")
            timeRange = d.get("timeRange", "N/A")
            sensor = d.get("sensor_name", "N/A")
            target = d.get("target_col", d.get("target_cols", "N/A"))
            model = d.get("modelSelected", "Unknown")
            preds = d.get("prediction", [])
            pred_len = len(preds) if isinstance(preds, list) else "N/A"

            # prediction preview
            if isinstance(preds, list) and len(preds) > 10:
                pred_text = f"first values {preds[:3]} ... last values {preds[-3:]} (total {len(preds)})"
            else:
                pred_text = str(preds)

            risk = d.get("risk", {})
            risk_level = risk.get("riskLevel", "unknown")

            return (
                f"This is the answer for request {taskId}. "
                f"The request covered {timeRange} on sensor {sensor}, targeting column {target}. "
                f"We selected model {model}, which produced {pred_len} predictions: {pred_text}. "
                f"The risk level was assessed as {risk_level}. "
                f"Based on the data, the current state is considered {risk_level}, "
                f"and no immediate action is required unless stated otherwise."
            )
        except Exception as e:
            return f"Could not generate narrative fallback: {e}"

    # 아래는 한국어 설명에 대한 함수들 (TBU)
    def _narrate(self, payload: Dict[str, Any]) -> str:
        """
        입력 JSON(payload)의 모든 필드를 '누락 없이' 한국어로 풀어쓴 설명문을 생성.
        - 1순위: LLM으로 생성 (형식 엄격 유도)
        - 실패 시: 결정적 폴백 포맷터로 즉시 반환
        """
        try:
            system_msg = (
                "너의 임무는 제조 예측 파이프라인의 JSON 결과를 한국어로 "
                "필드 누락 없이 그대로 설명문으로 풀어쓰는 것이다. "
                "절대 새로운 가정/수치/해석을 추가하지 말고 JSON에 있는 값만 사용하라. "
                "각 필드는 사람이 즉시 이해할 수 있도록 간결하게 항목별로 써라. "
                "출력은 순수 텍스트이며 Markdown 불릿을 사용하되 표나 코드블록은 쓰지 마라."
            )
            user_msg = (
                "다음 JSON의 모든 항목을 한국어로 자세히 설명하되, 아래 출력 형식을 지켜라.\n\n"
                "출력 형식:\n"
                "제목: PRISM 예측 결과 상세 보고\n"
                "\n"
                "섹션1: 요청 정보\n"
                "- taskId: ...\n"
                "- timeRange: ... (없으면 '정보 없음')\n"
                "- sensor_name / target_col(또는 target_cols): ...\n"
                "\n"
                "섹션2: 모델/예측\n"
                "- modelSelected: ...\n"
                "- pred_len: ...\n"
                "- prediction: 값이 많으면 앞 5개와 뒤 5개, 전체 개수 표기. (예: [앞 5] 1,2,3,4,5 / [뒤 5] ... / 총 N개)\n"
                "\n"
                "섹션3: 위험도\n"
                "- risk.riskLevel: ...\n"
                "- risk.exceedsThreshold 등 부가 필드가 있으면 모두 명시\n"
                "- suggestedActions가 있으면 모두 나열, 없으면 '정보 없음'\n"
                "\n"
                "섹션4: 변수 기여(설명)\n"
                "- explanation.importantFeatures: 이름과(있으면) 기여도, 최대 5개\n"
                "- explanation.method: ... (있으면)\n"
                "\n"
                "섹션5: 데이터/파이프라인 정보\n"
                "- csv_path, df_info(rows, cols), features_start_col_index_1based 등 데이터 관련 모든 필드 표시\n"
                "- feature_names는 개수가 많으면 첫 5개 + 총 개수 표기\n"
                "\n"
                "섹션6: 처리 이벤트 타임라인\n"
                "- events 전체를 시간순으로 한 줄씩 간결히 설명 (없으면 '정보 없음')\n"
                "\n"
                "섹션7: 결론\n"
                "- 현재 상태(정상/주의/위험 등 JSON 근거로) 한 줄\n"
                "- 바로 취해야 할 조치(있으면), 없으면 '특이 조치 없음'\n"
                "\n"
                "JSON:\n"
                f"{json.dumps(payload, ensure_ascii=False, indent=2)}"
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

    # deterministic fallback formatter
    def _fallback_ko(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("PRISM 예측 결과 상세 보고 (폴백 모드)")
        lines.append("")

        def fmt_scalar(v: Any) -> str:
            if v is None:
                return "정보 없음"
            if isinstance(v, float):
                return f"{v:.6g}"
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
                            item_key = f"{key}[{i}]"
                            walk(item_key, item, indent + 1)
            else:
                lines.append(f"{pad}{bullet}{key}: {fmt_scalar(val)}")

        top_keys = list(payload.keys())
        for first in ("data", "metadata", "code"):
            if first in top_keys:
                top_keys.remove(first)
                top_keys.insert(0, first)

        for k in top_keys:
            walk(k, payload[k], 0)

        return "\n".join(lines)
