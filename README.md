# PRISM-Pred: 예측 AI 에이전트

`PRISM-Pred`는 [PRISM-AGI](../README.md) 플랫폼의 예측을 담당하는 AI 에이전트로, 멀티모달 데이터를 통합 분석하여 공정 결과를 예측합니다.

---

## 1. 주요 기능

### Orchestration에서 응답을 자연어로 요청 받고, 최종 응답을 자연어로 반환하는 모듈
- 자연어 질의 처리 가능
- 도출 결과를 다시 자연어로 처리하여 반환 가능

### 멀티모달 예측 시스템
- 정형 데이터를 분석하는 전문가 모델
- 이미지, 텍스트 등 비정형 데이터를 분석하는 전문가 모델
- 다양한 형태의 데이터를 통합하여 종합적으로 분석하는 멀티모달 모델
- 주어진 과업에 가장 적합한 분석 전문가를 자동으로 할당하는 알고리즘

### 데이터 타입별 전문가 풀
- 테이블(Table) 데이터 처리 전문가 모델
- 이미지(Image) 데이터 분석 전문가 모델
- 텍스트(Text) 데이터 처리 전문가 모델
- 시계열(Time-series) 데이터 분석 전문가 모델
- 각 전문가 모델의 예측 결과를 융합하여 성능을 극대화하는 앙상블 및 협업 메커니즘

### 예측 신뢰도 관리
- 예측 결과의 신뢰도를 정량적으로 평가하는 시스템
- 모델의 불확실성을 측정하고 진단하는 모듈
- 새로운 데이터에 스스로 적응하며 성능을 개선하는 자가 발전 기능 (도메인 적응)
- 예측 결과에 따르는 잠재적 위험을 평가하는 기능

---

## 2. 성능 목표

| 기능             | 지표                   | 목표     |
| ---------------- | ---------------------- | -------- |
| **예측 정확도** | 각 데이터 타입별 예측 오차 | 5% 이내 |
| **신뢰도 관리** | 예측 위험 평가 상관계수 | 0.5 이상 |

---



## 3. 시연용 방법을 자세히 설명드립니다.

### 3-1. 먼저 의존성을 설치합니다.
```bash
pip install -r requirements.txt
```

### 3-2. 그리고, env 파일에 아래 양식으로 작성해줍니다. 이 때, 비아이매트릭스 측에서 배포해주신 모델들을 활용하고 싶다면 (A), 자체 openai api를 활용하고 싶다면 (B) 형식으로 작성해주시면 됩니다. 
##### (A)
(`.env`)

```env
MAX_LENGTH=512
BIMATRIX_BASE_URL=.
BIMATRIX_ID=.
BIMATRIX_PW=.
BIMATRIX_VERIFY=true
OPENAI_MODEL=/root/models/openai/gpt-oss-120b 
```
##### (B)
(`.env`)

```env
BIMATRIX_VERIFY=false
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini(변경 가능)
MAX_LENGTH=512
OPENAI_API_KEY = .

```

### 3-3. 이 후, 서버를 실행합니다.
```bash
uvicorn main:app --host 0.0.0.0 --port 8003 --reload
```

서버 시작 시 LLM 연결 상태가 로그로 표시됩니다:
```
[LLM] Connected → model=gpt-oss-120b
```

---

### 4-1. 서버를 실행한 이후, 예측 에이전트에 scenario의 input을 입력하면 output을 출력함을 확인할 수 있습니다.(--data @데이터 json 위치 경로 입력 시, 해당 json에 대한 응답이 출력되며, 현재 scenario 2,3,4,6,11 의 형식과 호환 완료된 상태입니다.)


```
curl -X POST "http://localhost:8003/api/v1/prediction/run-direct"   -H "Content-Type: application/json"   --data @scenarios/scenario02.json

```
### 4-2. 서버를 실행한 이후,직접 입력을 넣으셔도 작동합니다.

```bash
curl -X POST http://localhost:8003/api/v1/prediction/run-direct   -H "Content-Type: application/json"   --data-binary @- <<'JSON'
{
  "step_4_orchestration_to_prediction": {
    "from": "Orchestration",
    "to": "Predictive",
    "timestamp": "2025-05-01T14:20:07Z",
    "api_endpoint": "POST /api/v1/prediction/run-direct",
    "request": {
      "taskId": "ETCH_TASK_20250501_002_2",
      "timeRange": {
        "start": "2025-05-01T12:50:00Z",
        "end": "2025-05-01T14:20:00Z"
      },
      "sensor_name": "ETCH_CH1,ETCH_CH2,ETCH_CH3,ETCH_CH4",
      "target_cols": ["PRESSURE", "PROCESS_QUALITY_INDEX"],
      "feature_cols": ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE"],
      "prediction_horizon_minutes": 90,
      "prediction_interval_minutes": 5,
      "confidence_level": 0.95
    }
  }
}
JSON
```

### 엔드포인트
```
POST /api/v1/prediction/run-direct
```


#### 참고

- 현재 DB는 `./prism_prediction/Industrial_DB_sample/dataset_v3`의 하위 파일인 CMP/CVD/ETCH/ION/PHOTO 공정의 데이터를 사용합니다.
- 서버 로그는 uvicorn을 실행한 터미널에서 확인할 수 있습니다.

---

## 응답 예시 (실제 출력)

아래는 예측 API 호출 시의 예시 응답입니다.  
모델/버전/데이터에 따라 수치는 달라질 수 있습니다.

```json
{
  "code": "SUCCESS",
  "data": {
    "result": "# 산업 공정 예측 리포트\n\n## 1. 개요\n현재 압력이 8.7 mTorr로 정상 범위(5.0-7.0 mTorr)를 24.3% 초과하여 지속 상승 중입니다. 진공 펌프의 효율이 72.3%로 저하되어 압력 상승의 주요 원인으로 확인되었습니다. 현재 압력 상승률은 +0.028 mTorr/분입니다.\n\n## 2. 예측 결과\n향후 90분간의 압력 및 진공 펌프 효율 예측 결과는 다음과 같습니다.\n\n| 시간 (UTC) | 압력 (mTorr) | 진공 펌프 효율 (%) |\n|-------------|--------------|---------------------|\n| 2025-05-01 14:25 | 93.83 | 90.07 |\n| 2025-05-01 14:30 | 93.58 | 89.92 |\n| 2025-05-01 14:35 | 92.86 | 90.07 |\n| 2025-05-01 14:40 | 94.16 | 89.94 |\n| 2025-05-01 14:45 | 93.57 | 90.02 |\n| 2025-05-01 14:50 | 92.54 | 90.03 |\n| 2025-05-01 14:55 | 92.48 | 90.06 |\n| 2025-05-01 15:00 | 92.94 | 90.04 |\n| 2025-05-01 15:05 | 92.26 | 89.95 |\n| 2025-05-01 15:10 | 93.24 | 90.11 |\n| 2025-05-01 15:15 | 93.72 | 89.99 |\n| 2025-05-01 15:20 | 93.45 | 90.05 |\n| 2025-05-01 15:25 | 92.64 | 90.04 |\n| 2025-05-01 15:30 | 93.90 | 89.93 |\n| 2025-05-01 15:35 | 92.63 | 90.09 |\n| 2025-05-01 15:40 | 93.12 | 90.04 |\n| 2025-05-01 15:45 | 93.21 | 89.95 |\n| 2025-05-01 15:50 | 93.18 | 89.97 |\n\n## 3. 임계치 도달 분석\n- **임계치(10.0 mTorr) 도달 시점**: 예측된 압력은 90분 후에도 10.0 mTorr에 도달하지 않을 것으로 보입니다. 그러나 현재 압력이 이미 정상 범위를 초과하고 있어, 지속적인 모니터링이 필요합니다.\n- **인터록 작동 가능성**: 현재 위험 수준은 \"높음\"으로 평가되며, 압력이 계속 상승할 경우 인터록 작동 가능성이 존재합니다. 따라서 즉각적인 조치가 필요합니다.\n\n## 4. 결론\n압력 상승이 지속되고 있으며, 진공 펌프의 효율 저하가 주요 원인으로 확인되었습니다. 향후 90분간의 예측 결과에 따르면, 압력이 10.0 mTorr에 도달하지는 않겠지만, 현재의 높은 위험 수준을 고려할 때 즉각적인 조치가 필요합니다.",
    "raw": {
      "spec": {
        "taskId": "ETCH_TASK_20250501_002_2",
        "query": "모니터링 결과 압력이 8.7 mTorr로 정상 범위(5.0-7.0 mTorr)를 24.3% 초과하여 지속 상승 중입니다. 진공 펌프 효율이 72.3%로 저하되어 압력 상승의 주요 원인으로 확인되었습니다. 현재 상승률(+0.028 mTorr/분)이 계속될 경우 향후 90분간 압력과 공정 품질이 어떻게 전개될지 예측해주세요. 특히 임계치(10.0 mTorr) 도달 시점과 인터록 작동 가능성을 분석해주세요.",
        "timeRange": {
          "start": "2025-05-01T12:50:00Z",
          "end": "2025-05-01T14:20:00Z"
        },
        "sensor_name": "CHAMBER_E1,CHAMBER_E2,CHAMBER_E3,CHAMBER_E4",
        "target_cols": ["PRESSURE", "VACUUM_PUMP"],
        "feature_cols": ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE"],
        "prediction_horizon_minutes": 90,
        "prediction_interval_minutes": 5,
        "model_type": "lstm",
        "confidence_level": 0.95
      },
      "csv_path": "prism_prediction/Industrial_DB_sample/dataset_v3/test_scenarios/test_data/semiconductor/semiconductor_etch_002.csv",
      "df_info": { "rows": 5000, "cols": 11 },
      "feature_names": ["PRESSURE","VACUUM_PUMP","GAS_FLOW_RATE","RF_POWER","TEMPERATURE","ETCH_RATE","BIAS_VOLTAGE","CHAMBER_HUMIDITY","GAS_COMPOSITION"],
      "enc_in": 9,
      "target_col": "PRESSURE",
      "target_idx_in_features": 0,
      "pred_len": 18,
      "confidence_level": 0.95,
      "sensor_name": "CHAMBER_E1,CHAMBER_E2,CHAMBER_E3,CHAMBER_E4",
      "risk": { "riskLevel": "high", "exceedsThreshold": true },
      "explanation": {
        "importantFeatures": ["GAS_FLOW_RATE","ETCH_RATE","TEMPERATURE","RF_POWER","BIAS_VOLTAGE"],
        "method": "corr-proxy"
      }
    }
  },
  "metadata": {
    "timestamp": "2025-11-03T00:29:25Z",
    "request_id": "req_6be6d296"
  }
}
