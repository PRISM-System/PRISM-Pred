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



## 3. 설치 & 실행

### 3-1. 의존성 설치
```bash
pip install -r requirements.txt
```

### 3-2. 환경 변수 설정 (`.env`)
```env
OPENAI_BASE_URL=http://localhost:8000/v1
OPENAI_MODEL=gpt-oss-120b
OPENAI_API_KEY=EMPTY-KEY
```

### 3-3. 서버 실행
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

서버 시작 시 LLM 연결 상태가 로그로 표시됩니다:
```
[LLM] Connected → model=gpt-oss-120b
```

---

## 4. 예측 API

### 엔드포인트
```
POST /api/v1/prediction/run-direct
```

---

## 4-1. 실행 예시

### (A) 시나리오 JSON 파일로 실행
```bash
curl -X POST "http://localhost:8001/api/v1/prediction/run-direct"   -H "Content-Type: application/json"   --data @scenario/scenario02.json
```

### (B) 직접 JSON을 입력하여 실행
```bash
curl -X POST http://localhost:8001/api/v1/prediction/run-direct   -H "Content-Type: application/json"   --data-binary @- <<'JSON'
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


#### 참고

- 현재 DB는 `./prism_prediction/Industrial_DB_sample/dataset_v3`의 하위 파일인 CMP/CVD/ETCH/ION/PHOTO 공정의 데이터를 사용합니다.
- 서버 로그는 uvicorn을 실행한 터미널에서 확인할 수 있습니다.

---

## 응답 예시 (실제 출력)

아래는 예측 API 호출 시의 예시 응답입니다.  
모델/버전/데이터에 따라 수치는 달라질 수 있습니다.

```json
{
  "mode": "csv",
  "csv_path": "prism_prediction/Industrial_DB_sample/SEMI_CMP_SENSORS_predict.csv",
  "seq_len": 48,
  "label_len": 24,
  "pred_len": 11,
  "enc_in": 11,
  "c_out": 11,
  "eval_channel_idx": 8,
  "selection_metric": "val_rmse_focus",
  "best_model": "TimesNet",
  "best_val_rmse_focus": 3.2319672107696533,
  "best_val_rmse_all": 235.93780517578125,
  "results": {
    "Autoformer": { "val_rmse_all": 248.3970184326172, "val_rmse_focus": 108.75589752197266 },
    "DLinear":    { "val_rmse_all": 255.54910278320312, "val_rmse_focus": 3.3810741901397705 },
    "TimesNet":   { "val_rmse_all": 235.93780517578125, "val_rmse_focus": 3.2319672107696533 },
    "LightTS":    { "val_rmse_all": 1152.325439453125, "val_rmse_focus": 19.069955825805664 }
  },
  "prediction": [18.095299, 16.372711, 18.664963, 14.825407, 16.47625, 16.487608, 17.424318, 16.602634, 17.291752, 15.002139, 18.01297],
  "target_col_name": "MOTOR_CURRENT",
  "feature_start_col": 5
}
```
