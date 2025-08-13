# PRISM-Pred: 예측 AI 에이전트

`PRISM-Pred`는 [PRISM-AGI](../README.md) 플랫폼의 예측을 담당하는 AI 에이전트로, 멀티모달 데이터를 통합 분석하여 공정 결과를 예측합니다.

## 1. 주요 기능 ###

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

## 2. 성능 목표

| 기능 | 지표 | 목표 |
| --- | --- | --- |
| **예측 정확도** | 각 데이터 타입별 예측 오차 | 5% 이내 |
| **신뢰도 관리** | 예측 위험 평가 상관계수 | 0.5 이상 |

## 3. 실행 방법
- 시계열 전용 후보 모델 학습 과정 업데이트
- 아래 예시 코드로 실행 가능

### 3-1. 의존성 설치
> 프로젝트 루트에서 아래 명령으로 필요한 패키지를 설치합니다.
```
pip install -r requirements.txt

```
### 3-2. API 서버 실행
FastAPI 서버를 띄웁니다.
```
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

### 3-3. 예측 API 호출
```
curl -s -X POST "http://127.0.0.1:8001/api/v1/prediction/run-direct" -H "Content-Type: application/json" -d '{"taskId":"1","fromAgent":"orchestration","objective":"prediction","timeRange":"2025-08-20 09:00:00 - 09:10:00","sensor_name":"CMP","target_cols":"MOTOR_CURRENT","constraints":null,"userRole":"engineer"}' | jq .

```
#### 참고

- 기본 CSV 매핑은 prism_prediction/Industrial_DB_sample 하위 파일을 사용합니다.

- 서버 로그는 uvicorn을 실행한 터미널에서 확인할 수 있습니다.