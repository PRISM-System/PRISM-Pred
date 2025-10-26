# PRISM Test Scenarios

PRISM 시스템의 통합 테스트를 위한 시나리오 모음입니다.

## 개요 Overview

이 문서는 PRISM의 AI 에이전트 기반 자율제조 시스템을 검증하기 위한 테스트 시나리오를 정리합니다.

### 구성 요소
- 총 5개 산업 공정 환경 (반도체, 화학, 자동차, 배터리, 철강)
- 총 4가지 AI 에이전트 워크플로우 (4 레벨)
- 총 20개의 테스트 시나리오 (5개 산업 × 4개 워크플로우)
- 각 산업별 실제 공정 환경 기반 센서 데이터

## 워크플로우 Workflow Patterns

### 1. Monitoring Only (Level 1)
```
Query → Orchestration → Monitoring Agent → Orchestration → Answer
```
- 모니터링 에이전트만 사용
- 이상 감지 및 분석
- 예시: "CMP 공정의 Slurry 유량이 이상합니다"

### 2. Monitoring + Prediction (Level 2)
```
Query → Orchestration → Monitoring → Orchestration → Predictive → Orchestration → Answer
```
- 이상 감지 + 예측 분석 결합
- 미래 상태 예측 제공
- 예시: "현재 압력 증가율로 30분 후 임계치 도달 예상"

### 3. Monitoring + Prediction + AutoControl (Level 3)
```
Query → Orchestration → Monitoring → Orchestration → Predictive → Orchestration → AutoControl → Orchestration → Answer
```
- 예측 기반 자동 제어까지 수행
- 능동적인 조치 제안 및 실행
- 예시: "불량률 10% 감소, 공정 안정 5% 향상 달성"

### 4. Full Compliance Check (Level 4)
```
Query → Orchestration → Monitoring → Orchestration → Predictive → Orchestration → AutoControl → Orchestration → Compliance Check → Answer
```
- 조치 수행의 규제 및 안전 검증
- 표준 준수 여부 검사 수행
- 예시: "조치 수행이 규제 준수, 안전 기준 충족"

## 디렉토리 Directory Structure

```
test-scenarios/
├── README.md                    # 이 문서
├── QUICK_GUIDE.md               # 빠른 가이드 문서
├── SCENARIO_INDEX.json          # 전체 시나리오 인덱스 정보
├── api_endpoint.xlsx            # API 엔드포인트 명세
│
├── scenarios/                   # 시나리오 문서
│   ├── semiconductor/           # 반도체 공정 (4개 시나리오)
│   │   ├── SCENARIO_01.json     # Level 1: CMP 공정 모니터링
│   │   ├── SCENARIO_02.json     # Level 2: Etching 공정 예측
│   │   ├── SCENARIO_03.json     # Level 3: Deposition 공정 제어
│   │   └── SCENARIO_04.json     # Level 4: Full 워크플로우
│   │
│   ├── chemical/                # 화학 공정 (4개 시나리오)
│   │   ├── SCENARIO_05.json
│   │   ├── SCENARIO_06.json
│   │   ├── SCENARIO_07.json
│   │   └── SCENARIO_08.json
│   │
│   ├── automotive/              # 자동차 공정 (4개 시나리오)
│   │   ├── SCENARIO_09.json
│   │   ├── SCENARIO_10.json
│   │   ├── SCENARIO_11.json
│   │   └── SCENARIO_12.json
│   │
│   ├── battery/                 # 배터리 공정 (4개 시나리오)
│   │   ├── SCENARIO_13.json
│   │   ├── SCENARIO_14.json
│   │   ├── SCENARIO_15.json
│   │   └── SCENARIO_16.json
│   │
│   └── steel/                   # 철강 공정 (4개 시나리오)
│       ├── SCENARIO_17.json
│       ├── SCENARIO_18.json
│       ├── SCENARIO_19.json
│       └── SCENARIO_20.json
│
├── datasets/                    # 센서 데이터셋
│   ├── semiconductor_cmp_001.csv
│   ├── semiconductor_etch_002.csv
│   ├── chemical_reactor_001.csv
│   ├── automotive_welding_001.csv
│   ├── battery_formation_001.csv
│   └── steel_rolling_001.csv
│
└── test_data/                   # 데이터 생성 템플릿
    └── sample_data_templates.json

```

## 데이터셋 Dataset Specifications

### 센서 데이터 공통 규격
- **시간 간격**: 10초 간격 (실시간 수집 상황)
- **데이터 포인트**: 5,000개 이상
- **변수 개수**: 10개 이상
- **시작 시각**: 2025년 5월 1일 00:00:00
- **파일 형식**: CSV (UTF-8 인코딩)
- **타임스탬프**: ISO 8601 형식 (YYYY-MM-DDTHH:MM:SSZ)

### 산업별 센서 변수 예시

| 산업 | 센서 변수 |
|------|----------|
| **반도체 CMP** | MOTOR_CURRENT, HEAD_ROTATION, SLURRY_FLOW_RATE, PRESSURE, TEMPERATURE |
| **화학 반응기** | TEMPERATURE, PRESSURE, pH, CONCENTRATION, FEED_RATE, CATALYST_RATIO |
| **자동차 용접** | WELD_CURRENT, WELD_VOLTAGE, WIRE_SPEED, TRAVEL_SPEED, SHIELDING_GAS |
| **배터리 화성** | VOLTAGE, CURRENT, CAPACITY, TEMPERATURE, SOC, CYCLE |
| **철강 압연** | THICKNESS, ROLL_GAP, ROLLING_SPEED, TENSION, TEMPERATURE |

## 시나리오 Scenario File Structure

각 시나리오 JSON 파일은 다음 구조로 구성됩니다:

```json
{
  "scenario_id": "SCENARIO_01",
  "title": "반도체 CMP 공정 모니터 에이전트",
  "workflow_type": "monitoring_only",
  "industry": "semiconductor",
  "process": "CMP",

  "query": {
    "text": "최근 CMP 공정에서 Slurry 유량이 불안정합니다. 현재 상황을 분석해주세요.",
    "timestamp": "2025-05-01T13:50:00Z",
    "timeseries_data": {
      "file": "semiconductor_cmp_001.csv",
      "columns": ["TIMESTAMP", "MOTOR_CURRENT", "SLURRY_FLOW_RATE", ...],
      "time_range": {
        "start": "2025-05-01T00:00:00Z",
        "end": "2025-05-01T13:50:00Z"
      }
    }
  },

  "agent_workflow": {
    "orchestration_1": { ... },
    "monitoring_agent": { ... },
    "orchestration_2": { ... }
  },

  "expected_answer": {
    "summary": "...",
    "detected_issues": [...],
    "recommendations": [...]
  }
}
```

## 빠른 시작 Quick Start

### 1. 시나리오 전체 목록 확인
```bash
cat SCENARIO_INDEX.json | jq '.scenarios[] | {id, title, workflow_type}'
```

### 2. 특정 시나리오 읽기
```bash
cat scenarios/semiconductor/SCENARIO_01.json | jq
```

### 3. 센서 데이터 확인
```bash
head -20 datasets/semiconductor_cmp_001.csv
```

### 4. Python으로 시나리오 및 데이터 로드
```python
import json
import pandas as pd

# 시나리오 로드
with open('scenarios/semiconductor/SCENARIO_01.json') as f:
    scenario = json.load(f)

# 센서 데이터 로드
data_file = scenario['query']['timeseries_data']['file']
df = pd.read_csv(f'datasets/{data_file}')

# PRISM API 호출
# ...
```

## 명명 규칙 Naming Conventions

### 시나리오 파일명
- 형식: `SCENARIO_<번호>.json` (번호는 01부터 20까지 zero-padded)
- 위치: `scenarios/<industry>/`

### 데이터셋 파일명
- 형식: `<industry>_<process>_<번호>.csv`
- 예시: `semiconductor_cmp_001.csv`, `battery_formation_002.csv`

### 변수명 규칙
- 모두 대문자
- 단 단어는 언더스코어(_)
- 실제 공정 현장 용어 사용
- 예시: `MOTOR_CURRENT`, `SLURRY_FLOW_RATE`, `RF_POWER`

## 테스트 Test Guidelines

### 테스트 실행 절차
1. **시나리오 선택**: SCENARIO_INDEX.json에서 테스트할 시나리오 확인
2. **데이터 검증**: 해당 시나리오의 센서 데이터 파일 확인
3. **API 호출**: 자연어 쿼리와 데이터를 PRISM 시스템에 전달
4. **결과 비교**: 실제 응답과 expected_answer 비교
5. **성능 측정**: 응답 시간, 정확도, 규제 준수 여부

### 검증 항목
- 각 워크플로우 단계별 정확도
- 에이전트 간 데이터 전달 정확성
- 이상 감지 알고리즘 성능
- 예측 정확도의 신뢰도
- 조치 수행의 규제 준수 검증
- 전체 워크플로우 응답 시간

## 참고 문서 Reference Documents

- **QUICK_GUIDE.md**: 시나리오 생성 가이드라인
- **api_endpoint.xlsx**: PRISM API 엔드포인트 명세
- **test_data/sample_data_templates.json**: 데이터 생성 템플릿
- **ppt.pdf**: PRISM 시스템 전체 구조 설명

## 유지보수 Maintenance

### 시나리오 추가 방법
1. `scenarios/<industry>/` 디렉토리에 새 JSON 파일 생성
2. `SCENARIO_INDEX.json`에 인덱스 정보 추가
3. 필요시 센서 데이터 CSV 생성 후 `datasets/`에 저장
4. 테스트 실행 후 검증

### 데이터셋 업데이트
- 센서 데이터는 CSV 형식으로 저장
- 템플릿은 `test_data/sample_data_templates.json` 참조
- 이상치 삽입 시 실제 공정 상황을 반영하여 생성

## 문의 Contact

프로젝트 관련 문의:
- **주관기관**: 서울대학교 DSBA Lab
- **책임연구원**: 강필성 교수
- **과제번호**: RS-2025-02214591

---

**Last Updated**: 2025-10-14
**Version**: 1.0
