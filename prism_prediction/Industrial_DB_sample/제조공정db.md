# 반도체 공정 이상탐지 DB Schema

## 핵심 테이블 구조

### 1. 테이블명: SEMI_LOT_MANAGE -- 생산 LOT 관리 테이블

| 설명 | | |
|------|------|------|
| LOT별 생산 이력을 관리하는 중심 테이블. 이상탐지 시 LOT 단위 추적과 수율 이상 패턴 감지의 기준점 | | |

| **칼럼명** | **데이터타입** | **설명** |
|------------|----------------|----------|
| PNO | VARCHAR(PK) | LOT 관리 고유번호 |
| LOT_NO | VARCHAR | LOT 번호 |
| PRODUCT_NAME | VARCHAR | 제품명 |
| RECIPE_ID | VARCHAR | 사용 레시피 ID |
| START_QTY | INT | 시작 웨이퍼 수 |
| CURRENT_STEP | VARCHAR | 현재 공정 단계 |
| PRIORITY | VARCHAR | 우선순위(HOT/NORMAL) |
| CREDATE | DATE | LOT 생성일 |
| HOLDER | VARCHAR | 보류 사유(이상 발생 시) |
| FINAL_YIELD | FLOAT | 최종 수율(%) |
| GOOD_DIE | INT | 양품 다이 수 |
| TOTAL_DIE | INT | 전체 다이 수 |

### 2. 테이블명: SEMI_PROCESS_HISTORY -- 공정별 이력 추적 테이블

| **칼럼명** | **데이터타입** | **설명** |
|------------|----------------|----------|
| PNO | VARCHAR(PK) | 공정 이력 고유번호 |
| LOT_NO | VARCHAR | LOT 번호 |
| PROCESS_STEP | VARCHAR | 공정 단계 |
| EQUIPMENT_ID | VARCHAR | 사용 장비 ID |
| RECIPE_ID | VARCHAR | 적용 레시피 ID |
| OPERATOR | VARCHAR | 작업자 ID |
| START_TIME | DATETIME | 공정 시작 시각 |
| END_TIME | DATETIME | 공정 종료 시각 |
| IN_QTY | INT | 투입 수량 |
| OUT_QTY | INT | 산출 수량 |

### 3. 테이블명: SEMI_PARAM_MEASURE -- 공정 파라미터 측정 결과 테이블

| **칼럼명** | **데이터타입** | **설명** |
|------------|----------------|----------|
| PNO | VARCHAR(PK) | 측정 고유번호 |
| LOT_NO | VARCHAR | LOT 번호 |
| WAFER_ID | VARCHAR | 웨이퍼 ID |
| PROCESS_STEP | VARCHAR | 공정 단계 |
| EQUIPMENT_ID | VARCHAR | 측정 장비 ID |
| CATEGORY | VARCHAR | 측정 항목 분류(CD, THICKNESS 등) |
| PARAM_NAME | VARCHAR | 파라미터명 |
| UNIT | VARCHAR | 측정 단위(nm, Å, Ω/sq 등) |
| MEASURED_VAL | DECIMAL | 실제 측정값 |
| TARGET_VAL | DECIMAL | 목표값 |
| USL | DECIMAL | 상한 규격(Upper Spec Limit) |
| LSL | DECIMAL | 하한 규격(Lower Spec Limit) |
| MEASURE_TIME | DATETIME | 측정 시각 |

### 4. 테이블명: SEMI_EQUIPMENT_SENSOR -- 장비 센서 데이터 테이블

| **칼럼명** | **데이터타입** | **설명** |
|------------|----------------|----------|
| PNO | VARCHAR(PK) | 센서 데이터 고유번호 |
| EQUIPMENT_ID | VARCHAR | 장비 ID |
| LOT_NO | VARCHAR | 처리 중인 LOT 번호 |
| SENSOR_TYPE | VARCHAR | 센서 종류(TEMP/PRESSURE/FLOW 등) |
| SENSOR_VALUE | DECIMAL | 센서 측정값 |
| TIMESTAMP | DATETIME | 측정 시각 |
| CHAMBER_ID | VARCHAR | 챔버 ID(멀티 챔버 장비) |
| RECIPE_STEP | INT | 레시피 단계 |

## 공정별 센서 상세 스키마

### 5. 테이블명: SEMI_PHOTO_SENSORS -- 포토리소그래피 공정 센서

| **칼럼명** | **데이터타입** | **설명** | **정상범위** |
|------------|----------------|----------|--------------|
| PNO | VARCHAR(PK) | 측정 고유번호 | - |
| EQUIPMENT_ID | VARCHAR | 노광 장비 ID | - |
| LOT_NO | VARCHAR | 처리 LOT 번호 | - |
| WAFER_ID | VARCHAR | 웨이퍼 ID | - |
| TIMESTAMP | DATETIME | 측정 시각 | - |
| EXPOSURE_DOSE | DECIMAL | 노광 에너지 (mJ/cm²) | 20-40 |
| FOCUS_POSITION | DECIMAL | 포커스 위치 (nm) | ±50 |
| STAGE_TEMP | DECIMAL | 스테이지 온도 (°C) | 23±0.1 |
| BAROMETRIC_PRESSURE | DECIMAL | 대기압 (hPa) | - |
| HUMIDITY | DECIMAL | 습도 (%) | 45±5 |
| ALIGNMENT_ERROR_X | DECIMAL | X축 정렬 오차 (nm) | <3 |
| ALIGNMENT_ERROR_Y | DECIMAL | Y축 정렬 오차 (nm) | <3 |
| LENS_ABERRATION | DECIMAL | 렌즈 수차 (mλ) | <5 |
| ILLUMINATION_UNIFORMITY | DECIMAL | 조명 균일도 (%) | >98 |
| RETICLE_TEMP | DECIMAL | 레티클 온도 (°C) | 23±0.05 |

### 6. 테이블명: SEMI_ETCH_SENSORS -- 에칭 공정 센서

| **칼럼명** | **데이터타입** | **설명** | **정상범위** |
|------------|----------------|----------|--------------|
| PNO | VARCHAR(PK) | 측정 고유번호 | - |
| EQUIPMENT_ID | VARCHAR | 에칭 장비 ID | - |
| LOT_NO | VARCHAR | 처리 LOT 번호 | - |
| TIMESTAMP | DATETIME | 측정 시각 | - |
| RF_POWER_SOURCE | DECIMAL | Source RF 파워 (W) | 500-2000 |
| RF_POWER_BIAS | DECIMAL | Bias RF 파워 (W) | 50-500 |
| CHAMBER_PRESSURE | DECIMAL | 챔버 압력 (mTorr) | 5-200 |
| GAS_FLOW_CF4 | DECIMAL | CF4 가스 유량 (sccm) | 0-200 |
| GAS_FLOW_O2 | DECIMAL | O2 가스 유량 (sccm) | 0-100 |
| GAS_FLOW_AR | DECIMAL | Ar 가스 유량 (sccm) | 0-500 |
| GAS_FLOW_CL2 | DECIMAL | Cl2 가스 유량 (sccm) | 0-200 |
| ELECTRODE_TEMP | DECIMAL | 전극 온도 (°C) | 40-80 |
| CHAMBER_WALL_TEMP | DECIMAL | 챔버 벽 온도 (°C) | 60-80 |
| HELIUM_PRESSURE | DECIMAL | He 배압 (Torr) | 5-20 |
| ENDPOINT_SIGNAL | DECIMAL | 종점 검출 신호 강도 (a.u.) | - |
| PLASMA_DENSITY | DECIMAL | 플라즈마 밀도 (ions/cm³) | 1e10-1e12 |

### 7. 테이블명: SEMI_CVD_SENSORS -- 화학기상증착 공정 센서

| **칼럼명** | **데이터타입** | **설명** | **정상범위** |
|------------|----------------|----------|--------------|
| PNO | VARCHAR(PK) | 측정 고유번호 | - |
| EQUIPMENT_ID | VARCHAR | CVD 장비 ID | - |
| LOT_NO | VARCHAR | 처리 LOT 번호 | - |
| TIMESTAMP | DATETIME | 측정 시각 | - |
| SUSCEPTOR_TEMP | DECIMAL | 서셉터 온도 (°C) | 300-700 |
| CHAMBER_PRESSURE | DECIMAL | 챔버 압력 (Torr) | 0.1-760 |
| PRECURSOR_FLOW_TEOS | DECIMAL | TEOS 유량 (sccm) | 0-500 |
| PRECURSOR_FLOW_SILANE | DECIMAL | SiH4 유량 (sccm) | 0-1000 |
| PRECURSOR_FLOW_WF6 | DECIMAL | WF6 유량 (sccm) | 0-100 |
| CARRIER_GAS_N2 | DECIMAL | N2 캐리어 가스 유량 (slm) | 0-20 |
| CARRIER_GAS_H2 | DECIMAL | H2 캐리어 가스 유량 (slm) | 0-10 |
| SHOWERHEAD_TEMP | DECIMAL | 샤워헤드 온도 (°C) | 150-250 |
| LINER_TEMP | DECIMAL | 라이너 온도 (°C) | 100-200 |
| DEPOSITION_RATE | DECIMAL | 증착 속도 (Å/min) | - |
| FILM_STRESS | DECIMAL | 박막 스트레스 (MPa) | - |

### 8. 테이블명: SEMI_IMPLANT_SENSORS -- 이온주입 공정 센서

| **칼럼명** | **데이터타입** | **설명** | **정상범위** |
|------------|----------------|----------|--------------|
| PNO | VARCHAR(PK) | 측정 고유번호 | - |
| EQUIPMENT_ID | VARCHAR | 이온주입 장비 ID | - |
| LOT_NO | VARCHAR | 처리 LOT 번호 | - |
| TIMESTAMP | DATETIME | 측정 시각 | - |
| BEAM_CURRENT | DECIMAL | 빔 전류 (μA) | 0.1-5000 |
| BEAM_ENERGY | DECIMAL | 빔 에너지 (keV) | 0.2-3000 |
| DOSE_RATE | DECIMAL | 도즈율 (ions/cm²/s) | - |
| TOTAL_DOSE | DECIMAL | 총 도즈량 (ions/cm²) | 1e11-1e16 |
| IMPLANT_ANGLE | DECIMAL | 주입 각도 (도) | 0-45 |
| WAFER_ROTATION | DECIMAL | 웨이퍼 회전 속도 (rpm) | 0-1200 |
| SOURCE_PRESSURE | DECIMAL | 소스 압력 (Torr) | 1e-6-1e-4 |
| ANALYZER_PRESSURE | DECIMAL | 분석기 압력 (Torr) | 1e-7-1e-5 |
| END_STATION_PRESSURE | DECIMAL | 엔드스테이션 압력 (Torr) | 1e-7-1e-6 |
| BEAM_UNIFORMITY | DECIMAL | 빔 균일도 (%) | >98 |
| FARADAY_CUP_CURRENT | DECIMAL | 패러데이 컵 전류 (μA) | - |

### 9. 테이블명: SEMI_CMP_SENSORS -- 화학기계연마 공정 센서

| **칼럼명** | **데이터타입** | **설명** | **정상범위** |
|------------|----------------|----------|--------------|
| PNO | VARCHAR(PK) | 측정 고유번호 | - |
| EQUIPMENT_ID | VARCHAR | CMP 장비 ID | - |
| LOT_NO | VARCHAR | 처리 LOT 번호 | - |
| TIMESTAMP | DATETIME | 측정 시각 | - |
| HEAD_PRESSURE | DECIMAL | 헤드 압력 (psi) | 2-8 |
| RETAINER_PRESSURE | DECIMAL | 리테이너 압력 (psi) | 2-6 |
| PLATEN_ROTATION | DECIMAL | 플래튼 회전속도 (rpm) | 20-150 |
| HEAD_ROTATION | DECIMAL | 헤드 회전속도 (rpm) | 20-150 |
| SLURRY_FLOW_RATE | DECIMAL | 슬러리 유량 (ml/min) | 100-300 |
| SLURRY_TEMP | DECIMAL | 슬러리 온도 (°C) | 20-25 |
| PAD_TEMP | DECIMAL | 패드 온도 (°C) | 30-50 |
| REMOVAL_RATE | DECIMAL | 제거율 (Å/min) | - |
| MOTOR_CURRENT | DECIMAL | 모터 전류 (A) | - |
| CONDITIONER_PRESSURE | DECIMAL | 컨디셔너 압력 (lbs) | 5-9 |
| ENDPOINT_SIGNAL | DECIMAL | 종점 검출 신호 (a.u.) | - |

### 10. 테이블명: SEMI_SENSOR_ALERT_CONFIG -- 센서 알람 설정

| **칼럼명** | **데이터타입** | **설명** |
|------------|----------------|----------|
| CONFIG_ID | VARCHAR(PK) | 설정 고유 ID |
| SENSOR_ID | VARCHAR | 센서 ID |
| PARAM_NAME | VARCHAR | 파라미터명 |
| WARNING_UPPER | DECIMAL | 경고 상한값 |
| WARNING_LOWER | DECIMAL | 경고 하한값 |
| ALARM_UPPER | DECIMAL | 알람 상한값 |
| ALARM_LOWER | DECIMAL | 알람 하한값 |
| INTERLOCK_UPPER | DECIMAL | 인터락 상한값 (장비 정지) |
| INTERLOCK_LOWER | DECIMAL | 인터락 하한값 (장비 정지) |
| MOVING_AVG_WINDOW | INT | 이동평균 윈도우 크기 (초) |
| ALERT_TYPE | VARCHAR | 알람 타입 (INSTANT/AVERAGE/TREND) |
| ENABLED | BOOLEAN | 알람 활성화 여부 |

## 이상탐지 활용 시나리오

### 1. 실시간 파라미터 모니터링
- `SEMI_PARAM_MEASURE` + `SEMI_PROCESS_SPEC`을 조인하여 USL/LSL 벗어나는 측정값 실시간 감지
- `SEMI_EQUIPMENT_SENSOR`의 시계열 데이터로 장비 드리프트 패턴 학습

### 2. 수율 기반 이상탐지
- `SEMI_LOT_MANAGE`의 FINAL_YIELD 급락 LOT 식별
- `SEMI_WAFER_TEST`의 YIELD_RATE 분포 이상치 검출

### 3. 공정 간 상관관계 분석
- `SEMI_PROCESS_HISTORY`로 특정 장비/레시피의 불량 상관성 분석
- 센서 데이터 간 상관관계로 복합 이상 패턴 검출

### 4. 예측적 유지보수
- 센서값 트렌드로 장비 고장 예측
- CMP의 MOTOR_CURRENT 증가 → 패드 마모 예측

## 센서별 주요 이상탐지 포인트

### 포토 공정
- **FOCUS_POSITION** 드리프트 → CD 불량
- **STAGE_TEMP** 변동 → 오버레이 에러
- **ALIGNMENT_ERROR** 증가 → 수율 하락

### 에칭 공정
- **RF_POWER** 불안정 → 에칭률 변동
- **CHAMBER_PRESSURE** 이상 → 균일도 저하
- **ENDPOINT_SIGNAL** 지연 → 과도 에칭

### CVD 공정
- **SUSCEPTOR_TEMP** 편차 → 박막 두께 불균일
- **PRECURSOR_FLOW** 변동 → 조성 변화
- **DEPOSITION_RATE** 감소 → 장비 클리닝 필요

### 이온주입 공정
- **BEAM_CURRENT** 불안정 → 도즈량 편차
- **BEAM_UNIFORMITY** 저하 → 웨이퍼 내 불균일
- **SOURCE_PRESSURE** 상승 → 소스 교체 시기

### CMP 공정
- **REMOVAL_RATE** 감소 → 패드 컨디셔닝 필요
- **MOTOR_CURRENT** 증가 → 패드 수명 임박
- **SLURRY_FLOW** 불규칙 → 공급 시스템 점검