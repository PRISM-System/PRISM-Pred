#!/usr/bin/env bash
set -euo pipefail

BASE="http://127.0.0.1:8001"
CT='Content-Type: application/json'

echo "1) create task"
CREATE=$(curl -s -X POST "$BASE/api/v1/prediction/tasks" -H "$CT" \
  -d '{"taskObjective":"30분 내 과열 위험 탐지","timeRange":"2025-07-17T13:00:00~13:30:00","userRole":"engineer"}')
echo "$CREATE"

TASK=$(echo "$CREATE" | python3 -c "import sys,json; print(json.load(sys.stdin)['data']['taskId'])")
echo "TASK=$TASK"

echo "2) fetch-data"
curl -s -X POST "$BASE/api/v1/prediction/tasks/$TASK/fetch-data" -H "$CT" \
  -d '{"dataSources":["sensor1","sensor5"],"preprocessing":{"missing":"interpolation","scaling":"z-score"}}' \
  | python3 -m json.tool

echo "3) run"
curl -s -X POST "$BASE/api/v1/prediction/tasks/$TASK/run" -H "$CT" \
  -d '{"modelPreference":"auto"}' | python3 -m json.tool

echo "4) explain"
curl -s -X POST "$BASE/api/v1/prediction/tasks/$TASK/explain" -H "$CT" \
  -d '{"method":"SHAP"}' | python3 -m json.tool

echo "5) risk-assess"
curl -s -X POST "$BASE/api/v1/prediction/tasks/$TASK/risk-assess" -H "$CT" \
  -d '{}' | python3 -m json.tool

echo "6) result-summary"
curl -s "$BASE/api/v1/prediction/tasks/$TASK/result-summary" | python3 -m json.tool

echo "7) NL → 11 preds"
curl -s -X POST "$BASE/api/v1/prediction/nl" -H "$CT" \
  -d '{"query":"화학기계연마 공정 센서의 MOTOR_CURRENT 컬럼의 2024-02-04 09:00:00~09:10:00 의 값들을 예측해줘"}' \
  | python3 -m json.tool
