#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
각 시나리오 JSON 파일(automotive, battery, chemical, semiconductor, steel)을 읽어
파일명(SCENARIO_XX.json)을 key로 하고, value는 원본 JSON 전체 그대로 둔 상태로
prediction_scenarios.json 파일로 통합합니다.

정렬: SCENARIO 번호 순서
"""

import os
import re
import json
import argparse

DOMAINS_DEFAULT = ["automotive", "battery", "chemical", "semiconductor", "steel"]

def extract_scenario_num(filename: str) -> int:
    """SCENARIO_XX에서 숫자 부분을 추출"""
    m = re.search(r"SCENARIO_(\d+)", filename)
    return int(m.group(1)) if m else 999999

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True,
        help="Base dir for scenarios (e.g., /home/dmlab/AGI/prism_prediction/Industrial_DB_sample/dataset_v3/test-scenarios/scenarios)")
    parser.add_argument("--out", required=True,
        help="Output path for prediction_scenarios.json")
    parser.add_argument("--domains", nargs="*", default=DOMAINS_DEFAULT,
        help=f"Domains to include (default: {', '.join(DOMAINS_DEFAULT)})")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.base)
    out_path = os.path.abspath(args.out)
    domains = args.domains

    print(f"[INFO] Collecting scenario files from: {base_dir}")
    result = {}

    # --- 각 도메인 순회 ---
    for domain in domains:
        domain_dir = os.path.join(base_dir, domain)
        if not os.path.isdir(domain_dir):
            print(f"[WARN] Skip missing domain dir: {domain_dir}")
            continue

        for root, _, files in os.walk(domain_dir):
            for fn in sorted(files):
                if not fn.lower().endswith(".json"):
                    continue
                fpath = os.path.join(root, fn)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        obj = json.load(f)
                except Exception as e:
                    print(f"[ERROR] JSON load failed: {fpath} ({e})")
                    continue

                key_name = fn
                result[key_name] = obj

    # --- SCENARIO 번호 기준 정렬 ---
    sorted_items = sorted(result.items(), key=lambda x: extract_scenario_num(x[0]))
    ordered_result = {k: v for k, v in sorted_items}

    # --- 출력 저장 ---
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(ordered_result, f, ensure_ascii=False, indent=2)

    print(f"[OK] prediction_scenarios.json saved → {out_path}")
    print(f"Total scenarios: {len(ordered_result)}")

if __name__ == "__main__":
    main()
