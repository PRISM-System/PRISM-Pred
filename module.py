import pandas as pd, json, sys

def explain(feature_df: pd.DataFrame, target_col: str, preds):
    corr_s = feature_df.corr().get(target_col)
    important = []
    if corr_s is not None:
        important = [c for c in corr_s.abs().sort_values(ascending=False).index if c != target_col][:3]
    return {"importantFeatures": important, "method": "corr-proxy"}

if __name__ == "__main__":
    # argv[1] = csv_path, argv[2] = target_col
    csv, target = sys.argv[1], sys.argv[2]
    df = pd.read_csv(csv)
    out = explain(df.iloc[:,4:], target, [])
    with open("outputs/explain.json","w") as f:
        json.dump(out,f)


def risk(feature_df: pd.DataFrame, target_col: str, preds):
    series = pd.to_numeric(feature_df[target_col], errors="coerce").dropna()
    if series.size < 5:
        return {"riskLevel":"unknown","exceedsThreshold":False}
    z = (preds[-1] - series.mean())/(series.std() or 1)
    if abs(z)>2: level="high"
    elif abs(z)>1: level="medium"
    else: level="low"
    return {"riskLevel":level,"exceedsThreshold":level in ("medium","high")}

if __name__=="__main__":
    csv, target = sys.argv[1], sys.argv[2]
    df = pd.read_csv(csv)
    # preds는 생략: 데모라 랜덤
    preds = [float(series.mean()) for series in range(3)]
    out = assess(df.iloc[:,4:], target, preds)
    with open("outputs/risk.json","w") as f:
        json.dump(out,f)
