# run.py
# Usage:
#   (CSV 사용) python run.py --csv_path prism_prediction/Industrial_DB_sample/SEMI_CMP_SENSORS.csv \
#            --feature_start_col 5 --target_col_name MOTOR_CURRENT \
#            --seq_len 48 --label_len 24 --pred_len 11 --epochs 1 --device cpu --auto_eval_idx
#   (MOCK 사용) python run.py --seq_len 48 --label_len 24 --pred_len 12 --enc_in 4 --c_out 1 --epochs 1 --device cpu
import argparse, json, os, random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 추가: CSV 로딩용
import pandas as pd

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

# -----------------------------
# Configs object expected by models
# (Autoformer/Informer/TimesNet/DLinear/Mamba 공통 속성 포함)
# -----------------------------
@dataclass
class Configs:
    # task & lengths
    task_name: str = 'long_term_forecast'
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    # dims
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    # others (레포 공통 하이퍼)
    moving_avg: int = 25          # Autoformer/DLinear
    factor: int = 3               # Informer
    dropout: float = 0.1
    embed: str = 'timeF'
    freq: str = 'h'
    activation: str = 'gelu'
    distil: bool = True
    top_k: int = 3                # TimesNet
    num_kernels: int = 6          # TimesNet
    d_conv: int = 4               # Mamba
    expand: int = 2               # Mamba
    num_class: int = 1
    seg_len: int = 24 #SegRNN

# -----------------------------
# CSV 기반 시계열 데이터셋 (신규)
# -----------------------------
class CSVForecastDataset(Dataset):
    """
    - CSV를 읽고, 5번째(1-based) 컬럼부터 끝까지를 피처로 사용
    - target_col_name 으로 지정된 컬럼(피처 집합 내부)을 타깃으로 사용
    - (seq_len, label_len, pred_len) 구성에 맞춰 슬라이딩 윈도우 샘플 생성
    - x_mark_*는 데모 단계라 0으로 채움
    """
    def __init__(
        self,
        csv_path: str,
        seq_len: int,
        label_len: int,
        pred_len: int,
        feature_start_col: int = 5,   # 1-based
        target_col_name: Optional[str] = None,
        mark_dim: int = 4,
    ):
        super().__init__()
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        self.df_raw = df
        self.seq_len = seq_len
        self.label_len = label_len
        self.pred_len = pred_len
        self.mark_dim = mark_dim

        # 1-based -> 0-based
        start_idx0 = max(0, feature_start_col - 1)
        if df.shape[1] <= start_idx0:
            raise ValueError(f"feature_start_col={feature_start_col} is out of range for csv with {df.shape[1]} columns.")

        # 피처 프레임
        self.feature_df = df.iloc[:, start_idx0:].copy()
        self.feature_df = self.feature_df.apply(pd.to_numeric, errors="coerce")
        self.feature_df = self.feature_df.dropna(axis=0, how="any")  # 간단히 결측치 drop (데모 목적)
        self.feature_names = list(self.feature_df.columns)
        self.enc_in = self.feature_df.shape[1]

        # 타깃 인덱스 결정
        if target_col_name is None:
            # 명시 안하면 마지막 컬럼을 타깃
            self.target_col = self.feature_names[-1]
        else:
            if target_col_name not in self.feature_df.columns:
                raise ValueError(f"target_col_name '{target_col_name}' not in feature columns: {self.feature_names}")
            self.target_col = target_col_name
        self.target_idx = int(self.feature_df.columns.get_loc(self.target_col))

        # numpy 변환
        self.X = self.feature_df.to_numpy(dtype=np.float32)  # [T, enc_in]
        self.y = self.X[:, self.target_idx]                  # [T]

        # 슬라이딩 윈도우 인덱스 준비
        T = self.X.shape[0]
        self._windows: List[Tuple[int,int,int]] = []  # (enc_start, dec_start, y_start)
        total_need = self.seq_len + self.pred_len
        min_need = self.seq_len + self.label_len + self.pred_len  # x_dec 길이 확보
        if T < min_need:
            # 데이터가 부족하면 최소 1개 샘플이라도 만들어 보되, 부족분은 마지막 값 반복 (데모)
            # 단, 여기서는 간단히 예외 대신 경고성 축소 처리로 한 샘플 생성
            # enc: [0:seq_len], dec: [seq_len-label_len : seq_len+pred_len], y: [seq_len : seq_len+pred_len]
            pass

        max_start = max(0, T - (self.seq_len + self.label_len + self.pred_len))
        for s in range(0, max_start + 1):
            enc_start = s
            dec_start = s + self.seq_len - self.label_len
            y_start   = s + self.seq_len
            self._windows.append((enc_start, dec_start, y_start))

        # 하나도 못만들었으면 최소 한개 강제 생성(데모)
        if len(self._windows) == 0 and T >= self.seq_len + 1:
            self._windows.append((0, max(0, self.seq_len - self.label_len), self.seq_len - 1))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        enc_start, dec_start, y_start = self._windows[idx]

        # encoder input
        x_enc = self._safe_slice(self.X, enc_start, enc_start + self.seq_len)  # [seq_len, C]
        # decoder input (label_len + pred_len)
        x_dec = self._safe_slice(self.X, dec_start, dec_start + (self.label_len + self.pred_len))  # [label_len+pred_len, C]
        # time marks (zeros)
        x_mark_enc = np.zeros((x_enc.shape[0],  self.mark_dim), dtype=np.float32)
        x_mark_dec = np.zeros((x_dec.shape[0],  self.mark_dim), dtype=np.float32)
        # target
        y = self._safe_slice(self.y[:, None], y_start, y_start + self.pred_len)  # [pred_len, 1]

        # torch tensors
        return (
            torch.from_numpy(x_enc.copy()),
            torch.from_numpy(x_mark_enc),
            torch.from_numpy(x_dec.copy()),
            torch.from_numpy(x_mark_dec),
            torch.from_numpy(y.copy()),
        )

    def _safe_slice(self, arr: np.ndarray, s: int, e: int):
        """
        범위를 벗어나면 마지막 값을 반복(데모 목적).
        """
        length = e - s
        if s >= arr.shape[0]:
            last = arr[-1]
            return np.repeat(last[None, ...], length, axis=0)
        if e <= arr.shape[0]:
            return arr[s:e]
        # e가 넘친 경우
        head = arr[s:]
        last = arr[-1]
        tail = np.repeat(last[None, ...], e - arr.shape[0], axis=0)
        return np.concatenate([head, tail], axis=0)

# -----------------------------
# Mock dataset (zeros) -- CSV 미지정 시 사용
# -----------------------------
class ZeroTSDataset(Dataset):
    def __init__(self, num_samples: int, seq_len: int, label_len: int, pred_len: int,
                 enc_in: int, c_out: int, mark_dim:int=4):
        super().__init__()
        self.N = num_samples
        self.seq_len, self.label_len, self.pred_len = seq_len, label_len, pred_len
        self.enc_in, self.c_out, self.mark_dim = enc_in, c_out, mark_dim

    def __len__(self): return self.N

    def __getitem__(self, idx):
        # encoder input
        x_enc = torch.zeros(self.seq_len, self.enc_in, dtype=torch.float32)
        x_mark_enc = torch.zeros(self.seq_len, self.mark_dim, dtype=torch.float32)
        # decoder input: label_len + pred_len
        x_dec = torch.zeros(self.label_len + self.pred_len, self.enc_in, dtype=torch.float32)
        x_mark_dec = torch.zeros(self.label_len + self.pred_len, self.mark_dim, dtype=torch.float32)
        # target
        y = torch.zeros(self.pred_len, self.c_out, dtype=torch.float32)
        return x_enc, x_mark_enc, x_dec, x_mark_dec, y

# -----------------------------
# Import helpers
# -----------------------------
def import_model_module(name: str):
    """
    우선 'models.<Name>'로 import 시도, 실패하면 '<Name>'로 fallback.
    (업로드 경로/패키지 구조 상관없이 동작)
    """
    try:
        return __import__(f"models.{name}", fromlist=['Model'])
    except Exception:
        return __import__(name, fromlist=['Model'])

def build_model(name: str, cfg: Configs, device: torch.device):
    mod = import_model_module(name)
    model = mod.Model(cfg)
    return model.to(device)

def forward_for_all(model: nn.Module, x_enc, x_mark_enc, x_dec, x_mark_dec):
    # 업로드된 파일들 모두: forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None)
    return model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, c_out] 기대

# -----------------------------
# Trainer
# -----------------------------
def train_and_validate(
    name: str,
    cfg,
    train_loader,
    val_loader,
    device,
    epochs: int = 1,
    lr: float = 1e-3,
    eval_idx: int = 0,           # ← 타깃 채널 인덱스
):
    model = build_model(name, cfg, device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_focus_rmse = float("inf")
    last_rmse_all = float("inf")

    for ep in range(1, epochs+1):
        # ----- train (전체 채널로 손실 계산 유지) -----
        model.train()
        total_loss, n = 0.0, 0
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in train_loader:
            x_enc, x_mark_enc = x_enc.to(device), x_mark_enc.to(device)
            x_dec, x_mark_dec = x_dec.to(device), x_mark_dec.to(device)
            y = y.to(device)
            optim.zero_grad()
            yhat = forward_for_all(model, x_enc, x_mark_enc, x_dec, x_mark_dec)
            if yhat.dim() == 2:
                yhat = yhat.unsqueeze(1)      # [B, pred_len, C]
            loss = criterion(yhat, y)         # 전체 채널 기준 학습 유지
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * x_enc.size(0)
            n += x_enc.size(0)
        train_mse = total_loss / max(1, n)

        # ----- validate (포커스 채널 기준으로 베스트 선정) -----
        model.eval()
        Y, YH = [], []
        with torch.no_grad():
            for x_enc, x_mark_enc, x_dec, x_mark_dec, y in val_loader:
                x_enc, x_mark_enc = x_enc.to(device), x_mark_enc.to(device)
                x_dec, x_mark_dec = x_dec.to(device), x_mark_dec.to(device)
                y = y.to(device)
                yhat = forward_for_all(model, x_enc, x_mark_enc, x_dec, x_mark_dec)
                if yhat.dim() == 2:
                    yhat = yhat.unsqueeze(1)
                Y.append(y.cpu().numpy())
                YH.append(yhat.cpu().numpy())

        if len(Y) > 0:
            Y = np.concatenate(Y, axis=0)       # [N, H, C]
            YH = np.concatenate(YH, axis=0)     # [N, H, C]
            rmse_all = float(np.sqrt(np.mean((Y - YH) ** 2)))
            rmse_focus = float(np.sqrt(np.mean((Y[..., eval_idx] - YH[..., eval_idx]) ** 2)))
        else:
            rmse_all, rmse_focus = float("nan"), float("nan")

        print(f"[{name}] epoch {ep}/{epochs}  train_mse={train_mse:.6f}  "
              f"val_rmse_all={rmse_all:.6f}  val_rmse_focus(ch={eval_idx})={rmse_focus:.6f}")

        last_rmse_all = rmse_all
        if rmse_focus < best_focus_rmse:
            best_focus_rmse = rmse_focus
            # (선택) 체크포인트 저장 로직이 있다면 여기서 저장
            # torch.save(...)

    # 베스트 기준: focus RMSE
    metrics = {"val_rmse_all": last_rmse_all, "val_rmse_focus": best_focus_rmse}
    return best_focus_rmse, metrics, model

# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modality", type=str, default="timeseries_only")
    ap.add_argument("--seq_len", type=int, default=48)
    ap.add_argument("--label_len", type=int, default=24)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--enc_in", type=int, default=4)
    ap.add_argument("--c_out", type=int, default=1)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--e_layers", type=int, default=2)
    ap.add_argument("--d_layers", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--train_size", type=int, default=256)
    ap.add_argument("--val_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--models", type=str, default="Autoformer,DLinear,TimesNet,LightTS") #Informer,Mamba TBU
    # 기존: required=True 였지만 CSV 모드에서는 자동결정 가능하게 완화
    ap.add_argument("--eval_channel_idx", type=int, default=None, help="타깃 채널 인덱스 (0-based). CSV 모드에서 --auto_eval_idx면 자동 설정")
    ap.add_argument("--seg_len", type=int, default=12, help="SegRNN이 사용하는 segment 길이")

    # === 신규 (CSV 사용 시) ===
    ap.add_argument("--csv_path", type=str, default=None, help="CSV 파일 경로가 주어지면 Mock 대신 CSV 데이터 사용")
    ap.add_argument("--feature_start_col", type=int, default=5, help="피처 시작 컬럼 (1-based). 기본=5")
    ap.add_argument("--target_col_name", type=str, default=None, help="타깃(예측) 컬럼명. 예: MOTOR_CURRENT")
    ap.add_argument("--auto_eval_idx", action="store_true", help="CSV 모드에서 타깃 컬럼의 인덱스를 자동으로 eval_channel_idx로 사용")

    args = ap.parse_args()

    if args.modality != "timeseries_only":
        print(json.dumps({"status":"TBU", "message": f"modality {args.modality} not implemented"}))
        return

    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")

    # -----------------------------
    # 데이터셋 준비
    # -----------------------------
    csv_mode = args.csv_path is not None
    if csv_mode:
        # CSV 기반 데이터셋
        ds = CSVForecastDataset(
            csv_path=args.csv_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            feature_start_col=args.feature_start_col,
            target_col_name=args.target_col_name,
            mark_dim=4,
        )
        # enc_in/c_out 자동 보정
        enc_in = ds.enc_in
        c_out = 1
        # eval 채널 자동결정 옵션
        eval_idx = ds.target_idx if args.auto_eval_idx or args.eval_channel_idx is None else args.eval_channel_idx

        # 간단 split (80/20)
        n = len(ds)
        n_train = max(1, int(n * 0.8))
        n_val = max(1, n - n_train)
        train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))

        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True if n_train > args.batch_size else False)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

        # 공통 설정
        cfg = Configs(
            task_name='long_term_forecast',
            seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
            enc_in=enc_in, dec_in=enc_in, c_out=c_out,
            d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers, seg_len=args.seg_len
        )
    else:
        # MOCK 데이터셋
        train_ds = ZeroTSDataset(args.train_size, args.seq_len, args.label_len, args.pred_len, args.enc_in, args.c_out)
        val_ds   = ZeroTSDataset(args.val_size,   args.seq_len, args.label_len, args.pred_len, args.enc_in, args.c_out)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

        eval_idx = args.eval_channel_idx if args.eval_channel_idx is not None else 0

        cfg = Configs(
            task_name='long_term_forecast',
            seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
            enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.c_out,
            d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers, seg_len=args.seg_len
        )

    # -----------------------------
    # 후보 모델 학습 및 선택
    # -----------------------------
    candidates = [m.strip() for m in args.models.split(",") if m.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    best_name, best_metric = None, float("inf")

    for name in candidates:
        try:
            focus_rmse, metrics, _ = train_and_validate(
                name=name, cfg=cfg,
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=args.epochs,
                eval_idx=eval_idx,
            )
            results[name] = metrics
            if focus_rmse < best_metric:
                best_metric = focus_rmse
                best_name = name
        except Exception as e:
            results[name] = {"error": str(e)}
            print(f"[WARN] {name} failed: {e}")

    best_all = results.get(best_name, {}).get("val_rmse_all")

    summary = {
        "mode": "csv" if csv_mode else "mock",
        "csv_path": args.csv_path if csv_mode else None,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "enc_in": cfg.enc_in,
        "c_out": cfg.c_out,
        "eval_channel_idx": int(eval_idx),
        "selection_metric": "val_rmse_focus",
        "best_model": best_name,
        "best_val_rmse_focus": float(best_metric),
        "best_val_rmse_all": float(best_all) if best_all is not None else None,
        "results": results,
    }
    ensure_dir("outputs")
    with open(os.path.join("outputs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
