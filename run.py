# run.py
# Usage:
#   (CSV)  python run.py --csv_path prism_prediction/Industrial_DB_sample/SEMI_CMP_SENSORS.csv \
#            --feature_start_col 5 --target_col_name MOTOR_CURRENT \
#            --seq_len 48 --label_len 24 --pred_len 11 --epochs 1 --device cpu --models DLinear --auto_eval_idx
#   (MOCK) python run.py --seq_len 48 --label_len 24 --pred_len 12 --enc_in 4 --c_out 1 --epochs 1 --device cpu
import argparse, json, os, random, math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from importlib import import_module

_env = os.getenv("MAX_LENGTH", "").strip()
MAX_LENGTH = int(_env) if _env.isdigit() and int(_env) > 0 else None

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
# Configs
# -----------------------------
@dataclass
class Configs:
    task_name: str = 'long_term_forecast'
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 24
    enc_in: int = 1
    dec_in: int = 1
    c_out: int = 1
    d_model: int = 64
    n_heads: int = 4
    e_layers: int = 2
    d_layers: int = 1
    d_ff: int = 256
    moving_avg: int = 25
    factor: int = 3
    dropout: float = 0.1
    embed: str = 'timeF'
    freq: str = 'h'
    activation: str = 'gelu'
    distil: bool = True
    top_k: int = 3
    num_kernels: int = 6
    d_conv: int = 4
    expand: int = 2
    num_class: int = 1
    seg_len: int = 4

# -----------------------------
# CSV Dataset
# -----------------------------
class CSVForecastDataset(Dataset):
    """
    - CSV를 읽고, feature_start_col(1-based)부터 끝까지 피처로 사용
    - target_col_name(피처 집합 내부)을 타깃 채널로 사용 (평가/반환만 이 채널 사용)
    - (seq_len, label_len, pred_len)에 맞춰 슬라이딩 윈도우 생성
    - x_mark_*는 데모 단계라 0으로 채움
    - 학습 타깃 y는 "전체 채널" (C=enc_in). 모델 출력도 전체 채널로 맞춘 뒤,
      최종 응답 시 target 채널(eval_idx)만 선택해서 반환.
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

        start_idx0 = max(0, feature_start_col - 1)
        if df.shape[1] <= start_idx0:
            raise ValueError(f"feature_start_col={feature_start_col} out of range for csv with {df.shape[1]} columns.")

        self.feature_df = df.iloc[:, start_idx0:].copy()
        self.feature_df = self.feature_df.apply(pd.to_numeric, errors="coerce")
        # 간단 처리: 결측 있는 행 drop (데모 목적)
        self.feature_df = self.feature_df.dropna(axis=0, how="any")
        self.feature_names = list(self.feature_df.columns)
        self.enc_in = self.feature_df.shape[1]

        if target_col_name is None:
            self.target_col = self.feature_names[-1]
        else:
            if target_col_name not in self.feature_df.columns:
                raise ValueError(f"target_col_name '{target_col_name}' not in feature columns: {self.feature_names}")
            self.target_col = target_col_name
        self.target_idx = int(self.feature_df.columns.get_loc(self.target_col))

        self.X = self.feature_df.to_numpy(dtype=np.float32)  # [T, C]
        # fallback용 target series도 보관 (외삽 등)
        self.y_target_series = self.X[:, self.target_idx]    # [T]

        T = self.X.shape[0]
        self._windows: List[Tuple[int,int,int]] = []
        max_start = max(0, T - (self.seq_len + self.label_len + self.pred_len))
        for s in range(0, max_start + 1):
            enc_start = s
            dec_start = s + self.seq_len - self.label_len
            y_start   = s + self.seq_len
            self._windows.append((enc_start, dec_start, y_start))
        if len(self._windows) == 0 and T >= self.seq_len + 1:
            self._windows.append((0, max(0, self.seq_len - self.label_len), self.seq_len - 1))

    def __len__(self):
        return len(self._windows)

    def __getitem__(self, idx):
        enc_start, dec_start, y_start = self._windows[idx]
        x_enc = self._safe_slice(self.X, enc_start, enc_start + self.seq_len)  # [seq_len, C]
        x_dec = self._safe_slice(self.X, dec_start, dec_start + (self.label_len + self.pred_len))  # [label+pred, C]
        x_mark_enc = np.zeros((x_enc.shape[0],  self.mark_dim), dtype=np.float32)
        x_mark_dec = np.zeros((x_dec.shape[0],  self.mark_dim), dtype=np.float32)
        # 타깃은 전체 채널 -> [pred_len, C]
        y = self._safe_slice(self.X, y_start, y_start + self.pred_len)

        return (
            torch.from_numpy(x_enc.copy()),
            torch.from_numpy(x_mark_enc),
            torch.from_numpy(x_dec.copy()),
            torch.from_numpy(x_mark_dec),
            torch.from_numpy(y.copy()),
        )

    def _safe_slice(self, arr: np.ndarray, s: int, e: int):
        length = e - s
        if s >= arr.shape[0]:
            last = arr[-1]
            return np.repeat(last[None, ...], length, axis=0)
        if e <= arr.shape[0]:
            return arr[s:e]
        head = arr[s:]
        last = arr[-1]
        tail = np.repeat(last[None, ...], e - arr.shape[0], axis=0)
        return np.concatenate([head, tail], axis=0)

# -----------------------------
# Mock Dataset (fallback)
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
        x_enc = torch.zeros(self.seq_len, self.enc_in, dtype=torch.float32)
        x_mark_enc = torch.zeros(self.seq_len, self.mark_dim, dtype=torch.float32)
        x_dec = torch.zeros(self.label_len + self.pred_len, self.enc_in, dtype=torch.float32)
        x_mark_dec = torch.zeros(self.label_len + self.pred_len, self.mark_dim, dtype=torch.float32)
        y = torch.zeros(self.pred_len, self.c_out, dtype=torch.float32)
        return x_enc, x_mark_enc, x_dec, x_mark_dec, y

# -----------------------------
# Model import helpers
# -----------------------------
def import_model_module(name: str):
    for modname in (f"prism_prediction.models.{name}", f"models.{name}", name):
        try:
            return import_module(modname)
        except Exception:
            continue
    raise ImportError(f"Cannot import model module for '{name}'. Tried prism_prediction.models.{name}, models.{name}, and {name}")

def build_model(name: str, cfg: Configs, device: torch.device):
    mod = import_model_module(name)
    model = mod.Model(cfg)
    return model.to(device)

def forward_for_all(model: nn.Module, x_enc, x_mark_enc, x_dec, x_mark_dec):
    
    return model(x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, pred_len, C] 

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
    eval_idx: int = 0,
    save_dir: Optional[str] = None,
):
    model = build_model(name, cfg, device)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    best_ckpt_path = None

    best_focus_rmse = float("inf")
    last_rmse_all = float("inf")

    for ep in range(1, epochs+1):
        # ---- train ----
        model.train()
        total_loss, n = 0.0, 0
        for x_enc, x_mark_enc, x_dec, x_mark_dec, y in train_loader:
            x_enc, x_mark_enc = x_enc.to(device), x_mark_enc.to(device)
            x_dec, x_mark_dec = x_dec.to(device), x_mark_dec.to(device)
            y = y.to(device)                               # [B, H, C]
            optim.zero_grad()
            yhat = forward_for_all(model, x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, H, C] or [H, C]
            if yhat.dim() == 2:  # [H,C]
                yhat = yhat.unsqueeze(1)
            loss = criterion(yhat, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            total_loss += loss.item() * x_enc.size(0)
            n += x_enc.size(0)
        train_mse = total_loss / max(1, n)

        # ---- val ----
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
                Y.append(y.cpu().numpy()); YH.append(yhat.cpu().numpy())

        if len(Y) > 0:
            Y = np.concatenate(Y, axis=0)    # [N, H, C]
            YH = np.concatenate(YH, axis=0)  # [N, H, C]
            rmse_all = float(np.sqrt(np.mean((Y - YH) ** 2)))
            C = Y.shape[-1]
            safe_idx = max(0, min(eval_idx, C - 1))
            rmse_focus = float(np.sqrt(np.mean((Y[..., safe_idx] - YH[..., safe_idx]) ** 2)))
        else:
            rmse_all, rmse_focus = float("nan"), float("nan")

        print(f"[{name}] epoch {ep}/{epochs}  train_mse={train_mse:.6f}  "
              f"val_rmse_all={rmse_all:.6f}  val_rmse_focus(ch={eval_idx})={rmse_focus:.6f}")

        last_rmse_all = rmse_all
        if rmse_focus < best_focus_rmse:
            best_focus_rmse = rmse_focus
            # save best model
            if save_dir is not None:
                ensure_dir(save_dir)
                ckpt_path = os.path.join(save_dir, "ckpt.pt")
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "config": asdict(cfg),
                        "model_name": name,
                        "eval_idx": int(eval_idx),
                    },
                    ckpt_path,
                )
                best_ckpt_path = ckpt_path
         

    metrics = {"val_rmse_all": last_rmse_all, "val_rmse_focus": best_focus_rmse}
    return best_focus_rmse, metrics, model, best_ckpt_path

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
    ap.add_argument("--models", type=str, default="Autoformer,DLinear,TimesNet,LightTS,SegRNN")
    ap.add_argument("--eval_channel_idx", type=int, default=None)
    ap.add_argument("--seg_len", type=int, default=12)

    # CSV 모드
    ap.add_argument("--csv_path", type=str, default=None)
    ap.add_argument("--feature_start_col", type=int, default=5)
    ap.add_argument("--target_col_name", type=str, default=None)
    ap.add_argument("--auto_eval_idx", action="store_true")

    args = ap.parse_args()
    if args.modality != "timeseries_only":
        print(json.dumps({"status":"TBU", "message": f"modality {args.modality} not implemented"}))
        return

    set_seed(42)
    if args.device.lower().startswith("cuda") and torch.cuda.is_available():
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    csv_mode = args.csv_path is not None
    ds = None
    train_loader = val_loader = None

    if csv_mode:
        ds = CSVForecastDataset(
            csv_path=args.csv_path,
            seq_len=args.seq_len,
            label_len=args.label_len,
            pred_len=args.pred_len,
            feature_start_col=args.feature_start_col,
            target_col_name=args.target_col_name,
            mark_dim=4,
        )
        enc_in = ds.enc_in
        c_out = ds.enc_in                     
        eval_idx = ds.target_idx if (args.auto_eval_idx or args.eval_channel_idx is None) else args.eval_channel_idx
        eval_idx = max(0, min(eval_idx, c_out - 1))  # 안전 보정

        n = len(ds)
        g = torch.Generator().manual_seed(42)  

        if MAX_LENGTH is not None and n > MAX_LENGTH:
            idx = torch.randperm(n, generator=g)[:MAX_LENGTH]
            dataset = torch.utils.data.Subset(ds, idx.tolist())
        else:
            dataset = ds

        m = len(dataset)
        if m >= 2:
            # n_train ∈ [1, m-1] 
            n_train = max(1, min(m - 1, int(round(m * 0.8))))
            n_val = m - n_train
            train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=g)
        else:
            # 샘플이 1개뿐이면 동일 데이터셋 사용
            train_ds = dataset
            val_ds = dataset
            

        drop_last_train = (len(train_ds) > args.batch_size)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=drop_last_train)
        val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, drop_last=False)

        cfg = Configs(
            task_name='long_term_forecast',
            seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
            enc_in=enc_in, dec_in=enc_in, c_out=c_out,
            d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers, seg_len=args.seg_len
        )
    else:
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

    # 후보 학습 & 선택 + 베스트 모델 보관
    candidates = [m.strip() for m in args.models.split(",") if m.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    ckpt_paths: Dict[str, Optional[str]] = {}
    best_name, best_metric = None, float("inf")
    best_model_obj = None

    for name in candidates:
        try:
            focus_rmse, metrics, model_obj = train_and_validate(
                name=name, cfg=cfg,
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=args.epochs,
                eval_idx=eval_idx,
                save_dir=os.path.join("outputs", name),
            )
            results[name] = metrics
            if focus_rmse < best_metric:
                best_metric = focus_rmse
                best_name = name
                best_model_obj = model_obj
        except Exception as e:
            results[name] = {"error": str(e)}
            ckpt_paths[name] = None
            print(f"[WARN] {name} failed: {e}")

    # 예측 생성
    prediction: List[float] = []
    if csv_mode:
        try:
            if len(ds) == 0:
                raise RuntimeError("dataset has no windows")
            # ds의 마지막 윈도우로 예측
            x_enc, x_mark_enc, x_dec, x_mark_dec, _ = ds[len(ds)-1]
            x_enc = x_enc.unsqueeze(0).to(device)
            x_mark_enc = x_mark_enc.unsqueeze(0).to(device)
            x_dec = x_dec.unsqueeze(0).to(device)
            x_mark_dec = x_mark_dec.unsqueeze(0).to(device)

            if best_model_obj is not None:
                best_model_obj.eval()
                with torch.no_grad():
                    yhat = forward_for_all(best_model_obj, x_enc, x_mark_enc, x_dec, x_mark_dec)  # [B, H, C] 기대
                    if yhat.dim() == 2:  # [H, C]
                        yhat = yhat.unsqueeze(0)
                    C = yhat.size(-1)
                    ch = max(0, min(int(eval_idx), C - 1))  # MOTOR_CURRENT 채널
                    pred = yhat[0, :, ch].detach().cpu().numpy().tolist()
                    prediction = [float(round(p, 6)) for p in pred]
            else:
                # 베스트가 없으면 간단 외삽 fallback (타깃 채널 기준)
                series = ds.y_target_series
                delta = float(series[-1] - series[-2]) if len(series) >= 2 else 0.0
                start = float(series[-1])
                prediction = [round(start + (i+1)*delta, 6) for i in range(args.pred_len)]
        except Exception as e:
            results["inference_error"] = {"error": str(e)}
            # 최후 fallback: zeros
            prediction = [0.0 for _ in range(args.pred_len)]
    else:
        # mock 모드: zeros
        prediction = [0.0 for _ in range(args.pred_len)]


    if best_name is None:
        best_name = "Autoformer"
        
    best_all = results.get(best_name, {}).get("val_rmse_all") if best_name in results else None
    best_weight_path = ckpt_paths.get(best_name) if best_name is not None else ckpt_paths.get("Autoformer")

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
        "best_val_rmse_focus": float(best_metric) if best_metric != float("inf") else None,
        "best_val_rmse_all": float(best_all) if best_all is not None else None,
        "results": results,
        "prediction": prediction,             # MOTOR_CURRENT 채널만 반환
        "target_col_name": args.target_col_name,
        "feature_start_col": args.feature_start_col,
        "best_weight_path": best_weight_path,
    }
    ensure_dir("outputs")
    with open(os.path.join("outputs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
