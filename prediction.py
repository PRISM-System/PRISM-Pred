# run.py
# Usage:
#   python run.py --modality timeseries_only --seq_len 48 --label_len 24 --pred_len 12 \
#                 --enc_in 4 --c_out 1 --epochs 1 --device cpu
import argparse, json, math, os, sys, random
from dataclasses import dataclass
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

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
# Mock dataset (zeros) -- 이후 대체될 예정(데이터 셋에 맞게)
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
    eval_idx: int = 0,           # ← 추가
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
    ap.add_argument("--models", type=str, default="Autoformer,DLinear,TimesNet,LightTS,SegRNN") #Informer,Mamba TBU
    ap.add_argument("--eval_channel_idx", type=int, required=True, help="평가/베스트 선정에 사용할 타깃 채널 인덱스 (0-based)")
    ap.add_argument("--seg_len", type=int, default=12, help="SegRNN이 사용하는 segment 길이")
    args = ap.parse_args()

    if args.modality != "timeseries_only":
        print(json.dumps({"status":"TBU", "message": f"modality {args.modality} not implemented"}))
        return

    set_seed(42)
    device = torch.device(args.device if torch.cuda.is_available() or "cpu" not in args.device else "cpu")

    # 공통 설정
    cfg = Configs(
        task_name='long_term_forecast',
        seq_len=args.seq_len, label_len=args.label_len, pred_len=args.pred_len,
        enc_in=args.enc_in, dec_in=args.enc_in, c_out=args.c_out,
        d_model=args.d_model, n_heads=args.n_heads, e_layers=args.e_layers, d_layers=args.d_layers, seg_len=args.seg_len
    )

    # Mock 데이터 로더
    train_ds = ZeroTSDataset(args.train_size, args.seq_len, args.label_len, args.pred_len, args.enc_in, args.c_out)
    val_ds   = ZeroTSDataset(args.val_size,   args.seq_len, args.label_len, args.pred_len, args.enc_in, args.c_out)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False)

    # 후보 모델 학습 및 선택
    candidates = [m.strip() for m in args.models.split(",") if m.strip()]
    results: Dict[str, Dict[str, Any]] = {}
    best_name, best_metric = None, float("inf")
    results = {}

    for name in candidates:
        try:
            focus_rmse, metrics, _ = train_and_validate(
                name=name, cfg=cfg,
                train_loader=train_loader, val_loader=val_loader,
                device=device, epochs=args.epochs,
                eval_idx=args.eval_channel_idx,   # ← 추가
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
        "modality": args.modality,
        "seq_len": args.seq_len,
        "label_len": args.label_len,
        "pred_len": args.pred_len,
        "enc_in": args.enc_in,
        "c_out": args.c_out,

        # 평가/선정 기준 정보
        "eval_channel_idx": args.eval_channel_idx,
        "selection_metric": "val_rmse_focus",

        # 베스트 모델 및 점수(타깃 채널 기준)
        "best_model": best_name,
        "best_val_rmse_focus": float(best_metric),

        # (옵션) 베스트 모델의 전체 채널 RMSE도 참고용으로 저장
        "best_val_rmse_all": float(best_all) if best_all is not None else None,

        # 각 모델별 상세 결과: {"val_rmse_all": ..., "val_rmse_focus": ...}
        "results": results,
    }
    ensure_dir("outputs")
    with open(os.path.join("outputs", "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
