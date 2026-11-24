"""Train a ResNet-18 on PCam; select the checkpoint by validation AUC; evaluate it once on test.

M2 baseline:
    python train.py --name m2_baseline
M3 variants are flags on the same script, e.g.:
    python train.py --name rq2_scratch_10 --scratch --label-frac 0.10
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from pcam.augment import AUGMENTATIONS, normalize, to_float
from pcam.data import fits_on_device, iterate_batches, label_budget_indices, load_split
from pcam.metrics import binary_metrics
from pcam.model import build_resnet18


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--scratch", action="store_true", help="random init instead of ImageNet")
    p.add_argument("--label-frac", type=float, default=1.0)
    p.add_argument("--aug", default="geo", choices=list(AUGMENTATIONS))
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-frac", type=float, default=0.03)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--subset-seed", type=int, default=0)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=False,
                   help="mixed precision (bf16 if supported, else fp16 + GradScaler). Use on CUDA/Colab; no speedup measured on MPS, so off by default")
    p.add_argument("--gpu-data", action=argparse.BooleanOptionalAction, default=None,
                   help="keep the image arrays in GPU memory (default: on if CUDA has room for train+valid)")
    p.add_argument("--limit", type=int, default=0, help="debug: use only N samples per split")
    p.add_argument("--out", default="runs")
    return p.parse_args()


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def amp_dtype(device):
    # is_bf16_supported() is also True on a T4, where bf16 is only emulated (slow); require Ampere+.
    if device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8:
        return torch.bfloat16  # Ampere+ (L4, A100): native bf16, no loss scaling needed
    return torch.float16  # T4 / MPS: fp16 tensor cores + GradScaler


@torch.no_grad()
def predict(model, x, y, device, batch_size, amp):
    model.eval()
    probs = []
    for xb, _ in iterate_batches(x, y, batch_size, shuffle=False):
        xb = normalize(to_float(xb, device))
        with torch.autocast(device.type, dtype=amp_dtype(device), enabled=amp):
            logits = model(xb).squeeze(1)
        probs.append(torch.sigmoid(logits.float()).cpu().numpy())
    return np.concatenate(probs)


def main():
    a = parse_args()
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    device = get_device()
    torch.backends.cudnn.benchmark = device.type == "cuda"  # fixed 96x96 input: let cuDNN pick fastest kernels
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    gpu_data = fits_on_device(device) if a.gpu_data is None else a.gpu_data
    data_dev = device if gpu_data else None
    xtr, ytr = load_split("train", limit=a.limit, device=data_dev)
    xva, yva = load_split("valid", limit=a.limit, device=data_dev)
    train_idx = label_budget_indices(ytr, a.label_frac, a.subset_seed)
    print(f"loaded data in {time.time() - t0:.0f}s ({'GPU-resident' if gpu_data else 'memmap'}) | "
          f"train {len(train_idx)} (frac {a.label_frac}, pos {ytr[train_idx].mean():.3f}) | "
          f"valid {len(yva)} | device {device}", flush=True)

    model = build_resnet18(pretrained=not a.scratch).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)  # NHWC kernels are faster with AMP
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    steps_per_epoch = math.ceil(len(train_idx) / a.batch_size)
    total = a.epochs * steps_per_epoch
    warm = max(1, int(a.warmup_frac * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / (total - warm))))
    loss_fn = nn.BCEWithLogitsLoss()
    dtype = amp_dtype(device)
    scaler = torch.amp.GradScaler(device.type, enabled=a.amp and dtype == torch.float16)
    augment = AUGMENTATIONS[a.aug]

    config = {**vars(a), "gpu_data": gpu_data, "device": str(device),
              "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
              "torch": str(torch.__version__), "n_train": int(len(train_idx)), "steps_per_epoch": steps_per_epoch}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    history, best_auc, t_train = [], -1.0, time.time()
    for epoch in range(1, a.epochs + 1):
        model.train()
        t_ep, run_loss, n_seen = time.time(), 0.0, 0
        for step, (xb, yb) in enumerate(iterate_batches(xtr, ytr, a.batch_size, True, rng, train_idx)):
            xb = normalize(augment(to_float(xb, device)))
            yb = yb.to(device)
            with torch.autocast(device.type, dtype=dtype, enabled=a.amp):
                logits = model(xb).squeeze(1)
            loss = loss_fn(logits.float(), yb)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * len(yb)
            n_seen += len(yb)
            if step % 200 == 0:
                print(f"  ep {epoch} step {step}/{steps_per_epoch} loss {run_loss / n_seen:.4f} "
                      f"lr {sched.get_last_lr()[0]:.2e} ({n_seen / (time.time() - t_ep):.0f} img/s)", flush=True)

        pva = predict(model, xva, yva, device, 512, a.amp)
        vm = binary_metrics(yva, pva)
        vloss = float(-np.mean(yva * np.log(pva + 1e-7) + (1 - yva) * np.log(1 - pva + 1e-7)))
        rec = {"epoch": epoch, "train_loss": run_loss / n_seen, "val_loss": vloss,
               **{f"val_{k}": v for k, v in vm.items()}, "epoch_sec": time.time() - t_ep}
        history.append(rec)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        improved = vm["auc"] > best_auc
        if improved:
            best_auc = vm["auc"]
            torch.save({"model": model.state_dict(), "epoch": epoch, "config": config}, out / "best.pt")
            np.savez_compressed(out / "valid_preds.npz", y=yva, p=pva)  # for choosing thresholds on validation
        print(f"epoch {epoch}: train_loss {rec['train_loss']:.4f} val_loss {vloss:.4f} "
              f"val_auc {vm['auc']:.4f} val_acc {vm['accuracy']:.4f} val_recall {vm['recall']:.4f} "
              f"[{rec['epoch_sec']:.0f}s]{' *best*' if improved else ''}", flush=True)

    # Final: test set touched once, with the checkpoint chosen on validation.
    train_sec = time.time() - t_train
    del xtr
    if device.type == "cuda":
        torch.cuda.empty_cache()
    xte, yte = load_split("test", limit=a.limit, device=data_dev)
    ckpt = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    pte = predict(model, xte, yte, device, 512, a.amp)
    tm = binary_metrics(yte, pte)
    result = {"best_epoch": ckpt["epoch"], "best_val_auc": best_auc, "train_minutes": train_sec / 60, "test": tm}
    (out / "test_metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(out / "test_preds.npz", y=yte, p=pte)
    print(f"TEST (best epoch {ckpt['epoch']}): " + " ".join(
        f"{k} {tm[k]:.4f}" for k in ("auc", "accuracy", "precision", "recall", "specificity", "f1")))


if __name__ == "__main__":
    main()
