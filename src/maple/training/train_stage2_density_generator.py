from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from maple.core.stage2.density_generator import (
    Stage2DensityGenerator,
    Stage2NPZDataset,
    density_generation_loss,
)

CONDITION_COLUMNS = [
    "w_stiffness",
    "w_shear",
    "w_thermal",
    "vf_target",
    "stiffness",
    "shear",
    "thermal",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train the stage-II property-conditioned density generator.")
    p.add_argument("--manifest", type=str, required=True, help="Path to stage2_data/manifest.csv.")
    p.add_argument("--output_dir", type=str, default="./data/results/stage2_runs/density_generator", help="Training output directory.")
    p.add_argument("--target_key", type=str, default="rho_target_32", help="Supervision target key in each .npz file.")
    p.add_argument("--init_key", type=str, default="", help="Legacy smoothed seed-density input key.")
    p.add_argument("--use_rho_init", action="store_true", help="Legacy mode: train G(condition, rho_init) instead of G(condition).")
    p.add_argument("--condition_only", action="store_true", help="Deprecated compatibility flag; condition-only is now the default.")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=2.0e-4)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=20260510)
    p.add_argument("--device", type=str, default="auto", help="auto, cpu, cuda, cuda:<id>, or cuda:<id0>,<id1>,... for DataParallel.")
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--alpha_vol", type=float, default=1.0)
    p.add_argument("--beta_smooth", type=float, default=1.0e-3)
    p.add_argument("--out_res", type=int, default=32)
    p.add_argument("--base_res", type=int, default=4)
    p.add_argument("--cond_dim", type=int, default=7)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--base_channels", type=int, default=128)
    p.add_argument("--save_every", type=int, default=10)
    p.add_argument("--resume", type=str, default=None, help="Optional checkpoint path.")
    p.add_argument(
        "--resume_weights_only",
        action="store_true",
        help="Load only model weights from --resume and restart optimizer/scaler/epoch counters.",
    )
    p.add_argument("--require_target", action="store_true", help="Require target_available=1 in manifest.")
    p.add_argument("--allow_missing_target", dest="require_target", action="store_false", help="Do not filter by target_available.")
    p.set_defaults(require_target=None)
    return p.parse_args()


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(spec: str) -> torch.device:
    text = str(spec).strip().lower()
    if text == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if text == "cuda":
        return torch.device("cuda:0")
    if text.startswith("cuda:") and "," in text:
        first = text.split(":", 1)[1].split(",", 1)[0].strip()
        return torch.device(f"cuda:{int(first)}")
    return torch.device(text)


def parse_data_parallel_ids(spec: str) -> List[int]:
    text = str(spec).strip().lower()
    if not text.startswith("cuda:") or "," not in text:
        return []
    raw = text.split(":", 1)[1]
    ids = [int(part.strip()) for part in raw.split(",") if part.strip()]
    if len(ids) < 2:
        return []
    if not torch.cuda.is_available():
        raise RuntimeError(f"--device {spec!r} requests CUDA devices, but CUDA is unavailable")
    count = torch.cuda.device_count()
    bad = [idx for idx in ids if idx < 0 or idx >= count]
    if bad:
        raise ValueError(f"--device {spec!r} contains invalid CUDA ids {bad}; visible device_count={count}")
    return ids


def unwrap_model(model: torch.nn.Module) -> Stage2DensityGenerator:
    return model.module if isinstance(model, torch.nn.DataParallel) else model


def model_uses_noise_encoder(model: torch.nn.Module) -> bool:
    return bool(getattr(unwrap_model(model), "use_noise_encoder", False))


def load_model_state(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    base = unwrap_model(model)
    cleaned = {}
    for key, value in state.items():
        cleaned[key[7:] if key.startswith("module.") else key] = value
    base.load_state_dict(cleaned)


def split_dataset(dataset: Stage2NPZDataset, val_fraction: float, seed: int) -> Tuple[Subset, Subset]:
    n = len(dataset)
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    n_val = int(round(n * float(val_fraction)))
    if n > 1:
        n_val = max(1, min(n - 1, n_val))
    else:
        n_val = 0
    val_idx = indices[:n_val]
    train_idx = indices[n_val:]
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def append_log(path: Path, row: Dict[str, float]) -> None:
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def move_batch(batch: Dict[str, object], device: torch.device) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device=device, non_blocking=True)
        else:
            out[key] = value
    return out


def run_epoch(
    model: Stage2DensityGenerator,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer = None,
    scaler: torch.cuda.amp.GradScaler = None,
    *,
    use_amp: bool,
    alpha_vol: float,
    beta_smooth: float,
    grad_clip: float,
) -> Dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"loss": 0.0, "loss_rec": 0.0, "loss_vol": 0.0, "loss_smooth": 0.0}
    count = 0

    for batch in loader:
        batch = move_batch(batch, device)
        condition = batch["condition"]
        rho_init = batch.get("rho_init")
        rho_target = batch["rho_target"]
        vf_target = condition[:, 3]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            with torch.cuda.amp.autocast(enabled=use_amp):
                rho_pred = model(condition, rho_init if model_uses_noise_encoder(model) else None)
                losses = density_generation_loss(
                    rho_pred,
                    rho_target,
                    vf_target,
                    alpha_vol=alpha_vol,
                    beta_smooth=beta_smooth,
                )

            if training:
                if scaler is not None and use_amp:
                    scaler.scale(losses["loss"]).backward()
                    if grad_clip and grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    losses["loss"].backward()
                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))
                    optimizer.step()

        bs = int(condition.shape[0])
        count += bs
        for key in totals:
            totals[key] += float(losses[key].detach().item()) * bs

    denom = max(count, 1)
    return {key: value / denom for key, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    best_val_loss: float,
    args: argparse.Namespace,
) -> None:
    ckpt = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),
        "model": unwrap_model(model).state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "condition_columns": CONDITION_COLUMNS,
    }
    torch.save(ckpt, path)


def main() -> None:
    args = parse_args()
    setup_seed(int(args.seed))

    use_rho_init = bool(args.use_rho_init) and not bool(args.condition_only)
    args.condition_only = not use_rho_init
    if use_rho_init and str(args.init_key).strip() == "":
        raise ValueError("--use_rho_init requires --init_key, usually rho_init_32")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_csv = output_dir / "train_log.csv"
    config_json = output_dir / "config.json"
    config_payload = {**vars(args), "condition_columns": CONDITION_COLUMNS}
    config_json.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    data_parallel_ids = parse_data_parallel_ids(args.device)
    device = choose_device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    use_amp = bool(args.amp and device.type == "cuda")

    if args.require_target is None:
        require_target = str(args.target_key) != str(args.init_key)
    else:
        require_target = bool(args.require_target)

    dataset = Stage2NPZDataset(
        args.manifest,
        target_key=str(args.target_key),
        init_key=(str(args.init_key) if use_rho_init else None),
        require_target=require_target,
        require_init=use_rho_init,
    )
    if len(dataset) == 0:
        raise RuntimeError(
            "No training samples found. Check manifest path, target_available, and --target_key. "
            "Use maple.training.prepare_stage2_training_data without --no_target, and keep --target_key rho_target_32."
        )

    train_set, val_set = split_dataset(dataset, float(args.val_fraction), int(args.seed))
    if len(train_set) == 0:
        raise RuntimeError("Training split is empty.")

    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_set,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        pin_memory=pin_memory,
        drop_last=False,
    ) if len(val_set) > 0 else None

    base_model = Stage2DensityGenerator(
        cond_dim=int(args.cond_dim),
        out_res=int(args.out_res),
        base_res=int(args.base_res),
        latent_dim=int(args.latent_dim),
        base_channels=int(args.base_channels),
        use_noise_encoder=use_rho_init,
    ).to(device)

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    start_epoch = 1
    best_val_loss = float("inf")

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        load_model_state(base_model, ckpt["model"])

    if data_parallel_ids:
        model: torch.nn.Module = torch.nn.DataParallel(base_model, device_ids=data_parallel_ids, output_device=data_parallel_ids[0])
    else:
        model = base_model

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    if args.resume and not bool(args.resume_weights_only):
        optimizer.load_state_dict(ckpt["optimizer"])
        if ckpt.get("scaler") is not None:
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt.get("epoch", 0)) + 1
        best_val_loss = float(ckpt.get("best_val_loss", best_val_loss))

    print(f"[INFO] device={device} amp={use_amp}")
    if data_parallel_ids:
        print(f"[INFO] data_parallel_ids={data_parallel_ids}")
    print(f"[INFO] mode={'G(condition,rho_init)' if use_rho_init else 'G(condition)'}")
    print(f"[INFO] samples train={len(train_set)} val={len(val_set)}")
    print(f"[INFO] output_dir={output_dir}")

    for epoch in range(start_epoch, int(args.epochs) + 1):
        t0 = time.time()
        train_metrics = run_epoch(
            model,
            train_loader,
            device,
            optimizer,
            scaler,
            use_amp=use_amp,
            alpha_vol=float(args.alpha_vol),
            beta_smooth=float(args.beta_smooth),
            grad_clip=float(args.grad_clip),
        )

        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    val_loader,
                    device,
                    None,
                    None,
                    use_amp=use_amp,
                    alpha_vol=float(args.alpha_vol),
                    beta_smooth=float(args.beta_smooth),
                    grad_clip=0.0,
                )
        else:
            val_metrics = dict(train_metrics)

        elapsed = time.time() - t0
        row = {
            "epoch": int(epoch),
            "train_loss": train_metrics["loss"],
            "train_rec": train_metrics["loss_rec"],
            "train_vol": train_metrics["loss_vol"],
            "train_smooth": train_metrics["loss_smooth"],
            "val_loss": val_metrics["loss"],
            "val_rec": val_metrics["loss_rec"],
            "val_vol": val_metrics["loss_vol"],
            "val_smooth": val_metrics["loss_smooth"],
            "time_sec": elapsed,
        }
        append_log(log_csv, row)

        print(
            f"epoch={epoch:04d} "
            f"train={row['train_loss']:.6e} val={row['val_loss']:.6e} "
            f"rec={row['val_rec']:.6e} vol={row['val_vol']:.6e} "
            f"smooth={row['val_smooth']:.6e} time={elapsed:.1f}s"
        )

        save_checkpoint(output_dir / "latest.pt", model, optimizer, scaler, epoch, best_val_loss, args)
        if row["val_loss"] < best_val_loss:
            best_val_loss = float(row["val_loss"])
            save_checkpoint(output_dir / "best.pt", model, optimizer, scaler, epoch, best_val_loss, args)
        if int(args.save_every) > 0 and epoch % int(args.save_every) == 0:
            save_checkpoint(output_dir / f"epoch_{epoch:04d}.pt", model, optimizer, scaler, epoch, best_val_loss, args)

    print(f"[DONE] best_val_loss={best_val_loss:.6e}")
    print(f"[DONE] best_checkpoint={output_dir / 'best.pt'}")
    print(f"[DONE] train_log={log_csv}")


if __name__ == "__main__":
    main()
