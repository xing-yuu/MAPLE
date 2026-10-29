from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Iteratively densify a Pareto front with the property-conditioned 32^3 network. "
            "Each round generates N network-warm-start samples, appends them to the source "
            "CSV, prepares stage-II data, then fine-tunes from the previous checkpoint."
        )
    )
    p.add_argument(
        "--initial_source_csv",
        type=str,
        default="./data/datasets/multiphysics/dataset_log.csv",
        help="Initial Pareto/front dataset_log.csv. Defaults to ./data/datasets/multiphysics/dataset_log.csv.",
    )
    p.add_argument(
        "--initial_checkpoint",
        type=str,
        default=None,
        help="Optional initial condition-only stage-II checkpoint. If omitted or missing, the script trains it first from --initial_source_csv.",
    )
    p.add_argument("--output_root", type=str, default="./data/results/iterative-property-network-32")
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--samples_per_round", type=int, default=200)
    p.add_argument("--w_start", type=float, default=0.0)
    p.add_argument("--w_end", type=float, default=1.0)
    p.add_argument("--w_step", type=float, default=0.005)
    p.add_argument("--seeds_per_w", type=int, default=2)
    p.add_argument(
        "--sampling_mode",
        type=str,
        default="adaptive_front",
        choices=["uniform", "adaptive_front"],
        help="Sampling strategy for network densification. adaptive_front focuses on sparse Pareto-front segments.",
    )
    p.add_argument(
        "--front_objectives",
        type=str,
        default="e_kappa_vf",
        choices=["auto", "weighted", "e_kappa_vf"],
        help=(
            "Front geometry to densify. Use e_kappa_vf for the reviewer experiment: "
            "maximize E, maximize kappa, minimize volume fraction."
        ),
    )
    p.add_argument("--adaptive_min_gap_quantile", type=float, default=0.50)
    p.add_argument("--adaptive_gap_power", type=float, default=1.0)
    p.add_argument("--adaptive_front_eps", type=float, default=1.0e-9)
    p.add_argument("--tri_front_k", type=int, default=6)
    p.add_argument("--vf_objective", type=str, default="minimize", choices=["minimize", "maximize"])
    p.add_argument("--tri_weight_mode", type=str, default="source", choices=["cycle", "random", "source"])
    p.add_argument(
        "--tri_weight_values",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated w_E values for e_kappa_vf densification; w_kappa is 1-w_E.",
    )
    p.add_argument("--default_w_stiffness", type=float, default=0.5)
    p.add_argument("--default_w_shear", type=float, default=0.0)
    p.add_argument("--default_w_thermal", type=float, default=0.5)
    p.add_argument("--master_seed", type=int, default=20260528)
    p.add_argument("--seed_upper", type=int, default=2_147_483_647)
    p.add_argument("--volfrac", type=float, default=None, help="Optional fixed vf override for network densification.")
    p.add_argument("--exclude_exact", action="store_true")
    p.add_argument("--to_iters", type=int, default=2000)
    p.add_argument("--penal", type=float, default=3.0)
    p.add_argument("--rho_min", type=float, default=1.0e-5)
    p.add_argument("--move", type=float, default=0.05)
    p.add_argument("--xmin", type=float, default=1.0e-5)
    p.add_argument("--filter_k", type=int, default=3)
    p.add_argument("--symmetry", type=str, default="iso")
    p.add_argument("--lambda_print", type=float, default=0.9)
    p.add_argument("--overhang_deg", type=float, default=45.0)
    p.add_argument("--lambda_cavity", type=float, default=0.9)
    p.add_argument("--cavity_thresh", type=float, default=0.5)
    p.add_argument("--cavity_beta", type=float, default=20.0)
    p.add_argument("--min_channel_rad", type=int, default=3)
    p.add_argument("--cavity_rad", type=int, default=1)
    p.add_argument("--cavity_seed_faces", type=str, default="all", choices=["all", "x", "y", "z", "xy"])
    p.add_argument("--cavity_iters", type=int, default=128)
    p.add_argument("--cavity_tiles", type=int, default=2)
    p.add_argument("--cavity_tiles_xyz", type=str, default=None)
    p.add_argument("--no-cavity_focus_center", dest="cavity_focus_center", action="store_false")
    p.set_defaults(cavity_focus_center=True)
    p.add_argument("--gmg_smooth_iter", type=int, nargs="+", default=[2, 2, 3, 4, 5])
    p.add_argument("--gmg_max_cycle", type=int, default=1)
    p.add_argument("--mesh_res", type=int, default=128)
    p.add_argument("--mesh_iso", type=float, default=0.3)
    p.add_argument("--mesh_pad", type=int, default=2)
    p.add_argument("--warm_sigma", type=float, default=0.0)
    p.add_argument("--warm_smooth_passes", type=int, default=0)
    p.add_argument("--volume_iters", type=int, default=8)
    p.add_argument("--gpus", type=str, default="auto")
    p.add_argument("--workers_per_gpu", type=int, default=1)
    p.add_argument("--optimizer_verbose", action="store_true")
    p.add_argument("--save_iter_log", action="store_true")
    p.add_argument("--initial_epochs", type=int, default=200, help="Epochs for the first training pass when no initial checkpoint is available.")
    p.add_argument("--initial_lr", type=float, default=2.0e-4, help="Learning rate for the first training pass.")
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1.0e-4)
    p.add_argument("--weight_decay", type=float, default=1.0e-4)
    p.add_argument("--val_fraction", type=float, default=0.1)
    p.add_argument("--train_seed", type=int, default=20260528)
    p.add_argument("--device", type=str, default="auto", help="Training device: auto, cpu, cuda, or cuda:<id>.")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--alpha_vol", type=float, default=1.0)
    p.add_argument("--beta_smooth", type=float, default=1.0e-3)
    p.add_argument("--save_every", type=int, default=0)
    p.add_argument("--store_dtype", type=str, default="float16", choices=["float16", "float32"])
    p.add_argument("--overwrite", action="store_true", help="Regenerate/retrain existing round outputs.")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def run_command(cmd: List[str], *, cwd: Path, quiet: bool = False) -> None:
    if not quiet:
        print("[CMD] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_union_csv(path: Path, csv_paths: Iterable[Path]) -> int:
    rows: List[Dict[str, str]] = []
    fields: List[str] = []
    seen_fields = set()
    for csv_path in csv_paths:
        if not csv_path.exists():
            continue
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for name in reader.fieldnames or []:
                if name not in seen_fields:
                    fields.append(name)
                    seen_fields.add(name)
            for row in reader:
                rows.append(dict(row))

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fields})
    return len(rows)


def count_rows(path: Path) -> int:
    return len(read_csv_rows(path))


def csv_has_weight_columns(path: Path) -> bool:
    if not path.exists():
        return False
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
    return bool({"w_stiffness", "w_shear", "w_thermal", "w_kappa", "weight_idx", "objective"} & fields)


def append_summary(path: Path, row: Dict[str, object]) -> None:
    exists = path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def copy_initial_source(initial_source: Path, cumulative_csv: Path) -> None:
    cumulative_csv.parent.mkdir(parents=True, exist_ok=True)
    if not cumulative_csv.exists():
        shutil.copyfile(initial_source, cumulative_csv)


def add_optional(cmd: List[str], flag: str, value: Optional[object]) -> None:
    if value is not None:
        cmd.extend([flag, str(value)])


def prepare_stage2_data(
    *,
    repo: Path,
    dataset_log: Path,
    output_dir: Path,
    store_dtype: str,
    args: argparse.Namespace,
    quiet: bool,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "maple.training.prepare_stage2_training_data",
        "--dataset_log",
        str(dataset_log),
        "--output_dir",
        str(output_dir),
        "--out_res",
        "32",
        "--store_dtype",
        str(store_dtype),
        "--default_w_stiffness",
        str(args.default_w_stiffness),
        "--default_w_shear",
        str(args.default_w_shear),
        "--default_w_thermal",
        str(args.default_w_thermal),
        "--overwrite",
    ]
    run_command(cmd, cwd=repo, quiet=quiet)
    manifest = output_dir / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(f"Missing manifest after prepare: {manifest}")
    return manifest


def train_stage2_model(
    *,
    repo: Path,
    manifest: Path,
    output_dir: Path,
    epochs: int,
    lr: float,
    args: argparse.Namespace,
    seed: int,
    resume: Optional[Path],
    quiet: bool,
) -> Path:
    cmd = [
        sys.executable,
        "-m",
        "maple.training.train_stage2_density_generator",
        "--manifest",
        str(manifest),
        "--output_dir",
        str(output_dir),
        "--target_key",
        "rho_target_32",
        "--epochs",
        str(epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--lr",
        str(lr),
        "--weight_decay",
        str(args.weight_decay),
        "--val_fraction",
        str(args.val_fraction),
        "--seed",
        str(seed),
        "--device",
        str(args.device),
        "--alpha_vol",
        str(args.alpha_vol),
        "--beta_smooth",
        str(args.beta_smooth),
        "--save_every",
        str(args.save_every),
    ]
    if resume is not None:
        cmd.extend(["--resume", str(resume), "--resume_weights_only"])
    if args.amp:
        cmd.append("--amp")
    run_command(cmd, cwd=repo, quiet=quiet)

    best = output_dir / "best.pt"
    latest = output_dir / "latest.pt"
    checkpoint = best if best.exists() else latest
    if not checkpoint.exists():
        raise FileNotFoundError(f"Training did not produce a checkpoint in {output_dir}")
    return checkpoint


def main() -> None:
    args = parse_args()
    if int(args.rounds) <= 0:
        raise ValueError("--rounds must be positive")
    if int(args.samples_per_round) <= 0:
        raise ValueError("--samples_per_round must be positive")

    # Resolve subprocesses and relative dataset paths from the checkout where
    # the command was launched. The shell wrappers change to the repository
    # root before invoking this module, and this also works for editable or
    # wheel installations.
    repo = Path.cwd()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    initial_source_csv = Path(args.initial_source_csv).resolve()
    initial_checkpoint = Path(args.initial_checkpoint).resolve() if args.initial_checkpoint else None
    if (
        str(args.front_objectives) == "e_kappa_vf"
        and str(args.tri_weight_mode) == "source"
        and not csv_has_weight_columns(initial_source_csv)
    ):
        raise ValueError(
            "Paper-style e_kappa_vf densification requires source weights to be present in the CSV "
            "(w_stiffness/w_thermal or objective). The current initial_source_csv has no weight columns, "
            "so w_E/w_kappa cannot be obtained by front interpolation."
        )
    cumulative_csv = output_root / "cumulative_source.csv"
    summary_csv = output_root / "iteration_summary.csv"
    config_json = output_root / "config.json"
    config_json.write_text(json.dumps(vars(args), indent=2), encoding="utf-8")

    copy_initial_source(initial_source_csv, cumulative_csv)
    current_source_csv = cumulative_csv

    if initial_checkpoint is not None and initial_checkpoint.exists():
        current_checkpoint = initial_checkpoint
    else:
        if initial_checkpoint is not None:
            print(f"[INITIAL] checkpoint not found, training from initial_source_csv: {initial_checkpoint}", flush=True)
        else:
            print("[INITIAL] no initial checkpoint provided, training from initial_source_csv", flush=True)

        initial_stage2_dir = output_root / "initial_stage2_data"
        initial_train_dir = output_root / "initial_train"
        initial_best = initial_train_dir / "best.pt"
        initial_latest = initial_train_dir / "latest.pt"
        existing_initial = initial_best if initial_best.exists() else initial_latest
        if existing_initial.exists() and not args.overwrite:
            current_checkpoint = existing_initial
            print(f"[INITIAL] reuse trained checkpoint: {current_checkpoint}", flush=True)
        else:
            initial_manifest = prepare_stage2_data(
                repo=repo,
                dataset_log=initial_source_csv,
                output_dir=initial_stage2_dir,
                store_dtype=str(args.store_dtype),
                args=args,
                quiet=bool(args.quiet),
            )
            current_checkpoint = train_stage2_model(
                repo=repo,
                manifest=initial_manifest,
                output_dir=initial_train_dir,
                epochs=int(args.initial_epochs),
                lr=float(args.initial_lr),
                args=args,
                seed=int(args.train_seed),
                resume=None,
                quiet=bool(args.quiet),
            )

    generated_logs: List[Path] = []
    for round_idx in range(1, int(args.rounds) + 1):
        t0 = time.time()
        round_dir = output_root / f"round_{round_idx:02d}"
        generated_dir = round_dir / "generated"
        stage2_dir = round_dir / "stage2_data"
        finetune_dir = round_dir / "finetune"
        generated_log = generated_dir / "dataset_log.csv"
        best_checkpoint = finetune_dir / "best.pt"
        latest_checkpoint = finetune_dir / "latest.pt"

        print(
            f"[ROUND {round_idx:02d}] source={current_source_csv} checkpoint={current_checkpoint}",
            flush=True,
        )

        generated_count = 0 if args.overwrite else count_rows(generated_log)
        if args.overwrite or generated_count < int(args.samples_per_round):
            samples_to_generate = int(args.samples_per_round) if args.overwrite else int(args.samples_per_round) - generated_count
            densify_cmd = [
                sys.executable,
                "-m",
                "maple.pareto.densify_pareto_property_network_32",
                "--source_csv",
                str(current_source_csv),
                "--checkpoint",
                str(current_checkpoint),
                "--output_root",
                str(generated_dir),
                "--w_start",
                str(args.w_start),
                "--w_end",
                str(args.w_end),
                "--w_step",
                str(args.w_step),
                "--seeds_per_w",
                str(args.seeds_per_w),
                "--front_objectives",
                str(args.front_objectives),
                "--sampling_mode",
                str(args.sampling_mode),
                "--adaptive_min_gap_quantile",
                str(args.adaptive_min_gap_quantile),
                "--adaptive_gap_power",
                str(args.adaptive_gap_power),
                "--adaptive_front_eps",
                str(args.adaptive_front_eps),
                "--tri_front_k",
                str(args.tri_front_k),
                "--vf_objective",
                str(args.vf_objective),
                "--tri_weight_mode",
                str(args.tri_weight_mode),
                "--tri_weight_values",
                str(args.tri_weight_values),
                "--default_w_stiffness",
                str(args.default_w_stiffness),
                "--default_w_shear",
                str(args.default_w_shear),
                "--default_w_thermal",
                str(args.default_w_thermal),
                "--max_samples",
                str(samples_to_generate),
                "--master_seed",
                str(int(args.master_seed) + round_idx * 10_000),
                "--seed_upper",
                str(args.seed_upper),
                "--iters",
                str(args.to_iters),
                "--penal",
                str(args.penal),
                "--rho_min",
                str(args.rho_min),
                "--move",
                str(args.move),
                "--xmin",
                str(args.xmin),
                "--filter_k",
                str(args.filter_k),
                "--symmetry",
                str(args.symmetry),
                "--lambda_print",
                str(args.lambda_print),
                "--overhang_deg",
                str(args.overhang_deg),
                "--lambda_cavity",
                str(args.lambda_cavity),
                "--cavity_thresh",
                str(args.cavity_thresh),
                "--cavity_beta",
                str(args.cavity_beta),
                "--min_channel_rad",
                str(args.min_channel_rad),
                "--cavity_rad",
                str(args.cavity_rad),
                "--cavity_seed_faces",
                str(args.cavity_seed_faces),
                "--cavity_iters",
                str(args.cavity_iters),
                "--cavity_tiles",
                str(args.cavity_tiles),
                "--gmg_smooth_iter",
                *(str(x) for x in args.gmg_smooth_iter),
                "--gmg_max_cycle",
                str(args.gmg_max_cycle),
                "--mesh_res",
                str(args.mesh_res),
                "--mesh_iso",
                str(args.mesh_iso),
                "--mesh_pad",
                str(args.mesh_pad),
                "--warm_sigma",
                str(args.warm_sigma),
                "--warm_smooth_passes",
                str(args.warm_smooth_passes),
                "--volume_iters",
                str(args.volume_iters),
                "--gpus",
                str(args.gpus),
                "--workers_per_gpu",
                str(args.workers_per_gpu),
            ]
            add_optional(densify_cmd, "--volfrac", args.volfrac)
            add_optional(densify_cmd, "--cavity_tiles_xyz", args.cavity_tiles_xyz)
            if args.exclude_exact:
                densify_cmd.append("--exclude_exact")
            if not args.cavity_focus_center:
                densify_cmd.append("--no-cavity_focus_center")
            if args.optimizer_verbose:
                densify_cmd.append("--optimizer_verbose")
            if args.save_iter_log:
                densify_cmd.append("--save_iter_log")
            if args.overwrite:
                densify_cmd.append("--overwrite")
            if args.quiet:
                densify_cmd.append("--quiet")
            run_command(densify_cmd, cwd=repo, quiet=bool(args.quiet))
        else:
            print(f"[ROUND {round_idx:02d}] reuse generated log: {generated_log}", flush=True)

        if not generated_log.exists():
            raise FileNotFoundError(f"Round {round_idx} did not produce {generated_log}")

        generated_logs.append(generated_log)
        next_cumulative_csv = output_root / f"cumulative_source_round_{round_idx:02d}.csv"
        cumulative_rows = write_union_csv(next_cumulative_csv, [initial_source_csv, *generated_logs])
        shutil.copyfile(next_cumulative_csv, cumulative_csv)

        existing_checkpoint = best_checkpoint if best_checkpoint.exists() else latest_checkpoint
        if existing_checkpoint.exists() and not args.overwrite:
            next_checkpoint = existing_checkpoint
            manifest = stage2_dir / "manifest.csv"
            if not args.quiet:
                print(f"[ROUND {round_idx:02d}] reuse fine-tuned checkpoint: {next_checkpoint}", flush=True)
        else:
            prepare_cmd = [
                sys.executable,
                "-m",
                "maple.training.prepare_stage2_training_data",
                "--dataset_log",
                str(cumulative_csv),
                "--output_dir",
                str(stage2_dir),
                "--out_res",
                "32",
                "--store_dtype",
                str(args.store_dtype),
                "--default_w_stiffness",
                str(args.default_w_stiffness),
                "--default_w_shear",
                str(args.default_w_shear),
                "--default_w_thermal",
                str(args.default_w_thermal),
                "--overwrite",
            ]
            run_command(prepare_cmd, cwd=repo, quiet=bool(args.quiet))

            manifest = stage2_dir / "manifest.csv"
            if not manifest.exists():
                raise FileNotFoundError(f"Missing manifest after prepare: {manifest}")

            train_cmd = [
                sys.executable,
                "-m",
                "maple.training.train_stage2_density_generator",
                "--manifest",
                str(manifest),
                "--output_dir",
                str(finetune_dir),
                "--target_key",
                "rho_target_32",
                "--epochs",
                str(args.finetune_epochs),
                "--batch_size",
                str(args.batch_size),
                "--num_workers",
                str(args.num_workers),
                "--lr",
                str(args.lr),
                "--weight_decay",
                str(args.weight_decay),
                "--val_fraction",
                str(args.val_fraction),
                "--seed",
                str(int(args.train_seed) + round_idx),
                "--device",
                str(args.device),
                "--alpha_vol",
                str(args.alpha_vol),
                "--beta_smooth",
                str(args.beta_smooth),
                "--save_every",
                str(args.save_every),
                "--resume",
                str(current_checkpoint),
                "--resume_weights_only",
            ]
            if args.amp:
                train_cmd.append("--amp")
            run_command(train_cmd, cwd=repo, quiet=bool(args.quiet))

            next_checkpoint = best_checkpoint
            if not next_checkpoint.exists():
                next_checkpoint = latest_checkpoint
        if not next_checkpoint.exists():
            raise FileNotFoundError(f"Fine-tune did not produce a checkpoint in {finetune_dir}")

        append_summary(
            summary_csv,
            {
                "round": round_idx,
                "generated_log": str(generated_log),
                "generated_rows": count_rows(generated_log),
                "cumulative_csv": str(cumulative_csv),
                "cumulative_rows": cumulative_rows,
                "manifest": str(manifest),
                "checkpoint_in": str(current_checkpoint),
                "checkpoint_out": str(next_checkpoint),
                "time_sec": time.time() - t0,
            },
        )
        current_source_csv = cumulative_csv
        current_checkpoint = next_checkpoint

    print(f"[DONE] final_checkpoint={current_checkpoint}")
    print(f"[DONE] cumulative_source={cumulative_csv}")
    print(f"[DONE] summary={summary_csv}")


if __name__ == "__main__":
    main()
