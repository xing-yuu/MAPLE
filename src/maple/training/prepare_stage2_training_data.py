from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


CONDITION_COLUMNS = [
    "w_stiffness",
    "w_shear",
    "w_thermal",
    "vf_target",
    "stiffness",
    "shear",
    "thermal",
]
MANIFEST_CONDITION_COLUMNS = [f"cond_{name}" for name in CONDITION_COLUMNS]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Prepare stage-II network data from dataset_log.csv. For each row, "
            "save a property-conditioned vector "
            "[w_stiffness, w_shear, w_thermal, vf_target, stiffness, shear, thermal] "
            "and attach the optimized final density target at 32^3. The default "
            "network input is condition-only; rho_init_32 is stored only when "
            "--store_rho_init is passed for legacy experiments."
        )
    )
    p.add_argument("--dataset_log", type=str, required=True, help="Input dataset_log.csv from stage-I generation.")
    p.add_argument("--output_dir", type=str, required=True, help="Output directory for .npz samples and manifest.csv.")
    p.add_argument("--rho_column", type=str, default="rho_final_npy", help="Column containing optimized rho_final.npy path.")
    p.add_argument("--source_res", type=int, default=64, help="Legacy resolution used to generate smoothed seed noise.")
    p.add_argument("--out_res", type=int, default=32, help="Resolution saved for network training.")
    p.add_argument("--rho_min", type=float, default=1.0e-5, help="Minimum density after volume correction.")
    p.add_argument("--smooth_passes", type=int, default=4, help="Number of periodic binomial smoothing passes.")
    p.add_argument("--volume_iters", type=int, default=8, help="Mean-correction iterations after smoothing.")
    p.add_argument(
        "--store_rho_init",
        action="store_true",
        help="Also store the legacy rho_init_32 field. New property-conditioned training does not use it.",
    )
    p.add_argument(
        "--init_mode",
        type=str,
        default="seed_noise",
        choices=["seed_noise", "target_rho"],
        help="Legacy rho_init_32 source when --store_rho_init is set.",
    )
    p.add_argument("--default_vf", type=float, default=0.25, help="Fallback volume fraction when the CSV has no vf field.")
    p.add_argument("--default_w_stiffness", type=float, default=0.5, help="Fallback stiffness weight.")
    p.add_argument("--default_w_shear", type=float, default=0.0, help="Fallback shear weight.")
    p.add_argument("--default_w_thermal", type=float, default=0.5, help="Fallback thermal weight.")
    p.add_argument("--store_dtype", type=str, default="float16", choices=["float16", "float32"], help="Array dtype stored in .npz files.")
    p.add_argument("--no_target", action="store_true", help="Do not read/save optimized rho_final target fields.")
    p.add_argument("--overwrite", action="store_true", help="Regenerate existing sample .npz files.")
    return p.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def as_float(row: Dict[str, str], keys: Iterable[str], default: float) -> float:
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return float(default)


def parse_objective_weights(objective: str) -> Dict[str, float]:
    out = {"stiffness": 0.0, "shear": 0.0, "thermal": 0.0}
    text = str(objective or "").strip()
    if not text:
        return out

    text = text.replace("*", ":").replace("+", ",")
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            left, right = [x.strip().lower() for x in part.split(":", 1)]
            if left in out:
                out[left] = float(right)
            elif right in out:
                out[right] = float(left)
        elif part.lower() in out:
            out[part.lower()] = 1.0
    return out


def infer_weights(row: Dict[str, str], args: argparse.Namespace) -> Tuple[float, float, float]:
    objective_weights = parse_objective_weights(row.get("objective", ""))
    has_objective = any(name in str(row.get("objective", "")).lower() for name in ("stiffness", "shear", "thermal"))
    base_stiffness = objective_weights["stiffness"] if has_objective else args.default_w_stiffness
    base_shear = objective_weights["shear"] if has_objective else args.default_w_shear
    base_thermal = objective_weights["thermal"] if has_objective else args.default_w_thermal
    w_stiffness = as_float(row, ["w_stiffness"], base_stiffness)
    w_shear = as_float(row, ["w_shear"], base_shear)
    w_thermal = as_float(row, ["w_thermal"], base_thermal)
    total = w_stiffness + w_shear + w_thermal
    if total > 0:
        w_stiffness, w_shear, w_thermal = w_stiffness / total, w_shear / total, w_thermal / total
    return float(w_stiffness), float(w_shear), float(w_thermal)


def infer_vf(row: Dict[str, str], default_vf: float) -> float:
    if row.get("vf_target", "").strip():
        return float(row["vf_target"])
    if row.get("volfrac_target", "").strip():
        return float(row["volfrac_target"])
    if row.get("vf_percent", "").strip():
        return float(row["vf_percent"]) / 100.0
    if row.get("vf_final", "").strip():
        return float(row["vf_final"])
    return float(default_vf)


def infer_property(row: Dict[str, str], name: str, default: float = 0.0) -> float:
    keys = [name, f"prop_{name}", f"{name}_final", f"{name}_target"]
    if name == "thermal":
        keys.extend(["kappa", "prop_kappa", "kappa_final", "kappa_target"])
    return as_float(row, keys, default)


def build_condition(row: Dict[str, str], args: argparse.Namespace) -> np.ndarray:
    w_stiffness, w_shear, w_thermal = infer_weights(row, args)
    vf = infer_vf(row, args.default_vf)
    stiffness = infer_property(row, "stiffness", 0.0)
    shear = infer_property(row, "shear", 0.0)
    thermal = infer_property(row, "thermal", 0.0)
    cond = np.array(
        [
            w_stiffness,
            w_shear,
            w_thermal,
            vf,
            stiffness,
            shear,
            thermal,
        ],
        dtype=np.float32,
    )
    return cond


def binomial_smooth_periodic(vol: np.ndarray, passes: int) -> np.ndarray:
    out = np.asarray(vol, dtype=np.float32)
    if passes <= 0:
        return out

    weights = np.array([1.0, 4.0, 6.0, 4.0, 1.0], dtype=np.float32) / 16.0
    offsets = [-2, -1, 0, 1, 2]
    for _ in range(int(passes)):
        for axis in range(3):
            acc = np.zeros_like(out, dtype=np.float32)
            for weight, shift in zip(weights, offsets):
                acc += float(weight) * np.roll(out, shift=shift, axis=axis)
            out = acc
    return out


def match_mean_clamped(vol: np.ndarray, target_mean: float, rho_min: float, iters: int) -> np.ndarray:
    out = np.asarray(vol, dtype=np.float32)
    target = float(np.clip(target_mean, rho_min, 1.0))
    for _ in range(int(max(1, iters))):
        out = out - float(out.mean()) + target
        out = np.clip(out, float(rho_min), 1.0)
    return out.astype(np.float32, copy=False)


def make_smoothed_seed_density(seed: int, res: int, vf: float, rho_min: float, smooth_passes: int, volume_iters: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    rho = rng.random((res, res, res), dtype=np.float32)
    rho = match_mean_clamped(rho, vf, rho_min, volume_iters)
    rho = binomial_smooth_periodic(rho, smooth_passes)
    rho = match_mean_clamped(rho, vf, rho_min, volume_iters)
    return rho


def downsample_block_average(vol: np.ndarray, out_res: int) -> np.ndarray:
    arr = np.asarray(vol, dtype=np.float32)
    if arr.ndim != 3 or arr.shape[0] != arr.shape[1] or arr.shape[0] != arr.shape[2]:
        raise ValueError(f"expected cubic 3D volume, got {tuple(arr.shape)}")
    in_res = int(arr.shape[0])
    if in_res == int(out_res):
        return arr.astype(np.float32, copy=False)
    if in_res % int(out_res) != 0:
        raise ValueError(f"input resolution {in_res} must be divisible by output resolution {out_res}")
    factor = in_res // int(out_res)
    return arr.reshape(out_res, factor, out_res, factor, out_res, factor).mean(axis=(1, 3, 5)).astype(np.float32)


def relocated_candidates(path: Path, dataset_log: Path) -> List[Path]:
    candidates: List[Path] = []
    marker_names = [dataset_log.parent.name, "dataset-multiphysics"]
    parts = path.parts
    for marker in marker_names:
        for idx, part in enumerate(parts):
            if part != marker:
                continue
            suffix = Path(*parts[idx + 1 :])
            candidates.append(dataset_log.parent / suffix)
            candidates.append(dataset_log.parent.parent / marker / suffix)
    return candidates


def add_path_candidates(candidates: List[Path], raw_path: str, dataset_log: Path, *, append_rho: bool = False) -> None:
    raw = str(raw_path or "").strip()
    if not raw:
        return
    p = Path(raw)
    base = p if p.is_absolute() else dataset_log.parent / p
    candidates.append(base / "rho_final.npy" if append_rho else base)
    for relocated in relocated_candidates(p, dataset_log):
        candidates.append(relocated / "rho_final.npy" if append_rho else relocated)


def resolve_rho_path(row: Dict[str, str], dataset_log: Path, rho_column: str) -> Optional[Path]:
    candidates: List[Path] = []
    add_path_candidates(candidates, row.get(rho_column, ""), dataset_log)
    add_path_candidates(candidates, row.get("rho_final", ""), dataset_log)
    add_path_candidates(candidates, row.get("sample_dir", ""), dataset_log, append_rho=True)
    add_path_candidates(candidates, row.get("folder", ""), dataset_log, append_rho=True)

    seen = set()
    for p in candidates:
        key = str(p)
        if key in seen:
            continue
        seen.add(key)
        if p.exists():
            return p.resolve()
    return None


def load_target_rho_32(path: Path, out_res: int) -> np.ndarray:
    rho = np.asarray(np.load(path), dtype=np.float32)
    if rho.ndim == 1:
        res = round(rho.size ** (1.0 / 3.0))
        if res ** 3 != int(rho.size):
            raise ValueError(f"cannot reshape flat rho into cube: {path}")
        rho = rho.reshape(res, res, res)
    if rho.ndim != 3:
        raise ValueError(f"expected 3D rho in {path}, got {tuple(rho.shape)}")
    return downsample_block_average(rho, out_res)


def sample_name(index: int, row: Dict[str, str]) -> str:
    seed_text = str(row.get("seed", "")).strip()
    seed = int(float(seed_text)) if seed_text else int(index)
    prefix_parts = []
    for key in ("vf_percent", "weight_idx", "w1_idx", "w2_idx", "w3_idx"):
        value = str(row.get(key, "")).strip()
        if value:
            prefix_parts.append(f"{key}-{value}")
    prefix = "_".join(prefix_parts) if prefix_parts else f"sample-{index:06d}"
    return f"{prefix}_seed-{seed:010d}.npz"


def write_manifest_header(path: Path, source_fields: List[str]) -> None:
    fields = [
        "stage2_npz",
        "target_available",
        "condition_dim",
        *MANIFEST_CONDITION_COLUMNS,
        "source_rho_final_npy",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*source_fields, *fields])


def append_manifest_row(path: Path, source_fields: List[str], row: Dict[str, str], extra: Dict[str, object]) -> None:
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([*(row.get(k, "") for k in source_fields), *(extra[k] for k in extra)])


def main() -> None:
    args = parse_args()
    dataset_log = Path(args.dataset_log).resolve()
    output_dir = Path(args.output_dir).resolve()
    samples_dir = output_dir / "samples"
    manifest_csv = output_dir / "manifest.csv"
    metadata_json = output_dir / "metadata.json"

    if args.store_rho_init and args.source_res % args.out_res != 0:
        raise ValueError("--source_res must be divisible by --out_res")

    rows = read_csv_rows(dataset_log)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir.mkdir(parents=True, exist_ok=True)

    source_fields = list(rows[0].keys()) if rows else []
    write_manifest_header(manifest_csv, source_fields)

    dtype = np.float16 if args.store_dtype == "float16" else np.float32
    written = 0
    skipped = 0

    for index, row in enumerate(rows):
        if not row.get("seed", "").strip():
            skipped += 1
            continue

        seed = int(float(row["seed"]))
        vf = infer_vf(row, args.default_vf)
        condition = build_condition(row, args)
        out_path = samples_dir / sample_name(index, row)

        source_rho = resolve_rho_path(row, dataset_log, args.rho_column) if not args.no_target else None

        if out_path.exists() and not args.overwrite:
            skipped += 1
        else:
            arrays: Dict[str, np.ndarray] = {
                "condition": condition.astype(np.float32),
                "seed": np.array(seed, dtype=np.int64),
                "source_index": np.array(index, dtype=np.int64),
            }

            if args.store_rho_init:
                if str(args.init_mode) == "target_rho" and source_rho is not None:
                    rho_init_32 = load_target_rho_32(source_rho, int(args.out_res))
                else:
                    rho64 = make_smoothed_seed_density(
                        seed=seed,
                        res=int(args.source_res),
                        vf=vf,
                        rho_min=float(args.rho_min),
                        smooth_passes=int(args.smooth_passes),
                        volume_iters=int(args.volume_iters),
                    )
                    rho_init_32 = downsample_block_average(rho64, int(args.out_res))
                arrays["rho_init_32"] = rho_init_32[None].astype(dtype)

            if source_rho is not None:
                arrays["rho_target_32"] = load_target_rho_32(source_rho, int(args.out_res))[None].astype(dtype)

            np.savez_compressed(out_path, **arrays)
            written += 1

        extra = {
            "stage2_npz": str(out_path),
            "target_available": int(source_rho is not None),
            "condition_dim": int(condition.shape[0]),
            "cond_w_stiffness": float(condition[0]),
            "cond_w_shear": float(condition[1]),
            "cond_w_thermal": float(condition[2]),
            "cond_vf_target": float(condition[3]),
            "cond_stiffness": float(condition[4]),
            "cond_shear": float(condition[5]),
            "cond_thermal": float(condition[6]),
            "source_rho_final_npy": "" if source_rho is None else str(source_rho),
        }
        append_manifest_row(manifest_csv, source_fields, row, extra)

    metadata = {
        "dataset_log": str(dataset_log),
        "sample_count": len(rows),
        "written_count": written,
        "skipped_count": skipped,
        "source_res": int(args.source_res),
        "out_res": int(args.out_res),
        "smooth_passes": int(args.smooth_passes),
        "init_mode": str(args.init_mode),
        "condition_columns": CONDITION_COLUMNS,
        "store_dtype": str(args.store_dtype),
        "store_rho_init": bool(args.store_rho_init),
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[DONE] samples_dir={samples_dir}")
    print(f"[DONE] manifest={manifest_csv}")
    print(f"[DONE] metadata={metadata_json}")


if __name__ == "__main__":
    main()
