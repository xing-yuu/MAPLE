from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from maple.core.utils.printability_eval import printability_score_binary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate binary printability scores for a dataset of optimized voxel densities."
    )
    p.add_argument("--input_csv", type=str, required=True, help="Input dataset_log.csv path.")
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV path. Default: <input_csv stem>_printability.csv",
    )
    p.add_argument("--rho_column", type=str, default="rho_final_npy", help="CSV column that stores the density .npy path.")
    p.add_argument("--threshold", type=float, default=0.5, help="Density threshold used for binarization.")
    p.add_argument("--overhang_deg", type=float, default=45.0, help="Allowed overhang angle in degrees.")
    p.add_argument("--periodic_xy", dest="periodic_xy", action="store_true", help="Use periodic support in x/y for overhang checks.")
    p.add_argument("--no-periodic_xy", dest="periodic_xy", action="store_false", help="Disable periodic support in x/y.")
    p.set_defaults(periodic_xy=True)
    p.add_argument("--min_channel_rad", type=int, default=3, help="Minimum channel radius in voxels.")
    p.add_argument("--channel_tiles", type=int, default=3, help="Supercell tiling used for channel accessibility.")
    p.add_argument("--channel_tiles_xyz", type=str, default=None, help="Optional x,y,z override for channel tiling.")
    p.add_argument("--connectivity_tiles", type=int, default=3, help="Supercell tiling used for solid connectivity.")
    p.add_argument("--connectivity_tiles_xyz", type=str, default=None, help="Optional x,y,z override for connectivity tiling.")
    p.add_argument("--seed_faces", type=str, default="all", choices=["all", "x", "y", "z", "xy"], help="Boundary faces treated as outside for channel accessibility.")
    p.add_argument("--label_connectivity", type=int, default=1, choices=[1, 2, 3], help="Connected-component neighborhood for 3D labeling.")
    p.add_argument("--lambda_oh", type=float, default=1.0, help="Weight for the overhang penalty.")
    p.add_argument("--lambda_chan", type=float, default=1.0, help="Weight for the channel accessibility penalty.")
    p.add_argument("--lambda_conn", type=float, default=1.0, help="Weight for the solid-connectivity penalty.")
    p.add_argument("--oh_tol", type=float, default=0.0, help="Pass threshold for P_oh.")
    p.add_argument("--chan_tol", type=float, default=0.0, help="Pass threshold for P_chan.")
    p.add_argument("--quiet", action="store_true", help="Reduce per-row progress logs.")
    return p.parse_args()


def parse_tiles_xyz(text: Optional[str]) -> Optional[Tuple[int, int, int]]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",")]
    if len(parts) != 3:
        raise ValueError(f"expected x,y,z for tiles override, got {text!r}")
    vals = tuple(int(p) for p in parts)
    if any(v <= 0 for v in vals):
        raise ValueError(f"tiles override must be positive, got {vals!r}")
    return vals


def resolve_rho_path(row: Dict[str, str], csv_path: Path, rho_column: str) -> Path:
    candidates: List[Path] = []

    raw = str(row.get(rho_column, "")).strip()
    if raw:
        p = Path(raw)
        candidates.append(p if p.is_absolute() else (csv_path.parent / p))

    sample_dir = str(row.get("sample_dir", "")).strip()
    if sample_dir:
        p = Path(sample_dir)
        p = p if p.is_absolute() else (csv_path.parent / p)
        candidates.append(p / "rho_final.npy")

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    tried = ", ".join(str(p) for p in candidates) if candidates else "<none>"
    raise FileNotFoundError(f"could not resolve rho_final.npy for row; tried: {tried}")


def load_rho_grid(path: Path) -> np.ndarray:
    rho = np.load(path)
    rho = np.asarray(rho, dtype=np.float32)
    if rho.ndim == 3:
        return rho
    if rho.ndim == 1:
        res_f = round(rho.size ** (1.0 / 3.0))
        if res_f ** 3 != int(rho.size):
            raise ValueError(f"flat rho file is not a perfect cube: {path}")
        return rho.reshape(res_f, res_f, res_f)
    raise ValueError(f"unsupported rho array shape {tuple(rho.shape)} in {path}")


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    if not input_csv.exists():
        raise FileNotFoundError(f"input CSV not found: {input_csv}")

    output_csv = (
        Path(args.output_csv).resolve()
        if args.output_csv is not None
        else input_csv.with_name(f"{input_csv.stem}_printability.csv")
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    channel_tiles_xyz = parse_tiles_xyz(args.channel_tiles_xyz)
    connectivity_tiles_xyz = parse_tiles_xyz(args.connectivity_tiles_xyz)

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames_in = list(reader.fieldnames or [])
        rows = list(reader)

    extra_fields = [
        "rho_eval_path",
        "resolution",
        "binary_threshold",
        "vf_binary",
        "support_radius",
        "unsupported_voxels",
        "considered_solid_voxels",
        "P_oh",
        "S_oh",
        "trapped_void_voxels",
        "void_voxels",
        "boundary_reachable_void_voxels",
        "has_cavity",
        "sealed_boundary",
        "P_chan",
        "S_chan",
        "solid_component_count",
        "P_conn",
        "S_conn",
        "lambda_oh",
        "lambda_chan",
        "lambda_conn",
        "P_print",
        "S_print",
        "overhang_ok",
        "channel_ok",
        "connectivity_ok",
        "printable_ok",
        "score_error",
    ]
    fieldnames_out = fieldnames_in + [f for f in extra_fields if f not in fieldnames_in]

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_out)
        writer.writeheader()

        for idx, row in enumerate(rows, start=1):
            out_row: Dict[str, object] = dict(row)
            try:
                rho_path = resolve_rho_path(row, input_csv, args.rho_column)
                rho_grid = load_rho_grid(rho_path)
                metrics = printability_score_binary(
                    rho_grid,
                    threshold=float(args.threshold),
                    overhang_deg=float(args.overhang_deg),
                    periodic_xy=bool(args.periodic_xy),
                    min_channel_rad=int(args.min_channel_rad),
                    channel_tiles=int(args.channel_tiles),
                    channel_tiles_xyz=channel_tiles_xyz,
                    connectivity_tiles=int(args.connectivity_tiles),
                    connectivity_tiles_xyz=connectivity_tiles_xyz,
                    seed_faces=str(args.seed_faces),
                    label_connectivity=int(args.label_connectivity),
                    lambda_oh=float(args.lambda_oh),
                    lambda_chan=float(args.lambda_chan),
                    lambda_conn=float(args.lambda_conn),
                    oh_tol=float(args.oh_tol),
                    chan_tol=float(args.chan_tol),
                )
                out_row.update(metrics)
                out_row["rho_eval_path"] = str(rho_path)
                out_row["score_error"] = ""
                if not args.quiet:
                    print(
                        f"[{idx}/{len(rows)}] scored {rho_path.name} "
                        f"S_print={float(metrics['S_print']):.2f} "
                        f"P_oh={float(metrics['P_oh']):.4f} "
                        f"P_chan={float(metrics['P_chan']):.4f} "
                        f"has_cavity={bool(metrics['has_cavity'])} "
                        f"components={int(metrics['solid_component_count'])}"
                    )
            except Exception as exc:
                out_row["rho_eval_path"] = ""
                out_row["score_error"] = str(exc)
                if not args.quiet:
                    print(f"[{idx}/{len(rows)}] error: {exc}")

            writer.writerow(out_row)

    print(f"[DONE] wrote: {output_csv}")


if __name__ == "__main__":
    main()
