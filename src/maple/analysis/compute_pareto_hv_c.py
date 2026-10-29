from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


DEFAULT_INITIAL = "./data/datasets/multiphysics/dataset_log_printability.csv"
DEFAULT_COMBINED = [
    DEFAULT_INITIAL,
    "./data/results/iterative/cumulative_source_round_01.csv",
    "./data/results/iterative/cumulative_source_round_02.csv",
    "./data/results/iterative/cumulative_source_round_03.csv",
    "./data/results/iterative/cumulative_source_round_04.csv",
    "./data/results/iterative/cumulative_source_round_05.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Table-style Pareto-front HV and C-metric for the initial "
            "dataset and the merged iterative dataset."
        )
    )
    parser.add_argument("--initial_csv", type=str, default=DEFAULT_INITIAL)
    parser.add_argument("--combined_csvs", type=str, nargs="+", default=DEFAULT_COMBINED)
    parser.add_argument("--output_dir", type=str, default="./data/results/pareto_metric_outputs")
    parser.add_argument("--prefix", type=str, default="iterative_e_kappa_vf")
    parser.add_argument("--e_key", type=str, default="stiffness")
    parser.add_argument("--kappa_key", type=str, default="auto", help="auto, thermal, or kappa.")
    parser.add_argument("--vf_key", type=str, default="vf_target")
    parser.add_argument("--kappa_divisor", type=float, default=3.0)
    parser.add_argument(
        "--front_mode",
        type=str,
        default="tri",
        choices=["tri", "ekappa"],
        help="tri uses maximize [E, kappa/kappa_divisor, 1-vf]; ekappa uses maximize [E, kappa/kappa_divisor].",
    )
    parser.add_argument(
        "--feasible_only",
        action="store_true",
        help="Drop rows with printability_ok=False when that column is present. Blank/missing values are kept.",
    )
    return parser.parse_args()


def first_float(row: Dict[str, str], keys: Iterable[str]) -> float:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return float("nan")


def false_value(value: str) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "n"}


def metrics(row: Dict[str, str], args: argparse.Namespace) -> Tuple[float, float, float]:
    e_value = first_float(row, (args.e_key, "E", "youngs", "prop_stiffness"))
    if args.kappa_key == "auto":
        kappa_value = first_float(row, ("thermal", "kappa", "prop_thermal", "prop_kappa"))
    else:
        kappa_value = first_float(row, (args.kappa_key,))
    if math.isfinite(kappa_value) and args.kappa_divisor != 0:
        kappa_value /= float(args.kappa_divisor)
    vf_value = first_float(row, (args.vf_key, "vf_final", "volume_fraction", "vf_binary"))
    return e_value, kappa_value, vf_value


def row_key(row: Dict[str, str], args: argparse.Namespace) -> str:
    for key in ("sample_id", "rho_final_npy", "rho_eval_path", "sample_dir", "folder"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"{key}:{value}"
    e_value, kappa_value, vf_value = metrics(row, args)
    seed = str(row.get("seed", "")).strip()
    return f"fallback:{seed}:{vf_value:.8g}:{e_value:.8g}:{kappa_value:.8g}"


def load_dataset(paths: Sequence[Path], args: argparse.Namespace) -> Tuple[List[Dict[str, str]], np.ndarray]:
    merged: Dict[str, Dict[str, str]] = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                if args.feasible_only and false_value(row.get("printability_ok", "")):
                    continue
                key = row_key(row, args)
                if key not in merged:
                    merged[key] = dict(row)
                else:
                    existing = merged[key]
                    for field, value in row.items():
                        if str(existing.get(field, "")).strip() == "" and str(value).strip() != "":
                            existing[field] = value

    rows: List[Dict[str, str]] = []
    values: List[Tuple[float, float, float]] = []
    for row in merged.values():
        e_value, kappa_value, vf_value = metrics(row, args)
        if math.isfinite(e_value) and math.isfinite(kappa_value) and math.isfinite(vf_value):
            rows.append(row)
            values.append((e_value, kappa_value, vf_value))
    return rows, np.asarray(values, dtype=float)


def objective_values(values: np.ndarray, mode: str) -> np.ndarray:
    if mode == "tri":
        return np.column_stack([values[:, 0], values[:, 1], 1.0 - values[:, 2]])
    return values[:, :2]


def pareto_mask(objectives: np.ndarray) -> np.ndarray:
    keep = np.ones(objectives.shape[0], dtype=bool)
    eps = 1.0e-12
    for idx in range(objectives.shape[0]):
        if not keep[idx]:
            continue
        point = objectives[idx]
        dominated = np.all(objectives >= point - eps, axis=1) & np.any(objectives > point + eps, axis=1)
        dominated[idx] = False
        if np.any(dominated):
            keep[idx] = False
    return keep


def hypervolume(points: np.ndarray, reference: np.ndarray) -> float:
    if points.size == 0:
        return 0.0
    points = points[np.all(points > reference + 1.0e-15, axis=1)]
    if points.size == 0:
        return 0.0
    points = points[pareto_mask(points)]

    def recurse(active: np.ndarray, ref: np.ndarray) -> float:
        if active.size == 0:
            return 0.0
        if active.shape[1] == 1:
            return max(0.0, float(np.max(active[:, 0]) - ref[0]))

        coords = np.unique(np.concatenate(([ref[0]], active[:, 0])))
        coords.sort()
        total = 0.0
        for left, right in zip(coords[:-1], coords[1:]):
            width = float(right - left)
            if width <= 0:
                continue
            slab = active[active[:, 0] >= right - 1.0e-15][:, 1:]
            if slab.size:
                total += width * recurse(slab, ref[1:])
        return float(total)

    return recurse(points, reference)


def c_metric(front_a: np.ndarray, front_b: np.ndarray) -> float:
    if front_b.size == 0:
        return float("nan")
    covered = 0
    for point_b in front_b:
        if np.any(np.all(front_a >= point_b - 1.0e-12, axis=1)):
            covered += 1
    return float(covered / front_b.shape[0])


def summarize(
    name: str,
    rows: Sequence[Dict[str, str]],
    objectives: np.ndarray,
    front: np.ndarray,
    hv_raw: float,
    hv_normalized: float,
    c_dataset_initial: float,
    c_initial_dataset: float,
) -> Dict[str, object]:
    return {
        "dataset": name,
        "valid_unique_rows": len(rows),
        "pareto_rows": int(front.shape[0]),
        "HV_raw": hv_raw,
        "HV_normalized": hv_normalized,
        "C(dataset, initial)": c_dataset_initial,
        "C(initial, dataset)": c_initial_dataset,
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_paths = [Path(args.initial_csv).expanduser().resolve()]
    combined_paths = [Path(path).expanduser().resolve() for path in args.combined_csvs]

    initial_rows, initial_values = load_dataset(initial_paths, args)
    combined_rows, combined_values = load_dataset(combined_paths, args)

    initial_objectives = objective_values(initial_values, args.front_mode)
    combined_objectives = objective_values(combined_values, args.front_mode)
    initial_front = initial_objectives[pareto_mask(initial_objectives)]
    combined_front = combined_objectives[pareto_mask(combined_objectives)]

    raw_reference = np.zeros(initial_objectives.shape[1], dtype=float)
    all_objectives = np.vstack([initial_objectives, combined_objectives])
    obj_min = np.min(all_objectives, axis=0)
    obj_max = np.max(all_objectives, axis=0)
    span = np.maximum(obj_max - obj_min, 1.0e-15)
    initial_front_normalized = (initial_front - obj_min) / span
    combined_front_normalized = (combined_front - obj_min) / span

    initial_hv_raw = hypervolume(initial_front, raw_reference)
    combined_hv_raw = hypervolume(combined_front, raw_reference)
    initial_hv_normalized = hypervolume(initial_front_normalized, np.zeros_like(raw_reference))
    combined_hv_normalized = hypervolume(combined_front_normalized, np.zeros_like(raw_reference))

    table = [
        summarize(
            "Initial",
            initial_rows,
            initial_objectives,
            initial_front,
            initial_hv_raw,
            initial_hv_normalized,
            1.0,
            1.0,
        ),
        summarize(
            "Combined",
            combined_rows,
            combined_objectives,
            combined_front,
            combined_hv_raw,
            combined_hv_normalized,
            c_metric(combined_front, initial_front),
            c_metric(initial_front, combined_front),
        ),
    ]

    summary = {
        "front_mode": args.front_mode,
        "objective": (
            "maximize [E, kappa/kappa_divisor, 1-vf]"
            if args.front_mode == "tri"
            else "maximize [E, kappa/kappa_divisor]"
        ),
        "vf_key": args.vf_key,
        "kappa_divisor": float(args.kappa_divisor),
        "raw_reference": raw_reference.tolist(),
        "normalization_min": obj_min.tolist(),
        "normalization_max": obj_max.tolist(),
        "initial_csv": str(initial_paths[0]),
        "combined_csvs": [str(path) for path in combined_paths],
        "table": table,
    }

    json_path = output_dir / f"{args.prefix}_hv_c_metrics.json"
    csv_path = output_dir / f"{args.prefix}_hv_c_table.csv"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(table[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table)

    print(json.dumps(summary, indent=2))
    print(f"[DONE] metrics_json: {json_path}")
    print(f"[DONE] metrics_csv: {csv_path}")


if __name__ == "__main__":
    main()
