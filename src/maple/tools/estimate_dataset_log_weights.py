from __future__ import annotations

import argparse
import collections
import csv
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Estimate missing E/kappa objective weights for a dataset_log.csv. "
            "The estimate is based on local E-kappa Pareto-front slopes within each volume-fraction group."
        )
    )
    p.add_argument("--input_csv", type=str, required=True)
    p.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="Output CSV. If omitted, rewrite --input_csv in place.",
    )
    p.add_argument(
        "--backup",
        type=str,
        default=None,
        help="Optional backup path before in-place rewrite. Defaults to <input>.before_weight_estimation.csv.",
    )
    p.add_argument("--min_weight", type=float, default=0.02)
    p.add_argument("--max_weight", type=float, default=0.98)
    return p.parse_args()


def get_float(row: Dict[str, str], keys: Iterable[str], default: float = 0.0) -> float:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return float(default)


def e_value(row: Dict[str, str]) -> float:
    return get_float(row, ("stiffness", "E", "youngs", "prop_stiffness"))


def kappa_value(row: Dict[str, str]) -> float:
    return get_float(row, ("thermal", "kappa", "prop_thermal", "prop_kappa"))


def vf_group_key(row: Dict[str, str]) -> str:
    value = str(row.get("vf_percent", "")).strip()
    if value:
        return value
    return f"{get_float(row, ('vf_target', 'vf_final')):.8f}"


def pareto_2d(rows: Sequence[Dict[str, str]], indices: Sequence[int]) -> List[int]:
    out: List[int] = []
    for i in indices:
        e = e_value(rows[i])
        k = kappa_value(rows[i])
        dominated = False
        for j in indices:
            if i == j:
                continue
            ej = e_value(rows[j])
            kj = kappa_value(rows[j])
            if ej >= e - 1.0e-12 and kj >= k - 1.0e-12 and (ej > e + 1.0e-12 or kj > k + 1.0e-12):
                dominated = True
                break
        if not dominated:
            out.append(i)
    return sorted(out, key=lambda idx: (e_value(rows[idx]), kappa_value(rows[idx]), int(float(rows[idx].get("seed", idx) or idx))))


def estimate_weights(
    rows: Sequence[Dict[str, str]],
    *,
    min_weight: float,
    max_weight: float,
) -> Tuple[List[Tuple[float, float]], List[str], Dict[str, int]]:
    groups: Dict[str, List[int]] = collections.defaultdict(list)
    for idx, row in enumerate(rows):
        groups[vf_group_key(row)].append(idx)

    estimated: List[Tuple[float, float]] = [(0.5, 0.5) for _ in rows]
    methods: List[str] = ["fallback_0.5" for _ in rows]
    front_sizes: Dict[str, int] = {}

    lo = float(min_weight)
    hi = float(max_weight)
    if not (0.0 <= lo <= hi <= 1.0):
        raise ValueError(f"Expected 0 <= min_weight <= max_weight <= 1, got {lo}, {hi}")

    for group_key, indices in groups.items():
        front = pareto_2d(rows, indices)
        front_sizes[group_key] = len(front)
        if len(front) < 2:
            for idx in indices:
                estimated[idx] = (0.5, 0.5)
                methods[idx] = "fallback_0.5_front_lt2"
            continue

        e_vals = [e_value(rows[idx]) for idx in indices]
        k_vals = [kappa_value(rows[idx]) for idx in indices]
        e_min, e_max = min(e_vals), max(e_vals)
        k_min, k_max = min(k_vals), max(k_vals)
        e_span = max(e_max - e_min, 1.0e-12)
        k_span = max(k_max - k_min, 1.0e-12)

        e_front = [(e_value(rows[idx]) - e_min) / e_span for idx in front]
        k_front = [(kappa_value(rows[idx]) - k_min) / k_span for idx in front]

        front_weights: Dict[int, Tuple[float, float]] = {}
        for pos, idx in enumerate(front):
            if len(front) == 2:
                a, b = 0, 1
            elif pos == 0:
                a, b = 0, 1
            elif pos == len(front) - 1:
                a, b = len(front) - 2, len(front) - 1
            else:
                a, b = pos - 1, pos + 1

            dx = e_front[b] - e_front[a]
            dy = k_front[b] - k_front[a]
            if abs(dx) < 1.0e-9:
                w_e = 0.5
            else:
                slope = abs(dy / dx)
                w_e = slope / (1.0 + slope)
            w_e = max(lo, min(hi, float(w_e)))
            front_weights[idx] = (w_e, 1.0 - w_e)

        for idx in indices:
            e_norm = (e_value(rows[idx]) - e_min) / e_span
            k_norm = (kappa_value(rows[idx]) - k_min) / k_span
            nearest = min(
                front,
                key=lambda j: ((e_value(rows[j]) - e_min) / e_span - e_norm) ** 2
                + ((kappa_value(rows[j]) - k_min) / k_span - k_norm) ** 2,
            )
            estimated[idx] = front_weights[nearest]
            methods[idx] = "local_vf_pareto_slope"

    return estimated, methods, front_sizes


def insert_field(fields: List[str], name: str) -> None:
    if name in fields:
        return
    if name == "w_stiffness" and "seed" in fields:
        fields.insert(fields.index("seed"), name)
        return
    if name in ("w_shear", "w_thermal", "objective", "weight_estimation_method") and "w_stiffness" in fields:
        anchors = [x for x in ("w_stiffness", "w_shear", "w_thermal", "objective") if x in fields]
        fields.insert(max(fields.index(x) for x in anchors) + 1, name)
        return
    fields.append(name)


def main() -> None:
    args = parse_args()
    input_csv = Path(args.input_csv).resolve()
    output_csv = Path(args.output_csv).resolve() if args.output_csv else input_csv

    with input_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames or [])
    if not fields:
        raise ValueError(f"CSV has no header: {input_csv}")

    if output_csv == input_csv:
        backup = Path(args.backup).resolve() if args.backup else input_csv.with_name(input_csv.stem + ".before_weight_estimation.csv")
        if not backup.exists():
            shutil.copyfile(input_csv, backup)
        print(f"[BACKUP] {backup}")

    weights, methods, front_sizes = estimate_weights(
        rows,
        min_weight=float(args.min_weight),
        max_weight=float(args.max_weight),
    )

    for field in ("w_stiffness", "w_shear", "w_thermal", "objective", "weight_estimation_method"):
        insert_field(fields, field)

    for row, (w_e, w_kappa), method in zip(rows, weights, methods):
        row["w_stiffness"] = f"{w_e:.8f}"
        row["w_shear"] = "0.00000000"
        row["w_thermal"] = f"{w_kappa:.8f}"
        row["objective"] = f"stiffness:{w_e:.8f},thermal:{w_kappa:.8f}"
        row["weight_estimation_method"] = method

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field, "") for field in fields} for row in rows)

    w_e_values = [w[0] for w in weights]
    print(f"[DONE] output={output_csv}")
    print(f"[DONE] rows={len(rows)}")
    print(f"[DONE] w_stiffness min={min(w_e_values):.8f} max={max(w_e_values):.8f} mean={sum(w_e_values) / max(len(w_e_values), 1):.8f}")
    print(f"[DONE] front_size min={min(front_sizes.values()) if front_sizes else 0} max={max(front_sizes.values()) if front_sizes else 0}")


if __name__ == "__main__":
    main()
