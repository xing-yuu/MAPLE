from __future__ import annotations

import argparse
import csv
import os
from pathlib import PurePosixPath
from typing import Dict, Iterable, List, Optional


PATH_COLUMNS = (
    "sample_dir",
    "rho_final_npy",
    "final_obj",
    "rho_init_32_npy",
    "rho_net_32_npy",
    "rho_warmstart_64_npy",
    "rho_interp_64_npy",
    "source_rho_final_npy",
    "interp_lower_rho_npy",
    "interp_upper_rho_npy",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Rewrite absolute paths in dataset_log.csv after moving a dataset to a new server/path. "
            "The usual use is mapping an old dataset root to a new dataset root."
        )
    )
    p.add_argument("--input_csv", type=str, required=True, help="Original dataset_log.csv.")
    p.add_argument("--output_csv", type=str, required=True, help="Rewritten dataset_log.csv.")
    p.add_argument(
        "--old_root",
        type=str,
        default=None,
        help="Old dataset root. If omitted, it is inferred from the first non-empty sample_dir.",
    )
    p.add_argument(
        "--new_root",
        type=str,
        required=True,
        help="New dataset root, e.g. /public/home/lvlin001/project/IHD-0422/dataset-weight-sweep-v1.",
    )
    p.add_argument(
        "--check_exists",
        action="store_true",
        help="Print how many rewritten rho_final_npy/final_obj paths exist on the current machine.",
    )
    return p.parse_args()


def norm_posix(path: str) -> str:
    return str(PurePosixPath(path.strip()))


def infer_old_root(rows: List[Dict[str, str]]) -> str:
    for row in rows:
        sample_dir = row.get("sample_dir", "").strip()
        weight_idx = row.get("weight_idx", "").strip()
        seed = row.get("seed", "").strip()
        if not sample_dir or not weight_idx or not seed:
            continue
        sample = norm_posix(sample_dir)
        w_part = f"w_{int(float(weight_idx)):03d}"
        seed_part = f"seed_{int(float(seed)):010d}"
        suffix = f"/{w_part}/{seed_part}"
        if sample.endswith(suffix):
            return sample[: -len(suffix)]
    raise ValueError("Cannot infer old root. Pass --old_root explicitly.")


def rewrite_path(value: str, old_root: str, new_root: str) -> str:
    if not value or not value.strip():
        return value
    path = norm_posix(value)
    old = norm_posix(old_root).rstrip("/")
    new = norm_posix(new_root).rstrip("/")
    if path == old:
        return new
    if path.startswith(old + "/"):
        return new + path[len(old) :]
    return value


def rewrite_row(row: Dict[str, str], fieldnames: Iterable[str], old_root: str, new_root: str) -> Dict[str, str]:
    out = dict(row)
    fields = set(fieldnames)
    for col in PATH_COLUMNS:
        if col in fields:
            out[col] = rewrite_path(out.get(col, ""), old_root, new_root)
    return out


def count_existing(rows: List[Dict[str, str]]) -> None:
    for col in ("rho_final_npy", "final_obj"):
        total = sum(1 for row in rows if row.get(col, "").strip())
        exists = sum(1 for row in rows if row.get(col, "").strip() and os.path.exists(row[col]))
        print(f"[CHECK] {col}: {exists}/{total} exist")


def main() -> None:
    args = parse_args()
    with open(args.input_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames: Optional[List[str]] = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"CSV has no header: {args.input_csv}")
        rows = list(reader)

    old_root = norm_posix(args.old_root) if args.old_root else infer_old_root(rows)
    new_root = norm_posix(args.new_root)
    fixed = [rewrite_row(row, fieldnames, old_root, new_root) for row in rows]

    out_dir = os.path.dirname(os.path.abspath(args.output_csv))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed)

    print(f"[OK] rows: {len(fixed)}")
    print(f"[OK] old_root: {old_root}")
    print(f"[OK] new_root: {new_root}")
    print(f"[OK] output_csv: {os.path.abspath(args.output_csv)}")
    if args.check_exists:
        count_existing(fixed)


if __name__ == "__main__":
    main()
