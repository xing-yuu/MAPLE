from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


DEFAULT_INPUTS = [
    "./data/datasets/multiphysics/dataset_log_printability.csv",
    "./data/results/iterative/cumulative_source_round_01.csv",
    "./data/results/iterative/cumulative_source_round_02.csv",
    "./data/results/iterative/cumulative_source_round_03.csv",
    "./data/results/iterative/cumulative_source_round_04.csv",
    "./data/results/iterative/cumulative_source_round_05.csv",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Merge initial and iterative IHD dataset logs, compute Pareto fronts, "
            "and draw an E-kappa Pareto-front projection colored by volume fraction."
        )
    )
    p.add_argument("--inputs", type=str, nargs="+", default=DEFAULT_INPUTS)
    p.add_argument("--output_dir", type=str, default="./data/results/pareto_front_outputs")
    p.add_argument("--prefix", type=str, default="iterative_e_kappa_vf")
    p.add_argument("--front_mode", type=str, default="tri", choices=["tri", "ekappa"])
    p.add_argument("--vf_key", type=str, default="vf_target", help="Volume fraction used for coloring/front in tri mode.")
    p.add_argument("--e_key", type=str, default="stiffness")
    p.add_argument("--kappa_key", type=str, default="auto", help="auto, thermal, or kappa.")
    p.add_argument("--kappa_divisor", type=float, default=3.0, help="Divide raw kappa/thermal values by this factor before plotting/front computation.")
    p.add_argument("--feasible_only", action="store_true", help="Use only rows with printability_ok=True when that column is present.")
    p.add_argument("--show_initial", action="store_true", help="Draw initial dataset points with a distinct marker.")
    p.add_argument("--connect_2d_front", action="store_true", help="Connect Pareto points by E in the plotted 2D projection.")
    p.add_argument("--num_insets", type=int, default=0, help="Number of density thumbnails to place around the plot.")
    p.add_argument("--thumbnail", type=str, default="none", choices=["auto", "none"], help="auto uses rho_final.npy when reachable.")
    p.add_argument("--thumbnail_axis", type=int, default=2, choices=[0, 1, 2], help="Projection axis for rho thumbnails.")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--fig_width", type=float, default=7.2)
    p.add_argument("--fig_height", type=float, default=5.4)
    p.add_argument("--point_size", type=float, default=26.0)
    p.add_argument("--gray_size", type=float, default=12.0)
    p.add_argument("--alpha_all", type=float, default=0.16)
    return p.parse_args()


def read_rows(path: Path, stage: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fields = list(reader.fieldnames or [])
        rows = []
        for row in reader:
            item = dict(row)
            item["_source_file"] = str(path)
            item["_seen_stage"] = stage
            rows.append(item)
    return rows, fields


def infer_stage(path: Path, index: int) -> str:
    name = path.name
    match = re.search(r"round[_-]?(\d+)", name, flags=re.IGNORECASE)
    if match:
        return f"round_{int(match.group(1)):02d}"
    if "printability" in name.lower() or index == 0:
        return "initial"
    return path.stem


def first_float(row: Dict[str, str], keys: Iterable[str], default: float = float("nan")) -> float:
    for key in keys:
        value = str(row.get(key, "")).strip()
        if not value:
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return float(default)


def bool_value(text: str) -> Optional[bool]:
    value = str(text).strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    return None


def metric_values(row: Dict[str, str], args: argparse.Namespace) -> Tuple[float, float, float]:
    e = first_float(row, (args.e_key, "E", "youngs", "prop_stiffness"))
    if args.kappa_key == "auto":
        kappa = first_float(row, ("thermal", "kappa", "prop_thermal", "prop_kappa"))
    else:
        kappa = first_float(row, (args.kappa_key,))
    if math.isfinite(kappa) and args.kappa_divisor != 0:
        kappa = kappa / float(args.kappa_divisor)
    vf = first_float(row, (args.vf_key, "vf_final", "volume_fraction", "vf_binary"))
    return e, kappa, vf


def row_key(row: Dict[str, str], args: argparse.Namespace) -> str:
    for key in ("sample_id", "rho_final_npy", "rho_eval_path", "sample_dir", "folder"):
        value = str(row.get(key, "")).strip()
        if value:
            return f"{key}:{value}"
    e, kappa, vf = metric_values(row, args)
    seed = str(row.get("seed", "")).strip()
    return f"fallback:{seed}:{vf:.8g}:{e:.8g}:{kappa:.8g}"


def merge_rows(paths: Sequence[Path], args: argparse.Namespace) -> Tuple[List[Dict[str, str]], List[str]]:
    merged: Dict[str, Dict[str, str]] = {}
    fields: List[str] = []
    seen_fields = set()

    for idx, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(path)
        stage = infer_stage(path, idx)
        rows, file_fields = read_rows(path, stage)
        for field in [*file_fields, "_source_file", "_seen_stage", "first_seen_stage"]:
            if field not in seen_fields:
                fields.append(field)
                seen_fields.add(field)
        for row in rows:
            if args.feasible_only:
                flag = bool_value(row.get("printability_ok", ""))
                if flag is False:
                    continue
            key = row_key(row, args)
            if key not in merged:
                out = dict(row)
                out["first_seen_stage"] = stage
                merged[key] = out
            else:
                existing = merged[key]
                for field, value in row.items():
                    if str(existing.get(field, "")).strip() == "" and str(value).strip() != "":
                        existing[field] = value
    return list(merged.values()), fields


def finite_metrics(rows: Sequence[Dict[str, str]], args: argparse.Namespace) -> Tuple[List[Dict[str, str]], "object"]:
    import numpy as np

    valid = []
    values = []
    for row in rows:
        e, kappa, vf = metric_values(row, args)
        if math.isfinite(e) and math.isfinite(kappa) and math.isfinite(vf):
            row["_E_plot"] = f"{e:.10g}"
            row["_kappa_plot"] = f"{kappa:.10g}"
            row["_vf_plot"] = f"{vf:.10g}"
            valid.append(row)
            values.append((e, kappa, vf))
    return valid, np.asarray(values, dtype=float)


def pareto_mask(values: "object", mode: str) -> "object":
    import numpy as np

    if mode == "tri":
        objectives = np.column_stack([values[:, 0], values[:, 1], -values[:, 2]])
    else:
        objectives = values[:, :2]

    n = objectives.shape[0]
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if not keep[i]:
            continue
        p = objectives[i]
        dominates_i = np.all(objectives >= p - 1.0e-12, axis=1) & np.any(objectives > p + 1.0e-12, axis=1)
        dominates_i[i] = False
        if np.any(dominates_i):
            keep[i] = False
    return keep


def stage_sort_key(stage: str) -> Tuple[int, str]:
    if stage == "initial":
        return (0, stage)
    match = re.search(r"(\d+)", stage)
    if match:
        return (int(match.group(1)), stage)
    return (999, stage)


def write_csv(path: Path, rows: Sequence[Dict[str, str]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    final_fields = list(fields)
    for field in ("_E_plot", "_kappa_plot", "_vf_plot", "is_pareto"):
        if field not in final_fields:
            final_fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=final_fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in final_fields})


def resolve_path(raw: str, input_file: str, repo_root: Path) -> Optional[Path]:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    else:
        candidates.append(Path(input_file).parent / path)
        candidates.append(repo_root / path)

    parts = path.parts
    for marker in ("dataset-multiphysics", "iterative-e-kappa-vf-network-32", repo_root.name):
        if marker not in parts:
            continue
        idx = parts.index(marker)
        suffix = Path(*parts[idx + 1 :])
        if marker == repo_root.name:
            candidates.append(repo_root / suffix)
        else:
            candidates.append(repo_root / marker / suffix)
            # Public datasets live below data/datasets while older logs may
            # retain the original dataset-multiphysics marker.
            candidates.append(repo_root / "data" / "datasets" / "multiphysics" / suffix)

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_rho(row: Dict[str, str], repo_root: Path) -> Optional[Path]:
    input_file = row.get("_source_file", "")
    for key in ("rho_final_npy", "rho_eval_path", "rho_final"):
        found = resolve_path(row.get(key, ""), input_file, repo_root)
        if found is not None:
            return found
    sample_dir = resolve_path(row.get("sample_dir", ""), input_file, repo_root)
    if sample_dir is not None:
        candidate = sample_dir / "rho_final.npy"
        if candidate.exists():
            return candidate.resolve()
    return None


def thumbnail_from_rho(path: Path, axis: int) -> Optional["object"]:
    import numpy as np

    try:
        rho = np.asarray(np.load(path), dtype=float)
    except Exception:
        return None
    if rho.ndim == 1:
        res = round(rho.size ** (1.0 / 3.0))
        if res ** 3 != rho.size:
            return None
        rho = rho.reshape(res, res, res)
    if rho.ndim != 3:
        return None
    img = np.max(rho, axis=int(axis))
    img = img - float(np.nanmin(img))
    denom = float(np.nanmax(img))
    if denom > 0:
        img = img / denom
    rgb = np.zeros((*img.shape, 4), dtype=float)
    rgb[..., 0] = 0.52 * img + 0.20
    rgb[..., 1] = 0.72 * img + 0.24
    rgb[..., 2] = 0.68 * img + 0.24
    rgb[..., 3] = np.clip(0.18 + 0.82 * img, 0.0, 0.95)
    return np.rot90(rgb)


def choose_inset_rows(front_rows: Sequence[Dict[str, str]], args: argparse.Namespace, repo_root: Path) -> List[Dict[str, str]]:
    if args.thumbnail == "none" or args.num_insets <= 0:
        return []
    candidates = [row for row in front_rows if resolve_rho(row, repo_root) is not None]
    if not candidates:
        return []
    candidates = sorted(candidates, key=lambda r: float(r["_E_plot"]))
    if len(candidates) <= args.num_insets:
        return candidates
    picks = []
    for i in range(args.num_insets):
        idx = round(i * (len(candidates) - 1) / max(args.num_insets - 1, 1))
        picks.append(candidates[idx])
    return picks


def add_insets(ax: "object", front_rows: Sequence[Dict[str, str]], args: argparse.Namespace, repo_root: Path) -> int:
    from matplotlib.offsetbox import AnnotationBbox, OffsetImage

    positions = [
        (0.10, 0.88), (0.27, 0.95), (0.45, 0.95), (0.63, 0.95), (0.82, 0.94),
        (0.94, 0.68), (0.92, 0.47),
        (0.80, 0.08), (0.60, 0.08), (0.43, 0.08), (0.28, 0.08),
        (0.05, 0.62), (0.05, 0.42),
    ]
    count = 0
    for row, xybox in zip(choose_inset_rows(front_rows, args, repo_root), positions):
        rho_path = resolve_rho(row, repo_root)
        if rho_path is None:
            continue
        img = thumbnail_from_rho(rho_path, args.thumbnail_axis)
        if img is None:
            continue
        xy = (float(row["_E_plot"]), float(row["_kappa_plot"]))
        imagebox = OffsetImage(img, zoom=0.22)
        ab = AnnotationBbox(
            imagebox,
            xy=xy,
            xybox=xybox,
            xycoords="data",
            boxcoords=ax.transAxes,
            frameon=False,
            arrowprops=dict(arrowstyle="-", linestyle="--", linewidth=0.8, color="#1d5f86", alpha=0.9),
            pad=0.0,
        )
        ax.add_artist(ab)
        count += 1
    return count


def draw_plot(rows: Sequence[Dict[str, str]], values: "object", mask: "object", args: argparse.Namespace, output_dir: Path) -> Dict[str, object]:
    import matplotlib.pyplot as plt
    import numpy as np

    repo_root = Path(__file__).resolve().parents[3]
    front_rows = [row for row, keep in zip(rows, mask.tolist()) if keep]
    all_e = values[:, 0]
    all_k = values[:, 1]
    all_vf = values[:, 2]
    front_values = values[mask]

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "mathtext.fontset": "dejavuserif",
            "axes.linewidth": 0.8,
        }
    )
    fig, ax = plt.subplots(figsize=(args.fig_width, args.fig_height), constrained_layout=True)

    ax.scatter(all_e, all_k, s=args.gray_size, c="0.70", alpha=args.alpha_all, edgecolors="none", label="all candidates")

    if args.show_initial:
        initial_idx = [idx for idx, row in enumerate(rows) if row.get("first_seen_stage") == "initial"]
        if initial_idx:
            ax.scatter(all_e[initial_idx], all_k[initial_idx], s=12, c="0.35", alpha=0.20, marker="x", linewidths=0.4, label="initial")

    scatter = ax.scatter(
        front_values[:, 0],
        front_values[:, 1],
        s=args.point_size,
        c=front_values[:, 2],
        cmap="viridis_r",
        edgecolors="white",
        linewidths=0.25,
        zorder=5,
        label="non-dominated",
    )

    if args.connect_2d_front:
        order = np.argsort(front_values[:, 0])
        ax.plot(front_values[order, 0], front_values[order, 1], color="#1d5f86", linewidth=0.9, alpha=0.9, zorder=4)

    inset_count = add_insets(ax, front_rows, args, repo_root)

    ax.set_xlabel(r"$\bar{E}$", fontsize=14, loc="right")
    ax.set_ylabel(r"$\bar{\kappa}$", fontsize=14, rotation=0, loc="top")
    ax.grid(True, linestyle="--", linewidth=0.45, alpha=0.25)
    ax.set_axisbelow(True)
    xpad = max((float(np.nanmax(all_e)) - float(np.nanmin(all_e))) * 0.06, 1.0e-3)
    ypad = max((float(np.nanmax(all_k)) - float(np.nanmin(all_k))) * 0.06, 1.0e-3)
    ax.set_xlim(float(np.nanmin(all_e)) - xpad, float(np.nanmax(all_e)) + xpad)
    ax.set_ylim(float(np.nanmin(all_k)) - ypad, float(np.nanmax(all_k)) + ypad)
    cbar = fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.015)
    cbar.set_label(r"$v_f$", fontsize=14)

    title = "Pareto front projection: maximize E, maximize kappa"
    if args.front_mode == "tri":
        title += ", minimize vf"
    ax.set_title(title, fontsize=11)

    png = output_dir / f"{args.prefix}_pareto_front.png"
    pdf = output_dir / f"{args.prefix}_pareto_front.pdf"
    fig.savefig(png, dpi=int(args.dpi), bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf), "inset_count": inset_count}


def stage_stats(rows: Sequence[Dict[str, str]], mask: "object") -> Dict[str, Dict[str, int]]:
    stats: Dict[str, Dict[str, int]] = {}
    for row, keep in zip(rows, mask.tolist()):
        stage = row.get("first_seen_stage", "unknown") or "unknown"
        stats.setdefault(stage, {"count": 0, "pareto_count": 0})
        stats[stage]["count"] += 1
        if keep:
            stats[stage]["pareto_count"] += 1
    return {key: stats[key] for key in sorted(stats, key=stage_sort_key)}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    input_paths = [Path(path).expanduser().resolve() for path in args.inputs]

    rows, fields = merge_rows(input_paths, args)
    rows, values = finite_metrics(rows, args)
    mask = pareto_mask(values, args.front_mode)
    for row, keep in zip(rows, mask.tolist()):
        row["is_pareto"] = "True" if keep else "False"

    combined_csv = output_dir / f"{args.prefix}_combined_unique.csv"
    pareto_csv = output_dir / f"{args.prefix}_pareto_front.csv"
    write_csv(combined_csv, rows, fields)
    write_csv(pareto_csv, [row for row in rows if row.get("is_pareto") == "True"], fields)
    plot_outputs = draw_plot(rows, values, mask, args, output_dir)

    import numpy as np

    summary = {
        "inputs": [str(path) for path in input_paths],
        "front_mode": args.front_mode,
        "kappa_divisor": float(args.kappa_divisor),
        "unique_rows": len(rows),
        "pareto_rows": int(np.sum(mask)),
        "E_min": float(np.nanmin(values[:, 0])),
        "E_max": float(np.nanmax(values[:, 0])),
        "kappa_min": float(np.nanmin(values[:, 1])),
        "kappa_max": float(np.nanmax(values[:, 1])),
        "vf_min": float(np.nanmin(values[:, 2])),
        "vf_max": float(np.nanmax(values[:, 2])),
        "stage_stats": stage_stats(rows, mask),
        "outputs": {
            "combined_csv": str(combined_csv),
            "pareto_csv": str(pareto_csv),
            **plot_outputs,
        },
    }
    summary_json = output_dir / f"{args.prefix}_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[DONE] unique rows: {summary['unique_rows']}")
    print(f"[DONE] pareto rows: {summary['pareto_rows']}")
    print(f"[DONE] figure: {plot_outputs['png']}")
    print(f"[DONE] combined_csv: {combined_csv}")
    print(f"[DONE] pareto_csv: {pareto_csv}")
    print(f"[DONE] summary_json: {summary_json}")


if __name__ == "__main__":
    main()
