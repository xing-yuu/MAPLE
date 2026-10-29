from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import os
import random
import traceback
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch

from maple.core.main_IHD_v3 import (
    build_unitcell_batch,
    evaluate_homogenized_terms,
    optimize_inverse_homogenization,
    setup_seed,
)
from maple.core.stage2.density_generator import Stage2DensityGenerator
from maple.core.utils.model import IHD


CONDITION_COLUMNS = [
    "w_stiffness",
    "w_shear",
    "w_thermal",
    "vf_target",
    "stiffness",
    "shear",
    "thermal",
]

FIELDNAMES = [
    "sample_id",
    "front_objectives",
    "weight_tag",
    "weight_idx",
    "w_stiffness",
    "w_shear",
    "w_thermal",
    "objective",
    "vf_target",
    "vf_final",
    "seed",
    "sampling_mode",
    "front_gap",
    "front_interval_rank",
    "tri_front_edge",
    "condition_npy",
    "condition_stiffness",
    "condition_shear",
    "condition_thermal",
    "condition_kappa",
    "source_lower_w",
    "source_upper_w",
    "source_alpha",
    "source_lower_seed",
    "source_upper_seed",
    "source_lower_vf",
    "source_upper_vf",
    "source_lower_stiffness",
    "source_lower_shear",
    "source_lower_thermal",
    "source_lower_kappa",
    "source_upper_stiffness",
    "source_upper_shear",
    "source_upper_thermal",
    "source_upper_kappa",
    "rho_net_32_npy",
    "rho_warmstart_32_npy",
    "stiffness",
    "shear",
    "thermal",
    "kappa",
    "sample_dir",
    "rho_final_npy",
    "final_obj",
    "checkpoint",
    "source_csv",
]


@dataclass(frozen=True)
class SourceSample:
    weight: float
    w_stiffness: float
    w_shear: float
    w_thermal: float
    vf: float
    stiffness: float
    shear: float
    thermal: float
    seed: int
    row_index: int
    explicit_weight: bool = False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Use a property-conditioned stage-II network at 32^3. The script interpolates "
            "weights and homogenized properties on a source Pareto front, feeds "
            "[w_stiffness,w_shear,w_thermal,vf,stiffness,shear,thermal] to the network, "
            "then continues topology optimization from the generated density field."
        )
    )
    p.add_argument("--source_csv", type=str, required=True, help="Pareto/front dataset_log.csv used for condition interpolation.")
    p.add_argument("--checkpoint", type=str, required=True, help="Condition-only stage-II generator checkpoint.")
    p.add_argument("--output_root", type=str, default="./data/results/pareto-property-network-32")
    p.add_argument("--resolution", type=int, default=32, help="Topology-optimization resolution. This script requires 32.")
    p.add_argument("--volfrac", type=float, default=None, help="Optional fixed vf override. If omitted, vf is interpolated from source_csv.")
    p.add_argument("--w_start", type=float, default=0.0, help="Start stiffness weight.")
    p.add_argument("--w_end", type=float, default=1.0, help="End stiffness weight, inclusive.")
    p.add_argument("--w_step", type=float, default=0.005, help="Target stiffness-weight spacing.")
    p.add_argument("--seeds_per_w", type=int, default=10, help="Number of optimization runs per target weight.")
    p.add_argument("--max_samples", type=int, default=0, help="Maximum pending samples to generate; 0 means no limit.")
    p.add_argument(
        "--front_objectives",
        type=str,
        default="auto",
        choices=["auto", "weighted", "e_kappa_vf"],
        help=(
            "Front geometry to densify. weighted keeps the old one-dimensional weight sweep; "
            "e_kappa_vf treats the front as maximize E, maximize kappa, minimize volume fraction."
        ),
    )
    p.add_argument(
        "--sampling_mode",
        type=str,
        default="uniform",
        choices=["uniform", "adaptive_front"],
        help="uniform sweeps w; adaptive_front allocates samples to sparse segments of the current Pareto front.",
    )
    p.add_argument(
        "--adaptive_min_gap_quantile",
        type=float,
        default=0.50,
        help="Adaptive mode skips front segments below this gap quantile. 0 uses all segments; 0.5 skips the densest half.",
    )
    p.add_argument(
        "--adaptive_gap_power",
        type=float,
        default=1.0,
        help="Power applied to normalized front-segment gaps before allocating sample counts.",
    )
    p.add_argument("--adaptive_front_eps", type=float, default=1.0e-9, help="Dominance tolerance for extracting the Pareto front.")
    p.add_argument(
        "--tri_front_k",
        type=int,
        default=6,
        help="Number of nearest Pareto-front neighbors used to form sparse edges in e_kappa_vf mode.",
    )
    p.add_argument(
        "--vf_objective",
        type=str,
        default="minimize",
        choices=["minimize", "maximize"],
        help="Pareto sense for volume fraction in e_kappa_vf mode. Usually minimize.",
    )
    p.add_argument(
        "--tri_weight_mode",
        type=str,
        default="source",
        choices=["cycle", "random", "source"],
        help=(
            "How e_kappa_vf mode chooses physical objective weights. "
            "source follows the paper and interpolates source-row weights; cycle/random are explicit ablations."
        ),
    )
    p.add_argument(
        "--tri_weight_values",
        type=str,
        default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated w_E values used by --tri_weight_mode cycle/random. w_kappa is 1-w_E.",
    )
    p.add_argument("--default_w_stiffness", type=float, default=0.5, help="Fallback E objective weight for source rows without weights.")
    p.add_argument("--default_w_shear", type=float, default=0.0, help="Fallback shear objective weight for source rows without weights.")
    p.add_argument("--default_w_thermal", type=float, default=0.5, help="Fallback kappa objective weight for source rows without weights.")
    p.add_argument("--master_seed", type=int, default=20260527)
    p.add_argument("--seed_upper", type=int, default=2_147_483_647)
    p.add_argument("--exclude_exact", action="store_true", help="Force interpolation from neighboring weights even if target weight exists.")
    p.add_argument("--iters", type=int, default=2000)
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
    p.add_argument("--cavity_focus_center", dest="cavity_focus_center", action="store_true")
    p.add_argument("--no-cavity_focus_center", dest="cavity_focus_center", action="store_false")
    p.set_defaults(cavity_focus_center=True)
    p.add_argument("--gmg_smooth_iter", type=int, nargs="+", default=[2, 2, 3, 4, 5])
    p.add_argument("--gmg_max_cycle", type=int, default=1)
    p.add_argument("--mesh_res", type=int, default=128)
    p.add_argument("--mesh_iso", type=float, default=0.3)
    p.add_argument("--mesh_pad", type=int, default=2)
    p.add_argument("--warm_sigma", type=float, default=0.0, help="Optional Gaussian perturbation after network output.")
    p.add_argument("--warm_smooth_passes", type=int, default=0, help="Optional periodic binomial smoothing after network output.")
    p.add_argument("--volume_iters", type=int, default=8, help="Mean correction iterations for warm-start density.")
    p.add_argument("--no-match_volume", dest="match_volume", action="store_false", help="Do not correct generated mean density to target vf.")
    p.set_defaults(match_volume=True)
    p.add_argument("--optimizer_verbose", action="store_true")
    p.add_argument("--save_iter_log", action="store_true")
    p.add_argument("--gpus", type=str, default="auto", help="'auto', 'cpu', or comma-separated GPU ids.")
    p.add_argument("--workers_per_gpu", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()


def as_float(row: Dict[str, str], keys: Tuple[str, ...], default: float) -> float:
    for key in keys:
        value = row.get(key)
        if value is None or str(value).strip() == "":
            continue
        try:
            return float(value)
        except ValueError:
            continue
    return float(default)


def row_weight(row: Dict[str, str]) -> float:
    if str(row.get("w_stiffness", "")).strip():
        return float(row["w_stiffness"])
    if str(row.get("weight_idx", "")).strip():
        raw = float(row["weight_idx"])
        return raw / 10000.0 if raw > 100.0 else raw / 100.0
    obj = str(row.get("objective", ""))
    for part in obj.replace("+", ",").split(","):
        if "stiffness" not in part.lower():
            continue
        token = part.split(":")[-1].replace("*stiffness", "").strip()
        try:
            return float(token)
        except ValueError:
            pass
    raise ValueError(f"Cannot infer stiffness weight from row: {row}")


def row_has_weight_info(row: Dict[str, str]) -> bool:
    if any(str(row.get(key, "")).strip() for key in ("w_stiffness", "w_shear", "w_thermal", "w_kappa", "weight_idx")):
        return True
    objective = str(row.get("objective", "")).lower()
    return any(name in objective for name in ("stiffness", "thermal", "kappa", "shear"))


def row_weights(row: Dict[str, str], args: Optional[argparse.Namespace] = None) -> Tuple[float, float, float]:
    default_ws = float(getattr(args, "default_w_stiffness", 0.5))
    default_wsh = float(getattr(args, "default_w_shear", 0.0))
    default_wt = float(getattr(args, "default_w_thermal", 0.5))
    try:
        ws = row_weight(row)
    except ValueError:
        ws = default_ws
    wsh = as_float(row, ("w_shear", "weight_shear"), default_wsh)
    wt = as_float(row, ("w_thermal", "w_kappa", "weight_thermal", "weight_kappa"), default_wt)
    total = ws + wsh + wt
    if total > 0.0:
        ws, wsh, wt = ws / total, wsh / total, wt / total
    return float(ws), float(wsh), float(wt)


def row_vf(row: Dict[str, str]) -> float:
    return as_float(row, ("vf_target", "volfrac_target", "vf_final"), 0.25)


def row_thermal(row: Dict[str, str]) -> float:
    return as_float(row, ("thermal", "kappa", "thermal_final", "kappa_final", "prop_thermal", "prop_kappa"), 0.0)


def load_source_groups(source_csv: str, args: Optional[argparse.Namespace] = None) -> Dict[float, List[SourceSample]]:
    groups: Dict[float, List[SourceSample]] = {}
    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            try:
                ws, wsh, wt = row_weights(row, args)
            except Exception:
                continue
            weight = round(ws, 10)
            seed = int(float(row.get("seed", row_index) or row_index))
            sample = SourceSample(
                weight=weight,
                w_stiffness=ws,
                w_shear=wsh,
                w_thermal=wt,
                vf=row_vf(row),
                stiffness=as_float(row, ("stiffness", "stiffness_final", "prop_stiffness"), 0.0),
                shear=as_float(row, ("shear", "shear_final", "prop_shear"), 0.0),
                thermal=row_thermal(row),
                seed=seed,
                row_index=row_index,
                explicit_weight=row_has_weight_info(row),
            )
            groups.setdefault(weight, []).append(sample)

    if not groups:
        raise ValueError(f"No usable source rows found in {source_csv}")
    for samples in groups.values():
        samples.sort(key=lambda x: (x.seed, x.row_index))
    return groups


def load_source_samples(source_csv: str, args: argparse.Namespace) -> List[SourceSample]:
    samples: List[SourceSample] = []
    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row_index, row in enumerate(reader):
            try:
                ws, wsh, wt = row_weights(row, args)
                seed = int(float(row.get("seed", row_index) or row_index))
                explicit_weight = row_has_weight_info(row)
                if str(getattr(args, "tri_weight_mode", "source")) == "source" and not explicit_weight:
                    continue
                sample = SourceSample(
                    weight=round(ws, 10),
                    w_stiffness=ws,
                    w_shear=wsh,
                    w_thermal=wt,
                    vf=row_vf(row),
                    stiffness=as_float(row, ("stiffness", "stiffness_final", "prop_stiffness"), 0.0),
                    shear=as_float(row, ("shear", "shear_final", "prop_shear"), 0.0),
                    thermal=row_thermal(row),
                    seed=seed,
                    row_index=row_index,
                    explicit_weight=explicit_weight,
                )
            except Exception:
                continue
            samples.append(sample)
    if not samples:
        if str(getattr(args, "tri_weight_mode", "source")) == "source":
            raise ValueError(
                "No usable source rows with explicit weights found. The paper-style e_kappa_vf mode "
                "requires source CSV columns such as w_stiffness/w_thermal or objective. "
                "Regenerate/repair dataset_log.csv with weights, or explicitly use --tri_weight_mode cycle/random as an ablation."
            )
        raise ValueError(f"No usable source rows found in {source_csv}")
    samples.sort(key=lambda x: (x.vf, x.stiffness, x.thermal, x.seed, x.row_index))
    return samples


def all_source_samples(source_groups: Dict[float, List[SourceSample]]) -> List[SourceSample]:
    samples: List[SourceSample] = []
    for group in source_groups.values():
        samples.extend(group)
    return samples


def pareto_front_samples(samples: List[SourceSample], eps: float) -> List[SourceSample]:
    candidates = [
        s
        for s in samples
        if np.isfinite(s.stiffness) and np.isfinite(s.thermal)
    ]
    if len(candidates) < 2:
        return candidates

    # Maximization in (stiffness, thermal). Scan from high stiffness to low
    # stiffness and keep only points that improve thermal conductivity.
    candidates.sort(key=lambda s: (-s.stiffness, -s.thermal, s.row_index))
    front_desc: List[SourceSample] = []
    best_thermal = -float("inf")
    for sample in candidates:
        if sample.thermal > best_thermal + float(eps):
            front_desc.append(sample)
            best_thermal = sample.thermal

    front = sorted(front_desc, key=lambda s: (s.stiffness, s.thermal, s.row_index))
    deduped: List[SourceSample] = []
    for sample in front:
        if deduped and abs(sample.stiffness - deduped[-1].stiffness) <= float(eps):
            if sample.thermal > deduped[-1].thermal:
                deduped[-1] = sample
        else:
            deduped.append(sample)
    return deduped


def adaptive_front_intervals(
    front: List[SourceSample],
    *,
    min_gap_quantile: float,
    gap_power: float,
) -> List[Tuple[int, SourceSample, SourceSample, float]]:
    if len(front) < 2:
        return []

    stiffness_values = np.array([s.stiffness for s in front], dtype=np.float64)
    thermal_values = np.array([s.thermal for s in front], dtype=np.float64)
    sx = max(float(stiffness_values.max() - stiffness_values.min()), 1.0e-12)
    sy = max(float(thermal_values.max() - thermal_values.min()), 1.0e-12)

    intervals: List[Tuple[int, SourceSample, SourceSample, float]] = []
    gaps: List[float] = []
    for idx, (lower, upper) in enumerate(zip(front[:-1], front[1:])):
        dx = (upper.stiffness - lower.stiffness) / sx
        dy = (upper.thermal - lower.thermal) / sy
        gap = float(np.sqrt(dx * dx + dy * dy))
        intervals.append((idx, lower, upper, gap))
        gaps.append(gap)

    if not intervals:
        return []

    q = float(np.clip(min_gap_quantile, 0.0, 1.0))
    threshold = float(np.quantile(np.asarray(gaps, dtype=np.float64), q)) if q > 0.0 else 0.0
    selected = [item for item in intervals if item[3] >= threshold and item[3] > 0.0]
    if not selected:
        selected = [max(intervals, key=lambda item: item[3])]

    power = max(float(gap_power), 0.0)
    if power != 1.0:
        selected = [(idx, lo, hi, float(gap ** power)) for idx, lo, hi, gap in selected]
    return selected


def tri_objective_points(samples: List[SourceSample], vf_objective: str) -> np.ndarray:
    vf_sign = -1.0 if str(vf_objective) == "minimize" else 1.0
    return np.asarray(
        [[s.stiffness, s.thermal, vf_sign * s.vf] for s in samples],
        dtype=np.float64,
    )


def pareto_front_samples_e_kappa_vf(
    samples: List[SourceSample],
    *,
    eps: float,
    vf_objective: str,
) -> List[SourceSample]:
    candidates = [
        s
        for s in samples
        if np.isfinite(s.stiffness) and np.isfinite(s.thermal) and np.isfinite(s.vf)
    ]
    if len(candidates) < 2:
        return candidates

    points = tri_objective_points(candidates, vf_objective)
    keep: List[SourceSample] = []
    tol = float(eps)
    for idx, sample in enumerate(candidates):
        p = points[idx]
        dominates = np.all(points >= p - tol, axis=1) & np.any(points > p + tol, axis=1)
        dominates[idx] = False
        if not bool(np.any(dominates)):
            keep.append(sample)

    keep.sort(key=lambda s: (s.vf, s.stiffness, s.thermal, s.seed, s.row_index))
    return keep


def normalized_tri_points(samples: List[SourceSample], vf_objective: str) -> np.ndarray:
    points = tri_objective_points(samples, vf_objective)
    mins = points.min(axis=0)
    spans = points.max(axis=0) - mins
    spans[spans <= 1.0e-12] = 1.0
    return (points - mins) / spans


def adaptive_tri_front_edges(
    front: List[SourceSample],
    *,
    k_neighbors: int,
    min_gap_quantile: float,
    gap_power: float,
    vf_objective: str,
) -> List[Tuple[int, SourceSample, SourceSample, float]]:
    if len(front) < 2:
        return []

    pts = normalized_tri_points(front, vf_objective)
    k = max(1, min(int(k_neighbors), len(front) - 1))
    seen: Set[Tuple[int, int]] = set()
    edges: List[Tuple[int, SourceSample, SourceSample, float]] = []
    for idx in range(len(front)):
        dists = np.linalg.norm(pts - pts[idx], axis=1)
        order = np.argsort(dists)
        for nbr in order[1 : k + 1]:
            a, b = sorted((idx, int(nbr)))
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            gap = float(np.linalg.norm(pts[a] - pts[b]))
            if gap > 0.0:
                edges.append((len(edges), front[a], front[b], gap))

    if not edges:
        return []

    gaps = np.asarray([edge[3] for edge in edges], dtype=np.float64)
    q = float(np.clip(min_gap_quantile, 0.0, 1.0))
    threshold = float(np.quantile(gaps, q)) if q > 0.0 else 0.0
    selected = [edge for edge in edges if edge[3] >= threshold and edge[3] > 0.0]
    if not selected:
        selected = [max(edges, key=lambda item: item[3])]

    power = max(float(gap_power), 0.0)
    if power != 1.0:
        selected = [(idx, lo, hi, float(gap ** power)) for idx, lo, hi, gap in selected]
    return selected


def adaptive_tri_front_neighborhoods(
    front: List[SourceSample],
    *,
    k_neighbors: int,
    min_gap_quantile: float,
    gap_power: float,
    vf_objective: str,
) -> List[Tuple[int, List[SourceSample], float]]:
    if len(front) < 2:
        return []

    pts = normalized_tri_points(front, vf_objective)
    k = max(1, min(int(k_neighbors), len(front) - 1))
    neighborhoods: List[Tuple[int, List[SourceSample], float]] = []
    gaps: List[float] = []
    for idx in range(len(front)):
        dists = np.linalg.norm(pts - pts[idx], axis=1)
        order = np.argsort(dists)
        neighbor_ids = [int(j) for j in order[1 : k + 1]]
        group = [front[idx], *(front[j] for j in neighbor_ids)]
        gap = float(np.mean([dists[j] for j in neighbor_ids])) if neighbor_ids else 0.0
        if gap > 0.0:
            neighborhoods.append((idx, group, gap))
            gaps.append(gap)

    if not neighborhoods:
        return []

    q = float(np.clip(min_gap_quantile, 0.0, 1.0))
    threshold = float(np.quantile(np.asarray(gaps, dtype=np.float64), q)) if q > 0.0 else 0.0
    selected = [item for item in neighborhoods if item[2] >= threshold and item[2] > 0.0]
    if not selected:
        selected = [max(neighborhoods, key=lambda item: item[2])]

    power = max(float(gap_power), 0.0)
    if power != 1.0:
        selected = [(idx, group, float(gap ** power)) for idx, group, gap in selected]
    return selected


def weight_tag(w: float) -> str:
    return f"w_{int(round(float(w) * 10000.0)):05d}"


def weight_values(start: float, end: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError(f"w_step must be positive, got {step}")
    if end < start:
        raise ValueError(f"w_end must be >= w_start, got {start}>{end}")
    values: List[float] = []
    i = 0
    while True:
        w = round(float(start) + i * float(step), 10)
        if w > float(end) + 1.0e-10:
            break
        if -1.0e-10 <= w <= 1.0 + 1.0e-10:
            values.append(float(np.clip(w, 0.0, 1.0)))
        i += 1
    return values


def bracket_weights(target: float, source_weights: List[float], exclude_exact: bool) -> Tuple[float, float, float]:
    weights = sorted(source_weights)
    exact = [w for w in weights if abs(w - target) <= 1.0e-9]
    if exact and not exclude_exact:
        return exact[0], exact[0], 0.0
    lower_candidates = [w for w in weights if w < target - 1.0e-9]
    upper_candidates = [w for w in weights if w > target + 1.0e-9]
    lower = max(lower_candidates) if lower_candidates else weights[0]
    upper = min(upper_candidates) if upper_candidates else weights[-1]
    if abs(upper - lower) <= 1.0e-12:
        return lower, upper, 0.0
    alpha = (target - lower) / (upper - lower)
    return lower, upper, float(np.clip(alpha, 0.0, 1.0))


def sample_seeds(weight_key: int, count: int, master_seed: int, seed_upper: int) -> List[int]:
    if count <= 0:
        raise ValueError(f"seeds_per_w must be positive, got {count}")
    rng = random.Random(int(master_seed) + int(weight_key) * 1_000_003)
    return rng.sample(range(int(seed_upper)), int(count))


def lerp(a: float, b: float, alpha: float) -> float:
    return float((1.0 - alpha) * float(a) + alpha * float(b))


def normalize_weights(ws: float, wsh: float, wt: float) -> Tuple[float, float, float]:
    ws = max(0.0, float(ws))
    wsh = max(0.0, float(wsh))
    wt = max(0.0, float(wt))
    total = ws + wsh + wt
    if total <= 0.0:
        return 1.0, 0.0, 0.0
    return ws / total, wsh / total, wt / total


def objective_from_weights(ws: float, wsh: float, wt: float) -> str:
    ws, wsh, wt = normalize_weights(ws, wsh, wt)
    parts = []
    if ws > 1.0e-8:
        parts.append(f"stiffness:{ws:.6f}")
    if wsh > 1.0e-8:
        parts.append(f"shear:{wsh:.6f}")
    if wt > 1.0e-8:
        parts.append(f"thermal:{wt:.6f}")
    return ",".join(parts) if parts else "stiffness:1.000000"


def condition_from_sources(lower: SourceSample, upper: SourceSample, alpha: float, volfrac_override: Optional[float]) -> np.ndarray:
    ws = lerp(lower.w_stiffness, upper.w_stiffness, alpha)
    wsh = lerp(lower.w_shear, upper.w_shear, alpha)
    wt = lerp(lower.w_thermal, upper.w_thermal, alpha)
    ws, wsh, wt = normalize_weights(ws, wsh, wt)
    vf = float(volfrac_override) if volfrac_override is not None else lerp(lower.vf, upper.vf, alpha)
    return np.array(
        [
            ws,
            wsh,
            wt,
            vf,
            lerp(lower.stiffness, upper.stiffness, alpha),
            lerp(lower.shear, upper.shear, alpha),
            lerp(lower.thermal, upper.thermal, alpha),
        ],
        dtype=np.float32,
    )


def condition_from_sample(sample: SourceSample, volfrac_override: Optional[float]) -> np.ndarray:
    ws, wsh, wt = normalize_weights(sample.w_stiffness, sample.w_shear, sample.w_thermal)
    vf = float(volfrac_override) if volfrac_override is not None else float(sample.vf)
    return np.asarray(
        [ws, wsh, wt, vf, sample.stiffness, sample.shear, sample.thermal],
        dtype=np.float32,
    )


def convex_combine_conditions(conditions: np.ndarray, eta: np.ndarray) -> np.ndarray:
    eta_arr = np.asarray(eta, dtype=np.float32).reshape(-1, 1)
    cond = np.sum(np.asarray(conditions, dtype=np.float32) * eta_arr, axis=0)
    ws, wsh, wt = normalize_weights(float(cond[0]), float(cond[1]), float(cond[2]))
    cond[0] = ws
    cond[1] = wsh
    cond[2] = wt
    return cond.astype(np.float32)


def parse_tri_weight_values(text: str) -> List[float]:
    values: List[float] = []
    for part in str(text).replace(";", ",").split(","):
        item = part.strip()
        if not item:
            continue
        value = float(item)
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"tri_weight_values must stay in [0,1], got {value}")
        values.append(float(value))
    if not values:
        raise ValueError("--tri_weight_values cannot be empty")
    return values


def tri_sample_weights(args: argparse.Namespace, sample_index: int, seed: int, source_condition: np.ndarray) -> Tuple[float, float, float]:
    mode = str(args.tri_weight_mode)
    if mode == "source":
        return normalize_weights(float(source_condition[0]), float(source_condition[1]), float(source_condition[2]))

    values = parse_tri_weight_values(str(args.tri_weight_values))
    if mode == "random":
        rng = random.Random(int(args.master_seed) + int(seed) * 1_000_003 + int(sample_index) * 9_176)
        w_e = float(values[rng.randrange(len(values))])
    else:
        w_e = float(values[int(sample_index) % len(values)])
    w_kappa = 1.0 - w_e
    return normalize_weights(w_e, 0.0, w_kappa)


def override_condition_weights(condition: np.ndarray, ws: float, wsh: float, wt: float) -> np.ndarray:
    out = np.asarray(condition, dtype=np.float32).copy()
    ws, wsh, wt = normalize_weights(ws, wsh, wt)
    out[0] = float(ws)
    out[1] = float(wsh)
    out[2] = float(wt)
    return out


def build_sample_dir(output_root: str, w: float, seed: int) -> str:
    return os.path.join(output_root, weight_tag(w), f"seed_{int(seed):010d}")


def task_completed_key(task: Dict[str, object]) -> str:
    sample_id = str(task.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    return f"{task.get('weight_tag', '')}:seed_{int(task.get('seed', 0)):010d}"


def build_tasks(args: argparse.Namespace, source_groups: Dict[float, List[SourceSample]], completed: Set[str]) -> List[Dict[str, object]]:
    tasks: List[Dict[str, object]] = []
    source_weights = sorted(source_groups)
    for w in weight_values(float(args.w_start), float(args.w_end), float(args.w_step)):
        lower_w, upper_w, alpha = bracket_weights(w, source_weights, bool(args.exclude_exact))
        lower_group = source_groups[lower_w]
        upper_group = source_groups[upper_w]
        tag = weight_tag(w)
        wkey = int(round(w * 10000.0))
        seeds = sample_seeds(wkey, int(args.seeds_per_w), int(args.master_seed), int(args.seed_upper))
        for rank, seed in enumerate(seeds):
            sample_dir = build_sample_dir(str(args.output_root), w, seed)
            sample_id = f"{tag}_seed_{int(seed):010d}"
            probe_task = {"sample_id": sample_id, "weight_tag": tag, "seed": int(seed)}
            if not args.overwrite and task_completed_key(probe_task) in completed:
                if os.path.exists(os.path.join(sample_dir, "rho_final.npy")):
                    continue
            lower = lower_group[rank % len(lower_group)]
            upper = upper_group[rank % len(upper_group)]
            condition = condition_from_sources(lower, upper, alpha, args.volfrac)
            ws, wsh, wt = normalize_weights(float(condition[0]), float(condition[1]), float(condition[2]))
            tasks.append(
                {
                    "sample_id": sample_id,
                    "front_objectives": "weighted",
                    "weight": float(w),
                    "weight_tag": tag,
                    "weight_idx": wkey,
                    "seed": int(seed),
                    "sample_dir": sample_dir,
                    "lower": lower,
                    "upper": upper,
                    "alpha": alpha,
                    "condition": condition,
                    "w_stiffness": ws,
                    "w_shear": wsh,
                    "w_thermal": wt,
                    "objective": objective_from_weights(ws, wsh, wt),
                    "vf_target": float(condition[3]),
                    "sampling_mode": "uniform",
                    "front_gap": "",
                    "tri_front_edge": "",
                }
            )
    return tasks


def allocate_counts(weights: List[float], total: int) -> List[int]:
    if total <= 0:
        return [0 for _ in weights]
    arr = np.asarray(weights, dtype=np.float64)
    if arr.size == 0:
        return []
    if not np.isfinite(arr).all() or float(arr.sum()) <= 0.0:
        arr = np.ones_like(arr)
    raw = arr / float(arr.sum()) * int(total)
    counts = np.floor(raw).astype(np.int64)
    remainder = int(total) - int(counts.sum())
    if remainder > 0:
        order = np.argsort(-(raw - counts))
        for idx in order[:remainder]:
            counts[int(idx)] += 1
    return [int(x) for x in counts.tolist()]


def build_adaptive_front_tasks(
    args: argparse.Namespace,
    source_groups: Dict[float, List[SourceSample]],
    completed: Set[str],
) -> List[Dict[str, object]]:
    front = pareto_front_samples(all_source_samples(source_groups), eps=float(args.adaptive_front_eps))
    intervals = adaptive_front_intervals(
        front,
        min_gap_quantile=float(args.adaptive_min_gap_quantile),
        gap_power=float(args.adaptive_gap_power),
    )
    if not intervals:
        return build_tasks(args, source_groups, completed)

    target_count = int(args.max_samples) if int(args.max_samples) > 0 else max(1, len(intervals) * int(args.seeds_per_w))
    counts = allocate_counts([gap for _, _, _, gap in intervals], target_count)
    tasks: List[Dict[str, object]] = []

    for interval_rank, ((interval_idx, lower, upper, weighted_gap), count) in enumerate(zip(intervals, counts)):
        if count <= 0:
            continue
        seed_key = int(args.master_seed) + 17_000_003 * (interval_idx + 1)
        seeds = sample_seeds(seed_key, count, 1_000_003, int(args.seed_upper))
        for local_idx, seed in enumerate(seeds):
            alpha = float(local_idx + 1) / float(count + 1)
            condition = condition_from_sources(lower, upper, alpha, args.volfrac)
            ws, wsh, wt = normalize_weights(float(condition[0]), float(condition[1]), float(condition[2]))
            w = ws
            tag = weight_tag(w)
            sample_id = f"weighted_front_seg_{interval_idx:04d}_a_{int(round(alpha * 10000.0)):05d}_seed_{int(seed):010d}"
            sample_dir = os.path.join(
                str(args.output_root),
                "adaptive_front",
                f"seg_{interval_idx:04d}",
                f"a_{int(round(alpha * 10000.0)):05d}_seed_{int(seed):010d}",
            )
            probe_task = {"sample_id": sample_id, "weight_tag": tag, "seed": int(seed)}
            if not args.overwrite and task_completed_key(probe_task) in completed:
                if os.path.exists(os.path.join(sample_dir, "rho_final.npy")):
                    continue
            tasks.append(
                {
                    "sample_id": sample_id,
                    "front_objectives": "weighted",
                    "weight": float(w),
                    "weight_tag": tag,
                    "weight_idx": int(round(w * 10000.0)),
                    "seed": int(seed),
                    "sample_dir": sample_dir,
                    "lower": lower,
                    "upper": upper,
                    "alpha": alpha,
                    "condition": condition,
                    "w_stiffness": ws,
                    "w_shear": wsh,
                    "w_thermal": wt,
                    "objective": objective_from_weights(ws, wsh, wt),
                    "vf_target": float(condition[3]),
                    "sampling_mode": "adaptive_front",
                    "front_gap": float(weighted_gap),
                    "front_interval_rank": interval_rank,
                    "tri_front_edge": "",
                }
            )
    return tasks


def build_triobjective_front_tasks(
    args: argparse.Namespace,
    source_samples: List[SourceSample],
    completed: Set[str],
) -> List[Dict[str, object]]:
    front = pareto_front_samples_e_kappa_vf(
        source_samples,
        eps=float(args.adaptive_front_eps),
        vf_objective=str(args.vf_objective),
    )
    neighborhoods = adaptive_tri_front_neighborhoods(
        front,
        k_neighbors=int(args.tri_front_k),
        min_gap_quantile=float(args.adaptive_min_gap_quantile),
        gap_power=float(args.adaptive_gap_power),
        vf_objective=str(args.vf_objective),
    )
    if not neighborhoods:
        raise ValueError("Need at least two non-dominated E-kappa-vf samples to densify the tri-objective front.")

    target_count = int(args.max_samples) if int(args.max_samples) > 0 else max(1, len(neighborhoods) * int(args.seeds_per_w))
    counts = allocate_counts([gap for _, _, gap in neighborhoods], target_count)
    tasks: List[Dict[str, object]] = []
    candidate_index = 0

    for neighborhood_rank, ((neighborhood_idx, group, weighted_gap), count) in enumerate(zip(neighborhoods, counts)):
        if count <= 0:
            continue
        seed_key = int(args.master_seed) + 29_000_003 * (neighborhood_idx + 1)
        seeds = sample_seeds(seed_key, count, 1_000_003, int(args.seed_upper))
        for local_idx, seed in enumerate(seeds):
            rng = np.random.default_rng(int(args.master_seed) + int(seed) * 1_000_003 + candidate_index * 9_176)
            eta = rng.dirichlet(np.ones(len(group), dtype=np.float64)).astype(np.float32)
            latent_conditions = np.stack([condition_from_sample(sample, args.volfrac) for sample in group], axis=0)
            source_condition = convex_combine_conditions(latent_conditions, eta)
            if str(args.tri_weight_mode) == "source":
                condition = source_condition
                ws, wsh, wt = normalize_weights(float(condition[0]), float(condition[1]), float(condition[2]))
            else:
                ws, wsh, wt = tri_sample_weights(args, candidate_index, seed, source_condition)
                condition = override_condition_weights(source_condition, ws, wsh, wt)
            tag = weight_tag(ws)
            eta_tag = int(round(float(eta[0]) * 10000.0))
            w_tag = int(round(ws * 10000.0))
            sample_id = f"tri_e_kappa_vf_nb_{neighborhood_idx:04d}_e0_{eta_tag:05d}_we_{w_tag:05d}_seed_{int(seed):010d}"
            sample_dir = os.path.join(
                str(args.output_root),
                "tri_e_kappa_vf",
                f"nb_{neighborhood_idx:04d}",
                f"e0_{eta_tag:05d}_we_{w_tag:05d}_seed_{int(seed):010d}",
            )
            candidate_index += 1
            probe_task = {"sample_id": sample_id, "weight_tag": tag, "seed": int(seed)}
            if not args.overwrite and task_completed_key(probe_task) in completed:
                if os.path.exists(os.path.join(sample_dir, "rho_final.npy")):
                    continue
            lower = group[0]
            upper = group[1] if len(group) > 1 else group[0]
            tasks.append(
                {
                    "sample_id": sample_id,
                    "front_objectives": "e_kappa_vf",
                    "weight": float(ws),
                    "weight_tag": tag,
                    "weight_idx": int(round(ws * 10000.0)),
                    "seed": int(seed),
                    "sample_dir": sample_dir,
                    "lower": lower,
                    "upper": upper,
                    "alpha": float(eta[0]),
                    "condition": condition,
                    "latent_conditions": latent_conditions,
                    "latent_etas": eta,
                    "w_stiffness": ws,
                    "w_shear": wsh,
                    "w_thermal": wt,
                    "objective": objective_from_weights(ws, wsh, wt),
                    "vf_target": float(condition[3]),
                    "sampling_mode": "adaptive_front",
                    "front_gap": float(weighted_gap),
                    "front_interval_rank": neighborhood_rank,
                    "tri_front_edge": neighborhood_idx,
                }
            )
    return tasks


def completed_key_from_row(row: Dict[str, str]) -> str:
    sample_id = str(row.get("sample_id", "")).strip()
    if sample_id:
        return sample_id
    weight_tag_text = str(row.get("weight_tag", "")).strip()
    seed_text = str(row.get("seed", "")).strip()
    if weight_tag_text and seed_text:
        return f"{weight_tag_text}:seed_{int(float(seed_text)):010d}"
    return str(row.get("sample_dir", "")).strip()


def load_completed(log_csv: str) -> Set[str]:
    done: Set[str] = set()
    if not os.path.exists(log_csv):
        return done
    with open(log_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = completed_key_from_row(row)
            if key:
                done.add(key)
    return done


def ensure_dataset_log(log_csv: str) -> None:
    if os.path.exists(log_csv):
        return
    os.makedirs(os.path.dirname(log_csv), exist_ok=True)
    with open(log_csv, "w", encoding="utf-8", newline="") as f:
        csv.DictWriter(f, fieldnames=FIELDNAMES).writeheader()


def append_dataset_row(log_csv: str, row: Dict[str, object]) -> None:
    with open(log_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writerow({key: row.get(key, "") for key in FIELDNAMES})


def make_cfg(args: argparse.Namespace):
    cfg = type("Cfg", (), {})()
    cfg.resolution = int(args.resolution)
    cfg.GMG = type("GMG", (), {})()
    cfg.GMG.smooth_iter = list(args.gmg_smooth_iter)
    cfg.GMG.max_cycle = int(args.gmg_max_cycle)
    return cfg


def parse_gpu_ids(spec: str) -> List[Optional[int]]:
    text = str(spec).strip().lower()
    if text == "cpu":
        return [None]
    if text == "auto":
        return list(range(torch.cuda.device_count())) if torch.cuda.is_available() else [None]
    ids = [int(part.strip()) for part in str(spec).split(",") if part.strip()]
    if not ids:
        raise ValueError(f"Invalid --gpus value: {spec!r}")
    return ids


def load_generator(checkpoint: str, device: torch.device) -> Stage2DensityGenerator:
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model"]
    if any(key.startswith("noise_encoder.") for key in state):
        raise ValueError(
            "This script requires a condition-only checkpoint. Retrain without --use_rho_init."
        )
    train_args = dict(ckpt.get("args", {}))
    model = Stage2DensityGenerator(
        cond_dim=int(train_args.get("cond_dim", len(CONDITION_COLUMNS))),
        out_res=int(train_args.get("out_res", 32)),
        base_res=int(train_args.get("base_res", 4)),
        latent_dim=int(train_args.get("latent_dim", 256)),
        base_channels=int(train_args.get("base_channels", 128)),
        use_noise_encoder=False,
    ).to(device)
    if int(model.out_res) != 32:
        raise ValueError(f"This script requires a 32-resolution generator, got out_res={model.out_res}")
    model.load_state_dict(state)
    model.eval()
    return model


def infer_front_objectives(source_csv: str) -> str:
    with open(source_csv, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        has_property_front = "vf_target" in fieldnames and "stiffness" in fieldnames and (
            "thermal" in fieldnames or "kappa" in fieldnames
        )
        has_weight_front = bool({"weight_idx", "w_stiffness", "objective"} & fieldnames)
    if has_property_front and not has_weight_front:
        return "e_kappa_vf"
    return "weighted"


def match_mean_clamped_torch(vol: torch.Tensor, target_mean: float, rho_min: float, iters: int) -> torch.Tensor:
    out = vol.float()
    target = float(np.clip(target_mean, rho_min, 1.0))
    for _ in range(int(max(1, iters))):
        out = out - out.mean() + target
        out = out.clamp(float(rho_min), 1.0)
    return out


def binomial_smooth_periodic_torch(vol: torch.Tensor, passes: int) -> torch.Tensor:
    out = vol.float()
    if int(passes) <= 0:
        return out
    weights = [1.0 / 16.0, 4.0 / 16.0, 6.0 / 16.0, 4.0 / 16.0, 1.0 / 16.0]
    shifts = [-2, -1, 0, 1, 2]
    for _ in range(int(passes)):
        for axis in range(3):
            acc = torch.zeros_like(out)
            for weight, shift in zip(weights, shifts):
                acc = acc + float(weight) * torch.roll(out, shifts=shift, dims=axis)
            out = acc
    return out


def dataset_worker(
    worker_name: str,
    gpu_id: Optional[int],
    args: argparse.Namespace,
    task_queue: "mp.Queue",
    result_queue: "mp.Queue",
) -> None:
    try:
        if gpu_id is None:
            device = torch.device("cpu")
        else:
            if not torch.cuda.is_available():
                raise RuntimeError(f"{worker_name}: CUDA unavailable for gpu_id={gpu_id}")
            torch.cuda.set_device(gpu_id)
            device = torch.device(f"cuda:{gpu_id}")

        cfg = make_cfg(args)
        ihd_module = IHD(cfg, device).to(device)
        generator = load_generator(str(args.checkpoint), device)

        while True:
            task = task_queue.get()
            if task is None:
                break

            sample_dir = str(task["sample_dir"])
            seed = int(task["seed"])
            lower: SourceSample = task["lower"]
            upper: SourceSample = task["upper"]
            condition = np.asarray(task["condition"], dtype=np.float32)

            try:
                os.makedirs(sample_dir, exist_ok=True)
                setup_seed(seed)

                condition_t = torch.from_numpy(condition).to(device=device, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    if "latent_conditions" in task and "latent_etas" in task:
                        latent_conditions_np = np.asarray(task["latent_conditions"], dtype=np.float32)
                        latent_etas_np = np.asarray(task["latent_etas"], dtype=np.float32)
                        latent_conditions_t = torch.from_numpy(latent_conditions_np).to(device=device, dtype=torch.float32)
                        latent_etas_t = torch.from_numpy(latent_etas_np).to(device=device, dtype=torch.float32).reshape(-1, 1)
                        latent = generator.encode_latent(latent_conditions_t, None)
                        latent_mix = torch.sum(latent * latent_etas_t, dim=0, keepdim=True)
                        rho_net = generator.decode_latent(latent_mix)[0, 0]
                    else:
                        rho_net = generator(condition_t, None)[0, 0]
                    rho_warm = rho_net
                    if float(args.warm_sigma) > 0.0:
                        rho_warm = rho_warm + float(args.warm_sigma) * torch.randn_like(rho_warm)
                    if int(args.warm_smooth_passes) > 0:
                        rho_warm = binomial_smooth_periodic_torch(rho_warm, int(args.warm_smooth_passes))
                    if bool(args.match_volume):
                        rho_warm = match_mean_clamped_torch(
                            rho_warm,
                            target_mean=float(task["vf_target"]),
                            rho_min=float(args.rho_min),
                            iters=int(args.volume_iters),
                        )
                    else:
                        rho_warm = rho_warm.clamp(float(args.rho_min), 1.0)

                condition_npy = os.path.join(sample_dir, "condition.npy")
                rho_net_npy = os.path.join(sample_dir, "rho_net_32.npy")
                rho_warm_npy = os.path.join(sample_dir, "rho_warmstart_32.npy")
                np.save(condition_npy, condition.astype(np.float32))
                np.save(rho_net_npy, rho_net.detach().cpu().numpy().astype(np.float32))
                np.save(rho_warm_npy, rho_warm.detach().cpu().numpy().astype(np.float32))

                if not args.quiet:
                    print(
                        f"[START_OPT] worker={worker_name} gpu={gpu_id} {task['weight_tag']} "
                        f"seed={seed} objective={task['objective']} vf={float(task['vf_target']):.4f}",
                        flush=True,
                    )

                batch = build_unitcell_batch(device=device, res=32, rho_grid=rho_warm)
                result = optimize_inverse_homogenization(
                    ihd_module,
                    batch,
                    res=32,
                    objective=str(task["objective"]),
                    init_mode="voxel",
                    volfrac=float(task["vf_target"]),
                    penal=float(args.penal),
                    rho_min=float(args.rho_min),
                    move=float(args.move),
                    xmin=float(args.xmin),
                    iters=int(args.iters),
                    seed=seed,
                    symmetry=str(args.symmetry),
                    lambda_print=float(args.lambda_print),
                    overhang_deg=float(args.overhang_deg),
                    lambda_cavity=float(args.lambda_cavity),
                    cavity_thresh=float(args.cavity_thresh),
                    cavity_beta=float(args.cavity_beta),
                    min_channel_rad=int(args.min_channel_rad),
                    cavity_rad=int(args.cavity_rad),
                    cavity_seed_faces=str(args.cavity_seed_faces),
                    cavity_iters=int(args.cavity_iters),
                    cavity_tiles=int(args.cavity_tiles),
                    cavity_tiles_xyz=(None if args.cavity_tiles_xyz is None else str(args.cavity_tiles_xyz)),
                    cavity_focus_center=bool(args.cavity_focus_center),
                    filter_k=int(args.filter_k),
                    output_dir=sample_dir,
                    mesh_every=0,
                    mesh_res=int(args.mesh_res),
                    mesh_iso=float(args.mesh_iso),
                    mesh_pad=int(args.mesh_pad),
                    save_iter_log=bool(args.save_iter_log),
                    save_summary=False,
                    save_initial_mesh=False,
                    verbose=bool(args.optimizer_verbose),
                )

                rho_final_t = torch.from_numpy(result.rho_final).to(device=device, dtype=torch.float32).reshape(-1, 1)
                metrics = evaluate_homogenized_terms(
                    ihd_module,
                    batch,
                    rho_final_t,
                    terms=("stiffness", "shear", "thermal"),
                )

                result_queue.put(
                    {
                        "ok": True,
                        "worker": worker_name,
                        "gpu_id": gpu_id,
                        "sample_id": task.get("sample_id", ""),
                        "front_objectives": task.get("front_objectives", ""),
                        "weight_tag": task["weight_tag"],
                        "weight_idx": task["weight_idx"],
                        "w_stiffness": task["w_stiffness"],
                        "w_shear": task["w_shear"],
                        "w_thermal": task["w_thermal"],
                        "objective": task["objective"],
                        "vf_target": float(task["vf_target"]),
                        "vf_final": float(result.rho_final.mean()),
                        "seed": seed,
                        "sampling_mode": task.get("sampling_mode", ""),
                        "front_gap": task.get("front_gap", ""),
                        "front_interval_rank": task.get("front_interval_rank", ""),
                        "tri_front_edge": task.get("tri_front_edge", ""),
                        "condition_npy": os.path.abspath(condition_npy),
                        "condition_stiffness": float(condition[4]),
                        "condition_shear": float(condition[5]),
                        "condition_thermal": float(condition[6]),
                        "condition_kappa": float(condition[6]),
                        "source_lower_w": lower.weight,
                        "source_upper_w": upper.weight,
                        "source_alpha": float(task["alpha"]),
                        "source_lower_seed": lower.seed,
                        "source_upper_seed": upper.seed,
                        "source_lower_vf": lower.vf,
                        "source_upper_vf": upper.vf,
                        "source_lower_stiffness": lower.stiffness,
                        "source_lower_shear": lower.shear,
                        "source_lower_thermal": lower.thermal,
                        "source_lower_kappa": lower.thermal,
                        "source_upper_stiffness": upper.stiffness,
                        "source_upper_shear": upper.shear,
                        "source_upper_thermal": upper.thermal,
                        "source_upper_kappa": upper.thermal,
                        "rho_net_32_npy": os.path.abspath(rho_net_npy),
                        "rho_warmstart_32_npy": os.path.abspath(rho_warm_npy),
                        "stiffness": metrics["stiffness"],
                        "shear": metrics["shear"],
                        "thermal": metrics["thermal"],
                        "kappa": metrics["thermal"],
                        "sample_dir": os.path.abspath(sample_dir),
                        "rho_final_npy": os.path.abspath(result.rho_final_npy),
                        "final_obj": os.path.abspath(result.final_obj),
                        "checkpoint": os.path.abspath(str(args.checkpoint)),
                        "source_csv": os.path.abspath(str(args.source_csv)),
                    }
                )
            except Exception:
                result_queue.put(
                    {
                        "ok": False,
                        "worker": worker_name,
                        "gpu_id": gpu_id,
                        "weight_tag": task.get("weight_tag"),
                        "seed": task.get("seed"),
                        "traceback": traceback.format_exc(),
                    }
                )
            finally:
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    except Exception:
        result_queue.put(
            {
                "ok": False,
                "worker": worker_name,
                "gpu_id": gpu_id,
                "weight_tag": None,
                "seed": None,
                "traceback": traceback.format_exc(),
            }
        )


def main() -> None:
    args = parse_args()
    setup_seed(int(args.master_seed))
    args.source_csv = os.path.abspath(str(args.source_csv))
    args.checkpoint = os.path.abspath(str(args.checkpoint))
    args.output_root = os.path.abspath(str(args.output_root))

    if int(args.resolution) != 32:
        raise ValueError(f"This script is fixed to 32-resolution TO, got --resolution {args.resolution}")
    if args.volfrac is not None and not (0.0 < float(args.volfrac) <= 1.0):
        raise ValueError(f"--volfrac must be in (0,1], got {args.volfrac}")
    if int(args.workers_per_gpu) <= 0:
        raise ValueError(f"workers_per_gpu must be positive, got {args.workers_per_gpu}")

    front_objectives = str(args.front_objectives)
    if front_objectives == "auto":
        front_objectives = infer_front_objectives(args.source_csv)
    args.front_objectives = front_objectives

    os.makedirs(args.output_root, exist_ok=True)
    dataset_log = os.path.join(args.output_root, "dataset_log.csv")
    ensure_dataset_log(dataset_log)
    completed = load_completed(dataset_log) if not args.overwrite else set()

    source_groups: Optional[Dict[float, List[SourceSample]]] = None
    source_samples: Optional[List[SourceSample]] = None
    if front_objectives == "e_kappa_vf":
        source_samples = load_source_samples(args.source_csv, args)
        tasks = build_triobjective_front_tasks(args, source_samples, completed)
    else:
        source_groups = load_source_groups(args.source_csv, args)
        if str(args.sampling_mode) == "adaptive_front":
            tasks = build_adaptive_front_tasks(args, source_groups, completed)
        else:
            tasks = build_tasks(args, source_groups, completed)
        if str(args.sampling_mode) != "adaptive_front" and int(args.max_samples) > 0:
            tasks = tasks[: int(args.max_samples)]
    if not tasks:
        print("[INFO] no pending samples")
        return

    gpu_ids = parse_gpu_ids(str(args.gpus))
    worker_specs: List[Tuple[str, Optional[int]]] = []
    for gpu_id in gpu_ids:
        label = "cpu" if gpu_id is None else str(gpu_id)
        for worker_idx in range(int(args.workers_per_gpu)):
            worker_specs.append((f"gpu{label}-w{worker_idx}", gpu_id))

    if not args.quiet:
        print(f"[INFO] front_objectives={front_objectives}")
        if source_groups is not None:
            source_weights = sorted(source_groups)
            print(f"[INFO] source weights={len(source_weights)} range={source_weights[0]:.4f}..{source_weights[-1]:.4f}")
        if source_samples is not None:
            front = pareto_front_samples_e_kappa_vf(
                source_samples,
                eps=float(args.adaptive_front_eps),
                vf_objective=str(args.vf_objective),
            )
            print(f"[INFO] source samples={len(source_samples)} tri_pareto_front={len(front)} vf_objective={args.vf_objective}")
        print(f"[INFO] pending samples={len(tasks)} workers={len(worker_specs)} devices={gpu_ids}")

    ctx = mp.get_context("spawn")
    task_queue = ctx.Queue()
    result_queue = ctx.Queue()
    for task in tasks:
        task_queue.put(task)
    for _ in worker_specs:
        task_queue.put(None)

    workers: List[mp.Process] = []
    for worker_name, gpu_id in worker_specs:
        p = ctx.Process(target=dataset_worker, args=(worker_name, gpu_id, args, task_queue, result_queue), name=worker_name)
        p.start()
        workers.append(p)

    finished = 0
    while finished < len(tasks):
        msg = result_queue.get()
        finished += 1
        if msg["ok"]:
            append_dataset_row(dataset_log, msg)
            if not args.quiet:
                print(f"[DONE] {finished}/{len(tasks)} worker={msg['worker']} gpu={msg['gpu_id']} {msg['weight_tag']} seed={msg['seed']}")
        else:
            print(f"[ERROR] {finished}/{len(tasks)} worker={msg['worker']} gpu={msg['gpu_id']} weight={msg.get('weight_tag')} seed={msg.get('seed')}")
            print(msg["traceback"], end="")

    for p in workers:
        p.join()
        if p.exitcode not in (0, None):
            raise RuntimeError(f"Worker {p.name} exited with code {p.exitcode}")

    print(f"[DONE] dataset_log={dataset_log}")


if __name__ == "__main__":
    main()
