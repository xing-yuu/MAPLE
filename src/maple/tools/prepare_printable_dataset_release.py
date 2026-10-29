from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


DEFAULT_INPUT_CSVS = [
    "./data/datasets/multiphysics/dataset_log_printability.csv",
    "./data/results/iterative/dataset_log_printability.csv",
]

RELEASE_LOG_FIELDS = [
    "structure_id",
    "folder",
    "w_stiffness",
    "w_shear",
    "w_thermal",
    "stiffness",
    "kappa",
    "shear",
    "printability_score",
    "volume_fraction",
    "resolution",
    "rho_final",
    "final_obj",
]

TRUE_VALUES = {"1", "true", "t", "yes", "y"}
FALSE_VALUES = {"0", "false", "f", "no", "n", ""}


@dataclass(frozen=True)
class PathRemap:
    old_root: Path
    new_root: Path


@dataclass(frozen=True)
class ReleaseRecord:
    structure_id: str
    folder: str
    rho_source: Path
    obj_source: Path
    log_row: Dict[str, str]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Collect printable lattice structures from multiple printability CSV logs "
            "into one publication-ready dataset."
        )
    )
    p.add_argument(
        "--input_csv",
        nargs="+",
        default=None,
        help="Input printability CSV files. Defaults to the 15 configured project logs.",
    )
    p.add_argument("--output_root", required=True, help="Release dataset output directory.")
    p.add_argument(
        "--path_remap",
        action="append",
        default=[],
        metavar="OLD_ROOT=NEW_ROOT",
        help="Optional path-root rewrite; may be passed multiple times.",
    )
    p.add_argument(
        "--copy_mode",
        choices=["copy", "hardlink"],
        default="copy",
        help="How to place assets in the release. Use copy for a portable published dataset.",
    )
    p.add_argument(
        "--allow_missing",
        action="store_true",
        help="Skip printable rows with missing density or OBJ assets instead of failing.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing files in an existing release directory.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print a summary without creating the release.",
    )
    return p.parse_args()


def parse_bool(value: object) -> bool:
    text = str(value).strip().lower()
    if text in TRUE_VALUES:
        return True
    if text in FALSE_VALUES:
        return False
    try:
        return float(text) != 0.0
    except ValueError as exc:
        raise ValueError(f"cannot parse boolean value {value!r}") from exc


def parse_remaps(values: Sequence[str]) -> List[PathRemap]:
    remaps: List[PathRemap] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected OLD_ROOT=NEW_ROOT for --path_remap, got {value!r}")
        old, new = value.split("=", 1)
        if not old.strip() or not new.strip():
            raise ValueError(f"expected non-empty OLD_ROOT=NEW_ROOT, got {value!r}")
        remaps.append(PathRemap(Path(old.strip()), Path(new.strip())))
    return remaps


def first_nonempty(row: Mapping[str, str], names: Iterable[str]) -> str:
    for name in names:
        value = str(row.get(name, "")).strip()
        if value:
            return value
    return ""


def optional_float(row: Mapping[str, str], names: Iterable[str]) -> Optional[float]:
    for name in names:
        value = str(row.get(name, "")).strip()
        if not value:
            continue
        try:
            number = float(value)
        except ValueError:
            continue
        if math.isfinite(number):
            return number
    return None


def format_optional_float(value: Optional[float]) -> str:
    return "" if value is None else f"{value:.12g}"


def objective_weights(row: Mapping[str, str]) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    objective = str(row.get("objective", ""))
    for part in objective.split(","):
        if ":" not in part:
            continue
        name, value = part.split(":", 1)
        key = name.strip().lower()
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", value)
        if match and key in {"stiffness", "shear", "thermal", "kappa"}:
            weights["thermal" if key == "kappa" else key] = float(match.group(0))
    return weights


def normalized_weights(row: Mapping[str, str]) -> Tuple[float, float, float]:
    parsed = objective_weights(row)
    ws = optional_float(row, ("w_stiffness", "w1", "weight_stiffness"))
    wsh = optional_float(row, ("w_shear", "w2", "weight_shear"))
    wt = optional_float(row, ("w_thermal", "w3", "w_kappa", "weight_thermal"))

    ws = parsed.get("stiffness", 0.0) if ws is None else ws
    wsh = parsed.get("shear", 0.0) if wsh is None else wsh
    if wt is None:
        wt = parsed.get("thermal", 1.0 - ws - wsh)
    return float(ws), float(wsh), float(wt)


def printability_score(row: Mapping[str, str]) -> Optional[float]:
    direct = optional_float(
        row,
        ("S_print", "printability_score", "print_score", "printability"),
    )
    if direct is not None:
        return direct

    # Legacy evaluator format:
    #   S_oh: overhang score in [0, 100]
    #   S_trap: trapped-void penalty in [0, 1]
    #   N_cc: number of solid connected components
    # Reconstruct the equal-weight total score used by the current evaluator.
    s_oh = optional_float(row, ("S_oh",))
    p_chan = optional_float(row, ("S_trap", "P_chan"))
    component_count = optional_float(row, ("N_cc", "solid_component_count"))
    if s_oh is None or p_chan is None or component_count is None:
        return None

    p_oh = 1.0 - min(max(s_oh / 100.0, 0.0), 1.0)
    p_chan = min(max(p_chan, 0.0), 1.0)
    if component_count <= 0.0:
        p_conn = 1.0
    elif component_count <= 1.0:
        p_conn = 0.0
    else:
        p_conn = 1.0 - (1.0 / component_count)
    total_penalty = min(max((p_oh + p_chan + p_conn) / 3.0, 0.0), 1.0)
    return 100.0 * (1.0 - total_penalty)


def apply_remap(path: Path, remaps: Sequence[PathRemap]) -> Iterable[Path]:
    yield path
    path_text = str(path)
    for remap in remaps:
        old = str(remap.old_root).rstrip("/")
        if path_text == old:
            yield remap.new_root
        elif path_text.startswith(old + "/"):
            yield Path(str(remap.new_root).rstrip("/") + path_text[len(old) :])


def relocation_candidates(path: Path, csv_path: Path) -> Iterable[Path]:
    if not path.is_absolute():
        yield csv_path.parent / path
        return

    parts = path.parts
    for anchor in (csv_path.parent, *csv_path.parents):
        if not anchor.name:
            continue
        for index in range(len(parts) - 1, -1, -1):
            if parts[index] == anchor.name:
                yield anchor.joinpath(*parts[index + 1 :])


def path_candidates(raw: str, csv_path: Path, remaps: Sequence[PathRemap]) -> Iterable[Path]:
    if not raw.strip():
        return
    source = Path(raw.strip()).expanduser()
    seen: set[str] = set()
    for remapped in apply_remap(source, remaps):
        candidates = [remapped, *relocation_candidates(remapped, csv_path)]
        for candidate in candidates:
            key = os.path.normpath(str(candidate))
            if key not in seen:
                seen.add(key)
                yield Path(key)


def resolve_asset(
    row: Mapping[str, str],
    csv_path: Path,
    remaps: Sequence[PathRemap],
    *,
    kind: str,
) -> Optional[Path]:
    if kind == "rho":
        direct_columns = ("rho_final_npy", "rho_eval_path", "rho_final", "density_npy")
        sample_suffixes = ("rho_final.npy",)
    elif kind == "obj":
        direct_columns = ("final_obj", "obj_path", "mesh_obj")
        sample_suffixes = ("meshes/final.obj", "final.obj")
    else:
        raise ValueError(f"unsupported asset kind: {kind}")

    raw_candidates: List[str] = []
    for column in direct_columns:
        value = str(row.get(column, "")).strip()
        if value:
            raw_candidates.append(value)

    sample_dir = str(row.get("sample_dir", "")).strip()
    if sample_dir:
        for suffix in sample_suffixes:
            raw_candidates.append(str(Path(sample_dir) / suffix))

    for raw in raw_candidates:
        for candidate in path_candidates(raw, csv_path, remaps):
            if candidate.is_file():
                return candidate.resolve()
    return None


def read_printable_rows(csv_path: Path) -> Tuple[List[Dict[str, str]], int]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = list(reader)

    printable_rows: List[Dict[str, str]] = []
    for row_index, row in enumerate(rows, start=2):
        raw = first_nonempty(
            row,
            ("printable_ok", "printability_ok", "printability", "printable"),
        )
        if not raw:
            raise ValueError(
                f"{csv_path}:{row_index} has no printable_ok/printability_ok value"
            )
        if parse_bool(raw):
            printable_rows.append(dict(row))
    return printable_rows, len(rows)


def release_row(row: Mapping[str, str], structure_id: str, folder: str) -> Dict[str, str]:
    ws, wsh, wt = normalized_weights(row)
    stiffness = optional_float(row, ("stiffness", "stiffness_final", "prop_stiffness"))
    kappa = optional_float(row, ("kappa", "thermal", "thermal_final", "prop_thermal"))
    shear = optional_float(row, ("shear", "shear_final", "prop_shear"))
    print_score = printability_score(row)
    volume_fraction = optional_float(row, ("vf_final", "volume_fraction", "vf_target", "vf_binary"))
    resolution = first_nonempty(row, ("resolution",))

    if stiffness is None:
        raise ValueError("missing stiffness")
    if kappa is None:
        raise ValueError("missing thermal/kappa")
    if print_score is None:
        raise ValueError(
            "missing printability score fields; expected S_print or legacy S_oh/S_trap/N_cc"
        )

    return {
        "structure_id": structure_id,
        "folder": folder,
        "w_stiffness": format_optional_float(ws),
        "w_shear": format_optional_float(wsh),
        "w_thermal": format_optional_float(wt),
        "stiffness": format_optional_float(stiffness),
        "kappa": format_optional_float(kappa),
        "shear": format_optional_float(shear),
        "printability_score": format_optional_float(print_score),
        "volume_fraction": format_optional_float(volume_fraction),
        "resolution": resolution,
        "rho_final": f"{folder}/rho_final.npy",
        "final_obj": f"{folder}/final.obj",
    }


def collect_records(
    input_csvs: Sequence[Path],
    remaps: Sequence[PathRemap],
    allow_missing: bool,
) -> Tuple[List[ReleaseRecord], List[Dict[str, str]], Dict[str, object]]:
    records: List[ReleaseRecord] = []
    problems: List[str] = []
    total_rows = 0
    printable_rows = 0

    for csv_path in input_csvs:
        if not csv_path.is_file():
            problems.append(f"input CSV not found: {csv_path}")
            continue

        selected, row_count = read_printable_rows(csv_path)
        total_rows += row_count
        printable_rows += len(selected)

        for source_index, row in enumerate(selected, start=1):
            rho_source = resolve_asset(row, csv_path, remaps, kind="rho")
            obj_source = resolve_asset(row, csv_path, remaps, kind="obj")
            missing = []
            if rho_source is None:
                missing.append("rho_final.npy")
            if obj_source is None:
                missing.append("final.obj")

            structure_number = len(records) + 1
            structure_id = f"structure_{structure_number:07d}"
            folder = f"structures/{structure_id}"

            try:
                out_row = release_row(row, structure_id, folder)
            except Exception as exc:
                missing.append(str(exc))
                out_row = {}

            if missing:
                problems.append(
                    f"{csv_path}: printable row {source_index}: {', '.join(missing)}"
                )
                continue

            records.append(
                ReleaseRecord(
                    structure_id=structure_id,
                    folder=folder,
                    rho_source=rho_source,
                    obj_source=obj_source,
                    log_row=out_row,
                )
            )

    if problems and not allow_missing:
        detail = "\n".join(f"  - {problem}" for problem in problems[:50])
        suffix = "" if len(problems) <= 50 else f"\n  ... and {len(problems) - 50} more"
        raise RuntimeError(
            "release validation failed before copying files:\n"
            f"{detail}{suffix}\n"
            "Fix the paths/fields, pass --path_remap OLD=NEW, or use --allow_missing to skip them."
        )

    report: Dict[str, object] = {
        "input_csv_count": len(input_csvs),
        "input_row_count": total_rows,
        "printable_row_count": printable_rows,
        "released_structure_count": len(records),
        "skipped_problem_count": len(problems),
        "problems": problems,
    }
    return records, [record.log_row for record in records], report


def place_file(source: Path, destination: Path, copy_mode: str, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"destination exists; pass --overwrite to replace it: {destination}")
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "copy":
        shutil.copy2(source, destination)
    elif copy_mode == "hardlink":
        os.link(source, destination)
    else:
        raise ValueError(f"unsupported copy mode: {copy_mode}")


def write_release_log(output_root: Path, rows: Sequence[Mapping[str, str]]) -> None:
    output_csv = output_root / "dataset_log.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RELEASE_LOG_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def readme_text(structure_count: int) -> str:
    return f"""# Manufacturable Multi-Objective Lattice Structure Dataset

This release contains **{structure_count}** manufacturable topology-optimized lattice structures generated for multi-objective material-property design.

## Dataset Overview

The dataset combines stiffness-thermal and stiffness-shear-thermal design tasks into one structure collection without source-specific categories. Physical properties and manufacturability scores retain the values recorded by the original generation and evaluation pipelines.

## Layout

```text
.
|-- README.md
|-- dataset_log.csv
|-- release_report.json
`-- structures/
    |-- structure_0000001/
    |   |-- rho_final.npy
    |   `-- final.obj
    `-- ...
```

Each structure folder contains:

- `rho_final.npy`: final voxel density field.
- `final.obj`: final surface mesh exported by the optimization pipeline.

## Metadata

`dataset_log.csv` contains one row per released structure:

| Column | Description |
| --- | --- |
| `structure_id` | Unique structure identifier. |
| `folder` | Relative path to the structure folder. |
| `w_stiffness` | Stiffness objective weight. |
| `w_shear` | Shear objective weight. It is `0` for stiffness-thermal runs. |
| `w_thermal` | Thermal-conductivity objective weight. |
| `stiffness` | Final stiffness property copied from the source log. |
| `kappa` | Final thermal-conductivity property; renamed from source field `thermal`. |
| `shear` | Final shear property. Empty when it was not evaluated in the source run. |
| `printability_score` | Source `S_print`, or the equivalent score reconstructed from legacy `S_oh`, `S_trap`, and `N_cc`. |
| `volume_fraction` | Final volume fraction when available. |
| `resolution` | Density-grid resolution reported by the printability evaluation. |
| `rho_final` | Relative path to the density field. |
| `final_obj` | Relative path to the OBJ mesh. |

## Notes

- Density fields and meshes retain their original source resolution and geometry.
- Physical-property values retain the normalization and conventions used by the original optimization pipeline.
- `release_report.json` records aggregate packaging counts and any skipped rows.
- Add the appropriate license, citation, authorship, and contact information before public distribution.
"""


def write_release_files(
    output_root: Path,
    records: Sequence[ReleaseRecord],
    rows: Sequence[Mapping[str, str]],
    report: Dict[str, object],
    *,
    copy_mode: str,
    overwrite: bool,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    if (output_root / "dataset_log.csv").exists() and not overwrite:
        raise FileExistsError(
            f"release already exists: {output_root}. Pass --overwrite to replace release files."
        )

    for index, record in enumerate(records, start=1):
        structure_dir = output_root / record.folder
        place_file(record.rho_source, structure_dir / "rho_final.npy", copy_mode, overwrite)
        place_file(record.obj_source, structure_dir / "final.obj", copy_mode, overwrite)
        if index % 100 == 0 or index == len(records):
            print(f"[COPY] {index}/{len(records)} structures", flush=True)

    write_release_log(output_root, rows)
    (output_root / "README.md").write_text(readme_text(len(records)), encoding="utf-8")

    report_out = dict(report)
    report_out["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    report_out["copy_mode"] = copy_mode
    with (output_root / "release_report.json").open("w", encoding="utf-8") as f:
        json.dump(report_out, f, indent=2, ensure_ascii=True)
        f.write("\n")


def main() -> None:
    args = parse_args()
    input_csvs = [Path(path).expanduser().resolve() for path in (args.input_csv or DEFAULT_INPUT_CSVS)]
    output_root = Path(args.output_root).expanduser().resolve()
    remaps = parse_remaps(args.path_remap)

    records, rows, report = collect_records(input_csvs, remaps, bool(args.allow_missing))
    print(
        f"[VALID] input_rows={report['input_row_count']} "
        f"printable={report['printable_row_count']} "
        f"release={report['released_structure_count']} "
        f"skipped={report['skipped_problem_count']}"
    )

    if args.dry_run:
        print("[DRY RUN] validation complete; no files were written")
        return

    write_release_files(
        output_root,
        records,
        rows,
        report,
        copy_mode=str(args.copy_mode),
        overwrite=bool(args.overwrite),
    )
    print(f"[DONE] release dataset: {output_root}")
    print(f"[DONE] metadata: {output_root / 'dataset_log.csv'}")
    print(f"[DONE] README: {output_root / 'README.md'}")


if __name__ == "__main__":
    main()
