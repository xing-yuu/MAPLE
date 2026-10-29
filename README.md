# MAPLE: Lattice Structure Optimization for Additive Manufacturing

[中文 README](README.zh-CN.md)

This repository contains research code for the paper **Lattice Structure Optimization for Additive Manufacturing: Manufacturability-Driven Design and Pareto Front Construction**. It implements periodic inverse homogenization, density-based topology optimization, differentiable manufacturability penalties, a conditional density generator, and Pareto-front densification.

## Paper-to-code map

- `src/maple/core/`: inverse homogenization, finite-element solvers, objectives, printability, and Stage-II model code.
- `src/maple/data/`: multi-physics, bi-objective, and tri-objective dataset builders.
- `src/maple/training/`: Stage-II data preparation and model training.
- `src/maple/pareto/`: Pareto-front densification workflows.
- `src/maple/analysis/`: Pareto metrics, plotting, and printability evaluation.
- `src/maple/tools/`: dataset path repair, weight estimation, and release packaging.

The optimizer applies periodic filtering and OC updates to the design field. The `thermal` objective uses the conductivity tensor trace, and Stage-II data are normally converted from `64^3` to `32^3`.

## Repository layout

```text
src/maple/
├── core/              # Inverse homogenization, FE solvers, objectives, printability
├── data/              # Multi-physics, bi-objective, and tri-objective data builders
├── training/          # Stage-II data preparation and training
├── pareto/            # Pareto-front densification workflows
├── analysis/          # Metrics, plots, and printability evaluation
└── tools/             # Dataset path and release utilities
experiments/
├── README.md
├── biobjective/
│   ├── README.md
│   ├── run_weight_sweep.sh
│   ├── run_weight_sweep_unconstrained.sh
│   └── run_network_densification.sh
└── triobjective/
    ├── README.md
    ├── run_weight_sweep.sh
    └── run_e_kappa_vf_densification.sh
scripts/
└── training/run_stage2.sh
```

Each experiment directory contains its own usage notes and shell entry points.

| Experiment | Objective definition | Main entry point |
| --- | --- | --- |
| Bi-objective | Maximize `stiffness` and `thermal` at fixed volume fraction | `maple.data.build_weight_sweep_dataset` |
| Bi-objective baseline | Same two properties without manufacturability penalties | `maple.data.build_weight_sweep_dataset_unconstrained` |
| Tri-objective weighted sweep | Weighted `stiffness`, `shear`, and `thermal` at fixed volume fraction | `maple.data.build_triobjective_weight_dataset` |
| Tri-objective Pareto densification | Maximize stiffness and thermal conductivity while minimizing volume fraction | `--front_objectives e_kappa_vf` |

The code names correspond to the paper's E, G, and kappa. The weighted tri-objective sweep and the E-kappa-vf Pareto experiment are different workflows and should be reported separately.

The scalarized objectives are:

```text
Bi-objective:  J = w_E E + w_kappa kappa,         w_E + w_kappa = 1, w_G = 0
Tri-objective: J = w_E E + w_G G + w_kappa kappa, w_E + w_G + w_kappa = 1
```

`maple.data.build_multiphysics_dataset` generates an initial multi-physics dataset over volume fractions (default 1%–40%).

## Environment

The provided `environment.yml` targets Python 3.10, PyTorch 2.4, and CUDA 12.1 on Linux with an NVIDIA GPU:

```bash
conda env create -f environment.yml
conda activate ihd
python -m pip install -e .
```

`pymcubes`, `torch-scatter`, `scipy`, and `spconv` are optional packages for selected export or acceleration paths. CPU and macOS installations need a PyTorch build appropriate for the local platform.

Stage-II preparation normally converts input density fields to `32^3`; source cubic resolutions should therefore be divisible by 32. The `maple.pareto.densify_pareto_property_network_32` workflow and its condition-only checkpoint are fixed to `32^3`.

Check the available command-line interfaces after activating the environment:

```bash
python -m maple.core.main_IHD_v3 --help
python -m maple.data.build_weight_sweep_dataset --help
python -m maple.data.build_triobjective_weight_dataset --help
python -m maple.pareto.run_iterative_property_network_densification --help
python -m maple.data.build_multiphysics_dataset --help
```

## Running the experiments

### Bi-objective E-kappa

```bash
bash experiments/biobjective/run_weight_sweep.sh \
  --w_start 0 --w_end 100 --w_step 5 --seeds_per_w 10
```

The default target volume fraction is `0.25`. The constrained and unconstrained scripts use separate output roots so that the baseline can be compared with the manufacturability-aware run.

To run network densification, pass a local weighted CSV. A CSV produced by the preceding local sweep is suitable:

```bash
SOURCE_CSV=./data/results/biobjective-weight-sweep/dataset_log.csv \
OUT_ROOT=./data/results/biobjective-network \
bash experiments/biobjective/run_network_densification.sh
```

The source CSV must include `w_stiffness`/`w_thermal` or an `objective` column. Generated local sweep logs are stored under `data/results/`.

### Tri-objective E-G-kappa weighted sweep

```bash
bash experiments/triobjective/run_weight_sweep.sh \
  --w1_start 0 --w1_end 100 --w2_start 0 --w2_end 100 \
  --w_step 5 --seeds_per_combo 10
```

The script skips combinations with `w1+w2>100` and sets the third weight to `w3=1-w1-w2`. This is the fixed-volume-fraction weighted scalarization workflow; the E-kappa-vf Pareto workflow is provided by the network densification scripts.

### Tri-objective E-kappa-vf Pareto densification

The source CSV must have locally readable density paths, explicit objective weights, and preferably multiple volume fractions. One way to prepare a local weighted CSV from the included multi-physics log is:

```bash
python -m maple.tools.repair_dataset_log_paths \
  --input_csv ./data/datasets/multiphysics/dataset_log.csv \
  --output_csv ./data/datasets/multiphysics/dataset_log_local.csv \
  --old_root /public/home/lvlin013/parato/IHD-0422/dataset-multiphysics \
  --new_root "$PWD/data/datasets/multiphysics" --check_exists

python -m maple.tools.estimate_dataset_log_weights \
  --input_csv ./data/datasets/multiphysics/dataset_log_local.csv \
  --output_csv ./data/datasets/multiphysics/dataset_log_weighted.csv

SOURCE_CSV=./data/datasets/multiphysics/dataset_log_weighted.csv \
bash experiments/triobjective/run_e_kappa_vf_densification.sh
```

The default is five rounds with 200 generated samples per round. For ablations with source CSVs without weights, use `--tri_weight_mode cycle` or `random` to select the alternative weight-generation strategy.

The network wrappers require `SOURCE_CSV` explicitly. They validate the file and its weight columns before starting a long training run. The `GPUS` variable controls topology-optimization workers; Stage-II training uses the iterative script's `--device` option.

## Analysis

`plot_iterative_pareto_front.py` creates Pareto-front projections, and `compute_pareto_hv_c.py` computes hypervolume and C-metric values:

```bash
python -m maple.analysis.plot_iterative_pareto_front \
  --inputs ./data/datasets/multiphysics/dataset_log.csv \
  --front_mode ekappa \
  --output_dir ./data/results/biobjective-front

python -m maple.analysis.compute_pareto_hv_c \
  --initial_csv ./data/datasets/multiphysics/dataset_log.csv \
  --combined_csvs ./data/datasets/multiphysics/dataset_log.csv \
  --front_mode ekappa \
  --output_dir ./data/results/biobjective-metrics
```

For tri-objective analysis, use `--front_mode tri` and set `--vf_key`, `--e_key`, and `--kappa_key` to match the CSV. The analysis scripts divide `thermal`/`kappa` by 3 by default; set `--kappa_divisor 1` when the input is already normalized.

## Data and outputs

The dataset builders write CSV logs, density fields, meshes, manifests, and training checkpoints under a generated `data/` directory. Use `python -m maple.tools.repair_dataset_log_paths` for path relocation and `python -m maple.tools.prepare_printable_dataset_release` for release packaging.

Stage-II training uses objective-conditioned data: bi-objective runs use `w_shear=0`, while tri-objective runs use logs containing the shear target. The iterative pipeline stores cumulative CSVs and checkpoints under `OUT_ROOT`.

## License

This project is released under the MIT License. See [`LICENSE`](LICENSE).

## Citation

If you use the code or generated data, cite:

Paper: [Journal article](https://www.jcad.cn/article/doi/10.3724/SP.J.1089.2026-00157) · DOI: [10.3724/SP.J.1089.2026-00157](https://dx.doi.org/10.3724/SP.J.1089.2026-00157)

```bibtex
@article{xing2026lattice,
  author  = {Xing, Yu and Liu, Yang and Lu, Lin},
  title   = {Lattice Structure Optimization for Additive Manufacturing: Manufacturability-Driven Design and Pareto Front Construction},
  journal = {Journal of Computer-Aided Design & Computer Graphics},
  year    = {2026},
  volume  = {43},
  number  = {1},
  doi     = {10.3724/SP.J.1089.2026-00157}
}
```

