# MAPLE：面向增材制造的晶格结构优化

[English README](README.md)

本仓库整理了论文
**Lattice Structure Optimization for Additive Manufacturing: Manufacturability-Driven Design and Pareto Front Construction** 的研究代码，覆盖逆均质化拓扑优化、可制造性约束、条件密度生成网络和 Pareto 前沿增密。

论文中的主要方法在代码中的对应关系是：

- `src/maple/core/`：逆均质化、有限元求解器、目标函数、可制造性约束和 Stage-II 模型；
- `src/maple/data/`：多物理、双目标和三目标数据生成；
- `src/maple/training/`：Stage-II 数据准备和模型训练；
- `src/maple/pareto/`：Pareto 前沿增密流程；
- `src/maple/analysis/`：Pareto 指标、绘图和可制造性评估；
- `src/maple/tools/`：数据路径修复、权重估计和发布打包。

优化器对设计场执行周期滤波和 OC 更新；`thermal` 目标使用导热张量迹，Stage-II 数据通常由 `64^3` 整理为 `32^3`。

## 实验目录

```text
src/maple/
├── core/              # 逆均质化、有限元求解器、目标函数和可制造性约束
├── data/              # 多物理、双目标和三目标数据生成
├── training/          # Stage-II 数据准备和训练
├── pareto/            # Pareto 前沿增密流程
├── analysis/          # 指标、绘图和可制造性评估
└── tools/             # 数据路径和发布工具
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

每个实验目录都包含对应的使用说明和 Shell 入口。

双目标和三目标入口已经分开：

| 实验 | 目标定义 | 主要入口 |
| --- | --- | --- |
| 双目标 | 固定体积分数下最大化 `stiffness` 与 `thermal` | `maple.data.build_weight_sweep_dataset` |
| 双目标基线 | 同上，但不加入制造约束 | `maple.data.build_weight_sweep_dataset_unconstrained` |
| 三目标加权扫描 | 固定体积分数下加权 `stiffness`、`shear`、`thermal` | `maple.data.build_triobjective_weight_dataset` |
| 三目标 Pareto 增密 | 最大化刚度、最大化导热、最小化体积分数 | `--front_objectives e_kappa_vf` |

`maple.data.build_multiphysics_dataset` 是按体积分数（默认 1%–40%）生成初始多物理数据的
入口；它使用默认的刚度–导热加权目标，结果本身不是完整的 Pareto 前沿。

这里的 `stiffness`、`shear`、`thermal` 分别对应论文中的 E、G、κ。三目标加权扫描和
`E–κ–vf` Pareto 增密是两条独立的实验路径，分别对应物性加权扫描和三指标前沿增密。

双目标扫描使用 `J = w_E E + w_κ κ` 且 `w_E+w_κ=1`、`w_G=0`；三物性加权扫描使用
`J = w_E E + w_G G + w_κ κ` 且三个权重位于 simplex（和为 1）。这些是加权标量化
目标；`E–κ–vf` 路径则直接在三个指标上做非支配筛选。

## 环境安装

代码提供了 `environment.yml`。推荐在 Linux + NVIDIA GPU 环境中使用 Python 3.10 和 PyTorch 2.4：

```bash
conda env create -f environment.yml
conda activate ihd
python -m pip install -e .
```

`pymcubes`、`torch-scatter`、`scipy` 和 `spconv` 是可选依赖；某些网格导出或加速路径需要额外安装。CPU 或 macOS 环境需要根据本机 PyTorch 版本调整 `environment.yml` 中的 CUDA 依赖。

Stage-II 数据默认把输入密度场整理为 `32^3`；若从 `64^3` 以外的分辨率准备训练数据，
请确认原始立方分辨率能被 32 整除。`densify_pareto_property_network_32.py` 和对应
checkpoint 都固定使用 `32^3`。

先确认入口和参数可加载：

```bash
python -m maple.core.main_IHD_v3 --help
python -m maple.data.build_weight_sweep_dataset --help
python -m maple.data.build_triobjective_weight_dataset --help
python -m maple.pareto.run_iterative_property_network_densification --help
python -m maple.data.build_multiphysics_dataset --help
```

## 快速开始

### 双目标 E–κ

```bash
bash experiments/biobjective/run_weight_sweep.sh \
  --w_start 0 --w_end 100 --w_step 5 --seeds_per_w 10
```

这会在默认 `vf=0.25` 下生成刚度–导热权重扫描；可根据计算资源调整
`--seeds_per_w` 和 `--w_step`。

### 三目标 E–G–κ 加权扫描

```bash
bash experiments/triobjective/run_weight_sweep.sh \
  --w1_start 0 --w1_end 100 --w2_start 0 --w2_end 100 \
  --w_step 5 --seeds_per_combo 10
```

### 三目标 E–κ–vf 前沿增密

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

默认设置为 5 轮、每轮 200 个样本，适合论文规模实验；第一次运行会先训练初始
Stage-II 网络。若已有检查点，可以指定：

```bash
STAGE2_CHECKPOINT=/path/to/best.pt \
bash experiments/triobjective/run_e_kappa_vf_densification.sh
```

所有启动脚本都支持在命令末尾追加原始 Python 参数，也支持通过环境变量覆盖常用值：
`PYTHON_BIN`、`OUT_ROOT`、`SOURCE_CSV`、`VF`、`GPUS`、`ROUNDS`、
`SAMPLES_PER_ROUND`、`STAGE2_CHECKPOINT`。

`GPUS` 控制拓扑优化和增密 worker；Stage-II 训练使用迭代脚本自己的 `--device` 参数。
如果改变了源数据或要重新开始实验，请使用新的 `OUT_ROOT`；在已有目录上反复使用
`--overwrite` 可能会把重复行追加到日志中。

网络增密使用包含目标权重列的源 CSV。运行多物理数据生成流程后，可使用
`data/datasets/multiphysics/dataset_log.csv` 作为初始数据记录；权重插值使用带
`w_stiffness`/`w_thermal`/`objective` 列的日志，三目标路径使用覆盖多个体积分数的日志。
网络脚本通过 `SOURCE_CSV` 指定源文件。

对没有权重列的 CSV，可以显式传入 `--tri_weight_mode cycle` 或 `random` 做消融；这两种模式对应独立的权重生成策略。

## 结果分析

`plot_iterative_pareto_front.py` 用于绘制 Pareto 前沿投影，
`compute_pareto_hv_c.py` 用于计算 Hypervolume 和 C-metric。示例：

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

三目标分析时将 `--front_mode` 改为 `tri`，并根据 CSV 的字段选择 `--vf_key`、
`--e_key` 和 `--kappa_key`。代码默认把 `thermal`/`kappa` 除以 3 后参与绘图和指标，
这是当前分析脚本的归一化约定；如果你的数据已归一化，请显式设置
`--kappa_divisor 1`。

## 数据与输出

数据生成脚本会在运行时创建 `data/`，并输出 CSV 日志、密度场、网格、manifest 和训练 checkpoint。使用 `python -m maple.tools.repair_dataset_log_paths` 进行路径迁移，使用 `python -m maple.tools.prepare_printable_dataset_release` 进行发布整理。

Stage-II 训练使用目标条件数据：双目标流程使用 `w_shear=0`，三目标流程使用包含剪切目标的日志。迭代流程会在 `output_root` 下保存累计 CSV 和 checkpoint。

## 许可证

本项目采用 MIT License，详见 [`LICENSE`](LICENSE)。

## 引用

如果使用本代码或由此生成的数据，请引用：

论文链接：[Journal of Computer-Aided Design & Computer Graphics](https://www.jcad.cn/article/doi/10.3724/SP.J.1089.2026-00157)；DOI：[10.3724/SP.J.1089.2026-00157](https://dx.doi.org/10.3724/SP.J.1089.2026-00157)

```bibtex
@article{xing2026lattice,
  title   = {Lattice Structure Optimization for Additive Manufacturing: Manufacturability-Driven Design and Pareto Front Construction},
  author  = {Xing, Yu and Liu, Yang and Lu, Lin},
  journal = {Journal of Computer-Aided Design & Computer Graphics},
  year    = {2026},
  doi     = {10.3724/SP.J.1089.2026-00157}
}
```

