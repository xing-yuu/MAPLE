# 三目标实验

本目录包含两个目标定义不同的三目标流程。

## 1. 固定体积分数的 E–G–κ 加权扫描

`run_weight_sweep.sh` 调用 `maple.data.build_triobjective_weight_dataset`，扫描

```text
stiffness:w1, shear:w2, thermal:(1-w1-w2)
```

并固定 `volfrac`。这是三种物性之间的加权标量化数据生成流程，
`E–κ–vf` 三目标 Pareto 前沿由第二条流程构建。

```bash
bash experiments/triobjective/run_weight_sweep.sh \
  --w1_start 0 --w1_end 100 \
  --w2_start 0 --w2_end 100 \
  --w_step 5 --seeds_per_combo 10
```

脚本会自动跳过 `w1+w2>100` 的组合，第三个权重为 `w3=1-w1-w2`。

## 2. E–κ–vf Pareto 前沿增密

`run_e_kappa_vf_densification.sh` 调用
`maple.pareto.run_iterative_property_network_densification`，并显式设置
`--front_objectives e_kappa_vf --sampling_mode adaptive_front`。该路径最大化
刚度和导热性能，同时最小化体积分数；每轮将网络生成样本合并回源 CSV，再继续训练
Stage-II 网络。

```bash
SOURCE_CSV=./data/datasets/multiphysics/dataset_log_weighted.csv \
OUT_ROOT=./data/results/triobjective-e-kappa-vf \
bash experiments/triobjective/run_e_kappa_vf_densification.sh
```

`SOURCE_CSV` 使用包含目标权重列（`w_stiffness`/`w_thermal` 或 `objective`）的日志，
并覆盖多个体积分数。历史日志可先通过 `python -m maple.tools.repair_dataset_log_paths` 迁移路径，
再使用 `python -m maple.tools.estimate_dataset_log_weights` 生成 `dataset_log_weighted.csv`。

方法消融可以在命令末尾追加 `--tri_weight_mode cycle` 或 `random`，使用相应的权重生成策略。

论文实验规模可通过环境变量 `ROUNDS=5`、`SAMPLES_PER_ROUND=200` 设置。首次运行
未提供 `STAGE2_CHECKPOINT` 时，迭代脚本会先从初始 CSV 准备数据并训练初始网络。
