# 双目标实验：E–κ

本目录对应固定体积分数下的刚度–导热性能权衡。代码中使用的字段名是
`stiffness` 和 `thermal`；论文中的 E 和 κ 分别对应这两个字段。

双目标标量化目标可写为 `J = w_E E + w_κ κ`，其中 `w_E+w_κ=1`，并且
`w_G=0`。论文主实验常用固定 `vf=0.25`；实际运行时以命令行参数为准。

## 入口

- `run_weight_sweep.sh`：调用 `maple.data.build_weight_sweep_dataset`，在固定 `vf` 下扫描
  `stiffness:w,thermal:1-w`。
- `run_weight_sweep_unconstrained.sh`：调用
  `maple.data.build_weight_sweep_dataset_unconstrained`，生成不加入制造约束的双目标基线。
- `run_network_densification.sh`：调用
  `maple.pareto.run_iterative_property_network_densification --front_objectives weighted`，使用
  Stage-II 条件生成网络对已有前沿进行增密。

## 示例

```bash
# 先生成固定 vf 的初始双目标数据集（默认 vf=0.25）。
bash experiments/biobjective/run_weight_sweep.sh \
  --w_start 0 --w_end 100 --w_step 5 --seeds_per_w 10

# 可选：生成无制造约束的对照组。
bash experiments/biobjective/run_weight_sweep_unconstrained.sh \
  --w_start 0 --w_end 100 --w_step 5 --seeds_per_w 10

# 用已有的初始 CSV 训练/加载网络，并进行 5 轮前沿增密。
SOURCE_CSV=./data/results/biobjective-weight-sweep/dataset_log.csv \
OUT_ROOT=./data/results/biobjective-network \
bash experiments/biobjective/run_network_densification.sh
```

网络增密使用包含 `w_stiffness`/`w_thermal` 或 `objective` 列的本地 CSV，
并读取其中的 `rho_final_npy`/`final_obj` 路径。推荐使用上一步生成的
`./data/results/biobjective-weight-sweep/dataset_log.csv`；历史日志可通过
`python -m maple.tools.repair_dataset_log_paths` 迁移到本地数据目录。
