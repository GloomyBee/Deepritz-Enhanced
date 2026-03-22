# trial_space_value one_d

1D trial-space value 主线目录，用于评估 learned basis 作为固定试探空间时的数值价值。

当前活跃结构：

- `common.py`：路径、artifact、payload、落盘 helper
- `basis.py`：复用 shape_validation 的 1D basis 与评估接口
- `training.py`：复用 shape_validation 的 Phase A 训练接口
- `trial_space.py`：固定 basis Poisson 组装与系数求解
- `plotting.py`：case 级主图与诊断图
- `summary_plotting.py`：后续汇总图接口
- `poisson_compare/`：固定 basis Poisson 主入口

输出写入：

- `output/trial_space_value/one_d/poisson_compare/<case_name>/`
