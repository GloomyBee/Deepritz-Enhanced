# trial_space_value two_d

2D trial-space value 主线目录，用于评估 learned basis 作为 Poisson trial space 的数值价值。

当前活跃结构：

- `common.py`：路径、artifact、payload、落盘 helper
- `basis.py`：meshfree basis、RKPM、quadrature、basis-quality 评估
- `training.py`：Phase A basis 训练与 `frozen_w/joint` 的 Phase B 训练
- `trial_space.py`：固定 basis 组装、线性系统、trial-space 指标、方法调度
- `plotting.py`：case 级主图与 diagnostics
- `summary_plotting.py`：跨方法汇总图接口
- `poisson_compare/`：三种显式入口脚本

当前活跃入口：

- `deepritz_meshfree_kan_rkpm_2d_poisson_compare_classical_v1.py`
- `deepritz_meshfree_kan_rkpm_2d_poisson_compare_frozen_w_v1.py`
- `deepritz_meshfree_kan_rkpm_2d_poisson_compare_joint_v1.py`

输出写入：

- `output/trial_space_value/two_d/poisson_compare/<case_name>/`

每次运行固定产出：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/main_figure.png`
- `figures/diagnostics/*`
