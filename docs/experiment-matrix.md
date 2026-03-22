# 实验矩阵

本文档把“入口脚本 -> 科学问题 -> 关键变量 -> 指标与图”显式映射出来，便于复现实验、整理论文叙事和回看历史结果。

## Shape Validation

| 主线 | 入口脚本 | 研究问题 | 关键自变量 | 核心指标 | 主要输出图 |
| --- | --- | --- | --- | --- | --- |
| `shape_validation/one_d/uniform_nodes` | `experiments/shape_validation/one_d/uniform_nodes/deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py` | 规则节点下，learned 1D shape functions 是否满足分片统一性、线性重构和与 RKPM 的一致性 | `n_nodes`, `support_factor`, `variant`, `steps`, `hidden_dim`, `seed` | `PU residual`, `linear reproduction residual`, `Lambda_h`, learned vs RKPM shape discrepancy | `figures/main_figure.png`, `figures/diagnostics/*` |
| `shape_validation/one_d/nonuniform_nodes` | `experiments/shape_validation/one_d/nonuniform_nodes/deepritz_meshfree_kan_rkpm_1d_nonuniform_nodes_v1.py` | 非规则节点扰动下，1D shape functions 的稳健性如何 | `jitter_factor`, `seeds`, `representative_seed`, `n_nodes`, `support_factor`, `steps` | 同上，并关注跨 seed 波动 | 代表 case 的 `main_figure.png`，以及 group 级 summary 图 |
| `shape_validation/two_d/uniform_nodes` | `experiments/shape_validation/two_d/uniform_nodes/deepritz_meshfree_kan_rkpm_2d_uniform_nodes_v1.py` | 规则二维节点布置下，2D learned shape functions 是否满足结构性约束 | `n_side`, `kappa`, `variant`, `phase_a_steps`, `grid_resolution`, `quadrature_order`, `seed` | `PU residual`, `linear reproduction residual`, `Lambda_h`, learned vs RKPM shape discrepancy | `figures/main_figure.png`, `figures/diagnostics/representative_*`, `PU/linear/Lambda_h` heatmaps |
| `shape_validation/two_d/irregular_nodes` | `experiments/shape_validation/two_d/irregular_nodes/deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py` | 二维非规则节点扰动下，shape function 质量如何退化或保持稳定 | `jitters`, `n_side`, `kappa`, `phase_a_steps`, `seed` | 同上，并关注 jitter 对指标的影响 | 单 case `main_figure.png` + group 级 `summary_figure.png` |

## Trial Space Value

| 主线 | 入口脚本 | 研究问题 | 关键自变量 | 核心指标 | 主要输出图 |
| --- | --- | --- | --- | --- | --- |
| `trial_space_value/one_d/poisson_compare` | `experiments/trial_space_value/one_d/poisson_compare/deepritz_meshfree_kan_rkpm_1d_poisson_compare_v1.py` | learned 1D basis 固定后，作为 Poisson trial space 的效果如何 | `n_nodes`, `support_factor`, `variant`, `steps`, `quadrature_order`, `seed` | `basis_quality`, `relative_l2`, `relative_h1`, boundary residual, stiffness/solve diagnostics | `figures/main_figure.png`, `figures/diagnostics/*` |
| `trial_space_value/two_d/poisson_compare/classical` | `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_classical_v1.py` | 先学形函数，再用常规离散系统直接求系数，2D trial space 效果如何 | `n_side`, `kappa`, `phase_a_steps`, `phase_b_steps`, `quadrature_order`, `eval_resolution`, `seed` | `basis_quality`, `trial_space.relative_l2`, `trial_space.relative_h1`, BC residual, linear-system diagnostics | `figures/main_figure.png`, `figures/diagnostics/solution_triplet.png`, shape representative 图 |
| `trial_space_value/two_d/poisson_compare/frozen_w` | `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_frozen_w_v1.py` | basis 冻结，只优化 `w` 时，trial-space 求解效果如何 | `phase_b_steps`, `lr_w`, `warmup_w_steps`, 其余同上 | 同上，但重点关注 frozen-basis 优化 history | `main_figure.png`, `loss_curves.png`, `solution_triplet.png` |
| `trial_space_value/two_d/poisson_compare/joint` | `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_joint_v1.py` | Phase B 联合更新 basis 与 `w` 时，能否进一步提升 Poisson 解质量 | `phase_b_steps`, `lr_kan_b`, `lr_w`, `gamma_linear`, 其余同上 | 同上，并关注 joint retraining 带来的指标变化 | `main_figure.png`, `loss_curves.png`, `solution_triplet.png` |

## 运行形态

| 入口 | 运行形态 | 说明 |
| --- | --- | --- |
| `shape_validation/one_d/uniform_nodes` | 单 case | 一次运行生成一个 case 目录 |
| `shape_validation/one_d/nonuniform_nodes` | seed sweep | 一次运行按 `--seeds` 生成多 case，并汇总代表 seed |
| `shape_validation/two_d/uniform_nodes` | 单 case | 一次运行生成一个 case 目录 |
| `shape_validation/two_d/irregular_nodes` | jitter sweep | 一次运行按 `--jitters` 生成多 case，并输出 group summary |
| `trial_space_value/one_d/poisson_compare` | 单 case | 固定 basis 的 1D Poisson case |
| `trial_space_value/two_d/poisson_compare/classical` | 单方法单 case | 不再混合输出多方法 |
| `trial_space_value/two_d/poisson_compare/frozen_w` | 单方法单 case | 不再混合输出多方法 |
| `trial_space_value/two_d/poisson_compare/joint` | 单方法单 case | 不再混合输出多方法 |

## 推荐阅读顺序

第一次看实验，建议按下面顺序：

1. 先看 `shape_validation/one_d/uniform_nodes`
2. 再看 `shape_validation/two_d/uniform_nodes`
3. 再看 `shape_validation` 的非规则 sweep
4. 最后看 `trial_space_value`，把 learned basis 的“形函数质量”与“数值价值”连起来
