# meshfree_kan_rkpm_2d_validation

二维实验与 `output/meshfree_kan_rkpm_2d_validation/` 一一对应，按实验目标拆成独立子目录：

- `patch_test/`：验证 PU、一阶重构与 `Lambda_h` 热图。
- `poisson_convergence/`：二维 Poisson 主实验与节点加密收敛。
- `stability_ablation/`：`Softplus / No-Softplus / Teacher` 稳定性机制对比。
- `irregular_nodes/`：扰动节点下的鲁棒性实验。

每次运行默认生成：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/*.png`

推荐先跑：

1. `patch_test`
2. `stability_ablation`
3. `poisson_convergence`
4. `irregular_nodes`
