# shape_validation two_d

2D 形函数主线实验目录，输出写入 `output/shape_validation/two_d/`。

当前实验组：

- `patch_test/`
- `stability_ablation/`
- `poisson_convergence/`
- `irregular_nodes/`
- `consistency_constraints/`

单个 case 的主图协议：

- `figures/main_figure.png`
- `figures/summary_figure.png`：可选汇总图
- `figures/diagnostics/`：热图、误差分解、辅助诊断图

每次运行保留：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`

当前推荐推进顺序：

1. `patch_test`
2. `stability_ablation`
3. `poisson_convergence`
4. `irregular_nodes`
5. `consistency_constraints`
