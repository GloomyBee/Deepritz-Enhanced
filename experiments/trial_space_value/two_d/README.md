# trial_space_value two_d

2D trial-space value 主线实验目录，用于评估 learned shape functions 在 Poisson 求解中的实际数值收益。

当前实验组：

- `poisson_compare/`：在同一正弦 Poisson 问题上，对比 classical fixed-basis、frozen-`w` 和 joint Phase B

输出写入：

- `output/trial_space_value/two_d/`

case 级图像布局：

- 根 case 对比图：`figures/main_figure.png`
- 根 case 汇总图：`figures/summary_figure.png`（可选）
- 根 case 诊断图：`figures/diagnostics/`
- 各方法主图：`methods/<method>/figures/main_figure.png`
- 各方法诊断图：`methods/<method>/figures/diagnostics/`

每次运行仍保留：

- `config.json`
- `comparison_metrics.json`
- `comparison_summary.txt`
- `methods/<method>/metrics.json`
- `methods/<method>/curves.npz`
- `methods/<method>/summary.txt`
