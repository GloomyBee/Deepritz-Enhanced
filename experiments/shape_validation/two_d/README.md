# shape_validation two_d

2D 形函数验证主线目录，输出写入 `output/shape_validation/two_d/`。

当前活跃实验只保留两类节点分布：

- `uniform_nodes/`
- `irregular_nodes/`

其余旧 validation 家族已经迁移到 `archive/shape_validation/two_d/`，默认不再作为主视图入口扩展。

## 单个 case 的主图协议

每个 case 的 `figures/main_figure.png` 固定为 2x2：

- 左上：代表形函数对比（learned vs RKPM）
- 右上：代表 cut 上的 shape error
- 左下：主指标面板
- 右下：training history

2D 特有的补充图全部下沉到 `figures/diagnostics/`：

- representative shape heatmaps
- `PU` residual heatmap
- `linear_x` / `linear_y` residual heatmaps
- `Lambda_h` field
- global shape abs-error field
- loss curves

## 保留产物协议

每次运行保留：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/main_figure.png`
- `figures/diagnostics/*`

`irregular_nodes/` 额外在 group 根目录保留：

- `summary_figure.png`
- `irregular_summary.json`
- `irregular_summary.txt`

## 推荐命令

规则节点单 case：

```bash
python experiments/shape_validation/two_d/uniform_nodes/deepritz_meshfree_kan_rkpm_2d_uniform_nodes_v1.py --phase-a-steps 1 --output-tag smoke
```

非规则节点 sweep：

```bash
python experiments/shape_validation/two_d/irregular_nodes/deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py --jitters 0.0,0.1,0.2 --phase-a-steps 1 --output-tag smoke
```
