# Deepritz Enhanced

Deepritz Enhanced 是一个面向论文实验的轻量研究仓库，当前主线只围绕两件事展开：

- `shape_validation`：验证 learned meshfree shape functions 的分片统一性、线性重构能力、稳定性和与 RKPM 的差异。
- `trial_space_value`：评估 learned basis 作为 trial space 时，对 Poisson 问题离散求解的数值价值。

仓库已经从“脚本平铺 + 历史实验混杂”收口到“主线优先 + 薄共享层 + 历史归档隔离”的形态。新实验默认进入 `experiments/`；旧脚本、旧方法和不再活跃的变体统一放到 `archive/`。

## 当前结构

```text
Deepritz Enhanced/
├── core/
├── experiments/
│   ├── shape_validation/
│   │   ├── one_d/
│   │   └── two_d/
│   └── trial_space_value/
│       ├── one_d/
│       └── two_d/
├── archive/
│   ├── baselines/
│   ├── legacy/
│   ├── shape_validation/
│   └── trial_space_value/
├── docs/
├── output/
├── tests/
├── AGENTS.md
├── CLAUDE.md
└── requirements.txt
```

## 主线与目录语义

### `experiments/`

当前活跃入口都在这里。主线脚本按“研究问题 + 维度 + 叶子实验入口”组织。

活跃入口如下：

- `experiments/shape_validation/one_d/uniform_nodes/deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py`
- `experiments/shape_validation/one_d/nonuniform_nodes/deepritz_meshfree_kan_rkpm_1d_nonuniform_nodes_v1.py`
- `experiments/shape_validation/two_d/uniform_nodes/deepritz_meshfree_kan_rkpm_2d_uniform_nodes_v1.py`
- `experiments/shape_validation/two_d/irregular_nodes/deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py`
- `experiments/trial_space_value/one_d/poisson_compare/deepritz_meshfree_kan_rkpm_1d_poisson_compare_v1.py`
- `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_classical_v1.py`
- `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_frozen_w_v1.py`
- `experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_joint_v1.py`

### `core/`

`core/` 只应该承载已经被多条主线稳定复用的能力。当前最值得继续沉淀的边界是：

- `core/io_or_artifacts.py`：标准产物目录、`json/npz/txt` 落盘
- `core/plotting.py`：通用 figure 保存
- `core/splines.py`：当前 KAN 使用的 open uniform B-spline 基函数
- `core/numerics/quadrature.py`：通用积分点与权重
- `core/utils.py`：seed、路径和轻量解析 helper

仓库里仍保留了一些历史根模块，如 `problems.py`、`samplers.py`、`integrators.py`、`networks.py`、`trainers.py`、`visualizers.py`。它们目前主要服务旧代码与归档材料；新的主线逻辑不应继续往这些 catch-all 模块里堆。

### `archive/`

`archive/` 表示“可参考、非主线、默认不继续扩展”。这里放：

- 旧 `shape_validation` / `trial_space_value` 入口
- 历史 baseline 脚本
- 早期探索、一次性比较、旧输出叙事

如果你只是首次复现仓库，不需要先看 `archive/`。

## 标准输出协议

所有主线入口都写入标准 case 目录：

- `output/shape_validation/one_d/<group>/<case_name>/`
- `output/shape_validation/two_d/<group>/<case_name>/`
- `output/trial_space_value/one_d/<group>/<case_name>/`
- `output/trial_space_value/two_d/<group>/<case_name>/`

每个 case 目录固定包含：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/main_figure.png`
- `figures/diagnostics/*`

这套协议是当前论文实验的最小稳定接口。文档、测试和汇总脚本都默认依赖它。

## 环境安装

推荐环境：

- Python `>=3.11,<3.13`
- Windows PowerShell 或 Git Bash
- CPU 可运行；脚本会自动选择 `cuda` 或 `cpu`

安装步骤：

```powershell
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

如果你需要特定 CUDA 版本的 PyTorch，先按 PyTorch 官方方式安装匹配 wheel，再执行 `pip install -r requirements.txt`。

## 首次复现路径

建议按下面这条最短路径走，而不是一开始就跑 sweep。

### 1. 跑 1D smoke

```powershell
.\.venv\Scripts\python.exe experiments\shape_validation\one_d\uniform_nodes\deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py `
  --steps 80 `
  --n-nodes 11 `
  --support-factor 2.0 `
  --seed 42 `
  --output-tag readme_smoke
```

### 2. 跑 2D smoke

```powershell
.\.venv\Scripts\python.exe experiments\shape_validation\two_d\uniform_nodes\deepritz_meshfree_kan_rkpm_2d_uniform_nodes_v1.py `
  --phase-a-steps 80 `
  --n-side 9 `
  --grid-resolution 41 `
  --quadrature-order 4 `
  --seed 42 `
  --output-tag readme_smoke
```

### 3. 看结果

先看这两个文件：

- `output/.../summary.txt`
- `output/.../figures/main_figure.png`

再按需打开：

- `metrics.json`：结构化指标
- `figures/diagnostics/*`：补充图
- `curves.npz`：绘图和后处理数组

### 4. 再跑非规则节点和 trial-space 对比

1D 非规则节点 sweep：

```powershell
.\.venv\Scripts\python.exe experiments\shape_validation\one_d\nonuniform_nodes\deepritz_meshfree_kan_rkpm_1d_nonuniform_nodes_v1.py `
  --steps 120 `
  --n-nodes 11 `
  --support-factor 2.0 `
  --jitter-factor 0.1 `
  --seeds 42,43,44 `
  --representative-seed 42 `
  --output-tag readme_sweep
```

2D 非规则节点 sweep：

```powershell
.\.venv\Scripts\python.exe experiments\shape_validation\two_d\irregular_nodes\deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py `
  --jitters 0.0,0.1,0.2 `
  --phase-a-steps 80 `
  --n-side 9 `
  --grid-resolution 41 `
  --quadrature-order 4 `
  --seed 42 `
  --output-tag readme_sweep
```

2D trial-space classical：

```powershell
.\.venv\Scripts\python.exe experiments\trial_space_value\two_d\poisson_compare\deepritz_meshfree_kan_rkpm_2d_poisson_compare_classical_v1.py `
  --phase-a-steps 80 `
  --phase-b-steps 80 `
  --n-side 7 `
  --grid-resolution 41 `
  --eval-resolution 31 `
  --quadrature-order 4 `
  --seed 42 `
  --output-tag readme_trial
```

如果要比较 `frozen_w` 和 `joint`，只需把脚本名替换为对应入口。

## 主线实验说明

### `shape_validation`

目标是回答：“learned shape functions 本身是否满足应有的结构性质？”

当前只保留 4 个活跃入口：

- 1D `uniform_nodes`
- 1D `nonuniform_nodes`
- 2D `uniform_nodes`
- 2D `irregular_nodes`

当前主图语义已经在 1D/2D 之间统一为 2x2：

- 左上：代表形函数对比
- 右上：shape / consistency error
- 左下：主指标面板
- 右下：training history

### `trial_space_value`

目标是回答：“learned basis 作为 trial space，能否带来有意义的 Poisson 离散求解效果？”

当前主线是：

- 1D：fixed-basis Poisson
- 2D：`classical` / `frozen_w` / `joint`

其中：

- `classical`：先学形函数，再用常规离散系统直接求系数
- `frozen_w`：basis 冻结，只优化 `w`
- `joint`：Phase B 联合更新 basis 和 `w`

## 结果阅读顺序

第一次看单个 case，建议按这个顺序：

1. `summary.txt` 看结论性摘要
2. `figures/main_figure.png` 看主图叙事
3. `metrics.json` 看可程序化指标
4. `figures/diagnostics/*` 看维度特有诊断图

`shape_validation` 与 `trial_space_value` 的指标字段结构并不相同：

- `shape_validation` 更强调 `learned / rkpm / comparison`
- `trial_space_value` 更强调 `case / method / basis_quality / trial_space`

具体字段示例见 [docs/result-snapshot.md](docs/result-snapshot.md)。

## 文档索引

- [docs/experiment-matrix.md](docs/experiment-matrix.md)：脚本名到科学问题的映射
- [docs/result-snapshot.md](docs/result-snapshot.md)：最小结果快照与 `metrics.json` 字段说明
- [experiments/shape_validation/one_d/README.md](experiments/shape_validation/one_d/README.md)
- [experiments/shape_validation/two_d/README.md](experiments/shape_validation/two_d/README.md)
- [experiments/trial_space_value/one_d/README.md](experiments/trial_space_value/one_d/README.md)
- [experiments/trial_space_value/two_d/README.md](experiments/trial_space_value/two_d/README.md)

## 测试

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

推荐在调整下列内容后至少跑一次：

- `core/` 公共 helper
- 主线入口的导入路径
- 标准产物协议
- 主图与 diagnostics 绘图逻辑
