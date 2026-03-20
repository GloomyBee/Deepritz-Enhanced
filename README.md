# Deepritz Enhanced

面向论文实验开发的 Deep Ritz / meshfree 研究仓库。

当前主线已经收口为两条研究问题：

- `shape_validation`：先验证 learned shape functions 的结构正确性、一致性与几何鲁棒性
- `trial_space_value`：再评估 learned basis 作为 trial space 的数值价值

仓库默认采用“研究主线优先 + 薄共享层 + 历史归档隔离”的轻量结构，不再把所有复用逻辑继续堆在根目录。

## 当前结构

```text
Deepritz Enhanced/
├── core/
│   ├── io_or_artifacts.py
│   ├── plotting.py
│   ├── utils.py
│   ├── numerics/
│   ├── integrators.py
│   ├── networks.py
│   ├── problems.py
│   ├── samplers.py
│   ├── trainers.py
│   └── visualizers.py
├── experiments/
│   ├── shape_validation/
│   │   ├── one_d/
│   │   └── two_d/
│   └── trial_space_value/
│       ├── one_d/
│       └── two_d/
├── archive/
│   ├── baselines/
│   └── legacy/
├── output/
├── tests/
├── docs/
└── paper/
```

## 目录职责

### `core/`

只放已经被多个实验族证明稳定复用的能力，例如：

- 通用 artifacts 落盘
- 通用 plotting helper
- 基础数值积分和轻量工具
- 仍被多条主线共用的 problems / samplers / integrators / networks / trainers / visualizers

这里不是“大而全框架层”。强绑定某条实验叙事的逻辑，应回收到对应实验目录。

### `experiments/`

默认开发入口，所有活跃实验都放这里。

- `experiments/shape_validation/one_d/`
- `experiments/shape_validation/two_d/`
- `experiments/trial_space_value/one_d/`
- `experiments/trial_space_value/two_d/`

组织轴按研究问题划分，而不是按 `problem/network/trainer` 平铺。

### `archive/`

历史脚本、旧方法、低活跃材料。

- `archive/baselines/`：非当前论文主线的方法对比或探索脚本
- `archive/legacy/`：更早期的历史材料

归档内容默认不继续扩展，只保留可参考性。

## 实验组织规则

### 主线共享层

每条主线最多保留一个主共享层，再辅以少量具名局部模块。

例如：

- `experiments/shape_validation/one_d/common.py`
- `experiments/shape_validation/one_d/basis.py`
- `experiments/shape_validation/one_d/training.py`
- `experiments/shape_validation/one_d/plotting.py`

不再继续堆“主线 common + 子目录 common + 半公共 plotting/training”的多层套娃。

### 命名规则

- 目录名表达研究主题
- 文件名表达具体实验入口
- 版本号只用于实验入口脚本，如 `_v1`
- 公共模块禁止 `_v1/_v2`

典型入口：

```text
experiments/shape_validation/one_d/uniform_nodes/deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py
experiments/shape_validation/two_d/patch_test/deepritz_meshfree_kan_rkpm_2d_patch_test_v1.py
experiments/trial_space_value/one_d/poisson_compare/deepritz_meshfree_kan_rkpm_1d_poisson_compare_v1.py
experiments/trial_space_value/two_d/poisson_compare/deepritz_meshfree_kan_rkpm_2d_poisson_compare_v1.py
```

## 输出约定

`output/` 与实验结构镜像对齐：

```text
experiments/shape_validation/one_d/<group>/<script>.py
-> output/shape_validation/one_d/<group>/<case_name>/

experiments/shape_validation/two_d/<group>/<script>.py
-> output/shape_validation/two_d/<group>/<case_name>/

experiments/trial_space_value/one_d/<group>/<script>.py
-> output/trial_space_value/one_d/<group>/<case_name>/

experiments/trial_space_value/two_d/<group>/<script>.py
-> output/trial_space_value/two_d/<group>/<case_name>/
```

主线实验仍保留既有产物协议：

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/`

trial-space 的比较实验会额外保留：

- `comparison_metrics.json`
- `comparison_summary.txt`
- `methods/<method>/...`

## 快速开始

### 安装依赖

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 运行主线 smoke 示例

```bash
python experiments/shape_validation/one_d/uniform_nodes/deepritz_meshfree_kan_rkpm_1d_uniform_nodes_v1.py --steps 1 --output-tag smoke
python experiments/shape_validation/two_d/patch_test/deepritz_meshfree_kan_rkpm_2d_patch_test_v1.py --phase-a-steps 1 --phase-b-steps 1 --output-tag smoke
python experiments/trial_space_value/one_d/poisson_compare/deepritz_meshfree_kan_rkpm_1d_poisson_compare_v1.py --steps 1 --output-tag smoke
```

## 测试

优先用仓库虚拟环境执行：

```bash
.venv\Scripts\python.exe -m unittest discover tests
```

当前测试重点不是追求高覆盖率，而是保证：

- 新 `core/` 边界稳定
- 主线实验导入路径不回归
- 输出产物协议不回归

## 相关文档

- `experiments/shape_validation/two_d/README.md`
- `experiments/trial_space_value/one_d/README.md`
- `experiments/trial_space_value/two_d/README.md`
- `archive/baselines/README.md`
- `phase1_shape_function_validation_plan.md`
- `paper/`

## 参考文献

1. E, W., and Yu, B. The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems.
2. Chen, J. S., Wu, C. T., Yoon, S., and You, Y. A stabilized conforming nodal integration for Galerkin mesh-free methods.
3. Liu, W. K., Jun, S., and Zhang, Y. F. Reproducing kernel particle methods.
