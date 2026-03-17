# Deepritz Enhanced

基于 Deep Ritz 思想的 PDE / meshfree 研究项目，当前重点包含：

- Deep Ritz / PINN / VPINN 风格实验
- RKPM / SCNI / meshfree 相关验证
- KAN / RKPM 形函数学习与对比实验
- 统一的训练、积分、采样、可视化模块

## 项目特点

- 模块化：问题、采样、积分、网络、训练、可视化相互解耦
- 可扩展：根目录公共模块可被不同算例复用
- 面向实验：`examples/` 和 `output/` 采用对齐组织，便于横向比较
- 保留轨迹：算例脚本默认保留 `v1`、`v2`、`v3` 等版本演化

## 快速开始

### 1. 安装依赖

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 运行算例

```bash
python examples/meshfree_kan_rkpm_1d_validation/deepritz_meshfree_kan_rkpm_1d_validation_v1.py
```

## 项目结构

```text
Deepritz Enhanced/
├── problems.py
├── samplers.py
├── integrators.py
├── networks.py
├── trainers.py
├── visualizers.py
├── examples/
├── output/
├── tests/
├── docs/
├── paper/
└── legacy/
```

### 根目录模块职责

- `problems.py`：PDE 问题定义、解析解、源项、边界条件
- `samplers.py`：区域采样、边界采样、节点生成等
- `integrators.py`：Monte Carlo、Gauss、SCNI 等积分逻辑
- `networks.py`：MLP、KAN、RKPM 等网络或形函数相关结构
- `trainers.py`：训练流程、损失组织、误差统计
- `visualizers.py`：统一图像保存、训练历史和结果可视化

## 开发约定

下面这些规则用于统一后续开发，默认都按这套来。

### 1. 根目录放公共模块和工具类

- 只要逻辑具备复用价值，优先放到根目录公共模块，不要堆在单个算例脚本里。
- 算例脚本应尽量薄，只负责参数配置、实验组织和结果保存。

### 2. 算例统一放在 `examples/`

- 所有实验入口默认放在 `examples/` 下。
- 简单算例可以直接是单个脚本。
- 复杂主题建议拆成目录，并在目录内放一个 `common.py` 提供共享逻辑。
- `legacy/` 只保留历史代码和旧实验，不作为新开发主路径。

### 3. 命名规则

#### 基本规则

- 文件名统一小写
- 维度之间统一使用下划线 `_`
- 名称尽量完整表达方法、网络、积分方式、问题类型、实验主题
- 新版本默认追加 `_v1`、`_v2`、`_v3`，不直接覆盖旧版本

#### 常见格式

Deep Ritz / VPINN 这类需要显式积分器的脚本，推荐：

```text
{framework}_{network}_{integration}_{problem}_{variant}.py
```

PINN 这类直接计算残差的脚本，通常省略积分维度：

```text
{framework}_{network}_{problem}_{variant}.py
```

#### 词汇约定

- `framework`：`deepritz`、`pinn`、`vpinn`、`fem`
- `network`：`mlp`、`resnet`、`kan`、`kan_serial`、`rbf`、`rkpm`、`meshfree_kan_rkpm`
- `integration`：`mc`、`gauss`、`scni`
- `problem` / `variant`：`poisson`、`heat`、`linear`、`gaussian`、`validation`、`patch_test`、`poisson_convergence`、`stability_ablation`、`irregular_nodes`

#### 当前项目中的典型命名

```text
deepritz_meshfree_kan_rkpm_1d_validation_v1.py
deepritz_meshfree_kan_rkpm_2d_patch_test_v1.py
deepritz_meshfree_kan_rkpm_2d_poisson_convergence_v1.py
deepritz_meshfree_kan_rkpm_2d_stability_ablation_v1.py
deepritz_meshfree_kan_rkpm_2d_irregular_nodes_v1.py
```

### 4. `output/` 与 `examples/` 对齐

- 输出目录必须和算例目录结构对齐，避免实验结果混放。

#### 1D 单脚本类算例

```text
examples/meshfree_kan_rkpm_1d_validation/<script>.py
-> output/meshfree_kan_rkpm_1d_validation/<script>_output/
```

#### 2D 分组实验类算例

```text
examples/meshfree_kan_rkpm_2d_validation/<group>/<script>.py
-> output/meshfree_kan_rkpm_2d_validation/<group>/<case_name>/
```

#### 每次运行建议至少保存

- `config.json`
- `metrics.json`
- `curves.npz`
- `summary.txt`
- `figures/`

### 5. 脚本职责边界

- 算例脚本负责参数、调度、落盘
- 公共算法和共享工具放公共模块
- 同一主题的后续实验优先沿用既有目录和命名体系，保证可搜索、可对比、可批量统计

## 当前算例组织

### 1D 验证

目录：

```text
examples/meshfree_kan_rkpm_1d_validation/
```

特点：

- 同一主题下保留多个版本脚本，如 `v1` 到 `v5`
- 输出目录与脚本名直接对应

### 2D 验证

目录：

```text
examples/meshfree_kan_rkpm_2d_validation/
```

特点：

- 按实验主题拆分子目录
- 共享逻辑集中在 `common.py`
- 当前主题包括：
  - `patch_test`
  - `poisson_convergence`
  - `stability_ablation`
  - `irregular_nodes`

## 测试

当前测试主要放在：

```text
tests/
```

例如：

- `tests/test_meshfree_kan_rkpm_2d_common.py`

新增公共逻辑时，优先补对应测试，减少实验回归风险。

## 代码示例

```python
import torch

from problems import PoissonDisk
from networks import RitzNet
from integrators import MonteCarloIntegrator
from trainers import DeepRitzTrainer

problem = PoissonDisk(radius=1.0)
model = RitzNet(input_dim=2, hidden_dims=[100, 100, 100], output_dim=1)
integrator = MonteCarloIntegrator(n_points=8000, domain_bounds=(-1, 1, -1, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trainer = DeepRitzTrainer(
    model=model,
    problem=problem,
    integrator=integrator,
    optimizer=optimizer,
    device="cuda",
    beta_bc=1000.0,
)

history = trainer.train(n_steps=5000)
```

## 依赖环境

- Python >= 3.8
- PyTorch >= 1.10.0
- NumPy >= 1.21.0
- SciPy >= 1.9.0
- Matplotlib >= 3.6.0

## 文档

- `CLAUDE.md`：历史协作说明
- `examples/README.md`：示例脚本说明
- `docs/`：研究记录、架构说明、文献整理
- `paper/`：论文和汇报相关材料

## 参考文献

1. E, W., and Yu, B. The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems.
2. Chen, J. S., Wu, C. T., Yoon, S., and You, Y. A stabilized conforming nodal integration for Galerkin mesh-free methods.
3. Liu, W. K., Jun, S., and Zhang, Y. F. Reproducing kernel particle methods.

## 贡献

欢迎提交 Issue 和 Pull Request。
