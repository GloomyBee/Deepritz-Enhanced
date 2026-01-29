# Deepritz Enhanced - 物理信息神经网络研究项目

基于Deep Ritz方法的偏微分方程求解器，支持多种变分方法和网络架构。

## 项目特点

- ✅ **模块化设计**: 核心功能抽象为独立模块，易于扩展
- ✅ **多种方法**: Deep Ritz, RKPM, SCNI等前沿方法
- ✅ **丰富算例**: 从基础Poisson方程到高梯度奇异问题
- ✅ **完整文档**: 详细的中文注释和使用说明
- ✅ **可视化工具**: 内置训练历史和解对比可视化

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行示例

```bash
cd examples
python example_poisson_disk.py
```

## 项目结构

```
Deepritz Enhanced/
├── problems.py          # 问题定义模块
├── samplers.py          # 采样器模块
├── integrators.py       # 积分器模块
├── networks.py          # 神经网络模块
├── trainers.py          # 训练器模块
├── visualizers.py       # 可视化模块
│
├── examples/            # 示例脚本
│   ├── README.md
│   ├── example_poisson_disk.py
│   ├── example_sinusoidal_square.py
│   ├── example_linear_patch.py
│   └── example_gaussian_peak.py
│
├── legacy/              # 原始算例代码（已重构）
│
├── requirements.txt     # 依赖列表
├── CLAUDE.md           # Claude Code使用指南
└── README.md           # 本文件
```

## 核心模块

### problems.py - 问题定义
包含所有PDE问题的定义：
- `PoissonDisk`: 圆盘域Poisson方程
- `SinusoidalSquare`: 方形域正弦问题
- `HeatTrapezoid`: 梯形域热传导
- `LinearPatchProblem`: 线性分片试验
- `GaussianPeak2D`: 高斯峰问题

### samplers.py - 采样器
域采样和无网格节点生成：
- `DiskSampler`: 圆盘域采样
- `SquareSampler`: 方形域采样
- `TrapezoidSampler`: 梯形域采样
- `MeshfreeUtils`: RKPM节点和Voronoi图工具

### integrators.py - 积分器
数值积分方法：
- `MonteCarloIntegrator`: 蒙特卡洛积分
- `GaussIntegrator`: 高斯积分（背景网格）
- `SCNIIntegrator`: 稳定协调节点积分

### networks.py - 神经网络
多种网络架构：
- `RitzNet`: 全连接残差网络
- `RBFNet`: 径向基函数网络
- `KANNet`: Kolmogorov-Arnold网络
- `RKPMNet`: 再生核粒子法网络

### trainers.py - 训练器
统一的训练接口：
- `DeepRitzTrainer`: Deep Ritz方法训练器
- `SCNITrainer`: SCNI方法训练器（基于Voronoi的节点积分）
- 自动误差评估（L2, H1）

### visualizers.py - 可视化
结果可视化工具：
- `plot_training_history`: 训练历史曲线
- `plot_square_triplet`: 预测/解析/误差三联图

## 算例命名规范

为了系统化管理算例，本项目采用统一的命名规范，文件名包含完整的技术栈信息。

### 命名格式

**Deep Ritz / VPINN**（需要积分器）：
```
{framework}_{network}_{integration}_{problem}_{variant}.py
```

**PINN**（直接计算残差，省略积分维度）：
```
{framework}_{network}_{problem}_{variant}.py
```

### 维度词汇表

#### 1. Framework（基础框架）
- `deepritz` - Deep Ritz方法（变分形式，最小化能量泛函）
- `pinn` - 物理信息神经网络（强形式，最小化PDE残差）
- `vpinn` - 变分物理信息神经网络
- `fem` - 有限元法（参考基准）

#### 2. Network（神经网络结构）
- `mlp` - 多层感知机（全连接网络）
- `resnet` - 残差网络（带残差连接的MLP）
- `kan` - Kolmogorov-Arnold网络（B样条基函数）
- `kan_serial` - 串行KAN（粗糙+精细两阶段）
- `rbf` - 径向基函数网络
- `rkpm` - 再生核粒子法网络

#### 3. Integration（积分方式，PINN省略）
- `mc` - 蒙特卡洛积分（随机采样）
- `gauss` - 高斯积分（背景网格）
- `scni` - 稳定协调节点积分（Voronoi单元）

#### 4. Problem（问题类型）
- `poisson` - Poisson方程
- `heat` - 热传导方程
- `sinusoidal` - 正弦解析解问题
- `linear` - 线性问题
- `gaussian` - 高斯峰问题

#### 5. Variant（问题变体）
- `disk` - 圆盘域
- `square` - 方形域
- `trapezoid` - 梯形域
- `patch` - 分片试验
- `peak` - 高梯度峰值问题

### 命名示例

```python
# Deep Ritz方法示例
deepritz_mlp_mc_poisson_disk.py          # Deep Ritz + MLP + 蒙特卡洛 + Poisson + 圆盘域
deepritz_resnet_gauss_sinusoidal_square.py  # Deep Ritz + ResNet + 高斯积分 + 正弦 + 方形域
deepritz_rkpm_scni_linear_patch.py       # Deep Ritz + RKPM + SCNI + 线性 + 分片试验

# PINN方法示例（省略积分维度）
pinn_mlp_poisson_square.py               # PINN + MLP + Poisson + 方形域
pinn_kan_linear_patch.py                 # PINN + KAN + 线性 + 分片试验
pinn_kan_serial_sinusoidal_square.py    # PINN + 串行KAN + 正弦 + 方形域

# 高级组合
deepritz_kan_mc_gaussian_peak.py         # Deep Ritz + KAN + 蒙特卡洛 + 高斯峰
vpinn_resnet_gauss_heat_trapezoid.py    # VPINN + ResNet + 高斯积分 + 热传导 + 梯形域
```

### 命名规则

1. **全部小写**：所有字母使用小写
2. **下划线分隔**：各维度之间用下划线 `_` 连接
3. **不省略维度**：除PINN的积分维度外，其他维度必须填写
4. **词汇标准化**：严格使用词汇表中的标准词汇
5. **信息完整性**：文件名应包含足够信息以理解算例的完整技术栈

### 优点

- ✅ **信息完整**：一眼看出方法、网络、积分、问题的完整组合
- ✅ **易于对比**：方便对比不同方法在同一问题上的表现
- ✅ **便于搜索**：可以按任意维度搜索和筛选算例
- ✅ **规范统一**：避免命名混乱，便于团队协作

## 使用示例

```python
import torch
from problems import PoissonDisk
from networks import RitzNet
from integrators import MonteCarloIntegrator
from trainers import DeepRitzTrainer

# 1. 定义问题
problem = PoissonDisk(radius=1.0)

# 2. 创建网络
model = RitzNet(input_dim=2, hidden_dims=[100, 100, 100], output_dim=1)

# 3. 创建积分器
integrator = MonteCarloIntegrator(n_points=8000, domain_bounds=(-1, 1, -1, 1))

# 4. 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# 5. 训练
trainer = DeepRitzTrainer(
    model=model,
    problem=problem,
    integrator=integrator,
    optimizer=optimizer,
    device='cuda',
    beta_bc=1000.0
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

- [CLAUDE.md](CLAUDE.md) - Claude Code使用指南
- [examples/README.md](examples/README.md) - 示例脚本详细说明

## 参考文献

1. E, W., & Yu, B. (2018). The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. *Communications in Mathematics and Statistics*, 6(1), 1-12.

2. Chen, J. S., Wu, C. T., Yoon, S., & You, Y. (2001). A stabilized conforming nodal integration for Galerkin mesh-free methods. *International Journal for Numerical Methods in Engineering*, 50(2), 435-466.

3. Liu, W. K., Jun, S., & Zhang, Y. F. (1995). Reproducing kernel particle methods. *International Journal for Numerical Methods in Fluids*, 20(8‐9), 1081-1106.

## 许可证

本项目仅供学术研究使用。

## 贡献

欢迎提交Issue和Pull Request！
