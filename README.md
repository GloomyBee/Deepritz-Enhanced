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
