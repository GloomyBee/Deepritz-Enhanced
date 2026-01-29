# 示例脚本说明

本文件夹包含典型的PINNs求解示例，展示如何使用重构后的核心模块。

## 示例列表

### 1. example_poisson_disk.py - 圆盘域Poisson方程
**难度**: ⭐ 入门级

**问题描述**:
- 方程: -Δu = f(x,y) in Ω = {x ∈ R²: |x| < 1}
- 精确解: u = sin(πx)sin(πy)
- 方法: Deep Ritz + 全连接残差网络

**适合学习**:
- Deep Ritz方法基础
- 蒙特卡洛采样
- 能量泛函计算
- 边界惩罚项

**运行**:
```bash
cd examples
python example_poisson_disk.py
```

---

### 2. example_sinusoidal_square.py - 方形域正弦问题
**难度**: ⭐⭐ 基础级

**问题描述**:
- 方程: -Δu = f(x,y) in Ω = [0,1] × [0,1]
- 精确解: u = sin(πx)sin(πy)
- 边界: u = 0 on ∂Ω

**适合学习**:
- 方形域采样
- 多边界处理
- 精确误差计算

**运行**:
```bash
python example_sinusoidal_square.py
```

---

### 3. example_linear_patch.py - 线性分片试验
**难度**: ⭐⭐⭐ 进阶级

**问题描述**:
- 方程: -Δu = 0 in Ω = [0,1] × [0,1]
- 精确解: u = x + y
- 方法: RKPM + SCNI积分

**适合学习**:
- RKPM形函数构造
- SCNI节点积分
- Voronoi图构建
- 分片试验验证

**期望结果**:
- L2误差应 < 1e-8（机器精度）
- 如果误差过大，说明代码有Bug

**运行**:
```bash
python example_linear_patch.py
```

---

### 4. example_gaussian_peak.py - 高斯峰问题
**难度**: ⭐⭐⭐⭐ 挑战级

**问题描述**:
- 方程: -Δu = f(x,y) in Ω = [0,1] × [0,1]
- 精确解: u = exp(-r²/α²)，α = 0.1
- 特点: 在(0.5, 0.5)处有极陡的峰

**适合学习**:
- 高梯度问题处理
- 网络架构选择
- 采样策略优化
- 误差分析

**挑战**:
- 需要更深更宽的网络
- 需要更多采样点
- 训练时间较长

**运行**:
```bash
python example_gaussian_peak.py
```

---

## 如何编写新的算例

参考上述示例，编写新算例的步骤：

### 步骤1: 定义问题
```python
from problems import PoissonDisk

# 使用已有问题
problem = PoissonDisk(radius=1.0)

# 或自定义问题
from problems import BaseProblem

class MyProblem(BaseProblem):
    def exact_solution(self, x_tensor):
        # 实现精确解
        pass
    
    def source_term(self, x_tensor):
        # 实现源项
        pass
    
    def sample_boundary(self, n, device='cpu'):
        # 实现边界采样
        pass
```

### 步骤2: 创建积分器
```python
from integrators import GaussIntegrator

integrator = GaussIntegrator(nx=16, ny=16, domain_bounds=(0, 1, 0, 1), order=4)
```

### 步骤3: 创建神经网络
```python
from networks import RitzNet

model = RitzNet(
    input_dim=2,
    hidden_dims=[128, 128, 128, 128],
    output_dim=1
).to(device)
```

### 步骤4: 配置优化器
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

### 步骤5: 训练
```python
from trainers import DeepRitzTrainer

trainer = DeepRitzTrainer(
    model=model,
    problem=problem,
    integrator=integrator,
    optimizer=optimizer,
    device=device,
    beta_bc=500.0
)

history = trainer.train(n_steps=5000)
```

### 步骤6: 可视化
```python
from visualizers import plot_training_history

plot_training_history(history, filename='my_problem_history.png')
```

---

## 常见问题

### Q1: 训练不收敛怎么办？
- 检查边界惩罚系数 `beta_bc`（通常500-1000）
- 增加积分点数（如MonteCarloIntegrator的`n_points`）
- 降低学习率 `lr`
- 增加训练步数 `n_steps`

### Q2: 误差很大怎么办？
- 增加网络宽度和深度（`hidden_dims`）
- 增加积分点数或网格密度
- 尝试其他网络架构（RBF, KAN等）

### Q3: 如何使用GPU加速？
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```
确保PyTorch安装了CUDA支持。

### Q4: 如何保存和加载模型？
```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model.load_state_dict(torch.load('model.pth'))
```

---

## 进阶主题

### 使用不同的网络架构
```python
from networks import RBFNet, KANNet, RKPMNet

# RBF网络
centers = torch.rand(100, 2)
model = RBFNet(centers=centers, sigma=0.1)

# KAN网络
model = KANNet(input_dim=2, hidden_dim=32, output_dim=1, num_basis=5)

# RKPM网络
model = RKPMNet(nodes=nodes, support_factor=2.5)
```

### 使用不同的积分方法
```python
from integrators import MonteCarloIntegrator, GaussIntegrator, SCNIIntegrator

# 高斯积分
integrator = GaussIntegrator(nx=16, ny=16, domain_bounds=(0, 1, 0, 1), order=4)

# SCNI积分
integrator = SCNIIntegrator(nodes=nodes_np, domain_bounds=(0, 1, 0, 1))

# 蒙特卡洛积分
integrator = MonteCarloIntegrator(n_points=8000, domain_bounds=(0, 1, 0, 1))
```

---

## 参考文献

1. E, W., & Yu, B. (2018). The deep Ritz method: a deep learning-based numerical algorithm for solving variational problems. *Communications in Mathematics and Statistics*, 6(1), 1-12.

2. Chen, J. S., Wu, C. T., Yoon, S., & You, Y. (2001). A stabilized conforming nodal integration for Galerkin mesh-free methods. *International Journal for Numerical Methods in Engineering*, 50(2), 435-466.

3. Liu, W. K., Jun, S., & Zhang, Y. F. (1995). Reproducing kernel particle methods. *International Journal for Numerical Methods in Fluids*, 20(8‐9), 1081-1106.
