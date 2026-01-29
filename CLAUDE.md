# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

这是一个**物理信息神经网络（PINNs）研究项目**，使用深度学习方法求解偏微分方程（PDEs），特别是热传导方程和Poisson方程。项目实现了多种变分方法和网络架构的对比研究。

**核心方法**：
- **Deep Ritz方法**：将PDE转化为变分问题，使用神经网络最小化能量泛函
- **Hellinger-Reissner (HR)方法**：混合变分形式，同时求解温度场和热流场
- **有限元法（FEM）**：作为基准参考解

## 运行环境

**依赖安装**：
```bash
pip install -r requirements.txt
```

**核心依赖**：
- PyTorch >= 1.10.0（深度学习框架，自动微分）
- NumPy >= 1.21.0（数值计算）
- SciPy >= 1.9.0（Delaunay三角剖分、KDTree、Voronoi图）
- Matplotlib >= 3.6.0（可视化）

**运行单个求解器**：
```bash
python deepritz.py                    # 原始Deep Ritz基准（圆盘域Poisson方程）
python "heat_dr(exact sol).py"        # Deep Ritz求解热传导（解析解版本）
python heat_dr_rkpm_scni_patchtest(exact sol).py  # RKPM+SCNI方法分片试验
```

## 文件命名规范

### 方法前缀
- `deepritz.py`：原始Deep Ritz实现（圆盘域Poisson方程基准）
- `heat_dr_*.py`：Deep Ritz方法求解热传导问题
- `heat_hr_*.py`：Hellinger-Reissner混合变分方法
- `heat_fem.py`：有限元法基准解
- `fem_reference.py`：FEM参考解的插值工具类

### 网络架构标识
- `kernel`：使用高斯径向基函数（Gaussian RBF）激活
- `staticrbf`/`staticrbf2`/`staticrbf3`：静态RBF网络（Kansa方法）
- `moverbf`：可移动/可学习的RBF中心
- `kan`：Kolmogorov-Arnold Networks（KAN），使用B-样条基函数
- `ak_kan`：自适应KAN网络

### 无网格方法标识
- `rkpm`：Reproducing Kernel Particle Method（再生核粒子法）
  - `rkpm_gauss`：使用高斯核的RKPM
  - `rkpm_gauss_v2`：改进版本
  - `rkpm_gauss_learn`：可学习参数的RKPM
- `scni`：Stabilized Conforming Nodal Integration（稳定协调节点积分）
  - `kan_scni`：KAN网络 + SCNI积分
  - `rkpm_scni`：RKPM + SCNI积分

### 问题类型标识
- `patchtest`：分片试验（Patch Test），验证能否精确重构线性多项式
- `singularityproblem`：奇异性问题/高梯度问题（如高斯峰）

### 特殊后缀
- `(exact sol)`：使用解析解的版本，用于精确误差计算
  - 例如：`heat_dr(exact sol).py` 使用正弦函数解析解
  - 对比：`heat_dr.py` 使用FEM参考解
- `meshgrid`：使用网格化采样策略
- `multiscale`：多尺度方法
- `lazyupdate_nestingmesh`：惰性更新 + 嵌套网格策略

## 代码架构

### 标准求解器结构

每个求解器文件通常包含以下模块：

```python
# 1. 问题定义类
class HeatProblem / SinusoidalProblem / LinearPatchProblem:
    def exact_solution(x_tensor):      # 解析解
    def exact_gradient(x_tensor):      # 解析梯度
    def source_term(x_tensor):         # 源项
    def boundary_condition(x_tensor):  # 边界条件

# 2. 采样器类
class TrapezoidSampler / DataSampler:
    def sample_domain():       # 内部点采样
    def sample_left/top():     # 边界采样

# 3. 神经网络类
class HeatNet / RitzNet / MixedNet:
    class ResidualBlock       # 残差块
    def forward(x):           # 前向传播

# 4. 训练函数
def train():
    # 能量泛函计算
    # 边界惩罚项
    # 误差评估（L2, H1）
    # 可视化输出

# 5. 主函数
if __name__ == "__main__":
    main()
```

### 可视化模块

**`compare_viz.py`**（梯形域）：
- `plot_temperature_trap_with_nodes()`：带关键点标注的温度云图
- `plot_temperature_tri()`：FEM三角网格绘图
- `plot_error_history()`：误差收敛曲线
- 统一色标范围：20-28°C

**`compare_viz_square.py`**（方形域）：
- `plot_square_triplet()`：预测/解析/误差三联图
- 统一色标范围：0-1

### 问题域定义

**梯形域**（主要研究对象）：
- 几何：`x + y ≤ 2.0`，`x ∈ [0,2]`，`y ∈ [0,1]`
- 边界条件：
  - 左边界（x=0）：Dirichlet，T=20°C
  - 顶边界（y=1）：Neumann，q=-100 W/m²
  - 其他边界：绝热
- 物理参数：k=20 W/(m·°C)，s=50 W/m³

**方形域**（用于解析解验证）：
- 几何：`[0,1] × [0,1]`
- 解析解：`u = sin(πx)sin(πy)`

**圆盘域**（原始Deep Ritz基准）：
- 几何：`|x| < R`
- 解析解：`u = sin(πx)sin(πy)`

## 关键技术实现

### 能量泛函计算

Deep Ritz方法的核心是最小化能量泛函：

```
E(u) = ∫_Ω [1/2|∇u|² - fu] dx + λ∫_∂Ω |u-g|² ds
```

实现要点：
1. 使用 `torch.autograd.grad()` 计算梯度 `∇u`
2. 蒙特卡洛积分估计：`∫_Ω f dx ≈ Area(Ω) · mean(f)`
3. 边界惩罚系数 `λ` 通常设置为 500-1000

### 误差评估

**相对L2误差**：
```python
l2_error = ||u - u_exact||_L² / ||u_exact||_L²
```

**相对H1误差**：
```python
h1_error = sqrt(||u - u_exact||_L²² + ||∇u - ∇u_exact||_L²²) / sqrt(||u_exact||_H¹²)
```

注意：计算H1误差时必须保持梯度追踪（不能使用 `torch.no_grad()`）

### RKPM形函数

RKPM方法使用再生核粒子法构造形函数：

```python
# 1. 计算核函数 w(x - x_i)
# 2. 计算修正矩阵 M(x)
# 3. 形函数 ψ_i(x) = C(x) · w(x - x_i)
```

关键参数：
- `dilation`：核函数支撑域半径（通常为节点间距的2-3倍）
- `kernel_type`：核函数类型（cubic spline, quartic spline, Gaussian）

### SCNI积分

稳定协调节点积分使用Voronoi单元进行数值积分：

```python
# 1. 构建Voronoi图
vor = Voronoi(nodes)

# 2. 计算每个节点的Voronoi单元面积
area_i = compute_voronoi_area(node_i)

# 3. 节点积分
∫_Ω f dx ≈ Σ_i f(x_i) · area_i
```

## 开发指南

### 添加新的求解器

1. 复制最接近的现有求解器文件
2. 修改问题定义类（`exact_solution`, `source_term`, `boundary_condition`）
3. 调整网络架构参数（`width`, `depth`）
4. 修改训练参数（`lr`, `train_steps`, `penalty`）
5. 更新可视化调用（选择合适的绘图函数）

### 调试技巧

**检查梯度计算**：
```python
# 在能量泛函计算后添加
print(f"梯度范数: {torch.sqrt(grad_squared).mean().item():.6f}")
print(f"输出均值: {output.mean().item():.6f}")
```

**验证形函数分片性质**（RKPM）：
```python
# 所有形函数之和应为1
psi_sum = torch.sum(psi, dim=1)
print(f"形函数和: max={psi_sum.max():.6f}, min={psi_sum.min():.6f}")
```

**检查Voronoi单元面积**（SCNI）：
```python
# 所有单元面积之和应等于域面积
total_area = sum(voronoi_areas)
print(f"总面积: {total_area:.6f}, 理论值: {domain_area:.6f}")
```

### 性能优化

1. **批量大小**：`body_batch` 通常设置为 4096-8192
2. **采样频率**：每 100-200 步重新采样一次
3. **学习率调度**：使用 `StepLR`，每 500 步衰减 0.5
4. **设备选择**：优先使用 CUDA（`device = 'cuda' if torch.cuda.is_available() else 'cpu'`）

## 输出文件

**数据文件**：
- `fem_reference.npz`：FEM基准解数据

**输出目录**（代码中定义）：
- `output_heat/`：梯形域热传导结果
- `output_heat2/`：方形域解析解结果
- `output_scni_test/`：SCNI方法测试结果

**典型输出**：
- 温度场云图（PNG）
- 误差收敛曲线（PNG）
- 形函数可视化（PNG）
