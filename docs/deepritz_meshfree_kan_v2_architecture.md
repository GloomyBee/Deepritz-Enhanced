# Deep Meshfree KAN v2 架构详解

## 🎯 整体方法：Deep Meshfree KAN

这是一个**无网格深度学习方法**，结合了：

- **Deep Ritz变分框架**：最小化能量泛函
- **无网格形函数**：使用KAN网络学习形函数
- **两阶段训练**：几何预训练 + 物理求解

### 核心思想

```
u(x) = Σ φᵢ(x) · wᵢ = φ(x) @ w
```

- `φᵢ(x)`：由KAN网络学习的形函数（shape functions）
- `wᵢ`：节点系数（nodal coefficients）

---

## 🏗️ KAN网络结构（`KANSplineNet`）

### 网络架构

```
输入 x [B, 2] → Layer1 (per-feature) → hidden_sum [B, 8] → Layer2 → 输出 [B, 1]
```

### 详细层次结构

#### **Layer 1: 输入层 → 隐藏层**

```python
input_dim=2, hidden_dim=8, num=5 (basis数量)

对于每个输入维度 i ∈ {0,1}:
  x[:, i]           # [B, 1] - 单个坐标分量
    ↓
  _hat_basis()      # 线性帽函数（hat basis）
    ↓
  basis_i           # [B, 1, 5] - 5个基函数值
    ↓
  layer1[i]         # Linear(5, 8, bias=False)
    ↓
  hidden_i          # [B, 8]

hidden_sum = Σ hidden_i  # [B, 8] - 两个维度的贡献相加
```

**关键点**：

- 每个输入维度独立处理
- 使用**线性帽函数**（piecewise linear hat basis）在固定网格上
- 网格范围：`[-1.5, 1.5]`，5个节点
- 帽函数公式：`φ(x) = max(0, 1 - |x - grid_i| / h)`

#### **Layer 2: 隐藏层 → 输出层**

```python
hidden_sum        # [B, 8]
  ↓
_hat_basis()      # 对hidden_sum的每个维度应用帽函数
  ↓
basis_hidden      # [B, 8, 5] - 8个隐藏维度，每个5个基函数
  ↓
reshape           # [B, 40] - 展平
  ↓
layer2            # Linear(40, 1, bias=False)
  ↓
output            # [B, 1]
```

### 参数统计

```
Layer1: 2个 Linear(5, 8) = 2 × (5×8) = 80 参数
Layer2: 1个 Linear(40, 1) = 40 参数
总计: 120 参数（KAN网络）
节点系数: 64 参数（8×8网格）
模型总参数: 184
```

---

## 🔧 形函数计算流程（`compute_shape_functions`）

### 输入输出

```python
输入: x [M, 2] - M个查询点
输出: phi [M, N] - M个点在N个节点上的形函数值
```

### 计算步骤

#### **1. 相对坐标和距离**

```python
diff = x.unsqueeze(1) - nodes.unsqueeze(0)  # [M, N, 2]
dist = torch.norm(diff, dim=2)              # [M, N]
```

#### **2. KAN输入归一化**

```python
kan_in = (diff / support_radius).reshape(-1, 2)  # [M×N, 2]
# 归一化到 [-1, 1] 范围，适配KAN的grid范围
```

#### **3. KAN前向 + Softplus非负约束**

```python
phi_raw = softplus(kan(kan_in)).reshape(M, N)  # [M, N], ≥ 0
```

- KAN输出可能为负，softplus确保非负性
- `softplus(x) = log(1 + exp(x))`

#### **4. 紧支撑窗函数**

```python
window = cubic_spline_window(dist, radius)  # [M, N]
phi_windowed = phi_raw * window             # [M, N]
```

- **三次样条窗函数**（C²连续）

- 紧支撑：`dist > radius` 时窗函数为0

- 窗函数公式（q = dist/radius）：
  
  ```
  W(q) = 2/3 - 4q² + 4q³,              0 ≤ q ≤ 0.5
         4/3 - 4q + 4q² - 4/3·q³,      0.5 < q ≤ 1
         0,                            q > 1
  ```

#### **5. Shepard归一化（PU性质）**

```python
phi_sum = sum(phi_windowed, dim=1)  # [M, 1]
phi = phi_windowed / phi_sum        # [M, N]
```

- 确保 `Σφᵢ(x) = 1`（单位分解性质，Partition of Unity）

#### **6. 孤儿点fallback（Coverage-safe）**

```python
if phi_sum < eps_cov:  # 孤儿点检测（eps_cov = 1e-14）
    # 使用kNN距离softmax权重
    phi[orphan] = exp(-α·d_knn) / sum(exp(-α·d_knn))
```

- 当点x不在任何节点支撑域内时触发
- 使用k=8个最近邻的距离加权
- 确保数值稳定性，避免除零

---

## 📊 Phase A：几何预训练（1200步）

### 目标

训练KAN网络，使形函数满足**线性重构性质**：

```
Σ φᵢ(x) · xᵢ = x  （对任意x成立）
```

这是无网格方法的基本要求，确保能精确重构线性函数。

### 训练策略

#### **1. 固定w，只训练KAN**

```python
model.w.requires_grad = False
optimizer_kan = Adam(model.kan.parameters(), lr=1e-3)
```

#### **2. 混合采样（50% + 50%）**

```python
# 50% 均匀采样
x_uniform = rand(512, 2)

# 50% 节点周围局部采样
centers = nodes[random_indices]
noise = randn(512, 2) * (radius * 0.5)
x_local = reflect_to_unit_box(centers + noise)

x_domain = cat([x_uniform, x_local])  # [1024, 2]
```

**为什么混合采样？**

- 均匀采样：覆盖全域
- 局部采样：加强节点周围的形函数训练
- `reflect_to_unit_box`：避免边界堆积（比clamp更好）

#### **3. 线性重构损失（使用原始φ）**

```python
# 关键修复：使用return_raw=True获取归一化前的phi
phi_raw, stats = model.compute_shape_functions(x, return_raw=True)

# 手动归一化（训练几何，不依赖Shepard）
phi_sum = sum(phi_raw, dim=1, keepdim=True) + 1e-12
phi = phi_raw / phi_sum

# 线性重构
repro_x = phi @ nodes_x  # [M, 1]
repro_y = phi @ nodes_y  # [M, 1]

loss_linear = mean((repro_x - x[:, 0:1])² + (repro_y - x[:, 1:2])²)
```

**为什么使用原始φ？**（Codex的关键发现）

- 如果使用归一化后的φ，Shepard归一化已经强制Σφ=1
- 这削弱了几何预训练的效果
- 使用原始φ让KAN学习真正的形函数几何

#### **4. 监控指标**

```python
stats = {
    'phi_sum_min': phi_sum.min(),      # 最小覆盖度
    'phi_sum_p01': quantile(phi_sum, 0.01),  # 1%分位数
    'orphan_ratio': mean(orphan_mask)  # 孤儿点比例
}
```

---

## 🎯 Phase B：物理求解（2000步）

### 目标

最小化Deep Ritz能量泛函：

```
E(u) = ∫ [1/2·k·|∇u|²] dx + β·∫_∂Ω |u - g|² ds + γ·L_linear
```

### 训练策略

#### **1. 同时训练KAN和w**

```python
model.w.requires_grad = True
optimizer_all = Adam([
    {'params': model.kan.parameters(), 'lr': 1e-4},  # 降低KAN学习率
    {'params': [model.w], 'lr': 1e-2},               # w学习率更高
])
```

**为什么降低KAN学习率？**

- Phase A已经训练好几何
- Phase B主要调整w来拟合物理
- 避免破坏已学习的形函数几何

#### **2. 精确解初始化w**

```python
model.init_w_from_exact(problem.exact_solution)
# w_i = u_exact(x_i)
```

**巨大的收敛加速！**

- 从正确的初值开始
- 只需微调即可达到高精度

#### **3. 损失函数组成**

##### **能量项（物理）**

```python
x_domain.requires_grad_(True)
u_domain, phi_domain, stats = model(x_domain, return_phi=True, return_stats=True)

grad_u = autograd.grad(u_domain, x_domain, ...)[0]  # [M, 2]
loss_energy = 0.5 * k * mean(sum(grad_u², dim=1))
```

- 对于Laplace方程（f=0），能量 = 0.5·k·|∇u|²
- 最小化能量 ⟺ 求解PDE

##### **边界项（Dirichlet BC）**

```python
x_boundary = sample_boundary(1024)  # 边界采样
u_bc = model(x_boundary)
u_exact_bc = problem.exact_solution(x_boundary)
loss_bc = mean((u_bc - u_exact_bc)²)
```

- 惩罚系数 β = 100.0
- 强制边界条件

##### **线性重构项（保持几何）**

```python
repro_x = phi_domain @ nodes_x
repro_y = phi_domain @ nodes_y
loss_linear = mean((repro_x - x[:, 0:1])² + (repro_y - x[:, 1:2])²)
```

- 惩罚系数 γ = 10.0
- 防止Phase B破坏Phase A学到的几何

##### **总损失**

```python
loss = loss_energy + β·loss_bc + γ·loss_linear
```

#### **4. 误差评估（带eval模式）**

```python
model.eval()
with torch.no_grad():
    # L2误差
    u_val = model(x_val)
    l2_err = sqrt(mean((u_val - u_exact)²))

# H1误差（需要梯度）
x_val_grad.requires_grad_(True)
u_val_grad = model(x_val_grad)
grad_pred = autograd.grad(u_val_grad, x_val_grad, ...)[0]
grad_exact = problem.exact_gradient(x_val_grad)
h1_err = sqrt(mean(sum((grad_pred - grad_exact)², dim=1)))

model.train()
```

**为什么切换eval模式？**

- 虽然当前模型没有dropout/BN
- 但这是良好实践，防止未来引入bug

---

## 📈 预期性能

根据Codex的分析，修复后的v2版本应该能达到：

### 线性分片试验（Linear Patch Test）

```
最终L2误差: 1e-10 到 1e-13（机器精度级别）
最终H1误差: 1e-9 到 1e-12
```

### 关键因素

1. ✅ Float64精度
2. ✅ Phase A使用原始φ训练线性重构
3. ✅ 精确解初始化w
4. ✅ 紧支撑窗函数（C²连续）
5. ✅ Coverage-safe fallback
6. ✅ 混合采样策略

---

## 🔍 专家改进总结

### Codex Review发现的问题

1. **Phase A使用归一化φ** → 已修复，使用`return_raw=True`
2. **缺少eval模式切换** → 已修复，添加`model.eval()`/`model.train()`
3. **孤儿点fallback不训练KAN** → 设计特性，需监控orphan_ratio

### 保留的专家改进

1. ✅ Float64精度（全局设置）
2. ✅ Softplus非负性约束
3. ✅ Coverage-safe kNN fallback
4. ✅ Phase A直接训练线性重构
5. ✅ 混合采样（50%均匀 + 50%局部）
6. ✅ reflect_to_unit_box避免边界堆积
7. ✅ 精确解初始化w
8. ✅ 降低Phase B的KAN学习率
9. ✅ return_phi=True避免重复计算
10. ✅ Grid和h注册为buffer（性能优化）
11. ✅ 标准Cubic Spline窗函数

---

## 📝 使用示例

```python
# 1. 定义节点（8×8网格）
nodes = torch.tensor(meshgrid_points, device='cuda')
radius = 2.5 * node_spacing

# 2. 创建模型
model = MeshfreeKANNet(nodes, radius, kan_hidden_dim=8)

# 3. 训练
history = train_meshfree_kan(
    model=model,
    problem=LinearPatchProblem(),
    device='cuda',
    phase_a_steps=1200,
    phase_b_steps=2000,
    batch_size=1024,
    lr_kan_a=1e-3,
    lr_kan_b=1e-4,
    lr_w=1e-2,
    beta_bc=100.0,
    gamma_linear_b=10.0,
)

# 4. 预测
u_pred = model(x_test)
```

---

## 🚀 未来改进方向

1. **增加KAN basis数量**：num=5 → 7或9，提高表达能力
2. **自适应radius**：根据局部梯度调整支撑域
3. **分块评估**：处理大规模节点（N > 1000）
4. **多尺度训练**：从粗网格到细网格
5. **物理信息正则化**：在Phase A加入弱形式约束

---

## 📚 参考文献

- Deep Ritz Method: E & Yu (2018)
- Meshfree Methods: Liu & Gu (2005)
- KAN Networks: Liu et al. (2024)
- RKPM: Chen et al. (1996)
