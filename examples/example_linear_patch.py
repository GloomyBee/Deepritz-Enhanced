"""
Example 3: Linear patch test (RKPM + SCNI).
Equation: -Δu = 0 in [0,1]x[0,1]
Exact: u = x + y
"""

import sys
sys.path.append('..')

import torch
import numpy as np
from problems import LinearPatchProblem
from samplers import MeshfreeUtils
from networks import RKPMNet
from integrators import SCNIIntegrator
from trainers import SCNITrainer
from visualizers import plot_training_history

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================================
# 1. 定义问题
# ============================================================================
problem = LinearPatchProblem()
print(f"Problem: {problem.name}")
print("Exact: u = x + y")
print("Target error: < 1e-8")

# ============================================================================
# 2. 生成RKPM节点
# ============================================================================
nodes_np = MeshfreeUtils.sample_rkpm_nodes(
    n_total=625,  # 25×25网格
    x_range=(0, 1),
    y_range=(0, 1),
    noise_factor=0.4  # 添加扰动测试鲁棒性
)
print(f"\nRKPM nodes: {len(nodes_np)}")
nodes = torch.tensor(nodes_np, dtype=torch.float32, device=device)

# ============================================================================
# 3. 构建Voronoi图（用于SCNI积分）
# ============================================================================
# 4) RKPM network
model = RKPMNet(
    nodes=nodes,
    support_factor=2.5
).to(device)

# 5) SCNI integrator + optimizer
integrator = SCNIIntegrator(nodes_np, domain_bounds=(0, 1, 0, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================================
# 6. 训练
# ============================================================================
trainer = SCNITrainer(
    model=model,
    problem=problem,
    integrator=integrator,
    optimizer=optimizer,
    device=device,
    beta_bc=1000.0
)

print("\nTraining...")
history = trainer.train(n_steps=5000, log_interval=200, eval_interval=200)

# ============================================================================
# 7. 验证结果
# ============================================================================
final_l2_error = history['l2_error'][-1]
print(f"\nFinal L2 error: {final_l2_error:.2e}")

if final_l2_error < 1e-8:
    print("✓ Patch test passed (error < 1e-8)")
else:
    print("✗ Patch test failed (error > 1e-8)")
    print("  Possible causes:")
    print("  1) RKPM shape functions incorrect")
    print("  2) Support radius too small")

# 可视化
plot_training_history(history, filename="linear_patch_history.png")
