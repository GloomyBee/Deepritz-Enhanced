"""
Example 3: Linear patch test (RKPM + SCNI).
Equation: -Δu = 0 in [0,1]x[0,1]
Exact: u = x + y
"""

import sys
sys.path.append('..')

import os
import torch
import numpy as np
from pathlib import Path
from problems import LinearPatchProblem
from samplers import MeshfreeUtils
from networks import RKPMNet
from integrators import SCNIIntegrator
from trainers import SCNITrainer
from visualizers import plot_training_history, plot_square_triplet, get_example_output_subdir
import visualizers

# ============================================================================
# 0. 项目初始化（种子，设备，路径）
# ============================================================================
# Set random seed
torch.manual_seed(42)
np.random.seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Output directory management - use absolute path
ROOT = Path(__file__).resolve().parents[1]
visualizers.OUTPUT_DIR = str(ROOT / "output")
output_subdir = get_example_output_subdir(__file__)
print(f"Output directory: {visualizers.OUTPUT_DIR}/{output_subdir}/")

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
# 3.RKPM network
# ============================================================================
model = RKPMNet(
    nodes=nodes,
    support_factor=2.5
).to(device)

# ============================================================================
# 4 SCNI integrator + optimizer
# ============================================================================
integrator = SCNIIntegrator(nodes_np, domain_bounds=(0, 1, 0, 1))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ============================================================================
# 5. 训练
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
# 6. 验证结果
# ============================================================================
final_l2_error = history['l2_error'][-1]
print(f"\nFinal L2 error: {final_l2_error:.2e}")


# ============================================================================
# 7. 可视化
# ============================================================================
print("\nGenerating visualizations...")

# 生成测试网格
res = 100
x = np.linspace(0, 1, res)
y = np.linspace(0, 1, res)
X, Y = np.meshgrid(x, y)
x_test = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

# 预测和解析解
model.eval()
with torch.no_grad():
    u_pred, _, _ = model(x_test)  # RKPM返回 (u, u_x, u_y)
    u_pred = u_pred.cpu().numpy().reshape(res, res)
u_exact = problem.exact_solution(x_test).cpu().numpy().reshape(res, res)

# 绘制三联图
plot_square_triplet(
    X, Y, u_pred, u_exact,
    title_prefix="RKPM+SCNI (Linear Patch)",
    filename="solution_triplet.png",
    val_clim=(0.0, 2.0),  # u = x + y 范围 [0, 2]
    err_vmax=1e-6,  # 线性问题误差应该很小
    subdir=output_subdir
)

# Plot training history
plot_training_history(history, filename="training_history.png", subdir=output_subdir)

# Save training history data
trainer.save_history(os.path.join(visualizers.OUTPUT_DIR, output_subdir, "history.npz"))

print("\nDone.")
