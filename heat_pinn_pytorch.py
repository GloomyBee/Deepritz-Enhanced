import torch
import torch.nn as nn
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# 引入你的绘图工具
from compare_viz_square import plot_square_triplet, plot_error_history


# ==========================================
# 0. 基础工具
# ==========================================
def seed_all(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ==========================================
# 1. 问题定义: Patch Test (线性测试)
# ==========================================
class PatchTestProblem:
    """
    Patch Test: u(x,y) = x + y
    目的: 验证模型能否精确拟合线性函数，且二阶导数精确为0。
    """

    def __init__(self):
        self.area_domain = 1.0

    def exact_solution(self, x_tensor):
        x = x_tensor[:, 0:1]
        y = x_tensor[:, 1:2]
        return x + y

    def exact_gradient(self, x_tensor):
        # u = x + y -> du/dx = 1, du/dy = 1
        return torch.ones_like(x_tensor)

    def source_term(self, x_tensor):
        # -Laplace(u) = -(0 + 0) = 0
        return torch.zeros_like(x_tensor[:, 0:1])


# ==========================================
# 2. 采样器 (复用 heat_dr)
# ==========================================
class SquareSampler:
    @staticmethod
    def sample_domain(n_samples):
        return np.random.uniform(0, 1, (n_samples, 2))

    @staticmethod
    def sample_boundary(n_samples):
        n_side = n_samples // 4
        # Bottom, Top, Left, Right
        bot = np.column_stack([np.random.uniform(0, 1, n_side), np.zeros(n_side)])
        top = np.column_stack([np.random.uniform(0, 1, n_side), np.ones(n_side)])
        left = np.column_stack([np.zeros(n_side), np.random.uniform(0, 1, n_side)])
        right = np.column_stack([np.ones(n_side), np.random.uniform(0, 1, n_side)])
        return np.vstack([bot, top, left, right])


# ==========================================
# 3. KAN 组件
# ==========================================
def extend_grid(grid, k_extend=0):
    h = (grid[:, [-1]] - grid[:, [0]]) / (grid.shape[1] - 1)
    for i in range(k_extend):
        grid = torch.cat([grid[:, [0]] - h, grid], dim=1)
        grid = torch.cat([grid, grid[:, [-1]] + h], dim=1)
    return grid


def B_batch(x, grid, k=0):
    x = x.unsqueeze(dim=2)
    grid = grid.unsqueeze(dim=0)
    if k == 0:
        value = (x >= grid[:, :, :-1]) * (x < grid[:, :, 1:])
    else:
        B_km1 = B_batch(x[:, :, 0], grid=grid[0], k=k - 1)
        value = (x - grid[:, :, :-(k + 1)]) / (grid[:, :, k:-1] - grid[:, :, :-(k + 1)]) * B_km1[:, :, :-1] + \
                (grid[:, :, k + 1:] - x) / (grid[:, :, k + 1:] - grid[:, :, 1:(-k)]) * B_km1[:, :, 1:]
    return torch.nan_to_num(value)


class KANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num=5, k=3, grid_range=[0, 1]):
        super(KANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num = num
        self.k = k

        # 扩大一点 grid range 以防止边界数值不稳定
        grid = torch.linspace(grid_range[0], grid_range[1], steps=num + 1)[None, :].expand(self.in_dim, num + 1)
        grid = extend_grid(grid, k_extend=k)
        self.register_buffer('grid', grid)

        num_coef = self.grid.shape[1] - k - 1
        self.coef = nn.Parameter(torch.randn(in_dim, out_dim, num_coef) * 0.1)
        self.scale_base = nn.Parameter(torch.ones(in_dim, out_dim) * 0.1)
        self.base_fun = nn.SiLU()

    def forward(self, x):
        base = self.base_fun(x)
        y_base = torch.einsum('bi,io->bo', base, self.scale_base)
        b_splines = B_batch(x, self.grid, k=self.k)
        y_spline = torch.einsum('bik,iok->bo', b_splines, self.coef)
        return y_base + y_spline


class KAN_PINN(nn.Module):
    """
    单层 KAN 网络结构 (Patch Test 只需要简单结构)
    Structure: [2] -> KANLayer -> [8] -> Linear -> [1]
    """

    def __init__(self):
        super().__init__()
        # 即使是 Patch Test，我们也用稍微宽一点的网格来测试泛化性
        # grid_range 稍微扩大到 [-0.2, 1.2] 以覆盖 [0,1] 的输入
        self.kan1 = KANLayer(in_dim=2, out_dim=8, num=5, k=3, grid_range=[-0.2, 1.2])
        self.out = nn.Linear(8, 1, bias=False)

    def forward(self, x):
        h = self.kan1(x)
        u = self.out(h)
        return u


# ==========================================
# 4. 训练流程 (结构对齐 heat_dr)
# ==========================================
def train():
    seed_all(2024)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 1. Setup
    problem = PatchTestProblem()
    model = KAN_PINN().to(device)

    # KAN 通常需要稍大的 LR，或者特定的优化策略，这里先用 Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    # Config (Steps reduced for Patch Test)
    n_steps = 2000
    batch_domain = 1000
    batch_boundary = 400

    # BC 权重: 在 PINN 中通常需要给 BC 较高的权重
    beta = 100.0

    err_every = 200
    err_samples = 2000

    output_dir = "output_patch_test"
    os.makedirs(output_dir, exist_ok=True)

    # ---------------------------------------------
    # Step A: Pretrain (Boundary Anchoring)
    # ---------------------------------------------
    # KAN 的初始化往往很随机，先固定边界能极大稳定训练
    print(">>> Pretrain: Anchoring Dirichlet boundary (500 steps)")
    for i in range(501):
        x_bd = torch.tensor(SquareSampler.sample_boundary(batch_boundary), dtype=torch.float32, device=device)
        u_pred = model(x_bd)
        u_exact = problem.exact_solution(x_bd)

        loss_pre = torch.mean((u_pred - u_exact) ** 2)

        optimizer.zero_grad()
        loss_pre.backward()
        optimizer.step()

        if i % 100 == 0:
            print(f"Pretrain step {i}: BC loss={loss_pre.item():.6e}")

    # ---------------------------------------------
    # Step B: Main Loop (Physics Informed)
    # ---------------------------------------------
    print("\n>>> Main Loop: Physics + Boundary")
    err_steps, err_l2, err_h1 = [], [], []
    start_time = time.time()

    for step in range(n_steps + 1):
        # 1. Data
        x_dom = torch.tensor(SquareSampler.sample_domain(batch_domain), dtype=torch.float32, device=device)
        x_dom.requires_grad_(True)
        x_bd = torch.tensor(SquareSampler.sample_boundary(batch_boundary), dtype=torch.float32, device=device)

        # 2. PDE Residual (Strong Form for PINN)
        u_dom = model(x_dom)

        # 计算一阶导
        grads = torch.autograd.grad(u_dom, x_dom, torch.ones_like(u_dom), create_graph=True)[0]
        u_x, u_y = grads[:, 0:1], grads[:, 1:2]

        # 计算二阶导
        u_xx = torch.autograd.grad(u_x, x_dom, torch.ones_like(u_x), create_graph=True)[0][:, 0:1]
        u_yy = torch.autograd.grad(u_y, x_dom, torch.ones_like(u_y), create_graph=True)[0][:, 1:2]

        # Residual: -laplace(u) - f = 0
        f_val = problem.source_term(x_dom)
        res = -(u_xx + u_yy) - f_val
        loss_pde = torch.mean(res ** 2)

        # 3. BC Loss
        u_bd_pred = model(x_bd)
        u_bd_exact = problem.exact_solution(x_bd)
        loss_bc = torch.mean((u_bd_pred - u_bd_exact) ** 2)

        # Total Loss
        loss = loss_pde + beta * loss_bc

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 500 == 0 and step > 0:
            scheduler.step()  # 简单的 LR 衰减

        # Logging
        if step % 200 == 0:
            elapsed = time.time() - start_time
            print(f"Step {step}/{n_steps}: Loss={loss.item():.6f} (PDE={loss_pde.item():.6e}, BC={loss_bc.item():.6e})")

        # Evaluation
        if step % err_every == 0 or step == n_steps:
            with torch.no_grad():  # L2 不需要导数图
                pts = torch.tensor(SquareSampler.sample_domain(err_samples), dtype=torch.float32, device=device)
                u_pred = model(pts)
                u_ref = problem.exact_solution(pts)
                diff = u_pred - u_ref
                l2 = torch.sqrt(torch.mean(diff ** 2)).item()

            # H1 需要导数，重新计算
            with torch.enable_grad():
                pts_h1 = torch.tensor(SquareSampler.sample_domain(err_samples), dtype=torch.float32, device=device)
                pts_h1.requires_grad_(True)
                u_h1 = model(pts_h1)
                g_pred = torch.autograd.grad(u_h1, pts_h1, torch.ones_like(u_h1), create_graph=False)[0]
                g_ref = problem.exact_gradient(pts_h1)
                h1_diff = g_pred - g_ref
                h1 = torch.sqrt(torch.mean(torch.sum(h1_diff ** 2, dim=1))).item()

            err_steps.append(step)
            err_l2.append(l2)
            err_h1.append(h1)

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.1f}s")

    # ==========================================
    # 5. 可视化 (复用 compare_viz_square)
    # ==========================================
    x = np.linspace(0, 1, 200)
    y = np.linspace(0, 1, 200)
    X, Y = np.meshgrid(x, y)
    pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]), dtype=torch.float32, device=device)

    with torch.no_grad():
        T_pred = model(pts).cpu().numpy().reshape(X.shape)
        T_exact = problem.exact_solution(pts).cpu().numpy().reshape(X.shape)

    # 1. 场图 (注意 Patch Test 值域是 0~2)
    plot_square_triplet(
        X, Y, T_pred, T_exact,
        val_clim=(0.0, 2.0),  # Patch Test u=x+y, max=2
        err_vmax=0.01,  # 应该非常精确，误差上限设低一点
        title_prefix="KAN PINN (Patch Test)",
        filename=os.path.join(output_dir, "patch_field.png"),
    )
    print(f"Saved {output_dir}/patch_field.png")

    # 2. 误差曲线
    if len(err_steps) > 0:
        plot_error_history(
            np.array(err_steps), np.array(err_l2), np.array(err_h1),
            ylim=(1e-6, 1.0),  # Patch Test 应该能达到很高精度
            title="KAN PINN Patch Test Errors",
            filename=os.path.join(output_dir, "patch_errors.png"),
        )
        print(f"Saved {output_dir}/patch_errors.png")


if __name__ == "__main__":
    train()