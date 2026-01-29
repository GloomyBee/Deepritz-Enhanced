"""
训练器模块 (Trainers)
提供统一的训练框架用于Deep Ritz方法：
1. DeepRitzTrainer - 统一的训练器
2. LossComputer - 损失计算器
3. 能量泛函计算
4. 边界条件处理
5. 误差评估
"""

import torch
import torch.nn as nn
import numpy as np
import time
from typing import Optional


class GradientModelMixin:
    """Models that return (u, u_x, u_y) should set provides_gradients=True."""
    provides_gradients = True


class LossComputer:
    """
    损失计算器
    计算Deep Ritz方法的各项损失：能量泛函、边界条件、初始条件等
    """

    def __init__(self, problem, beta_bc=1000.0, beta_ic=1000.0):
        """
        Args:
            problem: 问题定义对象（包含exact_solution, source_term等方法）
            beta_bc: 边界条件惩罚系数
            beta_ic: 初始条件惩罚系数
        """
        self.problem = problem
        self.beta_bc = beta_bc
        self.beta_ic = beta_ic

    def compute_energy_functional(self, u, u_x, u_y, x_interior, weights=None):
        """
        计算能量泛函
        J = ∫ [0.5 * k * |∇u|² - f * u] dΩ

        Args:
            u: [N, 1] 位移/温度值
            u_x, u_y: [N, 1] 梯度分量
            x_interior: [N, 2] 内部积分点
            weights: [N, 1] 积分权重（可选）
        Returns:
            energy: 能量泛函值
        """
        # 应变能
        grad_sq = u_x ** 2 + u_y ** 2
        strain_energy = 0.5 * self.problem.k * grad_sq

        # 外力功
        f = self.problem.source_term(x_interior)
        force_work = f * u

        # 能量密度（乘以域指示函数）
        indicator = self.problem.domain_indicator(x_interior)
        energy_density = (strain_energy - force_work) * indicator

        # 积分
        if weights is not None:
            energy = torch.sum(energy_density * weights)
        else:
            # Monte Carlo积分：均值 * 区域面积
            energy = torch.mean(energy_density) * self.problem.area_domain

        return energy

    def compute_boundary_loss(self, model, x_boundary):
        """
        计算边界条件损失（Dirichlet边界）
        L_bc = β * mean((u_pred - u_exact)²)

        Args:
            model: 神经网络模型
            x_boundary: [M, 2] 边界点坐标
        Returns:
            loss_bc: 边界损失
        """
        # 预测值
        if getattr(model, "provides_gradients", False):
            # RKPM类型网络返回(u, u_x, u_y)
            u_pred, _, _ = model(x_boundary)
        else:
            u_pred = model(x_boundary)

        # 真实值
        u_exact = self.problem.boundary_condition(x_boundary)
        if u_exact is None:
            raise ValueError("boundary_condition returned None; please define Dirichlet boundary values.")

        # 均方误差
        loss_bc = torch.mean((u_pred - u_exact) ** 2) * self.beta_bc

        return loss_bc

    def compute_total_loss(self, model, x_interior, x_boundary, weights=None):
        """
        计算总损失
        Loss = Energy + Loss_BC

        Args:
            model: 神经网络模型
            x_interior: 内部积分点
            x_boundary: 边界点
            weights: 积分权重
        Returns:
            loss_dict: 包含各项损失的字典
        """
        # 内部能量
        if getattr(model, "provides_gradients", False):
            # RKPM类型：直接返回导数
            u, u_x, u_y = model(x_interior)
        else:
            # 标准网络：需要自动微分计算导数
            x_interior_grad = x_interior.clone().requires_grad_(True)
            u = model(x_interior_grad)
            grads = torch.autograd.grad(
                outputs=u, inputs=x_interior_grad,
                grad_outputs=torch.ones_like(u),
                create_graph=True, retain_graph=True
            )[0]
            u_x = grads[:, 0:1]
            u_y = grads[:, 1:2]

        energy = self.compute_energy_functional(u, u_x, u_y, x_interior, weights)

        # 边界条件
        loss_bc = self.compute_boundary_loss(model, x_boundary)

        # 总损失
        total_loss = energy + loss_bc

        return {
            'total': total_loss,
            'energy': energy,
            'boundary': loss_bc
        }


class ErrorEvaluator:
    """
    误差评估器
    计算L2误差和H1误差
    """

    def __init__(self, problem):
        """
        Args:
            problem: 问题定义对象
        """
        self.problem = problem

    def compute_l2_error(self, model, x_test, device='cpu'):
        """
        计算L2误差
        ||u - u_h||_L2 = sqrt(∫ (u - u_h)² dΩ)

        Args:
            model: 神经网络模型
            x_test: [N, 2] 测试点
            device: 计算设备
        Returns:
            l2_error: L2误差
        """
        with torch.no_grad():
            x_test = x_test.to(device)

            # 预测值
            if getattr(model, "provides_gradients", False):
                u_pred, _, _ = model(x_test)
            else:
                u_pred = model(x_test)

            # 真实值
            u_exact = self.problem.exact_solution(x_test)

            # L2误差
            diff = u_pred - u_exact
            indicator = self.problem.domain_indicator(x_test)
            l2_error = torch.sqrt(torch.mean((diff ** 2) * indicator) * self.problem.area_domain).item()

        return l2_error

    def compute_h1_error(self, model, x_test, device='cpu'):
        """
        计算H1半范数误差
        |u - u_h|_H1 = sqrt(∫ |∇u - ∇u_h|² dΩ)

        Args:
            model: 神经网络模型
            x_test: [N, 2] 测试点
            device: 计算设备
        Returns:
            h1_error: H1误差
        """
        x_test = x_test.to(device)

        # 预测梯度
        if getattr(model, "provides_gradients", False):
            # RKPM类型
            with torch.no_grad():
                _, u_x_pred, u_y_pred = model(x_test)
                grad_pred = torch.cat([u_x_pred, u_y_pred], dim=1)
        else:
            # 标准网络
            x_test_grad = x_test.clone().requires_grad_(True)
            u_pred = model(x_test_grad)
            grad_pred = torch.autograd.grad(
                outputs=u_pred, inputs=x_test_grad,
                grad_outputs=torch.ones_like(u_pred),
                create_graph=False, retain_graph=False
            )[0]

        # 真实梯度
        grad_exact = self.problem.exact_gradient(x_test)

        # H1误差
        diff_grad = grad_pred - grad_exact
        indicator = self.problem.domain_indicator(x_test)
        h1_error = torch.sqrt(torch.mean(torch.sum(diff_grad ** 2, dim=1, keepdim=True) * indicator)
                              * self.problem.area_domain).item()

        return h1_error


class DeepRitzTrainer:
    """
    Deep Ritz统一训练器
    支持多种网络架构和积分方法
    """

    def __init__(
        self,
        model: nn.Module,
        problem,
        integrator,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        beta_bc: float = 1000.0
    ):
        """
        Args:
            model: 神经网络模型
            problem: 问题定义
            integrator: 积分器（MonteCarloIntegrator/GaussIntegrator/SCNIIntegrator）
            optimizer: 优化器
            device: 计算设备
            beta_bc: 边界条件惩罚系数
        """
        self.model = model.to(device)
        self.problem = problem
        self.integrator = integrator
        self.optimizer = optimizer
        self.device = device

        self.loss_computer = LossComputer(problem, beta_bc=beta_bc)
        self.error_evaluator = ErrorEvaluator(problem)

        # 训练历史
        self.history = {
            'steps': [],
            'loss': [],
            'energy': [],
            'boundary': [],
            'l2_error': [],
            'h1_error': []
        }

    def prepare_boundary_points(self, n_boundary=200):
        """默认正方形域边界采样（备用）"""
        bd_pts = np.vstack([
            np.column_stack([np.linspace(0, 1, n_boundary), np.zeros(n_boundary)]),
            np.column_stack([np.linspace(0, 1, n_boundary), np.ones(n_boundary)]),
            np.column_stack([np.zeros(n_boundary), np.linspace(0, 1, n_boundary)]),
            np.column_stack([np.ones(n_boundary), np.linspace(0, 1, n_boundary)])
        ])
        return torch.tensor(bd_pts, dtype=torch.float32, device=self.device)

    def train_step(self, x_interior, x_boundary, weights=None):
        """
        单步训练
        """
        self.optimizer.zero_grad()

        # 计算损失
        loss_dict = self.loss_computer.compute_total_loss(
            self.model, x_interior, x_boundary, weights
        )

        # 反向传播
        loss_dict['total'].backward()
        self.optimizer.step()

        return loss_dict

    def train(
        self,
        n_steps: int = 3000,
        x_boundary: Optional[torch.Tensor] = None,
        n_boundary: int = 400,
        log_interval: int = 100,
        eval_interval: int = 100,
        n_eval_points: int = 500
    ):
        """
        训练主循环

        Args:
            n_steps: 训练步数
            x_boundary: 边界点（如果为None则自动生成）
            log_interval: 日志打印间隔
            eval_interval: 误差评估间隔
            n_eval_points: 评估点数量
        """
        print(f"开始训练 (设备: {self.device})...")
        start_time = time.time()

        # 准备边界点
        if x_boundary is None:
            if hasattr(self.problem, "sample_boundary"):
                x_boundary = self.problem.sample_boundary(n_boundary, device=self.device)
            else:
                x_boundary = self.prepare_boundary_points(n_boundary)

        # 准备积分点和权重
        if hasattr(self.integrator, 'get_points_tensor'):
            # GaussIntegrator
            x_interior = self.integrator.get_points_tensor(self.device)
            weights = self.integrator.get_weights_tensor(self.device)
        elif hasattr(self.integrator, 'sample_points'):
            # MonteCarloIntegrator
            x_interior = self.integrator.sample_points(self.device)
            weights = None
        else:
            # SCNIIntegrator - 特殊处理
            x_interior = None
            weights = None

        for step in range(n_steps):
            # 蒙特卡洛积分需要每步重新采样
            if hasattr(self.integrator, 'sample_points') and step > 0:
                x_interior = self.integrator.sample_points(self.device)

            # 训练步
            loss_dict = self.train_step(x_interior, x_boundary, weights)

            # 记录
            if step % log_interval == 0:
                print(f"Step {step}: Loss={loss_dict['total'].item():.5f}, "
                      f"Energy={loss_dict['energy'].item():.5f}, "
                      f"BC={loss_dict['boundary'].item():.5f}")

            # 误差评估
            if step % eval_interval == 0:
                x_eval = torch.rand(n_eval_points, 2, device=self.device)
                l2_err = self.error_evaluator.compute_l2_error(self.model, x_eval, self.device)
                h1_err = self.error_evaluator.compute_h1_error(self.model, x_eval, self.device)

                self.history['steps'].append(step)
                self.history['loss'].append(loss_dict['total'].item())
                self.history['energy'].append(loss_dict['energy'].item())
                self.history['boundary'].append(loss_dict['boundary'].item())
                self.history['l2_error'].append(l2_err)
                self.history['h1_error'].append(h1_err)

                print(f"   -> L2 Error: {l2_err:.2e}, H1 Error: {h1_err:.2e}")

        elapsed = time.time() - start_time
        print(f"训练完成！用时: {elapsed:.2f}s ({elapsed/n_steps*1000:.2f}ms/step)")

    def get_history(self):
        """返回训练历史"""
        return self.history


class SCNITrainer(DeepRitzTrainer):
    """
    SCNI专用训练器
    处理基于Voronoi单元的稳定化节点积分
    """

    def __init__(
        self,
        model: nn.Module,
        problem,
        integrator,  # SCNIIntegrator
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        beta_bc: float = 5000.0,
        scni_update_freq: int = 20
    ):
        super().__init__(model, problem, integrator, optimizer, device, beta_bc)
        self.scni_update_freq = scni_update_freq
        self.B_matrix = None
        self.node_areas = None

    def update_scni_geometry(self):
        """更新SCNI几何（B矩阵和面积）"""
        self.B_matrix, self.node_areas = self.integrator.compute_B_matrix(
            self.model.rkpm if hasattr(self.model, 'rkpm') else self.model.meshfree_layer,
            device=self.device
        )

    def train_step_scni(self, x_boundary):
        """SCNI专用训练步"""
        self.optimizer.zero_grad()

        # 获取Ritz系数
        if hasattr(self.model, 'linear'):
            W_coeffs = self.model.linear.weight.squeeze(0)
        else:
            raise ValueError("Model must have 'linear' layer for SCNI")

        # 计算节点处的梯度和位移
        grad_nodes = torch.einsum('nij, j -> ni', self.B_matrix, W_coeffs)

        nodes = self.model.rkpm.nodes if hasattr(self.model, 'rkpm') else self.model.meshfree_layer.nodes
        u_nodes, _, _ = self.model(nodes)
        u_nodes = u_nodes.squeeze(1)

        # 能量泛函
        grad_sq = torch.sum(grad_nodes ** 2, dim=1)
        energy_strain = 0.5 * self.problem.k * torch.sum(self.node_areas * grad_sq)

        f_nodes = self.problem.source_term(nodes).squeeze()
        energy_force = torch.sum(self.node_areas * f_nodes * u_nodes)

        # 边界条件
        loss_bc = self.loss_computer.compute_boundary_loss(self.model, x_boundary)

        # 总损失
        loss = energy_strain - energy_force + loss_bc

        loss.backward()
        self.optimizer.step()

        return {
            'total': loss,
            'energy': energy_strain - energy_force,
            'boundary': loss_bc
        }

    def train(self, n_steps=2000, x_boundary=None, log_interval=100, eval_interval=100):
        """SCNI训练主循环"""
        print(f"开始SCNI训练 (设备: {self.device})...")
        start_time = time.time()

        if x_boundary is None:
            if hasattr(self.problem, "sample_boundary"):
                x_boundary = self.problem.sample_boundary(400, device=self.device)
            else:
                x_boundary = self.prepare_boundary_points(400)

        # 初始化几何
        self.update_scni_geometry()

        for step in range(n_steps):
            # 定期更新几何
            if step % self.scni_update_freq == 0 and step > 0:
                self.update_scni_geometry()

            # 训练步
            loss_dict = self.train_step_scni(x_boundary)

            # 记录和评估
            if step % log_interval == 0:
                print(f"Step {step}: Loss={loss_dict['total'].item():.5f}")

            if step % eval_interval == 0:
                x_eval = torch.rand(500, 2, device=self.device)
                l2_err = self.error_evaluator.compute_l2_error(self.model, x_eval, self.device)
                h1_err = self.error_evaluator.compute_h1_error(self.model, x_eval, self.device)

                self.history['steps'].append(step)
                self.history['loss'].append(loss_dict['total'].item())
                self.history['l2_error'].append(l2_err)
                self.history['h1_error'].append(h1_err)

                print(f"   -> L2: {l2_err:.2e}, H1: {h1_err:.2e}")

        elapsed = time.time() - start_time
        print(f"训练完成！用时: {elapsed:.2f}s")


class PINNTrainer:
    """
    PINN训练器（强形式残差 + 边界约束）
    Loss = mean((Δu + f)^2) + beta_bc * mean((u - g)^2)
    """

    def __init__(
        self,
        model: nn.Module,
        problem,
        sampler,
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
        beta_bc: float = 100.0
    ):
        self.model = model.to(device)
        self.problem = problem
        self.sampler = sampler
        self.optimizer = optimizer
        self.device = device
        self.beta_bc = beta_bc

        self.error_evaluator = ErrorEvaluator(problem)
        self.history = {
            'steps': [],
            'loss': [],
            'pde': [],
            'boundary': [],
            'l2_error': [],
            'h1_error': []
        }

    @staticmethod
    def _laplacian(u: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Compute Laplacian of scalar u(x) with autograd."""
        grads = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
        lap = 0.0
        for d in range(grads.shape[1]):
            g = grads[:, d:d + 1]
            g2 = torch.autograd.grad(g, x, torch.ones_like(g), create_graph=True)[0][:, d:d + 1]
            lap = lap + g2
        return lap

    def train(
        self,
        n_steps: int = 2000,
        batch_domain: int = 1000,
        batch_boundary: int = 400,
        log_interval: int = 100,
        eval_interval: int = 200,
        n_eval_points: int = 2000
    ):
        print(f"开始PINN训练 (设备: {self.device})...")
        start_time = time.time()

        for step in range(n_steps + 1):
            x_dom = torch.tensor(self.sampler.sample_domain(batch_domain), dtype=torch.float32, device=self.device)
            x_dom.requires_grad_(True)

            if hasattr(self.problem, "sample_boundary"):
                x_bd = self.problem.sample_boundary(batch_boundary, device=self.device)
            else:
                x_bd = torch.tensor(self.sampler.sample_boundary(batch_boundary), dtype=torch.float32, device=self.device)

            u_dom = self.model(x_dom)
            lap = self._laplacian(u_dom, x_dom)
            f_val = self.problem.source_term(x_dom)
            res = -(lap) - f_val
            loss_pde = torch.mean(res ** 2)

            u_bd = self.model(x_bd)
            g_bd = self.problem.boundary_condition(x_bd)
            if g_bd is None:
                raise ValueError("boundary_condition returned None; please define Dirichlet boundary values.")
            loss_bc = torch.mean((u_bd - g_bd) ** 2)

            loss = loss_pde + self.beta_bc * loss_bc

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if step % log_interval == 0:
                print(f"Step {step}: Loss={loss.item():.6f} (PDE={loss_pde.item():.2e}, BC={loss_bc.item():.2e})")

            if step % eval_interval == 0:
                x_eval = torch.tensor(self.sampler.sample_domain(n_eval_points), dtype=torch.float32, device=self.device)
                l2_err = self.error_evaluator.compute_l2_error(self.model, x_eval, self.device)
                h1_err = self.error_evaluator.compute_h1_error(self.model, x_eval, self.device)

                self.history['steps'].append(step)
                self.history['loss'].append(loss.item())
                self.history['pde'].append(loss_pde.item())
                self.history['boundary'].append(loss_bc.item())
                self.history['l2_error'].append(l2_err)
                self.history['h1_error'].append(h1_err)

                print(f"   -> L2: {l2_err:.2e}, H1: {h1_err:.2e}")

        elapsed = time.time() - start_time
        print(f"训练完成！用时: {elapsed:.2f}s")

    def get_history(self):
        return self.history
