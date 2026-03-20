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
        print(f"Starting training (device: {self.device})...")
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
                print(f"Step {step:6d} | total={loss_dict['total'].item():.5e} | "
                      f"energy={loss_dict['energy'].item():.5e} | "
                      f"bc={loss_dict['boundary'].item():.5e}")

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

                print(f"           | L2={l2_err:.2e} | H1={h1_err:.2e}")

        elapsed = time.time() - start_time
        print(f"Training completed! Time: {elapsed:.2f}s ({elapsed/n_steps*1000:.2f}ms/step)")

        return self.history

    def get_history(self):
        """返回训练历史"""
        return self.history

    def save_history(self, filepath: str):
        """
        保存训练历史到文件

        Args:
            filepath: 保存路径（如 "output/example_output/history.npz"）
        """
        import os
        dirname = os.path.dirname(filepath)
        if dirname:  # Only create directory if dirname is not empty
            os.makedirs(dirname, exist_ok=True)
        np.savez(
            filepath,
            steps=self.history['steps'],
            loss=self.history['loss'],
            energy=self.history.get('energy', []),
            boundary=self.history.get('boundary', []),
            l2_error=self.history['l2_error'],
            h1_error=self.history['h1_error']
        )
        print(f"Training history saved: {filepath}")


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
        print(f"Starting SCNI training (device: {self.device})...")
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
                print(f"Step {step:6d} | total={loss_dict['total'].item():.5e} | "
                      f"energy={loss_dict['energy'].item():.5e} | "
                      f"bc={loss_dict['boundary'].item():.5e}")

            if step % eval_interval == 0:
                x_eval = torch.rand(500, 2, device=self.device)
                l2_err = self.error_evaluator.compute_l2_error(self.model, x_eval, self.device)
                h1_err = self.error_evaluator.compute_h1_error(self.model, x_eval, self.device)

                self.history['steps'].append(step)
                self.history['loss'].append(loss_dict['total'].item())
                self.history['l2_error'].append(l2_err)
                self.history['h1_error'].append(h1_err)

                print(f"           | L2={l2_err:.2e} | H1={h1_err:.2e}")

        elapsed = time.time() - start_time
        print(f"Training completed! Time: {elapsed:.2f}s")

        return self.history


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
        print(f"Starting PINN training (device: {self.device})...")
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
                print(f"Step {step:6d} | total={loss.item():.5e} | "
                      f"pde={loss_pde.item():.5e} | "
                      f"bc={loss_bc.item():.5e}")

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

                print(f"           | L2={l2_err:.2e} | H1={h1_err:.2e}")

        elapsed = time.time() - start_time
        print(f"Training completed! Time: {elapsed:.2f}s")

        return self.history

    def get_history(self):
        return self.history

    def save_history(self, filepath: str):
        """
        保存训练历史到文件

        Args:
            filepath: 保存路径（如 "output/example_output/history.npz"）
        """
        import os
        dirname = os.path.dirname(filepath)
        if dirname:  # Only create directory if dirname is not empty
            os.makedirs(dirname, exist_ok=True)
        np.savez(
            filepath,
            steps=self.history['steps'],
            loss=self.history['loss'],
            pde=self.history.get('pde', []),
            boundary=self.history.get('boundary', []),
            l2_error=self.history['l2_error'],
            h1_error=self.history['h1_error']
        )
        print(f"Training history saved: {filepath}")


class SerialKANTrainer:
    """
    Serial KAN PINN Trainer (Two-phase training: coarse + fine)
    Implements coarse-to-fine decomposition with frozen coarse network in phase 2.
    """

    def __init__(
        self,
        model: nn.Module,
        problem,
        sampler,
        optimizer_coarse: torch.optim.Optimizer,
        optimizer_fine: torch.optim.Optimizer,
        device: str = 'cpu',
        beta_bc: float = 100.0,
        lambda_fine_reg: float = 0.0,
        lambda_fine_bc: float = 1.0
    ):
        """
        Args:
            model: KANSerialPINN model with coarse/fine/total methods
            problem: Problem definition
            sampler: Domain/boundary sampler
            optimizer_coarse: Optimizer for coarse network
            optimizer_fine: Optimizer for fine network
            device: Computing device
            beta_bc: Boundary condition penalty coefficient
            lambda_fine_reg: Fine network regularization coefficient
            lambda_fine_bc: Fine network boundary penalty coefficient
        """
        self.model = model.to(device)
        self.problem = problem
        self.sampler = sampler
        self.optimizer_coarse = optimizer_coarse
        self.optimizer_fine = optimizer_fine
        self.device = device
        self.beta_bc = beta_bc
        self.lambda_fine_reg = lambda_fine_reg
        self.lambda_fine_bc = lambda_fine_bc

        self.error_evaluator = ErrorEvaluator(problem)
        self.history = {
            'steps': [],
            'loss': [],
            'pde': [],
            'boundary': [],
            'l2_error': [],
            'h1_error': []
        }

    def _evaluate_model(self, model_fn, x_eval):
        """
        Evaluate model using a specific forward function (coarse or total).

        Args:
            model_fn: Model function to use (e.g., self.model.coarse or self.model.total)
            x_eval: Evaluation points

        Returns:
            l2_error, h1_error: L2 and H1 errors
        """
        # L2 error
        self.model.eval()
        with torch.no_grad():
            u_pred = model_fn(x_eval)
            u_exact = self.problem.exact_solution(x_eval)
            diff = u_pred - u_exact
            indicator = self.problem.domain_indicator(x_eval)
            l2_error = torch.sqrt(torch.mean((diff ** 2) * indicator) * self.problem.area_domain).item()

        # H1 error
        x_eval_grad = x_eval.clone().requires_grad_(True)
        u_pred_grad = model_fn(x_eval_grad)
        grad_pred = torch.autograd.grad(
            outputs=u_pred_grad, inputs=x_eval_grad,
            grad_outputs=torch.ones_like(u_pred_grad),
            create_graph=False, retain_graph=False
        )[0]
        grad_exact = self.problem.exact_gradient(x_eval)
        diff_grad = grad_pred - grad_exact
        indicator = self.problem.domain_indicator(x_eval)
        h1_error = torch.sqrt(torch.mean(torch.sum(diff_grad ** 2, dim=1, keepdim=True) * indicator)
                              * self.problem.area_domain).item()

        self.model.train()
        return l2_error, h1_error

    def train_pretrain(self, n_steps: int, batch_boundary: int, log_interval: int = 100):
        """
        Pretrain coarse network on boundary conditions.

        Args:
            n_steps: Number of pretrain steps
            batch_boundary: Boundary batch size
            log_interval: Logging interval
        """
        print(f">>> Pretrain: {n_steps} steps")
        self.model.unfreeze_coarse_params()

        for step in range(n_steps + 1):
            x_bd = torch.tensor(self.sampler.sample_boundary(batch_boundary),
                              dtype=torch.float32, device=self.device)
            u_pred = self.model.coarse(x_bd)
            u_exact = self.problem.exact_solution(x_bd)
            loss_pre = torch.mean((u_pred - u_exact) ** 2)

            self.optimizer_coarse.zero_grad()
            loss_pre.backward()
            self.optimizer_coarse.step()

            if step % log_interval == 0:
                print(f"Pretrain step {step}: BC loss={loss_pre.item():.6e}")

    def train_phase1_coarse(
        self,
        n_steps: int,
        batch_domain: int,
        batch_boundary: int,
        log_interval: int = 100,
        eval_interval: int = 200,
        n_eval_points: int = 2000,
        pretrain_steps: int = 0
    ):
        """
        Phase 1: Train coarse network with PDE residual + boundary loss.

        Args:
            n_steps: Number of training steps
            batch_domain: Domain batch size
            batch_boundary: Boundary batch size
            log_interval: Logging interval
            eval_interval: Evaluation interval
            n_eval_points: Number of evaluation points
            pretrain_steps: Offset for step counting (to continue from pretrain)
        """
        print(f"\n>>> Phase 1: COARSE {n_steps} steps")

        for step in range(n_steps + 1):
            x_dom = torch.tensor(self.sampler.sample_domain(batch_domain),
                               dtype=torch.float32, device=self.device)
            x_dom.requires_grad_(True)
            x_bd = torch.tensor(self.sampler.sample_boundary(batch_boundary),
                              dtype=torch.float32, device=self.device)

            # PDE residual
            u_dom = self.model.coarse(x_dom)
            lap = PINNTrainer._laplacian(u_dom, x_dom)
            res = -lap - self.problem.source_term(x_dom)
            loss_pde = torch.mean(res ** 2)

            # Boundary loss
            u_bd = self.model.coarse(x_bd)
            loss_bc = torch.mean((u_bd - self.problem.exact_solution(x_bd)) ** 2)

            # Total loss
            loss = loss_pde + self.beta_bc * loss_bc

            self.optimizer_coarse.zero_grad()
            loss.backward()
            self.optimizer_coarse.step()

            # Logging
            if step % log_interval == 0:
                print(f"Step {step:6d} | total={loss.item():.5e} | "
                      f"pde={loss_pde.item():.5e} | "
                      f"bc={loss_bc.item():.5e}")

            # Evaluation
            if step % eval_interval == 0:
                x_eval = torch.tensor(self.sampler.sample_domain(n_eval_points),
                                    dtype=torch.float32, device=self.device)
                l2_err, h1_err = self._evaluate_model(self.model.coarse, x_eval)

                self.history['steps'].append(pretrain_steps + step)
                self.history['loss'].append(loss.item())
                self.history['pde'].append(loss_pde.item())
                self.history['boundary'].append(loss_bc.item())
                self.history['l2_error'].append(l2_err)
                self.history['h1_error'].append(h1_err)

                print(f"           | L2={l2_err:.2e} | H1={h1_err:.2e}")

    def train_phase2_fine(
        self,
        n_steps: int,
        batch_domain: int,
        batch_boundary: int,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_interval: int = 1000,
        log_interval: int = 100,
        eval_interval: int = 200,
        n_eval_points: int = 2000,
        phase1_steps: int = 0
    ):
        """
        Phase 2: Train fine network with coarse frozen.

        Args:
            n_steps: Number of training steps
            batch_domain: Domain batch size
            batch_boundary: Boundary batch size
            scheduler: Learning rate scheduler (optional)
            scheduler_interval: Scheduler step interval
            log_interval: Logging interval
            eval_interval: Evaluation interval
            n_eval_points: Number of evaluation points
            phase1_steps: Offset for step counting (to continue from phase 1)
        """
        print(f"\n>>> Phase 2: FINE {n_steps} steps")
        self.model.freeze_coarse_params()

        for step in range(n_steps + 1):
            x_dom = torch.tensor(self.sampler.sample_domain(batch_domain),
                               dtype=torch.float32, device=self.device)
            x_dom.requires_grad_(True)
            x_bd = torch.tensor(self.sampler.sample_boundary(batch_boundary),
                              dtype=torch.float32, device=self.device)

            # PDE residual (total = coarse + fine)
            u_dom_total = self.model.total(x_dom)
            lap = PINNTrainer._laplacian(u_dom_total, x_dom)
            res = -lap - self.problem.source_term(x_dom)
            loss_pde = torch.mean(res ** 2)

            # Boundary loss (total)
            u_bd_total = self.model.total(x_bd)
            loss_bc = torch.mean((u_bd_total - self.problem.exact_solution(x_bd)) ** 2)

            # Fine network regularization
            u_f_dom = self.model.fine(x_dom)
            u_f_bd = self.model.fine(x_bd)
            loss_freg = torch.mean(u_f_dom ** 2)
            loss_fbc = torch.mean(u_f_bd ** 2)

            # Total loss
            loss = loss_pde + self.beta_bc * loss_bc
            loss = loss + self.lambda_fine_reg * loss_freg + self.lambda_fine_bc * loss_fbc

            self.optimizer_fine.zero_grad()
            loss.backward()
            self.optimizer_fine.step()

            # Learning rate scheduling
            if scheduler is not None and step > 0 and step % scheduler_interval == 0:
                scheduler.step()

            # Logging
            if step % log_interval == 0:
                print(f"Step {step:6d} | total={loss.item():.5e} | "
                      f"pde={loss_pde.item():.5e} | "
                      f"bc={loss_bc.item():.5e}")

            # Evaluation
            if step % eval_interval == 0:
                x_eval = torch.tensor(self.sampler.sample_domain(n_eval_points),
                                    dtype=torch.float32, device=self.device)
                l2_err, h1_err = self._evaluate_model(self.model.total, x_eval)

                self.history['steps'].append(phase1_steps + step)
                self.history['loss'].append(loss.item())
                self.history['pde'].append(loss_pde.item())
                self.history['boundary'].append(loss_bc.item())
                self.history['l2_error'].append(l2_err)
                self.history['h1_error'].append(h1_err)

                print(f"           | L2={l2_err:.2e} | H1={h1_err:.2e}")

    def train(
        self,
        pretrain_steps: int = 500,
        phase1_steps: int = 1000,
        phase2_steps: int = 4000,
        batch_domain: int = 2000,
        batch_boundary: int = 500,
        scheduler_fine: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        scheduler_interval: int = 1000,
        log_interval: int = 100,
        eval_interval: int = 200,
        n_eval_points: int = 2000
    ):
        """
        Main training loop with three phases: pretrain → phase1_coarse → phase2_fine.

        Args:
            pretrain_steps: Number of pretrain steps
            phase1_steps: Number of phase 1 steps
            phase2_steps: Number of phase 2 steps
            batch_domain: Domain batch size
            batch_boundary: Boundary batch size
            scheduler_fine: Learning rate scheduler for fine network
            scheduler_interval: Scheduler step interval
            log_interval: Logging interval
            eval_interval: Evaluation interval
            n_eval_points: Number of evaluation points

        Returns:
            history: Training history dictionary
        """
        print(f"Starting Serial KAN PINN training (device: {self.device})...")
        start_time = time.time()

        # Phase 0: Pretrain
        self.train_pretrain(pretrain_steps, batch_boundary, log_interval=100)

        # Phase 1: Coarse
        self.train_phase1_coarse(
            phase1_steps, batch_domain, batch_boundary,
            log_interval, eval_interval, n_eval_points,
            pretrain_steps=pretrain_steps
        )

        # Phase 2: Fine
        self.train_phase2_fine(
            phase2_steps, batch_domain, batch_boundary,
            scheduler_fine, scheduler_interval,
            log_interval, eval_interval, n_eval_points,
            phase1_steps=pretrain_steps + phase1_steps
        )

        elapsed = time.time() - start_time
        print(f"Training completed! Time: {elapsed:.2f}s")

        return self.history

    def get_history(self):
        """Return training history."""
        return self.history

    def save_history(self, filepath: str):
        """
        Save training history to file.

        Args:
            filepath: Save path (e.g., "output/example_output/history.npz")
        """
        import os
        dirname = os.path.dirname(filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        np.savez(
            filepath,
            steps=self.history['steps'],
            loss=self.history['loss'],
            pde=self.history.get('pde', []),
            boundary=self.history.get('boundary', []),
            l2_error=self.history['l2_error'],
            h1_error=self.history['h1_error']
        )
        print(f"Training history saved: {filepath}")
