"""
问题定义模块

包含所有PDE问题的定义，每个问题类提供：
- exact_solution: 解析解
- exact_gradient: 解析梯度
- source_term: 源项
- boundary_condition: 边界条件
"""

import torch
import math
import numpy as np


# ============================================================================
# 基类定义
# ============================================================================

class BaseProblem:
    """所有问题的基类"""

    k = 1.0
    area_domain = 1.0

    def exact_solution(self, x_tensor):
        """
        计算精确解
        Args:
            x_tensor: 输入坐标 shape=(N, dim)
        Returns:
            精确解值 shape=(N, 1)
        """
        raise NotImplementedError

    def exact_gradient(self, x_tensor):
        """
        计算精确解的梯度
        Args:
            x_tensor: 输入坐标 shape=(N, dim)
        Returns:
            梯度值 shape=(N, dim)
        """
        raise NotImplementedError

    def source_term(self, x_tensor):
        """
        计算源项
        Args:
            x_tensor: 输入坐标 shape=(N, dim)
        Returns:
            源项值 shape=(N, 1)
        """
        raise NotImplementedError

    def boundary_condition(self, x_tensor):
        """
        计算边界条件（默认使用精确解）
        Args:
            x_tensor: 边界点坐标 shape=(N, dim)
        Returns:
            边界值 shape=(N, 1)
        """
        return self.exact_solution(x_tensor)

    def sample_boundary(self, n, device='cpu'):
        """采样边界点"""
        raise NotImplementedError

    def domain_indicator(self, x_tensor):
        """域指示函数（默认全域为1）"""
        return torch.ones((x_tensor.shape[0], 1), dtype=x_tensor.dtype, device=x_tensor.device)

# ============================================================================
# 1. Poisson2D - 圆盘域Poisson方程
# ============================================================================

class PoissonDisk(BaseProblem):
    """
    二维Poisson方程问题定义（圆盘域）

    方程：-Δu = f(x,y) in Ω = {x ∈ R²: |x| < R}
    边界：u = g(x,y) on ∂Ω

    测试问题：
        精确解：u(x,y) = sin(πx)sin(πy)
        源项：f(x,y) = 2π²sin(πx)sin(πy)  (由-Δu计算得到)
        边界：g(x,y) = sin(πx)sin(πy)  (在圆周上)
    """

    def __init__(self, radius: float = 1.0):
        """
        初始化Poisson问题

        Args:
            radius: 圆形区域半径
        """
        self.radius = radius
        self.k = 1.0
        self.area_domain = math.pi * radius ** 2
        self.name = "PoissonDisk"

    def source_term(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        源项 f(x,y)

        对于精确解 u = sin(πx)sin(πy)：
            ∂²u/∂x² = -π²sin(πx)sin(πy)
            ∂²u/∂y² = -π²sin(πx)sin(πy)
            Δu = ∂²u/∂x² + ∂²u/∂y² = -2π²sin(πx)sin(πy)

        因此 f = -Δu = 2π²sin(πx)sin(πy)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            源项值 shape=(N, 1)
        """
        return 2 * (math.pi ** 2) * torch.sin(math.pi * x_tensor[:, 0:1]) * \
               torch.sin(math.pi * x_tensor[:, 1:2])

    def exact_solution(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解 u_exact(x,y) = sin(πx)sin(πy)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            精确解值 shape=(N, 1)
        """
        return torch.sin(math.pi * x_tensor[:, 0:1]) * torch.sin(math.pi * x_tensor[:, 1:2])

    def exact_gradient(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解的梯度

        ∂u/∂x = π·cos(πx)·sin(πy)
        ∂u/∂y = π·sin(πx)·cos(πy)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            梯度值 shape=(N, 2)
        """
        dudx = math.pi * torch.cos(math.pi * x_tensor[:, 0:1]) * \
               torch.sin(math.pi * x_tensor[:, 1:2])
        dudy = math.pi * torch.sin(math.pi * x_tensor[:, 0:1]) * \
               torch.cos(math.pi * x_tensor[:, 1:2])
        return torch.cat([dudx, dudy], dim=1)

    def sample_boundary(self, n, device='cpu'):
        """圆盘边界采样"""
        theta = torch.rand(n, 1, device=device) * 2 * math.pi
        x = self.radius * torch.cos(theta)
        y = self.radius * torch.sin(theta)
        return torch.cat([x, y], dim=1)

    def domain_indicator(self, x_tensor):
        r2 = x_tensor[:, 0:1] ** 2 + x_tensor[:, 1:2] ** 2
        return (r2 <= self.radius ** 2).float()


# ============================================================================
# 2. HeatTrapezoid - 梯形域热传导问题
# ============================================================================

class HeatTrapezoid(BaseProblem):
    """
    梯形域热传导问题

    方程：-k·Δu = s in Ω (梯形域)
    边界条件：
        - 左边界：u = T_left (Dirichlet)
        - 上边界：-k·∂u/∂n = q_top (Neumann)
        - 其他边界：绝热边界

    参数：
        k: 导热系数 (默认 20.0)
        s: 热源强度 (默认 50.0)
        T_left: 左边界温度 (默认 20.0)
        q_top: 上边界热流 (默认 -100.0)
    """

    def __init__(self, k: float = 20.0, s: float = 50.0,
                 T_left: float = 20.0, q_top: float = -100.0):
        """
        初始化热传导问题

        Args:
            k: 导热系数
            s: 热源强度
            T_left: 左边界温度
            q_top: 上边界热流
        """
        self.k = k
        self.s = s
        self.T_left = T_left
        self.q_top = q_top
        self.area_domain = 1.5  # 梯形面积
        self.len_left = 1.0     # 左边界长度
        self.len_top = 1.0      # 上边界长度
        self.name = "HeatTrapezoid"

    def source_term(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        源项 f = s (常数热源)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            源项值 shape=(N, 1)
        """
        return torch.full((x_tensor.shape[0], 1), self.s,
                         dtype=x_tensor.dtype, device=x_tensor.device)

    def exact_solution(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解（无解析解，返回None）

        注意：此问题没有解析解，需要通过数值方法求解

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            None
        """
        return None

    def exact_gradient(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确梯度（无解析解，返回None）

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            None
        """
        return None

    def boundary_condition(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        边界条件（仅左边界有Dirichlet条件）

        Args:
            x_tensor: 边界点坐标 shape=(N, 2)

        Returns:
            边界值 shape=(N, 1)
        """
        # 左边界 (x=0): T = T_left
        return torch.full((x_tensor.shape[0], 1), self.T_left,
                         dtype=x_tensor.dtype, device=x_tensor.device)

    def sample_boundary(self, n, device='cpu'):
        """梯形边界采样"""
        # 边界长度
        len_bottom = 2.0
        len_left = 1.0
        len_top = 1.0
        len_slope = math.sqrt(2.0)
        total = len_bottom + len_left + len_top + len_slope

        n_bottom = int(n * len_bottom / total)
        n_left = int(n * len_left / total)
        n_top = int(n * len_top / total)
        n_slope = n - n_bottom - n_left - n_top

        x_bottom = torch.rand(n_bottom, 1, device=device) * 2.0
        y_bottom = torch.zeros_like(x_bottom)

        x_left = torch.zeros(n_left, 1, device=device)
        y_left = torch.rand(n_left, 1, device=device)

        x_top = torch.rand(n_top, 1, device=device)
        y_top = torch.ones_like(x_top)

        x_slope = torch.rand(n_slope, 1, device=device) + 1.0
        y_slope = 2.0 - x_slope

        return torch.cat([torch.cat([x_bottom, x_left, x_top, x_slope], dim=0),
                          torch.cat([y_bottom, y_left, y_top, y_slope], dim=0)], dim=1)

    def domain_indicator(self, x_tensor):
        return ((x_tensor[:, 0:1] >= 0.0) &
                (x_tensor[:, 0:1] <= 2.0) &
                (x_tensor[:, 1:2] >= 0.0) &
                (x_tensor[:, 1:2] <= 1.0) &
                (x_tensor[:, 0:1] + x_tensor[:, 1:2] <= 2.0)).float()


# ============================================================================
# 3. SinusoidalSquare - 方形域正弦解析解
# ============================================================================

class SinusoidalSquare(BaseProblem):
    """
    方形域正弦解析解问题

    方程：-k·Δu = f(x,y) in Ω = [0,1]×[0,1]
    边界：u = 0 on ∂Ω

    测试问题：
        精确解：u(x,y) = sin(πx)sin(πy)
        源项：f(x,y) = 2kπ²sin(πx)sin(πy)
    """

    def __init__(self, k: float = 1.0):
        """
        初始化问题

        Args:
            k: 扩散系数
        """
        self.k = k
        self.pi = math.pi
        self.area_domain = 1.0
        self.name = "SinusoidalSquare"

    def exact_solution(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解 u(x,y) = sin(πx)sin(πy)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            精确解值 shape=(N, 1)
        """
        x = x_tensor[:, 0:1]
        y = x_tensor[:, 1:2]
        return torch.sin(self.pi * x) * torch.sin(self.pi * y)

    def exact_gradient(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确梯度

        ∂u/∂x = π·cos(πx)·sin(πy)
        ∂u/∂y = π·sin(πx)·cos(πy)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            梯度值 shape=(N, 2)
        """
        x = x_tensor[:, 0:1]
        y = x_tensor[:, 1:2]
        dTdx = self.pi * torch.cos(self.pi * x) * torch.sin(self.pi * y)
        dTdy = self.pi * torch.sin(self.pi * x) * torch.cos(self.pi * y)
        return torch.cat([dTdx, dTdy], dim=1)

    def source_term(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        源项 f(x,y) = 2kπ²sin(πx)sin(πy)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            源项值 shape=(N, 1)
        """
        x = x_tensor[:, 0:1]
        y = x_tensor[:, 1:2]
        return 2 * self.k * (self.pi ** 2) * torch.sin(self.pi * x) * torch.sin(self.pi * y)

    def sample_boundary(self, n, device='cpu'):
        """正方形边界采样"""
        n_edge = n // 4
        n_rem = n - 4 * n_edge
        n_bottom = n_edge + (1 if n_rem > 0 else 0)
        n_top = n_edge + (1 if n_rem > 1 else 0)
        n_left = n_edge + (1 if n_rem > 2 else 0)
        n_right = n_edge

        x_bottom = torch.rand(n_bottom, 1, device=device)
        y_bottom = torch.zeros_like(x_bottom)

        x_top = torch.rand(n_top, 1, device=device)
        y_top = torch.ones_like(x_top)

        y_left = torch.rand(n_left, 1, device=device)
        x_left = torch.zeros_like(y_left)

        y_right = torch.rand(n_right, 1, device=device)
        x_right = torch.ones_like(y_right)

        return torch.cat([torch.cat([x_bottom, x_top, x_left, x_right], dim=0),
                          torch.cat([y_bottom, y_top, y_left, y_right], dim=0)], dim=1)


# ============================================================================
# 4. LinearPatchProblem - 线性分片试验
# ============================================================================

class LinearPatchProblem(BaseProblem):
    """
    分片试验：测试能否精确重构线性多项式 u = x + y

    方程：-Δu = 0 in Ω = [0,1]×[0,1]
    精确解：u = x + y

    如果线性基方法无法通过此测试（误差降不到机器精度），说明代码有Bug。
    """

    def __init__(self, k: float = 1.0):
        """
        初始化问题

        Args:
            k: 扩散系数（对于线性解，源项为0）
        """
        self.k = k
        self.area_domain = 1.0
        self.name = "LinearPatchProblem"

    def exact_solution(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解 u = x + y

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            精确解值 shape=(N, 1)
        """
        return x_tensor[:, 0:1] + x_tensor[:, 1:2]

    def exact_gradient(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确梯度

        ∂u/∂x = 1
        ∂u/∂y = 1

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            梯度值 shape=(N, 2)
        """
        return torch.ones_like(x_tensor)

    def source_term(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        源项 f = 0 (Laplacian(x + y) = 0)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            源项值 shape=(N, 1)
        """
        return torch.zeros_like(x_tensor[:, 0:1])

    def sample_boundary(self, n, device='cpu'):
        """正方形边界采样"""
        n_edge = n // 4
        n_rem = n - 4 * n_edge
        n_bottom = n_edge + (1 if n_rem > 0 else 0)
        n_top = n_edge + (1 if n_rem > 1 else 0)
        n_left = n_edge + (1 if n_rem > 2 else 0)
        n_right = n_edge

        x_bottom = torch.rand(n_bottom, 1, device=device)
        y_bottom = torch.zeros_like(x_bottom)

        x_top = torch.rand(n_top, 1, device=device)
        y_top = torch.ones_like(x_top)

        y_left = torch.rand(n_left, 1, device=device)
        x_left = torch.zeros_like(y_left)

        y_right = torch.rand(n_right, 1, device=device)
        x_right = torch.ones_like(y_right)

        return torch.cat([torch.cat([x_bottom, x_top, x_left, x_right], dim=0),
                          torch.cat([y_bottom, y_top, y_left, y_right], dim=0)], dim=1)


# ============================================================================
# 5. TwoDimHighGradientProblem - 二维高斯峰问题
# ============================================================================

class GaussianPeak2D(BaseProblem):
    """
    二维高斯峰问题（近奇异问题）

    方程：-Δu = f(x,y) in Ω = [0,1]×[0,1]
    精确解：u = exp(-r²/α²)，其中 r² = (x-xc)² + (y-yc)²

    参数：
        alpha: 控制峰的宽度，越小越陡峭（建议 0.1 或 0.05）
        xc, yc: 峰的中心位置（默认 0.5, 0.5）
    """

    def __init__(self, alpha: float = 0.1, xc: float = 0.5, yc: float = 0.5):
        """
        初始化问题

        Args:
            alpha: 峰宽度参数
            xc: 峰中心x坐标
            yc: 峰中心y坐标
        """
        self.alpha = alpha
        self.xc = xc
        self.yc = yc
        self.k = 1.0
        self.area_domain = 1.0
        self.name = "GaussianPeak2D"

    def exact_solution(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解 u = exp(-r²/α²)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            精确解值 shape=(N, 1)
        """
        x = x_tensor[:, 0]
        y = x_tensor[:, 1]
        r2 = (x - self.xc) ** 2 + (y - self.yc) ** 2
        return torch.exp(-r2 / (self.alpha ** 2)).unsqueeze(1)

    def exact_gradient(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确梯度

        ∂u/∂x = u · (-2(x-xc)/α²)
        ∂u/∂y = u · (-2(y-yc)/α²)

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            梯度值 shape=(N, 2)
        """
        x = x_tensor[:, 0]
        y = x_tensor[:, 1]
        u = self.exact_solution(x_tensor).squeeze()

        dudx = u * (-2 * (x - self.xc) / (self.alpha ** 2))
        dudy = u * (-2 * (y - self.yc) / (self.alpha ** 2))

        return torch.stack([dudx, dudy], dim=1)

    def source_term(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        源项 f = -Δu = (4/α² - 4r²/α⁴) · u = 4/α² · (1 - r²/α²) · u

        Args:
            x_tensor: 输入坐标 shape=(N, 2)

        Returns:
            源项值 shape=(N, 1)
        """
        x = x_tensor[:, 0]
        y = x_tensor[:, 1]
        r2 = (x - self.xc) ** 2 + (y - self.yc) ** 2
        u = torch.exp(-r2 / (self.alpha ** 2))

        f = (4.0 / self.alpha ** 2) * (1.0 - r2 / self.alpha ** 2) * u
        return f.unsqueeze(1)

    def sample_boundary(self, n, device='cpu'):
        """正方形边界采样"""
        n_edge = n // 4
        n_rem = n - 4 * n_edge
        n_bottom = n_edge + (1 if n_rem > 0 else 0)
        n_top = n_edge + (1 if n_rem > 1 else 0)
        n_left = n_edge + (1 if n_rem > 2 else 0)
        n_right = n_edge

        x_bottom = torch.rand(n_bottom, 1, device=device)
        y_bottom = torch.zeros_like(x_bottom)

        x_top = torch.rand(n_top, 1, device=device)
        y_top = torch.ones_like(x_top)

        y_left = torch.rand(n_left, 1, device=device)
        x_left = torch.zeros_like(y_left)

        y_right = torch.rand(n_right, 1, device=device)
        x_right = torch.ones_like(y_right)

        return torch.cat([torch.cat([x_bottom, x_top, x_left, x_right], dim=0),
                          torch.cat([y_bottom, y_top, y_left, y_right], dim=0)], dim=1)


# ============================================================================
# 6. HighGradientProblem - 1D高梯度问题（arctan）
# ============================================================================

class HighGradientProblem(BaseProblem):
    """
    一维高梯度问题

    方程：-u'' = g(x) in Ω = (-1, 1)
    精确解：u(x) = arctan(100x)

    这是一个具有陡峭梯度的测试问题，用于测试方法对高梯度的处理能力。
    """

    def __init__(self):
        """初始化问题"""
        self.domain = (-1.0, 1.0)
        self.k = 1.0
        self.area_domain = 2.0
        self.name = "HighGradientProblem"

    def exact_solution(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确解 u(x) = arctan(100x)

        Args:
            x_tensor: 输入坐标 shape=(N, 1) 或 (N,)

        Returns:
            精确解值 shape=(N, 1)
        """
        if x_tensor.dim() == 1:
            x = x_tensor
        else:
            x = x_tensor[:, 0] if x_tensor.shape[1] > 0 else x_tensor.squeeze()
        return torch.atan(100 * x).unsqueeze(-1) if x.dim() == 1 else torch.atan(100 * x)

    def exact_gradient(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        精确梯度 u'(x) = 100 / (1 + (100x)²)

        Args:
            x_tensor: 输入坐标 shape=(N, 1) 或 (N,)

        Returns:
            梯度值 shape=(N, 1)
        """
        if x_tensor.dim() == 1:
            x = x_tensor
        else:
            x = x_tensor[:, 0] if x_tensor.shape[1] > 0 else x_tensor.squeeze()
        grad = 100.0 / (1.0 + 10000.0 * x ** 2)
        return grad.unsqueeze(-1) if grad.dim() == 1 else grad

    def source_term(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        源项 g(x) = -u'' = (2×10⁶ · x) / (1 + 10⁴x²)²

        对于方程 -u'' = g，能量泛函为 J = 0.5∫(u')² - ∫g·u

        Args:
            x_tensor: 输入坐标 shape=(N, 1) 或 (N,)

        Returns:
            源项值 shape=(N, 1)
        """
        if x_tensor.dim() == 1:
            x = x_tensor
        else:
            x = x_tensor[:, 0] if x_tensor.shape[1] > 0 else x_tensor.squeeze()

        num = 2.0 * 10 ** 6 * x
        denom = (1.0 + 10 ** 4 * x ** 2) ** 2
        source = num / denom
        return source.unsqueeze(-1) if source.dim() == 1 else source

    def sample_boundary(self, n, device='cpu'):
        """一维边界采样"""
        n_left = n // 2
        n_right = n - n_left
        left = torch.full((n_left, 1), self.domain[0], device=device)
        right = torch.full((n_right, 1), self.domain[1], device=device)
        return torch.cat([left, right], dim=0)

    def domain_indicator(self, x_tensor):
        return ((x_tensor >= self.domain[0]) & (x_tensor <= self.domain[1])).float()

