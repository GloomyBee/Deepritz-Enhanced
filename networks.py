"""
神经网络模块 (Networks)
提供多种神经网络架构用于Deep Ritz方法：
1. RitzNet - 基础全连接网络
2. RBFNet - 径向基函数网络
3. KANNet - Kolmogorov-Arnold网络（B样条基函数）
4. RKPMNet - 再生核粒子法网络
"""

import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import cKDTree


# ==========================================
# 1. 残差块（用于深层网络）
# ==========================================
class ResidualBlock(nn.Module):
    """
    残差块
    实现 y = x + F(x) 的跳跃连接，有助于训练深层网络
    """

    def __init__(self, hidden_dim, activation=nn.Tanh()):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.activation = activation

    def forward(self, x):
        residual = x
        out = self.activation(self.linear1(x))
        out = self.linear2(out)
        out = out + residual
        return self.activation(out)


# ==========================================
# 2. 基础RitzNet（全连接网络）
# ==========================================
class RitzNet(nn.Module):
    """
    基础Deep Ritz网络
    标准的全连接前馈神经网络，使用Tanh激活函数

    网络结构: input -> hidden layers -> output
    """

    def __init__(self, input_dim=2, hidden_dims=[50, 50, 50], output_dim=1,
                 use_residual=False):
        """
        Args:
            input_dim: 输入维度（通常为2，表示x,y坐标）
            hidden_dims: 隐藏层维度列表
            output_dim: 输出维度（通常为1，表示温度或位移）
            use_residual: 是否使用残差连接
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        layers = []
        prev_dim = input_dim

        # 构建隐藏层
        for i, hidden_dim in enumerate(hidden_dims):
            if use_residual and i > 0 and hidden_dim == prev_dim:
                layers.append(ResidualBlock(hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.Tanh())
            prev_dim = hidden_dim

        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))

        self.network = nn.Sequential(*layers)

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        前向传播
        Args:
            x: [batch_size, input_dim] 输入坐标
        Returns:
            u: [batch_size, output_dim] 网络输出
        """
        return self.network(x)


# ==========================================
# 3. RBF网络（径向基函数网络）
# ==========================================
class RBFNet(nn.Module):
    """
    径向基函数网络
    使用高斯径向基函数作为激活函数

    u(x) = Σ w_i * φ(||x - c_i||)
    其中 φ(r) = exp(-r²/σ²) 是高斯RBF
    """

    def __init__(self, centers, sigma=0.1, trainable_centers=False):
        """
        Args:
            centers: [N, 2] RBF中心点坐标
            sigma: RBF宽度参数
            trainable_centers: 是否训练中心点位置
        """
        super().__init__()
        if trainable_centers:
            self.centers = nn.Parameter(centers.clone())
        else:
            self.register_buffer('centers', centers)

        self.sigma = sigma
        self.n_centers = centers.shape[0]

        # 输出层权重
        self.linear = nn.Linear(self.n_centers, 1, bias=True)
        nn.init.normal_(self.linear.weight, std=0.01)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 2] 输入坐标
        Returns:
            u: [batch_size, 1] 网络输出
        """
        # 计算距离: [batch, n_centers]
        diff = x.unsqueeze(1) - self.centers.unsqueeze(0)
        dist_sq = torch.sum(diff ** 2, dim=2)

        # 高斯RBF
        rbf_values = torch.exp(-dist_sq / (2 * self.sigma ** 2))

        # 线性组合
        u = self.linear(rbf_values)
        return u


# ==========================================
# 4. KAN网络（Kolmogorov-Arnold网络）
# ==========================================
class KANNet(nn.Module):
    """
    KAN网络（基于B样条基函数）
    使用可学习的样条函数代替传统的固定激活函数

    核心思想：
    - 每个连接不是简单的权重，而是一个可学习的单变量函数
    - 使用B样条基函数来参数化这些函数
    """

    def __init__(self, input_dim=2, hidden_dim=20, output_dim=1, num_basis=5):
        """
        Args:
            input_dim: 输入维度
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            num_basis: B样条基函数数量
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_basis = num_basis

        # B样条节点（均匀分布在[0,1]）
        self.register_buffer('knots', torch.linspace(0, 1, num_basis))
        self.sigma = 1.0 / (num_basis - 1)

        # 第一层：input -> hidden
        self.spline_coeffs_1 = nn.Parameter(
            torch.randn(input_dim, hidden_dim, num_basis) * 0.1
        )

        # 第二层：hidden -> output
        self.spline_coeffs_2 = nn.Parameter(
            torch.randn(hidden_dim, output_dim, num_basis) * 0.1
        )

    def _apply_kan_layer(self, x, coeffs):
        """
        应用KAN层
        Args:
            x: [batch, in_dim] 输入
            coeffs: [in_dim, out_dim, num_basis] 样条系数
        Returns:
            out: [batch, out_dim] 输出
        """
        batch_size, in_dim = x.shape
        out_dim = coeffs.shape[1]

        # 归一化输入到[0,1]
        x_norm = torch.sigmoid(x)  # [batch, in_dim]

        # 计算B样条基函数值
        # x_norm: [batch, in_dim, 1]
        # knots: [num_basis]
        x_expanded = x_norm.unsqueeze(2)  # [batch, in_dim, 1]
        knots_expanded = self.knots.view(1, 1, -1)  # [1, 1, num_basis]

        # 高斯基函数
        basis = torch.exp(-((x_expanded - knots_expanded) ** 2) / (2 * self.sigma ** 2))
        # basis: [batch, in_dim, num_basis]

        # 计算输出
        # basis: [batch, in_dim, num_basis]
        # coeffs: [in_dim, out_dim, num_basis]
        # 结果: [batch, out_dim]
        out = torch.einsum('bin,ion->bo', basis, coeffs)

        return out

    def forward(self, x):
        """
        Args:
            x: [batch_size, input_dim] 输入坐标
        Returns:
            u: [batch_size, output_dim] 网络输出
        """
        # 第一层
        h = self._apply_kan_layer(x, self.spline_coeffs_1)
        h = torch.tanh(h)  # 添加非线性

        # 第二层
        u = self._apply_kan_layer(h, self.spline_coeffs_2)

        return u


# ==========================================
# 5. RKPM网络（再生核粒子法网络）
# ==========================================
class RKPMLayer(nn.Module):
    """
    RKPM形函数层
    实现再生核粒子法的形函数及其导数计算

    核心公式：
    φ_I(x) = w(x-x_I) * P^T(x-x_I) * M^(-1)(x) * P(0)
    其中：
    - w: 核函数（三次样条）
    - P: 基函数向量 [1, x-x_I, y-y_I]
    - M: 矩量矩阵
    """

    def __init__(self, nodes, support_factor=2.5):
        """
        Args:
            nodes: [N, 2] 粒子坐标
            support_factor: 支持域因子（相对于平均粒子间距）
        """
        super().__init__()
        self.register_buffer('nodes', nodes)

        # 计算支持域半径
        nodes_np = nodes.cpu().numpy()
        tree = cKDTree(nodes_np)
        dists, _ = tree.query(nodes_np, k=2)
        h_avg = np.mean(dists[:, 1])
        self.dilation = support_factor * h_avg

        self.epsilon = 1e-5  # 正则化参数

    def _cubic_spline(self, r_norm):
        """
        三次样条核函数
        Args:
            r_norm: 归一化距离 r/a
        Returns:
            w: 核函数值
        """
        q = r_norm
        val = torch.zeros_like(q)
        m1 = (q <= 0.5)
        m2 = (q > 0.5) & (q <= 1.0)
        val[m1] = 2/3 - 4*q[m1]**2 + 4*q[m1]**3
        val[m2] = 4/3 - 4*q[m2] + 4*q[m2]**2 - 4/3*q[m2]**3
        return val

    def forward(self, x):
        """
        计算RKPM形函数及其导数
        Args:
            x: [batch_size, 2] 输入坐标
        Returns:
            phi: [batch_size, N_nodes] 形函数值
            phi_x: [batch_size, N_nodes] 形函数x方向导数
            phi_y: [batch_size, N_nodes] 形函数y方向导数
        """
        # 计算相对位置
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)  # [batch, N, 2]
        dx = diff[:, :, 0]
        dy = diff[:, :, 1]

        # 计算距离和核函数
        dist = torch.sqrt(dx**2 + dy**2 + 1e-10)
        r_norm = dist / self.dilation
        w = self._cubic_spline(r_norm) * (r_norm <= 1.0).float()

        # 构建基函数矩阵 P = [1, dx, dy]
        ones = torch.ones_like(dx)
        P = torch.stack([ones, dx, dy], dim=-1)  # [batch, N, 3]

        # 计算矩量矩阵 M = Σ P ⊗ P * w
        M = torch.einsum('bni, bnj, bn -> bij', P, P, w)  # [batch, 3, 3]

        # 正则化并求逆
        M_reg = M + torch.eye(3, device=x.device) * self.epsilon
        try:
            M_inv = torch.linalg.inv(M_reg)
        except:
            M_inv = torch.linalg.pinv(M_reg)

        # 计算形函数系数向量
        P_w = P * w.unsqueeze(-1)
        b_vec = torch.einsum('bij, bnj -> bni', M_inv, P_w)  # [batch, N, 3]

        # 提取形函数及其导数
        # 注意：由于泰勒展开的符号约定，导数需要取负号
        phi = b_vec[:, :, 0]
        phi_x = -b_vec[:, :, 1]
        phi_y = -b_vec[:, :, 2]

        return phi, phi_x, phi_y


class RKPMNet(nn.Module):
    """
    RKPM网络
    使用RKPM形函数作为基函数，通过线性组合得到解

    u(x) = Σ d_I * φ_I(x)
    """
    provides_gradients = True

    def __init__(self, nodes, support_factor=2.5):
        """
        Args:
            nodes: [N, 2] 粒子坐标（Tensor）
            support_factor: 支持域因子
        """
        super().__init__()
        self.rkpm = RKPMLayer(nodes, support_factor)
        self.linear = nn.Linear(nodes.shape[0], 1, bias=False)

        # 初始化系数
        nn.init.normal_(self.linear.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 2] 输入坐标
        Returns:
            u: [batch_size, 1] 位移/温度
            u_x: [batch_size, 1] x方向导数
            u_y: [batch_size, 1] y方向导数
        """
        phi, phi_x, phi_y = self.rkpm(x)

        u = self.linear(phi)
        u_x = self.linear(phi_x)
        u_y = self.linear(phi_y)

        return u, u_x, u_y


# ==========================================
# 6. 径向基KAN层（用于SCNI）
# ==========================================
class RadialBasisKANLayer(nn.Module):
    """
    径向基KAN层
    结合径向基函数和KAN的思想，用于无网格方法

    特点：
    - 使用径向距离作为输入
    - 通过可学习的基函数组合来构造形函数
    - 保持单位分解性质
    """

    def __init__(self, nodes, support_factor=2.5, num_basis=5):
        """
        Args:
            nodes: [N, 2] 粒子坐标
            support_factor: 支持域因子
            num_basis: 基函数数量
        """
        super().__init__()
        self.register_buffer('nodes', nodes)

        # 计算支持域
        nodes_np = nodes.cpu().numpy()
        tree = cKDTree(nodes_np)
        dists, _ = tree.query(nodes_np, k=2)
        h_avg = np.mean(dists[:, 1])
        self.register_buffer('dilation', torch.tensor(support_factor * h_avg, dtype=torch.float32))

        # 基函数中心
        self.register_buffer('centers', torch.linspace(0, 1.0, num_basis))
        self.sigma = 1.0 / (num_basis - 1)

        # 可学习的样条系数
        self.spline_coeffs = nn.Parameter(torch.ones(nodes.shape[0], num_basis) * 0.5)

    def forward(self, x):
        """
        Args:
            x: [batch_size, 2] 输入坐标
        Returns:
            phi: [batch_size, N_nodes] 形函数值
            None, None: 占位符（保持接口一致）
        """
        # 计算归一化距离
        diff = x.unsqueeze(1) - self.nodes.unsqueeze(0)
        r = torch.norm(diff, dim=2)  # [batch, N]
        q = r / self.dilation

        # 计算基函数值
        q_exp = q.unsqueeze(-1)  # [batch, N, 1]
        grid = self.centers.view(1, 1, -1)  # [1, 1, K]

        # 高斯基函数
        basis_values = torch.exp(-((q_exp - grid) ** 2) / (2 * self.sigma ** 2))

        # 线性组合
        coeffs = self.spline_coeffs.unsqueeze(0)  # [1, N, K]
        w_raw = torch.sum(basis_values * coeffs, dim=-1)  # [batch, N]

        # 紧支持
        mask = (q <= 1.0).float()
        w = w_raw * mask

        # 单位分解
        w_sum = torch.sum(w, dim=1, keepdim=True) + 1e-12
        phi = w / w_sum

        return phi, None, None
