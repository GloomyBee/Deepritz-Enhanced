"""
积分器模块 (Integrators)
提供三种积分方法用于Deep Ritz能量泛函计算：
1. MonteCarloIntegrator - 蒙特卡洛积分
2. GaussIntegrator - 高斯积分（背景网格）
3. SCNIIntegrator - SCNI积分（Voronoi单元）
"""

import torch
import numpy as np
from scipy.spatial import Voronoi

# ==========================================
# 1. 蒙特卡洛积分器
# ==========================================
class MonteCarloIntegrator:
    """
    蒙特卡洛积分器
    使用随机采样点进行积分，适用于简单问题的快速原型
    """

    def __init__(self, n_points=10000, domain_bounds=(0, 1, 0, 1)):
        """
        Args:
            n_points: 积分点数量
            domain_bounds: 定义域边界 (x_min, x_max, y_min, y_max)
        """
        self.n_points = n_points
        self.domain_bounds = domain_bounds
        x_min, x_max, y_min, y_max = domain_bounds
        self.area = (x_max - x_min) * (y_max - y_min)

    def sample_points(self, device='cpu'):
        """生成随机积分点"""
        x_min, x_max, y_min, y_max = self.domain_bounds
        x = torch.rand(self.n_points, 1, device=device) * (x_max - x_min) + x_min
        y = torch.rand(self.n_points, 1, device=device) * (y_max - y_min) + y_min
        return torch.cat([x, y], dim=1)

    def integrate(self, integrand_values):
        """
        计算积分
        Args:
            integrand_values: [N, 1] 被积函数在采样点的值
        Returns:
            积分值
        """
        return torch.mean(integrand_values) * self.area

# ==========================================
# 2. 高斯积分器
# ==========================================
class GaussIntegrator:
    """
    高斯积分器（背景网格）
    使用规则网格上的高斯积分点，精度高，适合RKPM等无网格方法
    """

    def __init__(self, nx=20, ny=20, domain_bounds=(0, 1, 0, 1), order=4):
        """
        Args:
            nx, ny: 背景网格划分数
            domain_bounds: 定义域边界
            order: 高斯积分阶数（2或4）
        """
        self.nx = nx
        self.ny = ny
        self.domain_bounds = domain_bounds
        self.order = order
        self.gauss_points, self.gauss_weights = self._generate_gauss_points()

    def _generate_gauss_points(self):
        """生成高斯积分��和权重"""
        if self.order == 2:
            # 2点高斯积分
            xi = np.array([-0.5773502692, 0.5773502692])
            wi = np.array([1.0, 1.0])
        elif self.order == 4:
            # 4点高斯积分
            xi = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])
            wi = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
        else:
            raise ValueError("Only order=2 or order=4 supported")

        gp_x, gp_y = np.meshgrid(xi, xi)
        gw_x, gw_y = np.meshgrid(wi, wi)
        gp_local = np.column_stack([gp_x.ravel(), gp_y.ravel()])
        gw_local = (gw_x * gw_y).ravel()

        x_min, x_max, y_min, y_max = self.domain_bounds
        dx = (x_max - x_min) / self.nx
        dy = (y_max - y_min) / self.ny
        area_cell = dx * dy

        all_points = []
        all_weights = []
        for i in range(self.nx):
            for j in range(self.ny):
                cx = x_min + (i + 0.5) * dx
                cy = y_min + (j + 0.5) * dy
                real_pts = np.zeros_like(gp_local)
                real_pts[:, 0] = cx + gp_local[:, 0] * (dx / 2)
                real_pts[:, 1] = cy + gp_local[:, 1] * (dy / 2)
                real_wts = gw_local * (area_cell / 4.0)
                all_points.append(real_pts)
                all_weights.append(real_wts)

        return np.vstack(all_points), np.hstack(all_weights)

    def get_points_tensor(self, device='cpu'):
        """返回积分点的Tensor"""
        return torch.tensor(self.gauss_points, dtype=torch.float32, device=device)

    def get_weights_tensor(self, device='cpu'):
        """返回积分权重的Tensor"""
        return torch.tensor(self.gauss_weights, dtype=torch.float32, device=device).unsqueeze(1)

    def integrate(self, integrand_values, weights=None):
        """
        计算积分
        Args:
            integrand_values: [N, 1] 被积函数在高斯点的值
            weights: 可选的权重张量，如果为None则使用内部权重
        Returns:
            积分值
        """
        if weights is None:
            weights = self.get_weights_tensor(integrand_values.device)
        return torch.sum(integrand_values * weights)

# ==========================================
# 3. Voronoi图及SCNI积分器
# ==========================================
class SCNIIntegrator:
    """
    SCNI积分器（Stabilized Conforming Nodal Integration）
    基于Voronoi单元的稳定化节点积分，适用于无网格方法

    核心思想：
    1. 对每个粒子构建Voronoi单元
    2. 使用散度定理将体积分转换为边界积分
    3. 计算B矩阵（应变-位移矩阵）用于梯度计算
    """

    def __init__(self, nodes, domain_bounds=(0, 1, 0, 1)):
        """
        Args:
            nodes: [N, 2] 粒子坐标（numpy数组）
            domain_bounds: 定义域边界 (x_min, x_max, y_min, y_max)
        """
        self.nodes = nodes
        self.domain_bounds = domain_bounds
        self.N_nodes = nodes.shape[0]

        # 预计算Voronoi几何
        self.scni_data = self._precompute_scni_topology()

    def _bounded_voronoi(self, points):
        """
        构建有界Voronoi图（镜像法）
        通过在边界外添加镜像点，确保边界粒子的Voronoi单元被正确裁剪
        """
        x_min, x_max, y_min, y_max = self.domain_bounds

        # 构造8个方向的镜像点
        p_left = points.copy()
        p_left[:, 0] = 2 * x_min - p_left[:, 0]
        p_right = points.copy()
        p_right[:, 0] = 2 * x_max - p_right[:, 0]
        p_down = points.copy()
        p_down[:, 1] = 2 * y_min - p_down[:, 1]
        p_up = points.copy()
        p_up[:, 1] = 2 * y_max - p_up[:, 1]

        # 角部镜像
        p_ld = p_left.copy()
        p_ld[:, 1] = 2 * y_min - p_ld[:, 1]
        p_rd = p_right.copy()
        p_rd[:, 1] = 2 * y_min - p_rd[:, 1]
        p_lu = p_left.copy()
        p_lu[:, 1] = 2 * y_max - p_lu[:, 1]
        p_ru = p_right.copy()
        p_ru[:, 1] = 2 * y_max - p_ru[:, 1]

        points_all = np.vstack([points, p_left, p_right, p_down, p_up,
                                p_ld, p_rd, p_lu, p_ru])

        vor = Voronoi(points_all)

        # 提取原始点的区域并裁剪顶点
        n_points = points.shape[0]
        new_vertices = np.copy(vor.vertices)
        new_vertices[:, 0] = np.clip(new_vertices[:, 0], x_min, x_max)
        new_vertices[:, 1] = np.clip(new_vertices[:, 1], y_min, y_max)

        filtered_regions = []
        for i in range(n_points):
            region_idx = vor.point_region[i]
            region = vor.regions[region_idx]
            filtered_regions.append(region)

        return filtered_regions, new_vertices

    @staticmethod
    def _compute_polygon_area(vertices):
        """使用鞋带公式计算多边形面积"""
        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    def _precompute_scni_topology(self):
        """
        预计算SCNI几何信息
        返回：
            edge_points: [M, 2] 所有边的中点
            edge_normals: [M, 2] 所有边的外法向量*长度
            edge_indices: [M] 每条边归属的节点索引
            node_areas: [N] 每个节点的控制体积
        """
        x_min, x_max, y_min, y_max = self.domain_bounds
        regions, vor_vertices = self._bounded_voronoi(self.nodes)

        midpoints = []
        normals_len = []
        owner_indices = []
        node_areas = np.zeros(self.N_nodes)

        for i in range(self.N_nodes):
            region_indices = regions[i]
            if -1 in region_indices or len(region_indices) < 3:
                continue

            poly_verts = vor_vertices[region_indices]

            # 裁剪到定义域
            poly_verts[:, 0] = np.clip(poly_verts[:, 0], x_min, x_max)
            poly_verts[:, 1] = np.clip(poly_verts[:, 1], y_min, y_max)

            # 计算面积
            area = self._compute_polygon_area(poly_verts)
            node_areas[i] = area

            if area < 1e-12:
                continue

            # 遍历多边形的每条边
            n_v = len(poly_verts)
            for k in range(n_v):
                p1 = poly_verts[k]
                p2 = poly_verts[(k + 1) % n_v]

                edge_vec = p2 - p1
                L = np.linalg.norm(edge_vec)
                if L < 1e-12:
                    continue

                mid = (p1 + p2) / 2.0
                # 外法线（逆时针顺序）: (dy, -dx)
                nx, ny = edge_vec[1], -edge_vec[0]

                midpoints.append(mid)
                normals_len.append([nx, ny])
                owner_indices.append(i)

        return {
            "edge_points": np.array(midpoints, dtype=np.float32),
            "edge_normals": np.array(normals_len, dtype=np.float32),
            "edge_indices": np.array(owner_indices, dtype=np.int64),
            "node_areas": np.array(node_areas, dtype=np.float32)
        }

    def compute_B_matrix(self, shape_function_evaluator, device='cpu'):
        """
        计算SCNI的B矩阵（应变-位移矩阵）

        B矩阵形状: [N_nodes, 2, N_nodes]
        B[I, 0, :] 是节点I处的 d/dx 算子向量
        B[I, 1, :] 是节点I处的 d/dy 算子向量

        使用散度定理: ∫_Ω ∂φ/∂x dΩ = ∫_∂Ω φ n_x dΓ

        Args:
            shape_function_evaluator: 形函数求值器，输入[K,2]，输出(phi, phi_x, phi_y)
            device: 计算设备
        Returns:
            B_matrix: [N_nodes, 2, N_nodes] 应变矩阵
            Areas: [N_nodes] 节点控制体积
        """
        edge_points = torch.tensor(self.scni_data['edge_points'], device=device)
        edge_normals = torch.tensor(self.scni_data['edge_normals'], device=device)
        edge_indices = torch.tensor(self.scni_data['edge_indices'], device=device)
        node_areas = torch.tensor(self.scni_data['node_areas'], device=device)

        # 计算所有边中点处的形函数值
        with torch.no_grad():
            phi_edges, _, _ = shape_function_evaluator(edge_points)  # [M, N_nodes]

        # 向量化计算B矩阵
        B_matrix = self._compute_B_vectorized(
            phi_edges, edge_normals, edge_indices, node_areas
        )

        return B_matrix, node_areas

    def _compute_B_vectorized(self, phi_edges, edge_normals, edge_indices, node_areas):
        """
        向量化计算B矩阵（GPU并行）

        Args:
            phi_edges: [M, N_nodes] 所有积分点处的形函数值
            edge_normals: [M, 2] 所有边的(法向量*长度)
            edge_indices: [M] 每条边归属的节点索引
            node_areas: [N_nodes] 节点面积
        Returns:
            B_matrix: [N_nodes, 2, N_nodes]
        """
        M, N = phi_edges.shape
        device = phi_edges.device

        # 计算被积项: phi * (n * L)
        # [M, N, 1] * [M, 1, 2] -> [M, N, 2]
        term = phi_edges.unsqueeze(2) * edge_normals.unsqueeze(1)

        # 初始化累加器
        B_accumulator = torch.zeros((N, N, 2), dtype=torch.float32, device=device)

        # Scatter-Add: 将边的贡献累加到对应节点
        idx_expanded = edge_indices.view(-1, 1, 1).expand(-1, N, 2)
        B_accumulator.scatter_add_(0, idx_expanded, term)

        # 除以面积（散度定理）
        areas_reshaped = node_areas.view(-1, 1, 1) + 1e-12
        B_final = B_accumulator / areas_reshaped

        # 调整为 [N_nodes, 2, N_nodes]
        return B_final.permute(0, 2, 1)

    def integrate_energy(self, B_matrix, node_areas, coefficients, source_term_values):
        """
        计算能量泛函（应变能 - 外力功）

        能量泛函: J = 0.5 * ∫ |∇u|² dΩ - ∫ f·u dΩ

        Args:
            B_matrix: [N, 2, N] 应变矩阵
            node_areas: [N] 节点控制体积
            coefficients: [N] Ritz系数
            source_term_values: [N] 源项在节点处的值
        Returns:
            energy_strain: 应变能
            energy_force: 外力功
        """
        # 计算节点处的梯度: grad[i, :] = B[i, :, :] @ coefficients
        grad_nodes = torch.einsum('nij, j -> ni', B_matrix, coefficients)

        # 应变能: 0.5 * ∑ A_i * |grad_i|²
        grad_sq = torch.sum(grad_nodes ** 2, dim=1)
        energy_strain = 0.5 * torch.sum(node_areas * grad_sq)

        # 计算节点处的位移（需要形函数值）
        # 这里假设 u_nodes 已经在外部计算
        # energy_force = torch.sum(node_areas * source_term_values * u_nodes)

        return energy_strain

    def get_areas_tensor(self, device='cpu'):
        """返回节点面积的Tensor"""
        return torch.tensor(self.scni_data['node_areas'], dtype=torch.float32, device=device)
