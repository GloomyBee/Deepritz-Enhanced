"""
采样器模块

包含所有域采样器，用于生成训练和测试数据点。

采样器类型：
- DiskSampler: 圆盘域采样
- SquareSampler: 方形域采样  
- TrapezoidSampler: 梯形域采样
- MeshfreeUtils: 无网格节点生成和Voronoi图工具
"""

import numpy as np
import torch
from scipy.spatial import Voronoi
import math


class BaseSampler:
    """采样器基类"""

    def sample_domain(self, num_samples):
        """从域内部采样点"""
        raise NotImplementedError

    def sample_boundary(self, num_samples):
        """从边界采样点"""
        raise NotImplementedError


# ============================================================================
# 1. 圆盘域采样器
# ============================================================================

class DiskSampler(BaseSampler):
    """圆盘域采样器 Ω = {x ∈ R²: |x| < R}"""

    def __init__(self, radius=1.0):
        self.radius = radius
        self.name = "DiskSampler"

    def sample_domain(self, num_samples):
        """从圆盘内部均匀采样点（极坐标变换）"""
        r = self.radius * np.sqrt(np.random.rand(num_samples))
        theta = 2 * np.pi * np.random.rand(num_samples)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.column_stack([x, y])

    def sample_boundary(self, num_samples):
        """从圆周边界均匀采样点"""
        theta = 2 * np.pi * np.random.rand(num_samples)
        x = self.radius * np.cos(theta)
        y = self.radius * np.sin(theta)
        return np.column_stack([x, y])


# ============================================================================
# 2. 方形域采样器
# ============================================================================

class SquareSampler(BaseSampler):
    """方形域采样器 Ω = [x_min, x_max] × [y_min, y_max]"""

    def __init__(self, x_range=(0.0, 1.0), y_range=(0.0, 1.0)):
        self.x_min, self.x_max = x_range
        self.y_min, self.y_max = y_range
        self.name = "SquareSampler"

    def sample_domain(self, num_samples):
        """从方形域内部均匀采样点"""
        x = np.random.uniform(self.x_min, self.x_max, num_samples)
        y = np.random.uniform(self.y_min, self.y_max, num_samples)
        return np.column_stack([x, y])

    def sample_boundary(self, num_samples):
        """从方形边界均匀采样点（按边长比例分配）"""
        width = self.x_max - self.x_min
        height = self.y_max - self.y_min
        perimeter = 2 * (width + height)

        n_bottom = int(num_samples * width / perimeter)
        n_right = int(num_samples * height / perimeter)
        n_top = int(num_samples * width / perimeter)
        n_left = num_samples - n_bottom - n_right - n_top

        x_bottom = np.random.uniform(self.x_min, self.x_max, n_bottom)
        y_bottom = np.full(n_bottom, self.y_min)

        x_right = np.full(n_right, self.x_max)
        y_right = np.random.uniform(self.y_min, self.y_max, n_right)

        x_top = np.random.uniform(self.x_min, self.x_max, n_top)
        y_top = np.full(n_top, self.y_max)

        x_left = np.full(n_left, self.x_min)
        y_left = np.random.uniform(self.y_min, self.y_max, n_left)

        x_all = np.concatenate([x_bottom, x_right, x_top, x_left])
        y_all = np.concatenate([y_bottom, y_right, y_top, y_left])
        return np.column_stack([x_all, y_all])

    def sample_left_boundary(self, num_samples):
        """采样左边界 (x = x_min)"""
        x = np.full(num_samples, self.x_min)
        y = np.random.uniform(self.y_min, self.y_max, num_samples)
        return np.column_stack([x, y])

    def sample_right_boundary(self, num_samples):
        """采样右边界 (x = x_max)"""
        x = np.full(num_samples, self.x_max)
        y = np.random.uniform(self.y_min, self.y_max, num_samples)
        return np.column_stack([x, y])

    def sample_bottom_boundary(self, num_samples):
        """采样底边界 (y = y_min)"""
        x = np.random.uniform(self.x_min, self.x_max, num_samples)
        y = np.full(num_samples, self.y_min)
        return np.column_stack([x, y])

    def sample_top_boundary(self, num_samples):
        """采样顶边界 (y = y_max)"""
        x = np.random.uniform(self.x_min, self.x_max, num_samples)
        y = np.full(num_samples, self.y_max)
        return np.column_stack([x, y])


# ============================================================================
# 3. 梯形域采样器
# ============================================================================

class TrapezoidSampler(BaseSampler):
    """梯形域采样器：x + y ≤ 2.0，x ∈ [0,2]，y ∈ [0,1]"""

    def __init__(self):
        self.name = "TrapezoidSampler"

    def sample_domain(self, num_samples):
        """从梯形域内部均匀采样点（拒绝采样法）"""
        samples = []
        while len(samples) < num_samples:
            x = np.random.uniform(0, 2, num_samples * 2)
            y = np.random.uniform(0, 1, num_samples * 2)
            mask = (x + y <= 2.0)
            valid_points = np.column_stack([x[mask], y[mask]])
            samples.append(valid_points)
        samples = np.vstack(samples)
        return samples[:num_samples]

    def sample_boundary(self, num_samples):
        """从梯形边界均匀采样点"""
        len_bottom = 2.0
        len_left = 1.0
        len_top = 1.0
        len_slope = np.sqrt(2.0)
        total_length = len_bottom + len_left + len_top + len_slope

        n_bottom = int(num_samples * len_bottom / total_length)
        n_left = int(num_samples * len_left / total_length)
        n_top = int(num_samples * len_top / total_length)
        n_slope = num_samples - n_bottom - n_left - n_top

        x_bottom = np.random.uniform(0, 2, n_bottom)
        y_bottom = np.zeros(n_bottom)

        x_left = np.zeros(n_left)
        y_left = np.random.uniform(0, 1, n_left)

        x_top = np.random.uniform(0, 1, n_top)
        y_top = np.ones(n_top)

        x_slope = np.random.uniform(1, 2, n_slope)
        y_slope = 2 - x_slope

        x_all = np.concatenate([x_bottom, x_left, x_top, x_slope])
        y_all = np.concatenate([y_bottom, y_left, y_top, y_slope])
        return np.column_stack([x_all, y_all])

    def sample_left_boundary(self, num_samples):
        """采样左边界 (x=0, y∈[0,1])"""
        x = np.zeros(num_samples)
        y = np.random.uniform(0, 1, num_samples)
        return np.column_stack([x, y])

    def sample_top_boundary(self, num_samples):
        """采样顶边界 (y=1, x∈[0,1])"""
        x = np.random.uniform(0, 1, num_samples)
        y = np.ones(num_samples)
        return np.column_stack([x, y])


# ============================================================================
# 4. 无网格工具类
# ============================================================================

class MeshfreeUtils:
    """无网格方法工具类：RKPM节点生成、Voronoi图构建"""

    @staticmethod
    def sample_rkpm_nodes(n_total=625, x_range=(0, 1), y_range=(0, 1), noise_factor=0.4):
        """生成带扰动的均匀无网格节点"""
        n_side = int(np.sqrt(n_total))
        x = np.linspace(x_range[0], x_range[1], n_side)
        y = np.linspace(y_range[0], y_range[1], n_side)
        X, Y = np.meshgrid(x, y)

        if noise_factor > 0:
            dx = (x_range[1] - x_range[0]) / n_side
            noise = (np.random.rand(*X.shape) - 0.5) * dx * noise_factor
            X += noise
            Y += noise

        X = np.clip(X, x_range[0], x_range[1])
        Y = np.clip(Y, y_range[0], y_range[1])
        return np.column_stack([X.ravel(), Y.ravel()])

    @staticmethod
    def bounded_voronoi(points, domain_bounds):
        """构建有界Voronoi图（使用镜像点技术）"""
        x_min, x_max, y_min, y_max = domain_bounds

        # 构造8个方向的镜像点
        p_left = points.copy()
        p_left[:, 0] = 2 * x_min - p_left[:, 0]
        p_right = points.copy()
        p_right[:, 0] = 2 * x_max - p_right[:, 0]
        p_down = points.copy()
        p_down[:, 1] = 2 * y_min - p_down[:, 1]
        p_up = points.copy()
        p_up[:, 1] = 2 * y_max - p_up[:, 1]

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
    def compute_polygon_area(vertices):
        """使用鞋带公式计算多边形面积"""
        x = vertices[:, 0]
        y = vertices[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        return area

    @staticmethod
    def compute_polygon_centroid(vertices):
        """计算多边形质心"""
        return np.mean(vertices, axis=0)


# ============================================================================
# 5. 误差计算工具
# ============================================================================

def compute_l2_error(output, target, area, relative=True):
    """计算L2误差（绝对或相对）"""
    squared_error = (output - target) ** 2
    l2_error_numerator = torch.sqrt(torch.mean(squared_error) * area)

    if relative:
        squared_target = target ** 2
        l2_norm_target = torch.sqrt(torch.mean(squared_target) * area)
        return (l2_error_numerator / l2_norm_target).item()
    else:
        return l2_error_numerator.item()


