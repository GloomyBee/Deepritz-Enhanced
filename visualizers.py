"""
可视化模块 (Visualizers)
提供统一的可视化工具用于Deep Ritz方法：
1. plot_solution - 解的可视化（2D等值线图）
2. plot_training_history - 训练历史曲线
3. plot_comparison - 预测/真实/误差三联图
4. plot_temperature_field - 温度场可视化（梯形域）
5. plot_particles - 粒子分布可视化
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from typing import Optional, Tuple


# ==========================================
# 全局配置
# ==========================================
OUTPUT_DIR = "output"
GLOBAL_FIGSIZE = (10, 8)
GLOBAL_CMAP = "inferno"

def _get_save_path(filename: str, subdir: Optional[str] = None) -> str:
    """
    内部辅助函��：组合输出路径
    Args:
        filename: 文件名
        subdir: 子目录（可选）
    Returns:
        完整路径
    """
    if subdir:
        full_dir = os.path.join(OUTPUT_DIR, subdir)
        os.makedirs(full_dir, exist_ok=True)
        return os.path.join(full_dir, filename)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return os.path.join(OUTPUT_DIR, filename)


# ==========================================
# 1. 正方形域可视化
# ==========================================
def plot_square_triplet(
    X: np.ndarray,
    Y: np.ndarray,
    pred: np.ndarray,
    exact: np.ndarray,
    *,
    title_prefix: str = "Deep Ritz",
    filename: str = "solution_triplet.png",
    cmap: str = "inferno",
    val_clim: Tuple[float, float] = (0.0, 1.0),
    err_vmax: float = 0.05,
    subdir: Optional[str] = None
):
    """
    预测/解析/误差三联图（正方形域）

    Args:
        X, Y: 网格坐标
        pred: 预测值
        exact: 真实值
        title_prefix: 标题前缀
        filename: 保存文件名
        cmap: 色图
        val_clim: 值域范围
        err_vmax: 误差最大值
        subdir: 子目录
    """
    error = np.abs(pred - exact)

    # 定义层级
    v_min, v_max = val_clim
    val_levels = np.linspace(v_min, v_max, 101)
    err_levels = np.linspace(0, err_vmax, 101)

    # 定义刻度
    val_ticks = np.linspace(v_min, v_max, 11)
    err_ticks = np.linspace(0, err_vmax, 6)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 子图1: 预测值
    c0 = axes[0].contourf(X, Y, pred, levels=val_levels, cmap=cmap,
                          vmin=v_min, vmax=v_max, extend='both')
    cb0 = plt.colorbar(c0, ax=axes[0], fraction=0.046, pad=0.04, ticks=val_ticks)
    axes[0].set_title(f"{title_prefix} 预测")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    # 子图2: 真实值
    c1 = axes[1].contourf(X, Y, exact, levels=val_levels, cmap=cmap,
                          vmin=v_min, vmax=v_max, extend='both')
    cb1 = plt.colorbar(c1, ax=axes[1], fraction=0.046, pad=0.04, ticks=val_ticks)
    axes[1].set_title("解析解")
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")

    # 子图3: 误差
    c2 = axes[2].contourf(X, Y, error, levels=err_levels, cmap="magma",
                          vmin=0, vmax=err_vmax, extend='max')
    cb2 = plt.colorbar(c2, ax=axes[2], fraction=0.046, pad=0.04, ticks=err_ticks)
    axes[2].set_title("绝对误差")
    axes[2].set_aspect("equal")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


def plot_solution(
    X: np.ndarray,
    Y: np.ndarray,
    U: np.ndarray,
    *,
    title: str = "解场",
    filename: str = "solution.png",
    cmap: str = GLOBAL_CMAP,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    subdir: Optional[str] = None
):
    """
    绘制单个解场

    Args:
        X, Y: 网格坐标
        U: 解值
        title: 标题
        filename: 文件名
        cmap: 色图
        vmin, vmax: 值域范围
        subdir: 子目录
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))

    c = ax.contourf(X, Y, U, levels=100, cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(c, ax=ax, label="值")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title, fontsize=14)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


# ==========================================
# 2. 训练历史可视化
# ==========================================
def plot_error_history(
    steps: np.ndarray,
    l2: np.ndarray,
    h1: np.ndarray,
    *,
    title: str = "误差收敛曲线",
    filename: str = "error_history.png",
    ylim: Optional[Tuple[float, float]] = None,
    subdir: Optional[str] = None
):
    """
    绘制误差收敛曲线

    Args:
        steps: 训练步数
        l2: L2误差
        h1: H1误差
        title: 标题
        filename: 文件名
        ylim: Y轴范围
        subdir: 子目录
    """
    steps = np.asarray(steps)
    l2 = np.asarray(l2)
    h1 = np.asarray(h1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    # 绘制曲线
    ax.semilogy(steps, np.maximum(l2, 1e-16), label="L2误差",
                linewidth=2, color='#1f77b4')
    ax.semilogy(steps, np.maximum(h1, 1e-16), label="H1误差（半范数）",
                linewidth=2, color='orange')

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel("训练步数", fontsize=12)
    ax.set_ylabel("误差", fontsize=12)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=12)

    if ylim is not None:
        ax.set_ylim(ylim)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


def plot_training_history(
    history: dict,
    *,
    filename: str = "training_history.png",
    subdir: Optional[str] = None
):
    """
    绘制完整训练历史（损失+误差）

    Args:
        history: 训练历史字典（包含steps, loss, l2_error, h1_error等）
        filename: 文件名
        subdir: 子目录
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    steps = np.array(history['steps'])

    # 子图1: 损失
    ax = axes[0]
    if 'loss' in history:
        ax.semilogy(steps, history['loss'], label="总损失", linewidth=2)
    if 'energy' in history:
        ax.semilogy(steps, np.abs(history['energy']), label="能量", linewidth=2)
    if 'boundary' in history:
        ax.semilogy(steps, history['boundary'], label="边界损失", linewidth=2)

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel("训练步数", fontsize=12)
    ax.set_ylabel("损失", fontsize=12)
    ax.set_title("损失曲线", fontsize=13)
    ax.legend(fontsize=10)

    # 子图2: 误差
    ax = axes[1]
    if 'l2_error' in history:
        ax.semilogy(steps, np.maximum(history['l2_error'], 1e-16),
                    label="L2误差", linewidth=2, color='#1f77b4')
    if 'h1_error' in history:
        ax.semilogy(steps, np.maximum(history['h1_error'], 1e-16),
                    label="H1误差", linewidth=2, color='orange')

    ax.grid(True, which="both", ls="--", alpha=0.5)
    ax.set_xlabel("训练步数", fontsize=12)
    ax.set_ylabel("误差", fontsize=12)
    ax.set_title("误差收敛", fontsize=13)
    ax.legend(fontsize=10)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


# ==========================================
# 3. 梯形域可视化（热传导问题专用）
# ==========================================
def trapezoid_mask(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """梯形域掩码"""
    return (X + Y) > 2.0 + 1e-5


def plot_domain_outline(ax):
    """绘制梯形边框"""
    ax.plot([0, 2, 1, 0, 0], [0, 0, 1, 1, 0], "k-", linewidth=1.5, alpha=0.5)


def plot_temperature_grid(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    *,
    title: str = "温度场",
    filename: str = "temperature.png",
    vmin: float = 20.0,
    vmax: float = 28.0,
    cmap: str = GLOBAL_CMAP,
    subdir: Optional[str] = None
):
    """
    绘制梯形域温度场

    Args:
        X, Y: 网格坐标
        T: 温度值
        title: 标题
        filename: 文件名
        vmin, vmax: 温度范围
        cmap: 色图
        subdir: 子目录
    """
    levels = np.linspace(vmin, vmax, 101)

    mask = trapezoid_mask(X, Y)
    T_plot = np.array(T, copy=True)
    T_plot[mask] = np.nan

    fig, ax = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE)

    c = ax.contourf(X, Y, T_plot, levels=levels, cmap=cmap, extend='both')

    cbar = plt.colorbar(c, ax=ax, label="温度 (°C)")
    cbar.set_ticks(np.linspace(vmin, vmax, 9))

    plot_domain_outline(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=14)

    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(-0.1, 1.6)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


def plot_temperature_tri(
    nodes: np.ndarray,
    elems: np.ndarray,
    U: np.ndarray,
    *,
    title: str = "FEM温度场",
    filename: str = "temperature_fem.png",
    vmin: float = 20.0,
    vmax: float = 28.0,
    cmap: str = GLOBAL_CMAP,
    subdir: Optional[str] = None
):
    """
    绘制FEM三角网格温度场

    Args:
        nodes: [N, 2] 节点坐标
        elems: [M, 3] 单元连接
        U: [N] 温度值
        title: 标题
        filename: 文件名
        vmin, vmax: 温度范围
        cmap: 色图
        subdir: 子目录
    """
    levels = np.linspace(vmin, vmax, 101)

    triang = mtri.Triangulation(nodes[:, 0], nodes[:, 1], elems)

    fig, ax = plt.subplots(1, 1, figsize=GLOBAL_FIGSIZE)

    tpc = ax.tricontourf(triang, U.reshape(-1), levels=levels, cmap=cmap, extend='both')

    cbar = plt.colorbar(tpc, ax=ax, label="温度 (°C)")
    cbar.set_ticks(np.linspace(vmin, vmax, 9))

    plot_domain_outline(ax)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title, fontsize=14)

    ax.set_xlim(-0.1, 2.3)
    ax.set_ylim(-0.1, 1.6)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


# ==========================================
# 4. 粒子/节点分布可视化
# ==========================================
def plot_particles(
    nodes: np.ndarray,
    *,
    title: str = "粒子分布",
    filename: str = "particles.png",
    marker_size: int = 10,
    subdir: Optional[str] = None
):
    """
    绘制粒子分布图

    Args:
        nodes: [N, 2] 粒子坐标
        title: 标题
        filename: 文件名
        marker_size: 标记大小
        subdir: 子目录
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))

    ax.scatter(nodes[:, 0], nodes[:, 1], s=marker_size, c='b', alpha=0.6)
    ax.set_title(f"{title} (N={nodes.shape[0]})", fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


def plot_voronoi_cells(
    nodes: np.ndarray,
    regions,
    vertices: np.ndarray,
    *,
    title: str = "Voronoi单元",
    filename: str = "voronoi.png",
    subdir: Optional[str] = None
):
    """
    绘制Voronoi单元图

    Args:
        nodes: [N, 2] 粒子坐标
        regions: Voronoi区域列表
        vertices: Voronoi顶点坐标
        title: 标题
        filename: 文件名
        subdir: 子目录
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # 绘制Voronoi单元
    for i, region in enumerate(regions):
        if -1 in region or len(region) < 3:
            continue
        polygon = vertices[region]
        ax.fill(polygon[:, 0], polygon[:, 1], alpha=0.2, edgecolor='k', linewidth=0.5)

    # 绘制粒子
    ax.scatter(nodes[:, 0], nodes[:, 1], s=20, c='r', zorder=10)

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)


# ==========================================
# 5. 形函数可视化
# ==========================================
def visualize_shape_function(
    model,
    node_idx: int,
    device: str = 'cpu',
    domain_bounds: Tuple[float, float, float, float] = (0, 1, 0, 1),
    *,
    filename: str = "shape_function.png",
    subdir: Optional[str] = None
):
    """
    可视化指定节点的形函数

    Args:
        model: 网络模型（必须有meshfree_layer或rkpm属性）
        node_idx: 节点索引
        device: 计算设备
        domain_bounds: 定义域边界
        filename: 文件名
        subdir: 子目录
    """
    import torch

    # 获取节点坐标
    if hasattr(model, 'rkpm'):
        nodes = model.rkpm.nodes.cpu().numpy()
        layer = model.rkpm
    elif hasattr(model, 'meshfree_layer'):
        nodes = model.meshfree_layer.nodes.cpu().numpy()
        layer = model.meshfree_layer
    else:
        raise ValueError("Model must have 'rkpm' or 'meshfree_layer' attribute")

    node_pos = nodes[node_idx]

    # 生成密集网格
    x_min, x_max, y_min, y_max = domain_bounds
    res = 100
    x = np.linspace(x_min, x_max, res)
    y = np.linspace(y_min, y_max, res)
    X, Y = np.meshgrid(x, y)
    grid_pts = torch.tensor(np.column_stack([X.ravel(), Y.ravel()]),
                           dtype=torch.float32, device=device)

    # 计算形函数值
    model.eval()
    with torch.no_grad():
        phi_vals, _, _ = layer(grid_pts)
        phi_target = phi_vals[:, node_idx].cpu().numpy().reshape(res, res)

    # 绘图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 左图：2D等值线
    ax = axes[0]
    c = ax.contourf(X, Y, phi_target, levels=50, cmap='viridis')
    plt.colorbar(c, ax=ax)
    ax.plot(node_pos[0], node_pos[1], 'r*', markersize=15, label='节点')
    ax.set_title(f"形函数 φ_{node_idx}(x,y)")
    ax.legend()
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")

    # 右图：径向剖面
    ax = axes[1]
    row_idx = np.argmin(np.abs(y - node_pos[1]))
    phi_slice = phi_target[row_idx, :]
    r_dist = x - node_pos[0]

    ax.plot(r_dist, phi_slice, linewidth=3, label='形函数剖面')
    ax.set_xlabel("距节点距离 (x - x_I)")
    ax.set_ylabel("φ 值")
    ax.set_title("径向剖面")
    ax.grid(True, linestyle='--', alpha=0.6)

    # 标记支持域
    if hasattr(layer, 'dilation'):
        if isinstance(layer.dilation, torch.Tensor):
            if layer.dilation.numel() == 1:
                dilation = layer.dilation.item()
            else:
                dilation = layer.dilation[node_idx].item()
        else:
            dilation = layer.dilation
        ax.axvline(dilation, color='k', linestyle=':', label='支持域半径')
        ax.axvline(-dilation, color='k', linestyle=':')

    ax.legend()

    fig.tight_layout()

    save_path = _get_save_path(filename, subdir)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"已保存: {save_path}")
    plt.close(fig)
