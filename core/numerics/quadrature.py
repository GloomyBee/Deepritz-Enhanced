from __future__ import annotations

import numpy as np


def gauss_legendre_interval_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    points, weights = np.polynomial.legendre.leggauss(order)
    points = 0.5 * (points + 1.0)
    weights = 0.5 * weights
    return points.astype(np.float64), weights.astype(np.float64)


def gauss_legendre_1d(order: int) -> tuple[np.ndarray, np.ndarray]:
    return gauss_legendre_interval_1d(order)


def square_domain_quadrature(order: int) -> tuple[np.ndarray, np.ndarray]:
    points_1d, weights_1d = gauss_legendre_1d(order)
    x_grid, y_grid = np.meshgrid(points_1d, points_1d, indexing="xy")
    w_grid = np.outer(weights_1d, weights_1d)
    points = np.column_stack([x_grid.ravel(), y_grid.ravel()])
    weights = w_grid.ravel()
    return points.astype(np.float64), weights.astype(np.float64)


def square_boundary_quadrature(order: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_1d, weights_1d = gauss_legendre_1d(order)
    left = np.column_stack([np.zeros_like(points_1d), points_1d])
    right = np.column_stack([np.ones_like(points_1d), points_1d])
    bottom = np.column_stack([points_1d, np.zeros_like(points_1d)])
    top = np.column_stack([points_1d, np.ones_like(points_1d)])
    points = np.vstack([left, right, bottom, top]).astype(np.float64)
    weights = np.tile(weights_1d, 4).astype(np.float64)
    normals = np.vstack(
        [
            np.tile(np.array([[-1.0, 0.0]], dtype=np.float64), (order, 1)),
            np.tile(np.array([[1.0, 0.0]], dtype=np.float64), (order, 1)),
            np.tile(np.array([[0.0, -1.0]], dtype=np.float64), (order, 1)),
            np.tile(np.array([[0.0, 1.0]], dtype=np.float64), (order, 1)),
        ]
    )
    return points, weights, normals
