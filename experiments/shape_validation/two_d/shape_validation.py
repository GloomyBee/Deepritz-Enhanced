from __future__ import annotations

from typing import Any

import numpy as np

from experiments.shape_validation.two_d.basis import (
    build_consistency_metrics_payload,
    compute_lambda_h_stats,
    evaluate_model_consistency_bundle_2d,
    evaluate_rkpm_consistency_bundle_2d,
    grid_points,
    rkpm_shape_matrix_2d,
)


def _infer_grid_shape(x_eval: np.ndarray) -> tuple[int, int]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    x_count = np.unique(np.round(x_eval[:, 0], decimals=12)).size
    y_count = np.unique(np.round(x_eval[:, 1], decimals=12)).size
    if x_count * y_count != x_eval.shape[0]:
        raise ValueError("x_eval must come from a tensor-product grid")
    return int(y_count), int(x_count)


def _select_representative_node_indices(nodes: np.ndarray) -> tuple[list[str], list[int]]:
    targets = [
        ("corner", np.array([0.0, 0.0], dtype=np.float64)),
        ("edge_mid", np.array([0.5, 0.0], dtype=np.float64)),
        ("center", np.array([0.5, 0.5], dtype=np.float64)),
    ]
    chosen: set[int] = set()
    labels: list[str] = []
    indices: list[int] = []
    for label, target in targets:
        order = np.argsort(np.linalg.norm(nodes - target[None, :], axis=1))
        pick = next((int(idx) for idx in order if int(idx) not in chosen), int(order[0]))
        chosen.add(pick)
        labels.append(label)
        indices.append(pick)
    return labels, indices


def _extract_representative_cuts(
    x_eval: np.ndarray,
    nodes: np.ndarray,
    phi_learned: np.ndarray,
    phi_rkpm: np.ndarray,
    representative_indices: list[int],
    representative_labels: list[str],
) -> tuple[dict[str, list[dict[str, float]]], dict[str, np.ndarray]]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    y_unique = np.unique(np.round(x_eval[:, 1], decimals=12))
    cut_x_rows: list[np.ndarray] = []
    cut_learned_rows: list[np.ndarray] = []
    cut_rkpm_rows: list[np.ndarray] = []
    cut_error_rows: list[np.ndarray] = []
    metadata: list[dict[str, float]] = []

    for label, index in zip(representative_labels, representative_indices):
        node_x = float(nodes[index, 0])
        node_y = float(nodes[index, 1])
        actual_y = float(y_unique[np.argmin(np.abs(y_unique - node_y))])
        mask = np.isclose(x_eval[:, 1], actual_y)
        cut_idx = np.flatnonzero(mask)
        order = np.argsort(x_eval[cut_idx, 0])
        ordered = cut_idx[order]
        cut_x_rows.append(x_eval[ordered, 0])
        cut_learned_rows.append(phi_learned[ordered, index])
        cut_rkpm_rows.append(phi_rkpm[ordered, index])
        cut_error_rows.append(np.abs(phi_learned[ordered, index] - phi_rkpm[ordered, index]))
        metadata.append(
            {
                "label": label,
                "node_index": int(index),
                "node_x": node_x,
                "node_y": node_y,
                "requested_cut_y": node_y,
                "actual_cut_y": actual_y,
            }
        )

    return {"representative_cut_metadata": metadata}, {
        "cut_x": np.stack(cut_x_rows, axis=0),
        "cut_learned": np.stack(cut_learned_rows, axis=0),
        "cut_rkpm": np.stack(cut_rkpm_rows, axis=0),
        "cut_abs_error": np.stack(cut_error_rows, axis=0),
    }


def evaluate_shape_metrics_2d(
    *,
    x_eval: np.ndarray,
    nodes: np.ndarray,
    phi_learned: np.ndarray,
    phi_rkpm: np.ndarray,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    x_eval = np.asarray(x_eval, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.float64)
    phi_learned = np.asarray(phi_learned, dtype=np.float64)
    phi_rkpm = np.asarray(phi_rkpm, dtype=np.float64)
    diff = phi_learned - phi_rkpm
    representative_labels, representative_indices = _select_representative_node_indices(nodes)
    metadata_payload, cut_arrays = _extract_representative_cuts(
        x_eval=x_eval,
        nodes=nodes,
        phi_learned=phi_learned,
        phi_rkpm=phi_rkpm,
        representative_indices=representative_indices,
        representative_labels=representative_labels,
    )

    y_count, x_count = _infer_grid_shape(x_eval)
    lambda_stats = compute_lambda_h_stats(phi_learned)
    pointwise_relative_error = np.linalg.norm(diff, axis=1) / np.maximum(np.linalg.norm(phi_rkpm, axis=1), 1.0e-16)
    linear_x = phi_learned @ nodes[:, 0] - x_eval[:, 0]
    linear_y = phi_learned @ nodes[:, 1] - x_eval[:, 1]
    pu = np.sum(phi_learned, axis=1) - 1.0

    representative_l2 = {}
    representative_linf = {}
    for label, index in zip(representative_labels, representative_indices):
        representative_l2[f"{label}_l2"] = float(np.sqrt(np.mean(diff[:, index] ** 2)))
        representative_linf[f"{label}_linf"] = float(np.max(np.abs(diff[:, index])))

    metrics: dict[str, Any] = {
        "global_l2": float(np.sqrt(np.mean(diff**2))),
        "global_linf": float(np.max(np.abs(diff))),
        "shape_relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(phi_rkpm), 1.0e-16)),
        "shape_relative_linf": float(np.max(np.abs(diff)) / max(float(np.max(np.abs(phi_rkpm))), 1.0e-16)),
        "pu_max_error": float(np.max(np.abs(pu))),
        "linear_x_rmse": float(np.sqrt(np.mean(linear_x**2))),
        "linear_y_rmse": float(np.sqrt(np.mean(linear_y**2))),
        "lambda_h_max": float(lambda_stats["lambda_h_max"]),
        "lambda_h_mean": float(lambda_stats["lambda_h_mean"]),
        "representative_labels": representative_labels,
        "representative_node_indices": [int(index) for index in representative_indices],
        **metadata_payload,
        **representative_l2,
        **representative_linf,
    }
    arrays = {
        "pointwise_relative_error": pointwise_relative_error,
        "global_abs_error_field": np.max(np.abs(diff), axis=1).reshape(y_count, x_count),
        "pu_field": pu.reshape(y_count, x_count),
        "linear_x_field": linear_x.reshape(y_count, x_count),
        "linear_y_field": linear_y.reshape(y_count, x_count),
        "lambda_h_field": np.sum(np.abs(phi_learned), axis=1).reshape(y_count, x_count),
        **cut_arrays,
    }
    return metrics, arrays


def build_shape_validation_metrics_payload_2d(
    *,
    case: dict[str, Any],
    learned_shape: dict[str, Any],
    rkpm_shape: dict[str, Any],
    learned_consistency: dict[str, Any],
    rkpm_consistency: dict[str, Any],
) -> dict[str, Any]:
    comparison: dict[str, float] = {}
    for key, value in learned_shape.items():
        if isinstance(value, (int, float)) and key in rkpm_shape and isinstance(rkpm_shape[key], (int, float)):
            comparison[f"learned_minus_rkpm_shape_{key}"] = float(value) - float(rkpm_shape[key])
            if abs(float(rkpm_shape[key])) > 1.0e-16:
                comparison[f"learned_over_rkpm_shape_{key}"] = float(value) / float(rkpm_shape[key])
    comparison.update(
        build_consistency_metrics_payload(case={}, learned=learned_consistency, rkpm=rkpm_consistency)["comparison"]
    )
    return {
        "case": case,
        "learned": {"shape": learned_shape, "consistency": learned_consistency},
        "rkpm": {"shape": rkpm_shape, "consistency": rkpm_consistency},
        "comparison": comparison,
    }


def build_shape_validation_summary_lines_2d(payload: dict[str, Any]) -> list[str]:
    case = payload["case"]
    learned_shape = payload["learned"]["shape"]
    learned_consistency = payload["learned"]["consistency"]
    return [
        f"case: {case}",
        f"shape_relative_l2: {learned_shape['shape_relative_l2']:.6e}",
        f"global_l2: {learned_shape['global_l2']:.6e}",
        f"corner_l2: {learned_shape['corner_l2']:.6e}",
        f"edge_mid_l2: {learned_shape['edge_mid_l2']:.6e}",
        f"center_l2: {learned_shape['center_l2']:.6e}",
        f"pu_max_error: {learned_shape['pu_max_error']:.6e}",
        f"linear_x_rmse: {learned_shape['linear_x_rmse']:.6e}",
        f"linear_y_rmse: {learned_shape['linear_y_rmse']:.6e}",
        f"lambda_h_max: {learned_shape['lambda_h_max']:.6e}",
        f"mass_sum_residual: {learned_consistency['mass_sum_residual']:.6e}",
        f"moment_matrix_residual_fro: {learned_consistency['moment_matrix_residual_fro']:.6e}",
        f"derivative_matrix_residual_fro: {learned_consistency['derivative_matrix_residual_fro']:.6e}",
    ]


def evaluate_shape_validation_case_2d(
    *,
    model: Any,
    nodes: np.ndarray,
    support_radius: float,
    grid_resolution: int,
    quadrature_order: int,
    device: str,
    case_meta: dict[str, Any],
) -> dict[str, Any]:
    x_grid, y_grid, x_eval = grid_points(grid_resolution)
    with np.errstate(all="ignore"):
        import torch

        x_eval_t = torch.tensor(x_eval, device=device, dtype=model.nodes.dtype)
        with torch.no_grad():
            phi_learned = model.compute_shape_functions(x_eval_t).cpu().numpy()
    phi_rkpm = rkpm_shape_matrix_2d(x_eval=x_eval, nodes=nodes, support_radius=support_radius)
    learned_shape_metrics, shape_arrays = evaluate_shape_metrics_2d(
        x_eval=x_eval,
        nodes=nodes,
        phi_learned=phi_learned,
        phi_rkpm=phi_rkpm,
    )
    rkpm_shape_metrics = {
        "global_l2": 0.0,
        "global_linf": 0.0,
        "shape_relative_l2": 0.0,
        "shape_relative_linf": 0.0,
        "corner_l2": 0.0,
        "corner_linf": 0.0,
        "edge_mid_l2": 0.0,
        "edge_mid_linf": 0.0,
        "center_l2": 0.0,
        "center_linf": 0.0,
        "pu_max_error": float(np.max(np.abs(np.sum(phi_rkpm, axis=1) - 1.0))),
        "linear_x_rmse": float(np.sqrt(np.mean((phi_rkpm @ nodes[:, 0] - x_eval[:, 0]) ** 2))),
        "linear_y_rmse": float(np.sqrt(np.mean((phi_rkpm @ nodes[:, 1] - x_eval[:, 1]) ** 2))),
        "lambda_h_max": float(np.max(np.sum(np.abs(phi_rkpm), axis=1))),
        "lambda_h_mean": float(np.mean(np.sum(np.abs(phi_rkpm), axis=1))),
        "representative_labels": learned_shape_metrics["representative_labels"],
        "representative_node_indices": learned_shape_metrics["representative_node_indices"],
        "representative_cut_metadata": learned_shape_metrics["representative_cut_metadata"],
    }
    learned_consistency_bundle = evaluate_model_consistency_bundle_2d(
        model=model,
        quadrature_order=quadrature_order,
        device=device,
    )
    rkpm_consistency_bundle = evaluate_rkpm_consistency_bundle_2d(
        nodes=nodes,
        support_radius=support_radius,
        quadrature_order=quadrature_order,
        device=device,
    )
    metrics = build_shape_validation_metrics_payload_2d(
        case=case_meta,
        learned_shape=learned_shape_metrics,
        rkpm_shape=rkpm_shape_metrics,
        learned_consistency=learned_consistency_bundle["metrics"],
        rkpm_consistency=rkpm_consistency_bundle["metrics"],
    )
    arrays: dict[str, np.ndarray] = {
        "x_grid": x_grid,
        "y_grid": y_grid,
        "x_eval": x_eval,
        "nodes": np.asarray(nodes, dtype=np.float64),
        "phi_learned": phi_learned,
        "phi_rkpm": phi_rkpm,
        "representative_node_indices": np.asarray(learned_shape_metrics["representative_node_indices"], dtype=np.int64),
        "representative_cut_y": np.asarray(
            [item["actual_cut_y"] for item in learned_shape_metrics["representative_cut_metadata"]],
            dtype=np.float64,
        ),
    }
    arrays.update(shape_arrays)
    return {
        "metrics": metrics,
        "arrays": arrays,
        "shape_arrays": shape_arrays,
        "learned_consistency_bundle": learned_consistency_bundle,
        "rkpm_consistency_bundle": rkpm_consistency_bundle,
    }
