from __future__ import annotations

from typing import Any

import numpy as np


def pca_2d(values: Any) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] < 2:
        raise ValueError("PCA requires a 2D matrix containing at least two samples")
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    _u, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    points = centered @ components.T
    if points.shape[1] == 1:
        points = np.column_stack([points[:, 0], np.zeros(matrix.shape[0])])
    total_variance = float(np.square(singular_values).sum())
    explained = (
        (np.square(singular_values[:2]) / total_variance).tolist()
        if total_variance > 0
        else [0.0, 0.0]
    )
    if len(explained) == 1:
        explained.append(0.0)
    return points, {"explained_variance_ratio": explained}


def reduce_2d(
    values: Any,
    method: str,
    *,
    seed: int = 42,
    perplexity: float = 30.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(values, dtype=np.float64)
    if method == "pca":
        return pca_2d(matrix)
    if method == "tsne":
        try:
            from sklearn.manifold import TSNE
        except ImportError as exc:
            raise RuntimeError("t-SNE requires the optional 'analysis' dependencies") from exc
        if matrix.shape[0] < 3:
            raise ValueError("t-SNE requires at least three samples")
        effective_perplexity = min(perplexity, max(1.0, matrix.shape[0] - 1.0))
        points = TSNE(
            n_components=2,
            perplexity=effective_perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(matrix)
        return points, {"perplexity": effective_perplexity}
    if method == "umap":
        try:
            import umap
        except ImportError as exc:
            raise RuntimeError("UMAP requires 'pip install umap-learn'") from exc
        points = umap.UMAP(n_components=2, random_state=seed).fit_transform(matrix)
        return points, {}
    raise ValueError(f"Unknown reducer: {method}")


def linear_cka(left: Any, right: Any) -> float:
    """Centered kernel alignment for cross-layer/model representation similarity."""
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError("CKA inputs must be 2D matrices with the same sample count")
    x -= x.mean(axis=0, keepdims=True)
    y -= y.mean(axis=0, keepdims=True)
    cross = np.linalg.norm(x.T @ y, ord="fro") ** 2
    denominator = np.linalg.norm(x.T @ x, ord="fro") * np.linalg.norm(y.T @ y, ord="fro")
    return float(cross / denominator) if denominator else 0.0
