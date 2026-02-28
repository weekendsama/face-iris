from __future__ import annotations

import numpy as np


def _run_kmeans(features: np.ndarray, num_clusters: int, num_iters: int) -> np.ndarray:
    num_samples = features.shape[0]
    if num_samples == 0:
        raise ValueError("features must contain at least one sample")

    num_clusters = max(1, min(num_clusters, num_samples))
    centroids = features[:num_clusters].copy()

    for _ in range(max(1, num_iters)):
        distances = np.sum((features[:, None, :] - centroids[None, :, :]) ** 2, axis=-1)
        assignments = np.argmin(distances, axis=1)

        next_centroids = centroids.copy()
        for idx in range(num_clusters):
            members = features[assignments == idx]
            if len(members) > 0:
                next_centroids[idx] = members.mean(axis=0)
        centroids = next_centroids

    return centroids


def fit_dsh_projections(
    features: np.ndarray,
    num_bits: int,
    num_clusters: int = 8,
    num_iters: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """Approximate DSH projections by separating centroid pairs with max-entropy splits."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")

    num_samples, feat_dim = features.shape
    if num_samples == 0:
        raise ValueError("features must contain at least one sample")

    centroids = _run_kmeans(features, num_clusters=num_clusters, num_iters=num_iters)

    candidates: list[tuple[float, np.ndarray, float]] = []
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            direction = centroids[i] - centroids[j]
            norm = np.linalg.norm(direction)
            if norm < 1e-6:
                continue

            normal = direction / norm
            midpoint = 0.5 * (centroids[i] + centroids[j])
            bias = -float(np.dot(normal, midpoint))
            logits = features @ normal + bias
            positive_ratio = np.mean(logits >= 0.0)
            positive_ratio = float(np.clip(positive_ratio, 1e-6, 1.0 - 1e-6))
            entropy = -(
                positive_ratio * np.log(positive_ratio)
                + (1.0 - positive_ratio) * np.log(1.0 - positive_ratio)
            )
            candidates.append((entropy, normal.astype(np.float32), bias))

    if not candidates:
        eye = np.eye(feat_dim, dtype=np.float32)
        repeats = int(np.ceil(num_bits / feat_dim))
        weights = np.tile(eye, (1, repeats))[:, :num_bits]
        bias = np.zeros((num_bits,), dtype=np.float32)
        return weights, bias

    candidates.sort(key=lambda item: item[0], reverse=True)
    top = candidates[:num_bits]

    weights = np.stack([item[1] for item in top], axis=1)
    bias = np.asarray([item[2] for item in top], dtype=np.float32)

    if weights.shape[1] < num_bits:
        repeats = int(np.ceil(num_bits / weights.shape[1]))
        weights = np.tile(weights, (1, repeats))[:, :num_bits]
        bias = np.tile(bias, repeats)[:num_bits]

    return weights.astype(np.float32), bias.astype(np.float32)
