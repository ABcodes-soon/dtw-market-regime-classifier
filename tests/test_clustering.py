"""Tests for the clustering engine — run against BOTH the original (src/) and
the fast (src_fast/) implementations via the `clustering_engine` fixture."""

import numpy as np
from sklearn.metrics import silhouette_score


def test_dtw_matrix_shape(clustering_engine):
    """The DTW distance matrix has the correct n x n shape."""
    data = np.random.rand(10, 100)  # 10 tickers, 100 days
    matrix = clustering_engine.compute_dtw_distance_matrix(data)
    assert matrix.shape == (10, 10)


def test_clustering_labels(clustering_engine):
    """Clustering returns one label per ticker and the requested number of groups."""
    data = np.random.rand(10, 100)
    labels = clustering_engine.run_clustering(data, 3)
    assert len(labels) == 10
    assert len(np.unique(labels)) == 3


def test_silhouette_score_valid(clustering_engine):
    """Silhouette scores are within the valid [-1, 1] range."""
    data = np.random.rand(10, 100)
    dtw_matrix = clustering_engine.compute_dtw_distance_matrix(data)
    labels = clustering_engine.run_clustering(data, 3)
    score = silhouette_score(dtw_matrix, labels, metric="precomputed")
    assert -1 <= score <= 1