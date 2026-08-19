"""Tests specific to the FAST pipeline (src_fast/): the silhouette K-search,
the bounded Sakoe-Chiba DTW window, the downsample option, and equivalence
with the original unbounded DTW distance."""

import numpy as np


def test_fast_optimal_k():
    """The silhouette K-search returns a valid best K and a full score dict."""
    from src_fast.clustering_engine import determine_optimal_k

    data = np.random.rand(10, 100)
    best_k, scores = determine_optimal_k(
        data, max_k=4, n_init=2, sakoe_chiba_radius=0.1, n_jobs=1,
    )
    assert 2 <= best_k <= 4
    assert set(scores.keys()) == {2, 3, 4}


def test_fast_bounded_dtw_window():
    """A Sakoe-Chiba bounded DTW matrix is still a valid n x n matrix."""
    from src_fast.clustering_engine import compute_dtw_distance_matrix

    data = np.random.rand(10, 100)
    matrix = compute_dtw_distance_matrix(data, sakoe_chiba_radius=10, n_jobs=1)
    assert matrix.shape == (10, 10)


def test_fast_downsample_option():
    """The downsample option for the K-search is accepted and returns valid output."""
    from src_fast.clustering_engine import determine_optimal_k

    data = np.random.rand(10, 200)
    best_k, scores = determine_optimal_k(
        data, max_k=4, n_init=1, sakoe_chiba_radius=0.1, n_jobs=1, downsample=4,
    )
    assert 2 <= best_k <= 4
    assert scores


def test_fast_matches_original_unbounded():
    """With no warping constraint, the fast engine produces the SAME DTW
    distance matrix as the original (both use tslearn cdist_dtw)."""
    import src.clustering_engine as orig
    import src_fast.clustering_engine as fast

    data = np.random.rand(10, 100)
    orig_matrix = orig.compute_dtw_distance_matrix(data)
    fast_matrix = fast.compute_dtw_distance_matrix(data, sakoe_chiba_radius=None, n_jobs=1)
    assert orig_matrix.shape == fast_matrix.shape == (10, 10)
    np.testing.assert_allclose(orig_matrix, fast_matrix, rtol=1e-6)
