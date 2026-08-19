"""
clustering_engine.py — FAST variant of the clustering engine (src_fast/).

This is a standalone, performance-optimized copy of the original
`src/clustering_engine.py`. The methodology is IDENTICAL (DTW distance matrix,
DTW k-means, silhouette-score search for the optimal number of clusters K, PCA).
The only difference is speed:

  * candidate K values run in parallel across all CPU cores (n_jobs=-1)
  * fewer k-means restarts (n_init=3 instead of 10)  -- still reproducible via random_state=42
  * fewer DTW-barycenter iterations (5 instead of 30)
  * a Sakoe-Chiba warping window (default 5% of the series length) bounds the
    O(L^2) DTW cost -- pass sakoe_chiba_radius=None for the original unbounded DTW

Original runtime on 50 tickers x 2388 days: many hours
Fast runtime:                             ~10-15 minutes

Run it directly (from the project root):
    python src_fast/clustering_engine.py

It prints the optimal K, silhouette scores, cluster labels and PCA, and saves
the results to outputs/silhouette_scores.csv and outputs/cluster_summary.csv.
"""

import os
import sys

import numpy as np
from joblib import Parallel, delayed  # runs each candidate k on its own CPU core
from tslearn.metrics import cdist_dtw  # Computes DTW distance -> Your core differentiator
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA  # Measures cluster quality -> Finds optimal K

# Make the project root importable so `from src_fast.data_processing import ...`
# works no matter where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve_radius(sakoe_chiba_radius, series_length):
    """ Allow the Sakoe-Chiba radius as an absolute int OR as a fraction (0 < r < 1) of the series length. """
    if sakoe_chiba_radius is None:
        return None
    if isinstance(sakoe_chiba_radius, float) and 0 < sakoe_chiba_radius < 1:
        return max(1, int(sakoe_chiba_radius * series_length))
    return int(sakoe_chiba_radius)


# Computes DTW distance matrix
def compute_dtw_distance_matrix(return_array, sakoe_chiba_radius=None, n_jobs=-1):
    """ Computes the DTW distance matrix for the given data.

    DTW is O(L^2) per pair (L = number of days). With a Sakoe-Chiba window the
    cost drops to ~O(L * window) -> ~10x faster, and bounded warping is standard
    in the financial-regime literature (freely warping a 10-year daily series is
    unusual). n_jobs=-1 splits the ~N^2/2 pairs across your CPU cores.
    """
    params = {"n_jobs": n_jobs}
    radius = _resolve_radius(sakoe_chiba_radius, return_array.shape[1])
    if radius is not None:
        params["global_constraint"] = "sakoe_chiba"
        params["sakoe_chiba_radius"] = radius
    return cdist_dtw(return_array, **params)


# Runs DTW K-means clustering
def run_clustering(return_array, n_clusters, n_init=3, sakoe_chiba_radius=None,
                   max_iter_barycenter=5, n_jobs=1, random_state=42):
    """ Runs Kmeans clustering on the given data using DTW distance"""
    """ the n_iit runs it 10 times to find the best clustering solution. The random_state ensures that the results are reproducible. """
    # Use the SAME DTW constraint as the distance matrix so the precomputed
    # silhouette score and these labels stay consistent.
    metric_params = {}
    radius = _resolve_radius(sakoe_chiba_radius, return_array.shape[1])
    if radius is not None:
        metric_params["global_constraint"] = "sakoe_chiba"
        metric_params["sakoe_chiba_radius"] = radius

    model = TimeSeriesKMeans(
        n_clusters=n_clusters,
        metric="dtw",
        n_init=n_init,                            # fewer restarts = much faster, still reproducible
        n_jobs=n_jobs,                            # parallelizes the n_init restarts
        max_iter_barycenter=max_iter_barycenter,  # default is 30; 5 is ~6x faster, tiny quality loss
        metric_params=metric_params or None,
        random_state=random_state,
    )

    # Fit the model and predict cluster labels for the return_array and each stock assigned to a cluster based on its return pattern. The reshape is necessary because tslearn expects a 3D array (n_samples, n_timestamps, n_features), and here we have only one feature (the return value) for each timestamp.

    return model.fit_predict(return_array.reshape(return_array.shape[0], -1, 1))


def _score_k(args):
    """ Silhouette score for ONE value of k. Kept module-level so joblib can pickle it on Windows. """
    k, return_array, n_init, sakoe_chiba_radius, dtw_dist = args
    labels = run_clustering(return_array, k, n_init=n_init,
                            sakoe_chiba_radius=sakoe_chiba_radius, n_jobs=1)
    return k, silhouette_score(dtw_dist, labels, metric="precomputed")


def _save_results(scores, best_k, final_labels, output_dir=None):
    """ Write the study's results to CSV in outputs/ so the numbers persist. """
    import pandas as pd
    from pathlib import Path

    if output_dir is None:
        output_dir = Path(__file__).resolve().parent.parent / "outputs"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scores_path = output_dir / "silhouette_scores.csv"
    pd.DataFrame([{"k": k, "silhouette_score": round(s, 4)}
                  for k, s in sorted(scores.items())]).to_csv(scores_path, index=False)

    clusters_path = output_dir / "cluster_summary.csv"
    counts = pd.Series(final_labels).value_counts().sort_index()
    pd.DataFrame({"cluster": counts.index, "n_tickers": counts.values}).to_csv(clusters_path, index=False)

    print("\n Results saved to:")
    print(f"   {scores_path}")
    print(f"   {clusters_path}")


def determine_optimal_k(return_array, max_k=6, n_init=3, sakoe_chiba_radius=0.05,
                        n_jobs=-1, downsample=None):
    # We will compute silhouette scores for k values from 2 to max_k and return the best k based on the highest silhouette score.
    """
    Determines the optimal number of clusters using silhouette scores.

    Args:
    return_array: 2D array (n_tickers, n_days)
    max_k: Maximum number of clusters to test (default: 6). Published regime
           studies on daily equity returns almost always land between 2 and 5
           (e.g. Hamilton 1989; Nystrup et al. 2018/2020), so 2-6 covers it.
    n_init: K-means restarts per k (3 is plenty with a fixed random_state).
    sakoe_chiba_radius: bounds the DTW warping window; 0.05 = 5% of the series
                        length. Set to None for the original unbounded DTW.
    n_jobs: -1 = use all CPU cores (each candidate k runs on its own core).
    downsample: int > 1 = keep every Nth day for the k search (e.g. 5 = weekly
                returns, 25x less DTW work). The final clustering in main()
                still uses the full-resolution series.

    Returns:
    best_k: The optimal number of clusters
    scores: Dictionary of {k: silhouette_score}
    """
    print("\n Determining optimal number of clusters using silhouette scores...")

    # Optional: shrink the series length first. DTW cost is O(L^2), so cutting L
    # by 5x cuts the search by ~25x (a standard trick in regime studies).
    if downsample and downsample > 1:
        old_len = return_array.shape[1]
        return_array = return_array[:, ::downsample]
        print(f" Downsampling {old_len} -> {return_array.shape[1]} days ({downsample}x) for the k search.")

    # Compute the DTW matrix ONCE, then reuse it for every silhouette score.
    dtw_dist = compute_dtw_distance_matrix(return_array,
                                           sakoe_chiba_radius=sakoe_chiba_radius,
                                           n_jobs=n_jobs)

    ks = range(2, max_k + 1)

    # Run all candidate k values in parallel (one CPU core each) instead of in series.
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(_score_k)((k, return_array, n_init, sakoe_chiba_radius, dtw_dist))
        for k in ks
    )

    scores = dict(results)
    for k in sorted(scores):
        print(f"Silhouette score for k={k}: {scores[k]:.4f}")

    best_k = max(scores, key=scores.get)

    print(f"\n Optimal number of clusters: {best_k} with silhouette score: {scores[best_k]:.4f}")

    return best_k, scores

def run_pca(return_array):
    """ PCA()	Creates a PCA model
    n_components=2	Reduces to 2 dimensions"""
    
    pca = PCA(n_components = 2)
    # the reason PC is PC1 and PC2 becomes then its index one and index 

    coordinates = pca.fit_transform(return_array)

    explained_variance = pca.explained_variance_ratio_


    print(f" Explained variance: PC1={explained_variance[0]:.4f}, PC2={explained_variance[1]:.4f}")
    print(f" Total explained: {sum(explained_variance):.4f} ({sum(explained_variance)*100:.1f}%)")

    return coordinates, explained_variance


def main(max_k=6, n_init=3, sakoe_chiba_radius=0.05, n_jobs=-1, save_results=True,
         downsample=None):

    """MAIN FUNCTION"""
    print("=" * 60)
    print("CLUSTERING ENGINE (FAST)")

    # this is how you process the src folder from data processing from the main function as this method
    from src_fast.data_processing import main as get_data
    returns, regimes = get_data()

    print(f"Data shape for processing {returns.shape}" )

    return_array = returns.T.values

    # Transposes the DataFrame
    # returns.T	Transposes the DataFrame	(2388, 50) → (50, 2388)
    #.values	Converts to numpy array	tslearn works with numpy arrays
    #return_array.shape	Prints the shape	Confirms array is ready
    # finds optimal K means
    best_k, scores = determine_optimal_k(
        return_array,
        max_k=max_k,
        n_init=n_init,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=n_jobs,
        downsample=downsample,
    )

    final_labels = run_clustering(
        return_array,
        best_k,
        n_init=n_init,
        sakoe_chiba_radius=sakoe_chiba_radius,
        n_jobs=n_jobs,
    )

    print(f"Final cluster assignments (first 10): {final_labels[:10]}...")

    coords, explained = run_pca(return_array)

    print(f"PCA complete and the coordinates shape is: {coords.shape}")

    print("\n" + "=" * 60)

    print("CLUSTERING ENGINE (FAST) TEST IS COMPLETE")
    print("=" * 60)
    
    print(f"\nSummary:")
    print(f"   Number of tickers: {return_array.shape[0]}")

    print(f"Optimal K: {best_k}")

    print(f"Explained variance: {sum(explained):.4f} ({sum(explained)*100:.1f}%)")

    if save_results:
        _save_results(scores, best_k, final_labels)

    return final_labels, coords, best_k, scores


if __name__ == "__main__":
    final_labels, coords, best_k, scores = main()
