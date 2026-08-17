import os
import sys

import numpy as np
from tslearn.metrics import cdist_dtw # Computes DTW distance -> Your core differentiator
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA # Measures cluster quality -> Finds optimal K

# Make the project root importable so `from src.data_processing import ...`
# works no matter where this script is run from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Computes DTW distance matrix
def compute_dtw_distance_matrix(return_array):
    """ Computes the DTW distance matrix for the given data """
    return cdist_dtw(return_array)


# Runs DTW K-means clustering
def run_clustering(return_array, n_clusters):
    """ Runs Kmeans clustering on the given data using DTW distance"""
    """ the n_iit runs it 10 times to find the best clustering solution. The random_state ensures that the results are reproducible. """
    model = TimeSeriesKMeans(n_clusters = n_clusters, metric = "dtw", n_init = 10, random_state = 42)

    # Fit the model and predict cluster labels for the return_array and each stock assigned to a cluster based on its return pattern. The reshape is necessary because tslearn expects a 3D array (n_samples, n_timestamps, n_features), and here we have only one feature (the return value) for each timestamp.

    return model.fit_predict(return_array.reshape(return_array.shape[0], -1, 1))


def determine_optimal_k(return_array, max_k = 9):
    # We will compute silhouette scores for k values from 2 to max_k and return the best k based on the highest silhouette score.
    """
    Determines the optimal number of clusters using silhouette scores.

    Args:
    return_array: 2D array (n_tickers, n_days)
    max_k: Maximum number of clusters to test (default: 9)

    Returns:
    best_k: The optimal number of clusters
    scores: Dictionary of {k: silhouette_score}
    """
    print("\n Determining optimal number of clusters using silhouette scores...")

    dtw_dist = compute_dtw_distance_matrix(return_array)

    scores = {}
    for k in range(2, max_k + 1):

        labels = run_clustering(return_array, k)

        score = silhouette_score(dtw_dist, labels, metric = "precomputed")

        scores[k] = score

        print(f"Silhouette score for k={k}: {score:.4f}")

    best_k = max(scores, key = scores.get)

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


def main():

    """MAIN FUNCTION"""
    print("=" * 60)
    print("CLUSTERING ENGINE")

    # this is how you process the src folder from data processing from the main function as this method
    from src.data_processing import main as get_data
    returns, regimes = get_data()

    print(f"Data shape for processing {returns.shape}" )

    return_array = returns.T.values

    # Transposes the DataFrame
    # returns.T	Transposes the DataFrame	(2388, 50) → (50, 2388)
    #.values	Converts to numpy array	tslearn works with numpy arrays
    #return_array.shape	Prints the shape	Confirms array is ready
    # finds optimal K means
    best_k, scores = determine_optimal_k(return_array) # so return_array and it will be the default variable.

    final_labels = run_clustering(return_array, best_k)

    print(f"Final cluster assignments (first 10): {final_labels[:10]}...")

    coords, explained = run_pca(return_array)

    print(f"PCA complete and the coordinates shape is: {coords.shape}")

    print("\n" + "=" * 60)

    print("CLUSTERING ENGINE TEST IS COMPLETE")
    print("=" * 60)
    
    print(f"\nSummary:")
    print(f"   Number of tickers: {return_array.shape[0]}")

    print(f"Optimal K: {best_k}")

    print(f"Explained variance: {sum(explained):.4f} ({sum(explained)*100:.1f}%)")

    return final_labels, coords, best_k, scores


if __name__ == "__main__":
    final_labels, coords, best_k, scores = main()

