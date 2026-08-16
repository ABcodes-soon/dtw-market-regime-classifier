import numpy as np
from tslearn.metrics import cdist_dtw # Computes DTW distance -> Your core differentiator
from tslearn.clustering import TimeSeriesKMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA # Measures cluster quality -> Finds optimal K


def compute_dtw_distance_matrix(return_array):
    """ Computes the DTW distance matrix for the given data """
    return cdist_dtw(return_array)

