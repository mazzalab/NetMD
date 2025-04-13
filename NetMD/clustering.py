
from plotUtils import plot_dendogram

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from tslearn.metrics import dtw_path
from typing import List, Dict
import pandas as pd
import numpy as np
import math


def elbow_method_cut(linkage_matrix: np.ndarray)->  float:
    '''
    Extract from teh linkage matrix the largest gap between distances to find the elbow point.
    Parameters:
        linkage_matrix (np.ndarray): The linkage matrix from hierarchical clustering.
    Returns:
        float: The index of the elbow point.
    '''
    distances = linkage_matrix[:, 2]
    
    all_gaps = np.diff(distances)

    # Find elbow point using gap differences
    max_diff_idx = np.argmax(all_gaps) 

    return max_diff_idx


# DTW is computed as the Euclidean distance between aligned time series
def normalized_dtw(ts1: List[float], ts2: List[float]):
    '''
    Computes the normalized distance between two time series using Dynamic Time Warping (DTW).
    Parameters:
        ts1 (List[float]): The first time series.
        ts2 (List[float]): The second time series.
    Returns:
        float: The normalized DTW distance between the two time series.
    '''
    _, sim_score = dtw_path(ts1, ts2)
    return sim_score / math.pow((len(ts1)**2 + len(ts2)**2), 0.25)


# Compute normalized DTW for all pairs
def compute_dtwmatrix(replica_ts: List[List[float]], meta_data: pd.DataFrame) -> pd.DataFrame:
    """
    Computes a Dynamic Time Warping (DTW) distance matrix for a set of time series.

    This function calculates the pairwise DTW distances between all time series in the input array.
    The resulting distance matrix is returned as a pandas DataFrame.

    Parameters:
        replica_ts (List[List[float]]): A NumPy array containing the time series data. The shape should be
                                 (num_series, num_frames, feature_dim).
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the time series,
                                  including a 'Rep' column for replica names.

    Returns:
        pd.DataFrame: A pandas DataFrame representing the DTW distance matrix. The index and
                      columns are the unique replica names from the meta_data DataFrame.
    """
    n_series = len(replica_ts)
    dtw_matrix = np.zeros((n_series, n_series))
    
    for i in range(n_series):
        for j in range(i, n_series):  # Only compute upper triangular part
            if i == j:
                dtw_matrix[i, j] = 0  # Distance with itself is 0
            else:
                dist = normalized_dtw(replica_ts[i], replica_ts[j])
                dtw_matrix[i, j] = dist
                dtw_matrix[j, i] = dist  # Symmetric

    dtw_matrix_df = pd.DataFrame(dtw_matrix, index=meta_data['Rep'].unique(), columns=meta_data['Rep'].unique())

    return dtw_matrix_df


def map_link_to_data(linkage_matrix: np.ndarray, rep_names: np.ndarray) -> Dict[int, List[str]]:
    """
    Maps the linkage matrix from hierarchical clustering to the original data points (replica names).

    This function takes a linkage matrix produced by hierarchical clustering and a list of replica
    names, and it constructs a dictionary that maps each step of the clustering process to the
    corresponding clusters of data points.

    Args:
        linkage_matrix (np.ndarray): A NumPy array representing the linkage matrix from hierarchical clustering.
        rep_names (np.ndarray): A NumPy array containing the original replica names.

    Returns:
        Dict: A dictionary where keys are the step indices of the linkage matrix,
                              and values are lists of two clusters (each cluster is a list of
                              replica names) merged at that step.
    """    
    clusters = {}

    def get_cluster_data(link, rep_names, cluster_index):
        
        cl_indexes = (int(link[cluster_index, 0]), int(link[cluster_index, 1]))
        cl_data = [None, None]

        for j in range(2):
            if cl_indexes[j] < len(rep_names):
                cl_data[j] = rep_names[cl_indexes[j]]
            else:
                cl_data[j] = get_cluster_data(link, rep_names, cl_indexes[j] - len(rep_names))
        
        # Combine data points
        return cl_data[0] + ' ' + cl_data[1]  

    # Redirecting to Original Data
    for i, link in enumerate(linkage_matrix):
        
        cl_indexes = (int(link[0]), int(link[1]))
        cl_data = [None, None]

        # Get data points for clusters
        for j in range(2):
        
            if cl_indexes[j] < len(rep_names):
                cl_data[j] = [rep_names[cl_indexes[j]]] # Direct index to original data
            else:
                # Recursive call
                cl_data[j] = get_cluster_data(linkage_matrix, rep_names, cl_indexes[j] - len(rep_names)).split(" ")  
        
        clusters[i] = cl_data
            
    return clusters


def hierarchical_clustering_rank(dtw_matrix_df: pd.DataFrame, out_path: str, plot_format: str, verbose: bool) -> Dict[str, int]:
    """
    Performs hierarchical clustering on a DTW distance matrix and ranks replicas based on cluster assignments.

    This function takes a DTW distance matrix, performs hierarchical clustering using Ward's linkage,
    determines an optimal cluster cut using the elbow method, plots the dendrogram, and assigns
    replicas to clusters. It then returns a dictionary mapping replica names to their cluster IDs.

    Args:
        dtw_matrix_df (pd.DataFrame): A pandas DataFrame representing the DTW distance matrix.
        out_path (str): The path to the output directory for saving the dendrogram plot.
        plot_format (str): The file format for saving the plot (e.g., 'png', 'pdf').
        verbose (bool): If True, enables verbose output (e.g., printing cluster information).

    Returns:
        Dict: A dictionary mapping replica names to their cluster IDs.
    """
    replica_names = dtw_matrix_df.index.to_numpy()
    condensed_dist = squareform(dtw_matrix_df.to_numpy())

    # Perform hierarchical clustering
    linkage_matrix = linkage(condensed_dist, method='ward', optimal_ordering=True)

    # Find the best cut thorugh the elbow method
    elbow_cut = elbow_method_cut(linkage_matrix)

    # Plot dendogram
    plot_dendogram(linkage_matrix, elbow_cut, replica_names, out_path, plot_format)

    # Recursive function to get data points for a cluster
    clusters = map_link_to_data(linkage_matrix, replica_names)

    if verbose:
        print("\nOptimal Clusters Cuts (linkage matrix format):\n")
        for i, cluster in enumerate(linkage_matrix):
            if i == elbow_cut + 1:
                print(f"------------------------------ <- Elbow Cut at gap {elbow_cut + 1}")
            # if i == larget_gap:
            #     print("------------------------------ <- Largest Gap Cut")
            print(f"Step {i+1}: Distance:{cluster[2]:.2f}\tMerging\t{clusters[i][0]}-{clusters[i][1]}")

    # Remove duplicates
    clusters_elbow = fcluster(linkage_matrix, linkage_matrix[elbow_cut, 2], criterion="distance")

    return {replica_names[i]:cluster for i, cluster in enumerate(clusters_elbow)}