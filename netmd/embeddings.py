from tslearn.barycenters import dtw_barycenter_averaging
from tslearn.metrics import dtw_path

from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import SpectralEmbedding
from typing import Dict, List, Callable
from sklearn.decomposition import PCA
import karateclub as kc 
import networkx as nx
import pandas as pd
import numpy as np
import math
import os


def g2v_fit_transform(g2v_config: Dict, subgraphs: List[nx.Graph]) -> np.ndarray:
    '''
    Fit the Graph2Vec model to the subgraphs and return the embeddings.\n
    Graph2Vec (https://arxiv.org/abs/1707.05005) implementation from https://github.com/benedekrozemberczki/karateclub.

    Parameters:
        g2v_config (Dict): Configuration dictionary for Graph2Vec.
        subgraphs (List[nx.Graph]): List of NetworkX graphs.

    Returns:
        np.ndarray: Graph embeddings.
    '''
    # Add self-loops to all nodes
    self_loops = [(node, node) for node in subgraphs[0].nodes()]

    for graph in subgraphs:
        graph.add_edges_from(self_loops)
   
    g2v_subgraphs = kc.Graph2Vec(**g2v_config)  
    g2v_subgraphs.fit(subgraphs)
    
    return g2v_subgraphs.get_embedding()



def entropy_filter(graphs: list[nx.Graph], entropies: pd.DataFrame, threshold: float) -> List[nx.Graph]:
    """
    Creates subgraphs from a list of graphs, including only edges that meet a specified entropy threshold.

    This function filters edges based on their entropy values across multiple frames. Edges are
    included in the subgraphs only if their entropy exceeds the given threshold in all replicas.

    Args:
        graphs (list[networkx.Graph]): A list of NetworkX graphs.
        entropies (pandas.DataFrame): A DataFrame containing edge entropy values, where rows
                                     represent edges and columns represent frames.
        treshold (float): The entropy threshold value.

    Returns:
        list[networkx.Graph]: A list of NetworkX subgraphs, each containing only the edges that
                             meet the entropy threshold.
    """

    selected_rows = (entropies > threshold).all(axis=1)
    edge_list = entropies[selected_rows].index.tolist()

    subgraphs = [nx.convert_node_labels_to_integers(g.edge_subgraph(edge_list), first_label=0, ordering='default') for g in graphs]  
    

    return subgraphs


def dim_reduction(subgraph_emb: np.ndarray, barycenter: np.ndarray, workers: int) -> np.ndarray:
    """
    Performs dimensionality reduction on subgraph embeddings and a barycenter using PCA and Spectral Embedding.

    This function first applies Principal Component Analysis (PCA) to reduce the dimensionality of
    the subgraph embeddings while retaining 90% of the variance. If the resulting number of
    components is less than 2, PCA is forced to reduce to 2 components. It then concatenates the
    subgraph embeddings and the barycenter, transforms them using the fitted PCA, and applies
    Spectral Embedding for further dimensionality reduction to 2 components.

    Args:
        subgraph_emb (np.ndarray): A NumPy array representing the subgraph embeddings.
        barycenter (np.ndarray): A NumPy array representing the barycenter embedding.
        workers (int): Number of workers that SpectralEmbedding is allowed to use.
    Returns:
        np.ndarray: A NumPy array representing the transformed embeddings after both PCA and
                    Spectral Embedding.
    """    
    # Fit PCA with 90% variance retention
    pca = PCA(n_components=0.9).fit(subgraph_emb)

    # If PCA drops below 2, force it to 2
    if 2 > pca.components_.shape[0]:  
        pca = PCA(n_components=2).fit(subgraph_emb)

    X = np.concatenate([subgraph_emb, barycenter], axis=0)

    transformed_emb = pca.transform(X)

    # Spectral: compute for subgraphs, estimate per barycenter
    transformed_emb = SpectralEmbedding(n_components=2, n_jobs=workers, random_state=42).fit_transform(transformed_emb)

    return transformed_emb


def iterative_dim_reduction(subgraph_emb: np.ndarray, barycenters: np.ndarray, workers: int) -> List[np.ndarray]:
    """
    Performs iterative dimensionality reduction on subgraph embeddings and a series of barycenters.

    This function applies PCA and Spectral Embedding to reduce the dimensionality of subgraph
    embeddings and each barycenter in a given list. It first fits PCA to the subgraph embeddings,
    ensuring at least 2 components are retained. Then, for each barycenter, it concatenates the
    subgraph embeddings with the barycenter, transforms them using the fitted PCA, and applies
    Spectral Embedding for further dimensionality reduction to 2 components.

    Args:
        subgraph_emb (np.ndarray): A NumPy array representing the subgraph embeddings.
        barycenters (np.ndarray): A NumPy array containing a list of barycenters.
        workers (int): Number of workers that SpectralEmbedding is allowed to use.

    Returns:
        List[np.ndarray]: A list of NumPy arrays, where each array represents the transformed
                          embeddings (subgraphs + barycenter) after both PCA and Spectral Embedding.
    """    
    reduced_embs = []

    # Fit PCA with 90% variance retention
    pca = PCA(n_components=0.9).fit(subgraph_emb)

    # If PCA drops below 2, force it to 2
    if 2 > pca.components_.shape[0]:  
        pca = PCA(n_components=2).fit(subgraph_emb)

    for bary in barycenters:

        X = np.concatenate([subgraph_emb, bary], axis=0)

        transformed_emb = pca.transform(X)

        # Spectral: compute for subgraphs, estimate per barycenter
        transformed_emb = SpectralEmbedding(n_components=2, n_jobs=workers, random_state=42).fit_transform(transformed_emb)
        reduced_embs.append(transformed_emb)

    return reduced_embs


def dtw_mapping(replicas_ts: List[List[float]], barycenter: np.ndarray, meta_data: pd.DataFrame, out_path: str) -> pd.DataFrame:
    """
    Performs Dynamic Time Warping (DTW) alignment between replicas and a barycenter, and saves the mapping.

    This function aligns each replica's time series data with a barycenter using DTW. It then
    generates a mapping that shows the alignment between frames in each replica and the barycenter,
    and writes this mapping along with per-frame distances to a text file.

    Args:
        replicas_ts (List[List[float]]): A 3D NumPy array containing the time series data for each replica.
                                  The shape should be (num_replicas, num_frames, feature_dim).
        barycenter (np.ndarray): A 2D NumPy array representing the barycenter time series.
                                 The shape should be (num_frames, feature_dim).
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the replicas,
                                  including a 'Rep' column for replica names.
        out_path (str): The path to the output directory where the mapping file will be saved.
    """
    alignment_string = ""

    with open(os.path.join(out_path, "dtw_mapping.txt"), "w") as file:
        pass

    # print(f"replicas_ts: {replicas_ts.shape}")
    for rep_idx, rep_name in enumerate(meta_data['Rep'].unique()):

        # def distance_alignemnt(time_series, reference, type_dist=np.min):
        path, _ = dtw_path(replicas_ts[rep_idx], barycenter)

        ts_ids, ref_ids = np.array(path).T # to shape (2, -1)
        aligned_ts = replicas_ts[rep_idx][ts_ids]
        aligned_ref = barycenter[ref_ids]
        
        per_frame_dist = ((aligned_ts - aligned_ref)**2).sum(axis=-1)**.5

        per_frame_dist =  pd.Series(per_frame_dist, index=ref_ids)

        # Find the max width for each column
        width = max(len(str(idx)) for idx in ref_ids)

        # Print the lists in a single horizontal line
        alignment_string += f'<{rep_name}>' + " - ".join(f"{a:>{width}}" for a in ts_ids) + '\n'
        alignment_string += '<barycenter>' + " - ".join(f"{a:>{width}}" for a in ref_ids) + '\n'
        alignment_string += '<score>' + " - ".join(f"{a:>{width}}" for a in per_frame_dist) + '\n\n'

        with open(os.path.join(out_path, "dtw_mapping.txt"), "a") as file:  
            file.write(alignment_string)

    return
	
def iterative_pruning(replicas_ts: List[List[float]], meta_data: pd.DataFrame) -> List[str]:
    """
    Iteratively removes a replica based on the greatest DTW distances, recalculating the barycenter
    and scores at each step, until only one replica remains.

    Parameters:
        replicas_ts (List[List[float]]): List of time series embeddings.
        meta_data (pd.DataFrame): Meta Data DataFrame with replica and frame information.

    Returns:
        Tuple: containing: a *DataFrame* ranking all the replicas from worst to best and the *list* of barycenters calculated at each iteration.
    """

    # DTW is computed as the Euclidean distance between aligned time series
    def normalized_dtw(ts1, ts2):
        _, sim_score = dtw_path(ts1, ts2)
        return sim_score / math.pow((len(ts1)**2 + len(ts2)**2), 0.25)

    replica_labels = meta_data['Rep'].unique()
    replica_ranks = pd.DataFrame(np.zeros((len(replica_labels), 2)), columns=['Rank', 'Barycenter_distance'], index=replica_labels)

    barycenters = []
    replica_index = { rep: i for i, rep in enumerate(replica_labels) }

    while len(replica_labels) > 1:
        # Step 1: Calculate scores with the current state of time_series

        barycenter = dtw_barycenter_averaging([replicas_ts[replica_index[rep]] for rep in replica_labels]) 
        barycenters.append(barycenter) # Add barycenter to barycenters list

        scores = np.array([normalized_dtw(replicas_ts[replica_index[rep]], barycenter) for rep in replica_labels])
        scores_df = pd.Series(scores, index=replica_labels)
        
        # Find the index of the worst replica based on the maximum score
        worst_replica = scores_df.idxmax()

        replica_ranks.loc[worst_replica, 'Barycenter_distance'] = scores_df[worst_replica]
        replica_ranks.loc[worst_replica, 'Rank'] = len(replica_labels) - 1
        
        # Remove the worst replica
        scores_df.drop(worst_replica, inplace=True)
        replica_labels = replica_labels[replica_labels != worst_replica]

    # Insert last values
    worst_replica = scores_df.idxmax()
    replica_ranks.loc[worst_replica, 'Barycenter_distance'] = scores_df[worst_replica]
    replica_ranks.loc[worst_replica, 'Rank'] = len(replica_labels) - 1
    replica_ranks.sort_values(by='Rank', ascending=True, inplace=True)


    # Return the final ranking
    return replica_ranks, barycenters