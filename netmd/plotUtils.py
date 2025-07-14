from .embeddings import iterative_dim_reduction

from scipy.cluster.hierarchy import dendrogram
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import matplotlib.cm as cmx
from matplotlib.collections import LineCollection
from typing import List
import numpy as np
import pandas as pd
import math
import os


def plot_emb_rep(emb_subg: np.ndarray, meta_data: pd.DataFrame, out_path: str, plot_format: str):
    """
    Plots the 2D embeddings of replicas, colored by replica and frame.

    This function generates two scatter plots: one showing the embeddings colored by replica,
    and the other showing the embeddings colored by frame number.

    Parameters:
        emb_subg (np.ndarray): A NumPy array containing the 2D embeddings of subgraphs.
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the embeddings.
        out_path (str): The path to the output directory where the plot will be saved.
        plot_format (str): The file format for saving the plot (e.g., 'png', 'pdf').
    """

    fig, axes = plt.subplots(1, 2, figsize=(19, 9))
    
    for replica in meta_data['Rep'].unique():
        if replica == 'barycenter': continue
        
        # Plot the embeddings per replica
        axes[0].scatter(emb_subg[meta_data['Rep']==replica, 0], 
                        emb_subg[meta_data['Rep']==replica, 1], 
                        marker='o', alpha = 0.25, s=10, 
                        label=f'{replica.strip().split("/")[-1]}')
        
        # Plot the embeddings per frame
        scatter = axes[1].scatter(emb_subg[meta_data['Rep']==replica,0], 
                        emb_subg[meta_data['Rep']==replica, 1], 
                        c=meta_data[meta_data['Rep']==replica]['Frame'], 
                        cmap="viridis", marker='o', alpha = 0.4, s=10, 
                        label=f'{replica.strip().split("/")[-1]}')
    
    #plt.scatter(transformed_embeddings_subg[:,0], transformed_embeddings_subg[:, 1], marker='o')
    axes[0].set_title("2D Embedding of Replicas", fontsize=14, fontweight='bold', color='black')
    axes[0].legend()
    axes[0].set_xlabel("Embedding Component 1", fontsize=14, color='black')
    axes[0].set_ylabel("Embedding Component 2", fontsize=14, color='black')
    axes[0].ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useOffset=False) 

    colorbar = plt.colorbar(scatter, ax=axes[1]) 
    colorbar.set_label("Frame", fontsize=12) 

    axes[1].set_title("2D Embedding of Replicas", fontsize=14, fontweight='bold', color='black')
    axes[1].set_xlabel("Embedding Component 1", fontsize=14, color='black')
    axes[1].set_ylabel("Embedding Component 2", fontsize=14, color='black')
    axes[1].ticklabel_format(axis='x', style='sci', scilimits=(-3,3), useOffset=False) 
    
    # Gridlines with light opacity for a cleaner look
    axes[0].grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.3)
    axes[1].grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.3)

    
    fig.tight_layout()

    plt.savefig(os.path.join(out_path, "replica_embedding." + plot_format), dpi=400, format=plot_format) 

    return


def plot_emb_bary(emb_data: np.ndarray, meta_data: pd.DataFrame, out_path: str, plot_format: str):
    """
    Plots the 2D embeddings of the barycenter, showing both points and a line representation.

    This function generates a figure with two subplots. The first subplot displays the barycenter points
    as scatter points, while the second subplot connects the points with a line, visualizing the
    trajectory of the barycenter embedding.

    Parameters:
        emb_data (np.ndarray): A NumPy array containing the 2D embeddings of the data, including the barycenter.
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the embeddings,
                                  including 'Rep' (replica name) and 'Frame' columns.
        out_path (str): The path to the output directory where the plot will be saved.
        plot_format (str): The file format for saving the plot (e.g., 'png', 'pdf').
    """
    fig, axes = plt.subplots(1, 2, figsize=(19, 9), gridspec_kw={'width_ratios': [1, 1.25]})

    plot_emb_bary_points(axes[0], emb_data, meta_data)
    plot_emb_bary_line(axes[1], emb_data, meta_data)
    
    # Adjust layout for tight presentation
    plt.tight_layout()

    # Save high-resolution plot
    plt.savefig(os.path.join(out_path, "barycenter_embedding." + plot_format), dpi=400, format=plot_format) 

    return 


def plot_emb_bary_points(ax: plt.Axes, emb_data: np.ndarray, meta_data: pd.DataFrame):
    """
    Plots the 2D embeddings of the barycenter, showing points representation.

    This function generates a figure displaying the barycenter points
    as scatter points, visualizing the trajectory of the barycenter embedding.

    Parameters:
        ax (plt.Axes): The axis used to plot the image.
        emb_data (np.ndarray): A NumPy array containing the 2D embeddings of the data, including the barycenter.
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the embeddings, including 'Rep' (replica name) and 'Frame' columns.
    """
    # Plot each replica with color mapped by frames
    for replica in meta_data['Rep'].unique():
        mask = meta_data['Rep'] == replica
        scatter = ax.scatter(
            emb_data[meta_data['Rep'] == replica, 0], 
            emb_data[meta_data['Rep'] == replica, 1],
            c=meta_data[meta_data['Rep'] == replica]['Frame'],
            cmap="RdBu",  # A nice perceptually uniform color map
            marker='o',
            alpha=0.7,  # More transparency
            s=60,  # Slightly larger marker size for better visibility
            edgecolor="grey",  # Black edge for better contrast
            #label=f'{replica.split("/")[-1]}'
        )

    # Highlight the barycenter with a distinct style
    ax.scatter(
        emb_data[meta_data['Rep'] == 'barycenter', 0], 
        emb_data[meta_data['Rep'] == 'barycenter', 1], 
        marker='o', 
        s=75,  # Larger size for the barycenter
        c='y',  # Distinct color for the barycenter
        edgecolor="grey",  # Edge color for clarity
        label="Barycenter",
        alpha=0.7
    )

    # Add a colorbar with a title
    cbar = plt.colorbar(scatter, orientation='vertical', ax=ax)
    cbar.set_label("Frames", fontsize=12, color="black", labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Adding title and axis labels with increased font size and bold text
    ax.set_title("Temporal Embedding of Graphs and Barycenter (points)", fontsize=14, fontweight='bold', color='black')
    ax.set_xlabel("Embedding Dimension 1", fontsize=14, color='black')
    ax.set_ylabel("Embedding Dimension 2", fontsize=14, color='black')

    # Enhance ticks for better visibility
    ax.tick_params(axis='x', labelsize=14, color='black')
    ax.tick_params(axis='y', labelsize=14, color='black')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) 

    # Gridlines with light opacity for a cleaner look
    ax.grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.3)
    ax.legend()

    return


def plot_emb_bary_line(ax: plt.Axes, emb_data: np.ndarray, meta_data: pd.DataFrame):
    """
    Plots the 2D embeddings of the barycenter, showing lines representation.

    This function generates a figure displaying the barycenter as a segmented line, visualizing the trajectory of the barycenter embedding.

    Parameters:
        ax (plt.Axes): The axis used to plot the image.
        emb_data (np.ndarray): A NumPy array containing the 2D embeddings of the data, including the barycenter.
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the embeddings, including 'Rep' (replica name) and 'Frame' columns.
    """
    # Plot each replica with color mapped by frames
    for replica in meta_data['Rep'].unique():
        mask = meta_data['Rep'] == replica
        scatter = ax.scatter(
            emb_data[meta_data['Rep'] == replica, 0], 
            emb_data[meta_data['Rep'] == replica, 1],
            c=meta_data[meta_data['Rep'] == replica]['Frame'],
            cmap="RdBu",  # A nice perceptually uniform color map
            marker='o',
            alpha=0.7,  # More transparency
            s=60,  # Slightly larger marker size for better visibility
            edgecolor="grey"  # Grey edge for better contrast
        )

    # Highlight the barycenter with a gradient line style
    bary_coords = emb_data[meta_data['Rep'] == 'barycenter']  # Barycenter coordinates

    # Create segments for the line
    segments = np.stack([bary_coords[:-1], bary_coords[1:]], axis=1)

    # Map a colormap to the gradient
    norm = mcolors.Normalize(vmin=0, vmax=len(bary_coords) - 1)
    cmap = plt.cm.viridis  # Choose your colormap
    lc = LineCollection(
        segments,
        cmap=cmap,
        norm=norm,
        linewidths=2,
        alpha=0.4
    )

    # Assign colors to each segment based on the index (frame number)
    lc.set_array(np.arange(len(bary_coords) - 1))

    # Add the gradient line to the plot
    ax.add_collection(lc)

    # Add a colorbar for the gradient line
    cbar_line = plt.colorbar(lc, orientation='vertical', ax=ax)
    cbar_line.set_label("Barycenter Gradient", fontsize=12, color="black", labelpad=10)
    cbar_line.ax.tick_params(labelsize=10)

    # Add a colorbar for the scatter plot
    cbar = plt.colorbar(scatter, orientation='vertical', ax=ax)
    cbar.set_label("Frames", fontsize=12, color="black", labelpad=10)
    cbar.ax.tick_params(labelsize=10)

    # Adding title and axis labels with increased font size and bold text
    ax.set_title("Temporal Embedding of Graphs and Barycenter (line)", fontsize=14, fontweight='bold', color='black', loc='left')
    ax.set_xlabel("Embedding Dimension 1", fontsize=14, color='black')
    ax.set_ylabel("Embedding Dimension 2", fontsize=14, color='black')

    # Enhance ticks for better visibility
    ax.tick_params(axis='x', labelsize=14, color='black')
    ax.tick_params(axis='y', labelsize=14, color='black')
    ax.ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) 

    # Gridlines with light opacity for a cleaner look
    ax.grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.3)

    return


def plot_pruning(subgraphs_emb: List[List[float]], barycenters: np.ndarray, meta_data: pd.DataFrame, replica_ranks: pd.DataFrame, out_path: str, plot_format: str, workers: int):
    """
    Plots the iterative pruning process, showing the embeddings of replicas and barycenters at each step.

    This function visualizes the iterative pruning of replicas. It generates a grid of subplots,
    where each subplot represents a step in the pruning process. It plots the embeddings of the
    remaining replicas and the barycenter at each step, using distinct colors for each replica.

    Parameters:
        subgraphs_emb (List[List[float]]): A list containing the embeddings of subgraphs for each replica.
        barycenters (np.ndarray): A NumPy array containing the barycenters calculated at each step.
        meta_data (pd.DataFrame): A pandas DataFrame containing metadata about the embeddings.
        replica_ranks (pd.DataFrame): A pandas DataFrame containing the ranking of replicas.
        out_path (str): The path to the output directory where the plot will be saved.
        plot_format (str): The file format for saving the plot (e.g., 'png', 'pdf').
        workers (int): Number of workers that SpectralEmbedding is allowed to use.

    """

	# Update metadata adding the barycenter info
    meta_data = pd.concat([meta_data, pd.DataFrame({'Rep': ['barycenter']*len(barycenters[0]), 'Frame': np.arange(0, len(barycenters[0]))})])

    multiple_reduced_emb = iterative_dim_reduction(subgraphs_emb, barycenters, workers)


    cols = 3
    rows = math.ceil(len(barycenters)/cols)
    row = 0
    
    # Choose a colormap and create an array of colors
    cmap = plt.cm.get_cmap('tab20', replica_ranks.shape[0]) 
    colors = {replica: cmap(i / replica_ranks.shape[0]) for i, replica in enumerate(replica_ranks.index)}

    fig, axes = plt.subplots(rows, cols,  figsize=(19, 15))

    for i, _ in enumerate(barycenters):
        
        if i == 0:
            replicas = replica_ranks.index
        else:
            replicas = replica_ranks[:-i].index

        reduced_embs = multiple_reduced_emb[i]
        
        col = i%cols

        if col == 0 and i>0:
            row += 1

        # Plot each replica with color mapped by frames
        for j,replica in enumerate(replicas):
            scatter = axes[row][col].scatter(
                reduced_embs[ meta_data['Rep'] == replica, 0], 
                reduced_embs[ meta_data['Rep'] == replica, 1],
                marker='o', alpha = 0.30, s=10, 
                label=f'{replica.strip().split("/")[-1]}',
                color=colors[replica]
                )
            
        
        # Highlight the barycenter with a distinct style
        axes[row][col].scatter(
            reduced_embs[meta_data['Rep'] == 'barycenter', 0], 
            reduced_embs[meta_data['Rep'] == 'barycenter', 1], 
            c='black',  # Distinct color for the barycenter
            edgecolor="w",  # Edge color for clarity
            label="Barycenter",
            marker='o', alpha = 0.65, s=15
            )
                
        # Adding title and axis labels with increased font size and bold text
        axes[row, col].set_title(f"Top {len(barycenters) - i + 1} Replicas", fontsize=16, fontweight='bold', color='black')
        axes[row, col].set_xlabel("Embedding Dimension 1", fontsize=14, color='black')
        axes[row, col].set_ylabel("Embedding Dimension 2", fontsize=14, color='black')
        
        # Enhance ticks for better visibility
        axes[row, col].tick_params(axis='x', labelsize=14, color='black')
        axes[row, col].tick_params(axis='y', labelsize=14, color='black')
        axes[row, col].ticklabel_format(axis='x', style='sci', scilimits=(-3,3)) 

        # Gridlines with light opacity for a cleaner look
        axes[row, col].grid(True, color='gray', linestyle='--', linewidth=0.6, alpha=0.3)

        axes[row, col].legend()


    for i in range(rows):
        for j in range(cols):
            if not axes[i, j].has_data():  # Check if the subplot has data
                fig.delaxes(axes[i, j]) 

    # Adjust layout for tight presentation
    fig.tight_layout()

    plt.savefig(os.path.join(out_path, "iterative_pruning." + plot_format), dpi=400, format=plot_format) 

    return


def plot_dendogram(linkage_matrix: np.ndarray, cut_distance_elbow: int, replica_names, out_path: str, plot_format: str):
    # Plot dendrogram with cut line
    _, ax = plt.subplots(figsize=(12, 8))
    threshold = linkage_matrix[cut_distance_elbow, 2]

    dendrogram(
        linkage_matrix,
        color_threshold = threshold + 1e-9,  # Set the threshold for coloring clusters #TODO: fix cutoff
        above_threshold_color="gray",  # Color for branches above the cut
        labels=replica_names,
        leaf_rotation=45, leaf_font_size=12
    )

    ax.axhline(y=threshold, color='red', linestyle='--', label=f'Cut Distance: {threshold:.3f}')
    ax.set_title("Dendrogram with Elbow Cut", fontsize=16, fontweight='bold', color='black')
    ax.set_xlabel("Replica Names", fontsize=14, color='black')
    ax.set_ylabel("Distance", fontsize=14, color='black')
    ax.tick_params(axis='x', labelsize=10, color='black')
    ax.tick_params(axis='y', labelsize=14, color='black')

    ax.legend()
    # Adjust the bottom margin
    plt.subplots_adjust(bottom=0.2)  # Adjust this value as needed
    plt.savefig(os.path.join(out_path, "dendogram." + plot_format), dpi=400, format=plot_format) 

    return
