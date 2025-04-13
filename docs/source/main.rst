Main 
====

=============================
Program Overview and Workflow
=============================

This program analyzes **molecular dynamics (MD)** replicas data to identify key structural changes and cluster similar states.

The workflow of the program proceeds as follows:

1. **Loading MD Replica Files**:
   All the MD replica files are loaded and a NetworkX representation is built for each frame.

2. **Preprocessing**:
   Each replica is preprocessed by removing edges with low entropy. Entropy is computed **intra-replica** (within each individual replica), while the filter is applied **inter-replica** (across all replicas).

3. **Graph2Vec Embedding**:
   The program uses **Graph2Vec** to compute meaningful embeddings for each frame, transforming the replicas into time series.

4. **Barycenter Calculation**:
   The barycenter of the time series is computed.

5. **Ranking Time Series**:
   Time series are ranked based on their similarity to the barycenter.

6. **Hierarchical Clustering**:
   Hierarchical clustering is performed on the time series.

Throughout the execution, several plots and files are saved in the *out_path* specified in the function :ref:`create_parser_function`.

----------------------
Plots Generated:
----------------------

- **2D Plot of the Replica Embeddings**:
  Visual representation of the embeddings of all replicas.

- **2D Plot of the Replica Embeddings with Barycenter**:
  The embeddings with the barycenter overlaid.

- **Iterative Pruning Process Plot**:
  Visualization of the iterative pruning process used to refine the replicas.

- **Dendrogram Plot**:
  A dendrogram plot with a cut line computed using the elbow method, showing the hierarchical clustering results.

----------------------
Files Created:
----------------------

- **`metadata.tsv`**:
  A DataFrame containing information about the replica and frame. Can be used to index the replica embeddings.

- **`subgraphs_emb.pkl`**:
  A list of subgraph embeddings belonging to different replicas.

- **`dtw_matrix.tsv`**:
  A squared matrix where each value represents the Euclidean distance between aligned time series.

- **`dtw_mapping.txt`**:
  This file contains the frame of each replica and the corresponding frame of the barycenter. It illustrates the dynamic time warping (DTW) indexing and the Euclidean distance between frames.

- **`iterative_ranks.tsv`**:
  A file containing the iterative pruning results of all replicas based on their distance from the barycenter.

.. note::

   For more information on how to specify the input and output paths, refer to the :ref:`create_parser_function` function.

.. _create_parser_function:

-----------------
``create_parser``
-----------------

Function that builds the argument parser from the ``argparse`` library.

- **-h, -\-help**

  Show this help message and exit.

- **-I INPUTPATH INPUTPATH, -\-InputPath INPUTPATH INPUTPATH**  *(required)*

  Specify the directory tree path followed by the standardized prefix of the contact file name.
  
  **Example:**
  
  .. code-block:: bash
     
     -i examples_dir contacts.tsv

- **-F FILES [FILES ...], -\-Files FILES [FILES ...]**  *(required)*

  Specify one or more contact file paths.

- **-f FEATURES, -\-features FEATURES** *(optional)*

  Specify the path to the input file containing node features. The file must be in tab-separated values (.tsv) format.
  If no path is provided, the unique chain identifier of each residue in the contact file will be used as the node feature.

- **-e EDGEFILTER, -\-edgeFilter EDGEFILTER** *(optional)*

  Specify the entropy threshold used to filter the graph edges. *(default: 0.1)*

- **-c CONFIGFILE, -\-configFile CONFIGFILE** *(optional)*

  Specify the path to the configuration file containing arguments for Graph2Vec.
  If no path is provided, default values will be used.

- **-o OUTPUTPATH, -\-outputPath OUTPUTPATH** *(optional)*

  Specify the output path. If no path is provided, the ``results`` folder will be used.

- **-p / -\-plotFormat {svg,png}** *(optional)*

  Specify the format of the image output. *(svg, png; default: svg)*

- **-\-verbose** *(optional)*

  Allow extra prints.

**Returns**

	*argparse.ArgumentParser*
    	Object for parsing command-line strings into Python objects.



-----------------
``check_entropy``
-----------------

Function that checks if the passed argument is a valid float between 0.0 and 1.0 (inclusive). If the argument does not meet the criteria, it raises an error.

- **argument (str)**: 
  A string representing the float that needs to be checked.

**Returns**

	*argument (str)*: 
  		The input argument is returned if it's a valid float between 0.0 and 1.0.

**Raises**

	*ValueError (argparse.ArgumentTypeError)**: 
		If the argument is not a valid float or is not within the specified range (0.0 to 1.0), an error is raised.

**Example Usage**

.. code-block:: python

    check_entropy("0.5")  # Valid value, returns 0.5
    check_entropy("1.2")  # Raises argparse.ArgumentTypeError
    check_entropy("abc")  # Raises argparse.ArgumentTypeError