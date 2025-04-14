from colorama import Fore, Style
from typing import Tuple, List, Dict 
import networkx as nx
import pandas as pd
import numpy as np
import os
import yaml

class InvalidConfigurationError(Exception):
    """Raised when the YAML routine encountered an error."""
    pass

def parse_config(config_path: str) -> Dict:
    '''
    Parses the *.yml* file containing all the required parameters for the graph embedding model. 
    \nThe required arguments with their default values are:
    * *wl_iterations*: Number of Weisfeiler-Lehman iterations. Default is 3.
    * *use_node_attribute*: Name of graph attribute. Default is "feature".
    * *dimensions*: Dimensionality of embedding. Default is 16.
    * *workers*: Number of cores. Default is 1.
    * *down_sampling*: Down sampling frequency. Default is 0.0.
    * *epochs*: Number of epochs. Default is 10.
    * *learning_rate* (HogWild): learning rate. Default is 0.025.
    * *min_count*: Minimal count of graph feature occurrences. Default is 5.
    * *seed*: Random seed for the model. Default is 42.
    * *erase_base_features*: Erasing the base features. Default is False.

    Parameters:
        config_path (str): Path to the *.yml* file.

    Returns:
        config (dict): Dictionary containing all the arguments for the graph embedding model.

    Raises:
        InvalidConfigurationError (Exception): If the *.yml* file does not contain the expected arguments.
    '''
    # Default dict that is updated and returned by the function
    default_config = {
            "wl_iterations" : 3,
            "use_node_attribute": "feature",
            "dimensions" : 16,
            "workers" : 1,
            "down_sampling" : 0.,
            "epochs" : 10,
            "learning_rate" : 0.025,
            "min_count" : 5,
            "seed" : 42,
            "erase_base_features" : False
        }

    if config_path != None:

        try:         
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)

        except yaml.YAMLError as e:
            raise InvalidConfigurationError(f"Error while parsing YAML file: {e}\n")

        # Update the dict if there are matching keys
        if config.keys() <= default_config.keys():
            default_config.update(config)
        else:
            raise InvalidConfigurationError(f"Error while parsing YAML file fields, please check file's fields: \n{config.keys()}\n")

    return default_config

def process_file(file_path: str) -> Tuple[pd.DataFrame, str]:
    '''
    Processes a contact file, extracting residue interaction information and preparing it for analysis.

    This function reads a contact file, filters out comment lines and empty lines, extracts
    the frame, interaction type, and atom identifiers, and then derives residue identifiers
    from the atom identifiers. It then cleans and formats the data into a pandas DataFrame,
    ready for further processing.

    Specifically, the function performs the following steps:

    1.  **File Reading and Initial Parsing:**
        * Reads the file line by line, skipping lines starting with '#' (comments) or empty lines.
        * Splits each line by tabs and extracts the first four columns: Frame, Interaction, Atom1, and Atom2.
    2.  **DataFrame Creation and Initial Cleaning:**
        * Creates a pandas DataFrame from the extracted data.
        * Fills any missing values with 0.
        * Assigns column names: "Frame", "Interaction", "Atom1", "Atom2".
    3.  **Residue Extraction:**
        * Extracts residue numbers from the "Atom1" and "Atom2" columns using regular expressions.
        * Converts the "Frame", "Res1", and "Res2" columns to integers.
    4.  **Residue Ordering:**
        * Ensures that "Res1" is always less than or equal to "Res2" by swapping values if necessary. This creates consistent residue pairs.
    5.  **Duplicate Removal and Reindexing:**
        * Removes duplicate rows based on "Frame", "Res1", and "Res2" columns.
        * Subtracts 1 from the residue numbers in "Res1" and "Res2" columns, effectively reindexing them.
    6.  **Replica Name Extraction:**
        * Extracts the replica name from the file path.

    Parameters:
        file_path (str): The path to the contact file.

    Returns:
        Tuple: containing: a *DataFrame* with columns "Frame", "Res1", and "Res2", representing residue interactions, and the *str* name of the replica, extracted from the file name.
    '''
    data = []
    replica_name = os.path.normpath(file_path).split(os.path.sep)[-1]

    # Read the file's content
    with open(file_path, "r") as file:
        for line in file.readlines(): 
            if not line.startswith('#') and not line.startswith('\n'):
                data.append(line.strip().split("\t")[:4])   
                  
    df = pd.DataFrame(data)
    df.fillna(0, inplace=True)
    df.columns = ["Frame", "Interaction","Atom1","Atom2"]

    # Extract Res1 and Res2 for df
    df['Res1'] = df['Atom1'].str.extract(r':(\d+):')
    df['Res2'] = df['Atom2'].str.extract(r':(\d+):')
    df[["Frame", "Res1", "Res2"]] = df[["Frame", "Res1", "Res2"]].astype(int)

    # Sort tuple Res1 and Res2
    df.loc[df["Res2"] < df["Res1"], ["Res1", "Res2"]] = df.loc[df["Res2"] < df["Res1"], ["Res2", "Res1"]].values

    # Clear duplicates and reindex residues
    df = df[["Frame", "Res1", "Res2"]].drop_duplicates()
    df[["Res1", "Res2"]] = df[["Res1", "Res2"]] - 1#(min(df['Res1'].min(), df['Res2'].min()))

    return df, replica_name


def compute_entropy(contacts_data: pd.DataFrame) -> pd.DataFrame:
    '''
    Computes edge entropy for a given set of residue contacts. Each edge in the contact data is assigned an entropy value based on the probability of its presence.

    Parameters:
        contacts_data (pd.DataFrame): DataFrame with columns 'Frame', 'Res1', and 'Res2' representing residue contacts.
 
    Returns:
        df (pandas.DataFrame): DataFrame containing the entropy value of each edge.

    '''
    # Set unique row of weight = 1 for each RES-RES in each frame
    edges_df = contacts_data.set_index(['Frame', 'Res1', 'Res2'])
    edges_df['weight'] = 1

    # Pivot RES labels as multicolumns, 
    adj_df = edges_df.unstack().unstack().fillna(0)

    mean_values = adj_df.mean(axis=0)

    with np.errstate(divide='ignore', over='ignore', under='ignore', invalid='ignore'):
        entropy = -mean_values.values * np.log2(mean_values.values) - (1 - mean_values.values) * np.log2(1 - mean_values.values)
    
    np.nan_to_num(entropy, copy=False)
    
    return pd.DataFrame(entropy, index=mean_values.index.droplevel())


def load_data(contacts_data: pd.DataFrame, features_data: pd.DataFrame, name: str ) -> Tuple[nx.Graph, pd.DataFrame, pd.DataFrame]:
    '''
    Constructs network representations of molecular dynamics frames and computes edge entropy.

    This function takes contact data (residue interactions), optional node features, and a replica name
    to generate a list of NetworkX graphs, metadata about the frames that is used to index the graph list, and the intra replica entropy of each edge.

    Parameters:
        contacts_data (pd.DataFrame): DataFrame with columns 'Frame', 'Res1', and 'Res2' representing residue contacts.
        features_data (pd.DataFrame): DataFrame containing features for each residue node (can be empty).
        name (str): The name of the replica.

    Returns:
        tuple: containing: a list of *NetworkX* graphs, one for each frame, a *DataFrame* with metadata about the frames (replica name, frame number), and
        a *DataFrame* containing the entropy of each edge.
    '''
    
    min_res = contacts_data['Res1'].min() # used for indexing
    res1 = set(contacts_data['Res1'].unique())
    res2 = set(contacts_data['Res2'].unique())
    all_res = res1.union(res2)

    contacts_df = contacts_data.groupby('Frame').apply(lambda df : (df[['Res1', 'Res2']] - min_res).apply(tuple, axis =1).values)#, include_groups=False) 

    graphs = [] 

    if features_data.empty:
        nodes = [(i, {"feature": res_num}) for i, res_num in enumerate(all_res)] 
    else:
        nodes = [(i, {"feature": features_data[features_data.index == res_num].values.flatten()}) for i, res_num in enumerate(all_res)] 

    for i in contacts_df.index:
        G = nx.Graph()

        G.add_nodes_from(nodes)
        G.add_edges_from(contacts_df.iloc[i])

        G = nx.convert_node_labels_to_integers(G, first_label=0, ordering='default')

        graphs.append(G)
        
        # numeric_indices = [index for index in range(G.number_of_nodes())]
        # node_indices = sorted([node for node in G.nodes()])
        # assert numeric_indices == node_indices, f"ASSERT IN GRAPH CREATION:\nindex: {i};\n nodes: {G.nodes(data=True)};\n edges:{G.edges()};\n"
    
    
    # Create a DataFrame for the replica and frame
    frame_list = list(contacts_df.index)  # List of frames
    rep_col = [name] * len(frame_list)  # Replicate the replica name for all frames

    # Build the DataFrame for metadata
    df = pd.DataFrame({'Rep': rep_col, 'Frame': frame_list, 'Res_num': len(all_res)})

    # Compute the entropy intra-replica
    contacts_data[['Res1', 'Res2']] = contacts_data[['Res1', 'Res2']] - min_res # apply reindexing again to the entropy edge list
    
    entropy = compute_entropy(contacts_data)
    entropy.columns = [name]

    return graphs, df, entropy

      
def crawl_replica_files(crawl_path: str, file_prefix: str, features_data: pd.DataFrame, verbose: bool) -> Tuple[List[nx.Graph], pd.DataFrame, pd.DataFrame]:
    """
    Recursively explores a directory to process contact files and construct network representations.

    This function traverses a directory tree, identifies files with a specified prefix,
    processes them using the `process_file` function to extract relevant data, and then loads the extracted data into NetworkX graphs,
    metadata DataFrames, and entropy DataFrames using the `load_data` function.

    Parameters:
        crawl_path (str): The path to the directory to crawl.
        file_prefix (str): The prefix of the files to process.
        features_data (pd.DataFrame): DataFrame containing features for each residue node.
        verbose (bool): If True, prints information about loaded files.

    Returns:
        Tuple: containing: a list of *NetworkX* graphs, one for each frame in all processed files, 
        a *DataFrame* containing metadata about the frames, and a *DataFrame* containing the entropy of each edge across all processed files.
    """
    glb_graph_list = []
    glb_entropy_df = pd.DataFrame()
    glb_metadata_df = pd.DataFrame()

    for item in os.listdir(crawl_path):
        if item.startswith('.') or item.startswith('..'):
            continue
        item_path = os.path.join(crawl_path, item)

        if os.path.isdir(item_path):
            # Capture returned values from the recursive call
            sub_graph_list, sub_metadata_df, sub_entropy_df = crawl_replica_files(item_path, file_prefix, features_data, verbose)

            # Combine the recursive results with the current results
            glb_graph_list += sub_graph_list
            glb_entropy_df = pd.concat([glb_entropy_df, sub_entropy_df], axis=1)
            glb_metadata_df = pd.concat([glb_metadata_df, sub_metadata_df])

        elif item.startswith(file_prefix):
            if verbose: 
                print(f"\nLoading data from {item}")
                
            # Process the current file
            replica_data, replica_name = process_file(item_path)
            graph_list, metadata_df, entropy = load_data(replica_data, features_data, replica_name)
            
            # Append current file results to global results
            glb_graph_list += graph_list
            glb_entropy_df = pd.concat([glb_entropy_df, entropy], axis=1)
            glb_metadata_df = pd.concat([glb_metadata_df, metadata_df])
            
            if verbose: 
                print(f'\n{replica_data}\n')

    # Fill NaN values in entropy DataFrame after processing all files
    glb_entropy_df.fillna(0.0, inplace=True)

    return glb_graph_list, glb_metadata_df, glb_entropy_df



# Quello giusto è quello sotto
# def iterate_replica_files(file_list: list[str], features_data: pd.DataFrame, verbose: bool) -> Tuple[List[nx.Graph], pd.DataFrame, pd.DataFrame]: # type: ignore
    
#     glb_graph_list = []
#     glb_entropy_df = pd.DataFrame()
#     glb_metadata_df = pd.DataFrame()

#     for file in file_list:

#         replica_data = pd.read_csv(file, sep='\t')
#         replica_data[["Res1", "Res2"]] = replica_data[["Res1", "Res2"]] - 1
        
#         if verbose: 
#             print(f"\nLoading data from {file}")
#             print(f'\n{replica_data}\n')

#         replica_name = os.path.basename(file)

#         graph_list, metadata_df, entropy = load_data(replica_data, features_data, replica_name)

#         glb_graph_list += graph_list
#         glb_entropy_df = pd.concat([glb_entropy_df, entropy], axis=1)
#         glb_metadata_df = pd.concat([glb_metadata_df, metadata_df])

#     glb_entropy_df.fillna(0., axis=1, inplace=True)

#     return glb_graph_list, glb_metadata_df, glb_entropy_df

def iterate_replica_files(file_list: list[str], features_data: pd.DataFrame, verbose: bool) -> Tuple[List[nx.Graph], pd.DataFrame, pd.DataFrame]: # type: ignore
    '''
    Iterates through the listed files, processes each file, and loads the resulting data.

    This function takes a list of file paths, processes each file using the `process_file`
    function to extract relevant data, and then loads the extracted data into NetworkX graphs,
    metadata DataFrames, and entropy DataFrames using the `load_data` function.

    Parameters:
        file_list (list[str, ...]): List of files path. 
        features_data (pd.DataFrame): DataFrame containing features for each residue node.
        verbose (bool): Allow extra prints.

    Returns:
        Tuple: containing: a list of *NetworkX* graphs, one for each frame in all processed files, 
        a *DataFrame* containing metadata about the frames, and a *DataFrame* containing the entropy of each edge across all processed files.
    '''

    glb_graph_list = []
    glb_entropy_df = pd.DataFrame()
    glb_metadata_df = pd.DataFrame()

    for file in file_list:
        
        if verbose: 
            print(f"Loading data from {file}")

        replica_data, replica_name = process_file(file)
        graph_list, metadata_df, entropy = load_data(replica_data, features_data, replica_name)

        glb_graph_list += graph_list
        glb_entropy_df = pd.concat([glb_entropy_df, entropy], axis=1)
        glb_metadata_df = pd.concat([glb_metadata_df, metadata_df])

        if verbose: 
            print(f'\n{replica_data}\n')

    glb_entropy_df.fillna(0., axis=1, inplace=True)

    return glb_graph_list, glb_metadata_df, glb_entropy_df




# Merge 2 raw files into one
# INTERNAL FUNCTION 
# def join_contacts(crawl_path):
#     """
#     Recursively explores a directory 
#     """

#     for item in os.listdir(crawl_path):

#         item_path = os.path.join(crawl_path, item)

#         if os.path.isdir(item_path):
#             join_contacts(item_path)  

#         else:

#             replica = "Joined_Replica_" + item[6:10] +".tsv"
#             joined_path = os.path.join(crawl_path, replica)
            
#             # Now join all files
#             joined_lines = []
#             last_frame = 0
            
#             for filename in sorted(os.listdir(crawl_path)):
#                 if filename.startswith("Rep"): break

#                 print(f"Reading: {os.path.join(crawl_path, filename)}")
#                 print(f"Last frame: {last_frame}")
#                 with open(os.path.join(crawl_path, filename),'r') as file:
#                     for i, line in enumerate(file.readlines()):
                        
#                         if line.startswith('#') or line.startswith('\n'): continue
                        
#                         row = line.split('\t')
#                         row[0] = str( last_frame + int(row[0]) )
#                         joined_lines.append(str.join('\t', row))

#                 with open(os.path.join(crawl_path, filename),'r') as file:
#                     last_frame = int(file.readlines()[-2].split('\t')[0]) + 1

#             with open(joined_path, 'w') as outfile:
#                 for line in joined_lines:
#                     outfile.write(line)
#                 outfile.write('\n')
            
#             break


#     return
# join_contacts('./data')