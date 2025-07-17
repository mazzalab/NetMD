from .embeddings import g2v_fit_transform, entropy_filter, dim_reduction, iterative_pruning, dtw_mapping
from .dataLoader import crawl_replica_files, iterate_replica_files, parse_config
from .clustering import compute_dtwmatrix, hierarchical_clustering_rank
from .plotUtils import plot_emb_rep, plot_emb_bary, plot_pruning

from tslearn.barycenters import dtw_barycenter_averaging
from colorama import Fore, Style

import pandas as pd
import numpy as np
import argparse
import os

import pickle as pk

# build parser 
def create_parser() -> argparse.ArgumentParser:
	'''
	Function that builds the argument parser from the argparse library.
	The command-line options are:

	* -h, --help:
		Show this help message and exit.
	* -I INPUTPATH INPUTPATH, --InputPath INPUTPATH INPUTPATH:
		[required] Specify the directory tree path followed by the standardized prefix of the contact file name.
		Example: -i examples_dir contacts.tsv
	* -F FILES [FILES ...], --Files FILES [FILES ...]:
		[required] Specify one or more contact file paths.
	* -f FEATURES, --features FEATURES:
		[optional] Specify the path to the input file containing node features. If provided, the file must be in tab-separated values (.tsv) format. If no path is provided, the unique chain identifier of each residue in the contact file will be used as the node feature.
	* -e EDGEFILTER, --edgeFilter EDGEFILTER:
		[optional] Specify the entropy threshold used to filter the graph edges. (default 0.1)
	* -c CONFIGFILE, --configFile CONFIGFILE:
		[optional] Specify the path to the configuration file containing arguments for Graph2Vec. If no path is provided, default values will be used.
	* -o OUTPUTPATH, --outputPath OUTPUTPATH:
		[optional] Specify the output path. If no path is provided, the "results" folder will be used.
	* -p {svg,png}, --plotFormat {svg,png}:
		[optional] Specify the format of the image output. (svg, png; default svg)
	* --verbose:
		[optional] Allow extra prints.

	Returns:
		replicas_ts (argparse.ArgumentParser): Object for parsing command-line strings into Python objects.
	'''
	
	#########parser_functions##########
	def check_entropy(argument: str) -> bool:
		'''
		Checks if the passed argument is a float between 0.0 and 1.0 (inclusive). If this condition is not met, an error is raised.

		Parameters:
			argument (str): String representing a float that needs to be checked.

		Returns:
			argument (str): The input argument is returned if it's a valid float between 0.0 and 1.0.

		Raises:
			ValueError (argparse.ArgumentTypeError): If the argument is not a valid float or is not within the specified range.
		'''
		try:
			argument = float(argument)
		except ValueError as exc:
			raise argparse.ArgumentTypeError(Fore.RED + Style.BRIGHT + argument + f"Invalid filter value {argument}" + Style.RESET_ALL ) from exc
		if argument > 1.0 or argument < 0.0:
			raise argparse.ArgumentTypeError(Fore.RED + Style.BRIGHT + argument + f"Invalid filter value {argument}" + Style.RESET_ALL )
		return argument
	###################################

	#### principal parser
	parser = argparse.ArgumentParser(description= Fore.WHITE + Style.BRIGHT +'''\n\n

	Parser testing 

	''' +  Style.RESET_ALL ,  usage = Fore.GREEN+"python3 " +Fore.WHITE+ "main.py" +Fore.MAGENTA+ " [-h] [-v] ( -i INPUTPATH INPUTPATH | -f [FILES ...]) [-e EDGEFILTER] [-o OUTPUTPATH] [-c CONFIGFILE]" +Style.RESET_ALL)
	
	group = parser.add_mutually_exclusive_group(required=True)
	group.add_argument('-I', '--InputPath', nargs=2, help='-[required] Specify the directory tree path followed by the standardized prefix of the contacts file name. Example: -i examples_dir contacts.tsv')
	group.add_argument('-F','--Files', nargs='+', help='-[required] Specify one or more contact file paths.')
	
	parser.add_argument('-f', '--features', help='-[optional] Specify the path to the input file containing node features. If provided, the file must be in tab-separated values (TSV) format. If no path is provided, the unique chain identifier of each residue in the contacts file will be used as the node feature', required=False)
	parser.add_argument('-e', '--edgeFilter', type=check_entropy, help='-[optional] Specify the entropy threshold used to filter the graph edges. (default 0.1)', default=0.1 ,required=False)
	parser.add_argument('-c', '--configFile', help='-[optional] Specify the path to the configuration file containing arguments for Graph2Vec. If no path is provided, default values will be used.', required=False)
	parser.add_argument('-o', '--outputPath', action='store', help='-[optional] Specify the output path. If no path is provided, the "results" folder will be used.', required=False, default='.'+os.sep+'results')
	parser.add_argument('-p', '--plotFormat',  action='store', type=str, default="svg", choices=['svg', 'png'], help='-[optional] Specify the format of the image output. (svg or png; default: svg)', required=False)
	parser.add_argument('--verbose', action='store_true', default=False, help='Allow extra prints.')

	parser._optionals.title = Fore.CYAN + Style.BRIGHT + "Arguments" + Style.RESET_ALL

	return parser


def main(args: argparse.Namespace) -> None:
	'''
	This program analyzes molecular dynamics (MD) replica data to identify key structural changes and cluster similar states.

	This function represents the main branch of the program. After parsing the command-line arguments, the program begins its workflow:

	* Loading all the MD replica files and building a NetworkX representation of each frame.
	* Preprocessing each replica by removing edges with low entropy. The entropy is computed intra-replica (within each individual replica), while the filter is applied inter-replica (across all replicas).
	* Using Graph2Vec to compute meaningful embeddings for each frame, transforming the replicas into time series.
	* Computing the barycenter of the time series.
	* Performing a ranking of the time series based on their similarity to the barycenter.
	* Computing hierarchical clustering on the time series.

	During execution, several plots and files are saved in the *out_path* specified in:py:func:`create_parser`.

	Plots:

	* 2D plot of the replica embeddings.
	* 2D plot of the replica embeddings with the barycenter.
	* Plot of the iterative pruning process.
	* Dendrogram plot with a cut line computed using the elbow method.

	Files:

	* `metadata.tsv`: DataFrame containing information about the replica and frame. Can be used to index the replica embeddings.
	* `subgraphs_emb.pkl`: List of subgraphs embeddings belonging to different replicas.
	* `dtw_matrix.tsv`: Squared matrix where each value represents the Euclidean distance between aligned time series.
	* `dtw_mapping.txt`: This file contains the frame of each replica and the corresponding frame of the barycenter. It illustrates the dynamic time warping indexing and the Euclidean distance between frames.
	* `iterative_ranks.tsv`: Iterative pruning of all replicas based on their distance from the barycenter.
	'''
	# Start phase 1 

	###### Graphs construction ######

	print(f"Function : {args}\n")
	
	# Check if the output directory exists
	out_path = os.path.join(args.outputPath)

	if not os.path.exists(args.outputPath): raise FileNotFoundError(f"Path '{args.outputPath}' does not exist.")

	#  Read configuration file (.yaml)
	g2v_config = parse_config(args.configFile)
	workers = g2v_config['workers']

	# Load files and compute graphs, meta data and entropies intra replica
	if args.features:
		features_df = pd.read_csv(args.features, sep='\t', index_col=0)
	else:
		features_df = pd.DataFrame()

	if args.verbose:
		print('\nSet of features:\n')
		print(f'\n{features_df}\n')
		if features_df.empty: 
			print("The features will default to unique identifiers.\n")

	if args.InputPath:
		graphs, meta_data, entropies = crawl_replica_files(args.InputPath[0], args.InputPath[1], features_df, args.verbose)
	else:
		graphs, meta_data, entropies = iterate_replica_files(args.Files, features_df, args.verbose)

	replica_names = meta_data['Rep'].unique()
	if len(replica_names) < 2:
		print(Fore.RED + Style.BRIGHT + f"[ERROR] Found {len(replica_names)} replica. At least two replicas should be provided. Please check the input files or directory structure." + Style.RESET_ALL)
		return 1
	elif len(replica_names) == 2:
		print(Fore.YELLOW + Style.BRIGHT + f"[WARNING] Only {len(replica_names)} replicas were found. The clustering and pruning will not provide meaningful results, try with three or more replicas." + Style.RESET_ALL)

	if args.verbose:
		print(f"\nFound {len(replica_names)} replicas\n")
		print("\nFiltering...\n")

	# Save the graphs, metadata, and entropies to files
	with  open(os.path.join(out_path, 'graphs.pkl'), 'wb') as f:
		pk.dump(graphs, f)

	entropies.to_csv(os.path.join(out_path, 'entropies.tsv'), sep='\t', index=False)

	# Filter the graphs based on the entropy threshold
	subgraphs = entropy_filter(graphs, entropies, args.edgeFilter)
	with  open(os.path.join(out_path, 'subgraphs.pkl'), 'wb') as f:
		pk.dump(subgraphs, f)

	# ###### Compute Graph2Vec Embeddings ######

	if args.verbose:
		print(f"Starting Graph2Vec{g2v_config}")


	### Run Grap2Vec ###

	# Shuffle the data and metadata
	meta_data.reset_index(drop=True, inplace=True)
	meta_data = meta_data.sample(frac=1, random_state=42)
	subgraphs = [subgraphs[i] for i in meta_data.index]

	subgraphs_emb = g2v_fit_transform(g2v_config, subgraphs)

	# Reorder the embeddings to match the original graphs
	meta_data = meta_data.reset_index(drop=True).sort_values(['Rep', 'Frame'])

	subgraphs_emb = subgraphs_emb[meta_data.index]



	###### Barycenter computation ######

	# Reshape the data to shape (replica, frames, G2V emb dim)
	replicas_ts = [subgraphs_emb[meta_data['Rep'] == replica] for replica in meta_data['Rep'].unique()]
	
	# Save subgraphs to file
	with open(os.path.join(out_path, 'subgraphs_emb.pkl'), 'wb') as f:
		pk.dump(replicas_ts, f)
	meta_data.to_csv(os.path.join(out_path, 'metadata.tsv'), sep='\t', index=False)
	
	# Compute barycenter 
	barycenter = dtw_barycenter_averaging(replicas_ts)
	
	# Compute and save dtw mapping and score
	dtw_mapping(replicas_ts, barycenter, meta_data, out_path) 

	# Fit PCA and Spectral on data 
	reduced_emb = dim_reduction(subgraphs_emb, barycenter, workers)

	# Update metadata adding the barycenter info
	meta_data = pd.concat([meta_data, pd.DataFrame({'Rep': ['barycenter']*len(barycenter), 'Frame': np.arange(0, len(barycenter))})])

	# Plot subgraphs embedding
	plot_emb_rep(reduced_emb, meta_data, out_path, args.plotFormat)
	# Plot subgraphs embedding with barycenter
	plot_emb_bary(reduced_emb, meta_data, out_path,  args.plotFormat)

	# End of phase 1
	
	###############################################################################

	# Phase 2

	###### Rank based on DTW distance ######

	# Reset metadata by removing the barycenter info
	meta_data = meta_data[:-len(barycenter)]

	# Perform iterative pruning based on the distance from the barycenter
	replica_ranks_df, barycenters = iterative_pruning(replicas_ts, meta_data)
	replica_ranks_df.to_csv(os.path.join(out_path, 'iterative_rank.tsv'), sep='\t')

	if args.verbose:
		print("\nIterative Pruning based on distance from barycenter:")
		print(f"\n{replica_ranks_df}\n")

	# Visualize the pruning process
	plot_pruning(subgraphs_emb, barycenters, meta_data, replica_ranks_df, out_path, args.plotFormat, workers)

	###### Rank based on clustering ######

	# Compute dtw distance for each time series
	dtw_matrix_df = compute_dtwmatrix(replicas_ts, meta_data)
	dtw_matrix_df.to_csv(os.path.join(out_path, 'dtw_matrix.tsv'), sep='\t')

	if args.verbose:
		print("\nDynamic Time Warping Distance Matrix:")
		print(f"\n{dtw_matrix_df}\n")

	# Perform hierarchical clustering 
	clusters_elbow = hierarchical_clustering_rank(dtw_matrix_df, out_path, args.plotFormat, args.verbose)
	
	# Insert the cluster information
	meta_data['Cluster'] = [clusters_elbow[rep] for rep in meta_data['Rep']]

	if args.verbose:
		print("\n\nReplica and clusters based on elbow cut of dendogram:")
		print(meta_data[['Rep', 'Cluster']].drop_duplicates().sort_values(by=['Cluster']).reset_index(drop=True))

	print(Style.BRIGHT + Fore.GREEN + "\nDone!\n" + Style.RESET_ALL)
	
	# End phase 2


def cli():
    parser = create_parser()
    args = parser.parse_args()
    main(args)


if __name__ == '__main__':
    cli()