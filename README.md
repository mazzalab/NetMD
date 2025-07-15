# NetMD - Synchronizing Graph-Embedded Molecular Dynamics Trajectories via Time-Warping

NetMD is a computational method for identifying consensus behavior across multiple molecular dynamics (MD) simulations. Using Graph-based Embeddings and Dynamic Time Warping, NetMD aligns trajectories that may be temporally out of sync and pinpoints the replicas that most faithfully represent the overall ensemble behavior. This enables consistent comparisons across simulations and supports reliable characterization of system variants, making it easier to detect shared patterns and reduce the influence of outliers or simulation artifacts.

[![Conda Version](https://img.shields.io/conda/vn/conda-forge/netmd.svg)](https://anaconda.org/conda-forge/netmd)
[![License](https://img.shields.io/github/license/mazzalab/netmd.svg)](LICENSE)

## 📦 Installation

Install via Conda:

```bash
conda install -c conda-forge netmd
```

## 🚀 Quickstart

Run NetMD on example trajectories:

```bash
netmd \
  -F example/GLUT1_WT/FullReplica1_WT.tsv example/GLUT1_WT/FullReplica2_WT.tsv example/GLUT1_WT/FullReplica3_WT.tsv \
  -o results \
  -e 0.1 \
  -c conf/config_g2v.yml \
  --verbose
```

**Flags:**

* `-F`: List of residue-residue contacts files extracted from MD trajectories with the Python package [GetContacts](https://github.com/getcontacts/getcontacts).
* `-e`: Entropy filter on graph's edges (value between 0. and 1.). Lower thresholds increase sensitivity by capturing more (including weaker) interactions, while higher thresholds prioritize only the strongest, most dynamic interactions, reducing noise.
* `-c`: graph2vec configuration YAML. 
* `-o`: Output directory.
* `--verbose`: Show more detailed logs.

Edit `config_g2v.yml` to customize graph2vec parameters, embedding dimensions, and other settings. See documentation for more options.

## ✨ Features

* **Smart Graph Embedding**: Transform complex molecular dynamics frames into meaningful low-dimensional vectors using the graph2vec algorithm. Each protein conformation becomes a compact, analyzable representation that captures essential structural information.

* **Consensus Discovery**: Automatically identify the most representative behaviors across multiple MD simulations using Dynamic Time Warping (DTW) barycenter averaging. Align trajectories that may be temporally out of sync and find the true consensus patterns.

* **Intelligent Clustering**: Hierarchically cluster molecular conformations to reveal shared structural patterns and detect outlier behaviors. Reduce noise from simulation artifacts and focus on biologically relevant states.

* **Entropy-Based Filtering**: Apply smart edge filtering based on entropy thresholds to eliminate irrelevant interactions and focus on the most informative structural features.

## 🛠️ Workflow 

<p align="center">
  <img src="./docs/source/_static/img/workflow/Figure1.svg" alt="Workflow diagram" width="70%"/>
</p>


## 📓 Interactive Analysis Notebook

Explore an end-to-end analysis of the GLUT1 case study within our Jupyter notebook, which outlines the complete NetMD workflow. We encourage you to experiment and customize it to fit your needs.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mazzalab/netmd/main?filepath=example/netmd_notebook.ipynb)


<!-- ## 📚 Documentation

Full docs are available at: [https://yourusername.github.io/netmd/](https://yourusername.github.io/netmd/) -->

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

<!-- ## 📖 Citation

If you use NetMD in your work, please cite:

```bibtex
@article{-----,
  author  = {Manuel Mangoni, Salvatore Daniele Bianco, Francesco Petrizzelli, Michele Pieroni, Pietro Hiram Guzzi, Viviana Caputo, Tommaso Biagini, Tommaso Mazza},
  title   = {Synchronizing Graph-Embedded Molecular Dynamics Trajectories via Time-Warping},
  journal = {XXX},
  year    = {XXXX},
  volume  = {XX},
  number  = {XXX},
  pages   = {X-X},
  url     = {http://yyy}
}
``` -->