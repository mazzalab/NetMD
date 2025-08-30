# NetMD - Synchronizing Graph-Embedded Molecular Dynamics Trajectories via Time-Warping

NetMD is a computational method for identifying consensus behavior across multiple molecular dynamics (MD) simulations. Using Graph-based Embeddings and Dynamic Time Warping, NetMD aligns trajectories that may be temporally out of sync and pinpoints the replicas that most faithfully represent the overall ensemble behavior. This enables consistent comparisons across simulations and supports reliable characterization of system variants, making it easier to detect shared patterns and reduce the influence of outliers or simulation artifacts.

[![Conda Version](https://anaconda.org/bioconda/netmd/badges/version.svg)](https://anaconda.org/bioconda/netmd)
[![Last updated](https://anaconda.org/bioconda/netmd/badges/latest_release_date.svg)]()
[![Platforms](https://anaconda.org/bioconda/netmd/badges/platforms.svg)]()
[![Download](https://anaconda.org/bioconda/netmd/badges/downloads.svg)]()
[![License](https://anaconda.org/bioconda/netmd/badges/license.svg)]()

## 📦 Installation

You can pull `netmd` without installing any dependencies by using the pre-built Docker image available on Quay.io.

```bash
docker pull quay.io/biocontainers/netmd:1.0.2--pyh3c853c9_0
```

Install via Conda:

```bash
# either create a new conda env and install NetMD
conda create -n netmd -c conda-forge bioconda::netmd

# or, if you already have an active conda env, just install
conda install -c conda-forge bioconda::netmd
```

## 🚀 Quickstart

Run NetMD on example trajectories:

```bash
netmd \
  -F example/GLUT1_WT/FullReplica10_WT.tsv example/GLUT1_WT/FullReplica2_WT.tsv example/GLUT1_WT/FullReplica3_WT.tsv \
  -o results \
  -e 0.1 \
  -c conf/config_g2v.yml \
  --verbose
```

or

```
docker run --rm \
  -v $(pwd):/data \
  quay.io/biocontainers/netmd:1.0.1--pyh3c853c9_0 \
  netmd \
    -F /data/example/GLUT1_WT/FullReplica10_WT.tsv /data/example/GLUT1_WT/FullReplica2_WT.tsv /data/example/GLUT1_WT/FullReplica3_WT.tsv \
    -o /data/results \
    -e 0.1 \
    -c /data/conf/config_g2v.yml \
    --verbose
```

> [!NOTE]
> Replace `$(pwd)` with the full path to your working directory if you're not on Unix-based systems (e.g., use `%cd%` on Windows PowerShell).

**Flags:**

- `-F`: List of residue-residue contacts files extracted from MD trajectories with the Python package [GetContacts](https://github.com/getcontacts/getcontacts).
- `-e`: Entropy filter on graph's edges (value between 0. and 1.). Lower thresholds increase sensitivity by capturing more (including weaker) interactions, while higher thresholds prioritize only the strongest, most dynamic interactions, reducing noise.
- `-c`: graph2vec configuration YAML.
- `-o`: Output directory.
- `--verbose`: Show more detailed logs.

Edit `config_g2v.yml` to customize graph2vec parameters, embedding dimensions, and other settings. See the [documentation](https://mazzalab.github.io/NetMD) for more options.

## ✨ Features

- **Smart Graph Embedding**: Transform complex molecular dynamics frames into meaningful low-dimensional vectors using the graph2vec algorithm. Each protein conformation becomes a compact, analyzable representation that captures essential structural information.

- **Consensus Discovery**: Automatically identify the most representative behaviors across multiple MD simulations using Dynamic Time Warping (DTW) barycenter averaging. Align trajectories that may be temporally out of sync and find the true consensus patterns.

- **Intelligent Clustering**: Hierarchically cluster molecular conformations to reveal shared structural patterns and detect outlier behaviors. Reduce noise from simulation artifacts and focus on biologically relevant states.

- **Entropy-Based Filtering**: Apply smart edge filtering based on entropy thresholds to eliminate irrelevant interactions and focus on the most informative structural features.

## 🛠️ Workflow

<p align="center">
  <img src="./docs/src_docs/source/_static/img/workflow/Figure1.svg" alt="Workflow diagram" width="90%"/>
</p>

## 📓 Interactive Analysis Notebook

Explore a complete case study analysis of the Glucose Transporter 1 (GLUT1) using our Jupyter [notebook](https://github.com/mazzalab/NetMD/blob/main/tutorial/netmd_notebook.ipynb), which demonstrates the full NetMD workflow. Feel free to experiment with it and adapt it to suit your needs.

<!-- [![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/mazzalab/netmd/main?filepath=example/netmd_notebook.ipynb) -->

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
