# scRNA Metabolic Entropy Analysis

## Overview

scRNA Metabolic Entropy Analysis is a Python package designed for the analysis of single-cell RNA sequencing data with a focus on entropy calculations within metabolic gene networks. The package offers tools for data preprocessing, network analysis, and entropy computation.

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/scRNA_Entropy_Analysis.git

cd scRNA_Entropy_Analysis
pip install .

```

## Usage

# Import the package:
```python
from entropy_analysis.data_preprocessing import load_and_preprocess_data, filter_genes, load_internal_data
from entropy_analysis.network_analysis import construct_network, get_subnetwork_matrices
from entropy_analysis.entropy_calculation import comp_srana
```

# Load internal data
```python
entrez_id, metabolic_Gene, network_data = load_internal_data()

```

# Load and preprocess data

```python
# Example usage

path = '/path/to/directory'
file_name = 'my_data_file.h5ad'
adata = load_and_preprocess_data(path, file_name) # adata is a normalized AnnData object.

''' note: please specify your groups to be calculated, e.g., 
cells = pd.read_csv("%s/cells.tsv" % PATH_I, sep="\t")
adata.obs["Diganosis"] =  cells["diagnosis"].values
adata.obs["celltype"] =  cells["celltype"].values
adata.obs["braak_diff"] =  cells["braak_diff"].values'''
```


# Filter genes and prepare network data
```python
adata_filtered, network_data_filtered = filter_genes(adata, entrez_id, metabolic_Gene, network_data)
```

# Construct network and get subnetwork matrices

```python
subgraph = construct_network(network_data_filtered)
adjMC_m, expMC_m = get_subnetwork_matrices(subgraph, adata_filtered)
```

# Compute entropy
```python
entropy_results = comp_srana(adjMC_m, expMC_m, local=True, mc_cores=16)
# The results now include 'SR', 'locS', 'nlocS', and 'expMC_var'
```
