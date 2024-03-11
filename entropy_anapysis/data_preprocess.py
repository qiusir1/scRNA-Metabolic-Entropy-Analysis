import pandas as pd
import scanpy as sc
import os


def load_and_preprocess_data(path_i, filename):
    # Construct the full path for the file
    full_path = f"{path_i}/{filename}"
    
    # Load the file
    adata = sc.read_h5ad(full_path)
    
    # Check if the data has already been normalized
    if "log1p" not in adata.uns:
        # Normalize the data
        sc.pp.normalize_total(adata, target_sum=1e4)

    return adata


def load_internal_data():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    
    entrez_id = pd.read_csv(os.path.join(data_dir, "entrez_to_name.tsv"), sep="\t")
    metabolic_Gene = pd.read_csv(os.path.join(data_dir, "metabolism_protein_gene_name.txt"), sep="\t")
    network_data = pd.read_csv(os.path.join(data_dir, "net17Jan16_matrix_GeneName.tsv"), sep="\t", index_col=0)
    
    return entrez_id, metabolic_Gene, network_data

def filter_genes(adata, entrez_id, metabolic_Gene, network_data):
    # find common genes within ppi and sc data
    gene_list = entrez_id["gene_name"].tolist()
    common_genes = [gene for gene in gene_list if gene in adata.var_names]
    adata_filtered = adata[:, common_genes]

    # network that only have the common genes
    network_data_filtered = network_data[network_data.columns.isin(common_genes)]
    network_data_filtered = network_data_filtered[network_data_filtered.index.isin(common_genes)]

    # Find the intersection with metabolic genes
    index_series = pd.Series(network_data_filtered.index)
    common_elements = index_series[index_series.isin(metabolic_Gene["gene_name"])].tolist()

    # network that only keep the metabolic genes in rows
    network_data_filtered = network_data_filtered[network_data_filtered.index.isin(common_elements)]

    return adata_filtered, network_data_filtered
