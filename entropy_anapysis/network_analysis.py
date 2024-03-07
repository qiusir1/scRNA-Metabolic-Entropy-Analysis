import networkx as nx
import pandas as pd
import scipy.sparse as sp
import numpy as np

def construct_network(data_frame):
    G = nx.from_pandas_adjacency(data_frame)
    largest_cc = max(nx.connected_components(G), key=len)
    subgraph = G.subgraph(largest_cc)
    return subgraph

def get_subnetwork_matrices(subgraph, adata_filtered):
    adjMC_m = nx.to_pandas_adjacency(subgraph)
    expMC_m = adata_filtered[:, adjMC_m.index]
    expMC_m.X = sp.csr_matrix(np.log2(expMC_m.X.toarray() + 1.1))
    return adjMC_m, expMC_m
