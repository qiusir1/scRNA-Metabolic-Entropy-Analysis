import numpy as np
import scipy.sparse as sp
import scanpy as sc
from multiprocessing import Pool
import datetime


def comp_max_sr(adj_matrix):
    # Find the largest eigenvalue of the adjacency matrix
    eigenvalues, _ = sp.linalg.eigs(adj_matrix, k=1, which='LM')
    max_sr = np.log(eigenvalues[0])
    return max_sr.real

def comp_s(p_v):
    # Compute local signaling entropy
    p_v_nonzero = p_v[p_v > 0]
    return -np.sum(p_v_nonzero * np.log(p_v_nonzero)) # sum (Pij * logPij)

def comp_ns(p_v):
    # Compute normalized local signaling entropy
    p_v_nonzero = p_v[p_v > 0]
    if len(p_v_nonzero) > 1:
        return -np.sum(p_v_nonzero * np.log(p_v_nonzero)) / np.log(len(p_v_nonzero)) # sum (Pij * logPij) / logK
    else:
        return 0

def comp_srana_prl(args):
    idx, exp_m, adj_m, local, max_sr = args

    exp_v = exp_m[:, idx]
    sumexp_v = adj_m.dot(exp_v) #A ^ X, one column

    inv_p_v = exp_v * sumexp_v # x(i)*A ^ X 
    nf = np.sum(inv_p_v) 
    inv_p_v /= nf # normalized invariant measure, inv_p_v: 0 - 1
    
    p_ij = (adj_m * exp_v) / sumexp_v[:,None] # Pij
    print(p_ij)
    
    s_v = np.apply_along_axis(comp_s, 1, p_ij) #local entropy
    ns_v = np.apply_along_axis(comp_ns, 1, p_ij) if local else None # normalized local entropy

    sr = np.sum(inv_p_v * s_v)

    if max_sr is not None:
        sr /= max_sr

    return sr, s_v, ns_v

def comp_srana(adjMC_m, expMC_m, local, mc_cores):
    adj_m = adjMC_m.values
    exp_m = expMC_m.X.T.toarray()
    features = expMC_m.var
    max_sr = comp_max_sr(adj_m)
    
    print("Start calculating...", datetime.datetime.now())
    with Pool(mc_cores) as pool:
        results = pool.map(comp_srana_prl, [(idx, exp_m, adj_m, local, max_sr) for idx in range(exp_m.shape[1])])
    
    sr, s_v, ns_v = zip(*results)
    return {
        'SR': np.array(sr),
        'locS': np.array(s_v),
        'nlocS': np.array(ns_v) if local else None,
        'expMC_var': np.array(features)
    }
    print("Done calculating...", datetime.datetime.now())

