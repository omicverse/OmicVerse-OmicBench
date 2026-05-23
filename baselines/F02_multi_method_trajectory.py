"""Hand-coded reference: scanpy DPT + PCA-Euclidean as two trajectory methods.

DPT uses diffusion eigenvectors. PCA-Euclidean from root centroid is a
genuinely-different distance metric. Two distinct pseudotimes.
"""
import numpy as np
import scanpy as sc


def run(adata):
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")

    # Method 1: DPT — writes obs['dpt_pseudotime']
    is_root = adata.obs["clusters"].astype(str) == "Ductal"
    adata.uns["iroot"] = int(np.where(is_root)[0][0]) if is_root.any() else 0
    sc.tl.diffmap(adata)
    sc.tl.dpt(adata)

    # Method 2: PCA-space Euclidean distance from Ductal centroid
    pca = adata.obsm["X_pca"]
    if is_root.any():
        centroid = pca[is_root].mean(axis=0)
    else:
        centroid = pca[adata.uns["iroot"]]
    eu = np.linalg.norm(pca - centroid, axis=1)
    if eu.max() > 0:
        eu = eu / eu.max()
    adata.obs["pseudotime_euclidean"] = eu.astype(float)

    adata.uns["trajectory_methods"] = ["dpt", "pca_euclidean"]

    # Identify terminal states: clusters with high mean pseudotime that aren't Ductal
    cluster_strs = adata.obs["clusters"].astype(str)
    cluster_dpt = {c: float(adata.obs.loc[cluster_strs == c, "dpt_pseudotime"]
                              .astype(float).median())
                   for c in cluster_strs.unique() if c != "Ductal"}
    # take top 2 by median pseudotime as terminals
    terminals = sorted(cluster_dpt.items(), key=lambda x: -x[1])[:3]
    adata.uns["terminal_states"] = [c for c, _ in terminals]

    return adata
