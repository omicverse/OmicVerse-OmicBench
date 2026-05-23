"""Hand-coded scvelo reference: stochastic + dynamical as two velocity modes."""
import scanpy as sc
import scvelo as scv


def run(adata):
    scv.pp.filter_genes(adata, min_shared_counts=10)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    # Mode 1: stochastic
    scv.tl.velocity(adata, mode="stochastic", vkey="velocity")
    scv.tl.velocity_graph(adata, vkey="velocity", n_jobs=1)
    scv.tl.velocity_embedding(adata, basis="umap", vkey="velocity")

    # Mode 2: dynamical
    scv.tl.recover_dynamics(adata, n_jobs=1)
    scv.tl.velocity(adata, mode="dynamical", vkey="velocity_alt")

    adata.uns["velocity_modes"] = ["stochastic", "dynamical"]
    return adata
