"""Hand-coded scanpy reference: subset T cells + re-cluster (CD4 / CD8 split)."""
import scanpy as sc


def run(adata):
    is_t = adata.obs["cell_type"].astype(str).str.contains("T cell", case=False)
    sub = adata[is_t].copy()
    sc.pp.neighbors(sub, n_neighbors=15, use_rep="X_pca")
    sc.tl.leiden(sub, resolution=0.5, key_added="t_subcluster",
                 flavor="igraph", n_iterations=2, directed=False)
    return sub
