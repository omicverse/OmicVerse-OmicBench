"""Hand-coded scanpy reference: PCA(50) + neighbors(15) + UMAP."""
import scanpy as sc


def run(adata):
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
    sc.tl.umap(adata)
    return adata
