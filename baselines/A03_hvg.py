"""Hand-coded scanpy reference: seurat_v3 HVG (uses raw counts layer)."""
import scanpy as sc


def run(adata):
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                 layer="counts", inplace=True)
    return adata
