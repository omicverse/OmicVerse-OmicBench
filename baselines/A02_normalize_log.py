"""Hand-coded scanpy reference: save raw counts, normalize_total 1e4, log1p."""
import scanpy as sc


def run(adata):
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    return adata
