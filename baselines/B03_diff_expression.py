"""Hand-coded reference for B03: multi-method DE — Wilcoxon + t-test."""
import scanpy as sc


def run(adata):
    # Method 1: Wilcoxon (default scanpy slot)
    sc.tl.rank_genes_groups(
        adata, groupby="leiden", method="wilcoxon",
        key_added="rank_genes_groups",
    )
    # Method 2: t-test, stored under a separate uns key
    sc.tl.rank_genes_groups(
        adata, groupby="leiden", method="t-test",
        key_added="rank_genes_t_test",
    )
    return adata
