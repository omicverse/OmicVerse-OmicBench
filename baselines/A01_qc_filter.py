"""Hand-coded scanpy reference for A01: QC + multi-feature percentages
+ best-effort doublet flagging.

omicdev does not ship scrublet / scDblFinder / doubletdetection. A real
bioinformatician without ov.Agent's registry would `pip install` one of
these — that's not possible inside a hand-coded reference. As fallback,
this baseline uses two count-statistic-based doublet heuristics
(high-total-counts and high-gene-count outlier) that correlate strongly
with real doublet detection. This faithfully reflects the "what a
hand-coded analyst gets without ov" floor.
"""
import numpy as np
import scanpy as sc


def run(adata):
    # Step 1-2: cell + gene filters
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # Step 3: mt + ribo + hb fractions
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    adata.var["hb"] = adata.var_names.str.contains(r"^HB[^P]", regex=True)
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"],
        percent_top=None, log1p=False, inplace=True,
    )

    # Step 4: mt% filter
    adata = adata[adata.obs.pct_counts_mt < 20].copy()

    # Step 5: two doublet heuristics (fallback without scrublet/scDblFinder)
    n = adata.n_obs
    # method 1: high-total-counts rank (doublets tend to have ≈2× counts)
    rank_counts = np.argsort(np.argsort(-adata.obs["total_counts"].values)) / n
    adata.obs["doublet_score_total_counts"] = 1.0 - rank_counts
    # method 2: high-gene-count rank (doublets tend to have more diverse genes)
    rank_genes = np.argsort(np.argsort(-adata.obs["n_genes_by_counts"].values)) / n
    adata.obs["doublet_score_n_genes"] = 1.0 - rank_genes
    # consensus: cells in top 8% by both methods
    thresh = 0.92
    mask1 = adata.obs["doublet_score_total_counts"] > thresh
    mask2 = adata.obs["doublet_score_n_genes"] > thresh
    adata.obs["doublet_consensus"] = (mask1 | mask2)
    # filter consensus doublets
    adata = adata[~adata.obs["doublet_consensus"]].copy()

    return adata
