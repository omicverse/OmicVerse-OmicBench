"""Hand-coded reference for B05: two annotation methods.

Method 1: marker-score based per-cluster annotation using canonical PBMC markers.
Method 2: nearest-cluster mapping using rank_genes_groups top markers.

This is the floor — what a competent analyst writes without celltypist /
SingleR / popV. ov.Agent has these methods built into its skill registry.
"""
import numpy as np
import pandas as pd
import scanpy as sc


PBMC_MARKER_PROFILES = {
    "CD4+ T cell":           ["CD3D", "CD3E", "CD4", "IL7R", "CCR7"],
    "Cytotoxic T cell":      ["CD3D", "CD8A", "CD8B", "GZMK", "CCL5"],
    "B cell":                ["CD79A", "CD79B", "MS4A1", "CD19"],
    "Natural killer cell":   ["NKG7", "GNLY", "NCAM1", "KLRD1", "GZMB"],
    "CD14+ monocyte":        ["CD14", "LYZ", "S100A8", "S100A9"],
    "CD16+ monocyte":        ["FCGR3A", "MS4A7"],
    "Dendritic cell":        ["CST3", "FCER1A", "CLEC10A"],
    "Plasmacytoid dendritic cell": ["IL3RA", "CLEC4C", "LILRA4"],
    "Megakaryocyte":         ["PPBP", "PF4", "GP9"],
}


def _expr_mean_in_cluster(adata, gene, cluster_mask):
    if gene not in adata.var_names:
        return 0.0
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    x = np.asarray(x).flatten()
    return float(x[cluster_mask].mean())


def run(adata):
    if "leiden" not in adata.obs.columns:
        sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")
        sc.tl.leiden(adata, resolution=1.0, flavor="igraph",
                     n_iterations=2, directed=False)

    # Method 1: marker-score (mean expression of canonical markers per cluster,
    # assign cluster to highest-scoring cell-type)
    leiden = adata.obs["leiden"].astype(str)
    cluster_to_type1 = {}
    for c in sorted(leiden.unique()):
        mask = (leiden == c).values
        scores = {}
        for ct, markers in PBMC_MARKER_PROFILES.items():
            scores[ct] = float(np.mean([_expr_mean_in_cluster(adata, g, mask)
                                         for g in markers]))
        cluster_to_type1[c] = max(scores, key=scores.get)
    adata.obs["cell_type_marker_score"] = leiden.map(cluster_to_type1).astype("category")

    # Method 2: rank_genes_groups DE → cluster's top-N markers' best canonical match
    if "rank_genes_groups" not in adata.uns:
        sc.tl.rank_genes_groups(adata, groupby="leiden", method="wilcoxon")
    names = adata.uns["rank_genes_groups"]["names"]
    cluster_to_type2 = {}
    for c in names.dtype.names:
        top10 = {str(g).upper() for g in list(names[c])[:10]}
        scores = {}
        for ct, markers in PBMC_MARKER_PROFILES.items():
            scores[ct] = len(top10 & {m.upper() for m in markers})
        cluster_to_type2[c] = max(scores, key=scores.get) if max(scores.values()) > 0 \
                              else cluster_to_type1[c]
    adata.obs["cell_type_de_top_markers"] = leiden.map(cluster_to_type2).astype("category")

    return adata
