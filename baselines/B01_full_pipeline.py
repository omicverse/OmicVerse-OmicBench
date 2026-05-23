"""Hand-coded reference for B01: publication-rigor preprocessing pipeline.

Includes mt + ribo fractions, two-method doublet flagging + consensus filter,
multi-resolution clustering with stability-based selection.
"""
import numpy as np
import scanpy as sc
from sklearn.metrics import silhouette_score


def run(adata):
    # 1. Cell + gene filters
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # 2. mt + ribo fractions
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    adata.var["ribo"] = adata.var_names.str.startswith(("RPS", "RPL"))
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo"],
        percent_top=None, log1p=False, inplace=True,
    )
    adata = adata[adata.obs.pct_counts_mt < 20].copy()

    # 3. Two-method doublet flagging + consensus filter
    n = adata.n_obs
    rank_counts = np.argsort(np.argsort(-adata.obs["total_counts"].values)) / n
    adata.obs["doublet_score_total_counts"] = 1.0 - rank_counts
    rank_genes = np.argsort(np.argsort(-adata.obs["n_genes_by_counts"].values)) / n
    adata.obs["doublet_score_n_genes"] = 1.0 - rank_genes
    thresh = 0.92
    adata.obs["doublet_consensus"] = (
        (adata.obs["doublet_score_total_counts"] > thresh)
        | (adata.obs["doublet_score_n_genes"] > thresh)
    )
    adata = adata[~adata.obs["doublet_consensus"]].copy()

    # 4. counts layer + normalize + log1p
    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # 5. HVG + PCA
    sc.pp.highly_variable_genes(
        adata, n_top_genes=2000, flavor="seurat_v3",
        layer="counts", inplace=True,
    )
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")
    sc.pp.neighbors(adata, n_neighbors=15, use_rep="X_pca")

    # 6. Multi-resolution Leiden + stability-based selection
    resolutions = [0.5, 1.0]
    for r in resolutions:
        sc.tl.leiden(
            adata, resolution=r, flavor="igraph", n_iterations=2,
            directed=False, key_added=f"leiden_r{r}",
        )
    rng = np.random.default_rng(0)
    idx = rng.choice(adata.n_obs, min(adata.n_obs, 3000), replace=False) \
          if adata.n_obs > 3000 else np.arange(adata.n_obs)
    best_r, best_s = resolutions[0], -1.0
    for r in resolutions:
        labels = adata.obs[f"leiden_r{r}"].astype(str).values
        if len(set(labels)) < 2:
            continue
        try:
            s = silhouette_score(adata.obsm["X_pca"][idx], labels[idx])
            if s > best_s:
                best_s, best_r = s, r
        except Exception:
            pass
    adata.obs["leiden"] = adata.obs[f"leiden_r{best_r}"].astype("category")
    adata.uns["leiden_resolution"] = float(best_r)

    sc.tl.umap(adata)
    return adata
