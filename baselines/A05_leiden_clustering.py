"""Hand-coded reference for A05: multi-resolution Leiden + stability-based selection.

A real practitioner scans ≥3 resolutions, computes pairwise ARI to
verify cluster structure is stable, and selects the resolution where
clusters look most "biologically chunky" by silhouette or modularity.
"""
import numpy as np
import scanpy as sc
from sklearn.metrics import adjusted_rand_score, silhouette_score


def run(adata):
    resolutions = [0.4, 0.8, 1.2]

    # Run Leiden at each resolution
    for r in resolutions:
        key = f"leiden_r{r}"
        sc.tl.leiden(
            adata, resolution=r, flavor="igraph",
            n_iterations=2, directed=False, key_added=key,
        )

    # Pick the resolution with the highest cell-type silhouette on X_pca
    # (proxy for "biologically chunky"). Real workflows would also look at
    # ARI vs known annotations, modularity, or visual UMAP coherence.
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
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
    return adata
