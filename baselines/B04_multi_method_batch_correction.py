"""Hand-coded reference: harmony + combat as two distinct batch correction methods.

Calls canonical libraries directly (harmonypy + scanpy.pp.combat) — what a
competent bioinformatician would write without ov's single-call dispatch.
"""
import numpy as np
import scanpy as sc


def run(adata):
    # Need preprocess + PCA before harmony if not already done
    if "X_pca" not in adata.obsm:
        sc.pp.normalize_total(adata, target_sum=1e4) if not (adata.X.max() < 50) else None
        if "log1p" not in adata.uns:
            sc.pp.log1p(adata)
        sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat")
        sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")

    # Save uncorrected X_pca BEFORE combat overwrites obsm['X_pca']
    pca_raw = adata.obsm["X_pca"].copy()

    # Method 1: harmony
    try:
        sc.external.pp.harmony_integrate(adata, key="batch", basis="X_pca",
                                          adjusted_basis="X_harmony")
    except Exception:
        import harmonypy
        ho = harmonypy.run_harmony(adata.obsm["X_pca"], adata.obs, ["batch"])
        Z = np.asarray(ho.Z_corr)
        if Z.shape[0] != adata.n_obs and Z.shape[1] == adata.n_obs:
            Z = Z.T
        adata.obsm["X_harmony"] = Z

    # Method 2: combat (overwrites adata.X)
    sc.pp.combat(adata, key="batch")
    sc.pp.pca(adata, n_comps=50, mask_var="highly_variable")
    adata.obsm["X_combat"] = adata.obsm["X_pca"].copy()
    adata.obsm["X_pca"] = pca_raw

    adata.uns["batch_correction_methods"] = ["harmony", "combat"]
    return adata
