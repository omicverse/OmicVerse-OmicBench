"""Hand-coded reference for G01: alpha + beta + group statistical tests.

Pure-Python (numpy + scipy). Tests group-level alpha diversity differences
(Kruskal-Wallis on shannon by group) and beta diversity (PERMANOVA-style
F-statistic via scipy permutations).
"""
import numpy as np
import scipy.spatial.distance as ssd
from scipy.stats import kruskal


def run(adata):
    X = adata.X
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X, dtype=float)

    # Alpha diversity per sample
    p = X / np.maximum(X.sum(axis=1, keepdims=True), 1e-12)
    p_safe = np.where(p > 0, p, 1e-12)
    adata.obs["shannon"] = -(p * np.log(p_safe)).sum(axis=1)
    adata.obs["simpson"] = 1 - (p ** 2).sum(axis=1)
    adata.obs["observed_otus"] = (X > 0).sum(axis=1)

    # Beta diversity: pairwise Bray-Curtis
    bc = ssd.squareform(ssd.pdist(X, metric="braycurtis"))
    adata.obsp["bray_curtis"] = bc

    # Group statistical tests (uses obs["group"] if present)
    if "group" in adata.obs.columns:
        groups = adata.obs["group"].astype(str).values
        unique_groups = list(set(groups))
        # Alpha: Kruskal-Wallis on shannon
        if len(unique_groups) >= 2:
            samples_per_group = [adata.obs.loc[groups == g, "shannon"].astype(float).values
                                  for g in unique_groups]
            samples_per_group = [s for s in samples_per_group if len(s) >= 2]
            if len(samples_per_group) >= 2:
                stat, p_alpha = kruskal(*samples_per_group)
                adata.uns["alpha_test_kruskal"] = {
                    "metric": "shannon", "statistic": float(stat),
                    "pvalue": float(p_alpha), "groups": unique_groups,
                }
        # Beta: PERMANOVA-style F-stat with permutations on Bray-Curtis
        F_obs, p_beta = _permanova(bc, groups, n_permutations=999)
        adata.uns["beta_permanova"] = {
            "metric": "bray_curtis", "pseudo_F": float(F_obs),
            "pvalue": float(p_beta), "groups": unique_groups,
        }
    return adata


def _permanova(D, group_labels, n_permutations=999, rng_seed=0):
    """Pseudo-F + p-value via permutation, on a square distance matrix."""
    rng = np.random.default_rng(rng_seed)
    n = D.shape[0]
    groups = np.asarray(group_labels)
    SS_total = (D ** 2).sum() / (2 * n)
    def pseudo_F(g):
        SS_within = 0.0
        ng = 0
        for u in set(g):
            idx = np.where(g == u)[0]
            if len(idx) < 2:
                continue
            ng += 1
            sub = D[np.ix_(idx, idx)]
            SS_within += (sub ** 2).sum() / (2 * len(idx))
        SS_between = SS_total - SS_within
        df_b = max(ng - 1, 1)
        df_w = max(n - ng, 1)
        return (SS_between / df_b) / max(SS_within / df_w, 1e-12)
    F_obs = pseudo_F(groups)
    F_perm = np.zeros(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(groups)
        F_perm[i] = pseudo_F(perm)
    p = float((np.sum(F_perm >= F_obs) + 1) / (n_permutations + 1))
    return F_obs, p
