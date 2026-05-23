"""Statistical helpers — bootstrap CI, paired McNemar, BH-FDR.

These follow the SWE-bench Live convention: percentile bootstrap (n=10 000)
for binary Pass@1, exact two-sided binomial McNemar for paired comparisons,
Benjamini-Hochberg false discovery control at q=0.10 across the family of
pairwise system contrasts.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bootstrap_ci(passes: np.ndarray, n_boot: int = 10_000, alpha: float = 0.05,
                 rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Percentile bootstrap CI for a binary proportion. Returns (mean, lo, hi)."""
    if rng is None:
        rng = np.random.default_rng(0)
    n = len(passes)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(passes.mean())
    if n == 1:
        return (mean, mean, mean)
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = passes[idx].mean(axis=1)
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return (mean, lo, hi)


def mcnemar_paired(a_pass: pd.Series, b_pass: pd.Series) -> tuple[int, int, float | None]:
    """Paired McNemar on binary pass/fail series indexed by matched task IDs.

    Returns (b01, b10, p_two_sided). Uses exact binomial — appropriate for
    small n_discordant (≤ ~25). If n_discordant == 0, p is None.
    """
    common = a_pass.index.intersection(b_pass.index)
    a = a_pass.loc[common].astype(bool).values
    b = b_pass.loc[common].astype(bool).values
    b01 = int(((a == True) & (b == False)).sum())
    b10 = int(((a == False) & (b == True)).sum())
    n = b01 + b10
    if n == 0:
        return (b01, b10, None)
    k = min(b01, b10)
    p_one_side = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    p_two_side = min(1.0, 2 * p_one_side)
    return (b01, b10, p_two_side)


def benjamini_hochberg(pvals: list[float | None], q: float = 0.10) -> list[bool]:
    """BH-FDR rejection at level q. None p-values are not rejected.

    Returns a list of booleans aligned with input order.
    """
    items = [(i, p) for i, p in enumerate(pvals) if p is not None]
    if not items:
        return [False] * len(pvals)
    items.sort(key=lambda kv: kv[1])
    m = len(items)
    reject = [False] * len(pvals)
    cutoff = -1
    for rank, (_i, p) in enumerate(items, start=1):
        if p <= q * rank / m:
            cutoff = rank
    for rank, (i, _p) in enumerate(items, start=1):
        if rank <= cutoff:
            reject[i] = True
    return reject
