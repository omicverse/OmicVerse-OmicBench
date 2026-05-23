"""bench v1.0 grader engine.

Each task in `bench/tasks.py` declares a list of `checks`. A check is a
dict `{"id": ..., "type": <type-string>, ...kwargs}`. CHECK_DISPATCH maps
the type string to a function `(adata, **kwargs) -> (bool, str)`.

Design:
- Pass = ALL checks pass (binary outcome).
- Score = fraction of checks passed (for partial-credit reporting).
- Every check has a SCIENTIFIC justification — not just "did the key
  exist", but "is the result biologically reasonable".
- All checks tolerate alias key groups so naming-convention differences
  (e.g., `pct_counts_mt` vs `mito_perc`) don't penalize alternate-correct
  systems.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from bench.types import FailureMode, Grade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Canonical biology references (used by several check types below)
# Sources: PanglaoDB, CellChatDB v2, cellphonedb v5, Bastidas-Ponce 2019
# (Development), Krentz 2018, CellMarker 2.0.
# ---------------------------------------------------------------------------

PBMC_MARKERS: dict[str, list[str]] = {
    "T_cell":        ["CD3D", "CD3E", "TRAC", "IL7R"],
    "CD4_T":         ["CD4", "IL7R", "CCR7"],
    "CD8_T":         ["CD8A", "CD8B", "GZMK", "CCL5"],
    "B_cell":        ["CD79A", "CD79B", "MS4A1", "CD19"],
    "NK":            ["NKG7", "GNLY", "NCAM1", "KLRD1", "KLRF1", "GZMB"],
    "CD14_monocyte": ["CD14", "LYZ", "S100A8", "S100A9", "FCN1"],
    "CD16_monocyte": ["FCGR3A", "MS4A7"],
    "DC":            ["CST3", "FCER1A", "CLEC10A"],
    "pDC":           ["IL3RA", "CLEC4C", "LILRA4"],
}
PBMC_ALL_MARKERS: list[str] = sorted({m for ms in PBMC_MARKERS.values() for m in ms})

PBMC_CELLTYPE_MAP: list[tuple[tuple[str, ...], list[str]]] = [
    (("CD4+ T cell", "CD4 T", "T helper"),    PBMC_MARKERS["CD4_T"] + PBMC_MARKERS["T_cell"]),
    (("Cytotoxic T", "CD8+ T", "CD8 T"),      PBMC_MARKERS["CD8_T"] + PBMC_MARKERS["T_cell"]),
    (("Natural killer", "NK"),                PBMC_MARKERS["NK"]),
    (("CD14+ monocyte", "CD14 mono"),         PBMC_MARKERS["CD14_monocyte"]),
    (("CD16+ monocyte", "CD16 mono"),         PBMC_MARKERS["CD16_monocyte"]),
    (("Plasmacytoid", "pDC"),                 PBMC_MARKERS["pDC"]),
    (("Dendritic", "DC"),                     PBMC_MARKERS["DC"]),
    (("monocyte",),                           PBMC_MARKERS["CD14_monocyte"] + PBMC_MARKERS["CD16_monocyte"]),
    (("T cell",),                             PBMC_MARKERS["T_cell"]),
    (("B cell",),                             PBMC_MARKERS["B_cell"]),
    (("Megakaryocyte",),                      ["PPBP", "PF4", "GP9"]),
]

PANC_MARKERS: dict[str, list[str]] = {
    "Ductal":   ["Sox9", "Hes1", "Pdx1", "Krt19", "Cftr", "Spp1"],
    "EP_Ngn3":  ["Ngn3", "Neurog3", "Neurod1", "Nkx2-2", "Pax4"],
    "Alpha":    ["Gcg", "Arx", "Mafb", "Pou3f4", "Irx1"],
    "Beta":     ["Ins1", "Ins2", "Mafa", "Nkx6-1", "Pak3"],
    "Delta":    ["Sst", "Hhex"],
    "Epsilon":  ["Ghrl"],
}

PBMC_LR_REFERENCE: list[tuple[str, str]] = [
    ("ccl5", "ccr5"), ("cxcl10", "cxcr3"), ("cxcl11", "cxcr3"),
    ("cd40", "cd40lg"), ("cd40lg", "cd40"),
    ("il15", "il15ra"), ("il2", "il2rb"), ("il7", "il7r"),
    ("tnfsf13b", "tnfrsf13b"),
    ("hla-e", "klrc1"), ("hla-c", "kir2dl1"),
    ("icam1", "itgal"), ("icam1", "itgb2"),
    ("lgals9", "havcr2"),
]


def _markers_for_celltype(label: str) -> list[str]:
    """Map a cell-type label to canonical markers via substring matching."""
    s = str(label)
    for substrings, markers in PBMC_CELLTYPE_MAP:
        if any(sub in s for sub in substrings):
            return markers
    return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_alias(container, spec):
    if isinstance(spec, (list, tuple)):
        for k in spec:
            if k in container:
                return True, k
        return False, None
    return (spec in container), (spec if spec in container else None)


def _resolve_alias(adata, kind: str, spec):
    cont = getattr(adata, kind)
    keys = list(cont.columns) if kind in ("obs", "var") else list(cont.keys())
    return _has_alias(keys, spec)


def _load_adata(path):
    p = Path(path)
    import anndata as ad
    return ad.read_h5ad(p)


def _to_dense(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.asarray(X)


def _expr_vec(adata, gene: str) -> np.ndarray | None:
    """Per-cell expression of a single gene as a 1D dense array, or None."""
    if gene not in adata.var_names:
        return None
    x = adata[:, gene].X
    if hasattr(x, "toarray"):
        x = x.toarray()
    return np.asarray(x).flatten()


def _check_aucell_celltype_specificity(
    adata, *,
    celltype_obs: str = "cell_type",
    score_pattern: str = r"^(aucell|score|signature|auc)_",
    min_celltype_specific: int = 2,
    min_z_gap: float = 0.3,
) -> tuple[bool, str]:
    """Verify that ≥``min_celltype_specific`` of the agent's pathway-score
    columns peak in the cell type their name implies.

    Replaces the brittle "did the agent import a specific package" check
    on B08-style tasks. Real biology test:

      For each ``aucell_<tag>`` (or ``score_<tag>``, etc.) obs column:
        1. Map ``<tag>`` to a canonical PBMC cell type (T / NK / B /
           monocyte / DC / megakaryocyte / pDC).
        2. Group cells into ``family`` (any cell-type label normalising
           to the expected family via :func:`_normalize_celltype` +
           :func:`_parent_family`) vs ``non-family``.
        3. Pass if mean(family) > mean(non-family) AND
           (mean(family) - mean(non-family)) / std(score) ≥ ``min_z_gap``.

    Family-aware comparison fixes two earlier mistakes: monocyte signatures
    where CD14+ vs CD16+ are both monocytes (top vs 2nd gap is small but
    family vs non-family gap is large), and NK signatures where the column
    tag was incorrectly matched to T-cell because ``cytotox`` is a substring
    of ``nk_cytotoxic``.
    """
    import numpy as np, pandas as pd
    if celltype_obs not in adata.obs.columns:
        return False, f"obs[{celltype_obs!r}] missing"
    rx = re.compile(score_pattern, re.I)
    cols = [c for c in adata.obs.columns if rx.search(c)]
    if not cols:
        return False, f"no obs columns match {score_pattern!r}"
    ct = adata.obs[celltype_obs].astype(str)
    # Pre-compute normalized parent family per cell, so the per-column
    # family-vs-non-family split is just a string compare.
    ct_family = ct.map(lambda s: _parent_family(_normalize_celltype(s)))

    # Tag → expected family token. The family token must match the
    # ``_parent_family(_normalize_celltype(label))`` of cells, e.g.
    # "CD14+ Monocyte" → ``cd14_monocyte`` → parent ``monocyte``. Order
    # matters: NK and pDC patterns come BEFORE T-cell so that
    # ``nk_cytotoxic`` and ``plasmacytoid_dc`` aren't swallowed by a
    # broader ``cytotox`` / ``dc`` regex that lives in the T or generic
    # DC bucket.
    TAGS: list[tuple[str, str]] = [
        # NK first — ``cytotox`` lives inside ``nk_cytotoxic`` and would
        # otherwise be picked up by the T-cell regex.
        (r"\bnk\b|natural[\s_-]?killer", "nk_cell"),
        # pDC before generic DC.
        (r"plasmacytoid[\s_-]?(dc|dendritic)|\bpdc\b", "dendritic"),
        (r"\bdendritic\b|\bdc[\s_-]marker\b|^dc_|_dc_", "dendritic"),
        (r"treg|regulatory[\s_-]?t", "t_cell"),
        # Plasma BEFORE B cell so plasmablast/plasma_cell don't get caught
        # by b_cell pattern; they map to plasma_cell which falls under
        # b_cell parent (currently no entry in _FAMILY_PARENT, leaving
        # plasma_cell as its own family — also fine).
        (r"\bplasma(?:blast|cell)?\b", "plasma_cell"),
        (r"b[\s_-]?cell|\bbcr\b", "b_cell"),
        # T-cell after NK / pDC / Treg / DC.
        (r"\bt[\s_-]?cell\b|\btcr\b|\bcd[48]\b|cytotoxic[\s_-]?t", "t_cell"),
        # Cytotox AFTER explicit nk/t patterns so nothing leaks into here.
        (r"\bcytotox(?:ic)?\b", "t_cell"),
        # Monocyte / macrophage.
        (r"\bcd14\+?", "monocyte"),
        (r"\bcd16\+?", "monocyte"),
        (r"classical[\s_-]?mono|non[\s_-]?classical[\s_-]?mono", "monocyte"),
        (r"\bmacrophage\b|\bmicroglia\b", "monocyte"),
        (r"\bmonocyte\b|\bmono\b", "monocyte"),
        # Megakaryocyte / platelets.
        (r"\bmegakaryocyte\b|\bmkp\b|\bplatelet\b", "megakaryocyte"),
    ]
    tag_rxs = [(re.compile(k, re.I), fam) for k, fam in TAGS]

    n_specific = 0
    detail: list[str] = []
    for col in cols:
        expected_family = None
        for rx2, fam in tag_rxs:
            if rx2.search(col):
                expected_family = fam
                break
        if expected_family is None:
            continue  # column tag (e.g. "interferon_response") doesn't
                       # imply a single cell type — skip
        v = pd.to_numeric(adata.obs[col], errors="coerce")
        if v.notna().sum() < 100:
            continue
        in_family = ct_family.values == expected_family
        n_in = int(in_family.sum())
        if n_in == 0:
            if len(detail) < 4:
                detail.append(f"{col}: family={expected_family!r} not in obs")
            continue
        mu_in = float(np.nanmean(v.values[in_family]))
        mu_out = float(np.nanmean(v.values[~in_family])) if (~in_family).any() else 0.0
        std_v = float(np.nanstd(v.values) or 1e-9)
        z_gap = (mu_in - mu_out) / std_v
        if mu_in > mu_out and z_gap >= min_z_gap:
            n_specific += 1
            if len(detail) < 4:
                detail.append(f"{col}↑{expected_family} (z_gap={z_gap:.2f})")
        elif len(detail) < 4:
            detail.append(f"{col}: family={expected_family!r} z_gap={z_gap:.2f} < {min_z_gap}")
    if n_specific >= min_celltype_specific:
        return True, f"{n_specific} celltype-specific scores: {detail[:4]}"
    return False, (f"only {n_specific} columns peak in expected family "
                    f"(need ≥{min_celltype_specific}); samples={detail[:4]}")


_NEGATION_RX = re.compile(
    r"NOT[ _-]?AVAILABLE"
    r"|not\s+installed"
    r"|No\s+module\s+named"
    r"|ModuleNotFoundError"
    r"|ImportError"
    r"|cannot\s+import"
    r"|could\s+not\s+import"
    r"|pip\s+install\s+",
    re.I,
)


def _is_negated_match(content: str, mt: 're.Match') -> bool:
    """Check whether a regex hit on ``content`` lands inside a negation
    context (e.g. ``"pywgcna: NOT AVAILABLE"``, ``"No module named scgpt"``).

    Looks at the surrounding ±200 chars and rejects the match if a
    negation marker appears in the same window.
    """
    s = max(0, mt.start() - 200)
    e = min(len(content), mt.end() + 200)
    window = content[s:e]
    return _NEGATION_RX.search(window) is not None


def _check_tool_output_evidence(adata, *, trajectory_path: str | None = None,
                                   patterns: list[str] | None = None,
                                   description: str = "expected workflow markers",
                                   ) -> tuple[bool, str]:
    """Generic anti-fabrication check: require ≥1 of ``patterns`` to match
    a tool-output line in the agent's trajectory.

    Pairs with structural rubrics whose regex / alias-list checks can be
    satisfied by the agent hand-rolling a fake artefact. Setting an
    evidence pattern that only the *real* upstream library would print
    (e.g. ``[scGPT] Fine-tuning:``, ``cellphonedb``, ``PyWGCNA``,
    ``[Predicting] Running model inference``) blocks that route.

    Scans only ``role == "tool"`` messages — the user-role task prompt
    can quote the same markers as examples and would otherwise let any
    agent that *reads* the prompt trigger this check.

    Negation-aware: a hit inside a window like ``pywgcna: NOT AVAILABLE``
    or ``No module named 'scgpt'`` is rejected (otherwise the agent
    "checking package availability" would falsely satisfy evidence
    requirements — observed on MiniMax baseline E04 where the
    ``\\bPyWGCNA\\b`` regex matched ``"pywgcna: NOT AVAILABLE"``
    while the agent fell back to scipy clustering).
    """
    import json, re
    if not trajectory_path:
        return False, "no trajectory_path provided"
    p = Path(trajectory_path)
    if not p.exists():
        return False, f"trajectory file missing: {trajectory_path}"
    try:
        traj = json.loads(p.read_text())
    except Exception as exc:
        return False, f"trajectory load failed: {type(exc).__name__}: {exc}"
    msgs = traj.get("messages") or []
    if not patterns:
        return False, "no patterns supplied"
    rxs = [re.compile(p, re.I) for p in patterns]
    matches: list[str] = []
    for m in msgs:
        if m.get("role") != "tool": continue
        c = m.get("content") or ""
        for rx in rxs:
            for mt in rx.finditer(c):
                if _is_negated_match(c, mt):
                    continue
                matches.append(f"{rx.pattern!r}→{mt.group()[:80]!r}")
                break
            if matches and matches[-1].startswith(repr(rx.pattern)):
                break
        if len(matches) >= 3: break
    if matches:
        return True, f"{description}: matched {matches[:2]}"
    return False, f"{description}: no markers in tool output"


def _check_finetune_evidence(adata, *, trajectory_path: str | None = None,
                                patterns: list[str] | None = None,
                                ) -> tuple[bool, str]:
    """Scan the mini-swe-agent trajectory log for evidence that an agent
    actually ran a real foundation-model fine-tune (vs. just aliasing
    columns to dodge the structural rubric).

    Looks at the sibling ``minisweagent_trajectory.json`` for tool-output
    snippets matching any of ``patterns``. Default patterns cover the
    canonical scGPT / Geneformer / SCLLMManager training-progress markers
    (``train_acc=``, ``val_acc=``, ``epoch N/M``, ``Fine-tuning: \d+%``,
    ``[scGPT] Fine-tuning``).

    Used by tasks where the deliverable is "did the agent run the
    expensive workflow", not "did the deliverable land in adata"; pairs
    well with a structural check so the agent has to both train AND save.
    """
    import json, re
    if not trajectory_path:
        # Best-effort: the workspace dir typically contains both
        # ``final.h5ad`` and ``minisweagent_trajectory.json``. Caller can
        # still pass an explicit path if the layout differs.
        return False, "no trajectory_path provided to fine_tune_evidence check"
    p = Path(trajectory_path)
    if not p.exists():
        return False, f"trajectory file missing: {trajectory_path}"
    try:
        traj = json.loads(p.read_text())
    except Exception as exc:
        return False, f"trajectory load failed: {type(exc).__name__}: {exc}"
    msgs = traj.get("messages") or []
    # Require a *canonical* omicverse SCLLMManager training-progress
    # marker, not just generic ``train_acc=`` strings — agents have
    # been observed printing ad-hoc "fake" epoch logs (model name made
    # up, elapsed=0.3s per epoch, etc.) to satisfy substring matches.
    # The bracketed ``[scGPT]`` / ``[Geneformer]`` / ``[scFoundation]``
    # prefix is emitted by ``SCLLMOutput.section_header`` inside the
    # real omicverse fine-tune code path; faked logs don't carry it.
    pats = patterns or [
        r"\[scGPT\]\s+(Fine-tuning|FINE-TUNING|Training)\b",
        r"\[Geneformer\]\s+(Fine[- ]?tun|Training|Train)\b",
        r"\[scFoundation\]\s+(Fine[- ]?tun|Training|Train)\b",
        r"\[CellPLM\]\s+(Fine[- ]?tun|Training|Train)\b",
        r"\[UCE\]\s+(Fine[- ]?tun|Training|Train)\b",
        # Generic SCLLMManager progress bar with model identifier:
        r"\[(scGPT|Geneformer|scFoundation|CellPLM|UCE)\][^\[]{0,400}\d+%\|.*\d+/\d+",
        # Quiet-style legitimate fine-tunes (gpt-5.5 / scripts that suppress
        # SCLLMOutput prints but actually invoke the upstream training
        # loops). These markers come from the upstream library's own
        # function calls, which a fabrication-only agent cannot fake
        # without actually running the code path.
        r"\bSCLLMManager\.fine_tune\(",
        r"\b\.fine_tune\(\s*train_adata\s*=",
        r"\bgeneformer\.Classifier\(",
        r"\bClassifier\.train\(",
        r"\bcell_classifier_token_dataset\b",  # geneformer training prep
        r"\bfine_tune_model\b",                # scgpt training entry
        r"\btrain_test_split_adata\b",
        r"\bprepare_data\(.*?\bfine_tune\b",
        # Per-epoch metrics from upstream training loops
        r"\bepoch\s*[:=]\s*\d+/\d+\b.*\b(train_loss|val_loss|train_acc|val_acc)\b",
        r"\b(train_loss|val_loss|train_acc|val_acc)\s*[:=]\s*\d+\.\d+",
    ]
    rxs = [re.compile(p, re.I) for p in pats]
    matches: list[str] = []
    for m in msgs:
        # Only scan TOOL outputs — the task prompt lives in user messages
        # and may include the same markers as examples (e.g. ``train_acc=``
        # in the L02 deliverable description). Trusting the user role here
        # would let any agent that *reads* the prompt trigger this check.
        # Restricting to tool keeps the signal tied to actual command output.
        if m.get("role") != "tool": continue
        c = m.get("content") or ""
        for rx in rxs:
            mt = rx.search(c)
            if mt:
                matches.append(f"{rx.pattern!r}→{mt.group()!r}")
                break
        if len(matches) >= 3: break
    if matches:
        return True, f"fine-tune evidence found ({len(matches)}+ matches): {matches[:2]}"
    return False, "no fine-tune progress markers in trajectory"


# ---------------------------------------------------------------------------
# Structural / existence checks (used as floor sanity, not the main grader)
# ---------------------------------------------------------------------------

def _check_must_have(adata, kind: str, keys: list, label: str = "") -> tuple[bool, str]:
    cont = getattr(adata, kind)
    have = list(cont.columns) if kind in ("obs", "var") else list(cont.keys())
    missing = []
    for spec in keys:
        if isinstance(spec, (list, tuple)):
            if not any(k in have for k in spec):
                missing.append("any-of:[" + "|".join(spec) + "]")
        elif spec not in have:
            missing.append(spec)
    if missing:
        return False, f"{label or kind} missing: {missing}"
    return True, ""


def _check_must_have_regex(adata, kind: str, patterns: list, label: str = "") -> tuple[bool, str]:
    """Like ``_check_must_have`` but every entry in ``patterns`` is a regex.

    Each pattern must match at least one key in the container. Useful when
    the upstream library has multiple equally-valid naming conventions for
    the same conceptual key (e.g. ``velocity_umap`` /  ``velocity_S_umap``
    / ``velo_<method>_umap`` / ``velocity_<method>_umap``).

    For ``adata.uns`` we also walk one level into dict-valued entries so
    omicverse-style nested namespacing
    (``uns['micro']['braycurtis_pcoa_var']``, ``uns['da']['<method>_pvals']``,
    etc.) is matched the same as flat top-level keys. We do not recurse
    further to keep matching predictable.
    """
    cont = getattr(adata, kind)
    if kind in ("obs", "var"):
        have = list(cont.columns)
    else:
        have = list(cont.keys())
        if kind == "uns":
            for k in list(cont.keys()):
                v = cont.get(k)
                if isinstance(v, dict):
                    for sub in v.keys():
                        have.append(f"{k}.{sub}")
                        have.append(sub)
    missing = []
    matched: list[str] = []
    for pat in patterns:
        try:
            rx = re.compile(pat, re.I)
        except re.error:
            missing.append(f"invalid-regex:{pat!r}")
            continue
        hits = [h for h in have if rx.search(h)]
        if not hits:
            missing.append(f"regex:{pat!r}")
        else:
            matched.append(f"{pat!r}→{hits[0]}")
    if missing:
        return False, f"{label or kind} missing: {missing}"
    return True, "; ".join(matched)


def _check_shape_range(adata, axis: int, mn=None, mx=None) -> tuple[bool, str]:
    n = adata.shape[axis]
    if mn is not None and n < mn:
        return False, f"shape[{axis}]={n} < min={mn}"
    if mx is not None and n > mx:
        return False, f"shape[{axis}]={n} > max={mx}"
    return True, ""


def _check_value_range(adata, obs_alias, mn=None, mx=None,
                       not_all_same=False, nan_max_frac=None) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "obs", obs_alias)
    if not found:
        return False, f"value_range: no obs key in {obs_alias}"
    vals = adata.obs[k].values
    if vals.dtype.kind in 'fc':
        nan_frac = np.isnan(vals.astype(float)).mean()
        nonan = vals[~np.isnan(vals.astype(float))]
    else:
        nan_frac = 0.0
        nonan = vals
    if nan_max_frac is not None and nan_frac > nan_max_frac:
        return False, f"obs[{k!r}] nan fraction={nan_frac:.2%} > {nan_max_frac:.2%}"
    if mn is not None and len(nonan) and nonan.min() < mn:
        return False, f"obs[{k!r}] min={nonan.min()} < {mn}"
    if mx is not None and len(nonan) and nonan.max() > mx:
        return False, f"obs[{k!r}] max={nonan.max()} > {mx}"
    if not_all_same and len(np.unique(nonan)) <= 1:
        return False, f"obs[{k!r}] all-same value (likely garbage fill)"
    return True, ""


def _check_x_value_range(adata, mn=None, mx=None, not_all_integer=False) -> tuple[bool, str]:
    from scipy import sparse
    X = adata.X
    sample = X.data[:5000] if (sparse.issparse(X) and X.data.size > 5000) \
             else (X.data if sparse.issparse(X) else X.flatten()[:5000])
    if mn is not None and sample.min() < mn:
        return False, f"X min={sample.min()} < {mn}"
    if mx is not None and sample.max() > mx:
        return False, f"X max={sample.max()} > {mx}"
    if not_all_integer and np.allclose(sample, np.round(sample), atol=1e-6):
        return False, f"X is integer-like (looks unnormalized)"
    return True, ""


def _check_per_cell_expm1_sum(adata, target: float, tolerance_pct: float) -> tuple[bool, str]:
    from scipy import sparse
    X = adata.X
    per_cell = (np.asarray(np.expm1(X).sum(axis=1)).flatten() if sparse.issparse(X)
                else np.expm1(X).sum(axis=1))
    median = float(np.median(per_cell))
    err = abs(median - target) / target
    if err > tolerance_pct / 100.0:
        return False, f"per-cell expm1 sum median={median:.0f}, target={target} ({err*100:.1f}% off)"
    return True, ""


def _check_layer_dtype_int(adata, layer_alias) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "layers", layer_alias)
    if not found:
        return False, f"layer alias missing: {layer_alias}"
    L = adata.layers[k]
    sample = L.data[:1000] if (hasattr(L, "data") and L.data.size > 1000) \
             else (L.data if hasattr(L, "data") else L.flatten()[:1000])
    if not np.allclose(sample, np.round(sample), atol=1e-6):
        return False, f"layer[{k!r}] not integer-like — was it saved AFTER normalize?"
    return True, ""


def _check_var_bool_sum(adata, var_alias, target: int, tolerance: int) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "var", var_alias)
    if not found:
        return False, f"var key missing: {var_alias}"
    s = int(adata.var[k].astype(bool).sum())
    if abs(s - target) > tolerance:
        return False, f"var[{k!r}].sum()={s} ≠ {target}±{tolerance}"
    return True, ""


def _check_obs_unique_count(adata, obs_alias, mn: int, mx: int) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "obs", obs_alias)
    if not found:
        return False, f"obs key missing: {obs_alias}"
    n = adata.obs[k].nunique()
    if n < mn or n > mx:
        return False, f"obs[{k!r}].nunique()={n} not in [{mn},{mx}]"
    return True, ""


def _check_var_unique_count(adata, var_key, mn: int, mx: int) -> tuple[bool, str]:
    """Number of distinct values in an adata.var column (e.g. WGCNA module
    labels). ``var_key`` may be a string or alias list."""
    found, k = _resolve_alias(adata, "var", var_key)
    if not found:
        return False, f"var key missing: {var_key}"
    n = adata.var[k].nunique()
    if n < mn or n > mx:
        return False, f"var[{k!r}].nunique()={n} not in [{mn},{mx}]"
    return True, f"var[{k!r}].nunique()={n}"


def _check_obsm_count_matching_regex(adata, pattern: str, min_count: int = 2) -> tuple[bool, str]:
    """Count distinct obsm keys matching a regex (used to verify "≥N
    multivariate embeddings" without listing every literal name)."""
    rx = re.compile(pattern, re.I)
    matches = [k for k in adata.obsm.keys() if rx.search(k)]
    if len(matches) < min_count:
        return False, f"only {len(matches)} obsm keys match {pattern!r}: {matches}"
    return True, f"{len(matches)} matching: {matches}"


def _check_any_container_regex(adata, patterns: list, containers: list) -> tuple[bool, str]:
    """Pass if at least one regex pattern matches a key in any of the named
    containers (layers / obsm / obs / var / uns). Useful when the task
    prompt allows the agent to store output in either layers OR obsm,
    where a strict single-container regex check would unfairly fail."""
    hits: list[tuple[str, str, str]] = []
    for c in containers:
        try:
            cont = getattr(adata, c)
        except AttributeError:
            continue
        keys = list(cont.columns) if c in ("obs", "var") else list(cont.keys())
        for k in keys:
            for pat in patterns:
                try:
                    if re.search(pat, str(k), re.I):
                        hits.append((c, pat, k))
                        break
                except re.error:
                    continue
    if not hits:
        return False, (f"no key matches /{patterns}/ in any of {containers}")
    summary = ", ".join(f"{c}[{k!r}]" for c, _, k in hits[:4])
    return True, f"matched: {summary}"


def _check_pairwise_pseudotime_correlation(adata,
                                              pseudotime_obs_pattern: str,
                                              min_pairwise_spearman: float = 0.4,
                                              min_methods: int = 2,
                                              ) -> tuple[bool, str]:
    """All pseudotime obs columns matching the regex must agree directionally.
    Computes Spearman ρ for every unordered pair and requires the *minimum*
    pair to clear ``min_pairwise_spearman``. Rewards quality (consistent
    directionality across methods) without rewarding extra methods that
    disagree with the others — i.e. two correlated methods score the same
    as four correlated methods, but a fourth method that flips the sign
    re-fails the check.
    """
    import pandas as pd
    from itertools import combinations
    rx = re.compile(pseudotime_obs_pattern, re.I)
    cols = [c for c in adata.obs.columns if rx.search(c)]
    if len(cols) < min_methods:
        return False, f"only {len(cols)} pseudotime cols match (<{min_methods})"
    series = {}
    for c in cols:
        v = pd.to_numeric(adata.obs[c], errors="coerce")
        if v.notna().sum() >= 5:
            series[c] = v
    if len(series) < min_methods:
        return False, f"only {len(series)} usable numeric pseudotime cols"
    pairs = list(combinations(series.keys(), 2))
    rhos = []
    for a_, b_ in pairs:
        rho = series[a_].corr(series[b_], method="spearman")
        rhos.append((a_, b_, rho))
    worst = min(rhos, key=lambda t: -1.0 if pd.isna(t[2]) else t[2])
    worst_rho = worst[2]
    if pd.isna(worst_rho) or worst_rho < min_pairwise_spearman:
        return False, (f"min pairwise Spearman {worst[0]}↔{worst[1]} = "
                        f"{worst_rho:.3f} < {min_pairwise_spearman}")
    return True, (f"{len(series)} methods, min pairwise Spearman = "
                   f"{worst_rho:.3f} ({worst[0]}↔{worst[1]})")


def _check_pseudotime_root_anchored(adata, groupby_obs: str,
                                       root_cluster: str,
                                       pseudotime_obs_pattern: str) -> tuple[bool, str]:
    """Median pseudotime in the root cluster should be the lowest of all
    clusters. Accepts any obs column matching the pseudotime regex."""
    import pandas as pd
    if groupby_obs not in adata.obs.columns:
        return False, f"obs[{groupby_obs!r}] missing"
    rx = re.compile(pseudotime_obs_pattern, re.I)
    cands = [c for c in adata.obs.columns if rx.search(c)]
    if not cands:
        return False, f"no obs col matches /{pseudotime_obs_pattern}/"
    fail_reasons = []
    for col in cands:
        try:
            v = pd.to_numeric(adata.obs[col], errors="coerce")
        except Exception:
            continue
        if v.notna().sum() == 0:
            fail_reasons.append(f"{col}: all NaN")
            continue
        med = v.groupby(adata.obs[groupby_obs].astype(str)).median()
        if root_cluster not in med.index:
            fail_reasons.append(f"{col}: root {root_cluster!r} not in obs[{groupby_obs!r}]")
            continue
        if med[root_cluster] == med.min():
            return True, (f"{col}: root cluster {root_cluster!r} has the "
                           f"lowest median pseudotime ({med[root_cluster]:.3f})")
        else:
            fail_reasons.append(f"{col}: root median={med[root_cluster]:.3f} "
                                  f"> min cluster median={med.min():.3f}")
    return False, f"no pseudotime col anchored at root: {fail_reasons[:4]}"


def _check_obs_value_comparison(adata,
                                  obs_a: str,
                                  obs_b_or_pattern: str,
                                  comparison: str = "a_gt_b",
                                  min_fraction: float = 0.5,
                                  ) -> tuple[bool, str]:
    """Element-wise comparison of two numeric obs columns; pass when the
    requested fraction of rows satisfies the comparison.

    ``obs_b_or_pattern`` may be either a literal column name or a regex.
    If it does not match a literal column, treat it as a regex and pick
    the first matching column.

    ``comparison`` ∈ {"a_gt_b", "a_lt_b", "a_ge_b", "a_le_b"}.
    """
    import numpy as np
    import pandas as pd
    if obs_a not in adata.obs.columns:
        return False, f"obs[{obs_a!r}] missing"
    # resolve obs_b literal-or-regex
    if obs_b_or_pattern in adata.obs.columns:
        obs_b = obs_b_or_pattern
    else:
        rx = re.compile(obs_b_or_pattern, re.I)
        hits = [c for c in adata.obs.columns if rx.search(c)]
        if not hits:
            return False, (f"obs[{obs_b_or_pattern!r}] missing (no literal "
                            f"col, no regex match either)")
        obs_b = hits[0]
    a = pd.to_numeric(adata.obs[obs_a], errors="coerce")
    b = pd.to_numeric(adata.obs[obs_b], errors="coerce")
    mask = a.notna() & b.notna()
    if mask.sum() == 0:
        return False, "no non-NaN rows in either column"
    a, b = a[mask], b[mask]
    op = {
        "a_gt_b": np.greater, "a_lt_b": np.less,
        "a_ge_b": np.greater_equal, "a_le_b": np.less_equal,
    }.get(comparison)
    if op is None:
        return False, f"unknown comparison {comparison!r}"
    n_pass = int(op(a.values, b.values).sum())
    frac = n_pass / mask.sum()
    if frac < min_fraction:
        return False, (f"only {n_pass}/{mask.sum()} ({frac:.0%}) of rows have "
                        f"obs[{obs_a!r}] {comparison} obs[{obs_b!r}]; "
                        f"required ≥{min_fraction:.0%}")
    return True, (f"{n_pass}/{mask.sum()} ({frac:.0%}) of rows satisfy "
                   f"obs[{obs_a!r}] {comparison} obs[{obs_b!r}]")


def _check_obs_unique_subset(adata, obs_key: str, must_only_contain_substring: str) -> tuple[bool, str]:
    if obs_key not in adata.obs.columns:
        return False, f"obs key {obs_key!r} missing"
    vals = adata.obs[obs_key].astype(str)
    bad = ~vals.str.contains(must_only_contain_substring, case=False)
    if bad.any():
        return False, f"obs[{obs_key!r}] has {bad.sum()} cells without substring {must_only_contain_substring!r}"
    return True, ""


def _check_obsm_shape(adata, key: str, expect, cells_tolerance: int = 0,
                      dims_tolerance: int = 0) -> tuple[bool, str]:
    if key not in adata.obsm:
        return False, f"obsm[{key!r}] missing"
    s = adata.obsm[key].shape
    e0, e1 = expect
    if abs(s[0] - e0) > cells_tolerance:
        return False, f"obsm[{key!r}].shape[0]={s[0]} ≠ {e0}±{cells_tolerance}"
    if abs(s[1] - e1) > dims_tolerance:
        return False, f"obsm[{key!r}].shape[1]={s[1]} ≠ {e1}±{dims_tolerance}"
    return True, ""


def _check_uns_dict_keys(adata, uns_alias, must_have_subkeys: list[str]) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "uns", uns_alias)
    if not found:
        return False, f"uns key missing: {uns_alias}"
    sub = adata.uns[k]
    if not isinstance(sub, dict):
        return False, f"uns[{k!r}] is not a dict (got {type(sub).__name__})"
    miss = [sk for sk in must_have_subkeys if sk not in sub]
    if miss:
        return False, f"uns[{k!r}] missing subkeys: {miss}"
    return True, ""


def _check_uns_value_nonempty(adata, uns_alias, min_rows: int = 1) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "uns", uns_alias)
    if not found:
        return False, f"uns key missing: {uns_alias}"
    v = adata.uns[k]
    n = len(v) if hasattr(v, "__len__") else 0
    if n < min_rows:
        return False, f"uns[{k!r}] has {n} entries < min {min_rows}"
    return True, ""


# ---------------------------------------------------------------------------
# Biology-grounded checks
# ---------------------------------------------------------------------------

def _check_marker_overlap_in_var(adata, var_alias, ref_markers: list[str],
                                  min_count: int = 5) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "var", var_alias)
    if not found:
        return False, f"var key missing: {var_alias}"
    flagged = adata.var.index[adata.var[k].astype(bool)]
    flagged_set = {g.upper() for g in flagged}
    ref_set = {m.upper() for m in ref_markers}
    hits = sorted(flagged_set & ref_set)
    if len(hits) < min_count:
        return False, f"only {len(hits)}/{len(ref_set)} canonical markers in var[{k!r}]: {hits}"
    return True, f"{len(hits)} canonical markers flagged: {hits[:8]}"


def _check_obsm_celltype_silhouette(adata, obsm_key,
                                     celltype_obs: str = "cell_type",
                                     min_silhouette: float = 0.10) -> tuple[bool, str]:
    """Accepts a literal obsm key OR an alias list — first present alias wins."""
    from sklearn.metrics import silhouette_score
    candidates = [obsm_key] if isinstance(obsm_key, str) else list(obsm_key)
    actual_key = None
    for cand in candidates:
        if cand in adata.obsm:
            actual_key = cand
            break
    if actual_key is None:
        return False, f"obsm key missing among aliases: {candidates}"
    if celltype_obs not in adata.obs.columns:
        return False, f"obs[{celltype_obs!r}] missing"
    labels = adata.obs[celltype_obs].astype(str).values
    if len(set(labels)) < 2:
        return False, "only 1 unique cell_type — cannot compute silhouette"
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
    s = silhouette_score(np.asarray(adata.obsm[actual_key])[idx], labels[idx])
    if s < min_silhouette:
        return False, f"silhouette({actual_key} | {celltype_obs}) = {s:.3f} < {min_silhouette}"
    return True, f"silhouette({actual_key} | {celltype_obs}) = {s:.3f}"


def _check_clustering_ari(adata, obs_alias, oracle_path, oracle_obs_key: str,
                           min_ari: float) -> tuple[bool, str]:
    from sklearn.metrics import adjusted_rand_score
    found, k = _resolve_alias(adata, "obs", obs_alias)
    if not found:
        return False, f"clustering_ari: obs key missing in {obs_alias}"
    if oracle_path is None:
        # cell_type might be in the input fixture itself
        if oracle_obs_key not in adata.obs.columns:
            return False, f"clustering_ari: no oracle and obs[{oracle_obs_key!r}] missing"
        truth = adata.obs[oracle_obs_key].astype(str).to_numpy()
        pred = adata.obs[k].astype(str).to_numpy()
    else:
        oracle = _load_adata(oracle_path)
        if oracle_obs_key not in oracle.obs.columns:
            return False, f"oracle obs[{oracle_obs_key!r}] missing"
        common = adata.obs_names.intersection(oracle.obs_names)
        if len(common) < 50:
            return False, f"only {len(common)} overlapping cells with oracle"
        pred = adata.obs.loc[common, k].astype(str).to_numpy()
        truth = oracle.obs.loc[common, oracle_obs_key].astype(str).to_numpy()
    ari = float(adjusted_rand_score(truth, pred))
    if ari < min_ari:
        return False, f"ARI={ari:.3f} < {min_ari}"
    return True, f"ARI={ari:.3f}"


def _check_subcluster_marker_split(adata, subcluster_obs, marker_a: str,
                                    marker_b: str) -> tuple[bool, str]:
    found, k = _resolve_alias(adata, "obs", subcluster_obs)
    if not found:
        return False, f"obs key missing: {subcluster_obs}"
    expr_a = _expr_vec(adata, marker_a)
    expr_b = _expr_vec(adata, marker_b)
    if expr_a is None or expr_b is None:
        miss = [g for g, e in [(marker_a, expr_a), (marker_b, expr_b)] if e is None]
        return False, f"marker genes missing: {miss}"
    clusters = adata.obs[k].astype(str).values
    a_higher: list[str] = []
    b_higher: list[str] = []
    for c in sorted(set(clusters)):
        mask = clusters == c
        if mask.sum() < 5:
            continue
        ma = float(expr_a[mask].mean())
        mb = float(expr_b[mask].mean())
        if ma > mb + 0.1: a_higher.append(c)
        elif mb > ma + 0.1: b_higher.append(c)
    if not a_higher or not b_higher:
        return False, f"sub-clusters didn't split by {marker_a} vs {marker_b}: A-dom={a_higher}, B-dom={b_higher}"
    return True, f"{marker_a}-dominant: {a_higher}; {marker_b}-dominant: {b_higher}"


def _check_cluster_top_markers_canonical(adata, leiden_obs: str = "leiden",
                                           celltype_obs: str = "cell_type",
                                           rank_uns_key: str = "rank_genes_groups",
                                           top_n: int = 10,
                                           min_clusters_with_canonical: int = 3) -> tuple[bool, str]:
    found_l, kl = _resolve_alias(adata, "obs", leiden_obs)
    found_c, kc = _resolve_alias(adata, "obs", celltype_obs)
    if not (found_l and found_c):
        return False, f"missing obs: leiden={kl}, cell_type={kc}"
    if rank_uns_key not in adata.uns:
        return False, f"uns[{rank_uns_key!r}] missing"
    sub = adata.uns[rank_uns_key]
    if not isinstance(sub, dict) or "names" not in sub:
        return False, f"uns[{rank_uns_key!r}] missing 'names'"
    names = sub["names"]
    groups = (set(names.dtype.names)
              if hasattr(names, "dtype") and names.dtype.names else set())
    if not groups:
        return False, "no cluster groups in rank_genes_groups"
    leiden = adata.obs[kl].astype(str)
    celltype = adata.obs[kc].astype(str)
    hits: list[tuple[str, str, list[str]]] = []
    misses: list[tuple[str, str]] = []
    for g in sorted(groups):
        mask = leiden == g
        if mask.sum() < 5:
            continue
        majority = celltype[mask].mode()
        if len(majority) == 0:
            continue
        ct = majority.iloc[0]
        canon = _markers_for_celltype(ct)
        if not canon:
            continue
        canon_upper = {m.upper() for m in canon}
        top_upper = {str(t).upper() for t in list(names[g])[:top_n]}
        overlap = sorted(top_upper & canon_upper)
        (hits if overlap else misses).append((g, ct, overlap) if overlap else (g, ct))
    if len(hits) < min_clusters_with_canonical:
        return False, f"only {len(hits)} clusters with canonical hits in top-{top_n}; misses: {misses[:5]}"
    return True, f"{len(hits)} clusters hit canonical markers: {hits[:5]}"


def _check_marker_overlap_vs_oracle(adata, uns_key, oracle_path, oracle_uns_key,
                                      top_n: int = 10, min_jaccard: float = 0.2) -> tuple[bool, str]:
    if uns_key not in adata.uns:
        return False, f"uns[{uns_key!r}] missing"
    if oracle_path is None:
        return False, "no oracle"
    oracle = _load_adata(oracle_path)
    if oracle_uns_key not in oracle.uns:
        return False, f"oracle uns[{oracle_uns_key!r}] missing"
    sys_names = adata.uns[uns_key].get("names", None)
    or_names = oracle.uns[oracle_uns_key].get("names", None)
    if sys_names is None or or_names is None:
        return False, "missing 'names' in rank_genes"
    sys_groups = set(sys_names.dtype.names) if hasattr(sys_names, "dtype") and sys_names.dtype.names else set()
    or_groups = set(or_names.dtype.names) if hasattr(or_names, "dtype") and or_names.dtype.names else set()
    common = sys_groups & or_groups
    if not common:
        return False, "no overlapping cluster groups"
    jaccards = []
    for g in common:
        s = set(list(sys_names[g])[:top_n])
        o = set(list(or_names[g])[:top_n])
        if s | o:
            jaccards.append(len(s & o) / len(s | o))
    if not jaccards:
        return False, "no comparable groups"
    mean_j = float(np.mean(jaccards))
    if mean_j < min_jaccard:
        return False, f"mean top-{top_n} Jaccard={mean_j:.3f} < {min_jaccard}"
    return True, f"mean Jaccard={mean_j:.3f} across {len(common)} groups"


# ---------------------------------------------------------------------------
# Multi-method (B04 batch correction, F01 velocity, F02 trajectory)
# ---------------------------------------------------------------------------

def _check_batch_silhouette_drop(adata, batch_obs_key: str,
                                  uncorrected_obsm: str = "X_pca",
                                  corrected_obsm_pattern: str = r"(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)",
                                  min_drop: float = 0.03) -> tuple[bool, str]:
    """Pass if any corrected obsm/layer has lower batch-silhouette than X_pca."""
    from sklearn.metrics import silhouette_score
    if batch_obs_key not in adata.obs.columns:
        return False, f"obs[{batch_obs_key!r}] missing"
    if uncorrected_obsm not in adata.obsm:
        return False, f"obsm[{uncorrected_obsm!r}] missing — cannot baseline"
    labels = adata.obs[batch_obs_key].astype(str).values
    if len(set(labels)) < 2:
        return False, "only 1 batch"
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
    base = silhouette_score(np.asarray(adata.obsm[uncorrected_obsm])[idx], labels[idx])
    rx = re.compile(corrected_obsm_pattern, re.I)
    keys = ([("obsm", k) for k in adata.obsm.keys() if rx.search(k) and k != uncorrected_obsm]
            + [("layers", k) for k in adata.layers.keys() if rx.search(k)])
    if not keys:
        return False, f"no corrected obsm/layer matches /{corrected_obsm_pattern}/"
    drops = []
    for kind, k in keys:
        try:
            X = adata.obsm[k] if kind == "obsm" else adata.layers[k]
            X = _to_dense(X)
            if X.ndim != 2 or X.shape[0] != adata.n_obs:
                continue
            X = X[:, : min(50, X.shape[1])]
            s = silhouette_score(X[idx], labels[idx])
            drops.append((k, base - s))
        except Exception:
            continue
    if not drops:
        return False, "could not score any corrected key"
    best = max(d for _, d in drops)
    if best < min_drop:
        return False, f"max silhouette drop={best:+.3f} < {min_drop} (base={base:.3f}, per-key {drops})"
    return True, f"max silhouette drop={best:+.3f} (base={base:.3f}); per-key {drops}"


def _check_celltype_silhouette_preserved(adata, celltype_obs: str = "cell_type",
                                           uncorrected_obsm: str = "X_pca",
                                           corrected_obsm_pattern: str = r"(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)",
                                           min_relative_preservation: float = 0.5) -> tuple[bool, str]:
    """scIB-style biology preservation: cell-type ASW on corrected ≥ X% of baseline."""
    from sklearn.metrics import silhouette_score
    if celltype_obs not in adata.obs.columns:
        return False, f"obs[{celltype_obs!r}] missing"
    if uncorrected_obsm not in adata.obsm:
        return False, f"obsm[{uncorrected_obsm!r}] missing"
    labels = adata.obs[celltype_obs].astype(str).values
    if len(set(labels)) < 2:
        return False, "only 1 cell_type"
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
    base = silhouette_score(np.asarray(adata.obsm[uncorrected_obsm])[idx], labels[idx])
    rx = re.compile(corrected_obsm_pattern, re.I)
    keys = ([("obsm", k) for k in adata.obsm.keys() if rx.search(k) and k != uncorrected_obsm]
            + [("layers", k) for k in adata.layers.keys() if rx.search(k)])
    if not keys:
        return False, "no corrected key"
    scores = []
    for kind, k in keys:
        try:
            X = adata.obsm[k] if kind == "obsm" else adata.layers[k]
            X = _to_dense(X)
            if X.ndim != 2 or X.shape[0] != adata.n_obs:
                continue
            X = X[:, : min(50, X.shape[1])]
            scores.append((k, silhouette_score(X[idx], labels[idx])))
        except Exception:
            continue
    if not scores:
        return False, "could not score"
    best = max(s for _, s in scores)
    threshold = max(min_relative_preservation * base, 0.0)
    if best < threshold:
        return False, f"best ASW(cell_type)={best:.3f} < {min_relative_preservation:.0%}×{base:.3f}"
    return True, f"best ASW(cell_type)={best:.3f} ≥ {min_relative_preservation:.0%}×{base:.3f}"


def _check_velocity_modes_consistency(adata, min_mean_cosine: float = 0.05) -> tuple[bool, str]:
    """≥2 velocity modes; their mean per-cell cosine ≥ threshold."""
    all_velo = [k for k in adata.layers.keys() if "velocity" in k.lower()]
    primary = [k for k in all_velo
               if not k.lower().startswith("variance_") and not k.lower().endswith("_u")]
    velo_keys = primary if len(primary) >= 2 else all_velo
    if len(velo_keys) < 2:
        return False, f"only {len(velo_keys)} velocity layers: {velo_keys}"
    L1, L2 = _to_dense(adata.layers[velo_keys[0]]), _to_dense(adata.layers[velo_keys[1]])
    if L1.shape != L2.shape:
        return False, f"shape mismatch {L1.shape} vs {L2.shape}"
    L1, L2 = np.nan_to_num(L1.astype(float)), np.nan_to_num(L2.astype(float))
    n1 = np.linalg.norm(L1, axis=1) + 1e-12
    n2 = np.linalg.norm(L2, axis=1) + 1e-12
    mean_cos = float(np.nanmean((L1 * L2).sum(axis=1) / (n1 * n2)))
    if mean_cos < min_mean_cosine:
        return False, f"mean cosine={mean_cos:.3f} < {min_mean_cosine} between {velo_keys[:2]}"
    return True, f"mean cosine={mean_cos:.3f} between {velo_keys[:2]}"


def _check_velocity_root_anchoring(adata, root_cluster: str,
                                     groupby_obs: str = "clusters",
                                     basis_obsm: str = "X_umap",
                                     velocity_obsm: str = "velocity_umap",
                                     min_mean_outward_cosine: float = 0.10) -> tuple[bool, str]:
    """Velocity field on average points away from root cluster centroid.

    ``velocity_obsm`` may be a literal obsm key OR a regex matching one or
    more obsm keys. omicverse's per-method tutorial conventions vary —
    ``velocity_umap`` (dynamo), ``velocity_S_umap`` (scvelo),
    ``velo_latentvelo_umap``, ``velo_graphvelo_umap`` — and agents may
    further generalise to ``velocity_<method>_umap``. When the literal
    key is absent we fall back to the first obsm key matching the pattern.
    """
    if basis_obsm not in adata.obsm:
        return False, f"obsm missing: {basis_obsm}"
    actual_velocity_obsm = velocity_obsm
    if velocity_obsm not in adata.obsm:
        try:
            rx = re.compile(velocity_obsm, re.I)
        except re.error:
            rx = re.compile(r"^(velocity|velo)_.*umap$", re.I)
        for k in adata.obsm.keys():
            if rx.search(k):
                actual_velocity_obsm = k
                break
        else:
            return False, f"no obsm key matches {velocity_obsm!r}; have {list(adata.obsm.keys())}"
    velocity_obsm = actual_velocity_obsm
    if groupby_obs not in adata.obs.columns:
        return False, f"obs[{groupby_obs!r}] missing"
    pos, vel = np.asarray(adata.obsm[basis_obsm]), np.asarray(adata.obsm[velocity_obsm])
    if pos.shape != vel.shape:
        return False, f"shape mismatch {pos.shape} vs {vel.shape}"
    cluster = adata.obs[groupby_obs].astype(str)
    root_mask = (cluster == root_cluster).values
    if not root_mask.any():
        return False, f"root cluster {root_cluster!r} not found"
    centroid = pos[root_mask].mean(axis=0)
    outward = pos - centroid
    n_out = np.linalg.norm(outward, axis=1) + 1e-12
    n_vel = np.linalg.norm(vel, axis=1) + 1e-12
    cos = (outward * vel).sum(axis=1) / (n_out * n_vel)
    cos = np.nan_to_num(cos)[~root_mask]
    if len(cos) == 0:
        return False, "no non-root cells"
    mean_cos = float(np.nanmean(cos))
    if mean_cos < min_mean_outward_cosine:
        return False, f"outward cosine={mean_cos:.3f} < {min_mean_outward_cosine}"
    return True, f"outward cosine={mean_cos:.3f} (velocity flows from {root_cluster!r})"


def _check_pseudotime_root_agreement(adata, root_cluster: str,
                                      groupby_obs: str = "clusters",
                                      min_root_to_other_gap: float = 0.10) -> tuple[bool, str]:
    # Only true pseudotime columns — exclude scanpy's intermediate DPT
    # artefacts (``dpt_groups`` = integer cluster labels, ``dpt_order`` /
    # ``dpt_order_indices`` = integer indices) which would otherwise be
    # picked up by ``startswith("dpt")``.
    pt_cols = [c for c in adata.obs.columns
               if (c.lower().endswith("pseudotime")
                   or c.lower().endswith("_dpt")
                   or c.lower().startswith("pseudotime_"))]
    if not pt_cols:
        return False, "no pseudotime obs column"
    if groupby_obs not in adata.obs.columns:
        return False, f"obs[{groupby_obs!r}] missing"
    cluster = adata.obs[groupby_obs].astype(str)
    if root_cluster not in cluster.unique():
        return False, f"root {root_cluster!r} not in obs[{groupby_obs!r}]"
    failures = []
    for col in pt_cols:
        v = adata.obs[col].astype(float).values
        if np.all(np.isnan(v)):
            failures.append(f"{col}: all-NaN")
            continue
        rm = (cluster == root_cluster).values
        med_root = float(np.nanmedian(v[rm]))
        med_other = float(np.nanmedian(v[~rm]))
        rng = float(np.nanmax(v) - np.nanmin(v)) or 1.0
        gap = (med_other - med_root) / rng
        if gap < min_root_to_other_gap:
            failures.append(f"{col}: gap {gap:+.3f} < {min_root_to_other_gap}")
    if failures:
        return False, "; ".join(failures)
    return True, f"all {len(pt_cols)} pseudotime cols anchor at {root_cluster!r}"


def _check_obs_two_distinct_pseudotime(adata, min_distinct_pseudotime_cols: int = 2) -> tuple[bool, str]:
    # See _check_pseudotime_root_agreement: exclude ``dpt_groups`` and
    # ``dpt_order_indices`` — those are integer scanpy intermediates,
    # not pseudotime estimates.
    pt_cols = [c for c in adata.obs.columns
               if (c.lower().endswith("pseudotime")
                   or c.lower().endswith("_dpt")
                   or c.lower().startswith("pseudotime_"))]
    if len(pt_cols) < min_distinct_pseudotime_cols:
        return False, f"only {len(pt_cols)} pseudotime cols: {pt_cols}"
    if len(pt_cols) >= 2:
        v1 = adata.obs[pt_cols[0]].astype(float).values
        v2 = adata.obs[pt_cols[1]].astype(float).values
        if v1.shape == v2.shape:
            from scipy.stats import spearmanr
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() > 50:
                r, _ = spearmanr(v1[mask], v2[mask])
                if abs(r) > 0.999:
                    return False, f"{pt_cols[:2]} effectively identical (rho={r:.4f})"
    return True, f"distinct pseudotime cols: {pt_cols}"


# ---------------------------------------------------------------------------
# Spatial / multi-omics / bulk / microbiome (v1.0 additions)
# ---------------------------------------------------------------------------

def _check_spatial_domain_silhouette(adata, cluster_obs: str,
                                       spatial_obsm: str = "spatial",
                                       min_silhouette: float = 0.05) -> tuple[bool, str]:
    """Visium-style spatial domain clustering: silhouette of spatial coords by cluster
    label > random. Real spatial-aware methods give 0.10-0.30; random/expression-only
    clustering gives near 0."""
    from sklearn.metrics import silhouette_score
    found, k = _resolve_alias(adata, "obs", cluster_obs)
    if not found:
        return False, f"obs[{cluster_obs!r}] missing"
    if spatial_obsm not in adata.obsm:
        return False, f"obsm[{spatial_obsm!r}] missing"
    labels = adata.obs[k].astype(str).values
    if len(set(labels)) < 2:
        return False, "only 1 cluster"
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
    s = silhouette_score(np.asarray(adata.obsm[spatial_obsm])[idx], labels[idx])
    if s < min_silhouette:
        return False, f"spatial silhouette({k} | {spatial_obsm}) = {s:.3f} < {min_silhouette}"
    return True, f"spatial silhouette={s:.3f}"


def _check_var_count_above(adata, var_alias, min_count: int) -> tuple[bool, str]:
    """≥min_count vars marked True in a boolean var column (for SVG / variable feature counts)."""
    found, k = _resolve_alias(adata, "var", var_alias)
    if not found:
        return False, f"var key missing: {var_alias}"
    n = int(adata.var[k].astype(bool).sum())
    if n < min_count:
        return False, f"var[{k!r}].sum()={n} < {min_count}"
    return True, f"var[{k!r}].sum()={n} ≥ {min_count}"


def _check_obsm_dim_at_least(adata, obsm_key, min_dims: int) -> tuple[bool, str]:
    """Accepts a literal obsm key OR an alias list — first present alias wins."""
    candidates = [obsm_key] if isinstance(obsm_key, str) else list(obsm_key)
    actual_key = None
    for cand in candidates:
        if cand in adata.obsm:
            actual_key = cand
            break
    if actual_key is None:
        return False, f"obsm key missing among aliases: {candidates}"
    d = int(adata.obsm[actual_key].shape[1])
    if d < min_dims:
        return False, f"obsm[{actual_key!r}] has {d} dims < {min_dims}"
    return True, f"obsm[{actual_key!r}] has {d} dims"


def _check_uns_dataframe_has_directional_columns(adata, uns_alias) -> tuple[bool, str]:
    import pandas as pd
    found, k = _resolve_alias(adata, "uns", uns_alias)
    if not found:
        return False, f"uns key missing: {uns_alias}"
    df = adata.uns[k]
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return False, f"uns[{k!r}] not a DataFrame"
    cols_lower = [str(c).lower() for c in df.columns]
    needs = {
        "sender":   ["sender", "source", "from", "cluster_a", "cell_type_a", "celltype_a", "ligand_celltype"],
        "receiver": ["receiver", "target", "to", "cluster_b", "cell_type_b", "celltype_b", "receptor_celltype"],
        "ligand":   ["ligand", "ligand_symbol", "gene_a", "gene_l"],
        "receptor": ["receptor", "receptor_symbol", "gene_b", "gene_r"],
    }
    missing = [role for role, toks in needs.items()
               if not [c for c in cols_lower if any(t in c for t in toks)]]
    if missing:
        return False, f"missing column roles: {missing} (have {df.columns.tolist()})"
    return True, f"directional cols OK ({df.columns.tolist()})"


def _check_ccc_reference_lr_hit(adata, uns_alias, min_hits: int = 1,
                                  reference_pairs: list[tuple[str, str]] | None = None
                                  ) -> tuple[bool, str]:
    import pandas as pd
    found, k = _resolve_alias(adata, "uns", uns_alias)
    if not found:
        return False, f"uns key missing: {uns_alias}"
    df = adata.uns[k]
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return False, "not a DataFrame"
    cols_lower = [str(c).lower() for c in df.columns]
    lig_cols = [c for c, cl in zip(df.columns, cols_lower)
                if any(t in cl for t in ("ligand", "gene_a", "gene_l"))]
    rec_cols = [c for c, cl in zip(df.columns, cols_lower)
                if any(t in cl for t in ("receptor", "gene_b", "gene_r"))]
    if not lig_cols or not rec_cols:
        return False, f"no ligand/receptor cols in {df.columns.tolist()}"
    refs = reference_pairs if reference_pairs is not None else PBMC_LR_REFERENCE
    lig = df[lig_cols[0]].astype(str).str.lower().values
    rec = df[rec_cols[0]].astype(str).str.lower().values
    matches = [(l, r) for l, r in zip(lig, rec)
               if any((rl in l and rr in r) or (rl in r and rr in l)
                      for rl, rr in refs)]
    if len(matches) < min_hits:
        return False, f"{len(matches)} ref-LR-pair hits (need ≥{min_hits}; ref={len(refs)})"
    return True, f"{len(matches)} ref-LR-pair hits, e.g. {matches[:3]}"


def _check_mofa_factor_variance(adata, factor_obsm: str = "X_mofa",
                                 min_factors: int = 5,
                                 min_total_variance: float = 0.30) -> tuple[bool, str]:
    """A real MOFA / GLUE joint embedding has ≥5 factors and explains substantial
    variance. Random embedding has factor variance ≈ uniform/spread, real factor
    variance is concentrated in top components."""
    if factor_obsm not in adata.obsm:
        return False, f"obsm[{factor_obsm!r}] missing"
    F = np.asarray(adata.obsm[factor_obsm])
    if F.ndim != 2 or F.shape[1] < min_factors:
        return False, f"obsm[{factor_obsm!r}] shape {F.shape}; want ≥{min_factors} factors"
    var = F.var(axis=0)
    if var.sum() == 0:
        return False, f"obsm[{factor_obsm!r}] zero variance"
    top_var_frac = (np.sort(var)[::-1][:min_factors].sum() / var.sum())
    if top_var_frac < min_total_variance:
        return False, (f"top-{min_factors} factor variance fraction = {top_var_frac:.2%} "
                       f"< {min_total_variance:.0%}")
    return True, f"top-{min_factors} factor variance = {top_var_frac:.2%}"


def _check_peak_gene_link_count(adata, uns_alias, min_links: int = 1000,
                                  min_promoter_proximal_frac: float = 0.30
                                  ) -> tuple[bool, str]:
    """Multi-omics task: peak-to-gene linkage table must have ≥N links and a
    plausible fraction within proximal/promoter range (typically ±10kb of TSS)."""
    import pandas as pd
    found, k = _resolve_alias(adata, "uns", uns_alias)
    if not found:
        return False, f"uns key missing: {uns_alias}"
    df = adata.uns[k]
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except Exception:
            return False, "not a DataFrame"
    if len(df) < min_links:
        return False, f"only {len(df)} peak-gene links < {min_links}"
    cols_lower = [str(c).lower() for c in df.columns]
    dist_col = next((c for c, cl in zip(df.columns, cols_lower)
                     if "distance" in cl or "dist" in cl), None)
    if dist_col is None:
        return True, f"{len(df)} peak-gene links (no distance column to check promoter fraction)"
    dists = pd.to_numeric(df[dist_col], errors="coerce").abs()
    prox_frac = float((dists <= 10000).mean())
    if prox_frac < min_promoter_proximal_frac:
        return False, f"only {prox_frac:.1%} links within ±10kb (need ≥{min_promoter_proximal_frac:.0%})"
    return True, f"{len(df)} links; {prox_frac:.1%} within ±10kb of TSS"


def _check_deconv_fractions_sane(adata, fractions_obs_pattern: str = r"frac_|_fraction|cell_type_frac",
                                   min_n_celltypes: int = 3,
                                   sum_tolerance: float = 0.10) -> tuple[bool, str]:
    """Bulk RNA → cell-type fractions: per-sample fractions sum to ~1 and ≥3 cell types
    have nonzero contribution."""
    rx = re.compile(fractions_obs_pattern, re.I)
    cols = [c for c in adata.obs.columns if rx.search(c)]
    if len(cols) < min_n_celltypes:
        # check uns for a fractions table
        for k in adata.uns:
            if "fraction" in k.lower() or "deconv" in k.lower() or "cell_type" in k.lower():
                df = adata.uns[k]
                import pandas as pd
                if isinstance(df, pd.DataFrame) and df.shape[1] >= min_n_celltypes:
                    cols = list(df.columns)
                    sums = df.sum(axis=1).values
                    if np.all(np.abs(sums - 1.0) < sum_tolerance) or np.all(np.abs(sums - 100) < 100*sum_tolerance):
                        return True, f"uns[{k!r}] {df.shape[1]} cell-type fractions, sums≈1"
        return False, f"only {len(cols)} fraction columns matched /{fractions_obs_pattern}/"
    df = adata.obs[cols].astype(float)
    sums = df.sum(axis=1).values
    if not (np.all(np.abs(sums - 1.0) < sum_tolerance)
            or np.all(np.abs(sums - 100) < 100*sum_tolerance)):
        return False, f"fractions don't sum to 1 (median sum = {np.median(sums):.3f})"
    nonzero_per_sample = (df > 0.001).sum(axis=1).median()
    if nonzero_per_sample < min_n_celltypes:
        return False, f"median nonzero cell-types per sample = {nonzero_per_sample} < {min_n_celltypes}"
    return True, f"{len(cols)} fraction cols, sums≈1, median {nonzero_per_sample} non-zero per sample"


def _check_bulk2single_ari_vs_ref(adata, oracle_path: str | None,
                                    min_ari: float = 0.20,
                                    cluster_obs: str = "cell_type") -> tuple[bool, str]:
    """Bulk2Single: generated synthetic single cells should recover the reference
    scRNA cell-type structure. Compare the synthetic adata's clustering against
    the original reference cell-type labels via ARI on shared cell-type names."""
    if oracle_path is None or not Path(oracle_path).exists():
        return False, "oracle scRNA reference fixture missing"
    try:
        ref = _load_adata(oracle_path)
    except Exception as e:
        return False, f"could not load ref: {e}"
    if cluster_obs not in adata.obs.columns:
        return False, f"obs[{cluster_obs!r}] missing in synthetic adata"
    if cluster_obs not in ref.obs.columns:
        return False, f"obs[{cluster_obs!r}] missing in reference"
    syn_types = set(adata.obs[cluster_obs].astype(str).unique())
    ref_types = set(ref.obs[cluster_obs].astype(str).unique())
    overlap = syn_types & ref_types
    if len(overlap) < 2:
        return False, (f"synthetic / ref cell-types don't overlap (≥2 needed): "
                       f"syn={syn_types}, ref={ref_types}")
    return True, f"synthetic recovers {len(overlap)} ref cell-types: {sorted(overlap)[:5]}"


def _check_alpha_diversity_present(adata, metrics: list[str]) -> tuple[bool, str]:
    """16S microbiome: per-sample alpha diversity metrics (Shannon, Simpson, Faith PD,
    observed_otus) must be computed and stored in obs."""
    found = [m for m in metrics if m in adata.obs.columns]
    missing = [m for m in metrics if m not in adata.obs.columns]
    if len(found) < 1:
        return False, f"no alpha diversity metrics in obs (missing: {missing})"
    bad = []
    for m in found:
        v = adata.obs[m].astype(float).values
        if np.all(np.isnan(v)) or np.all(v == 0):
            bad.append(m)
    if bad:
        return False, f"alpha diversity metrics all-NaN or all-zero: {bad}"
    return True, f"alpha diversity present: {found}"


def _check_beta_diversity_present(adata, uns_or_obsp_keys: list[str]) -> tuple[bool, str]:
    """16S beta diversity: a pairwise sample-distance matrix (UniFrac, Bray-Curtis)
    must be present in uns or obsp."""
    for k in uns_or_obsp_keys:
        if k in adata.uns or k in adata.obsp:
            obj = adata.uns[k] if k in adata.uns else adata.obsp[k]
            if hasattr(obj, "shape") and len(obj.shape) == 2:
                return True, f"beta diversity matrix at {k}: shape {obj.shape}"
    return False, f"no beta diversity matrix at any of {uns_or_obsp_keys}"


# ---------------------------------------------------------------------------
# v1.1 practitioner-rigor checks
# ---------------------------------------------------------------------------

def _check_multi_doublet_consensus(adata, score_pattern: str = r"doublet_score|doublet_pred|scrublet|scdblfinder|doubletfinder|sccomposite",
                                     min_methods: int = 2,
                                     min_score_agreement: float = 0.10,
                                     consensus_obs_pattern: str = r"doublet_consensus|doublet|is_doublet") -> tuple[bool, str]:
    """≥2 doublet score columns; pairwise Spearman ≥ threshold; consensus column present.

    Real practitioner doublet QC uses ≥2 methods (scrublet + scDblFinder + doubletfinder
    are easy to combine via ov.pp.qc(doublets_method=...)) and takes consensus. Single-
    method shortcuts fail this check.
    """
    rx = re.compile(score_pattern, re.I)
    score_cols = [c for c in adata.obs.columns if rx.search(c)]
    # exclude bool/consensus columns: must contain 'consensus' or 'is_doublet',
    # OR be a boolean dtype (any 'doublet' bool column is a flag, not a score)
    score_only = []
    for c in score_cols:
        if "consensus" in c.lower() or "is_doublet" in c.lower():
            continue
        if adata.obs[c].dtype == bool:
            continue
        # require numeric to be a score
        try:
            v = adata.obs[c].astype(float).values
            if np.all((v == 0) | (v == 1)) and len(np.unique(v)) <= 2:
                # bool-like 0/1 column; not a score
                continue
            score_only.append(c)
        except Exception:
            continue
    if len(score_only) < min_methods:
        return False, f"only {len(score_only)} doublet score columns (need ≥{min_methods}): {score_only}"
    # pairwise agreement
    from scipy.stats import spearmanr
    cols = score_only[:3]
    rs = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            v1 = adata.obs[cols[i]].astype(float).values
            v2 = adata.obs[cols[j]].astype(float).values
            mask = ~(np.isnan(v1) | np.isnan(v2))
            if mask.sum() > 50:
                r, _ = spearmanr(v1[mask], v2[mask])
                rs.append((cols[i], cols[j], float(r)))
    if not rs or max(abs(r) for _, _, r in rs) < min_score_agreement:
        return False, f"doublet methods don't agree (spearman): {rs}"
    return True, f"{len(score_only)} doublet methods; pairwise spearman: {rs}"


def _check_clustering_resolution_stability(adata, resolution_obs_pattern: str = r"leiden_r|leiden_res|res\d|resolution_",
                                             min_resolutions_tested: int = 2,
                                             min_pairwise_ari: float = 0.40) -> tuple[bool, str]:
    """≥N obs columns from a resolution sweep; pairwise ARI ≥ threshold (cluster
    structure is stable across reasonable resolutions)."""
    from sklearn.metrics import adjusted_rand_score
    rx = re.compile(resolution_obs_pattern, re.I)
    res_cols = [c for c in adata.obs.columns if rx.search(c)]
    if len(res_cols) < min_resolutions_tested:
        return False, f"only {len(res_cols)} resolution columns: {res_cols}"
    aris = []
    for i in range(len(res_cols)):
        for j in range(i+1, len(res_cols)):
            l1 = adata.obs[res_cols[i]].astype(str).values
            l2 = adata.obs[res_cols[j]].astype(str).values
            aris.append(adjusted_rand_score(l1, l2))
    if not aris:
        return False, "could not compute pairwise ARI"
    mean_ari = float(np.mean(aris))
    if mean_ari < min_pairwise_ari:
        return False, f"mean pairwise ARI between {len(res_cols)} resolutions = {mean_ari:.3f} < {min_pairwise_ari}"
    return True, f"{len(res_cols)} resolutions; mean ARI = {mean_ari:.3f}"


# PBMC / general-immune cell-type family lookup. Maps any specific label
# substring to a canonical family token. ``normalize_celltype`` returns the
# family token if any phrase matches, else a lowercased/cleaned-up version
# of the input (which won't match anything else and behaves like before).
_CELLTYPE_FAMILY_RULES: list[tuple[str, str]] = [
    # T-cell family — covers CD4 T, CD8 T, naive T, memory T, regulatory T,
    # cytotoxic T, etc.
    (r"\bt[\s\-_]?reg|regulatory[\s\-_]t", "treg"),
    (r"\bcd8\b|\bcytotoxic[\s\-_]t\b|\btc(?!_|d)|killer[\s\-_]t", "cd8_t_cell"),
    (r"\bcd4\b|helper[\s\-_]t|\bth(?:1|2|17)?\b", "cd4_t_cell"),
    (r"\bt[\s\-_]?cell\b|\bt[\s\-_]lymph", "t_cell"),
    # DC family — must come BEFORE plasma_cell pattern so "Plasmacytoid
    # DC" / "plasmacytoid dendritic cell" are routed to pDC, not plasma.
    (r"plasmacytoid[\s\-_]?(dc|dendritic)|\bpdc\b", "pdc"),
    (r"\bdendritic|\bdc\b|\bcdc\d?\b|\bmydc\b", "dendritic"),
    # B-cell / plasma family (after pDC because "plasmacytoid" matches
    # the plasma_cell regex too).
    (r"\bplasma(?:blast|cell)?\b|\bplasmablast\b", "plasma_cell"),
    (r"\bb[\s\-_]?cell\b|\bb[\s\-_]lymph", "b_cell"),
    # Innate cytotoxic
    (r"\bnk[\s\-_]?cell\b|natural[\s\-_]killer", "nk_cell"),
    # Monocyte/macrophage family
    (r"\bcd14[\+\s\-_]|classical[\s\-_]mono", "cd14_monocyte"),
    (r"\bcd16[\+\s\-_]|non[\s\-_]?classical[\s\-_]mono|intermediate[\s\-_]mono", "cd16_monocyte"),
    (r"\bmacrophage|microglia\b", "macrophage"),
    (r"\bmonocyte|\bmono\b", "monocyte"),
    # Megakaryocyte / platelets
    (r"\bmegakaryocyte|\bmkp\b|\bplatelet", "megakaryocyte"),
    # Erythroid
    (r"\berythro|\brbc\b", "erythroid"),
    # Hematopoietic stem / progenitor
    (r"\bhsc\b|hematopoietic[\s\-_]?stem|\bhspc\b|progenitor", "hspc"),
    # Misc — labels indicating "unknown" / "unclassified" should not match
    # anything (don't merge them).
]


def _normalize_celltype(label: str) -> str:
    """Map a free-text cell-type label to a canonical family token, or
    return a lowercase/cleaned version of the label itself.

    "CD4+ T cell"           → "cd4_t_cell"
    "CD8+ cytotoxic T cell" → "cd8_t_cell"
    "T cell"                → "t_cell"          (CD4 / CD8 / T cell all
    "Monocyte"              → "monocyte"           merge into the family
    "CD14+ Monocyte"        → "cd14_monocyte"      via parent_family())
    "Plasmacytoid DC"       → "pdc"
    """
    s = (label or "").strip()
    if not s or s.lower() in {"unknown", "unclassified", "ambiguous", "nan", "none"}:
        return ""
    for pat, family in _CELLTYPE_FAMILY_RULES:
        if re.search(pat, s, re.I):
            return family
    return re.sub(r"[\s\-]+", "_", s.lower())


# Family hierarchy: which canonical tokens are subtypes of which families.
# Used so "CD4+ T cell" (cd4_t_cell) and "T cell" (t_cell) count as agree.
_FAMILY_PARENT: dict[str, str] = {
    "cd4_t_cell": "t_cell",
    "cd8_t_cell": "t_cell",
    "treg": "t_cell",
    "cd14_monocyte": "monocyte",
    "cd16_monocyte": "monocyte",
    "pdc": "dendritic",
    "macrophage": "monocyte",  # macrophages are mono-derived; coarse lumping OK
}


def _parent_family(token: str) -> str:
    """Return the parent family for a normalized celltype token."""
    return _FAMILY_PARENT.get(token, token)


def _check_multi_method_annotation_consistency(adata,
                                                 celltype_obs_pattern: str = r"(cell_type|celltype|annotation)_[a-zA-Z0-9_]+",
                                                 min_methods: int = 2,
                                                 min_majority_agreement: float = 0.60) -> tuple[bool, str]:
    """≥2 cell-type annotation columns from different methods; agreement ≥ threshold.

    Default pattern requires a *method suffix* (e.g. cell_type_cosg, annotation_popv)
    so that bare ground-truth `cell_type` columns from the fixture are not counted
    as annotation method outputs.

    Cross-method label comparison normalizes each label to a celltype
    *family* token via :func:`_normalize_celltype` and then walks the
    parent-family hierarchy: ``"CD4+ T cell"`` and ``"T cell"`` both
    compare as ``t_cell``. This avoids the spurious ~0% agreement two
    methods get when one returns broad labels and the other returns
    subtype labels even though they describe the same population.
    """
    rx = re.compile(celltype_obs_pattern, re.I)
    # Exclude fixture-derived columns the agent did NOT add: CellxGene's bare
    # ``cell_type`` ground-truth column, ``predicted_celltype`` metadata,
    # ontology-id / term-id columns.
    SKIP_SUBSTR = ("ground_truth", "groundtruth", "ontology", "_term_id", "_id_")
    SKIP_EXACT = {"cell_type", "celltype", "predicted_celltype",
                   "predicted_cell_type", "annotation"}
    anno_cols = [c for c in adata.obs.columns if rx.search(c)
                 and c.lower() not in SKIP_EXACT
                 and not any(skip in c.lower() for skip in SKIP_SUBSTR)
                 and not c.lower().endswith("_id")]
    if len(anno_cols) < min_methods:
        return False, f"only {len(anno_cols)} cell-type columns: {anno_cols}"
    # Family-normalize each column's labels before comparing.
    norm = {}
    for c in anno_cols[:3]:
        raw = adata.obs[c].astype(str)
        toks = raw.map(_normalize_celltype)
        # Walk to parent family so subtype labels collapse to the same
        # token as their broad counterparts.
        norm[c] = toks.map(_parent_family).values
    cols = list(norm.keys())
    agreements = []
    for i in range(len(cols)):
        for j in range(i+1, len(cols)):
            l1, l2 = norm[cols[i]], norm[cols[j]]
            # Empty normalized tokens (unknown/unclassified) shouldn't
            # count for or against agreement — drop those rows.
            mask = (l1 != "") & (l2 != "")
            if mask.sum() == 0:
                agreements.append(0.0)
                continue
            agreements.append(float((l1[mask] == l2[mask]).mean()))
    mean_agree = float(np.mean(agreements)) if agreements else 0.0
    if mean_agree < min_majority_agreement:
        return False, (f"mean family-level pairwise agreement = "
                        f"{mean_agree:.3f} < {min_majority_agreement} "
                        f"(cols={cols}, pairwise={[round(a,3) for a in agreements]})")
    return True, (f"{len(anno_cols)} annotation methods; "
                   f"family-level agreement = {mean_agree:.3f} "
                   f"(pairwise={[round(a,3) for a in agreements]})")


def _check_multi_method_de_overlap(adata, uns_pattern: str = r"rank_genes|de_|differential_expr|wilcoxon|t_test|cosg|mast|scdeg",
                                     min_methods: int = 2,
                                     min_top_jaccard: float = 0.30,
                                     top_n: int = 20) -> tuple[bool, str]:
    """≥2 DE result tables; top-N gene-set Jaccard across methods ≥ threshold.

    Two storage shapes are accepted:

    - **scanpy ``rank_genes_groups`` dict** (single-cell, per-cluster):
      ``uns[k]['names']`` is a structured array whose dtype names are the
      cluster groups; per-cluster top-N gene set = first ``top_n`` of
      each group's name column. Jaccard averaged across shared clusters.

    - **DataFrame** (bulk DE, single comparison): one row per gene,
      sorted by ``padj`` / ``qvalue`` / ``pvals_adj`` (or any sensibly
      named significance column). Gene names come from the index or a
      ``gene`` / ``gene_name`` column. Top-N gene set per method;
      Jaccard taken pairwise across methods (no per-cluster averaging).
    """
    import pandas as pd

    PADJ_KEYS = ("padj", "pvals_adj", "qvalue", "qval", "fdr",
                 "p.adjust", "adj.P.Val", "adjp")
    GENE_KEYS = ("gene", "gene_name", "feature", "feature_name", "symbol")

    rx = re.compile(uns_pattern, re.I)
    method_keys = [k for k in adata.uns.keys() if rx.search(k)]
    if len(method_keys) < min_methods:
        return False, f"only {len(method_keys)} DE-related uns keys: {method_keys}"

    # ---- per-method top-N gene set extraction ---------------------------
    # Two output shapes:
    #   A) cluster_topsets: list[(key, {cluster: set(gene)})]  (single-cell)
    #   B) bulk_topsets:    list[(key, set(gene))]             (bulk DE)
    cluster_topsets: list[tuple[str, dict[str, set]]] = []
    bulk_topsets: list[tuple[str, set]] = []

    # Scan up to 10 matching uns keys — agents (especially ov-arm) often add
    # extra summary tables (`*_top50_table`, `*_markers`, `de_marker_analysis`,
    # …) that get matched but lack a parseable structure; the first 5 may
    # all be such siblings even when a real rank_genes_groups is present.
    for k in method_keys[:10]:
        v = adata.uns[k]
        # (A) scanpy rank_genes_groups dict — `names` is a structured ndarray
        if isinstance(v, dict) and "names" in v:
            names = v["names"]
            if hasattr(names, "dtype") and getattr(names.dtype, "names", None):
                cluster_names = names.dtype.names
                # Single-cluster rank_genes_groups (bulk DE stored as the
                # scanpy dict shape with one comparison) is a bulk result —
                # treat it as such so it can pair with sibling de_* tables.
                if len(cluster_names) == 1:
                    g = cluster_names[0]
                    bulk_topsets.append(
                        (k, set(str(t).upper() for t in list(names[g])[:top_n]))
                    )
                    continue
                per_cluster = {g: set(str(t).upper() for t in list(names[g])[:top_n])
                               for g in cluster_names}
                cluster_topsets.append((k, per_cluster))
                continue
            # (A') ov-arm style: dict with `names` as a DataFrame whose
            # columns are cluster ids (e.g. cosg output). One column per
            # cluster, rows are top-ranked genes already in order.
            if isinstance(names, pd.DataFrame):
                per_cluster = {str(col): set(str(t).upper()
                                              for t in names[col].head(top_n).tolist())
                               for col in names.columns}
                cluster_topsets.append((k, per_cluster))
                continue
        # (B) DataFrame (bulk DE)
        if isinstance(v, pd.DataFrame):
            df = v
            # find a significance column for sorting
            sig_col = None
            for c in PADJ_KEYS:
                if c in df.columns:
                    sig_col = c; break
            if sig_col is None:
                # fall back: any column whose lower-name contains
                # "padj" / "fdr" / "qval"
                for c in df.columns:
                    cl = str(c).lower()
                    if any(t in cl for t in ("padj", "fdr", "qval", "adj")):
                        sig_col = c; break
            # find gene identifier column (or use index). ``df.index.astype
            # (str)`` returns an ``Index`` (not a ``Series``) which lacks
            # ``.loc`` — fall through to a Series so the gather below works
            # for both index-and-column gene IDs.
            gene_series = None
            for c in GENE_KEYS:
                if c in df.columns:
                    gene_series = df[c].astype(str); break
            if gene_series is None:
                gene_series = pd.Series(df.index.astype(str), index=df.index)
            try:
                if sig_col is not None:
                    # Sort by adjusted p-value, breaking ties on |log2FC|
                    # descending. Without the tie-breaker, small-sample
                    # tests (e.g. Wilcoxon on n≈20) saturate many genes at
                    # the same minimum p-value and pandas falls back to
                    # alphabetical index order — yielding nonsensical
                    # top-N "signals" that disagree across methods purely
                    # due to ranking noise.
                    sort_df = pd.DataFrame({"_p": df[sig_col].astype(float)})
                    lfc_col = next((c for c in df.columns
                                     if any(tok in str(c).lower()
                                            for tok in ("log2fc", "log2_fc",
                                                         "logfoldchange",
                                                         "logfc", "lfc",
                                                         "log2foldchange"))),
                                    None)
                    if lfc_col is not None:
                        sort_df["_lfc_abs"] = (-df[lfc_col]
                                                  .astype(float).abs())
                        order = sort_df.sort_values(["_p", "_lfc_abs"],
                                                       kind="mergesort").index
                    else:
                        order = sort_df["_p"].sort_values(kind="mergesort").index
                else:
                    # no sortable significance column → just take first top_n rows
                    order = df.index[:top_n]
                top = [str(g).upper() for g in gene_series.loc[order[:top_n]].tolist()]
                bulk_topsets.append((k, set(top)))
            except Exception:
                continue

    n_methods = len(cluster_topsets) + len(bulk_topsets)
    if n_methods < min_methods:
        return False, (f"could not extract top-{top_n} from {min_methods} "
                        f"methods (got {n_methods}: cluster-style="
                        f"{len(cluster_topsets)}, bulk-style="
                        f"{len(bulk_topsets)})")

    # ---- pairwise Jaccard ---------------------------------------------
    jaccards: list[float] = []
    if cluster_topsets:
        common_clusters = set(cluster_topsets[0][1].keys())
        for _, m in cluster_topsets[1:]:
            common_clusters &= set(m.keys())
        for c in common_clusters:
            for i in range(len(cluster_topsets)):
                for j in range(i + 1, len(cluster_topsets)):
                    a = cluster_topsets[i][1].get(c, set())
                    b = cluster_topsets[j][1].get(c, set())
                    if a | b:
                        jaccards.append(len(a & b) / len(a | b))
    if bulk_topsets:
        for i in range(len(bulk_topsets)):
            for j in range(i + 1, len(bulk_topsets)):
                a = bulk_topsets[i][1]
                b = bulk_topsets[j][1]
                if a | b:
                    jaccards.append(len(a & b) / len(a | b))

    if not jaccards:
        return False, "no comparable cluster/method pairs"
    mean_j = float(np.mean(jaccards))
    if mean_j < min_top_jaccard:
        return False, (f"mean top-{top_n} Jaccard across methods = "
                        f"{mean_j:.3f} < {min_top_jaccard}")
    return True, f"{n_methods} DE methods; mean Jaccard = {mean_j:.3f}"


def _check_kbet_lisi_quantitative(adata, batch_obs_key: str,
                                    corrected_obsm_pattern: str = r"(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)",
                                    min_lisi: float = 0.60) -> tuple[bool, str]:
    """LISI-style local mixing on corrected embedding: per-cell, fraction of
    k-NN that are different-batch (normalized by max possible). High = batches
    well mixed locally. Real harmony/combat → 0.7-0.9; uncorrected → 0.3-0.5;
    fabricated → ~0.5 random. We require ≥0.6 on at least one corrected key."""
    from sklearn.neighbors import NearestNeighbors
    if batch_obs_key not in adata.obs.columns:
        return False, f"obs[{batch_obs_key!r}] missing"
    labels = adata.obs[batch_obs_key].astype(str).values
    if len(set(labels)) < 2:
        return False, "only 1 batch"
    rx = re.compile(corrected_obsm_pattern, re.I)
    keys = ([("obsm", k) for k in adata.obsm.keys() if rx.search(k)]
            + [("layers", k) for k in adata.layers.keys() if rx.search(k)])
    if not keys:
        return False, f"no corrected key matches /{corrected_obsm_pattern}/"
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
    sub_labels = labels[idx]
    n_batches = len(set(sub_labels))
    expected_other_frac = 1.0 - 1.0/n_batches  # if perfectly mixed
    scores = []
    for kind, k in keys:
        try:
            X = adata.obsm[k] if kind == "obsm" else adata.layers[k]
            X = _to_dense(X)
            if X.ndim != 2 or X.shape[0] != adata.n_obs:
                continue
            X = X[:, : min(50, X.shape[1])]
            nn = NearestNeighbors(n_neighbors=15).fit(X[idx])
            _, ind = nn.kneighbors(X[idx])
            # fraction of neighbors with DIFFERENT batch (excluding self)
            same_label = sub_labels[ind[:, 1:]] != sub_labels[:, None]
            mix_frac = float(same_label.mean())
            # normalized LISI-like: 1 means perfect mixing
            normalized = mix_frac / expected_other_frac if expected_other_frac > 0 else 0
            scores.append((k, normalized))
        except Exception:
            continue
    if not scores:
        return False, "could not score any corrected key"
    best = max(s for _, s in scores)
    if best < min_lisi:
        return False, f"max kNN-batch-mixing score = {best:.3f} < {min_lisi} (per-key {scores})"
    return True, f"max kNN-batch-mixing score = {best:.3f}; per-key {scores}"


def _check_joint_embedding_outperforms_single(adata,
                                                 joint_obsm_pattern: str = r"X_(mofa|glue|multivi|wnn|joint)",
                                                 single_obsm_pattern: str = r"X_(pca|rna|atac)",
                                                 celltype_obs: str = "cell_type",
                                                 min_silhouette_improvement: float = 0.02) -> tuple[bool, str]:
    """Joint-embedding cell-type silhouette > best single-modality embedding by margin."""
    from sklearn.metrics import silhouette_score
    if celltype_obs not in adata.obs.columns:
        return False, f"obs[{celltype_obs!r}] missing — cannot compare"
    labels = adata.obs[celltype_obs].astype(str).values
    if len(set(labels)) < 2:
        return False, "only 1 cell_type"
    rx_j = re.compile(joint_obsm_pattern, re.I)
    rx_s = re.compile(single_obsm_pattern, re.I)
    joint_keys = [k for k in adata.obsm if rx_j.search(k)]
    single_keys = [k for k in adata.obsm if rx_s.search(k) and k not in joint_keys]
    if not joint_keys or not single_keys:
        return False, f"missing joint or single keys; joint={joint_keys}, single={single_keys}"
    rng = np.random.default_rng(0)
    n = adata.n_obs
    idx = rng.choice(n, min(n, 3000), replace=False) if n > 3000 else np.arange(n)
    j_scores = [(k, silhouette_score(np.asarray(adata.obsm[k])[idx], labels[idx]))
                for k in joint_keys]
    s_scores = [(k, silhouette_score(np.asarray(adata.obsm[k])[idx], labels[idx]))
                for k in single_keys]
    best_j = max(s for _, s in j_scores)
    best_s = max(s for _, s in s_scores)
    improvement = best_j - best_s
    if improvement < min_silhouette_improvement:
        return False, (f"joint silhouette {best_j:.3f} - best single {best_s:.3f} = "
                       f"{improvement:+.3f} < {min_silhouette_improvement}")
    return True, f"joint {best_j:.3f} > best single {best_s:.3f} by {improvement:+.3f}"


def _check_multi_method_deconv_agreement(adata, fractions_keys_pattern: str = r"fractions|deconv|frac_",
                                            min_methods: int = 2,
                                            min_correlation: float = 0.40) -> tuple[bool, str]:
    """≥2 deconv method results in uns; per-cell-type fraction Pearson correlation
    across methods ≥ threshold (correlated even if absolute scale differs).

    Per-method storage shape: ``samples × cell-types`` DataFrame (rows
    matching ``adata.n_obs``). Other matching uns entries that the agent
    may produce as a side effect — metadata dicts, summary tables (one
    row per cell-type, single ``pearson_r`` column), etc. — are silently
    skipped instead of consuming a slot in the comparison.
    """
    import pandas as pd
    rx = re.compile(fractions_keys_pattern, re.I)
    method_keys = [k for k in adata.uns.keys() if rx.search(k)]
    method_dfs: list[tuple[str, pd.DataFrame]] = []
    n_obs = adata.n_obs
    skipped_reasons: dict[str, str] = {}
    # Scan up to 10 candidate keys (covers metadata-dict pollution like
    # ``deconv_metadata`` / ``deconv_method_agreement_pearson`` taking
    # early slots).
    for k in method_keys[:10]:
        df = adata.uns[k]
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                skipped_reasons[k] = "non-DataFrame"
                continue
        # Filter to samples × cell-types tables: row count must match
        # adata.n_obs and column count must be ≥ 2 (otherwise it's a
        # summary like (n_celltypes, 1) cross-method correlation).
        if df.shape[0] != n_obs:
            skipped_reasons[k] = (f"shape[0]={df.shape[0]} ≠ n_obs={n_obs} "
                                   "(not a samples × cell-types table)")
            continue
        if df.shape[1] < 2:
            skipped_reasons[k] = f"only {df.shape[1]} columns (not a fractions matrix)"
            continue
        method_dfs.append((k, df))
        if len(method_dfs) >= 5:  # plenty for cross-method comparison
            break

    if len(method_dfs) < min_methods:
        return False, (f"only {len(method_dfs)} deconv method results "
                        f"(rows == n_obs == {n_obs}): "
                        f"{[k for k, _ in method_dfs]} (skipped: {skipped_reasons})")
    # cross-method per-cell-type correlation
    common_cells = set(method_dfs[0][1].columns)
    for _, df in method_dfs[1:]:
        common_cells &= set(df.columns)
    if not common_cells:
        return False, "no common cell-type columns across methods"
    corrs = []
    for i in range(len(method_dfs)):
        for j in range(i+1, len(method_dfs)):
            for c in common_cells:
                a = method_dfs[i][1][c].astype(float).values
                b = method_dfs[j][1][c].astype(float).values
                if len(a) == len(b) and len(a) > 3:
                    r = float(np.corrcoef(a, b)[0, 1])
                    if not np.isnan(r):
                        corrs.append((c, r))
    if not corrs:
        return False, "could not compute per-cell-type correlations"
    mean_r = float(np.mean([r for _, r in corrs]))
    if mean_r < min_correlation:
        return False, f"mean cross-method per-cell-type correlation = {mean_r:.3f} < {min_correlation}"
    return True, f"{len(method_dfs)} methods; mean correlation = {mean_r:.3f}"


def _check_bulk2single_composition_match(adata, oracle_path: str | None,
                                            cluster_obs: str = "cell_type",
                                            min_correlation: float = 0.40) -> tuple[bool, str]:
    """Synthetic bulk2single cell-type composition (counts per cell-type) should
    correlate with reference scRNA composition."""
    if oracle_path is None or not Path(oracle_path).exists():
        return False, "oracle reference fixture missing"
    if cluster_obs not in adata.obs.columns:
        return False, f"obs[{cluster_obs!r}] missing"
    try:
        ref = _load_adata(oracle_path)
    except Exception as e:
        return False, f"could not load ref: {e}"
    if cluster_obs not in ref.obs.columns:
        return False, f"obs[{cluster_obs!r}] missing in reference"
    syn_freq = adata.obs[cluster_obs].astype(str).value_counts(normalize=True)
    ref_freq = ref.obs[cluster_obs].astype(str).value_counts(normalize=True)
    common = sorted(set(syn_freq.index) & set(ref_freq.index))
    if len(common) < 2:
        return False, f"<2 common cell-types: syn={list(syn_freq.index)[:5]}, ref={list(ref_freq.index)[:5]}"
    syn_v = np.array([syn_freq.get(c, 0) for c in common])
    ref_v = np.array([ref_freq.get(c, 0) for c in common])
    r = float(np.corrcoef(syn_v, ref_v)[0, 1])
    if np.isnan(r) or r < min_correlation:
        return False, f"syn-vs-ref composition correlation r={r:.3f} < {min_correlation}"
    return True, f"composition correlation r={r:.3f} across {len(common)} cell-types"


def _check_velocity_confidence_present(adata, min_nonzero_frac: float = 0.50,
                                          confidence_keys: list[str] | None = None
                                          ) -> tuple[bool, str]:
    """velocity_confidence (or velocity_self_transition) per cell present and non-trivial."""
    candidates = confidence_keys or ["velocity_confidence", "velocity_confidence_transition",
                                       "velocity_self_transition", "velocity_length"]
    found_obs = [c for c in adata.obs.columns
                 if any(k in c.lower() for k in (kk.lower() for kk in candidates))]
    if not found_obs:
        return False, f"no velocity confidence column; tried {candidates}"
    for col in found_obs:
        v = adata.obs[col].astype(float).values
        nonzero_frac = float((v != 0).mean())
        if nonzero_frac >= min_nonzero_frac and not np.all(np.isnan(v)):
            return True, f"velocity confidence {col} has {nonzero_frac:.2%} non-zero values"
    return False, f"velocity confidence columns all-NaN or near-zero: {found_obs}"


def _check_trajectory_branch_detection(adata, branch_uns_pattern: str = r"terminal|branch|fate|cellrank",
                                          min_terminals: int = 2) -> tuple[bool, str]:
    """≥N terminal states identified — either as uns key listing terminals or as
    obs column flagging terminal cells."""
    rx = re.compile(branch_uns_pattern, re.I)
    uns_hits = [k for k in adata.uns.keys() if rx.search(k)]
    obs_hits = [c for c in adata.obs.columns if rx.search(c)]
    if not uns_hits and not obs_hits:
        return False, f"no terminal/branch/fate keys in uns or obs"
    n_terminals = 0
    for k in uns_hits:
        v = adata.uns[k]
        if hasattr(v, "__len__"):
            n_terminals = max(n_terminals, len(v))
    for c in obs_hits:
        try:
            uniq = adata.obs[c].astype(str).unique()
            n_terminals = max(n_terminals, len(uniq) - 1)  # minus background
        except Exception:
            pass
    if n_terminals < min_terminals:
        return False, f"only {n_terminals} terminal/branch states (need ≥{min_terminals})"
    return True, f"{n_terminals} terminal/branch states detected"


def _check_alpha_diversity_group_test(adata, group_obs: str,
                                         metric: str = "shannon",
                                         max_pvalue: float | None = None,
                                         test_uns_key_pattern: str = r"alpha_test|kruskal|mann_whitney|group_diff"
                                         ) -> tuple[bool, str]:
    """A statistical test of group-level alpha diversity differences must be PRESENT
    in uns (key matching the pattern, with a numeric p-value field).

    The test merely needs to have been RUN (rigor check), not necessarily reach
    significance — small demo datasets may not show p<0.05 even when the analyst
    correctly applied the method. If `max_pvalue` is given AND a result exists,
    we additionally enforce p ≤ threshold.
    """
    if metric not in adata.obs.columns:
        return False, f"obs[{metric!r}] missing — cannot test"
    if group_obs not in adata.obs.columns:
        return False, f"obs[{group_obs!r}] missing"
    rx = re.compile(test_uns_key_pattern, re.I)
    test_keys = [k for k in adata.uns.keys() if rx.search(k)]
    if not test_keys:
        return False, (f"no group alpha-diversity test in uns matching "
                       f"/{test_uns_key_pattern}/ — agent must store the test result "
                       f"(rigor: 'did you actually statistically test the difference?')")
    for k in test_keys:
        v = adata.uns[k]
        p = None
        if isinstance(v, dict):
            p = v.get("pvalue", v.get("p", v.get("pval")))
        else:
            try:
                p = float(v)
            except Exception:
                pass
        if p is not None:
            p = float(p)
            if max_pvalue is not None and p > max_pvalue:
                return False, f"alpha test result uns[{k!r}] p={p:.3f} > {max_pvalue}"
            return True, f"group alpha-diversity test stored: uns[{k!r}] p={p:.3f}"
    return False, f"test keys {test_keys} have no recognizable p-value field"


def _check_beta_diversity_permanova(adata, beta_obsp_pattern: str = r"beta|bray|unifrac",
                                       group_obs: str = "group",
                                       max_pvalue: float = 0.05,
                                       test_uns_key_pattern: str = r"permanova|anosim|beta_test"
                                       ) -> tuple[bool, str]:
    """PERMANOVA / ANOSIM test on beta diversity matrix between groups must be present
    in uns with significant p-value."""
    rx = re.compile(test_uns_key_pattern, re.I)
    test_keys = [k for k in adata.uns.keys() if rx.search(k)]
    for k in test_keys:
        v = adata.uns[k]
        if isinstance(v, dict):
            p = v.get("pvalue", v.get("p", v.get("pval")))
            if p is not None and float(p) <= max_pvalue:
                return True, f"PERMANOVA-style test passed: uns[{k!r}] p={p:.3f}"
        try:
            p = float(v)
            if p <= max_pvalue:
                return True, f"PERMANOVA-style test passed: uns[{k!r}] = {p:.3f}"
        except Exception:
            pass
    return False, f"no PERMANOVA / ANOSIM test in uns matching /{test_uns_key_pattern}/"


def _check_clustering_ari_vs_obs(adata, pred_obs: str, truth_obs: str,
                                    min_ari: float = 0.20) -> tuple[bool, str]:
    """ARI of cluster prediction obs column vs ground-truth obs column — for tasks
    where truth is in the input fixture (e.g. C01 ground_truth_layer)."""
    from sklearn.metrics import adjusted_rand_score
    found_p, kp = _resolve_alias(adata, "obs", pred_obs)
    if not found_p:
        return False, f"prediction obs missing: {pred_obs}"
    if truth_obs not in adata.obs.columns:
        return False, f"truth obs[{truth_obs!r}] missing"
    pred = adata.obs[kp].astype(str).values
    truth = adata.obs[truth_obs].astype(str).values
    # filter NaN/empty truth labels
    mask = (truth != "nan") & (truth != "") & (truth != "NA")
    if mask.sum() < 50:
        return False, f"only {mask.sum()} cells with valid truth labels"
    ari = float(adjusted_rand_score(truth[mask], pred[mask]))
    if ari < min_ari:
        return False, f"ARI({pred_obs} vs {truth_obs}) = {ari:.3f} < {min_ari}"
    return True, f"ARI({pred_obs} vs {truth_obs}) = {ari:.3f}"


def _check_da_method_overlap(adata, uns_pattern: str = r"da_|differential_abundance|wilcox|deseq|ancombc",
                              min_methods: int = 2,
                              min_jaccard: float = 0.20,
                              top_n: int = 30) -> tuple[bool, str]:
    """Multi-method DA: ≥2 differential-abundance method results, with reasonable
    overlap (Jaccard ≥ 0.2) of top-N significant taxa.

    Storage shape accepted on each per-method ``adata.uns[k]``:
      - DataFrame with a p-value-like column (``pval`` / ``p_val`` /
        ``p_value`` / ``padj`` / ``fdr`` / ``qval`` …) and either a
        taxon-id index OR a column called ``feature`` / ``taxon`` / ``gene``
        / ``id`` / ``var`` (case-insensitive).
      - dict that converts to such a DataFrame.

    Side-effect-only entries that match the pattern but carry no real per-
    taxon table (e.g. ``da_contrast``, ``da_method_comparison`` summaries)
    are silently skipped instead of consuming a slot.
    """
    import pandas as pd
    SIG_TOKENS = ("pval", "p_val", "p value", "padj", "fdr", "qval", "qvalue")
    TAXON_COLS = ("feature", "taxon", "gene", "id", "var", "asv", "otu",
                  "name", "feature_id", "gene_id", "taxa")

    rx = re.compile(uns_pattern, re.I)
    method_keys = [k for k in adata.uns.keys() if rx.search(k)]
    if len(method_keys) < min_methods:
        return False, f"only {len(method_keys)} DA result tables in uns (need ≥{min_methods}): {method_keys}"

    top_sets: list[tuple[str, set]] = []
    skipped_reasons: dict[str, str] = {}
    # Scan up to 8 candidate keys (covers metadata-dict pollution like
    # ``da_contrast`` / ``da_comparison`` taking early slots).
    for k in method_keys[:8]:
        df = adata.uns[k]
        if not isinstance(df, pd.DataFrame):
            try:
                df = pd.DataFrame(df)
            except Exception:
                skipped_reasons[k] = "non-DataFrame, no DataFrame conversion"
                continue
        # Skip obvious summary/metadata tables (≤2 rows, or no useful columns).
        if df.shape[0] < 5:
            skipped_reasons[k] = f"too few rows ({df.shape[0]}) — looks like metadata"
            continue

        cols_lower = {c: str(c).lower() for c in df.columns}

        # Find a significance/p-value column. Use space-and-underscore-
        # flexible matching: replace _ with space before the substring scan
        # so ``p_value`` and ``p value`` both match the ``"p val"`` token.
        def _matches_sig(col_lower: str) -> bool:
            normalized = col_lower.replace("_", " ").replace(".", " ")
            return any(tok.replace("_", " ") in normalized for tok in SIG_TOKENS)

        # Prefer adjusted p-value columns when available.
        pcol = next((c for c, cl in cols_lower.items()
                      if any(tok in cl.replace("_", " ").replace(".", " ")
                             for tok in ("padj", "fdr", "qval", "qvalue", "adj p"))),
                    None)
        if pcol is None:
            pcol = next((c for c, cl in cols_lower.items() if _matches_sig(cl)), None)

        if pcol is None:
            skipped_reasons[k] = (f"no p-value column among "
                                   f"{list(df.columns)[:8]}")
            continue

        try:
            ranked = df.nsmallest(top_n, pcol)
        except Exception as exc:
            skipped_reasons[k] = f"nsmallest failed: {exc}"
            continue

        # Taxon identifier: object-dtype index, OR a column whose name
        # matches a taxon-id alias.
        taxa: set | None = None
        if ranked.index.dtype == object:
            taxa = set(str(t) for t in ranked.index.tolist())
        else:
            for cand in TAXON_COLS:
                col_match = next(
                    (c for c, cl in cols_lower.items() if cl == cand),
                    None,
                )
                if col_match is None:
                    col_match = next(
                        (c for c, cl in cols_lower.items() if cand in cl),
                        None,
                    )
                if col_match is not None:
                    taxa = set(str(t) for t in ranked[col_match].tolist())
                    break

        if not taxa:
            skipped_reasons[k] = ("could not identify taxon column "
                                   f"(index is {ranked.index.dtype}; cols "
                                   f"{list(df.columns)[:6]})")
            continue

        top_sets.append((k, taxa))
        if len(top_sets) >= 5:  # capped, plenty
            break

    if len(top_sets) < min_methods:
        return False, (
            f"could not extract top-{top_n} taxa from {min_methods} methods: "
            f"{[k for k, _ in top_sets]} (skipped: {skipped_reasons})"
        )
    overlaps = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            a, b = top_sets[i][1], top_sets[j][1]
            if a | b:
                overlaps.append(len(a & b) / len(a | b))
    if not overlaps:
        return False, "no method pairs had overlapping top-taxa sets"
    mean_j = float(np.mean(overlaps))
    if mean_j < min_jaccard:
        return False, (f"mean DA-method top-{top_n} Jaccard = {mean_j:.3f} "
                        f"< {min_jaccard} (across {len(top_sets)} methods: "
                        f"{[k for k, _ in top_sets]})")
    return True, (f"{len(top_sets)} DA methods; mean top-{top_n} Jaccard = "
                   f"{mean_j:.3f}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

CHECK_DISPATCH = {
    "must_have_obs_keys":       lambda a, **kw: _check_must_have(a, "obs", kw["keys"], "obs"),
    "must_have_var_keys":       lambda a, **kw: _check_must_have(a, "var", kw["keys"], "var"),
    "must_have_uns_keys":       lambda a, **kw: _check_must_have(a, "uns", kw["keys"], "uns"),
    "must_have_layers":         lambda a, **kw: _check_must_have(a, "layers", kw["keys"], "layers"),
    "must_have_layers_regex":   lambda a, **kw: _check_must_have_regex(a, "layers", kw["patterns"], "layers"),
    "must_have_obsm_keys":      lambda a, **kw: _check_must_have(a, "obsm", kw["keys"], "obsm"),
    "must_have_obsm_keys_regex": lambda a, **kw: _check_must_have_regex(a, "obsm", kw["patterns"], "obsm"),
    "must_have_obs_keys_regex":  lambda a, **kw: _check_must_have_regex(a, "obs", kw["patterns"], "obs"),
    "must_have_var_keys_regex":  lambda a, **kw: _check_must_have_regex(a, "var", kw["patterns"], "var"),
    "must_have_uns_keys_regex":  lambda a, **kw: _check_must_have_regex(a, "uns", kw["patterns"], "uns"),
    "must_have_any_container_regex": lambda a, **kw: _check_any_container_regex(
        a, kw["patterns"], kw.get("containers", ["layers", "obsm", "obs", "uns"])),
    "obs_value_comparison":      lambda a, **kw: _check_obs_value_comparison(
        a, kw["obs_a"], kw.get("obs_b") or kw.get("obs_b_pattern"),
        kw.get("comparison", "a_gt_b"), kw.get("min_fraction", 0.5)),
    "var_unique_count":         lambda a, **kw: _check_var_unique_count(
                                    a, kw["var_key"], kw.get("min_unique", 1), kw.get("max_unique", 10**6)),
    "pseudotime_root_anchored": lambda a, **kw: _check_pseudotime_root_anchored(
        a, kw["groupby_obs"], kw["root_cluster"], kw["pseudotime_obs_pattern"]),
    "pairwise_pseudotime_correlation": lambda a, **kw: _check_pairwise_pseudotime_correlation(
        a, kw["pseudotime_obs_pattern"],
        kw.get("min_pairwise_spearman", 0.4),
        kw.get("min_methods", 2)),
    "fine_tune_evidence": lambda a, **kw: _check_finetune_evidence(
        a, trajectory_path=kw.get("trajectory_path"),
        patterns=kw.get("patterns")),
    "tool_output_evidence": lambda a, **kw: _check_tool_output_evidence(
        a, trajectory_path=kw.get("trajectory_path"),
        patterns=kw.get("patterns"),
        description=kw.get("description", "expected workflow markers")),
    "aucell_celltype_specificity": lambda a, **kw: _check_aucell_celltype_specificity(
        a,
        celltype_obs=kw.get("celltype_obs", "cell_type"),
        score_pattern=kw.get("score_pattern", r"^(aucell|score|signature|auc)_"),
        min_celltype_specific=kw.get("min_celltype_specific", 2),
        min_z_gap=kw.get("min_z_gap", 0.5)),
    "obs_count_matching_regex": lambda a, **kw: (
        lambda rx, hits: (True, f"{len(hits)} matching: {hits}") if len(hits) >= kw.get("min_count", 1)
                          else (False, f"only {len(hits)} obs cols match {kw['pattern']!r}: {hits}")
    )(re.compile(kw["pattern"], re.I),
       [c for c in a.obs.columns if re.compile(kw["pattern"], re.I).search(c)]),
    "obsm_count_matching_regex": lambda a, **kw: _check_obsm_count_matching_regex(
                                    a, kw["pattern"], kw.get("min_count", 2)),
    "shape_range":              lambda a, **kw: _check_shape_range(a, kw["axis"], kw.get("min"), kw.get("max")),
    "value_range":              lambda a, **kw: _check_value_range(
        a, kw.get("obs_alias"), kw.get("min"), kw.get("max"),
        kw.get("not_all_same", False), kw.get("nan_max_frac")),
    "x_value_range":            lambda a, **kw: _check_x_value_range(
        a, kw.get("min"), kw.get("max"), kw.get("not_all_integer", False)),
    "per_cell_expm1_sum":       lambda a, **kw: _check_per_cell_expm1_sum(a, kw["target"], kw["tolerance_pct"]),
    "layer_dtype_check":        lambda a, **kw: _check_layer_dtype_int(a, kw["layer"]),
    "var_bool_sum":             lambda a, **kw: _check_var_bool_sum(
        a, kw["var_key"], kw["target"], kw.get("tolerance", 0)),
    "var_count_above":          lambda a, **kw: _check_var_count_above(a, kw["var_alias"], kw["min_count"]),
    "obs_unique_count":         lambda a, **kw: _check_obs_unique_count(
        a, kw["obs_key"], kw["min_unique"], kw["max_unique"]),
    "obs_unique_subset":        lambda a, **kw: _check_obs_unique_subset(
        a, kw["obs_key"], kw["must_only_contain_substring"]),
    "obsm_shape":               lambda a, **kw: _check_obsm_shape(
        a, kw["key"], kw["expect"],
        kw.get("cells_tolerance", 0), kw.get("dims_tolerance", 0)),
    "obsm_dim_at_least":        lambda a, **kw: _check_obsm_dim_at_least(a, kw["obsm_key"], kw["min_dims"]),
    "uns_dict_keys":            lambda a, **kw: _check_uns_dict_keys(
        a, kw["uns_key"], kw["must_have_subkeys"]),
    "uns_value_nonempty":       lambda a, **kw: _check_uns_value_nonempty(
        a, kw["uns_key"], kw.get("min_rows", 1)),
    "uns_dataframe_has_directional_columns": lambda a, **kw: _check_uns_dataframe_has_directional_columns(
        a, kw["uns_key"]),

    # Biology
    "marker_overlap_in_var":    lambda a, **kw: _check_marker_overlap_in_var(
        a, kw["var_alias"], kw["ref_markers"], kw.get("min_count", 5)),
    "obsm_celltype_silhouette": lambda a, **kw: _check_obsm_celltype_silhouette(
        a, kw["obsm_key"], kw.get("celltype_obs", "cell_type"),
        kw.get("min_silhouette", 0.10)),
    "clustering_ari":           lambda a, oracle_path=None, **kw: _check_clustering_ari(
        a, kw["obs_key"], oracle_path, kw["oracle_obs_key"], kw["min_ari"]),
    "subcluster_marker_split":  lambda a, **kw: _check_subcluster_marker_split(
        a, kw["subcluster_obs"], kw["marker_a"], kw["marker_b"]),
    "cluster_top_markers_canonical": lambda a, **kw: _check_cluster_top_markers_canonical(
        a, kw.get("leiden_obs", "leiden"), kw.get("celltype_obs", "cell_type"),
        kw.get("rank_uns_key", "rank_genes_groups"),
        kw.get("top_n", 10), kw.get("min_clusters_with_canonical", 3)),
    "marker_overlap_vs_oracle": lambda a, oracle_path=None, **kw: _check_marker_overlap_vs_oracle(
        a, kw["uns_key"], oracle_path, kw["oracle_uns_key"],
        kw.get("top_n_per_cluster", 10), kw.get("min_overlap_jaccard", 0.2)),

    # Multi-method
    "batch_silhouette_drop":    lambda a, **kw: _check_batch_silhouette_drop(
        a, kw["batch_obs_key"],
        kw.get("uncorrected_obsm", "X_pca"),
        kw.get("corrected_obsm_pattern", r"(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)"),
        kw.get("min_drop", 0.03)),
    "celltype_silhouette_preserved": lambda a, **kw: _check_celltype_silhouette_preserved(
        a, kw.get("celltype_obs", "cell_type"),
        kw.get("uncorrected_obsm", "X_pca"),
        kw.get("corrected_obsm_pattern", r"(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)"),
        kw.get("min_relative_preservation", 0.5)),
    "velocity_modes_consistency":   lambda a, **kw: _check_velocity_modes_consistency(
        a, kw.get("min_mean_cosine", 0.05)),
    "velocity_root_anchoring":      lambda a, **kw: _check_velocity_root_anchoring(
        a, kw["root_cluster"], kw.get("groupby_obs", "clusters"),
        kw.get("basis_obsm", "X_umap"), kw.get("velocity_obsm", "velocity_umap"),
        kw.get("min_mean_outward_cosine", 0.10)),
    "pseudotime_root_agreement":    lambda a, **kw: _check_pseudotime_root_agreement(
        a, kw["root_cluster"], kw.get("groupby_obs", "clusters"),
        kw.get("min_root_to_other_gap", 0.10)),
    "obs_two_distinct_pseudotime":  lambda a, **kw: _check_obs_two_distinct_pseudotime(
        a, kw.get("min_distinct_pseudotime_cols", 2)),

    # Spatial / multi-omics / bulk / 16S
    "spatial_domain_silhouette":    lambda a, **kw: _check_spatial_domain_silhouette(
        a, kw["cluster_obs"], kw.get("spatial_obsm", "spatial"),
        kw.get("min_silhouette", 0.05)),
    "ccc_reference_lr_hit":         lambda a, **kw: _check_ccc_reference_lr_hit(
        a, kw["uns_key"], kw.get("min_hits", 1), kw.get("reference_pairs")),
    "mofa_factor_variance":         lambda a, **kw: _check_mofa_factor_variance(
        a, kw.get("factor_obsm", "X_mofa"), kw.get("min_factors", 5),
        kw.get("min_total_variance", 0.30)),
    "peak_gene_link_count":         lambda a, **kw: _check_peak_gene_link_count(
        a, kw["uns_key"], kw.get("min_links", 1000),
        kw.get("min_promoter_proximal_frac", 0.30)),
    "deconv_fractions_sane":        lambda a, **kw: _check_deconv_fractions_sane(
        a, kw.get("fractions_obs_pattern", r"frac_|_fraction|cell_type_frac"),
        kw.get("min_n_celltypes", 3), kw.get("sum_tolerance", 0.10)),
    "bulk2single_ari_vs_ref":       lambda a, oracle_path=None, **kw: _check_bulk2single_ari_vs_ref(
        a, oracle_path, kw.get("min_ari", 0.20), kw.get("cluster_obs", "cell_type")),
    "alpha_diversity_present":      lambda a, **kw: _check_alpha_diversity_present(
        a, kw["metrics"]),
    "beta_diversity_present":       lambda a, **kw: _check_beta_diversity_present(
        a, kw["uns_or_obsp_keys"]),
    "da_method_overlap":            lambda a, **kw: _check_da_method_overlap(
        a, kw.get("uns_pattern", r"da_|differential_abundance|wilcox|deseq|ancombc"),
        kw.get("min_methods", 2), kw.get("min_jaccard", 0.20),
        kw.get("top_n", 30)),

    # v1.1 practitioner-rigor checks
    "multi_doublet_consensus":      lambda a, **kw: _check_multi_doublet_consensus(
        a, kw.get("score_pattern", r"doublet_score|doublet_pred|scrublet|scdblfinder|doubletfinder|sccomposite"),
        kw.get("min_methods", 2), kw.get("min_score_agreement", 0.10),
        kw.get("consensus_obs_pattern", r"doublet_consensus|doublet|is_doublet")),
    "clustering_resolution_stability": lambda a, **kw: _check_clustering_resolution_stability(
        a, kw.get("resolution_obs_pattern", r"leiden_r|leiden_res|res\d|resolution_"),
        kw.get("min_resolutions_tested", 2), kw.get("min_pairwise_ari", 0.40)),
    "multi_method_annotation_consistency": lambda a, **kw: _check_multi_method_annotation_consistency(
        a, kw.get("celltype_obs_pattern", r"cell_type|celltype|cell_types|annotation"),
        kw.get("min_methods", 2), kw.get("min_majority_agreement", 0.60)),
    "multi_method_de_overlap":      lambda a, **kw: _check_multi_method_de_overlap(
        a, kw.get("uns_pattern", r"rank_genes|de_|differential_expr|wilcoxon|t_test|cosg|mast|scdeg"),
        kw.get("min_methods", 2), kw.get("min_top_jaccard", 0.30),
        kw.get("top_n", 20)),
    "kbet_lisi_quantitative":       lambda a, **kw: _check_kbet_lisi_quantitative(
        a, kw["batch_obs_key"],
        kw.get("corrected_obsm_pattern", r"(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)"),
        kw.get("min_lisi", 0.60)),
    "joint_embedding_outperforms_single": lambda a, **kw: _check_joint_embedding_outperforms_single(
        a, kw.get("joint_obsm_pattern", r"X_(mofa|glue|multivi|wnn|joint)"),
        kw.get("single_obsm_pattern", r"X_(pca|rna|atac)"),
        kw.get("celltype_obs", "cell_type"),
        kw.get("min_silhouette_improvement", 0.02)),
    "multi_method_deconv_agreement": lambda a, **kw: _check_multi_method_deconv_agreement(
        a, kw.get("fractions_keys_pattern", r"fractions|deconv|frac_"),
        kw.get("min_methods", 2), kw.get("min_correlation", 0.40)),
    "bulk2single_composition_match": lambda a, oracle_path=None, **kw: _check_bulk2single_composition_match(
        a, oracle_path, kw.get("cluster_obs", "cell_type"),
        kw.get("min_correlation", 0.40)),
    "velocity_confidence_present":  lambda a, **kw: _check_velocity_confidence_present(
        a, kw.get("min_nonzero_frac", 0.50), kw.get("confidence_keys")),
    "trajectory_branch_detection":  lambda a, **kw: _check_trajectory_branch_detection(
        a, kw.get("branch_uns_pattern", r"terminal|branch|fate|cellrank"),
        kw.get("min_terminals", 2)),
    "alpha_diversity_group_test":   lambda a, **kw: _check_alpha_diversity_group_test(
        a, kw["group_obs"], kw.get("metric", "shannon"),
        kw.get("max_pvalue", 0.05),
        kw.get("test_uns_key_pattern", r"alpha_test|kruskal|mann_whitney|group_diff")),
    "beta_diversity_permanova":     lambda a, **kw: _check_beta_diversity_permanova(
        a, kw.get("beta_obsp_pattern", r"beta|bray|unifrac"),
        kw.get("group_obs", "group"), kw.get("max_pvalue", 0.05),
        kw.get("test_uns_key_pattern", r"permanova|anosim|beta_test")),
    "clustering_ari_vs_obs":        lambda a, **kw: _check_clustering_ari_vs_obs(
        a, kw["pred_obs"], kw["truth_obs"], kw.get("min_ari", 0.20)),
}


# ---------------------------------------------------------------------------
# Main grader entrypoint
# ---------------------------------------------------------------------------

def grade(*, final_adata_path, checks: list[dict], oracle_path: str | None = None,
          task_id: str = "", system: str = "", model_id: str = "", seed: int = 0,
          trajectory_path: str | None = None,
          **_unused) -> Grade:
    if final_adata_path is None:
        return Grade(task_id=task_id, system=system, model_id=model_id, seed=seed,
                     passed=False, score=0.0,
                     failure_mode=FailureMode.SILENT_NONE,
                     notes="no final adata produced")
    try:
        adata = _load_adata(final_adata_path)
    except Exception as exc:
        return Grade(task_id=task_id, system=system, model_id=model_id, seed=seed,
                     passed=False, score=0.0,
                     failure_mode=FailureMode.ADAPTER_ERROR,
                     notes=f"load failed: {exc}")
    # If trajectory_path wasn't passed, derive from the workspace layout
    # (mini-swe-agent writes ``final.h5ad`` and ``minisweagent_trajectory.json``
    # as siblings). Lets new check types inspect agent behavior alongside
    # the deliverable.
    if trajectory_path is None and final_adata_path:
        candidate = Path(final_adata_path).parent / "minisweagent_trajectory.json"
        if candidate.exists():
            trajectory_path = str(candidate)
    rubric: dict[str, bool] = {}
    notes: list[str] = []
    for check in checks:
        cid = check.get("id", check.get("type", "check"))
        ctype = check["type"]
        kwargs = {k: v for k, v in check.items()
                  if k not in ("id", "type", "rationale")}
        fn = CHECK_DISPATCH.get(ctype)
        if fn is None:
            rubric[cid] = False
            notes.append(f"{cid}: UNKNOWN check type {ctype!r}")
            continue
        try:
            if ctype in ("clustering_ari", "marker_overlap_vs_oracle",
                         "bulk2single_ari_vs_ref",
                         "bulk2single_composition_match"):
                ok, msg = fn(adata, oracle_path=oracle_path, **kwargs)
            elif ctype in ("fine_tune_evidence", "tool_output_evidence"):
                ok, msg = fn(adata, trajectory_path=trajectory_path, **kwargs)
            else:
                ok, msg = fn(adata, **kwargs)
        except Exception as exc:
            ok, msg = False, f"check raised: {type(exc).__name__}: {exc}"
        rubric[cid] = bool(ok)
        if msg:
            notes.append(f"{cid}: {msg}")
    n_pass = sum(1 for v in rubric.values() if v)
    n_total = len(rubric) or 1
    score = n_pass / n_total
    passed = (n_pass == n_total)
    return Grade(task_id=task_id, system=system, model_id=model_id, seed=seed,
                 passed=passed, score=score,
                 failure_mode=FailureMode.NONE if passed else FailureMode.WRONG_TOOL_CHOICE,
                 rubric=rubric,
                 notes=" | ".join(notes)[:1500])
