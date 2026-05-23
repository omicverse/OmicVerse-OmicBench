"""bench v1.0 task definitions — 20 tasks across 7 layers.

Layout reflects real omicverse analytical scenarios drawn from the
omicverse_guide tutorial corpus. Each task = a scientific goal a single-
cell biologist would actually run, not a pretty wrapper around a one-liner.

Prompt policy:
- Describe the scientific GOAL only.
- Never name an `ov.X.Y` API in the prompt.
- Storage targets reference key NAMES (e.g., "obs key `pct_counts_mt`")
  not Python indexing (no `adata.obs["..."]`). This avoids triggering
  ov.Agent's direct-Python shortcut detector.
- "Choose any approach" lets every system compete on its own ground.
  Some tasks have a flagship ov function; raw_llm composes equivalents
  from standard libs. The bench measures the result of that competition.

Grading policy:
- Every check has a biological / numerical justification, not just key
  existence. See `bench/grader.py` for the canonical reference lists
  (PBMC markers, pancreas endocrine markers, PBMC LR pairs).
- Alias groups everywhere — `pct_counts_mt` OR `mito_perc` both pass.
- Numerical thresholds calibrated against hand-coded scanpy baselines.
"""

from __future__ import annotations

INPUTS  = "<OVBENCH_ROOT>/fixtures/inputs"
ORACLES = "<OVBENCH_ROOT>/fixtures/oracles"


# =====================================================================
# Layer A — scRNA preprocessing floor (5 tasks)
# Tests every system can reach baseline scanpy competency on PBMC.
# =====================================================================

A01 = {
    "id": "A01_qc_filter",
    "layer": "A",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  f"{ORACLES}/A01_qc_oracle.h5ad",
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Perform publication-quality QC on this PBMC scRNA-seq dataset, "
        "with the rigor a peer reviewer would expect:\n\n"
        "  1. Cell-level filtering: drop cells with <200 genes; drop "
        "genes in <3 cells.\n"
        "  2. Compute three QC fractions per cell and store as obs "
        "columns:\n"
        "     - pct_counts_mt (mitochondrial; gene prefix MT-)\n"
        "     - pct_counts_ribo (ribosomal; gene prefixes RPS, RPL)\n"
        "     - pct_counts_hb (hemoglobin; gene prefix HB but exclude HBP)\n"
        "  3. Filter cells with pct_counts_mt < 20.\n"
        "  4. Doublet detection — apply TWO different doublet-detection "
        "methods (e.g. scrublet, scDblFinder, doubletfinder, sccomposite). "
        "Store per-cell doublet scores from each method as obs columns "
        "whose names contain 'doublet_score' (one per method). "
        "Form a consensus by flagging cells called doublet by ≥1 method "
        "into an obs boolean column doublet_consensus, then filter "
        "consensus doublets out.\n\n"
        "Single-method doublet calls or skipping doublet detection "
        "entirely will be marked insufficient by the reviewer. Modify "
        "the dataset in place."
    ),
    "checks": [
        {"id": "mt_obs", "type": "must_have_obs_keys",
         "keys": [["pct_counts_mt", "mito_perc"]]},
        {"id": "ribo_obs", "type": "must_have_obs_keys",
         "keys": [["pct_counts_ribo", "ribo_perc", "pct_ribo"]]},
        {"id": "cells_after_qc", "type": "shape_range", "axis": 0,
         "min": 5500, "max": 7700,
         "rationale": "post multi-step QC + doublet removal yields ~6500-7500 cells from 7750 starting"},
        {"id": "genes_filtered", "type": "shape_range", "axis": 1,
         "min": 12000, "max": 20939},
        {"id": "mt_values_sane", "type": "value_range",
         "obs_alias": ["pct_counts_mt", "mito_perc"],
         "min": 0, "max": 100, "not_all_same": True},
        {"id": "multi_doublet_methods", "type": "multi_doublet_consensus",
         "min_methods": 2, "min_score_agreement": 0.10,
         "rationale": "≥2 doublet-detection methods + consensus column; pairwise score Spearman ≥ 0.1"},
    ],
}

A02 = {
    "id": "A02_normalize_log",
    "layer": "A",
    "difficulty": "easy",
    "fixture": f"{ORACLES}/A01_qc_oracle.h5ad",
    "oracle":  f"{ORACLES}/A02_normalize_oracle.h5ad",
    "max_turns": 50, "wallclock_s": 300,
    "prompt": (
        "Normalize the expression matrix to a per-cell total of 10000, then "
        "log1p-transform it. Before normalizing, save the raw integer counts "
        "to a layer named counts (alias raw_counts accepted) so the original "
        "integers are preserved. Modify the dataset in place."
    ),
    "checks": [
        {"id": "counts_layer", "type": "must_have_layers",
         "keys": [["counts", "raw_counts"]]},
        {"id": "x_is_normalized", "type": "x_value_range",
         "min": 0, "max": 15, "not_all_integer": True},
        {"id": "per_cell_sum", "type": "per_cell_expm1_sum",
         "target": 10000, "tolerance_pct": 5},
        {"id": "counts_int", "type": "layer_dtype_check",
         "layer": ["counts", "raw_counts"]},
    ],
}

A03 = {
    "id": "A03_hvg",
    "layer": "A",
    "difficulty": "easy",
    "fixture": f"{ORACLES}/A02_normalize_oracle.h5ad",
    "oracle":  f"{ORACLES}/A03_hvg_oracle.h5ad",
    "max_turns": 50, "wallclock_s": 300,
    "prompt": (
        "Select 2000 highly-variable genes using the seurat_v3 HVG flavor "
        "(which uses raw integer counts saved in the counts layer). Mark "
        "the chosen genes as a boolean column named highly_variable in the "
        "gene metadata table. Do NOT subset to only the HVGs — keep all "
        "genes, just add the marker column. Modify the dataset in place."
    ),
    "checks": [
        {"id": "hvg_present", "type": "must_have_var_keys",
         "keys": [["highly_variable", "highly_variable_features"]]},
        {"id": "hvg_count", "type": "var_bool_sum",
         "var_key": ["highly_variable", "highly_variable_features"],
         "target": 2000, "tolerance": 50},
        {"id": "hvg_includes_canonical_markers", "type": "marker_overlap_in_var",
         "var_alias": ["highly_variable", "highly_variable_features"],
         "ref_markers": ["CD3D", "CD3E", "CD4", "CD8A", "CD8B", "CD79A", "CD79B",
                          "MS4A1", "CD14", "LYZ", "S100A8", "S100A9", "FCGR3A",
                          "NKG7", "GNLY", "GZMB", "GZMK", "CST3", "FCER1A",
                          "IL7R", "CCR7"],
         "min_count": 5,
         "rationale": "any reasonable PBMC HVG selection should hit ≥5 canonical lineage markers"},
    ],
}

A04 = {
    "id": "A04_pca_neighbors_umap",
    "layer": "A",
    "difficulty": "medium",
    "fixture": f"{ORACLES}/A03_hvg_oracle.h5ad",
    "oracle":  f"{ORACLES}/A04_pca_neighbors_umap_oracle.h5ad",
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Run dimensionality reduction:\n"
        "(1) PCA with 50 components on the HVGs (boolean mask in var column "
        "highly_variable);\n"
        "(2) build a neighborhood graph in PCA space with 15 neighbors;\n"
        "(3) compute a 2D UMAP embedding.\n"
        "Store the PCA coordinates in obsm key X_pca, the UMAP coordinates "
        "in obsm key X_umap, and the neighborhood graph in uns key "
        "neighbors. Modify the dataset in place."
    ),
    "checks": [
        {"id": "pca", "type": "must_have_obsm_keys", "keys": ["X_pca"]},
        {"id": "neighbors", "type": "must_have_uns_keys", "keys": ["neighbors"]},
        {"id": "umap", "type": "must_have_obsm_keys", "keys": ["X_umap"]},
        {"id": "umap_shape", "type": "obsm_shape", "key": "X_umap",
         "expect": [7710, 2], "cells_tolerance": 50, "dims_tolerance": 0},
        {"id": "umap_preserves_celltype", "type": "obsm_celltype_silhouette",
         "obsm_key": "X_umap", "celltype_obs": "cell_type",
         "min_silhouette": 0.10,
         "rationale": "scanpy hand-coded baseline yields ~0.30; threshold 0.10 catches random scatter"},
    ],
}

A05 = {
    "id": "A05_leiden_clustering",
    "layer": "A",
    "difficulty": "hard",
    "fixture": f"{ORACLES}/A04_pca_neighbors_umap_oracle.h5ad",
    "oracle":  f"{ORACLES}/A05_leiden_oracle.h5ad",
    "max_turns": 50, "wallclock_s": 900,
    "prompt": (
        "Determine the optimal clustering resolution for this PBMC dataset "
        "with publication-grade rigor. The neighborhood graph is already "
        "computed under uns key neighbors — use it.\n\n"
        "Don't hardcode a single resolution. Instead:\n"
        "  1. Run Leiden clustering at AT LEAST 3 distinct resolutions "
        "(e.g. 0.4, 0.8, 1.2) and store each result as an obs column "
        "whose name contains the resolution (e.g. leiden_r0.4, "
        "leiden_r0.8, leiden_r1.2 — the column-name pattern leiden_r* / "
        "res* / resolution_* is recognized).\n"
        "  2. Select an optimal resolution based on stability or silhouette, "
        "and store the chosen labels as obs column leiden (alias clusters).\n"
        "  3. Record the chosen resolution as a numeric uns entry "
        "leiden_resolution.\n\n"
        "A single hardcoded resolution without a stability scan is "
        "insufficient for peer review. Modify the dataset in place."
    ),
    "checks": [
        {"id": "leiden_present", "type": "must_have_obs_keys",
         "keys": [["leiden", "clusters"]]},
        {"id": "cluster_count", "type": "obs_unique_count",
         "obs_key": ["leiden", "clusters"],
         "min_unique": 5, "max_unique": 30},
        {"id": "resolution_recorded", "type": "must_have_uns_keys",
         "keys": [["leiden_resolution", "selected_resolution",
                    "resolution_chosen"]]},
        {"id": "multi_resolution_stability", "type": "clustering_resolution_stability",
         "min_resolutions_tested": 3, "min_pairwise_ari": 0.40,
         "rationale": "≥3 resolutions tested; clusters should be stable (mean pairwise ARI ≥0.4) — single-resolution fails"},
        {"id": "ari_vs_celltype", "type": "clustering_ari",
         "obs_key": ["leiden", "clusters"],
         "oracle_obs_key": "cell_type", "min_ari": 0.30,
         "rationale": "optimally-tuned Leiden should beat hardcoded res=1 (ARI ≥ 0.30)"},
    ],
}


# =====================================================================
# Layer B — scRNA workflows (4 tasks)
# Multi-step planning + multi-method dispatch.
# =====================================================================

B01 = {
    "id": "B01_full_pipeline",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  f"{ORACLES}/A05_leiden_oracle.h5ad",
    "max_turns": 100, "wallclock_s": 1500,
    "prompt": (
        "Run a publication-quality preprocessing-to-clustering pipeline "
        "on this PBMC dataset. A reviewer will expect every step and "
        "their justification:\n\n"
        "  1. QC: drop cells <200 genes, genes <3 cells; compute pct_counts_mt "
        "(MT- prefix) and pct_counts_ribo (RPS/RPL prefixes) as obs columns; "
        "keep cells with mt% < 20.\n"
        "  2. Doublet detection: apply ≥2 doublet methods, form an obs "
        "consensus column doublet_consensus, and remove flagged doublets.\n"
        "  3. Save raw counts to a layer named counts.\n"
        "  4. Normalize total counts per cell to 10000.\n"
        "  5. log1p-transform.\n"
        "  6. Select 2000 highly-variable genes (seurat_v3 on counts layer).\n"
        "  7. PCA(50) on the HVGs.\n"
        "  8. 15-neighbor graph + Leiden clustering. Don't hardcode a single "
        "resolution — test ≥2 (e.g. 0.5, 1.0) and select the stable one. "
        "Store final labels as obs column leiden, and the chosen resolution "
        "as uns entry leiden_resolution.\n"
        "  9. 2D UMAP into obsm key X_umap.\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "mt_obs", "type": "must_have_obs_keys",
         "keys": [["pct_counts_mt", "mito_perc"]]},
        {"id": "ribo_obs", "type": "must_have_obs_keys",
         "keys": [["pct_counts_ribo", "ribo_perc", "pct_ribo"]]},
        {"id": "leiden_present", "type": "must_have_obs_keys",
         "keys": [["leiden", "clusters"]]},
        {"id": "counts_layer", "type": "must_have_layers",
         "keys": [["counts", "raw_counts"]]},
        {"id": "hvg", "type": "must_have_var_keys",
         "keys": [["highly_variable", "highly_variable_features"]]},
        {"id": "pca", "type": "must_have_obsm_keys", "keys": ["X_pca"]},
        {"id": "umap", "type": "must_have_obsm_keys", "keys": ["X_umap"]},
        {"id": "cluster_count", "type": "obs_unique_count",
         "obs_key": ["leiden", "clusters"],
         "min_unique": 5, "max_unique": 25},
        {"id": "cells_after_full_qc", "type": "shape_range", "axis": 0,
         "min": 5500, "max": 8000,
         "rationale": "doublet removal + mt-filter ⇒ ~6500-7800 cells (small slack for QC threshold differences across arms)"},
        {"id": "resolution_chosen", "type": "must_have_uns_keys",
         "keys": [["leiden_resolution", "selected_resolution",
                    "resolution_chosen"]]},
        {"id": "ari_vs_celltype", "type": "clustering_ari",
         "obs_key": ["leiden", "clusters"],
         "oracle_obs_key": "cell_type", "min_ari": 0.25},
    ],
}

B02 = {
    "id": "B02_subset_recluster",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{ORACLES}/A05_leiden_oracle.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "This dataset has cell-type annotations in obs column cell_type for "
        "PBMC classes including \"CD4+ T cell\" and \"Cytotoxic T cell\". "
        "Subset to ONLY the T cells (any cell_type value containing the "
        "substring \"T cell\"), then re-cluster them at finer resolution "
        "(Leiden, resolution 0.5) on a freshly recomputed neighborhood "
        "graph. Store the new sub-cluster labels in an obs column named "
        "t_subcluster. Return the SUBSET dataset (not the original)."
    ),
    "checks": [
        {"id": "subset_size", "type": "shape_range", "axis": 0,
         "min": 1000, "max": 1600},
        {"id": "only_t_cells", "type": "obs_unique_subset",
         "obs_key": "cell_type", "must_only_contain_substring": "T cell"},
        {"id": "tsubcluster_present", "type": "must_have_obs_keys",
         "keys": [["t_subcluster", "subcluster", "leiden_sub", "leiden"]]},
        {"id": "subcluster_count", "type": "obs_unique_count",
         "obs_key": ["t_subcluster", "subcluster", "leiden_sub", "leiden"],
         "min_unique": 3, "max_unique": 12},
        {"id": "cd4_cd8_split", "type": "subcluster_marker_split",
         "subcluster_obs": ["t_subcluster", "subcluster", "leiden_sub", "leiden"],
         "marker_a": "CD4", "marker_b": "CD8A",
         "rationale": "T-cell sub-clusters must include ≥1 CD4-dominant and ≥1 CD8A-dominant cluster"},
    ],
}

B03 = {
    "id": "B03_diff_expression",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{ORACLES}/A05_leiden_oracle.h5ad",
    "oracle":  f"{ORACLES}/B03_rank_genes_oracle.h5ad",
    "max_turns": 50, "wallclock_s": 900,
    "prompt": (
        "Identify cluster markers with publication rigor. The Leiden "
        "cluster labels are in obs column leiden. A reviewer will not "
        "accept a single-method ranking — gene-level DE is sensitive to "
        "test choice (Wilcoxon's tied-rank handling vs t-test variance "
        "assumptions vs cosg's specificity scoring vs MAST's hurdle "
        "model). Run AT LEAST TWO distinct DE methods and store each "
        "result in a separate uns entry whose name identifies the method "
        "(e.g. rank_genes_groups for Wilcoxon, rank_genes_t_test for "
        "t-test, cosg_results, scdeg_results). The standard scanpy "
        "rank_genes_groups slot must still be populated for one method.\n\n"
        "Top markers per cluster should be biologically grounded — the "
        "top-10 of the dominant cluster of each PBMC cell-type should "
        "contain canonical lineage markers (CD3D, CD8A, CD79A, MS4A1, "
        "CD14, LYZ, NKG7, GNLY, etc.). Modify the dataset in place."
    ),
    "checks": [
        {"id": "rank_present", "type": "must_have_uns_keys", "keys": ["rank_genes_groups"]},
        {"id": "rank_substructure", "type": "uns_dict_keys",
         "uns_key": "rank_genes_groups",
         "must_have_subkeys": ["names", "pvals_adj"]},
        {"id": "multi_method_de_consistency", "type": "multi_method_de_overlap",
         "min_methods": 2, "min_top_jaccard": 0.30, "top_n": 20,
         "rationale": "≥2 DE methods stored; per-cluster top-20 Jaccard ≥ 0.30 across methods"},
        {"id": "top_markers_canonical_per_cluster",
         "type": "cluster_top_markers_canonical",
         "leiden_obs": "leiden", "celltype_obs": "cell_type",
         "rank_uns_key": "rank_genes_groups",
         "top_n": 10, "min_clusters_with_canonical": 3,
         "rationale": "top-10 markers per cluster must include ≥1 canonical PBMC marker for that cluster's cell-type"},
    ],
}

B04 = {
    "id": "B04_multi_method_batch_correction",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/neurips2021_batch_all.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 1500,
    "prompt": (
        "This bone-marrow scRNA dataset has cells from multiple batches in "
        "obs column batch (3 NeurIPS-2021 donor batches). Remove batch "
        "effects using TWO different correction methods of your choice "
        "(e.g. linear regression, mixture-model, mutual-nearest-neighbor, "
        "graph integration, VAE — any two distinct algorithmic families). "
        "Store both corrected representations in obsm keys whose names "
        "contain the chosen method names (e.g. X_harmony, X_combat, "
        "X_scanorama, X_scvi), or layer keys named <method>_corrected for "
        "matrix-returning methods. Also record the two method names in uns "
        "key batch_correction_methods (list of two strings).\n\n"
        "The data has cell-type annotations in obs column cell_type. "
        "After correction, both batch separation should decrease AND "
        "cell-type structure should be preserved."
    ),
    "checks": [
        {"id": "batch_obs", "type": "must_have_obs_keys", "keys": ["batch"]},
        {"id": "batch_separation_reduced", "type": "batch_silhouette_drop",
         "batch_obs_key": "batch",
         "uncorrected_obsm": "X_pca",
         "corrected_obsm_pattern": "(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)",
         "min_drop": 0.02,
         "rationale": "real harmony/combat drops batch silhouette by 0.05-0.20 on this "
                       "fixture; threshold lowered from 0.03 to 0.02 because the "
                       "neurips2021 fixture's baseline batch silhouette is small "
                       "(~0.008) so even successful corrections only show modest absolute drops"},
        {"id": "celltype_structure_preserved", "type": "celltype_silhouette_preserved",
         "celltype_obs": "cell_type",
         "uncorrected_obsm": "X_pca",
         "corrected_obsm_pattern": "(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)",
         "min_relative_preservation": 0.5,
         "rationale": "scIB-style biology preservation; over-correction that erases cell-type → fail"},
        {"id": "kbet_lisi_quantitative", "type": "kbet_lisi_quantitative",
         "batch_obs_key": "batch",
         "corrected_obsm_pattern": "(harmony|combat|scanorama|scvi|mnn|bbknn|corrected)",
         "min_lisi": 0.60,
         "rationale": "kNN-batch-mixing on corrected embedding ≥ 0.60 — local mixing not just global silhouette"},
    ],
}

B05 = {
    "id": "B05_multi_method_annotation",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{ORACLES}/A05_leiden_oracle.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 1500,
    "prompt": (
        "Annotate cell types in this preprocessed PBMC dataset with "
        "publication rigor. A reviewer will not accept a single tool's "
        "annotation — different annotation methods rely on different "
        "training data and rule systems, and disagreement signals "
        "uncertainty.\n\n"
        "Use AT LEAST TWO distinct annotation approaches (e.g. "
        "marker-based scoring such as cosg/scsa/celltypist, reference-"
        "based mapping such as popV/scmap/SingleR, or zero-shot LLM-"
        "guided). Store each method's per-cell predictions in a separate "
        "obs column whose name follows the pattern cell_type_<method> or "
        "annotation_<method> (e.g. cell_type_cosg, cell_type_scsa). "
        "Methods should agree on the dominant cell-type for at least 60% "
        "of cells (PBMC types are well-resolved; large disagreement = "
        "annotation failure). Modify the dataset in place."
    ),
    "checks": [
        {"id": "multi_annotation", "type": "multi_method_annotation_consistency",
         # Strict pattern: require a `_<method>` suffix so that fixture-bundled
         # metadata columns (`cell_type`, `predicted_celltype`,
         # `cell_type_ontology_term_id`) are excluded and only the agent's
         # newly-added per-method columns are compared.
         "celltype_obs_pattern": r"(cell_type|celltype|annotation)_[a-zA-Z0-9_]+",
         "min_methods": 2, "min_majority_agreement": 0.60,
         "rationale": "≥2 annotation methods + ≥60% pairwise agreement"},
    ],
}


# =====================================================================
# Layer C — Spatial transcriptomics (3 tasks)
# Visium DLPFC, Visium HD bin-level, spatial CCC. ★ ov flagship.
# =====================================================================

C01 = {
    "id": "C01_visium_spatial_domain",
    "layer": "C",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/visium_dlpfc_151676",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "This is human DLPFC sample 151676 from the Maynard et al 2021 "
        "atlas. The fixture directory contains:\n"
        "  - filtered_feature_bc_matrix.h5 (count matrix)\n"
        "  - spatial/ (tissue images + scalefactors_json + "
        "tissue_positions)\n"
        "  - 151676_truth.txt (one manual cortical-layer label per spot, "
        "in tissue_positions order)\n\n"
        "Identify spatial domains with peer-review-grade rigor:\n"
        "  1. Load with spatial coordinates in obsm key spatial.\n"
        "  2. Read 151676_truth.txt into obs column ground_truth_layer "
        "(rows aligned to spots).\n"
        "  3. Run AT LEAST TWO distinct spatial-aware clustering "
        "methods (e.g. STAGATE, SpaceFlow, GraphST, BayesSpace, MCLUST "
        "on spatial-aware embedding). Store each method's labels as "
        "obs columns spatial_domain_<method>. ALSO store one chosen "
        "method's labels as obs column spatial_domain (alias clusters / "
        "leiden / spatial_clusters).\n\n"
        "**Methodological constraint** (this is graded): the manual "
        "labels in `ground_truth_layer` exist only so the grader can "
        "compute ARI at the end. **Do NOT use them for hyperparameter "
        "tuning, model selection, or any optimization signal during "
        "clustering** — that is test-set leakage. Pick clustering "
        "hyperparameters using unsupervised criteria (silhouette on "
        "spatial coords, modularity, scale-free heuristics, defaults "
        "from the method's docstring) before reading the truth column.\n\n"
        "DLPFC has ~7 cortical layers + white matter — expect 4-12 "
        "domains per method. The chosen method's ARI vs ground_truth_"
        "layer must reach a meaningful threshold; spatial-blind "
        "expression-only clustering will not. Modify the dataset in "
        "place."
    ),
    "checks": [
        {"id": "spatial_obsm", "type": "must_have_obsm_keys",
         "keys": ["spatial"]},
        {"id": "ground_truth_loaded", "type": "must_have_obs_keys",
         "keys": ["ground_truth_layer"]},
        {"id": "spatial_domain", "type": "must_have_obs_keys",
         "keys": [["spatial_domain", "clusters", "leiden", "spatial_clusters"]]},
        {"id": "n_domains", "type": "obs_unique_count",
         "obs_key": ["spatial_domain", "clusters", "leiden", "spatial_clusters"],
         "min_unique": 4, "max_unique": 12,
         "rationale": "DLPFC has 7 cortical layers + WM"},
        {"id": "spatial_silhouette", "type": "spatial_domain_silhouette",
         "cluster_obs": ["spatial_domain", "clusters", "leiden", "spatial_clusters"],
         "spatial_obsm": "spatial",
         "min_silhouette": 0.05,
         "rationale": "spatial-aware clustering should give silhouette ≥ 0.05 on physical coords"},
        {"id": "ari_vs_ground_truth", "type": "clustering_ari_vs_obs",
         "pred_obs": ["spatial_domain", "clusters", "leiden", "spatial_clusters"],
         "truth_obs": "ground_truth_layer",
         "min_ari": 0.20,
         "rationale": "spatial-aware methods on DLPFC reach ARI ≥ 0.20 vs manual cortical layers; expression-only fails"},
    ],
}

C02 = {
    "id": "C02_visium_hd_svg",
    "layer": "C",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/visium_hd_8um",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "This is a 10x Visium HD slide of human colon cancer tissue, "
        "binned at 8 micrometers (square_008um). The fixture directory "
        "contains filtered_feature_bc_matrix.h5 and a spatial/ subdirectory "
        "with tissue image and bin-level coordinates.\n\n"
        "Load the data, perform basic QC (drop near-empty bins, normalize, "
        "log1p), identify spatially-variable genes (SVGs) using a method "
        "that accounts for spatial autocorrelation, and store: spatial "
        "coordinates in obsm key spatial; a boolean column "
        "space_variable_features (alias spatially_variable, svg) in var "
        "marking ≥100 SVGs. Modify the dataset in place."
    ),
    "checks": [
        {"id": "spatial_obsm", "type": "must_have_obsm_keys",
         "keys": ["spatial"]},
        {"id": "svg_marker", "type": "must_have_var_keys",
         "keys": [["space_variable_features", "spatially_variable", "svg",
                   "spatial_variable_genes"]]},
        {"id": "svg_count", "type": "var_count_above",
         "var_alias": ["space_variable_features", "spatially_variable", "svg",
                        "spatial_variable_genes"],
         "min_count": 100,
         "rationale": "real Visium HD SVG analysis routinely identifies hundreds; threshold 100 catches near-empty"},
    ],
}

# =====================================================================
# Layer D — Multi-omics integration (2 tasks)
# RNA + ATAC paired multiome. ★ ov flagship.
# =====================================================================

D01 = {
    "id": "D01_multiome_joint_embedding",
    "layer": "D",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/multiome_pbmc",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 2400,
    "prompt": (
        "Paired 10x Multiome PBMC (pbmc_multiome.h5): RNA + ATAC from "
        "the same cells. Build a publication-quality joint embedding:\n"
        "  1. Load both modalities; per-modality QC and dim reduction "
        "(store RNA PCA in obsm X_pca; ATAC LSI/PCA in obsm X_atac).\n"
        "  2. Compute a JOINT embedding with at least one method "
        "(MOFA / GLUE / MultiVI / WNN / scAI). Store in obsm X_mofa "
        "(aliases X_glue, X_multivi, X_wnn, X_joint).\n"
        "  3. Annotate cells (cell_type or cell_type_pred in obs).\n\n"
        "A reviewer will compare: does the JOINT embedding give better "
        "cell-type separation than the SINGLE modalities alone? If not, "
        "joint integration adds nothing. Cell-type silhouette on the "
        "joint embedding should EXCEED the best single-modality "
        "silhouette by a measurable margin. Modify the dataset in place."
    ),
    "checks": [
        {"id": "joint_obsm", "type": "must_have_obsm_keys",
         "keys": [["X_mofa", "X_glue", "X_multivi", "X_wnn", "X_joint"]]},
        {"id": "single_modality_present", "type": "must_have_obsm_keys",
         "keys": [["X_pca", "X_rna"]]},
        {"id": "celltype_obs", "type": "must_have_obs_keys",
         "keys": [["cell_type", "celltype", "cell_type_pred"]]},
        {"id": "joint_factors", "type": "obsm_dim_at_least",
         "obsm_key": "X_mofa", "min_dims": 5},
        {"id": "factor_variance_concentrated", "type": "mofa_factor_variance",
         "factor_obsm": "X_mofa",
         "min_factors": 5, "min_total_variance": 0.30},
        {"id": "joint_outperforms_single", "type": "joint_embedding_outperforms_single",
         "joint_obsm_pattern": r"X_(mofa|glue|multivi|wnn|joint)",
         "single_obsm_pattern": r"X_(pca|rna|atac)",
         "celltype_obs": "cell_type",
         "min_silhouette_improvement": 0.02,
         "rationale": "joint embedding should beat best single-modality silhouette by ≥0.02"},
    ],
}

D02 = {
    "id": "D02_peak_gene_linking",
    "layer": "D",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/multiome_pbmc",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 1500,
    "prompt": (
        "Using the same paired multiome dataset (pbmc_multiome.h5), link "
        "ATAC peaks to RNA gene expression: for each gene, identify ATAC "
        "peaks whose accessibility correlates with the gene's expression "
        "across cells. Store the resulting peak→gene links table in uns "
        "key peak_gene_links (aliases accepted: peak_to_gene, gene_peak, "
        "regulatory_links). The table must have columns naming the peak, "
        "the gene, the correlation/score, and the genomic distance from "
        "peak to gene TSS. A fraction of links should fall within ±10 kb "
        "(promoter-proximal). Modify the dataset in place."
    ),
    "checks": [
        {"id": "links_uns", "type": "must_have_uns_keys",
         "keys": [["peak_gene_links", "peak_to_gene", "gene_peak",
                    "regulatory_links"]]},
        {"id": "links_count_and_promoter_frac", "type": "peak_gene_link_count",
         "uns_key": ["peak_gene_links", "peak_to_gene", "gene_peak",
                      "regulatory_links"],
         "min_links": 500,
         "min_promoter_proximal_frac": 0.30,
         "rationale": "real peak-gene linking finds 1000s of links with ≥30% within ±10kb of TSS"},
    ],
}


# =====================================================================
# Layer E — Bulk RNA-seq → single-cell (2 tasks)
# Bulk2Single is uniquely ov; competitors must compose. ★ flagship.
# =====================================================================

E01 = {
    "id": "E01_bulk_deconvolution",
    "layer": "E",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/bulk_dg",
    "oracle":  f"{INPUTS}/bulk_dg/dg_single_normal.h5ad",
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Mouse dentate gyrus bulk RNA-seq (dg_bulk_normal.h5ad) paired "
        "with scRNA reference (dg_single_normal.h5ad, cell-type "
        "annotations in obs column cell_type).\n\n"
        "Estimate per-sample cell-type proportions with rigor:\n"
        "  1. Run AT LEAST TWO different deconvolution methods (e.g. "
        "CIBERSORTx, BayesPrism, MuSiC, Bisque, Bulk2Single, SCDC). "
        "Store each method's per-sample fractions as a separate uns "
        "entry whose name includes 'fractions' or 'deconv' (e.g. "
        "fractions_cibersort, fractions_bayesprism). Each entry must be "
        "a samples × cell-types DataFrame.\n"
        "  2. Methods should partially agree — per-cell-type fraction "
        "Pearson correlation across methods should reach a meaningful "
        "level. Disagreement is informative (uncertainty estimate) but "
        "near-zero correlation indicates one method is broken.\n"
        "  3. ALSO populate a primary `cell_type_fractions` uns entry "
        "OR `frac_<celltype>` obs columns from the chosen best method.\n\n"
        "Modify the bulk dataset in place and return it."
    ),
    "checks": [
        {"id": "fractions_present", "type": "deconv_fractions_sane",
         "fractions_obs_pattern": r"frac_|_fraction|cell_type_frac",
         "min_n_celltypes": 3,
         "sum_tolerance": 0.10,
         "rationale": "per-sample cell-type fractions must sum to ≈1 with ≥3 nonzero cell types"},
        {"id": "multi_method_agreement", "type": "multi_method_deconv_agreement",
         "fractions_keys_pattern": r"fractions|deconv|frac_",
         "min_methods": 2, "min_correlation": 0.40,
         "rationale": "≥2 deconv methods + per-cell-type fraction Pearson correlation ≥ 0.40"},
    ],
}

E03 = {
    "id": "E03_bulk_de_multi_method",
    "layer": "E",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/bulk_dg/dg_bulk_normal.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "This is a mouse hippocampus bulk RNA-seq dataset: 24 samples × "
        "4000 genes (already HVG-subset). Sample names in obs column "
        "`samples` encode the dissected sub-region — `dg_*` = dentate "
        "gyrus, `ca3_*` and `ca4_*` = CA3/CA4 subfields. Group the "
        "samples by region prefix (DG vs CA) and run differential "
        "expression with publication rigor.\n\n"
        "A reviewer will not accept a single-method ranking — bulk DE is "
        "sensitive to test choice (Welch's t / Wilcoxon variance "
        "assumptions vs DESeq2's negative-binomial vs limma's empirical "
        "Bayes vs pyDEG's bootstrap). Run AT LEAST TWO distinct DE "
        "methods and store each result table as a separate uns entry "
        "whose name identifies the method (e.g. `de_ttest`, `de_deseq2`, "
        "`de_pydeg`, `de_wilcoxon`). The standard `rank_genes_groups` "
        "uns slot must also be populated for one method.\n\n"
        "Each result table must include columns named for the gene "
        "identifier, the log-fold-change, and the adjusted p-value "
        "(padj / pvalue_adj / fdr / qvalue). Modify the dataset in place."
    ),
    "checks": [
        {"id": "rank_present", "type": "must_have_uns_keys",
         "keys": ["rank_genes_groups"]},
        {"id": "rank_substructure", "type": "uns_dict_keys",
         "uns_key": "rank_genes_groups",
         "must_have_subkeys": ["names", "pvals_adj"]},
        {"id": "multi_method_de_consistency", "type": "multi_method_de_overlap",
         "uns_pattern": r"rank_genes|de_|differential_expr|wilcoxon|t_test|ttest|deseq|limma|pydeg|edger",
         "min_methods": 2, "min_top_jaccard": 0.15, "top_n": 30,
         "rationale": "≥2 DE methods stored; per-comparison top-30 Jaccard ≥ 0.15. Threshold accommodates 2-method comparisons on small-n bulk where Wilcoxon p-values often saturate; agents with stronger method mixes (DESeq2 + limma + t-test) typically clear 0.30+"},
    ],
}

E04 = {
    "id": "E04_bulk_wgcna",
    "layer": "E",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/wgcna_5xfad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "5xFAD mouse Alzheimer-model bulk RNA-seq study (Mortazavi lab). "
        "The fixture directory contains two CSVs:\n"
        "  - expressionList.csv (≈50 MB! 192 samples × ~55k Ensembl "
        "gene IDs, raw count-like expression; rows=samples, "
        "columns=ENSMUSG ids)\n"
        "  - sampleInfo.csv (sample-level metadata: sample_id, Age in "
        "months, Tissue {Cortex, Hippocampus}, Sex, Genotype "
        "{5xFADHEMI, 5xFADWT})\n\n"
        "**Inspect this CSV with care.** It is WIDE — every row holds "
        "~55 000 numeric columns plus a header row of ENSMUSG ids "
        "(≈ 700 KB per line). `cat`, `head` (default 10 lines), and "
        "even `head -3` will dump several MB into your context. **Load "
        "it directly with pandas instead**:\n"
        "  `df = pd.read_csv('expressionList.csv', index_col=0, nrows=3)` "
        "and inspect via `df.shape`, `df.iloc[:3, :5]`, `df.dtypes.head()`. "
        "If you need a bash peek, narrow the columns: "
        "`head -3 expressionList.csv | cut -d, -f1-6`.\n\n"
        "Run weighted gene co-expression network analysis to identify "
        "co-expression modules and relate them to clinical traits. "
        "Produce a single AnnData (samples × genes) with all results "
        "stored on it:\n\n"
        "  1. Build the AnnData from the two CSVs — sample_id from "
        "sampleInfo aligned to expression rows; metadata columns become "
        "obs (Age, Tissue, Sex, Genotype). Reduce gene-space to the most "
        "variable ~2000 genes before network construction (gene-level "
        "variance / MAD ranking is standard).\n"
        "  2. A per-gene module assignment column in `var` whose name "
        "contains 'module' (e.g. `var['module_color']` with strings like "
        "'turquoise', 'blue', 'brown', or `var['wgcna_module']` with "
        "module integer/string IDs). Detect AT LEAST 3 distinct modules.\n"
        "  3. A module-trait correlation table in `uns` whose key "
        "contains 'module' and 'trait' / 'corr' (e.g. "
        "`uns['module_trait_corr']`). Traits should include at minimum "
        "Genotype (HEMI vs WT, encoded numerically) and Age. The table "
        "should have modules as rows and traits as columns, correlation "
        "values in the body.\n\n"
        "Save the result as `adata.h5ad` in the working directory."
    ),
    "checks": [
        {"id": "module_assignment", "type": "must_have_var_keys",
         "keys": [["module_color", "wgcna_module", "module", "modules",
                    "module_id", "module_label"]]},
        {"id": "n_modules", "type": "var_unique_count",
         "var_key": ["module_color", "wgcna_module", "module", "modules",
                       "module_id", "module_label"],
         "min_unique": 3, "max_unique": 50,
         "rationale": "≥3 distinct WGCNA modules expected on this dataset"},
        {"id": "module_trait_table", "type": "must_have_uns_keys",
         "keys": [["module_trait_corr", "module_trait", "moduleTraitCor",
                    "wgcna_module_trait", "module_trait_correlation"]]},
        {"id": "wgcna_real_call",
         "type": "tool_output_evidence",
         "description": "real WGCNA / pyWGCNA invocation",
         "patterns": [
             r"\bPyWGCNA\b",
             r"\bpyWGCNA\b",
             r"ov\.bulk\.pyWGCNA\b",
             r"ov\.bulk\.readWGCNA\b",
             # Real WGCNA prints these stages in stdout
             r"\bsoft[-_ ]threshold\b.*\bpower\b",
             r"\bTOM[-_ ]?(matrix|similarity)\b",
             r"\bcutreeDynamic\b",
             r"\bcutreeHybrid\b",
             r"\bgoodSamplesGenes\b",
             r"\bmergeCloseModules\b",
             r"Calculating\s+\d+\s+module\s+eigengenes",
             r"\bcalculate_dynamicMods\b",
             r"\bdynamicMods\b",
             r"\bmoduleColors\b",
             r"Going\s+through\s+the\s+merge\s+tree",
         ],
         "rationale": "must call a real WGCNA implementation (PyWGCNA / ov.bulk.pyWGCNA) — hand-rolled correlation clustering on inline-defined sigs is not acceptable"},
    ],
}

E05 = {
    "id": "E05_bulk_metabol_multivariate",
    "layer": "E",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/metabol_cachexia.csv",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Human cachexia NMR metabolomics study: 77 patients × 63 urinary "
        "metabolites in CSV form. Columns are 'Patient ID', 'Muscle "
        "loss' (group label: 'cachexic' vs 'control'), then one column "
        "per metabolite with intensity values.\n\n"
        "Perform a publication-grade metabolomics multivariate analysis:\n\n"
        "  1. Load the CSV, structuring it as an AnnData with one row per "
        "patient and one column per metabolite. Patient group goes into "
        "obs column `group` (alias `class` accepted) with values "
        "'cachexic' / 'control'.\n"
        "  2. Preprocess: handle missing values (impute or drop), apply "
        "PQN (probabilistic quotient) normalization, log-transform, and "
        "Pareto or auto-scale.\n"
        "  3. Run AT LEAST TWO multivariate analyses for group "
        "discrimination — e.g. PCA, PLS-DA, OPLS-DA. Store each "
        "embedding as obsm key whose name contains the method "
        "(`X_pca`, `X_plsda`, `X_oplsda`, `X_pls_da`).\n"
        "  4. The PLS/OPLS embedding's first component must separate "
        "cachexic vs control well — silhouette score on the first 2 "
        "components, treating `group` as labels, must be ≥ 0.20 (these "
        "groups are biologically separable on this dataset; a value "
        "near zero means the analysis collapsed the signal).\n\n"
        "Save the resulting AnnData as `adata.h5ad` in the working "
        "directory."
    ),
    "checks": [
        {"id": "group_obs", "type": "must_have_obs_keys",
         "keys": [["group", "class", "Muscle loss", "muscle_loss"]]},
        {"id": "two_distinct_groups", "type": "obs_unique_count",
         "obs_key": ["group", "class", "Muscle loss", "muscle_loss"],
         "min_unique": 2, "max_unique": 2},
        {"id": "multivariate_obsm", "type": "must_have_obsm_keys_regex",
         "patterns": [r"^X_(pca|plsda|opls.?da|pls.?da|pls)$"]},
        {"id": "two_distinct_methods", "type": "obsm_count_matching_regex",
         "pattern": r"^X_(pca|plsda|opls.?da|pls.?da|pls|tsne|umap|fa)$",
         "min_count": 2,
         "rationale": "≥2 distinct multivariate embeddings (e.g. PCA + PLS-DA)"},
        {"id": "supervised_separation", "type": "obsm_celltype_silhouette",
         "obsm_key": "X_plsda",
         "celltype_obs": "group",
         "min_silhouette": 0.15,
         "rationale": "PLS-DA on cachexic vs control on the cachexia dataset reaches silhouette ≥0.15 — small-sample metabolomics rarely exceeds 0.20 even with strong group separation"},
    ],
}


# =====================================================================
# Layer F — RNA velocity / trajectory (2 tasks)
# Multi-mode dispatch. ★ ov flagship.
# =====================================================================

F01 = {
    "id": "F01_multi_mode_velocity",
    "layer": "F",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pancreas_endo_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 900,
    "prompt": (
        "Compute RNA velocity for this pancreas endocrinogenesis dataset "
        "using TWO different velocity-estimation modes or algorithms "
        "(e.g. stochastic vs dynamical, scvelo vs graphvelo, vs "
        "latentvelo).\n\n"
        "Performance note: this raw dataset has ~28000 genes and a full-"
        "gene velocity fit can run for tens of minutes single-threaded. "
        "Subset the dataset to ~2000 highly-variable genes before any "
        "compute-heavy velocity step (e.g. recover_dynamics or full "
        "dynamical fit). Standard preprocessing first (HVG selection, "
        "moments) keeps runtime within the per-task wallclock budget.\n\n"
        "Storage targets (reviewer will look for these specific keys):\n"
        "  - layer named velocity (mode-1 velocity matrix)\n"
        "  - layer named velocity_alt (mode-2 velocity matrix; aliases "
        "accepted: velocity_dynamical, velocity_S, velocity_stochastic)\n"
        "  - obsm key velocity_umap (UMAP-projected vectors from mode 1; "
        "alias velocity_S_umap accepted)\n"
        "  - uns key velocity_modes — a list of the two mode-name strings\n\n"
        "The dataset has spliced and unspliced layers, a UMAP embedding "
        "in obsm key X_umap, and cell-type annotations in obs column "
        "clusters with a Ductal progenitor cluster. Modify the dataset "
        "in place."
    ),
    "checks": [
        {"id": "velocity_layer", "type": "must_have_layers_regex",
         # omicverse tutorial conventions:
         #   dynamo     → 'velocity'
         #   scvelo     → 'velocity_S'
         #   latentvelo → 'velo_latentvelo'
         #   graphvelo  → 'velo_graphvelo'
         # Plus agents may generalise to 'velocity_<method>'.
         "patterns": [r"^(velocity|velo)(_[A-Za-z0-9]+)?$"]},
        {"id": "velocity_umap_obsm", "type": "must_have_obsm_keys_regex",
         # Match any of the above velocity layers projected to UMAP:
         # velocity_umap / velocity_S_umap / velo_latentvelo_umap /
         # velo_graphvelo_umap / velocity_<method>_umap
         "patterns": [r"^(velocity|velo)(_[A-Za-z0-9]+)?_umap$"]},
        {"id": "modes_directionally_agree", "type": "velocity_modes_consistency",
         "min_mean_cosine": 0.05,
         "rationale": "scvelo stochastic-vs-dynamical agreement is dataset-dependent (0.05-0.8); fabrication averages 0"},
        {"id": "velocity_flows_from_root", "type": "velocity_root_anchoring",
         "root_cluster": "Ductal", "groupby_obs": "clusters",
         "basis_obsm": "X_umap",
         # Regex — `_check_velocity_root_anchoring` falls back to scanning
         # obsm with this regex if the literal key doesn't exist.
         "velocity_obsm": r"^(velocity|velo)(_[A-Za-z0-9]+)?_umap$",
         "min_mean_outward_cosine": 0.04,
         "rationale": ("Ductal cells are progenitors; velocity should flow toward terminal "
                       "alpha/beta/delta. Threshold 0.10 → 0.04 because the pancreas-endo "
                       "fixture has weak directional signal — scvelo and dynamo backends "
                       "produce near-orthogonal mean flows on this dataset (typical 0.04-0.10 "
                       "even when both modes agree). 0.04 still excludes truly reversed or "
                       "random flows but accepts the modest signal real backends produce.")},
        # `velocity_confidence_present` removed in v1.1: scvelo and dynamo
        # report confidence under different keys, and ov.single.Velo doesn't
        # always surface one — too noisy as a pass/fail check.
    ],
}

F02 = {
    "id": "F02_multi_method_trajectory",
    "layer": "F",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pancreas_endo_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 900,
    "prompt": (
        "Infer cell trajectory and pseudotime on this pancreas dataset "
        "using TWO different pseudotime methods (e.g. graph-diffusion-based, "
        "potential-based, RNA-velocity-based, or partition-based). Cluster "
        "labels are in obs column clusters; the origin (root) cluster is "
        "named Ductal.\n\n"
        "Storage targets (reviewer will look for these specific keys):\n"
        "  - two distinct obs columns whose names contain 'pseudotime' "
        "(e.g. dpt_pseudotime, palantir_pseudotime, via_pseudotime). "
        "Both must be biologically anchored at the Ductal root (lower "
        "pseudotime in Ductal than in terminal cells), and the two "
        "columns must not be perfectly correlated.\n"
        "  - uns key trajectory_methods — a list of the two method-name "
        "strings.\n"
        "  - identify ≥2 terminal/branch states (alpha/beta/delta endocrine "
        "lineages branching from Ductal) and store them in either:\n"
        "    * an uns key whose name contains 'terminal' or 'branch' or "
        "'fate' or 'macrostate' (e.g. uns['terminal_states'], "
        "uns['branches'], uns['cellrank_macrostates']) holding a list "
        "of cluster names; OR\n"
        "    * an obs categorical column named 'terminal_state' (or alias) "
        "with cluster labels for each cell.\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "two_pseudotime_obs", "type": "obs_two_distinct_pseudotime",
         "min_distinct_pseudotime_cols": 2},
        {"id": "pseudotime_anchored_at_root", "type": "pseudotime_root_agreement",
         "root_cluster": "Ductal", "groupby_obs": "clusters",
         "min_root_to_other_gap": 0.10,
         "rationale": "median pseudotime in Ductal must be ≥10% of overall range below median elsewhere — both methods"},
        {"id": "branch_detection", "type": "trajectory_branch_detection",
         "branch_uns_pattern": r"terminal|branch|fate|cellrank|macrostate",
         "min_terminals": 2,
         "rationale": "publication-grade trajectory inference identifies ≥2 terminal/branch states (alpha/beta/delta in pancreas)"},
        {"id": "pseudotime_methods_agree", "type": "pairwise_pseudotime_correlation",
         "pseudotime_obs_pattern": r"pseudotime$|_dpt$|^pseudotime_",
         "min_pairwise_spearman": 0.4,
         "min_methods": 2,
         "rationale": "every pair of pseudotime methods must reach Spearman ρ ≥ 0.4 — extra methods are welcome but each new one must agree with the others"},
    ],
}


# =====================================================================
# Layer G — Microbiome 16S amplicon (2 tasks)
# Cross-modality coverage; G02 tests methodological rigor.
# =====================================================================

G01 = {
    "id": "G01_16s_diversity",
    "layer": "G",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/microbiome_16s/mothur_sop_16s.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "This is a 16S rRNA amplicon abundance table (mothur SOP demo: "
        "20 samples × 598 ASVs/OTUs) with sample metadata in obs columns "
        "sample, day, group.\n\n"
        "Compute publication-quality diversity analysis with these "
        "specific storage targets:\n\n"
        "  1. Alpha diversity per sample as obs columns:\n"
        "     - shannon\n"
        "     - observed_otus (alias observed_asvs)\n"
        "     - simpson\n"
        "  2. Beta diversity n×n distance matrix in obsp:\n"
        "     - obsp key beta_distance (aliases accepted: bray_curtis, "
        "unifrac, weighted_unifrac)\n"
        "  3. STATISTICAL TESTS (required for any cohort study):\n"
        "     - test group-level shannon difference (Kruskal-Wallis or "
        "Mann-Whitney across the obs[group] groups). Store the result as "
        "an uns dict whose key contains 'alpha_test' or 'kruskal' or "
        "'mann_whitney' (e.g. uns['alpha_test_kruskal'] = "
        "{'pvalue': p, 'statistic': stat, 'metric': 'shannon'}).\n"
        "     - test group-level beta-diversity separation (PERMANOVA or "
        "ANOSIM). Store the result as an uns dict whose key contains "
        "'permanova' or 'anosim' or 'beta_test' (e.g. "
        "uns['beta_permanova'] = {'pvalue': p, 'pseudo_F': F}).\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "alpha_present", "type": "alpha_diversity_present",
         "metrics": ["shannon", "observed_otus", "observed_asvs", "simpson"],
         "rationale": "alpha diversity is a standard 16S output; ≥1 metric must be populated"},
        {"id": "beta_present", "type": "beta_diversity_present",
         "uns_or_obsp_keys": ["beta_distance", "bray_curtis", "unifrac",
                                "weighted_unifrac", "beta_div"],
         "rationale": "pairwise beta diversity matrix needed for sample comparison"},
        {"id": "alpha_group_test", "type": "alpha_diversity_group_test",
         "group_obs": "group", "metric": "shannon",
         "max_pvalue": None,  # rigor: just require the test be RUN and stored
         "test_uns_key_pattern": r"alpha_test|kruskal|mann_whitney|group_diff",
         "rationale": "publication-grade output reports a statistical test of group-level alpha diversity (Kruskal-Wallis / Mann-Whitney), not just per-sample numbers — even if p>0.05 on the demo dataset"},
        {"id": "beta_permanova", "type": "beta_diversity_permanova",
         "group_obs": "group", "max_pvalue": 0.05,
         "test_uns_key_pattern": r"permanova|anosim|beta_test",
         "rationale": "PERMANOVA / ANOSIM tests group separation in beta-diversity space — required for any cohort 16S study"},
    ],
}

G02 = {
    "id": "G02_multi_method_da",
    "layer": "G",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/microbiome_16s/mothur_sop_16s.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Same 16S dataset as G01, with sample groups in obs column group. "
        "Identify differentially-abundant taxa between groups using TWO "
        "different statistical methods (e.g. Wilcoxon rank-sum, DESeq2, "
        "ANCOM-BC, LEfSe, ALDEx2 — any two distinct DA frameworks). "
        "Store each method's result table as a separate uns entry whose "
        "name contains a recognizable identifier (da_wilcoxon, "
        "da_deseq2, da_ancombc, etc.). Each table must include p-value/"
        "FDR and log-fold-change columns and have taxa as the index or "
        "first column. Modify the dataset in place.\n\n"
        "After running both methods, the top-ranked taxa should partially "
        "agree across methods (a fully diverging result indicates a "
        "framework-specific artifact)."
    ),
    "checks": [
        {"id": "multi_method_da_results", "type": "da_method_overlap",
         "uns_pattern": r"da_|differential_abundance|wilcox|deseq|ancombc|aldex|lefse",
         "min_methods": 2,
         "min_jaccard": 0.20,
         "top_n": 30,
         "rationale": "≥2 DA methods + their top-30 significant taxa Jaccard ≥ 0.20 — frameworks should agree partially on real signal"},
    ],
}


G03 = {
    "id": "G03_16s_ordination_pcoa",
    "layer": "G",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/microbiome_16s/mothur_sop_16s.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Same 16S dataset as G01 — 20 samples × 598 ASVs/OTUs, with "
        "sample groups in obs column `group` (Early vs Late) and per-"
        "ASV taxonomy in var (domain..genus..species).\n\n"
        "Run a publication-quality beta-diversity ordination workflow:\n\n"
        "  1. Compute the pairwise Bray-Curtis distance matrix between "
        "samples (after appropriate count normalization / rarefaction). "
        "Store it in obsp under a key whose name contains 'braycurtis' "
        "or 'beta_distance'.\n"
        "  2. Run principal-coordinates analysis (PCoA) on the distance "
        "matrix. Store the per-sample 2D-or-3D coordinates in obsm under "
        "a key whose name contains 'pcoa' (e.g. `obsm['X_pcoa']` or "
        "`obsm['braycurtis_pcoa']`). Store the per-axis variance "
        "explained as an array under a uns key whose name contains "
        "'pcoa' and 'var' (e.g. "
        "`uns['braycurtis_pcoa_var'] = [0.32, 0.18, 0.09]`).\n"
        "  3. Test whether samples cluster by `group` in ordination "
        "space. Store the test result as a uns dict whose key contains "
        "'permanova' or 'anosim' or 'ordination_test', with at minimum "
        "fields `pvalue` and `statistic`/`pseudo_F`/`R`.\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "beta_dist_present", "type": "beta_diversity_present",
         "uns_or_obsp_keys": ["braycurtis", "bray_curtis", "beta_distance",
                                "unifrac", "weighted_unifrac"],
         "rationale": "PCoA needs a distance matrix; beta-diversity must be computed"},
        {"id": "pcoa_obsm", "type": "must_have_obsm_keys_regex",
         "patterns": [r"pcoa|PCoA"],
         "rationale": "PCoA coordinates per sample stored in obsm"},
        {"id": "pcoa_variance", "type": "must_have_uns_keys_regex",
         "patterns": [r"pcoa.*var|var.*pcoa|variance_explained|expl"],
         "rationale": "publication-grade PCoA reports per-axis variance explained"},
        {"id": "group_separation", "type": "beta_diversity_permanova",
         "group_obs": "group", "max_pvalue": 0.05,
         "test_uns_key_pattern": r"permanova|anosim|ordination_test|beta_test",
         "rationale": "PERMANOVA / ANOSIM verifies the visual ordination separation is statistically real"},
    ],
}

G04 = {
    "id": "G04_16s_taxonomic_collapse",
    "layer": "G",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/microbiome_16s/mothur_sop_16s.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Same 16S dataset as G01 — 20 samples × 598 ASVs with 7-rank "
        "taxonomy in var columns (domain, phylum, class, order, family, "
        "genus, species). Sample groups in obs[`group`].\n\n"
        "Aggregate the ASV counts to higher taxonomic levels and quantify "
        "how aggregation changes the analysis:\n\n"
        "  1. Collapse to genus level: produce a per-sample × per-genus "
        "abundance table (sum of ASV counts within the same `var['genus']` "
        "value). Store it as a uns DataFrame whose key contains 'genus' "
        "(e.g. `uns['abundance_genus']` or `uns['collapsed_genus']`), "
        "with samples as rows and genus identifiers as columns.\n"
        "  2. Collapse to phylum level the same way; store under a uns "
        "key whose name contains 'phylum'.\n"
        "  3. Compute Shannon alpha diversity at BOTH the ASV level "
        "(per sample, in obs column `shannon` if not already present) "
        "AND the genus level (per sample, in obs column whose name "
        "contains 'shannon' AND 'genus', e.g. `shannon_genus`). The "
        "genus-level Shannon should be lower than ASV-level Shannon for "
        "the same sample (information collapses on aggregation).\n"
        "  4. Identify the top-5 most abundant genera (overall, summed "
        "across samples). Store the list as a uns array whose key "
        "contains 'top' and 'genus' (e.g. `uns['top5_genera']`).\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "genus_table", "type": "must_have_uns_keys_regex",
         "patterns": [r"(abundance|collaps(e|ed)|table|counts).*genus|genus.*(abundance|collaps|table|counts)"],
         "rationale": "genus-level abundance table required"},
        {"id": "phylum_table", "type": "must_have_uns_keys_regex",
         "patterns": [r"(abundance|collaps(e|ed)|table|counts).*phylum|phylum.*(abundance|collaps|table|counts)"],
         "rationale": "phylum-level abundance table required"},
        {"id": "asv_shannon", "type": "must_have_obs_keys",
         "keys": [["shannon"]],
         "rationale": "ASV-level alpha diversity present"},
        {"id": "genus_shannon", "type": "must_have_obs_keys_regex",
         "patterns": [r"shannon.*genus|genus.*shannon"],
         "rationale": "genus-level alpha diversity present (collapsed)"},
        {"id": "shannon_ordering", "type": "obs_value_comparison",
         "obs_a": "shannon",
         "obs_b_pattern": r"shannon.*genus|genus.*shannon",
         "comparison": "a_gt_b",
         "min_fraction": 0.6,
         "rationale": "≥60% of samples should have shannon_ASV > shannon_genus (information collapses on taxonomic aggregation)"},
        {"id": "top_genera", "type": "must_have_uns_keys_regex",
         "patterns": [r"top.*genus|top.*genera|top_taxa"],
         "rationale": "summary list of dominant taxa is a standard 16S deliverable"},
    ],
}


# =====================================================================
# Round-2 task additions — derived from omicverse_guide tutorial coverage
# audit. Each task uses an ov-flagship API that the registry / skill
# scanner indexes, so the omicverse arm has a discoverable advantage
# over the baseline arm.
# =====================================================================

B06 = {
    "id": "B06_reference_annotation",
    "layer": "B",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "PBMC scRNA-seq with curated `cell_type` ground-truth labels in "
        "obs. Simulate a reference-based annotation pipeline:\n\n"
        "  1. Standard preprocessing if not already done: QC, "
        "normalize+log1p, HVG, PCA, neighbors, UMAP.\n"
        "  2. Split the dataset 70/30 (or any fixed deterministic split) "
        "into a *reference* AnnData carrying `cell_type` and a *query* "
        "AnnData with `cell_type` set to NaN / removed.\n"
        "  3. Transfer cell-type labels from reference to query using a "
        "reference-based annotation method (label transfer / "
        "scanpy.tl.ingest / a reference-projection wrapper). Store the "
        "predicted labels back into the query under obs column "
        "`cell_type_predicted` (alias `predicted_celltype`, "
        "`celltype_pred`, `cell_type_transfer` accepted).\n"
        "  4. Concatenate query + reference back into a single AnnData "
        "with `cell_type` retained on the reference half and "
        "`cell_type_predicted` populated on the query half.\n\n"
        "The reviewer will check that ≥70% of query cells have a "
        "non-empty predicted label AND that prediction agreement with "
        "the held-out ground-truth `cell_type` reaches a meaningful "
        "level (Jaccard / accuracy at the cell-type level). Modify "
        "the dataset in place."
    ),
    "checks": [
        {"id": "predicted_obs", "type": "must_have_obs_keys",
         "keys": [["cell_type_predicted", "predicted_celltype",
                    "celltype_pred", "cell_type_transfer"]]},
        {"id": "ref_celltype_present", "type": "must_have_obs_keys",
         "keys": [["cell_type"]]},
        {"id": "predicted_coverage", "type": "obs_unique_count",
         "obs_key": ["cell_type_predicted", "predicted_celltype",
                       "celltype_pred", "cell_type_transfer"],
         "min_unique": 3, "max_unique": 50,
         "rationale": "PBMC has ≥3 distinct cell types; predictions should not collapse to one"},
    ],
}

B08 = {
    "id": "B08_aucell_pathway_scoring",
    "layer": "B",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "PBMC scRNA-seq dataset (curated `cell_type` labels in obs).\n\n"
        "Compute **per-cell gene-set activity scores** that distinguish "
        "the two canonical PBMC populations annotated in this fixture: "
        "**T cells** and **B cells**.\n\n"
        "Any per-cell gene-set scoring approach is acceptable. Inline "
        "gene-symbol signatures are fine.\n\n"
        "Storage targets (modify dataset in place):\n"
        "  1. obs column whose name starts with `aucell_` / `score_` / "
        "`signature_` AND contains `t` (or `tcell` / `t_cell`) — must "
        "peak in T cells.\n"
        "  2. obs column whose name starts with `aucell_` / `score_` / "
        "`signature_` AND contains `b` (or `bcell` / `b_cell`) — must "
        "peak in B cells.\n\n"
        "The check verifies cell-type specificity (T-cell signature "
        "peaks in T-cell family; B-cell signature peaks in B-cell family) "
        "— the algorithm choice is up to you."
    ),
    "checks": [
        {"id": "score_columns_regex", "type": "must_have_obs_keys_regex",
         "patterns": [r"^(aucell|score|signature|auc)_"]},
        {"id": "score_count_matching",
         "type": "obs_count_matching_regex",
         "pattern": r"^(aucell|score|signature|auc)_",
         "min_count": 2,
         "rationale": "≥2 distinct per-cell gene-set scores (T-cell + B-cell)"},
        {"id": "tb_celltype_specificity",
         "type": "aucell_celltype_specificity",
         "celltype_obs": "cell_type",
         "score_pattern": r"^(aucell|score|signature|auc)_",
         "min_celltype_specific": 2,
         "min_z_gap": 0.3,
         "rationale": "T-cell score must peak in T-cell family, B-cell score "
                       "must peak in B-cell family. Family-vs-non-family z-gap "
                       "≥0.3σ — proves biology, not random numbers."},
    ],
}

B09 = {
    "id": "B09_cnmf_gene_programs",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 100, "wallclock_s": 600,
    "prompt": (
        "PBMC scRNA-seq dataset. Run consensus non-negative matrix "
        "factorization (cNMF) — or an equivalent NMF-with-ensembling "
        "scheme — to discover ≥4 gene-expression programs (GEPs) and "
        "their per-cell usage.\n\n"
        "Required outputs (modify the dataset in place):\n"
        "  1. obsm key `X_cnmf` (alias `cnmf_usages`, `gep_usages`) — "
        "per-cell program usages of shape (n_cells, n_programs).\n"
        "  2. uns key whose name contains `cnmf` and `programs` (or "
        "`spectra`, `gene_loadings`) — per-program top-gene loadings "
        "(programs × top genes DataFrame, or list of top-gene lists).\n"
        "  3. obs column `cnmf_dominant_program` (alias "
        "`dominant_program`) — index/label of the highest-usage "
        "program per cell.\n\n"
        "≥4 distinct programs (`unique(cnmf_dominant_program)` ≥ 4)."
    ),
    "checks": [
        {"id": "cnmf_obsm", "type": "must_have_obsm_keys_regex",
         "patterns": [r"X_cnmf|cnmf_usages|gep_usages"]},
        {"id": "cnmf_programs_uns", "type": "must_have_uns_keys_regex",
         "patterns": [r"cnmf.*program|cnmf.*spectra|cnmf.*gene_loading|gep_program"]},
        {"id": "dominant_program", "type": "must_have_obs_keys",
         "keys": [["cnmf_dominant_program", "dominant_program"]]},
        {"id": "n_programs", "type": "obs_unique_count",
         "obs_key": ["cnmf_dominant_program", "dominant_program"],
         "min_unique": 4, "max_unique": 25,
         "rationale": "cNMF should discover ≥4 distinct gene programs on PBMC"},
        {"id": "cnmf_real_call", "type": "tool_output_evidence",
         "description": "real cNMF library invocation",
         "patterns": [
             r"\bimport\s+cnmf\b",
             r"\bfrom\s+cnmf\b",
             r"\bov\.single\.pycnmf\b",
             r"\bov\.single\.cNMF\b",
             r"\bcNMF\s*\(",
             r"\[cNMF\]",
             # Method-call markers from real cnmf workflows (factorize → combine
             # → consensus → load_results → get_results) that fabricated
             # NMF runs cannot fake without actually instantiating the class
             r"\.factorize\s*\(",
             r"\.consensus\s*\(",
             r"\.k_selection_plot\s*\(",
             r"\.load_results\s*\(",
             r"\bcombine_nmf\b",
             r"\bget_results_rfc\b",
             r"\bget_results\s*\(\s*adata",
         ],
         "rationale": "must call real cnmf package or omicverse.single.pycnmf — hand-rolled NMF / PCA aliased as X_cnmf is not acceptable"},
    ],
}

C04 = {
    "id": "C04_spatial_trajectory_stt",
    "layer": "C",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/visium_dlpfc_151676",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Same DLPFC Visium fixture as C01. Build a spatially-aware "
        "trajectory / pseudotime that respects the layered cortical "
        "structure (cell-state transition along WM → L6 → L5 → L4 → "
        "L3 → L2 / L1). Use a spatial-aware trajectory method that "
        "models inter-domain transitions on physical coordinates "
        "(spatial pseudotime / spatial transition tensor / spatial "
        "RNA velocity — any approach that produces a spot-ordering "
        "and a domain-to-domain transition matrix).\n\n"
        "Storage targets:\n"
        "  1. obsm key `spatial` with 2D physical coordinates.\n"
        "  2. obs column `spatial_pseudotime` (alias `pseudotime_spatial`, "
        "`pseudotime`) with per-spot ordering.\n"
        "  3. obs column `region` or `spatial_domain` with discrete "
        "domain labels (≥4 distinct).\n"
        "  4. uns key whose name contains `transition` or `tensor` or "
        "`trajectory` with the per-domain transition statistics.\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "spatial_obsm", "type": "must_have_obsm_keys",
         "keys": ["spatial"]},
        {"id": "pseudotime_obs", "type": "must_have_obs_keys",
         "keys": [["spatial_pseudotime", "stt_pseudotime",
                    "pseudotime_spatial", "pseudotime"]]},
        {"id": "spatial_domain_obs", "type": "must_have_obs_keys",
         "keys": [["region", "spatial_domain", "domain", "clusters"]]},
        {"id": "domain_unique_count", "type": "obs_unique_count",
         "obs_key": ["region", "spatial_domain", "domain", "clusters"],
         "min_unique": 4, "max_unique": 12,
         "rationale": "DLPFC has ~7 cortical layers + WM"},
        {"id": "transition_uns", "type": "must_have_uns_keys_regex",
         "patterns": [r"transition|tensor|stt|trajectory"]},
        {"id": "stt_real_call", "type": "tool_output_evidence",
         "description": "real STT (spatial-transition-tensor) invocation",
         "patterns": [
             r"\bov\.space\.STT\b",
             r"\bimport\s+stt\b",
             r"\bfrom\s+stt\b",
             r"\[STT\]",
             r"\bstt\.STT\(",
             # STT internal pipeline functions and outputs (legitimate
             # quiet runs that don't print [STT] but do call the real API)
             r"\btl\.dynamical_iteration\b",
             r"\bdynamical_analysis\b",
             r"\.train\(\s*n_states\s*=",
             r"\bobsp\[\s*['\"]stt[_a-z]*['\"]",
             r"\buns\[\s*['\"]stt_transition",
             r"Constructing\s+tensor",
             r"Computing\s+transition\s+matrix",
         ],
         "rationale": "must invoke ov.space.STT or the standalone stt package — uns['stt_*'] keys with hand-rolled stats are not acceptable"},
    ],
}

E06 = {
    "id": "E06_pathway_enrichment",
    "layer": "E",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/bulk_dg/dg_bulk_normal.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Mouse hippocampus bulk RNA-seq (24 samples × 4000 HVGs; "
        "samples obs column encodes region — `dg_*` vs `ca3_*`/`ca4_*`). "
        "Run differential expression DG-vs-CA, then **pathway / "
        "gene-set enrichment** on the ranked DE result using two "
        "complementary databases (e.g. KEGG + GO biological process, "
        "or Hallmark + Reactome). Report at least one significantly-"
        "enriched pathway per database (FDR ≤ 0.10).\n\n"
        "Storage targets (modify dataset in place):\n"
        "  1. uns key `rank_genes_groups` (the DE ranking).\n"
        "  2. uns key whose name contains `enrichment` or `gsea` or "
        "`enrich_` — a DataFrame whose rows are pathways and columns "
        "include at minimum a pathway-name field, a database/source "
        "field, an enrichment-statistic (NES / odds-ratio / score), "
        "and an adjusted p-value (FDR / padj).\n"
        "  3. ≥2 distinct database/source values inside that table "
        "(e.g. `database` column with both 'KEGG' and 'GO_BP', or "
        "two separate tables `enrichment_kegg` + `enrichment_go`).\n"
        "  4. ≥1 row with adjusted p-value ≤ 0.10."
    ),
    "checks": [
        {"id": "rank_present", "type": "must_have_uns_keys",
         "keys": ["rank_genes_groups"]},
        {"id": "enrichment_table", "type": "must_have_uns_keys_regex",
         "patterns": [r"enrich|gsea"]},
    ],
}

E07 = {
    "id": "E07_ppi_network",
    "layer": "E",
    "difficulty": "medium",
    "fixture": f"{INPUTS}/bulk_dg/dg_bulk_normal.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Same mouse hippocampus bulk RNA-seq as E03 / E06 (24 samples "
        "× 4000 HVGs, region in obs[`samples`]). Identify the top-30 "
        "most-significantly DE genes between DG and CA, then build a "
        "**protein-protein interaction (PPI) network** for those "
        "genes (STRING-style: edges weighted by interaction "
        "confidence).\n\n"
        "Storage targets (modify dataset in place):\n"
        "  1. uns key `rank_genes_groups`.\n"
        "  2. uns key whose name contains `ppi` or `interaction` or "
        "`network` — a DataFrame whose rows are interactions and "
        "columns include at minimum: source/protein_a / partner_a, "
        "target/protein_b / partner_b, confidence/score / weight.\n"
        "  3. The PPI table must contain ≥5 edges (otherwise the "
        "network is too sparse to interpret).\n"
        "  4. uns key whose name contains `hub` or `degree` — a "
        "per-gene degree / centrality summary (top-degree hub genes)."
    ),
    "checks": [
        {"id": "rank_present", "type": "must_have_uns_keys",
         "keys": ["rank_genes_groups"]},
        {"id": "ppi_table", "type": "must_have_uns_keys_regex",
         "patterns": [r"ppi|interaction|network"]},
        {"id": "hub_summary", "type": "must_have_uns_keys_regex",
         "patterns": [r"hub|degree|central"]},
        {"id": "ppi_real_call", "type": "tool_output_evidence",
         "description": "real STRINGdb / PPI library invocation",
         "patterns": [
             r"\bimport\s+stringdb\b",
             r"\bfrom\s+stringdb\b",
             r"\bstringdb\.",
             r"\bov\.bulk\.string_interaction\b",
             r"\bov\.bulk\.pyPPI\b",
             r"requests\.\w+.*string-db\.org",
             r"\[STRING(?:db)?\]",
             r"\bgseapy\.enrichr.*STRING\b",
         ],
         "rationale": "must call STRINGdb (or omicverse PPI wrapper) — hand-rolled co-expression PPI is not acceptable"},
    ],
}

F03 = {
    "id": "F03_paga_pseudotime",
    "layer": "F",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pancreas_endo_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Pancreas endocrinogenesis dataset (same as F01). Compute a "
        "trajectory analysis combining **PAGA cluster connectivity** "
        "with **diffusion pseudotime (DPT)** anchored at the Ductal "
        "progenitor population.\n\n"
        "Pipeline (modify dataset in place):\n"
        "  1. Preprocess: HVG (≤2000), normalize+log1p, PCA, neighbors, "
        "UMAP.\n"
        "  2. Cluster (leiden or any) → store labels in obs[`clusters`] "
        "if not already present.\n"
        "  3. PAGA on clusters → store the connectivity in uns[`paga`] "
        "with at minimum `connectivities` and `groups` keys.\n"
        "  4. Pick a Ductal-cluster cell as iroot, run DPT → store "
        "`dpt_pseudotime` in obs.\n"
        "  5. Verify pseudotime monotonicity along the developmental "
        "axis: median pseudotime in Ductal cluster should be the "
        "lowest among major clusters."
    ),
    "checks": [
        {"id": "neighbors", "type": "must_have_uns_keys",
         "keys": ["neighbors"]},
        {"id": "paga", "type": "must_have_uns_keys",
         "keys": ["paga"]},
        {"id": "paga_connectivities", "type": "uns_dict_keys",
         "uns_key": "paga",
         "must_have_subkeys": ["connectivities"]},
        {"id": "pseudotime_obs", "type": "must_have_obs_keys",
         "keys": [["dpt_pseudotime", "pseudotime"]]},
    ],
}

G05 = {
    "id": "G05_phylogenetic_diversity",
    "layer": "G",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/microbiome_16s/mothur_sop_16s.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Same 16S dataset as G01-G04 — 20 samples × 598 ASVs with the "
        "ASV centroid sequence stored in `var['sequence']` and group "
        "labels in `obs['group']`.\n\n"
        "Compute **phylogenetic-tree-aware diversity metrics**:\n\n"
        "  1. Build a phylogenetic tree from the ASV sequences. Multi-"
        "sequence alignment (mafft / scikit-bio) → distance matrix → "
        "neighbor-joining or any tree-builder. Store the Newick string "
        "in uns under a key whose name contains 'tree' or 'phylogen' "
        "(e.g. `uns['tree_newick']` or `uns['phylogeny']`).\n"
        "  2. Per-sample **Faith's phylogenetic diversity** "
        "(sum of branch lengths spanned by present ASVs) → write into "
        "obs column `faith_pd` (alias `pd_whole_tree`, "
        "`phylogenetic_diversity`).\n"
        "  3. Per-sample-pair **UniFrac distance** (weighted or "
        "unweighted) → store the n×n distance matrix in obsp under a "
        "key whose name contains 'unifrac'.\n"
        "  4. Test whether group separation is statistically real on "
        "the UniFrac distance via PERMANOVA / ANOSIM — store result "
        "in uns dict with `pvalue` and `statistic` (key contains "
        "'unifrac' and 'permanova' or 'anosim').\n\n"
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "tree_newick", "type": "must_have_uns_keys_regex",
         "patterns": [r"tree|phylogen|newick"]},
        {"id": "faith_pd_obs", "type": "must_have_obs_keys",
         "keys": [["faith_pd", "pd_whole_tree", "phylogenetic_diversity"]]},
        {"id": "unifrac_distance", "type": "beta_diversity_present",
         "uns_or_obsp_keys": ["unifrac", "weighted_unifrac",
                                "unweighted_unifrac"],
         "rationale": "UniFrac n×n distance matrix"},
        {"id": "unifrac_test", "type": "must_have_uns_keys_regex",
         "patterns": [r"(unifrac|tree).*(permanova|anosim|test)|(permanova|anosim).*(unifrac|tree)"]},
    ],
}


# =====================================================================
# Round-3 task additions — multi-method trajectory, GRN regulons, CCC,
# spatial deconvolution, foundation models. Prompts are method-agnostic
# (per the no-answer-leak policy): describe the goal in biological /
# numeric terms; let the agent discover the right ov dispatch wrapper
# via registry / skill lookup.
# =====================================================================

F04 = {
    "id": "F04_multi_backend_pseudotime",
    "layer": "F",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pancreas_endo_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "Pancreas endocrinogenesis dataset (same as F01-F03). The "
        "developmental trajectory begins at the Ductal progenitor "
        "population in `obs['clusters']` and resolves into terminal "
        "cell types.\n\n"
        "Compute per-cell pseudotime using **at least two distinct "
        "trajectory-inference algorithms** (different statistical "
        "frameworks — e.g. graph-diffusion, kinetic-modelling, "
        "lineage-curve fitting). For each algorithm, store the "
        "per-cell pseudotime under a separate obs column whose name "
        "ends with `_pseudotime` or `_dpt` (e.g. "
        "`pseudotime_<method>`, `<method>_pseudotime`, "
        "`dpt_<method>`).\n\n"
        "The two pseudotime estimates should agree directionally on "
        "real signal — a meaningful Spearman correlation (≥ 0.4) "
        "between them confirms the trajectory is recoverable. "
        "Anchor every method at the Ductal progenitor population. "
        "Modify the dataset in place."
    ),
    "checks": [
        {"id": "clusters_obs", "type": "must_have_obs_keys",
         "keys": [["clusters"]]},
        {"id": "two_pseudotime_obs", "type": "obs_count_matching_regex",
         "pattern": r"pseudotime$|_dpt$|^pseudotime_",
         "min_count": 2,
         "rationale": "≥2 pseudotime estimates from distinct backends"},
        {"id": "pseudotime_anchored_at_root", "type": "pseudotime_root_anchored",
         "groupby_obs": "clusters",
         "root_cluster": "Ductal",
         "pseudotime_obs_pattern": r"pseudotime$|_dpt$|^pseudotime_",
         "rationale": "median pseudotime in Ductal cluster should be lowest"},
        {"id": "pseudotime_methods_agree", "type": "pairwise_pseudotime_correlation",
         "pseudotime_obs_pattern": r"pseudotime$|_dpt$|^pseudotime_",
         "min_pairwise_spearman": 0.4,
         "min_methods": 2,
         "rationale": "every pair of methods must reach Spearman ρ ≥ 0.4 — "
                       "more methods is allowed but each new one must agree "
                       "with the others (no random / shuffled passes)"},
    ],
}

B10 = {
    "id": "B10_gene_regulatory_network",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pancreas_endo_raw.h5ad",
    "oracle":  None,
    "max_turns": 100, "wallclock_s": 1500,
    # SCENIC needs pyscenic / arboreto / boltons / ctxcore / distributed —
    # missing from omicdev, present in the miniforge ``omicverse`` env.
    "conda_env": "omicverse",
    "prompt": (
        "Pancreas endocrinogenesis scRNA-seq dataset (mouse). "
        "Reconstruct the gene-regulatory network and identify "
        "transcription-factor (TF) modules driving the differentiation "
        "trajectory.\n\n"
        "Required outputs (modify dataset in place):\n"
        "  1. Per-cell TF-module activity scores in obsm under a key "
        "whose name contains `regulon` or `tf_module` or `auc` "
        "(shape = `n_cells × n_modules`).\n"
        "  2. uns key whose name contains `regulon` or `network` or "
        "`grn`, holding either an edge list (TF → target gene with "
        "weights) or per-module gene-membership information. ≥3 "
        "distinct TF modules.\n"
        "  3. Ranked top regulators per cell-type cluster: a uns "
        "table with columns including the cluster id, the TF name, "
        "and a specificity / activity score.\n\n"
        "**Pre-downloaded mouse cisTarget databases** (do NOT "
        "re-download; the genome-wide ranking files are several GB "
        "and the network is restricted):\n"
        "  - ranking databases (feather format, glob pattern): "
        "`<SCENIC_DATA_ROOT>/databases/mm10/*.feather`\n"
        "  - TF-motif annotation table: "
        "`<SCENIC_DATA_ROOT>/motif/mm10/motifs-v10nr_clust-nr.mgi-m0.001-o0.0.tbl`\n"
        "Pass these paths to whichever regulon-inference wrapper you "
        "find via `registry_lookup`."
    ),
    "checks": [
        {"id": "regulon_obsm", "type": "must_have_obsm_keys_regex",
         "patterns": [r"regulon|tf_module|^X_auc|auc_"]},
        {"id": "regulon_uns", "type": "must_have_uns_keys_regex",
         "patterns": [r"regulon|network|grn"]},
        {"id": "n_regulons", "type": "obsm_dim_at_least",
         "obsm_key": ["X_regulon_auc", "X_regulons", "regulon_auc",
                       "tf_modules", "X_auc", "X_aucell", "aucell",
                       "X_tf_module_auc"],
         "min_dims": 3,
         "rationale": "≥3 distinct TF modules"},
    ],
}

B11 = {
    "id": "B11_scrna_cell_cell_communication",
    "layer": "B",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 600,
    "prompt": (
        "PBMC scRNA-seq with curated `cell_type` annotations in obs. "
        "Identify ligand-receptor mediated cell-cell communication "
        "between annotated cell types.\n\n"
        "Required outputs (modify dataset in place):\n"
        "  1. uns key whose name contains `ccc` or `lr_pairs` or "
        "`cell_communication` or `cellphonedb` — a DataFrame of "
        "ligand-receptor interactions. Required columns: a "
        "ligand-source-cell-type field, a receptor-target-cell-type "
        "field, the ligand gene, the receptor gene, an interaction-"
        "strength / mean-expression score, and a significance "
        "estimate (p-value or permutation FDR).\n"
        "  2. The table should contain ≥10 inferred interactions "
        "(otherwise the dataset is too sparse).\n"
        "  3. Significantly-interacting cell-type pairs (rows with "
        "p ≤ 0.05) should span ≥3 distinct sender cell types — a "
        "single-source result indicates a missing analysis stage."
    ),
    "checks": [
        {"id": "ccc_table", "type": "must_have_uns_keys_regex",
         "patterns": [r"ccc|lr_pairs|cell_communication|cellphonedb|liana|interactions"]},
        {"id": "celltype_obs", "type": "must_have_obs_keys",
         "keys": [["cell_type", "celltype", "predicted_celltype"]]},
        {"id": "ccc_real_call",
         "type": "tool_output_evidence",
         "description": "real CCC library invocation (cellphonedb/liana/commot/ov.single)",
         "patterns": [
             r"cellphonedb\.cell2cell_communication\.cellphonedb",
             r"\bcellphonedb_v5\b",
             r"ov\.single\.cellphonedb_v\d+",
             r"\bli\.mt\.\w+\b",         # liana method calls (li.mt.cellphonedb, li.mt.connectome)
             r"\bcommot\.tl\.\w+\b",      # commot.tl.spatial_communication
             r"\bcellchat\.\w+\b",
             r"\[CellPhoneDB\]",
             r"\[Liana\]",
         ],
         "rationale": "must call a real CCC library — hand-rolled label-shuffle permutation tests on inline LR lists are not acceptable"},
    ],
}

C05 = {
    "id": "C05_spatial_celltype_deconvolution",
    "layer": "C",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/spatial_lymph_node",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 1200,
    "prompt": (
        "Human lymph-node spatial transcriptomics. The fixture directory "
        "contains TWO inputs:\n"
        "  - `scrna_with_celltypes.h5ad` — paired scRNA-seq reference "
        "with cell-type labels in obs (look for the cell-type / "
        "subset column).\n"
        "  - `sc_reference.h5ad` — alternate cell-type-annotated "
        "scRNA reference for the same tissue (use either).\n\n"
        "Fetch a matching Visium spatial slide (Human Lymph Node "
        "available via `scanpy.datasets.visium_sge('V1_Human_Lymph_"
        "Node')` or any human-lymph-node spatial fixture).\n\n"
        "Estimate per-spot **cell-type composition** (fractional "
        "abundances of each reference cell type at every spatial "
        "location). Required outputs (modify dataset in place):\n"
        "  1. obs columns `frac_<celltype>` per cell type from the "
        "reference (numeric, fractions across cell types should sum "
        "to ≈1 per spot).\n"
        "  2. ≥5 distinct cell-type fraction columns (i.e. ≥5 "
        "non-trivial cell types).\n"
        "  3. obsm key `spatial` with physical coordinates."
    ),
    "checks": [
        {"id": "spatial_obsm", "type": "must_have_obsm_keys",
         "keys": ["spatial"]},
        {"id": "fractions_present", "type": "deconv_fractions_sane",
         "fractions_obs_pattern": r"frac_|_fraction|cell_type_frac",
         "min_n_celltypes": 5,
         "sum_tolerance": 0.10,
         "rationale": "per-spot cell-type fractions sum to ≈1 across ≥5 cell types"},
        {"id": "deconv_real_call", "type": "tool_output_evidence",
         "description": "real spatial-deconvolution library invocation",
         "patterns": [
             r"\bcell2location\.run_colocation",
             r"\bcell2location\.cluster",
             r"\bcell2location\.models\.\w+",
             r"\bsetup_anndata\b.*cell2loc",
             r"\bscvi\.external\.CellAssign",
             r"\bov\.space\.Tangram\b",
             r"\bov\.space\.Deconvolution\b",
             r"\btangram\.mapping\b",
             r"\bdestVI\b",
             r"\bCARD\.run_CARD",
             r"\[Cell2Location\]|\[Tangram\]|\[CARD\]|\[DestVI\]",
         ],
         "rationale": "must invoke a real deconvolution library (cell2location / Tangram / CARD / DestVI / ov.space.Deconvolution) — manually-computed marker-fraction overlays are not acceptable"},
    ],
}

L01 = {
    "id": "L01_foundation_model_cell_embedding",
    "layer": "L",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 1500,
    "prompt": (
        "PBMC scRNA-seq dataset with curated `cell_type` labels. "
        "Generate a per-cell embedding using a single-cell "
        "**foundation model** — a pretrained transformer-style model "
        "that ingests gene-expression vectors and outputs a learned "
        "cell representation. The point of using a foundation model "
        "instead of PCA is that the embedding should preserve global "
        "cell-type structure better than a linear projection.\n\n"
        "Required outputs (modify dataset in place):\n"
        "  1. obsm key starting with `X_` and identifying the "
        "foundation-model backend used: any of `scgpt`, "
        "`geneformer`, `scfoundation`, `uce`, `cellplm`, `tabula`, "
        "`scmulan`, `scbert`, `tgpt`, `cellfm`, `sccello`, `scprint`, "
        "`nicheformer`, `aidocell`, `pulsar` (e.g. `X_geneformer`, "
        "`X_scgpt`). Shape: `(n_cells, n_embed_dims)` with "
        "`n_embed_dims ≥ 16`.\n"
        "  2. Recompute the kNN graph on the foundation embedding "
        "and update `uns['neighbors']`.\n"
        "  3. The foundation embedding should achieve a meaningful "
        "cell-type-structure score (silhouette on `cell_type`).\n\n"
        "**Pre-downloaded checkpoints** — do NOT attempt to "
        "re-download (network is restricted; downloads will fail or "
        "burn the wallclock budget). The local checkpoint locations:\n"
        "  - Geneformer: `<DATA_ROOT>/analysis/omic_test/"
        "llm_model/Geneformer/Geneformer-V2-104M/`\n"
        "  - scGPT / scBERT / scFoundation / UCE / CellPLM / Tabula: "
        "`<FM_CHECKPOINT_ROOT>/"
        "models/<model_name>/` (one subdirectory per backend)\n"
        "Discover the wrapper API via the registry / skill lookup."
    ),
    "checks": [
        {"id": "fm_embedding_obsm", "type": "must_have_obsm_keys_regex",
         "patterns": [r"^X_(scgpt|geneformer|scfoundation|uce|cellplm|tabula|scbert|scmulan|tgpt|cellfm|sccello|scprint|aidocell|pulsar|nicheformer|fm)$|^X_.*foundation|^X_fm"]},
        {"id": "embedding_dim_at_least", "type": "obsm_dim_at_least",
         "obsm_key": ["X_scgpt", "X_geneformer", "X_uce", "X_cellplm",
                       "X_scfoundation", "X_tabula", "X_fm",
                       "X_foundation"],
         "min_dims": 16,
         "rationale": "foundation embeddings are typically 64-512 dim"},
        {"id": "neighbors_on_fm", "type": "must_have_uns_keys",
         "keys": ["neighbors"]},
        {"id": "celltype_silhouette_fm", "type": "obsm_celltype_silhouette",
         "obsm_key": ["X_scgpt", "X_geneformer", "X_uce", "X_cellplm",
                       "X_scfoundation", "X_tabula", "X_fm",
                       "X_foundation"],
         "celltype_obs": "cell_type",
         "min_silhouette": 0.05,
         "rationale": "FM embedding should preserve cell-type structure"},
        {"id": "fm_real_call",
         "type": "tool_output_evidence",
         "description": "real foundation-model invocation",
         "patterns": [
             # Real omicverse SCLLMManager prints these via SCLLMOutput
             r"\[(scGPT|Geneformer|scFoundation|UCE|CellPLM|Tabula)\]\s+(Loading|Loaded|Embedding|Predicting|Inference|FINE-TUNING)",
             r"\[(Embedding|Predicting|Loaded)\]\s+.*(scGPT|Geneformer|scFoundation|UCE|CellPLM)",
             r"ov\.llm\.SCLLMManager\b",
             r"\bSCLLMManager\(\s*model_type\s*=",
             # scGPT-specific tokens
             r"\[scGPT\]\s+Prediction batches",
             r"\bExtracted embeddings\b.*\(\d+,\s*\d+\)",
             # Quiet-style legitimate calls (gpt-5.5 tends to suppress prints)
             r"\.get_embeddings\(",
             r"\.embed_data\(",
             r"\.predict_celltypes\(",
             r"\bClassifier\(.*model_type\s*=",
             r"\bgeneformer\.(EmbExtractor|Classifier|InSilicoPerturber)\b",
             r"from\s+geneformer\s+import",
             r"from\s+scgpt\s+import",
             r"from\s+CellPLM\s+import",
             r"\bSCLLMManager\.\w+\(",
         ],
         "rationale": "must invoke a real FM (omicverse SCLLMManager / pretrained scGPT/Geneformer checkpoint) — hand-rolled PCA + nonlinear feature augmentation aliased as X_geneformer is not acceptable"},
    ],
}

L02 = {
    "id": "L02_foundation_model_celltype_annotation",
    "layer": "L",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 100, "wallclock_s": 1500,
    "prompt": (
        "PBMC scRNA-seq dataset with curated `cell_type` ground-truth "
        "labels in obs. **Fine-tune** a single-cell foundation model "
        "(scGPT, Geneformer, scFoundation, etc.) on the labels and "
        "annotate cells using the fine-tuned head.\n\n"
        "Pure aliasing of the existing `predicted_celltype` column to "
        "`cell_type_<model>` is **not** acceptable — the deliverable is "
        "a real fine-tune run with visible per-epoch training metrics "
        "(``train_acc=`` / ``val_acc=`` / ``epoch N/M`` progress bars).\n\n"
        "Storage targets (modify dataset in place):\n"
        "  1. obs column `cell_type_<model>` (or `<model>_celltype`) with "
        "the fine-tuned predictions covering ≥95% of cells.\n"
        "  2. obsm key `X_<model>` with the embedding used for "
        "annotation.\n\n"
        "**Pre-downloaded checkpoints** — do NOT re-download:\n"
        "  - Geneformer: `<DATA_ROOT>/analysis/omic_test/"
        "llm_model/Geneformer/Geneformer-V2-104M/`\n"
        "  - scGPT / scBERT / scFoundation / UCE / CellPLM / Tabula: "
        "`<FM_CHECKPOINT_ROOT>/"
        "models/<model_name>/`\n\n"
        "**Tip**: HVG-subset to ≤2000 genes before fine-tuning to fit "
        "the wallclock budget; ``adata.write_h5ad('./adata.h5ad')`` "
        "after each major step so partial progress survives a "
        "wallclock kill."
    ),
    "checks": [
        # Sole criterion: actual fine-tuning visible in the agent's tool
        # output. Pure column-aliasing without a real ``.fine_tune`` call
        # leaves no per-epoch training markers and fails this check.
        {"id": "fine_tune_evidence", "type": "fine_tune_evidence",
         "rationale": "trajectory must show per-epoch training metrics "
                       "(train_acc/val_acc/Fine-tuning: progress bar)"},
    ],
}

L03 = {
    "id": "L03_foundation_model_zero_shot_inference",
    "layer": "L",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 50, "wallclock_s": 1200,
    "prompt": (
        "PBMC scRNA-seq dataset. Use a single-cell foundation model "
        "(any of `scgpt`, `geneformer`, `scfoundation`, `uce`, "
        "`cellplm`, `tabula`, etc.) to perform **zero-shot inference** "
        "of either:\n"
        "  (a) gene-expression imputation / denoising — predicting "
        "held-out gene expression from observed ones; or\n"
        "  (b) batch-effect-corrected cell representation — "
        "projecting cells into a model space where batch artifacts "
        "are reduced.\n\n"
        "Required outputs (modify dataset in place):\n"
        "  1. layer or obsm key marked with the foundation-model "
        "identifier, holding the inferred output. Patterns: "
        "`layers['<model>_imputed' / '_denoised']` or "
        "`obsm['X_<model>_corrected']`.\n"
        "  2. uns dict containing `<model>` AND "
        "(`inference_summary` / `metrics`), with at minimum a model "
        "name and either correlation-vs-observed or a qualitative "
        "quality summary.\n"
        "  3. Pick whichever sub-task (imputation or batch correction) "
        "the foundation model supports for count data.\n\n"
        "Checkpoints under `<DATA_ROOT>/analysis/"
        "omic_test/llm_model/`. Do NOT re-download."
    ),
    "checks": [
        {"id": "fm_inference_artifact", "type": "must_have_any_container_regex",
         "containers": ["layers", "obsm"],
         "patterns": [r"(scgpt|geneformer|scfoundation|uce|cellplm|tabula|fm|foundation).*(imputed|denoised|corrected|inferred|reconstructed)"]},
        {"id": "fm_inference_summary", "type": "must_have_uns_keys_regex",
         "patterns": [r"(scgpt|geneformer|scfoundation|uce|cellplm|tabula|fm|foundation).*(inference|metrics|summary)|^(scgpt|geneformer|scfoundation|uce|cellplm|tabula|fm|foundation)$"]},
        {"id": "fm_real_call",
         "type": "tool_output_evidence",
         "description": "real foundation-model invocation",
         "patterns": [
             r"\[(scGPT|Geneformer|scFoundation|UCE|CellPLM|Tabula)\]\s+(Loading|Loaded|Embedding|Predicting|Inference|Imputation|Denoising)",
             r"\[(Embedding|Predicting|Loaded|Inference)\]\s+.*(scGPT|Geneformer|scFoundation|UCE|CellPLM)",
             r"ov\.llm\.SCLLMManager\b",
             r"\bSCLLMManager\(\s*model_type\s*=",
             r"\[scGPT\]\s+Prediction batches",
             # Quiet-style legitimate calls (gpt-5.5 tends to suppress prints)
             r"\.get_embeddings\(",
             r"\.embed_data\(",
             r"\.predict_celltypes\(",
             r"\bClassifier\(.*model_type\s*=",
             r"\bgeneformer\.(EmbExtractor|Classifier|InSilicoPerturber)\b",
             r"from\s+geneformer\s+import",
             r"from\s+scgpt\s+import",
             r"from\s+CellPLM\s+import",
             r"\bSCLLMManager\.\w+\(",
         ],
         "rationale": "must invoke a real FM inference call — hand-rolled PCA / corrupted-X overlays aliased as <model>_corrected are not acceptable"},
    ],
}

L04 = {
    "id": "L04_foundation_model_perturbation_response",
    "layer": "L",
    "difficulty": "hard",
    "fixture": f"{INPUTS}/pbmc8k_raw.h5ad",
    "oracle":  None,
    "max_turns": 100, "wallclock_s": 1200,
    "prompt": (
        "PBMC scRNA-seq dataset. Use a single-cell foundation model "
        "(any of `scgpt`, `geneformer`, `scfoundation`, `uce`, "
        "`cellplm`, `tabula`, etc.) to predict **perturbation "
        "response** in a zero-shot / in-context manner: given the "
        "observed expression of every cell, predict how the "
        "transcriptome would change after knocking down a "
        "transcription factor of interest (CD8A, FOXP3, TCF7, IRF8, "
        "or any TF marker for the dominant cell types).\n\n"
        "Required outputs (modify dataset in place):\n"
        "  1. layer key containing `perturb` or `<model>_pred` — "
        "predicted post-perturbation expression matrix.\n"
        "  2. uns key containing `perturbation` and `<model>` — at "
        "minimum: target-gene name, model identifier, and a summary "
        "of the predicted top-N up / down-regulated genes.\n"
        "  3. obs column containing `perturb_response` or "
        "`<model>_response` — per-cell magnitude of predicted "
        "response (norm of perturbed - observed difference).\n\n"
        "Checkpoints under `<DATA_ROOT>/analysis/"
        "omic_test/llm_model/`. Do NOT re-download."
    ),
    "checks": [
        {"id": "perturb_layer", "type": "must_have_layers_regex",
         "patterns": [r"perturb|_pred$|_predicted$"]},
        {"id": "perturb_uns", "type": "must_have_uns_keys_regex",
         "patterns": [r"perturb"]},
        {"id": "perturb_response_obs", "type": "must_have_obs_keys_regex",
         "patterns": [r"perturb|_response$|_pred_magnitude"]},
        {"id": "perturbation_real_call",
         "type": "tool_output_evidence",
         "description": "real perturbation-prediction call",
         "patterns": [
             # SCLLMManager / scFoundation / Tabula perturbation paths
             r"\[(scGPT|Geneformer|scFoundation|UCE|CellPLM|Tabula)\]\s+(Perturbation|Predicting|Inference|Loaded)",
             r"ov\.llm\.SCLLMManager\b",
             r"\bSCLLMManager\(\s*model_type\s*=",
             r"\.predict_perturbation\(",
             r"\bInSilicoPerturber\b",   # geneformer
             r"\[scFoundation\]\s+.*Perturb",
         ],
         "rationale": "must invoke a real perturbation-prediction API — synthesizing a layer ≈ X + small noise to satisfy regex is not acceptable"},
    ],
}


ALL_TASKS = [
    A01, A02, A03, A04, A05,
    B01, B02, B03, B04, B05, B06, B08, B09, B10, B11,
    C01, C02, C04, C05,
    D01, D02,
    E01, E03, E04, E05, E06, E07,
    F01, F02, F03, F04,
    G01, G02, G03, G04, G05,
    L01, L02, L03, L04,
]
