# baselines — hand-coded reference scripts

Each `<TASK_ID>.py` exports `run(adata) -> adata | str`. Loaded by the
`human_scanpy` adapter (`bench/adapters/human_scanpy.py`).

This is the **C1 anchor**: the paper's first claim is "ov.Agent improves
single-cell productivity over hand-coded scanpy". That claim is
unfalsifiable without measuring hand-coded scanpy on the same task suite
with the same grader — these scripts are that measurement.

## Coverage

| Task | Has baseline? | Notes |
|---|:-:|---|
| A01-A05 | ✓ | scanpy QC / norm / HVG / PCA+UMAP / Leiden |
| B01 | ✓ | full pipeline (A01-A05 chained) |
| B02 | ✓ | T-cell subset + re-cluster (CD4/CD8 split) |
| B03 | ✓ | Wilcoxon rank_genes_groups |
| B04 | ✓ | harmony + combat as two batch correction methods |
| C01 visium spatial domain | — | requires loading Visium 10x format + spatial-aware clustering; non-trivial in pure scanpy |
| C02 visium HD | — | Visium HD bin-level loading; ov-flagship |
| C03 spatial CCC | — | distance-weighted CCC requires squidpy + cellphonedb composition; ov-flagship |
| D01 multiome joint | — | RNA + ATAC integration via MOFA/GLUE/MultiVI; ov-flagship |
| D02 peak-gene linking | — | requires composing scvi-tools + chromVAR; ov-flagship |
| E01 bulk deconvolution | — | bulk → cell-type fractions via deep-learning model; ov-flagship |
| E02 Bulk2Single | — | uniquely ov.bulk2single; competitors must compose |
| F01 multi-mode velocity | ✓ | scvelo stochastic + dynamical |
| F02 multi-method trajectory | ✓ | scanpy DPT + PCA-euclidean |
| G01 16S diversity | ✓ | numpy/scipy alpha + Bray-Curtis beta |
| G02 multi-method DA | — | requires composing wilcoxon + DESeq2 + ANCOM-BC frameworks; non-trivial |

Tasks without baselines return `failure_mode=no_baseline` — the report
excludes those rows from the human_scanpy aggregate. This is documented
honestly in the paper's methods section.

## Editing

Each script must define a top-level `run(adata)` function. Treat as
deterministic — no randomness without a fixed seed.
