"""Shared constants + per-run output path resolution."""
from __future__ import annotations

import re
from pathlib import Path

ENDPOINT = "http://127.0.0.1:11434/v1"
API_KEY = "sk-ollama-fake-12345"

OMICDEV_PY = "<CONDA_PREFIX>/bin/python"
BIOMNI_PY  = "<BIOMNI_PREFIX>/bin/python"

REPO_ROOT = Path(__file__).resolve().parents[1]

# Legacy roots kept for backward-compat with older trajectories on disk.
RESULTS = REPO_ROOT / "results"
RUNS_DIR = RESULTS / "runs"
RESULTS.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)


def model_short(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", model_id).strip("_")


def run_dir(system: str, task_id: str, model_id: str, seed: int) -> Path:
    """Legacy per-(system,task,seed) directory under ``results/runs/``.

    Used by adapters that haven't been migrated to the run-name layout. The
    adapter's ``final.h5ad``, ``log.txt``, ``raw_trace.json`` etc. land here
    and the runner copies / references the artefacts when assembling the
    trajectory JSON.
    """
    p = RUNS_DIR / system / f"{task_id}__{model_short(model_id)}_seed{seed}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def workspace_dir(run_name: str, task_id: str, system: str,
                  model_id: str, seed: int) -> Path:
    """Per-trajectory sandbox under ``data/workspace/<run_name>/``.

    Matches the bixbench layout: workspaces are scoped to ``run_name`` so they
    can be wiped per-run without affecting other sweeps.
    """
    p = (REPO_ROOT / "data" / "workspace" / run_name /
         f"{task_id}__{system}__{model_short(model_id)}__seed{seed}")
    p.mkdir(parents=True, exist_ok=True)
    return p
