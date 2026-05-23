"""human_scanpy adapter — runs a deterministic hand-coded reference script.

Each task may have a reference script under `baselines/<TASK_ID>.py` that
defines `run(adata) -> adata|str`. This is the C1 anchor: "ov.Agent vs
hand-coded scanpy". When no script exists (e.g. CellPhoneDB / CellFateGenie
/ knowledge tasks), the row gets failure_mode=no_baseline and is excluded
from human_scanpy pass-rate aggregations.
"""

from __future__ import annotations

import importlib.util
import time
import traceback
from pathlib import Path

import anndata as ad

from bench._paths import REPO_ROOT, run_dir

BASELINE_DIR = REPO_ROOT / "baselines"


def run(task: dict, model_id: str, seed: int) -> dict:
    out_dir = run_dir("human_scanpy", task["id"], model_id, seed)
    final_path, final_text, err = None, None, None
    t0 = time.time()

    script = BASELINE_DIR / f"{task['id']}.py"
    if not script.exists():
        return {
            "final_adata_path": None,
            "final_text": None,
            "n_turns": 0,
            "wallclock_s": time.time() - t0,
            "error": f"no_baseline:{script.name}",
        }

    try:
        spec = importlib.util.spec_from_file_location(f"baseline_{task['id']}", script)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if not hasattr(mod, "run"):
            return {
                "final_adata_path": None, "final_text": None, "n_turns": 0,
                "wallclock_s": time.time() - t0,
                "error": "baseline missing `run(adata) -> adata|str`",
            }
        adata_in = ad.read_h5ad(task["fixture"]) if task["fixture"] else None
        result = mod.run(adata_in)
        if isinstance(result, str):
            final_text = result
        elif result is not None:
            final_path = out_dir / "final.h5ad"
            result.write_h5ad(final_path)
    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        with (out_dir / "log.txt").open("a") as f:
            traceback.print_exc(file=f)
    return {
        "final_adata_path": str(final_path) if final_path and final_path.exists() else None,
        "final_text": final_text,
        "n_turns": 0,
        "wallclock_s": time.time() - t0,
        "error": err,
    }
