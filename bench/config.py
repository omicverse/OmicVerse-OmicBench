"""Run-config loader.

A YAML in ``configs/`` describes one reproducible benchmark run. ``load_config``
resolves it into a normalised dict and computes the per-run output paths. The
runner cross-products ``scope.systems × scope.task_ids × scope.seeds`` and
emits one trajectory JSON per run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from bench._paths import REPO_ROOT
from bench.tasks import ALL_TASKS

VALID_SYSTEMS = {
    "mini_swe_omicverse", "mini_swe_baseline",
    "gpt_omicverse", "gpt_baseline",
    "deepseek_omicverse", "deepseek_baseline",
    "openrouter_omicverse", "openrouter_baseline",
    "gemini_omicverse", "gemini_baseline",
    "glm_omicverse", "glm_baseline",
    "minimax_omicverse", "minimax_baseline",
    "gpt_omicverse_no_registry", "deepseek_omicverse_no_registry",
    "gpt_omicverse_doc_rag", "deepseek_omicverse_doc_rag",
    "human_scanpy",
}


def load_config(config_path: str | os.PathLike) -> dict[str, Any]:
    """Load + lightly validate a run-config YAML."""
    path = Path(config_path).resolve()
    with path.open() as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping")

    if "run_name" not in cfg or not str(cfg["run_name"]).strip():
        raise ValueError(f"{path}: missing required field `run_name`")

    cfg.setdefault("description", "")
    cfg.setdefault("llm", {})
    cfg.setdefault("agent", {})
    cfg.setdefault("scope", {})
    cfg.setdefault("paths", {})

    llm = cfg["llm"]
    llm.setdefault("model", "qwen3.6:35b-a3b")
    llm.setdefault("models", None)
    llm.setdefault("endpoint", "http://127.0.0.1:11434/v1")
    llm.setdefault("api_key_env", "OLLAMA_API_KEY")
    llm.setdefault("temperature", 0.0)

    agent = cfg["agent"]
    agent.setdefault("max_agent_turns", 35)
    agent.setdefault("shortcut", "on")

    scope = cfg["scope"]
    scope.setdefault("systems", ["mini_swe_omicverse", "mini_swe_baseline", "human_scanpy"])
    scope.setdefault("task_ids", [])
    scope.setdefault("layers", [])
    scope.setdefault("difficulties", [])
    scope.setdefault("seeds", [0])
    scope.setdefault("skip_completed", True)

    bad = [s for s in scope["systems"] if s not in VALID_SYSTEMS]
    if bad:
        raise ValueError(f"{path}: unknown systems {bad}; valid: {sorted(VALID_SYSTEMS)}")

    paths = cfg["paths"]
    paths.setdefault("trajectories_dir", "trajectories")
    paths.setdefault("results_dir", "results")
    paths.setdefault("workspace_dir", "data/workspace")
    paths.setdefault("logs_dir", "logs")

    cfg["_config_path"] = str(path)
    return cfg


def resolve_models(cfg: dict[str, Any]) -> list[str]:
    """Return the list of models to sweep. ``llm.models`` (list) overrides
    ``llm.model`` (single str)."""
    models = cfg["llm"].get("models")
    if models:
        return list(models)
    return [cfg["llm"]["model"]]


def resolve_tasks(cfg: dict[str, Any]) -> list[dict[str, Any]]:
    """Filter ``ALL_TASKS`` by config scope."""
    scope = cfg["scope"]
    tasks = list(ALL_TASKS)
    ids = list(scope.get("task_ids") or [])
    if ids:
        tasks = [t for t in tasks if t["id"] in ids]
    layers = list(scope.get("layers") or [])
    if layers:
        tasks = [t for t in tasks if t["layer"] in layers]
    diffs = list(scope.get("difficulties") or [])
    if diffs:
        tasks = [t for t in tasks if t["difficulty"] in diffs]
    return tasks


def run_paths(cfg: dict[str, Any]) -> dict[str, Path]:
    """Per-run output paths, sharded by ``run_name``. Created on demand."""
    run_name = cfg["run_name"]
    p = cfg["paths"]
    out = {
        "trajectories": REPO_ROOT / p["trajectories_dir"] / run_name,
        "results":      REPO_ROOT / p["results_dir"]      / run_name,
        "workspace":    REPO_ROOT / p["workspace_dir"]    / run_name,
        "logs":         REPO_ROOT / p["logs_dir"]         / run_name,
    }
    for v in out.values():
        v.mkdir(parents=True, exist_ok=True)
    return out


def trajectory_path(rp: dict[str, Path], task_id: str, system: str,
                    model_id: str, seed: int) -> Path:
    """Per-trajectory JSON path under ``trajectories/<run_name>/``.

    Filename pattern: ``<task_id>__<system>__<model_short>__seed<n>.json``.
    Multiple-model sweeps land in the same run dir but differ by ``model_short``.
    """
    from bench._paths import model_short
    fname = f"{task_id}__{system}__{model_short(model_id)}__seed{seed}.json"
    return rp["trajectories"] / fname
