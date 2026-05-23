"""Generate per-task markdown specs in tasks_md/ from bench.tasks.ALL_TASKS.

Each ``<TASK_ID>.md`` has YAML frontmatter (so other tooling can parse the
metadata without round-tripping through Python) followed by IMRAD-style
sections: Question, Data, Evaluation mode, Checks, Rationale, Scope.

Run::

    <CONDA_PREFIX>/bin/python scripts/build_task_md.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from bench.tasks import ALL_TASKS  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "tasks_md"
OUT.mkdir(exist_ok=True)


# Per-layer human-facing description (also used in the index).
LAYER_META: list[tuple[str, str, str]] = [
    ("A", "scRNA preprocessing floor",
     "Standard single-cell preprocessing — every system should pass these. "
     "Failure here = real capability deficiency, not task obscurity."),
    ("B", "scRNA workflow",
     "Multi-step planning + multi-method dispatch. Tests whether the system "
     "can sequence ≥2 methodologically-distinct approaches and reconcile them."),
    ("C", "Spatial transcriptomics",
     "Visium DLPFC, Visium HD bin-level, spatial cell-cell communication. "
     "Tasks where omicverse exposes built-in spatial domain / SVG / CCC paths "
     "that the baseline arm has to rediscover from general-purpose libraries."),
    ("D", "Paired multi-omics (RNA + ATAC)",
     "Joint embedding + regulatory peak-gene linking. omicverse has built-in "
     "MOFA / GLUE / WNN paths; the baseline arm must compose them."),
    ("E", "Bulk RNA-seq",
     "Bulk-to-single-cell deconvolution + synthesis. Bulk2Single is uniquely "
     "omicverse; the baseline arm must reach it without being told."),
    ("F", "RNA velocity / trajectory",
     "Multi-mode velocity dispatch + multi-method pseudotime. Tests whether "
     "the system can run ≥2 velocity / trajectory algorithms and grade their "
     "agreement."),
    ("G", "Microbiome 16S",
     "Cross-modality coverage + methodological rigor (multi-method "
     "differential abundance)."),
]
LAYER_BLURB: dict[str, str] = {L: blurb for L, _, blurb in LAYER_META}
LAYER_TITLE: dict[str, str] = {L: title for L, title, _ in LAYER_META}


def _yaml_quote(s: str) -> str:
    """Single-line YAML scalar — escape if it contains awkward chars."""
    if any(c in s for c in [':', '#', "'", '"', '\n']):
        return '"' + s.replace('\\', '\\\\').replace('"', '\\"') + '"'
    return s


def _frontmatter(t: dict) -> str:
    fm = [
        "---",
        f"id: {t['id']}",
        f"layer: {t['layer']}",
        f"difficulty: {t['difficulty']}",
        f"modality: {_yaml_quote(LAYER_TITLE.get(t['layer'], ''))}",
        f"max_turns: {t['max_turns']}",
        f"wallclock_s_limit: {t['wallclock_s']}",
    ]
    if t.get("fixture"):
        fm.append(f"fixture: {_yaml_quote(t['fixture'])}")
    if t.get("oracle"):
        fm.append(f"oracle: {_yaml_quote(t['oracle'])}")
    fm.append(
        f"eval_mode: "
        f"{'knowledge_rubric' if 'knowledge_check' in t else 'adata_checks'}"
    )
    fm.append("---")
    return "\n".join(fm)


def render_task(t: dict) -> str:
    body: list[str] = []
    body.append(_frontmatter(t))
    body.append("")
    body.append(f"# {t['id']}")
    body.append("")
    body.append(f"_{LAYER_TITLE.get(t['layer'], 'Task')} · "
                f"layer {t['layer']} · {t['difficulty']}_")
    body.append("")

    # Question -----------------------------------------------------------
    body.append("## Question")
    body.append("")
    body.append("Verbatim prompt sent to every system:")
    body.append("")
    body.append("> " + t["prompt"].replace("\n", "\n> "))
    body.append("")

    # Data ---------------------------------------------------------------
    body.append("## Data")
    body.append("")
    if t.get("fixture"):
        body.append(f"- **Fixture**: `{t['fixture']}`")
    else:
        body.append("- **Fixture**: (none — knowledge-only task)")
    if t.get("oracle"):
        body.append(f"- **Oracle**: `{t['oracle']}`  (used by graders that "
                    "reference a gold reference)")
    else:
        body.append("- **Oracle**: (none — checks are alias-key existence + "
                    "value sanity only)")
    body.append(f"- **Budget**: {t['max_turns']} agent turns, "
                f"{t['wallclock_s']}s wallclock")
    body.append("")

    # Evaluation mode + Checks ------------------------------------------
    body.append("## Evaluation")
    body.append("")
    if "knowledge_check" in t:
        kc = t["knowledge_check"]
        body.append(f"`eval_mode: knowledge_rubric` — pass threshold "
                    f"**{kc['pass_threshold']} of {len(kc['judge_criteria'])}** "
                    f"criteria.")
        body.append("")
        body.append("Criteria:")
        for i, c in enumerate(kc["judge_criteria"]):
            if isinstance(c, dict):
                body.append(f"{i+1}. **{c.get('description', f'C{i+1}')}** — "
                            f"required-any: {c.get('required_any', [])}")
            else:
                body.append(f"{i+1}. {c}")
        body.append("")
    elif "checks" in t:
        body.append("`eval_mode: adata_checks` — every check must pass; "
                    "score = fraction of checks passed.")
        body.append("")
        body.append("| id | type | params |")
        body.append("|---|---|---|")
        for c in t["checks"]:
            params = {k: v for k, v in c.items()
                      if k not in ("id", "type", "rationale")}
            body.append(f"| `{c['id']}` | `{c['type']}` | "
                        f"`{json.dumps(params)}` |")
        rats = [c for c in t["checks"] if c.get("rationale")]
        if rats:
            body.append("")
            body.append("### Rationale")
            body.append("")
            for c in rats:
                body.append(f"- **`{c['id']}`** — {c['rationale']}")
        body.append("")

    # Why this task ------------------------------------------------------
    body.append("## Why this task")
    body.append("")
    body.append(LAYER_BLURB.get(t["layer"], ""))
    body.append("")

    # Scope of changes (governance) -------------------------------------
    body.append("## Scope of changes")
    body.append("")
    body.append("Bug fixes that admit a legitimate solution can be made "
                "without prior approval — oracle generation, grader "
                "implementation, formatting.")
    body.append("")
    body.append("**Approval required** before:")
    body.append("- adding or removing this task,")
    body.append("- changing the prompt text (the model-facing string),")
    body.append("- lowering grader thresholds after a system has failed "
                "(no goalpost-moving).")
    body.append("")
    return "\n".join(body)


def render_index() -> str:
    idx: list[str] = []
    idx.append("# Task suite — index")
    idx.append("")
    idx.append(f"**{len(ALL_TASKS)} tasks across {len({t['layer'] for t in ALL_TASKS})} layers.**")
    idx.append("")
    idx.append("Read each task spec end-to-end before approving a sweep. If a "
               "prompt is ambiguous, a grader threshold is wrong, or an oracle "
               "is incorrect — flag it.")
    idx.append("")

    for L, title, blurb in LAYER_META:
        layer_tasks = [t for t in ALL_TASKS if t["layer"] == L]
        if not layer_tasks:
            continue
        idx.append(f"## Layer {L} — {title}")
        idx.append("")
        idx.append(blurb)
        idx.append("")
        idx.append("| ID | Title | Difficulty | Fixture | Oracle |")
        idx.append("|---|---|:-:|---|---|")
        for t in layer_tasks:
            title_short = t["id"].split("_", 1)[1].replace("_", " ")
            fix = Path(t["fixture"]).name if t.get("fixture") else "—"
            ora = Path(t["oracle"]).name if t.get("oracle") else "—"
            idx.append(f"| [{t['id']}]({t['id']}.md) | {title_short} | "
                       f"{t['difficulty']} | `{fix}` | `{ora}` |")
        idx.append("")

    idx.append("## Cross-cutting policy")
    idx.append("")
    idx.append(f"1. **Prompts describe the goal, not the API.** No code-style "
               f"indexing (e.g. no `adata.obs[\"foo\"]`), no library names. "
               f"Reference target keys by name (\"the obs key "
               f"`pct_counts_mt`\"). This leaves ov.Agent's "
               f"`_detect_direct_python_request` shortcut detector inactive "
               f"across all {len(ALL_TASKS)} prompts.")
    idx.append("2. **Grader accepts alias groups.** v1 only accepted scanpy "
               "naming → unfairly penalized ov-conventions. v3 takes any-of "
               "`[pct_counts_mt | mito_perc]`.")
    idx.append("3. **Numerical graders where possible** (ARI, Jaccard, "
               "silhouette, value range, cosine), not just key existence.")
    idx.append("4. **Oracle adata for every chained-A task** — pre-computed "
               "from `omicverse.datasets.*`, regenerable via "
               "`fixtures/build_oracles.py`.")
    idx.append("5. **Knowledge tasks (rubric) use substring scoring** with "
               "explicit `required_any` lists — replaceable with an LLM "
               "judge in v1.1.")
    idx.append("")
    return "\n".join(idx)


def main() -> None:
    for t in ALL_TASKS:
        (OUT / f"{t['id']}.md").write_text(render_task(t))
    (OUT / "README.md").write_text(render_index())
    print(f"wrote {len(ALL_TASKS)} task markdowns + index → {OUT}")


if __name__ == "__main__":
    main()
