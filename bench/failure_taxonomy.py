"""7-class failure-mode tagger from trace + grader output (per design plan).

Order matters: more specific tags before generic ones.
"""
from __future__ import annotations


def classify(*, run_result: dict, grade_passed: bool, task: dict, system: str) -> str:
    if grade_passed:
        return "none"

    err = (run_result.get("error") or "").lower()
    final_path = run_result.get("final_adata_path")
    final_text = run_result.get("final_text") or ""

    # Adapter / infra failures (NOT system's fault) ----------------------
    if err.startswith("no_baseline"):
        return "no_baseline"
    if "biomni driver exit" in err or "biomni timed out" in err:
        return "adapter_error"
    if "load failed" in err and "iospec" in err:
        return "adapter_error"

    # Hard timeouts / max-turns ------------------------------------------
    if "timed out" in err or "timeout" in err:
        return "exceeded_turns"
    if run_result.get("n_turns", 0) >= task["max_turns"]:
        return "exceeded_turns"

    # Runtime errors during code execution -------------------------------
    if "runtime error" in err:
        return "code_runtime_error"
    if "attributeerror" in err and "module" in err:
        return "hallucinated_fn"

    # Knowledge tasks (D-layer) ------------------------------------------
    if task["layer"] == "D":
        if not final_text.strip():
            return "silent_none"
        return "judge_rejected"

    # Adata tasks: produced no final adata --------------------------------
    if final_path is None:
        return "silent_none"

    # Default: distinguish workflow-order (B*) from generic tool-pick -----
    if task["layer"] == "B":
        return "wrong_workflow_order"
    return "wrong_tool_choice"
