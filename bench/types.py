"""Shared dataclasses + enums used across grader / runner / report.

Lightweight stdlib-only dataclasses (no Pydantic) — the heavyweight Pydantic
schema in legacy/ovagent_bench_pkg/tasks/schema.py was abandoned with v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class FailureMode(str, Enum):
    NONE = "none"
    TOOL_GAP = "tool_gap"
    HALLUCINATED_FN = "hallucinated_fn"
    WRONG_TOOL_CHOICE = "wrong_tool_choice"
    WRONG_WORKFLOW_ORDER = "wrong_workflow_order"
    CODE_RUNTIME_ERROR = "code_runtime_error"
    SILENT_NONE = "silent_none"
    EXCEEDED_TURNS = "exceeded_turns"
    JUDGE_REJECTED = "judge_rejected"
    ADAPTER_ERROR = "adapter_error"
    NO_BASELINE = "no_baseline"


@dataclass
class Grade:
    task_id: str = ""
    system: str = ""
    model_id: str = ""
    seed: int = 0
    passed: bool = False
    score: float = 0.0
    failure_mode: FailureMode = FailureMode.NONE
    rubric: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
