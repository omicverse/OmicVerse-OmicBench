"""Per-system invokers.

Each entry exports ``run(task, model_id, seed) -> dict`` with keys
``final_adata_path / final_text / n_turns / wallclock_s / error``.

The v1 benchmark uses three systems sharing a clean control structure:

- ``mini_swe_omicverse``  — mini-swe-agent + omicverse domain prompt (treatment)
- ``mini_swe_baseline``   — mini-swe-agent + bare system prompt (control)
- ``human_scanpy``        — deterministic hand-coded scanpy / scvelo / numpy
                            reference scripts (ceiling reference)

Both mini-swe arms run the **same loop, same tools (bash only), same model,
same data**. The only difference between them is whether the system prompt
contains ``prompts/omicverse_system.md``. This is the cleanest "domain-prompt
ablation" the benchmark can offer.

Legacy adapters (``ov_agent``, ``raw_llm``, ``biomni``) are preserved under
``_legacy/`` for archive but are not registered.
"""

from . import human_scanpy, mini_swe

SYSTEMS = {
    # qwen3.6 via local ollama
    "mini_swe_omicverse": mini_swe.run_omicverse,
    "mini_swe_baseline":  mini_swe.run_baseline,
    # gpt-5.5 via ChatGPT OAuth (Codex CLI tokens reused)
    "gpt_omicverse":    mini_swe.run_gpt_omicverse,
    "gpt_baseline":     mini_swe.run_gpt_baseline,
    # DeepSeek API (OpenAI-compatible)
    "deepseek_omicverse": mini_swe.run_deepseek_omicverse,
    "deepseek_baseline":  mini_swe.run_deepseek_baseline,
    # OpenRouter (OpenAI-compatible) — same qwen model, different backend so
    # we can re-run qwen tasks without competing against the local Ollama.
    "openrouter_omicverse": mini_swe.run_openrouter_omicverse,
    "openrouter_baseline":  mini_swe.run_openrouter_baseline,
    # Gemini (Google's OpenAI-compat endpoint) — 4th LLM provider arm.
    "gemini_omicverse":     mini_swe.run_gemini_omicverse,
    "gemini_baseline":      mini_swe.run_gemini_baseline,
    # Zhipu GLM-5.1 (BigModel coding-plan endpoint, OpenAI-compatible) — 5th arm.
    "glm_omicverse":        mini_swe.run_glm_omicverse,
    "glm_baseline":         mini_swe.run_glm_baseline,
    # MiniMax M-series (OpenAI-compatible) — 6th arm.
    "minimax_omicverse":    mini_swe.run_minimax_omicverse,
    "minimax_baseline":     mini_swe.run_minimax_baseline,
    # Ablation arms: omicverse prompt MINUS the registry-lookup section.
    # Only on gpt-5.5 (via Codex CLI OAuth) and deepseek-v4-flash for the registry-channel
    # attribution study.
    "gpt_omicverse_no_registry":    mini_swe.run_gpt_omicverse_no_registry,
    "deepseek_omicverse_no_registry": mini_swe.run_deepseek_omicverse_no_registry,
    # Ablation arms: omicverse prompt + vanilla doc-RAG (embedding lookup
    # over raw docstrings) instead of Beacon's structured registry. Tests
    # whether Beacon's library-side contract (aliases, requires/produces,
    # prerequisites, auto_fix, skill metadata) outperforms generic RAG.
    "gpt_omicverse_doc_rag":        mini_swe.run_gpt_omicverse_doc_rag,
    "deepseek_omicverse_doc_rag":     mini_swe.run_deepseek_omicverse_doc_rag,
    # deterministic hand-coded ceiling reference
    "human_scanpy":       human_scanpy.run,
}
