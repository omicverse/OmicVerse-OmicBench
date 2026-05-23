"""ovagent-bench — benchmark for ov.Agent vs Biomni vs raw_llm vs human_scanpy.

Top-level layout:
  tasks_md/    human-readable task spec markdowns
  baselines/   hand-coded scanpy reference scripts
  fixtures/    inputs/ + oracles/
  scripts/     ollama_up.sh, sweep.sh, analyze.sh
  bench/       this python package (tasks, grader, runner, adapters, stats, report, figures)
  results/     symlink → /scratch/.../ovagent_bench_data/results/
  legacy/      archived earlier iterations (not used)
"""
__version__ = "3.1"
