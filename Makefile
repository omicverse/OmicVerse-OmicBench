SHELL := /bin/bash
PYTHON ?= python
ENV    := source bench-env.sh &&

.PHONY: help fetch sweep grade analyze radar clean-traj

help:
	@echo "OmicVerse-OmicBench targets"
	@echo "  fetch              — pull OmicBench tasks + fixtures from HF (~10.7 GB)"
	@echo "  sweep NAME=<yaml>  — run a sweep (writes trajectories/<run>/*.json)"
	@echo "  grade NAME=<run>   — grade a finished run (writes results/<run>/grades.csv)"
	@echo "  analyze            — regenerate cross-run figures + summary"
	@echo "  radar              — regenerate analysis/ovagent_radar_native.png"
	@echo "  clean-traj         — delete trajectories/<run>/  (results stay)"

fetch:
	bash scripts/fetch_omicbench.sh

sweep:
	@if [ -z "$(NAME)" ]; then echo "usage: make sweep NAME=<config.yaml>"; exit 1; fi
	bash scripts/run.sh configs/$(NAME)

grade:
	@if [ -z "$(NAME)" ]; then echo "usage: make grade NAME=<run_name>"; exit 1; fi
	bash scripts/grade.sh $(NAME)

analyze:
	bash scripts/analyze.sh

radar:
	$(PYTHON) analysis/radar_native.py

clean-traj:
	@if [ -z "$(NAME)" ]; then echo "usage: make clean-traj NAME=<run_name>"; exit 1; fi
	rm -rf trajectories/$(NAME)
