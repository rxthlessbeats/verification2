SHELL := /bin/bash
.DELETE_ON_ERROR:

.PHONY: help setup setup-system web-setup web-serve test test-unit test-integration test-e2e check
help:
	@echo 'Setup: setup, setup-system, web-setup ARCHIVES=/path/to/assets, then web-serve'
	@echo 'Checks: test, test-unit, test-integration, test-e2e, check'
	@echo 'Experiments: scripts/experiment.sh NAME [STEP...]; see scripts/experiment.sh --help'

setup:
	uv sync --no-cache --locked
	uv run --no-cache python -m verification2.benchmarks.setup

setup-system:
	bash scripts/setup_system.sh

web-setup:
	uv run --no-cache python -m verification2.benchmarks.web_setup $(if $(ARCHIVES),--archives $(ARCHIVES),)

web-serve:
	uv run --no-cache python -m verification2.benchmarks.web_setup --serve

test:
	uv run --no-cache --offline --no-sync pytest -q
test-unit:
	uv run --no-cache --offline --no-sync pytest tests/unit -q
test-integration:
	uv run --no-cache --offline --no-sync pytest tests/integration -q
test-e2e:
	uv run --no-cache --offline --no-sync pytest tests/e2e -q
check:
	uv run --no-cache --offline --no-sync ruff check src tests scripts
	uv run --no-cache --offline --no-sync ruff format --check --quiet src tests scripts
	for script in scripts/*.sh; do bash -n "$$script" || exit; done
	git diff --check
