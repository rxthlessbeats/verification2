SHELL := /bin/bash
.DELETE_ON_ERROR:

BENCHMARKS ?= $(BENCHMARK)
empty :=
space := $(empty) $(empty)
RUN ?= full-$(subst $(space),-,$(strip $(BENCHMARKS)))
JOBS ?= 1
STAGE ?= all
VIEWS ?= QA QEA QRA QERA

.PHONY: help setup setup-system test test-unit test-integration test-e2e check full-train full-test
help:
	@echo 'Setup: setup, setup-system'
	@echo 'Checks: test, test-unit, test-integration, test-e2e, check'
	@echo 'Experiments: full-train, full-test BENCHMARK=bird (or BENCHMARKS="bird appworld"), STAGE=all|solver|jev|verifier, VIEWS="QA QEA QRA QERA"'

full-train full-test: require-benchmark

.PHONY: require-benchmark
require-benchmark:
	$(if $(strip $(BENCHMARKS)),,$(error Choose a benchmark with BENCHMARK=bird or BENCHMARKS="bird appworld"))

full-train:
	bash scripts/experiment.sh "$(RUN)" pilot train --benchmarks "$(BENCHMARKS)" --train all --stage "$(STAGE)" --conditions "$(VIEWS)" --jobs "$(JOBS)"

full-test:
	bash scripts/experiment.sh "$(RUN)" freeze test --benchmarks "$(BENCHMARKS)" --test all --stage "$(STAGE)" --conditions "$(VIEWS)" --jobs "$(JOBS)"

setup:
	uv sync --no-cache --locked
	uv run --no-cache python -m verification2.benchmarks.setup

setup-system:
	bash scripts/setup_system.sh

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
