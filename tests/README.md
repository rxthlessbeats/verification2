# Tests

| Folder | Scope | Command |
|---|---|---|
| `unit/` | Parsing, validation, rendering and metrics | `make test-unit` |
| `integration/` | Files, adapters, CLI, transport and native interfaces | `make test-integration` |
| `e2e/` | Three-stage workflows, resume and frozen held-out checks | `make test-e2e` |

`make test` runs all tests; `make check` runs lint, formatting and shell checks.
Fixtures stay byte-for-byte unchanged. Shared builders live in `support.py`.
Scripted native/provider responses make no paid calls and do not establish
benchmark quality.

For a live run, complete [setup](../docs/FRESH_RUN.md), then
`scripts/experiment.sh smoke doctor pilot --benchmarks tau2_bench`.
Inspect `outputs/smoke/pilot/report.json`, `readiness.json`, native episode
logs and Jev/verifier request/result files. A blocked preflight makes no model
requests. Formal collection requires a passing three-benchmark reasoning pilot.
