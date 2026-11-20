# Run

Requires a repository checkout, Python 3.12, uv and GNU Make.

## Setup

On a fresh checkout, copy `.env.example` to `.env`, set permissions to `600`,
and fill in `OPENCODE_API_KEY` and `JEV_API_KEY`. `.env` picks a model for each
role; each model's endpoints, key variable and limits are in
[`models.yaml`](../src/verification2/core/models.yaml). Keep an existing `.env`
intact.

```bash
make setup
make setup-system
```

Setup reuses pinned sources and task caches, creates native environments under
`raw/`, and saves dependency locks. Keep those locks with run
artifacts. Native dependencies may include SDKs/tokenizers; the runners do not
use local token or cost estimates.

`setup-system` uses sudo on Debian/Ubuntu to install Docker and Compose.
If Make is missing, first run `bash scripts/setup_system.sh`, then
repeat `make setup-system` after native setup. Other operating systems need the
equivalent host packages. Verify `docker info` and `docker compose version`.

## Experiment

```bash
make full-train BENCHMARK=bird
make full-test BENCHMARK=bird
```

Choose one benchmark with `BENCHMARK` or several with `BENCHMARKS="bird appworld"`. Both targets use all four views and share a run folder named for the selection. With the underlying script, `NAME` selects `outputs/NAME/`, whose first run needs `--benchmarks`; each run writes a
timestamped `run.log` and a `timing.txt` summary. `--detach` keeps a run going
after the terminal closes; check it with `status` and end it with `stop`, which
also removes its SWE containers. Only one run per experiment folder can be
active. See `scripts/experiment.sh --help`.

Doctor checks sources, locks, imports, Docker, keys, DNS and services;
it makes no paid calls. A blocked pilot also makes no model requests. Each
benchmark must pass the pilot with a native score and recorded readable reasoning.

Tasks come from the frozen split in `data/splits/<benchmark>.json`: τ²'s own
train/test split, or for benchmarks without one, all eligible tasks split
50%/50% by a shuffle seeded with 42, written the first time they are used. `--train N` and `--test N` run the first N tasks per
benchmark, balanced over domains (default all); the pilot is the first train task.
Selection precedes solver outcomes. Do not inspect test records during development.
See the [protocol](FULL_RUN_PLAN.md) for scoring and paired comparisons.

Copy `src/verification2/core/prompts.yaml` to version training changes. Pass
`--prompts path/to/prompts.yaml` to each run. Verifier-only changes reuse Jev
outputs; Jev changes create another classified version. Solver changes solve
every task again and need a new pilot. Freeze the selected source, locks, prompts, models,
output settings and four views before testing.

Runs resume completed stages. Failed attempts need diagnosis, then
`--retry-failed`. Earlier attempts and API usage stay saved; completed solver
episodes never rerun. Partial batches publish only valid rows and report
failures/coverage. Invalid inputs go to `quarantine.jsonl`. Formal collection
requires a passing pilot.

For Messages APIs, the required output maximum is the
model's `max_output_tokens` in `models.yaml`; no smaller local cap is added.

## Logs

Under `outputs/NAME/<pilot|train|test>/`:

| Path | Contents |
|---|---|
| `report.json`, `readiness.json` | Run status and setup failures |
| `solver/episodes/` | Native logs, API calls, scores, frozen records and receipts |
| `solver/task_results.jsonl` | Planned/completed/failed tasks |
| `jev*/` | Requests, decisions, overrides, classified records and provenance |
| `verifier*/` | Blinded requests, verdicts, API usage/cost and timing |
| `analysis*/` | Error rates, coverage, paired intervals and comparisons |
| `receipts.json` | Reusable stage fingerprints and file hashes |

The shared `api_cache/` is under the experiment root. Keep cache and logs private;
they contain full benchmark text. Missing usage/cost stays unknown. Preserve
failed attempts when accounting for cost. Context overflows remain errors.

Use `make check` and `make test` for offline validation. Automated tests use
scripted native/provider responses; they do not replace the live pilot.
