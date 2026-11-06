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
`raw/`, saves dependency locks and installs Chromium. Keep those locks with run
artifacts. Native dependencies may include SDKs/tokenizers; the runners do not
use local token or cost estimates.

`setup-system` uses sudo on Debian/Ubuntu to install Docker, Compose and browser
libraries. If Make is missing, first run `bash scripts/setup_system.sh`, then
repeat `make setup-system` after native setup. Other operating systems need the
equivalent host packages. Verify `docker info` and `docker compose version`.

## WebArena services

Task JSON and saved trajectories do not contain the website databases. Reuse or
download these original assets into one directory:

| Asset | Source |
|---|---|
| `shopping_final_0712.tar` | [Shopping](https://archive.org/download/webarena-env-shopping-image) |
| `shopping_admin_final_0719.tar` | [Admin](https://archive.org/download/webarena-env-shopping-admin-image) |
| `postmill-populated-exposed-withimg.tar` | [Forum](https://archive.org/download/webarena-env-forum-image) |
| `gitlab-populated-final-port8023.tar` | [GitLab](https://archive.org/download/webarena-env-gitlab-image) |
| `wikipedia_en_all_maxi_2022-05.zim` | [Wikipedia](https://archive.org/download/webarena-env-wiki-image) |
| `openstreetmap-website-db.tar.gz` | [Map database](https://zenodo.org/records/12636845/files/openstreetmap-website-db.tar.gz) |
| `openstreetmap-website-web.tar.gz` | [Map server](https://zenodo.org/records/12636845/files/openstreetmap-website-web.tar.gz) |
| `openstreetmap-website.tar.gz` | [Map source](https://zenodo.org/records/12636845/files/openstreetmap-website.tar.gz) |

`make setup` does not download these large assets. See the
[pinned deployment instructions](https://github.com/gasse/webarena-setup/blob/8d0e0a20d50e4a8c02d1520195182e19fc3c080f/webarena/README.md).

```bash
make web-setup ARCHIVES=/path/to/assets
make web-serve  # Keep running in another terminal.
```

Setup records ownership of seven containers and refuses conflicts. Episode
resets recreate those benchmark containers and reset task state. Run one
experiment at a time. The native map also uses external map services.

Generated URLs are saved in `raw/websites/.env`; experiments load them before `.env`.
Use the `WA_*` settings in `.env.example` for existing services with a compatible
native reset endpoint. Resets affect the configured benchmark services.

## Experiment

```bash
scripts/experiment.sh tau2 doctor --benchmarks tau2_bench
scripts/experiment.sh tau2 pilot train --train 50 --jobs 8 --detach   # Training, QERA.
scripts/experiment.sh tau2 train --train 50 --conditions "QA QEA QRA QERA"  # Comparisons.
scripts/experiment.sh tau2 freeze test --jobs 8                       # After final settings.
```

`NAME` selects `outputs/NAME/`, whose first run needs `--benchmarks`; each run writes a
timestamped `run.log` and a `timing.txt` summary. `--detach` keeps a run going
after the terminal closes; check it with `status` and end it with `stop`, which
also removes its SWE containers. Only one run per experiment folder can be
active. See `scripts/experiment.sh --help`.

Doctor checks sources, locks, imports, Docker, Chromium, keys, DNS and services;
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
