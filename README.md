# verification2

Study which trajectory information improves agent verification.

**Solver → Jev E/R labels → verifier.** The main experiment uses τ²-Bench,
AgentDojo, AppWorld, BIRD and ScienceWorld. Compare QA, QEA, QRA and QERA on
the same trajectories in both partitions. Each benchmark has a frozen split.
SWE-bench Verified remains available as an optional integration.

- [Setup and commands](docs/FRESH_RUN.md)
- [Experiment protocol](docs/FULL_RUN_PLAN.md)
- [Data and prompt contract](docs/SPEC.md)
- [Benchmark sources](docs/BENCHMARKS.md)
- [Experiment results and CSV tables](results/README.md)
- [Tests](tests/README.md)

## Quick start

Needs Ubuntu/Debian with sudo, Python 3.12, [uv](https://docs.astral.sh/uv/)
and GNU Make. Details for each step are in the [setup guide](docs/FRESH_RUN.md).

```bash
# 1. Install
git clone git@github.com:rxthlessbeats/verification2.git && cd verification2
cp .env.example .env && chmod 600 .env   # fill OPENCODE_API_KEY and JEV_API_KEY
make setup                               # project, native benchmark runtimes, task data
make setup-system                        # sudo: Docker and Compose
docker info && docker compose version    # must work without sudo

# 2. Run the selected full training split, then freeze and evaluate its full test split.
make full-train BENCHMARK=bird
make full-test BENCHMARK=bird
```

Choose `tau2_bench`, `agentdojo`, `appworld`, `bird`, or `scienceworld`.
For several benchmarks, use `BENCHMARKS="bird appworld"` on both commands.
The default run folder follows the selection, for example `outputs/full-bird/`.
Use the same selection and ordering for train and test. Full training includes the required first-task pilot.

Use `STAGE=solver`, `STAGE=jev`, or `STAGE=verifier` on either Make target to run one part.
Jev reads the saved solver trajectories. The verifier reads the matching Jev output from
the same run and split. On a fresh run, execute the stages in that order.
`STAGE=all` is the default. `VIEWS` selects the verifier checks and defaults to all four.

```bash
make full-train BENCHMARK=bird STAGE=solver
make full-train BENCHMARK=bird STAGE=jev
make full-train BENCHMARK=bird STAGE=verifier VIEWS="QA QEA"
```

Use `full-test` for the held-out split. Its first invocation freezes the allowed views.
Later verifier invocations can choose a subset of those views.

Each benchmark has one frozen train/test split in `data/splits/<benchmark>.json`:
τ² uses its own split (178 train, 100 test). SWE-bench Verified has no official
split, so all 500 tasks are split 50%/50% (250/250) by a shuffle seeded with 42.
BIRD keeps every Mini-Dev question and a seed-42 train sample of that size.
ScienceWorld's full split is a seed-42 sample of 500 train and 500 test
variations, balanced over task types.
Each file is written the first time its benchmark is used. `--train N` and
`--test N` run the first N tasks per benchmark, balanced over domains (default
all), and a larger N later only adds tasks. The pilot is each benchmark's first
train task. `--jobs N` solves tasks and sends verifier checks N at a time.

`NAME` is the experiment folder `outputs/NAME/`; its first run needs
`--benchmarks`, which the folder then keeps. It holds a timestamped `run.log`, a
`timing.txt` summary and `<pilot|train|test>/report.json`. Metrics are in
`analysis*/summary.csv` in the same folder: false rejection, false acceptance and
coverage per benchmark. Steps (`doctor pilot train freeze test`) run in the order
given; a failed step stops the run, and a second run in the same folder is
refused. `stop` ends a `--detach` run and removes its SWE containers. After a
failure, fix the cause and add `--retry-failed`; completed episodes and API
responses are reused. `scripts/experiment.sh --help` lists every option.

Solved trajectories are stored once per task and solver model in
`data/trajectory/<benchmark>/<model>/`, and later runs only solve tasks that are
missing there. To solve stored tasks again, for example after changing solver
code, add `--output outputs/<new> --rerun all` (or `--rerun "swebench_verified"`);
a stored trajectory is replaced only when the new solve succeeds. Stored
trajectories are keyed by solver model and prompt, so after a solver prompt edit
every task is solved again; Jev and verifier prompt edits reuse them.

Pydantic validates inputs and model answers. Edit repo-owned prompts in
[`prompts.yaml`](src/verification2/core/prompts.yaml). `.env` picks a model for
each role (`SOLVER_MODEL`, `JEV_MODEL`, `VERIFIER_MODEL`); every model's
endpoints, API key variable, limits and reasoning levels are in
[`models.yaml`](src/verification2/core/models.yaml).

```text
src/verification2/
  agents/       solver, Jev classification, verifier
  benchmarks/   native runners, setup, readiness checks
  core/         settings, schemas, prompts, API cache
  data/         input validation, train/test splits
  experiments/  collection, orchestration, protocol, reports
```
