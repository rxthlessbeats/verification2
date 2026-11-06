# verification2

Study which trajectory information improves agent verification.

**Solver → Jev E/R labels → verifier.** Use fresh τ²-Bench, SWE-bench Verified
and ARB/WebArena episodes. Each benchmark has a frozen train/test split, and its
first train task is the pilot. Develop with QERA on train; compare QA/QEA/QRA/QERA
once on the held-out test set. Optionally, Jev also labels each E/R piece useful,
misleading or irrelevant, and the QJevA view keeps only the useful ones
(`--conditions "QA QEA QRA QERA QJevA"`).

- [Setup and commands](docs/FRESH_RUN.md)
- [Experiment protocol](docs/FULL_RUN_PLAN.md)
- [Data and prompt contract](docs/SPEC.md)
- [Benchmark sources](docs/BENCHMARKS.md)
- [Tests](tests/README.md)

## Quick start

Needs Ubuntu/Debian with sudo, Python 3.12, [uv](https://docs.astral.sh/uv/)
and GNU Make. Details for each step are in the [setup guide](docs/FRESH_RUN.md).

```bash
# 1. Install
git clone git@github.com:rxthlessbeats/verification2.git && cd verification2
cp .env.example .env && chmod 600 .env   # fill OPENCODE_API_KEY and JEV_API_KEY
make setup                               # project, native benchmark runtimes, task data
make setup-system                        # sudo: Docker, Compose, Chromium libraries
docker info && docker compose version    # must work without sudo

# 2. WebArena sites: download the 8 assets listed in docs/FRESH_RUN.md into one folder
make web-setup ARCHIVES=/path/to/assets
make web-serve                           # keep running in a second terminal

# 3. Check readiness (no paid calls)
scripts/experiment.sh tau2 doctor --benchmarks tau2_bench

# 4. Run (paid API calls); --detach keeps going after the terminal closes
scripts/experiment.sh tau2 pilot train --train 50 --jobs 8 --detach  # pilot must pass; QERA
scripts/experiment.sh tau2 status                    # running or not, timing, latest lines
scripts/experiment.sh tau2 train --train 50 --conditions "QA QEA QRA QERA"
scripts/experiment.sh tau2 freeze test --jobs 8      # lock settings, then held-out test once
```

Each benchmark has one frozen train/test split in `data/splits/<benchmark>.json`:
τ² uses its own split (178 train, 100 test). SWE-bench Verified has no official
split, so all 500 tasks are split 50%/50% (250/250) by a shuffle seeded with 42.
WebArena's 694 tasks with model-free grading are split the same way (347/347);
AgentRewardBench's official split covers only 100 of them.
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
## Results: τ²-Bench

All 278 tasks of τ²'s official split, one fresh solver episode each (run of
2026-11-06 in `outputs/tau2`). Solver and verifier are `qwen3.8-flash` at the
provider's default reasoning level, temperature 0, seed 42; Jev is
`typesafe-ai/jev`. The verifier sees the question (Q), the final answer (A) and
optionally the environment evidence (E) and the solver's reasoning (R) as labelled
by Jev. QJevA shows Q, A and only the E/R pieces Jev's selector labelled useful
(73% of 3,445 test pieces; 18% irrelevant, 8% misleading). False rejection is the share of natively correct episodes the verifier
calls incorrect; false acceptance is the share of natively incorrect episodes it
calls correct. Brackets are 95% bootstrap intervals over tasks (2,000 draws,
conditions resampled together). Every check returned a valid verdict (coverage
100%).

| Split | View | Native correct / incorrect | False rejection | False acceptance |
|---|---|---|---|---|
| Train (178) | QERA | 159 / 19 | 11.3% [6.4, 16.8] | 63.2% [40.9, 83.3] |
| Test (100) | QA | 87 / 13 | 9.2% [3.5, 15.7] | 69.2% [40.0, 93.3] |
| Test (100) | QEA | 87 / 13 | 11.5% [5.6, 18.6] | 30.8% [7.1, 58.3] |
| Test (100) | QRA | 87 / 13 | 12.6% [6.1, 20.3] | 61.5% [33.3, 88.9] |
| Test (100) | QERA | 87 / 13 | 11.5% [5.6, 18.8] | 69.2% [41.7, 93.3] |
| Test (100) | QJevA | 87 / 13 | 11.5% [4.8, 18.4] | 53.8% [27.3, 81.8] |

Paired changes on the test set, in percentage points:

| Change | False rejection | False acceptance |
|---|---|---|
| Add E (QA → QEA) | +2.3 [−3.4, +8.4] | −38.5 [−72.7, 0.0] |
| Add R (QA → QRA) | +3.4 [−1.2, +8.6] | −7.7 [−35.7, +20.0] |
| Add R to E (QEA → QERA) | 0.0 [−3.5, +3.4] | **+38.5 [+12.5, +66.7]** |
| Add E to R (QRA → QERA) | −1.1 [−6.5, +3.5] | +7.7 [−18.2, +35.7] |
| Jev selection (QERA → QJevA) | 0.0 [−5.7, +5.5] | −15.4 [−53.8, +22.2] |
